/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Trung Nguyen (UChicago) with Claude Opus 4.8
------------------------------------------------------------------------- */

#include "pair_table_ucg_chirality_gpu.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "info.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "suffix.h"

#include <cmath>

using namespace LAMMPS_NS;

// must match the (anonymous) cv_mode enum in pair_table_ucg_chirality.cpp
enum { NEIGHBOR_INDEPENDENT, NEIGHBOR_DEPENDENT };

// GPU library neighbor modes (see fix_gpu / Device::gpu_mode)
enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

// External functions from the GPU library

int tuc_gpu_init(const int ntypes, double **cutsq, double ***table_coeffs,
                 double **table_data, double *special_lj, const int inum,
                 const int nall, const int max_nbors, const int maxspecial,
                 const double cell_size, int &gpu_mode, FILE *screen,
                 int tabstyle, int ntables, int tablength, const int max_states,
                 const int n_actual_types, const int n_total_states,
                 const int cv_mode, const double distcut, int *nstates_per_type,
                 int *actual_types_from_state, double *cv_thresholds,
                 double *threshold_radii, double *prob_scale,
                 double *prob_prefactor);
void tuc_gpu_clear();
void tuc_gpu_compute_prob(const int ago, const int inum_full, const int nall,
                          double **host_x, int *host_type, int *ilist,
                          int *numj, int **firstneigh, double *host_chirality,
                          int &host_start, const double cpu_time, bool &success,
                          void **prob_ptr);
void tuc_gpu_compute_forces(const int eflag, const int vflag, const int inum,
                            const int nall, int *ilist, double **host_prob,
                            double **host_partial);
double tuc_gpu_bytes();

/* ---------------------------------------------------------------------- */

PairTableUCGChiralityGPU::PairTableUCGChiralityGPU(LAMMPS *lmp) :
    PairTableUCGChirality(lmp), acc_float(false), cpu_time(0.0)
{
  suffix_flag |= Suffix::GPU;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ---------------------------------------------------------------------- */

PairTableUCGChiralityGPU::~PairTableUCGChiralityGPU()
{
  tuc_gpu_clear();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTableUCGChiralityGPU::init_style()
{
  // base init_style: requires atom_style chiral + newton off, requests a full
  // neighbor list, and extracts the thermostat temperature.  The same host
  // full neighbor list drives both the device kernels and the host Phase-3.
  PairTableUCGChirality::init_style();

  // M2 device kernel does not implement the one-body entropy / chemical
  // potential terms (both are hardwired off in the host style).
  if (using_state_entropy || using_state_chemical_potential)
    error->all(FLERR, "pair table/ucg/chirality/gpu does not support "
                      "state entropy or chemical potentials");
  if (max_states_per_type > 2)
    error->all(FLERR, "pair table/ucg/chirality/gpu supports at most 2 "
                      "substates per type");

  acc_float = Info::has_accelerator_feature("GPU", "precision", "single");

  const int ntypes = atom->ntypes;

  // cutsq is filled by Pair::init() only after init_style(), so recompute it
  // here to size the GPU neighbor binning (mirrors pair_table_gpu).
  double maxcut = -1.0;
  double cut;
  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
        cut = init_one(i, j);
        cut *= cut;
        if (cut > maxcut) maxcut = cut;
        cutsq[i][j] = cutsq[j][i] = cut;
      } else
        cutsq[i][j] = cutsq[j][i] = 0.0;
    }
  }
  double cell_size = sqrt(maxcut) + neighbor->skin;

  // pack the tabulated potentials exactly like pair_table_gpu so the device
  // can look them up by state-pair (alpha,beta) index.
  double ***table_coeffs = nullptr;
  double **table_data = nullptr;
  memory->create(table_coeffs, ntypes + 1, ntypes + 1, 6, "table:coeffs");

  for (int i = 1; i <= ntypes; i++)
    for (int j = 1; j <= ntypes; j++) {
      int n = tabindex[i][j];
      auto *tb = &tables[n];
      table_coeffs[i][j][0] = n;
      table_coeffs[i][j][1] = tb->nshiftbits;
      table_coeffs[i][j][2] = tb->nmask;
      table_coeffs[i][j][3] = tb->innersq;
      table_coeffs[i][j][4] = tb->invdelta;
      table_coeffs[i][j][5] = tb->deltasq6;
    }

  if (tabstyle != BITMAP) {
    memory->create(table_data, ntables, 6 * tablength, "table:data");
    for (int n = 0; n < ntables; n++) {
      auto *tb = &tables[n];
      if (tabstyle == LOOKUP) {
        for (int k = 0; k < tablength - 1; k++) {
          table_data[n][6 * k + 1] = tb->e[k];
          table_data[n][6 * k + 2] = tb->f[k];
        }
      } else if (tabstyle == LINEAR) {
        for (int k = 0; k < tablength; k++) {
          table_data[n][6 * k + 0] = tb->rsq[k];
          table_data[n][6 * k + 1] = tb->e[k];
          table_data[n][6 * k + 2] = tb->f[k];
          if (k < tablength - 1) {
            table_data[n][6 * k + 3] = tb->de[k];
            table_data[n][6 * k + 4] = tb->df[k];
          }
        }
      } else if (tabstyle == SPLINE) {
        for (int k = 0; k < tablength; k++) {
          table_data[n][6 * k + 0] = tb->rsq[k];
          table_data[n][6 * k + 1] = tb->e[k];
          table_data[n][6 * k + 2] = tb->f[k];
          table_data[n][6 * k + 3] = tb->e2[k];
          table_data[n][6 * k + 4] = tb->f2[k];
        }
      }
    }
  } else {
    int ntable = 1 << tablength;
    memory->create(table_data, ntables, 6 * ntable, "table:data");
    for (int n = 0; n < ntables; n++) {
      auto *tb = &tables[n];
      for (int k = 0; k < ntable; k++) {
        table_data[n][6 * k + 0] = tb->rsq[k];
        table_data[n][6 * k + 1] = tb->e[k];
        table_data[n][6 * k + 2] = tb->f[k];
        table_data[n][6 * k + 3] = tb->de[k];
        table_data[n][6 * k + 4] = tb->df[k];
        table_data[n][6 * k + 5] = tb->drsq[k];
      }
    }
  }

  int maxspecial = 0;
  if (atom->molecular != Atom::ATOMIC) maxspecial = atom->maxspecial;
  int mnf = 5e-2 * neighbor->oneatom;

  // distcut is hardcoded to 4.0 in calculate_substate_prob() on the host
  const double distcut = 4.0;
  const int cv_dependent = (cv_mode == NEIGHBOR_DEPENDENT) ? 1 : 0;

  int gpu_mode;
  int success = tuc_gpu_init(
      ntypes + 1, cutsq, table_coeffs, table_data, force->special_lj,
      atom->nlocal, atom->nlocal + atom->nghost, mnf, maxspecial, cell_size,
      gpu_mode, screen, tabstyle, ntables, tablength, max_states_per_type,
      n_actual_types, n_total_states, cv_dependent, distcut, n_states_per_type,
      actual_types_from_state, cv_thresholds, threshold_radii,
      probability_scaling_factor, probability_pre_factor);
  GPU_EXTRA::check_flag(success, error, world);

  memory->destroy(table_coeffs);
  memory->destroy(table_data);

  // M2 only implements the host-neighboring (GPU_FORCE) path.
  if (gpu_mode != GPU_FORCE)
    error->all(FLERR,
               "Pair style table/ucg/chirality/gpu requires host neighbor "
               "list building; use 'package gpu <N> neigh no'");
}

/* ----------------------------------------------------------------------
   Phase 1 on the device: compute per-atom substate probabilities and copy
   them back into the host arrays.
------------------------------------------------------------------------- */

void PairTableUCGChiralityGPU::compute_substate_prob(int /*cv_mode*/)
{
  const int nall = atom->nlocal + atom->nghost;
  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  int host_start;
  bool success = true;
  void *prob_pinned = nullptr;

  tuc_gpu_compute_prob(neighbor->ago, inum, nall, atom->x, atom->type, ilist,
                       numneigh, firstneigh, atom->chirality, host_start,
                       cpu_time, success, &prob_pinned);
  if (!success) error->one(FLERR, "Insufficient memory on accelerator");
  if (host_start < inum)
    error->one(FLERR,
               "pair style table/ucg/chirality/gpu currently requires all "
               "local atoms on the GPU; use 'package gpu 1 split 1.0'");

  // unpack packed probabilities into host arrays.
  // packed layout per row ii (atom i = ilist[ii], stride = 1 + 2*max_states):
  //   [ ee, p[0..ms-1], dp[0..ms-1] ]
  const int ms = max_states_per_type;
  const int stride = 1 + 2 * ms;

  if (acc_float) {
    auto p = (float *) prob_pinned;
    for (int ii = 0; ii < inum; ii++) {
      const int i = ilist[ii];
      const int base = ii * stride;
      enantiomeric_excess[i] = p[base];
      for (int s = 0; s < ms; s++) {
        substate_probability[i][s] = p[base + 1 + s];
        substate_probability_partial[i][s] = p[base + 1 + ms + s];
      }
    }
  } else {
    auto p = (double *) prob_pinned;
    for (int ii = 0; ii < inum; ii++) {
      const int i = ilist[ii];
      const int base = ii * stride;
      enantiomeric_excess[i] = p[base];
      for (int s = 0; s < ms; s++) {
        substate_probability[i][s] = p[base + 1 + s];
        substate_probability_partial[i][s] = p[base + 1 + ms + s];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Phases 2+3 on the device (fused): the tabulated pair forces AND the CV
   back-force are computed in one kernel and added to atom->f / eng_vdwl /
   virial by fix gpu.  substate_probability_force stays resident on the device,
   so there is no device->host->device round-trip and the host Phase-3 loop is
   skipped (the base compute() gates it on this method's return value).
   Probabilities have already been forward-communicated to ghosts by the caller.
------------------------------------------------------------------------- */

bool PairTableUCGChiralityGPU::device_pair_forces(int /*eflag*/, int /*vflag*/)
{
  const int nall = atom->nlocal + atom->nghost;
  const int inum = list->inum;
  int *ilist = list->ilist;

  // kernel energy/virial codes: 0 none, 1 global, 2 per-atom
  const int ef = eflag_atom ? 2 : (eflag_either ? 1 : 0);
  const int vf = vflag_atom ? 2 : (vflag_either ? 1 : 0);

  tuc_gpu_compute_forces(ef, vf, inum, nall, ilist, substate_probability,
                         substate_probability_partial);

  return true;
}

/* ---------------------------------------------------------------------- */

double PairTableUCGChiralityGPU::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes + tuc_gpu_bytes();
}
