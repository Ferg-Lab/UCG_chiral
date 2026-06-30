/***************************************************************************
                       lal_table_ucg_chirality_ext.cpp
                             -------------------
                            Trung Dac Nguyen

  Functions for LAMMPS access to table/ucg/chirality acceleration routines.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : ndactrung@gmail.com
 ***************************************************************************/

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_table_ucg_chirality.h"

using namespace std;
using namespace LAMMPS_AL;

static TableUCGChirality<PRECISION,ACC_PRECISION> TUCMF;

// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int tuc_gpu_init(const int ntypes, double **cutsq, double ***table_coeffs,
                 double **table_data, double *special_lj, const int inum,
                 const int nall, const int max_nbors, const int maxspecial,
                 const double cell_size, int &gpu_mode, FILE *screen,
                 int tabstyle, int ntables, int tablength, const int max_states,
                 const int n_actual_types, const int n_total_states,
                 const int cv_mode, const double distcut, int *nstates_per_type,
                 int *actual_types_from_state, double *cv_thresholds,
                 double *threshold_radii, double *prob_scale,
                 double *prob_prefactor) {
  TUCMF.clear();
  gpu_mode=TUCMF.device->gpu_mode();
  double gpu_split=TUCMF.device->particle_split();
  int first_gpu=TUCMF.device->first_device();
  int last_gpu=TUCMF.device->last_device();
  int world_me=TUCMF.device->world_me();
  int gpu_rank=TUCMF.device->gpu_rank();
  int procs_per_gpu=TUCMF.device->procs_per_gpu();

  TUCMF.device->init_message(screen,"table/ucg/chirality",first_gpu,last_gpu);

  bool message=false;
  if (TUCMF.device->replica_me()==0 && screen)
    message=true;

  if (message) {
    fprintf(screen,"Initializing Device and compiling on process 0...");
    fflush(screen);
  }

  int init_ok=0;
  if (world_me==0)
    init_ok=TUCMF.init(ntypes, cutsq, table_coeffs, table_data, special_lj,
                       inum, nall, max_nbors, maxspecial, cell_size, gpu_split,
                       screen, tabstyle, ntables, tablength, max_states,
                       n_actual_types, n_total_states, cv_mode, distcut,
                       nstates_per_type, actual_types_from_state, cv_thresholds,
                       threshold_radii, prob_scale, prob_prefactor);

  TUCMF.device->world_barrier();
  if (message)
    fprintf(screen,"Done.\n");

  for (int i=0; i<procs_per_gpu; i++) {
    if (message) {
      if (last_gpu-first_gpu==0)
        fprintf(screen,"Initializing Device %d on core %d...",first_gpu,i);
      else
        fprintf(screen,"Initializing Devices %d-%d on core %d...",first_gpu,
                last_gpu,i);
      fflush(screen);
    }
    if (gpu_rank==i && world_me!=0)
      init_ok=TUCMF.init(ntypes, cutsq, table_coeffs, table_data, special_lj,
                         inum, nall, max_nbors, maxspecial, cell_size, gpu_split,
                         screen, tabstyle, ntables, tablength, max_states,
                         n_actual_types, n_total_states, cv_mode, distcut,
                         nstates_per_type, actual_types_from_state,
                         cv_thresholds, threshold_radii, prob_scale,
                         prob_prefactor);

    TUCMF.device->serialize_init();
    if (message)
      fprintf(screen,"Done.\n");
  }
  if (message)
    fprintf(screen,"\n");

  if (init_ok==0)
    TUCMF.estimate_gpu_overhead();
  return init_ok;
}

void tuc_gpu_clear() {
  TUCMF.clear();
}

// Phase 1: per-atom substate probabilities (device -> host).
void tuc_gpu_compute_prob(const int ago, const int inum_full, const int nall,
                          double **host_x, int *host_type, int *ilist,
                          int *numj, int **firstneigh, double *host_chirality,
                          int &host_start, const double cpu_time, bool &success,
                          void **prob_ptr) {
  TUCMF.compute_probabilities(ago, inum_full, nall, host_x, host_type, ilist,
                              numj, firstneigh, host_chirality, host_start,
                              cpu_time, success, prob_ptr);
}

// Phases 2+3 (fused): tabulated pair forces + CV back-force on the device.
void tuc_gpu_compute_forces(const int eflag, const int vflag, const int inum,
                            const int nall, int *ilist, double **host_prob,
                            double **host_partial) {
  TUCMF.compute_forces(eflag, vflag, inum, nall, ilist, host_prob,
                       host_partial);
}

double tuc_gpu_bytes() {
  return TUCMF.host_memory_usage();
}
