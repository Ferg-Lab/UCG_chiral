/***************************************************************************
                          lal_table_ucg_chirality.h
                             -------------------
                            Trung Dac Nguyen

  Class for acceleration of the table/ucg/chirality pair style.

  M1 offloaded Phase 1 (per-atom substate probabilities) to the device.
  M2 additionally offloads Phase 2 (the tabulated pair-force loop and the
  per-atom substate_probability_force reduction).  Phase 3 (the CV back-force
  loop) is still computed on the host.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : ndactrung@gmail.com
 ***************************************************************************/

#ifndef LAL_TABLE_UCG_CHIRALITY_H
#define LAL_TABLE_UCG_CHIRALITY_H

#include "lal_base_atomic.h"

namespace LAMMPS_AL {

template <class numtyp, class acctyp>
class TableUCGChirality : public BaseAtomic<numtyp, acctyp> {
 public:
  TableUCGChirality();
  ~TableUCGChirality();

  /// Clear any previous data and set up for a new LAMMPS run
  /** Returns:
    * -  0 if successful
    * - -1 if fix gpu not found
    * - -3 if there is an out of memory error
    * - -4 if the GPU library was not compiled for GPU
    * - -5 Double precision is not supported on card **/
  int init(const int ntypes, double **cutsq, double ***host_table_coeffs,
           double **host_table_data, double *host_special_lj,
           const int nlocal, const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           const double gpu_split, FILE *screen,
           int tabstyle, int ntables, int tablength,
           const int max_states, const int n_actual_types,
           const int n_total_states, const int cv_mode, const double distcut,
           int *host_nstates_per_type, int *host_actual_types_from_state,
           double *host_cv_thresholds, double *host_threshold_radii,
           double *host_prob_scale, double *host_prob_prefactor);

  /// Clear all host and device data
  void clear();

  /// Returns memory usage on device per atom
  int bytes_per_atom(const int max_nbors) const;

  /// Total host memory used by library for pair style
  double host_memory_usage() const;

  /// Phase 1: compute per-atom substate probabilities on the device and copy
  /// them back to the host (packed: [ee, p[0..ms-1], dp[0..ms-1]] per atom).
  void compute_probabilities(const int ago, const int inum_full, const int nall,
                             double **host_x, int *host_type, int *ilist,
                             int *numj, int **firstneigh, double *host_chirality,
                             int &host_start, const double cpu_time,
                             bool &success, void **prob_ptr);

  /// Phases 2+3 (fused): with ghost-filled probabilities (host
  /// substate_probability / _partial), compute the tabulated pair forces AND
  /// the CV back-force + energy + virial on the device.  The per-atom
  /// substate_probability_force stays resident on the device (consumed by the
  /// fused Phase-3 part of the kernel).  Reuses the atom data and neighbor list
  /// already cast in compute_probabilities().
  void compute_forces(const int eflag, const int vflag, const int inum,
                      const int nall, int *ilist, double **host_prob,
                      double **host_partial);

  // ------------------------- DEVICE KERNELS -------------------------

  UCL_Kernel k_prob;    ///< Phase-1 substate-probability kernel
  UCL_Kernel k_force;   ///< Phase-2 tabulated-force kernel

  // ------------------------ UCG STATE CONSTANTS ---------------------

  UCL_D_Vec<int> nstates_per_type, actual_types_from_state;
  UCL_D_Vec<numtyp> cv_thresholds, threshold_radii;
  UCL_D_Vec<numtyp> prob_scale, prob_prefactor;

  // --------------------------- TABLE DATA ---------------------------

  UCL_D_Vec<int> tabindex, nshiftbits, nmask;
  UCL_D_Vec<numtyp4> coeff2;   ///< x=innersq, y=invdelta, z=deltasq6
  UCL_D_Vec<numtyp4> coeff3;   ///< x=rsq, y=e, z=f
  UCL_D_Vec<numtyp4> coeff4;   ///< x=de/e2, y=df/f2, z=drsq (bitmap)
  UCL_D_Vec<numtyp> cutsq;     ///< per-type-pair cutoff^2 (lj_types^2)
  UCL_D_Vec<numtyp> sp_lj;     ///< special LJ values

  // --------------------------- ATOM DATA ----------------------------

  /// Per-atom chirality input (host filled, copied to device each step)
  UCL_Vector<numtyp, numtyp> chi;

  /// Packed per-atom probability output (device computed, copied to host).
  UCL_Vector<acctyp, acctyp> prob_out;

  /// Per-atom probabilities and partials for ALL atoms (local+ghost), flat
  /// [nall*max_states], pushed to the device after the host forward_comm.
  UCL_Vector<numtyp, numtyp> prob_in, partial_in;

  bool shared_types;
  int _lj_types;

  // ----------------------------- UCG STATE --------------------------

  int _max_states, _n_actual_types, _n_total_states;
  int _cv_mode;                ///< 1 = NEIGHBOR_DEPENDENT, 0 = INDEPENDENT
  numtyp _distcutsq;

  int _tabstyle, _tablength, _ntables;

  /// strides / capacities of the per-atom buffers
  int _prob_stride, _max_prob_size, _max_chi_size;
  int _max_probin_size;

 private:
  bool _allocated;

  int loop(const int eflag, const int vflag) override;
};

}

#endif
