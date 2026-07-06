/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(table/ucg/chirality/gpu,PairTableUCGChiralityGPU);
// clang-format on
#else

#ifndef LMP_PAIR_TABLE_UCG_CHIRALITY_GPU_H
#define LMP_PAIR_TABLE_UCG_CHIRALITY_GPU_H

#include "pair_table_ucg_chirality.h"

namespace LAMMPS_NS {

class PairTableUCGChiralityGPU : public PairTableUCGChirality {
 public:
  PairTableUCGChiralityGPU(LAMMPS *lmp);
  ~PairTableUCGChiralityGPU() override;
  void init_style() override;
  double memory_usage() override;

 protected:
  // Phase-1 override: compute the per-atom substate probabilities on the
  // device and copy them back into the host arrays.
  void compute_substate_prob(int cv_mode) override;

  // Phase-2 override: compute the tabulated pair forces on the device and copy
  // the per-atom substate_probability_force back to the host for Phase 3.
  bool device_pair_forces(int eflag, int vflag) override;

  bool acc_float;
  double cpu_time;
};

}    // namespace LAMMPS_NS
#endif
#endif
