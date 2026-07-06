/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(update/chirality,FixUpdateChirality);
// clang-format on
#else

#ifndef LMP_FIX_UPDATE_CHIRALITY_H
#define LMP_FIX_UPDATE_CHIRALITY_H

#include "fix.h"

namespace LAMMPS_NS {

class FixUpdateChirality : public Fix {
 public:
  FixUpdateChirality(class LAMMPS *, int, char **); // Constructor
  ~FixUpdateChirality() override; // Destructor

  int setmask() override;
  void setup(int) override;
  void pre_force(int) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  class AtomVecChiral *avec;

 protected:
  int nevery, nmax;                // to be invoked every time steps
  double cutsq;              // squared cutoff distance for chirality
  double threshold_radii, probability_scaling_factor, cv_thresholds, probability_pre_factor;
  double prob_cutoff, prob_tolerance;
  double *updated_chirality, *prob_i;  
  double compute_proximity_function(double, double); // SD
  double threshold_prob_from_cv(double); // SD
  class RanMars *random;
  int seed;
  int commflag;

  class NeighList *neigh_list;
  class PairTableUCGChirality* pair_table_ucg;

  void self_consistent_update();
  void chirality_energy_minimize();
  int update_mode, cv_mode;
  int first_nelements;
  double distcut;
  int maxiter;
};

}    // namespace LAMMPS_NS

#endif
#endif
