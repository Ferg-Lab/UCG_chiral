/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(lj/smooth/linear/chiral,PairLJSmoothLinearChiral)

#else

#ifndef PAIR_LJ_SMOOTH_LINEAR_CHIRAL_H
#define PAIR_LJ_SMOOTH_LINEAR_CHIRAL_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJSmoothLinearChiral : public Pair {
 public:
  PairLJSmoothLinearChiral(class LAMMPS *);
  virtual ~PairLJSmoothLinearChiral();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void  computezeta();

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage();

 protected:
  int nmax;
  double *zeta, *dzetax, *dzetay, *dzetaz;
  double *i1_vec, *i2_vec, *i3_vec, *i4_vec;
  double cut_global;
  double **cut;
  double **epsilon,**sigma;
  double **ljcut,**dljcut;
  double **lj1,**lj2,**lj3,**lj4;
  double bias_global;
  double **bias;

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

*/
