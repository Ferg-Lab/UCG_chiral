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

PairStyle(table/ucg/chirality,PairTableUCGChirality)

#else

#ifndef LMP_PAIR_TABLE_UCG_CHIRALITY_H
#define LMP_PAIR_TABLE_UCG_CHIRALITY_H

#include "pair.h"

namespace LAMMPS_NS {

class PairTableUCGChirality : public Pair {
 public:
 PairTableUCGChirality(class LAMMPS *);
  virtual ~PairTableUCGChirality();

  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);
  void finish() override;

 protected:
  enum{LOOKUP,LINEAR,SPLINE,BITMAP};
  double *cut_respa;

  void calculate_substate_prob(int cv_mode);
  // Phase-1 seam: by default runs calculate_substate_prob() on the host;
  // accelerator subclasses (e.g. table/ucg/chirality/gpu) override this to
  // compute the per-atom substate probabilities on the device instead.
  virtual void compute_substate_prob(int cv_mode);

  // Phase-2 seam: by default returns false, so the base computes the tabulated
  // pair forces (and the substate_probability_force reduction) on the host.
  // Accelerator subclasses override this to compute Phase 2 on the device and
  // return true; on return substate_probability_force[][] must be filled for
  // local atoms (Phase 3 consumes it).
  virtual bool device_pair_forces(int eflag, int vflag);

  int tabstyle,tablength;
  struct Table {
    int ninput,rflag,fpflag,match,ntablebits;
    int nshiftbits,nmask;
    double rlo,rhi,fplo,fphi,cut;
    double *rfile,*efile,*ffile;
    double *e2file,*f2file;
    double innersq,delta,invdelta,deltasq6;
    double *rsq,*drsq,*e,*de,*f,*df,*e2,*f2;
  };
  int ntables;
  Table *tables;

  int **tabindex;
  // Normal PairTable functions
  void allocate();
  void read_table(Table *, char *, char *);
  void param_extract(Table *, char *);
  void bcast_table(Table *);
  void spline_table(Table *);
  void compute_table(Table *);
  void null_table(Table *);
  void free_table(Table *);
  void spline(double *, double *, int, double, double, double *);
  double splint(double *, double *, double *, int, double);

  // RLEUCG functions and variables
  double T, kT;     //  target temperature

  int n_total_states;
  int n_actual_types;
  int max_states_per_type;
  int state_params_allocated;
  int *n_states_per_type;
  int *actual_types_from_state;
  int *use_state_entropy;
  double *chemical_potentials;
  double *cv_thresholds;
  double *threshold_radii;
  double *probability_scaling_factor; // SD
  double *probability_pre_factor; // SD

  inline float fast_tanh(float x) {
    float e = expf(-2.0f * fabsf(x));
    float t = (1.0f - e) / (1.0f + e);
    return copysignf(t, x);
  }

  // Compute local chirality scaled proximity function (tanh functions)
  inline double compute_proximity_function(double distance, double threshold, double neg_chirality) { // SD
    double tanh_factor = fast_tanh((distance - threshold) / (0.1 * threshold));
    return neg_chirality * 0.5 * (1.0 - tanh_factor);
  }

  inline double compute_proximity_function_der(double distance, double threshold, double neg_chirality) { // SD
    double tanh_factor = fast_tanh((distance - threshold) / (0.1 * threshold));
    return neg_chirality * 0.5 * (1.0 - tanh_factor * tanh_factor) / (0.1 * threshold);
  }

  void read_state_settings(const char *);
  void threshold_prob_and_partial_from_cv(int, double, double&, double&);
  double **substate_probability, **substate_probability_partial, **substate_probability_force;
  double **substate_cv_backforce;
  //double *num_l_d_neighbors; // SD
  double *enantiomeric_excess; // SD

  int nmax;

  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);

  int cv_mode;
  bool using_state_entropy;
  bool using_state_chemical_potential;

  double time_substate_prob, time_substate_force, time_pair_forces;
  std::string mystyle;       // text label for style

};

}

#endif
#endif

/* ERROR/WARNING messages:
E: Pair distance < table inner cutoff
Two atoms are closer together than the pairwise table allows.
E: Pair distance > table outer cutoff
Two atoms are further apart than the pairwise table allows.
E: Illegal ... command
Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.
E: Unknown table style in pair_style command
Style of table is invalid for use with pair_style table command.
E: Illegal number of pair table entries
There must be at least 2 table entries.
E: Invalid pair table length
Length of read-in pair table is invalid
E: Invalid pair table cutoff
Cutoffs in pair_coeff command are not valid with read-in pair table.
E: Bitmapped table in file does not match requested table
Setting for bitmapped table in pair_coeff command must match table
in file exactly.
E: All pair coeffs are not set
All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.
E: Cannot open file %s
The specified file cannot be opened.  Check that the path and name are
correct. If the file is a compressed file, also check that the gzip
executable can be found and run.
E: Did not find keyword in table file
Keyword used in pair_coeff command was not found in table file.
E: Bitmapped table is incorrect length in table file
Number of table entries is not a correct power of 2.
E: Invalid keyword in pair table parameters
Keyword used in list of table parameters is not recognized.
E: Pair table parameters did not set N
List of pair table parameters must include N setting.
E: Pair table cutoffs must all be equal to use with KSpace
When using pair style table with a long-range KSpace solver, the
cutoffs for all atom type pairs must all be the same, since the
long-range solver starts at that cutoff.
*/
