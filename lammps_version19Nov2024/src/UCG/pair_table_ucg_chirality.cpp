/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Siva Dasetty and Trung Nguyen (UChicago)
------------------------------------------------------------------------- */

#include "pair_table_ucg_chirality.h"

#include "atom.h"
#include "atom_vec_chiral.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "integrate.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "respa.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <cstdlib>

using namespace LAMMPS_NS;
using namespace MathConst;

enum {NONE, RLINEAR, RSQ, BMP};
enum {ENANTIOMER_EXCESS, SUBSTATE_PROP};
enum {NEIGHBOR_INDEPENDENT, NEIGHBOR_DEPENDENT};

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairTableUCGChirality::PairTableUCGChirality(LAMMPS *lmp) : Pair(lmp)
{
  ntables = 0;
  tables = nullptr;

  nmax = 0;

  substate_probability = nullptr;
  substate_probability_partial = nullptr;
  substate_probability_force = nullptr; // SD
  substate_cv_backforce = nullptr;
  state_params_allocated = 0;
  enantiomeric_excess = nullptr; // SD

  using_state_entropy = false;
  using_state_chemical_potential = false;

  // set max comm sizes needed by this pair

  comm_reverse = 3;
  comm_forward = 3;

  mystyle = "ucg/chirality";
  
}

/* ---------------------------------------------------------------------- */

PairTableUCGChirality::~PairTableUCGChirality()
{
  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

  memory->destroy(substate_probability);
  memory->destroy(substate_probability_partial);
  memory->destroy(substate_probability_force); // SD
  memory->destroy(substate_cv_backforce);
  memory->destroy(enantiomeric_excess); // SD

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(tabindex);
  }
  if (state_params_allocated) {
    memory->destroy(n_states_per_type);
    memory->destroy(actual_types_from_state);
    memory->destroy(use_state_entropy);
    memory->destroy(chemical_potentials);
    memory->destroy(cv_thresholds);
    memory->destroy(threshold_radii);
    memory->destroy(probability_scaling_factor); //SD
    memory->destroy(probability_pre_factor); //SD
  }
}

/* ---------------------------------------------------------------------- */

void PairTableUCGChirality::threshold_prob_and_partial_from_cv(int type, double cv, double &prob, double &partial) 
{
  if (type == 1){
    double a = probability_scaling_factor[type];
    double b = cv_thresholds[type];
    double c = probability_pre_factor[type];
   
    //printf("type: %d a: %f b: %f c: %f cv: %f \n", type, a, b, c, cv);

    // sigmoidal activation based probability function and its derivative with respect to CV.
    prob = c / (1.0 + exp( -1.0 * (a * (cv - b )) ) );
    partial = ( c * a * exp(-a*( cv - b)) )/( (1.0 + exp(-a*(cv - b)) ) * ( 1.0 + exp(-a*(cv - b)) ) );
  }
  else {
    error->one(FLERR,"Declared type does not exist.");
  }
}

/* ---------------------------------------------------------------------- */

int PairTableUCGChirality::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = substate_cv_backforce[i][0];
    buf[m++] = substate_cv_backforce[i][1];
    buf[m++] = substate_cv_backforce[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairTableUCGChirality::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    substate_cv_backforce[j][0] += buf[m++];
    substate_cv_backforce[j][1] += buf[m++];
    substate_cv_backforce[j][2] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int PairTableUCGChirality::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m,jsubstate;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = enantiomeric_excess[j]; // SD
    for (jsubstate = 0; jsubstate < max_states_per_type; jsubstate++) {
      buf[m++] = substate_probability[j][jsubstate];
      buf[m++] = substate_probability_partial[j][jsubstate];
      buf[m++] = substate_probability_force[j][jsubstate]; // SD [This should be [i][isubstate] to be consistent.
    }
  }
  
  return m;
}


/* ---------------------------------------------------------------------- */

void PairTableUCGChirality::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last,isubstate;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    enantiomeric_excess[i] = buf[m++]; // SD
    for (isubstate = 0; isubstate < max_states_per_type; isubstate++) { // SD
      substate_probability[i][isubstate] = buf[m++];
      substate_probability_partial[i][isubstate] = buf[m++];
      substate_probability_force[i][isubstate] = buf[m++]; // SD
    }
  }
}

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

void PairTableUCGChirality::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype,itable;
  int itype_actual, jtype_actual;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,factor_lj,fraction,value,a,b;
  double cv_force; 
  double substate_cv_forces; // SD
  
  if (update->ntimestep <= update->beginstep+1) {
    time_substate_prob = time_substate_force = time_pair_forces = 0.0;
  }

  int isubstate,jsubstate,ksubstate,alpha,beta;
  double alphaprob,betaprob;
  int *ilist,*jlist,*numneigh,**firstneigh;
  Table *tb;

  double pair_force; // For updating in the ev_tally_full routine
  double energy_lj; // Energy routine for ev_tally_full routine

  union_int_float_t rsq_lookup;
  int tlm1 = tablength - 1;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag; // [SD]
  double *chirality = atom->chirality;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  //#ifndef DEBUG_PROBABILITIES
  //#define DEBUG_PROBABILITIES

  // DEBUGGING OUTPUT FILES
  #ifdef DEBUG_PROBABILITIES 
  FILE* densityoutfile = fopen("ucg_debug_local_ee.out", "a");
  if (!densityoutfile) {
    char str[128];
    sprintf(str, "Cannot open file %s", densityoutfile);
  }

  FILE* steponeprobabilityoutfile = fopen("ucg_debug_forcefield_substate_part_forces.out", "a");
  if (!steponeprobabilityoutfile) {
    char str[128];
    sprintf(str, "Cannot open file %s", steponeprobabilityoutfile);
  }

  FILE* probabilityenergyoutfile = fopen("ucg_debug_remain_substate_energy_and_total.out", "a");
  if (!probabilityenergyoutfile) {
    char str[128];
    sprintf(str, "Cannot open file %s", probabilityenergyoutfile);
  }

  FILE* probabilityforceoutfile = fopen("ucg_debug_remain_substate_forces_and_total.out", "a");
  if (!probabilityforceoutfile) {
    char str[128];
    sprintf(str, "Cannot open file %s", probabilityforceoutfile);
  }
  #endif

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  #ifdef DEBUG_PROBABILITIES 
  fprintf(densityoutfile, "Number of neighbor lists or inum: %d \n", inum);
  #endif

  if (atom->nmax > nmax) {
    memory->destroy(substate_probability);
    memory->destroy(substate_probability_partial);
    memory->destroy(substate_probability_force); // SD
    memory->destroy(substate_cv_backforce);
    memory->destroy(enantiomeric_excess); // SD
    nmax = atom->nmax;
    memory->create(substate_probability, nmax, max_states_per_type , "pair:substate_probability"); // SD removed max_states_per_type - 1
    memory->create(substate_probability_partial, nmax, max_states_per_type , "pair:substate_probability_partial"); // SD see above
    memory->create(substate_probability_force, nmax, max_states_per_type , "pair:substate_probability_force"); // SD; see above
    memory->create(substate_cv_backforce, nmax, 3, "pair:substate_cv_backforce");
    memory->create(enantiomeric_excess, nmax, "pair:enantiomeric_excess"); // SD
  }

  int nall = atom->nlocal + atom->nghost;
  for(i = 0; i < nlocal; i++) {
    for (j = 0; j < max_states_per_type ; j++) { // SD
      substate_probability[i][j] = 0.0;
      substate_probability_partial[i][j] = 0.0;
      substate_probability_force[i][j] = 0.0; // SD
    }
    substate_cv_backforce[i][0] = 0.0;
    substate_cv_backforce[i][1] = 0.0;
    substate_cv_backforce[i][2] = 0.0;
    enantiomeric_excess[i] = 0.0; // SD
  }

  // time stamps
  double time0, time1, time2, time3, time4;

  // First loop: Calculate the substate probabilities of each bead i. // SD
  time0 = platform::walltime();
  calculate_substate_prob(cv_mode);
  
  time1 = platform::walltime();

  // Communicate enantiomeric_excess and state probabilities with neighboring procs
  comm->forward_comm(this);

  // Second loop: Calculate all forces that do not depend on 
  // probability derivatives and against the state distribution 
  // as well.
  time2 = platform::walltime();
  // [SD]
  // Specifically, calculate p(s_i | R) x p(s_j | R) x grad U_si_sj (R_ij)
  // Also, keep track of (1) \sum_j \sum_s_j p(s_j | R) x U_si_sj (R_ij). 
  // and determine (2) \sum_j \sum_s_j  U_si_sj x der(p(s_j | R), CV) x der(CV, dx).
  double temp_total_potenergy = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itype_actual = actual_types_from_state[itype];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // Compute single-body forces conjugate to state change.
    // Apply to each of the state probabilities except the last,
    // which is always kept implicit.
    // SD: No entropy and no chemical potentials needed for now anyway.

    if (using_state_entropy || using_state_chemical_potential) {

      if (n_states_per_type[itype_actual] > 1) {
        double i_prob_accounted = 0.0;

        // For each sub-state but the last, calculate the derivative
        // of the free energy with respect to probability.
        for (isubstate = 0; isubstate < n_states_per_type[itype_actual] - 1; isubstate++) {
          // Calculate one-body-state entropic forces.
          // SD --- wrong sign? +? [Not corrected/not used]
          if (using_state_entropy && use_state_entropy[itype_actual]) {
            substate_probability_force[i][isubstate] -= kT * log(substate_probability[i][isubstate]);
          }

          // Calculate one-body-state potential forces.
          if (using_state_chemical_potential)
            substate_probability_force[i][isubstate] -= chemical_potentials[itype + isubstate];

          i_prob_accounted += substate_probability[i][isubstate];
        }
        // For the last substate, use conservation of probability to write
        // its effect as force mediated through the other probabilities.

        if (using_state_entropy && use_state_entropy[itype_actual]) {
          for (isubstate = 0; isubstate < n_states_per_type[itype_actual] - 1; isubstate++) {
            substate_probability_force[i][isubstate] += kT * log(1 - i_prob_accounted);
          }
        }
      }
    }

    double threshold_itype = threshold_radii[itype_actual];
    int nstates_i = n_states_per_type[itype_actual];
    
    // Compute two-body forces at fixed state and effects of the
    // two body potential on state change.
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      energy_lj = 0.0;
      pair_force = 0.0;
      if (rsq < cutsq[itype][jtype]) {
        jtype_actual = actual_types_from_state[jtype];
        double distance = sqrt(rsq); // SD
        double rinv = 1.0/distance; // SD
        double threshold_jtype = threshold_radii[jtype_actual];
        int nstates_j = n_states_per_type[jtype_actual];

        // Loop over all possible substates of particle i.
        for (isubstate = 0; isubstate < nstates_i; isubstate++) {
          alpha = itype + isubstate;
          if (nstates_i > 1) {
            alphaprob = substate_probability[i][isubstate];
          } else {
            alphaprob = 1.0;
          }
          
          // Iterate over all possible substates of particle j.
          for (jsubstate = 0; jsubstate < nstates_j; jsubstate++) {
            beta = jtype + jsubstate;
            if (nstates_j > 1) {
              betaprob = substate_probability[j][jsubstate];
            } else {
              betaprob = 1.0;
            }

            double substate_prob_partial_j = substate_probability_partial[j][jsubstate];

            tb = &tables[tabindex[alpha][beta]];
            if (rsq < tb->innersq)
              error->one(FLERR,"Pair distance < table inner cutoff");
            if (tabstyle == LOOKUP) {
              itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
              if (itable >= tlm1)
                error->one(FLERR,"Pair distance > table outer cutoff");
              fpair = factor_lj * tb->f[itable];
            } else if (tabstyle == LINEAR) {
              itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
              if (itable >= tlm1)
                error->one(FLERR,"Pair distance > table outer cutoff");
              fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
              value = tb->f[itable] + fraction*tb->df[itable];
              fpair = factor_lj * value;
            } else if (tabstyle == SPLINE) {
              itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
              if (itable >= tlm1)
                error->one(FLERR,"Pair distance > table outer cutoff");
              b = (rsq - tb->rsq[itable]) * tb->invdelta;
              a = 1.0 - b;
              value = a * tb->f[itable] + b * tb->f[itable+1] +
                ((a*a*a-a)*tb->f2[itable] + (b*b*b-b)*tb->f2[itable+1]) *
                tb->deltasq6;
              fpair = factor_lj * value;
            } else {
              rsq_lookup.f = rsq;
              itable = rsq_lookup.i & tb->nmask;
              itable >>= tb->nshiftbits;
              fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
              value = tb->f[itable] + fraction*tb->df[itable];
              fpair = factor_lj * value;
            }
            // Scale the pair force with current state weights
            #ifdef DEBUG_PROBABILITIES
            fprintf(probabilityenergyoutfile, "Step: " BIGINT_FORMAT "i: %d j: %d isub: %d jsub: alphaprob: %lg betaprob: %lg force/r: %lg final fpair: %lg \n",
              update->ntimestep, tag[i], tag[j], isubstate, jsubstate, alphaprob, betaprob, fpair, fpair * alphaprob * betaprob);
            #endif

            fpair = fpair * alphaprob * betaprob; // [SD]: p(s_i | R) x p(s_j | R) x ∇U_si_sj(R_ij)
            
            // Accumulate
            // SD: Accumulate force field forces from all particles j; substate i, substate j.
            pair_force += fpair;

            // Energy calculation from the tabulated potential
            if (eflag) {
              if (tabstyle == LOOKUP)
                evdwl = tb->e[itable];
              else if (tabstyle == LINEAR || tabstyle == BITMAP)
                evdwl = tb->e[itable] + fraction*tb->de[itable];
              else
                evdwl = a * tb->e[itable] + b * tb->e[itable+1] +
                  ((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) *
                  tb->deltasq6;
              evdwl *= factor_lj;
            }

            energy_lj += evdwl * alphaprob * betaprob; // old comment? -> halved to prevent double counting energy contribution.
            temp_total_potenergy +=  evdwl * alphaprob * betaprob;

            #ifdef DEBUG_PROBABILITIES
            fprintf(probabilityenergyoutfile, "Step: " BIGINT_FORMAT "nlocal: %d j: %d tag-j: %d evdwl: %lg elj: %lg pair_force: %lg\n",
              update->ntimestep, nlocal, j, tag[j], evdwl, energy_lj, pair_force);
            #endif

            // SD -- separated from else condition [This is needed for local and ghost atoms]
            if (nstates_i > 1) {
              // SD
              substate_probability_force[i][isubstate] += betaprob * evdwl; // Sum over all j and all s_j
           
              // SD: Accumulate part of the contribution from substate cv force to force 
              substate_cv_forces = compute_proximity_function_der(distance, threshold_jtype, -1.0*chirality[i]);
              substate_cv_forces *= evdwl * substate_prob_partial_j * rinv; // * (1/num_l_d_neighbors[j]);

              pair_force += substate_cv_forces; // SD [Trung - Do we need to add this?] - Yes

              #ifdef DEBUG_PROBABILITIES
              fprintf(probabilityforceoutfile, "Step: " BIGINT_FORMAT "part: %lg substate pf: %lg subcvff: %lg f[0]: %lg f[1]: %lg f[2]: %lg \n",
                update->ntimestep, betaprob * evdwl, substate_probability_force[i][isubstate], substate_cv_forces, f[i][0], f[i][1], f[i][2]);
              #endif
            }
            
          } // end of jsubstate
        } // end of isubstate

        #ifdef DEBUG_PROBABILITIES
        fprintf(probabilityforceoutfile, "Step: " BIGINT_FORMAT "\t nlocal: %d \t final scaled energy: %lg \t final energy_lj: %lg \t pf: %lg\n",
          update->ntimestep ,nlocal, evdwl, energy_lj, pair_force);
        #endif

        f[i][0] += pair_force * delx;
        f[i][1] += pair_force * dely;
        f[i][2] += pair_force * delz;
        if (newton_pair) {
          f[j][0] -= pair_force * delx;
          f[j][1] -= pair_force * dely;
          f[j][2] -= pair_force * delz;  
        }

        //printf("Energy i %d energy_lj %f \n", tag[i], energy_lj);
        if (evflag) ev_tally_full(i, energy_lj, 0.0, pair_force, delx, dely, delz); // SD

      } // j within cutsq

    } // end of jj
  } // end of ii

  //printf("Energy %f\n", temp_total_potenergy);

  time3 = platform::walltime();

  // Third loop: Calculate forces from probability derivatives on local atoms.
  // substate_cv_backforce on local atoms need to be accumulated from neighboring procs using reverse comm <-- this actually is not correct
  //   only need reverse comm if ghost atoms' substate_cv_backforce are computed

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itype_actual = actual_types_from_state[itype];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    double threshold_itype = threshold_radii[itype_actual];
    int nstates_i = n_states_per_type[itype_actual];

    // Apply the force against the CV, in this case density.
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK; // [SD]
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      
      // Distribute the force down to every pair of particles
      // contributing to the density.
      if (rsq < cutsq[itype][jtype]) {

        double distance = sqrt(rsq);
        double fx = 0;
        double fy = 0;
        double fz = 0;
        #pragma unroll // Added 03/11/2026 - Merge edits from gitlab
        for (isubstate = 0; isubstate < nstates_i; isubstate++) { // SD removed -1

          // Convert force w.r.t state to force w.r.t. CV
          // by using the partial of state w.r.t. CV.
          cv_force = substate_probability_force[i][isubstate] * substate_probability_partial[i][isubstate];
          fpair = cv_force * compute_proximity_function_der(distance, threshold_itype, -1.0*chirality[j]); // SD
          fpair /= distance; // SD

          fx += fpair * delx;
          fy += fpair * dely;
          fz += fpair * delz;
          
          #ifdef DEBUG_PROBABILITIES
            fprintf(probabilityforceoutfile, "Step: " BIGINT_FORMAT "remaining fpair: %lg\n", update->ntimestep , fpair); 
          #endif

          // [SD] OLD CODE: This is not required for full neighbor list.
          //fpair = cv_force * compute_proximity_function_der(jtype_actual, distance, -1.0*chirality[i]); // SD
          //fpair /= distance; // SD
          //fpair /= num_l_d_neighbors[j]; //SD
          //pair_force += fpair; // SD [double counting in ev_tally?]
          //substate_cv_backforce[j][0] -= fpair * delx;
          //substate_cv_backforce[j][1] -= fpair * dely;
          //substate_cv_backforce[j][2] -= fpair * delz;

          ev_tally_full(i, 0.0, 0.0, fpair, delx, dely, delz); // SD

        } // end of isubstate

        substate_cv_backforce[i][0] += fx;
        substate_cv_backforce[i][1] += fy;
        substate_cv_backforce[i][2] += fz;

      } // end if rsq < cutsq
    } // end of jj
  } // end of ii

  time4 = platform::walltime();

  // accumulate substate_cv_backforce from neighboring procs into local atoms:
  // this comm is not needed because only local atoms' substate_cv_backforce are computed
  //comm->reverse_comm(this);

  // Add the CV forces to other forces.
  for (i = 0; i < nlocal; i++) {  
    f[i][0] += substate_cv_backforce[i][0];
    f[i][1] += substate_cv_backforce[i][1];
    f[i][2] += substate_cv_backforce[i][2];
  }

  if (vflag_fdotr) virial_fdotr_compute();

  time_substate_prob  += time1 - time0;
  time_substate_force += time3 - time2;
  time_pair_forces    += time4 - time3;

  #ifdef DEBUG_PROBABILITIES
  fclose(densityoutfile);
  fclose(steponeprobabilityoutfile);
  fclose(probabilityenergyoutfile);
  fclose(probabilityforceoutfile);
  #endif 
}

/* ----------------------------------------------------------------------
   calculate substate probabilities
------------------------------------------------------------------------- */
void PairTableUCGChirality::calculate_substate_prob(int cv_mode)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  int itype_actual, jtype_actual;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq;
  //factor_lj,fraction,value,a,b;
  double distance; 
  double substate_cv_forces; // SD
  
  int isubstate,jsubstate,ksubstate,alpha,beta;
  double alphaprob,betaprob;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  int *type = atom->type;
  tagint *tag = atom->tag; // [SD]
  double *chirality = atom->chirality;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double distcut = 4.0;                  // acquiring this value from fix update/chiarlity
  double distcut_sq = distcut*distcut;

  // First loop: Calculate the substate probabilities of each bead i. // SD
  if (cv_mode == NEIGHBOR_INDEPENDENT) {
    
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      itype_actual = actual_types_from_state[itype]; // (Notes) SD: state ids to actual type ids of the particle (for parameters?)
      jlist = firstneigh[i];
      jnum = numneigh[i];

      //printf("Chirality values: %d \t %f\n", tag[i], chirality[i]);
      #ifdef DEBUG_PROBABILITIES 
      fprintf(densityoutfile, "Number of neighbors of i: %d or tag i: %d are jnum: %d \n", jnum);
      #endif

      if (n_states_per_type[itype_actual] > 1) {

        // SD -- added this loop to calculate all substate probabilities. However, this is just for
        // keeping it consistent with rest of the code in next two stages. Note: The code only supports 2 substate systems.
        for (isubstate = 0; isubstate < n_states_per_type[itype_actual]; isubstate++) {
          
          enantiomeric_excess[i] = chirality[i];
          if (isubstate == 0) {
            substate_probability[i][isubstate] = enantiomeric_excess[i];
            substate_probability_partial[i][isubstate] = 0.0;
          }
          else {
            substate_probability[i][isubstate] = 1 - substate_probability[i][0];
            substate_probability_partial[i][isubstate] = 0.0; //-0.0;
          }
        }
      } else {
        // For types without substates, simply assign p0 = 1.
        // (No partial derivatives.)
        substate_probability[i][0] = 1.0;
      }

      // SD
      #ifdef DEBUG_PROBABILITIES
      for (j = 0; j < max_states_per_type; j++) {
      fprintf(steponeprobabilityoutfile, "step: "BIGINT_FORMAT "i:%d tag-i: %d isubstate:%d substate_probability[i][j]:%f partial:%f \n",
        update->ntimestep, i, tag[i], j, substate_probability[i][j], substate_probability_partial[i][j]);
      }
      #endif
    }

  } else if (cv_mode == NEIGHBOR_DEPENDENT) {

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      itype_actual = actual_types_from_state[itype]; // (Notes) SD: state ids to actual type ids of the particle (for parameters?)
      jlist = firstneigh[i];
      jnum = numneigh[i];

      //printf("Chirality values: %d \t %f\n", tag[i], chirality[i]);
      #ifdef DEBUG_PROBABILITIES 
      fprintf(densityoutfile, "Number of neighbors of i: %d or tag i: %d are jnum: %d \n", jnum);
      #endif

      if (n_states_per_type[itype_actual] > 1) {

        // SD -- added this loop to calculate all substate probabilities. However, this is just for
        // keeping it consistent with rest of the code in next two stages. Note: The code only supports 2 substate systems.
        for (isubstate = 0; isubstate < n_states_per_type[itype_actual]; isubstate++) {

          // For each particle with states, calculate the CV that 
          // we use to measure state probability (currently, local enantiomeric excess) [SD]
          int local_neigh = 0;
          for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK; // [SD]
            jtype = type[j];
            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;

            // Trung: This if check is to consider only neighbors that are within the cutoff distance,
            //        because there are neighbors that are between [cutoff; cutoff+skin] distance.
            if (rsq < distcut_sq){ // cutsq[itype][jtype])  // May be replace with custom cutoff and not pair cutoff.
              distance = sqrt(rsq);

              if (isubstate == 0) {
                local_neigh += 1;
                enantiomeric_excess[i] += compute_proximity_function(itype_actual, distance, chirality[j]);
              }

              #ifdef DEBUG_PROBABILITIES
                // SD: log number densities for debugging.
                fprintf(densityoutfile, BIGINT_FORMAT " Step: %d tag-i: %d tag-j: %d itype: %d jtype: %d itype actual: %d isubstate:%d pf: %lg ee: %lg \n",
                  update->ntimestep, tag[i], tag[j], itype, jtype, itype_actual, isubstate, compute_proximity_function(itype_actual, distance, -1.0*chirality[j]), enantiomeric_excess[i]);
              #endif
            }
          }
          if (isubstate == 0) {
            threshold_prob_and_partial_from_cv(itype_actual, enantiomeric_excess[i]/local_neigh, substate_probability[i][isubstate], substate_probability_partial[i][isubstate]);// [SD]
          }
          else {
            // Probability of second substate is 1 - probability of first substate
            substate_probability[i][isubstate] = 1 - substate_probability[i][0];
            substate_probability_partial[i][isubstate] = -substate_probability_partial[i][0];
          }
          
        }
      } else {
        // For types without substates, simply assign p0 = 1.
        // (No partial derivatives.)
          substate_probability[i][0] = 1.0;
        }

      // SD
      #ifdef DEBUG_PROBABILITIES
      for (j = 0; j < max_states_per_type; j++) {
      fprintf(steponeprobabilityoutfile, "step: "BIGINT_FORMAT "i:%d tag-i: %d isubstate:%d substate_probability[i][j]:%f partial:%f \n",
        update->ntimestep, i, tag[i], j, substate_probability[i][j], substate_probability_partial[i][j]);
      }
      #endif
    }

  }
  
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairTableUCGChirality::allocate()
{
  allocated = 1;
  const int nt = atom->ntypes + 1;

  memory->create(setflag,nt,nt,"pair:setflag");
  memory->create(cutsq,nt,nt,"pair:cutsq");
  memory->create(tabindex,nt,nt,"pair:tabindex");

  memset(&setflag[0][0],0,nt*nt*sizeof(int));
  memset(&cutsq[0][0],0,nt*nt*sizeof(double));
  memset(&tabindex[0][0],0,nt*nt*sizeof(int));
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTableUCGChirality::settings(int narg, char **arg)
{
  if (narg < 2) error->all(FLERR,"Illegal pair_style command");

  // Tabulated potential settings

  if (strcmp(arg[0],"lookup") == 0) tabstyle = LOOKUP;
  else if (strcmp(arg[0],"linear") == 0) tabstyle = LINEAR;
  else if (strcmp(arg[0],"spline") == 0) tabstyle = SPLINE;
  else if (strcmp(arg[0],"bitmap") == 0) tabstyle = BITMAP;
  else error->all(FLERR,"Unknown table style in pair_style command");

  tablength = utils::numeric(FLERR, arg[1], false, lmp);
  if (tablength < 2) error->all(FLERR,"Illegal number of pair table entries");

  // optional keywords
  // assert the tabulation is compatible with a specific long-range solver

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"ewald") == 0) ewaldflag = 1;
    else if (strcmp(arg[iarg],"pppm") == 0) pppmflag = 1;
    else if (strcmp(arg[iarg],"msm") == 0) msmflag = 1;
    else if (strcmp(arg[iarg],"dispersion") == 0) dispersionflag = 1;
    else if (strcmp(arg[iarg],"tip4p") == 0) tip4pflag = 1;
    else error->all(FLERR,"Illegal pair_style command");
    iarg++;
  }

  // Read in a state definition file
  read_state_settings(arg[2]);

  // delete old tables, since cannot just change settings

  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);
  
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(tabindex);
  }
  allocated = 0;

  ntables = 0;
  tables = NULL;
}

void PairTableUCGChirality::read_state_settings(const char *file) {
  char *eof;
  char line[MAXLINE];
  char state_type[MAXLINE];
  char entropy_spec[MAXLINE];

  // Open the state settings file.
  FILE* fp = fopen(file, "r");
  if (fp == NULL) {
    char str[128];
    sprintf(str, "Cannot open file %s", file);
    error->one(FLERR, str);
  }

  // Read the total number of actual types and total number of states.
  eof = fgets(line, MAXLINE, fp);
  if (eof == NULL) error->one(FLERR,"Unexpected end of RLEUCG state settings file");
  sscanf(line,"%d %d", &n_actual_types, &n_total_states);

  // Allocate space for storing state settings based on the number
  // of actual types.
  memory->create(n_states_per_type, n_actual_types + 1, "pair:n_states_per_type");
  memory->create(actual_types_from_state, n_total_states + 1, "pair:n_states_per_type");
  memory->create(use_state_entropy, n_actual_types + 1, "pair:n_states_per_type");
  memory->create(chemical_potentials, n_total_states + 1, "pair:n_states_per_type");
  memory->create(cv_thresholds, n_actual_types + 1, "pair:n_states_per_type");
  memory->create(threshold_radii, n_actual_types + 1, "pair:n_states_per_type");
  memory->create(probability_scaling_factor, n_actual_types + 1, "pair:n_states_per_type");
  memory->create(probability_pre_factor, n_actual_types + 1, "pair:n_states_per_type");
  
  state_params_allocated = 1;

  for (int i = 0; i <= n_total_states; i++) {
    chemical_potentials[i] = 0.0;
    actual_types_from_state[i] = 0;
  }
  for (int i = 0; i <= n_actual_types; i++) {
    n_states_per_type[i] = 0;
    use_state_entropy[i] = 0;
    cv_thresholds[i] = 0.0;
    threshold_radii[i] = 0.0;
    probability_scaling_factor[i] = 1.0;
    probability_pre_factor[i] = 0.0;
  }

  // For each actual atom type, read the number of states for that type, and
  // (if more than one) the density threshold, the threshold radius for
  // the density, and the one-body chemical potential for the state.
  int curr_state = 1;
  max_states_per_type = 2;
  for (int i = 1; i <= n_actual_types; i++) {
    // Read the number of states and way that they are assigned.
    eof = fgets(line, MAXLINE, fp);
    if (eof == NULL) error->one(FLERR,"Unexpected end of RLEUCG state settings file");
    sscanf(line, "%d %s %s", &n_states_per_type[i], state_type, entropy_spec);
    max_states_per_type = std::max(max_states_per_type, n_states_per_type[i]);
    if (strcmp(entropy_spec, "use_entropy") == 0) {
      use_state_entropy[i] = 1;
    } else if (strcmp(entropy_spec, "no_entropy") == 0) {
      use_state_entropy[i] = 0;
    }

    // If this type has more than one state, read further state parameters.
    if (n_states_per_type[i] > 1) {
      // Read state probability assignment parameters.
      if (strcmp(state_type, "density") == 0) {
        eof = fgets(line, MAXLINE, fp);
        if (eof == NULL) error->one(FLERR,"Unexpected end of RLEUCG state settings file");
        sscanf(line, "%lg %lg", &cv_thresholds[i], &threshold_radii[i]);
        // Read additional parameters for probability function; // SD
        eof = fgets(line, MAXLINE, fp);
        if (eof == NULL) error->one(FLERR,"Unexpected end of RLEUCG state settings file when reading additional parameters");
        sscanf(line, "%lg %lg", &probability_scaling_factor[i], &probability_pre_factor[i]);
      } else {
        error->one(FLERR,"Unknown state assignment type for RLEUCG");
      }

      // Read state chemical potentials.
      eof = fgets(line, MAXLINE, fp);
      if (eof == NULL) error->one(FLERR,"Unexpected end of RLEUCG state settings file");
      char *p = strtok(line, " ");
      for (int j = 0; j < n_states_per_type[i] - 1; j++) {
        sscanf(p, "%lg", &chemical_potentials[i + j]);
        p = strtok(NULL, " ");
      }

      // Read CV mode (INDEPENDENT or DEPENDENT)
      eof = fgets(line, MAXLINE, fp);
      if (eof == NULL) error->one(FLERR,"Unexpected end of RLEUCG state settings file");
      int cv_mode_value;
      sscanf(line, "%d", &cv_mode_value);
      if (cv_mode_value == 1) {
        cv_mode = NEIGHBOR_INDEPENDENT;
      } else if (cv_mode_value == 0) {
        cv_mode = NEIGHBOR_DEPENDENT;
      }
    }

    if (comm->me == 0) { // Added 03/11/2026 - Merge edits from gitlab
      printf("\n=====================================================================\n");
      printf("\n\nProbability scaling factor, entropy spec, and chemical potentials\n");
      printf("cv_thresholds[0]: %lg \t cv_thresholds[1]: %lg \n", cv_thresholds[0], cv_thresholds[1]);
      printf("threshold_radii[0]: %lg \t threshold_radii[1] %g \n", threshold_radii[0], threshold_radii[1]);
      printf("probability_scaling_factor[0]: %g \t probability_scaling_factor[1]: %lg \n", probability_scaling_factor[0], probability_scaling_factor[1]);
      printf("probability_pre_factor[0]: %lg \t probability_pre_factor[1]: %lg \n", probability_pre_factor[0], probability_pre_factor[1]);
      printf("use_state_entropy[0]: %d \t use_state_entropy[1]: %d \n", use_state_entropy[0], use_state_entropy[1]);
      printf("chemical_potentials[0]: %lg \t chemical_potentials[1]: %lg \t chemical_potentials[2]: %lg \n\n", chemical_potentials[0], chemical_potentials[1], chemical_potentials[2]);  
      printf("cv mode value %d and cv mode: %d\n",NEIGHBOR_INDEPENDENT,cv_mode);
      printf("\n=====================================================================\n");
    }

    // Keep an up-to-date back-map from state ids to actual type ids.
    for (int j = 0; j < n_states_per_type[i]; j++) {
      actual_types_from_state[curr_state] = i;
      curr_state++;
    }
  }
  //comm_forward = 3 * (max_states_per_type - 1);
  comm_forward = 2 + 3 * (max_states_per_type) ; // SD

  // Close after finishing.
  fclose(fp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTableUCGChirality::coeff(int narg, char **arg)
{
  if (narg != 4 && narg != 5) error->all(FLERR,"Illegal pair_coeff command");
  if (!allocated) allocate();
  
  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  int me;
  MPI_Comm_rank(world,&me);
  tables = (Table *)
    memory->srealloc(tables,(ntables+1)*sizeof(Table),"pair:tables");
  Table *tb = &tables[ntables];
  null_table(tb);
  if (me == 0) read_table(tb,arg[2],arg[3]);
  bcast_table(tb);

  // set table cutoff

  if (narg == 5) tb->cut = utils::numeric(FLERR, arg[4], false, lmp);
  else if (tb->rflag) tb->cut = tb->rhi;
  else tb->cut = tb->rfile[tb->ninput-1];

  // error check on table parameters
  // insure cutoff is within table
  // for BITMAP tables, file values can be in non-ascending order

  if (tb->ninput <= 1) error->one(FLERR,"Invalid pair table length");
  double rlo,rhi;
  if (tb->rflag == 0) {
    rlo = tb->rfile[0];
    rhi = tb->rfile[tb->ninput-1];
  } else {
    rlo = tb->rlo;
    rhi = tb->rhi;
  }
  if (tb->cut <= rlo || tb->cut > rhi)
    error->all(FLERR,"Invalid pair table cutoff");
  if (rlo <= 0.0) error->all(FLERR,"Invalid pair table cutoff");

  // match = 1 if don't need to spline read-in tables
  // this is only the case if r values needed by final tables
  //   exactly match r values read from file
  // for tabstyle SPLINE, always need to build spline tables

  tb->match = 0;
  if (tabstyle == LINEAR && tb->ninput == tablength &&
      tb->rflag == RSQ && tb->rhi == tb->cut) tb->match = 1;
  if (tabstyle == BITMAP && tb->ninput == 1 << tablength &&
      tb->rflag == BMP && tb->rhi == tb->cut) tb->match = 1;
  if (tb->rflag == BMP && tb->match == 0)
    error->all(FLERR,"Bitmapped table in file does not match requested table");

  // spline read-in values and compute r,e,f vectors within table
  
  if (tb->match == 0) spline_table(tb);

  compute_table(tb);

  // store ptr to table in tabindex

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      tabindex[i][j] = ntables;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Illegal pair_coeff command");
  ntables++;
}

/* ----------------------------------------------------------------------
   init style needed to declare the full neighborlist and kT term (one-body) on
------------------------------------------------------------------------- */

void PairTableUCGChirality::init_style()
{
  if (atom->chirality_flag != 1)
    error->all(FLERR, "Pair table/ucg/chirality requires atom style chirality");

  // request regular or rRESPA neighbor lists

  int list_style = NeighConst::REQ_FULL; //FULL; //DEFAULT;  [SD]

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style, "^respa")) {
    auto respa = dynamic_cast<Respa *>(update->integrate);
    if (respa->level_inner >= 0) list_style = NeighConst::REQ_RESPA_INOUT;
    if (respa->level_middle >= 0) list_style = NeighConst::REQ_RESPA_ALL;
  }
  neighbor->add_request(this, list_style);

  // set rRESPA cutoffs

  if (utils::strmatch(update->integrate_style, "^respa") &&
      (dynamic_cast<Respa *>(update->integrate))->level_inner >= 0)
    cut_respa = (dynamic_cast<Respa *>(update->integrate))->cutoff;
  else
    cut_respa = nullptr;

  /*---YP--- Obtain thermostat temperature */

  double *pT = NULL;
  int pdim;

  for(int ifix = 0; ifix < modify->nfix; ifix++)
  {
    pT = (double*) modify->fix[ifix]->extract("t_target", pdim);
    if(pT) { T = (*pT); break; }
  }
  kT = force->boltz * T;

  if (force->newton_pair != 0)
    error->all(FLERR, "Newton pair is turned on. It has to be turned off in local density UCG simulation.");

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTableUCGChirality::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  tabindex[j][i] = tabindex[i][j];

  return tables[tabindex[i][j]].cut;
}

/* ----------------------------------------------------------------------
   read a table section from a tabulated potential file
   only called by proc 0
   this function sets these values in Table:
     ninput,rfile,efile,ffile,rflag,rlo,rhi,fpflag,fplo,fphi,ntablebits
------------------------------------------------------------------------- */

void PairTableUCGChirality::read_table(Table *tb, char *file, char *keyword)
{
  char line[MAXLINE];

  // open file

  FILE *fp = utils::open_potential(file, lmp, nullptr);
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open file %s",file);
    error->one(FLERR,str);
  }

  // loop until section found with matching keyword

  while (1) {
    if (fgets(line,MAXLINE,fp) == NULL)
      error->one(FLERR,"Did not find keyword in table file");
    if (strspn(line," \t\n\r") == strlen(line)) continue;  // blank line
    if (line[0] == '#') continue;                          // comment
    char *word = strtok(line," \t\n\r");
    if (strcmp(word,keyword) == 0) break;           // matching keyword
    fgets(line,MAXLINE,fp);                         // no match, skip section
    param_extract(tb,line);
    fgets(line,MAXLINE,fp);
    for (int i = 0; i < tb->ninput; i++) fgets(line,MAXLINE,fp);
  }

  // read args on 2nd line of section
  // allocate table arrays for file values

  fgets(line,MAXLINE,fp);
  param_extract(tb,line);
  memory->create(tb->rfile,tb->ninput,"pair:rfile");
  memory->create(tb->efile,tb->ninput,"pair:efile");
  memory->create(tb->ffile,tb->ninput,"pair:ffile");

  // setup bitmap parameters for table to read in

  tb->ntablebits = 0;
  int masklo,maskhi,nmask,nshiftbits;
  if (tb->rflag == BMP) {
    while (1 << tb->ntablebits < tb->ninput) tb->ntablebits++;
    if (1 << tb->ntablebits != tb->ninput)
      error->one(FLERR,"Bitmapped table is incorrect length in table file");
    init_bitmap(tb->rlo,tb->rhi,tb->ntablebits,masklo,maskhi,nmask,nshiftbits);
  }

  // read r,e,f table values from file
  // if rflag set, compute r
  // if rflag not set, use r from file

  int itmp;
  double rtmp;
  union_int_float_t rsq_lookup;

  fgets(line,MAXLINE,fp);
  for (int i = 0; i < tb->ninput; i++) {
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d %lg %lg %lg",&itmp,&rtmp,&tb->efile[i],&tb->ffile[i]);

    if (tb->rflag == RLINEAR)
      rtmp = tb->rlo + (tb->rhi - tb->rlo)*i/(tb->ninput-1);
    else if (tb->rflag == RSQ) {
      rtmp = tb->rlo*tb->rlo +
        (tb->rhi*tb->rhi - tb->rlo*tb->rlo)*i/(tb->ninput-1);
      rtmp = sqrt(rtmp);
    } else if (tb->rflag == BMP) {
      rsq_lookup.i = i << nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->rlo*tb->rlo) {
        rsq_lookup.i = i << nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      rtmp = sqrtf(rsq_lookup.f);
    }

    tb->rfile[i] = rtmp;
  }

  // close file
  fclose(fp);
}

/* ----------------------------------------------------------------------
   broadcast read-in table info from proc 0 to other procs
   this function communicates these values in Table:
     ninput,rfile,efile,ffile,rflag,rlo,rhi,fpflag,fplo,fphi
------------------------------------------------------------------------- */

void PairTableUCGChirality::bcast_table(Table *tb)
{
  MPI_Bcast(&tb->ninput,1,MPI_INT,0,world);

  int me;
  MPI_Comm_rank(world,&me);
  if (me > 0) {
    memory->create(tb->rfile,tb->ninput,"pair:rfile");
    memory->create(tb->efile,tb->ninput,"pair:efile");
    memory->create(tb->ffile,tb->ninput,"pair:ffile");
  }

  MPI_Bcast(tb->rfile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->efile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->ffile,tb->ninput,MPI_DOUBLE,0,world);

  MPI_Bcast(&tb->rflag,1,MPI_INT,0,world);
  if (tb->rflag) {
    MPI_Bcast(&tb->rlo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->rhi,1,MPI_DOUBLE,0,world);
  }
  MPI_Bcast(&tb->fpflag,1,MPI_INT,0,world);
  if (tb->fpflag) {
    MPI_Bcast(&tb->fplo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->fphi,1,MPI_DOUBLE,0,world);
  }
}

/* ----------------------------------------------------------------------
   build spline representation of e,f over entire range of read-in table
   this function sets these values in Table: e2file,f2file
------------------------------------------------------------------------- */

void PairTableUCGChirality::spline_table(Table *tb)
{
  memory->create(tb->e2file,tb->ninput,"pair:e2file");
  memory->create(tb->f2file,tb->ninput,"pair:f2file");

  double ep0 = - tb->ffile[0];
  double epn = - tb->ffile[tb->ninput-1];
  spline(tb->rfile,tb->efile,tb->ninput,ep0,epn,tb->e2file);

  if (tb->fpflag == 0) {
    tb->fplo = (tb->ffile[1] - tb->ffile[0]) / (tb->rfile[1] - tb->rfile[0]);
    tb->fphi = (tb->ffile[tb->ninput-1] - tb->ffile[tb->ninput-2]) /
      (tb->rfile[tb->ninput-1] - tb->rfile[tb->ninput-2]);
  }
  
  double fp0 = tb->fplo;
  double fpn = tb->fphi;
  spline(tb->rfile,tb->ffile,tb->ninput,fp0,fpn,tb->f2file);
}

/* ----------------------------------------------------------------------
   extract attributes from parameter line in table section
   format of line: N value R/RSQ/BITMAP lo hi FP fplo fphi
   N is required, other params are optional
------------------------------------------------------------------------- */

void PairTableUCGChirality::param_extract(Table *tb, char *line)
{
  tb->ninput = 0;
  tb->rflag = NONE;
  tb->fpflag = 0;

  char *word = strtok(line," \t\n\r\f");
  while (word) {
    if (strcmp(word,"N") == 0) {
      word = strtok(NULL," \t\n\r\f");
      tb->ninput = atoi(word);
    } else if (strcmp(word,"R") == 0 || strcmp(word,"RSQ") == 0 ||
               strcmp(word,"BITMAP") == 0) {
      if (strcmp(word,"R") == 0) tb->rflag = RLINEAR;
      else if (strcmp(word,"RSQ") == 0) tb->rflag = RSQ;
      else if (strcmp(word,"BITMAP") == 0) tb->rflag = BMP;
      word = strtok(NULL," \t\n\r\f");
      tb->rlo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->rhi = atof(word);
    } else if (strcmp(word,"FP") == 0) {
      tb->fpflag = 1;
      word = strtok(NULL," \t\n\r\f");
      tb->fplo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->fphi = atof(word);
    } else {
      printf("WORD: %s\n",word);
      error->one(FLERR,"Invalid keyword in pair table parameters");
    }
    word = strtok(NULL," \t\n\r\f");
  }

  if (tb->ninput == 0) error->one(FLERR,"Pair table parameters did not set N");
}

/* ----------------------------------------------------------------------
   compute r,e,f vectors from splined values
------------------------------------------------------------------------- */

void PairTableUCGChirality::compute_table(Table *tb)
{

  int tlm1 = tablength-1;

  // inner = inner table bound
  // cut = outer table bound
  // delta = table spacing in rsq for N-1 bins

  double inner;
  if (tb->rflag) inner = tb->rlo;
  else inner = tb->rfile[0];
  tb->innersq = inner*inner;
  tb->delta = (tb->cut*tb->cut - tb->innersq) / tlm1;
  tb->invdelta = 1.0/tb->delta;

  // direct lookup tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // e,f = value at midpt of bin
  // e,f are N-1 in length since store 1 value at bin midpt
  // f is converted to f/r when stored in f[i]
  // e,f are never a match to read-in values, always computed via spline interp

  if (tabstyle == LOOKUP) {
    memory->create(tb->e,tlm1,"pair:e");
    memory->create(tb->f,tlm1,"pair:f");

    double r,rsq;
    for (int i = 0; i < tlm1; i++) {
      rsq = tb->innersq + (i+0.5)*tb->delta;
      r = sqrt(rsq);
      tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
      tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
    }
  }

  // linear tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // de,df values = delta from lower edge to upper edge of bin
  // rsq,e,f are N in length so de,df arrays can compute difference
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == LINEAR) {
    memory->create(tb->rsq,tablength,"pair:rsq");
    memory->create(tb->e,tablength,"pair:e");
    memory->create(tb->f,tablength,"pair:f");
    memory->create(tb->de,tlm1,"pair:de");
    memory->create(tb->df,tlm1,"pair:df");

    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i]/r;
      } else {
        tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
    }

    for (int i = 0; i < tlm1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
    }
  }

  // cubic spline tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // e2,f2 = spline coefficient for each bin
  // rsq,e,f,e2,f2 are N in length so have N-1 spline bins
  // f is converted to f/r after e is splined
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == SPLINE) {
    memory->create(tb->rsq,tablength,"pair:rsq");
    memory->create(tb->e,tablength,"pair:e");
    memory->create(tb->f,tablength,"pair:f");
    memory->create(tb->e2,tablength,"pair:e2");
    memory->create(tb->f2,tablength,"pair:f2");

    tb->deltasq6 = tb->delta*tb->delta / 6.0;

    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i]/r;
      } else {
        tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r);
      }
    }

    // ep0,epn = dh/dg at inner and at cut
    // h(r) = e(r) and g(r) = r^2
    // dh/dg = (de/dr) / 2r = -f/2r

    double ep0 = - tb->f[0] / (2.0 * sqrt(tb->innersq));
    double epn = - tb->f[tlm1] / (2.0 * tb->cut);
    spline(tb->rsq,tb->e,tablength,ep0,epn,tb->e2);

    // fp0,fpn = dh/dg at inner and at cut
    // h(r) = f(r)/r and g(r) = r^2
    // dh/dg = (1/r df/dr - f/r^2) / 2r
    // dh/dg in secant approx = (f(r2)/r2 - f(r1)/r1) / (g(r2) - g(r1))

    double fp0,fpn;
    double secant_factor = 0.1;
    if (tb->fpflag) fp0 = (tb->fplo/sqrt(tb->innersq) - tb->f[0]/tb->innersq) /
      (2.0 * sqrt(tb->innersq));
    else {
      double rsq1 = tb->innersq;
      double rsq2 = rsq1 + secant_factor*tb->delta;
      fp0 = (splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,sqrt(rsq2)) /
             sqrt(rsq2) - tb->f[0] / sqrt(rsq1)) / (secant_factor*tb->delta);
    }

    if (tb->fpflag && tb->cut == tb->rfile[tb->ninput-1]) fpn =
      (tb->fphi/tb->cut - tb->f[tlm1]/(tb->cut*tb->cut)) / (2.0 * tb->cut);
    else {
      double rsq2 = tb->cut * tb->cut;
      double rsq1 = rsq2 - secant_factor*tb->delta;
      fpn = (tb->f[tlm1] / sqrt(rsq2) -
             splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,sqrt(rsq1)) /
             sqrt(rsq1)) / (secant_factor*tb->delta);
    }

    for (int i = 0; i < tablength; i++) tb->f[i] /= sqrt(tb->rsq[i]);
    spline(tb->rsq,tb->f,tablength,fp0,fpn,tb->f2);
  }

  // bitmapped linear tables
  // 2^N bins from inner to cut, spaced in bitmapped manner
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == BITMAP) {
    double r;
    union_int_float_t rsq_lookup;
    int masklo,maskhi;

    // linear lookup tables of length ntable = 2^n
    // stored value = value at lower edge of bin

    init_bitmap(inner,tb->cut,tablength,masklo,maskhi,tb->nmask,tb->nshiftbits);
    int ntable = 1 << tablength;
    int ntablem1 = ntable - 1;

    memory->create(tb->rsq,ntable,"pair:rsq");
    memory->create(tb->e,ntable,"pair:e");
    memory->create(tb->f,ntable,"pair:f");
    memory->create(tb->de,ntable,"pair:de");
    memory->create(tb->df,ntable,"pair:df");
    memory->create(tb->drsq,ntable,"pair:drsq");

    union_int_float_t minrsq_lookup;
    minrsq_lookup.i = 0 << tb->nshiftbits;
    minrsq_lookup.i |= maskhi;

    for (int i = 0; i < ntable; i++) {
      rsq_lookup.i = i << tb->nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->innersq) {
        rsq_lookup.i = i << tb->nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      r = sqrtf(rsq_lookup.f);
      tb->rsq[i] = rsq_lookup.f;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i]/r;
      } else {
        tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
      minrsq_lookup.f = MIN(minrsq_lookup.f,rsq_lookup.f);
    }

    tb->innersq = minrsq_lookup.f;

    for (int i = 0; i < ntablem1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
      tb->drsq[i] = 1.0/(tb->rsq[i+1] - tb->rsq[i]);
    }

    // get the delta values for the last table entries
    // tables are connected periodically between 0 and ntablem1

    tb->de[ntablem1] = tb->e[0] - tb->e[ntablem1];
    tb->df[ntablem1] = tb->f[0] - tb->f[ntablem1];
    tb->drsq[ntablem1] = 1.0/(tb->rsq[0] - tb->rsq[ntablem1]);

    // get the correct delta values at itablemax
    // smallest r is in bin itablemin
    // largest r is in bin itablemax, which is itablemin-1,
    //   or ntablem1 if itablemin=0

    // deltas at itablemax only needed if corresponding rsq < cut*cut
    // if so, compute deltas between rsq and cut*cut
    //   if tb->match, data at cut*cut is unavailable, so we'll take
    //   deltas at itablemax-1 as a good approximation

    double e_tmp,f_tmp;
    int itablemin = minrsq_lookup.i & tb->nmask;
    itablemin >>= tb->nshiftbits;
    int itablemax = itablemin - 1;
    if (itablemin == 0) itablemax = ntablem1;
    int itablemaxm1 = itablemax - 1;
    if (itablemax == 0) itablemaxm1 = ntablem1;
    rsq_lookup.i = itablemax << tb->nshiftbits;
    rsq_lookup.i |= maskhi;
    if (rsq_lookup.f < tb->cut*tb->cut) {
      if (tb->match) {
        tb->de[itablemax] = tb->de[itablemaxm1];
        tb->df[itablemax] = tb->df[itablemaxm1];
        tb->drsq[itablemax] = tb->drsq[itablemaxm1];
      } else {
            rsq_lookup.f = tb->cut*tb->cut;
        r = sqrtf(rsq_lookup.f);
        e_tmp = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        f_tmp = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
        tb->de[itablemax] = e_tmp - tb->e[itablemax];
        tb->df[itablemax] = f_tmp - tb->f[itablemax];
        tb->drsq[itablemax] = 1.0/(rsq_lookup.f - tb->rsq[itablemax]);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   set all ptrs in a table to NULL, so can be freed safely
------------------------------------------------------------------------- */

void PairTableUCGChirality::null_table(Table *tb)
{
  tb->rfile = tb->efile = tb->ffile = NULL;
  tb->e2file = tb->f2file = NULL;
  tb->rsq = tb->drsq = tb->e = tb->de = NULL;
  tb->f = tb->df = tb->e2 = tb->f2 = NULL;
}

/* ----------------------------------------------------------------------
   free all arrays in a table
------------------------------------------------------------------------- */

void PairTableUCGChirality::free_table(Table *tb)
{
  memory->destroy(tb->rfile);
  memory->destroy(tb->efile);
  memory->destroy(tb->ffile);
  memory->destroy(tb->e2file);
  memory->destroy(tb->f2file);

  memory->destroy(tb->rsq);
  memory->destroy(tb->drsq);
  memory->destroy(tb->e);
  memory->destroy(tb->de);
  memory->destroy(tb->f);
  memory->destroy(tb->df);
  memory->destroy(tb->e2);
  memory->destroy(tb->f2);
}

/* ----------------------------------------------------------------------
   spline and splint routines modified from Numerical Recipes
------------------------------------------------------------------------- */

void PairTableUCGChirality::spline(double *x, double *y, int n,
                       double yp1, double ypn, double *y2)
{
  int i,k;
  double p,qn,sig,un;
  double *u = new double[n];

  if (yp1 > 0.99e30) y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0]) / (x[1]-x[0]) - yp1);
  }
  for (i = 1; i < n-1; i++) {
    sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
    p = sig*y2[i-1] + 2.0;
    y2[i] = (sig-1.0) / p;
    u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1]);
    u[i] = (6.0*u[i] / (x[i+1]-x[i-1]) - sig*u[i-1]) / p;
  }
  if (ypn > 0.99e30) qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0/(x[n-1]-x[n-2])) * (ypn - (y[n-1]-y[n-2]) / (x[n-1]-x[n-2]));
  }
  y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2] + 1.0);
  for (k = n-2; k >= 0; k--) y2[k] = y2[k]*y2[k+1] + u[k];

  delete [] u;
}

/* ---------------------------------------------------------------------- */

double PairTableUCGChirality::splint(double *xa, double *ya, double *y2a, int n, double x)
{
  int klo,khi,k;
  double h,b,a,y;

  klo = 0;
  khi = n-1;
  while (khi-klo > 1) {
    k = (khi+klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  h = xa[khi]-xa[klo];
  a = (xa[khi]-x) / h;
  b = (x-xa[klo]) / h;
  y = a*ya[klo] + b*ya[khi] +
    ((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi]) * (h*h)/6.0;
  return y;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTableUCGChirality::write_restart(FILE *fp)
{
  write_restart_settings(fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTableUCGChirality::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTableUCGChirality::write_restart_settings(FILE *fp)
{
  fwrite(&tabstyle,sizeof(int),1,fp);
  fwrite(&tablength,sizeof(int),1,fp);
  fwrite(&ewaldflag,sizeof(int),1,fp);
  fwrite(&pppmflag,sizeof(int),1,fp);
  fwrite(&msmflag,sizeof(int),1,fp);
  fwrite(&dispersionflag,sizeof(int),1,fp);
  fwrite(&tip4pflag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTableUCGChirality::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&tabstyle,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tablength,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&ewaldflag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&pppmflag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&msmflag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&dispersionflag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tip4pflag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&tabstyle,1,MPI_INT,0,world);
  MPI_Bcast(&tablength,1,MPI_INT,0,world);
  MPI_Bcast(&ewaldflag,1,MPI_INT,0,world);
  MPI_Bcast(&pppmflag,1,MPI_INT,0,world);
  MPI_Bcast(&msmflag,1,MPI_INT,0,world);
  MPI_Bcast(&dispersionflag,1,MPI_INT,0,world);
  MPI_Bcast(&tip4pflag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairTableUCGChirality::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  int itable;
  double fraction,value,a,b,phi;
  int tlm1 = tablength - 1;

  Table *tb = &tables[tabindex[itype][jtype]];
  if (rsq < tb->innersq) error->one(FLERR,"Pair distance < table inner cutoff");

  if (tabstyle == LOOKUP) {
    itable = static_cast<int> ((rsq-tb->innersq) * tb->invdelta);
    if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
    fforce = factor_lj * tb->f[itable];
  } else if (tabstyle == LINEAR) {
    itable = static_cast<int> ((rsq-tb->innersq) * tb->invdelta);
    if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
    value = tb->f[itable] + fraction*tb->df[itable];
    fforce = factor_lj * value;
  } else if (tabstyle == SPLINE) {
    itable = static_cast<int> ((rsq-tb->innersq) * tb->invdelta);
    if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
    b = (rsq - tb->rsq[itable]) * tb->invdelta;
    a = 1.0 - b;
    value = a * tb->f[itable] + b * tb->f[itable+1] +
      ((a*a*a-a)*tb->f2[itable] + (b*b*b-b)*tb->f2[itable+1]) *
      tb->deltasq6;
    fforce = factor_lj * value;
  } else {
    union_int_float_t rsq_lookup;
    rsq_lookup.f = rsq;
    itable = rsq_lookup.i & tb->nmask;
    itable >>= tb->nshiftbits;
    fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
    value = tb->f[itable] + fraction*tb->df[itable];
    fforce = factor_lj * value;
  }

  if (tabstyle == LOOKUP)
    phi = tb->e[itable];
  else if (tabstyle == LINEAR || tabstyle == BITMAP)
    phi = tb->e[itable] + fraction*tb->de[itable];
  else
    phi = a * tb->e[itable] + b * tb->e[itable+1] +
      ((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) * tb->deltasq6;
  return factor_lj*phi;
}

/* ----------------------------------------------------------------------
   return the Coulomb cutoff for tabulated potentials
   called by KSpace solvers which require that all pairwise cutoffs be the same
   loop over all tables not just those indexed by tabindex[i][j] since
     no way to know which tables are active since pair::init() not yet called
------------------------------------------------------------------------- */

void *PairTableUCGChirality::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") != 0) return NULL;
  if (ntables == 0) error->all(FLERR,"All pair coeffs are not set");

  double cut_coul = tables[0].cut;
  for (int m = 1; m < ntables; m++)
    if (tables[m].cut != cut_coul)
      error->all(FLERR,
                 "Pair table cutoffs must all be equal to use with KSpace");
  dim = 0;
  return &tables[0].cut;
}

/* ----------------------------------------------------------------------
   print out timing info at end of run
------------------------------------------------------------------------- */
void PairTableUCGChirality::finish()
{
  double ave;
  MPI_Allreduce(&time_substate_prob,&ave,1,MPI_DOUBLE,MPI_SUM,world);
  time_substate_prob = ave/comm->nprocs;

  MPI_Allreduce(&time_substate_force,&ave,1,MPI_DOUBLE,MPI_SUM,world);
  time_substate_force = ave/comm->nprocs;

  MPI_Allreduce(&time_pair_forces,&ave,1,MPI_DOUBLE,MPI_SUM,world);
  time_pair_forces = ave/comm->nprocs;

  double time_total = (time_substate_prob + time_substate_force + time_pair_forces) / 100.0;

  if (comm->me == 0) {
    utils::logmesg(lmp,"\n{} timing breakdown:\n", utils::uppercase(mystyle));
    utils::logmesg(lmp,"  Substate probablities (1st loop) time: {:<12.6g} {:6.2f}%\n", time_substate_prob, time_substate_prob/time_total);
    utils::logmesg(lmp,"  Substate forces (2nd loop) time      : {:<12.6g} {:6.2f}%\n", time_substate_force, time_substate_force/time_total);
    utils::logmesg(lmp,"  Pair forces (3rd loop) time          : {:<12.6g} {:6.2f}%\n", time_pair_forces, time_pair_forces/time_total);
    utils::logmesg(lmp,"  Total   time                         : {:.6g}\n",time_total * 100.0);
  }
}
