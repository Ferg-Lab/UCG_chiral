/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Siva Dasetty and Trung Nguyen (UChicago)
------------------------------------------------------------------------- */

#include "fix_update_chirality.h"

#include "atom.h"
#include "atom_vec_chiral.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "math_const.h"
#include "neigh_list.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "pair_table_ucg_chirality.h"
#include "update.h"
#include "memory.h"
#include "random_mars.h"

#include <cmath>
#include <cstring>

#include <algorithm>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_2PI;
using MathConst::MY_4PI;

//enum {PROBABILITY_UPDATE, CHIRALITY_UPDATE, ITERATIVE_UPDATE};
enum {CHIRALITY_UPDATE, ITERATIVE_UPDATE};
enum {SELF_CONSISTENT_UPDATE, ENERGY_MINIMIZATION};
enum {NEIGHBOR_INDEPENDENT, NEIGHBOR_DEPENDENT};

//#define CHIRALITY_DEBUG

/* ---------------------------------------------------------------------- */

FixUpdateChirality::FixUpdateChirality(LAMMPS *_lmp, int narg, char **arg) :
  Fix(_lmp, narg, arg), random(nullptr), neigh_list(nullptr), pair_table_ucg(nullptr)
{
  if (narg < 4) error->all(FLERR, "Illegal fix update/chirality command");

  avec = dynamic_cast<AtomVecChiral *>(atom->style_match("chiral"));
  if (!avec) error->all(FLERR, "Fix update/chirality requires atom style chiral");

  // parse required arguments

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery < 0) error->all(FLERR, "Illegal fix update/chirality command");

  threshold_radii = utils::numeric(FLERR, arg[4], false, lmp); // rth parameter in proximity function
  probability_scaling_factor = utils::numeric(FLERR, arg[5], false, lmp); // a parameter in sigmoid
  cv_thresholds = utils::numeric(FLERR, arg[6], false, lmp); // b parameter in sigmoid
  probability_pre_factor = utils::numeric(FLERR, arg[7], false, lmp); // c parameter in sigmoid
  
  seed = utils::inumeric(FLERR, arg[8], false, lmp); // random seed

  cv_mode = NEIGHBOR_INDEPENDENT;  // default
  if (utils::inumeric(FLERR, arg[9], false, lmp) == 1) {
    cv_mode = NEIGHBOR_INDEPENDENT;
  } else if (utils::inumeric(FLERR, arg[9], false, lmp) == 0) {
    cv_mode = NEIGHBOR_DEPENDENT;
  }
  //printf("cv_mode: %d \n", cv_mode);

  random = new RanMars(lmp, seed + comm->me);

  distcut = utils::numeric(FLERR, arg[10], false, lmp);
  first_nelements = utils::inumeric(FLERR, arg[11], false, lmp);
  maxiter = utils::inumeric(FLERR, arg[12], false, lmp);

  //printf("\nReading fix args..\n");
  //printf("nevery %d threshold_radii %f probability_scaling_factor %f cv_thresholds %f probability_pre_factor %f prob_cutoff %f prob_tolerance %f\n\n", nevery, threshold_radii, probability_scaling_factor, cv_thresholds, probability_pre_factor, prob_cutoff, prob_tolerance);

  comm_forward = 1;
  nmax = 0;
  updated_chirality = nullptr;
  update_mode = SELF_CONSISTENT_UPDATE;
}

/* ---------------------------------------------------------------------- */
FixUpdateChirality::~FixUpdateChirality()
{
  memory->destroy(updated_chirality); // SD
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixUpdateChirality::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixUpdateChirality::setup(int /*vflag*/)
{
  if (force->pair_match("^hybrid",0) != nullptr) {
    auto hybrid = dynamic_cast<PairHybrid *>(force->pair);
    for (int i = 0; i < hybrid->nstyles; i++)
      if (!utils::strmatch(hybrid->keywords[i],"^table/ucg")) {
        neigh_list = hybrid->styles[i]->list;
        pair_table_ucg = (PairTableUCGChirality*)(hybrid->styles[i]);
      }  
  } else {
    neigh_list = force->pair->list;
    pair_table_ucg = (PairTableUCGChirality*)force->pair;
  }

  if (neigh_list == nullptr)
    error->all(FLERR,"Fix update/chirality requires a pair style ucg with neighbor list");

  int itmp = 0;
  auto p_cutoff = (double *) force->pair->extract("cut_coul",itmp);
  if (p_cutoff == nullptr)
    error->all(FLERR,"Pair style does not provide a cutoff");
  double cutoff = *p_cutoff;
  cutsq = cutoff * cutoff;
}

/* ---------------------------------------------------------------------- */
// Compute local chirality scaled proximity function (tanh functions)
double FixUpdateChirality::compute_proximity_function(double distance, double neg_chirality) // SD
{
  double tanh_factor = tanh((distance - threshold_radii) / (0.1 * threshold_radii));
  return neg_chirality * 0.5 * (1.0 - tanh_factor);
}

/* ---------------------------------------------------------------------- */
double FixUpdateChirality::threshold_prob_from_cv(double cv)
{ 
  double a = probability_scaling_factor;
  double b = cv_thresholds;
  double c = probability_pre_factor;
  //printf("type: %d a: %f b: %f c: %f cv: %f \n", type, a, b, c, cv);
  // sigmoidal activation based probability function and its derivative with respect to CV.
  return c / (1.0 + exp( -1.0 * (a * (cv - b )) ) );
}

/* ---------------------------------------------------------------------- */

void FixUpdateChirality::pre_force(int)
{
  if (nevery == 0) return;
  if (update->ntimestep % nevery) return;

  if (update_mode == ENERGY_MINIMIZATION)
    chirality_energy_minimize();
  else if (update_mode == SELF_CONSISTENT_UPDATE)
    self_consistent_update();
  else {
    // not implemented
  }
}

/* ---------------------------------------------------------------------- */

void FixUpdateChirality::self_consistent_update()
{
  int *ilist, *jlist, *numneigh, **firstneigh;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int i,j,ii,jj,inum,jnum,itype,jtype;
  tagint *tag = atom->tag; // [SD]
 
  double **x = atom->x;
  double *chirality = atom->chirality;
  int *type = atom->type;

  inum = neigh_list->inum;
  ilist = neigh_list->ilist;
  numneigh = neigh_list->numneigh;
  firstneigh = neigh_list->firstneigh;

  double enantiomeric_excess;
  double distance, prob, rn;
  double average_prob_j;


  if (atom->nmax > nmax) {
    memory->destroy(updated_chirality); // SD
    nmax = atom->nmax;
    memory->create(updated_chirality, nmax, "fix:updated_chirality"); // SD 
  }

  int nlocal = atom->nlocal;
  for (i = 0; i < nlocal; i++) {
    updated_chirality[i] = chirality[i]; // Initial guess
  }

  // communicate the initial updated chirality values to other procs
  // prob_i are computed in the iterations, no need to communicate at initialization.

  commflag = ITERATIVE_UPDATE;
  comm->forward_comm(this);

  if (cv_mode == NEIGHBOR_INDEPENDENT) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];
      // assign random number; doesn't depend on anything.
      rn = random->uniform(); // draw a random uniform number
      if (i < nlocal) {
        chirality[i] = rn;
      }
    }
  }
  else if (cv_mode == NEIGHBOR_DEPENDENT) {
    double chiral_diff_tolerance = 1e-5;

    for (int iter = 0; iter < maxiter; iter++) {

      //Step 1: Assign based on given sigmoid function with x as sum of prob of neighbors evaluated using updated_chirality.
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        // iterate over all neighbors of atom i
        //enantiomeric_excess = 0.0; // Evaluate ee for particle i
        std::vector<std::pair<double, size_t>> dist_ij_j_indices;
        int num_neigh = 0;
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;
          jtype = type[j];
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx * delx + dely * dely + delz * delz;
          //if (rsq < distcut*distcut) { //cutsq) {
          //if (num_neigh <= first_nelements) { 
            //distance = sqrt(rsq);
            //dist_ij_j_indices.emplace_back(distance, j);
            dist_ij_j_indices.emplace_back(rsq, j);
            num_neigh += 1;
          //}
        }

        //printf("Num neigh %d %d %d\n", update->ntimestep, tag[i], num_neigh);
       
        if (num_neigh > 0) {

          double mean_neigh_p = 0.0;
          // SD: num_neigh in the revised code (line 259- keeping all distances) should not be less than first_nelements. So below never enters.
          if (num_neigh < first_nelements) {
            std::nth_element(dist_ij_j_indices.begin(), dist_ij_j_indices.begin() + num_neigh, dist_ij_j_indices.end());
            //std::partial_sort(dist_ij_j_indices.begin(), dist_ij_j_indices.begin() + num_neigh, dist_ij_j_indices.end());
            for (size_t n = 0; n < num_neigh; ++n) {
              size_t j_idx = dist_ij_j_indices[n].second;
              //printf("idx %d %d %d %f %d\n", iter, tag[i], n, dist_ij_j_indices[n].first, j_idx);
              mean_neigh_p += updated_chirality[j_idx];
            }
            mean_neigh_p = mean_neigh_p / num_neigh;
          }
          else {
            std::nth_element(dist_ij_j_indices.begin(), dist_ij_j_indices.begin() + first_nelements, dist_ij_j_indices.end());
           // std::partial_sort(dist_ij_j_indices.begin(), dist_ij_j_indices.begin() + first_nelements, dist_ij_j_indices.end());
            for (size_t n = 0; n < first_nelements; ++n) {
              size_t j_idx = dist_ij_j_indices[n].second;
              //printf("fidx %d %d %d %f %d %f\n", iter, tag[i], n, dist_ij_j_indices[n].first, j_idx, updated_chirality[j_idx]);
              mean_neigh_p += updated_chirality[j_idx];
            }
            mean_neigh_p = mean_neigh_p / first_nelements;
          }

          // Evaluate probability function
          prob = threshold_prob_from_cv(mean_neigh_p);
          //printf("Beigh atoms: %d %d %d %d %d %f %f %f\n", iter, tag[i], jnum, num_neigh, first_nelements, mean_neigh_p, prob, updated_chirality[i]);
          if (i < nlocal) {
            // Use prob to assign probability of bead as 1 or 0.
            // updated_chirality[i] = prob;
            rn = random->uniform();
            if (rn < prob) {
              updated_chirality[i] = 1.0; //prob;
            }
            else {
              updated_chirality[i] = 0.0;
            }
          }
          //printf("Neigh atoms: %d %d %d %d %d %f %f %f\n", iter, tag[i], jnum, num_neigh, first_nelements, mean_neigh_p, rn, updated_chirality[i]);
          //printf("NProb %f \n", prob); 
        }
        else {
          /* No neighbors */
          //printf("Iter %d %d %f %f\n",iter,tag[i],updated_chirality[i],chirality[i]);
          rn = random->uniform();
          prob = 0.5;
          if (rn < prob) {
            updated_chirality[i] = 1.0; //prob; //1.0;
          }
          else {
            updated_chirality[i] = 0.0;
          }
          //printf("0Prob %f \n", prob);
        }
      }
      //printf("updated\n");
      // Communicate updated_chirality to other procs.
      commflag = ITERATIVE_UPDATE;
      comm->forward_comm(this);    

      // Step 2: Find the bead with maximum difference between updated_chirality and chirality.
      double max_difference, local_diff = 0.0;
      int nlocal = atom->nlocal;
      for (i = 0; i < nlocal; i++) {
        double diff = std::abs( std::abs(updated_chirality[i]) - std::abs(chirality[i]));
        if (local_diff < diff) {
          local_diff = diff;
        }
      }
      MPI_Allreduce(&local_diff, &max_difference, 1, MPI_DOUBLE, MPI_MAX, world);
      #ifdef CHIRALITY_DEBUG
        if (comm->me == 0) printf("step = %ld: iter = %d, max_difference = %f\n",
        update->ntimestep, iter, max_difference);
      #endif

      // store chirality with the updated values
      for (i = 0; i < nlocal; i++) {
        #ifdef CHIRALITY_DEBUG
          printf("Step: %ld Iter: %d tag: %d chirality: %f, updated chirality: %f; max_difference = %f \n", 
               update->ntimestep, iter, tag[i], chirality[i], updated_chirality[i], max_difference);
        #endif
        chirality[i] = updated_chirality[i];
      }

      // Step 3: Terminate if converged, chirality is already updated above
      if (max_difference < chiral_diff_tolerance)
        break;
    }
  } 

  // communicate converged chirality values to neighboring procs
  commflag = CHIRALITY_UPDATE;
  comm->forward_comm(this);

}

/* ----------------------------------------------------------------------
   update chiralities by minimizing the chirality energy
------------------------------------------------------------------------- */

void FixUpdateChirality::chirality_energy_minimize()
{
  // TODO: these params can be specified via fix_modify 
  int maxiter = 1000;
  double chiral_diff_tolerance = 1e-5;
  double energy_tolerance = 1e-4;

  int eflag = 1;
  int vflag = 0;
  pair_table_ucg->compute(eflag, vflag);
  double current_energy = pair_table_ucg->eng_vdwl;
  double last_energy = current_energy;

  if (atom->nmax > nmax) {
    memory->destroy(updated_chirality); // SD
    nmax = atom->nmax;
    memory->create(updated_chirality, nmax, "fix:updated_chirality"); // SD 
  }

  double *chirality = atom->chirality;
  int nlocal = atom->nlocal;

  // store the current chiralities
  for (int i = 0; i < nlocal; i++) 
    updated_chirality[i] = chirality[i];
  
  for (int iter = 0; iter < maxiter; iter++) {

    // update chirality values and communicate with neighboring procs
    // via conjugate gradient or steepest descent
    for (int i = 0; i < nlocal; i++) {
      // chirality[i] = ...
    }

    // communicate chirality with neighboring procs.
    commflag = CHIRALITY_UPDATE;
    comm->forward_comm(this);  

    // compute energy of the new chiralities
    pair_table_ucg->compute(eflag, vflag);
    current_energy = pair_table_ucg->eng_vdwl;

    // compute the chirality diff
    double max_difference, local_diff = 0.0;
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) {
      double diff = std::abs( std::abs(updated_chirality[i]) - std::abs(chirality[i]));
      if (local_diff < diff) {
        local_diff = diff;
      }
    }
    MPI_Allreduce(&local_diff, &max_difference, 1, MPI_DOUBLE, MPI_MAX, world);
    
    // check for convergence in chirality  diff and in energy diff
    if (std::abs(current_energy - last_energy) < energy_tolerance &&
        max_difference < chiral_diff_tolerance)
      break;
 
    // store the current chiralities
    for (int i = 0; i < nlocal; i++) 
      updated_chirality[i] = chirality[i];
  
    // and energy
    last_energy = current_energy;
  }
}

/* ---------------------------------------------------------------------- */

int FixUpdateChirality::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
  int * /*pbc*/)
{
  int m;
  int i,j;
  //if (commflag == PROBABILITY_UPDATE) {
  //  m = 0;
  //  for (i = 0; i < n; i++) {
  //    j = list[i];
      //buf[m++] = prob_i[i];
  //  }
  //} 
  if (commflag == ITERATIVE_UPDATE) {
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = updated_chirality[i];
    }
  } else { 
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = atom->chirality[j];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixUpdateChirality::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;
  int last;
  //if (commflag == PROBABILITY_UPDATE) {
  //  m = 0;
  //  last = first + n;
  //  for (i = first; i < last; i++) {
  //    //prob_i[i] = buf[m++];
  //  }
  //}
  if (commflag == ITERATIVE_UPDATE) {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      updated_chirality[i] = buf[m++];
    }
  } else {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++ ) {
      atom->chirality[i] = buf[m];
    }
  }
}
