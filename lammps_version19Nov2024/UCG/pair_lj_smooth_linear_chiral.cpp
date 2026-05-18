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
   Contributing author: Nikolai Petsev
   Based on pair_lj_smooth_linear by Jonathan Zimmerman
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "pair_lj_smooth_linear_chiral.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "neigh_request.h" 
#include "memory.h"
#include "error.h"
#include "neighbor.h" 
#include "molecule.h" 
#include "update.h"
#include "string.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJSmoothLinearChiral::PairLJSmoothLinearChiral(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;

  zeta = nullptr;
  dzetax = nullptr;
  dzetay = nullptr;
  dzetaz = nullptr;
  i1_vec = nullptr;
  i2_vec = nullptr;
  i3_vec = nullptr;
  i4_vec = nullptr;
  nmax = 0;

  // set comm size needed by this pair
  comm_forward = 8;
  comm_reverse = 8;
}

/* ---------------------------------------------------------------------- */

PairLJSmoothLinearChiral::~PairLJSmoothLinearChiral()
{	
  memory->destroy(zeta);
  memory->destroy(dzetax);
  memory->destroy(dzetay);
  memory->destroy(dzetaz);
  memory->destroy(i1_vec);
  memory->destroy(i2_vec);
  memory->destroy(i3_vec);
  memory->destroy(i4_vec);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(ljcut);
    memory->destroy(dljcut);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(bias);
  }
}

/* ----------------------------------------------------------------------
   compute energy and force for chiral model
------------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  int k,m,k1,k2,k3,k4,m1,m2,m3,m4;
  int ind1,ind2,ind3,ind4,jnd1,jnd2,jnd3,jnd4;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair,evdwl_tmp;
  double rsq,r2inv,r6inv,forcelj,factor_lj;
  double r,rinv;
  int *ilist,*jlist,*numneigh,**firstneigh;

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *special_lj = force->special_lj;

  int *type = atom->type;  
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Check if skin is large enough to include whole tetramers
  // In LJ units 1.5 sigma is a good choice
  double skin = neighbor->skin;
  if (strcmp(update->unit_style,"lj") == 0) {
    if (skin < sigma[1][1]*1.5) {
      error->all(FLERR,"Neighbor list skin is too small - increase it above 1.5 sigma - use, for instance, 'neighbor        1.5 bin'");
    }
  }

  // zero energy
  evdwl = 0.0;
  evdwl_tmp = 0.0;

  // update per-atom arrays for zeta and zeta gradient
  computezeta(); 

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    // choose particle i
    i = ilist[ii];

    // particle i type
    itype = type[i];

    // save particle i coordinates
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // 3 body term - identify local ID of monomers in tetramer A using mapping arrays
    ind1 = static_cast<int>(i1_vec[i]);
    ind2 = static_cast<int>(i2_vec[i]);
    ind3 = static_cast<int>(i3_vec[i]);
    ind4 = static_cast<int>(i4_vec[i]);
    
    k1 = atom->map(ind1);
    k2 = atom->map(ind2);
    k3 = atom->map(ind3);
    k4 = atom->map(ind4);

    jlist = firstneigh[i];
    jnum = numneigh[i]; 
    for (jj = 0; jj < jnum; jj++) {
      // choose particle j
      j = jlist[jj];

      // particle j type
      jtype = type[j];
	
      // 3 body term - identify local ID of monomers in tetramer B using mapping arrays
      jnd1 = static_cast<int>(i1_vec[j]);
      jnd2 = static_cast<int>(i2_vec[j]);
      jnd3 = static_cast<int>(i3_vec[j]);
      jnd4 = static_cast<int>(i4_vec[j]);

      m1 = atom->map(jnd1);
      m2 = atom->map(jnd2);
      m3 = atom->map(jnd3);
      m4 = atom->map(jnd4);

      // find LJ factor, strip additional info
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      // compute displacement vector
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
    
      // compute separation
      rsq = delx*delx + dely*dely + delz*delz;

      // check if we are within cutoff
      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        rinv  = sqrt(r2inv);

        forcelj = r6inv*(lj1[itype][jtype]*r6inv-lj2[itype][jtype]);
        forcelj = rinv*forcelj;

        // chiral renormalization term
        double bias_lj = 1.0 + bias_global*zeta[i]*zeta[j];	  
	      
        // find LJ force
        fpair = bias_lj*factor_lj*forcelj*rinv;
	      
        // find non-LJ force pre-factor
        r = sqrt(rsq);
        evdwl_tmp = factor_lj*r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]);
        evdwl_tmp -= ljcut[itype][jtype];
		
        // append LJ force for particle i		
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;

        // append non-LJ force for monomers in tetramer A, check that local ID exists
        if (k1 > -1) {
          f[k1][0] += evdwl_tmp*bias_global*zeta[j]*dzetax[k1]; 
          f[k1][1] += evdwl_tmp*bias_global*zeta[j]*dzetay[k1]; 
          f[k1][2] += evdwl_tmp*bias_global*zeta[j]*dzetaz[k1]; 
        }
        if (k2 > -1) {
          f[k2][0] += evdwl_tmp*bias_global*zeta[j]*dzetax[k2]; 
          f[k2][1] += evdwl_tmp*bias_global*zeta[j]*dzetay[k2]; 
          f[k2][2] += evdwl_tmp*bias_global*zeta[j]*dzetaz[k2]; 
        }
        if (k3 > -1) {
          f[k3][0] += evdwl_tmp*bias_global*zeta[j]*dzetax[k3]; 
          f[k3][1] += evdwl_tmp*bias_global*zeta[j]*dzetay[k3]; 
          f[k3][2] += evdwl_tmp*bias_global*zeta[j]*dzetaz[k3]; 
        }
        if (k4 > -1) {
          f[k4][0] += evdwl_tmp*bias_global*zeta[j]*dzetax[k4]; 
          f[k4][1] += evdwl_tmp*bias_global*zeta[j]*dzetay[k4]; 
          f[k4][2] += evdwl_tmp*bias_global*zeta[j]*dzetaz[k4]; 
        }
	  
        if (newton_pair || j < nlocal) {
          // append force for particle j
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;

          // append non-LJ force for monomers in tetramer B, check that local ID exists
          if (m1 > -1) {
            f[m1][0] += evdwl_tmp*bias_global*zeta[i]*dzetax[m1]; 
            f[m1][1] += evdwl_tmp*bias_global*zeta[i]*dzetay[m1]; 
            f[m1][2] += evdwl_tmp*bias_global*zeta[i]*dzetaz[m1]; 
          }
          if (m2 > -1) {
            f[m2][0] += evdwl_tmp*bias_global*zeta[i]*dzetax[m2]; 
            f[m2][1] += evdwl_tmp*bias_global*zeta[i]*dzetay[m2]; 
            f[m2][2] += evdwl_tmp*bias_global*zeta[i]*dzetaz[m2]; 
          }
          if (m3 > -1) {
            f[m3][0] += evdwl_tmp*bias_global*zeta[i]*dzetax[m3]; 
            f[m3][1] += evdwl_tmp*bias_global*zeta[i]*dzetay[m3]; 
            f[m3][2] += evdwl_tmp*bias_global*zeta[i]*dzetaz[m3]; 
          }
          if (m4 > -1) {
            f[m4][0] += evdwl_tmp*bias_global*zeta[i]*dzetax[m4]; 
            f[m4][1] += evdwl_tmp*bias_global*zeta[i]*dzetay[m4]; 
            f[m4][2] += evdwl_tmp*bias_global*zeta[i]*dzetaz[m4]; 
          }
        }

        if (eflag) evdwl = evdwl_tmp * bias_lj;
	
        if (evflag) ev_tally(i,j,nlocal,newton_pair,
          evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(ljcut,n+1,n+1,"pair:ljcut");
  memory->create(dljcut,n+1,n+1,"pair:dljcut");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(bias,n+1,n+1,"pair:bias");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Illegal pair_style command");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  bias_global = utils::numeric(FLERR, arg[1], false, lmp);

  // reset cutoffs that have been explicitly set
  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++){
      for (j = i+1; j <= atom->ntypes; j++){
        if (setflag[i][j]){
          cut[i][j] = cut_global;
          bias[i][j] = bias_global;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::coeff(int narg, char **arg)
{
  if (narg !=4 && narg != 5 && narg != 6)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
  double sigma_one = utils::numeric(FLERR, arg[3], false, lmp);

  double cut_one = cut_global;
  if (narg == 5) {
    cut_one = utils::numeric(FLERR, arg[4], false, lmp);
  }

  double bias_one = bias_global;
  if (narg == 6) {
    bias_one = utils::numeric(FLERR, arg[5], false, lmp);
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      bias[i][j] = bias_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJSmoothLinearChiral::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
    bias[i][j] = mix_energy(bias[i][i],bias[j][j],sigma[i][i],sigma[j][j]);//folarin latinwo
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  double cut6inv = pow(cut[i][j],-6.0);
  double cutinv  = 1.0/cut[i][j];
  ljcut[i][j]  = cut6inv*(lj3[i][j]*cut6inv-lj4[i][j]);
  dljcut[i][j] = cutinv*cut6inv*(lj1[i][j]*cut6inv-lj2[i][j]);

  cut[j][i] = cut[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  cut[j][i] = cut[i][j];
  ljcut[j][i] = ljcut[i][j];
  dljcut[j][i] = dljcut[i][j];
  bias[j][i] = bias[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
        fwrite(&bias[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
          fread(&bias[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&bias[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&bias_global,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&bias_global,sizeof(double),1,fp);

  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&bias_global,1,MPI_DOUBLE,0,world);

}

/* ---------------------------------------------------------------------- */

double PairLJSmoothLinearChiral::single(int i, int j, int itype, int jtype,
  double rsq, double factor_coul, double factor_lj, double &fforce)
{
  double r2inv,r6inv,forcelj,philj,r,rinv;
  double bias_lj = 1.0 + bias_global * zeta[i] * zeta[j];	  

  r2inv = 1.0/rsq;
  r6inv = r2inv*r2inv*r2inv;
  rinv  = sqrt(r2inv);
  r     = sqrt(rsq);
  forcelj = r6inv*(lj1[itype][jtype]*r6inv-lj2[itype][jtype]);
  forcelj = rinv*forcelj;
  fforce = bias_lj*factor_lj*forcelj;

  philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]);
  philj = philj - ljcut[itype][jtype];

  return factor_lj*philj*bias_lj;
}

/* ----------------------------------------------------------------------
   zeta and zeta gradient computation
------------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::computezeta() 
{
  int i1,i2,i3,i4,i,n,m;
  double vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y,vb3z;
  double tag1, tag2, tag3, tag4;
  double sb1,sb2,sb3,b1mag2,b1mag,b2mag2;
  double b2mag,b3mag2,b3mag;
  double e23e34[3], e12e34[3], e12e23[3], e23e34j[3], e12e34j[3], e12e23j[3]; 
  double gradterm12x, gradterm12y, gradterm12z; 
  double gradterm23x, gradterm23y, gradterm23z; 
  double gradterm34x, gradterm34y, gradterm34z;
  double c1f,c2f,c3f,zetaf; 

  tagint *tag = atom->tag;
  double **x = atom->x;
  int **dihedrallist = neighbor->dihedrallist;
  int ndihedrallist = neighbor->ndihedrallist;
  int newton_bond = force->newton_bond;
  int nlocal = atom->nlocal;

  // grow local atom-based arrays as needed
  if (atom->nmax > nmax) {
    memory->destroy(zeta);
    memory->destroy(dzetax);
    memory->destroy(dzetay);
    memory->destroy(dzetaz);
    memory->destroy(i1_vec);
    memory->destroy(i2_vec);
    memory->destroy(i3_vec);
    memory->destroy(i4_vec);
    nmax = atom->nmax;
    memory->create(zeta,nmax,"pair:zeta");
    memory->create(dzetax,nmax,"pair:dzetax");
    memory->create(dzetay,nmax,"pair:dzetay");
    memory->create(dzetaz,nmax,"pair:dzetaz");
    memory->create(i1_vec,nmax,"pair:i1_vec");
    memory->create(i2_vec,nmax,"pair:i2_vec");
    memory->create(i3_vec,nmax,"pair:i3_vec");
    memory->create(i4_vec,nmax,"pair:i4_vec");
  }

  // initialize zeta/dzeta and mapping arrays to zero
  if (newton_bond) {
    m = nlocal + atom->nghost;
    for (i = 0; i < m; i++) {
      zeta[i] = 0.0;
      dzetax[i] = 0.0;
      dzetay[i] = 0.0;
      dzetaz[i] = 0.0;
      i1_vec[i] = 0.0;
      i2_vec[i] = 0.0;
      i3_vec[i] = 0.0;
      i4_vec[i] = 0.0;
    }
  } else {
    for (i = 0; i < nlocal; i++) {
      zeta[i] = 0.0;
      dzetax[i] = 0.0;
      dzetay[i] = 0.0;
      dzetaz[i] = 0.0;
      i1_vec[i] = 0.0;
      i2_vec[i] = 0.0;
      i3_vec[i] = 0.0;
      i4_vec[i] = 0.0;
    }
  }	  
    
  // zeta/dzeta calculation
  for (n = 0; n < ndihedrallist; n++) { 
    i1 = dihedrallist[n][0];
    i2 = dihedrallist[n][1];
    i3 = dihedrallist[n][2];
    i4 = dihedrallist[n][3];

    // 1st bond
    vb1x = x[i2][0] - x[i1][0];
    vb1y = x[i2][1] - x[i1][1];
    vb1z = x[i2][2] - x[i1][2];

    // 2nd bond
    vb2x = x[i3][0] - x[i2][0];
    vb2y = x[i3][1] - x[i2][1];
    vb2z = x[i3][2] - x[i2][2];

    // 3rd bond
    vb3x = x[i4][0] - x[i3][0];
    vb3y = x[i4][1] - x[i3][1];
    vb3z = x[i4][2] - x[i3][2];

    // 1st and 2nd angle
    b1mag2 = vb1x*vb1x + vb1y*vb1y + vb1z*vb1z;
    b1mag = sqrt(b1mag2);
    b2mag2 = vb2x*vb2x + vb2y*vb2y + vb2z*vb2z;
    b2mag = sqrt(b2mag2);
    b3mag2 = vb3x*vb3x + vb3y*vb3y + vb3z*vb3z;
    b3mag = sqrt(b3mag2);

    c1f = (vb2y*vb3z - vb2z*vb3y);
    c2f = (vb2z*vb3x - vb2x*vb3z);
    c3f = (vb2x*vb3y - vb2y*vb3x);

    zetaf = vb1x*c1f + vb1y*c2f + vb1z*c3f;
    zetaf = zetaf/(b1mag*b2mag*b3mag);
	
    // bond length for normalization
    sb1 = 1.0 / b1mag;
    sb2 = 1.0 / b2mag;
    sb3 = 1.0 / b3mag;	
	
    // compute cross products of normalized bond vectors
    e23e34[0] = (vb2y*vb3z-vb2z*vb3y)*sb2*sb3;
    e23e34[1] = (vb2z*vb3x-vb2x*vb3z)*sb2*sb3;
    e23e34[2] = (vb2x*vb3y-vb2y*vb3x)*sb2*sb3;
    e12e34[0] = (vb1y*vb3z-vb1z*vb3y)*sb1*sb3;
    e12e34[1] = (vb1z*vb3x-vb1x*vb3z)*sb1*sb3;
    e12e34[2] = (vb1x*vb3y-vb1y*vb3x)*sb1*sb3;
    e12e23[0] = (vb1y*vb2z-vb1z*vb2y)*sb1*sb2;
    e12e23[1] = (vb1z*vb2x-vb1x*vb2z)*sb1*sb2;
    e12e23[2] = (vb1x*vb2y-vb1y*vb2x)*sb1*sb2;

    // find derivative of zeta for each monomer
    // 1-2 bond
    gradterm12x = e23e34[0]*sb1;
    gradterm12y = e23e34[1]*sb1;
    gradterm12z = e23e34[2]*sb1;
    gradterm12x -= (e23e34[0]*vb1x + e23e34[1]*vb1y + e23e34[2]*vb1z)*vb1x*sb1*sb1*sb1;
    gradterm12y -= (e23e34[0]*vb1x + e23e34[1]*vb1y + e23e34[2]*vb1z)*vb1y*sb1*sb1*sb1;
    gradterm12z -= (e23e34[0]*vb1x + e23e34[1]*vb1y + e23e34[2]*vb1z)*vb1z*sb1*sb1*sb1;     

    // 2-3 bond
    gradterm23x = e12e34[0]*sb2;
    gradterm23y = e12e34[1]*sb2;
    gradterm23z = e12e34[2]*sb2; 
    gradterm23x -= (e12e34[0]*vb2x + e12e34[1]*vb2y + e12e34[2]*vb2z)*vb2x*sb2*sb2*sb2;
    gradterm23y -= (e12e34[0]*vb2x + e12e34[1]*vb2y + e12e34[2]*vb2z)*vb2y*sb2*sb2*sb2;
    gradterm23z -= (e12e34[0]*vb2x + e12e34[1]*vb2y + e12e34[2]*vb2z)*vb2z*sb2*sb2*sb2; 

    // 3-4 bond
    gradterm34x = e12e23[0]*sb3;
    gradterm34y = e12e23[1]*sb3;
    gradterm34z = e12e23[2]*sb3;
    gradterm34x -= (e12e23[0]*vb3x + e12e23[1]*vb3y + e12e23[2]*vb3z)*vb3x*sb3*sb3*sb3;
    gradterm34y -= (e12e23[0]*vb3x + e12e23[1]*vb3y + e12e23[2]*vb3z)*vb3y*sb3*sb3*sb3;
    gradterm34z -= (e12e23[0]*vb3x + e12e23[1]*vb3y + e12e23[2]*vb3z)*vb3z*sb3*sb3*sb3;

    // store zeta and zeta gradient arrays
    zeta[i1] = zetaf;
    dzetax[i1] = gradterm12x;
    dzetay[i1] = gradterm12y;
    dzetaz[i1] = gradterm12z;

    zeta[i2] = zetaf;
    dzetax[i2] = - gradterm23x - gradterm12x;
    dzetay[i2] = - gradterm23y - gradterm12y;
    dzetaz[i2] = - gradterm23z - gradterm12z;

    zeta[i3] = zetaf;
    dzetax[i3] = gradterm34x + gradterm23x;
    dzetay[i3] = gradterm34y + gradterm23y;
    dzetaz[i3] = gradterm34z + gradterm23z;

    zeta[i4] = zetaf;    
    dzetax[i4] = - gradterm34x;
    dzetay[i4] = - gradterm34y;
    dzetaz[i4] = - gradterm34z;

    // create mapping arrays of IDs for easy access during force compute
    tag1 = static_cast<double>(tag[i1]);
    tag2 = static_cast<double>(tag[i2]);
    tag3 = static_cast<double>(tag[i3]);
    tag4 = static_cast<double>(tag[i4]);

    i1_vec[i1] = i1_vec[i2] = i1_vec[i3] = i1_vec[i4] = tag1;
    i2_vec[i1] = i2_vec[i2] = i2_vec[i3] = i2_vec[i4] = tag2;
    i3_vec[i1] = i3_vec[i2] = i3_vec[i3] = i3_vec[i4] = tag3;
    i4_vec[i1] = i4_vec[i2] = i4_vec[i3] = i4_vec[i4] = tag4;
  }
  
  // communicate zeta/dzeta values between procs
  if (newton_bond) comm->reverse_comm(this);
  comm->forward_comm(this);
}

/* ----------------------------------------------------------------------
   forward/reverse packing and unpacking routines for proc communication
------------------------------------------------------------------------- */
int PairLJSmoothLinearChiral::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, m;
	    
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = zeta[j];
    buf[m++] = dzetax[j];
    buf[m++] = dzetay[j];
    buf[m++] = dzetaz[j];
    buf[m++] = i1_vec[j];
    buf[m++] = i2_vec[j];
    buf[m++] = i3_vec[j];
    buf[m++] = i4_vec[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */
void PairLJSmoothLinearChiral::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    zeta[i] = buf[m++];
    dzetax[i] = buf[m++];
    dzetay[i] = buf[m++];
    dzetaz[i] = buf[m++];
    i1_vec[i] = buf[m++];
    i2_vec[i] = buf[m++];
    i3_vec[i] = buf[m++];
    i4_vec[i] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int PairLJSmoothLinearChiral::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = zeta[i];
    buf[m++] = dzetax[i];
    buf[m++] = dzetay[i];
    buf[m++] = dzetaz[i];
    buf[m++] = i1_vec[i];
    buf[m++] = i2_vec[i];
    buf[m++] = i3_vec[i];
    buf[m++] = i4_vec[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairLJSmoothLinearChiral::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    zeta[j] += buf[m++];
    dzetax[j] += buf[m++];
    dzetay[j] += buf[m++];
    dzetaz[j] += buf[m++];
    i1_vec[j] += buf[m++];
    i2_vec[j] += buf[m++];
    i3_vec[j] += buf[m++];
    i4_vec[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairLJSmoothLinearChiral::memory_usage()
{
  double bytes = 8 * nmax * sizeof(double);
  return bytes;
}
