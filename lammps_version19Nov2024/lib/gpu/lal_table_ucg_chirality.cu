// **************************************************************************
//                         lal_table_ucg_chirality.cu
//                             -------------------
//                            Trung Dac Nguyen
//
//  Device code for acceleration of the table/ucg/chirality pair style.
//
//  Milestone 1: this kernel computes only Phase 1 of the host compute --
//  the per-atom substate probabilities (and their partials and the local
//  enantiomeric excess).  The result is packed into prob_out and copied back
//  to the host, analogous to how the per-atom field (fieldp) is returned by
//  the AMOEBA/HIPPO styles.  The tabulated force loop and the CV back-force
//  loop are still done on the host, so no table data is referenced here.
//
// __________________________________________________________________________
//    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
// __________________________________________________________________________
//
//    begin                :
//    email                : ndactrung@gmail.com
// ***************************************************************************

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_aux_fun1.h"
#ifndef _DOUBLE_DOUBLE
_texture( pos_tex,float4);
#else
_texture_2d( pos_tex,int4);
#endif
#else
#define pos_tex x_
#endif

#define LOOKUP 0
#define LINEAR 1
#define SPLINE 2
#define BITMAP 3

#ifndef __UNION_INT_FLOAT
#define __UNION_INT_FLOAT
typedef union {
  int i;
  float f;
} union_int_float;
#endif

// fast hyperbolic tangent, mirroring PairTableUCGChirality::fast_tanh on host
ucl_inline numtyp ucg_fast_tanh(numtyp x) {
  numtyp ax = (x >= (numtyp)0.0) ? x : -x;
  numtyp e = ucl_exp((numtyp)(-2.0) * ax);
  numtyp t = ((numtyp)1.0 - e) / ((numtyp)1.0 + e);
  return (x < (numtyp)0.0) ? -t : t;
}

// ---------------------------------------------------------------------------
// Phase-1 substate probability kernel.
//
//   cv_dependent == 0 : NEIGHBOR_INDEPENDENT
//       ee[i] = chi[i];  p[0] = chi[i];  dp[0] = 0;  p[1] = 1-p[0];  dp[1] = 0
//   cv_dependent == 1 : NEIGHBOR_DEPENDENT
//       ee[i] = sum_{j in cutoff} proximity(itype_actual, r_ij, chi[j])
//       cv    = ee[i]/local_neigh
//       p[0],dp[0] = sigmoid(cv);  p[1] = 1-p[0];  dp[1] = -dp[0]
//
//   NOTE (host parity): the host calls
//       compute_proximity_function(itype_actual, distance, chirality[j])
//   against the signature (distance, threshold, neg_chirality), so on the host
//   the "distance" argument is itype_actual and the "threshold" argument is the
//   actual pair distance.  We replicate that exact argument order here so the
//   GPU result matches the CPU reference bit-for-bit (threshold_radii is
//   therefore unused in Phase 1).
// ---------------------------------------------------------------------------

#define UCG_PROB_KERNEL_BODY                                                  \
  int tid, ii, offset;                                                        \
  atom_info(t_per_atom,ii,tid,offset);                                        \
                                                                              \
  __local acctyp red_acc[2][BLOCK_PAIR];                                      \
  int n_stride;                                                               \
                                                                              \
  if (ii<inum) {                                                              \
    int nbor, nbor_end, i, numj;                                             \
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,     \
              n_stride,nbor_end,nbor);                                        \
                                                                              \
    numtyp4 ix; fetch4(ix,i,pos_tex);                                         \
    int itype = ix.w;                                                         \
    int itype_actual = actual_types_from_state[itype];                        \
    int nstates_i = nstates_per_type[itype_actual];                           \
                                                                              \
    acctyp ee = (acctyp)0;                                                    \
    acctyp lneigh = (acctyp)0;                                                \
                                                                              \
    if (cv_dependent && nstates_i > 1) {                                      \
      for ( ; nbor<nbor_end; nbor+=n_stride) {                                \
        int j = dev_packed[nbor];                                             \
        j &= NEIGHMASK;                                                       \
        numtyp4 jx; fetch4(jx,j,pos_tex);                                     \
        numtyp delx = ix.x-jx.x;                                              \
        numtyp dely = ix.y-jx.y;                                              \
        numtyp delz = ix.z-jx.z;                                              \
        numtyp rsq = delx*delx+dely*dely+delz*delz;                           \
        if (rsq < distcutsq) {                                                \
          numtyp distance = ucl_sqrt(rsq);                                    \
          numtyp arg = (((numtyp)itype_actual) - distance) /                  \
                       ((numtyp)0.1 * distance);                              \
          numtyp tf = ucg_fast_tanh(arg);                                     \
          ee += chi[j] * (numtyp)0.5 * ((numtyp)1.0 - tf);                    \
          lneigh += (acctyp)1.0;                                             \
        }                                                                     \
      }                                                                       \
    }                                                                         \
                                                                              \
    if (t_per_atom > 1) {                                                     \
      red_acc[0][tid] = ee;                                                   \
      red_acc[1][tid] = lneigh;                                               \
      for (unsigned int s=t_per_atom/2; s>0; s>>=1) {                         \
        simdsync();                                                           \
        if (offset < s) {                                                     \
          red_acc[0][tid] += red_acc[0][tid+s];                              \
          red_acc[1][tid] += red_acc[1][tid+s];                              \
        }                                                                     \
      }                                                                       \
      ee = red_acc[0][tid];                                                   \
      lneigh = red_acc[1][tid];                                               \
    }                                                                         \
                                                                              \
    if (offset == 0) {                                                        \
      int base = ii * prob_stride;                                            \
      for (int s=0; s<prob_stride; s++) prob_out[base+s] = (acctyp)0;         \
      if (nstates_i > 1) {                                                    \
        numtyp p0, dp0, ee_store;                                            \
        if (cv_dependent) {                                                   \
          numtyp cv = ee / lneigh;                                            \
          numtyp a = prob_scale[itype_actual];                               \
          numtyp b = cv_thresholds[itype_actual];                            \
          numtyp c = prob_prefactor[itype_actual];                           \
          numtyp ex = ucl_exp(-a * (cv - b));                                \
          numtyp denom = (numtyp)1.0 + ex;                                    \
          p0 = c / denom;                                                     \
          dp0 = (c * a * ex) / (denom * denom);                              \
          ee_store = (numtyp)ee;                                              \
        } else {                                                              \
          numtyp chir = chi[i];                                               \
          p0 = chir;                                                          \
          dp0 = (numtyp)0;                                                    \
          ee_store = chir;                                                    \
        }                                                                     \
        prob_out[base+0] = ee_store;                                          \
        prob_out[base+1] = p0;                                                \
        prob_out[base+2] = (numtyp)1.0 - p0;                                  \
        prob_out[base+1+max_states]   = dp0;                                  \
        prob_out[base+1+max_states+1] = cv_dependent ? -dp0 : (numtyp)0;      \
      } else {                                                                \
        prob_out[base+1] = (numtyp)1.0;                                       \
      }                                                                       \
    }                                                                         \
  }

__kernel void k_ucg_prob(const __global numtyp4 *restrict x_,
                         const __global numtyp *restrict chi,
                         const __global int *restrict nstates_per_type,
                         const __global int *restrict actual_types_from_state,
                         const __global numtyp *restrict cv_thresholds,
                         const __global numtyp *restrict prob_scale,
                         const __global numtyp *restrict prob_prefactor,
                         const __global int *dev_nbor,
                         const __global int *dev_packed,
                         __global acctyp *restrict prob_out,
                         const int cv_dependent, const int max_states,
                         const numtyp distcutsq, const int inum,
                         const int nbor_pitch, const int t_per_atom,
                         const int prob_stride) {
  UCG_PROB_KERNEL_BODY
}

// Identical to k_ucg_prob.  The state-parameter tables are tiny, so there is
// no shared-memory "fast" specialization to do; this variant exists only so
// BaseAtomic::compile_kernels can resolve the "_fast" symbol it always loads.
__kernel void k_ucg_prob_fast(const __global numtyp4 *restrict x_,
                              const __global numtyp *restrict chi,
                              const __global int *restrict nstates_per_type,
                              const __global int *restrict actual_types_from_state,
                              const __global numtyp *restrict cv_thresholds,
                              const __global numtyp *restrict prob_scale,
                              const __global numtyp *restrict prob_prefactor,
                              const __global int *dev_nbor,
                              const __global int *dev_packed,
                              __global acctyp *restrict prob_out,
                              const int cv_dependent, const int max_states,
                              const numtyp distcutsq, const int inum,
                              const int nbor_pitch, const int t_per_atom,
                              const int prob_stride) {
  UCG_PROB_KERNEL_BODY
}

// ---------------------------------------------------------------------------
// Phase-2 tabulated pair-force kernel.  For each local atom i it loops its full
// neighbor list and, for every substate pair (alpha,beta), looks up the
// tabulated force/energy, scales by p(s_i)*p(s_j), adds the CV cross-force, and
// accumulates: the pair force on i, the (halved-by-store_answers) energy/virial,
// and the per-atom substate_probability_force reduction spf[isub].
//
// HOST PARITY (important): the host computes evdwl only inside `if (eflag)`, so
// when energy is not requested the substate_probability_force AND the CV
// cross-force terms are zero.  We gate evdwl on `eflag` here to match exactly.
// ---------------------------------------------------------------------------
__kernel void k_ucg_force(const __global numtyp4 *restrict x_,
                          const __global numtyp *restrict chi,
                          const __global int *restrict tabindex,
                          const __global numtyp4 *restrict coeff2,
                          const __global numtyp4 *restrict coeff3,
                          const __global numtyp4 *restrict coeff4,
                          const __global int *restrict nshiftbits,
                          const __global int *restrict nmask,
                          const int lj_types,
                          const __global numtyp *restrict cutsq,
                          const __global numtyp *restrict sp_lj_in,
                          const __global numtyp *restrict prob,
                          const __global numtyp *restrict partial,
                          const __global int *restrict nstates_per_type,
                          const __global int *restrict actual_types_from_state,
                          const __global numtyp *restrict threshold_radii,
                          const __global int *dev_nbor,
                          const __global int *dev_packed,
                          __global acctyp3 *restrict ans,
                          __global acctyp *restrict engv,
                          const int eflag, const int vflag, const int inum,
                          const int nbor_pitch, const int t_per_atom,
                          const int tabstyle, const int tablength,
                          const int max_states) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  __local numtyp sp_lj[4];
  __local acctyp spf_acc[2][BLOCK_PAIR];
  int n_stride;
  local_allocate_store_pair();

  sp_lj[0]=sp_lj_in[0];
  sp_lj[1]=sp_lj_in[1];
  sp_lj[2]=sp_lj_in[2];
  sp_lj[3]=sp_lj_in[3];

  acctyp3 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp energy, virial[6];
  if (EVFLAG) {
    energy=(acctyp)0;
    for (int i=0; i<6; i++) virial[i]=(acctyp)0;
  }
  acctyp spf0=(acctyp)0, spf1=(acctyp)0;

  int tlm1 = tablength - 1;

  // per-atom state hoisted out of the ii<inum guard so the t_per_atom
  // reduction (all lanes must participate) and the Phase-3 loop can use them
  int i=0, numj, itype=0, itype_actual=0, nstates_i=0;
  int nbor, nbor_end=0, nbor_begin=0;
  numtyp4 ix;

  if (ii<inum) {
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);
    nbor_begin=nbor;

    fetch4(ix,i,pos_tex);
    itype=ix.w;
    itype_actual=actual_types_from_state[itype];
    nstates_i=nstates_per_type[itype_actual];
    numtyp chi_i=chi[i];

    for ( ; nbor<nbor_end; nbor+=n_stride) {
      int j=dev_packed[nbor];
      numtyp factor_lj=sp_lj[sbmask(j)];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex);
      int jtype=jx.w;
      int mtype_cut=itype*lj_types+jtype;

      numtyp delx=ix.x-jx.x;
      numtyp dely=ix.y-jx.y;
      numtyp delz=ix.z-jx.z;
      numtyp rsq=delx*delx+dely*dely+delz*delz;

      if (rsq<cutsq[mtype_cut]) {
        int jtype_actual=actual_types_from_state[jtype];
        numtyp distance=ucl_sqrt(rsq);
        numtyp rinv=ucl_recip(distance);
        numtyp threshold_jtype=threshold_radii[jtype_actual];
        int nstates_j=nstates_per_type[jtype_actual];

        numtyp pair_force=(numtyp)0;
        numtyp energy_lj=(numtyp)0;

        for (int isub=0; isub<nstates_i; isub++) {
          int alpha=itype+isub;
          numtyp alphaprob=(nstates_i>1)?prob[i*max_states+isub]:(numtyp)1.0;
          acctyp spf_isub=(acctyp)0;

          for (int jsub=0; jsub<nstates_j; jsub++) {
            int beta=jtype+jsub;
            numtyp betaprob=(nstates_j>1)?prob[j*max_states+jsub]:(numtyp)1.0;
            numtyp partial_j=partial[j*max_states+jsub];

            int ab=alpha*lj_types+beta;
            int tbindex=tabindex[ab];
            numtyp innersq=coeff2[ab].x;
            numtyp invdelta=coeff2[ab].y;
            numtyp deltasq6=coeff2[ab].z;

            numtyp fforce=(numtyp)0, evdwl=(numtyp)0;
            int itable;
            numtyp fraction, aa, bb;

            if (tabstyle==LOOKUP) {
              itable=(int)((rsq-innersq)*invdelta);
              if (itable<tlm1) {
                int idx=itable+tbindex*tablength;
                fforce=coeff3[idx].z;
                if (eflag) evdwl=coeff3[idx].y;
              }
            } else if (tabstyle==LINEAR) {
              itable=(int)((rsq-innersq)*invdelta);
              if (itable<tlm1) {
                int idx=itable+tbindex*tablength;
                fraction=(rsq-coeff3[idx].x)*invdelta;
                fforce=coeff3[idx].z+fraction*coeff4[idx].z;
                if (eflag) evdwl=coeff3[idx].y+fraction*coeff4[idx].y;
              }
            } else if (tabstyle==SPLINE) {
              itable=(int)((rsq-innersq)*invdelta);
              if (itable<tlm1) {
                int idx=itable+tbindex*tablength;
                bb=(rsq-coeff3[idx].x)*invdelta;
                aa=(numtyp)1.0-bb;
                fforce=aa*coeff3[idx].z + bb*coeff3[idx+1].z +
                       ((aa*aa*aa-aa)*coeff4[idx].z +
                        (bb*bb*bb-bb)*coeff4[idx+1].z)*deltasq6;
                if (eflag)
                  evdwl=aa*coeff3[idx].y + bb*coeff3[idx+1].y +
                        ((aa*aa*aa-aa)*coeff4[idx].y +
                         (bb*bb*bb-bb)*coeff4[idx+1].y)*deltasq6;
              }
            } else { // BITMAP
              union_int_float rsq_lookup;
              rsq_lookup.f=rsq;
              itable=rsq_lookup.i & nmask[ab];
              itable >>= nshiftbits[ab];
              int idx=itable+tbindex*tablength;
              fraction=(rsq_lookup.f-coeff3[idx].x)*coeff4[idx].w;
              fforce=coeff3[idx].z+fraction*coeff4[idx].z;
              if (eflag) evdwl=coeff3[idx].y+fraction*coeff4[idx].y;
            }

            fforce*=factor_lj;
            if (eflag) evdwl*=factor_lj;

            pair_force += fforce*alphaprob*betaprob;
            energy_lj  += evdwl*alphaprob*betaprob;

            if (nstates_i>1) {
              spf_isub += betaprob*evdwl;
              // substate_cv_forces = der(distance,threshold_jtype,-chi_i)
              //                      * evdwl * partial_j / distance
              numtyp denom=(numtyp)0.1*threshold_jtype;
              numtyp tf=ucg_fast_tanh((distance-threshold_jtype)/denom);
              numtyp der=(-chi_i)*(numtyp)0.5*((numtyp)1.0-tf*tf)/denom;
              pair_force += der*evdwl*partial_j*rinv;
            }
          } // jsub

          if (isub==0) spf0 += spf_isub;
          else         spf1 += spf_isub;
        } // isub

        f.x += pair_force*delx;
        f.y += pair_force*dely;
        f.z += pair_force*delz;

        if (EVFLAG && eflag) energy += energy_lj;
        if (EVFLAG && vflag) {
          virial[0] += delx*delx*pair_force;
          virial[1] += dely*dely*pair_force;
          virial[2] += delz*delz*pair_force;
          virial[3] += delx*dely*pair_force;
          virial[4] += delx*delz*pair_force;
          virial[5] += dely*delz*pair_force;
        }
      } // within cutsq
    } // for nbor
  } // if ii

  // reduce substate_probability_force across t_per_atom lanes and broadcast
  // the full per-atom value to every lane (the Phase-3 loop below is strided)
  if (t_per_atom>1) {
    spf_acc[0][tid]=spf0;
    spf_acc[1][tid]=spf1;
    for (unsigned int s=t_per_atom/2; s>0; s>>=1) {
      simdsync();
      if (offset<s) {
        spf_acc[0][tid]+=spf_acc[0][tid+s];
        spf_acc[1][tid]+=spf_acc[1][tid+s];
      }
    }
    simdsync();
    spf0=spf_acc[0][tid-offset];
    spf1=spf_acc[1][tid-offset];
  }

  // ---- Phase 3: CV back-force on atom i using its now-complete spf ----
  //   substate_cv_backforce += sum_s spf[i][s]*partial[i][s]
  //                            * der(distance,threshold_itype,-chi[j]) / distance
  if (ii<inum && nstates_i>1) {
    numtyp threshold_itype=threshold_radii[itype_actual];
    numtyp denom=(numtyp)0.1*threshold_itype;
    numtyp cvsum=spf0*partial[i*max_states+0];
    if (max_states>1) cvsum+=spf1*partial[i*max_states+1];
    for (nbor=nbor_begin; nbor<nbor_end; nbor+=n_stride) {
      int j=dev_packed[nbor];
      j &= NEIGHMASK;
      numtyp4 jx; fetch4(jx,j,pos_tex);
      int jtype=jx.w;
      numtyp delx=ix.x-jx.x;
      numtyp dely=ix.y-jx.y;
      numtyp delz=ix.z-jx.z;
      numtyp rsq=delx*delx+dely*dely+delz*delz;
      if (rsq<cutsq[itype*lj_types+jtype]) {
        numtyp distance=ucl_sqrt(rsq);
        numtyp rinv=ucl_recip(distance);
        numtyp chi_j=chi[j];
        numtyp tf=ucg_fast_tanh((distance-threshold_itype)/denom);
        numtyp der=(-chi_j)*(numtyp)0.5*((numtyp)1.0-tf*tf)/denom;
        numtyp fpair=cvsum*der*rinv;
        f.x+=fpair*delx;
        f.y+=fpair*dely;
        f.z+=fpair*delz;
        if (EVFLAG && vflag) {
          virial[0]+=delx*delx*fpair;
          virial[1]+=dely*dely*fpair;
          virial[2]+=delz*delz*fpair;
          virial[3]+=delx*dely*fpair;
          virial[4]+=delx*delz*fpair;
          virial[5]+=dely*delz*fpair;
        }
      }
    }
  }

  store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                ans,engv);
}
