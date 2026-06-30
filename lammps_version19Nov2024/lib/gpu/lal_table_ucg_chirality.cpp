/***************************************************************************
                         lal_table_ucg_chirality.cpp
                             -------------------
                            Trung Dac Nguyen

  Class for acceleration of the table/ucg/chirality pair style.

  M1: per-atom substate probabilities (Phase 1) on the device.
  M2: tabulated pair forces + substate_probability_force reduction (Phase 2)
      on the device.  Phase 3 (CV back-force) remains on the host.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : ndactrung@gmail.com
 ***************************************************************************/

#if defined(USE_OPENCL)
#include "table_ucg_chirality_cl.h"
#elif defined(USE_CUDART)
const char *table_ucg_chirality=0;
#else
#include "table_ucg_chirality_cubin.h"
#endif

#include "lal_table_ucg_chirality.h"
#include <cassert>
#include <cmath>
namespace LAMMPS_AL {
#define TableUCGChiralityT TableUCGChirality<numtyp, acctyp>

#define LOOKUP 0
#define LINEAR 1
#define SPLINE 2
#define BITMAP 3

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
TableUCGChiralityT::TableUCGChirality() : BaseAtomic<numtyp,acctyp>(),
  _max_prob_size(0), _max_chi_size(0), _max_probin_size(0),
  _allocated(false) {
}

template <class numtyp, class acctyp>
TableUCGChiralityT::~TableUCGChirality() {
  clear();
}

template <class numtyp, class acctyp>
int TableUCGChiralityT::bytes_per_atom(const int max_nbors) const {
  return this->bytes_per_atom_atomic(max_nbors);
}

template <class numtyp, class acctyp>
int TableUCGChiralityT::init(const int ntypes, double **host_cutsq,
                 double ***host_table_coeffs, double **host_table_data,
                 double *host_special_lj, const int nlocal,
                 const int nall, const int max_nbors,
                 const int maxspecial, const double cell_size,
                 const double gpu_split, FILE *_screen,
                 int tabstyle, int ntables, int tablength,
                 const int max_states, const int n_actual_types,
                 const int n_total_states, const int cv_mode,
                 const double distcut,
                 int *host_nstates_per_type, int *host_actual_types_from_state,
                 double *host_cv_thresholds, double *host_threshold_radii,
                 double *host_prob_scale, double *host_prob_prefactor) {
  int success;
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,
                            gpu_split,_screen,table_ucg_chirality,"k_ucg_prob");
  if (success!=0)
    return success;

  k_prob.set_function(*(this->pair_program),"k_ucg_prob");
  k_force.set_function(*(this->pair_program),"k_ucg_force");

  // If atom type constants fit in shared memory use fast kernel (force kernel
  // here always uses the global path, but keep lj_types consistent).
  int lj_types=ntypes;
  shared_types=false;
  int max_shared_types=this->device->max_shared_types();
  if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
    lj_types=max_shared_types;
    shared_types=true;
  }
  _lj_types=lj_types;

  _max_states      = max_states;
  _n_actual_types  = n_actual_types;
  _n_total_states  = n_total_states;
  _cv_mode         = cv_mode;          // 1 = NEIGHBOR_DEPENDENT, 0 = INDEPENDENT
  _distcutsq       = (numtyp)(distcut*distcut);
  _prob_stride     = 1 + 2*max_states;

  _tabstyle = tabstyle;
  _ntables = ntables;
  if (tabstyle != BITMAP) _tablength = tablength;
  else _tablength = 1 << tablength;

  const int na = n_actual_types + 1;   // arrays indexed by actual type id
  const int nt = n_total_states + 1;   // arrays indexed by state id

  // ---- state-parameter constants (device read-only) ----

  UCL_H_Vec<int> h_na_int(na,*(this->ucl_device),UCL_WRITE_ONLY);
  UCL_H_Vec<int> h_nt_int(nt,*(this->ucl_device),UCL_WRITE_ONLY);
  UCL_H_Vec<numtyp> h_na(na,*(this->ucl_device),UCL_WRITE_ONLY);

  nstates_per_type.alloc(na,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0;i<na;i++) h_na_int[i]=host_nstates_per_type[i];
  ucl_copy(nstates_per_type,h_na_int,false);

  actual_types_from_state.alloc(nt,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0;i<nt;i++) h_nt_int[i]=host_actual_types_from_state[i];
  ucl_copy(actual_types_from_state,h_nt_int,false);

  cv_thresholds.alloc(na,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0;i<na;i++) h_na[i]=(numtyp)host_cv_thresholds[i];
  ucl_copy(cv_thresholds,h_na,false);

  threshold_radii.alloc(na,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0;i<na;i++) h_na[i]=(numtyp)host_threshold_radii[i];
  ucl_copy(threshold_radii,h_na,false);

  prob_scale.alloc(na,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0;i<na;i++) h_na[i]=(numtyp)host_prob_scale[i];
  ucl_copy(prob_scale,h_na,false);

  prob_prefactor.alloc(na,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0;i<na;i++) h_na[i]=(numtyp)host_prob_prefactor[i];
  ucl_copy(prob_prefactor,h_na,false);

  // ---- table data (indexed by state-pair alpha,beta; same packing as
  //      pair_table_gpu / lal_table) ----

  UCL_H_Vec<int> hw_int(lj_types*lj_types,*(this->ucl_device),UCL_WRITE_ONLY);
  for (int i=0;i<lj_types*lj_types;i++) hw_int[i]=0;

  tabindex.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  nshiftbits.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  nmask.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);

  for (int ix=1; ix<ntypes; ix++)
    for (int iy=1; iy<ntypes; iy++)
      hw_int[ix*lj_types+iy] = (int)host_table_coeffs[ix][iy][0];
  ucl_copy(tabindex,hw_int,false);
  for (int ix=1; ix<ntypes; ix++)
    for (int iy=1; iy<ntypes; iy++)
      hw_int[ix*lj_types+iy] = (int)host_table_coeffs[ix][iy][1];
  ucl_copy(nshiftbits,hw_int,false);
  for (int ix=1; ix<ntypes; ix++)
    for (int iy=1; iy<ntypes; iy++)
      hw_int[ix*lj_types+iy] = (int)host_table_coeffs[ix][iy][2];
  ucl_copy(nmask,hw_int,false);

  UCL_H_Vec<numtyp4> hw(lj_types*lj_types,*(this->ucl_device),UCL_WRITE_ONLY);
  coeff2.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  for (int ix=1; ix<ntypes; ix++)
    for (int iy=1; iy<ntypes; iy++) {
      hw[ix*lj_types+iy].x = host_table_coeffs[ix][iy][3]; // innersq
      hw[ix*lj_types+iy].y = host_table_coeffs[ix][iy][4]; // invdelta
      hw[ix*lj_types+iy].z = host_table_coeffs[ix][iy][5]; // deltasq6
      hw[ix*lj_types+iy].w = (numtyp)0.0;
    }
  ucl_copy(coeff2,hw,false);

  UCL_H_Vec<numtyp4> hw2(_ntables*_tablength,*(this->ucl_device),UCL_WRITE_ONLY);
  for (int i=0;i<_ntables*_tablength;i++) {
    hw2[i].x=0.0; hw2[i].y=0.0; hw2[i].z=0.0; hw2[i].w=0.0;
  }
  coeff3.alloc(_ntables*_tablength,*(this->ucl_device),UCL_READ_ONLY);
  for (int n=0; n<_ntables; n++) {
    if (tabstyle == LOOKUP) {
      for (int k=0; k<_tablength-1; k++) {
        hw2[n*_tablength+k].x = (numtyp)0;
        hw2[n*_tablength+k].y = host_table_data[n][6*k+1]; // e
        hw2[n*_tablength+k].z = host_table_data[n][6*k+2]; // f
        hw2[n*_tablength+k].w = (numtyp)0;
      }
    } else {
      for (int k=0; k<_tablength; k++) {
        hw2[n*_tablength+k].x = host_table_data[n][6*k+0]; // rsq
        hw2[n*_tablength+k].y = host_table_data[n][6*k+1]; // e
        hw2[n*_tablength+k].z = host_table_data[n][6*k+2]; // f
        hw2[n*_tablength+k].w = (numtyp)0;
      }
    }
  }
  ucl_copy(coeff3,hw2,false);

  coeff4.alloc(_ntables*_tablength,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0;i<_ntables*_tablength;i++) {
    hw2[i].x=0.0; hw2[i].y=0.0; hw2[i].z=0.0; hw2[i].w=0.0;
  }
  for (int n=0; n<_ntables; n++) {
    if (tabstyle == LINEAR) {
      for (int k=0; k<_tablength-1; k++) {
        hw2[n*_tablength+k].y = host_table_data[n][6*k+3]; // de
        hw2[n*_tablength+k].z = host_table_data[n][6*k+4]; // df
      }
    } else if (tabstyle == SPLINE) {
      for (int k=0; k<_tablength; k++) {
        hw2[n*_tablength+k].y = host_table_data[n][6*k+3]; // e2
        hw2[n*_tablength+k].z = host_table_data[n][6*k+4]; // f2
      }
    } else if (tabstyle == BITMAP) {
      for (int k=0; k<_tablength; k++) {
        hw2[n*_tablength+k].y = host_table_data[n][6*k+3]; // de
        hw2[n*_tablength+k].z = host_table_data[n][6*k+4]; // df
        hw2[n*_tablength+k].w = host_table_data[n][6*k+5]; // drsq
      }
    }
  }
  ucl_copy(coeff4,hw2,false);

  UCL_H_Vec<numtyp> host_rsq(lj_types*lj_types,*(this->ucl_device),UCL_WRITE_ONLY);
  cutsq.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack1(ntypes,lj_types,cutsq,host_rsq,host_cutsq);

  UCL_H_Vec<double> dview;
  sp_lj.alloc(4,*(this->ucl_device),UCL_READ_ONLY);
  dview.view(host_special_lj,4,*(this->ucl_device));
  ucl_copy(sp_lj,dview,false);

  // ---- per-atom buffers ----

  _max_chi_size=static_cast<int>(static_cast<double>(nall)*1.10);
  chi.alloc(_max_chi_size,*(this->ucl_device),UCL_READ_WRITE,UCL_READ_WRITE);

  _max_probin_size=static_cast<int>(static_cast<double>(nall)*1.10);
  prob_in.alloc(_max_probin_size*max_states,*(this->ucl_device),
                UCL_READ_WRITE,UCL_READ_WRITE);
  partial_in.alloc(_max_probin_size*max_states,*(this->ucl_device),
                   UCL_READ_WRITE,UCL_READ_WRITE);

  _max_prob_size=static_cast<int>(static_cast<double>(nlocal)*1.10);
  if (_max_prob_size<1) _max_prob_size=1;
  prob_out.alloc(_max_prob_size*_prob_stride,*(this->ucl_device),
                 UCL_READ_WRITE,UCL_READ_WRITE);

  _allocated=true;
  this->_max_bytes=nstates_per_type.row_bytes()+
    actual_types_from_state.row_bytes()+cv_thresholds.row_bytes()+
    threshold_radii.row_bytes()+prob_scale.row_bytes()+
    prob_prefactor.row_bytes()+sp_lj.row_bytes()+
    tabindex.row_bytes()+nshiftbits.row_bytes()+nmask.row_bytes()+
    coeff2.row_bytes()+coeff3.row_bytes()+coeff4.row_bytes()+cutsq.row_bytes()+
    chi.device.row_bytes()+prob_out.device.row_bytes()+
    prob_in.device.row_bytes()+partial_in.device.row_bytes();
  return 0;
}

template <class numtyp, class acctyp>
void TableUCGChiralityT::clear() {
  if (!_allocated)
    return;
  _allocated=false;

  nstates_per_type.clear();
  actual_types_from_state.clear();
  cv_thresholds.clear();
  threshold_radii.clear();
  prob_scale.clear();
  prob_prefactor.clear();
  tabindex.clear();
  nshiftbits.clear();
  nmask.clear();
  coeff2.clear();
  coeff3.clear();
  coeff4.clear();
  cutsq.clear();
  sp_lj.clear();
  chi.clear();
  prob_out.clear();
  prob_in.clear();
  partial_in.clear();

  k_prob.clear();
  k_force.clear();

  this->clear_atomic();
}

template <class numtyp, class acctyp>
double TableUCGChiralityT::host_memory_usage() const {
  return this->host_memory_usage_atomic()+
         sizeof(TableUCGChirality<numtyp,acctyp>);
}

// ---------------------------------------------------------------------------
// Phase 1: copy nbor list from host, cast atom data + chirality to device, run
// the substate-probability kernel, copy the packed result back to the host.
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
void TableUCGChiralityT::compute_probabilities(const int f_ago,
                  const int inum_full, const int nall, double **host_x,
                  int *host_type, int *ilist, int *numj, int **firstneigh,
                  double *host_chirality, int &host_start,
                  const double cpu_time, bool &success, void **prob_ptr) {
  this->acc_timers();

  if (inum_full>_max_prob_size) {
    _max_prob_size=static_cast<int>(static_cast<double>(inum_full)*1.10);
    prob_out.resize(_max_prob_size*_prob_stride);
  }
  *prob_ptr=prob_out.host.begin();

  if (nall>_max_chi_size) {
    _max_chi_size=static_cast<int>(static_cast<double>(nall)*1.10);
    chi.resize(_max_chi_size);
  }
  if (nall>_max_probin_size) {
    _max_probin_size=static_cast<int>(static_cast<double>(nall)*1.10);
    prob_in.resize(_max_probin_size*_max_states);
    partial_in.resize(_max_probin_size*_max_states);
  }

  if (inum_full==0) {
    host_start=0;
    this->resize_atom(0,nall,success);
    this->zero_timers();
    return;
  }

  int ago=this->hd_balancer.ago_first(f_ago);
  int inum=this->hd_balancer.balance(ago,inum_full,cpu_time);
  this->ans->inum(inum);
  host_start=inum;

  if (ago==0) {
    this->reset_nbors(nall, inum, ilist, numj, firstneigh, success);
    if (!success)
      return;
  }

  this->atom->cast_x_data(host_x,host_type);
  this->atom->add_x_data(host_x,host_type);

  // cast chirality for all atoms (local + ghost) and copy to device
  for (int k=0; k<nall; k++) chi[k]=(numtyp)host_chirality[k];
  chi.update_device(nall,false);

  // launch the Phase-1 kernel
  const int BX=this->block_size();
  int GX=static_cast<int>(ceil(static_cast<double>(inum)/
                               (BX/this->_threads_per_atom)));
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  k_prob.set_size(GX,BX);
  k_prob.run(&this->atom->x, &chi, &nstates_per_type, &actual_types_from_state,
             &cv_thresholds, &prob_scale, &prob_prefactor,
             &this->nbor->dev_nbor, &this->_nbor_data->begin(),
             &prob_out, &_cv_mode, &_max_states, &_distcutsq, &inum,
             &nbor_pitch, &this->_threads_per_atom, &_prob_stride);
  this->time_pair.stop();

  prob_out.update_host(inum*_prob_stride,false);
}

// ---------------------------------------------------------------------------
// Phase 2: push ghost-filled probabilities to device, run the tabulated force
// kernel (forces/energy/virial -> ans, registered for fix_gpu accumulation),
// and copy the per-atom substate_probability_force reduction back to the host.
// Reuses the atom/neighbor data already cast in compute_probabilities().
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
void TableUCGChiralityT::compute_forces(const int eflag, const int vflag,
                  const int inum, const int nall, int *ilist, double **host_prob,
                  double **host_partial) {
  const int ms=_max_states;

  // pack probabilities/partials for all atoms (local+ghost) and push to device
  for (int k=0; k<nall; k++)
    for (int s=0; s<ms; s++) {
      prob_in[k*ms+s]=(numtyp)host_prob[k][s];
      partial_in[k*ms+s]=(numtyp)host_partial[k][s];
    }
  prob_in.update_device(nall*ms,false);
  partial_in.update_device(nall*ms,false);

  // Phases 2 and 3 are fused in k_ucg_force: it produces the tabulated pair
  // forces AND the CV back-force, so substate_probability_force never leaves
  // the device.  Forces/energy/virial are registered for fix gpu accumulation.
  this->ans->inum(inum);
  int red_blocks=loop(eflag,vflag);
  this->ans->copy_answers(eflag!=0,vflag!=0,eflag==2,vflag==2,ilist,red_blocks);
  this->device->add_ans_object(this->ans);
}

// Launch the Phase-2 tabulated-force kernel.
template <class numtyp, class acctyp>
int TableUCGChiralityT::loop(const int eflag, const int vflag) {
  const int BX=this->block_size();
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));
  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  k_force.set_size(GX,BX);
  k_force.run(&this->atom->x, &chi, &tabindex, &coeff2, &coeff3, &coeff4,
              &nshiftbits, &nmask, &_lj_types, &cutsq, &sp_lj,
              &prob_in, &partial_in, &nstates_per_type, &actual_types_from_state,
              &threshold_radii,
              &this->nbor->dev_nbor, &this->_nbor_data->begin(),
              &this->ans->force, &this->ans->engv, &eflag, &vflag, &ainum,
              &nbor_pitch, &this->_threads_per_atom, &_tabstyle, &_tablength,
              &_max_states);
  this->time_pair.stop();
  return GX;
}

template class TableUCGChirality<PRECISION,ACC_PRECISION>;
}
