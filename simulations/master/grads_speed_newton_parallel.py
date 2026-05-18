import numpy as np
from typing import Iterable, Dict, Tuple

import numpy as np
from functools import lru_cache
from typing import Dict

from bspline import bspline_design_matrix, eval_bspline
from scipy.interpolate import BSpline

from prob import *
from write_potentials import load_models

# --- fast + parallel ensemble pass over many frames (drop-in replacement) ---

from functools import lru_cache
from typing import Dict, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor as Pool

from mpi4py import MPI

# ---------- Cached indexers (avoid per-frame boolean masks) ----------
@lru_cache(maxsize=None)
def _upper_idx(N: int):
    # i<j (no diagonal)
    return np.triu_indices(N, k=1)

@lru_cache(maxsize=None)
def _offdiag_idx(N: int):
    # all i!=j (both (i,j) and (j,i))
    ii, jj = np.indices((N, N))
    m = (ii != jj)
    return ii[m].astype(np.int32), jj[m].astype(np.int32)

def _get_indices(N: int, use_upper_triangle_only: bool):
    return _upper_idx(N) if use_upper_triangle_only else _offdiag_idx(N)

def build_bspline_basis(t, degree):
    nbasis = len(t) - degree - 1
    eye = np.eye(nbasis)
    return [BSpline(t, eye[j], degree, extrapolate=False)
            for j in range(nbasis)]

def frame_feature_sums_from_p_matrix(
    R: np.ndarray,           # (N,N) distance matrix (symmetric, zeros on diag)
    p: np.ndarray,           # (N,) per-particle probabilities (used to build p_neigh)
    sigmoid: float,          # probability parameter "a"
    model_like,              # Parameters of like interactions
    model_unlike,            # Parameters of unlike interactions
    nneigh,                  # Probability neighbor cutoff
    cutoff,                  # Probability neighbor cutoff distance
    pairs,
    B_like_rcut, 
    B_unlike_rcut
) -> Dict[str, Dict[str, np.ndarray]]:

    # --- ensure shapes/dtypes ---
    p = np.asarray(p, dtype=float).ravel()

    i_idx, j_idx = pairs  # each (M,)

    # distances in chosen ordering
    rij = R[i_idx, j_idx]  # (M,)

    # basis wrt coefficients at these distances
    #Phi_like = bspline_design_matrix(rij, model_like.t, model_like.degree)         # (M, nb_like)
    #Phi_unlike = bspline_design_matrix(rij, model_unlike.t, model_unlike.degree)  # (M, nb_unlike)

    # B-spline at these distances
    U_like = eval_bspline(rij, model_like.t, model_like.degree, model_like.c)         # (M,)
    U_unlike = eval_bspline(rij, model_unlike.t, model_unlike.degree, model_unlike.c) # (M,)

    #U_like_rcut   = float(np.dot(model_like.c,   B_like_rcut))
    #U_unlike_rcut = float(np.dot(model_unlike.c, B_unlike_rcut))

    #U_like   = U_like   - U_like_rcut
    #U_unlike = U_unlike - U_unlike_rcut


    # sigmoid probabilities + derivative wrt 'sigmoid' (p_neigh fixed)
    p_neigh, neigh_idx = rho_neigh_from_p(p, R, nneigh, cutoff)  # (N,)
    p_g, p_gradient = p_grad(sigmoid, p_neigh)                   # each (N,)

    # --- per-pair gathers (no pairs_stacked to save alloc) ---
    pi = p_g[i_idx] #p_g[i_idx]
    pj = p_g[j_idx] # p_g[j_idx]
    dpi = p_gradient[i_idx]
    dpj = p_gradient[j_idx]

    delta_U = U_like - U_unlike  # (M,)

    # pair weights
    w_like = 1.0 - pi - pj + 2.0 * pi * pj
    w_unlike = 1.0 - w_like  # slightly cheaper & numerically consistent

    M_pairs = float(w_like.size)

    sum_delta_U = float(np.abs(delta_U).sum()) / M_pairs

    sum_w_like = float(w_like.sum()) / M_pairs # for effective samples of like
    sum_w_unlike = float(w_unlike.sum()) / M_pairs # for effective samples of unlike

    # dw_like/da per pair
    w_prime = (2.0 * pj - 1.0) * dpi + (2.0 * pi - 1.0) * dpj

    sum_abs_wprime = float(np.abs(w_prime).sum()) / M_pairs

    # --- derivatives wrt 'sigmoid' (a) ---
    grad_sigmoid = np.dot(delta_U, w_prime)          # scalar

    # --- derivatives wrt spline coefficients (use einsum; no w[:,None] alloc) ---
    #grad_like = np.einsum("m,mk->k", w_like, Phi_like, optimize=True)
    #grad_unlike = np.einsum("m,ml->l", w_unlike, Phi_unlike, optimize=True)


    nb_like = model_like.nbasis
    nb_unlike = model_unlike.nbasis
    
    grad_like = np.zeros(nb_like)
    grad_unlike = np.zeros(nb_unlike)

    sum_B_like = np.zeros(nb_like)
    sum_B_unlike = np.zeros(nb_unlike)

    for k, bk in enumerate(model_like.basis):
        Bk_raw = np.nan_to_num(bk(rij), 0.0)
        Bk = Bk_raw #- B_like_rcut[k]          # anchor basis
        grad_like[k] = np.dot(w_like, Bk)
        sum_B_like[k] = Bk_raw.sum()  

    for l, bl in enumerate(model_unlike.basis):
        Bl_raw = np.nan_to_num(bl(rij), 0.0)
        Bl = Bl_raw #- B_unlike_rcut[l]          # anchor basis
        grad_unlike[l] = np.dot(w_unlike, Bl)
        sum_B_unlike[l] = Bl_raw.sum()       


    deriv_a = {"grad_a": grad_sigmoid}
    deriv_c_like = {"grad_like": grad_like}
    deriv_c_unlike = {"grad_unlike": grad_unlike}

    return {"deriv_a": deriv_a, "deriv_c_like": deriv_c_like, "deriv_c_unlike": deriv_c_unlike, 
            "sum_w_like": sum_w_like, "sum_w_unlike": sum_w_unlike, "sum_abs_wprime": sum_abs_wprime,
            "sum_delta_U": sum_delta_U, "sum_B_like": sum_B_like,
            "sum_B_unlike": sum_B_unlike, "n_pairs": M_pairs}


def ensemble_feature_means_from_p_matrix_parallel(
    frames: Iterable[Dict[str, np.ndarray]],
    model_like,
    model_unlike,
    sigmoid,
    nneigh,
    cutoff,
    N,
    use_upper_triangle_only: bool = True,
    stride: int = 1,
    max_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Efficiently computes ensemble averages over frames of:
      - <dU/dλ>                     (Eq. 51 ingredients)
      - <d²U/dλ_i dλ_j>             (Eq. 52 first two terms)  [energy-level second derivatives]
      - <(dU/dλ_i)(dU/dλ_j)>        (Eq. 52 covariance term ingredient)

    Parameter ordering (full vector):
      λ = [ a , c_like(0..K-1), c_unlike(0..M-1) ]
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    K, M = model_like.nbasis, model_unlike.nbasis
    D = 1 + K + M  # total parameter dimension

    # ---- one-time initialization ----
    model_like.basis = build_bspline_basis(model_like.t, model_like.degree)
    model_unlike.basis = build_bspline_basis(model_unlike.t, model_unlike.degree)

    pairs = _get_indices(N, use_upper_triangle_only)

    # --- NEW: cutoff anchoring constants (Option B1) ---
    rcut = float(4.0)  # or model_like.rcut if you store it there
    B_like_at_rcut = np.empty(K, dtype=float)
    B_unlike_at_rcut = np.empty(M, dtype=float)

    for k, bk in enumerate(model_like.basis):
        val = bk(rcut)
        B_like_at_rcut[k] = 0.0 if not np.isfinite(val) else float(val)

    for l, bl in enumerate(model_unlike.basis):
        val = bl(rcut)
        B_unlike_at_rcut[l] = 0.0 if not np.isfinite(val) else float(val)


    #print(f'K, M, D: {K} {M} {D}')

    # Accumulators
    acc_g = np.zeros(D, dtype=float)          # sum of gradients

    acc_sum_w_like = 0.0 # for effective samples
    acc_sum_w_unlike = 0.0 # for effective samples
    acc_sum_abs_wprime = 0.0
    acc_sum_delta_U = 0.0 

    acc_sum_B_like = np.zeros(K, dtype=float)
    acc_sum_B_unlike = np.zeros(M, dtype=float)
    acc_n_pairs = 0.0

    n = 0

    def _iter_strided():
        for k, fr in enumerate(frames):
            if (k % stride) == 0:
                yield fr

    def _worker(fr):
        return frame_feature_sums_from_p_matrix(
            fr["R"], fr["p"], sigmoid, model_like, model_unlike, nneigh, cutoff, pairs, B_like_at_rcut, B_unlike_at_rcut
        )

    #with ThreadPoolExecutor(max_workers=max_workers) as ex:
    #with ProcessPoolExecutor(max_workers=max_workers) as ex:
    #with Pool(max_workers=max_workers) as ex:
    #    for out in ex.map(_worker, _iter_strided()):
    #        # Pull per-frame derivatives (avoid extra dict lookups)
    #        ga = float(out["deriv_a"]["grad_a"])
    #        gL = np.asarray(out["deriv_c_like"]["grad_like"], dtype=float)
    #        gU = np.asarray(out["deriv_c_unlike"]["grad_unlike"], dtype=float)

    #        # Build full gradient vector g = [ga, gL, gU] (one concat per frame)
    #        #g = np.empty(D, dtype=float)
    #        #g[0] = ga
    #        #g[1:1 + K] = gL
    #        #g[1 + K:] = gU

    #        ## Accumulate <g>
    #        #acc_g += g

    #        acc_g[0] += ga
    #        acc_g[1:1 + K] += gL
    #        acc_g[1 + K:] += gU

    #        n += 1

    for out in map(_worker, _iter_strided()):
        ga = float(out["deriv_a"]["grad_a"])
        gL = np.asarray(out["deriv_c_like"]["grad_like"], dtype=float)
        gU = np.asarray(out["deriv_c_unlike"]["grad_unlike"], dtype=float)
    
        acc_g[0] += ga
        acc_g[1:1 + K] += gL
        acc_g[1 + K:] += gU

        acc_sum_w_like += float(out["sum_w_like"])
        acc_sum_w_unlike += float(out["sum_w_unlike"])
        acc_sum_abs_wprime += float(out["sum_abs_wprime"])
        acc_sum_delta_U += float(out["sum_delta_U"])

        acc_sum_B_like += np.asarray(out["sum_B_like"], dtype=float)
        acc_sum_B_unlike += np.asarray(out["sum_B_unlike"], dtype=float)
        acc_n_pairs += float(out["n_pairs"])



        n += 1

        if (n % 10) == 0 and rank == 0:
            print(f"[rank {rank}] processed {n} frames", flush=True)
    

    #print('acc_g {acc_g}')

    if n == 0:
        raise ValueError("No frames processed. Check stride / input frames.")

    invn = 1.0 / float(n)

    mean_g = acc_g * invn

    mean_sum_w_like = acc_sum_w_like * invn
    mean_sum_w_unlike = acc_sum_w_unlike * invn
    mean_sum_abs_wprime  = acc_sum_abs_wprime * invn
    mean_sum_delta_U = acc_sum_delta_U * invn

    mean_sum_B_like = acc_sum_B_like * invn
    mean_sum_B_unlike = acc_sum_B_unlike * invn
    mean_n_pairs = acc_n_pairs * invn

    mean_basis_occupancy_like = mean_sum_B_like / mean_n_pairs
    mean_basis_occupancy_unlike = mean_sum_B_unlike / mean_n_pairs


    # Keep your familiar outputs too (views, no copies)
    mean_grad_sigmoid = mean_g[0]
    mean_grad_like = mean_g[1:1 + K]
    mean_grad_unlike = mean_g[1 + K:]

    return {
        # Eq (51) ingredients (means of first derivatives)
        "mean_grad_sigmoid": mean_grad_sigmoid,
        "mean_grad_like": mean_grad_like,
        "mean_grad_unlike": mean_grad_unlike,

        # Full mean gradient vector (useful for assembling Eq 51 compactly)
        "mean_g": mean_g,

        "mean_sum_w_like": mean_sum_w_like,
        "mean_sum_w_unlike": mean_sum_w_unlike,
        "mean_sum_abs_wprime": mean_sum_abs_wprime,
        "mean_sum_delta_U": mean_sum_delta_U,

        # NEW
        "mean_basis_occupancy_like": mean_basis_occupancy_like,
        "mean_basis_occupancy_unlike": mean_basis_occupancy_unlike,
        "mean_n_pairs": mean_n_pairs,

        "n_frames": n,
        "dim": D,
        "nb_like": K,
        "nb_unlike": M,
    }




## Minimal test
#R=np.array([[0,2,6],[2,0,3],[6,3,0]])
##p=np.array([0.5,0.5,0.25])
#p=np.array([1,1,1])
#nneigh=12
#cutoff=4
#use_upper_triangle_only = True
#sigmoid=4.645
#
#p_neigh, neigh_idx = rho_neigh_from_p(p, R, nneigh, cutoff)
#p_gradient = p_grad(sigmoid, p_neigh)
#
#model_like, model_unlike = load_models('../init_potentials/iter_0/models.pkl')
#
#N = p.shape[0]
#pairs = _get_indices(N, use_upper_triangle_only)
#i_idx, j_idx = pairs
#rij = R[i_idx, j_idx] 
#
##Phi_like = bspline_design_matrix(rij, model_like.t, model_like.degree)
##Phi_unlike = bspline_design_matrix(rij, model_unlike.t, model_unlike.degree)
#
###print(rij)
###print(model_like.t, model_like.degree)
###print(Phi_like)
###print(Phi_unlike)
#
## B-spline at these distances
#U_like = eval_bspline(rij, model_like.t, model_like.degree, model_like.c)
#U_unlike = eval_bspline(rij, model_unlike.t, model_unlike.degree, model_unlike.c)
# 
#print(rij)
#print(U_like)
#print(U_unlike)
#print(p_gradient)
#print(p)
#print(p_neigh)
#
## accumulate sufficient statistics
#pairs_stacked = np.stack([i_idx, j_idx], axis=1)
#grad_sigmoid = sum_like_unlike_pairs(p_gradient, pairs_stacked, U_like, U_unlike) # scalar
#
#print(grad_sigmoid)







