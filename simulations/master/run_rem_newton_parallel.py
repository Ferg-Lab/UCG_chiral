import os
import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple, Optional
from time import perf_counter
from mpi4py import MPI
from mpi4py.util import pkl5
import socket


from process_traj import *
from write_potentials import * #load_models, load_coeffs
#from grads_speed import ensemble_feature_means_from_p_matrix, ensemble_feature_means_from_p_matrix_parallel
from grads_speed_newton_parallel import ensemble_feature_means_from_p_matrix_parallel
from bspline import eval_bspline
from scipy.interpolate import BSpline

#from bspline import *
#from grads import *
#from grads_speed import *


# ---------- REM objective (linear in coefficients) ----------
from typing import Dict, Tuple
import numpy as np

def fixed_end_indices(n, nfix_left=0, nfix_right=0):
    """
    Return indices [0..nfix_left-1] U [n-nfix_right .. n-1], clipped safely.
    """
    nfix_left = int(max(0, nfix_left))
    nfix_right = int(max(0, nfix_right))

    left = np.arange(min(nfix_left, n), dtype=int)
    right_start = max(n - nfix_right, 0)
    right = np.arange(right_start, n, dtype=int)

    if left.size == 0:
        return right
    if right.size == 0:
        return left
    return np.unique(np.concatenate([left, right]))


def build_fixed_idx_ends(K, M, nfix_like_left=6, nfix_like_right=4, nfix_unlike_left=6, nfix_unlike_right=4, fix_a=False):
    fixed = []

    if fix_a:
        fixed.append(0)

    fixed_like = fixed_end_indices(K, nfix_like_left, nfix_like_right)      # in [0..K-1]
    fixed_unlike = fixed_end_indices(M, nfix_unlike_left, nfix_unlike_right)    # in [0..M-1]

    # map to global indices
    fixed.extend(1 + fixed_like)
    fixed.extend(1 + K + fixed_unlike)

    return np.asarray(sorted(set(fixed)), dtype=int)


# ---------- REM objective + gradient (Eq. 51) + Hessian (Eq. 52) ----------
def rem_grads(
    aa_stats: Dict[str, np.ndarray],
    cg_stats: Dict[str, np.ndarray],
    beta,
    K,
    M,
    free_mask
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Implements:
      Eq. (51):  ∇ S_rel = β <∂U/∂λ>_FP  - β <∂U/∂λ>_CG
      Eq. (52):  H = β <∂²U/∂λi∂λj>_FP - β <∂²U/∂λi∂λj>_CG
                      + β² ( <g g^T>_CG - <g>_CG <g>_CG^T )

    Here:
      - "FP" corresponds to aa_stats (reference)
      - "CG" corresponds to cg_stats (current model)
      - g := ∂U/∂λ  (energy gradient vector)

    Inputs:
      aa_stats, cg_stats are outputs of ensemble_feature_means_from_p_matrix_parallel(...)
      beta can be set to 1.0 if you already absorb β elsewhere.

    Returns:
      loss_dict, grad_Srel (full vector), H_Srel (full matrix)
    """

    # --- Eq. (51): gradient of S_rel ---
    # Full gradient vectors <g> for AA/FP and CG ensembles
    g_aa = np.asarray(aa_stats["mean_g"], dtype=float)  # (D,)
    g_cg = np.asarray(cg_stats["mean_g"], dtype=float)  # (D,)

    grad_Srel = beta * (g_aa - g_cg)  # (D,)

    # Also keep your component-wise outputs (same as before, just scaled by beta if you want)
    #d_grad_like = beta * (aa_stats["mean_grad_like"] - cg_stats["mean_grad_like"])
    #d_grad_unlike = beta * (aa_stats["mean_grad_unlike"] - cg_stats["mean_grad_unlike"])
    #d_grad_sigmoid = beta * (aa_stats["mean_grad_sigmoid"] - cg_stats["mean_grad_sigmoid"])

    # For backwards compatibility / inspection
    loss = {
        "aa_mean_basis_occupancy_like": aa_stats["mean_basis_occupancy_like"],
        "aa_mean_basis_occupancy_unlike": aa_stats["mean_basis_occupancy_unlike"],
        "cg_mean_basis_occupancy_like": cg_stats["mean_basis_occupancy_like"],
        "cg_mean_basis_occupancy_unlike": cg_stats["mean_basis_occupancy_unlike"],

        "d_grad_sigmoid_aa": g_aa[0],
        "d_grad_like_aa": g_aa[1:1 + K],
        "d_grad_unlike_aa": g_aa[1 + K:],
        "d_grad_sigmoid_cg": g_cg[0],
        "d_grad_like_cg": g_cg[1:1 + K],
        "d_grad_unlike_cg": g_cg[1 + K:],

        "d_grad_like": grad_Srel[1:1+K],
        "d_grad_unlike": grad_Srel[1+K:],
        "d_grad_sigmoid": grad_Srel[0],
        "grad_Srel": grad_Srel[free_mask],

        "S_like_cg": cg_stats.get("mean_sum_w_like", None),
        "S_unlike_cg": cg_stats.get("mean_sum_w_unlike", None),
        "S_like_aa": aa_stats.get("mean_sum_w_like", None),
        "S_unlike_aa": aa_stats.get("mean_sum_w_unlike", None),

        "mean_wprime_cg": cg_stats.get("mean_sum_abs_wprime", None),
        "mean_wprime_aa": aa_stats.get("mean_sum_abs_wprime", None),

        "mean_sum_delta_U_cg": cg_stats.get("mean_sum_delta_U", None),
        "mean_sum_delta_U_aa": aa_stats.get("mean_sum_delta_U", None)
         

    }

    return loss, grad_Srel


# ---- one Newton-like REM step:  λ^{k+1} = λ^k - χ H^{-1} ∇S_rel  ----
#def rem_step(c_like, c_unlike, sigmoid, aa_stats, cg_stats, chi=1.0, beta=1.0, damping=1e-8):
def rem_step(c_like, c_unlike, sigmoid, aa_stats, cg_stats, chi=1.0, beta=1.0, damping=1e-8, nfix_like_left=6, nfix_like_right=4, nfix_unlike_left=6, nfix_unlike_right=4):
    """
    Uses Eq. (50): λ^{k+1} = λ^k - χ H^{-1} ∇_λ S_rel
    where ∇S_rel and H are from Eq. (51) and Eq. (52).

    IMPORTANT:
      - Full H from Eq. (52) is generally dense due to the covariance (Fisher) term,
        but if you choose to *ignore* covariance you get a block-sparse matrix with
        zero diagonal spline blocks.
      - Here we use the *full* H_Srel returned by rem_grads(), then add damping
        to ensure solvability.
    """

    # Parameter sizes / layout must match ensemble_feature_means_from_p_matrix_parallel:
    # λ = [ a , c_like(0..K-1), c_unlike(0..M-1) ]
    K = c_like.shape[0]
    M = c_unlike.shape[0]
    D_expected = 1 + K + M

   
    # Build fixed indices (global)
    fixed_idx = build_fixed_idx_ends(K, M, nfix_like_left=nfix_like_left, nfix_like_right=nfix_like_right, nfix_unlike_left=nfix_unlike_left, nfix_unlike_right=nfix_unlike_right)

     # Free mask
    free_mask = np.ones(D_expected, dtype=bool)
    free_mask[fixed_idx] = False


    # Get full gradient and Hessian of S_rel (Eq. 51 & 52)
    loss, grad_Srel = rem_grads(aa_stats, cg_stats, beta, K, M, free_mask)


    #if grad_Srel.shape[0] != D_expected or H_Srel.shape != (D_expected, D_expected):
    #    raise ValueError(
    #        f"Dimension mismatch: expected D={D_expected}, got grad {grad_Srel.shape}, Hess {H_Srel.shape}"
    #    )

    # Damped solve on free subspace only
    g = grad_Srel

    #gf = g[free_mask]


    ###### >>> BEGIN ADD: support-based diagonal scaling (AA + CG)
    #####eps = 1e-12

    #####S_like = float(cg_stats["mean_sum_w_like"]) + float(aa_stats["mean_sum_w_like"])
    #####S_unlike = float(cg_stats["mean_sum_w_unlike"]) + float(aa_stats["mean_sum_w_unlike"])

    #####sA = float(cg_stats["mean_sum_abs_wprime"]) + float(aa_stats["mean_sum_abs_wprime"])


    #####scale = np.ones(D_expected, dtype=float)

    ###### scale[0] = 1.0 / (sA + eps)
    #####scale[1:1 + K] = 1.0 / (S_like + eps)
    #####scale[1 + K:] = 1.0 / (S_unlike + eps)

    #####delta = scale * g              # scale full gradient
    #####delta[~free_mask] = 0.0        # respect fixed parameters


    # >>> BEGIN REPLACE: block-gated steepest-descent (no scaling)
    #tol = 1e-2
    
    # supports are fractions in [0,1] (pair-averaged)
    #S_like = float(cg_stats["mean_sum_w_like"])     # or AA+CG if you insist, but CG is the identifiability limiter
    #S_unlike = float(cg_stats["mean_sum_w_unlike"])
    
    delta = g.copy()
    delta[~free_mask] = 0.0

    #S_target = 0.05     # composition mismatch threshold
    
    #dS = abs(float(cg_stats["mean_sum_w_like"]) -  float(aa_stats["mean_sum_w_like"]))
    
    #if dS > S_target:
        # --- mismatch large: update ONLY a ---
    #    delta[1:] = 0.0


    # --- NEW: occupancy-based freezing (wall-safe) ---
    #occ_eps = 1e-4   # tune; start 1e-4 to 1e-3

    #occ_like_aa = np.asarray(aa_stats["mean_basis_occupancy_like"], dtype=float)   # (K,)
    #occ_like_cg = np.asarray(cg_stats["mean_basis_occupancy_like"], dtype=float)   # (K,)
    #occ_unlike_aa = np.asarray(aa_stats["mean_basis_occupancy_unlike"], dtype=float) # (M,)
    #occ_unlike_cg = np.asarray(cg_stats["mean_basis_occupancy_unlike"], dtype=float) # (M,)
   
    #print("min/max occ_like_cg", occ_like_cg.min(), occ_like_cg.max(), flush=True)
    #print("min/max occ_like_aa", occ_like_aa.min(), occ_like_aa.max(), flush=True)
    #print("min/max occ_unlike_cg", occ_unlike_cg.min(), occ_unlike_cg.max(), flush=True)
    #print("min/max occ_unlike_aa", occ_unlike_aa.min(), occ_unlike_aa.max(), flush=True)

    # "Supported" = both ensembles actually put probability mass there.
    # This avoids training bases that sit on the hard wall / rarely sampled.
    #like_supported = (occ_like_aa > occ_eps) & (occ_like_cg > occ_eps)
    #unlike_supported = (occ_unlike_aa > occ_eps) & (occ_unlike_cg > occ_eps)

    # Apply to delta (delta indices align with lam layout)
    #delta_like = delta[1:1 + K]
    #delta_unlike = delta[1 + K:]

    #delta_like[~like_supported] = 0.0
    #delta_unlike[~unlike_supported] = 0.0

    #delta[1:1 + K] = delta_like
    #delta[1 + K:] = delta_unlike



    # --- NEW: per-iteration step clipping (trust region in parameter space) ---
    # Limits how much each parameter can change in one iteration:
    #   |Δa| <= da_max
    #   |Δc_k| <= dc_like_max / dc_unlike_max
    da_max = 0.005           # tune
    dc_like_max = 0.02     # tune (in same units as c_like)
    dc_unlike_max = 0.02    # tune

    # delta is the gradient; actual step is chi * delta
    # so clip delta so that chi*delta respects the max step size
    if chi > 0:
        delta[0] = np.clip(delta[0], -da_max / chi, da_max / chi)

        delta[1:1 + K] = np.clip(delta[1:1 + K], -dc_like_max / chi, dc_like_max / chi)
        delta[1 + K:]  = np.clip(delta[1 + K:],  -dc_unlike_max / chi, dc_unlike_max / chi)

    #da = -chi * float(delta[0])  # because lam_new = lam - chi*delta

    
    # gate spline blocks if support too small
    #if S_like < tol:
    #    delta[1:1 + K] = 0.0
    
    #if S_unlike < tol:
    #    delta[1 + K:] = 0.0
    
    # (optional) gate 'a' using derivative-support if you added it; otherwise leave 'a' ungated
    # sA = float(cg_stats["mean_abs_wprime"])
    # if sA < tol_a:
    #     delta[0] = 0.0
    # >>> END REPLACE



    # add damping in free space (better conditioned than full-space damping)
    #delta_f = gf
    #print(delta_f)
    #print('Delta completed')

    # assemble full delta with zeros in fixed slots
    #delta = np.zeros(D_expected, dtype=float)
    #print(delta)
    #delta[free_mask] = delta_f
    #print(delta_f)

    ## Damping: because some curvature directions can be near-zero / noisy
    ## (also covers the 'zero diagonal' issue if you are near the energy-Hessian-only regime)
    #H = H_Srel + damping * np.eye(D_expected)

    ## Solve H * delta = grad, then λ_new = λ - chi * delta
    ## (Prefer solve over explicit inverse)
    #delta_e = np.linalg.solve(H, grad_Srel)
    #print(delta_e)
    #print('end delta e')

    # Current parameter vector
    lam = np.empty(D_expected, dtype=float)


    lam[0] = float(sigmoid)
    lam[1:1 + K] = np.asarray(c_like, dtype=float)
    lam[1 + K:] = np.asarray(c_unlike, dtype=float)

    lam_new = lam - chi * delta

    sigmoid_new = float(lam_new[0])
    c_like_new = lam_new[1:1 + K].copy()
    c_unlike_new = lam_new[1 + K:].copy()

    return c_like_new, c_unlike_new, sigmoid_new, loss



# ============================================================
if __name__ == "__main__":

    RUNSTEP = SSSS

    kt_reduced = TTTT
    lr   = 0.002 #5
    step = RUNSTEP
    LOGPREVIOUSPATH = 'iter_' + str(int(step)-1)
    LOGPATH = 'iter_' + str(step)
    nneigh = 12
    cutoff = 4

    ref_dump_path = '/project2/andrewferguson/sivadasetty/doe/ucg/chiral_tetramer/SystemSize1000tetramers/T-'+str(kt_reduced)+'_freqdouble_racemic/dump_rerun_files/ucg_traj_reformat_joinfrag_vmd_fixed_forces_binary2_chirality.lammpstrj'
    model_dump_path = LOGPREVIOUSPATH + '/UCG_chiral.lammpstrj'

    #t0 = perf_counter()

    comm_world = MPI.COMM_WORLD
    comm = pkl5.Intracomm(comm_world)   # <-- important for large Python objects
    print(f"rank {comm_world.Get_rank()}/{comm_world.Get_size()} on {socket.gethostname()} pid={os.getpid()}", flush=True)

    rank = comm.Get_rank()
    size = comm.Get_size()
    #print(f'rank: {rank} size: {size}')
    
    if rank == 0:
        full_aa_traj = dump_to_batch(ref_dump_path, True)
        aa_traj  = [x for i, x in enumerate(full_aa_traj) if (i % 10) < 2]
        n = len(aa_traj)
        # build a plain Python list of exactly `size` chunks (contiguous slices)
        chunks = [aa_traj[(r*n)//size : ((r+1)*n)//size] for r in range(size)]
        assert len(chunks) == size
    else:
        chunks = None
    
    aa_traj_local = comm.scatter(chunks, root=0)

    # avoid printing the whole trajectory chunk (huge)
    print(f"[rank {rank}/{size}] n_local_frames={len(aa_traj_local)}", flush=True)

    if rank == 0:
        cg_traj = dump_to_batch(model_dump_path, False)
        NCG = len(cg_traj)
        cg_chunks = [cg_traj[(r*NCG)//size : ((r+1)*NCG)//size] for r in range(size)]
        assert len(cg_chunks) == size
    else:
        cg_chunks = None

    cg_traj_local = comm.scatter(cg_chunks, root=0)

    #t1 = perf_counter()
    #local_time = t1 - t0

    #if rank == 0:
    #    print(f'local time at the end of reading trajs: {local_time}', flush=True)

    model_ll, model_ld = load_models(filename=LOGPREVIOUSPATH + '/models.pkl')
    saved_c_ll, saved_c_ld, saved_sigmoid = load_coeffs(LOGPREVIOUSPATH +"/coeff.pkl")
    c_like = saved_c_ll
    c_unlike = saved_c_ld
    sigmoid = saved_sigmoid

    #t0 = perf_counter()
    N = 1000

    # --- your per-rank frame processing here ---
    aa_local = ensemble_feature_means_from_p_matrix_parallel(aa_traj_local, model_ll, model_ld, sigmoid, nneigh, cutoff, N, True, 1, 1)

    # --- combine per-rank means into global mean ---
    n_local = int(aa_local["n_frames"])
   
    # convert local means -> local sums
    sum_g_local = aa_local["mean_g"] * float(n_local)   # (D,)
    sum_w_like_local = aa_local["mean_sum_w_like"] * float(n_local)
    sum_w_unlike_local = aa_local["mean_sum_w_unlike"] * float(n_local)
    sum_abs_wprime_local = aa_local["mean_sum_abs_wprime"] * float(n_local)
    sum_delta_U_local = aa_local["mean_sum_delta_U"] * float(n_local)

    mean_basis_occupancy_like_local = aa_local["mean_basis_occupancy_like"] * float(n_local)
    mean_basis_occupancy_unlike_local = aa_local["mean_basis_occupancy_unlike"] * float(n_local)
    mean_n_pairs_local = aa_local["mean_n_pairs"] * float(n_local)

    # global sums across all ranks
    sum_g_global = comm.allreduce(sum_g_local, op=MPI.SUM)
    sum_w_like_global = comm.allreduce(sum_w_like_local, op=MPI.SUM)
    sum_w_unlike_global = comm.allreduce(sum_w_unlike_local, op=MPI.SUM)
    sum_abs_wprime_global = comm.allreduce(sum_abs_wprime_local, op=MPI.SUM)
    sum_delta_U_global = comm.allreduce(sum_delta_U_local, op=MPI.SUM)
    mean_basis_occupancy_like_global = comm.allreduce(mean_basis_occupancy_like_local, op=MPI.SUM)
    mean_basis_occupancy_unlike_global = comm.allreduce(mean_basis_occupancy_unlike_local, op=MPI.SUM)
    mean_n_pairs_global = comm.allreduce(mean_n_pairs_local, op=MPI.SUM)
    
    n_global = comm.allreduce(n_local, op=MPI.SUM)
   
    if n_global == 0:
        raise ValueError("No frames processed across all ranks.")
   
    mean_g_global = sum_g_global / float(n_global)
    mean_w_like_global = sum_w_like_global / float(n_global)
    mean_w_unlike_global = sum_w_unlike_global / float(n_global) 
    mean_abs_wprime_global = sum_abs_wprime_global / float(n_global)
    mean_delta_U_global = sum_delta_U_global / float(n_global)
    mean_basis_occupancy_like_global = mean_basis_occupancy_like_global / float(n_global)
    mean_basis_occupancy_unlike_global = mean_basis_occupancy_unlike_global / float(n_global)
    mean_n_pairs_global =  mean_n_pairs_global / float(n_global)

    # rebuild aa_stats in the same format your downstream code expects
    K = aa_local["nb_like"]
    M = aa_local["nb_unlike"]
    aa_stats = {
        "mean_grad_sigmoid": mean_g_global[0],
        "mean_grad_like": mean_g_global[1:1+K],
        "mean_grad_unlike": mean_g_global[1+K:],
        "mean_g": mean_g_global,
        "mean_sum_w_like": mean_w_like_global,
        "mean_sum_w_unlike": mean_w_unlike_global,
        "mean_sum_abs_wprime": mean_abs_wprime_global,
        "mean_sum_delta_U": mean_delta_U_global,
        "mean_basis_occupancy_like": mean_basis_occupancy_like_global,
        "mean_basis_occupancy_unlike": mean_basis_occupancy_unlike_global,
        "mean_n_pairs":  mean_n_pairs_global,
        "n_frames": n_global,
        "dim": aa_local["dim"],
        "nb_like": K,
        "nb_unlike": M,
    }

    #t1 = perf_counter()
    #local_time = t1 - t0

    #print(f'local time:{local_time}', flush=True)

    #if (rank == 0):
    #    print('AA STATS FINAL', flush=True)
    #    print(aa_stats)

    cg_local = ensemble_feature_means_from_p_matrix_parallel(cg_traj_local, model_ll, model_ld, sigmoid, nneigh, cutoff, N, True, 1, 1)

    #if (rank == 0):
    #    print('N CG local frames: {int(cg_local["n_frames"])}', flush=True)

    # --- combine per-rank means into global mean ---
    n_cg_local = int(cg_local["n_frames"])

    # convert local means -> local sums
    sum_cg_g_local = cg_local["mean_g"] * float(n_cg_local)   # (D,)
    sum_cg_w_like_local = cg_local["mean_sum_w_like"] * float(n_cg_local)
    sum_cg_w_unlike_local = cg_local["mean_sum_w_unlike"] * float(n_cg_local)
    sum_cg_abs_wprime_local = cg_local["mean_sum_abs_wprime"] * float(n_cg_local)
    sum_cg_delta_U_local = cg_local["mean_sum_delta_U"] * float(n_cg_local)

    mean_cg_basis_occupancy_like_local = cg_local["mean_basis_occupancy_like"] * float(n_cg_local)
    mean_cg_basis_occupancy_unlike_local = cg_local["mean_basis_occupancy_unlike"] * float(n_cg_local)
    mean_cg_n_pairs_local = cg_local["mean_n_pairs"] * float(n_cg_local)

    # global sums across all ranks
    sum_cg_g_global = comm.allreduce(sum_cg_g_local, op=MPI.SUM)
    sum_cg_w_like_global = comm.allreduce(sum_cg_w_like_local, op=MPI.SUM)
    sum_cg_w_unlike_global = comm.allreduce(sum_cg_w_unlike_local, op=MPI.SUM)
    sum_cg_abs_wprime_global = comm.allreduce(sum_cg_abs_wprime_local, op=MPI.SUM)
    sum_cg_delta_U_global = comm.allreduce(sum_cg_delta_U_local, op=MPI.SUM)

    mean_cg_basis_occupancy_like_global = comm.allreduce(mean_cg_basis_occupancy_like_local, op=MPI.SUM)
    mean_cg_basis_occupancy_unlike_global = comm.allreduce(mean_cg_basis_occupancy_unlike_local, op=MPI.SUM)
    mean_cg_n_pairs_global = comm.allreduce(mean_cg_n_pairs_local, op=MPI.SUM)

    n_cg_global = comm.allreduce(n_cg_local, op=MPI.SUM)

    if n_cg_global == 0:
        raise ValueError("No frames processed across all ranks.")

    mean_cg_g_global = sum_cg_g_global / float(n_cg_global)
    mean_cg_w_like_global = sum_cg_w_like_global / float(n_cg_global)
    mean_cg_w_unlike_global = sum_cg_w_unlike_global / float(n_cg_global)
    mean_cg_abs_wprime_global = sum_cg_abs_wprime_global / float(n_cg_global)
    mean_cg_delta_U_global = sum_cg_delta_U_global / float(n_cg_global)
    mean_cg_basis_occupancy_like_global = mean_cg_basis_occupancy_like_global / float(n_cg_global)
    mean_cg_basis_occupancy_unlike_global = mean_cg_basis_occupancy_unlike_global / float(n_cg_global)
    mean_cg_n_pairs_global =  mean_cg_n_pairs_global / float(n_cg_global)
    


    # rebuild aa_stats in the same format your downstream code expects
    K_cg = cg_local["nb_like"]
    M_cg = cg_local["nb_unlike"]
    cg_stats = {
        "mean_grad_sigmoid": mean_cg_g_global[0],
        "mean_grad_like": mean_cg_g_global[1:1+K],
        "mean_grad_unlike": mean_cg_g_global[1+K:],
        "mean_g": mean_cg_g_global,
        "mean_sum_w_like": mean_cg_w_like_global,
        "mean_sum_w_unlike": mean_cg_w_unlike_global,  
        "mean_sum_abs_wprime": mean_cg_abs_wprime_global, 
        "mean_sum_delta_U": mean_cg_delta_U_global,
        "mean_basis_occupancy_like": mean_cg_basis_occupancy_like_global,
        "mean_basis_occupancy_unlike": mean_cg_basis_occupancy_unlike_global,
        "mean_cg_n_pairs": mean_cg_n_pairs_global,
        "n_frames": n_cg_global,
        "dim": cg_local["dim"],
        "nb_like": K_cg,
        "nb_unlike": M_cg,
    }


    print(aa_stats, flush=True)
    print(cg_stats, flush=True)

    #t1 = perf_counter()
    #local_time = t1 - t0

    #print(f'CG local time:{local_time}', flush=True)

    #if (rank == 0):
    #    print('CG STATS FINAL', flush=True)
    #    print(cg_stats)

    #c_like_new, c_unlike_new, sigmoid_new, loss = rem_step(c_like, c_unlike, sigmoid, aa_stats, cg_stats, lr=lr)
    
    c_like_new, c_unlike_new, sigmoid_new, loss =  rem_step(c_like, c_unlike, sigmoid, aa_stats, cg_stats, chi=lr, beta=1/kt_reduced, damping=1e-8, nfix_like_left=7, nfix_like_right=6, nfix_unlike_left=9, nfix_unlike_right=6) 

    #c_like_new, c_unlike_new, sigmoid_new, loss =  rem_step(c_like, c_unlike, sigmoid, aa_stats, cg_stats, chi=lr, beta=1/kt_reduced, damping=1e-8, nfix_like_left=6, nfix_like_right=6, nfix_unlike_left=8, nfix_unlike_right=6) 


    #if (rank == 0):
    #    print(f'c_like_new {c_like_new} c_unlike_new {c_unlike_new} sigmoid_new {sigmoid_new} loss {loss}', flush=True)


    model_ll.c = c_like_new
    model_ld.c = c_unlike_new

    xld = np.arange(0.1,6,0.25)
    xll = np.arange(0.1,6,0.25)

    U_ld_pred =eval_bspline(xld, model_ld.t, model_ld.degree, model_ld.c)
    U_ll_pred =eval_bspline(xll, model_ll.t, model_ll.degree, model_ll.c)

    ull_smoothened = np.zeros_like(U_ll_pred)
    for n in range(1, len(U_ll_pred)-1):
        ull_smoothened[n] = (U_ll_pred[n-1] + U_ll_pred[n] + U_ll_pred[n+1]) / 3
    ull_smoothened[0] = U_ll_pred[0]  # or some other rule
    ull_smoothened[-1] = U_ll_pred[-1]
    ull_smoothened -= ull_smoothened[-1]

    uld_smoothened = np.zeros_like(U_ld_pred)
    for n in range(1, len(U_ld_pred)-1):
        uld_smoothened[n] = (U_ld_pred[n-1] + U_ld_pred[n] + U_ld_pred[n+1]) / 3
    uld_smoothened[0] = U_ld_pred[0]  # or some other rule
    uld_smoothened[-1] = U_ld_pred[-1]
    uld_smoothened -= uld_smoothened[-1]

    ull_force = -np.gradient(ull_smoothened,xll)
    ull_force -= ull_force[-1]
    uld_force = -np.gradient(uld_smoothened,xld)
    uld_force -= uld_force[-1]

    # Save potentials and model and also coefficients (extra)
    os.makedirs(LOGPATH, exist_ok=True)
    write_potentials(xll, ull_smoothened, ull_force, xld, uld_smoothened, uld_force, LOGPATH)
    save_models(model_ll, model_ld, filename=LOGPATH+"/models.pkl")
    save_coeffs(c_like_new, c_unlike_new, sigmoid_new, filename=LOGPATH+"/coeff.pkl") 
    save_loss(loss, filename=LOGPATH+"/loss.pkl")
    

