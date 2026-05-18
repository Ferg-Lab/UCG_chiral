import numpy as np
from typing import Iterable, Dict, Tuple

import numpy as np
from functools import lru_cache
from typing import Dict


#def p_grad(p, p_neigh):
#    """
#    Compute dp/da for given p and average of p_neigh.
#
#    Parameters
#    ----------
#    p : float
#        p of all i
#    p_neigh : float 
#        Value of average of p_neigh,i
#
#    Returns
#    -------
#    dp_da : float or ndarray
#        Derivative wrt a
#    """
#    z = np.asarray(p_neigh) - 0.5
#    dp_da = p * (1 - p) * z
#    return dp_da

def p_grad(a, p_neigh):
    """
    Compute p(a) and dp/da for the logistic function:
        p = 1 / (1 + exp(-a * (p_neigh - 0.5)))

    Parameters
    ----------
    a : float or ndarray
        Parameter a in the logistic function
    p_neigh : float or ndarray
        Average neighbor value(s)

    Returns
    -------
    p : float or ndarray
        Logistic probability at given a and p_neigh
    dp_da : float or ndarray
        Derivative of p wrt a
    """
    z = np.asarray(p_neigh) - 0.5
    p = 1.0 / (1.0 + np.exp(-a * z))
    dp_da = p * (1 - p) * z
    return p, dp_da



def rho_neigh_from_p(p: np.ndarray,
                     R: np.ndarray,
                     N: int,
                     cutoff: float
                     ):
    """
    Compute rho_neigh[i] = mean_j p[j] over the N closest neighbors of i within 'cutoff'.

    Parameters
    ----------
    p : (M,) array
        p-values for all points i=0..M-1
    R : (M,M) array
        Pairwise distances R_ij (R[i,i] can be 0)
    N : int
        Number of closest neighbors to consider (within cutoff)
    cutoff : float
        Maximum neighbor distance

    Returns
    -------
    rho_neigh : (M,) array
        Average neighbor p for each i
    neigh_idx : list[np.ndarray]
        For each i, the indices of the neighbors used in the average
    """
    p = np.asarray(p, dtype=float).ravel()
    R = np.asarray(R, dtype=float)
    M = p.size
    assert R.shape == (M, M), "R must be (M,M) with same M as p.size"

    # Sort neighbor indices by distance for each i
    idx_sorted = np.argsort(R, axis=1)

    rho_neigh = np.zeros(M) #np.full(M, fill_value, dtype=float)
    neigh_idx = []

    for i in range(M):
        # Skip self: ensure i is not included
        row_sorted = idx_sorted[i]
        row_sorted = row_sorted[row_sorted != i]

        # Apply cutoff, keep only neighbors within cutoff
        within = row_sorted[R[i, row_sorted] <= cutoff]

        # Take N closest among those within cutoff
        chosen = within[:N]
        neigh_idx.append(chosen)

        #if i >= 520 and i < 650:
        #    print(f'chosen: {i} {chosen}')

        if chosen.size != 0:
           #print("Empty!")
            rho_neigh[i] = p[chosen].mean()
        else:
            #print("Empty!")
            rho_neigh[i] = p[i] #.mean()
            #print(p[i])
            #print('END\n')

    return rho_neigh, neigh_idx


