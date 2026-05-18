import os
import numpy as np
import pandas as pd
import pickle
#from scipy.interpolate import CubicSpline
import sys

sys.path.append('../')
from bspline import SplineModel, eval_bspline
from write_potentials import write_potentials, save_models, save_coeffs
from scipy.interpolate import BSpline



def soft_wall(x, x0, V0, dV0):
    n = -dV0 * x0 / V0
    n = np.clip(n, 0.5, 8.0)
    A = V0 * x0**n
    return A / x**n, A, n


# ============================================================
if __name__ == "__main__":

    kt_reduced = 4.6
    iter = 0
    step = 0

    init_guess = 'ibi_guess/prior_model'

    saved_pot_ll = pd.read_csv(init_guess + '/Pair_L-L.table', skiprows=4, 
                               sep='\s+', names=['r', 'u', 'f'])
    saved_pot_dd = pd.read_csv(init_guess + '/Pair_D-D.table', skiprows=4, 
                               sep='\s+', names=['r', 'u', 'f'])
    saved_pot_ld = pd.read_csv(init_guess + '/Pair_L-D.table', skiprows=4, 
                               sep='\s+', names=['r', 'u', 'f'])

    xll = saved_pot_ll.r
    xdd = saved_pot_ll.r
    xld = saved_pot_ld.r

    #print(xll)
    #print(xld)

    U_ll_pred = saved_pot_ll.u.values
    U_ld_pred = saved_pot_ld.u.values

    start_match = 0.7 # 1.2

    r_ext = np.arange(0.1, start_match, 0.1)  

    x0 = start_match
    find_match_index_ld = np.argwhere(np.isclose(xld, start_match, atol=0.06) == True)[0][0]
    V0 = U_ld_pred[find_match_index_ld]
    dV0 = np.gradient(U_ld_pred,xld)[find_match_index_ld]
    U_ld_ext, A_ld, n_ld = soft_wall(r_ext, start_match, V0, dV0)

    x0 = start_match
    find_match_index = np.argwhere(np.isclose(xll, start_match, atol=0.06) == True)[0][0]
    V0 = U_ll_pred[find_match_index]
    dV0 = np.gradient(U_ll_pred, xll)[find_match_index]
    U_ll_ext, A_ll, n_ll = soft_wall(r_ext, start_match, V0, dV0)

    print(f'A_ll:{A_ll: .2f} n_ll:{n_ll: .2f} A_ld:{A_ld: .2f} n_ld:{n_ld: .2f}')

    xll = np.concatenate(( r_ext, xll[find_match_index:]))
    xld = np.concatenate(( r_ext, xld[find_match_index_ld:]))
    U_ll_pred = np.concatenate(( U_ll_ext, U_ll_pred[find_match_index:]))
    U_ld_pred = np.concatenate(( U_ld_ext, U_ld_pred[find_match_index_ld:]))

    U_ld_pred -= U_ld_pred[-1]
    U_ll_pred -= U_ll_pred[-1]


    # Smoothen the potentials using Equation 7 in [http://dx.doi.org/10.1063/1.4880555]
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

    U_ll_pred = ull_smoothened
    U_ld_pred = uld_smoothened

    # Cut at 6.0
    mask = (xll <= 6.0)
    xll = xll[mask]
    xld = xld[mask]
    U_ll_pred = U_ll_pred[mask]
    U_ld_pred = U_ld_pred[mask]


    # Write smoothened potentials before B-spline fit
    ull_force = -np.gradient(U_ll_pred,xll)
    ull_force -= ull_force[-1]
    uld_force = -np.gradient(U_ld_pred,xld)
    uld_force -= uld_force[-1]
    write_potentials(xll, U_ll_pred, ull_force, xld, U_ld_pred, uld_force, 'iter_'+str(step) + '/prior_fit')

    ## Fit new splines
    interior = [0.5, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.5, 4., 4.5, 5.]
    degree = 3
    
    model_ld = SplineModel(xmin=0.1, xmax=6.0, degree=degree, interior_knots=interior)
    model_ld = model_ld.fit(xld, U_ld_pred)

    model_ll = SplineModel(xmin=0.1, xmax=6.0, degree=degree, interior_knots=interior)
    model_ll = model_ll.fit(xll, U_ll_pred)

    xld = np.arange(0.1,6,0.25)#15)
    xll = np.arange(0.1,6,0.25)#15)

    U_ld_pred =eval_bspline(xld, model_ld.t, model_ld.degree, model_ld.c)
    U_ll_pred =eval_bspline(xll, model_ll.t, model_ll.degree, model_ll.c)

    # Write after fit
    ull_force = -np.gradient(U_ll_pred,xll)
    ull_force -= ull_force[-1]
    uld_force = -np.gradient(U_ld_pred,xld)
    uld_force -= uld_force[-1]
    write_potentials(xll, U_ll_pred, ull_force, xld, U_ld_pred, uld_force, 'iter_'+str(step) + '/post_fit')

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
    #ull_smoothened = U_ll_pred
    #uld_smoothened = U_ld_pred

    ull_force = -np.gradient(ull_smoothened,xll)
    ull_force -= ull_force[-1]
    uld_force = -np.gradient(uld_smoothened,xld)
    uld_force -= uld_force[-1]

    # Save potential
    write_potentials(xll, ull_smoothened, ull_force, xld, uld_smoothened, uld_force, 'iter_'+str(step))

    # Save model
    save_models(model_ll, model_ld, filename='iter_'+str(step)+"/models.pkl")    

    # Save coefficients #4.645, 6.872
    save_coeffs(model_ll.c, model_ld.c, 4.645, filename='iter_'+str(step)+"/coeff.pkl")

















