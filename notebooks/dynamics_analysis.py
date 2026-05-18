import numpy as np
from scipy.optimize import curve_fit

def autocorr_ee_direct(ee, max_lag=None):
    """
    Compute normalized autocorrelation C_ee(Δt)

    max_lag: maximum lag (in index units)

    Returns:
        lags (array)
        C (array)
    """
    ee = np.asarray(ee)
    n = len(ee)

    if max_lag is None:
        max_lag = n - 1

    denom = np.mean(ee * ee)   # <ee(t)^2>

    C = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        num = np.mean(ee[:n - lag] * ee[lag:])
        C[lag] = num / denom

    lags = np.arange(max_lag + 1)

    return lags, C


def exp_decay(t, tau):
    return np.exp(-t / tau)

# def fit_tau_nonlinear(time, C, fit_until=0.2):
#     mask = (C > fit_until)
#     t_fit = time[mask]
#     C_fit = C[mask]

#     popt, pcov = curve_fit(exp_decay, t_fit, C_fit, p0=[t_fit[1]])
#     tau = popt[0]
#     tau_err = np.sqrt(np.diag(pcov))[0]
#     return tau, tau_err

def fit_tau_nonlinear(time, C, fit_until=0.2, lag_max=None):

    mask = (C > fit_until)
    if lag_max is not None:
        mask = mask & (time <= lag_max)

    t_fit = time[mask]
    C_fit = C[mask]

    popt, pcov = curve_fit(exp_decay, t_fit, C_fit, p0=[t_fit[1]])
    tau = popt[0]
    tau_err = np.sqrt(np.diag(pcov))[0]
    return tau, tau_err, t_fit, C_fit



def mfpt_direct_core(x, dt, L_state, D_state):
    """
    Compute MFPT from A -> B.

    Parameters
    ----------
    x : array
        ee(t)
    dt : float
        time spacing between samples
    L_state : function
        returns True if x is inside L state
    D_state : function
        returns True if x is inside D state

    Returns
    -------
    mfpt : float
        mean first passage time
    passage_times : array
        all measured first passage times
    """

    x = np.asarray(x)
    n = len(x)

    passage_times = []

    i = 0
    while i < n:

        # wait until trajectory enters L state
        if L_state(x[i]):

            start = i
            i += 1

            # propagate until D_state reached
            while i < n and not D_state(x[i]):
                i += 1

            if i < n:
                t_pass = (i - start) * dt
                passage_times.append(t_pass)

        i += 1

    passage_times = np.array(passage_times)

    mfpt = np.mean(passage_times) if len(passage_times) > 0 else np.nan

    return mfpt, passage_times



