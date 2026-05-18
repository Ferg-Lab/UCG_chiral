import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
from scipy.interpolate import BSpline, make_lsq_spline


def make_clamped_knots_from_interior(xmin: float, xmax: float, degree: int,
                                     interior_knots: Sequence[float]) -> np.ndarray:
    """
    Build a clamped knot vector from explicit *interior* knots.
    interior_knots must be strictly within (xmin, xmax) and strictly increasing.
    """
    interior = np.asarray(interior_knots, dtype=float)
    if interior.size:
        if not (np.all(interior > xmin) and np.all(interior < xmax)):
            raise ValueError("All interior_knots must lie strictly inside (xmin, xmax).")
        if np.any(np.diff(interior) <= 0):
            raise ValueError("interior_knots must be strictly increasing.")
    t = np.concatenate([
        np.full(degree + 1, xmin),
        interior,
        np.full(degree + 1, xmax)
    ])
    return t

# ---------- Spline wrapper ----------
@dataclass
class SplineModel:
    xmin: float
    xmax: float
    degree: int = 3
    interior_knots: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None          # full internal knot vector
    c: Optional[np.ndarray] = None          # coefficients

    @property
    def nbasis(self) -> int:
        if self.t is None:
            return 0
        return len(self.t) - self.degree - 1

    def build_knots(self):
        """
        Establish the knot vector in the following priority:
        assume interior_knots provided: build clamped t from interior;
        """
        self.t = make_clamped_knots_from_interior(self.xmin, self.xmax, self.degree, self.interior_knots)

    def fit(self, x: np.ndarray, y: np.ndarray,
            w: Optional[np.ndarray] = None,
            lam: float = 0.0) -> np.ndarray:
        """
        Fit coefficients by (weighted) least squares 
        Returns coefficients c (length = nbasis) and stores BSpline.
        """
        self.build_knots()

        # Fast path: SciPy LSQ (no penalty)
        interior = self.t #self.t[self.degree+1: -(self.degree+1)]  # SciPy expects interior only
        self.bspline = make_lsq_spline(np.asarray(x, float), np.asarray(y, float), interior, k=self.degree)
        self.c = self.bspline.c
        #self.bspline = BSpline(self.t, self.c, self.degree, extrapolate=False)
        return self # self.c, interior

def bspline_design_matrix(x: np.ndarray, t: np.ndarray, degree: int) -> np.ndarray:
    """
    Design matrix (Jacobian wrt coefficients) for B-spline.

    Parameters
    ----------
    x : array_like, shape (M,)
        Points where the spline is evaluated
    t : array_like
        Full knot vector (with end repeats for clamping)
    degree : int
        Degree of the B-spline

    Returns
    -------
    J : ndarray of shape (M, nbasis)
        J[i, j] = B_j(x_i)
    """
    x = np.asarray(x, dtype=float).ravel()
    nbasis = len(t) - degree - 1
    J = np.empty((x.size, nbasis), dtype=float)

    eye = np.eye(nbasis)
    for j in range(nbasis):
        bj = BSpline(t, eye[j], degree, extrapolate=False)
        J[:, j] = bj(x)   # vectorized evaluation

    return np.nan_to_num(J, 0.0)


def eval_bspline(x: np.ndarray, t: np.ndarray, degree: int, coeffs: np.ndarray) -> np.ndarray:
    """
    Evaluate a B-spline with given coefficients at x.

    Parameters
    ----------
    x : array_like, shape (M,)
        Points where the spline is evaluated
    t : array_like
        Full knot vector (with end repeats for clamping)
    degree : int
        Degree of the B-spline
    coeffs : array_like, shape (nbasis,)
        Coefficients for each basis function

    Returns
    -------
    f : ndarray of shape (M,)
        Values of the spline at x
    """
    Phi = bspline_design_matrix(x, t, degree)  # (M, nbasis)
    coeffs = np.asarray(coeffs, dtype=float).ravel()
    return Phi @ coeffs

