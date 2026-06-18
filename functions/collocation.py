"""collocation.py - Legendre collocation helpers.

Provides ``collocation_points(d, scheme)`` and ``collocation_coeff(tau)``,
matching MATLAB's CasADi helpers used by the MLTP transcription:

    tau        = collocation_points(d, 'legendre')   % d points in (0,1)
    [C, D, B]  = collocation_coeff(tau)

with the exact shapes the transcription relies on:
    C : (d+1) x d   differentiation matrix      ->  dPi = Z * C      (Z is nx x (d+1))
    D : (d+1) x 1   continuity / end-point row  ->  Xk_end = Z * D
    B : d x 1       quadrature (Gauss) weights   ->  J += Qk * B * ds  (Qk is 1 x d)

When the installed CasADi exposes the native helpers they are used directly
(guaranteeing identical numerics to the MATLAB run); otherwise a self-contained
Lagrange-polynomial implementation is used. The augmented node set is
``[0, tau_1, ..., tau_d]`` for C and D; the quadrature weights B use the Lagrange
basis over the d collocation points only (the standard Gauss-Legendre weights).
"""

import numpy as np

try:
    import casadi as _ca
    _HAS_CASADI = True
except Exception:                                       # pragma: no cover
    _HAS_CASADI = False


def collocation_points(d, scheme="legendre"):
    """Return the d collocation points in the open interval (0,1)."""
    if _HAS_CASADI:
        return list(_ca.collocation_points(d, scheme))
    if scheme != "legendre":
        raise NotImplementedError(
            "numpy fallback only implements the 'legendre' scheme; "
            "install casadi for 'radau'/'chebyshev'.")
    # Gauss-Legendre nodes on [-1,1] shifted to [0,1]
    x, _ = np.polynomial.legendre.leggauss(d)
    return np.sort((x + 1.0) / 2.0).tolist()


def collocation_coeff(tau):
    """Return (C, D, B) for the collocation points tau (see module docstring)."""
    tau = [float(t) for t in np.asarray(tau).reshape(-1)]
    d = len(tau)

    if _HAS_CASADI:
        C, D, B = _ca.collocation_coeff(tau)
        C = np.array(_ca.DM(C).full(), dtype=float).reshape(d + 1, d)
        D = np.array(_ca.DM(D).full(), dtype=float).reshape(d + 1, 1)
        B = np.array(_ca.DM(B).full(), dtype=float).reshape(d, 1)   # native returns 1xd
        return C, D, B

    # ---- numpy fallback ----------------------------------------------------
    tr = [0.0] + tau                                    # augmented nodes [0, tau]
    C = np.zeros((d + 1, d))
    D = np.zeros((d + 1, 1))
    for j in range(d + 1):
        # Lagrange basis polynomial L_j over the augmented nodes (degree d)
        p = np.poly1d([1.0])
        for r in range(d + 1):
            if r != j:
                p = p * np.poly1d([1.0, -tr[r]]) / (tr[j] - tr[r])
        D[j, 0] = np.polyval(p, 1.0)                    # continuity: L_j(1)
        dp = np.polyder(p)
        for r in range(d):                              # derivative at the d collocation points
            C[j, r] = np.polyval(dp, tr[r + 1])

    B = np.zeros((d, 1))
    for j in range(d):
        # Lagrange basis over the collocation points only (degree d-1) -> Gauss weights
        p = np.poly1d([1.0])
        for r in range(d):
            if r != j:
                p = p * np.poly1d([1.0, -tau[r]]) / (tau[j] - tau[r])
        ip = np.polyint(p)
        B[j, 0] = np.polyval(ip, 1.0)

    return C, D, B
