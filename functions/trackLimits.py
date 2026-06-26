"""trackLimits.py - port of Functions/trackLimits.m

Left/right track-limit polylines from centre line (x0,y0) and width w
(scalar or per-point).
"""

import numpy as np


def trackLimits(x0, y0, w):
    X0 = np.column_stack((np.asarray(x0, float).reshape(-1),
                          np.asarray(y0, float).reshape(-1)))
    dx = np.diff(X0, axis=0)
    norms = np.linalg.norm(dx, axis=1)
    n_vec = np.column_stack((-dx[:, 1], dx[:, 0])) / norms[:, None]

    half = np.abs(np.asarray(w, float) / 2.0)            # abs(w(:)/2)
    if half.ndim == 0:
        factor = float(half)                             # scalar broadcast
    else:
        factor = half.reshape(-1)[:n_vec.shape[0]][:, None]

    Xl = X0[:-1, :] + factor * n_vec
    Xr = X0[:-1, :] - factor * n_vec

    # extrapolate the final point: Xl(end+1,:) = 2*Xl(end,:)-Xl(end-1,:)
    Xl = np.vstack((Xl, 2 * Xl[-1, :] - Xl[-2, :]))
    Xr = np.vstack((Xr, 2 * Xr[-1, :] - Xr[-2, :]))

    return Xl, Xr
