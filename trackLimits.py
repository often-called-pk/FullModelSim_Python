"""trackLimits.py - direct port of Functions/trackLimits.m

Find cartesian coordinates of the left and right track limits from the centre
line (x0,y0) and the width w of the track (scalar or per-point array).

MATLAB original:
    X0 = [x0(:) y0(:)];
    dx = diff(X0);
    n_vec = [-dx(:,2) dx(:,1)]./vecnorm(dx')';
    Xl = X0(1:end-1,:) + abs(w(:)/2).*n_vec;
    Xr = X0(1:end-1,:) - abs(w(:)/2).*n_vec;
    Xl(end+1,:) = 2*Xl(end,:)-Xl(end-1,:);
    Xr(end+1,:) = 2*Xr(end,:)-Xr(end-1,:);
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

    # linear extrapolation of the final point (Xl(end+1,:) = 2*Xl(end,:)-Xl(end-1,:))
    Xl = np.vstack((Xl, 2 * Xl[-1, :] - Xl[-2, :]))
    Xr = np.vstack((Xr, 2 * Xr[-1, :] - Xr[-2, :]))

    return Xl, Xr
