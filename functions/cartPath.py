"""cartPath.py - direct port of Functions/cartPath.m

Retrieve cartesian coordinates of the vehicle trajectory from the centre-line
coordinates (x0,y0) and the signed normal distance to the centre line (n).

MATLAB original:
    X0 = [x0(:) y0(:)];
    dx = diff(X0);
    n_vec = [-dx(:,2) dx(:,1)]./vecnorm(dx')';
    X = X0 + n(:).*[n_vec; n_vec(end,:)];
    x = X(:,1); y = X(:,2);
"""

import numpy as np


def cartPath(x0, y0, n):
    X0 = np.column_stack((np.asarray(x0, float).reshape(-1),
                          np.asarray(y0, float).reshape(-1)))     # [x0(:) y0(:)]
    dx = np.diff(X0, axis=0)                                      # diff(X0)
    norms = np.linalg.norm(dx, axis=1)                            # vecnorm(dx')'
    n_vec = np.column_stack((-dx[:, 1], dx[:, 0])) / norms[:, None]

    # [n_vec; n_vec(end,:)]  -> pad last row so length matches X0
    n_vec_pad = np.vstack((n_vec, n_vec[-1:]))

    nn = np.asarray(n, float).reshape(-1, 1)                      # n(:)
    X = X0 + nn * n_vec_pad

    return X[:, 0], X[:, 1]
