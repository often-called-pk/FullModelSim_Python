"""curv2cart.py - direct port of Functions/curv2cart.m

Reconstruct a track in cartesian coordinates (x,y) from curvilinear data (s,k).

    s     : cumulative distance along the curve (m)
    k     : curvature, with sign (1/m)
    o     : +1 or -1. By default (o=+1) k>0 is a left-hand turn; o=-1 reverses it.
    theta : orientation of the first segment in the cartesian plane (default 0).

Faithful translation of the MATLAB loop, including the 1e-4 straight-line
threshold and the chord/angle update used on curved segments.
"""

import numpy as np
from .rotatePoint2D import rotatePoint2D


def _pol2cart(theta, rho):
    """MATLAB pol2cart: (theta, rho) -> (x, y)."""
    return rho * np.cos(theta), rho * np.sin(theta)


def curv2cart(s, k, o=1, theta=0.0):
    thr = 1e-4  # threshold for the curvature to be considered a straight

    if o == -1:
        k = -np.asarray(k, dtype=float)
    else:
        k = np.asarray(k, dtype=float)

    s = np.asarray(s, dtype=float).reshape(-1)          # s = s(:)'
    N = s.size
    ds = np.concatenate(([0.0], np.diff(s)))            # ds = [0 diff(s)]

    X = np.zeros((N, 2))
    X[0, :] = [0.0, 0.0]                                # X(1,:) = [0 0]
    if N > 1:
        X[1, :] = rotatePoint2D(theta, np.array([s[1], 0.0]))  # X(2,:) = rotatePoint2D(theta,[s(2) 0])

    # MATLAB: for i = 3:N  ->  python index j = i-1 spans 2 .. N-1
    for j in range(2, N):
        if abs(k[j - 1]) < thr:                         # straights (k(i-1))
            d = X[j - 1, :] - X[j - 2, :]
            X[j, :] = X[j - 1, :] + d / np.linalg.norm(d) * ds[j]
        else:
            a = np.arcsin(k[j - 1] * ds[j - 1] / 2.0) + np.arcsin(k[j - 1] * ds[j] / 2.0)
            b = np.arctan2(X[j - 1, 1] - X[j - 2, 1], X[j - 1, 0] - X[j - 2, 0])
            u, v = _pol2cart(b + a, ds[j])
            X[j, :] = X[j - 1, :] + np.array([u, v])

    x = X[:, 0]
    y = X[:, 1]
    return x, y
