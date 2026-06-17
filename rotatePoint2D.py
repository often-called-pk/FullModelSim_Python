"""rotatePoint2D.py - direct port of Functions/rotatePoint2D.m

Rotate a 2D point P about the origin.

MATLAB original:
    function rotatedPoint = rotatePoint2D(alpha, P)
    P = reshape(P,1,[]);
    s = sin(alpha);  c = cos(alpha);
    R = [c, -s;  s, c];
    rotatedPoint = P * R;

Note: the MATLAB header comment says ``alpha in degrees`` but the code applies
sin/cos directly (i.e. radians). The behaviour is preserved exactly here. In the
only call site (curv2cart) the angle is 0 by default, so R is the identity.
"""

import numpy as np


def rotatePoint2D(alpha, P):
    P = np.asarray(P, dtype=float).reshape(-1)          # P = reshape(P,1,[])
    if np.size(alpha) != 1:
        raise ValueError("Angle of rotation must be a scalar.")
    alpha = float(np.asarray(alpha).reshape(()))

    s = np.sin(alpha)
    c = np.cos(alpha)
    R = np.array([[c, -s],
                  [s,  c]])

    # rotatedPoint = P * R   (1x2 row times 2x2)
    return P @ R
