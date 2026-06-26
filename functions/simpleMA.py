"""simpleMA.py - port of Functions/simpleMA.m

Moving-average smoother for synthetic curvature signals.

    s_in : input signal
    N    : moving-average window (intervals)
    M    : number of passes (default 1)

Port notes: filter(b,1,x) -> scipy lfilter, circshift -> np.roll, MATLAB round
(half away from zero) -> floor(x+0.5) for the positive args used here.
"""

import numpy as np
from scipy.signal import lfilter


def _matlab_round(x):
    return int(np.floor(float(x) + 0.5))


def simpleMA(s_in, N, M=1):
    s_out = np.asarray(s_in, dtype=float).reshape(-1).copy()
    shift = _matlab_round(N / 2.0)
    n_taps = _matlab_round(N)
    for _ in range(int(M)):
        coeff = np.ones(n_taps) / N                 # ones(1,round(N))/N
        s_out = lfilter(coeff, [1.0], s_out)        # filter(coeff,1,s_out)
        s_out = np.roll(s_out, -shift)              # circshift(s_out,-round(N/2))
    return s_out
