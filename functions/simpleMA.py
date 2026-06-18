"""simpleMA.py - direct port of Functions/simpleMA.m

Moving-average smoother used to clean synthetic curvature signals.

    s_in : input signal
    N    : number of intervals used for the moving average
    M    : number of times to apply the moving average (default 1)

MATLAB original:
    s_out = s_in;
    for i=1:M
        coeff = ones(1,round(N))/(N);
        s_out = filter(coeff,1,s_out);          % causal FIR
        s_out = circshift(s_out, -round(N/2));  % undo the FIR delay
    end

``filter(b,1,x)`` is reproduced with scipy.signal.lfilter and ``circshift`` with
np.roll. MATLAB ``round`` (half away from zero) is reproduced with floor(x+0.5)
for the positive arguments used here.
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
