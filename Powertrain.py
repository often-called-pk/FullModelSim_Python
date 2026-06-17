"""Powertrain.py - direct port of Powertrain.m

Defines the powertrain parameter namespace ``pt`` and, exactly as the MATLAB
script does, sets ``vp.Rw`` (wheel radius) and the derived ``vp.gear`` (final
drive ratio from the top-speed requirement). Called from userOpts before
vehParams (which re-asserts the same vp.Rw value).

MATLAB original:
    pt.Pmax  = 2*228*1000;        % max continuous power e-motor (W)   [2x SPM242-176]
    pt.Tmax  = 2*301;             % max continuous torque e-motor (Nm)
    pt.OMmax = 17750*(pi/30);     % max angular velocity e-motor (rad/s)
    pt.Vmax  = 290/3.6;           % required top speed (m/s) - used for gear selection
    pt.eff   = 0.9;               % efficiency e-motor (-)
    vp.Rw    = 0.3142857;         % wheel radius (m)
    vp.gear  = (pt.OMmax*vp.Rw)/pt.Vmax;   % final drive ratio
"""

import numpy as np
from types import SimpleNamespace


def Powertrain(ctx):
    if not hasattr(ctx, "vp") or ctx.vp is None:
        ctx.vp = SimpleNamespace()

    pt = SimpleNamespace()
    pt.Pmax = 2 * 228 * 1000.0          # max continuous power e-motor (W)
    pt.Tmax = 2 * 301.0                 # max continuous torque e-motor (Nm)
    pt.OMmax = 17750 * (np.pi / 30.0)   # max angular velocity e-motor (rad/s)
    pt.Vmax = 290.0 / 3.6               # required top speed (m/s)
    pt.eff = 0.9                        # efficiency e-motor (-)

    ctx.vp.Rw = 0.3142857               # wheel radius (m)
    ctx.vp.gear = (pt.OMmax * ctx.vp.Rw) / pt.Vmax   # final drive ratio

    ctx.pt = pt
    return ctx
