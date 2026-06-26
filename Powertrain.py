"""Powertrain.py - direct port of Powertrain.m

Defines the powertrain namespace ``pt`` and, as in the MATLAB script, sets
``vp.Rw`` and the derived ``vp.gear`` (final drive ratio from top speed).
Called from userOpts before vehParams, which later overwrites vp.Rw with 0.355
(a different value) but keeps this gear — see the wheel-radius quirk.

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
    pt.Pmax = 3 * 150 * 1000.0          # max continuous power e-motor (W)
    pt.Tmax = 3 * 143.0                 # max continuous torque e-motor (Nm)
    pt.OMmax = 25000 * (np.pi / 30.0)   # max angular velocity e-motor (rad/s)
    pt.Vmax = 380.0 / 3.6               # max top speed (m/s) calculated using vmax = {(2 × Pmax)/(Rho × Cd × A)}^(1/3)
    pt.eff = 0.9                        # efficiency e-motor (-)

    ctx.vp.Rw = 0.35               # wheel radius (m)
    ctx.vp.gear = (pt.OMmax * ctx.vp.Rw) / pt.Vmax   # final drive ratio

    ctx.pt = pt
    return ctx
