import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
# Plan Task 5 (docs/superpowers/plans/2026-10-18-physics-fixes.md): A5 EM4 motor speed
# (corrected Om_motor_i = Om_i*gear), A6 toe indices (corrected), Steering / CamberGain
# options and the CG_p table. Checks marked "corrected" run only with physics='corrected'.
# Plain script: top-level asserts, one PASS line per check.
# Run: python tests/test_physics_em4_kinematics.py
import warnings

import numpy as np
import casadi as ca
from functions.context import Ctx
from userOpts import userOpts
from vehModel import vehModel


# ---- shared test recipe (plan: "Shared test recipe", verbatim) --------------------
def model(physics="matlab", **kw):          # kw -> userOpts (circuit, ATD, Electric_4Motors, vp_overrides, Steering, CamberGain)
    c = Ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        userOpts(c, **dict(dict(circuit="Sturn", physics=physics), **kw))
    vehModel(c)
    return c

def sym_time(c, names=()):
    """Physical symbols Xp (nx), Up (nu), K; dxdt = physical time derivative (nx); sigs = m attributes."""
    m = c.m23
    xs, us = ca.DM(np.asarray(m.x_s, float)), ca.DM(np.asarray(m.u_s, float))
    F = ca.Function("F", [m.x, m.u, m.pv], [m.dx, m.sf] + [getattr(m, n) for n in names])
    Xp, Up, K = ca.SX.sym("Xp", m.nx), ca.SX.sym("Up", m.nu), ca.SX.sym("K")
    o = F(Xp / xs, Up / us, K)
    return Xp, Up, K, o[0] * xs / o[1], list(o[2:])

def static_state(c, vx):
    vp = c.vp
    X = np.zeros(c.m23.nx)
    X[0] = vx
    X[5:9] = vx / vp.Rw
    X[19:23] = [vp.xti_fl, vp.xti_fr, vp.xti_rl, vp.xti_rr]
    return X
# state order nx=23: 0 vx, 1 vy, 2 r, 3 n, 4 eps, 5-8 Om_fl/fr/rl/rr, 9 zs, 10 zsdot, 11 theta,
# 12 thetadot, 13 phi, 14 phidot, 15-18 wu_fl..rr, 19-22 zt_fl..rr. theta > 0 = nose down,
# phi > 0 lifts left side (fl, rl). xs > 0 = compression. f_lift_i > 0 = downforce.
# inputs (Static, ATD On, EM4 Off, nu=7): T_motor, T_brake, ATD_FL, ATD_FR, ATD_RL, ATD_RR, delta.
# coast input: zeros except ATD_* = 0.25.


# ---- helpers for this file --------------------------------------------------------
def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)))


def sig(c, names):
    """g(X, u) -> list of floats: named m attributes in physical units, straight road (kappa = 0)."""
    Xp, Up, K, _, sigs = sym_time(c, names)
    f = ca.Function("sig", [Xp, Up, K], sigs)
    return lambda X, u: [float(v) for v in f.call([X, u, 0.0])]


TOE = dict(toe_front=0.2, toe_rear=0.1)     # degrees (vehParams primaries)
SA = ("sa_fl", "sa_fr", "sa_rl", "sa_rr")   # tyre slip angles, rad


# ---- checks -------------------------------------------------------------------------
print("1. A5 EM4 motor speed: corrected Om_motor_i = Om_i*gear; MATLAB Om_motor_i = Om_i/gear")
ce = model("corrected", ATD="Off", Electric_4Motors="On")
me = model("matlab", ATD="Off", Electric_4Motors="On")
Omc = sig(ce, ("Om_motor_fl", "Om_fl"))(static_state(ce, 20.0), np.zeros(ce.m23.nu))
Omm = sig(me, ("Om_motor_fl", "Om_fl"))(static_state(me, 20.0), np.zeros(me.m23.nu))
ok("corrected EM4: Om_motor_fl == gear*Om_fl at static vx = 20 (rel 1e-12)",
   rel(Omc[0], ce.vp.gear * Omc[1]) < 1e-12)
ok("matlab EM4: Om_motor_fl == Om_fl/gear at static vx = 20 (rel 1e-12)",
   rel(Omm[0], Omm[1] / me.vp.gear) < 1e-12)

print("2. A6 toe indices, Steering NA: corrected sa_rr uses -toe_rear; MATLAB sa_rr uses -toe_front")
ct = model("corrected", vp_overrides=TOE)
mt = model("matlab", vp_overrides=TOE)
sc = sig(ct, SA)(static_state(ct, 20.0), np.zeros(ct.m23.nu))
sm = sig(mt, SA)(static_state(mt, 20.0), np.zeros(mt.m23.nu))
tf, tr = ct.vp.toe_front_rad, ct.vp.toe_rear_rad
ok("corrected: sa_rl == +toe_rear_rad (abs 1e-12)", abs(sc[2] - tr) <= 1e-12)
ok("corrected: sa_rr == -toe_rear_rad (abs 1e-12)", abs(sc[3] + tr) <= 1e-12)
ok("matlab: sa_rr == -toe_front_rad (abs 1e-12)", abs(sm[3] + tf) <= 1e-12)

print("3. A6 toe indices, Steering Ackermann: corrected front uses toe_front; MATLAB sa_fl uses toe_rear")
ak = model("corrected", vp_overrides=TOE, Steering="Ackermann")
am = model("matlab", vp_overrides=TOE, Steering="Ackermann")
sak = sig(ak, SA)(static_state(ak, 20.0), np.zeros(ak.m23.nu))
sam = sig(am, SA)(static_state(am, 20.0), np.zeros(am.m23.nu))
ok("corrected Ackermann: sa_fl == +toe_front_rad (abs 1e-12)", abs(sak[0] - tf) <= 1e-12)
ok("corrected Ackermann: sa_fr == -toe_front_rad (abs 1e-12)", abs(sak[1] + tf) <= 1e-12)
ok("matlab Ackermann: sa_fl == +toe_rear_rad (abs 1e-12)", abs(sam[0] - tr) <= 1e-12)

print("4. CamberGain='Table': builds under both physics; CG_p table has 14 finite values")
tm = model("matlab", CamberGain="Table")
tc = model("corrected", CamberGain="Table")
ok("CamberGain='Table' builds under matlab and corrected", tm.m23.nx == 23 and tc.m23.nx == 23)
tabs = [np.asarray(c.vp.CG_p_deg_per_deg_table, dtype=float) for c in (tm, tc)]
ok("CG_p_deg_per_deg_table has 14 finite values (matlab and corrected)",
   all(t.size == 14 and bool(np.all(np.isfinite(t))) for t in tabs))

print("5. userOpts Steering / CamberGain options: defaults and validation")
cd = model()
ok("default ctx.Steering == 'NA'", cd.Steering == "NA")
ok("default ctx.CamberGain == 'Off'", cd.CamberGain == "Off")
try:
    model(Steering="x")
    raised = False
except ValueError:
    raised = True
ok("Steering='x' raises ValueError", raised)

print("ALL physics EM4 / kinematics (Task 5) TESTS PASSED")
