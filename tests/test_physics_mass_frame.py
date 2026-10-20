"""Physics A1 (mass bookkeeping) and A3 (frame terms): plan Task 3, tests 1-7.

physics='corrected' (opt-in) changes the MATLAB mass and frame terms:
  A1  per-corner unsprung mass muf/2 (front) and mur/2 (rear), planar m, lever-rule static
      corner loads Wi0 and spring preloads xsi_i (sprung front share l_r/l);
  A3  body equations without the attitude coupling: no g sin(theta) in dvx, no
      g sin(phi) cos(theta) in dvy, no thetadot*vx or phidot*vy in dzsdot.
physics='matlab' (default) keeps the MATLAB terms; tests 2-5 pin them.
Test 6 runs the QSS screen (MLTP_screen, no NLP solve). Test 7 checks quasi_static_states.
Derivatives are symbolic (casadi) at the state, on a straight line (kappa = 0), coast input.
Needs casadi and Data/DATA_AA.mat (prints SKIP and exits 0 otherwise). Run from the repo root:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_physics_mass_frame.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import os
import sys
import warnings

import numpy as np

DATA = os.path.join(_bootstrap.ROOT, "Data")
try:
    import casadi as ca
except ImportError as exc:                                      # pragma: no cover
    print(f"SKIP test_physics_mass_frame: casadi not importable ({exc})")
    sys.exit(0)
if not os.path.exists(os.path.join(DATA, "DATA_AA.mat")):       # pragma: no cover
    print("SKIP test_physics_mass_frame: Data/DATA_AA.mat missing")
    sys.exit(0)

from functions.context import Ctx
from functions.ladder import quasi_static_states
from MLTP_screen import MLTP_screen
from userOpts import userOpts
from vehModel import vehModel


# ---- shared test recipe (plan "Shared test recipe"), verbatim ----------------
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
# ------------------------------------------------------------------------------


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def close(a, b, rel_tol=0.0, abs_tol=0.0):
    """|a - b| <= max(abs_tol, rel_tol * |b|)."""
    return abs(a - b) <= max(abs_tol, rel_tol * abs(b))


COAST = np.array([0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0])   # T_motor, T_brake, ATD_FL..RR, delta


def evaluate(c, X, U, names=()):
    """(dxdt, {name: value}) of c.m23 at state X, input U, kappa 0."""
    Xp, Up, K, dxdt, sig = sym_time(c, names)
    f = ca.Function("f", [Xp, Up, K], [dxdt] + sig)
    res = f(np.asarray(X, float).reshape(-1, 1), np.asarray(U, float).reshape(-1, 1), 0.0)
    res = res if isinstance(res, tuple) else (res,)
    vals = [np.array(r, dtype=float).reshape(-1) for r in res]
    return vals[0], {n: vals[i + 1][0] for i, n in enumerate(names)}


def jacobian(c, X, U):
    """d dxdt / d x, 23 x 23, at state X, input U, kappa 0."""
    Xp, Up, K, dxdt, _ = sym_time(c)
    f = ca.Function("J", [Xp, Up, K], [ca.jacobian(dxdt, Xp)])
    return np.array(f(np.asarray(X, float).reshape(-1, 1), np.asarray(U, float).reshape(-1, 1), 0.0))


def hessian_zsdot(c, X, U):
    """d2 dxdt[10] / (d x_i d x_j), 23 x 23, at state X, input U, kappa 0."""
    Xp, Up, K, dxdt, _ = sym_time(c)
    f = ca.Function("H", [Xp, Up, K], [ca.jacobian(ca.jacobian(dxdt[10], Xp), Xp)])
    return np.array(f(np.asarray(X, float).reshape(-1, 1), np.asarray(U, float).reshape(-1, 1), 0.0))


def attitude_state(c):
    """vx=30, vy=0.3, r=0.1, theta=0.01, phi=0.02; zt = xti and Om = vx/Rw from static_state."""
    X = static_state(c, 30.0)
    X[1], X[2], X[11], X[13] = 0.3, 0.1, 0.01, 0.02
    return X


c_c = model("corrected")
c_m = model("matlab")
vp_c, vp_m = c_c.vp, c_m.vp
# Static checks (tests 1, 2, 7) need a tyre with no force at zero slip: MF205 has horizontal
# and vertical shifts (pHy1/2, pVy1/2), about 390 N of side force per corner at zero slip.
NOSHIFT = {"pHy1": 0.0, "pHy2": 0.0, "pVy1": 0.0, "pVy2": 0.0}
c_c0 = model("corrected", vp_overrides=NOSHIFT)
c_m0 = model("matlab", vp_overrides=NOSHIFT)

# =============================================================================
print("1. corrected, static at vx=0.5, coast: wheels, body rates and tyre loads (A1)")
d, sg = evaluate(c_c0, static_state(c_c0, 0.5), COAST, ("fz_fl", "fz_fr", "fz_rl", "fz_rr"))
ok(f"|dxdt[10]| = {abs(d[10]):.2e} < 1e-3 (zsdot rate)", abs(d[10]) < 1e-3)
ok(f"|dxdt[12]| = {abs(d[12]):.2e} < 1e-3 (thetadot rate)", abs(d[12]) < 1e-3)
ok(f"|dxdt[14]| = {abs(d[14]):.2e} < 1e-3 (phidot rate)", abs(d[14]) < 1e-3)
ok(f"max |dxdt[15:19]| = {np.max(np.abs(d[15:19])):.2e} < 1e-3 (wheel rates)",
   np.all(np.abs(d[15:19]) < 1e-3))
fz = np.array([sg["fz_fl"], sg["fz_fr"], sg["fz_rl"], sg["fz_rr"]])
ok(f"sum fz = {fz.sum():.3f} N == m g = {vp_c.m * vp_c.g:.3f} N (rel 1e-3)",
   close(fz.sum(), vp_c.m * vp_c.g, rel_tol=1e-3))
share = (fz[0] + fz[1]) / fz.sum()
front = (vp_c.muf + vp_c.ms * vp_c.l_r / vp_c.l) / vp_c.m          # front unsprung + lever-rule sprung
ok(f"(fz_fl + fz_fr)/sum = {share:.5f} == (muf + ms l_r/l)/m = {front:.5f} (abs 1e-3)",
   close(share, front, abs_tol=1e-3))

# =============================================================================
print("2. matlab, same static state: wheel rate follows the MATLAB formula, strongly negative")
dm, _ = evaluate(c_m0, static_state(c_m0, 0.5), COAST)
expect = (vp_m.kt * vp_m.xti_fl - vp_m.mus * vp_m.g - vp_m.k_fl * vp_m.xsi_fl) / vp_m.mus
ok(f"dxdt[15] = {dm[15]:.4f} == (kt xti_fl - mus g - k_fl xsi_fl)/mus = {expect:.4f} (abs 1e-3)",
   close(dm[15], expect, abs_tol=1e-3))
ok(f"dxdt[15] = {dm[15]:.4f} < -5 (quirk pinned)", dm[15] < -5)

# =============================================================================
print("3. A1 force balance, coast: sum of tyre forces = m (corrected) or ms (matlab)")
FX = ("fx_fl", "fx_fr", "fx_rl", "fx_rr", "f_drag", "f_roll")
FY = ("fy_fl", "fy_fr", "fy_rl", "fy_rr", "f_side")

d, s = evaluate(c_c, static_state(c_c, 30.0), COAST, FX)
r = (sum(s[n] for n in FX[:4]) - s["f_drag"] - s["f_roll"]) / d[0]
ok(f"corrected, vx=30: (sum fx - f_drag - f_roll)/dxdt[0] = {r:.6f} == m = {vp_c.m:.6f} (rel 1e-9)",
   close(r, vp_c.m, rel_tol=1e-9))
X = static_state(c_c, 30.0)
X[1] = 0.5
d, s = evaluate(c_c, X, COAST, FY)
r = (sum(s[n] for n in FY[:4]) + s["f_side"]) / d[1]
ok(f"corrected, vx=30, vy=0.5, r=0: (sum fy + f_side)/dxdt[1] = {r:.6f} == m = {vp_c.m:.6f} (rel 1e-9)",
   close(r, vp_c.m, rel_tol=1e-9))

d, s = evaluate(c_m, static_state(c_m, 30.0), COAST, FX)
r = (sum(s[n] for n in FX[:4]) - s["f_drag"] - s["f_roll"]) / d[0]
ok(f"matlab, vx=30: (sum fx - f_drag - f_roll)/dxdt[0] = {r:.6f} == ms = {vp_m.ms:.6f} (rel 1e-9)",
   close(r, vp_m.ms, rel_tol=1e-9))
X = static_state(c_m, 30.0)
X[1] = 0.5
d, s = evaluate(c_m, X, COAST, FY)
r = (sum(s[n] for n in FY[:4]) + s["f_side"]) / d[1]
ok(f"matlab, vx=30, vy=0.5, r=0: (sum fy + f_side)/dxdt[1] = {r:.6f} == ms = {vp_m.ms:.6f} (rel 1e-9)",
   close(r, vp_m.ms, rel_tol=1e-9))

# =============================================================================
print("4. A3 attitude: theta enters dvx and phi enters dvy only through gravity")
Jc = jacobian(c_c, attitude_state(c_c), COAST)
Jm = jacobian(c_m, attitude_state(c_m), COAST)
ok(f"corrected: d dxdt[0]/d theta = {Jc[0, 11]:.3e} == 0 (abs 1e-12)", abs(Jc[0, 11]) <= 1e-12)
ok(f"corrected: d dxdt[1]/d phi = {Jc[1, 13]:.3e} == 0 (abs 1e-12)", abs(Jc[1, 13]) <= 1e-12)
ok(f"matlab: d dxdt[0]/d theta = {Jm[0, 11]:.4f} > 9", Jm[0, 11] > 9)
ok(f"matlab: d dxdt[1]/d phi = {Jm[1, 13]:.4f} < -9", Jm[1, 13] < -9)

# =============================================================================
print("5. A3 second-order terms in dzsdot: thetadot*vx and phidot*vy")
Hc = hessian_zsdot(c_c, attitude_state(c_c), COAST)
Hm = hessian_zsdot(c_m, attitude_state(c_m), COAST)
ok(f"corrected: d2 dxdt[10]/(d vx d thetadot) = {Hc[0, 12]:.3e} == 0 (abs 1e-9)",
   abs(Hc[0, 12]) <= 1e-9)
ok(f"corrected: d2 dxdt[10]/(d vy d phidot) = {Hc[1, 14]:.3e} == 0 (abs 1e-9)",
   abs(Hc[1, 14]) <= 1e-9)
ok(f"matlab: d2 dxdt[10]/(d vx d thetadot) = {Hm[0, 12]:.6f} == +1 (abs 1e-9)",
   abs(Hm[0, 12] - 1.0) <= 1e-9)
ok(f"matlab: d2 dxdt[10]/(d vy d phidot) = {Hm[1, 14]:.6f} == -1 (abs 1e-9)",
   abs(Hm[1, 14] + 1.0) <= 1e-9)

# =============================================================================
print("6. QSS screen, Sturn: corrected lap / matlab lap in [1.07, 1.13]")
lap_m = MLTP_screen(circuit="Sturn", save=False, verbose=False).data.lap_time
lap_c = MLTP_screen(circuit="Sturn", save=False, verbose=False, physics="corrected").data.lap_time
ratio = lap_c / lap_m
ok(f"lap corrected/matlab = {ratio:.4f} ({lap_c:.3f} s / {lap_m:.3f} s) in [1.07, 1.13]",
   1.07 <= ratio <= 1.13)

# =============================================================================
print("7. quasi_static_states(vp, 20, physics='corrected') is a corrected equilibrium")
q = quasi_static_states(c_c0.vp, 20.0, physics="corrected")[:, 0]  # rows 5-22 of the state
X7 = np.concatenate([[20.0, 0.0, 0.0, 0.0, 0.0], q])
d7, _ = evaluate(c_c0, X7, COAST)
ok(f"corrected: max |dxdt[15:19]| = {np.max(np.abs(d7[15:19])):.3f} < 0.05",
   np.all(np.abs(d7[15:19]) < 0.05))

print("ALL physics mass/frame TESTS PASSED")
