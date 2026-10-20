import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
# Plan Task 4 (docs/superpowers/plans/2026-10-18-physics-fixes.md): A2 suspension (damper
# formula, spring to wheel, smooth stops, no torque cutoff) and A4 aero load path (aero acts
# through the body, not through fzs). Checks marked "corrected" run only with physics='corrected'.
# Plain script: top-level asserts, one PASS line per check.
# Run: python tests/test_physics_suspension_aero.py
import warnings

import numpy as np
import casadi as ca
from functions.context import Ctx
from functions.ggv import aero_coefficients, _load_basis
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


def coast(c):
    """Coast input (Static, ATD On, EM4 Off): zeros, ATD_* = 0.25."""
    u = np.zeros(c.m23.nu)
    u[2:6] = 0.25
    return u


def sig(c, names):
    """g(X, u) -> list of floats: named m attributes in physical units, straight road (kappa = 0)."""
    Xp, Up, K, _, sigs = sym_time(c, names)
    f = ca.Function("sig", [Xp, Up, K], sigs)
    return lambda X, u: [float(v) for v in f.call([X, u, 0.0])]


def dxdt_of(c):
    """g(X, u) -> physical dxdt (nx,), straight road (kappa = 0)."""
    Xp, Up, K, dxdt, _ = sym_time(c)
    f = ca.Function("dxdt", [Xp, Up, K], [dxdt])
    return lambda X, u: np.array(f.call([X, u, 0.0])[0], dtype=float).ravel()


def kin_residual(c):
    """g(X, u) -> d(xs_i)/dt - xsdot_i for the four corners, d/dt taken through dxdt."""
    Xp, Up, K, dxdt, s = sym_time(c, ("xs_fl", "xs_fr", "xs_rl", "xs_rr",
                                     "xsdot_fl", "xsdot_fr", "xsdot_rl", "xsdot_rr"))
    J = ca.jacobian(ca.vertcat(*s[:4]), Xp)
    f = ca.Function("kin", [Xp, Up, K], [ca.mtimes(J, dxdt) - ca.vertcat(*s[4:])])
    return lambda X, u: np.array(f.call([X, u, 0.0])[0], dtype=float).ravel()


def random_state(c, rng):
    """Random state with |xs_i| < 0.04 at every corner: |zs| <= 0.02, |theta|, |phi| <= 0.004
    (lever arms 1.5 m, half track 0.9 m) and |zt_i - xti_i| <= 0.004, so |xs_i| <= 0.034."""
    X = static_state(c, rng.uniform(5.0, 40.0))
    X[1:5] = rng.uniform(-0.2, 0.2, 4)                                          # vy, r, n, eps
    X[5:9] += rng.uniform(-1.0, 1.0, 4)                                         # Om_i
    X[9:15] = rng.uniform(-1.0, 1.0, 6) * [0.02, 0.2, 0.004, 0.1, 0.004, 0.1]   # zs zsdot theta thetadot phi phidot
    X[15:19] = rng.uniform(-0.5, 0.5, 4)                                        # wu_i
    X[19:23] += rng.uniform(-0.004, 0.004, 4)                                   # zt_i
    return X


def block_eigs(c):
    """Eigenvalues of the 14x14 block [9:23, 9:23] of d(dxdt)/dX at static vx = 20, coast input."""
    Xp, Up, K, dxdt, _ = sym_time(c)
    A = ca.Function("A", [Xp, Up, K], [ca.jacobian(dxdt, Xp)])
    M = np.array(A.call([static_state(c, 20.0), coast(c), 0.0])[0], dtype=float)
    return np.linalg.eigvals(M[9:23, 9:23])


def dfzs_dvx(c):
    """g(X, u) -> d(fzs_fl)/d(vx) at X, straight road."""
    Xp, Up, K, _, (fzs,) = sym_time(c, ("fzs_fl",))
    f = ca.Function("dfz", [Xp, Up, K], [ca.jacobian(fzs, Xp)[0]])
    return lambda X, u: float(f.call([X, u, 0.0])[0])


# ---- checks -------------------------------------------------------------------------
cc = model("corrected")
cm = model("matlab")

print("1. A2 sprung corner damping: c_i = 2*zeta_i*sqrt(k_i*m_c_i) (corrected); MATLAB unchanged")
ok("corrected: c_fl == 2*0.7*sqrt(75000*ms*l_r/(2*l)) (rel 1e-12)",
   rel(cc.vp.c_fl, 2 * 0.7 * np.sqrt(75000 * cc.vp.ms * cc.vp.l_r / (2 * cc.vp.l))) < 1e-12)
ok("corrected: c_rl == 2*0.7*sqrt(80000*ms*l_f/(2*l)) (rel 1e-12)",
   rel(cc.vp.c_rl, 2 * 0.7 * np.sqrt(80000 * cc.vp.ms * cc.vp.l_f / (2 * cc.vp.l))) < 1e-12)
ok("matlab: c_fl == 0.7*2*sqrt(m*(1-wB)/2)*75000 (rel 1e-12)",
   rel(cm.vp.c_fl, 0.7 * 2 * np.sqrt(cm.vp.m * (1 - cm.vp.wB) / 2) * 75000) < 1e-12)

print("2. A2 kinematics: xsdot_i == d(xs_i)/dt (corrected); MATLAB xs_fl has no wheel travel")
kc, km = kin_residual(cc), kin_residual(cm)
rng = np.random.default_rng(4)
worst = 0.0
for _ in range(5):
    worst = max(worst, float(np.max(np.abs(kc(random_state(cc, rng), coast(cc))))))
ok(f"corrected: jacobian(xs_i, Xp) @ dxdt == xsdot_i, 5 random states, max error {worst:.1e} (abs 1e-9)",
   worst < 1e-9)
X = static_state(cm, 20.0)
X[15] = 0.5                                                                     # wu_fl = 0.5
mismatch = abs(km(X, coast(cm))[0])
ok(f"matlab, wu_fl = 0.5: xs_fl kinematic mismatch {mismatch:.3f} > 0.1", mismatch > 0.1)

print("3. A2 linear modes at vx = 20 (block [9:23, 9:23])")
ev_c = block_eigs(cc)
ok(f"corrected: max|eig| = {np.max(np.abs(ev_c)):.0f} < 2000", np.max(np.abs(ev_c)) < 2000)
hz = [abs(e.imag) / (2 * np.pi) for e in ev_c if abs(e.imag) > 1e-9]
hz_txt = ", ".join(f"{x:.2f}" for x in hz)
ok(f"corrected: a complex eig at [{hz_txt}] Hz has |Im|/(2 pi) in [1.0, 4.0] Hz",
   any(1.0 <= x <= 4.0 for x in hz))
ev_m = block_eigs(cm)
ok(f"matlab: max|eig| = {np.max(np.abs(ev_m)):.0f} > 5000", np.max(np.abs(ev_m)) > 5000)

print("4. A2 smooth stops (corrected) and no torque cutoff at xs >= 0.075")
fz = sig(cc, ("fzs_fl",))
u0 = coast(cc)
X = static_state(cc, 20.0)
fz_sweep = []
for zs in np.linspace(-0.3, 0.3, 6001):                                         # step 1e-4 m
    X[9] = zs
    fz_sweep.append(fz(X, u0)[0])
fz_sweep = np.array(fz_sweep)
ok("corrected: fzs_fl finite for zs in [-0.3, 0.3]", bool(np.all(np.isfinite(fz_sweep))))
step_max = float(np.max(np.abs(np.diff(fz_sweep))))
ok(f"corrected: max adjacent |dfzs_fl| = {step_max:.1f} N < 500 N", step_max < 500.0)
u_t = coast(cc)
u_t[0], u_t[2] = 100.0, 1.0                                                     # T_motor = 100 Nm, ATD_FL = 1
X = static_state(cc, 20.0)
X[9] = -0.08                                                                    # xs_fl = +0.08 (past 0.075)
T_c = sig(cc, ("T_fl",))(X, u_t)[0]
ok(f"corrected: T_fl = {T_c:.3f} == 100*gear = {100.0 * cc.vp.gear:.3f} at xs_fl = +0.08 (rel 1e-9)",
   rel(T_c, 100.0 * cc.vp.gear) < 1e-9)
X = static_state(cm, 20.0)
X[9] = -0.08
T_m = sig(cm, ("T_fl",))(X, u_t)[0]
ok(f"matlab: T_fl = {T_m:.3g} == 0 at the same state (xs_fl >= 0.075 cutoff)", abs(T_m) <= 1e-12)

print("5. A4 aero load path: corrected fzs_fl has no f_lift_fl; MATLAB fzs_fl has it")
ok("corrected: d fzs_fl / d vx == 0 at static vx = 40 (abs 1e-12)",
   abs(dfzs_dvx(cc)(static_state(cc, 40.0), coast(cc))) <= 1e-12)
ok("matlab: |d fzs_fl / d vx| > 1 at static vx = 40",
   abs(dfzs_dvx(cm)(static_state(cm, 40.0), coast(cm))) > 1.0)

print("6. A4 aero pitch moment (static, coast): dxdt[12] between vx = 10 and vx = 60")
dth_c, dth_m = dxdt_of(cc), dxdt_of(cm)
d_c = dth_c(static_state(cc, 60.0), coast(cc))[12] - dth_c(static_state(cc, 10.0), coast(cc))[12]
d_m = dth_m(static_state(cm, 60.0), coast(cm))[12] - dth_m(static_state(cm, 10.0), coast(cm))[12]
ok(f"corrected: dxdt[12](vx=60) - dxdt[12](vx=10) = {d_c:.3g} > 0", d_c > 0)
ok(f"matlab: dxdt[12](vx=60) - dxdt[12](vx=10) = {d_m:.3g} < 0", d_m < 0)

print("7. A4 ggv load basis: corrected vehModel aero_front == nominal Cl split")
aero_c = aero_coefficients(cc.vp, cc.aero)
front_v = _load_basis(cc.vp, aero_c, "vehModel", physics="corrected")[3]
front_n = _load_basis(cc.vp, aero_c, "nominal")[3]
ok(f"aero_front: vehModel/corrected {front_v:.6f} == nominal {front_n:.6f} (rel 1e-12)",
   rel(front_v, front_n) < 1e-12)

print("ALL physics suspension/aero (Task 4) TESTS PASSED")
