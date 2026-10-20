"""MF 5.2 scaling factors: peak grip LMUX and LMUY, nominal load LFZO, slip stiffness LKX and LKY (plans
docs/superpowers/plans/2026-10-20-mf-peak-grip-scaling.md, Task 1, and
docs/superpowers/plans/2026-10-20-finish-partials.md, Task 1 (C6)).

Plain script, no pytest. Each check prints PASS or FAIL, and the first FAIL stops the file. Run from the
repo root:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_mf_scaling.py

  1. defaults (MF205 and CopyB give 1.0 for all five), vp_overrides plumbing, unknown key raises
  2. full model, both physics, camber on (no solve):
       mu_*_x ratio 0.9 and mu_*_y ratio 0.8 at 3 operating states,
       peak fx over kappa (alpha 0) ratio 0.9 and peak fy over alpha (kappa 0) ratio 0.8 (bounded search),
       dfx/dkappa and dfy/dalpha at zero slip (central difference, step 1e-6, pHy1 = pHy2 = 0),
       vertical shifts Svx and Svy at zero slip, with the peak zeroed (zero-peak probe),
       camber off and pVy1 = pVy2 = 0: the fy zero crossing alpha = -SHy does not move, and Svyk scales
       (zero-crossing probe)
  2b. full model, both physics, camber on: LFZO 0.9 equals the Fz0 = 0.9 x 4905 override (mu, fx, fy);
       LKX 1.2 scales dfx/dkappa at zero slip and moves no peak, mu_x, mu_y or fy;
       LKY 1.2 scales dfy/dalpha at zero slip (pHy1 = pHy2 = 0) and moves no peak, mu_x, mu_y or fx
  3. QSS screener: peak_mu_x and peak_mu_y scale, the lap moves, factors 1.0 give the same lap;
       LFZO 0.9 gives the Fz0 = 0.9 x 4905 override lap; LKX and LKY leave the lap unchanged
  4. 7-state model: builds with the factors and matches the default build exactly, LFZO, LKX, LKY included
  5. parity export: no LMUX or LMUY while both are 1.0, both exported when one of them is not 1.0;
       the five MF scales are dropped or kept together

Sections 2 to 5 need casadi and Data/DATA_AA.mat. Without DATA_AA they print SKIP and exit 0.
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import sys
import warnings

import numpy as np

from functions.context import Ctx
from vehParams import vehParams


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *args, **kwargs):
    """fn(*args, **kwargs) with stdout swallowed and warnings ignored."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def rel_err(a, b, ratio=1.0):
    """Max relative error of a against ratio * b."""
    a, b = np.asarray(a, float), ratio * np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


# =============================================================================
print("1. defaults and override plumbing")
for tyre_set in ("MF205", "CopyB"):
    c = Ctx()
    quiet(vehParams, c, tyre_set=tyre_set)
    lx, ly = getattr(c.mf, "LMUX", None), getattr(c.mf, "LMUY", None)
    ok(f"{tyre_set}: mf.LMUX and mf.LMUY are float 1.0",
       isinstance(lx, float) and isinstance(ly, float) and lx == 1.0 and ly == 1.0)
    ok(f"{tyre_set}: mf.LFZO, mf.LKX and mf.LKY are float 1.0",
       all(isinstance(getattr(c.mf, k, None), float) and getattr(c.mf, k) == 1.0 for k in ("LFZO", "LKX", "LKY")))
c = Ctx()
quiet(vehParams, c, vp_overrides={"LMUX": 0.9, "LMUY": 1.1})
ok("vp_overrides LMUX 0.9 and LMUY 1.1 land in ctx.mf",
   getattr(c.mf, "LMUX", None) == 0.9 and getattr(c.mf, "LMUY", None) == 1.1)
c = Ctx()
quiet(vehParams, c, vp_overrides={"LFZO": 0.9, "LKX": 1.1, "LKY": 1.2})
ok("vp_overrides LFZO 0.9, LKX 1.1 and LKY 1.2 land in ctx.mf and ctx.mf_overrides",
   (c.mf.LFZO, c.mf.LKX, c.mf.LKY) == (0.9, 1.1, 1.2) and c.mf_overrides == ["LFZO", "LKX", "LKY"])
try:
    quiet(vehParams, Ctx(), vp_overrides={"LMUZ": 1.0})
    raised = False
except ValueError:
    raised = True
ok("unknown override key LMUZ still raises ValueError", raised)

if not os.path.exists(os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")):
    print("SKIP sections 2-5 (test_mf_scaling): Data/DATA_AA.mat missing")
    sys.exit(0)

# model imports come after the DATA_AA guard
import casadi as ca  # noqa: E402
from scipy.optimize import brentq, minimize_scalar  # noqa: E402
from MLTP_screen import MLTP_screen  # noqa: E402
from functions.ggv import peak_mu_x, peak_mu_y  # noqa: E402
from userOpts import userOpts  # noqa: E402
from vehModel import vehModel  # noqa: E402
from vehModel_initial import vehModel_initial  # noqa: E402
import validation.py_export as py_export  # noqa: E402

KW = dict(circuit="Sturn", vi=60.0, AeroConfig="Static", ATD="On", Electric_4Motors="Off", mesh="uniform")
W = ("fl", "fr", "rl", "rr")
CAMBER = {"gamma_fl": -2.0, "gamma_rl": -1.5}                  # deg; vehParams mirrors the right side
FAC = {"LMUX": 0.9, "LMUY": 0.8}
PEAK0 = {"pDx1": 0.0, "pDx2": 0.0, "pDy1": 0.0, "pDy2": 0.0}  # zero peak: the zero-slip force is the shift
SHIFT = {"pVx1": 0.03, "pVx2": 0.02}                          # the default sets have pVx1 = pVx2 = 0
# MF 5.2 does not scale the horizontal shift SHy = pHy1 + pHy2 dfz. With SHy != 0, the slope at zero slip
# changes with By = Kya/(Cy Dy), so the slope probe sets pHy1 = pHy2 = 0. The camber part of SHy is then
# -Svy_gamma/Kya (MF205: pKy6 = pKy7 = 0). Svy_gamma and Dy both scale with LMUY, so By SHy does not change.
HY0 = {"pHy1": 0.0, "pHy2": 0.0}
# Zero-crossing probe, camber off (the default). pVy1 = pVy2 = 0 gives Svy = 0, so at kappa 0 the force fy is
# zero at alpha = -SHy = -(pHy1 + pHy2 dfz), which does not scale. There Fy0 = 0, so at kappa 0.05 fy = Svyk,
# which scales with LMUY through mu_y. The default sets have rVy5 = -8.8e-11 (Svyk near 0): the probe sets 0.05.
CROSS = {"pVy1": 0.0, "pVy2": 0.0, "rVy5": 0.05}
VX0, H = 30.0, 1e-6
ONE = (1.0, 1.0, 1.0, 1.0)
LOADS_B = (1.4, 0.7, 1.3, 0.8)
# three operating points: vx, vy, r, wheel slip per corner, load factor per corner, steer (fraction of u_s)
OPS = (
    dict(vx=30.0, vy=4.2, r=0.2, kap=(0.08, -0.06, 0.04, -0.03), load=LOADS_B, steer=0.02),   # side slip ~0.15 rad
    dict(vx=45.0, vy=-3.0, r=-0.3, kap=(-0.04, 0.02, 0.06, -0.06), load=(0.8, 1.3, 0.9, 1.2), steer=0.0),
    dict(vx=38.0, vy=-5.5, r=0.1, kap=(0.02, -0.02, 0.05, -0.05), load=ONE, steer=-0.02),      # side slip ~-0.15 rad
)


def build(physics, ov):
    """Full 23-state model: userOpts and vehModel."""
    c = Ctx()
    quiet(userOpts, c, physics=physics, vp_overrides=ov, **KW)
    quiet(vehModel, c)
    return c


def state(c, vx, vy=0.0, r=0.0, kap=(0.0, 0.0, 0.0, 0.0), load=ONE, steer=0.0):
    """Normalised (x, u, pv) at one operating point: wheel speeds from the slip ratios, tyre
    deflections from the load factors (static x factor), ATD coast inputs, and steer."""
    m, vp = c.m23, c.vp
    X = np.zeros(m.nx)
    X[0], X[1], X[2] = vx, vy, r
    X[5:9] = vx / vp.Rw * (1.0 + np.asarray(kap, float))
    X[19:23] = np.asarray(load, float) * np.array([vp.xti_fl, vp.xti_fr, vp.xti_rl, vp.xti_rr])
    U = np.zeros(m.nu)
    U[2:6] = 0.25
    U[-1] = steer
    return X / np.asarray(m.x_s, float), U / np.asarray(m.u_s, float), 0.005


def op_states(c):
    return [state(c, o["vx"], o["vy"], o["r"], o["kap"], o["load"], o["steer"]) for o in OPS]


def fn_of(c, names):
    """casadi Function (x, u, pv) -> model signals, built on this model's own graph."""
    m = c.m23
    return ca.Function("F", [m.x, m.u, m.pv], [getattr(m, n) for n in names])


def values(c, names, states):
    """Model signals at each state, shape (number of states, number of names)."""
    F = fn_of(c, names)
    return np.array([[float(v) for v in F(*s)] for s in states])


def sweep(c, kind, load=ONE, kap=0.0):
    """Function s -> the four tyre forces at slip s. kind 'x': fx at kappa = s, alpha 0. kind 'y': fy at
    alpha = s and kappa = kap. vehModel has sa_fl = -atan(vy / vx) at r = 0, so alpha needs vy = -vx tan(alpha)."""
    F = fn_of(c, [f"f{kind}_{w}" for w in W])

    def f(s):
        st = (state(c, VX0, kap=(s,) * 4, load=load) if kind == "x"
              else state(c, VX0, vy=-VX0 * np.tan(s), kap=(kap,) * 4, load=load))
        return np.array([float(v) for v in F(*st)])
    return f


def zero_slip_slopes(c, kind):
    """Central-difference slopes of the four tyre forces at zero slip."""
    f = sweep(c, kind)
    return (f(H) - f(-H)) / (2.0 * H)


def peaks(c, kind):
    """Peak force per corner at the loads LOADS_B: bounded search over kappa 0..0.4 or alpha 0..0.5 rad."""
    f, hi = sweep(c, kind, LOADS_B), (0.4 if kind == "x" else 0.5)
    return np.array([-minimize_scalar(lambda s: -f(s)[i], bounds=(0.0, hi), method="bounded",
                                      options={"xatol": 1e-9}).fun for i in range(4)])


def crossing(c):
    """Per corner at the loads LOADS_B: the alpha where fy = 0 at kappa 0, and fy at that alpha and kappa 0.05."""
    f0, f1 = sweep(c, "y", LOADS_B), sweep(c, "y", LOADS_B, kap=0.05)
    alpha = [brentq(lambda s: f0(s)[i], -0.05, 0.05, xtol=1e-15) for i in range(4)]
    return np.array(alpha), np.array([f1(a)[i] for i, a in enumerate(alpha)])


print("2. full model, both physics, camber on (the zero-crossing probe has camber off)")
for physics in ("matlab", "corrected"):
    c_def, c_fac = build(physics, CAMBER), build(physics, {**CAMBER, **FAC})
    S = op_states(c_def)
    for ax, r in (("x", 0.9), ("y", 0.8)):
        names = [f"mu_{w}_{ax}" for w in W]
        e = rel_err(values(c_fac, names, S), values(c_def, names, S), r)
        ok(f"{physics}: mu_*_{ax} = {r} x default at 3 operating states (max rel err {e:.1e} <= 1e-12)",
           e <= 1e-12)
    # m.mu_* cannot show a factor applied a second time, on D: the peak force can
    for ax, r, over in (("x", 0.9, "kappa, alpha 0"), ("y", 0.8, "alpha, kappa 0")):
        e = rel_err(peaks(c_fac, ax), peaks(c_def, ax), r)
        ok(f"{physics}: peak f{ax} = {r} x default at 4 loads (bounded search over {over}, "
           f"max rel err {e:.1e} <= 1e-6)", e <= 1e-6)

    c_def, c_fac = build(physics, {**CAMBER, **HY0}), build(physics, {**CAMBER, **HY0, **FAC})
    e = rel_err(zero_slip_slopes(c_fac, "x"), zero_slip_slopes(c_def, "x"))
    ok(f"{physics}: dfx/dkappa at zero slip equal to default (central difference h=1e-6, "
       f"max rel err {e:.1e} <= 1e-6)", e <= 1e-6)
    e = rel_err(zero_slip_slopes(c_fac, "y"), zero_slip_slopes(c_def, "y"))
    ok(f"{physics}: dfy/dalpha at zero slip equal to default (pHy1 = pHy2 = 0, central difference h=1e-6, "
       f"max rel err {e:.1e} <= 1e-6)", e <= 1e-6)

    c_def = build(physics, {**CAMBER, **PEAK0, **SHIFT})
    c_fac = build(physics, {**CAMBER, **PEAK0, **SHIFT, **FAC})
    names = [f"fx_{w}" for w in W] + [f"fy_{w}" for w in W]
    Z = [state(c_def, VX0, load=ld) for ld in (ONE, LOADS_B)]
    d, f = values(c_def, names, Z), values(c_fac, names, Z)
    smin_x, smin_y = float(np.min(np.abs(d[:, :4]))), float(np.min(np.abs(d[:, 4:])))
    ok(f"{physics}: zero-peak probe, shifts are nonzero (min |Svx| {smin_x:.1f} N, min |Svy| {smin_y:.1f} N)",
       smin_x > 1.0 and smin_y > 1.0)
    e = rel_err(f[:, :4], d[:, :4], 0.9)
    ok(f"{physics}: vertical shift Svx = 0.9 x default at zero slip (zero-peak probe, "
       f"max rel err {e:.1e} <= 1e-12)", e <= 1e-12)
    e = rel_err(f[:, 4:], d[:, 4:], 0.8)
    ok(f"{physics}: vertical shift Svy = 0.8 x default at zero slip (zero-peak probe, "
       f"max rel err {e:.1e} <= 1e-12)", e <= 1e-12)

    (a_def, v_def), (a_fac, v_fac) = crossing(build(physics, CROSS)), crossing(build(physics, {**CROSS, **FAC}))
    e, amin = rel_err(a_fac, a_def), float(np.min(np.abs(a_def)))
    ok(f"{physics}: fy zero crossing alpha = -SHy equal to default (camber off, pVy1 = pVy2 = 0, kappa 0, "
       f"min |alpha| {amin:.1e} rad, max rel err {e:.1e} <= 1e-9)", amin > 1e-3 and e <= 1e-9)
    e, vmin = rel_err(v_fac, v_def, 0.8), float(np.min(np.abs(v_def)))
    ok(f"{physics}: Svyk = 0.8 x default (fy at that alpha and kappa 0.05, rVy5 = 0.05, min |Svyk| {vmin:.1f} N, "
       f"max rel err {e:.1e} <= 1e-9)", vmin > 1.0 and e <= 1e-9)


print("2b. LFZO, LKX, LKY (C6): full model, both physics, camber on")
for physics in ("matlab", "corrected"):
    c_def = build(physics, CAMBER)
    S = op_states(c_def)
    names = [f"mu_{w}_x" for w in W] + [f"mu_{w}_y" for w in W] + [f"fx_{w}" for w in W] + [f"fy_{w}" for w in W]
    c_l, c_f = build(physics, {**CAMBER, "LFZO": 0.9}), build(physics, {**CAMBER, "Fz0": 0.9 * 4905.0})
    d0, d1 = values(c_def, names, S), values(c_l, names, S)
    e = rel_err(d1, values(c_f, names, S))
    ok(f"{physics}: LFZO = 0.9 equals the Fz0 = 0.9 x 4905 override in mu, fx and fy at 3 states "
       f"(max rel err {e:.1e} <= 1e-12)", e <= 1e-12)
    ok(f"{physics}: LFZO = 0.9 moves mu and the forces (max rel change {np.max(np.abs(d1 / d0 - 1)):.2e} > 1e-3)",
       np.max(np.abs(d1 / d0 - 1)) > 1e-3)

    c_k = build(physics, {**CAMBER, "LKX": 1.2})
    keep = [f"mu_{w}_x" for w in W] + [f"mu_{w}_y" for w in W] + [f"fy_{w}" for w in W]
    e = rel_err(values(c_k, keep, S), values(c_def, keep, S))
    ok(f"{physics}: LKX = 1.2 leaves mu_x, mu_y and fy at 3 states unchanged (max rel err {e:.1e} <= 1e-12)",
       e <= 1e-12)
    e = rel_err(zero_slip_slopes(c_k, "x"), zero_slip_slopes(c_def, "x"), 1.2)
    ok(f"{physics}: LKX = 1.2 gives dfx/dkappa at zero slip = 1.2 x default (central difference h=1e-6, "
       f"max rel err {e:.1e} <= 1e-6)", e <= 1e-6)
    e = rel_err(peaks(c_k, "x"), peaks(c_def, "x"))
    ok(f"{physics}: LKX = 1.2 leaves the peak fx unchanged (bounded search over kappa, "
       f"max rel err {e:.1e} <= 1e-6)", e <= 1e-6)

    c_h = build(physics, {**CAMBER, **HY0})
    c_y = build(physics, {**CAMBER, **HY0, "LKY": 1.2})
    keep = [f"mu_{w}_x" for w in W] + [f"mu_{w}_y" for w in W] + [f"fx_{w}" for w in W]
    e = rel_err(values(c_y, keep, S), values(c_h, keep, S))
    ok(f"{physics}: LKY = 1.2 leaves mu_x, mu_y and fx at 3 states unchanged (max rel err {e:.1e} <= 1e-12)",
       e <= 1e-12)
    e = rel_err(zero_slip_slopes(c_y, "y"), zero_slip_slopes(c_h, "y"), 1.2)
    ok(f"{physics}: LKY = 1.2 gives dfy/dalpha at zero slip = 1.2 x default (pHy1 = pHy2 = 0, central difference "
       f"h=1e-6, max rel err {e:.1e} <= 1e-6)", e <= 1e-6)
    e = rel_err(peaks(c_y, "y"), peaks(c_h, "y"))
    ok(f"{physics}: LKY = 1.2 leaves the peak fy unchanged (bounded search over alpha, "
       f"max rel err {e:.1e} <= 1e-6)", e <= 1e-6)


# =============================================================================
print("3. QSS screener")
c0, c1 = Ctx(), Ctx()
quiet(userOpts, c0, circuit="Sturn")
quiet(userOpts, c1, circuit="Sturn", vp_overrides=FAC)
Fz = np.linspace(800.0, 6000.0, 9)
gam = np.linspace(-0.05, 0.05, 9)
e = rel_err(peak_mu_x(Fz, c1.mf, c1.vp.Fz0), peak_mu_x(Fz, c0.mf, c0.vp.Fz0), 0.9)
ok(f"QSS peak_mu_x = 0.9 x default over Fz 0.8-6 kN (max rel err {e:.1e} <= 1e-12)", e <= 1e-12)
e = rel_err(peak_mu_y(Fz, c1.mf, c1.vp.Fz0, gam), peak_mu_y(Fz, c0.mf, c0.vp.Fz0, gam), 0.8)
ok(f"QSS peak_mu_y = 0.8 x default over Fz and camber (max rel err {e:.1e} <= 1e-12)", e <= 1e-12)


def qss_lap(ov):
    return quiet(MLTP_screen, circuit="Sturn", save=False, verbose=False, vp_overrides=ov).data.lap_time


lap0, lap95, lap1 = qss_lap({}), qss_lap({"LMUX": 0.95, "LMUY": 0.95}), qss_lap({"LMUX": 1.0, "LMUY": 1.0})
ok(f"QSS lap at LMUX = LMUY = 0.95 ({lap95:.3f} s) is longer than default ({lap0:.3f} s)", lap95 > lap0)
ok(f"QSS lap at LMUX = LMUY = 1.0 equals default exactly ({lap1!r} == {lap0!r})", lap1 == lap0)
lapF, lapK = qss_lap({"LFZO": 0.9}), qss_lap({"Fz0": 0.9 * 4905.0})
ok(f"QSS lap at LFZO = 0.9 ({lapF:.5f} s) equals the Fz0 = 0.9 x 4905 override lap ({lapK:.5f} s) "
   f"and differs from default ({lap0:.5f} s)", lapF == lapK and lapF != lap0)
lapX = qss_lap({"LKX": 1.2, "LKY": 1.2})
ok(f"QSS lap at LKX = LKY = 1.2 equals default exactly ({lapX!r} == {lap0!r}): the screener reads the peak only",
   lapX == lap0)


# =============================================================================
print("4. 7-state model")
a, b = Ctx(), Ctx()
for c, ov in ((a, None), (b, FAC)):
    quiet(userOpts, c, circuit="Sturn", vp_overrides=ov)
    quiet(vehModel_initial, c)
ok("7-state build with LMUX 0.9 and LMUY 0.8: ctx.mf carries the factors",
   getattr(b.mf, "LMUX", None) == 0.9 and getattr(b.mf, "LMUY", None) == 0.8)
# the 7-state model has one aux variable, y = ltx (longitudinal load transfer): it is an input too
pts = [(np.array([0.3, 0.05, 0.1, 0.0, 0.01, 0.3, 0.3]), np.array([0.5, 0.0, 0.1]), np.array([0.1]), 0.005),
       (np.array([0.45, -0.02, -0.05, 0.0, -0.02, 0.45, 0.45]), np.array([0.2, 0.3, -0.1]), np.array([-0.2]), 0.01)]
Fa = ca.Function("Fa", [a.m7.x, a.m7.u, a.m7.y, a.m7.pv], [a.m7.dx])
Fb = ca.Function("Fb", [b.m7.x, b.m7.u, b.m7.y, b.m7.pv], [b.m7.dx])
diff = max(float(np.max(np.abs(np.array(Fa(x, u, y, k)) - np.array(Fb(x, u, y, k))))) for x, u, y, k in pts)
ok(f"7-state dx with the factors equals the default build (max abs diff {diff!r} == 0.0)", diff == 0.0)
d = Ctx()
quiet(userOpts, d, circuit="Sturn", vp_overrides={"LFZO": 0.9, "LKX": 1.1, "LKY": 1.2})
quiet(vehModel_initial, d)
Fd = ca.Function("Fd", [d.m7.x, d.m7.u, d.m7.y, d.m7.pv], [d.m7.dx])
diff = max(float(np.max(np.abs(np.array(Fa(x, u, y, k)) - np.array(Fd(x, u, y, k))))) for x, u, y, k in pts)
ok(f"7-state dx with LFZO 0.9, LKX 1.1 and LKY 1.2 equals the default build (max abs diff {diff!r} == 0.0)",
   diff == 0.0)


# =============================================================================
def export(ov):
    """validation.py_export.params with vp_overrides. params() reads its userOpts keyword
    arguments from PARITY, so PARITY gets the key for the call."""
    saved = py_export.PARITY
    py_export.PARITY = {**saved, "vp_overrides": ov}
    try:
        return quiet(py_export.params, "Sturn", 30)
    finally:
        py_export.PARITY = saved


print("5. parity export")
mf0 = export(None)["mf"]
ok("parity export at default: no LMUX or LMUY in the mf block", "LMUX" not in mf0 and "LMUY" not in mf0)
ok("parity export sanity: the mf block holds the MF coefficients (pDy1 present)", "pDy1" in mf0)
ok("parity export at default: no LFZO, LKX or LKY in the mf block", not any(k in mf0 for k in ("LFZO", "LKX", "LKY")))
# both directions: a filter that tests one factor only hides the other factor from gate 1
for ov in ({"LMUY": 0.9}, {"LMUX": 0.9}):
    mf1, want = export(ov)["mf"], {"LMUX": 1.0, "LMUY": 1.0, **ov}
    got = {k: mf1.get(k) for k in want}
    ok(f"parity export, vp_overrides {ov}: the mf block has {got} == {want}", got == want)
for ov in ({"LFZO": 0.9}, {"LKX": 1.1}, {"LKY": 1.2}, {"LMUY": 0.9}):
    mf1 = export(ov)["mf"]
    want = {"LMUX": 1.0, "LMUY": 1.0, "LFZO": 1.0, "LKX": 1.0, "LKY": 1.0, **ov}
    got = {k: mf1.get(k) for k in want}
    ok(f"parity export, vp_overrides {ov}: all five MF scales in the mf block, {got} == {want}", got == want)

print("ALL mf_scaling TESTS PASSED")
