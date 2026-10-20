"""MF 5.2 peak-grip scaling factors LMUX and LMUY (plan
docs/superpowers/plans/2026-10-20-mf-peak-grip-scaling.md, Task 1).

Plain script, no pytest. Each check prints PASS or FAIL. The file prints the PASS and FAIL
count per section and exits 1 when any check fails. Run from the repo root:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_mf_scaling.py

  1. defaults (MF205 and CopyB give 1.0), vp_overrides plumbing, unknown key raises
  2. full model, both physics, camber on (no solve):
       mu_*_x ratio 0.9 and mu_*_y ratio 0.8 at 3 operating states,
       Kxk and Kya equal to default (only if vehModel exposes them on ctx.m23, else SKIP),
       dfx/dkappa and dfy/dalpha at zero slip (central difference, step 1e-6, pHy1 = pHy2 = 0),
       vertical shifts Svx and Svy at zero slip, with the peak zeroed (zero-peak probe)
  3. QSS screener: peak_mu_x and peak_mu_y scale, the lap moves, factors 1.0 give the same lap
  4. 7-state model: builds with the factors and matches the default build exactly
  5. parity export: no LMUX or LMUY at 1.0, mf.LMUY exported at 0.9

Sections 2 to 5 need casadi and Data/DATA_AA.mat. Without DATA_AA they print SKIP and exit 0.
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import sys
import traceback
import warnings

import numpy as np

from functions.context import Ctx
from vehParams import vehParams

HAVE_DATA = os.path.exists(os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat"))
W = ("fl", "fr", "rl", "rr")
RESULTS = {}      # section -> [PASS count, FAIL count]
FAILS = []        # (section, check name)
SKIPS = []        # (section, note)
_SEC = ["1"]
_CACHE = {}       # built models, shared by the checks of one run


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    RESULTS.setdefault(_SEC[0], [0, 0])[0 if cond else 1] += 1
    if not cond:
        FAILS.append((_SEC[0], name))


def skip(note):
    print(f"  [SKIP] {note}")
    SKIPS.append((_SEC[0], note))


def section(num, title):
    _SEC[0] = num
    print(f"{num}. {title}")


def guard(label, fn, *args):
    """Run one group of checks. An exception fails that group only, the others still run."""
    try:
        fn(*args)
    except Exception as exc:
        traceback.print_exc()
        ok(f"{label} ran to the end ({type(exc).__name__}: {exc})", False)


def quiet(fn, *args, **kwargs):
    """fn(*args, **kwargs) with stdout swallowed and warnings ignored."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def memo(key, make):
    """Build once per key."""
    if key not in _CACHE:
        _CACHE[key] = quiet(make)
    return _CACHE[key]


def items(ov):
    return tuple(sorted((ov or {}).items()))


def rel_err(a, b, ratio=1.0):
    """Max relative error of a against ratio * b."""
    a, b = np.asarray(a, float), ratio * np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def report():
    print("summary, PASS and FAIL per section:")
    for sec in sorted(RESULTS):
        p, f = RESULTS[sec]
        print(f"  section {sec}: {p} PASS, {f} FAIL")
    for sec, note in SKIPS:
        print(f"  SKIP (section {sec}): {note}")
    if FAILS:
        for sec, name in FAILS:
            print(f"  FAIL (section {sec}): {name}")
        print(f"{len(FAILS)} FAIL")
        sys.exit(1)


# =============================================================================
def sec1():
    for tyre_set in ("MF205", "CopyB"):
        c = Ctx()
        quiet(vehParams, c, tyre_set=tyre_set)
        lx, ly = getattr(c.mf, "LMUX", None), getattr(c.mf, "LMUY", None)
        ok(f"{tyre_set}: mf.LMUX and mf.LMUY are float 1.0",
           isinstance(lx, float) and isinstance(ly, float) and lx == 1.0 and ly == 1.0)
    c = Ctx()
    quiet(vehParams, c, vp_overrides={"LMUX": 0.9, "LMUY": 1.1})
    ok("vp_overrides LMUX 0.9 and LMUY 1.1 land in ctx.mf",
       getattr(c.mf, "LMUX", None) == 0.9 and getattr(c.mf, "LMUY", None) == 1.1)
    try:
        quiet(vehParams, Ctx(), vp_overrides={"LMUZ": 1.0})
        raised = False
    except ValueError:
        raised = True
    ok("unknown override key LMUZ still raises ValueError", raised)


section("1", "defaults and override plumbing")
guard("section 1 checks", sec1)

if not HAVE_DATA:
    print("SKIP sections 2-5 (test_mf_scaling): Data/DATA_AA.mat missing")
    report()
    sys.exit(0)

# model imports come after the DATA_AA guard
import casadi as ca  # noqa: E402
from MLTP_screen import MLTP_screen  # noqa: E402
from functions.ggv import peak_mu_x, peak_mu_y  # noqa: E402
from userOpts import userOpts  # noqa: E402
from vehModel import vehModel  # noqa: E402
from vehModel_initial import vehModel_initial  # noqa: E402
import validation.py_export as py_export  # noqa: E402

KW = dict(circuit="Sturn", vi=60.0, AeroConfig="Static", ATD="On", Electric_4Motors="Off", mesh="uniform")
CAMBER = {"gamma_fl": -2.0, "gamma_rl": -1.5}                  # deg; vehParams mirrors the right side
FAC = {"LMUX": 0.9, "LMUY": 0.8}
PEAK0 = {"pDx1": 0.0, "pDx2": 0.0, "pDy1": 0.0, "pDy2": 0.0}  # zero peak: the zero-slip force is the shift
SHIFT = {"pVx1": 0.03, "pVx2": 0.02}                          # the default sets have pVx1 = pVx2 = 0
# MF 5.2 does not scale the horizontal shift SHy = pHy1 + pHy2 dfz. With SHy != 0, the slope at zero slip
# changes with By = Kya/(Cy Dy), so the slope probe sets pHy1 = pHy2 = 0. The camber part of SHy is then
# -Svy_gamma/Kya (MF205: pKy6 = pKy7 = 0). Svy_gamma and Dy both scale with LMUY, so By SHy does not change.
HY0 = {"pHy1": 0.0, "pHy2": 0.0}
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
    def make():
        c = Ctx()
        userOpts(c, physics=physics, vp_overrides=ov, **KW)
        vehModel(c)
        return c
    return memo(("full", physics, items(ov)), make)


def screen_ctx(ov):
    def make():
        c = Ctx()
        userOpts(c, circuit="Sturn", vp_overrides=ov)
        return c
    return memo(("screen", items(ov)), make)


def ctx7(ov):
    def make():
        c = Ctx()
        userOpts(c, circuit="Sturn", vp_overrides=ov)
        vehModel_initial(c)
        return c
    return memo(("init7", items(ov)), make)


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


def zero_slip_slopes(c, kind):
    """Central-difference slopes of the four tyre forces at zero slip. kind 'x': dfx/dkappa through
    the wheel speeds. kind 'y': dfy/dalpha through the side slip. vehModel has sa_fl = -atan(vy / vx)
    at r = 0, so the slip angle alpha needs vy = -vx tan(alpha)."""
    F = fn_of(c, [f"f{kind}_{w}" for w in W])

    def f(s):
        st = state(c, VX0, kap=(s,) * 4) if kind == "x" else state(c, VX0, vy=-VX0 * np.tan(s))
        return np.array([float(v) for v in F(*st)])

    return (f(H) - f(-H)) / (2.0 * H)


def sec2_mu(physics):
    c_def, c_fac = build(physics, CAMBER), build(physics, {**CAMBER, **FAC})
    S = op_states(c_def)
    mux, muy = [f"mu_{w}_x" for w in W], [f"mu_{w}_y" for w in W]
    e = rel_err(values(c_fac, mux, S), values(c_def, mux, S), 0.9)
    ok(f"{physics}: mu_*_x = 0.9 x default at 3 operating states (max rel err {e:.1e} <= 1e-12)", e <= 1e-12)
    e = rel_err(values(c_fac, muy, S), values(c_def, muy, S), 0.8)
    ok(f"{physics}: mu_*_y = 0.8 x default at 3 operating states (max rel err {e:.1e} <= 1e-12)", e <= 1e-12)


def sec2_kxk_kya(physics):
    c_def, c_fac = build(physics, CAMBER), build(physics, {**CAMBER, **FAC})
    if not all(hasattr(c_def.m23, f"{k}_{w}") for k in ("Kxk", "Kya") for w in W):
        skip(f"{physics}: Kxk and Kya are not on ctx.m23 (vehModel keeps them local), equality not checked")
        return
    S = op_states(c_def)
    for k in ("Kxk", "Kya"):
        names = [f"{k}_{w}" for w in W]
        e = rel_err(values(c_fac, names, S), values(c_def, names, S))
        ok(f"{physics}: {k} equal to default at 3 operating states (max rel err {e:.1e} <= 1e-12)", e <= 1e-12)


def sec2_slopes(physics):
    c_def, c_fac = build(physics, {**CAMBER, **HY0}), build(physics, {**CAMBER, **HY0, **FAC})
    e = rel_err(zero_slip_slopes(c_fac, "x"), zero_slip_slopes(c_def, "x"))
    ok(f"{physics}: dfx/dkappa at zero slip equal to default (central difference h=1e-6, "
       f"max rel err {e:.1e} <= 1e-6)", e <= 1e-6)
    e = rel_err(zero_slip_slopes(c_fac, "y"), zero_slip_slopes(c_def, "y"))
    ok(f"{physics}: dfy/dalpha at zero slip equal to default (pHy1 = pHy2 = 0, central difference h=1e-6, "
       f"max rel err {e:.1e} <= 1e-6)", e <= 1e-6)


def sec2_shifts(physics):
    p_def = build(physics, {**CAMBER, **PEAK0, **SHIFT})
    p_fac = build(physics, {**CAMBER, **PEAK0, **SHIFT, **FAC})
    names = [f"fx_{w}" for w in W] + [f"fy_{w}" for w in W]
    Z = [state(p_def, VX0, load=ld) for ld in (ONE, LOADS_B)]
    d, f = values(p_def, names, Z), values(p_fac, names, Z)
    smin_x, smin_y = float(np.min(np.abs(d[:, :4]))), float(np.min(np.abs(d[:, 4:])))
    ok(f"{physics}: zero-peak probe, shifts are nonzero (min |Svx| {smin_x:.1f} N, min |Svy| {smin_y:.1f} N)",
       smin_x > 1.0 and smin_y > 1.0)
    e = rel_err(f[:, :4], d[:, :4], 0.9)
    ok(f"{physics}: vertical shift Svx = 0.9 x default at zero slip (zero-peak probe, "
       f"max rel err {e:.1e} <= 1e-12)", e <= 1e-12)
    e = rel_err(f[:, 4:], d[:, 4:], 0.8)
    ok(f"{physics}: vertical shift Svy = 0.8 x default at zero slip (zero-peak probe, "
       f"max rel err {e:.1e} <= 1e-12)", e <= 1e-12)


section("2", "full model, both physics, camber on")
for _ph in ("matlab", "corrected"):
    guard(f"{_ph}: mu checks", sec2_mu, _ph)
    guard(f"{_ph}: Kxk and Kya checks", sec2_kxk_kya, _ph)
    guard(f"{_ph}: zero-slip slope checks", sec2_slopes, _ph)
    guard(f"{_ph}: zero-peak shift checks", sec2_shifts, _ph)


# =============================================================================
def sec3_peaks():
    c0, c1 = screen_ctx({}), screen_ctx(FAC)
    Fz = np.linspace(800.0, 6000.0, 9)
    gam = np.linspace(-0.05, 0.05, 9)
    e = rel_err(peak_mu_x(Fz, c1.mf, c1.vp.Fz0), peak_mu_x(Fz, c0.mf, c0.vp.Fz0), 0.9)
    ok(f"QSS peak_mu_x = 0.9 x default over Fz 0.8-6 kN (max rel err {e:.1e} <= 1e-12)", e <= 1e-12)
    e = rel_err(peak_mu_y(Fz, c1.mf, c1.vp.Fz0, gam), peak_mu_y(Fz, c0.mf, c0.vp.Fz0, gam), 0.8)
    ok(f"QSS peak_mu_y = 0.8 x default over Fz and camber (max rel err {e:.1e} <= 1e-12)", e <= 1e-12)


def qss_lap(ov):
    def run():
        return MLTP_screen(circuit="Sturn", save=False, verbose=False, vp_overrides=ov).data.lap_time
    return quiet(run)


def sec3_laps():
    lap0 = qss_lap({})
    lap95 = qss_lap({"LMUX": 0.95, "LMUY": 0.95})
    lap1 = qss_lap({"LMUX": 1.0, "LMUY": 1.0})
    ok(f"QSS lap at LMUX = LMUY = 0.95 ({lap95:.3f} s) is longer than default ({lap0:.3f} s)", lap95 > lap0)
    ok(f"QSS lap at LMUX = LMUY = 1.0 equals default exactly ({lap1!r} == {lap0!r})", lap1 == lap0)


section("3", "QSS screener")
guard("QSS peak checks", sec3_peaks)
guard("QSS lap checks", sec3_laps)


# =============================================================================
def sec4():
    a, b = ctx7({}), ctx7(FAC)
    ok("7-state build with LMUX 0.9 and LMUY 0.8: ctx.mf carries the factors",
       getattr(b.mf, "LMUX", None) == 0.9 and getattr(b.mf, "LMUY", None) == 0.8)
    # the 7-state model has one aux variable, y = ltx (longitudinal load transfer): it is an input too
    pts = [(np.array([0.3, 0.05, 0.1, 0.0, 0.01, 0.3, 0.3]), np.array([0.5, 0.0, 0.1]), np.array([0.1]), 0.005),
           (np.array([0.45, -0.02, -0.05, 0.0, -0.02, 0.45, 0.45]), np.array([0.2, 0.3, -0.1]), np.array([-0.2]), 0.01)]
    Fa = ca.Function("Fa", [a.m7.x, a.m7.u, a.m7.y, a.m7.pv], [a.m7.dx])
    Fb = ca.Function("Fb", [b.m7.x, b.m7.u, b.m7.y, b.m7.pv], [b.m7.dx])
    diff = max(float(np.max(np.abs(np.array(Fa(x, u, y, k)) - np.array(Fb(x, u, y, k)))))
               for x, u, y, k in pts)
    ok(f"7-state dx with the factors equals the default build (max abs diff {diff!r} == 0.0)", diff == 0.0)


section("4", "7-state model")
guard("7-state checks", sec4)


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


def sec5_default():
    d0 = export(None)
    ok("parity export at default: no LMUX or LMUY in the mf block",
       "LMUX" not in d0["mf"] and "LMUY" not in d0["mf"])
    ok("parity export sanity: the mf block holds the MF coefficients (pDy1 present)", "pDy1" in d0["mf"])


def sec5_lmuy():
    d1 = export({"LMUY": 0.9})
    ok("parity export, vp_overrides LMUY 0.9: mf.LMUY exported as 0.9", d1["mf"].get("LMUY") == 0.9)


section("5", "parity export")
guard("parity export at default", sec5_default)
guard("parity export with LMUY 0.9", sec5_lmuy)

report()
print("ALL mf_scaling TESTS PASSED")
