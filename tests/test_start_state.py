"""Start state options (plan docs/superpowers/plans/2026-10-18-phase2.md, Task 4: B1 / B2 engine).

userOpts(start_state=...) boundary boxes and the periodic flag, result_stem() suffixes,
nlp_structure / structure_mismatch / plan_full_warm_start with n_periodic and the start_state
guard, and the NLP rows: the Xi_tol start box and the periodic rows appended last to g.
Sections 1-5 build no NLP. Sections 6-8 build the real NLP through a fake solver (no IPOPT)
and need casadi + Data/DATA_AA.mat (SKIP, exit 0 otherwise). Section 8: start_state=None (resolver, car start). Run from anywhere:

    python tests/test_start_state.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import copy
import inspect
import io
import os
import sys
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np

from functions import carfile
from functions.context import Ctx
from functions.importfile import load_solution, result_stem
from functions.warmstart import nlp_record, nlp_structure, plan_full_warm_start, structure_mismatch
from userOpts import K_JOIN_MAX, resolve_start_state, userOpts

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
STURN_RESULT = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
nan = np.nan


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout and warnings swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def raised(exc, fn, *a, **kw):
    """str(exc) of the exception that fn(*a, **kw) raises, or None when it raises none."""
    try:
        quiet(fn, *a, **kw)
    except exc as e:
        return str(e)
    return None


# ---- 1. userOpts, matlab default: the boxes stay as before -------------------------
print("userOpts start_state: matlab (default)")
c = Ctx()
quiet(userOpts, c, circuit="Sturn")
ok("start_state defaults to 'matlab'", c.start_state == "matlab")
ok("periodic defaults to False", c.periodic is False)
ok("Xi_tol defaults to None", c.Xi_tol is None)
ok("Xi unchanged: vx 60, vy 0, zs..phidot 0, rest free",
   np.array_equal(c.Xi, [60, 0] + [nan] * 7 + [0] * 6 + [nan] * 8, equal_nan=True))
ok("Xi_init unchanged", np.array_equal(c.Xi_init, [60, 0] + [nan] * 5, equal_nan=True))
ok("Xf free", np.isnan(c.Xf).all())
for bad in ("Rolling", "foo"):
    msg = raised(ValueError, userOpts, Ctx(), circuit="Sturn", start_state=bad)
    ok(f"start_state={bad!r} raises ValueError", msg is not None and "start_state must be one of" in msg)

# ---- 2. userOpts, rolling: r = 0, wheel speeds at vi / Rw, Xi_tol box -------------
print("userOpts start_state: rolling")
c = Ctx()
quiet(userOpts, c, circuit="Sturn", start_state="rolling")
ok("Xi_init: r 0, wheel speeds vi / Rw, n and eps free",
   np.array_equal(c.Xi_init, [60, 0, 0, nan, nan, 60 / c.vp.Rw_f, 60 / c.vp.Rw_r], equal_nan=True))
ok("Xi: vx = vi", c.Xi[0] == 60)
ok("Xi: vy = 0", c.Xi[1] == 0)
ok("Xi: r = 0", c.Xi[2] == 0)
ok("Xi: wheel speeds free (MLTP fills them)", np.isnan(c.Xi[5:9]).all())
ok("Xi_tol: OPT_e on rows 0-8, 1e-3 on rows 9-22",
   np.array_equal(c.Xi_tol, np.r_[np.full(9, 1e-2), np.full(14, 1e-3)]))
ok("periodic is False", c.periodic is False)

# ---- 3. userOpts, periodic: closed circuits only, closing sample ------------------
print("userOpts start_state: periodic")
for circ in ("Sturn", "Hairpin", "Straight", "ZigZag", "VirtualTrack", "BCN_S1"):
    msg = raised(ValueError, userOpts, Ctx(), circuit=circ, start_state="periodic")
    ok(f"periodic on open circuit {circ} raises, names the closed set", msg is not None and "BCN" in msg)
c = Ctx()
quiet(userOpts, c, circuit="BCN", start_state="periodic")
ok("BCN periodic: Xi free", np.isnan(c.Xi).all())
ok("BCN periodic: Xf free", np.isnan(c.Xf).all())
ok("BCN periodic: Xi_init free", np.isnan(c.Xi_init).all())
ok("BCN periodic: Xf_init free", np.isnan(c.Xf_init).all())
ok("BCN periodic: periodic True", c.periodic is True)
ok("BCN periodic: Xi_tol None", c.Xi_tol is None)
ok("BCN periodic: track already closed, 4651 samples", c.track.s.size == 4651)

c0 = Ctx()
quiet(userOpts, c0, circuit="Spa")
s0, x0, y0 = c0.track.s, c0.track.x, c0.track.y
ok("default Spa keeps 1401 samples", s0.size == 1401)
c = Ctx()
quiet(userOpts, c, circuit="Spa", start_state="periodic")
gap = np.hypot(x0[-1] - x0[0], y0[-1] - y0[0])
ok("Spa periodic: one closing sample (1402)", c.track.s.size == 1402)
ok("Spa periodic: s runs on by the closing gap", abs((c.track.s[-1] - s0[-1]) - gap) < 1e-9)
ok("Spa periodic: x closed", c.track.x[-1] == c.track.x[0])
ok("Spa periodic: y closed", c.track.y[-1] == c.track.y[0])
ok("Spa periodic: k closed", c.track.k[-1] == c.track.k[0])

c = Ctx()
quiet(userOpts, c, circuit="Circle", start_state="periodic")
ok("Circle periodic: synthetic, no closing sample (500)", c.track.s.size == 500)

c = Ctx()
quiet(userOpts, c, circuit="BCN", start_state="periodic", ni=1.0)
ok("periodic keeps ni at Xi[3] and Xi_init[3]", c.Xi[3] == 1.0 and c.Xi_init[3] == 1.0)

# ---- 4. result stems ----------------------------------------------------------------
print("result_stem start_state suffix")
ok("matlab stem unchanged", result_stem("BCN", "Static_ATDOn_EM4Off") == "BCN_Static_ATDOn_EM4Off")
ok("rolling adds _startRolling",
   result_stem("BCN", "Static_ATDOn_EM4Off", start_state="rolling") == "BCN_Static_ATDOn_EM4Off_startRolling")
ok("periodic stem: tyre, physics, start, mesh in that order",
   result_stem("BCN", "Static_ATDOn_EM4Off", "CopyB", "curvature", "corrected", "periodic")
   == "BCN_Static_ATDOn_EM4Off_CopyB_physCorrected_startPeriodic_meshCurvature")

# ---- 5. NLP structure record and warm-start planning --------------------------------
print("nlp_structure / plan_full_warm_start with n_periodic")
st = nlp_structure(23, 7, 0, 18, 3, 4)
ok("plain Sturn structure: n_g 1904", st["n_g"] == 1904)
ok("plain Sturn structure: n_periodic 0", st["n_periodic"] == 0)
sp = nlp_structure(23, 7, 0, 18, 3, 4, n_periodic=30)
ok("periodic structure: n_g 1934", sp["n_g"] == 1934)
ok("periodic structure: n_w 1812", sp["n_w"] == 1812)
ok("periodic structure: n_periodic 30", sp["n_periodic"] == 30)
ok("structure_mismatch reports n_periodic 30 -> 0",
   any(r.startswith("n_periodic 30 -> 0") for r in structure_mismatch(dict(sp), dict(st))))
sv = dict(st)
sv.pop("n_periodic")
ok("saved structure without n_periodic counts as 0", structure_mismatch(sv, st) == [])

KEYS = ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"]   # Static / ATD On / EM4 Off
sf = np.linspace(0.0, 540.0, 73)                                   # N 18, OPT_d 3: 4 N + 1 points
rs = np.random.RandomState(0)


def saved_record(struct):
    res = {"w_opt": rs.rand(struct["n_w"]), "lam_g": rs.randn(struct["n_g"]),
           "lam_x": rs.randn(struct["n_w"]), "structure": struct,
           "sym_type": "SX", "linear_solver": "ma57"}
    return nlp_record(res, {"iter_count": 3, "return_status": "Solve_Succeeded"})


rec_p, rec_s = saved_record(sp), saved_record(st)
ok("nlp_record keeps n_periodic", rec_p["structure"]["n_periodic"] == 30)
src_p = SimpleNamespace(nlp=rec_p, input_keys=KEYS, s_full=sf, tyre_set="MF205", start_state="periodic")
src_s = SimpleNamespace(nlp=rec_s, input_keys=KEYS, s_full=sf, tyre_set="MF205")   # no start_state: matlab
plain = dict(st, input_keys=KEYS, s_full=sf, tyre_set="MF205")
per = dict(sp, input_keys=KEYS, s_full=sf, tyre_set="MF205", start_state="periodic")
rolling = dict(st, input_keys=KEYS, s_full=sf, tyre_set="MF205", start_state="rolling")

warm, mode, _ = plan_full_warm_start(src_p, plain)
ok("periodic source into plain target: full-interp", mode == "full-interp" and warm is None)
warm, mode, _ = plan_full_warm_start(src_p, per)
ok("periodic source into periodic target: full+duals", mode == "full+duals")
ok("... re-injects 1934 multipliers", warm["lam_g0"].size == 1934)
warm, mode, note = plan_full_warm_start(src_s, rolling)
ok("matlab source (no key) into rolling target: full-interp, note names start_state",
   mode == "full-interp" and "start_state" in note)
src_m = SimpleNamespace(nlp=rec_s, input_keys=KEYS, s_full=sf, tyre_set="MF205", start_state="matlab")
warm, mode, note = plan_full_warm_start(src_m, rolling)
ok("explicit matlab source into rolling target: full-interp, note names start_state",
   mode == "full-interp" and "start_state" in note)
src_r = SimpleNamespace(nlp=rec_s, input_keys=KEYS, s_full=sf, tyre_set="MF205", start_state="rolling")
warm, mode, _ = plan_full_warm_start(src_r, rolling)
ok("rolling into rolling: full+duals", mode == "full+duals")
warm, mode, _ = plan_full_warm_start(src_s, plain)
ok("target without start_state: no check, full+duals", mode == "full+duals")
if os.path.exists(STURN_RESULT):
    d = load_solution(STURN_RESULT)
    warm, mode, _ = plan_full_warm_start(
        d, dict(st, input_keys=d.input_keys, s_full=d.s_full, tyre_set="MF205", start_state="matlab"))
    ok("stored Sturn result (no n_periodic, no start_state) into matlab target: full+duals",
       mode == "full+duals")
else:
    print(f"  SKIP stored Sturn result: {STURN_RESULT} missing")

# ---- 6-7. the real NLP rows: fake solver, no IPOPT ----------------------------------
if not os.path.exists(DATA_AA):
    print("SKIP test_start_state sections 6-7: Data/DATA_AA.mat missing")
    sys.exit(0)

import casadi as ca
import MLTP as mltp_mod
import MLTP_initial as mi_mod
import functions.transcription as T

_SIG = inspect.signature(T.build_and_solve_nlp)


class _Captured(Exception):
    pass


def capture(module, fn, **kw):
    """module.fn(**kw) up to its build_and_solve_nlp call: the bound arguments (only those
    passed). The stand-in raises before the build, so nothing is assembled or solved."""
    seen = {}

    def stand_in(*a, **k):
        seen.update(_SIG.bind(*a, **k).arguments)
        raise _Captured()

    real, module.build_and_solve_nlp = module.build_and_solve_nlp, stand_in
    try:
        quiet(getattr(module, fn), **kw)
    except _Captured:
        pass
    finally:
        module.build_and_solve_nlp = real
    return seen


@contextlib.contextmanager
def fake_ipopt(rec):
    """T._make_solver -> fake: the solver returns x = x0 with zero duals; rec gets 'nlp' and the
    1-D x0, lbx, ubx, lbg, ubg it was called with."""
    def fake(ca_, nlp, opts, info=None):
        rec["nlp"] = nlp
        if info is not None:
            info["linear_solver"] = "fake"

        def solver(**k):
            rec.update({key: np.asarray(v, float).reshape(-1) for key, v in k.items()})
            return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                    "lam_x": np.zeros(nlp["x"].numel())}

        solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
        return solver

    real, T._make_solver = T._make_solver, fake
    try:
        yield rec
    finally:
        T._make_solver = real


def build_only(args, **over):
    """The real build_and_solve_nlp on the captured args plus overrides, IPOPT faked: (res, rec)."""
    rec = {}
    with fake_ipopt(rec):
        res = quiet(T.build_and_solve_nlp, **dict(args, **over))
    return res, rec


X7 = np.vstack([np.linspace(30, 50, 10), np.zeros((4, 10)), np.full((2, 10), 100.0)])
U7 = np.vstack([np.full(10, 300.0), np.zeros((2, 10))])
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=U7)}   # in-memory 7-state init (warm_start=INIT)


def periodic_rows(cap, label, nx, nu, ny):
    """g ends with (knot N - knot 0) for every state, then every input; both bounds zero."""
    N, nh, d = cap["disc"]["N"], np.size(cap["h_lb"]), cap["OPT_d"]
    n_p = nx + nu
    rF, qF = build_only(cap)
    rT, qT = build_only(cap, periodic=True)
    ok(f"{label}: periodic adds {n_p} rows", rT["n_g"] == rF["n_g"] + n_p)
    ok(f"{label}: structure records n_periodic {n_p}",
       rT["structure"] == nlp_structure(nx, nu, ny, N, d, nh, n_periodic=n_p))
    ok(f"{label}: plain build records n_periodic 0", rF["structure"]["n_periodic"] == 0)
    ok(f"{label}: plain rows keep their lower bounds", np.array_equal(qT["lbg"][:rF["n_g"]], qF["lbg"]))
    ok(f"{label}: plain rows keep their upper bounds", np.array_equal(qT["ubg"][:rF["n_g"]], qF["ubg"]))
    ok(f"{label}: periodic rows have zero lower bounds", np.all(qT["lbg"][-n_p:] == 0))
    ok(f"{label}: periodic rows have zero upper bounds", np.all(qT["ubg"][-n_p:] == 0))
    w = np.random.default_rng(1).normal(size=rT["n_w"])
    gw = np.asarray(ca.Function("g", [qT["nlp"]["x"]], [qT["nlp"]["g"]])(w)).ravel()
    Xk = w[:nx * (N + 1)].reshape(nx, N + 1, order="F")
    Uk = w[nx * (N + 1):n_p * (N + 1)].reshape(nu, N + 1, order="F")
    ok(f"{label}: periodic state rows = knot N - knot 0",
       np.allclose(gw[-n_p:-nu], Xk[:, N] - Xk[:, 0], rtol=0, atol=1e-12))
    ok(f"{label}: periodic input rows = knot N - knot 0",
       np.allclose(gw[-nu:], Uk[:, N] - Uk[:, 0], rtol=0, atol=1e-12))


print("periodic rows: build_and_solve_nlp(periodic=True)")
a7 = capture(mi_mod, "MLTP_initial", circuit="Circle", OPT_ds=100, save=False)
periodic_rows(a7, "7-state", 7, 3, 1)
a23 = capture(mltp_mod, "MLTP", circuit="Circle", OPT_ds=100, warm_start=INIT,
              mesh="uniform", save=False, plot=False)
periodic_rows(a23, "23-state", 23, 7, 0)

print("start box: Xi_tol and the default boxes (Sturn, 23-state)")
a = capture(mltp_mod, "MLTP", circuit="Sturn", warm_start=INIT, mesh="uniform", save=False, plot=False)
m, Xi = a["m"], a["Xi"]
r0, q0 = build_only(a)
ok("default build: 1904 constraints", r0["n_g"] == 1904)
ok("default start rows: lbg = max(x_min, Xi/x_s - OPT_e)",
   np.array_equal(q0["lbg"][:23], np.fmax(m.x_min, Xi / m.x_s - 0.01)))
ok("default start rows: ubg = min(x_max, Xi/x_s + OPT_e)",
   np.array_equal(q0["ubg"][:23], np.fmin(m.x_max, Xi / m.x_s + 0.01)))
tol = np.r_[np.full(9, 0.01), np.full(14, 1e-3)]
X2 = Xi.copy()
X2[9] = -0.0128
X2[15:19] = 0.0      # wheel speeds at rest (NaN would keep the full [-1, 1] box)
X2[19:23] = 0.0195
r2, q2 = build_only(a, Xi=X2, Xi_tol=tol)
wd = q2["ubg"][:23] - q2["lbg"][:23]
ok("Xi_tol build: 1904 constraints", r2["n_g"] == 1904)
ok("Xi_tol rows 9-22: box width 2e-3", np.allclose(wd[9:23], 2e-3, rtol=0, atol=1e-15))
ok("Xi_tol rows 9-22: box centred on Xi",
   np.allclose((q2["ubg"][9:23] + q2["lbg"][9:23]) / 2, X2[9:23], rtol=0, atol=1e-15))
ok("Xi_tol rows 0-1: box width 0.02 (OPT_e kept)", np.allclose(wd[[0, 1]], 0.02, rtol=0, atol=1e-15))
ok("Xi_tol: end box unchanged", np.array_equal(q2["lbg"][23:46], q0["lbg"][23:46]))

# ---- synthetic car fixture (copied from tests/test_cs_wiring.py) ------------------------------
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair. An over or omit cid is a file cid.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}

# ---- synthetic car: literal cids and tags (as in tests/test_cs_car.py) -------------------------------
# Tags: 1 scalar, 2 array, 6 byte, 7 curve, 9 map, 10 hardpoints, 11 spring-like, 12 damper, 13 tyre.
TYRE = {  # dims (d1..d7), radius m, toe deg (0: a toe force jacks the body through a 2 m roll centre), camber, castor, drag, kz N/m
    "fl": dict(dims=(3, 2, 3, 2, 2, 4, 4), radius=0.300, toe=0.0, camber=-1.0, castor=2.0, drag=0.011, kz=1.60e5),
    "fr": dict(dims=(2, 2, 3, 2, 2, 4, 4), radius=0.305, toe=0.0, camber=-1.5, castor=2.5, drag=0.012, kz=1.62e5),
    "rl": dict(dims=(4, 2, 3, 3, 2, 4, 4), radius=0.310, toe=0.0, camber=-0.5, castor=1.5, drag=0.0135, kz=1.40e5),
    "rr": dict(dims=(2, 2, 3, 2, 3, 4, 4), radius=0.315, toe=0.0, camber=-0.75, castor=1.0, drag=0.014, kz=1.42e5),
}
TYRE_TEMPS = (300.0, 350.0)
TYRE_BG = 325.0
PRESSURE = 2.1e5
VEL = np.array([0.0, 100.0, 200.0, 400.0, 800.0])        # mm/s, from 0 up (the damper curve starts at the origin)
BUMP = np.array([0.0, 900.0, 1800.0, 2700.0, 3600.0])    # N, zero at zero velocity
REB = np.array([0.0, 600.0, 1200.0, 1800.0, 2400.0])     # N, positive as the file stores it
SPRING = {1: (1.23e5, 0.020), 2: (1.31e5, 0.021), 3: (0.97e5, 0.019), 4: (1.02e5, 0.0205),
          5: (0.0, 0.0), 6: (0.0, 0.0)}
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}
ARB = {19: 2.0e4, 21: 1.5e4}
HARD = (49, 50, 51, 52)                  # front left, front right, rear left, rear right
# Left-side hardpoints (x rearward, y left +, z up, m), rows: lower inner front, lower inner aft, lower outer,
# upper inner front, upper inner aft, upper outer. The right side is y mirrored. Front and rear are the same.
HP_LEFT = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.68, 0.11],
                    [-0.15, 0.30, 0.28], [0.15, 0.30, 0.28], [0.0, 0.64, 0.33]])
HP_RIGHT = HP_LEFT * np.array([1.0, -1.0, 1.0])
BUMP_STEER_X = np.linspace(-0.02, 0.02, 11)   # m of wheel travel; toe change zero (cids 74-77)
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: 1.52, 92: 1.49, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,            # ride heights, mm
    100: 0.9, 102: 1000.0, 103: 9000.0,                     # efficiency, idle and limit rpm
    107: 14000.0, 108: 0.6, 109: 0.05,                      # brake force N, front bias, delay s
    111: 0.0, 112: 1.0, 113: 1.0,                           # torque split (0 = rear drive), diff types (1 = open)
    55: 0.03, 59: 0.03, 63: 0.03, 67: 0.03,                # roll-centre heights per corner, m
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,                # CLA, CDA, offset, rho
    390: 350.0,
}
GAP = {7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.0, 12: 0.0}   # rubber ground gaps, m; third gaps 0 (no third element)
EXPLICIT.update(GAP)
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})        # DRS
EXPLICIT.update({142: 0.5 * (EXPLICIT[117] + EXPLICIT[118]) * 1e-3,
                 144: 0.5 * (EXPLICIT[119] + EXPLICIT[120]) * 1e-3, 56: 0.0, 60: 0.0, 64: 0.0, 68: 0.0})
EXPLICIT.update({c: 0.0 for c in (55, 59, 63, 67) if c not in EXPLICIT})
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
AERO_X = 0.005 * np.arange(7)            # rear ride height nodes, m
AERO_Z = 0.002 + 0.005 * np.arange(7)    # front ride height nodes, m
MAPS = {114: ((4, 4), "diff_f"), 116: ((4, 4), "diff_r")}   # casadi bspline: >= 4 per axis
MIRROR = ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
          (80, 81), (82, 83), (117, 118), (119, 120))                  # left cid -> right cid


def arb(n, s):
    """n arbitrary finite numbers, different for each seed s."""
    return np.sin(np.arange(n) * 0.9 + s) * 7.3 - 0.61


def tyre_sizes(d):
    """f64 count of the 13 tables of a tyre (carfile docstring: 2D(k) = 2k, 3D(a, b) = a + b + a b)."""
    d1, d2, d3, d4, d5, d6, d7 = d

    def g(a, b):
        return a + b + a * b
    return [2 * d3, 2 * d4, 2 * d4, g(d1, d2), g(d6, d7), 2 * d6, 2 * d4, g(d1, d2), 2 * d3, 2 * d6, 2 * d5, 2 * d5, 2 * d5]


def tyre_record(i, corner, toe=None):
    t = TYRE[corner]
    dims = t["dims"]
    d1, d2, d3, _, d5, d6, d7 = dims
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(dims))]
    A, B = arb(d1, 20 + i), arb(d1, 30 + i)
    load = np.linspace(1000.0, 25000.0, d1)              # N, up to 25 kN (the load floor is tested at 20 kN)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, np.stack([A, B], axis=1).ravel()])
    tabs[0][:d3] = np.linspace(0.0, 2.0, d3)                 # slip abscissae: from 0, increasing
    tabs[0][d3:] = np.array([0.0, 0.97, 0.97])[:d3]          # slip values: from 0, flat after the knee at 1
    tabs[5][:d6] = np.linspace(-3.0, 3.0, d6)                # camber abscissae (deg), increasing
    tabs[4] = np.concatenate([np.linspace(-3.0, 3.0, d6), np.linspace(0.0, 25000.0, d7), np.ones(d6 * d7)])  # flat 1.0 grid
    px = np.linspace(150.0, 250.0, d5)
    py = t["kz"] * (1.0 + 0.01 * np.arange(d5))
    tabs[10] = np.concatenate([px, py])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = (t["toe"] if toe is None else toe), t["camber"], t["castor"], t["radius"], PRESSURE, TYRE_BG
    sc[7], sc[8], sc[9], sc[11], sc[15] = t["drag"], 0.11 + 0.01 * i, 0.12 + 0.01 * i, 0.7 + 0.1 * i, 1.05 + 0.1 * i
    return 13, 80 + i, (list(dims), sc, tabs)


def mirror_car(R, toe=0.0):
    """Copy each left record onto its right partner and set every toe [deg] to toe: a left-right symmetric car."""
    idx = {c: i for i, (_, c, _) in enumerate(R)}
    for right, left in MIRROR:      # file cids: the right wheel is the lower one
        t, _, v = R[idx[left]]
        R[idx[right]] = (t, right, copy.deepcopy(v))
    for i, (t, c, v) in enumerate(R):
        if t == 13:
            dims, sc, tabs = v
            sc = np.array(sc, float)
            sc[0] = toe
            R[i] = (t, c, (dims, sc, tabs))
    return R


def build_records(arb_front=None, mirror=False, toe=0.0, toes=None, over=None):
    R = []

    def add(tag, cid, v):
        cid = FILE_CID.get(cid, cid)
        R.append((tag, cid, v))
    for cid, v in EXPLICIT.items():
        add(1, cid, v)
    for cid in (53, 61, 69):
        add(6, cid, 0)                                   # geometry mode 0 (53, 61), no live axle (69)
    for cid, (rate, pre) in SPRING.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.05]), np.array([0.0, 500.0])))
    for cid, (rate, pre) in RUBBER.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.02]), np.array([0.0, 800.0])))
    arb_rates = dict(ARB)
    if arb_front is not None:
        arb_rates[19] = arb_front
    for cid, rate in arb_rates.items():
        add(11, cid, (1, 0.0, rate, np.array([0.0, 0.1]), np.array([0.0, 300.0])))
    for cid in range(23, 31):
        add(2, cid, np.array([1.0 + 0.05 * np.sin(cid), 0.0, 0.0, 0.0, 0.0]))   # constant motion ratio, near 1
    for cid in range(366, 372):
        add(6, cid, 10 + cid - 366)
    for cid in range(204, 210):
        add(6, cid, 1)                                   # rubber gap flags (fl..rr, third f/r): 1 = ground gap
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    for cid in (127, 131, 135):
        add(9, cid, (AERO_X, AERO_Z, np.zeros(49) if mirror else arb(49, cid)))   # the mirror car has no aero: no lift at the static point
    for cid, ((nx, nz), _) in MAPS.items():
        add(9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid)))
    for cid in range(31, 36):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    for cid, hp in zip(HARD, (HP_LEFT, HP_RIGHT, HP_LEFT, HP_RIGHT)):
        add(10, cid, (0, hp.T.ravel()))                  # column-major (6, 3): the 3 coordinate rows of 6 points
    for cid in range(74, 78):
        add(7, cid, (BUMP_STEER_X, np.zeros(BUMP_STEER_X.size)))
    for i, c in enumerate(CORNERS):
        add(*tyre_record(i, c, (toes or {}).get(c)))
    if mirror:
        R = mirror_car(R, toe)
    if over:
        R = [(t, c, over[c]) if c in over else (t, c, v) for t, c, v in R]
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


# ---- 8. start_state=None (lane S): the resolver table and the car start --------------------
# The synthetic car below is copied from tests/test_cs_wiring.py (its record tables, not imported).
# Ring: the curvature joins at the line (k(0) = k(L)). Open ramp: k(L) = 1.5 k(0), far over K_JOIN_MAX.
CIRC_JOIN, CIRC_OPEN = "Synth Ring", "Synth Ramp"
S_RING = np.linspace(0.0, 600.0, 121)
K_RING = (2.0 * np.pi / 600.0) * (1.0 + 0.5 * np.sin(2.0 * np.pi * S_RING / 600.0))
K_RAMP = (2.0 * np.pi / 600.0) * (1.0 + 0.5 * S_RING / 600.0)
TMP = tempfile.TemporaryDirectory()


def write_install(root, name, k):
    """A ChassisSim folder with one 600 m circuit of curvature k on S_RING (the 'k s' rows of a ChassisSim file)."""
    circ = os.path.join(root, "Circuits")
    os.makedirs(circ)
    ini = ("[Total Circ prop]\r\nTotal circuits = 1\r\n\r\n"
           f"[Circuit0]\r\nCircuit name = {name}\r\nCircuit file name = tr_synth.txt\r\nBase Grip factor = 1.0\r\n"
           "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n")
    with open(os.path.join(root, "Circ_prop.ini"), "w", encoding="latin-1", newline="") as f:
        f.write(ini)
    rows = [f"{float(kk)!r}   {float(s)!r}" for kk, s in zip(k, S_RING)]
    with open(os.path.join(circ, "tr_synth.txt"), "w", encoding="latin-1", newline="") as f:
        f.write("\r\n".join(rows) + "\r\n")
    return root


INST = write_install(os.path.join(TMP.name, "ChassisSim"), CIRC_JOIN, K_RING)
INST_OPEN = write_install(os.path.join(TMP.name, "ChassisSimOpen"), CIRC_OPEN, K_RAMP)
CAR_PATH = os.path.join(TMP.name, "start_car.car")
carfile.write(build_records(), CAR_PATH)
CAR = f"chassissim:{CAR_PATH}"
CS_JOIN, CS_OPEN = f"chassissim:{CIRC_JOIN}", f"chassissim:{CIRC_OPEN}"
K_OPEN_0, K_OPEN_L = float(K_RAMP[0]), float(K_RAMP[-1])

print("resolve_start_state: the table")
ok("4EM car on Sturn with None -> matlab", resolve_start_state(None, "Sturn", None) == "matlab")
ok("4EM car on BCN with None -> matlab", resolve_start_state(None, "BCN", None) == "matlab")
ok("car on Sturn with None -> matlab (not a ChassisSim circuit)", resolve_start_state(None, "Sturn", CAR) == "matlab")
ok("car on the joining ring with None -> periodic",
   resolve_start_state(None, CS_JOIN, CAR, INST) == "periodic")
ok("4EM run on the joining ring with None -> matlab (no car)",
   resolve_start_state(None, CS_JOIN, None, INST) == "matlab")
ok("explicit rolling wins on the joining ring", resolve_start_state("rolling", CS_JOIN, CAR, INST) == "rolling")
ok("explicit matlab wins on the joining ring", resolve_start_state("matlab", CS_JOIN, CAR, INST) == "matlab")
ok("explicit periodic on Sturn is returned (userOpts refuses it: not a closed circuit)",
   resolve_start_state("periodic", "Sturn", None) == "periodic")
with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    r_open = resolve_start_state(None, CS_OPEN, CAR, INST_OPEN)
join_warn = [w for w in wl if "does not join" in str(w.message)]
ok("a circuit that fails the join check with None -> matlab", r_open == "matlab")
ok("... with one warning naming the join", len(join_warn) == 1)
msg = raised(ValueError, resolve_start_state, "periodic", CS_OPEN, CAR, INST_OPEN)
ok("explicit periodic on the open ramp raises ValueError", msg is not None)
ok("... naming both curvature values k(0) and k(L)",
   msg is not None and f"{K_OPEN_0:.3e}" in msg and f"{K_OPEN_L:.3e}" in msg)
ok("the join bound is 1e-3 1/m", K_JOIN_MAX == 1e-3)

print("start box of a car on Sturn: wheel speeds at vi / R")
a_car = capture(mltp_mod, "MLTP", circuit="Sturn", car=CAR, chassissim_dir=INST, physics="corrected",
                save=False, plot=False, mesh="uniform")
m_car, Xi_car = a_car["m"], a_car["Xi"]
vi_car = Xi_car[0]
R_CAR = np.array([0.300, 0.300, 0.310, 0.310])          # fl, fl, rl, rl: the car model uses the fl and rl radius
ok("car on Sturn: Xi rows 5-8 = vi / R (fronts fl radius, rears rl radius)",
   np.allclose(Xi_car[5:9], vi_car / R_CAR, rtol=0, atol=1e-12))
r_car, q_car = build_only(a_car)
lo_exp = np.fmax(m_car.x_min, Xi_car / m_car.x_s - 0.01)
hi_exp = np.fmin(m_car.x_max, Xi_car / m_car.x_s + 0.01)
ok("car on Sturn: rows 5-8 lower bound = Xi / x_s - OPT_e", np.array_equal(q_car["lbg"][5:9], lo_exp[5:9]))
ok("car on Sturn: rows 5-8 upper bound = Xi / x_s + OPT_e", np.array_equal(q_car["ubg"][5:9], hi_exp[5:9]))
ok("car on Sturn: rows 5-8 box is 2 OPT_e wide (no clip at the state bounds)",
   np.allclose((q_car["ubg"] - q_car["lbg"])[5:9], 0.02, rtol=0, atol=1e-12))

print("start box of the 4EM car on Sturn: wheel speeds free (parity box)")
a_4em = capture(mltp_mod, "MLTP", circuit="Sturn", save=False, plot=False, mesh="uniform")
m_4, Xi_4 = a_4em["m"], a_4em["Xi"]
r_4, q_4 = build_only(a_4em)
ok("4EM car on Sturn: Xi rows 5-8 are NaN (free)", np.all(np.isnan(Xi_4[5:9])))
ok("4EM car on Sturn: rows 5-8 span the state bounds",
   np.array_equal(q_4["lbg"][5:9], m_4.x_min[5:9]) and np.array_equal(q_4["ubg"][5:9], m_4.x_max[5:9]))

print("car on the joining ring: periodic start, free start box")
a_ring = capture(mltp_mod, "MLTP", circuit=CS_JOIN, car=CAR, chassissim_dir=INST, physics="corrected",
                 save=False, plot=False)
m_ring, Xi_ring = a_ring["m"], a_ring["Xi"]
r_ring, q_ring = build_only(a_ring)
ok("car on the joining ring: n_periodic = nx + nu (23 + 3)",
   r_ring["structure"]["n_periodic"] == int(m_ring.nx) + int(m_ring.nu) == 26)
ok("car on the joining ring: every start row free but row 3 (ni)", np.all(np.isnan(np.delete(Xi_ring, 3))))
free = [i for i in range(23) if i != 3]
ok("car on the joining ring: free start rows span the state bounds",
   np.array_equal(q_ring["lbg"][free], m_ring.x_min[free]) and np.array_equal(q_ring["ubg"][free], m_ring.x_max[free]))

print("test_start_state: done")
