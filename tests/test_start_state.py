"""Start state options (plan docs/superpowers/plans/2026-10-18-phase2.md, Task 4: B1 / B2 engine).

userOpts(start_state=...) boundary boxes and the periodic flag, result_stem() suffixes,
nlp_structure / structure_mismatch / plan_full_warm_start with n_periodic and the start_state
guard, and the NLP rows: the Xi_tol start box and the periodic rows appended last to g.
Sections 1-5 build no NLP. Sections 6-7 build the real NLP through a fake solver (no IPOPT)
and need casadi + Data/DATA_AA.mat (SKIP, exit 0 otherwise). Run from anywhere:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_start_state.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import inspect
import io
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np

from functions.context import Ctx
from functions.importfile import load_solution, result_stem
from functions.warmstart import nlp_record, nlp_structure, plan_full_warm_start, structure_mismatch
from userOpts import userOpts

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

print("test_start_state: done")
