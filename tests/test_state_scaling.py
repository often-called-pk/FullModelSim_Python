"""B7: userOpts(state_scaling='matlab' | 'typical') scales the 14 chassis rows 9-22 of the 23-state model.

The default keeps MATLAB's scale 1.0. 'typical' builds the same NLP in other units. The start box keeps
its physical width. A saved w or multiplier is refused across the two scalings, as tyre_set and start_state
are. Sections 1-2 build no NLP. Sections 3-6 need casadi and Data/DATA_AA.mat (SKIP, exit 0 otherwise).
Sections 3-5 use build_only or a stand-in that stops before the solver. Section 6 replaces the IPOPT solver
with a fake and starts from Results/Sturn_Static_ATDOn_EM4Off.mat (SKIP if it is missing). No section runs
IPOPT. Run from anywhere:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_state_scaling.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import inspect
import io
import os
import sys
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np

from functions.context import Ctx
from functions.importfile import load_solution
from functions.warmstart import nlp_structure, plan_full_warm_start
from userOpts import ROLL_TOL, STATE_SCALINGS, TYPICAL_CHASSIS_SCALE, userOpts

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
PRESET = np.array([1e-2, 1e-1, 1e-2, 1e-1, 1e-2, 1e-1] + [3e-1] * 4 + [2e-2] * 4)


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def raised(fn, *a, **kw):
    try:
        quiet(fn, *a, **kw)
    except ValueError as e:
        return str(e)
    return None


print("1. userOpts")
ref = Ctx()
quiet(userOpts, ref, circuit="Sturn")
ok("default: matlab, no scale vector, no start-box override",
   ref.state_scaling == "matlab" and ref.chassis_scale is None and ref.Xi_tol is None)
ok("choices: matlab and typical, preset pinned",
   tuple(STATE_SCALINGS) == ("matlab", "typical") and np.array_equal(TYPICAL_CHASSIS_SCALE, PRESET))
ok("unknown name raises",
   "state_scaling" in (raised(userOpts, Ctx(), circuit="Sturn", state_scaling="Typical") or ""))
c = Ctx()
quiet(userOpts, c, circuit="Sturn", state_scaling="typical")
ok("typical: preset on the chassis rows, physical start values",
   c.state_scaling == "typical" and np.array_equal(c.chassis_scale, PRESET)
   and np.array_equal(c.Xi, ref.Xi, equal_nan=True))
ok("typical: start box keeps OPT_e in physical units (rows 9-22 OPT_e / scale)",
   np.array_equal(c.Xi_tol, np.r_[np.full(9, 1e-2), 1e-2 / PRESET]))
c = Ctx()
quiet(userOpts, c, circuit="Sturn", state_scaling="typical", start_state="rolling")
ok("typical + rolling: ROLL_TOL / scale on rows 9-22", np.array_equal(c.Xi_tol[9:], ROLL_TOL / PRESET))
c = Ctx()
quiet(userOpts, c, circuit="Circle", state_scaling="typical", start_state="periodic")
ok("typical + periodic: start rows stay free", np.isnan(c.Xi).all() and c.state_scaling == "typical")

print("2. warm-start planner")
st = nlp_structure(23, 7, 0, 18, 3, 4)


def full_result(sc):
    """A saved full result of structure st. sc None: no state_scaling key, as in a default result."""
    res = dict(input_keys=["u"] * 7, s_full=np.arange(73.0),
               nlp=dict(structure=st, w_opt=np.zeros(st["n_w"]), lam_g=np.zeros(st["n_g"]),
                        lam_x=np.zeros(st["n_w"]), return_status="Solve_Succeeded"))
    return dict(res, state_scaling=sc) if sc else res


def target(sc):
    """The expected structure of the new NLP. sc None: the caller does not pass the key."""
    exp = dict(st, input_keys=["u"] * 7, s_full=np.arange(73.0))
    return dict(exp, state_scaling=sc) if sc else exp


for src, dst, mode in ((None, "matlab", "full+duals"), ("typical", "typical", "full+duals"),
                       (None, "typical", "full-interp"), ("typical", "matlab", "full-interp")):
    w, m, note = plan_full_warm_start(full_result(src), target(dst))
    ok(f"source {src or 'no key (matlab)'} -> target {dst}: {mode}",
       m == mode and (mode == "full+duals" or (w is None and "state_scaling" in note)))
ok("target without the key skips the check (older callers)",
   plan_full_warm_start(full_result("typical"), target(None))[1] == "full+duals")

if not os.path.exists(DATA_AA):
    print("SKIP test_state_scaling sections 3-6: Data/DATA_AA.mat missing")
    sys.exit(0)

import casadi as ca  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import functions.transcription as T  # noqa: E402
import MLTP as mltp_mod  # noqa: E402
import MLTP_paramOptim as mp  # noqa: E402
from functions.sensitivity import nlp_index  # noqa: E402


class _Captured(Exception):
    """Stops a call once the value under test has been seen."""


def expected_of(module, name, call):
    """The expected dict that call() hands to module.<name>. The planner stops the call, so no solve runs."""
    seen = {}

    def stop(src, expected, *a, **kw):
        seen.update(expected)
        raise _Captured()

    real = getattr(module, name)
    setattr(module, name, stop)
    try:
        quiet(call)
    except _Captured:
        pass
    finally:
        setattr(module, name, real)
    return seen


def planner_scaling(expected):
    """The state_scaling the planner got, or None when the planner was not reached."""
    return expected.get("state_scaling") if expected else None


print("3. MLTP and optimise_design hand state_scaling to the planner (Sturn, stopped before the solve)")
SRC = {"x_opt": np.zeros((23, 2)), "u_opt": np.zeros((7, 2))}      # a full result: the planner is the first stop
ok("MLTP: typical reaches the planner", planner_scaling(expected_of(
    mltp_mod, "plan_full_warm_start",
    lambda: mltp_mod.MLTP(circuit="Sturn", warm_start=SRC, save=False, plot=False,
                          state_scaling="typical"))) == "typical")
ok("MLTP: default reaches the planner as matlab", planner_scaling(expected_of(
    mltp_mod, "plan_full_warm_start",
    lambda: mltp_mod.MLTP(circuit="Sturn", warm_start=SRC, save=False, plot=False))) == "matlab")
ok("optimise_design: typical reaches the planner", planner_scaling(expected_of(
    mp, "plan_design_warm_start",
    lambda: mp.optimise_design([("alpha_RW", -0.05, 0.05)], "t6", circuit="Sturn", warm_start=SRC,
                               build_only=True, state_scaling="typical"))) == "typical")
ok("optimise_design: default reaches the planner as matlab", planner_scaling(expected_of(
    mp, "plan_design_warm_start",
    lambda: mp.optimise_design([("alpha_RW", -0.05, 0.05)], "t6", circuit="Sturn", warm_start=SRC,
                               build_only=True))) == "matlab")

X7 = np.vstack([np.linspace(30, 50, 10), np.zeros((4, 10)), np.full((2, 10), 100.0)])
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=np.vstack([np.full(10, 300.0), np.zeros((2, 10))]))}
_SIG = inspect.signature(T.build_and_solve_nlp)


def build(**kw):
    """The NLP MLTP hands to build_and_solve_nlp, built without a solver: (captured args, result)."""
    seen = {}

    def stand_in(*a, **k):
        seen.update(_SIG.bind(*a, **k).arguments)
        raise _Captured()

    real, mltp_mod.build_and_solve_nlp = mltp_mod.build_and_solve_nlp, stand_in
    try:
        quiet(mltp_mod.MLTP, circuit="Sturn", warm_start=INIT, save=False, plot=False, **kw)
    except _Captured:
        pass
    finally:
        mltp_mod.build_and_solve_nlp = real
    return seen, quiet(T.build_and_solve_nlp, **seen, build_only=True)


def evaluate(res, w):
    """f, g and the Jacobian of g at w, from the built NLP."""
    nlp = res["nlp"]
    F = ca.Function("F", [nlp["x"]], [nlp["f"], nlp["g"], ca.jacobian(nlp["g"], nlp["x"])])
    f, g, J = F(w)
    return float(f), np.asarray(g).ravel(), sp.csr_matrix(J.full())


print("4. the typical NLP is the default NLP in other units (Sturn, build_only)")
ad, rd = build()
at, rt = build(state_scaling="typical")
md, mt = ad["m"], at["m"]
nx = 23
ix = nlp_index(rd["structure"])
ok("sizes: Sturn NLP 1812 x 1904 in both scalings",
   (rd["nlp"]["x"].numel(), rd["nlp"]["g"].numel()) == (rt["nlp"]["x"].numel(), rt["nlp"]["g"].numel()) == (1812, 1904))
ok("x_s: rows 9-22 are 1.0 by default and the preset when typical",
   np.all(md.x_s[9:] == 1.0) and np.array_equal(mt.x_s[9:], PRESET))
ok("x_s rows 0-8 and u_s unchanged", np.array_equal(md.x_s[:9], mt.x_s[:9]) and np.array_equal(md.u_s, mt.u_s))
ok("physical state bounds unchanged",
   np.allclose(md.x_min * md.x_s, mt.x_min * mt.x_s, rtol=1e-14, atol=0)
   and np.allclose(md.x_max * md.x_s, mt.x_max * mt.x_s, rtol=1e-14, atol=0))
c14 = mt.x_s / md.x_s                                  # w_typical = w_default / c14 on the state rows
cw, rg = np.ones(rd["structure"]["n_w"]), np.ones(rd["structure"]["n_g"])
for i in range(nx):
    cw[ix["X"][i]] = cw[ix["XC"][i]] = c14[i]
    for key in ("defect", "cont", "start", "end"):     # g_typical = g_default / c14 on the state rows
        rg[ix[key][i]] = 1.0 / c14[i]
wd = np.asarray(rd["w0"], float).ravel()
ok("cold seed: w0 typical = w0 default / c", np.allclose(np.asarray(rt["w0"], float).ravel(), wd / cw,
                                                          rtol=1e-12, atol=1e-15))
fd, gd, Jd = evaluate(rd, wd)
ft, gt, Jt = evaluate(rt, wd / cw)
ok("f equal", abs(fd - ft) <= 1e-12 * abs(fd))
ok("g equal up to the row factors: g' = R g", np.allclose(gt, rg * gd, rtol=1e-9, atol=1e-10))
ok("J equal up to the row and column factors: J' = R J C",
   abs(Jt - sp.diags(rg) @ Jd @ sp.diags(cw)).max() <= 1e-9 * abs(Jd).max())
ok("w bounds: lbw / c and ubw / c",
   np.allclose(rt["lbw"].ravel(), rd["lbw"].ravel() / cw, rtol=1e-14, atol=0)
   and np.allclose(rt["ubw"].ravel(), rd["ubw"].ravel() / cw, rtol=1e-14, atol=0))
ok("g bounds: lbg * r and ubg * r (the start box keeps its physical width)",
   np.allclose(rt["lbg"].ravel(), rd["lbg"].ravel() * rg, rtol=1e-14, atol=1e-15)
   and np.allclose(rt["ubg"].ravel(), rd["ubg"].ravel() * rg, rtol=1e-14, atol=1e-15))

print("5. rolling start keeps its physical box (Sturn, build_only)")
a0, r0 = build(start_state="rolling")
a1, r1 = build(start_state="rolling", state_scaling="typical")


def edges(a, r):
    """Physical start-box edges of the 23 states. The first 23 rows of g are the start rows."""
    return np.r_[r["lbg"].ravel()[:nx] * a["m"].x_s, r["ubg"].ravel()[:nx] * a["m"].x_s]


ok("rolling: physical start-box edges equal (rows 0-22)",
   np.allclose(edges(a1, r1), edges(a0, r0), rtol=1e-12, atol=1e-15))

print("6. results record state_scaling (Sturn, IPOPT faked, no solve)")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
if not os.path.exists(STURN):
    print(f"SKIP test_state_scaling section 6: {STURN} missing")
    sys.exit(0)
import setup_sweep as SS  # noqa: E402


@contextlib.contextmanager
def fake_ipopt():
    """Replaces the IPOPT solver: x = x0 and zero multipliers."""
    def fake(ca_, nlp, opts, info=None):
        if info is not None:
            info["linear_solver"] = "fake"

        def solver(**k):
            return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                    "lam_x": np.zeros(nlp["x"].numel())}

        solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
        return solver

    real, T._make_solver = T._make_solver, fake
    try:
        yield
    finally:
        T._make_solver = real


def faked(fn, *a, **kw):
    """fn(*a, circuit='Sturn', save=False, **kw) through the fake solver."""
    with fake_ipopt():
        return quiet(fn, *a, **dict(dict(circuit="Sturn", save=False), **kw))


def precheck(src, scaling):
    """The SweepError text of setup_sweep._precheck_base for a Sturn sweep of this scaling, or None."""
    base = SimpleNamespace(input_keys=list(typ.input_keys), OPT_d=3, tyre_set="MF205", start_state="matlab",
                           state_scaling=scaling)
    try:
        SS._precheck_base(src, base, {"N": int(src.N), "s_full": np.asarray(src.s_full)}, 4, "Sturn", "base")
    except SS.SweepError as e:
        return str(e)
    return None


DESIGN = ([("alpha_RW", 4.0, 16.0)], "t6")
with tempfile.TemporaryDirectory() as tmp:
    typ = faked(mltp_mod.MLTP, warm_start=STURN, plot=False, state_scaling="typical", save=True, results_dir=tmp)
    back = load_solution(typ.out_path)
    ok("MLTP typical from the stored matlab result: full-interp, data.state_scaling 'typical', also after the "
       ".mat round trip",
       typ.elapsed["warm_start"] == "full-interp" and getattr(typ.data, "state_scaling", None) == "typical"
       and str(getattr(back, "state_scaling", None)) == "typical")
    dflt = faked(mltp_mod.MLTP, warm_start=typ.out_path, plot=False)
    ok("MLTP default from the typical result: full-interp, no state_scaling field (absent means 'matlab')",
       dflt.elapsed["warm_start"] == "full-interp" and not hasattr(dflt.data, "state_scaling"))
    d_typ = faked(mp.optimise_design, *DESIGN, warm_start=typ.out_path, state_scaling="typical")
    d_dfl = faked(mp.optimise_design, *DESIGN, warm_start=typ.out_path)
ok("optimise_design from the typical result: typical gives full+duals and data.state_scaling 'typical', "
   "default gives full-interp and no field",
   d_typ.elapsed["warm_start"] == "full+duals" and getattr(d_typ.data, "state_scaling", None) == "typical"
   and d_dfl.elapsed["warm_start"] == "full-interp" and not hasattr(d_dfl.data, "state_scaling"))
stored = load_solution(STURN)
ok("setup_sweep base check: a base of the sweep's scaling passes, a base of the other scaling raises SweepError",
   precheck(back, "typical") is None and "state_scaling" in (precheck(back, "matlab") or "")
   and precheck(stored, "matlab") is None and "state_scaling" in (precheck(stored, "typical") or ""))
print("test_state_scaling: done")
