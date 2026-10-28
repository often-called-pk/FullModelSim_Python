"""B3 userOpts(between_knots=...): linear inputs between the knots and power rows at the collocation points.

  'matlab'        (default) MATLAB's arithmetic. At the collocation points the dynamics see
                  U_k + duk * tau, a near hold of U_k. The path rows apply at the knots only.
  'linear'        The dynamics see the linear input U_k + (U_k+1 - U_k) * tau. Same rows and variables.
  'linear+power'  'linear', plus the motor power rows and the pack row at the N * OPT_d collocation
                  points. These rows come last in g, after the periodic rows.

Plain script, no pytest. Each check prints PASS or FAIL. The first FAIL stops the file.

  1. userOpts: values, flags and errors
  2. nlp_structure, structure_mismatch and the between_knots guard of the warm start
  3. refine.nlp_inputs and _input_affine: the inputs inside an interval
  4. nlp_index: the pathcol rows
  5. rpm cap: gear * Vmax / Rw < OMmax, so B3 adds no rpm row
  6. NLP rows through MLTP (Sturn, fake solver)
  7. EM4, regen and pack rows, and the periodic lap
  8. warm-start modes, channels, saved fields, Pmax price and refusals
  9. refine: the reseed and the defect indicator use the linear input

Sections 1 to 5 need no casadi. Sections 6 to 9 need casadi and Data/DATA_AA.mat. Without them the
file prints SKIP and exits 0. No IPOPT runs here. Run from any directory:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_between_knots.py
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
from functions.refine import _input_affine, nlp_inputs
from functions.sensitivity import nlp_index
from functions.transcription import discretise, interp_inputs
from functions.warmstart import STRUCT_KEYS, nlp_record, nlp_structure, plan_full_warm_start, structure_mismatch
from userOpts import BETWEEN_KNOTS, userOpts

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
STURN_RESULT = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout and warnings swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def raised(exc, fn, *a, **kw):
    """str(exc) of the exception that fn(*a, **kw) raises, or None if it raises none."""
    try:
        quiet(fn, *a, **kw)
    except exc as e:
        return str(e)
    return None


# ---- 1. userOpts -------------------------------------------------------------------------
print("1. userOpts: between_knots values, flags and errors")
ok("choices are matlab, linear, linear+power", BETWEEN_KNOTS == ("matlab", "linear", "linear+power"))
c0 = Ctx()
quiet(userOpts, c0, circuit="Sturn")
ok("default: matlab, no linear inputs, no power rows at the collocation points",
   c0.between_knots == "matlab" and c0.linear_inputs is False and c0.power_at_col is False)
for value, lin, pw in (("matlab", False, False), ("linear", True, False), ("linear+power", True, True)):
    c = Ctx()
    quiet(userOpts, c, circuit="Sturn", between_knots=value)
    ok(f"{value}: linear_inputs {lin}, power_at_col {pw}, OPT_uinter stays 'linear'",
       c.between_knots == value and c.linear_inputs is lin and c.power_at_col is pw and c.OPT_uinter == "linear")
for bad in ("Linear", "power", "", None):
    msg = raised(ValueError, userOpts, Ctx(), circuit="Sturn", between_knots=bad)
    ok(f"between_knots={bad!r} raises ValueError that names the knob", msg is not None and "between_knots" in msg)

# ---- 2. structure record and warm-start guard ---------------------------------------------
print("2. nlp_structure, structure_mismatch and the between_knots guard of the warm start")
st = nlp_structure(23, 7, 0, 18, 3, 4)
sp = nlp_structure(23, 7, 0, 18, 3, 4, n_pathcol=54)
ok("n_pathcol is a structure key with default 0, Sturn n_g 1904 -> 1958, n_w unchanged",
   "n_pathcol" in STRUCT_KEYS and st["n_pathcol"] == 0 and st["n_g"] == 1904
   and sp["n_g"] == 1958 and sp["n_w"] == st["n_w"] == 1812)
ok("periodic rows and path rows add up",
   nlp_structure(23, 7, 0, 18, 3, 4, n_periodic=30, n_pathcol=54)["n_g"] == 1988)
sv = dict(st)
del sv["n_pathcol"]
ok("a saved structure without n_pathcol counts as 0", structure_mismatch(sv, st) == [])
ok("structure_mismatch names 'n_pathcol 54 -> 0'",
   any(r.startswith("n_pathcol 54 -> 0") for r in structure_mismatch(dict(sp), dict(st))))

KEYS = ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"]
SF = np.linspace(0.0, 540.0, 73)          # s_full of N = 18, d = 3
rs = np.random.RandomState(0)


def saved(struct, **extra):
    """A full result as MLTP saves it: the NLP record of struct, the input keys and the grid, plus extra."""
    res = {"w_opt": rs.rand(struct["n_w"]), "lam_g": rs.randn(struct["n_g"]),
           "lam_x": rs.randn(struct["n_w"]), "structure": struct}
    return SimpleNamespace(nlp=nlp_record(res, {"iter_count": 3, "return_status": "Solve_Succeeded"}),
                           input_keys=KEYS, s_full=SF, tyre_set="MF205", **extra)


def target(struct, between=None):
    """The expected structure of a new NLP. between None means the target has no between_knots key."""
    extra = {} if between is None else {"between_knots": between}
    return dict(struct, input_keys=KEYS, s_full=SF, tyre_set="MF205", **extra)


def mode(src, exp):
    return plan_full_warm_start(src, exp)[1]


src_m = saved(st)                         # no field: read as 'matlab'
src_l = saved(st, between_knots="linear")
src_p = saved(sp, between_knots="linear+power")
ok("nlp_record keeps n_pathcol", src_p.nlp["structure"]["n_pathcol"] == 54 and src_m.nlp["structure"]["n_pathcol"] == 0)
ok("matlab source (no field) into a matlab target: full+duals", mode(src_m, target(st, "matlab")) == "full+duals")
warm, md, note = plan_full_warm_start(src_m, target(st, "linear"))
ok("matlab source into a linear target: full-interp, no multipliers, the note names both values",
   warm is None and md == "full-interp" and all(t in note for t in ("between_knots", "matlab", "linear")))
ok("linear source into a linear target: full+duals", mode(src_l, target(st, "linear")) == "full+duals")
ok("linear source into a linear+power target: full-interp (between_knots differ)",
   mode(src_l, target(sp, "linear+power")) == "full-interp")
ok("linear+power source into a linear+power target: full+duals",
   mode(src_p, target(sp, "linear+power")) == "full+duals")
ok("a target without a between_knots key skips the check (sweeps, old callers)",
   mode(src_m, target(st)) == "full+duals")

# ---- 3. refine.nlp_inputs and _input_affine: the inputs inside an interval ----------------
print("3. refine.nlp_inputs and _input_affine: the inputs inside an interval")
tr = SimpleNamespace(s=np.linspace(0.0, 540.0, 301), k=np.zeros(301))
disc = discretise(tr, 30, 3)
N, s_knot, s_col, dsk, tau = disc["N"], disc["s_knot"], disc["s_col"], disc["dsk"], disc["tau"]
U = rs.randn(3, N + 1)
k_of = np.repeat(np.arange(N), 3)
near = U[:, k_of] + ((U[:, 1:] - U[:, :-1]) / dsk)[:, k_of] * np.tile(tau, N)
ok("default: U_k + duk * tau with duk per metre (MATLAB's near hold)",
   np.allclose(nlp_inputs(disc, U, s_col, "linear"), near, rtol=0, atol=1e-13))
ok("linear_inputs: equals np.interp at the collocation points",
   np.allclose(nlp_inputs(disc, U, s_col, "linear", linear_inputs=True),
               interp_inputs(U, s_knot, s_col, "linear"), rtol=0, atol=1e-12))
s_all = np.linspace(0.0, 540.0, 1001)
ok("linear_inputs: equals np.interp at any s, the last knot included",
   np.allclose(nlp_inputs(disc, U, s_all, "linear", linear_inputs=True),
               interp_inputs(U, s_knot, s_all, "linear"), rtol=0, atol=1e-12))
ok("both modes are exact at every knot, U_N included",
   np.array_equal(nlp_inputs(disc, U, s_knot, "linear", linear_inputs=True), U)
   and np.array_equal(nlp_inputs(disc, U, s_knot), U))
a0, b0 = _input_affine(U, dsk, "linear")
a1, b1 = _input_affine(U, dsk, "linear", True)
ok("_input_affine: same a, and b is the increment (the per-metre slope times dsk)",
   np.array_equal(a0, a1) and np.allclose(b0 * dsk, b1, rtol=0, atol=1e-13))

# ---- 4. sensitivity.nlp_index: the pathcol rows -------------------------------------------
print("4. nlp_index: pathcol rows last in g, after the periodic rows")
ix0 = nlp_index(st)
ix1 = nlp_index(nlp_structure(23, 7, 0, 18, 3, 4, n_periodic=30, n_pathcol=54))
ok("no path rows: pathcol is empty, nh 4", ix0["pathcol"].size == 0 and ix0["nh"] == 4)
ok("h, rate and periodic rows keep their places",
   ix1["nh"] == 4 and np.array_equal(ix1["h"], ix0["h"]) and np.array_equal(ix1["rate"], ix0["rate"])
   and np.array_equal(ix1["periodic"], ix0["rate"].max() + 1 + np.arange(30)))
ok("pathcol: the last 54 rows of g", np.array_equal(ix1["pathcol"], ix1["n_g"] - 54 + np.arange(54)))
ok("a structure without n_pathcol counts as 0", nlp_index(sv)["pathcol"].size == 0)

# ---- 5. rpm cap: the Om box keeps the motor speed below OMmax ------------------------------
print("5. rpm cap: gear * Vmax / Rw < OMmax, so B3 adds no rpm row")
c = Ctx()
quiet(userOpts, c, circuit="Sturn")
cap = c.vp.gear * c.pt.Vmax / c.vp.Rw
ok(f"gear * Vmax / Rw = {cap:.1f} rad/s = {cap / c.pt.OMmax:.4f} OMmax (below OMmax, 0.8853)",
   cap < c.pt.OMmax and abs(cap / c.pt.OMmax - 0.8853) < 1e-3)

# ---- SKIP guard: the model sections need the aero file and casadi -------------------------
if not os.path.exists(DATA_AA):                                  # pragma: no cover
    print("SKIP sections 6-9: Data/DATA_AA.mat missing")
    sys.exit(0)
import casadi as ca  # noqa: E402
import functions.transcription as T  # noqa: E402
import MLTP as mltp_mod  # noqa: E402
import MLTP_initial as init_mod  # noqa: E402
import MLTP_paramOptim as po_mod  # noqa: E402
import setup_sweep as SS  # noqa: E402
from functions.importfile import importfile  # noqa: E402
from functions.sensitivity import limit_prices  # noqa: E402

X7 = np.vstack([np.linspace(30, 50, 10), np.zeros((4, 10)), np.full((2, 10), 100.0)])
U7 = np.vstack([np.full(10, 300.0), np.zeros((2, 10))])
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=U7)}


@contextlib.contextmanager
def fake_ipopt(rec, status="Fake"):
    """Replaces the IPOPT solver: x = x0, zero duals and the IPOPT status ``status``.
    rec gets the NLP and the vectors of the solve call."""
    def fake(ca_, nlp, opts, info=None):
        rec["nlp"] = nlp
        if info is not None:
            info["linear_solver"] = "fake"

        def solver(**k):
            rec.update({key: np.asarray(v, float).reshape(-1) for key, v in k.items()})
            return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                    "lam_x": np.zeros(nlp["x"].numel())}
        solver.stats = lambda: {"iter_count": 0, "return_status": status}
        return solver
    real, T._make_solver = T._make_solver, fake
    try:
        yield rec
    finally:
        T._make_solver = real


def mltp(circuit="Sturn", warm_start=INIT, status="Fake", **kw):
    """MLTP through the fake solver with the IPOPT status ``status``. Returns (ctx, solve record).
    rec["warnings"] holds the texts of the warnings. The call saves nothing unless save=True."""
    rec = {}
    kw = dict(dict(save=False, plot=False), **kw)
    with fake_ipopt(rec, status), contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctx = mltp_mod.MLTP(circuit=circuit, warm_start=warm_start, **kw)
    rec["warnings"] = [str(w.message) for w in caught]
    return ctx, rec


def g_of(rec):
    """Function (w) -> (f, g) of the NLP that the solve call received."""
    nlp = rec["nlp"]
    return ca.Function("fg", [nlp["x"]], [nlp["f"], nlp["g"]])


def at_points(ctx, w):
    """States and linear input at the collocation points of a decision vector w of ctx's NLP.

    The function is numpy only and independent of the NLP code. It returns (Xkj, u), with shapes
    (nx, N * d) and (nu, N * d).
    """
    m, d = ctx.m23, ctx.OPT_d
    grid = discretise(ctx.track, ctx.OPT_ds, d, mesh=ctx.mesh)
    n = grid["N"]
    Uk = w[m.nx * (n + 1):(m.nx + m.nu) * (n + 1)].reshape(m.nu, n + 1, order="F")
    Xkj = w[(m.nx + m.nu) * (n + 1):].reshape(m.nx, n * d, order="F")
    return Xkj, interp_inputs(Uk, grid["s_knot"], grid["s_col"], "linear")


rng = np.random.RandomState(1)

# ---- 6. NLP rows through MLTP (Sturn, fake solver) ----------------------------------------
print("6. NLP rows through MLTP (Sturn, fake solver)")
ctx_m, rec_m = mltp()
ctx_l, rec_l = mltp(between_knots="linear")
ctx_p, rec_p = mltp(between_knots="linear+power")
st_m, st_l, st_p = (x.data.nlp["structure"] for x in (ctx_m, ctx_l, ctx_p))
n_g0 = st_m["n_g"]
ok("matlab: n_g 1904 and n_pathcol 0, the default NLP keeps its size", n_g0 == 1904 and st_m["n_pathcol"] == 0)
ok("linear: the matlab sizes", st_l["n_g"] == n_g0 and st_l["n_w"] == st_m["n_w"] and st_l["n_pathcol"] == 0)
ok("linear+power: n_g + 54 (one row per collocation point), the record equals nlp_structure",
   st_p == nlp_structure(23, 7, 0, 18, 3, 4, n_pathcol=54))

f_m, f_l, f_p = g_of(rec_m), g_of(rec_l), g_of(rec_p)
w = rec_m["x0"] * (1 + 0.05 * rng.randn(rec_m["x0"].size))   # a generic point, no row is trivially zero
g_m, g_l, g_p = (np.asarray(f(w)[1]).ravel() for f in (f_m, f_l, f_p))
ix = nlp_index(st_p)
other = np.setdiff1d(np.arange(n_g0), ix["defect"].ravel())
ok("linear: only the defect rows differ from the matlab build",
   np.array_equal(g_l[other], g_m[other]) and np.abs(g_l - g_m).max() > 1e-3)
ok("linear+power: its first n_g rows are the linear build's rows", np.array_equal(g_p[:n_g0], g_l))

m = ctx_p.m23
nx, nu, d = m.nx, m.nu, ctx_p.OPT_d
N = st_p["N"]
Xk = w[:nx * (N + 1)].reshape(nx, N + 1, order="F")
Uk = w[nx * (N + 1):(nx + nu) * (N + 1)].reshape(nu, N + 1, order="F")
Xkj = w[(nx + nu) * (N + 1):].reshape(nx, N * d, order="F")
grid = discretise(ctx_p.track, ctx_p.OPT_ds, d, mesh=ctx_p.mesh)
u_pts = interp_inputs(Uk, grid["s_knot"], grid["s_col"], "linear")
f_dyn = ca.Function("f_dyn", [m.x, m.u, m.pv], [m.dx, m.sf])
dX = np.asarray(f_dyn.map(N * d)(Xkj, u_pts, grid["pv_col"])[0])
C, dsk7 = np.asarray(grid["C"], float), np.asarray(grid["dsk"], float)
ref = np.empty((nx, N * d))
for k in range(N):
    cols = slice(d * k, d * k + d)
    ref[:, cols] = dsk7[k] * dX[:, cols] - np.hstack([Xk[:, k:k + 1], Xkj[:, cols]]) @ C
ok("linear: the defect rows equal the dynamics at the np.interp inputs (numpy, 1e-9)",
   np.abs(g_l[ix["defect"]] - ref).max() < 1e-9)

pt, vp = ctx_p.pt, ctx_p.vp
Om = Xkj[5:9] * np.asarray(m.x_s)[5:9, None]                 # wheel speeds [rad/s]
P_pts = u_pts[0] * m.u_s[0] * Om.sum(axis=0) / 4 * vp.gear   # T * Om_motor [W], Om_motor = gear * mean(Om)
ok("linear+power: the new rows are (Pmax - T Om_motor) / Pmax at the collocation points, in point order (1e-12)",
   np.abs(g_p[n_g0:] - (pt.Pmax - P_pts) / pt.Pmax).max() < 1e-12)
ok("the new rows have the bounds of the knot power row, [0, 1]",
   np.all(rec_p["lbg"][n_g0:] == 0.0) and np.all(rec_p["ubg"][n_g0:] == 1.0))


def init_nlp(**kw):
    """Serialised (f, g) function of the 7-state init NLP that MLTP_initial builds with kw."""
    rec = {}
    with fake_ipopt(rec), contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_mod.MLTP_initial(circuit="Sturn", save=False, **kw)
    nlp = rec["nlp"]
    return ca.Function("fg", [nlp["x"]], [nlp["f"], nlp["g"]]).serialize()


ok("the 7-state init builds the same NLP in every mode (MATLAB arithmetic, no extra row)",
   init_nlp() == init_nlp(between_knots="linear") == init_nlp(between_knots="linear+power"))

# ---- 7. EM4, regen and pack rows, and the periodic lap ------------------------------------
print("7. rows for EM4, regen and pack, and the periodic lap")
ctx_e, _ = mltp(between_knots="linear+power", Electric_4Motors="On", ATD="Off")
ok("EM4 under physics 'matlab': no rows, n_g 2038 (the linear build's size)",
   ctx_e.data.nlp["structure"]["n_pathcol"] == 0 and ctx_e.data.nlp["structure"]["n_g"] == 2038)

ctx_c, rec_c = mltp(between_knots="linear+power", Electric_4Motors="On", ATD="Off", physics="corrected")
st_c = ctx_c.data.nlp["structure"]
ok("EM4 under physics 'corrected': 4 rows per point, n_g 2254",
   st_c == nlp_structure(23, 6, 0, 18, 3, 12, n_pathcol=216) and st_c["n_g"] == 2254)
w_c = rec_c["x0"] * (1 + 0.05 * rng.randn(rec_c["x0"].size))
Xc, uc = at_points(ctx_c, w_c)
mc = ctx_c.m23
P4 = np.array([uc[i] * mc.u_s[i] * Xc[5 + i] * mc.x_s[5 + i] * ctx_c.vp.gear for i in range(4)])  # T_i Om_i gear
rows_c = np.asarray(g_of(rec_c)(w_c)[1]).ravel()[st_c["n_g"] - 216:].reshape(-1, 4)
ok("EM4 rows: (Pmax - T_i Om_i gear) / Pmax, the four motors of a point together (1e-12)",
   np.abs(rows_c - (ctx_c.pt.Pmax - P4.T) / ctx_c.pt.Pmax).max() < 1e-12)

ctx_r, rec_r = mltp(between_knots="linear+power", physics="corrected",
                    vp_overrides=dict(Preg=50e3, Ppack=300e3))
st_r = ctx_r.data.nlp["structure"]
w_r = rec_r["x0"] * (1 + 0.05 * rng.randn(rec_r["x0"].size))
Xr, ur = at_points(ctx_r, w_r)
mr = ctx_r.m23
P_r = ur[0] * mr.u_s[0] * (Xr[5:9] * np.asarray(mr.x_s)[5:9, None]).sum(axis=0) / 4 * ctx_r.vp.gear
rows_r = np.asarray(g_of(rec_r)(w_r)[1]).ravel()[st_r["n_g"] - 108:].reshape(-1, 2)
ok("regen and pack rows: (Pmax - P) / Pmax, then (Ppack - P) / Ppack, at each point (1e-12)",
   np.abs(rows_r[:, 0] - (ctx_r.pt.Pmax - P_r) / ctx_r.pt.Pmax).max() < 1e-12
   and np.abs(rows_r[:, 1] - (300e3 - P_r) / 300e3).max() < 1e-12)
ok("regen and pack rows: 108 rows, lower bounds 0, upper bounds 1 + Preg / Pmax and inf",
   st_r["n_pathcol"] == 108 and np.array_equal(rec_r["lbg"][-108:], np.zeros(108))
   and np.allclose(rec_r["ubg"][-108:][0::2], 1 + 50e3 / ctx_r.pt.Pmax)
   and np.all(np.isinf(rec_r["ubg"][-108:][1::2])))

ctx_o, rec_o = mltp(circuit="Circle", between_knots="linear+power", start_state="periodic")
ctx_o1, rec_o1 = mltp(circuit="Circle", between_knots="linear", start_state="periodic")
st_o = ctx_o.data.nlp["structure"]
Nc = st_o["N"]
w_o = rec_o["x0"] * (1 + 0.05 * rng.randn(rec_o["x0"].size))
g_o = np.asarray(g_of(rec_o)(w_o)[1]).ravel()
g_o1 = np.asarray(g_of(rec_o1)(w_o)[1]).ravel()
ok("periodic lap: structure n_periodic 30, n_pathcol 3 N",
   st_o == nlp_structure(23, 7, 0, Nc, 3, 4, n_periodic=30, n_pathcol=3 * Nc))
ok("periodic lap: g is the linear build's g, then the path rows last",
   g_o.size == g_o1.size + 3 * Nc and np.array_equal(g_o[:g_o1.size], g_o1))

# ---- 8. warm-start modes, channels, saved fields, Pmax price, refusals --------------------
print("8. warm-start modes, channels, saved fields, Pmax price and refusals")
if os.path.exists(STURN_RESULT):
    cm, _ = mltp(warm_start=STURN_RESULT)
    cl, _ = mltp(warm_start=STURN_RESULT, between_knots="linear")
    ok("stored default result: full+duals into a matlab solve, full-interp into a linear solve",
       cm.elapsed["warm_start"] == "full+duals" and cl.elapsed["warm_start"] == "full-interp")

    def torque(x):
        """Motor torque [Nm] at every point, from the vehicle channels (throttle is torque / Tmax)."""
        return np.asarray(x.data.vehicle_full["throttle"]) * x.m23.u_s[0]
    ok("linear: the channels at every point use the linear input, so they equal u_full",
       np.abs(torque(cl) - np.asarray(cl.data.u_full)[0]).max() < 1e-8)
    ok("matlab: the channels use the near hold, which differs from u_full (E1 stage M)",
       np.abs(torque(cm) - np.asarray(cm.data.u_full)[0]).max() > 1.0)
else:                                                             # pragma: no cover
    print(f"  SKIP warm-start and channel checks: {STURN_RESULT} missing")

ok("matlab: the result has no between_knots field (absent means 'matlab')", not hasattr(ctx_m.data, "between_knots"))
with tempfile.TemporaryDirectory() as tmp:
    mltp(warm_start=INIT, between_knots="linear+power", save=True, results_dir=tmp)
    mats = [f for f in os.listdir(tmp) if f.endswith(".mat")]
    back = importfile(os.path.join(tmp, mats[0]))["data"] if len(mats) == 1 else None
    ok("the saved result holds between_knots 'linear+power' and n_pathcol 54",
       back is not None and str(back.between_knots) == "linear+power"
       and int(back.nlp.structure.n_pathcol) == 54)
    c2, r2 = mltp(warm_start=os.path.join(tmp, mats[0]), between_knots="linear+power")
    ok("the saved linear+power result seeds a linear+power solve with its multipliers, no 'out of sync' warning",
       c2.elapsed["warm_start"] == "full+duals" and not any("out of sync" in w for w in r2["warnings"]))

ok("limit_prices: Pmax is NaN with path rows at the collocation points, finite without them",
   np.isnan(limit_prices(ctx_p.data)["Pmax"]) and np.isfinite(limit_prices(ctx_l.data)["Pmax"]))

with fake_ipopt({}):
    msg = raised(ValueError, po_mod.optimise_design, [("brkB", 0.0, 1.0)], "x", circuit="Sturn",
                 between_knots="linear", save=False)
ok("optimise_design raises ValueError for between_knots 'linear'", msg is not None and "between_knots" in msg)
design = ([("brkB", 0.0, 1.0)], "x")
built = quiet(po_mod.optimise_design, *design, circuit="Sturn", warm_start=ctx_m, build_only=True, save=False)
msg = raised(ValueError, po_mod.optimise_design, *design, circuit="Sturn", warm_start=ctx_l, build_only=True,
             save=False)
ok("optimise_design: a matlab source gives multipliers, a linear source of the same size gives none (ValueError)",
   "lam_g0" in built and msg is not None and "between_knots" in msg)

with tempfile.TemporaryDirectory() as tmp, fake_ipopt({}):
    msg = raised(SS.SweepError, SS.setup_sweep, [("alpha_RW", 4, 16)], 4, circuit="Sturn", results_root=tmp,
                 confirm=False, plot=False, verbose=False, workers=0, between_knots="linear")
    ok("setup_sweep raises SweepError for between_knots 'linear' and writes no file",
       msg is not None and "between_knots" in msg and os.listdir(tmp) == [])
base = SimpleNamespace(input_keys=list(ctx_m.input_keys), OPT_d=3, tyre_set="MF205", start_state="matlab")
grid_m = {"N": 18, "s_full": np.asarray(ctx_m.data.s_full)}
msg = raised(SS.SweepError, SS._precheck_base, ctx_l.data, base, grid_m, 4, "Sturn", "base")
ok("setup_sweep base check: a matlab result passes, a linear result of the same size raises SweepError",
   raised(SS.SweepError, SS._precheck_base, ctx_m.data, base, grid_m, 4, "Sturn", "base") is None
   and msg is not None and "between_knots" in msg)

# ---- 9. refine: the reseed and the defect indicator follow the flag -----------------------
print("9. refine: the reseed and the defect indicator use the linear input")
seen = {"defect_errors": [], "nlp_inputs": []}


def spy(name):
    """mltp_mod.<name> that logs the linear_inputs value of each call, the default value included."""
    fn = getattr(mltp_mod, name)
    sig = inspect.signature(fn)

    def logged(*a, **kw):
        seen[name].append(sig.bind(*a, **kw).arguments.get("linear_inputs", False))
        return fn(*a, **kw)
    return logged


real_fns = mltp_mod.defect_errors, mltp_mod.nlp_inputs
mltp_mod.defect_errors, mltp_mod.nlp_inputs = spy("defect_errors"), spy("nlp_inputs")
try:
    mltp(between_knots="linear", refine=dict(passes=1), status="Solve_Succeeded")
finally:
    mltp_mod.defect_errors, mltp_mod.nlp_inputs = real_fns
ok("linear + refine: each defect_errors call and each nlp_inputs call (the reseed and the channels) gets "
   "linear_inputs=True", len(seen["defect_errors"]) >= 1 and set(seen["defect_errors"]) == {True}
   and len(seen["nlp_inputs"]) >= 2 and set(seen["nlp_inputs"]) == {True})

print("ALL B3 between-knots TESTS PASSED")
