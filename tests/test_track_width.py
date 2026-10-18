"""Track width / per-point lateral bounds (plan docs/superpowers/plans/2026-10-18-phase2.md, Task 6: C1).

userOpts(n_bounds=...) turns a width W (band -W..W) or a (lo, hi) pair (scalars or arrays on
track.s) into ctx.track.n_lo / n_hi and the file-name tag ctx.n_tag. discretise() carries the band
to the collocation grid, build_and_solve_nlp patches the n bounds per point (the default path is
unchanged) and raises ValueError on an empty start / end box, reconstruct_track draws the edges at
the band, and result_stem / MLTP / MLTP_initial / MLTP_screen put the tag in the file name.

Sections 1-4, 10, 13a and 15 are casadi-free. Section 16 needs Data/DATA_AA.mat (the screen
envelope reads ctx.aero) and prints SKIP without it. Sections 5-9, 11-14 (and 13b) need casadi and
Data/DATA_AA.mat (SKIP, exit 0 otherwise). Sections 5-8, 12 and 14 build the real NLP through a fake
solver, section 13b runs MLTP through stand-ins, and no section runs IPOPT. Run from anywhere:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_track_width.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import atexit
import contextlib
import inspect
import io
import os
import shutil
import sys
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np

import functions.transcription as T
from functions.context import Ctx
from functions.importfile import load_solution, result_stem
from functions.trackLimits import trackLimits
from functions.transcription import discretise, reconstruct_track
from functions.warmstart import nlp_structure, plan_full_warm_start
from MLTP_screen import MLTP_screen
from userOpts import userOpts

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
TMP = tempfile.mkdtemp(prefix="test_track_width_")
atexit.register(shutil.rmtree, TMP, True)


def _no_ipopt(*a, **k):
    raise RuntimeError("test_track_width never runs IPOPT: a build reached the real _make_solver")


T._make_solver = _no_ipopt  # fake_ipopt() swaps in its own solver for build_only()


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


def opts(**kw):  # userOpts only: no vehModel, no casadi
    c = Ctx()
    quiet(userOpts, c, **dict(dict(circuit="Sturn"), **kw))
    return c


# ---- 1. default: no band, no tag (casadi-free) ------------------------------------
print("1. default Sturn: no band, no tag")
c = opts()
ok("no n_lo on the track by default", getattr(c.track, "n_lo", None) is None)
ok("discretise carries no n_bounds by default", discretise(c.track, 30, 3)["n_bounds"] is None)
ok("n_tag is empty by default", c.n_tag == "")

# ---- 2. the accepted forms (casadi-free) ------------------------------------------
print("2. n_bounds forms: width, (lo, hi) scalars, (lo, hi) arrays")
L = len(c.track.s)
c = opts(n_bounds=3.5)
ok("W=3.5: n_lo == -3.5 on track.s", np.array_equal(c.track.n_lo, np.full(len(c.track.s), -3.5)))
ok("W=3.5: n_hi == 3.5 on track.s", np.array_equal(c.track.n_hi, np.full(len(c.track.s), 3.5)))
ok("W=3.5: n_tag hw3p5", c.n_tag == "hw3p5")
ok("W=4 (int): n_tag hw4", opts(n_bounds=4).n_tag == "hw4")
c = opts(n_bounds=(-2.0, 5.0))
ok("(-2, 5): full arrays, n_lo -2 and n_hi 5",
   np.array_equal(c.track.n_lo, np.full(L, -2.0)) and np.array_equal(c.track.n_hi, np.full(L, 5.0)))
ok("(-2, 5): n_tag nbCustom", c.n_tag == "nbCustom")
lo_arr, hi_arr = -np.linspace(3.0, 5.0, L), np.linspace(3.0, 5.0, L)
c = opts(n_bounds=(lo_arr, hi_arr))
ok("(lo, hi) arrays of len(track.s) kept",
   np.array_equal(c.track.n_lo, lo_arr) and np.array_equal(c.track.n_hi, hi_arr))
ok("(lo, hi) arrays: n_tag nbCustom", c.n_tag == "nbCustom")

# ---- 3. invalid values raise ValueError (casadi-free) -----------------------------
print("3. invalid n_bounds raise ValueError")
bad = [("0", 0), ("-1", -1), ("nan", np.nan), ("inf", np.inf), ("True (bool)", True), ("'4' (str)", "4"),
       ("lo > hi", (1.0, 0.5)), ("lo == hi (scalars)", (1.0, 1.0)),
       ("lo == hi at one point", (np.zeros(L), np.r_[0.0, np.ones(L - 1)])),  # fixed line
       ("wrong array length", (np.zeros(5), np.ones(5))),
       ("3-tuple", (0, 1, 2))]
for label, v in bad:
    ok(f"n_bounds={label}: ValueError",
       raised(ValueError, userOpts, Ctx(), circuit="Sturn", n_bounds=v) is not None)

# ---- 4. discretise carries the band to the collocation grid (casadi-free) ---------
print("4. discretise: n_bounds on s_full")
s = c.track.s
hi = np.linspace(3.0, 5.0, len(s))
c = opts(n_bounds=(-hi, hi))
disc = discretise(c.track, 30, 3)
ok("n_bounds[1] == interp(s_full, track.s, hi)",
   np.array_equal(disc["n_bounds"][1], np.interp(disc["s_full"], c.track.s, hi)))
ok("n_bounds[0] == interp(s_full, track.s, lo)",
   np.array_equal(disc["n_bounds"][0], np.interp(disc["s_full"], c.track.s, -hi)))

# ---- 10. reconstruct_track edges (casadi-free) -------------------------------------
print("10. reconstruct_track: edges at the band, or the fixed 8 m trackLimits")
s10 = np.linspace(0.0, 100.0, 101)
zero10 = np.zeros(101)
s_full10 = np.linspace(0.0, 100.0, 41)
band = SimpleNamespace(s=s10, x=s10, y=zero10, k=zero10,
                       n_lo=np.full(101, -1.0), n_hi=np.full(101, 3.0))
rec = reconstruct_track(band, s_full10, np.zeros(41), 0.0, 8.0)
ok("band: left edge at n_hi (y = 3)", np.allclose(rec["Xl"][:, 1], 3, atol=1e-12))
ok("band: right edge at n_lo (y = -1)", np.allclose(rec["Xr"][:, 1], -1, atol=1e-12))
ok("band: returned dict has n_lo and n_hi", "n_lo" in rec and "n_hi" in rec)
plain = SimpleNamespace(s=s10, x=s10, y=zero10, k=zero10)
rec0 = reconstruct_track(plain, s_full10, np.zeros(41), 0.0, 8.0)
xi, yi = np.interp(s_full10, s10, s10), np.interp(s_full10, s10, zero10)
ok("no band: Xl == trackLimits(8 m) bit for bit", np.array_equal(rec0["Xl"], trackLimits(xi, yi, 8.0)[0]))
ok("no band: Xr == trackLimits(8 m) bit for bit", np.array_equal(rec0["Xr"], trackLimits(xi, yi, 8.0)[1]))

# ---- 13a. result_stem tag (casadi-free) -------------------------------------------
print("13a. result_stem: n_tag after the start suffix, before the mesh suffix")
ok("n_tag only: BCN_Static_ATDOn_EM4Off_hw3p5",
   result_stem("BCN", "Static_ATDOn_EM4Off", n_tag="hw3p5") == "BCN_Static_ATDOn_EM4Off_hw3p5")
ok("rolling + hw4p2 + curvature stem",
   result_stem("BCN", "Static_ATDOn_EM4Off", "MF205", "curvature", "matlab", "rolling", "hw4p2")
   == "BCN_Static_ATDOn_EM4Off_startRolling_hw4p2_meshCurvature")
ok("default stem unchanged", result_stem("BCN", "Static_ATDOn_EM4Off") == "BCN_Static_ATDOn_EM4Off")

# ---- 15. periodic Spa: band length on the closed-lap grid (casadi-free) -----------
print("15. periodic Spa: band arrays on the 1402-point closed grid")
cs = quiet(userOpts, Ctx(), circuit="Spa", start_state="periodic",
           n_bounds=(np.full(1402, -3.0), np.full(1402, 3.0)))
ok("periodic Spa: n_lo has 1402 points (file 1401 + closing sample)", cs.track.n_lo.size == 1402)
msg15 = raised(ValueError, userOpts, Ctx(), circuit="Spa", start_state="periodic",
               n_bounds=(np.full(1401, -3.0), np.full(1401, 3.0)))
ok("periodic Spa: 1401 arrays raise ValueError naming 1402", msg15 is not None and "1402" in msg15)

# ---- 16. screen file tag (needs the aero file: ctx.aero) --------------------------
print("16. MLTP_screen file name carries the band tag")
if os.path.exists(DATA_AA):
    quiet(MLTP_screen, circuit="Sturn", n_bounds=3.5, save=True, results_dir=TMP, verbose=False)
    ok("screen writes Sturn_Static_ATDOn_EM4Off_hw3p5_qss.mat",
       os.path.exists(os.path.join(TMP, "Sturn_Static_ATDOn_EM4Off_hw3p5_qss.mat")))
    ok("screen leaves the default _qss stem alone",
       not os.path.exists(os.path.join(TMP, "Sturn_Static_ATDOn_EM4Off_qss.mat")))
else:
    print("  SKIP 16: Data/DATA_AA.mat missing (the screen envelope needs ctx.aero)")

# ---- casadi sections: need the aero data file -------------------------------------
if not os.path.exists(DATA_AA):
    print("SKIP test_track_width casadi sections (5-9, 11, 12, 13b, 14): Data/DATA_AA.mat missing")
    sys.exit(0)

import casadi as ca                             # noqa: E402
import MLTP as mltp_mod                         # noqa: E402
import MLTP_initial as mi_mod                   # noqa: E402
from vehModel import vehModel                   # noqa: E402
from vehModel_initial import vehModel_initial   # noqa: E402


def model(**kw):  # userOpts + the 23-state model
    c = Ctx()
    quiet(userOpts, c, **dict(dict(circuit="Sturn"), **kw))
    vehModel(c)
    return c


_SIG = inspect.signature(T.build_and_solve_nlp)


class _Captured(Exception):
    pass


def capture(module, fn, **kw):
    """module.fn(**kw) up to its build_and_solve_nlp call: the bound arguments (only those passed).
    The stand-in raises before the build, so nothing is assembled or solved."""
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
    def fake(ca_, nlp, iopts, info=None):
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
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=U7)}  # in-memory 7-state init (warm_start=INIT)


def run_stub(calls=None, **kw):
    """MLTP(**kw) with MLTP_initial -> INIT and build_and_solve_nlp -> the packed guesses as the
    solution (zero duals, Solve_Succeeded); calls gets every build's bound arguments.
    Defaults: Sturn, uniform mesh, save/plot off."""
    calls = [] if calls is None else calls

    def init_stub(**k):
        return SimpleNamespace(data=SimpleNamespace(init=INIT["init"]),
                               elapsed=dict(ipopt_iters=1, return_status="Solve_Succeeded", qss=0.0))

    def nlp_stub(*a, **k):
        b = _SIG.bind(*a, **k).arguments
        calls.append(b)
        m, disc_, g = b["m"], b["disc"], b["guesses"]
        extra = {"n_periodic": m.nx + m.nu} if b.get("periodic") else {}
        st = nlp_structure(m.nx, m.nu, m.ny, disc_["N"], b["OPT_d"], np.size(b["h_lb"]), **extra)
        w = np.concatenate([g["x0"].reshape(-1, order="F"), g["u0"].reshape(-1, order="F"),
                            np.asarray(g["xc0"]).reshape(-1, order="F")])
        stats = {"iter_count": 7, "return_status": "Solve_Succeeded"}
        return dict(sol={"x": w, "f": 0.0}, solver=SimpleNamespace(stats=lambda: dict(stats)),
                    w_opt=w, lam_g=np.zeros(st["n_g"]), lam_x=np.zeros(st["n_w"]), structure=st,
                    warm_info=dict(x0=False, lam_g0=False, lam_x0=False, ipopt=False, duals=False),
                    sym_type="SX", linear_solver="fake", N=disc_["N"])

    real = (mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp)
    mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp = init_stub, nlp_stub
    try:
        return quiet(mltp_mod.MLTP, **dict(dict(circuit="Sturn", save=False, plot=False, mesh="uniform"), **kw))
    finally:
        mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp = real


# ---- 5. n_bounds=4.0 == default, bit for bit (casadi, fake IPOPT) -----------------
print("5. n_bounds=4.0 == default bounds (the per-point patch at the model's own +-4 m)")
a0 = capture(mltp_mod, "MLTP", circuit="Sturn", warm_start=INIT, mesh="uniform", save=False, plot=False)
a4 = capture(mltp_mod, "MLTP", circuit="Sturn", n_bounds=4.0, warm_start=INIT, mesh="uniform",
             save=False, plot=False)
_, q0 = build_only(a0)
_, q4 = build_only(a4)
for key in ("lbx", "ubx", "lbg", "ubg"):
    ok(f"n_bounds=4.0: {key} identical to the default build", np.array_equal(q4[key], q0[key]))

# ---- 6. n_bounds=3.5 on the 23-state NLP (casadi, fake IPOPT) ---------------------
print("6. n_bounds=3.5, 23-state: n rows at knots and collocation points, start / end box rows")
b = 3.5 * (1.0 / 5.0)
a = capture(mltp_mod, "MLTP", circuit="Sturn", n_bounds=3.5, warm_start=INIT, mesh="uniform",
            save=False, plot=False)
_, q = build_only(a)
N = a["disc"]["N"]
ok("Sturn grid is N = 18, OPT_d = 3", N == 18 and a["OPT_d"] == 3)
knot = 3 + 23 * np.arange(N + 1)                          # w rows of n at the knots
colloc = (23 + 7) * (N + 1) + 3 + 23 * np.arange(3 * N)   # w rows of n at the collocation points
ok("knot rows: ubx == b and lbx == -b", np.all(q["ubx"][knot] == b) and np.all(q["lbx"][knot] == -b))
ok("collocation rows: ubx == b and lbx == -b",
   np.all(q["ubx"][colloc] == b) and np.all(q["lbx"][colloc] == -b))
ok("start box row of n: lbg == -b and ubg == b", q["lbg"][3] == -b and q["ubg"][3] == b)
ok("end box row of n: lbg == -b and ubg == b", q["lbg"][26] == -b and q["ubg"][26] == b)
mx = np.zeros(q["lbx"].size, dtype=bool)
mx[knot] = True
mx[colloc] = True
mg = np.zeros(q["lbg"].size, dtype=bool)
mg[[3, 26]] = True
ok("every other lbx entry == default build", np.array_equal(q["lbx"][~mx], q0["lbx"][~mx]))
ok("every other ubx entry == default build", np.array_equal(q["ubx"][~mx], q0["ubx"][~mx]))
ok("every other lbg entry == default build", np.array_equal(q["lbg"][~mg], q0["lbg"][~mg]))
ok("every other ubg entry == default build", np.array_equal(q["ubg"][~mg], q0["ubg"][~mg]))

# ---- 7. 7-state init: the same band on its own layout (casadi, fake IPOPT) --------
print("7. n_bounds=3.5, 7-state init (MLTP_initial)")
a7 = capture(mi_mod, "MLTP_initial", circuit="Sturn", n_bounds=3.5, save=False)
_, q7 = build_only(a7)
N7 = a7["disc"]["N"]
k7 = 3 + 7 * np.arange(N7 + 1)
c7 = (7 + 3 + 1) * (N7 + 1) + 3 + 7 * np.arange(3 * N7)
ok("7-state knot rows: ubx == b", np.all(q7["ubx"][k7] == b))
ok("7-state collocation rows: ubx == b", np.all(q7["ubx"][c7] == b))
ok("7-state knot and collocation rows: lbx == -b", np.all(q7["lbx"][k7] == -b) and np.all(q7["lbx"][c7] == -b))

# ---- 8. start box centred on ni; ramp band on knots and collocation points --------
print("8. ni=1.0 start box; ramp band exact on knots and collocation points")
a = capture(mltp_mod, "MLTP", circuit="Sturn", ni=1.0, n_bounds=3.5, warm_start=INIT, mesh="uniform",
            save=False, plot=False)
_, q = build_only(a)
ok("ni=1.0: start box lower row == 0.19", abs(q["lbg"][3] - 0.19) < 1e-15)
ok("ni=1.0: start box upper row == 0.21", abs(q["ubg"][3] - 0.21) < 1e-15)
s8 = opts().track.s
hi8 = np.linspace(3.0, 5.0, len(s8))
ar = capture(mltp_mod, "MLTP", circuit="Sturn", n_bounds=(-hi8, hi8), warm_start=INIT, mesh="uniform",
             save=False, plot=False)
_, qr = build_only(ar)
dr = ar["disc"]
kn = 3 + 23 * np.arange(dr["N"] + 1)
cl = (23 + 7) * (dr["N"] + 1) + 3 + 23 * np.arange(3 * dr["N"])
ok("ramp: knot ubx == interp(s_knot, hi) * (1/5) exactly",
   np.array_equal(qr["ubx"][kn], np.interp(dr["s_knot"], s8, hi8) * (1.0 / 5.0)))
ok("ramp: collocation ubx == interp(s_col, hi) * (1/5) exactly",
   np.array_equal(qr["ubx"][cl], np.interp(dr["s_col"], s8, hi8) * (1.0 / 5.0)))

# ---- 9. row 3 is n in both models (casadi) ----------------------------------------
print("9. state row 3 is n in both models")
c9 = model()
ok("23-state x[3] is n_n", str(c9.m23.x[3]) == "n_n")
vehModel_initial(c9)
ok("7-state x[3] is n_n", str(c9.m7.x[3]) == "n_n")

# ---- 11. the band is not NLP structure: stored duals still re-inject (casadi) ----
print("11. warm start with n_bounds=4.1: full+duals (bounds are not structure)")
d11 = load_solution(STURN)
c41 = opts(n_bounds=4.1)
disc41 = discretise(c41.track, c41.OPT_ds, c41.OPT_d, mesh=c41.mesh, mesh_opts=c41.mesh_opts)
expected = dict(nlp_structure(23, 7, 0, 18, 3, 4), input_keys=d11.input_keys, s_full=disc41["s_full"],
                tyre_set="MF205", start_state="matlab")
_, mode11, _ = plan_full_warm_start(d11, expected)
ok("n_bounds=4.1: warm start mode full+duals", mode11 == "full+duals")

# ---- 12. empty start / end box raises, always (casadi, fake IPOPT) ----------------
print("12. empty start box raises ValueError")
a = capture(mltp_mod, "MLTP", circuit="Sturn", warm_start=INIT, mesh="uniform", save=False, plot=False)
Xi5 = np.array(a["Xi"], dtype=float).copy()
Xi5[3] = 5.0                                              # start n outside even the model's +-4 m
msg12 = raised(ValueError, build_only, a, Xi=Xi5)
ok("ni=5 m (default bounds): ValueError naming an empty box", msg12 is not None and "empty" in msg12)
a3 = capture(mltp_mod, "MLTP", circuit="Sturn", ni=3.0, n_bounds=2.5, warm_start=INIT, mesh="uniform",
             save=False, plot=False)
msg12b = raised(ValueError, build_only, a3)
ok("ni=3 m with a 2.5 m band: ValueError naming an empty box", msg12b is not None and "empty" in msg12b)
a2 = capture(mltp_mod, "MLTP", circuit="Sturn", ni=2.0, n_bounds=2.5, warm_start=INIT, mesh="uniform",
             save=False, plot=False)
_, q2 = build_only(a2)
ok("ni=2 m with a 2.5 m band: builds", "lbx" in q2)

# ---- 13b. save path: tag in the file name, band on the saved track (casadi) -------
print("13b. MLTP save with n_bounds=3.5: tag in the file name, band on the saved track")
run_stub(n_bounds=3.5, save=True, results_dir=TMP, mesh="auto")
path13 = os.path.join(TMP, "Sturn_Static_ATDOn_EM4Off_hw3p5.mat")
ok("file Sturn_Static_ATDOn_EM4Off_hw3p5.mat written", os.path.exists(path13))
d13 = load_solution(path13)
ok("saved data.n_tag == hw3p5", str(d13.n_tag) == "hw3p5")
ok("saved data.track.n_hi == 3.5 everywhere", np.all(d13.track.n_hi == 3.5))

# ---- 14. narrow band around a sampled line: continuity stays feasible (casadi) ----
print("14. narrow band (+-0.05 m) around a sampled line: continuity holds after projection")
S = opts().track.s
nr = 3.0 * np.sin(2 * np.pi * S / 100.0)
a = capture(mltp_mod, "MLTP", circuit="Sturn", n_bounds=(nr - 0.05, nr + 0.05), warm_start=INIT,
            mesh="uniform", save=False, plot=False)
_, q = build_only(a)
d14 = a["disc"]
N14 = d14["N"]
D14 = np.asarray(d14["D"], dtype=float).ravel()
mid = np.interp(d14["s_full"], S, nr) / 5.0               # the line in scaled units (x_s[3] = 5)
knot14 = 4 * np.arange(N14 + 1)
col14 = np.delete(np.arange(4 * N14 + 1), knot14)
nK, nC = mid[knot14], mid[col14].reshape(N14, 3)
res14 = D14[0] * nK[:-1] + nC @ D14[1:] - nK[1:]          # continuity residual of the line
nC = nC - res14[:, None] * np.sign(D14[1:]) / np.abs(D14[1:]).sum()   # projected, still in band
w = q["x0"].copy()
w[3 + 23 * np.arange(N14 + 1)] = nK
w[573 + 23 * np.arange(3 * N14)] = nC.ravel()
gw = np.asarray(ca.Function("g", [q["nlp"]["x"]], [q["nlp"]["g"]])(w)).ravel()
ok("n continuity rows vanish after the projection (|g| < 1e-12)",
   np.abs(gw[118 + 92 * np.arange(N14)]).max() < 1e-12)
nidx = np.r_[3 + 23 * np.arange(N14 + 1), 573 + 23 * np.arange(3 * N14)]
ok("projected n entries inside the band", np.all(q["lbx"][nidx] <= w[nidx]) and np.all(w[nidx] <= q["ubx"][nidx]))
