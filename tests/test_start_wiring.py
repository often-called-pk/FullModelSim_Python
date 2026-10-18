"""Start state wiring (plan docs/superpowers/plans/2026-10-18-phase2.md, Task 5: B1 / B2 wiring
and the rolling equilibrium).

rolling_start() is the straight-line coast equilibrium of the 23-state model, not
quasi_static_states: the residual rows, the pinned matlab / corrected values, the variants, the
bounds and the Newton failures. MLTP, MLTP_initial and optimise_design forward Xi_tol, periodic
and start_state to the NLP build; MLTP fills ctx.Xi for a rolling start, warns on a curved one,
passes Xi_tol to the refine passes; the sweep precheck checks the periodic rows and start_state;
the result files get the start suffix.

Run order: sections 7 and 10 first (casadi-free), then sections 1-6, 8, 9 and 11 in number order.
Sections 1-6, 8, 9 and 11 build the real NLP through a stand-in (no IPOPT) and need casadi and
Data/DATA_AA.mat (SKIP, exit 0 otherwise). Run from anywhere:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_start_wiring.py
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
from functions.ladder import boundary_box
from functions.transcription import discretise
from functions.warmstart import nlp_structure
from MLTP_screen import MLTP_screen
from userOpts import userOpts
import setup_sweep as SS

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
nan = np.nan


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


WARNS = []


def quiet(fn, *a, **kw):
    """fn(*a, **kw), stdout swallowed; warning texts recorded into WARNS (not shown)."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            return fn(*a, **kw)
        finally:
            WARNS[:] = [str(x.message) for x in w]


def raised(exc, fn, *a, **kw):
    """str(exc) of the exception that fn(*a, **kw) raises, or None when it raises none."""
    try:
        quiet(fn, *a, **kw)
    except exc as e:
        return str(e)
    return None


# ---- 7. sweep precheck: periodic rows and start_state (casadi-free) ---------------
print("7. setup_sweep._precheck_base: n_periodic and start_state")
bc = quiet(userOpts, Ctx(), circuit="Circle")
disc = discretise(bc.track, bc.OPT_ds, bc.OPT_d, mesh=bc.mesh)
N = disc["N"]
n_in = len(bc.input_keys)


def base_src(structure, start_state=None):
    """A solution-shaped stand-in for a base result (the fields _precheck_base reads)."""
    src = SimpleNamespace(x_opt=np.zeros((23, N + 1)),
                          nlp=SimpleNamespace(structure=structure, w_opt=np.zeros(structure["n_w"])),
                          input_keys=list(bc.input_keys), s_full=disc["s_full"],
                          tyre_set="MF205", circuit="Circle")
    if start_state is not None:
        src.start_state = start_state
    return src


st_per = nlp_structure(23, n_in, 0, N, bc.OPT_d, 4, n_periodic=30)
msg = raised(SS.SweepError, SS._precheck_base, base_src(st_per, "periodic"), bc, disc, 4, "Circle", "<x>")
ok("periodic base into plain sweep: SweepError names n_periodic", msg is not None and "n_periodic" in msg)
bp = quiet(userOpts, Ctx(), circuit="Circle", start_state="periodic")
ok("periodic base into periodic sweep: no error",
   raised(SS.SweepError, SS._precheck_base, base_src(st_per, "periodic"), bp, disc, 4, "Circle", "<x>") is None)
br = quiet(userOpts, Ctx(), circuit="Circle", start_state="rolling")
st_plain = nlp_structure(23, n_in, 0, N, bc.OPT_d, 4)
msg = raised(SS.SweepError, SS._precheck_base, base_src(st_plain), br, disc, 4, "Circle", "<x>")
ok("plain base (no start_state) into rolling sweep: SweepError names start_state",
   msg is not None and "start_state" in msg)

# ---- 10. QSS screen stem with start_state (casadi-free, needs the aero file) ------
# functions/ggv.py aero_coefficients raises without ctx.aero, so the screen needs DATA_AA.mat
print("10. MLTP_screen stem with start_state")
if os.path.exists(DATA_AA):
    with tempfile.TemporaryDirectory() as TMP:
        quiet(MLTP_screen, circuit="Spa", start_state="periodic", save=True, results_dir=TMP, verbose=False)
        ok("periodic screen writes Spa_Static_ATDOn_EM4Off_startPeriodic_qss.mat",
           os.path.exists(os.path.join(TMP, "Spa_Static_ATDOn_EM4Off_startPeriodic_qss.mat")))
        ok("periodic screen leaves the default _qss stem alone",
           not os.path.exists(os.path.join(TMP, "Spa_Static_ATDOn_EM4Off_qss.mat")))
else:
    print("  SKIP section 10: Data/DATA_AA.mat missing (the screen envelope needs ctx.aero)")

# ---- casadi sections: need the aero data file -------------------------------------
if not os.path.exists(DATA_AA):
    print("SKIP test_start_wiring sections 1-6, 8, 9, 11: Data/DATA_AA.mat missing")
    sys.exit(0)

import casadi as ca
import MLTP as mltp_mod
import MLTP_initial as mi_mod
import MLTP_paramOptim as po_mod
import functions.transcription as T
import vehModel as VM
from vehModel import rolling_start, vehModel

_SIG = inspect.signature(T.build_and_solve_nlp)


class _Captured(Exception):
    pass


def model(physics="matlab", **kw):
    """Sturn model of these userOpts kwargs, vehModel built, no solve."""
    c = Ctx()
    quiet(userOpts, c, **dict(dict(circuit="Sturn", physics=physics), **kw))
    vehModel(c)
    return c


def eq_state(c, vx):
    """X of rolling_start(c, c.m23, vx). A RuntimeError (no equilibrium) gives an all-NaN X,
    so the checks that read it fail and the script goes on."""
    try:
        return rolling_start(c, c.m23, vx)[0]
    except RuntimeError:
        return np.full(c.m23.nx, nan)


def sym_time(c, names=()):                             # Phase 1 recipe
    m = c.m23
    xs, us = ca.DM(np.asarray(m.x_s, float)), ca.DM(np.asarray(m.u_s, float))
    F = ca.Function("F", [m.x, m.u, m.pv], [m.dx, m.sf] + [getattr(m, n) for n in names])
    Xp, Up, K = ca.SX.sym("Xp", m.nx), ca.SX.sym("Up", m.nu), ca.SX.sym("K")
    o = F(Xp / xs, Up / us, K)
    return Xp, Up, K, o[0] * xs / o[1], list(o[2:])  # dxdt = physical time derivative


def capture(module, fn, **kw):
    """module.fn(**kw) up to its build_and_solve_nlp call: the bound arguments (only those
    passed). Errors raised before the build propagate."""
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
    """T._make_solver -> fake: no IPOPT. The solver returns x = x0, zero duals;
    rec gets 'nlp' and the 1-D x0, lbx, ubx, lbg, ubg (+ lam_*0) it was called with."""
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
    """The real build_and_solve_nlp on captured args (+ overrides), IPOPT faked: (res, rec)."""
    rec = {}
    with fake_ipopt(rec):
        res = quiet(T.build_and_solve_nlp, **dict(args, **over))
    return res, rec


X7 = np.vstack([np.linspace(30, 50, 10), np.zeros((4, 10)), np.full((2, 10), 100.0)])
U7 = np.vstack([np.full(10, 300.0), np.zeros((2, 10))])
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=U7)}   # in-memory 7-state init (warm_start=INIT)


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


# ---- 1. rolling_start: residual rows and the pinned values per physics ------------
print("1. rolling_start: straight-line coast equilibrium at 60 m/s")
c_m = model("matlab")
X_m, U_m = rolling_start(c_m, c_m.m23, 60.0)
c_c = model("corrected")
X_c, U_c = rolling_start(c_c, c_c.m23, 60.0)
for label, c, X, U in (("matlab", c_m, X_m, U_m), ("corrected", c_c, X_c, U_c)):
    Xp, Up, K, dxdt, _ = sym_time(c)
    G = ca.Function("G", [Xp, Up, K], [dxdt])
    r = np.asarray(G(np.nan_to_num(X), U, 0.0)).ravel()
    ok(f"{label}: residual rows {list(VM.ROLL_RESIDUAL)} below 1e-8",
       np.abs(r[list(VM.ROLL_RESIDUAL)]).max() < 1e-8)
    ok(f"{label}: vx = 60 m/s", X[0] == 60)
    ok(f"{label}: wheel speeds at vx / Rw",
       np.array_equal(X[5:7], [60 / c.vp.Rw_f] * 2) and np.array_equal(X[7:9], [60 / c.vp.Rw_r] * 2))
    ok(f"{label}: r, zsdot, thetadot, phidot and wheel rows 15-18 at 0",
       np.all(X[[2, 10, 12, 14, 15, 16, 17, 18]] == 0))
    ok(f"{label}: n and eps not set (NaN)", np.isnan(X[[3, 4]]).all())
    ok(f"{label}: coast input T_motor 0, T_brake 0, ATD 0.25",
       U[0] == 0 and U[1] == 0 and np.all(U[2:6] == 0.25))
ok("matlab: |zs| < 2e-4 m", abs(X_m[9]) < 2e-4)
ok("matlab: pitch between -1.75 and -1.60 mrad", -1.75e-3 < X_m[11] < -1.60e-3)
ok("matlab: |phi| < 1e-9 rad", abs(X_m[13]) < 1e-9)
ok("matlab: tyre deflections equal (ptp < 1e-9)", np.ptp(X_m[19:23]) < 1e-9)
ok("matlab: front-left deflection above Wfl0 / kt (mus quirk, not quasi_static_states)",
   X_m[19] - c_m.vp.Wfl0 / c_m.vp.kt > 6e-3)
ok("matlab: vy between 0.29 and 0.33 m/s", 0.29 < X_m[1] < 0.33)
ok("corrected: zs between -9.5 and -9.3 mm (aero sink)", -9.5e-3 < X_c[9] < -9.3e-3)
ok("corrected: pitch between 2.2 and 2.3 mrad", 2.2e-3 < X_c[11] < 2.3e-3)
ok("corrected: |phi| < 1e-9 rad", abs(X_c[13]) < 1e-9)
ok("corrected: front tyre deflections equal", abs(X_c[19] - X_c[20]) < 1e-9)
ok("corrected: rear tyre deflections equal", abs(X_c[21] - X_c[22]) < 1e-9)
ok("corrected: front deflection above rear", X_c[19] > X_c[21])
ok("corrected: vy between 0.27 and 0.31 m/s", 0.27 < X_c[1] < 0.31)

# ---- 2. rolling_start: variants, the 70 m/s box, the failure paths ----------------
print("2. rolling_start: variants, bounds, failures")
X_cb = eq_state(model("matlab", tyre_set="CopyB"), 60.0)
ok("CopyB: |vy| < 1e-7 m/s (zero cornering stiffness, delta seed 1e-6)", abs(X_cb[1]) < 1e-7)
ok("CopyB: rows 9-22 match MF205 within 1e-9", np.allclose(X_cb[9:23], X_m[9:23], rtol=0, atol=1e-9))
X_ack = eq_state(model("matlab", Steering="Ackermann"), 60.0)
ok("Ackermann: converges (delta seed 1e-6, not the NaN at 0)", np.isfinite(X_ack[9:23]).all())
ok("Ackermann: rows 9-22 match NA within 1e-9", np.allclose(X_ack[9:23], X_m[9:23], rtol=0, atol=1e-9))
X_ack_cb = eq_state(model("matlab", Steering="Ackermann", tyre_set="CopyB"), 60.0)
ok("Ackermann + CopyB: converges", np.isfinite(X_ack_cb[9:23]).all())
ok("Ackermann + CopyB: rows 9-22 match within 1e-9",
   np.allclose(X_ack_cb[9:23], X_m[9:23], rtol=0, atol=1e-9))
X_ack_c = eq_state(model("corrected", Steering="Ackermann"), 60.0)
ok("corrected Ackermann: converges", np.isfinite(X_ack_c[9:23]).all())
X_aalb = eq_state(model("matlab", AeroConfig="AALB"), 60.0)
ok("AALB: rows 9-22 match Static within 1e-9", np.allclose(X_aalb[9:23], X_m[9:23], rtol=0, atol=1e-9))
X_act = eq_state(model("matlab", AeroConfig="Active", ATD="Off"), 60.0)
ok("Active, ATD Off: rows 9-22 match Static within 1e-9",
   np.allclose(X_act[9:23], X_m[9:23], rtol=0, atol=1e-9))
X_em4 = eq_state(model("matlab", Electric_4Motors="On", ATD="Off"), 60.0)
ok("EM4, ATD Off: converges", np.isfinite(X_em4[9:23]).all())
c70 = model("corrected", vi=70.0)
X70 = rolling_start(c70, c70.m23, 70.0)[0]
ok("corrected 70 m/s: zs below -10 mm (outside the legacy start box)", X70[9] < -0.01)
lo70, _ = boundary_box(c70.m23.x_min, c70.m23.x_max, c70.Xi, c70.m23.x_s, 0.01)
ok("legacy box (OPT_e around Xi, zs = 0) excludes the 70 m/s equilibrium",
   lo70[9] > X70[9] / c70.m23.x_s[9])
msg = raised(RuntimeError, rolling_start, c_m, c_m.m23, 60.0, max_iter=1)
ok("max_iter 1: RuntimeError names vx = 60 and the residual",
   msg is not None and "vx = 60" in msg and "residual" in msg)
for physics in ("matlab", "corrected"):
    c = model(physics)
    msg = raised(RuntimeError, rolling_start, c, c.m23, 85.0)
    ok(f"{physics}: 85 m/s (above Vmax) raises at rows [0, 5, 6, 7, 8]",
       msg is not None and "[0, 5, 6, 7, 8]" in msg)

# ---- 3. MLTP periodic wiring: Xi free, Xi_tol None, n_periodic rows ---------------
print("3. MLTP periodic wiring (Circle, OPT_ds 100)")
a3 = capture(mltp_mod, "MLTP", circuit="Circle", OPT_ds=100, start_state="periodic",
             warm_start=INIT, mesh="uniform", save=False, plot=False)
ok("periodic: periodic True passed to the build", a3["periodic"] is True)
ok("periodic: Xi free (NaN, ni default)", np.isnan(a3["Xi"]).all())
ok("periodic: Xi_tol None", a3["Xi_tol"] is None)
r_per, _ = build_only(a3)
r_plain, _ = build_only(a3, periodic=False)
ok("periodic: 30 more constraint rows (nx + nu)", r_per["n_g"] == r_plain["n_g"] + 30)
ok("periodic: structure records n_periodic 30", r_per["structure"]["n_periodic"] == 30)
recs3 = []
for vi3 in (55.0, 65.0):
    cap = capture(mltp_mod, "MLTP", circuit="Circle", OPT_ds=100, start_state="periodic", vi=vi3,
                  warm_start=INIT, mesh="uniform", save=False, plot=False)
    recs3.append(build_only(cap)[1])
ok("vi only seeds: lbx, ubx, lbg, ubg identical at vi 55 and 65",
   all(np.array_equal(recs3[0][k], recs3[1][k]) for k in ("lbx", "ubx", "lbg", "ubg")))

# ---- 4. MLTP rolling wiring: Xi from the equilibrium, box widths, Xi_tol ----------
print("4. MLTP rolling wiring (Sturn, corrected, 70 m/s)")
a4 = capture(mltp_mod, "MLTP", circuit="Sturn", start_state="rolling", physics="corrected", vi=70.0,
             warm_start=INIT, mesh="uniform", save=False, plot=False)
w4 = list(WARNS)                    # kept for section 9 (the next quiet call overwrites WARNS)
ok("rolling: Xi_tol = OPT_e on rows 0-8, ROLL_TOL on rows 9-22",
   np.array_equal(a4["Xi_tol"], np.r_[np.full(9, 0.01), np.full(14, 1e-3)]))
ok("rolling: periodic False", a4["periodic"] is False)
ok("rolling: Xi[9] = the 70 m/s equilibrium zs", abs(a4["Xi"][9] - X70[9]) < 1e-12)
ok("rolling: Xi[2] = 0 (r)", a4["Xi"][2] == 0)
r4, q4 = build_only(a4)
wd4 = q4["ubg"][:23] - q4["lbg"][:23]
ok("rolling: 1904 constraints (no periodic rows)", r4["n_g"] == 1904)
ok("rolling: rows 9-22 box width 2e-3", np.allclose(wd4[9:23], 2e-3, rtol=0, atol=1e-15))
ok("rolling: rows 9-22 box centred on Xi",
   np.allclose((q4["ubg"][9:23] + q4["lbg"][9:23]) / 2, a4["Xi"][9:23], rtol=0, atol=1e-15))
ok("rolling: rows 0, 1, 2, 5-8 box width 0.02",
   np.allclose(wd4[[0, 1, 2, 5, 6, 7, 8]], 0.02, rtol=0, atol=1e-12))

# ---- 5. optimise_design and MLTP_initial wiring -----------------------------------
print("5. optimise_design and MLTP_initial wiring")
b5 = capture(po_mod, "MLTP_paramOptim", circuit="Circle", start_state="periodic", warm_start=INIT, save=False)
ok("MLTP_paramOptim periodic: periodic True", b5["periodic"] is True)
ok("MLTP_paramOptim periodic: promoted parameters passed (param not None)", b5["param"] is not None)
msg = raised(ValueError, capture, po_mod, "MLTP_paramOptim", circuit="Sturn", start_state="rolling",
             warm_start=INIT, save=False)
ok("MLTP_paramOptim rolling: ValueError before any solve, names rolling",
   msg is not None and "rolling" in msg)
c7 = capture(mi_mod, "MLTP_initial", circuit="Circle", OPT_ds=100, start_state="periodic", save=False)
ok("MLTP_initial periodic: periodic True", c7["periodic"] is True)
ok("MLTP_initial periodic: Xi free (NaN)", np.isnan(c7["Xi"]).all())
r7 = capture(mi_mod, "MLTP_initial", circuit="Sturn", start_state="rolling", save=False)
ok("MLTP_initial rolling: periodic False", r7["periodic"] is False)
ok("MLTP_initial rolling: no Xi_tol passed (7-state keeps the OPT_e box)", "Xi_tol" not in r7)
ok("MLTP_initial rolling: Xi_init = vi, r 0, wheel speeds at vi / Rw",
   np.array_equal(r7["Xi"], [60, 0, 0, nan, nan, 60 / c_m.vp.Rw_f, 60 / c_m.vp.Rw_r], equal_nan=True))

# ---- 6. refine passes inherit the start state -------------------------------------
print("6. refine passes: Xi_tol and periodic forwarded")
calls6 = []
run_stub(calls6, start_state="rolling", refine=dict(passes=1, tol=1e-12))
ok("refine: at least one pass re-solved (2+ builds)", len(calls6) >= 2)
ok("refine: every build gets the same Xi_tol",
   all(np.array_equal(cl["Xi_tol"], calls6[0]["Xi_tol"]) for cl in calls6))
ok("refine: every build periodic False", all(cl["periodic"] is False for cl in calls6))

# ---- 8. result files: start suffix, saved Xi, n_periodic --------------------------
print("8. result files: start suffix, saved Xi and n_periodic")
with tempfile.TemporaryDirectory() as TMP:
    c8 = run_stub(start_state="rolling", save=True, results_dir=TMP, mesh="auto")
    p8 = os.path.join(TMP, "Sturn_Static_ATDOn_EM4Off_startRolling.mat")
    ok("rolling: Sturn_Static_ATDOn_EM4Off_startRolling.mat written", os.path.exists(p8))
    d8 = load_solution(p8)
    ok("rolling: data.start_state = 'rolling'", str(d8.start_state) == "rolling")
    ok("rolling: data.Xi equals ctx.Xi (NaN in place)",
       np.allclose(d8.Xi, c8.Xi, rtol=0, atol=0, equal_nan=True))
    run_stub(circuit="Circle", start_state="periodic", save=True, results_dir=TMP, mesh="auto")
    p8b = os.path.join(TMP, "Circle_Static_ATDOn_EM4Off_startPeriodic.mat")
    ok("periodic: Circle_Static_ATDOn_EM4Off_startPeriodic.mat written", os.path.exists(p8b))
    d8b = load_solution(p8b)
    ok("periodic: data.nlp.structure.n_periodic = 30", int(d8b.nlp.structure.n_periodic) == 30)
    ok("periodic: no 'out of sync' warning", not any("out of sync" in w for w in WARNS))

# ---- 9. straight-start warning ----------------------------------------------------
print("9. rolling on a curved start warns, a straight start does not")
capture(mltp_mod, "MLTP", circuit="Circle", start_state="rolling", warm_start=INIT,
        mesh="uniform", save=False, plot=False)
ok("Circle rolling: warns 'straight start'", any("straight start" in w for w in WARNS))
ok("Sturn rolling (section 4): no 'straight start' warning",
   not any("straight start" in w for w in w4))

# ---- 11. real build with the fake solver ----------------------------------------
print("11. real build with the fake solver: saved records, warm-start modes, MLTP_initial and optimise_design")
import shutil
from plotSDI import plotSDI

RES_TMP = tempfile.mkdtemp(prefix="start_wiring_res_")
PLOTS_TMP = tempfile.mkdtemp(prefix="start_wiring_plots_")
try:
    with fake_ipopt({}):
        # MLTP: one saved record per start_state (refine adds _meshAdaptive only after an accepted pass)
        for name, kw in (("matlab Sturn", dict(circuit="Sturn")),
                         ("rolling Sturn", dict(circuit="Sturn", start_state="rolling")),
                         ("periodic Circle", dict(circuit="Circle", OPT_ds=100, start_state="periodic")),
                         ("periodic Circle + refine", dict(circuit="Circle", OPT_ds=100, start_state="periodic",
                                                           refine=dict(passes=1, tol=1e-12)))):
            c = quiet(mltp_mod.MLTP, warm_start=INIT, save=True, plot=False, mesh="auto",
                      results_dir=RES_TMP, **kw)
            ok(f"{name}: no out-of-sync warning", not any("out of sync" in w for w in WARNS))
            st = kw.get("start_state", "matlab")
            stem = f"{kw['circuit']}_Static_ATDOn_EM4Off" + ("" if st == "matlab" else "_start" + st.capitalize())
            if "refine" in kw:
                stem += "_meshAdaptive" if c.mesh_requested == "adaptive" else ""
            p = os.path.join(RES_TMP, stem + ".mat")
            ok(f"{name}: saved {stem}.mat", os.path.isfile(p))
            d = load_solution(p)
            ok(f"{name}: data.start_state", d.start_state == st)
            ok(f"{name}: data.Xi (23,)", np.shape(d.Xi) == (23,))
            ok(f"{name}: structure.n_periodic", int(d.nlp.structure.n_periodic) == (30 if st == "periodic" else 0))
            if st == "rolling":
                ok("rolling Xi: r = 0, zs near equilibrium, n free",
                   d.Xi[2] == 0 and abs(d.Xi[9]) < 2e-4 and np.isnan(d.Xi[3]))
            if st == "periodic":
                ok(f"{name}: periodic Xi all NaN", np.isnan(d.Xi).all())

        # warm starts: another start_state interpolates the primal only, the same one keeps the duals
        base = os.path.join(RES_TMP, "Sturn_Static_ATDOn_EM4Off.mat")
        c1 = quiet(mltp_mod.MLTP, warm_start=base, circuit="Sturn", save=False, plot=False, mesh="uniform")
        ok("warm matlab <- matlab: full+duals", c1.elapsed["warm_start"] == "full+duals")
        c2 = quiet(mltp_mod.MLTP, warm_start=base, circuit="Sturn", start_state="rolling", save=False,
                   plot=False, mesh="uniform")
        ok("warm rolling <- matlab: full-interp", c2.elapsed["warm_start"] == "full-interp")
        rolling_src = os.path.join(RES_TMP, "Sturn_Static_ATDOn_EM4Off_startRolling.mat")
        c3 = quiet(mltp_mod.MLTP, warm_start=rolling_src, circuit="Sturn", start_state="rolling", save=False,
                   plot=False, mesh="uniform")
        ok("warm rolling <- rolling: full+duals", c3.elapsed["warm_start"] == "full+duals")

        # MLTP_initial: start_state in the file name and in the record; Xi is the 7-state start box
        for name, kw in (("matlab", {}), ("rolling", dict(start_state="rolling")),
                         ("periodic", dict(circuit="Circle", OPT_ds=100, start_state="periodic"))):
            kw = dict(dict(circuit="Sturn"), **kw)
            quiet(mi_mod.MLTP_initial, save=True, results_dir=RES_TMP, **kw)
            st = kw.get("start_state", "matlab")
            fn = f"init_{kw['circuit']}" + ("" if st == "matlab" else "_start" + st.capitalize()) + ".mat"
            ok(f"MLTP_initial {name}: saved {fn}", os.path.isfile(os.path.join(RES_TMP, fn)))
            d = load_solution(os.path.join(RES_TMP, fn))
            ok(f"MLTP_initial {name}: init.start_state {st}, Xi (7,)",
               d.start_state == st and np.shape(d.Xi) == (7,))
            ok(f"MLTP_initial {name}: no out-of-sync warning", not any("out of sync" in w for w in WARNS))

        # optimise_design, called through MLTP_paramOptim: start_state, Xi and n_periodic in the record
        for name, kw in (("matlab Sturn", dict(circuit="Sturn")),
                         ("periodic Circle", dict(circuit="Circle", OPT_ds=100, start_state="periodic"))):
            quiet(po_mod.MLTP_paramOptim, warm_start=INIT, save=True, results_dir=RES_TMP, **kw)
            ok(f"{name}: no out-of-sync warning", not any("out of sync" in w for w in WARNS))
            st = kw.get("start_state", "matlab")
            fn = f"{kw['circuit']}_paramOptim" + ("" if st == "matlab" else "_start" + st.capitalize()) + ".mat"
            ok(f"{name}: saved {fn}", os.path.isfile(os.path.join(RES_TMP, fn)))
            d = load_solution(os.path.join(RES_TMP, fn))
            ok(f"{name}: start_state + Xi (23,)", d.start_state == st and np.shape(d.Xi) == (23,))
            ok(f"{name}: structure n_periodic", int(d.nlp.structure.n_periodic) == (30 if st == "periodic" else 0))
        pp = os.path.join(RES_TMP, "Circle_Static_ATDOn_EM4Off_startPeriodic.mat")
        c4 = quiet(po_mod.MLTP_paramOptim, circuit="Circle", OPT_ds=100, start_state="periodic",
                   warm_start=pp, save=False)
        ok("periodic MLTP -> periodic paramOptim: full+duals", c4.elapsed["warm_start"] == "full+duals")
        c5 = quiet(po_mod.MLTP_paramOptim, circuit="Circle", OPT_ds=100, warm_start=pp, save=False)
        ok("periodic MLTP -> plain paramOptim: full-interp", c5.elapsed["warm_start"] == "full-interp")

    # plotSDI names the folder from the start state: <plots>/Sturn/Static_ATDOn_EM4Off_startRolling
    quiet(plotSDI, os.path.join(RES_TMP, "Sturn_Static_ATDOn_EM4Off_startRolling.mat"), save_dir=PLOTS_TMP)
    ok("plotSDI rolling start: Sturn/Static_ATDOn_EM4Off_startRolling/racing_line.html written",
       os.path.isfile(os.path.join(PLOTS_TMP, "Sturn", "Static_ATDOn_EM4Off_startRolling",
                                   "racing_line.html")))
finally:
    shutil.rmtree(RES_TMP, ignore_errors=True)
    shutil.rmtree(PLOTS_TMP, ignore_errors=True)

print("test_start_wiring: done")
