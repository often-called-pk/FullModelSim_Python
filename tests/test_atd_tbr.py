"""C4: userOpts(atd_tbr=...) bounds the torque bias of both axle differentials. ATD On, Electric_4Motors Off only.

The wheel torque of each corner is T_i = ATD_i (gear T_motor + 2 T_brake). With k = (TBR - 1) / (TBR + 1), each axle
obeys |T_l - T_r| <= k |T_l + T_r| for drive and brake alike (T_brake <= 0). The rows are linear in the inputs. They
exist only when atd_tbr is set. TBR = 1 (k = 0) gives one equality row per axle. TBR > 1 gives two inequality rows per
axle. The warm start compares the path-row names, so the 8 PureSlip rows do not seed the 8 TBR rows with multipliers.

Sections 1 and 2 build no NLP. Sections 3 to 6 need casadi and Data/DATA_AA.mat. Without them, these sections print
SKIP and the file exits 0. Sections 3 to 5 use a stand-in build or the QSS. Section 6 replaces the IPOPT solver with a
fake and starts from Results/Sturn_Static_ATDOn_EM4Off.mat (SKIP if it is missing). No IPOPT runs. Run from any
directory:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_atd_tbr.py
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
from userOpts import userOpts

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
INPUT_KEYS = ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"]    # ATD On, EM4 Off, Static aero


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def raised(**kw):
    """Message of the ValueError that userOpts raises for Sturn with kw, None if it does not raise."""
    try:
        quiet(userOpts, Ctx(), circuit="Sturn", **kw)
    except ValueError as e:
        return str(e)
    return None


print("1. userOpts")
c = Ctx()
quiet(userOpts, c, circuit="Sturn")
ok("default: no atd_tbr and no pt.atd_k", getattr(c, "atd_tbr", None) is None and not hasattr(c.pt, "atd_k"))
for tbr, k in ((1.0, 0.0), (2.0, 1 / 3), (3, 0.5), (2.5, 3 / 7)):
    c = Ctx()
    quiet(userOpts, c, circuit="Sturn", atd_tbr=tbr)
    ok(f"TBR {tbr}: k = {k:.4f}", abs(c.pt.atd_k - k) < 1e-15)
ok("ATD Off raises", "atd_tbr" in (raised(ATD="Off", atd_tbr=2.0) or ""))
ok("4 motors raise", "atd_tbr" in (raised(Electric_4Motors="On", atd_tbr=2.0) or ""))
for bad in (0.5, True, float("inf"), "x", -1):
    ok(f"atd_tbr={bad!r} raises", "atd_tbr" in (raised(atd_tbr=bad) or ""))

print("2. warm-start planner (structure only)")


def saved(nh, names=None):
    """A saved full result of an NLP with nh path rows (Sturn N = 18, 7 inputs). names sets data.constraints."""
    s = nlp_structure(23, 7, 0, 18, 3, nh)
    r = dict(input_keys=INPUT_KEYS, s_full=np.arange(73.0),
             nlp=dict(structure=s, w_opt=np.zeros(s["n_w"]), lam_g=np.zeros(s["n_g"]),
                      lam_x=np.zeros(s["n_w"]), return_status="Solve_Succeeded"))
    if names is not None:
        r["constraints"] = {n: np.zeros(19) for n in names}
    return r


def expected(nh, names=None):
    """The structure that a new NLP with nh path rows (named names) hands to the planner."""
    e = dict(nlp_structure(23, 7, 0, 18, 3, nh), input_keys=INPUT_KEYS, s_full=np.arange(73.0))
    if names is not None:
        e["hnames"] = names
    return e


for a, b, mode in ((4, 4, "full+duals"), (4, 8, "full-interp"), (8, 8, "full+duals"), (8, 6, "full-interp")):
    ok(f"nh {a} -> {b}: {mode}", plan_full_warm_start(saved(a), expected(b))[1] == mode)
PURE = ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr", "motor_power", "motor_rpm", "BrTh_1", "ATD_eq"]
TBR = ["motor_power", "motor_rpm", "BrTh_1", "ATD_eq", "diff_F_l", "diff_F_r", "diff_R_l", "diff_R_r"]
_, md, note = plan_full_warm_start(saved(8, PURE), expected(8, TBR))
ok("PureSlip 8 rows -> TBR 2 8 rows: other row names, full-interp, the note names the rows",
   md == "full-interp" and "path rows" in note and "diff_F_l" in note)
ok("TBR 2 -> TBR 3: same row names, full+duals",
   plan_full_warm_start(saved(8, TBR), expected(8, TBR))[1] == "full+duals")
ok("source without data.constraints: names not compared, full+duals",
   plan_full_warm_start(saved(8), expected(8, TBR))[1] == "full+duals")

try:
    import casadi as ca  # noqa: E402
except ImportError:  # pragma: no cover
    ca = None
if ca is None or not os.path.exists(DATA_AA):
    print("SKIP test_atd_tbr sections 3-6: casadi or Data/DATA_AA.mat missing")
    sys.exit(0)

import functions.transcription as T  # noqa: E402
import MLTP as mltp_mod  # noqa: E402
from MLTP import build_path_constraints  # noqa: E402
from MLTP_screen import MLTP_screen  # noqa: E402
from vehModel import vehModel  # noqa: E402

print("3. the rows (Sturn model)")


def rows(**kw):
    """(context, row names, row expressions, lower bounds, upper bounds) of the Sturn path constraints for kw."""
    c = Ctx()
    quiet(userOpts, c, circuit="Sturn", **kw)
    quiet(vehModel, c)
    names, h, lb, ub = build_path_constraints(ca, c.m23, c.pt)
    return c, names, h, lb, ub


hn0 = rows()[1]
ok("default rows: motor_power, motor_rpm, BrTh_1, ATD_eq", hn0 == ["motor_power", "motor_rpm", "BrTh_1", "ATD_eq"])
c2, hn2, h2, lb2, ub2 = rows(atd_tbr=2.0)
ok("TBR 2: four inequality rows after ATD_eq, bounds [0, inf)",
   hn2[4:] == ["diff_F_l", "diff_F_r", "diff_R_l", "diff_R_r"] and np.all(lb2[4:] == 0) and np.all(np.isinf(ub2[4:])))
c1, hn1, h1, lb1, ub1 = rows(atd_tbr=1.0)
ok("TBR 1: one equality row per axle, bounds 0",
   hn1[4:] == ["diff_F", "diff_R"] and np.all(lb1[4:] == 0) and np.all(ub1[4:] == 0))

m = c2.m23
ia = [i for i, key in enumerate(c2.input_keys) if key == "ATD"]
iM = c2.input_keys.index("T_motor")


def at(split, c=c2, h=h2):
    """Values of the path rows h of context c with the ATD inputs set to split = (FL, FR, RL, RR), other inputs 0."""
    mc = c.m23
    u = np.zeros(mc.nu)
    u[ia] = split
    return np.asarray(ca.Function("h", [mc.x, mc.u, mc.pv], [h])(np.zeros(mc.nx), u / mc.u_s, 0)).ravel()


ok("TBR 1 row values: a - b per axle, 0.3 front and 0.1 rear for the split (.4, .1, .3, .2)",
   np.allclose(at((.4, .1, .3, .2), c1, h1)[4:], [.3, .1], atol=1e-12))


ok("equal split satisfies TBR 2", np.all(at((.25, .25, .25, .25))[4:] >= 0))
v = at((.4, .1, .3, .2))
ok("FL 0.4 / FR 0.1 breaks the front axle, the rear 0.3 / 0.2 holds", v[4] < 0 and v[5] > 0 and np.all(v[6:] > 0))
k = (2.0 - 1.0) / (2.0 + 1.0)
ok("row values: k(a+b) - (a-b) and k(a+b) + (a-b) per axle",
   np.allclose(v[4:], [k * .5 - .3, k * .5 + .3, k * .5 - .1, k * .5 + .1], atol=1e-12))
ok("all torque on one wheel breaks TBR 2", at((1, 0, 0, 0))[4] < 0)

Tw = ca.Function("Tw", [m.x, m.u, m.pv], [m.T_fl, m.T_fr, m.T_rl, m.T_rr])
u = np.zeros(m.nu)
u[ia] = (.4, .1, .3, .2)
u[iM] = 300.0
tq = np.array([float(t) for t in Tw(np.zeros(m.nx), u / m.u_s, 0)])
ok("wheel torque = ATD_i gear T_motor", np.allclose(tq, np.array([.4, .1, .3, .2]) * 300.0 * c2.vp.gear, rtol=1e-12))

print("4. NLP sizes (build_only, no solver)")
X7 = np.vstack([np.linspace(30, 50, 10), np.zeros((4, 10)), np.full((2, 10), 100.0)])
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=np.vstack([np.full(10, 300.0), np.zeros((2, 10))]))}
_SIG = inspect.signature(T.build_and_solve_nlp)


class _Captured(Exception):
    """Stops MLTP once the call under test has been seen."""


def sizes(**kw):
    """(n_w, n_g, structure n_g) of the NLP that MLTP hands to build_and_solve_nlp for kw. Built, not solved."""
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
    r = quiet(T.build_and_solve_nlp, **seen, build_only=True)
    return r["nlp"]["x"].numel(), r["nlp"]["g"].numel(), r["structure"]["n_g"]


ok("default: Sturn NLP 1812 x 1904, structure n_g 1904", sizes() == (1812, 1904, 1904))
ok("TBR 1: two rows per knot more (19 knots), 1942", sizes(atd_tbr=1.0) == (1812, 1942, 1942))
ok("TBR 2: four rows per knot more (19 knots), 1980", sizes(atd_tbr=2.0) == (1812, 1980, 1980))

print("5. QSS: the knob changes nothing there (the QSS has no left-right split)")
lap = [quiet(MLTP_screen, circuit="Sturn", save=False, verbose=False, **kw).data.lap_time
       for kw in ({}, {"atd_tbr": 1.0}, {"atd_tbr": 2.0})]
ok("QSS lap identical with atd_tbr 1 and 2", lap[1] == lap[0] and lap[2] == lap[0])

print("6. results record atd_tbr, the warm start compares the row names (Sturn, IPOPT faked, no solve)")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
if not os.path.exists(STURN):
    print(f"SKIP test_atd_tbr section 6: {STURN} missing")
    sys.exit(0)
import MLTP_paramOptim as mp  # noqa: E402


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


DESIGN = ([("alpha_RW", 4.0, 16.0)], "t7")
with tempfile.TemporaryDirectory() as tmp:
    t2 = faked(mltp_mod.MLTP, warm_start=STURN, plot=False, atd_tbr=2.0, save=True, results_dir=tmp)
    back = load_solution(t2.out_path)
    ok("MLTP TBR 2: data.atd_tbr 2.0, also after the .mat round trip",
       getattr(t2.data, "atd_tbr", None) == 2.0 and float(getattr(back, "atd_tbr", np.nan)) == 2.0)
    ps = faked(mltp_mod.MLTP, warm_start=t2.out_path, plot=False, TyreModel="PureSlip", save=True, results_dir=tmp)
    ok("MLTP PureSlip from the TBR 2 file (8 rows each, other names): full-interp, no atd_tbr field",
       ps.elapsed["warm_start"] == "full-interp" and not hasattr(ps.data, "atd_tbr"))
    t3 = faked(mltp_mod.MLTP, warm_start=t2.out_path, plot=False, atd_tbr=3.0)
    ok("MLTP TBR 3 from the TBR 2 file: the row names survive the .mat round trip, full+duals",
       t3.elapsed["warm_start"] == "full+duals")
    d2 = faked(mp.optimise_design, *DESIGN, warm_start=t2.out_path, atd_tbr=2.0)
    dps = faked(mp.optimise_design, *DESIGN, warm_start=ps.out_path, atd_tbr=2.0)
    d0 = faked(mp.optimise_design, *DESIGN, warm_start=t2.out_path)
ok("optimise_design TBR 2: full+duals and data.atd_tbr 2.0 from the TBR 2 file, full-interp from the PureSlip file",
   d2.elapsed["warm_start"] == "full+duals" and getattr(d2.data, "atd_tbr", None) == 2.0
   and dps.elapsed["warm_start"] == "full-interp")
ok("optimise_design default from the TBR 2 file: full-interp, no atd_tbr field",
   d0.elapsed["warm_start"] == "full-interp" and not hasattr(d0.data, "atd_tbr"))
print("test_atd_tbr: done")
