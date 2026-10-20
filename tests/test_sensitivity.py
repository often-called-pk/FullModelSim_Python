"""Limit prices from stored multipliers (plan docs/superpowers/plans/2026-10-18-phase2.md, Task 7: C2).

functions/sensitivity.py prices the bounds of a saved full result from its NLP multipliers: the track
width W (n_hi, n_lo), the torque and brake torque limits, the power rating Pmax, the start speed vi,
the per-point lateral price map and the continuity costates. Sections 1-5 are numpy only. Section 5
reads the stored Sturn and BCN results in Results/ and prints SKIP for a file that is absent.
Sections 6-7 rebuild the stored Sturn NLP through a stand-in solver (no IPOPT) and compare the prices
with finite differences of the rebuilt Lagrangian. They need casadi and Data/DATA_AA.mat (SKIP,
exit 0 otherwise). No section runs IPOPT. Run from anywhere:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_sensitivity.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np

import functions.transcription as T
from functions.importfile import load_solution
from functions.sensitivity import costates, lateral_prices, limit_prices, nlp_index
from functions.warmstart import nlp_structure

STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
BCN = os.path.join(_bootstrap.ROOT, "Results", "BCN_Static_ATDOn_EM4Off.mat")
DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")


def _no_ipopt(*a, **k):
    raise RuntimeError("test_sensitivity never runs IPOPT: a build reached the real _make_solver")


T._make_solver = _no_ipopt  # fake_ipopt() swaps in its own solver for the rebuilds


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout and warnings swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def raised(exc, fn, *a, **kw):
    """True when fn(*a, **kw) raises exc."""
    try:
        quiet(fn, *a, **kw)
    except exc:
        return True
    return False


def close(a, b, rel=1e-15):
    return abs(a - b) <= rel * abs(b)


ST = nlp_structure(23, 7, 0, 18, 3, 4)  # the default Sturn NLP: 1812 w, 1904 g, 4 path rows
I = nlp_index(ST)
KEYS = ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"]


def record():
    """An all-zero full result of the Sturn sizes, no Xi (the matlab start)."""
    x_s = np.ones(23)
    x_s[0], x_s[3] = 100.0, 5.0
    nlp = SimpleNamespace(lam_g=np.zeros(ST["n_g"]), lam_x=np.zeros(ST["n_w"]), structure=ST,
                          x_s=x_s, u_s=np.array([602.0, 4000.0, 1.0, 1.0, 1.0, 1.0, np.pi / 8]))
    cons = {k: np.ones(19) for k in ("motor_power", "motor_rpm", "BrTh_1", "ATD_eq")}
    return SimpleNamespace(nlp=nlp, input_keys=list(KEYS), s_full=np.linspace(0, 540, 73),
                           constraints=cons)


def n_record():
    """record() with the n multipliers of section 2: X[3,1] = 2, XC[3,4] = -1, end = 0.5, start = 0.25."""
    r = record()
    r.nlp.lam_x[I["X"][3, 1]] = 2.0
    r.nlp.lam_x[I["XC"][3, 4]] = -1.0
    r.nlp.lam_g[I["end"][3]] = 0.5
    r.nlp.lam_g[I["start"][3]] = 0.25
    return r


# ---- 1. nlp_index: the packing of build_and_solve_nlp ------------------------------
print("1. nlp_index: rows of w and g")
ok("1a nh == 4", I["nh"] == 4)
ok("1b X, U, XC partition arange(1812)",
   np.array_equal(np.sort(np.concatenate([I[k].ravel() for k in ("X", "U", "XC")])), np.arange(1812)))
ok("1c start, end, defect, cont, h, rate partition arange(1904)",
   np.array_equal(np.sort(np.concatenate([I[k].ravel() for k in
                                          ("start", "end", "defect", "cont", "h", "rate")])),
                  np.arange(1904)))
ok("1d X[3, 1] == 26", I["X"][3, 1] == 26)
ok("1e U[0, 0] == 437", I["U"][0, 0] == 437)
ok("1f XC[3, 0] == 573", I["XC"][3, 0] == 573)
ok("1g cont[0, 0] == 115", I["cont"][0, 0] == 115)
ok("1h h[0, 0] == 1702", I["h"][0, 0] == 1702)
ok("1i h[0, 18] == 1774", I["h"][0, 18] == 1774)
ok("1j rate[0, 0] == 1778", I["rate"][0, 0] == 1778)
ok("1k rate[6, 17] == 1903", I["rate"][6, 17] == 1903)
per = nlp_index(nlp_structure(23, 7, 0, 18, 3, 4, n_periodic=30))
ok("1l periodic rows == arange(1904, 1934)", np.array_equal(per["periodic"], np.arange(1904, 1934)))
st7 = nlp_structure(7, 3, 1, 10, 3, 6)
I7 = nlp_index(st7)
ok("1m 7-state X, U, Y, XC partition arange(n_w)",
   np.array_equal(np.sort(np.concatenate([I7[k].ravel() for k in ("X", "U", "Y", "XC")])),
                  np.arange(st7["n_w"])))
par = nlp_index(nlp_structure(23, 7, 0, 18, 3, 4, n_param=6))
ok("1n P == arange(1812, 1818)", np.array_equal(par["P"], np.arange(1812, 1818)))
bad = dict(ST, n_g=ST["n_g"] + 1)
ok("1o n_g off by 1 -> ValueError", raised(ValueError, nlp_index, bad))

# ---- 2. limit_prices: W from the n multipliers ------------------------------------
print("2. limit_prices: W counts the start row only when n(0) is free")
r = n_record()
p = limit_prices(r)
ok("2a W == -(2 + 1 + 0.5 + 0.25)/5", close(p["W"], -(2 + 1 + 0.5 + 0.25) / 5))
ok("2b n_hi == -(2.5 + 0.25)/5", close(p["n_hi"], -(2.5 + 0.25) / 5))
ok("2c n_lo == 1/5", close(p["n_lo"], 1 / 5))
r.Xi = np.full(23, np.nan)
r.Xi[0], r.Xi[3] = 60.0, 1.0
ok("2d Xi[3] = 1.0 (pinned start): W == -3.5/5", close(limit_prices(r)["W"], -3.5 / 5))

# ---- 3. limit_prices: Tmax, Tbrake_max, Pmax, vi and the NaN rules ----------------
print("3. limit_prices: torque, brake, power and start speed")
r = record()
r.nlp.lam_x[I["U"][0, 0]], r.nlp.lam_x[I["U"][0, 1]] = 3.0, -7.0
ok("3a Tmax == -3/602", close(limit_prices(r)["Tmax"], -3 / 602))
r = record()
r.nlp.lam_x[I["U"][1, 2]], r.nlp.lam_x[I["U"][1, 3]] = -4.0, 9.0
ok("3b Tbrake_max == -4/4000", close(limit_prices(r)["Tbrake_max"], -4 / 4000))
r = record()
r.constraints["motor_power"][5], r.constraints["motor_power"][6] = 0.0, 1.0
r.nlp.lam_g[I["h"][0, 5]], r.nlp.lam_g[I["h"][0, 6]] = -0.2, 0.3
ok("3c Pmax == -0.2/456e3", close(limit_prices(r)["Pmax"], -0.2 / 456e3))
r = record()
r.nlp.lam_g[I["start"][0]] = 1.5
ok("3d vi == -0.015", close(limit_prices(r)["vi"], -0.015))
r.Xi = np.zeros(23)
r.Xi[0] = np.nan
ok("3e Xi[0] NaN: vi is NaN", np.isnan(limit_prices(r)["vi"]))
r = record()
del r.constraints
ok("3f no constraints: Pmax is NaN", np.isnan(limit_prices(r)["Pmax"]))

# ---- 4. lateral_prices and costates -----------------------------------------------
print("4. lateral_prices and costates")
r = n_record()
lp = lateral_prices(r)
W = limit_prices(r)["W"]
ok("4a sum(hi - lo) == W (rel 1e-12)", abs(np.sum(lp["hi"] - lp["lo"]) - W) <= 1e-12 * abs(W))
c = record()
c.nlp.lam_g[I["cont"][0, 0]] = 0.7
c.nlp.lam_g[I["cont"][3, 5]] = -0.4
P = costates(c)["P"]
ok("4b P[0, 0] == 0.7/100", close(P[0, 0], 0.7 / 100))
ok("4c P[3, 5] == -0.4/5", close(P[3, 5], -0.4 / 5))
ok("4d only those two costates are nonzero", np.count_nonzero(P) == 2)
nonlp = record()
del nonlp.nlp
ok("4e no data.nlp -> ValueError", raised(ValueError, costates, nonlp))
short = record()
short.nlp.lam_g = short.nlp.lam_g[:-1]
ok("4f lam_g[:-1] -> ValueError", raised(ValueError, costates, short))

# ---- 5. stored results: prices of the saved Sturn and BCN solves ------------------
print("5. stored results (Results/)")
if os.path.exists(STURN):
    p = limit_prices(load_solution(STURN))
    ok("5a Sturn -0.25 < W < -0.15", -0.25 < p["W"] < -0.15)
    ok("5b Sturn -0.02 < vi < -0.005", -0.02 < p["vi"] < -0.005)
    ok("5c Sturn Tmax < 0", p["Tmax"] < 0)
    ok("5d Sturn Tbrake_max < 0", p["Tbrake_max"] < 0)
    ok("5e Sturn Pmax < 0", p["Pmax"] < 0)
else:
    print("  SKIP 5a-5e: Results/Sturn_Static_ATDOn_EM4Off.mat missing")
if os.path.exists(BCN):
    d = load_solution(BCN)
    p = limit_prices(d)
    ok("5f BCN -2.1 < W < -1.6", -2.1 < p["W"] < -1.6)
    ok("5g BCN -0.08 < vi < -0.06", -0.08 < p["vi"] < -0.06)
    ok("5h BCN -8e-3 < Tmax < -5e-3", -8e-3 < p["Tmax"] < -5e-3)
    ok("5i BCN Pmax < 0", p["Pmax"] < 0)
    ok("5j BCN costate P[0, 0] < 0", costates(d)["P"][0, 0] < 0)
else:
    print("  SKIP 5f-5j: Results/BCN_Static_ATDOn_EM4Off.mat missing")

# ---- casadi sections: the rebuilds need the aero file and the stored Sturn solution -
if not os.path.exists(DATA_AA) or not os.path.exists(STURN):
    print("SKIP test_sensitivity sections 6-7: Data/DATA_AA.mat or "
          "Results/Sturn_Static_ATDOn_EM4Off.mat missing")
    sys.exit(0)

import casadi as ca                     # noqa: E402
import MLTP as mltp_mod                 # noqa: E402

os.environ["MLTP_KEEP_NLP"] = "1"       # MLTP keeps ctx.nlp_fg and ctx.nlp_bounds (never saved)


@contextlib.contextmanager
def fake_ipopt():
    """T._make_solver -> a stand-in: the solver returns x = x0 with zero duals. No IPOPT."""
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


def rebuild(**kw):
    """MLTP on the stored Sturn solution with the stand-in solver: the NLP is built, not solved."""
    with fake_ipopt():
        return quiet(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False, **kw)


def rebuild_pt(attr, factor):
    """rebuild() with ctx.pt.<attr> times factor right after userOpts (vehModel reads pt)."""
    real = mltp_mod.userOpts

    def scaled(ctx, *a, **k):
        out = real(ctx, *a, **k)
        setattr(ctx.pt, attr, getattr(ctx.pt, attr) * factor)
        return out

    mltp_mod.userOpts = scaled
    try:
        return rebuild()
    finally:
        mltp_mod.userOpts = real


# ---- 6. zero-solve rebuild of the stored Sturn NLP --------------------------------
print("6. rebuild of the stored Sturn NLP: full+duals start, KKT conditions, R share")
d = load_solution(STURN)
w, lg, lx = d.nlp.w_opt, d.nlp.lam_g, d.nlp.lam_x
c0 = rebuild()
ok("6a warm start x0 is the stored w_opt (full+duals)", np.array_equal(c0.nlp_bounds["w0"], w))
Ws = ca.SX.sym("w", w.size)
J, g = c0.nlp_fg(Ws)
G = ca.Function("G", [Ws], [ca.gradient(J, Ws), ca.jacobian(g, Ws)])
gJ, Jg = G(w)
gJ = np.asarray(gJ).ravel()
JgT_lg = np.asarray(ca.mtimes(Jg.T, ca.DM(lg))).ravel()
ok("6b grad J + Jg' lam_g + lam_x == 0 at the stored point (< 1e-5)",
   np.abs(gJ + JgT_lg + lx).max() < 1e-5)
ok("6c the same check with lam_g and lam_x negated fails (> 1)",
   np.abs(gJ - JgT_lg - lx).max() > 1)
ok("6d R share: 0.04 < J - lap_time < 0.06", 0.04 < float(c0.nlp_fg(w)[0]) - d.lap_time < 0.06)

# ---- 7. oracle: finite differences of the rebuilt Lagrangian ----------------------
print("7. oracle: rebuilt Lagrangian differences == limit prices")
P = limit_prices(d)


def bnd(b, lam, lo, hi):
    """The bound that each multiplier sees: the upper one where lam > 0, the lower one where lam < 0."""
    return np.where(lam > 0, b[hi], np.where(lam < 0, b[lo], 0.0))


def dV(cA, cB):
    """First-order change of the optimal value from build A to build B, at the stored point w."""
    JA, gA = cA.nlp_fg(w)
    JB, gB = cB.nlp_fg(w)
    JA, JB = float(JA), float(JB)
    gA, gB = np.asarray(gA).ravel(), np.asarray(gB).ravel()
    bA, bB = cA.nlp_bounds, cB.nlp_bounds
    return ((JB - JA) + lg @ (gB - gA)
            - lg @ (bnd(bB, lg, "lbg", "ubg") - bnd(bA, lg, "lbg", "ubg"))
            - lx @ (bnd(bB, lx, "lbw", "ubw") - bnd(bA, lx, "lbw", "ubw")))


c_a, c_b = rebuild(), rebuild(vi=60.5)
ok("7a vi: dV / 0.5 == limit_prices vi (rel 1e-4)", close(dV(c_a, c_b) / 0.5, P["vi"], 1e-4))
del c_a, c_b

c_a, c_b = rebuild_pt("Pmax", 1 - 1e-6), rebuild_pt("Pmax", 1 + 1e-6)
ok("7b Pmax: dV / (2e-6 Pmax) == limit_prices Pmax (rel 1e-4)",
   close(dV(c_a, c_b) / (2e-6 * c0.pt.Pmax), P["Pmax"], 1e-4))
del c_a, c_b

c_a, c_b = rebuild_pt("Tmax", 1 - 1e-6), rebuild_pt("Tmax", 1 + 1e-6)
ok("7c Tmax: dV / (2e-6 Tmax) == limit_prices Tmax (rel 1e-4)",
   close(dV(c_a, c_b) / (2e-6 * c0.pt.Tmax), P["Tmax"], 1e-4))
del c_a, c_b

Tb = c0.vp.Tbrake_max
c_a = rebuild(vp_overrides={"Tbrake_max": Tb * (1 - 1e-6)})
c_b = rebuild(vp_overrides={"Tbrake_max": Tb * (1 + 1e-6)})
ok("7d Tbrake_max: dV / (2e-6 Tb) == limit_prices Tbrake_max (rel 1e-4)",
   close(dV(c_a, c_b) / (2e-6 * Tb), P["Tbrake_max"], 1e-4))
del c_a, c_b

c_a, c_b = rebuild(n_bounds=3.99), rebuild(n_bounds=4.01)
ok("7e W: dV / 0.02 == limit_prices W (rel 1e-4)", close(dV(c_a, c_b) / 0.02, P["W"], 1e-4))
del c_a, c_b

print("test_sensitivity: all asserts passed")
