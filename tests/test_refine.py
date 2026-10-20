"""Adaptive h-refinement of the collocation mesh: functions/refine.py and MLTP(refine=...)
(plain script, no pytest). Run from the repo root:

    venv\\Scripts\\python.exe tests\\test_refine.py

  1. lagrange_basis against collocation_coeff (C, D)
  2. defect_errors is zero (to rounding) for a solution that is a polynomial of degree
     <= OPT_d, with the (x, u, pv) and the aux (x, u, y, pv) signature
  3. ... O(h^(d+1)) for a smooth solution (harmonic oscillator at three resolutions)
  4. ... and localised: a curvature step inside one interval flags that interval only
  5. nlp_inputs / defect_errors use the NLP's own in-interval input arithmetic (pinned
     by one tiny real NLP solve, < 0.1 s)
  6. refine_knots: nested bisection, ds_min, max_N, max_split, merge (merged knots count
     towards max_N in the same pass)
  7. discretise(s_knot=refined) and the reseed helpers on the refined grid
  8. run_refinement stop reasons with fake solve / indicator callbacks (eta_max, not the
     mean, drives the stop; the lap-rise rejection; no merge-only re-solve)
  9. refine_options validation
 10. refine_record savemat round trip (single attempt included), result_stem(..., 'adaptive')
 11. MLTP(refine=...) wiring with a stand-in build_and_solve_nlp (no 23-state NLP solve):
     nested passes, capped max_iter, the reseed at kept and new knots, the indicator and lap
     against an independent evaluation, the cold retry, the lap-rise rejection, records,
     save=True, time accounting, bad state names before any solve, refine=None

Needs casadi (prints SKIP and exits 0 otherwise); section 11 also needs Data/DATA_AA.mat.
"""
import contextlib
import inspect
import io
import os
import sys
import tempfile
import time
import warnings
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
ROOT = _bootstrap.ROOT

try:
    import casadi as ca
except ImportError as exc:                                      # pragma: no cover
    print(f"SKIP test_refine: casadi not importable ({exc})")
    sys.exit(0)

from functions.collocation import collocation_points, collocation_coeff
from functions.importfile import load_solution, result_stem
from functions.refine import (RECORD_COLUMNS, REFINE_DEFAULTS, defect_errors, eval_states,
                              interval_polynomials, lagrange_basis, lap_rise, nlp_inputs,
                              pass_accepted, refine_knots, refine_options, refine_record,
                              resolve_max_N, run_refinement, state_rows)
from functions.transcription import (build_and_solve_nlp, compute_time, discretise, interp_inputs,
                                     unpack_solution)
from functions.warmstart import nlp_structure


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc:
        return True
    return False


d = 3
tau = np.asarray(collocation_points(d, "legendre"), dtype=float)
C, D, B = collocation_coeff(tau)
nodes = np.concatenate(([0.0], tau))
u = ca.SX.sym("u", 1)
pv = ca.SX.sym("pv", 1)

# =============================================================================
print("1. lagrange_basis on the collocation nodes [0, tau_1..tau_d]")
L_tau, dL_tau = lagrange_basis(nodes, tau)
L_one, _ = lagrange_basis(nodes, [1.0])
t_any = np.linspace(-0.3, 1.4, 17)
L_any, dL_any = lagrange_basis(nodes, t_any)
ok("dL at the collocation points == collocation_coeff's C", np.allclose(dL_tau, C, rtol=0, atol=1e-12))
ok("L at t = 1 == collocation_coeff's D", np.allclose(L_one[:, 0], D[:, 0], rtol=0, atol=1e-12))
ok("columns sum to 1 (dL columns to 0) and L at the nodes is the identity",
   np.allclose(L_any.sum(axis=0), 1.0, atol=1e-12) and np.allclose(dL_any.sum(axis=0), 0.0, atol=1e-11)
   and np.allclose(lagrange_basis(nodes, nodes)[0], np.eye(d + 1), atol=1e-14))

# =============================================================================
print("2. exactness: a polynomial solution of degree <= OPT_d has zero defect")
x4 = ca.SX.sym("x", 4)
f4 = ca.Function("f4", [x4, u, pv], [ca.vertcat(x4[1], x4[2], u, pv), x4[0]], ["x", "u", "pv"], ["dx", "L"])
a_k, b_k, c3 = 0.01, 2e-4, 0.3                   # kappa = a + b s (linear), x1''' = u = c3
s_lin = np.linspace(0.0, 120.0, 241)
track_lin = SimpleNamespace(s=s_lin, k=a_k + b_k * s_lin)


def exact4(s):
    s = np.asarray(s, dtype=float)
    return np.vstack([1.0 - 0.1 * s + 0.1 * s**2 + c3 * s**3 / 6, -0.1 + 0.2 * s + c3 * s**2 / 2,
                      0.2 + c3 * s, 0.5 + a_k * s + b_k * s**2 / 2])


disc4 = discretise(track_lin, 30, d)
N4 = disc4["N"]
X4, X4c = exact4(disc4["s_knot"]), exact4(disc4["s_col"])
U4 = np.full((1, N4 + 1), c3)
de4 = defect_errors(f4, disc4, track_lin, X4, X4c, U4)
xmax = float(np.abs(X4).max())
ok(f"cubic / quadratic exact solution, N={N4}: max E = {de4['E'].max():.1e} <= 1e-12 * max|X| "
   f"({xmax:.0f})", de4["E"].shape == (4, N4) and de4["E"].max() <= 1e-12 * xmax)
ok(f"lap-time quadrature defect dT ~ 0 (L = x1, a cubic: both quadratures exact), "
   f"max |dT| = {np.abs(de4['dT']).max():.1e}",
   de4["dT"].shape == (N4,) and np.abs(de4["dT"]).max() <= 1e-12 * 30.0 * xmax)
ok("n_eval = N * (2d sub-segments x (d+1) Gauss points + d collocation points) = 27 N",
   de4["n_eval"] == N4 * (2 * d * (d + 1) + d))
y1 = ca.SX.sym("y", 1)
f4y = ca.Function("f4y", [x4, u, y1, pv], [ca.vertcat(x4[1], x4[2], u + y1, pv), x4[0]],
                  ["x", "u", "y", "pv"], ["dx", "L"])
U4y, Y4y = np.full((1, N4 + 1), 0.2), np.full((1, N4 + 1), c3 - 0.2)
de4y = defect_errors(f4y, disc4, track_lin, X4, X4c, U4y, Yk=Y4y)
ok(f"aux signature f(x, u, y, pv) (the 7-state model's): max E = {de4y['E'].max():.1e}, same bound",
   de4y["E"].max() <= 1e-12 * xmax)
ok("... the aux values are used: y = 0 leaves a defect",
   defect_errors(f4y, disc4, track_lin, X4, X4c, U4y, Yk=0.0 * Y4y)["E"].max() > 1e-3)
ok("f_dyn signature and Yk must agree (ValueError)",
   raises(ValueError, defect_errors, f4y, disc4, track_lin, X4, X4c, U4y)
   and raises(ValueError, defect_errors, f4, disc4, track_lin, X4, X4c, U4, Yk=Y4y))

# =============================================================================
print("3. order: a smooth solution sampled exactly at the nodes, h = 40 / 20 / 10 m")
xo = ca.SX.sym("xo", 2)
w_o = 0.05
f_osc = ca.Function("fo", [xo, u, pv], [ca.vertcat(xo[1], -w_o**2 * xo[0]), 1 + 0 * xo[0]])
track_flat = SimpleNamespace(s=np.linspace(0.0, 240.0, 481), k=np.zeros(481))


def exact_osc(s):
    return np.vstack([np.cos(w_o * s), -w_o * np.sin(w_o * s)])


E_h = []
for ds in (40, 20, 10):
    dd = discretise(track_flat, ds, d)
    E_h.append(defect_errors(f_osc, dd, track_flat, exact_osc(dd["s_knot"]), exact_osc(dd["s_col"]),
                             np.zeros((1, dd["N"] + 1)))["E"].max())
ratios = [E_h[0] / E_h[1], E_h[1] / E_h[2]]
ok(f"max E {E_h[0]:.2e} / {E_h[1]:.2e} / {E_h[2]:.2e}: ratios {ratios[0]:.1f}, {ratios[1]:.1f} per halving "
   "in [10, 24] (O(h^(d+1)) = 16)", all(10.0 <= r <= 24.0 for r in ratios))

# =============================================================================
print("4. localisation: deps/ds = -kappa with a curvature step at 97 m inside [90, 120]")
xe = ca.SX.sym("xe", 1)
f_eps = ca.Function("fe", [xe, u, pv], [-pv, 1 + 0 * xe])
s_step = np.linspace(0.0, 240.0, 2401)
track_step = SimpleNamespace(s=s_step, k=np.where(s_step < 97.0, 0.0, 0.05))
disc_st = discretise(track_step, 30, d)


def exact_eps(s):
    return (-0.05 * np.clip(np.asarray(s, dtype=float) - 97.0, 0.0, None)).reshape(1, -1)


E_st = defect_errors(f_eps, disc_st, track_step, exact_eps(disc_st["s_knot"]), exact_eps(disc_st["s_col"]),
                     np.zeros((1, disc_st["N"] + 1)))["E"][0]
k_hot = int(np.argmax(E_st))
ok(f"argmax E is interval {k_hot} = [{disc_st['s_knot'][k_hot]:.0f}, {disc_st['s_knot'][k_hot + 1]:.0f}] m "
   f"(E = {E_st[k_hot]:.1e}), the one holding the step",
   disc_st["s_knot"][k_hot] < 97.0 < disc_st["s_knot"][k_hot + 1] and E_st[k_hot] > 1e-2)
ok(f"every other interval < 1e-12 (max {np.delete(E_st, k_hot).max():.1e})", np.delete(E_st, k_hot).max() < 1e-12)

# =============================================================================
print("5. the NLP's in-interval input arithmetic (one tiny real solve)")
xs1, us1, ps1 = ca.SX.sym("x1", 1), ca.SX.sym("u1", 1), ca.SX.sym("p1", 1)
f1 = ca.Function("f1", [xs1, us1, ps1], [us1, xs1**2], ["x", "u", "pv"], ["dx", "L"])
f1_sf = ca.Function("sf", [xs1, ps1], [xs1**2 + 0 * ps1], ["x", "kappa"], ["sf"])
h1 = ca.Function("h", [xs1, us1, ps1], [xs1], ["x", "u", "pv"], ["h"])
m1 = SimpleNamespace(nx=1, nu=1, ny=0, x_min=np.array([-10.0]), x_max=np.array([10.0]),
                     u_min=np.array([-0.05]), u_max=np.array([0.05]), x_s=np.array([1.0]),
                     u_s=np.array([1.0]))
track1 = SimpleNamespace(s=np.linspace(0.0, 40.0, 81), k=np.zeros(81))
disc1 = discretise(track1, 10, d)
N1 = disc1["N"]
g1 = {"x0": np.ones((1, N1 + 1)), "u0": np.zeros((1, N1 + 1)), "xc0": np.ones((1, N1 * d))}
reg1 = {"ru": np.zeros(1), "rdu": np.zeros(1), "rdu2": np.zeros(1)}
opts1 = {"ipopt": {"linear_solver": "mumps", "print_level": 0, "max_iter": 200, "tol": 1e-10, "sb": "yes"},
         "print_time": False}
with contextlib.redirect_stdout(io.StringIO()):
    r1 = build_and_solve_nlp(ca, m1, f1, f1_sf, h1, np.array([-10.0]), np.array([10.0]), disc1, g1, reg1,
                             np.array([-1e3]), np.array([1e3]), np.array([1.0]), np.array([np.nan]),
                             d, "linear", 1e-2, opts1)
X1, U1, _, X1c = unpack_solution(r1["w_opt"], 1, 1, 0, N1, d, [1.0], [1.0], None)
status1 = r1["solver"].stats()["return_status"]
ok(f"toy x' = u, min int x^2, |u| <= 0.05, N={N1}: {status1}, the inputs vary (U = {np.round(U1[0], 4)})",
   status1 == "Solve_Succeeded" and np.ptp(U1) > 1e-2)
Z1 = interval_polynomials(disc1, X1, X1c)
u_nlp = nlp_inputs(disc1, U1, disc1["s_col"])
u_lin = interp_inputs(U1, disc1["s_knot"], disc1["s_col"], "linear")


def node_residual(u_col):
    return max(float(np.max(np.abs(Z1[:, :, k] @ dL_tau - disc1["dsk"][k] * u_col[:, k * d:(k + 1) * d])))
               for k in range(N1))


res_nlp, res_lin = node_residual(u_nlp), node_residual(u_lin)
ok(f"collocation residual with refine.nlp_inputs {res_nlp:.1e} <= 1e-8, with the true linear "
   f"interpolant {res_lin:.1e} > 1e-2 (transcription and refine agree on U_k + duk * tau)",
   res_nlp <= 1e-8 and res_lin > 1e-2)
E1 = defect_errors(f1, disc1, track1, X1, X1c, U1)["E"].max()
ok(f"defect_errors uses the same arithmetic: E = {E1:.1e} <= 1e-8 on the converged toy", E1 <= 1e-8)
ok("nlp_inputs is exact at every knot, U_N included",
   np.array_equal(nlp_inputs(disc1, U1, disc1["s_knot"]), U1))
ok("'constant' holds U_k inside interval k",
   np.array_equal(nlp_inputs(disc1, U1, disc1["s_col"], "constant"), np.repeat(U1[:, :N1], d, axis=1)))

# =============================================================================
print("6. refine_knots")
s12 = np.linspace(0.0, 540.0, 13)               # N = 12, h = 45 m
eta12 = np.array([1e-3, 5e-3, 2e-2, 0.11, 0.08, 9e-3, 1e-2, 0.03, 1e-4, 1e-4, 1e-4, 0.05])
over = eta12 > 1e-2                             # eta == tol is not refined
s_b, info_b = refine_knots(s12, eta12, 1e-2)
added = np.setdiff1d(s_b, s12)
ok(f"bisection: {int(over.sum())} intervals over tol -> N {info_b['N_old']} -> {info_b['N_new']} "
   "= N + n_split, n_split counted, the split mask is eta > tol",
   info_b["n_split"] == over.sum() == 5 and s_b.size - 1 == 12 + 5 and info_b["N_new"] == 17
   and np.array_equal(info_b["split"], over))
ok("nested (every old knot kept), strictly increasing, end points fixed",
   np.all(np.isin(s12, s_b)) and np.all(np.diff(s_b) > 0) and s_b[0] == s12[0] and s_b[-1] == s12[-1])
ok("the new knots are exactly the midpoints of the intervals over tol",
   np.allclose(added, 0.5 * (s12[:-1] + s12[1:])[over], rtol=0, atol=1e-12))
s_n, info_n = refine_knots(s12, eta12, 1e-2, ds_min=30.0)
ok("ds_min = 30 m blocks every bisection of a 45 m interval (22.5 m parts): knots unchanged, 5 blocked",
   np.array_equal(s_n, s12) and info_n["n_split"] == 0 and info_n["blocked"] == 5)
s_nu = np.array([0.0, 80.0, 125.0, 170.0])
s_p, info_p = refine_knots(s_nu, np.full(3, 1.0), 1e-2, ds_min=30.0)
ok("non-uniform knots: only the 80 m interval clears ds_min = 30 m", np.array_equal(s_p, [0, 40, 80, 125, 170])
   and info_p["blocked"] == 2)
s_c, info_c = refine_knots(s12, eta12, 1e-2, max_N=14)
ok("max_N = 14: the two worst intervals (eta 0.11, 0.08) split first, capped flagged",
   s_c.size - 1 == 14 and info_c["capped"] and np.allclose(np.setdiff1d(s_c, s12), [157.5, 202.5]))
ok("max_N not binding -> not capped", not refine_knots(s12, eta12, 1e-2, max_N=17)[1]["capped"])
eta4 = np.full(12, 1e-4)
eta4[2], eta4[7] = 1.0, 0.2                     # (eta/tol)^(1/4) = 3.16 -> 4 parts, 2.11 -> 3 parts
s_4, info_4 = refine_knots(s12, eta4, 1e-2, max_split=4)
ok("max_split = 4: ceil((eta/tol)^(1/(d+1))) equal parts (4 and 3), nested",
   list(info_4["parts"][[2, 7]]) == [4, 3] and s_4.size - 1 == 12 + 3 + 2 and np.all(np.isin(s12, s_4))
   and np.allclose(np.diff(s_4[(s_4 >= 90) & (s_4 <= 135)]), 11.25)
   and np.allclose(np.diff(s_4[(s_4 >= 315) & (s_4 <= 360)]), 15.0))
ok("max_split = 4 with ds_min = 12: 4 parts of 11.25 m are too short -> 3 parts of 15 m",
   refine_knots(s12, eta4, 1e-2, max_split=4, ds_min=12.0)[1]["parts"][2] == 3)
ok("max_split = 2 always bisects", refine_knots(s12, eta4, 1e-2)[1]["parts"].max() == 2)
eta_m = np.array([1e-4, 1e-4, 1e-4, 0.05, 1e-4, 2e-4, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
s_m, info_m = refine_knots(s12, eta_m, 1e-2, merge=True)
ok(f"merge: pairs both below tol/32 joined left to right, each interval once, unsplit only "
   f"({info_m['n_merged']} merges: 0-90, 180-270, 315-405, 405-495 m)",
   info_m["n_merged"] == 4 and np.array_equal(np.setdiff1d(s12, s_m), [45.0, 225.0, 360.0, 450.0])
   and np.array_equal(np.setdiff1d(s_m, s12), [157.5]) and s_m.size - 1 == 12 + 1 - 4
   and s_m[0] == 0.0 and s_m[-1] == 540.0
   and np.array_equal(np.flatnonzero(info_m["merged"]), [0, 1, 4, 5, 7, 8, 9, 10]))
ok("merge respects ds_max (a 90 m union with ds_max = 80 m is not joined)",
   refine_knots(s12, eta_m, 1e-2, merge=True, ds_max=80.0)[1]["n_merged"] == 0)
ok("merge never joins a split interval or one at/above merge_ratio * tol",
   refine_knots(s12, np.full(12, 5e-3), 1e-2, merge=True)[1]["n_merged"] == 0
   and refine_knots(s12, eta_m, 1e-2, merge=True, merge_ratio=1e-3)[1]["n_merged"] == 0)
ok("merge=False never removes a knot (any eta)",
   all(np.all(np.isin(s12, refine_knots(s12, e, 1e-2)[0])) for e in (eta12, eta_m, eta4, np.zeros(12))))
mid12 = 0.5 * (s12[:-1] + s12[1:])
eta_mb = np.where((mid12 > 150.0) & (mid12 < 330.0), 0.08, 1e-6)    # 4 hot intervals, 8 cold
s_mb, info_mb = refine_knots(s12, eta_mb, 1e-2, max_N=12, merge=True)
ok(f"merge + max_N = N: the {info_mb['n_merged']} knots the merges free are spent on hot intervals in the "
   f"same pass ({info_mb['n_split']} split, N stays {info_mb['N_new']}, capped), not a coarsening-only pass",
   info_mb["n_merged"] == 3 and info_mb["n_split"] == 3 and info_mb["capped"] and info_mb["N_new"] == 12
   and np.allclose(np.setdiff1d(s_mb, s12), [157.5, 202.5, 247.5]))
ok("merge never joins an interval over tol, even one blocked by ds_min (merge_ratio >= 1)",
   refine_knots(s12, eta_mb, 1e-2, merge=True, merge_ratio=100.0, ds_min=30.0)[1]["merged"][3:7].sum() == 0)
ok("bad inputs raise (eta length, non-increasing knots, max_split < 2)",
   raises(ValueError, refine_knots, s12, eta12[:-1], 1e-2)
   and raises(ValueError, refine_knots, s12[::-1], eta12, 1e-2)
   and raises(ValueError, refine_knots, s12, eta12, 1e-2, max_split=1))

# =============================================================================
print("7. discretise(s_knot=refined) and the reseed helpers on the refined grid")
s_tr7 = np.linspace(0.0, 540.0, 271)
track7 = SimpleNamespace(s=s_tr7, k=0.02 * np.sin(s_tr7 / 40.0))
disc_a = discretise(track7, 45, d)
s_r, _ = refine_knots(disc_a["s_knot"], eta12, 1e-2)
disc_r = discretise(track7, 45, d, s_knot=s_r)
Nr = disc_r["N"]
s_col_r = disc_r["s_col"].reshape(d, Nr, order="F")
ok(f"N = {Nr} = len(s_knot) - 1, mesh 'custom', s_full[::d+1] == s_knot, s_full has (d+1) N + 1 points",
   Nr == s_r.size - 1 and disc_r["mesh"] == "custom" and np.array_equal(disc_r["s_full"][::d + 1], s_r)
   and disc_r["s_full"].size == (d + 1) * Nr + 1)
ok("collocation points inside their own interval, k_col == interp(track)",
   np.all(s_col_r > s_r[None, :-1]) and np.all(s_col_r < s_r[None, 1:])
   and np.allclose(disc_r["k_col"], np.interp(disc_r["s_col"], track7.s, track7.k), rtol=0, atol=0))
rs = np.random.RandomState(3)
Xa, Xac, Ua = rs.randn(5, 13), rs.randn(5, 12 * d), rs.randn(2, 13)
Za = interval_polynomials(disc_a, Xa, Xac)
x_knots = eval_states(disc_a, Za, s_r, x_end=Xa[:, -1])
x_cols = eval_states(disc_a, Za, disc_r["s_col"])
kept = np.isin(s_r, disc_a["s_knot"])
ok(f"eval_states: ({x_knots.shape[0]}, {Nr + 1}) at the knots, ({x_cols.shape[0]}, {Nr * d}) at the "
   "collocation points; the kept knots return the old knot states exactly (X_N included)",
   x_knots.shape == (5, Nr + 1) and x_cols.shape == (5, Nr * d)
   and np.array_equal(x_knots[:, kept], Xa))
ok("eval_states at the old collocation points returns X_kj (the polynomial interpolates its nodes)",
   np.allclose(eval_states(disc_a, Za, disc_a["s_col"]), Xac, rtol=0, atol=1e-12))
u_knots = nlp_inputs(disc_a, Ua, s_r)
k0 = int(np.flatnonzero(over)[0])
mid = 0.5 * (disc_a["s_knot"][k0] + disc_a["s_knot"][k0 + 1])
ok("nlp_inputs on the refined knots: exact at the kept ones, U_k + duk * 0.5 at a new midpoint",
   u_knots.shape == (2, Nr + 1) and np.array_equal(u_knots[:, kept], Ua)
   and np.allclose(u_knots[:, np.flatnonzero(s_r == mid)[0]],
                   Ua[:, k0] + (Ua[:, k0 + 1] - Ua[:, k0]) / disc_a["dsk"][k0] * 0.5, rtol=0, atol=1e-14))

# =============================================================================
print("8. run_refinement (fake solve and indicator)")
H0 = 45.0


def fake_eval(r):                     # eta_k = C_k (h_k / h0)^4, hot spot on 150-330 m
    s = np.asarray(r["s_knot"])
    hk, smid = np.diff(s), 0.5 * (s[:-1] + s[1:])
    return np.where((smid > 150.0) & (smid < 330.0), 0.08, 0.004) * (hk / H0) ** 4, {"dT": np.zeros(hk.size)}


def make_step(statuses=None, attempts=False):
    steps = []

    def step(cur, s_new):
        status = statuses.pop(0) if statuses else "Solve_Succeeded"
        steps.append(np.array(s_new))
        out = dict(status=status, iters=10, wall=0.5, lap=17.9, seed="reseed")
        if attempts:
            out["seed"] = "cold"
            out["attempts"] = [dict(status="Maximum_Iterations_Exceeded", iters=1000, wall=9.0,
                                    lap=np.nan, seed="reseed")]
        return out
    return step, steps


def base():
    return dict(status="Solve_Succeeded", iters=289, wall=20.0, lap=18.2, seed="base", warm_start="cold",
                s_knot=np.linspace(0.0, 540.0, 13))


step, steps = make_step()
first = base()
final, log, stop = run_refinement(first, step, fake_eval, refine_options(dict(tol=1e-2), H0))
hot = np.array([157.5, 202.5, 247.5, 292.5])     # midpoints of the 4 intervals over tol
ok("tol: one pass bisects exactly the hot intervals, then every eta <= tol -> 'tol'",
   stop == "tol" and len(steps) == 1 and np.allclose(np.setdiff1d(steps[0], first["s_knot"]), hot)
   and final["status"] == "Solve_Succeeded" and final is not first)
ok("log: base row + one pass row (pass, seed, N, n_split, eta evaluated on both)",
   [e["pass"] for e in log] == [0, 1] and [e["seed"] for e in log] == ["base", "reseed"]
   and [e["N"] for e in log] == [12, 16] and [e["n_split"] for e in log] == [0, 4]
   and all(e["eta"] is not None and e["converged"] for e in log)
   and log[0]["n_over"] == 4 and log[1]["eta_max"] <= 1e-2 and log[0]["s_argmax"] in hot)
step, steps = make_step()
final, log, stop = run_refinement(base(), step, fake_eval, refine_options(dict(tol=1e-9, passes=2), H0))
ok("tol = 1e-9, passes = 2: exactly 2 steps (N 12 -> 24 -> 48 = max_N) then 'passes'",
   stop == "passes" and len(steps) == 2 and [e["N"] for e in log] == [12, 24, 48])
step, steps = make_step()
final, log, stop = run_refinement(base(), step, fake_eval, refine_options(dict(tol=1e-9, passes=5, max_N=30), H0))
ok("max_N = 30: 12 -> 24 -> 30 (worst-first, capped) then the knots cannot change -> 'max_N'",
   stop == "max_N" and len(steps) == 2 and [e["N"] for e in log] == [12, 24, 30]
   and [e["capped"] for e in log] == [False, False, True] and final["s_knot"].size == 31)
step, steps = make_step()
final, log, stop = run_refinement(base(), step, fake_eval, refine_options(dict(tol=1e-9, ds_min=30.0), H0))
ok("ds_min = 30 m on 45 m intervals: nothing can be split -> 'no-split', no solve",
   stop == "no-split" and len(steps) == 0 and len(log) == 1 and final["s_knot"].size == 13)
step, steps = make_step(["Maximum_Iterations_Exceeded"])
first = base()
final, log, stop = run_refinement(first, step, fake_eval, refine_options(dict(tol=1e-2), H0))
ok("a pass that does not converge: previous result returned, 'solve-failed', the attempt logged",
   stop == "solve-failed" and final is first and len(log) == 2 and not log[1]["converged"]
   and log[1]["status"] == "Maximum_Iterations_Exceeded" and log[1]["eta"] is None
   and np.isnan(log[1]["eta_max"]))
step, steps = make_step(attempts=True)
final, log, stop = run_refinement(base(), step, fake_eval, refine_options(dict(tol=1e-2), H0))
ok("failed attempts listed by a step (reseed before a cold retry) are logged before the accepted one",
   stop == "tol" and [e["pass"] for e in log] == [0, 1, 1] and [e["seed"] for e in log] == ["base", "reseed", "cold"]
   and [e["converged"] for e in log] == [True, False, True] and log[1]["N"] == log[2]["N"] == 16)
step, steps = make_step()
evals = []
first = dict(base(), status="Infeasible_Problem_Detected")
final, log, stop = run_refinement(first, step, lambda r: evals.append(1) or fake_eval(r),
                                  refine_options(True, H0))
ok("base not converged: 'base-failed', no step and no evaluation",
   stop == "base-failed" and final is first and not steps and not evals and len(log) == 1
   and not log[0]["converged"])
step, steps = make_step()
final, log, stop = run_refinement(base(), step, fake_eval, refine_options(dict(passes=0), H0))
ok("passes = 0: base only, evaluated, stop 'passes'",
   stop == "passes" and not steps and len(log) == 1 and log[0]["eta"] is not None)


def one_hot_eval(r):                  # one hot interval (135-180 m): eta_mean <= tol < eta_max
    s = np.asarray(r["s_knot"])
    hk, smid = np.diff(s), 0.5 * (s[:-1] + s[1:])
    return np.where((smid > 135.0) & (smid < 180.0), 0.05, 1e-3) * (hk / H0) ** 4, {}


eta_oh = one_hot_eval(base())[0]
step, steps = make_step()
final, log, stop = run_refinement(base(), step, one_hot_eval, refine_options(dict(tol=1e-2), H0))
ok(f"eta_mean {eta_oh.mean():.4f} <= tol < eta_max {eta_oh.max():.2f}: the max drives the stop (one pass, "
   "then 'tol'); the log holds the exact max, mean and count over tol",
   stop == "tol" and len(steps) == 1 and log[0]["eta_max"] == eta_oh.max() == 0.05
   and log[0]["eta_mean"] == eta_oh.mean() and log[0]["n_over"] == 1 and log[1]["eta_max"] <= 1e-2)

ok("lap_rise / pass_accepted: (lap - parent) / parent; converged and at most max_lap_rise slower; "
   "an unknown lap is not judged",
   np.isclose(lap_rise(dict(lap=18.18), dict(lap=18.0)), 0.01)
   and np.isnan(lap_rise(dict(lap=np.nan), dict(lap=18.0))) and np.isnan(lap_rise(dict(), dict(lap=18.0)))
   and pass_accepted(dict(status="Solve_Succeeded", lap=18.05), dict(lap=18.0), dict(max_lap_rise=3e-3))
   and not pass_accepted(dict(status="Solve_Succeeded", lap=18.06), dict(lap=18.0), dict(max_lap_rise=3e-3))
   and pass_accepted(dict(status="Solve_Succeeded", lap=19.0), dict(lap=18.0), dict(max_lap_rise=np.inf))
   and not pass_accepted(dict(status="Maximum_Iterations_Exceeded", lap=17.0), dict(lap=18.0), {})
   and pass_accepted(dict(status="Solve_Succeeded"), dict(lap=18.0), dict(max_lap_rise=0.0)))


def lap_step(laps):                   # converged steps with the given laps, in order
    laps = list(laps)

    def step(cur, s_new):
        return dict(status="Solve_Succeeded", iters=10, wall=0.5, lap=laps.pop(0), seed="reseed")
    return step


first = base()
final, log, stop = run_refinement(first, lap_step([18.3]), fake_eval, refine_options(dict(tol=1e-9), H0))
ok("a converged pass 0.55% slower than the pass it refines (> max_lap_rise 0.3%): rejected, 'lap-rise', "
   "the base returned; the row is converged but not accepted and is never evaluated",
   stop == "lap-rise" and final is first and [e["converged"] for e in log] == [True, True]
   and [e["accepted"] for e in log] == [True, False] and log[1]["eta"] is None and log[1]["lap"] == 18.3)
final, log, stop = run_refinement(base(), lap_step([18.25, 18.3]), fake_eval,
                                  refine_options(dict(tol=1e-9, passes=2), H0))
ok("a pass 0.27% slower is accepted and judged against its own parent next (18.25 -> 18.3 = +0.27%)",
   stop == "passes" and [e["accepted"] for e in log] == [True, True, True] and final["lap"] == 18.3)
final, log, stop = run_refinement(base(), lap_step([19.0]), fake_eval,
                                  refine_options(dict(tol=1e-9, passes=1, max_lap_rise=np.inf), H0))
ok("max_lap_rise = inf switches the check off", stop == "passes" and log[1]["accepted"] and final["lap"] == 19.0)

def quiet_eval(r):                    # fake_eval's hot spot over a 1e-6 background (mergeable)
    s = np.asarray(r["s_knot"])
    hk, smid = np.diff(s), 0.5 * (s[:-1] + s[1:])
    return np.where((smid > 150.0) & (smid < 330.0), 0.08, 1e-6) * (hk / H0) ** 4, {}


step, steps = make_step()
final, log, stop = run_refinement(base(), step, quiet_eval, refine_options(dict(max_N=12, merge=True, passes=3), H0))
ok(f"merge=True with max_N = N0 = 12 (4 hot intervals): N {[e['N'] for e in log]}, stop {stop!r}: freed knots "
   "refine the hot spots, the run meets tol without exceeding max_N or a coarsening-only pass",
   stop == "tol" and all(e["N"] <= 12 for e in log) and all(e["n_split"] > 0 for e in log[1:])
   and log[-1]["eta_max"] <= 1e-2)
step, steps = make_step()
final, log, stop = run_refinement(base(), step, fake_eval, refine_options(dict(merge=True, ds_min=30.0), H0))
ok("merge=True but every interval over tol blocked by ds_min: 'no-split' and no solve (no merge-only pass)",
   stop == "no-split" and not steps and len(log) == 1)
fail_log = run_refinement(base(), make_step(attempts=True)[0], fake_eval, refine_options(dict(tol=1e-2), H0))

# =============================================================================
print("9. refine_options")
o_def = refine_options(True, 30.0)
ok("None / False -> None (refinement off)", refine_options(None, 30) is None and refine_options(False, 30) is None)
ok("True -> the defaults: passes 2, tol 1e-2, max_N None (4 N0), merge off, bisection, 1000 its, n/eps, "
   "max_lap_rise 0.3%",
   o_def["passes"] == 2 and o_def["tol"] == 1e-2 and o_def["max_N"] is None and o_def["merge"] is False
   and o_def["max_split"] == 2 and o_def["pass_max_iter"] == 1000 and o_def["states"] == ("n", "eps")
   and o_def["max_lap_rise"] == 3e-3 and set(o_def) == set(REFINE_DEFAULTS)
   and refine_options(dict(max_lap_rise=np.inf), 30.0)["max_lap_rise"] == np.inf
   and refine_options(dict(max_lap_rise=0), 30.0)["max_lap_rise"] == 0.0)
ok("ds_min / ds_max default to 0.125 / 2.5 x OPT_ds; resolve_max_N gives 4 N0 unless set",
   o_def["ds_min"] == 3.75 and o_def["ds_max"] == 75.0 and refine_options(True, 45.0)["ds_min"] == 5.625
   and resolve_max_N(o_def, 18) == 72 and resolve_max_N(refine_options(dict(max_N=50), 30), 18) == 50)
o_d = refine_options(dict(passes=3, tol=5e-3, states="vx", merge=True, ds_max=np.inf), 30.0)
ok("a dict overrides only its keys (a single state name becomes a tuple, ds_max may be inf)",
   o_d["passes"] == 3 and o_d["tol"] == 5e-3 and o_d["states"] == ("vx",) and o_d["merge"] is True
   and o_d["ds_max"] == np.inf and o_d["ds_min"] == 3.75)
bad = [dict(passes=-1), dict(tol=0.0), dict(tol=-1e-3), dict(tol=np.nan), dict(max_split=1),
       dict(ds_min=100.0, ds_max=50.0), dict(ds_min=50.0, ds_max=50.0), dict(passes=True),
       dict(passes=1.5), dict(max_N=0), dict(merge="no"), dict(states=()), dict(pass_max_iter=0),
       dict(max_lap_rise=-1e-3), dict(max_lap_rise=np.nan), dict(max_lap_rise="x"), dict(max_lap_rise=True),
       dict(bogus=1), "yes", 3]
ok(f"unknown key / bad value -> ValueError ({len(bad)} cases)",
   all(raises(ValueError, refine_options, b, 30.0) for b in bad))

# =============================================================================
print("10. refine_record savemat round trip; result_stem(..., 'adaptive')")
final, log, stop = fail_log
opts_r = refine_options(dict(tol=1e-2), H0)
rec = refine_record(log, opts_r, stop, "uniform")


def has_none(v):
    if v is None:
        return True
    if isinstance(v, dict):
        return any(has_none(x) for x in v.values())
    return False


ok("record: no None anywhere, max_N resolved (4 x 12), one entry per attempt (converged / accepted)",
   not has_none(rec) and rec["opts"]["max_N"] == 48 and rec["n_attempts"] == 3
   and rec["N"].tolist() == [12, 16, 16] and rec["converged"].tolist() == [1, 0, 1]
   and rec["accepted"].tolist() == [1, 0, 1] and rec["pass_no"].tolist() == [0, 1, 1]
   and rec["seed"] == "base, reseed, cold" and rec["stop_reason"] == "tol"
   and rec["opts"]["max_lap_rise"] == 3e-3 and set(RECORD_COLUMNS) <= set(rec))
tmp = tempfile.mkdtemp()
path_r = os.path.join(tmp, "refine_record.mat")
sio.savemat(path_r, {"data": {"lap_time": 1.0, "refine": rec}}, do_compression=True)
back = load_solution(path_r).refine
ok("savemat -> load_solution: strings, ints and arrays come back (NaN for the unevaluated attempt)",
   back.stop_reason == "tol" and back.base_mesh == "uniform" and back.status == rec["status"]
   and back.seed == rec["seed"] and np.array_equal(back.N, rec["N"])
   and np.array_equal(back.pass_no, rec["pass_no"]) and np.array_equal(back.accepted, rec["accepted"])
   and np.array_equal(back.converged, rec["converged"])
   and np.isnan(back.eta_max[1]) and np.allclose(back.eta, rec["eta"])
   and np.array_equal(back.s_knot0, rec["s_knot0"]) and int(back.opts.max_N) == 48
   and back.opts.states == "n,eps" and float(back.opts.tol) == 1e-2)
final1, log1, stop1 = run_refinement(base(), make_step()[0], fake_eval, refine_options(dict(passes=0), H0))
path_1 = os.path.join(tmp, "refine_record_1.mat")
sio.savemat(path_1, {"data": {"lap_time": 1.0, "refine": refine_record(log1, refine_options(dict(passes=0), H0),
                                                                        stop1, "uniform")}})
back1 = load_solution(path_1).refine
ok("a single-attempt record (passes=0) reloads with 1-D arrays of length 1, not scalars",
   all(np.ndim(getattr(back1, k)) == 1 and np.size(getattr(back1, k)) == 1 for k in RECORD_COLUMNS)
   and back1.N.tolist() == [12] and back1.eta.size == 12 and back1.s_knot0.size == 13)
cfg = "Static_ATDOn_EM4Off"
ok("result_stem(..., mesh_requested='adaptive') ends with '_meshAdaptive' (Results/ and Plots/ agree)",
   result_stem("Sturn", cfg, "MF205", "adaptive") == "Sturn_Static_ATDOn_EM4Off_meshAdaptive"
   and result_stem("Sturn", cfg, "CopyB", "adaptive").endswith("_CopyB_meshAdaptive"))

# =============================================================================
print("11. MLTP(refine=...) wiring (stand-in build_and_solve_nlp: no NLP is solved)")
if not os.path.exists(os.path.join(ROOT, "Data", "DATA_AA.mat")):   # pragma: no cover
    print("  [SKIP] Data/DATA_AA.mat missing")
    print("\nALL refine TESTS PASSED (MLTP wiring skipped)")
    sys.exit(0)

import MLTP as mltp_mod                                   # noqa: E402

ok("MLTP(refine=None) is the default", inspect.signature(mltp_mod.MLTP).parameters["refine"].default is None)

N7 = 9                                                    # a 7-state init held in memory
s7 = np.linspace(0.0, 1.0, N7 + 1)
X7 = np.vstack([np.linspace(30, 50, N7 + 1), np.zeros(N7 + 1), np.full(N7 + 1, 0.05),
                0.5 * np.sin(6 * s7), 0.02 * np.cos(4 * s7), np.full(N7 + 1, 100.0), np.full(N7 + 1, 100.0)])
U7 = np.vstack([np.full(N7 + 1, 300.0), np.linspace(0, -500, N7 + 1), 0.05 * np.sin(5 * s7)])
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=U7)}
_SIG = inspect.signature(build_and_solve_nlp)


def run_mltp(statuses=None, calls=None, perturb=0.0, vx_scale=None, **kw):
    """MLTP(...) with build_and_solve_nlp replaced by a stand-in that returns the packed
    guesses as the solution (zero duals) with the next status of ``statuses``. perturb > 0
    moves the 'solution' off the guesses: rows 5-22 (wheel, suspension, tyre) + perturb at
    the knots and collocation points, eps + perturb / 50 at the collocation points only (a
    non-constant eps polynomial in every interval). vx_scale = {solve index: factor}
    scales that solve's vx (a slower / faster lap). ``calls`` is appended to even when MLTP
    raises. Defaults: Sturn, warm_start = the in-memory 7-state init, mesh 'uniform', no
    save / plot (each overridable through kw)."""
    calls = [] if calls is None else calls
    statuses, vx_scale = list(statuses or []), dict(vx_scale or {})

    def stand_in(*args, **kwargs):
        a = _SIG.bind(*args, **kwargs).arguments
        m, disc, g, opts = a["m"], a["disc"], a["guesses"], a["opts"]
        st = nlp_structure(m.nx, m.nu, m.ny, disc["N"], a["OPT_d"], np.size(a["h_lb"]))
        x0, xc0 = np.array(g["x0"], dtype=float), np.array(g["xc0"], dtype=float)
        x0[5:] += perturb
        xc0[5:] += perturb
        xc0[4] += perturb / 50.0
        x0[0] *= vx_scale.get(len(calls), 1.0)
        xc0[0] *= vx_scale.get(len(calls), 1.0)
        w = np.concatenate([x0.reshape(-1, order="F"), np.asarray(g["u0"]).reshape(-1, order="F"),
                            xc0.reshape(-1, order="F")])
        status = statuses.pop(0) if statuses else "Solve_Succeeded"
        calls.append(dict(N=disc["N"], s_knot=np.array(disc["s_knot"]), max_iter=opts["ipopt"]["max_iter"],
                          warm=a.get("warm"), guesses=g, status=status, w=w, disc=disc))
        stats = {"iter_count": 7, "return_status": status}
        return dict(sol={"x": w}, solver=SimpleNamespace(stats=lambda: dict(stats)), w_opt=w,
                    lam_g=np.zeros(st["n_g"]), lam_x=np.zeros(st["n_w"]), structure=st,
                    warm_info=dict(x0=False, lam_g0=False, lam_x0=False, ipopt=False, duals=False),
                    sym_type="SX", linear_solver="ma57", N=disc["N"])

    real = mltp_mod.build_and_solve_nlp
    mltp_mod.build_and_solve_nlp = stand_in
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx = mltp_mod.MLTP(**dict(dict(circuit="Sturn", warm_start=INIT, save=False, plot=False,
                                            mesh="uniform"), **kw))
    finally:
        mltp_mod.build_and_solve_nlp = real
    return ctx, calls, buf.getvalue()


ctx_r, calls, out = run_mltp(refine=dict(passes=2, tol=1e-9), perturb=0.05)
dr, rr = ctx_r.data, ctx_r.data.refine
ok(f"refine=dict(passes=2, tol=1e-9): {len(calls)} solves with N {[c['N'] for c in calls]} = 18 -> 36 -> 72",
   [c["N"] for c in calls] == [18, 36, 72])
ok("each knot vector keeps every knot of the previous one (nested bisection)",
   all(np.all(np.isin(calls[i]["s_knot"], calls[i + 1]["s_knot"])) for i in range(2)))
ok("base solve with the configured max_iter (6000), passes 1-2 with min(6000, pass_max_iter) = 1000, "
   "all cold (no warm vectors)",
   [c["max_iter"] for c in calls] == [6000, 1000, 1000] and all(c["warm"] is None for c in calls))
g0, g1 = calls[0]["guesses"], calls[1]["guesses"]
ms = ctx_r.m23
x_s = np.asarray(ms.x_s)
unit_x, unit_u = np.ones(ms.nx), np.ones(ms.nu)


def sol_of(c):                        # (Xk, Uk, Xkj) scaled, of the stand-in's solution of call c
    x_, u_, _, xc_ = unpack_solution(c["w"], ms.nx, ms.nu, 0, c["N"], d, unit_x, unit_u, None)
    return x_, u_, xc_


ok("reseed: vx, vy, r, n, eps and the inputs carried over exactly at the kept knots",
   np.array_equal(g1["x0"][:5, ::2], g0["x0"][:5]) and np.array_equal(g1["u0"][:, ::2], g0["u0"]))
xb, ub, xcb = sol_of(calls[0])
Zb = interval_polynomials(calls[0]["disc"], xb, xcb)
s_new1, s_col1 = calls[1]["s_knot"], calls[1]["disc"]["s_col"]
ok("reseed at the new knots and collocation points: vx..eps = the previous state polynomials "
   "(eval_states), inputs = the previous NLP's in-interval arithmetic (nlp_inputs), not np.interp / a hold",
   np.allclose(g1["x0"][:5], eval_states(calls[0]["disc"], Zb, s_new1, x_end=xb[:, -1])[:5], rtol=0, atol=1e-13)
   and np.allclose(g1["xc0"][:5], eval_states(calls[0]["disc"], Zb, s_col1)[:5], rtol=0, atol=1e-13)
   and np.allclose(g1["u0"], nlp_inputs(calls[0]["disc"], ub, s_new1), rtol=0, atol=1e-13)
   and not np.allclose(g1["u0"][:, 1::2], 0.5 * (ub[:, :-1] + ub[:, 1:]), rtol=0, atol=1e-6)
   and not np.allclose(g1["xc0"][4], np.repeat(g1["x0"][4, :-1], d), rtol=0, atol=1e-6))
vx1 = g1["x0"][0] * x_s[0]
ok("reseed: wheel speeds, suspension and tyre deflections re-seeded quasi-statically from vx "
   "(the previous solution's rows 5-22 sit 0.05 off quasi-static)",
   np.allclose(g1["x0"][5] * x_s[5], vx1 / ctx_r.vp.Rw_f, rtol=1e-13)
   and np.allclose(g1["x0"][7] * x_s[7], vx1 / ctx_r.vp.Rw_r, rtol=1e-13)
   and np.all(g1["x0"][9:19] == 0.0) and np.allclose(g1["x0"][19] * x_s[19], ctx_r.vp.Wfl0 / ctx_r.vp.kt)
   and g1["xc0"].shape == (23, 36 * d) and np.all(g1["xc0"][9:19] == 0.0)
   and np.allclose(g1["xc0"][5] * x_s[5], g1["xc0"][0] * x_s[0] / ctx_r.vp.Rw_f, rtol=1e-13)
   and np.all(sol_of(calls[0])[0][9:19] == 0.05))
f_dyn_t = ca.Function("f_dyn_t", [ms.x, ms.u, ms.pv], [ms.dx, ms.sf])
f_sf_t = ca.Function("f_sf_t", [ms.x, ms.kappa], [ms.sf])


def eta_of(c):                        # independent indicator: rows n, eps of defect_errors
    x_, u_, xc_ = sol_of(c)
    return defect_errors(f_dyn_t, c["disc"], ctx_r.track, x_, xc_, u_)["E"][[3, 4]].max(axis=0)


eta_b, eta_f = eta_of(calls[0]), eta_of(calls[2])
laps = [float(compute_time(ca, f_sf_t, c["disc"], sol_of(c)[2], unit_x)[-1]) for c in calls]
ok(f"indicator: data.refine.eta_max {np.round(rr['eta_max'], 4)} / eta / eta_mean == the n, eps rows of "
   "defect_errors(f_dyn, scaled solution, scaled inputs, track curvature) recomputed here",
   np.allclose(rr["eta_max"], [eta_of(c).max() for c in calls], rtol=1e-12, atol=0)
   and np.allclose(rr["eta_mean"][0], eta_b.mean(), rtol=1e-12, atol=0)
   and np.allclose(rr["eta"], eta_f, rtol=1e-12, atol=0) and eta_b.size == 18)
ok(f"lap: data.refine.lap_time {np.round(rr['lap_time'], 4)} == compute_time of each pass's solution; "
   "the last == data.lap_time",
   np.allclose(rr["lap_time"], laps, rtol=1e-12, atol=0)
   and np.isclose(rr["lap_time"][-1], dr.lap_time, rtol=1e-12, atol=0) and laps[0] > 1.0)
ctx_y, calls_y, _ = run_mltp(refine=dict(passes=1, tol=1e-9, states=("r", "n")), perturb=0.05)
x_y, u_y, xc_y = sol_of(calls_y[0])
eta_y = defect_errors(f_dyn_t, calls_y[0]["disc"], ctx_y.track, x_y, xc_y, u_y)["E"][[2, 3]].max(axis=0)
ok("states=('r', 'n'): the indicator uses those rows (yaw rate via the 'r' alias), and the yaw-rate row "
   "depends on the inputs, which reach f_dyn scaled",
   np.isclose(ctx_y.data.refine["eta_max"][0], eta_y.max(), rtol=1e-12, atol=0)
   and not np.isclose(eta_y.max(), defect_errors(f_dyn_t, calls_y[0]["disc"], ctx_y.track, x_y, xc_y,
                                                  u_y * np.asarray(ms.u_s)[:, None])["E"][[2, 3]].max(),
                      rtol=1e-6, atol=0))
ok("ctx.mesh_requested == data.mesh == data.mesh_requested == 'adaptive' (ctx.mesh stays the base mesh)",
   ctx_r.mesh_requested == dr.mesh == dr.mesh_requested == "adaptive" and ctx_r.mesh == "uniform")
ok("the result is the last pass: data.N = 72, s_full on its knots, data.nlp from that pass ('refine')",
   dr.N == 72 and np.array_equal(dr.s_full[::d + 1], calls[2]["s_knot"]) and dr.nlp["warm_start"] == "refine"
   and dr.nlp["structure"]["N"] == 72 and dr.x_opt.shape == (23, 73))
ok("data['refine']: stop 'passes', arrays per attempt (pass_no, N, n_split, iters, converged, accepted), "
   "base mesh",
   rr["stop_reason"] == "passes" and rr["pass_no"].tolist() == [0, 1, 2] and rr["N"].tolist() == [18, 36, 72]
   and rr["n_split"].tolist() == [0, 18, 36] and rr["iters"].tolist() == [7, 7, 7]
   and rr["converged"].tolist() == [1, 1, 1] and rr["accepted"].tolist() == [1, 1, 1]
   and rr["base_mesh"] == "uniform" and rr["seed"] == "base, reseed, reseed"
   and np.array_equal(rr["s_knot0"], calls[0]["s_knot"]) and rr["eta"].size == 72
   and np.all(np.isfinite(rr["eta_max"])) and rr["opts"]["max_N"] == 72)
ok("ctx.refine_log has the same rows; elapsed: final / total iterations, passes, stop reason",
   [e["N"] for e in ctx_r.refine_log] == [18, 36, 72] and ctx_r.elapsed["ipopt_iters"] == 7
   and ctx_r.elapsed["ipopt_iters_total"] == 21 and ctx_r.elapsed["refine_passes"] == 2
   and ctx_r.elapsed["refine_stop"] == "passes" and ctx_r.elapsed["warm_start"] == "refine")
ok("the [MLTP] line reports the refinement chain (N, lap, eta_max, stop)",
   "refine: N 18->36->72, lap " + "->".join(f"{x:.3f}" for x in laps) + " s, eta_max" in out
   and "stop=passes" in out)
ok("the saved stem would be <circuit>_<cfg>_meshAdaptive",
   result_stem("Sturn", "Static_ATDOn_EM4Off", ctx_r.tyre_set, ctx_r.mesh_requested).endswith("_meshAdaptive"))

ctx_c, calls_c, out_c = run_mltp(["Solve_Succeeded", "Maximum_Iterations_Exceeded", "Solve_Succeeded"],
                                 refine=dict(passes=1, tol=1e-9))
g_init = mltp_mod.warmstart_guesses(ctx_c, ctx_c.m23, X7, U7, calls_c[2]["s_knot"])
ok("a reseeded pass that fails is retried once on the same mesh from the 7-state init guesses",
   [c["N"] for c in calls_c] == [18, 36, 36] and calls_c[2]["max_iter"] == 1000
   and all(np.array_equal(calls_c[2]["guesses"][k], g_init[k]) for k in ("x0", "u0", "xc0")))
ok("... accepted: data.nlp.warm_start 'init7', log rows base / reseed (failed) / cold, 'adaptive'",
   ctx_c.data.nlp["warm_start"] == "init7" and ctx_c.data.refine["seed"] == "base, reseed, cold"
   and ctx_c.data.refine["converged"].tolist() == [1, 0, 1] and ctx_c.data.N == 36
   and ctx_c.mesh_requested == "adaptive" and "retrying once from the 7-state init" in out_c)
ctx_f, calls_f, _ = run_mltp(["Solve_Succeeded", "Maximum_Iterations_Exceeded", "Maximum_Iterations_Exceeded"],
                             refine=dict(passes=1, tol=1e-9))
ok("reseed and retry both fail: 'solve-failed', the base solution is kept, the mesh request unchanged",
   len(calls_f) == 3 and ctx_f.data.refine["stop_reason"] == "solve-failed" and ctx_f.data.N == 18
   and ctx_f.mesh_requested == "uniform" and ctx_f.data.mesh == "uniform"
   and ctx_f.elapsed["refine_passes"] == 0 and ctx_f.data.refine["converged"].tolist() == [1, 0, 0])

ctx_l, calls_l, out_l = run_mltp(refine=dict(passes=1, tol=1e-9), vx_scale={1: 0.9})
rl = ctx_l.data.refine
ok(f"a reseeded pass that CONVERGES to a slower lap ({rl['lap_time'][1]:.3f} s vs {rl['lap_time'][0]:.3f} s) "
   "is treated as failed: cold retry on the same mesh, accepted ('init7'); the reseed row converged, not accepted",
   [c["N"] for c in calls_l] == [18, 36, 36] and rl["seed"] == "base, reseed, cold"
   and rl["converged"].tolist() == [1, 1, 1] and rl["accepted"].tolist() == [1, 0, 1]
   and rl["lap_time"][1] > 1.05 * rl["lap_time"][0] and ctx_l.data.nlp["warm_start"] == "init7"
   and ctx_l.data.N == 36 and ctx_l.mesh_requested == "adaptive" and "slower than the pass it refines" in out_l
   and "retrying once from the 7-state init" in out_l)
ctx_l2, calls_l2, out_l2 = run_mltp(refine=dict(passes=1, tol=1e-9), vx_scale={1: 0.9, 2: 0.9})
rl2 = ctx_l2.data.refine
ok("... and when the cold retry is slower too: stop 'lap-rise', the base kept (N 18, its lap, mesh request "
   "'uniform', warm start 'init7' of the base solve), no row after the base accepted",
   len(calls_l2) == 3 and rl2["stop_reason"] == "lap-rise" and ctx_l2.data.N == 18
   and np.isclose(ctx_l2.data.lap_time, rl2["lap_time"][0], rtol=1e-12, atol=0)
   and rl2["accepted"].tolist() == [1, 0, 0] and rl2["converged"].tolist() == [1, 1, 1]
   and ctx_l2.mesh_requested == "uniform" and ctx_l2.data.mesh == "uniform"
   and ctx_l2.elapsed["refine_passes"] == 0 and ctx_l2.data.nlp["warm_start"] == "init7"
   and "stop=lap-rise" in out_l2)
ctx_l3, calls_l3, _ = run_mltp(refine=dict(passes=1, tol=1e-9, max_lap_rise=np.inf), vx_scale={1: 0.9})
ctx_l4, calls_l4, _ = run_mltp(refine=dict(passes=1, tol=1e-9), vx_scale={1: 1.1})
ok("max_lap_rise=inf accepts the slower reseed; a FASTER reseed is always accepted (no retry)",
   len(calls_l3) == 2 and ctx_l3.data.refine["accepted"].tolist() == [1, 1] and ctx_l3.data.N == 36
   and len(calls_l4) == 2 and ctx_l4.data.refine["accepted"].tolist() == [1, 1]
   and ctx_l4.data.refine["lap_time"][1] < ctx_l4.data.refine["lap_time"][0])

ctx_b, calls_b, _ = run_mltp(["Infeasible_Problem_Detected"], refine=True)
ok("base solve not converged: one solve, 'base-failed', no refinement",
   len(calls_b) == 1 and ctx_b.data.refine["stop_reason"] == "base-failed" and ctx_b.data.N == 18)
ctx_n, calls_n, out_n = run_mltp(refine=None)
ok("refine=None: exactly one solve, no 'refine' key, no refine log, mesh_requested unchanged",
   len(calls_n) == 1 and calls_n[0]["max_iter"] == 6000 and "refine" not in vars(ctx_n.data)
   and not hasattr(ctx_n, "refine_log") and ctx_n.mesh_requested == "uniform"
   and ctx_n.data.mesh_requested == "uniform" and "ipopt_iters_total" not in ctx_n.elapsed
   and "refine:" not in out_n)

tmp_res = tempfile.mkdtemp()
ctx_s, calls_s, out_s = run_mltp(refine=dict(passes=1, tol=1e-9), save=True, results_dir=tmp_res)
path_s = os.path.join(tmp_res, "Sturn_Static_ATDOn_EM4Off_meshAdaptive.mat")
back_s = load_solution(path_s) if os.path.exists(path_s) else None
ok("save=True writes <stem>_meshAdaptive.mat; load_solution gives data.refine with 1-D per-attempt arrays",
   back_s is not None and back_s.mesh == back_s.mesh_requested == "adaptive" and int(back_s.N) == 36
   and back_s.refine.N.tolist() == [18, 36] and back_s.refine.pass_no.tolist() == [0, 1]
   and back_s.refine.accepted.tolist() == [1, 1] and back_s.refine.stop_reason == "passes"
   and np.isclose(back_s.refine.lap_time[-1], float(back_s.lap_time), rtol=1e-12, atol=0)
   and back_s.nlp.warm_start == "refine" and back_s.refine.eta.size == 36)

real_seed = mltp_mod.seed_m23


def slow_seed_m23(*a, **k):           # a cold start whose seed takes 0.4 s
    time.sleep(0.4)
    return real_seed(*a, **k)


mltp_mod.seed_m23 = slow_seed_m23
try:
    ctx_t, calls_t, _ = run_mltp(["Solve_Succeeded", "Maximum_Iterations_Exceeded", "Solve_Succeeded"],
                                 warm_start=None, ladder="qss23", refine=dict(passes=1, tol=1e-9))
finally:
    mltp_mod.seed_m23 = real_seed
el_t = ctx_t.elapsed
ok(f"time accounting: the cold retry's seed time goes to 'init' only (init {el_t['init']:.2f} s; solve_base "
   f"{el_t['solve_base']:.2f} + refine {el_t['refine']:.2f} = solve {el_t['solve']:.2f} s; cold attempt wall "
   f"{ctx_t.data.refine['wall'][2]:.2f} s)",
   len(calls_t) == 3 and el_t["init"] >= 0.75 and abs(el_t["solve_base"] + el_t["refine"] - el_t["solve"]) < 0.2
   and ctx_t.data.refine["wall"][2] < 0.35)

calls_v, called_i = [], []
real_i = mltp_mod.MLTP_initial


def init_stand_in(**k):               # a working 7-state init, so a late check would show as calls
    called_i.append(k)
    return SimpleNamespace(data=SimpleNamespace(init=SimpleNamespace(x_opt=X7, u_opt=U7, lap_time=19.5)),
                           elapsed=dict(ipopt_iters=1, return_status="Solve_Succeeded", qss=0.0))


mltp_mod.MLTP_initial = init_stand_in
try:
    bad_opts = raises(ValueError, run_mltp, calls=calls_v, refine=dict(tol=0.0)) \
        and raises(ValueError, run_mltp, calls=calls_v, refine=dict(bogus=1))
    bad_state = raises(ValueError, run_mltp, calls=calls_v, warm_start=None, refine=dict(states=("n", "bogus")))
    bad_hom = raises(ValueError, run_mltp, calls=calls_v, warm_start=None, homotopy=(1.2, 1.0),
                     refine=dict(states=("vz",)))
finally:
    mltp_mod.MLTP_initial = real_i
ok("a bad refine option or a state name the model does not have (also with homotopy) raises ValueError "
   "before any solve: no NLP solve, no 7-state init",
   bad_opts and bad_state and bad_hom and not calls_v and not called_i)
ok("state_rows: n / eps are rows 3 / 4 of the 23-state model ('r' aliases the yaw-rate symbol)",
   state_rows(ms, ("n", "eps")) == [3, 4] and state_rows(ms, ("vx", "r")) == [0, 2]
   and raises(ValueError, state_rows, ms, ("bogus",)))

print("\nALL refine TESTS PASSED")
