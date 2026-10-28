"""B4 userOpts(collocation='radau'): the Radau IIA collocation points.

Radau IIA ends each interval on the next knot. So s_full repeats every interior knot, while the
NLP keeps its Legendre size. Sections 1 to 6b need CasADi and no model. Sections 7 to 9 run MLTP
(section 8 also MLTP_paramOptim) on the stored Sturn point with the IPOPT solver replaced by a fake
(no solve). They need
Data/DATA_AA.mat and Results/Sturn_Static_ATDOn_EM4Off.mat. Without them the file prints SKIP
and exits 0. Run from any directory:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_radau.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

import functions.telemetry as P
import functions.transcription as T
from functions.collocation import collocation_coeff, collocation_points
from functions.context import Ctx
from functions.importfile import load_solution
from functions.transcription import discretise, reconstruct_track, time_at_points
from functions.warmstart import guesses_from_full, interp_rows
from userOpts import userOpts

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
MODEL = os.path.exists(DATA_AA) and os.path.exists(STURN)


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout and warnings swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


@contextlib.contextmanager
def fake_ipopt(rec=None, status="Fake"):
    """Replaces the IPOPT solver: x = x0 and zero duals. rec, if given, gets the NLP."""
    def fake(ca_, nlp, opts, info=None):
        if rec is not None:
            rec["nlp"] = nlp
        if info is not None:
            info["linear_solver"] = "fake"

        def solver(**k):
            if rec is not None:
                rec.update(lbx=np.ravel(k["lbx"]), ubx=np.ravel(k["ubx"]))
            return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                    "lam_x": np.zeros(nlp["x"].numel())}

        solver.stats = lambda: {"iter_count": 0, "return_status": status}
        return solver

    real, T._make_solver = T._make_solver, fake
    try:
        yield
    finally:
        T._make_solver = real


# ---- 1. Radau IIA points, D, B and stability (CasADi, d = 3) ------------------------------
print("1. Radau IIA points, coefficients and stability, d = 3")
r6 = np.sqrt(6.0)
tau = np.asarray(collocation_points(3, "radau"), float)
C, D, B = collocation_coeff(tau)
ok("points (4 - sqrt 6)/10, (4 + sqrt 6)/10 and 1, with the last point exactly 1",
   np.allclose(tau, [(4 - r6) / 10, (4 + r6) / 10, 1.0], rtol=0, atol=1e-15) and tau[-1] == 1.0)
ok("D = [0, 0, 0, 1] exactly: the continuity value is the last point",
   np.array_equal(D.ravel(), [0, 0, 0, 1.0]))
ok("B = the Radau weights (16 - sqrt 6)/36, (16 + sqrt 6)/36 and 1/9",
   np.allclose(B.ravel(), [(16 - r6) / 36, (16 + r6) / 36, 1 / 9], rtol=0, atol=1e-15))
nodes = np.r_[0.0, tau]
ok("C differentiates a cubic exactly at the points",
   np.allclose((nodes ** 3 + 2 * nodes) @ C, 3 * tau ** 2 + 2, rtol=0, atol=1e-12))


def amp(z, scheme):
    """Amplification of one collocation step of y' = lam y, with z = h lam."""
    t_ = np.asarray(collocation_points(3, scheme), float)
    C_, D_, _ = collocation_coeff(t_)
    Z = np.linalg.solve(C_[1:, :].T - z * np.eye(3), -C_[0, :])
    return D_[0, 0] + D_[1:, 0] @ Z


g3, r3 = amp(-1e3, "legendre"), amp(-1e3, "radau")
ok(f"stiff decay, h lam = -1e3: Gauss {g3:+.3f} keeps its size and flips the sign, "
   f"Radau {r3:+.4f} is damped", g3 < -0.9 and abs(r3) < 1e-2)

# ---- 2. discretise(scheme='radau'): exact duplicates at the interval ends -----------------
print("2. discretise(scheme='radau'): each end point is the next knot, exactly")
track = SimpleNamespace(s=np.linspace(0.0, 540.0, 270), k=np.zeros(270))
grid_l, grid_r = discretise(track, 30, 3), discretise(track, 30, 3, scheme="radau")
N = grid_l["N"]
same = discretise(track, 30, 3, scheme="legendre")
ok("default scheme is 'legendre': same output, bit for bit, as scheme='legendre'",
   all(np.array_equal(np.asarray(grid_l[k]), np.asarray(v)) for k, v in same.items() if v is not None))
ok("legendre: s_full strictly increasing", bool(np.all(np.diff(grid_l["s_full"]) > 0)))
ok(f"radau: s_full non-decreasing, exactly {N} zero steps (one per interval)",
   bool(np.all(np.diff(grid_r["s_full"]) >= 0)) and int(np.sum(np.diff(grid_r["s_full"]) == 0)) == N)
ok("radau: s_full[::4] are the knots, s_full[3::4] are the next knots, exactly",
   np.array_equal(grid_r["s_full"][::4], grid_r["s_knot"])
   and np.array_equal(grid_r["s_full"][3::4], grid_r["s_knot"][1:]))
rng = np.random.default_rng(1)
exact, plain_off = True, False
for trial in range(4):
    knots = np.sort(np.concatenate(([0.0, 540.0], rng.uniform(1, 539, 17 + trial))))
    g = discretise(track, 30, 3, s_knot=knots, scheme="radau")
    exact &= bool(np.array_equal(g["s_full"][3::4], knots[1:]))
    start = knots[0] + np.concatenate(([0.0], np.cumsum(np.diff(knots)[:-1])))
    plain_off |= bool(np.any(start + np.diff(knots) != knots[1:]))
ok("4 random knot vectors: every end point equals its next knot (a plain cumulative sum "
   "misses one by an ulp)", exact and plain_off)

# ---- 3. reconstruct_track on the Radau grid: finite, one answer per station ---------------
print("3. reconstruct_track on the Radau grid: finite, one answer per station")
syn = SimpleNamespace(s=track.s, k=0.004 * np.sin(2 * np.pi * track.s / 540.0))        # curved, no x, y
R = 100.0
circ = SimpleNamespace(s=track.s, k=np.full(270, 1 / R), x=R * np.sin(track.s / R),
                       y=R * (1 - np.cos(track.s / R)))
keys = ("x", "y", "xopt", "yopt", "Xl", "Xr")
for name, trk in (("synthetic (curv2cart)", syn), ("x, y given", circ)):
    g = discretise(trk, 30, 3, scheme="radau")
    s_f, n_f = g["s_full"], np.sin(g["s_full"] / 50.0)
    t = reconstruct_track(trk, s_f, g["k_full"], n_f, 8.0)
    u = np.unique(s_f)
    idx = np.searchsorted(u, s_f)
    tu = reconstruct_track(trk, u, np.interp(u, trk.s, trk.k), np.sin(u / 50.0), 8.0)
    ok(f"{name}: finite, full length, equal to the call on the {u.size} unique stations",
       np.array_equal(t["s"], s_f) and all(
           np.all(np.isfinite(t[k])) and np.shape(t[k])[0] == s_f.size and np.array_equal(t[k], tu[k][idx])
           for k in keys))

# ---- 4. guesses_from_full takes a Radau result as the source ------------------------------
print("4. guesses_from_full: a Radau result is a valid source, interp_rows stays strict")
g = discretise(track, 30, 3, scheme="radau")
f = lambda s: np.vstack([2 + 0.5 * s, 10 - 0.01 * s, 0.001 * s])          # linear in s: exact to interpolate
src = SimpleNamespace(x_opt=f(g["s_knot"]), x_full=f(g["s_full"]), s_full=g["s_full"], OPT_d=3,
                      u_opt=np.vstack([1 + 0.001 * g["s_knot"], 0.1 + 0 * g["s_knot"]]),
                      input_keys=["T_motor", "T_brake"])
new = discretise(track, 15, 3)
out = guesses_from_full(src, new["s_knot"], new["s_col"], ["T_motor", "T_brake"], np.ones(3), np.ones(2))
ok("x0 and xc0 equal the linear source on the new knots and points",
   np.allclose(out["x0"], f(new["s_knot"]), rtol=0, atol=1e-12)
   and np.allclose(out["xc0"], f(new["s_col"]), rtol=0, atol=1e-12))
try:
    interp_rows(g["s_full"], src.x_full, new["s_col"])
    refused = False
except ValueError:
    refused = True
ok("interp_rows still refuses a repeated station", refused)

# ---- 5. telemetry: the cubic reader takes the nodes of data.collocation -------------------
print("5. telemetry: the cubic reader takes the nodes of data.collocation")
D_, N_ = 3, 5
s_knot5 = np.concatenate(([0.0], np.cumsum([30, 25, 40, 30, 35.0])))
dsk5 = np.diff(s_knot5)
cub = lambda s: 1.0 + 0.3 * s - 2e-3 * s ** 2 + 1e-5 * s ** 3
sf = lambda s: 0.016 + 1e-4 * s + 2e-7 * s ** 2
Sf = lambda s: 0.016 * s + 0.5e-4 * s ** 2 + 2e-7 / 3 * s ** 3


def node_data(scheme, field):
    """A result on the nodes of scheme. field adds data.collocation, as MLTP saves a radau result."""
    t_ = np.asarray(collocation_points(D_, scheme), float)
    s_full = np.concatenate([s_knot5[k] + np.r_[0.0, t_] * dsk5[k] for k in range(N_)] + [[s_knot5[-1]]])
    if scheme == "radau":
        s_full[D_::D_ + 1][:N_] = s_knot5[1:]                # each end point is the next knot, exactly
    vf = {n: cub(s_full) for n in ("delta", "throttle", "om_gear", "xs_fl", "xs_fr", "xs_rl", "xs_rr")}
    vf["ax"] = vf["ay"] = np.zeros(s_full.size)
    t_full = time_at_points(dict(tau=t_, dsk=dsk5, s_knot=s_knot5, N=N_), sf(s_full), Sf(s_full)[::D_ + 1])
    data = SimpleNamespace(OPT_d=D_, s_full=s_full, t_full=t_full, vehicle_full=vf,
                           x_full=np.vstack([cub(s_full)] * 5))
    if field:
        data.collocation = scheme
    return data


data_r, data_l = node_data("radau", True), node_data("legendre", False)
ok("time_at_points: t equals the exact integral at every point, end points included",
   np.abs(data_r.t_full - Sf(data_r.s_full)).max() < 1e-12)
rs = P.resample(data_r, "distance", 1.0)
err = np.abs(rs["vx"] - cub(rs["s"])).max()
ok(f"data.collocation 'radau': vx is the cubic to 1e-9 (max error {err:.1e})", err < 1e-9)
rl = P.resample(data_l, "distance", 1.0)
err_l = np.abs(rl["vx"] - cub(rl["s"])).max()
ok(f"no field: the Legendre nodes are used, vx is the cubic to 1e-9 (max error {err_l:.1e})",
   err_l < 1e-9)

# ---- 6. userOpts(collocation=...) ---------------------------------------------------------
print("6. userOpts(collocation=...): values and errors")
c0, c1 = Ctx(), Ctx()
quiet(userOpts, c0, circuit="Sturn")
quiet(userOpts, c1, circuit="Sturn", collocation="radau")
ok("default 'legendre', 'radau' accepted", c0.collocation == "legendre" and c1.collocation == "radau")
for bad in ("Radau", "gauss", "chebyshev", None):
    named = False
    try:
        quiet(userOpts, Ctx(), circuit="Sturn", collocation=bad)
    except ValueError as exc:
        named = "legendre" in str(exc) and "radau" in str(exc)
    ok(f"collocation={bad!r} raises ValueError that names both schemes", named)


# ---- 6b. build_and_solve_nlp: path rows on a Radau grid need the linear input -------------
print("6b. build_and_solve_nlp: h_col_rows on a Radau grid needs linear_inputs=True")


def guard(scheme, lin):
    """True when the builder raises its ValueError before any work (the other arguments are stand-ins)."""
    try:
        T.build_and_solve_nlp(None, None, None, None, None, None, None,
                              dict(tau=np.asarray(collocation_points(3, scheme), float)), None, None, None, None,
                              None, None, 3, "linear", 0.1, {}, linear_inputs=lin, h_col_rows=[0])
    except ValueError as exc:
        return "linear_inputs" in str(exc)
    except Exception:                   # past the check: a stand-in argument fails
        return False
    return False


ok("radau + near hold raises, radau + linear and legendre + near hold pass the check",
   guard("radau", False) and not guard("radau", True) and not guard("legendre", False))

# ---- 7-9. MLTP on Sturn with the IPOPT solver faked ---------------------------------------
if MODEL:
    import casadi as ca
    import MLTP as mltp_mod
    from functions.sensitivity import nlp_index

    def run(rec=None, status="Fake", **kw):
        """MLTP on Sturn through the fake solver, warm-started from the stored point. Saves nothing."""
        kw = dict(dict(warm_start=STURN, save=False, plot=False), **kw)
        with fake_ipopt(rec, status):
            return quiet(mltp_mod.MLTP, circuit="Sturn", **kw)

    print("7. MLTP on Sturn, IPOPT faked: sizes, repeated stations, end-point inputs")
    rl_rec, rr_rec = {}, {}
    leg = run(rl_rec, collocation="legendre")
    rad = run(rr_rec, collocation="radau")
    dl, dr = leg.data, rad.data
    D, Nn = int(dr.OPT_d), int(dr.N)
    ok("legendre run: no 'collocation' field in the result (the default result does not change)",
       not hasattr(dl, "collocation"))
    ok("radau run: data.collocation == 'radau'", dr.collocation == "radau")
    st_l, st_r = dl.nlp["structure"], dr.nlp["structure"]
    ok(f"n_w {st_r['n_w']} and n_g {st_r['n_g']} equal the Legendre NLP "
       "(the end point is one variable and one row)",
       (st_l["n_w"], st_l["n_g"]) == (st_r["n_w"], st_r["n_g"]))
    ok("warm start: Legendre re-injects duals from the stored point. Radau differs, so it is primal only",
       leg.elapsed["warm_start"] == "full+duals" and rad.elapsed["warm_start"] == "full-interp")
    ok(f"radau s_full: {Nn} zero steps, none negative, knots strictly increasing",
       int(np.sum(np.diff(dr.s_full) == 0)) == Nn and bool(np.all(np.diff(dr.s_full) >= 0))
       and bool(np.all(np.diff(dr.s_full[::D + 1]) > 0)))
    ok("track x, y, xopt, yopt, Xl, Xr: no NaN",
       all(np.all(np.isfinite(np.asarray(dr.track[k], float))) for k in ("x", "y", "xopt", "yopt", "Xl", "Xr")))
    ok("t_full at each end point equals t_opt at the next knot (1e-12 s)",
       np.abs(np.asarray(dr.t_full)[D::D + 1][:Nn] - np.asarray(dr.t_opt)[1:]).max() < 1e-12)
    u_opt = np.asarray(dr.u_opt, float)
    row = [i for i, k in enumerate(rad.input_keys) if str(k).strip() == "T_motor"][0]
    u_s = float(rad.m23.u_s[row])
    duk = (u_opt[row, 1:] - u_opt[row, :-1]) / np.diff(dr.s_full[::D + 1])
    thr_end = np.asarray(dr.vehicle_full["throttle"], float)[D::D + 1][:Nn]
    ok("vehicle_full throttle at each end point is the input the NLP used there, (U_k + duk) / u_s",
       np.allclose(thr_end, (u_opt[row, :-1] + duk) / u_s, rtol=0, atol=1e-12))
    ok("... and not the next knot input U_(k+1) / u_s (the input jumps in this data)",
       np.abs(thr_end - u_opt[row, 1:] / u_s).max() > 0.1)
    ok("telemetry.resample on the radau result: finite channels",
       all(np.all(np.isfinite(v)) for v in P.resample(dr, "distance", 1.0).values()))

    def cont_nnz(rec, data):
        """Nonzeros per continuity row of g, from the NLP that the solve call received."""
        w, g_, f_ = rec["nlp"]["x"], rec["nlp"]["g"], rec["nlp"]["f"]
        rows = np.array(ca.Function("fg", [w], [f_, g_]).sparsity_jac(0, 1).row(), dtype=int)
        cont = nlp_index(data.nlp["structure"])["cont"].ravel()
        return np.unique(np.bincount(rows, minlength=g_.numel())[cont])

    ok("continuity rows: 5 nonzeros for Legendre, 2 for Radau (X_end - X_next)",
       cont_nnz(rl_rec, dl).tolist() == [5] and cont_nnz(rr_rec, dr).tolist() == [2])
    m23 = leg.m23
    box = {b: np.concatenate([np.tile(np.ravel(getattr(m23, f"x_{b}")), Nn + 1),
                              np.tile(np.ravel(getattr(m23, f"u_{b}")), Nn + 1),
                              np.tile(np.ravel(getattr(m23, f"x_{b}")), Nn * D)]) for b in ("min", "max")}
    end = np.zeros(box["min"].size, bool)           # the last point of each interval (on Radau, the next knot)
    end[((m23.nx + m23.nu) * (Nn + 1) + m23.nx * np.arange(D - 1, Nn * D, D)[:, None] + np.arange(m23.nx)).ravel()] = True
    band_rec = {}
    run(band_rec, collocation="radau", n_bounds=3.5)
    ok("lbx / ubx: Legendre = the model box everywhere. Radau = the same, but -inf / +inf at the last point of each "
       "interval (the copy of the next knot), also with an n band",
       np.array_equal(rl_rec["lbx"], box["min"]) and np.array_equal(rl_rec["ubx"], box["max"])
       and np.array_equal(rr_rec["lbx"][~end], box["min"][~end]) and np.array_equal(rr_rec["ubx"][~end], box["max"][~end])
       and np.all(rr_rec["lbx"][end] == -np.inf) and np.all(rr_rec["ubx"][end] == np.inf)
       and np.all(band_rec["lbx"][end] == -np.inf) and np.all(band_rec["ubx"][end] == np.inf))
    with tempfile.TemporaryDirectory() as tmp:
        f_mat = os.path.join(tmp, "sturn_radau.mat")
        sio.savemat(f_mat, {"data": dict(vars(dr))}, do_compression=True)
        ok(".mat round trip: data.collocation survives", str(load_solution(f_mat).collocation) == "radau")
    rlp_rec = {}
    rlp = run(rlp_rec, collocation="radau", between_knots="linear+power")
    llp = run(collocation="legendre", between_knots="linear+power")
    st_rp, st_lp, d_rp = rlp.data.nlp["structure"], llp.data.nlp["structure"], rlp.data
    ok("radau + linear+power: one power row fewer per interval in n_pathcol and in g "
       "(the tau = 1 row repeats the next knot row)",
       st_rp["n_pathcol"] == st_lp["n_pathcol"] - Nn and st_rp["n_g"] == st_lp["n_g"] - Nn)
    m_rp, nx, nu = rlp.m23, rlp.m23.nx, rlp.m23.nu
    w = np.asarray(d_rp.nlp["w_opt"], float).ravel()
    w = w * (1 + 0.05 * np.random.default_rng(3).standard_normal(w.size))   # the rows differ from point to point
    g_rp = np.asarray(ca.Function("g", [rlp_rec["nlp"]["x"]], [rlp_rec["nlp"]["g"]])(w)).ravel()
    Uk = w[nx * (Nn + 1):(nx + nu) * (Nn + 1)].reshape(nu, Nn + 1, order="F")
    Xkj = w[(nx + nu) * (Nn + 1):].reshape(nx, Nn * D, order="F")
    tau_r = np.asarray(collocation_points(D, "radau"), float)
    rows_exp = []
    for k in range(Nn):
        for j in range(D - 1):
            T_ = (Uk[row, k] + (Uk[row, k + 1] - Uk[row, k]) * tau_r[j]) * m_rp.u_s[row]
            Om = rlp.vp.gear * np.mean(Xkj[5:9, D * k + j] * np.asarray(m_rp.x_s)[5:9])
            rows_exp.append((rlp.pt.Pmax - T_ * Om) / rlp.pt.Pmax)
    ok("the power rows are (Pmax - T Om_motor) / Pmax at the points j < OPT_d - 1, with the linear input (1e-12)",
       np.abs(g_rp[-st_rp["n_pathcol"]:] - np.array(rows_exp)).max() < 1e-12)
    u_rp = np.asarray(d_rp.u_opt, float)
    thr_rp = np.asarray(d_rp.vehicle_full["throttle"], float)[D::D + 1][:Nn]
    ok("linear+power: vehicle_full throttle at each end point is (U_k + (U_(k+1) - U_k)) / u_s, exactly",
       np.array_equal(thr_rp, (u_rp[row, :-1] + (u_rp[row, 1:] - u_rp[row, :-1])) / u_s))

    print("8. warm starts with a Radau result")
    r_same = run(collocation="radau", warm_start=rad)
    r_leg = run(collocation="legendre", warm_start=rad)
    ok("radau -> radau on the same grid: primal and duals re-injected (full+duals)",
       r_same.elapsed["warm_start"] == "full+duals")
    ok("radau -> legendre: interpolated by s, primal only (full-interp), no error",
       r_leg.elapsed["warm_start"] == "full-interp")
    import MLTP_paramOptim as po_mod
    with fake_ipopt():
        des = quiet(po_mod.MLTP_paramOptim, circuit="Sturn", warm_start=rad, collocation="radau", save=False)
    ok("MLTP_paramOptim radau from the radau result: full+duals, N zero steps, data.collocation 'radau'",
       des.elapsed["warm_start"] == "full+duals" and int(np.sum(np.diff(des.data.s_full) == 0)) == int(des.data.N)
       and getattr(des.data, "collocation", None) == "radau")

    print("9. MLTP(refine=...) keeps the scheme on every pass")
    rf = run(status="Solve_Succeeded", collocation="radau", mesh="uniform",
             refine=dict(passes=1, tol=1e-9, max_lap_rise=np.inf))
    d_rf = rf.data
    ok(f"refined radau result: N = {int(d_rf.N)} (> {Nn}), one zero step per interval, collocation 'radau'",
       int(d_rf.N) > Nn and int(np.sum(np.diff(d_rf.s_full) == 0)) == int(d_rf.N)
       and d_rf.collocation == "radau")
else:
    print("SKIP sections 7-9 (casadi model): Data/DATA_AA.mat or the stored Sturn result is missing")

print("\nALL radau TESTS PASSED" if MODEL else "\nnumpy radau tests PASSED (model sections skipped)")
