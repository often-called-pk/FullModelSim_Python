"""E1 channels at the collocation points, the ChassisSim monster file and the 1 m CSV
(plan docs/superpowers/plans/2026-10-20-finish-partials.md, Task 6).

Sections 1-3 are numpy only: time_at_points (functions/transcription.py) and the resampled channels
and exports (functions/telemetry.py) on a synthetic result. Sections 4-6 need casadi and
Data/DATA_AA.mat. They run MLTP on the stored Sturn point with IPOPT faked (no solve) and check the
wiring of data.vehicle_full and data.t_full, then an EM4 On and a physics='corrected' stub, then a
.mat round trip. Without DATA_AA they print SKIP and the run still exits 0. Run from any directory:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_telemetry.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import inspect
import io
import os
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

import functions.telemetry as P
from functions.collocation import collocation_points
from functions.importfile import load_solution
from functions.refine import interval_polynomials, lagrange_basis
from functions.transcription import time_at_points

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
INIT7 = SimpleNamespace(x_opt=np.vstack([np.linspace(30, 50, 10), np.zeros((4, 10)), np.full((2, 10), 100.0)]),
                        u_opt=np.vstack([np.full(10, 300.0), np.zeros((2, 10))]))   # in-memory 7-state init


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def channels(veh):
    """data.vehicle or data.vehicle_full as a dict: in memory a dict, reloaded from a .mat a namespace."""
    return veh if isinstance(veh, dict) else vars(veh)


def read(path):
    with open(path) as fh:
        return fh.read()


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout and warnings swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


@contextlib.contextmanager
def fake_ipopt():
    """transcription._make_solver -> a solver that returns x = x0 and zero duals (no IPOPT)."""
    import functions.transcription as T

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


def run_stub(**kw):
    """MLTP(**kw) on Sturn (uniform mesh, save and plot off) with MLTP_initial -> INIT7 and
    build_and_solve_nlp -> the packed guesses as the solution (zero duals, no IPOPT)."""
    import MLTP as mltp_mod
    import functions.transcription as T
    from functions.warmstart import nlp_structure
    sig = inspect.signature(T.build_and_solve_nlp)

    def init_stub(**k):
        return SimpleNamespace(data=SimpleNamespace(init=INIT7),
                               elapsed=dict(ipopt_iters=1, return_status="Solve_Succeeded", qss=0.0))

    def nlp_stub(*a, **k):
        b = sig.bind(*a, **k).arguments
        m, disc, g = b["m"], b["disc"], b["guesses"]
        st = nlp_structure(m.nx, m.nu, m.ny, disc["N"], b["OPT_d"], np.size(b["h_lb"]))
        w = np.concatenate([g["x0"].reshape(-1, order="F"), g["u0"].reshape(-1, order="F"),
                            np.asarray(g["xc0"]).reshape(-1, order="F")])
        stats = {"iter_count": 7, "return_status": "Solve_Succeeded"}
        return dict(sol={"x": w, "f": 0.0}, solver=SimpleNamespace(stats=lambda: dict(stats)), w_opt=w,
                    lam_g=np.zeros(st["n_g"]), lam_x=np.zeros(st["n_w"]), structure=st,
                    warm_info=dict(x0=False, lam_g0=False, lam_x0=False, ipopt=False, duals=False),
                    sym_type="SX", linear_solver="fake", N=disc["N"])

    real = (mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp)
    mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp = init_stub, nlp_stub
    try:
        return quiet(mltp_mod.MLTP, circuit="Sturn", save=False, plot=False, mesh="uniform", **kw)
    finally:
        mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp = real


# =============================================================================================
print("1. time_at_points: exact for sf of degree <= d-1 inside each interval (d = 3, N = 5)")
D, N = 3, 5
tau = np.asarray(collocation_points(D, "legendre"), float)
s_knot = np.concatenate(([0.0], np.cumsum([30, 25, 40, 30, 35.0])))
dsk = np.diff(s_knot)
s_full = np.concatenate([s_knot[k] + np.r_[0.0, tau] * dsk[k] for k in range(N)] + [[s_knot[-1]]])
disc = dict(tau=tau, dsk=dsk, s_knot=s_knot, N=N)
sf = lambda s: 0.016 + 1e-4 * s + 2e-7 * s ** 2
Sf = lambda s: 0.016 * s + 0.5e-4 * s ** 2 + 2e-7 / 3 * s ** 3
t_exact = Sf(s_full)
t_full = time_at_points(disc, sf(s_full), t_exact[::D + 1])
err_t = np.abs(t_full - t_exact).max()
ok(f"max |t_full - exact| = {err_t:.1e} s <= 1e-12", err_t <= 1e-12)
ok("t_full at the knots equals t_opt bit for bit", np.array_equal(t_full[::D + 1], t_exact[::D + 1]))

# =============================================================================================
print("2. resample: cubic states exact, input step kept at a knot, ax / ay inside the node range")
cub = lambda s: 1.0 + 0.3 * s - 2e-3 * s ** 2 + 1e-5 * s ** 3
vf = {n: cub(s_full) for n in ("delta", "throttle", "om_gear", "xs_fl", "xs_fr", "xs_rl", "xs_rr")}
vf["delta"] = np.where(np.arange(s_full.size) < 2 * (D + 1), 1.0, 2.0)     # step at knot 2 (55 m)
vf["throttle"] = 0.5 - s_full / 80.0                                         # < 0 after 40 m (regen)
wig = np.where(np.arange(s_full.size) % 2 == 0, -1.0, 1.0)                 # node pattern a cubic overshoots
vf["ax"], vf["ay"] = 5 * wig, 3 * wig
data = SimpleNamespace(OPT_d=D, s_full=s_full, t_full=t_full, vehicle_full=vf,
                       x_full=np.vstack([cub(s_full)] * 5))
rs = P.resample(data, "distance", 1.0)
ok(f"1 m stations: {rs['s'].size} rows from 0 to {s_full[-1]:g} m",
   rs["s"].size == 161 and rs["s"][0] == 0.0 and rs["s"][-1] == s_full[-1])
ok("vx is the cubic to 1e-9", np.abs(rs["vx"] - cub(rs["s"])).max() < 1e-9)
ok("delta: 1.0 before the knot at 55 m, 2.0 from it (no smearing)",
   np.allclose(rs["delta"][rs["s"] < s_knot[2] - 1e-9], 1.0, rtol=0, atol=1e-12)
   and np.allclose(rs["delta"][rs["s"] >= s_knot[2]], 2.0, rtol=0, atol=1e-12))
ok("ax, ay stay inside the node range (no overshoot)",
   np.abs(rs["ax"]).max() <= 5 + 1e-12 and np.abs(rs["ay"]).max() <= 3 + 1e-12)
ok("t is monotone and ends at t_full[-1]",
   bool(np.all(np.diff(rs["t"]) > 0)) and abs(rs["t"][-1] - t_full[-1]) < 1e-12)

# =============================================================================================
print("3. 50 Hz stations, ChassisSim monster file and the 1 m CSV")
rt = P.resample(data, "time", 0.02)
ok(f"50 Hz: {rt['t'].size} rows = floor(T / 0.02) + 1", rt["t"].size == int(t_full[-1] / 0.02) + 1)
ok("50 Hz: t is the 0.02 s grid to 1e-6 s", np.abs(rt["t"] - 0.02 * np.arange(rt["t"].size)).max() < 1e-6)
with tempfile.TemporaryDirectory() as tmp:
    mon, mon_pos, csv_path = (os.path.join(tmp, f) for f in ("monster.txt", "monster_pos.txt", "csv.csv"))
    P.export_monster(data, mon)
    P.export_monster(data, mon_pos, sign=(1.0, 1.0))
    P.export_csv(data, csv_path, 1.0)
    text = read(mon)
    lines = text.splitlines()
    back = np.genfromtxt(mon, delimiter="\t")
    pos = np.genfromtxt(mon_pos, delimiter="\t")
    hdr = read(csv_path).splitlines()[0].split(",")
    csv = np.loadtxt(csv_path, delimiter=",", skiprows=1)
ok(f"monster: no header, 11 tab-separated columns per row, distance starts at 0 ({back.shape[0]} rows)",
   not any(ch.isalpha() for ch in text) and all(x.count("\t") == 10 for x in lines)
   and back.shape == (rt["t"].size, 11) and back[0, 0] == 0.0)
ok("monster: distance column = integral of the speed column (0.01 m)",
   abs(back[-1, 0] - np.trapezoid(back[:, 10] / 3.6, dx=0.02)) < 0.01)
ok("monster: a sign flip changes only ay (column 2) and steer (column 8)",
   [i for i in range(11) if not np.allclose(back[:, i], pos[:, i])] == [2, 8])
ok("monster: throttle column = 100 * max(throttle, 0) (a regen torque < 0 gives 0 %)",
   rt["throttle"].min() < 0 < rt["throttle"].max()
   and np.abs(back[:, 9] - 100 * np.maximum(rt["throttle"], 0)).max() <= 0.05 + 1e-9)
names = ["s", "t", "vx", "vy", "r", "n", "eps", "delta", "throttle", "om_gear",
         "xs_fl", "xs_fr", "xs_rl", "xs_rr", "ax", "ay"]
ok("csv: header row = the 16 channel names, one row per 1 m station",
   hdr == names and csv.shape == (rs["s"].size, 16))

# =============================================================================================
if os.path.exists(DATA_AA):
    import MLTP as mltp_mod

    print("4. wiring on the stored Sturn point (fake IPOPT: x = stored w_opt)")
    with fake_ipopt():
        c = quiet(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False)
    d = c.data
    D = int(d.OPT_d)
    n_pts = np.size(d.s_full)
    veh, full = channels(d.vehicle), channels(d.vehicle_full)
    new = {"om_gear", "throttle", "xs_fl", "xs_fr", "xs_rl", "xs_rr", "sf"}
    ok("vehicle_full holds the 7 new channels and every knot channel of vehicle except E_motor",
       new <= set(full) and set(veh) - {"E_motor"} <= set(full))
    ok(f"vehicle_full: every channel has the {n_pts} full-grid points",
       all(np.size(a) == n_pts for a in full.values()))
    ok("vehicle[k] == vehicle_full[k][::d+1] bitwise for every knot channel",
       all(np.asarray(veh[k]).tobytes() == np.asarray(full[k])[::D + 1].tobytes()
           for k in veh if k != "E_motor"))
    ok("t_full[::d+1] == t_opt exactly and t_full[-1] == lap_time",
       np.array_equal(d.t_full[::D + 1], d.t_opt) and d.t_full[-1] == d.lap_time)

    # state-polynomial kinematics at the collocation points, from the NLP's own states and sf
    s_m = np.asarray(d.s_full, float)
    knot = np.arange(0, n_pts, D + 1)                     # knot indices on the full grid
    cn = np.setdiff1d(np.arange(n_pts), knot)             # collocation indices, interval by interval
    s_knot_m = s_m[knot]
    Z = interval_polynomials(dict(s_knot=s_knot_m, dsk=np.diff(s_knot_m), tau=tau), d.x_opt, d.xc_opt)
    _, dL = lagrange_basis(np.r_[0.0, tau], tau)          # dL[j, r] = l_j'(tau_r) on the nodes [0, tau]
    Zk = np.transpose(Z, (0, 2, 1))                       # (nx, N, d + 1): node values of each interval
    nx = Zk.shape[0]
    X = Zk[:, :, 1:].reshape(nx, -1)                      # states at the collocation points
    dX = ((Zk @ dL) / np.diff(s_knot_m)[None, :, None]).reshape(nx, -1)   # d/ds at the same points
    ok("reference nodes equal the stored collocation states",
       np.allclose(X, np.asarray(d.x_full)[:, cn], rtol=1e-12, atol=1e-12))
    vx, vy, r, n_lat, eps = X[:5]
    kap = np.asarray(d.k_full, float)[cn]
    sf_ref = (1 - n_lat * kap) / (vx * np.cos(eps) - vy * np.sin(eps))
    ax_ref, ay_ref = dX[0] / sf_ref - vy * r, dX[1] / sf_ref + vx * r
    err_ax = np.abs(np.asarray(full["ax"], float)[cn] - ax_ref).max()
    err_ay = np.abs(np.asarray(full["ay"], float)[cn] - ay_ref).max()
    rms_ax = np.sqrt(np.mean(ax_ref ** 2))
    ok(f"stored point is not trivial: rms ax at the nodes {rms_ax:.2f} m/s^2 > 1", rms_ax > 1)
    ok(f"ax at the collocation nodes = state-polynomial kinematics (max {err_ax:.1e} <= 1e-4 m/s^2)",
       err_ax <= 1e-4)
    ok(f"ay at the collocation nodes = state-polynomial kinematics (max {err_ay:.1e} <= 1e-5 m/s^2)",
       err_ay <= 1e-5)

    print("5. EM4 On (physics matlab) and physics='corrected' stubs (no solve)")
    c4 = run_stub(Electric_4Motors="On", ATD="Off")
    f4 = channels(c4.data.vehicle_full)
    om_w = np.mean([f4[f"Om_{w}"] for w in ("fl", "fr", "rl", "rr")], axis=0)
    ok("EM4 On: om_gear = mean wheel speed times gear (motor speed, not Om_motor_*)",
       np.allclose(f4["om_gear"], om_w * c4.vp.gear, rtol=1e-12, atol=1e-9))
    cc = run_stub(physics="corrected")
    fc = channels(cc.data.vehicle_full)
    ok("physics corrected: the new channels are finite and t_full[-1] == lap_time",
       all(np.all(np.isfinite(fc[k])) for k in new | {"ax", "ay"}) and cc.data.t_full[-1] == cc.data.lap_time)

    print("6. .mat round trip: the exports of the reloaded result equal the in-memory exports")
    with tempfile.TemporaryDirectory() as tmp:
        f_mat = os.path.join(tmp, "sturn_e1.mat")
        sio.savemat(f_mat, {"data": dict(vars(d))}, do_compression=True)
        reloaded = load_solution(f_mat)
        out = {}
        for tag, src in (("mem", d), ("mat", reloaded)):
            P.export_monster(src, os.path.join(tmp, f"{tag}.txt"))
            P.export_csv(src, os.path.join(tmp, f"{tag}.csv"), 1.0)
            out[tag] = read(os.path.join(tmp, f"{tag}.txt")) + read(os.path.join(tmp, f"{tag}.csv"))
    ok("exports of the reloaded result equal the in-memory exports", out["mem"] == out["mat"])
else:
    print("SKIP sections 4-6 (casadi): Data/DATA_AA.mat missing")

print("\nALL telemetry TESTS PASSED" if os.path.exists(DATA_AA)
      else "\nnumpy telemetry tests PASSED (casadi sections skipped)")
