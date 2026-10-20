"""E1 output channels and the energy unit (plan docs/superpowers/plans/2026-10-18-phase2.md, Task 3).

m.ax / m.ay are the road-plane CG accelerations (outside dx). data.vehicle gains ax, ay, delta and
Om_fl..Om_rr. E_motor stays in Wh, P_motor in W. plotSDI.accel_channels returns (ax, ay) from a
result's vehicle channels, derives them from Lon_acc / Lat_acc and x_opt for a result saved before
E1, or returns (None, None).

Sections 1-3 need casadi and Data/DATA_AA.mat. They solve nothing: IPOPT is faked, section 2 reads
the stored Sturn point and section 3 runs a stand-in. Sections 4-5 are numpy only; section 5 reads
Results/BCN_Static_ATDOn_EM4Off.mat. Without DATA_AA the casadi sections print SKIP and the run
still exits 0. Run from any directory:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_output_channels.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import inspect
import io
import os
import warnings
from types import SimpleNamespace

import numpy as np

import plotSDI
from functions.importfile import load_solution
from gg_plots import vehicle_gg_from_solution

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
BCN = os.path.join(_bootstrap.ROOT, "Results", "BCN_Static_ATDOn_EM4Off.mat")
INIT7 = SimpleNamespace(x_opt=np.vstack([np.linspace(30, 50, 10), np.zeros((4, 10)), np.full((2, 10), 100.0)]),
                        u_opt=np.vstack([np.full(10, 300.0), np.zeros((2, 10))]))   # in-memory 7-state init


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout and warnings swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def channels(veh):
    """data.vehicle as a dict: an in-memory result holds a dict, a loaded .mat a namespace."""
    return veh if isinstance(veh, dict) else vars(veh)


def scaled_err(res, ref):
    """|res| / max(1, |ref|): the plan's tolerance form |res| <= tol * max(1, |ref|)."""
    return abs(res) / max(1.0, abs(ref))


def energy_wh(P, t):
    """Energy [Wh] of a power trace P [W] over t [s] at the 0.9 motor efficiency (MLTP.py E_motor)."""
    return np.sum(0.5 * (P[1:] + P[:-1]) * np.diff(t)) / 3600 / 0.9


# ---- casadi helpers (sections 1-3 only) ----------------------------------------------------------
def model(physics):
    """Sturn, Static, ATD On, 4 Motors Off: userOpts + vehModel, no solve."""
    from functions.context import Ctx
    from userOpts import userOpts
    from vehModel import vehModel
    c = Ctx()
    quiet(userOpts, c, circuit="Sturn", physics=physics)
    vehModel(c)
    return c


def sym_time(c, names):
    """Named signals of c.m23 as functions of a physical state and input (Phase 1 recipe).
    Returns (Xp, Up, K, dxdt, signals)."""
    import casadi as ca
    m = c.m23
    xs, us = ca.DM(np.asarray(m.x_s, float)), ca.DM(np.asarray(m.u_s, float))
    F = ca.Function("F", [m.x, m.u, m.pv], [m.dx, m.sf] + [getattr(m, n) for n in names])
    Xp, Up, K = ca.SX.sym("Xp", m.nx), ca.SX.sym("Up", m.nu), ca.SX.sym("K")
    o = F(Xp / xs, Up / us, K)
    return Xp, Up, K, o[0] * xs / o[1], list(o[2:])


def static_state(c, vx):
    """Wheel speeds on the rolling radius, tyre deflections at rest, the rest zero (Phase 1 recipe)."""
    vp = c.vp
    X = np.zeros(c.m23.nx)
    X[0] = vx
    X[5:9] = vx / vp.Rw
    X[19:23] = [vp.xti_fl, vp.xti_fr, vp.xti_rl, vp.xti_rr]
    return X


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


if os.path.exists(DATA_AA):
    import casadi as ca
    import MLTP as mltp_mod

    # =========================================================================================
    print("1. ax, ay identities on the symbolic model (5 random states per physics, seed 0)")
    rng = np.random.default_rng(0)
    NAMES = (["ax", "ay", "Lon_acc", "Lat_acc", "vx", "vy", "r", "delta", "f_drag", "f_roll", "f_side",
              "theta", "phi"] + [f"f{d}_{w}" for d in ("x", "y") for w in ("fl", "fr", "rl", "rr")])
    for phys in ("matlab", "corrected"):
        c = model(phys)
        m, vp = c.m23, c.vp
        Xp, Up, K, _, sigs = sym_time(c, NAMES)
        G = ca.Function("G", [Xp, Up, K], sigs)
        worst = dict(ax=0.0, ay=0.0, Fx=0.0, Fy=0.0)
        for _ in range(5):
            X = static_state(c, rng.uniform(20, 60))
            X[1] = rng.uniform(-0.5, 0.5)
            X[2] = rng.uniform(-0.3, 0.3)
            X[10] = rng.uniform(-0.1, 0.1)
            X[11] = rng.uniform(-0.01, 0.01)
            X[12] = rng.uniform(-0.1, 0.1)
            X[13] = rng.uniform(-0.01, 0.01)
            X[5:9] *= 1 + rng.uniform(-0.05, 0.05, 4)
            U = np.zeros(m.nu)
            U[0] = rng.uniform(0, 300)
            U[2:6] = 0.25
            U[-1] = rng.uniform(-0.05, 0.05)
            d = dict(zip(NAMES, [float(v) for v in G(X, U, 0.0)]))
            cd, sd = np.cos(d["delta"]), np.sin(d["delta"])
            sum_fx = (d["fx_fl"] + d["fx_fr"]) * cd - (d["fy_fl"] + d["fy_fr"]) * sd + d["fx_rl"] + d["fx_rr"]
            sum_fy = (d["fy_fl"] + d["fy_fr"]) * cd + (d["fx_fl"] + d["fx_fr"]) * sd + d["fy_rl"] + d["fy_rr"]
            ax, ay = d["ax"], d["ay"]
            worst["ax"] = max(worst["ax"], scaled_err(ax - (d["Lon_acc"] - d["vy"] * d["r"]), ax))
            worst["ay"] = max(worst["ay"], scaled_err(ay - (d["Lat_acc"] + d["vx"] * d["r"]), ay))
            if phys == "corrected":
                fx_err = ax * vp.m - (sum_fx - d["f_drag"] - d["f_roll"])
                fy_err = ay * vp.m - (sum_fy + d["f_side"])
                fx_ref, fy_ref = ax * vp.m, ay * vp.m
            else:
                ms, g = vp.ms, vp.g
                fx_err = ax * ms - (sum_fx - d["f_drag"] - d["f_roll"] - ms * X[12] * X[10] + ms * g * np.sin(X[11]))
                fy_err = ay * ms - (sum_fy + d["f_side"] - ms * g * np.sin(X[13]) * np.cos(X[11]))
                fx_ref, fy_ref = ax * ms, ay * ms
            worst["Fx"] = max(worst["Fx"], scaled_err(fx_err, fx_ref))
            worst["Fy"] = max(worst["Fy"], scaled_err(fy_err, fy_ref))
        ok(f"{phys}: ay = Lat_acc + vx*r (worst scaled error {worst['ay']:.1e} <= 1e-12)",
           worst["ay"] <= 1e-12)
        ok(f"{phys}: ax = Lon_acc - vy*r (worst scaled error {worst['ax']:.1e} <= 1e-12)",
           worst["ax"] <= 1e-12)
        if phys == "corrected":
            ok(f"corrected: m*ax = SumFxg - f_drag - f_roll (worst {worst['Fx']:.1e} <= 1e-9)",
               worst["Fx"] <= 1e-9)
            ok(f"corrected: m*ay = SumFyg + f_side (worst {worst['Fy']:.1e} <= 1e-9)",
               worst["Fy"] <= 1e-9)
        else:
            ok(f"matlab: ms*ax = SumFxg - f_drag - f_roll - ms*thetadot*zsdot + ms*g*sin(theta) "
               f"(worst {worst['Fx']:.1e} <= 1e-9)", worst["Fx"] <= 1e-9)
            ok(f"matlab: ms*ay = SumFyg + f_side - ms*g*sin(phi)*cos(theta) "
               f"(worst {worst['Fy']:.1e} <= 1e-9)", worst["Fy"] <= 1e-9)

    # =========================================================================================
    print("2. data.vehicle on the stored Sturn point (fake IPOPT: x = stored w_opt)")
    with fake_ipopt():
        c2 = quiet(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False)
    v = channels(c2.data.vehicle)
    X, U = np.asarray(c2.data.x_opt, float), np.asarray(c2.data.u_opt, float)
    ok("stored point is not trivial: max|vy| > 1 m/s", np.abs(X[1]).max() > 1)
    ok("stored point is not trivial: max|r| > 0.5 rad/s", np.abs(X[2]).max() > 0.5)
    ok("stored point is not trivial: max|steer row| > 0.05", np.abs(U[-1]).max() > 0.05)
    ok("stored point is not trivial: max|Om_fl - Om_fr| > 1 rad/s", np.abs(X[5] - X[6]).max() > 1)
    need = {"ax", "ay", "delta", "Om_fl", "Om_fr", "Om_rl", "Om_rr", "Lon_acc", "Lat_acc", "E_motor"}
    ok("data.vehicle holds ax, ay, delta, Om_fl..Om_rr, Lon_acc, Lat_acc, E_motor", need <= set(v))
    ok("ay = Lat_acc + vx*r", np.allclose(v["ay"], v["Lat_acc"] + X[0] * X[2], rtol=1e-12, atol=1e-9))
    ok("ax = Lon_acc - vy*r", np.allclose(v["ax"], v["Lon_acc"] - X[1] * X[2], rtol=1e-12, atol=1e-9))
    ok("delta = 35 deg * u_opt[-1] / (pi/8)",
       np.allclose(v["delta"], U[-1] / (np.pi / 8) * np.deg2rad(35.0), rtol=1e-12, atol=1e-15))
    for w, row in (("fl", 5), ("fr", 6), ("rl", 7), ("rr", 8)):
        ok(f"Om_{w} = x_opt[{row}] (rad/s)", np.allclose(v[f"Om_{w}"], X[row], rtol=1e-12, atol=0))

    # =========================================================================================
    print("3. E_motor in Wh: trapezoid of P over t_opt, / 3600 s/h, / efficiency 0.9")
    E, P = np.asarray(v["E_motor"], float), np.asarray(v["P_motor"], float)
    t = np.asarray(c2.data.t_opt, float)
    ok("EM4 Off: E_motor[-1] = trapz(P_motor over t_opt) / 3600 / 0.9 [Wh]",
       abs(E[-1] - energy_wh(P, t)) <= 1e-4 * abs(E[-1]) + 1e-9)
    c4 = run_stub(Electric_4Motors="On", ATD="Off")
    v4 = channels(c4.data.vehicle)
    P4 = sum(np.asarray(v4[f"P_motor_{w}"], float) for w in ("fl", "fr", "rl", "rr"))
    E4, t4 = np.asarray(v4["E_motor"], float), np.asarray(c4.data.t_opt, float)
    ok("EM4 On: E_motor[-1] = trapz(sum of P_motor_fl..rr over t_opt) / 3600 / 0.9 [Wh]",
       abs(E4[-1] - energy_wh(P4, t4)) <= 1e-4 * abs(E4[-1]) + 1e-9)
else:
    print("SKIP sections 1-3 (casadi): Data/DATA_AA.mat missing")

# =============================================================================================
print("4. accel_channels: channels, derived for a result saved before E1, or (None, None)")
rng4 = np.random.default_rng(0)
a4, b4 = rng4.normal(size=4), rng4.normal(size=4)
x4 = rng4.normal(size=(23, 4))
s4 = np.linspace(0.0, 90.0, 13)
old = SimpleNamespace(N=3, OPT_d=3, s_full=s4, x_opt=x4, vehicle={"Lon_acc": a4, "Lat_acc": b4})
ax_old, ay_old = plotSDI.accel_channels(old)
ok("pre-E1 result: ax = Lon_acc - vy*r", np.array_equal(ax_old, a4 - x4[1] * x4[2]))
ok("pre-E1 result: ay = Lat_acc + vx*r", np.array_equal(ay_old, b4 + x4[0] * x4[2]))
p4, q4 = rng4.normal(size=4), rng4.normal(size=4)
new = SimpleNamespace(N=3, OPT_d=3, s_full=s4, x_opt=x4,
                      vehicle={"ax": p4, "ay": q4, "Lon_acc": a4, "Lat_acc": b4})
ax_new, ay_new = plotSDI.accel_channels(new)
ok("E1 result: stored ax returned unchanged", np.array_equal(ax_new, p4))
ok("E1 result: stored ay returned unchanged", np.array_equal(ay_new, q4))
names_old = {tr.name for tr in plotSDI.plot_speed(old).data}
ok("plot_speed: traces named ax and ay", {"ax", "ay"} <= names_old)
gg = vehicle_gg_from_solution(old)
ok("g-g: ay_g = (Lat_acc + vx*r) / 9.81", np.allclose(gg["ay_g"], (b4 + x4[0] * x4[2]) / 9.81, rtol=1e-12))
none = SimpleNamespace(N=3, OPT_d=3, s_full=s4, x_opt=x4, vehicle={})
res_none = plotSDI.accel_channels(none)
ok("no acceleration channels and no source: (None, None)", res_none[0] is None and res_none[1] is None)
names_none, raised = set(), False
try:
    names_none = {tr.name for tr in plotSDI.plot_speed(none).data}
except Exception:
    raised = True
ok("plot_speed without acceleration channels raises nothing", not raised)
ok("plot_speed without acceleration channels: no ax or ay trace", not ({"ax", "ay"} & names_none))

# =============================================================================================
print("5. stored BCN result (before E1): RMS(vx*r) and peak |ay| in g")
if os.path.exists(BCN):
    d5 = load_solution(BCN)
    X5 = np.asarray(d5.x_opt, float)
    rms_vr = float(np.sqrt(np.mean((X5[0] * X5[2]) ** 2)))
    ok(f"RMS(vx*r) = {rms_vr:.3f} m/s^2 in (10.5, 10.8)", 10.5 < rms_vr < 10.8)
    peak_g = float(np.abs(plotSDI.accel_channels(d5)[1]).max() / 9.81)
    ok(f"peak |ay| = {peak_g:.3f} g in (1.3, 1.6)", 1.3 < peak_g < 1.6)
else:
    print("  SKIP section 5: Results/BCN_Static_ATDOn_EM4Off.mat missing")

print("\nALL output channels TESTS PASSED" if os.path.exists(DATA_AA)
      else "\nnumpy output channel tests PASSED (casadi sections skipped)")
