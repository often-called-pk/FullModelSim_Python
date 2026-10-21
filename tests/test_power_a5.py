"""A5 powertrain knobs as vp_overrides keys (vehParams.PT_KEYS): per-motor ratings, gear, regen, pack row.

Plain script, no pytest. Each check prints PASS or FAIL, and the first FAIL stops the file. Run from any directory:

    python tests/test_power_a5.py

  1. defaults: no new attribute, rows and bounds unchanged, QSS lap pinned
  2. Tmax and Pmax (ratings of one motor), refusals by value, bare ctx, refusal under physics='matlab'
  3. gear: corrected EM4 motor speed, QSS rpm limit and force, Rw quirk kept
  4. Preg: torque and power bounds, QSS regen force anchored to vehModel wheel torques, QSS lap
  5. Ppack: pack row last with bounds [0, inf], QSS power cap, QSS lap
  6. energy, motor by motor: drive costs P / eff, regen returns eff * |P| (stand-in solve)
  7. NLP size and bounds with a fake IPOPT (Sturn, N = 18)

Sections 1 to 7 need casadi and Data/DATA_AA.mat. Without Data/DATA_AA.mat the file prints SKIP and exits 0.
No IPOPT runs here.
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import inspect
import io
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np

if not os.path.exists(os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")):   # pragma: no cover
    print("SKIP test_power_a5: Data/DATA_AA.mat missing")
    sys.exit(0)

import casadi as ca
import functions.transcription as T
import MLTP as mltp_mod
from functions.context import Ctx
from functions.ggv import build_envelope
from MLTP import build_path_constraints
from MLTP_screen import screen_sweep
from setup_sweep import _vp_snapshot
from userOpts import userOpts
from vehModel import vehModel


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout swallowed and warnings ignored."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


EM4 = dict(physics="corrected", Electric_4Motors="On", ATD="Off")
ONE = dict(physics="corrected", Electric_4Motors="Off", ATD="On")


def build(**kw):
    """Sturn userOpts and vehModel. Corrected EM4 with ATD Off unless kw says otherwise."""
    c = Ctx()
    quiet(userOpts, c, **dict(dict(circuit="Sturn", **EM4), **kw))
    vehModel(c)
    return c


def rows(c):
    """(model, row names, row function, lower bounds, upper bounds) of the path constraints."""
    m = c.m23
    names, h, lb, ub = build_path_constraints(ca, m, c.pt)
    return m, names, ca.Function("h", [m.x, m.u, m.pv], [h]), lb, ub


def state(c, vx):
    """Straight-line rolling state: wheel speed vx / Rw, tyres at rest, the rest zero."""
    X = np.zeros(c.m23.nx)
    X[0] = vx
    X[5:9] = vx / c.vp.Rw
    X[19:23] = [c.vp.xti_fl, c.vp.xti_fr, c.vp.xti_rl, c.vp.xti_rr]
    return X


def a5_error(**kw):
    """Message of the ValueError that build(**kw) raises, or "" if none.

    An unknown-key error does not count. It means the key is not routed at all.
    """
    try:
        build(**kw)
    except ValueError as exc:
        return "" if "Unknown" in str(exc) else str(exc)
    return ""


def lap(ov, **kw):
    """QSS lap time of one vp_overrides dict. Nothing is saved. Base: corrected EM4."""
    return screen_sweep("Sturn", [ov], **dict(EM4, **kw))[0]["lap_time"]


# =============================================================================
print("1. defaults: nothing changes")
c0 = build()
m0, hn0, _, lb0, ub0 = rows(c0)
ok("ctx.pt has no Preg and no Ppack", not hasattr(c0.pt, "Preg") and not hasattr(c0.pt, "Ppack"))
ok("12 EM4 rows, upper bounds all 1, lower bounds 0 (rpm, power) and -1 (BrTh)",
   len(hn0) == 12 and np.all(ub0 == 1.0) and np.array_equal(lb0, np.r_[np.zeros(8), -np.ones(4)]))
ok("motor torque bounds are [0, 1] with no negative zero (the NLP bound bytes stay)",
   np.array_equal(m0.u_min[:4], np.zeros(4)) and not np.signbit(m0.u_min[:4]).any() and np.all(m0.u_max[:4] == 1.0))
env0 = build_envelope(c0)
ok("QSS envelope: P_regen is None and to_dict() has no P_regen key",
   env0.P_regen is None and "P_regen" not in env0.to_dict())
ok("QSS envelope: F_brk(v) is the friction cap", env0.F_brk(30.0) == env0.F_brk_trq)
lap0 = lap({})
# Pin re-based on 5e882bf. A4 moved the corrected QSS lap by -0.003 %. It was 18.9951 on 185331b.
ok(f"QSS Sturn corrected EM4 lap {lap0:.5f} s == 18.99446 s", abs(lap0 - 18.99446) < 6e-5)
ok("explicit Preg = 0 and a huge Ppack leave the lap bit for bit",
   lap({"Preg": 0.0}) == lap0 and lap({"Ppack": 1e9}) == lap0)
ok("explicit Preg = 0 gives the default upper bounds",
   np.array_equal(rows(build(vp_overrides={"Preg": 0.0}))[4], ub0))

# =============================================================================
print("2. per-motor ratings (vp_overrides Tmax, Pmax) and refusals")
c1 = build(vp_overrides={"Tmax": 301.0, "Pmax": 228e3})
m1, _, f1, _, _ = rows(c1)
ok("pt.Tmax and pt.Pmax set, the four motor inputs scale with Tmax",
   c1.pt.Tmax == 301.0 and c1.pt.Pmax == 228e3 and np.all(m1.u_s[:4] == 301.0) and np.all(m1.u_max[:4] == 1.0))
X, U = state(c1, 40.0), np.zeros(m1.nu)
U[0] = 0.5 * 301.0
Om_m = X[5] * c1.vp.gear
got = float(f1(X / m1.x_s, U / m1.u_s, 0.0)[0])
ok(f"motor_power_fl = (Pmax - T Om_motor) / Pmax = {(228e3 - U[0] * Om_m) / 228e3:.6f}",
   abs(got - (228e3 - U[0] * Om_m) / 228e3) < 1e-12)
e1 = build_envelope(c1)
ok("QSS: P_mot = 4 Pmax, F_mot_trq = 4 Tmax gear / Rw",
   abs(e1.P_mot - 4 * 228e3) < 1e-6 and abs(e1.F_mot_trq - 4 * 301.0 * c1.vp.gear / c1.vp.Rw) < 1e-6)
ok("QSS lap with half ratings is 0.9% to 1.3% slower",
   0.009 < lap({"Tmax": 301.0, "Pmax": 228e3}) / lap0 - 1 < 0.013)
ok("Tmax <= 0, Pmax <= 0, Preg < 0, gear NaN and Ppack inf are refused by value (inf gives NaN rows)",
   all(a5_error(vp_overrides={k: v}) for k, v in
       (("Tmax", 0.0), ("Pmax", -1.0), ("Preg", -1.0), ("gear", float("nan")), ("Ppack", float("inf")))))
snap = quiet(_vp_snapshot, {"Tmax": 301.0, "gear": 8.0}, {"mb": 1900.0}, "Data", "MF205", "corrected")
ok("a bare ctx takes the keys (setup_sweep._vp_snapshot has no pt)",
   snap["gear"] == 8.0 and snap["mb"] == 1900.0)
ok("every key is refused under physics='matlab' and the message names corrected",
   all("corrected" in a5_error(physics="matlab", vp_overrides={k: v}) for k, v in
       (("Tmax", 301.0), ("Pmax", 228e3), ("gear", 8.0), ("Preg", 1e3), ("Ppack", 1e5))))

# =============================================================================
print("3. gear as a vp_overrides key")
c2 = build(vp_overrides={"gear": 8.0})
m2 = c2.m23
f2 = ca.Function("f2", [m2.x, m2.u, m2.pv], [m2.Om_motor_fl, m2.Om_fl])
a, b = (float(v) for v in f2(state(c2, 30.0) / m2.x_s, np.zeros(m2.nu), 0.0))
ok("vp.gear = 8, corrected EM4 Om_motor_fl = 8 Om_fl", c2.vp.gear == 8.0 and abs(a - 8.0 * b) < 1e-9 * a)
e2 = build_envelope(c2)
ok("QSS: v_rpm = OMmax Rw / gear, F_mot_trq = 4 Tmax gear / Rw",
   abs(e2.v_rpm - c2.pt.OMmax * c2.vp.Rw / 8.0) < 1e-9
   and abs(e2.F_mot_trq - 4 * c2.pt.Tmax * 8.0 / c2.vp.Rw) < 1e-6)
ok("default gear is still OMmax * 0.3142857 / Vmax (the Rw quirk stays)",
   abs(c0.vp.gear - c0.pt.OMmax * 0.3142857 / c0.pt.Vmax) < 1e-12)
ok("a gear 10% up speeds up the single-motor QSS lap",
   lap({"gear": 1.1 * c0.vp.gear}, **ONE) < lap({}, **ONE))

# =============================================================================
print("4. regenerative braking (Preg)")
Preg = 50e3
c3 = build(vp_overrides={"Preg": Preg})
m3, _, f3, lb3, ub3 = rows(c3)
ok("EM4: motor torque bounds [-1, 1] on the 4 inputs",
   np.all(m3.u_min[:4] == -1.0) and np.all(m3.u_max[:4] == 1.0))
ok("EM4: power rows ub = 1 + Preg / Pmax, BrTh rows ub = inf, rpm rows and every lower bound unchanged",
   np.allclose(ub3[:4], 1 + Preg / c3.pt.Pmax, rtol=0, atol=1e-15) and np.all(ub3[4:8] == 1.0)
   and np.all(np.isinf(ub3[8:])) and np.array_equal(lb3, lb0))
X = state(c3, 40.0)
Om_m = X[5] * c3.vp.gear
for frac, inside in ((-0.05, True), (-0.20, False)):
    U = np.zeros(m3.nu)
    U[0] = frac * c3.pt.Tmax
    r = float(f3(X / m3.x_s, U / m3.u_s, 0.0)[0])
    ok(f"fl motor at {frac:+.2f} Tmax = {U[0] * Om_m / 1e3:+.1f} kW: row {r:.4f} "
       f"{'inside' if inside else 'outside'} [0, {ub3[0]:.4f}]",
       (lb3[0] <= r <= ub3[0]) == inside)
c3s = build(vp_overrides={"Preg": Preg}, Electric_4Motors="Off", ATD="On")
m3s, _, _, lb3s, ub3s = rows(c3s)
ok("single motor + ATD: u_min = -1, power ub = 1 + Preg / Pmax, BrTh_1 ub = inf, ATD_eq ub 1e-3",
   m3s.u_min[0] == -1.0 and abs(ub3s[0] - (1 + Preg / c3s.pt.Pmax)) < 1e-15 and ub3s[1] == 1.0
   and np.isinf(ub3s[2]) and ub3s[3] == 1e-3)
env3 = build_envelope(c3)
ok("QSS: P_regen = 4 Preg with EM4 and Preg with one motor",
   env3.P_regen == 4 * Preg and build_envelope(c3s).P_regen == Preg)
# Anchor: the QSS regen force equals the vehModel wheel torques at the motor limit min(Tmax, Preg / Om_motor).
f_T = ca.Function("f_T", [m3.x, m3.u], [m3.T_fl + m3.T_fr + m3.T_rl + m3.T_rr, m3.Om_motor_fl])
anchored = True
for v in (10.0, 25.0, 40.0, 70.0):
    U = np.zeros(m3.nu)
    U[:4] = -min(c3.pt.Tmax, Preg / (state(c3, v)[5] * c3.vp.gear))
    T_sum, _ = (float(z) for z in f_T(state(c3, v) / m3.x_s, U / m3.u_s))
    anchored &= abs(-T_sum / c3.vp.Rw - (float(env3.F_brk(v)) - env3.F_brk_trq)) < 1e-6 * env3.F_brk_trq
ok("QSS regen force == vehModel wheel torques at the motor limit (v = 10, 25, 40, 70 m/s)", anchored)
ok("QSS lap with regen is 0.3% to 0.9% faster (the friction brakes sit near the tyre limit)",
   -0.009 < lap({"Preg": 456e3}) / lap0 - 1 < -0.003)

# =============================================================================
print("5. pack power row (Ppack)")
c4 = build(vp_overrides={"Ppack": 600e3})
m4, hn4, f4, lb4, ub4 = rows(c4)
ok("EM4: 13 rows, pack_power last, lower bound 0, upper bound inf",
   len(hn4) == 13 and hn4[-1] == "pack_power" and lb4[-1] == 0 and np.isinf(ub4[-1]))
X, U = state(c4, 40.0), np.zeros(m4.nu)
U[:4] = np.array([0.5, 0.4, 0.3, 0.2]) * c4.pt.Tmax
P_sum = float(np.sum(U[:4]) * X[5] * c4.vp.gear)
ok(f"pack_power = (Ppack - sum of the motor powers) / Ppack = {(600e3 - P_sum) / 600e3:.5f}",
   abs(float(f4(X / m4.x_s, U / m4.u_s, 0.0)[-1]) - (600e3 - P_sum) / 600e3) < 1e-12)
ok("single motor + ATD: rows end ATD_eq, pack_power",
   rows(build(vp_overrides={"Ppack": 300e3}, Electric_4Motors="Off", ATD="On"))[1][-2:] == ["ATD_eq", "pack_power"])
ok("QSS: P_mot = min(4 Pmax, Ppack)", abs(build_envelope(c4).P_mot - 600e3) < 1e-6)
ok("QSS lap with Ppack 456 kW is 4.5% to 5.3% slower", 0.045 < lap({"Ppack": 456e3}) / lap0 - 1 < 0.053)

# =============================================================================
print("6. energy channel, motor by motor: drive costs P / eff, regen returns eff * |P| (stand-in solve, no IPOPT)")
X7 = np.vstack([np.linspace(30, 50, 10), np.zeros((4, 10)), np.full((2, 10), 100.0)])
U7 = np.vstack([np.linspace(250.0, -150.0, 10), np.zeros((2, 10))])     # motor torque crosses zero
sig = inspect.signature(T.build_and_solve_nlp)


def init_stub(**k):
    return SimpleNamespace(data=SimpleNamespace(init=SimpleNamespace(x_opt=X7, u_opt=U7)),
                           elapsed=dict(ipopt_iters=1, return_status="Solve_Succeeded", qss=0.0))


def nlp_stub(*a, **k):
    """Returns the start guess as the solution. The energy channel is what this checks."""
    from functions.warmstart import nlp_structure
    b = sig.bind(*a, **k).arguments
    m, disc, g = b["m"], b["disc"], b["guesses"]
    st = nlp_structure(m.nx, m.nu, m.ny, disc["N"], b["OPT_d"], np.size(b["h_lb"]))
    u0 = np.array(g["u0"], dtype=float)
    u0[1] *= -1.0                       # motor fr against the other three: mixed signs, as in torque vectoring
    w = np.concatenate([g["x0"].reshape(-1, order="F"), u0.reshape(-1, order="F"),
                        np.asarray(g["xc0"]).reshape(-1, order="F")])
    return dict(sol={"x": w, "f": 0.0},
                solver=SimpleNamespace(stats=lambda: {"iter_count": 7, "return_status": "Solve_Succeeded"}),
                w_opt=w, lam_g=np.zeros(st["n_g"]), lam_x=np.zeros(st["n_w"]), structure=st,
                warm_info=dict(x0=False, lam_g0=False, lam_x0=False, ipopt=False, duals=False),
                sym_type="SX", linear_solver="fake", N=disc["N"])


real = (mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp)
mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp = init_stub, nlp_stub
try:
    def run(ov, **kw):
        return quiet(mltp_mod.MLTP, circuit="Sturn", save=False, plot=False, mesh="uniform",
                     vp_overrides=ov, **dict(EM4, **kw))
    cr, cd = run({"Preg": 60e3}), run({})
finally:
    mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp = real
for tag, cc, regen in (("regen", cr, True), ("default", cd, False)):
    v, t, eff = cc.data.vehicle, np.asarray(cc.data.t_opt, float), cc.pt.eff
    Ps = [np.asarray(v[f"P_motor_{w}"], float) for w in ("fl", "fr", "rl", "rr")]
    P = sum(Ps)
    Pb = sum(np.where(p < 0, p * eff, p / eff) for p in Ps) if regen else P / eff
    want = np.r_[0.0, np.cumsum(0.5 * (Pb[:-1] + Pb[1:]) * np.diff(t))] / 3600.0
    rule = "per motor: eff * P when P < 0, else P / eff" if regen else "P / eff"
    mixed = np.any((np.min(Ps, axis=0) < 0) & (np.max(Ps, axis=0) > 0))
    ok(f"{tag}: E_motor == trapezoid of the pack power ({rule}), min P {P.min() / 1e3:+.0f} kW, motor signs mixed",
       np.allclose(np.asarray(v["E_motor"], float), want, rtol=1e-3, atol=1e-6) and P.min() < 0 and mixed)
ok("regen lap energy is larger than the default lap energy (the regen credit)",
   cr.data.vehicle["E_motor"][-1] > cd.data.vehicle["E_motor"][-1])

# =============================================================================
print("7. NLP size and bounds with a fake IPOPT (Sturn, N = 18)")
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=np.vstack([np.full(10, 300.0), np.zeros((2, 10))]))}


@contextlib.contextmanager
def fake_ipopt(rec):
    """Replaces the IPOPT solver. rec gets the bound and start vectors of the solve call."""
    def fake(ca_, nlp, opts, info=None):
        if info is not None:
            info["linear_solver"] = "fake"

        def solver(**k):
            rec.update({key: np.asarray(v, float).reshape(-1) for key, v in k.items()})
            return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                    "lam_x": np.zeros(nlp["x"].numel())}
        solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
        return solver
    real_, T._make_solver = T._make_solver, fake
    try:
        yield rec
    finally:
        T._make_solver = real_


class _Captured(Exception):
    pass


def nlp_bounds(ov, **kw):
    """Bounds and start vectors of the NLP that MLTP builds for ov. IPOPT is not run."""
    seen = {}

    def stand_in(*a, **k):
        seen.update(sig.bind(*a, **k).arguments)
        raise _Captured()
    real_ = mltp_mod.build_and_solve_nlp
    mltp_mod.build_and_solve_nlp = stand_in
    try:
        quiet(mltp_mod.MLTP, circuit="Sturn", warm_start=INIT, save=False, plot=False, vp_overrides=ov,
              **dict(EM4, **kw))
    except _Captured:
        pass
    finally:
        mltp_mod.build_and_solve_nlp = real_
    rec = {}
    with fake_ipopt(rec):
        quiet(T.build_and_solve_nlp, **seen)
    return rec


b0 = nlp_bounds({})
N1 = 19                                                                   # N + 1 nodes
bp = nlp_bounds({"Preg": 50e3})
ok("regen: same n_w and n_g, 4 (N+1) lower bounds -1 (were 0), 8 (N+1) upper g bounds changed",
   bp["x0"].size == b0["x0"].size and bp["lbg"].size == b0["lbg"].size
   and np.count_nonzero(bp["lbx"] != b0["lbx"]) == 4 * N1 and np.all(bp["lbx"][bp["lbx"] != b0["lbx"]] == -1.0)
   and np.count_nonzero(bp["ubg"] != b0["ubg"]) == 8 * N1 and np.array_equal(bp["lbg"], b0["lbg"]))
bk = nlp_bounds({"Ppack": 600e3})
ok("pack: n_w unchanged, n_g + (N+1), the new rows have bounds [0, inf]",
   bk["x0"].size == b0["x0"].size and bk["lbg"].size == b0["lbg"].size + N1
   and np.isinf(bk["ubg"]).sum() == N1 and np.all(bk["lbg"][np.isinf(bk["ubg"])] == 0))
br = nlp_bounds({"Tmax": 301.0, "Pmax": 228e3})
dr = br["ubg"] != b0["ubg"]
ok("ratings: same sizes, only the 4 N motor rate rows change, 2e4 / 602 -> 2e4 / 301 (the physical 2e4 Nm/s stays)",
   br["x0"].size == b0["x0"].size and br["lbg"].size == b0["lbg"].size and dr.sum() == 4 * (N1 - 1)
   and np.allclose(b0["ubg"][dr], 2e4 / 602.0) and np.allclose(br["ubg"][dr], 2e4 / 301.0))

print("ALL A5 power TESTS PASSED")
