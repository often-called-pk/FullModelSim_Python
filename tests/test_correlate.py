"""correlate.py (E6 stage S, E7a grip, E8a drag, E8b downforce): fit QSS knobs to a logged lap.

Plain script. Sections 1 and 1b are numpy only. The model sections need Data/DATA_AA.mat. Without it
the file prints SKIP and exits 0 after section 1b. Section 5 (lift_fit equilibria and the downforce
stage) and the V9 check of section 7 also need casadi. The logs are synthetic: QSS laps and straight
runs of a car with known LMUX, LMUY and Cd_scale. No IPOPT and no solve. Run from any directory:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_correlate.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np

import functions.correlate as C
from functions.collocation import collocation_points
from functions.ggv import build_envelope, march

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


@contextlib.contextmanager
def quiet():
    """Swallow stdout and warnings. userOpts prints and warns."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def raises(exc, fn, *args, text=None, **kw):
    """True if fn(*args, **kw) raises exc. text, when given, must be in the message."""
    try:
        with quiet():
            fn(*args, **kw)
    except exc as err:
        return text is None or text in str(err)
    return False


def strict_json(text):
    """json.loads that refuses NaN and Infinity. Python accepts both by default."""
    def refuse(token):
        raise ValueError(token)
    return json.loads(text, parse_constant=refuse)


def load_strict(path):
    """The parsed ledger at path, or None when it holds NaN or Infinity."""
    with open(path) as fh:
        text = fh.read()
    try:
        return strict_json(text)
    except ValueError:
        return None


def make_ctx(circuit, physics="matlab", **ov):
    """Ctx of the car on ``circuit``. ov are vp_overrides. MF keys such as LMUX go to ctx.mf."""
    from functions.context import Ctx
    from userOpts import userOpts
    ctx = Ctx()
    with quiet():
        userOpts(ctx, circuit=circuit, vp_overrides=ov, physics=physics)
    return ctx


def lap_qss(ctx):
    """QSS lap of ctx from 60 m/s on a 1 m march, the speed that run_stage uses."""
    return march(build_envelope(ctx), ctx.track.s, ctx.track.k, 60.0)


def qss_log(circuit, dt=0.02, noise=(0.0, 0.0, 0.0), seed=0, **ov):
    """50 Hz log of the QSS lap of the car on ``circuit``. noise is the sigma of vx, ax and ay."""
    p = lap_qss(make_ctx(circuit, **ov))
    t = np.arange(0.0, p["t"][-1], dt)
    log = dict(t=t, vx=np.interp(t, p["t"], p["v"]), ax=np.interp(t, p["t"], p["ax"]),
               ay=np.interp(t, p["t"], p["ay"]))
    rng = np.random.default_rng(seed)
    for key, sigma in zip(("vx", "ax", "ay"), noise):
        log[key] = log[key] + sigma * rng.standard_normal(t.size)
    return log


def straight_log(car, thr, v0, v1, dt=0.02, sv=0.0, st=0.0, sa=None, down=False, seed=0):
    """Straight run of ``car`` from v0 to v1. down means v1 < v0. Motor torque is thr * min(Tmax,
    Pmax / motor speed). dv/dt = (gear Tm / Rw - F_res) / M_eff, by Euler. Noise: sv on vx, st
    (relative) on Tm and sa on ax. The log has ax only when sa is given."""
    vp, pt = car.vp, car.pt
    env = build_envelope(car)
    v, vs, Ts, As = v0, [], [], []
    while (v > v1 if down else v < v1) and len(vs) < 400000:
        Tm = thr * min(pt.Tmax, pt.Pmax / (v / vp.Rw * vp.gear))
        a = (vp.gear * Tm / vp.Rw - float(env.F_res(v))) / env.M_eff
        vs.append(v)
        Ts.append(Tm)
        As.append(a)
        v += a * dt
    n, rng = len(vs), np.random.default_rng(seed)
    log = dict(t=dt * np.arange(n), vx=np.array(vs) + sv * rng.standard_normal(n),
               Tm=np.array(Ts) * (1 + st * rng.standard_normal(n)), ay=np.zeros(n), brake=np.zeros(n))
    if sa is not None:
        log["ax"] = np.array(As) + sa * rng.standard_normal(n)
    return log


def near_limit_speeds(ctx, log):
    """Speeds of the log samples above 90 % of the lateral limit of ctx."""
    env = build_envelope(ctx)
    v = log["vx"]
    return v[np.abs(log["ay"]) / np.interp(v, env.v_grid, env.ay_max) > 0.9]


def lap_data(ctx, p, h=2.0, d=3):
    """Data of MLTP (OPT_d, s_full, t_full, x_full, vehicle_full) for the QSS lap p of ctx. The nodes
    are the Legendre points of knots h m apart. The channels are the QSS values at the nodes."""
    tau = np.asarray(collocation_points(d, "legendre"), float)
    s_knot = np.linspace(p["s"][0], p["s"][-1], max(int(round((p["s"][-1] - p["s"][0]) / h)), 1) + 1)
    dsk = np.diff(s_knot)
    s_full = np.concatenate([s_knot[k] + np.r_[0.0, tau] * dsk[k] for k in range(dsk.size)] + [[s_knot[-1]]])
    vx = np.interp(s_full, p["s"], p["v"])
    z = np.zeros_like(s_full)
    channels = dict(delta=z, throttle=np.full_like(s_full, 0.5), om_gear=vx / ctx.vp.Rw * ctx.vp.gear,
                    xs_fl=z, xs_fr=z, xs_rl=z, xs_rr=z,
                    ax=np.interp(s_full, p["s"], p["ax"]), ay=np.interp(s_full, p["s"], p["ay"]))
    return SimpleNamespace(OPT_d=d, s_full=s_full, t_full=np.interp(s_full, p["s"], p["t"]),
                           x_full=np.vstack([vx, z, z, z, z]), vehicle_full=channels)


# =============================================================================================
print("1. gamma: 1 for orthogonal columns, inf for a zero or a repeated column, 1 / sqrt(1 - cos) for two")
ok("orthogonal columns: 1", abs(C.gamma(np.eye(3)) - 1.0) < 1e-12)
ok("repeated column: inf", C.gamma(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])) == np.inf)
ok("zero column: inf", C.gamma(np.array([[1.0, 0.0], [2.0, 0.0]])) == np.inf)
S = np.array([[1.0, 0.96], [0.0, np.sqrt(1 - 0.96 ** 2)]])
ok("two unit columns, cos 0.96: 1 / sqrt(1 - 0.96) = 5", abs(C.gamma(S) - 5.0) < 1e-9)
ok("one column: 1", C.gamma(np.array([[3.0], [4.0]])) == 1.0)

print("1b. functions/correlate.py imports with casadi blocked")
code = ("import sys; sys.modules['casadi'] = None; sys.path.insert(0, %r); "
        "import functions.correlate; print('import ok')" % _bootstrap.ROOT)
r = subprocess.run([sys.executable, "-c", code], cwd=_bootstrap.ROOT, capture_output=True, text=True, timeout=300)
ok("import with casadi blocked: rc 0 and 'import ok' printed", r.returncode == 0 and "import ok" in r.stdout)

if not os.path.exists(DATA_AA):
    print("SKIP sections 2-7: Data/DATA_AA.mat missing")
    print("\nnumpy correlate checks PASSED (model sections skipped)")
    sys.exit(0)

try:
    import casadi  # noqa: F401
    HAVE_CA = True
except ImportError:
    HAVE_CA = False

# =============================================================================================
print("2. identifiable: the near-limit samples of the QSS lap (Sturn, BCN)")
ok("NEVER names LFZO, Fz0_shift, LKX and LKY", set(C.NEVER) == {"LFZO", "Fz0_shift", "LKX", "LKY"})
for circ in ("Sturn", "BCN"):
    ctx = make_ctx(circ)
    vs = near_limit_speeds(ctx, qss_log(circ))

    def obs(c, vs=vs):
        e = build_envelope(c)
        return np.interp(vs, e.v_grid, e.ay_max)

    r1 = C.identifiable(ctx, ["LMUY"], obs)
    r2 = C.identifiable(ctx, ["LMUY", "LFZO"], obs)
    r3 = C.identifiable(ctx, ["LMUY", "Fz0_shift"], obs)
    r4 = C.identifiable(ctx, ["LMUY", "Cl_scale"], obs)
    r5 = C.identifiable(ctx, ["LMUY", "pDy2"], obs)
    r6 = C.identifiable(ctx, ["LFZO"], obs)
    ok(f"{circ}: LMUY alone: gamma 1 and ok", r1["gamma"] == 1.0 and r1["ok"])
    ok(f"{circ}: LFZO alone: gamma 1, never ['LFZO'], not ok (the NEVER rule alone refuses it)",
       abs(r6["gamma"] - 1.0) < 1e-12 and r6["never"] == ["LFZO"] and not r6["ok"])
    ok(f"{circ}: LMUY with pDy2 (no NEVER name): gamma {r5['gamma']:.1f} above 15, not ok",
       r5["gamma"] > 15 and r5["never"] == [] and not r5["ok"])
    ok(f"{circ}: LFZO with LMUY: gamma {r2['gamma']:.0f} above 15, never ['LFZO'], not ok",
       r2["gamma"] > 15 and r2["never"] == ["LFZO"] and not r2["ok"])
    ok(f"{circ}: Fz0_shift with LMUY: gamma {r3['gamma']:.3g} above 1e4, not ok",
       r3["gamma"] > 1e4 and not r3["ok"])
    if circ == "BCN":
        ok(f"BCN: Cl_scale apart from LMUY: gamma {r4['gamma']:.2f} below 5", r4["gamma"] < 5)
    else:
        ok(f"Sturn: Cl_scale not apart from LMUY: se {r4['se']['Cl_scale']:.2f} above 0.1",
           r4["se"]["Cl_scale"] > 0.1)
    ok(f"{circ}: ctx knobs back at 1 after identifiable",
       ctx.mf.LMUY == 1.0 and ctx.mf.LFZO == 1.0 and ctx.vp.Fz0_shift == 1.0 and ctx.vp.Cl_scale == 1.0)
ok("lap time alone cannot tell LMUX from LMUY (one row: gamma inf)",
   C.identifiable(make_ctx("Sturn"), ["LMUX", "LMUY"], lambda c: [lap_qss(c)["lap_time"]])["gamma"] == np.inf)

# =============================================================================================
print("3. grip_fit on QSS logs of a known LMUX and LMUY")
for circ in ("Sturn", "BCN"):
    model = make_ctx(circ)
    for tx, ty in ((1.0, 1.0), (0.9, 0.9), (0.95, 1.05), (1.1, 0.85)):
        r = C.grip_fit(qss_log(circ, LMUX=tx, LMUY=ty), model)
        ok(f"{circ} truth LMUX {tx}, LMUY {ty}: fit LMUX {r['LMUX']:.4f}, LMUY {r['LMUY']:.4f}",
           abs(r["LMUY"] - ty) < 0.01 and abs(r["LMUX"] - tx) < 0.03)
    r = C.grip_fit(qss_log(circ, noise=(0.1, 0.05, 0.05), seed=4, LMUX=0.92, LMUY=0.92), model)
    ok(f"{circ} noise (0.1, 0.05, 0.05), truth 0.92: fit LMUY {r['LMUY']:.4f}, LMUX {r['LMUX']:.4f}",
       abs(r["LMUY"] - 0.92) < 0.012 and abs(r["LMUX"] - 0.92) < 0.05)
    ok(f"{circ}: rms after the fit below rms before", r["rms_after"] < r["rms_before"])
    if circ == "BCN":
        ok(f"BCN: n_events {r['n_events']} from 3 to 20, below the {len(r['vt'])} samples of the limit set",
           3 <= r["n_events"] <= 20 and r["n_events"] < len(r["vt"]))
    ok(f"{circ}: ctx knobs unchanged by grip_fit", model.mf.LMUY == 1.0 and model.mf.LMUX == 1.0)

car = make_ctx("BCN")
r = C.grip_fit(straight_log(car, 1.0, 30.0, 70.0, sa=0.0), car)
ok(f"a straight run: LMUY nan with a reason ('{r['why_y']}')",
   np.isnan(r["LMUY"]) and "fewer than 20" in r["why_y"])
slow = qss_log("BCN", LMUY=0.9)
slow["ay"] = slow["ay"] * 0.3                          # lateral acceleration at 30 % of the limit
r = C.grip_fit(slow, make_ctx("BCN"))
ok(f"a log at 30 % of the limit: LMUY nan with a reason ('{r['why_y']}')",
   np.isnan(r["LMUY"]) and r["why_y"] != "")
r = C.grip_fit(qss_log("BCN", LMUX=0.9, LMUY=0.9), make_ctx("BCN"), lam=(1.2, 2.0))
ok(f"a bracket (1.2, 2.0) without the truth 0.9: LMUY and LMUX nan, why_x '{r['why_x']}'",
   np.isnan(r["LMUY"]) and r["why_y"].startswith("no root") and np.isnan(r["LMUX"]) and r["why_x"] == "no root")
lg93 = qss_log("BCN", LMUX=0.93, LMUY=0.93)
real_brentq, calls = C.brentq, []


def pass1_far(f, a, b, **kw):
    """brentq with the pass-1 LMUX root at 10: pass 2 then finds fewer than 20 tyre-limited samples."""
    calls.append(1)
    return 10.0 if len(calls) == 2 else real_brentq(f, a, b, **kw)


C.brentq = pass1_far
try:
    r = C.grip_fit(lg93, make_ctx("BCN"))
finally:
    C.brentq = real_brentq
ok(f"pass 2 stops after pass 1 set LMUX: LMUX nan, why_x '{r['why_x'][:20]} ...'",
   len(calls) == 2 and np.isnan(r["LMUX"]) and r["why_x"].startswith("fewer than 20"))
r = C.grip_fit(lg93, make_ctx("BCN"))
i = np.flatnonzero(np.isin(lg93["vx"], r["vb"]))[0]    # a raw sample of the trail braking limit set
spike = lg93["ax"].copy()
spike[i], spike[i + 1] = spike[i] + 13.0, spike[i + 1] - 13.0      # a kerb spike: up, then down
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = C.grip_fit(dict(lg93, ax=spike), make_ctx("BCN"))
ok(f"a raw ax of {spike[i]:+.1f} m/s^2 in the trail braking limit set: no exception, LMUX {r['LMUX']:.3f} finite",
   spike[i] > 0 and np.isfinite(r["LMUX"]))

print("3b. grip_from_lap: the lap ratio of run V9 (Sturn, LMUX = LMUY = 0.95: 18.24010 s, 18.01633 s)")
car = make_ctx("Sturn")
s = C.grip_from_lap(car, 18.24010 / 18.01633)
ok(f"V9 ratio gives s {s:.4f}, within 0.002 of the solved 0.95", abs(s - 0.95) < 0.002)
ok("grip_from_lap leaves ctx at LMUX = LMUY = 1", car.mf.LMUX == 1.0 and car.mf.LMUY == 1.0)
ok("a ratio outside the bounds: nan", np.isnan(C.grip_from_lap(car, 3.0)))
ratio_09 = lap_qss(make_ctx("Sturn", LMUX=0.9, LMUY=0.9))["lap_time"] / lap_qss(car)["lap_time"]
s09 = C.grip_from_lap(car, ratio_09)
ok(f"QSS-only truth: the lap ratio of s = 0.9 gives s {s09:.6f}, within 1e-4 of 0.9",
   abs(s09 - 0.9) < 1e-4)
with quiet():
    rec = C.run_stage("grip", None, "Sturn", lap=(18.24010, 18.01633), tags={"driver": "synthetic"})
ok(f"lap-only grip stage: method 'lap', LMUY apply {rec['apply']['LMUY']:.4f} within 0.002 of 0.95",
   rec["method"] == "lap" and abs(rec["apply"]["LMUY"] - 0.95) < 0.002)

print("3c. a ChassisSim monster file: export_monster, then read_monster, then grip_fit")
from functions.telemetry import export_monster, read_monster
car = make_ctx("BCN", LMUX=0.93, LMUY=0.93)
data = lap_data(car, lap_qss(car))
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "monster.txt")
    export_monster(data, path)
    log = read_monster(path)
r = C.grip_fit(log, make_ctx("BCN"))
ok(f"monster file: fit LMUY {r['LMUY']:.4f} within 0.012 of 0.93, LMUX {r['LMUX']:.4f} within 0.05 of 0.93",
   abs(r["LMUY"] - 0.93) < 0.012 and abs(r["LMUX"] - 0.93) < 0.05)

# =============================================================================================
print("4. drag_fit on straight runs of a known Cd_scale (the fit model at Cd_scale 1)")
model = make_ctx("BCN")
for cd in (0.8, 1.2, 1.5):
    car = make_ctx("BCN", Cd_scale=cd)
    r = C.drag_fit(straight_log(car, 1.0, 30.0, 75.0, sa=0.0), model, win=0.2)
    ok(f"Cd_scale {cd}, exact ax: fit {r['Cd_scale']:.5f} within 1e-4, f {r['f']:.4f} within 1e-4 of 0.01",
       abs(r["Cd_scale"] - cd) < 1e-4 and abs(r["f"] - 0.01) < 1e-4)
    r = C.drag_fit(straight_log(car, 1.0, 30.0, 75.0), model, win=2.0)
    ok(f"Cd_scale {cd}, speed only: fit {r['Cd_scale']:.4f} within 0.01", abs(r["Cd_scale"] - cd) < 0.01)
    r = C.drag_fit(straight_log(car, 0.7, 30.0, 75.0, sv=0.05, st=0.02, sa=0.05, seed=1), model, win=0.2)
    ok(f"Cd_scale {cd}, noise (vx 0.05 m/s, torque 2 %, ax 0.05 m/s^2): fit {r['Cd_scale']:.3f} "
       f"within 3 se + 0.01 (se {r['se_Cd_scale']:.3f}), ok",
       abs(r["Cd_scale"] - cd) < 3 * r["se_Cd_scale"] + 0.01 and r["ok"])
    r = C.drag_fit(straight_log(car, 0.0, 75.0, 25.0, down=True, sv=0.05, sa=0.05, seed=2), model,
                   win=0.2, coast=True)
    ok(f"Cd_scale {cd}, coast-down with noise: fit {r['Cd_scale']:.3f} within 0.03", abs(r["Cd_scale"] - cd) < 0.03)
r = C.drag_fit(straight_log(make_ctx("BCN", Cd_scale=1.5), 1.0, 30.0, 75.0, sa=0.0), make_ctx("BCN", Cd_scale=1.2),
               win=0.2)
ok(f"the fit model at Cd_scale 1.2: fit {r['Cd_scale']:.5f} within 1e-4 of the truth 1.5 (an absolute value)",
   abs(r["Cd_scale"] - 1.5) < 1e-4)

lg = straight_log(model, 1.0, 30.0, 75.0, sa=0.0)
lg["Tm"] = lg["Tm"] / 0.95                             # logged torque 5 % high
r = C.drag_fit(lg, model, win=0.2)
ok(f"torque 5 % high: fit Cd_scale {r['Cd_scale']:.3f} in (0.84, 0.88) (14 % low by design)",
   0.84 < r["Cd_scale"] < 0.88)
r = C.drag_fit(straight_log(model, 1.0, 60.0, 64.0, sa=0.0), model, win=0.05)
ok(f"60-64 m/s run, {r['n']} windows: gamma {r['gamma']:.0f} above 15 and ok False",
   r["n"] >= 8 and r["gamma"] > 15 and not r["ok"])
r = C.drag_fit(straight_log(model, 1.0, 30.0, 75.0, sa=0.0), model, win=2.0)
ok(f"few windows: n {r['n']} from 3 to 7, exact fit {r['Cd_scale']:.4f}, ok False",
   3 <= r["n"] < 8 and abs(r["Cd_scale"] - 1.0) < 1e-3 and not r["ok"])
r = C.drag_fit(dict(t=np.arange(100) * 0.02, vx=np.full(100, 50.0), Tm=np.zeros(100), ay=np.full(100, 5.0)),
               model)
ok("no straight window (ay 5 m/s^2 throughout): Cd_scale nan and ok False",
   np.isnan(r["Cd_scale"]) and not r["ok"])

from functions.context import Ctx
from userOpts import userOpts
car = make_ctx("BCN", Cd_scale=1.2)
up = straight_log(car, 1.0, 30.0, 75.0, sa=0.0)
r1 = C.drag_fit(up, model, win=0.2)
em4 = Ctx()
with quiet():
    userOpts(em4, circuit="BCN", Electric_4Motors="On", ATD="Off")
thr = {k: x for k, x in up.items() if k != "Tm"}
r_one = C.drag_fit(dict(thr, throttle=up["Tm"] / model.pt.Tmax), model, win=0.2)
r_four = C.drag_fit(dict(thr, throttle=up["Tm"] / (4 * em4.pt.Tmax)), em4, win=0.2)
ok(f"throttle instead of Tm: one motor {r_one['Cd_scale']:.6f}, four motors (mean throttle) "
   f"{r_four['Cd_scale']:.6f}, both equal to the Tm fit {r1['Cd_scale']:.6f}",
   em4.pt.EM4 == 1 and abs(r_one["Cd_scale"] - r1["Cd_scale"]) < 1e-9
   and abs(r_four["Cd_scale"] - r1["Cd_scale"]) < 1e-9)
down = straight_log(car, 0.0, 75.0, 25.0, down=True, sa=0.0)
mix = {k: np.r_[up[k] / (0.95 if k == "Tm" else 1.0), down[k]] for k in ("vx", "Tm", "ay", "brake", "ax")}
mix["t"] = 0.02 * np.arange(mix["vx"].size)                        # torque 5 % high on the powered part
r_c, r_p = C.drag_fit(mix, model, win=0.2, coast=True), C.drag_fit(mix, model, win=0.2)
ok(f"powered run (torque 5 % high), then coast-down: coast=True {r_c['Cd_scale']:.4f} within 0.02 of 1.2, "
   f"all windows {r_p['Cd_scale']:.4f} off by more than 0.04, {r_c['n']} windows below {r_p['n']}",
   abs(r_c["Cd_scale"] - 1.2) < 0.02 and abs(r_p["Cd_scale"] - 1.2) > 0.04 and r_c["n"] < r_p["n"])
vb = np.arange(up["vx"][-1], 30.0, -12.0 * 0.02)                   # braking at 12 m/s^2, no torque
z = np.zeros(vb.size)
stop = {k: np.r_[up[k], x] for k, x in (("vx", vb), ("Tm", z), ("ay", z), ("ax", z - 12.0), ("brake", z + 1.0))}
stop["t"] = 0.02 * np.arange(stop["vx"].size)
r_b = C.drag_fit(stop, model, win=0.2)
r_n = C.drag_fit({k: x for k, x in stop.items() if k != "brake"}, model, win=0.2)
r_0 = C.drag_fit(dict(stop, brake=0.0 * stop["brake"]), model, win=0.2)
ok(f"run, then braking: with brake {r_b['Cd_scale']:.5f} (n {r_b['n']}), without a brake channel "
   f"{r_n['Cd_scale']:.5f} (n {r_n['n']}), both within 1e-4 of 1.2",
   abs(r_b["Cd_scale"] - 1.2) < 1e-4 and abs(r_n["Cd_scale"] - 1.2) < 1e-4)
ok(f"a brake channel at 0 during the braking: fit {r_0['Cd_scale']:.2f}, more than 0.5 from 1.2 (brake is read)",
   abs(r_0["Cd_scale"] - 1.2) > 0.5)

print("4a. check_ax: the sign gate on ax")
lg = qss_log("BCN", noise=(0.05, 0.02, 0.02), seed=1)
ok(f"check_ax on a good log: correlation {C.check_ax(lg):.3f} above 0.9", C.check_ax(lg) > 0.9)
ok("grip_fit on a log with ax flipped: ValueError 'ax does not follow'",
   raises(ValueError, C.grip_fit, dict(lg, ax=-lg["ax"]), make_ctx("BCN"), text="ax does not follow"))
sl = straight_log(model, 1.0, 30.0, 75.0, sa=0.05)
ok("drag_fit on a log with ax flipped: ValueError", raises(ValueError, C.drag_fit, dict(sl, ax=-sl["ax"]), model))
ok("no t in the log: check_ax skips (nan)", np.isnan(C.check_ax(dict(vx=np.ones(5), ax=np.ones(5)))))
ok("no ax in the log: check_ax skips (nan)", np.isnan(C.check_ax(dict(t=np.arange(5.0), vx=np.ones(5)))))
t60 = 0.02 * np.arange(60)
ok("60 samples (10 usable) with ax flipped: check_ax skips (nan), no ValueError",
   np.isnan(C.check_ax(dict(t=t60, vx=10.0 + 0.5 * t60 ** 2, ax=-t60))))
noise = np.random.default_rng(0).standard_normal(500)
ok("constant speed (1e-9 m/s noise): check_ax skips (nan)",
   np.isnan(C.check_ax(dict(t=0.02 * np.arange(500), vx=30.0 + 1e-9 * noise, ax=noise))))

print("4b. 1 % NaN dropouts in vx, ax, ay (and Tm) do not break the fits")
rng = np.random.default_rng(0)


def dropouts(log, keys):
    for key in keys:
        log[key][rng.choice(log[key].size, size=log[key].size // 100, replace=False)] = np.nan
    return log


lg = dropouts(qss_log("BCN", noise=(0.05, 0.02, 0.02), seed=1, LMUX=0.93, LMUY=0.93), ("vx", "ax", "ay"))
r = C.grip_fit(lg, make_ctx("BCN"))
ok(f"grip_fit with NaN: LMUY {r['LMUY']:.4f} within 0.012 of 0.93", abs(r["LMUY"] - 0.93) < 0.012)
sl = dropouts(straight_log(make_ctx("BCN", Cd_scale=1.2), 1.0, 30.0, 75.0, sv=0.05, st=0.02, sa=0.05, seed=2),
              ("vx", "ax", "Tm"))
r = C.drag_fit(sl, make_ctx("BCN"), win=0.5)
ok(f"drag_fit with NaN: Cd_scale {r['Cd_scale']:.3f} within 3 se + 0.02 of 1.2 "
   f"(se {r['se_Cd_scale']:.3f}, n {r['n']})",
   abs(r["Cd_scale"] - 1.2) < 3 * r["se_Cd_scale"] + 0.02)

# =============================================================================================
print("5. lift_fit: matlab physics refused, corrected equilibria give Cl_scale")
ok("physics matlab: ValueError", raises(ValueError, C.lift_fit, {}, make_ctx("Sturn")))
if HAVE_CA:
    from vehModel import rolling_start, vehModel
    speeds = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    for cl in (1.0, 1.3):
        car = make_ctx("Sturn", physics="corrected", Cl_scale=cl)
        with quiet():
            vehModel(car)
        vp = car.vp
        rows = []
        for v in speeds:
            X, _ = rolling_start(car, car.m23, float(v))
            zs, th, ph, zt = X[9], X[11], X[13], X[19:23]
            rows.append([-(zs - vp.l_f * th + vp.t / 2 * ph) - (zt[0] - vp.xti_fl),
                         -(zs - vp.l_f * th - vp.t / 2 * ph) - (zt[1] - vp.xti_fr),
                         -(zs + vp.l_r * th + vp.t / 2 * ph) - (zt[2] - vp.xti_rl),
                         -(zs + vp.l_r * th - vp.t / 2 * ph) - (zt[3] - vp.xti_rr)])
        R = np.array(rows)
        log = dict(vx=speeds, ax=np.zeros(5), ay=np.zeros(5), xs_fl=R[:, 0], xs_fr=R[:, 1],
                   xs_rl=R[:, 2], xs_rr=R[:, 3])
        r = C.lift_fit(log, make_ctx("Sturn", physics="corrected"))              # the fit model at Cl_scale 1
        ok(f"corrected equilibria, truth Cl_scale {cl}: front {r['front']:.3f}, rear {r['rear']:.3f}, "
           f"mean {r['mean']:.3f} within 4 %", abs(r["mean"] - cl) < 0.04 * cl)
    r12 = C.lift_fit(log, make_ctx("Sturn", physics="corrected", Cl_scale=1.2))
    ok(f"the fit model at Cl_scale 1.2: mean {r12['mean']:.3f}, the same absolute value as with the model at 1",
       abs(r12["mean"] - r["mean"]) < 1e-9)
    big = {k: np.tile(x, 20) for k, x in log.items()}              # 100 samples
    big["xs_fl"][7] = np.nan                                       # a 1 % dropout in one damper channel
    r_nan = C.lift_fit(big, make_ctx("Sturn", physics="corrected"))
    r_cut = C.lift_fit({k: np.delete(x, 7) for k, x in big.items()}, make_ctx("Sturn", physics="corrected"))
    ok(f"1 % NaN in xs_fl: the fit drops that sample (mean {r_nan['mean']:.3f}, the fit without the sample)",
       np.isfinite(r_nan["mean"]) and abs(r_nan["mean"] - r_cut["mean"]) < 1e-12)
    with quiet():                                       # the log of truth 1.3, with and without ax noise
        rec = C.run_stage("downforce", dict(log, ax=0.2 * np.random.default_rng(3).standard_normal(5)), "Sturn",
                          physics="corrected")
        flat = C.run_stage("downforce", log, "Sturn", physics="corrected")
    ok(f"downforce stage, ax noise 0.2 m/s^2: apply Cl_scale {rec['apply'].get('Cl_scale', np.nan):.3f} "
       f"within 4 % of 1.3, gamma {rec['identifiable']['gamma']:.2f}, qss_ratio {rec['qss_ratio']}",
       rec["identifiable"]["ok"] and abs(rec["apply"]["Cl_scale"] - 1.3) < 0.04 * 1.3 and rec["qss_ratio"] is not None)
    ok("downforce stage, ax 0 everywhere (a zero column): gamma inf, the gate refuses, apply {}",
       flat["identifiable"]["gamma"] == np.inf and flat["apply"] == {} and flat["qss_ratio"] is None)
else:
    print("  SKIP corrected equilibria (casadi missing)")

print("6. run_stage and the ledger: grip then drag in one file, grip frozen into drag, stale marks")
with tempfile.TemporaryDirectory() as tmp:
    led = os.path.join(tmp, "bcn.json")
    car = make_ctx("BCN", Cd_scale=1.2, LMUX=0.93, LMUY=0.93)
    lg_grip = qss_log("BCN", noise=(0.05, 0.02, 0.02), seed=7, LMUX=0.93, LMUY=0.93)
    lg_drag = straight_log(car, 0.8, 30.0, 75.0, sv=0.05, st=0.01, sa=0.03, seed=3)
    with quiet():
        g = C.run_stage("grip", lg_grip, "BCN", ledger=led)
        d = C.run_stage("drag", lg_drag, "BCN", ledger=led)
    ok(f"grip: apply LMUY {g['apply']['LMUY']:.4f} within 0.01 of 0.93, qss_ratio {g['qss_ratio']:.4f} above 1",
       abs(g["apply"]["LMUY"] - 0.93) < 0.01 and g["qss_ratio"] > 1)
    ok(f"grip: identifiable ok (gamma {g['identifiable']['gamma']:.2f})", g["identifiable"]["ok"])
    ok(f"drag: Cd_scale {d['fit']['Cd_scale']:.3f} within 0.06 of 1.2, grip LMUY frozen in base_overrides",
       abs(d["fit"]["Cd_scale"] - 1.2) < 0.06 and d["base_overrides"].get("LMUY") == g["apply"]["LMUY"])
    L = load_strict(led)
    ok("ledger: grip then drag, none stale, strict JSON (no NaN, no Infinity)",
       L is not None and [st["stage"] for st in L["stages"]] == ["grip", "drag"]
       and not any(st.get("stale") for st in L["stages"]))
    with quiet():
        g2 = C.run_stage("grip", lg_grip, "BCN", ledger=led)
    L = load_strict(led)
    ok("grip run again: drag marked stale, the new grip not",
       L is not None and [st.get("stale", False) for st in L["stages"]] == [False, True, False])
    fr = C._frozen(led, "downforce")
    ok(f"frozen values for downforce: the new grip, no Cd_scale of the stale drag ({sorted(fr)})",
       "Cd_scale" not in fr and fr["LMUY"] == g2["apply"]["LMUY"])
    with quiet():
        g3 = C.run_stage("grip", slow, "BCN", ledger=led)
    ok("a later grip run that identifies nothing (30 % of the limit): apply {}, it wins, frozen values for drag {}",
       g3["apply"] == {} and C._frozen(led, "drag") == {})
    ok("a ledger of one car refuses another physics mode (ValueError)",
       raises(ValueError, C.run_stage, "grip", lg_grip, "BCN", ledger=led, physics="corrected"))
    ok("a ledger of one car refuses four motors, another config (ValueError: drag_fit reads 4 x throttle there)",
       raises(ValueError, C.run_stage, "grip", lg_grip, "BCN", ledger=led, Electric_4Motors="On", ATD="Off"))
    ok("the record keeps config and the userOpts kwargs, and the ledger key holds config",
       g["config"] == "Static_ATDOn_EM4Off" and g["useropts"] == {}
       and load_strict(led)["key"] == ["BCN", "Static_ATDOn_EM4Off", "matlab", "MF205"])
    led2 = os.path.join(tmp, "refused.json")
    with quiet():
        d0 = C.run_stage("drag", dict(t=np.arange(100) * 0.02, vx=np.full(100, 50.0), Tm=np.zeros(100),
                                      ay=np.full(100, 5.0)), "BCN", ledger=led2)
    ok("a drag stage that the gate refuses (no straight window): apply {}, qss_ratio None, strict JSON ledger",
       not d0["identifiable"]["ok"] and d0["apply"] == {} and d0["qss_ratio"] is None and load_strict(led2) is not None)
ok("an unknown stage: ValueError", raises(ValueError, C.run_stage, "balance", lg_grip, "BCN"))

print("6b. run_stage grip paths: link, the lap ratio after the accelerations, the gate")
lg_split = qss_log("BCN", LMUX=0.95, LMUY=1.05)
with quiet():
    a = C.run_stage("grip", lg_split, "BCN")
    b = C.run_stage("grip", lg_split, "BCN", link=False)
ok(f"link=True: LMUX = LMUY = {a['apply']['LMUY']:.4f}, the gate tests LMUY alone (gamma 1)",
   a["apply"]["LMUX"] == a["apply"]["LMUY"] and abs(a["identifiable"]["gamma"] - 1.0) < 1e-12)
ok(f"link=False: LMUX {b['apply']['LMUX']:.4f} within 0.03 of 0.95, LMUY {b['apply']['LMUY']:.4f} within 0.01 "
   f"of 1.05, the gate tests both (gamma {b['identifiable']['gamma']:.2f})",
   abs(b["apply"]["LMUX"] - 0.95) < 0.03 and abs(b["apply"]["LMUY"] - 1.05) < 0.01
   and set(b["identifiable"]["se"]) == {"LMUY", "LMUX"} and b["identifiable"]["ok"])


def fake_fit(ly, lx):
    """A grip_fit stand-in: LMUY ly and LMUX lx on 30 limit samples in 3 corners."""
    return lambda log, ctx, **kw: dict(LMUY=ly, LMUX=lx, n_y=30, n_x=30, n_events=3, why_y="", why_x="",
                                       rms_before=np.nan, rms_after=np.nan, vt=np.linspace(15.0, 45.0, 30),
                                       vb=np.linspace(20.0, 40.0, 30), ab=np.full(30, 9.0))


stub_log, V9 = dict(vx=np.ones(5), ay=np.ones(5)), (18.24010, 18.01633)
real_fit, real_idn = C.grip_fit, C.identifiable
try:
    C.grip_fit = fake_fit(np.nan, 0.9)
    with quiet():
        r_lap = C.run_stage("grip", stub_log, "Sturn", lap=V9)
        r_none = C.run_stage("grip", stub_log, "Sturn")
    ok(f"LMUY nan, LMUX 0.9: the lap ratio sets LMUY {r_lap['apply']['LMUY']:.4f} (method lap), no lap: apply {{}}",
       r_lap["method"] == "lap" and abs(r_lap["apply"]["LMUY"] - 0.95) < 0.002 and r_none["apply"] == {}
       and r_none["method"] is None)
    C.grip_fit = fake_fit(0.95, np.nan)
    with quiet():
        r = C.run_stage("grip", stub_log, "Sturn", link=False)
    ok(f"link=False, LMUX not identified: apply {r['apply']} (LMUX keeps its base value)",
       r["method"] == "accel" and r["apply"] == {"LMUY": 0.95})
    C.grip_fit = fake_fit(0.95, 0.9)
    C.identifiable = lambda *args, **kw: dict(gamma=99.0, se={}, never=[], ok=False)
    with quiet():
        r_lap = C.run_stage("grip", stub_log, "Sturn", lap=V9)
        r_none = C.run_stage("grip", stub_log, "Sturn")
    ok(f"the gate refuses the accelerations: the lap ratio sets LMUY {r_lap['apply']['LMUY']:.4f}, no lap: apply {{}}",
       r_lap["method"] == "lap" and abs(r_lap["apply"]["LMUY"] - 0.95) < 0.002 and r_none["apply"] == {})
finally:
    C.grip_fit, C.identifiable = real_fit, real_idn

# =============================================================================================
print("7. nlp_forecast: stand-in gradients, then run V9 (Sturn, LMUX = LMUY = 0.95: +0.22377 s solved), no solve")
import functions.sensitivity as SE
real_sg, SE.setup_gradients = SE.setup_gradients, lambda data, params, **kw: {
    "LMUY": -3.0, "LMUX": -1.0, "INFLPRES": 1.5e-6, "alpha_FL": -4.6e-4}
try:
    f0 = C.nlp_forecast("stored.mat", {"LMUY": 0.9, "LMUX": 0.9})
    f1 = C.nlp_forecast("stored.mat", {"LMUY": 0.9, "LMUX": 0.9}, vp_overrides={"LMUY": 0.95})
    f2 = C.nlp_forecast("stored.mat", {"INFLPRES": 242000.0, "alpha_FL": 11.0})
finally:
    SE.setup_gradients = real_sg
ok(f"gradients -3 and -1 s: forecast {f0:+.2f} s from 1.0, {f1:+.2f} s with LMUY 0.95 in vp_overrides",
   abs(f0 - 0.4) < 1e-12 and abs(f1 - 0.25) < 1e-12)
ok(f"knobs with a default other than 1: INFLPRES 220000 -> 242000 Pa, alpha_FL 10 -> 11 deg give {f2:+.5f} s "
   "(the steps from the model values)", abs(f2 - (1.5e-6 * 22000.0 - 4.6e-4)) < 1e-12)
if HAVE_CA:
    import functions.transcription as T

    def no_solver(*args, **kw):
        raise RuntimeError("nlp_forecast must not solve")

    real, T._make_solver = T._make_solver, no_solver
    try:
        with quiet():
            f = C.nlp_forecast(STURN, {"LMUX": 0.95, "LMUY": 0.95}, circuit="Sturn")
    finally:
        T._make_solver = real
    ok(f"first-order forecast {f:+.5f} s within 3 % of the solved +0.22377 s",
       abs(f - 0.22377) < 0.03 * 0.22377)
else:
    print("  SKIP nlp_forecast (casadi missing)")

print("\nALL correlate TESTS PASSED")
