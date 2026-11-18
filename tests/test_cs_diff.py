"""ChassisSim phase 2 task 7: rear differentials of a ChassisSim car (vehModel_cs.py, D8).

The rear drive force P (T_drive) splits between the two rear wheels. The faster wheel loses T_x to the slower
one. Types: 1 open, 2 super (split by the Fx peaks), 3 spool, 4 custom map, 5 Detroit. The checks build the
synthetic car once per type with MLTP and a fake IPOPT (no solve), then evaluate the model outputs at chosen
states. The vendor section (last) runs only when CHASSISSIM_DIR names the ChassisSim install folder. It
builds the GP3 2013 car on Eastern Creek, reads its rear map and writes nothing into the vendor folders.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_diff.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_diff.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import tempfile
import warnings
from pathlib import Path

import casadi as ca
import numpy as np

import functions.transcription as T
from functions import carfile
from MLTP import MLTP
from vehModel_cs import K_SPOOL, _lock_map, drive_scale

PASS = 0
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"
CORNERS = ("fl", "fr", "rl", "rr")
NAMES = ["T_rl", "T_rr", "T_drive", "T_brake", "mu_rl_x", "mu_rr_x", "fz_rl", "fz_rr"]
ZERO_TOL = 0.05     # N, the one-way and deadband floor: the map clamp width is 0.1 N, so its zero sits within 0.05 N


def ok(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASS += 1


def quiet(fn, *a, **kw):
    """Call fn with stdout and warnings swallowed, so the output stays short."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def fake(ca_, nlp, opts, info=None):
    """Stand-in for functions.transcription._make_solver: returns the start point, no iterations."""
    def solver(**k):
        return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()), "lam_x": np.zeros(nlp["x"].numel())}
    solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
    if info is not None:
        info["linear_solver"] = "fake"
    return solver


T._make_solver = fake


def arb(n, s):
    """n arbitrary finite numbers, different for each seed s."""
    return np.sin(np.arange(n) * 0.9 + s) * 7.3 - 0.61


# ---- synthetic car: physical hardpoints, rc heights 0.03 m, zero bump steer, flat table 4 -------------
# Left side, car frame: x rearward, y left +, z up. Right side is the mirror in y.
HP_LEFT = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.68, 0.11],
                    [-0.15, 0.30, 0.28], [0.15, 0.30, 0.28], [0.0, 0.64, 0.33]])
HP_RIGHT = HP_LEFT * np.array([1.0, -1.0, 1.0])
TRACK = 1.5
RC_STATIC = 0.03
INI = ("[Total Circ prop]\r\nTotal circuits = 1\r\n\r\n"
       f"[Circuit0]\r\nCircuit name = {CIRC}\r\nCircuit file name = tr_synth.txt\r\nBase Grip factor = 1.0\r\n"
       "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n")
S_TRACK = np.linspace(0.0, 600.0, 121)
K_TRACK = (2.0 * np.pi / 600.0) * (1.0 + 0.5 * np.sin(2.0 * np.pi * S_TRACK / 600.0))
DIMS = (3, 2, 3, 2, 2, 5, 4)      # d6 = 5 cambers, d7 = 4 loads (table 4 has >= 4 points per axis)
TYRE_TEMPS = (300.0, 350.0)
TYRE_BG = 325.0
PRESSURE = 2.1e5
RADIUS = {"fl": 0.300, "fr": 0.305, "rl": 0.310, "rr": 0.315}
DRAG = {"fl": 0.011, "fr": 0.012, "rl": 0.0135, "rr": 0.014}
KZ = {"fl": 1.60e5, "fr": 1.62e5, "rl": 1.40e5, "rr": 1.42e5}
VEL = np.array([0.0, 100.0, 200.0, 400.0, 800.0])
BUMP = np.array([0.0, 900.0, 1800.0, 2700.0, 3600.0])
REB = np.array([0.0, 600.0, 1200.0, 1800.0, 2400.0])
BUMP_X = np.linspace(-0.02, 0.02, 11)
SPRING = {1: (1.23e5, 0.020), 2: (1.31e5, 0.021), 3: (0.97e5, 0.019), 4: (1.02e5, 0.0205),
          5: (0.0, 0.0), 6: (0.0, 0.0)}
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}
ARB = {19: 2.0e4, 21: 1.5e4}
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: TRACK, 92: TRACK, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,
    100: 0.9, 102: 1000.0, 103: 9000.0,
    107: 14000.0, 108: 0.6, 109: 0.05,
    111: 0.0, 112: 1.0, 113: 0.0,
    55: RC_STATIC, 59: RC_STATIC, 63: RC_STATIC, 67: RC_STATIC,
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,
    390: 350.0,
}
EXPLICIT.update({7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.0, 12: 0.0})   # third gaps 0: no third element
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
AERO_X = 0.005 * np.arange(7)
AERO_Z = 0.002 + 0.005 * np.arange(7)
# rear differential map (cid 116): force axis N, wheel-speed difference axis m/s, data[ix, iz] in N, as the
# ChassisSim GP3 map: 0 for force <= 0 and for speed < 0.1 m/s, else 0.1 x force
MAP_X = np.array([-10000.0, -5000.0, -3333.0, -1667.0, 0.0, 1667.0, 3333.0, 5000.0, 6667.0, 8333.0, 10000.0])
MAP_Z = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
MAP_DATA = np.where((MAP_X > 0)[:, None] & (MAP_Z >= 0.1)[None, :], 0.1 * MAP_X[:, None], 0.0)


def tyre_sizes(d):
    """f64 count of the 13 tables of a tyre (carfile docstring: 2D(k) = 2k, 3D(a, b) = a + b + a b)."""
    d1, d2, d3, d4, d5, d6, d7 = d

    def g(a, b):
        return a + b + a * b
    return [2 * d3, 2 * d4, 2 * d4, g(d1, d2), g(d6, d7), 2 * d6, 2 * d4, g(d1, d2), 2 * d3, 2 * d6, 2 * d5, 2 * d5, 2 * d5]


def tyre_record(i, corner):
    """Tyre record (tag 13). Table 4 (Fx / Fy peak ratio) is flat 1.0, so the Fx peak equals the Fy peak."""
    d1, d2, d3, _, d5, d6, d7 = DIMS
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(DIMS))]
    load = np.linspace(1000.0, 25000.0, d1)
    grid = (1.4 * load[:, None] + 100.0 * np.arange(d2)[None, :]).ravel()    # positive peak, rises with load
    tabs[3] = np.concatenate([load, TYRE_TEMPS, grid])
    tabs[0] = np.concatenate([np.linspace(0.0, 2.0, d3), np.array([0.0, 0.8, 0.8])[:d3]])
    cam = np.linspace(-6.0, 6.0, d6)
    tabs[5] = np.concatenate([cam, 1.0 - 0.01 * cam ** 2])
    loads4 = 1000.0 * np.arange(1, d7 + 1)
    tabs[4] = np.concatenate([cam, loads4, np.ones(d6 * d7)])
    px = np.linspace(150.0, 250.0, d5)
    tabs[10] = np.concatenate([px, KZ[corner] * (1.0 + 0.01 * np.arange(d5))])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = 0.0, 0.0, 2.0, RADIUS[corner], PRESSURE, TYRE_BG
    sc[7], sc[8], sc[9], sc[11], sc[15] = DRAG[corner], 0.11, 0.12, 3.0, 1.05
    return 13, 80 + i, (list(DIMS), sc, tabs)


def build_records(rear_type, omit=()):
    """Records of the synthetic car. rear_type goes to cid 113; cid 113 is left out when it is in omit."""
    R = []

    def add(tag, cid, v):
        if v is not None and cid not in omit:
            R.append((tag, cid, v))
    vals = dict(EXPLICIT)
    vals[113] = rear_type
    for cid, v in vals.items():
        add(1, cid, v)
    for cid, (rate, pre) in SPRING.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.05]), np.array([0.0, 500.0])))
    for cid, (rate, pre) in RUBBER.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.02]), np.array([0.0, 800.0])))
    for cid, rate in ARB.items():
        add(11, cid, (1, 0.0, rate, np.array([0.0, 0.1]), np.array([0.0, 300.0])))
    for cid in range(23, 31):
        add(2, cid, np.array([1.0 + 0.05 * np.sin(cid), 0.0, 0.0, 0.0, 0.0]))
    for cid in range(366, 372):
        add(6, cid, 10 + cid - 366)
    for cid in range(204, 208):
        add(6, cid, 1)
    for cid in (74, 75, 76, 77):
        add(7, cid, (BUMP_X, np.zeros(len(BUMP_X))))      # bump steer, all zero
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    add(9, 114, (0.5 * np.arange(4), 1.0 * np.arange(4) + 0.1, arb(16, 114)))
    add(9, 116, (MAP_X, MAP_Z, MAP_DATA.ravel()))         # C order: the reader reshapes to (nx, nz)
    for cid in (127, 131, 135):
        add(9, cid, (AERO_X, AERO_Z, arb(49, cid)))
    for cid in range(31, 36):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    for cid, rows in ((49, HP_LEFT), (50, HP_RIGHT), (51, HP_LEFT), (52, HP_RIGHT)):
        add(10, cid, (0, np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])))
    for i, c in enumerate(CORNERS):
        add(*tyre_record(i, c))
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


def write_car(name, rear_type, omit=()):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(rear_type, omit), path)
    return path


def write_install(root):
    circ = os.path.join(root, "Circuits")
    os.makedirs(circ)
    with open(os.path.join(root, "Circ_prop.ini"), "w", encoding="latin-1", newline="") as f:
        f.write(INI)
    rows = [f"{float(k)!r}   {float(s)!r}" for k, s in zip(K_TRACK, S_TRACK)]
    with open(os.path.join(circ, "tr_synth.txt"), "w", encoding="latin-1", newline="") as f:
        f.write("\r\n".join(rows) + "\r\n")
    return root


INSTALL = write_install(os.path.join(TMP.name, "ChassisSim"))


def build(car_path, chassis_dir=INSTALL, circuit=CIRC):
    """MLTP on a car with a fake IPOPT (no solve). Returns ctx with ctx.m23 and ctx.cs."""
    return quiet(MLTP, car=f"chassissim:{car_path}", circuit=f"chassissim:{circuit}", chassissim_dir=chassis_dir,
                 physics="corrected", save=False, plot=False, linear_solver="mumps")


def evaluator(ctx):
    """Return ev(x, P, Tb): the model outputs NAMES at the physical state x, drive force P [N], brake force Tb [N]."""
    m = ctx.m23
    f = ca.Function("f", [m.x, m.u, m.pv], [ca.vertcat(*[ca.SX(getattr(m, n)) for n in NAMES])])

    def ev(x, P, Tb=0.0):
        u = np.array([P / m.u_s[0], Tb / m.u_s[1], 0.0])     # inputs are scaled by the model's u_s
        return dict(zip(NAMES, np.array(f(np.asarray(x, float) / m.x_s, u, 0.0), float).ravel()))
    return ev


def state(ctx, vx=20.0, dv=0.0, phi=0.0, dz=0.0):
    """Physical state on a straight. dv = Rw_r (Om_rl - Om_rr) [m/s], phi roll [rad]. Tyre states at W0 / kt,
    except that dz [N] moves load from the right rear tyre to the left (the tyre load follows its zt state)."""
    vp, cs = ctx.vp, ctx.cs
    Rr = cs.tyre.rl["radius"]
    x = np.zeros(23)
    x[0] = vx
    x[5] = x[6] = vx / cs.tyre.fl["radius"]
    x[7] = vx / Rr + dv / (2 * Rr)
    x[8] = vx / Rr - dv / (2 * Rr)
    x[13] = phi
    x[19:23] = [getattr(vp, f"W{c}0") / vp.kt for c in CORNERS]
    x[21] = (vp.Wrl0 + dz) / vp.kt
    x[22] = (vp.Wrr0 - dz) / vp.kt
    return x


def rear(ev, ctx, P, dv, phi=0.0, dz=0.0):
    """Rear wheel forces [N] (no brake) and the raw outputs. F_rl = T_rl / Rw_r."""
    o = ev(state(ctx, dv=dv, phi=phi, dz=dz), P)
    Rr = ctx.cs.tyre.rl["radius"]
    return o["T_rl"] / Rr, o["T_rr"] / Rr, o


def tx_of(ev, ctx, P, dv, phi=0.0):
    """Transfer T_x to the slower wheel [N] = (F_rr - F_rl) / 2 (positive when the left wheel is faster)."""
    fl, fr, _ = rear(ev, ctx, P, dv, phi)
    return 0.5 * (fr - fl)


def lock_grid_range(tab):
    """(min, max) of _lock_map over a dense grid that runs past both axes of the map (clamp case included)."""
    x, z = np.asarray(tab["x"], float), np.asarray(tab["z"], float)
    Pg, dg = np.meshgrid(np.linspace(x[0] - 2000.0, x[-1] + 2000.0, 101), np.linspace(0.0, z[-1] + 1.0, 101))
    Pv, dvv = Pg.ravel(), dg.ravel()
    P, d = ca.SX.sym("P"), ca.SX.sym("d")
    g = ca.Function("g", [P, d], [_lock_map(tab)(P, d)]).map(Pv.size)
    vals = np.array(g(Pv.reshape(1, -1), dvv.reshape(1, -1)), float).ravel()
    return vals.min(), vals.max()


def custom_checks(ctx, label):
    """Item 3 of the brief for a car with rear type 4: sum, sign, mirror, one-way, deadband, locked, clamp."""
    ev = evaluator(ctx)
    tab = ctx.cs.diff.rear_map
    dmax = float(np.max(tab["data"]))
    print(f"[cs_diff] custom map (type 4) on {label}")
    worst = 0.0
    for P in (500.0, 2000.0, 5000.0, 9000.0):
        for dv in (-0.5, -0.12, 0.0, 0.03, 0.2, 1.0):
            fl, fr, _ = rear(ev, ctx, P, dv)
            worst = max(worst, abs(fl + fr - P) / P)
    ok(f"{label}: F_rl + F_rr = P on 24 states (worst rel {worst:.1e})", worst <= 1e-9)
    fl, fr, _ = rear(ev, ctx, 5000.0, 0.3)
    ok(f"{label}: faster left wheel loses at dv +0.3 (F_rl {fl:.1f} < F_rr {fr:.1f} N)", fl < fr)
    fl_m, fr_m, _ = rear(ev, ctx, 5000.0, -0.3)
    ok(f"{label}: mirror symmetry, dv -> -dv swaps F_rl and F_rr (max err {max(abs(fl - fr_m), abs(fr - fl_m)):.2e} N)",
       abs(fl - fr_m) < 1e-9 * 5000.0 and abs(fr - fl_m) < 1e-9 * 5000.0)
    for P in (0.0, -5000.0):
        t = tx_of(ev, ctx, P, 0.3)
        ok(f"{label}: no transfer at P {P:.0f} N (|T_x| {abs(t):.2e} N, floor {ZERO_TOL} N)", abs(t) < ZERO_TOL)
    for dv in (0.03, -0.03):
        t = tx_of(ev, ctx, 5000.0, dv)
        ok(f"{label}: deadband at |dv| 0.03 m/s, |T_x| {abs(t):.3f} N < 2 % of 0.1 P (500 N)", abs(t) < 0.02 * 0.1 * 5000.0)
    for P in (2000.0, 5000.0, 10000.0):
        t = tx_of(ev, ctx, P, 0.5)
        ok(f"{label}: locked at dv 0.5, P {P:.0f}: T_x {t:.1f} N within 2 % of 0.1 P", abs(t - 0.1 * P) <= 0.02 * 0.1 * P)
    worst = max(abs(tx_of(ev, ctx, P, dv) - 0.1 * P) / (0.1 * P)
                for P in (2000.0, 5000.0, 10000.0) for dv in (0.12, 0.3, 1.0, 2.5))
    ok(f"{label}: 0.1 P within 1 % at dv >= 0.12 and P 2000-10000 N (worst {100 * worst:.3f} %)", worst <= 0.01)
    lo, hi = lock_grid_range(tab)
    ok(f"{label}: T_lock >= 0 over the dense clamped grid (min {lo:.3e} N)", lo >= -1e-9)
    ok(f"{label}: T_lock <= max(data) = {dmax:.0f} N over the dense clamped grid (max {hi:.3f} N)", hi <= dmax + 1e-9)


print("[cs_diff] build the synthetic cars (fake IPOPT, no solve)")
BUILD = {}
for t in (1, 2, 3, 4, 5):
    BUILD[t] = build(write_car(f"diff_{t}.car", float(t)))
ev = {t: evaluator(c) for t, c in BUILD.items()}
ok("five synthetic cars build with cid 113 = 1..5", all(BUILD[t].cs.diff.rear_type == float(t) for t in BUILD))

print("[cs_diff] every type splits the drive force")
for t in BUILD:
    worst = 0.0
    for P in (500.0, 2000.0, 5000.0, 9000.0):
        for dv in (-0.5, -0.03, 0.0, 0.2, 1.0):
            fl, fr, _ = rear(ev[t], BUILD[t], P, dv)
            worst = max(worst, abs(fl + fr - P) / P)
    ok(f"type {t}: F_rl + F_rr = P within 1e-9 relative (worst {worst:.1e})", worst <= 1e-9)

print("[cs_diff] open (type 1)")
worst = 0.0
for P in (500.0, 2000.0, 5000.0, 9000.0):
    for dv in (-1.0, -0.3, 0.05, 0.6):
        fl, fr, _ = rear(ev[1], BUILD[1], P, dv)
        worst = max(worst, abs(fl - P / 2) / P, abs(fr - P / 2) / P)
ok(f"open: F_rl = F_rr = P/2 for any dv (worst rel {worst:.1e})", worst <= 1e-9)

custom_checks(BUILD[4], "synthetic")

print("[cs_diff] spool (type 3)")
worst = 0.0
for P in (0.0, 3000.0):
    for dv in (-0.4, -0.05, 0.1, 0.5):
        t = tx_of(ev[3], BUILD[3], P, dv)
        worst = max(worst, abs(t - K_SPOOL * dv) / (K_SPOOL * abs(dv)))
ok(f"spool: T_x = K_SPOOL x dv (K {K_SPOOL:.0f} N s/m, worst rel {worst:.1e})", worst <= 1e-9)
fl, fr, _ = rear(ev[3], BUILD[3], 3000.0, 0.1)
ok("spool: faster left wheel loses (F_rl < F_rr at dv +0.1)", fl < fr)

print("[cs_diff] Detroit (type 5)")
# E_DET = F_ref / (2 K_SPOOL) is the tanh width of type 5 (vehModel_cs.py). F_ref = drive_scale of the car.
F_REF = float(drive_scale(BUILD[5].cs, BUILD[5].cs.tyre.rl["radius"]))
E_DET = F_REF / (2 * K_SPOOL)
h = 1e-6
slope0 = (tx_of(ev[5], BUILD[5], F_REF, h) - tx_of(ev[5], BUILD[5], F_REF, -h)) / (2 * h)
ok(f"Detroit slope at dv 0 and P = F_ref ({F_REF:.0f} N): {slope0:.0f} N s/m <= K_SPOOL {K_SPOOL:.0f} (1 % margin)",
   slope0 <= 1.01 * K_SPOOL)
for dv in (20.0 * E_DET, 0.5):
    t = tx_of(ev[5], BUILD[5], 5000.0, dv)
    ok(f"Detroit under drive, dv {dv:.3f} m/s ({dv / E_DET:.0f} x E_DET {E_DET:.4f}): |T_x| {t:.1f} N -> 0.5 P = 2500 N",
       abs(t) / (0.5 * 5000.0) > 0.999 and abs(t - 0.5 * 5000.0) / 2500.0 < 1e-3)
t0 = tx_of(ev[5], BUILD[5], 0.0, 0.5)
ok(f"Detroit under coast (P = 0): T_x = 0 (|T_x| {abs(t0):.2e} N)", abs(t0) < 1e-9)

print("[cs_diff] super (type 2)")
ev2, car2 = ev[2], BUILD[2]
fl, fr, o = rear(ev2, car2, 5000.0, 0.3, phi=0.02, dz=300.0)     # 300 N of rear load moved to the left tyre
fze = {c: 0.5 * (o[f"fz_{c}"] + np.sqrt(o[f"fz_{c}"] ** 2 + 40.0 ** 2)) for c in ("rl", "rr")}   # the model's load floor
Crl, Crr = o["mu_rl_x"] * fze["rl"], o["mu_rr_x"] * fze["rr"]
ok(f"super: the rear loads are unequal at this state (fz_rl {o['fz_rl']:.0f}, fz_rr {o['fz_rr']:.0f} N)",
   abs(o["fz_rl"] - o["fz_rr"]) > 0.01 * o["fz_rl"])
ok(f"super: F_c / C_c equal for both rear wheels (rel {abs(fl / Crl - fr / Crr) / (fl / Crl):.1e})",
   abs(fl / Crl - fr / Crr) <= 1e-9 * abs(fl / Crl))
ok(f"super: C_c = Fx peak of the corner (C_rl {Crl:.0f}, C_rr {Crr:.0f} N, both > 0)", Crl > 0 and Crr > 0)
ok(f"super: F_rl + F_rr = P at the unequal loads (err {abs(fl + fr - 5000.0):.1e} N)",
   abs(fl + fr - 5000.0) <= 1e-9 * 5000.0)

print("[cs_diff] unknown differential types")
for label, rt, omit, tok in (("0", 0.0, (), "differential type 0.0"), ("6", 6.0, (), "differential type 6.0"),
                             ("None", None, (113,), "lacks the fields ['diff.rear_type']")):
    msg = ""
    try:
        build(write_car(f"diff_bad_{label}.car", rt, omit))
    except ValueError as e:
        msg = str(e)
    ok(f"rear type {label}: ValueError naming '{tok}' ({msg[:60]!r})", tok in msg)

print("[cs_diff] open-diff graph unchanged")
car1, ev1 = BUILD[1], ev[1]
bias = car1.cs.brake.front_bias
Rr = car1.cs.tyre.rl["radius"]
worst = 0.0
for P, Tb, dv, phi in ((0.0, 0.0, 0.0, 0.0), (1500.0, 0.0, 0.4, 0.0), (6000.0, -4000.0, -0.2, 0.02),
                       (3000.0, -9000.0, 0.0, -0.02)):
    o = ev1(state(car1, dv=dv, phi=phi), P, Tb)
    exp_rl = P * Rr / 2 + Tb * (1 - bias) / 2 * Rr
    worst = max(worst, abs(o["T_rl"] - exp_rl) / max(1.0, abs(exp_rl)), abs(o["T_rr"] - exp_rl) / max(1.0, abs(exp_rl)))
ok(f"open: T_rl = T_rr = T_drive Rw/2 + rear brake part at 4 states (worst rel {worst:.1e})", worst <= 1e-9)

print(f"{PASS} checks passed (synthetic)")


# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs_dir):
    print("[cs_diff] vendor GP3 2013 Eastern Creek")
    car_path = Path(cs_dir).parent / "Models" / "GP3 2013" / "Eastern Creek, AU" / "def_GP3 2013.car"
    ok("GP3 2013 Eastern Creek car file found", car_path.is_file())
    ctx = build(car_path, chassis_dir=cs_dir, circuit="Eastern Creek")
    ok(f"GP3 rear differential type is 4 (read {ctx.cs.diff.rear_type})", ctx.cs.diff.rear_type == 4.0)
    custom_checks(ctx, "GP3")
    ev_g = evaluator(ctx)
    worst = 0.0
    for P, dv in ((2000.0, 0.3), (5000.0, -0.6), (9000.0, 0.0)):
        fl, fr, _ = rear(ev_g, ctx, P, dv)
        worst = max(worst, abs(fl + fr - P) / P)
    ok(f"GP3 model: F_rl + F_rr = P at 3 states (worst rel {worst:.1e})", worst <= 1e-9)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
print(f"{PASS} checks passed")
