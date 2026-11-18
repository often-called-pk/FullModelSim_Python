"""vehModel_cs.py: the ChassisSim car model and its build path (ChassisSim phase 2, task 3).

A synthetic car file (arbitrary values, written with carfile.write; the cids and record tags are literal numbers
here) and a synthetic ChassisSim install with one short circuit go through MLTP(car=..., circuit=...,
physics='corrected') with a fake IPOPT (no solve). Checks: the build (sizes, input keys, path rows), the
refusals, the static point, left-right symmetry on a straight, the front bar law, the gear envelope, the 4EM
default NLP snapshot (subprocess), and the GP3 2013 car on Eastern Creek when CHASSISSIM_DIR names the install
(else SKIP). Nothing is written into the vendor folders.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_vehmodel_cs.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_vehmodel_cs.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import copy
import io
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import casadi as ca
import numpy as np

import functions.transcription as T
from functions import carfile
from MLTP import MLTP
import vehModel_cs as V

PASS = 0
CORNERS = ("fl", "fr", "rl", "rr")
G = 9.81
ROWS_VERT = [10, 12, 14, 15, 16, 17, 18]     # zsdot, thetadot, phidot and the wheel rows (vertical states)
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"


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


def raises(exc, fn, *a, **kw):
    try:
        quiet(fn, *a, **kw)
    except exc:
        return True
    return False


def fake(ca_, nlp, opts, info=None):
    """Stand-in for functions.transcription._make_solver: returns the start point, no iterations."""
    def solver(**k):
        return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()), "lam_x": np.zeros(nlp["x"].numel())}
    solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
    if info is not None:
        info["linear_solver"] = "fake"
    return solver


T._make_solver = fake


# ---- synthetic ChassisSim install: one circuit, 600 m, one lap of turning -----------------------------
S = np.linspace(0.0, 600.0, 121)
K = (2.0 * np.pi / 600.0) * (1.0 + 0.5 * np.sin(2.0 * np.pi * S / 600.0))
INI = ("[Total Circ prop]\r\nTotal circuits = 1\r\n\r\n"
       f"[Circuit0]\r\nCircuit name = {CIRC}\r\nCircuit file name = tr_synth.txt\r\nBase Grip factor = 1.0\r\n"
       "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n")


def write_text(path, text):
    with open(path, "w", encoding="latin-1", newline="") as f:
        f.write(text)


def build_install(root):
    circ = os.path.join(root, "Circuits")
    os.makedirs(circ)
    write_text(os.path.join(root, "Circ_prop.ini"), INI)
    rows = [f"{float(k)!r}   {float(s)!r}" for k, s in zip(K, S)]    # 'k s' rows as the ChassisSim files
    write_text(os.path.join(circ, "tr_synth.txt"), "\r\n".join(rows) + "\r\n")
    return root


INSTALL = build_install(os.path.join(TMP.name, "ChassisSim"))


# ---- synthetic car: literal cids and tags (as in tests/test_cs_car.py) -------------------------------
# Tags: 1 scalar, 2 array, 6 byte, 7 curve, 9 map, 10 hardpoints, 11 spring-like, 12 damper, 13 tyre.
TYRE = {  # dims (d1..d7), radius m, toe deg (0: a toe force jacks the body through a 2 m roll centre), camber, castor, drag, kz N/m
    "fl": dict(dims=(3, 2, 3, 2, 2, 2, 2), radius=0.300, toe=0.0, camber=-1.0, castor=2.0, drag=0.011, kz=1.60e5),
    "fr": dict(dims=(2, 2, 3, 2, 2, 3, 2), radius=0.305, toe=0.0, camber=-1.5, castor=2.5, drag=0.012, kz=1.62e5),
    "rl": dict(dims=(4, 2, 3, 3, 2, 2, 2), radius=0.310, toe=0.0, camber=-0.5, castor=1.5, drag=0.0135, kz=1.40e5),
    "rr": dict(dims=(2, 2, 3, 2, 3, 2, 3), radius=0.315, toe=0.0, camber=-0.75, castor=1.0, drag=0.014, kz=1.42e5),
}
TYRE_TEMPS = (300.0, 350.0)
TYRE_BG = 325.0
PRESSURE = 2.1e5
VEL = np.array([0.0, 100.0, 200.0, 400.0, 800.0])        # mm/s, from 0 up (the damper curve starts at the origin)
BUMP = np.array([0.0, 900.0, 1800.0, 2700.0, 3600.0])    # N, zero at zero velocity
REB = np.array([0.0, 600.0, 1200.0, 1800.0, 2400.0])     # N, positive as the file stores it
SPRING = {1: (1.23e5, 0.020), 2: (1.31e5, 0.021), 3: (0.97e5, 0.019), 4: (1.02e5, 0.0205),
          5: (9.0e4, 0.0), 6: (8.0e4, 0.0)}
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}
ARB = {19: 2.0e4, 21: 1.5e4}
HARD = (49, 50, 51, 52)
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: 1.52, 92: 1.49, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,            # ride heights, mm
    100: 0.9, 102: 1000.0, 103: 9000.0,                     # efficiency, idle and limit rpm
    107: 14000.0, 108: 0.6, 109: 0.05,                      # brake force N, front bias, delay s
    111: 0.0, 112: 1.0, 113: 0.0,                           # torque split (0 = rear drive), diff types
    55: 0.060, 59: 0.065, 63: 0.095, 67: 0.100,            # roll-centre heights per corner, m
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,                # CLA, CDA, offset, rho
    390: 350.0,
}
EXPLICIT.update({c: 0.5 + 0.01 * c for c in range(7, 13)})            # suspension gaps
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})        # DRS
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
AERO_X = 0.005 * np.arange(7)            # rear ride height nodes, m
AERO_Z = 0.002 + 0.005 * np.arange(7)    # front ride height nodes, m
MAPS = {114: ((4, 4), "diff_f"), 116: ((4, 4), "diff_r")}   # casadi bspline: >= 4 per axis
MIRROR = ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
          (80, 81), (82, 83), (117, 118), (119, 120))                  # left cid -> right cid


def arb(n, s):
    """n arbitrary finite numbers, different for each seed s."""
    return np.sin(np.arange(n) * 0.9 + s) * 7.3 - 0.61


def tyre_sizes(d):
    """f64 count of the 13 tables of a tyre (carfile docstring: 2D(k) = 2k, 3D(a, b) = a + b + a b)."""
    d1, d2, d3, d4, d5, d6, d7 = d

    def g(a, b):
        return a + b + a * b
    return [2 * d3, 2 * d4, 2 * d4, g(d1, d2), g(d6, d7), 2 * d6, 2 * d4, g(d1, d2), 2 * d3, 2 * d6, 2 * d5, 2 * d5, 2 * d5]


def tyre_record(i, corner):
    t = TYRE[corner]
    dims = t["dims"]
    d1, d2, d3, _, d5, d6, _ = dims
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(dims))]
    A, B = arb(d1, 20 + i), arb(d1, 30 + i)
    load = 1000.0 * np.arange(1, d1 + 1)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, np.stack([A, B], axis=1).ravel()])
    tabs[0][:d3] = np.linspace(0.0, 2.0, d3)                 # slip abscissae: from 0, increasing
    tabs[0][d3:] = np.array([0.0, 0.97, 0.97])[:d3]          # slip values: from 0, flat after the knee at 1
    tabs[5][:d6] = np.linspace(-3.0, 3.0, d6)                # camber abscissae (deg), increasing
    px = np.linspace(150.0, 250.0, d5)
    py = t["kz"] * (1.0 + 0.01 * np.arange(d5))
    tabs[10] = np.concatenate([px, py])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = t["toe"], t["camber"], t["castor"], t["radius"], PRESSURE, TYRE_BG
    sc[7], sc[8], sc[9], sc[11], sc[15] = t["drag"], 0.11 + 0.01 * i, 0.12 + 0.01 * i, 0.7 + 0.1 * i, 1.05 + 0.1 * i
    return 13, 80 + i, (list(dims), sc, tabs)


def mirror_car(R, toe=0.0):
    """Copy each left record onto its right partner and set every toe [deg] to toe: a left-right symmetric car."""
    idx = {c: i for i, (_, c, _) in enumerate(R)}
    for left, right in MIRROR:
        t, _, v = R[idx[left]]
        R[idx[right]] = (t, right, copy.deepcopy(v))
    for i, (t, c, v) in enumerate(R):
        if t == 13:
            dims, sc, tabs = v
            sc = np.array(sc, float)
            sc[0] = toe
            R[i] = (t, c, (dims, sc, tabs))
    return R


def build_records(arb_front=None, mirror=False, toe=0.0):
    R = []

    def add(tag, cid, v):
        R.append((tag, cid, v))
    for cid, v in EXPLICIT.items():
        add(1, cid, v)
    for cid, (rate, pre) in SPRING.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.05]), np.array([0.0, 500.0])))
    for cid, (rate, pre) in RUBBER.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.02]), np.array([0.0, 800.0])))
    arb_rates = dict(ARB)
    if arb_front is not None:
        arb_rates[19] = arb_front
    for cid, rate in arb_rates.items():
        add(11, cid, (1, 0.0, rate, np.array([0.0, 0.1]), np.array([0.0, 300.0])))
    for cid in range(23, 31):
        add(2, cid, np.array([1.0, 0.1, -0.05, 0.02, 0.0]) * (1 + 0.05 * np.sin(cid)))   # motion ratio, positive
    for cid in range(366, 372):
        add(6, cid, 10 + cid - 366)
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    for cid in (127, 131, 135):
        add(9, cid, (AERO_X, AERO_Z, np.zeros(49) if mirror else arb(49, cid)))   # the mirror car has no aero: no lift at the static point
    for cid, ((nx, nz), _) in MAPS.items():
        add(9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid)))
    for cid in range(31, 36):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    for cid in HARD:
        add(10, cid, (0, arb(18, cid)))
    for i, c in enumerate(CORNERS):
        add(*tyre_record(i, c))
    if mirror:
        R = mirror_car(R, toe)
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


def write_car(name, arb_front=None, mirror=False, toe=0.0):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(arb_front, mirror, toe), path)
    return path


def build(car_path, **kw):
    """MLTP on the synthetic install with a fake IPOPT. Keyword arguments override the defaults."""
    opts = dict(car=f"chassissim:{car_path}", circuit=f"chassissim:{CIRC}", chassissim_dir=INSTALL,
                physics="corrected", save=False, plot=False, linear_solver="mumps")
    opts.update(kw)
    return quiet(MLTP, **opts)


# ---- model helpers --------------------------------------------------------------------------------------
def static_x(ctx, v=0.5):
    """The static point of the car: straight line at v, zs = theta = phi = 0, tyre states at W0 / kt."""
    vp, cs = ctx.vp, ctx.cs
    x = np.zeros(23)
    x[0] = v
    x[5:7] = v / cs.tyre.fl["radius"]
    x[7:9] = v / cs.tyre.rl["radius"]
    x[19:23] = [getattr(vp, f"W{c}0") / vp.kt for c in CORNERS]
    return x


def rates(ctx, x):
    """Time rates of all 23 states (per s, = d/ds divided by sf) at the physical state x on a straight."""
    m = ctx.m23
    f = ca.Function("f", [m.x, m.u, m.pv], [m.dx, m.sf])
    dxs, sf = f.call([np.asarray(x, float) / m.x_s, np.zeros(m.nu), 0.0])
    return np.asarray(dxs, float).ravel() * m.x_s / float(sf)


def tyre_loads(ctx, x):
    m = ctx.m23
    fz = ca.Function("c", [m.x, m.u, m.pv], [m.fz_fl, m.fz_fr, m.fz_rl, m.fz_rr])
    return np.array([float(v) for v in fz.call([np.asarray(x, float) / m.x_s, np.zeros(m.nu), 0.0])])


def corner_weights(cs):
    """Static corner weights of the car [N]: the front share of the total on the front axle, half per corner."""
    front = cs.mass.front_share * cs.mass.total * G / 2
    rear = (1.0 - cs.mass.front_share) * cs.mass.total * G / 2
    return np.array([front, front, rear, rear])


def roll_stiffness(ctx, h=1e-6):
    """Roll stiffness [N m/rad] = -(d phidot / d phi) x Ix, by a central difference at the static point."""
    x = static_x(ctx)
    xp, xm = x.copy(), x.copy()
    xp[13] += h
    xm[13] -= h
    d = rates(ctx, xp)[14] - rates(ctx, xm)[14]
    return -d / (2 * h) * ctx.vp.I_x


print("[vehmodel_cs] synthetic car: build")
BASE = write_car("base.car")
ctx = build(BASE)
m = ctx.m23
ok("nx 23 and nu 3", m.nx == 23 and m.nu == 3)
ok("input keys exactly T_drive, T_brake, delta", list(ctx.input_keys) == ["T_drive", "T_brake", "delta"])
ok("path row names exactly drive_env, BrTh_1", list(ctx.data.constraints) == ["drive_env", "BrTh_1"])

print("[vehmodel_cs] refusals")
ok("physics='matlab' with a car raises ValueError", raises(ValueError, build, BASE, physics="matlab"))
ok("Electric_4Motors='On' with a car raises ValueError", raises(ValueError, build, BASE, Electric_4Motors="On"))
ok("AeroConfig other than 'Static' with a car raises ValueError", raises(ValueError, build, BASE, AeroConfig="AALB"))

print("[vehmodel_cs] static point")
x0 = static_x(ctx)
d0 = rates(ctx, x0)
ok("f_dyn finite at the static point", bool(np.all(np.isfinite(d0))))
loads = tyre_loads(ctx, x0)
w = corner_weights(ctx.cs)
err = float(np.max(np.abs(loads - w) / w))
ok(f"wheel loads equal the car corner weights within 0.5 %: max {100 * err:.4f} % "
   f"(loads {np.round(loads, 1)} N)", err < 0.005)
vert = np.abs(d0[ROWS_VERT])
ok(f"vertical rows small at the static point (< 1 per s): max {vert.max():.3f}", bool(vert.max() < 1.0))

print("[vehmodel_cs] gear envelope")
Fe = ca.Function("F", [m.x, m.u, m.pv], [m.F_env])
vs = np.linspace(5.0, 60.0, 23)
F = np.array([float(Fe.call([static_x(ctx, v) / m.x_s, np.zeros(m.nu), 0.0])[0]) for v in vs])
eng, Rr = ctx.cs.engine, ctx.cs.tyre.rl["radius"]
bound = float(np.max(eng.efficiency * np.max(eng.torque_curve["y"]) * np.asarray(eng.gears) / Rr)) + 1.0
ok(f"F_env > 0 for v in 5..60 m/s (min {F.min():.1f} N)", bool(np.all(F > 0)))
ok(f"F_env <= max over gears of eff max(T) G / R + 1 N: max F {F.max():.1f} N, bound {bound:.1f} N",
   bool(F.max() <= bound))
F40 = float(Fe.call([static_x(ctx, 40.0) / m.x_s, np.zeros(m.nu), 0.0])[0])
F60 = float(Fe.call([static_x(ctx, 60.0) / m.x_s, np.zeros(m.nu), 0.0])[0])
ok(f"F_env falls at high speed (60 m/s {F60:.1f} N below 40 m/s {F40:.1f} N)", F60 < F40)

print("[vehmodel_cs] symmetry on a straight (left-right mirror car)")
SYM = write_car("sym.car", mirror=True)
cs_sym = build(SYM)
rs = rates(cs_sym, static_x(cs_sym))
ok(f"vy, r and phidot rates zero at vy = r = delta = 0 (|dvy|, |dr|, |dphidot| = "
   f"{abs(rs[1]):.1e}, {abs(rs[2]):.1e}, {abs(rs[14]):.1e} per s)",
   abs(rs[1]) < 1e-9 and abs(rs[2]) < 1e-9 and abs(rs[14]) < 1e-9)

print("[vehmodel_cs] front bar law")
BAR0 = write_car("bar0.car", arb_front=0.0)
ctx_bar0 = build(BAR0)
S_k, S_0 = roll_stiffness(ctx), roll_stiffness(ctx_bar0)
k = ctx.cs.susp.arb_f["rate"]
mr0 = float(np.atleast_1d(ctx.cs.susp.mr_arb_f)[0])
t_f = ctx.cs.geom.track_front
expect = 0.5 * k * mr0 ** 2 * t_f ** 2          # bar = k mr0^2 / 2 per wheel, roll arm t_f (the model's law)
dS = S_k - S_0
ok(f"front bar adds k mr0^2 t_f^2 / 2 to the roll stiffness within 2 %: dS {dS:.4e}, "
   f"expected {expect:.4e} N m/rad", abs(dS - expect) < 0.02 * abs(expect))

print("[vehmodel_cs] 4EM default snapshot")
SNAP = os.path.join(_bootstrap.ROOT, ".superpowers", "sdd", "chassissim", "map", "nlp_snapshot.py")
BASE_JSON = os.path.join(_bootstrap.ROOT, ".superpowers", "sdd", "chassissim", "map", "nlp_base.json")
if not (os.path.isfile(SNAP) and os.path.isfile(BASE_JSON)):
    print("  NOTE: nlp_snapshot.py or nlp_base.json not found, snapshot check skipped")
else:
    out = os.path.join(TMP.name, "snapshot.json")
    run = subprocess.run([sys.executable, SNAP, _bootstrap.ROOT, out, BASE_JSON],
                         capture_output=True, text=True, timeout=3600)
    ok("4EM default NLP snapshot: SNAPSHOT DIFF: none", "SNAPSHOT DIFF: none" in run.stdout)


print("[vehmodel_cs] review pins")
# 1. aero axis order: rear ride height on X, front on Z. Static ride heights 17.75 (front) and 20.75 mm (rear),
#    zs = -0.75 mm gives 17.0 and 20.0 mm, both on the map nodes (rear 0.005 k, front 0.002 + 0.005 k).
x = static_x(ctx, 30.0)
x[9] = -0.00075
lift = float(ca.Function("L", [m.x, m.u, m.pv], [m.f_lift]).call([x / m.x_s, np.zeros(m.nu), 0.0])[0])
cla = ctx.cs.aero.cla_map["data"]
exp_lift = 0.5 * ctx.cs.aero.rho * 30.0 ** 2 * cla[4, 3]
ok(f"lift at the nodes: rear 20 mm on X, front 17 mm on Z (lift {lift:.3f}, map entry {exp_lift:.3f} N)",
   abs(lift - exp_lift) < 1e-2 * abs(exp_lift))

# 2. tyre law, rear left: sa at the peak (sa_n = 1), zero slip ratio: |Fy| = 0.9 F' cam (lateral cap)
rl = ctx.cs.tyre.rl
x = static_x(ctx, 30.0)
x[1] = -30.0 * np.tan(rl["peak_slip_angle"])                 # sa_rl = atan(-vy / vx) = peak
fy, mu, fz, sa, sx = [float(v) for v in ca.Function("T", [m.x, m.u, m.pv], [m.fy_rl, m.mu_rl_x, m.fz_rl, m.sa_rl, m.sx_rl])
                      .call([x / m.x_s, np.zeros(m.nu), 0.0])]
cam = float(np.interp(rl["camber"], rl["camber_x"], rl["camber_y"]))
exp_fy = 0.9 * mu * fz * cam
ok(f"pure lateral at the peak slip angle: |Fy| = 0.9 F' cam ({abs(fy):.4f} vs {exp_fy:.4f} N)",
   abs(abs(fy) - exp_fy) < 1e-3 * exp_fy and abs(sa / rl["peak_slip_angle"] - 1) < 1e-6 and abs(sx) < 1e-9)

# 2b. locked wheel (Om = 0, slip ratio -1): |Fx| <= F' and not zero (no growth past the table end)
x = static_x(ctx, 30.0)
x[1] = 0.0
x[7] = 0.0
fx, mu, fz, sx = [float(v) for v in ca.Function("T", [m.x, m.u, m.pv], [m.fx_rl, m.mu_rl_x, m.fz_rl, m.sx_rl])
                  .call([x / m.x_s, np.zeros(m.nu), 0.0])]
ok(f"locked wheel (SR {sx:.3f}): 0.9 F' <= |Fx| <= F' ({abs(fx):.4f} vs F' {mu * fz:.4f} N)",
   abs(sx + 1) < 1e-9 and 0.9 * mu * fz <= abs(fx) <= mu * fz * (1 + 1e-6))

# 3. brake split under pure brake (T_brake_n = -1, no drive): front / rear wheel torque ratio, left = right
x = static_x(ctx, 30.0)
Tfl, Tfr, Trl, Trr = [float(v) for v in ca.Function("B", [m.x, m.u, m.pv], [m.T_fl, m.T_fr, m.T_rl, m.T_rr])
                      .call([x / m.x_s, np.array([0.0, -1.0, 0.0]), 0.0])]
bias = ctx.cs.brake.front_bias
ratio = bias * ctx.cs.tyre.fl["radius"] / ((1 - bias) * ctx.cs.tyre.rl["radius"])
ok(f"brake torque ratio front/rear = bias Rf / ((1 - bias) Rr) = {ratio:.6f} (model {Tfl / Trl:.6f})",
   abs(Tfl / Trl - ratio) < 1e-9 * abs(ratio))
Rf, Rr = ctx.cs.tyre.fl["radius"], ctx.cs.tyre.rl["radius"]
tot = -ctx.cs.brake.force * (bias * Rf + (1 - bias) * Rr)      # sum of wheel torques = T_brake (bias Rf + (1 - bias) Rr)
ok("front left and front right torques equal, the four sum to the brake torque",
   abs(Tfl - Tfr) < 1e-9 * abs(Tfl) and abs(Tfl + Tfr + Trl + Trr - tot) < 1e-6 * abs(tot))

# 4. symmetric non-zero toe (same toe on both sides, mirror car) on a straight: dvy = dr = 0
cs_toe = build(write_car("sym_toe.car", mirror=True, toe=0.3))
rt = rates(cs_toe, static_x(cs_toe, 30.0))
ok(f"symmetric toe 0.3 deg on a straight: dvy, dr zero (|dvy| {abs(rt[1]):.1e}, |dr| {abs(rt[2]):.1e} per s)",
   abs(rt[1]) < 1e-9 and abs(rt[2]) < 1e-9)

# 5. rubber table (flag 1): force at the travel past the gap (wheel frame: table at mr0 (travel - gap), times mr0)
mr0 = float(ctx.cs.susp.mr_fl[0])
gap = float(ctx.cs.susp.gap_fl)
rub = ctx.cs.susp.rubber_fl
xs = ca.SX.sym("xs")
fr_fun = ca.Function("R", [xs], [V._rubber(ctx.cs, "fl", mr0)(xs)])
f_r = float(fr_fun(gap + 0.01))
exp_r = mr0 * float(np.interp(mr0 * 0.01, np.r_[0.0, rub["x"]], np.r_[0.0, rub["y"]]))
ok(f"rubber table (flag {rub['flag']}) at 10 mm past the gap: {f_r:.2f} N vs table {exp_r:.2f} N",
   rub["flag"] == 1 and abs(f_r - exp_r) < 0.02 * abs(exp_r))

# 6. residual gate: mirror car, toe 0, vertical rows at the static point below 1e-2 per s
rv = np.abs(rates(cs_sym, static_x(cs_sym))[ROWS_VERT])
ok(f"mirror car, toe 0: vertical rows at the static point < 1e-2 per s (max {rv.max():.2e})", rv.max() < 1e-2)

# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs_dir):
    print("[vehmodel_cs] vendor GP3 2013 Eastern Creek")
    car_path = Path(cs_dir).parent / "Models" / "GP3 2013" / "Eastern Creek, AU" / "def_GP3 2013.car"
    ok("GP3 2013 Eastern Creek car file found", car_path.is_file())
    c = build(str(car_path), circuit="chassissim:Eastern Creek", chassissim_dir=cs_dir)
    ok("GP3 builds: nx 23 and nu 3", c.m23.nx == 23 and c.m23.nu == 3)
    x = static_x(c)
    ld = tyre_loads(c, x)
    wt = corner_weights(c.cs)
    e = float(np.max(np.abs(ld - wt) / wt))
    ok(f"GP3 static loads equal the car corner weights within 0.5 %: max {100 * e:.4f} %", e < 0.005)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
print(f"{PASS} checks passed")
