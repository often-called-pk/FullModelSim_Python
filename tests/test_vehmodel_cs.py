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

import time
from types import SimpleNamespace

import functions.transcription as T
from functions import carfile
from MLTP import MLTP
import vehModel_cs as V
from scipy.interpolate import PchipInterpolator

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
GAP = {7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.010, 12: 0.010}   # rubber ground gaps, m (a few mm)
EXPLICIT.update(GAP)
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


def tyre_record(i, corner, toe=None):
    t = TYRE[corner]
    dims = t["dims"]
    d1, d2, d3, _, d5, d6, _ = dims
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(dims))]
    A, B = arb(d1, 20 + i), arb(d1, 30 + i)
    load = np.linspace(1000.0, 25000.0, d1)              # N, up to 25 kN (the load floor is tested at 20 kN)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, np.stack([A, B], axis=1).ravel()])
    tabs[0][:d3] = np.linspace(0.0, 2.0, d3)                 # slip abscissae: from 0, increasing
    tabs[0][d3:] = np.array([0.0, 0.97, 0.97])[:d3]          # slip values: from 0, flat after the knee at 1
    tabs[5][:d6] = np.linspace(-3.0, 3.0, d6)                # camber abscissae (deg), increasing
    px = np.linspace(150.0, 250.0, d5)
    py = t["kz"] * (1.0 + 0.01 * np.arange(d5))
    tabs[10] = np.concatenate([px, py])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = (t["toe"] if toe is None else toe), t["camber"], t["castor"], t["radius"], PRESSURE, TYRE_BG
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


def build_records(arb_front=None, mirror=False, toe=0.0, toes=None, over=None):
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
        add(2, cid, np.array([1.0 + 0.05 * np.sin(cid), 0.0, 0.0, 0.0, 0.0]))   # constant motion ratio, near 1
    for cid in range(366, 372):
        add(6, cid, 10 + cid - 366)
    for cid in range(204, 208):
        add(6, cid, 1)                                   # rubber gap flags: 1 = ground gap
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
        add(*tyre_record(i, c, (toes or {}).get(c)))
    if mirror:
        R = mirror_car(R, toe)
    if over:
        R = [(t, c, over[c]) if c in over else (t, c, v) for t, c, v in R]
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


def write_car(name, arb_front=None, mirror=False, toe=0.0, toes=None, over=None):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(arb_front, mirror, toe, toes, over), path)
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

# 6. residual gate: mirror car, toe 0, vertical rows at the static point below 1e-2 per s
rv = np.abs(rates(cs_sym, static_x(cs_sym))[ROWS_VERT])
ok(f"mirror car, toe 0: vertical rows at the static point < 1e-2 per s (max {rv.max():.2e})", rv.max() < 1e-2)

# 1. slip law: locked-ish wheel at normalised slip ~11 gives |Fx| = fn_end F' (1 %); h(1e-3) = fn'(0) (1 %)
rl = ctx.cs.tyre.rl
x = static_x(ctx, 30.0)
x[7] = -1.6 * 30.0 / rl["radius"]                              # slip ratio -1.6: normalised slip 1.6 / 0.14 ~ 11
fx, mu, fz, sx = [float(v) for v in ca.Function("T", [m.x, m.u, m.pv], [m.fx_rl, m.mu_rl_x, m.fz_rl, m.sx_rl])
                  .call([x / m.x_s, np.zeros(m.nu), 0.0])]
fn_end = float(rl["slip_y"][-1])
ok(f"normalised slip {abs(sx) / rl['peak_slip_ratio']:.1f} >= 10: |Fx| = fn_end F' ({abs(fx):.4f} vs {fn_end * mu * fz:.4f} N)",
   abs(sx / rl["peak_slip_ratio"]) >= 10 and abs(abs(fx) / (fn_end * mu * fz) - 1) < 0.01)
xk, yk = np.asarray(rl["slip_x"], float), np.asarray(rl["slip_y"], float)
odd = PchipInterpolator(np.r_[-xk[::-1], xk[1:]], np.r_[-yk[::-1], yk[1:]])
slope0 = float(odd.derivative()(0.0))
h1 = float(V._slip_curve(rl)(1e-3))
ok(f"h(1e-3) = fn'(0) within 1 % ({h1:.5f} vs {slope0:.5f})", abs(h1 / slope0 - 1) < 0.01)

# 2. load floor: fze at fz = 0 is the floor (finite); at fz = 20 kN the load peak is evaluated at fz = fze, so
#    Fp / mu = fz within 0.1 % (the load table reaches 25 kN; the PCHIP reference is the table itself)
x = static_x(ctx, 30.0)
x[21] = 0.0                                                    # zt_rl = 0: fz = 0
mu0 = float(ca.Function("M", [m.x, m.u, m.pv], [m.mu_rl_x]).call([x / m.x_s, np.zeros(m.nu), 0.0])[0])
fp0 = float(PchipInterpolator(rl["load"], rl["fy_peak"])(rl["load"][0])) * rl["grip"]
fze0 = fp0 / mu0
x = static_x(ctx, 30.0)
x[21] = 20000.0 / ctx.vp.kt                                    # zt_rl: fz = 20 kN
mu20 = float(ca.Function("M", [m.x, m.u, m.pv], [m.mu_rl_x]).call([x / m.x_s, np.zeros(m.nu), 0.0])[0])
fp20 = float(PchipInterpolator(rl["load"], rl["fy_peak"])(20000.0)) * rl["grip"]
fze20 = fp20 / mu20
ok(f"load floor: fze at fz 0 is ~20 N ({fze0:.3f}, finite), at 20 kN fze equals fz ({fze20:.1f} vs 20000)",
   np.isfinite(mu0) and 19.5 < fze0 < 20.5 and abs(fze20 / 20000.0 - 1) < 1e-3)

# 3. slip table with decreasing x raises ValueError
ok("slip table with decreasing x raises ValueError",
   raises(ValueError, V._slip_curve, {"slip_x": [0.0, 2.0, 1.0], "slip_y": [0.0, 1.0, 1.0]}))

# 4. T_brake rate bounds: the 4EM normalised ramp (torque rate / Tbrake_max) times the brake force
ub_ref = float(np.ravel(ctx.c.ub.T_brake)[0]) / ctx.vp.Tbrake_max * ctx.cs.brake.force
lb_ref = float(np.ravel(ctx.c.lb.T_brake)[0]) / ctx.vp.Tbrake_max * ctx.cs.brake.force
ok(f"T_brake rate bounds equal the 4EM ramp x brake force (ub {float(np.ravel(ctx.duk_ub)[1]):.4g} vs {ub_ref:.4g}, "
   f"lb {float(np.ravel(ctx.duk_lb)[1]):.4g} vs {lb_ref:.4g})",
   abs(float(np.ravel(ctx.duk_ub)[1]) - ub_ref) < 1e-9 * abs(ub_ref) and abs(float(np.ravel(ctx.duk_lb)[1]) - lb_ref) < 1e-9 * abs(lb_ref))

# 5. refusals: cid 111 = 0.3 and a gap flag 0 raise ValueError
ok("torque split 0.3 (cid 111) raises ValueError", raises(ValueError, build, write_car("split.car", over={111: 0.3})))
ok("rubber gap flag 0 (cid 204) raises ValueError", raises(ValueError, build, write_car("flag.car", over={204: 0})))

# 6. throttle channel = T_drive / F_env (share of the drive available at this speed)
x = static_x(ctx, 30.0)
thr_expr = dict(m.veh_extra)["throttle"]
thr = float(ca.Function("t", [m.x, m.u, m.pv], [thr_expr]).call([x / m.x_s, np.array([0.4, 0.0, 0.0]), 0.0])[0])
fenv = float(ca.Function("F", [m.x, m.u, m.pv], [m.F_env]).call([x / m.x_s, np.zeros(m.nu), 0.0])[0])
f_ref = V.drive_scale(ctx.cs, ctx.cs.tyre.rl["radius"])
ok(f"throttle = T_drive / F_env: {thr:.6f} vs {0.4 * f_ref / fenv:.6f}",
   abs(thr - 0.4 * f_ref / fenv) < 1e-9)

# 7. toe per wheel: fl 0.5 deg, fr 0.2 deg. On a straight each slip angle carries its own toe
ct = build(write_car("toe.car", toes={"fl": 0.5, "fr": 0.2}))
x = static_x(ct, 30.0)
sa_fl, sa_fr = [float(v) for v in ca.Function("S", [ct.m23.x, ct.m23.u, ct.m23.pv], [ct.m23.sa_fl, ct.m23.sa_fr])
                .call([x / ct.m23.x_s, np.zeros(ct.m23.nu), 0.0])]
t_fl, t_fr = ct.cs.tyre.fl["toe"], ct.cs.tyre.fr["toe"]
ok(f"own toe per wheel: sa_fl = +toe_fl ({sa_fl:.5f} vs {t_fl:.5f}), sa_fr = -toe_fr ({sa_fr:.5f} vs {-t_fr:.5f})",
   abs(sa_fl - t_fl) < 1e-9 and abs(sa_fr + t_fr) < 1e-9 and abs(t_fl - t_fr) > 1e-3)


# 8. clamp: 50 mm below the front grid (and the rear grid) lift and drag stay non-zero and hold the edge value
def aero_at(ctx_, zs):
    mm_ = ctx_.m23
    xx = static_x(ctx_, 30.0)
    xx[9] = zs
    lift, drag = [float(v) for v in ca.Function("A", [mm_.x, mm_.u, mm_.pv], [mm_.f_lift, mm_.f_drag])
                  .call([xx / mm_.x_s, np.zeros(mm_.nu), 0.0])]
    return lift, drag


L1, D1 = aero_at(ctx, -0.06575)           # front -48 mm (grid edge 2 mm), rear -45 mm
L2, D2 = aero_at(ctx, -0.07775)           # front -60 mm, rear -57 mm
ok(f"aero held outside the grid: lift {L1:.2f} / {L2:.2f} N, drag {D1:.3f} / {D2:.3f} N (nonzero, within 1 %)",
   abs(L1) > 1 and abs(D1) > 1e-3 and abs(L1 / L2 - 1) < 0.01 and abs(D1 / D2 - 1) < 0.01)

# 8b. damper beyond the table: the force at 6 m/s is non-zero; the extension is linear (end slope)
mm = ctx.m23
fz_fl = ca.Function("Z", [mm.x, mm.u, mm.pv], [mm.fzs_fl])
x0 = static_x(ctx, 30.0)
F0 = float(fz_fl.call([x0 / mm.x_s, np.zeros(mm.nu), 0.0])[0])


def fzs_at(v):
    xx = static_x(ctx, 30.0)
    xx[15] = v                                 # wheel speed: the damper velocity of the front left
    return float(fz_fl.call([xx / mm.x_s, np.zeros(mm.nu), 0.0])[0]) - F0


g6 = fzs_at(6.0)
slope_a = (fzs_at(3.0) - fzs_at(1.0)) / 2.0
slope_b = (fzs_at(1.02) - fzs_at(1.0)) / 0.02
ok(f"damper at 6 m/s non-zero ({g6:.1f} N); linear end extension (slope {slope_a:.1f} vs {slope_b:.1f} N per m/s)",
   abs(g6) > 1.0 and abs(slope_a / slope_b - 1) < 0.02)

# 9. rubber frame: compression = MR x travel - gap. No force 10 onset widths before the gap; table past it
mr0 = float(ctx.cs.susp.mr_fl[0])
gap = float(ctx.cs.susp.gap_fl)
xs = ca.SX.sym("xs")
fr_fun = ca.Function("R", [xs], [V._rubber(ctx.cs, "fl")(xs)])
f_off = float(fr_fun((gap - 10 * V.W_RUBBER) / mr0))
rub = ctx.cs.susp.rubber_fl
f_on = float(fr_fun((gap + 0.01) / mr0))
exp_on = mr0 * float(np.interp(0.01, np.r_[0.0, rub["x"]], np.r_[0.0, rub["y"]]))
ok(f"rubber frame: {f_off:.4f} N 10 onset widths before the gap; {f_on:.2f} N at 10 mm past it (table {exp_on:.2f} N)",
   abs(f_off) < 0.01 and abs(f_on / exp_on - 1) < 0.02)

# 1. non-constant motion ratio: sm(x) = 0.9 x + 0.25 x^2, mr(x) = 0.9 + 0.5 x on the front left.
#    The spring force is (F0 / mr0 + k sm(x)) mr(x); the rubber force is mr(x) table(sm(x) - gap), both within 1 %.
mr_poly = np.array([0.9, 0.5, 0.0, 0.0, 0.0])
ctx_mr = build(write_car("mr_poly.car", over={23: mr_poly}))
mm_r, vp_r, cs_r = ctx_mr.m23, ctx_mr.vp, ctx_mr.cs
fzs_fl_fun = ca.Function("Z", [mm_r.x, mm_r.u, mm_r.pv], [mm_r.fzs_fl])


def fzs_at_travel(xtrav):
    """fzs_fl at strut travel xtrav (zs = -xtrav, theta = phi = 0, zt static)."""
    xx = static_x(ctx_mr, 30.0)
    xx[9] = -xtrav
    return float(fzs_fl_fun.call([xx / mm_r.x_s, np.zeros(mm_r.nu), 0.0])[0])


def sm_p(x):
    return 0.9 * x + 0.25 * x * x


def mr_p(x):
    return 0.9 + 0.5 * x


F0_r = vp_r.m_c_fl * vp_r.g
k_r = float(cs_r.susp.spring_fl["rate"])
spring_exp = (F0_r / 0.9 + k_r * sm_p(-0.02)) * mr_p(-0.02)
spring_model = fzs_at_travel(-0.02)                    # extension: no rubber, no damper force
ok(f"polynomial MR, spring force at travel -20 mm: {spring_model:.2f} vs (F0/mr0 + k sm) mr = {spring_exp:.2f} N",
   abs(spring_model - spring_exp) < 0.01 * abs(spring_exp))
rub_tab = cs_r.susp.rubber_fl
gap_r = float(cs_r.susp.gap_fl)
spring_exp_p = (F0_r / 0.9 + k_r * sm_p(0.02)) * mr_p(0.02)
rub_model = fzs_at_travel(0.02) - spring_exp_p
rub_exp = mr_p(0.02) * float(np.interp(sm_p(0.02) - gap_r, np.r_[0.0, rub_tab["x"]], np.r_[0.0, rub_tab["y"]]))
ok(f"polynomial MR, rubber force at travel +20 mm: {rub_model:.2f} vs mr table(sm - gap) = {rub_exp:.2f} N",
   rub_exp > 1.0 and abs(rub_model - rub_exp) < 0.01 * abs(rub_exp))

# 2. slip table that ends rising (last point highest): fn and the capped lateral curve stay at or below max(y)
#    and LAT_CAP beyond the table end
Tt = {"slip_x": [0.0, 1.0, 2.0], "slip_y": [0.0, 1.0, 1.2]}
hv_u = V._slip_curve(Tt)
hv_c = V._slip_curve(Tt, V.LAT_CAP)
vs_t = np.linspace(0.01, 9.5, 300)
fn_u = np.array([float(hv_u(v)) * v for v in vs_t])
fn_c = np.array([float(hv_c(v)) * v for v in vs_t])
ok(f"rising slip table: fn <= max(y) = 1.2 everywhere (max {fn_u.max():.7f}; the B-spline rings by "
   f"{(fn_u.max() / 1.2 - 1) * 1e6:.1f} ppm at the knee)", fn_u.max() <= 1.2 * (1 + 1e-5))
ok(f"rising slip table: capped lateral curve <= LAT_CAP beyond the table end (max past 2: {fn_c[vs_t >= 2].max():.4f})",
   fn_c.max() <= V.LAT_CAP * (1 + 1e-5) and fn_c[vs_t >= 2].max() <= V.LAT_CAP * (1 + 1e-5))

# 3. _curve node grid: step inside the table at most half the smallest spacing (2 mm/s here); a close pair
#    (spacing 1e-3 over a range of 3) builds fast, with the node count capped
_orig_interp = ca.interpolant
_grids = []


def _spy_interp(name, solver, grid, values, *a, **k):
    _grids.append(np.asarray(grid[0], float))
    return _orig_interp(name, solver, grid, values, *a, **k)


ca.interpolant = _spy_interp
try:
    xt = np.arange(0.0, 0.8 + 1e-9, 0.002)
    V._curve(xt, np.sin(xt))
    g_in = _grids[-1][(_grids[-1] >= 0.0) & (_grids[-1] <= 0.8)]
    max_step = float(np.max(np.diff(g_in)))
    ok(f"node step inside a 2 mm/s table {max_step * 1e3:.3f} mm <= half the spacing (1 mm)", max_step <= 0.5 * 0.002 + 1e-12)
    xc = np.array([0.0, 1.0, 1.0 + 1e-3, 2.0, 3.0])
    t0 = time.time()
    V._curve(xc, np.cos(xc))
    dt = time.time() - t0
    ok(f"close pair (1e-3 over range 3) builds in {dt:.2f} s with {_grids[-1].size} nodes", dt < 2.0 and _grids[-1].size <= 5100)
finally:
    ca.interpolant = _orig_interp

# 4. cs_seed_inputs takes M and the resistance from prof['env'] (not from the 4EM drag of ladder.lon_model)
mm_s = ctx.m23
prof_a = {"s": np.array([0.0, 100.0]), "v": np.array([20.0, 20.0]), "ax": np.array([1.0, 1.0]),
          "env": SimpleNamespace(M=700.0, F_res=lambda v: 100.0 + 0.5 * np.asarray(v) ** 2)}
prof_b = dict(prof_a, env=SimpleNamespace(M=900.0, F_res=lambda v: 100.0 + 0.5 * np.asarray(v) ** 2))
ua = V.cs_seed_inputs(ctx, mm_s, prof_a, [50.0])
ub = V.cs_seed_inputs(ctx, mm_s, prof_b, [50.0])
ua_exp = (700.0 * 1.0 + 100.0 + 0.5 * 400.0) / mm_s.u_s[0]
ub_exp = (900.0 * 1.0 + 100.0 + 0.5 * 400.0) / mm_s.u_s[0]
ok(f"cs_seed_inputs drive = (M a + c0 + c2 v^2) / u_s from prof env: {ua[0, 0]:.5f} vs {ua_exp:.5f}, "
   f"M 900: {ub[0, 0]:.5f} vs {ub_exp:.5f}",
   abs(ua[0, 0] - ua_exp) < 1e-9 and abs(ub[0, 0] - ub_exp) < 1e-9 and abs(ua[1, 0]) < 1e-12 and abs(ua[2, 0]) < 1e-12)

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
