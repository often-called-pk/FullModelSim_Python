"""ChassisSim phase 2 task 6: suspension kinematics, the tyre table 4 decode and the dynamic camber.

Covers functions/chassissim.axle_kinematics and suspension_curves, the tyre table 4 read in _tyre_decode,
the dynamic camber and the Fx camber factor of vehModel_cs, and the QSS envelope of two cars. The synthetic
checks use physical hardpoints (left side: x rearward, y left +, z up) written as the car file holds them.
The model checks build a synthetic car with MLTP (a fake IPOPT, no solve) and MLTP_screen (no solve).
The vendor section (last) runs only when CHASSISSIM_DIR names the ChassisSim install folder. It reads the
GP3 2013 car and its log and prints SKIP otherwise. Nothing is written into the vendor folders.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_kinematics.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_kinematics.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import sys
import tempfile
import warnings

import casadi as ca
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import functions.transcription as T
from functions import carfile
from functions.chassissim import _tyre_decode, axle_kinematics, load_car, suspension_curves
from functions.ggv import build_envelope
from functions.telemetry import read_ld
from MLTP import MLTP
from MLTP_screen import MLTP_screen

PASS = 0
G = 9.81
CORNERS = ("fl", "fr", "rl", "rr")
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


# ---- synthetic geometry (physical) -------------------------------------------------------------------
# Left side, car frame: x rearward, y left +, z up. Rows: lower inner front, lower inner aft, lower outer,
# upper inner front, upper inner aft, upper outer. The right side is the mirror in y.
HP_LEFT = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.68, 0.11],
                    [-0.15, 0.30, 0.28], [0.15, 0.30, 0.28], [0.0, 0.64, 0.33]])
HP_LOW = HP_LEFT.copy()
HP_LOW[5, 2] = 0.25                  # upper outer lower: another camber gain
HP_RIGHT = HP_LEFT * np.array([1.0, -1.0, 1.0])
TRACK = 1.5
RC_STATIC = 0.03

# parallel, equal-length horizontal arms: the upper outer straight above the lower outer
HP_PARALLEL = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.65, 0.10],
                        [-0.15, 0.25, 0.30], [0.15, 0.25, 0.30], [0.0, 0.65, 0.30]])


def circle_pts(c0, r0, c1, r1):
    """The two intersections of two circles in the plane (independent of functions/chassissim)."""
    d = c1 - c0
    dn = np.hypot(*d)
    a = (r0 ** 2 - r1 ** 2 + dn ** 2) / (2 * dn)
    hh = np.sqrt(r0 ** 2 - a ** 2)
    m = c0 + a * d / dn
    perp = np.array([-d[1], d[0]]) / dn
    return [m + s * hh * perp for s in (1.0, -1.0)]


def fd_camber_gain(hp, track, zg, h=1e-6):
    """Camber gain dcamber/dw [rad/m] at w = 0, by a central difference of the upright angle. The upright is
    rigid, the contact patch is fixed to it and w = patch height - zg (bump positive). Computed from the
    hardpoints alone, with the lower arm turned by +-h."""
    li, lo = hp[:2, 1:].mean(0), hp[2, 1:]
    ui, uo = hp[3:5, 1:].mean(0), hp[5, 1:]
    L1, L2, Lu = np.hypot(*(lo - li)), np.hypot(*(uo - ui)), np.hypot(*(uo - lo))
    th0 = np.arctan2(*(lo - li)[::-1])

    def upright(dl):
        LO = li + L1 * np.array([np.cos(th0 + dl), np.sin(th0 + dl)])
        cands = circle_pts(ui, L2, LO, Lu)
        UO = min(cands, key=lambda p: np.hypot(*(p - uo)))
        u = UO - LO
        da = np.arctan2(-u[0], u[1]) - np.arctan2(-(uo - lo)[0], (uo - lo)[1])
        R = np.array([[np.cos(da), -np.sin(da)], [np.sin(da), np.cos(da)]])
        cp = LO + R @ (np.array([track / 2, zg]) - lo)
        return da, cp[1] - zg
    da_p, w_p = upright(h)
    da_m, w_m = upright(-h)
    return (da_p - da_m) / (w_p - w_m)


print("[cs_kinematics] synthetic kinematics")
PAR = axle_kinematics(HP_PARALLEL, TRACK, RC_STATIC, name="parallel")
wp = np.linspace(-0.03, 0.03, 61)
ok(f"parallel equal arms: camber change over +-30 mm below 1e-9 rad (max {np.max(np.abs(PAR.camber(wp))):.2e})",
   bool(np.max(np.abs(PAR.camber(wp))) < 1e-9))
ok(f"parallel equal arms: above-ground RC at w = 0 below 1e-9 m (|{PAR.rc_h(0.0):.2e}|)",
   bool(abs(PAR.rc_h(0.0)) < 1e-9))

SYN = axle_kinematics(HP_LEFT, TRACK, RC_STATIC, name="synthetic")
ok(f"synthetic set: ground datum zg {SYN.zg * 1e3:.3f} mm (expected 5.4 mm)", abs(SYN.zg - 0.0054) < 1e-4)
ok(f"synthetic set: above-ground static RC {SYN.rc_h(0.0) * 1e3:.2f} mm (expected 24.6 mm)",
   abs(SYN.rc_h(0.0) - 0.0246) < 1e-4)
chassis_rc = SYN.rc_h(0.0) + SYN.zg          # rc_h is above the contact patch: chassis RC = above-ground RC + zg
ok(f"synthetic set: chassis-frame RC at w = 0 equals rc_static (|{chassis_rc - RC_STATIC:.2e}| m)",
   abs(chassis_rc - RC_STATIC) < 1e-9)
ok(f"synthetic set: above-ground RC at w = 0 equals chassis RC - zg (|{SYN.rc_h(0.0) - (RC_STATIC - SYN.zg):.2e}| m)",
   abs(SYN.rc_h(0.0) - (RC_STATIC - SYN.zg)) < 1e-9)

gain_fd = fd_camber_gain(HP_LEFT, TRACK, SYN.zg)
gain_fit = SYN.camber.deriv()(0.0)
ok(f"synthetic set: camber gain at w = 0 {np.degrees(gain_fit) * 1e-3:.5f} deg/mm equals the finite difference "
   f"{np.degrees(gain_fd) * 1e-3:.5f} (within 2 %)", abs(gain_fit - gain_fd) <= 0.02 * abs(gain_fd))

# analytic: the instant centre of the upright (arm lines meet) and the contact patch; dtheta/dz = 1 / (lateral offset)
li, lo, ui, uo = HP_LEFT[:2, 1:].mean(0), HP_LEFT[2, 1:], HP_LEFT[3:5, 1:].mean(0), HP_LEFT[5, 1:]
t_ic = np.linalg.solve(np.array([lo - li, -(uo - ui)]).T, ui - li)
IC = li + t_ic[0] * (lo - li)
patch = np.array([TRACK / 2, SYN.zg])
swing = np.hypot(*(patch - IC))
ok(f"synthetic set: camber gain {abs(gain_fit):.5f} rad/m equals 1 / swing-arm length {1 / swing:.5f} (within 3 %)",
   abs(abs(gain_fit) - 1.0 / swing) <= 0.03 / swing)

print("[cs_kinematics] unsolvable and flat-datum geometries")
UNSOLV = HP_LEFT.copy()
UNSOLV[2] = [0.0, 0.27, 0.10]        # lower arm 2 cm long: no solution at 50 mm travel
try:
    axle_kinematics(UNSOLV, TRACK, RC_STATIC, name="front")
    raised = ""
except ValueError as e:
    raised = str(e)
ok("unsolvable geometry raises ValueError naming the front axle", "front axle" in raised)

li0, lo0, ui0 = HP_LEFT[0:2, 1:].mean(0), HP_LEFT[2, 1:], HP_LEFT[3:5, 1:].mean(0)
P0 = li0 + (0.0 - li0[0]) / (lo0[0] - li0[0]) * (lo0 - li0)     # lower arm line on the centreline
uo_flat = ui0 + (0.64 - ui0[0]) / (0.0 - ui0[0]) * (P0 - ui0)   # upper arm line through the same centreline point
FLAT = HP_LEFT.copy()
FLAT[5, 1:] = uo_flat
try:
    axle_kinematics(FLAT, TRACK, RC_STATIC, name="rear")
    raised = ""
except ValueError as e:
    raised = str(e)
ok("instant centre on the centreline raises ValueError naming the rear axle", "rear axle" in raised)


print("[cs_kinematics] tyre table 4 decode")
DIMS4 = (3, 2, 3, 2, 2, 3, 2)     # d1..d7: table 4 has d6 = 3 cambers and d7 = 2 loads


def tyre_sizes(d):
    """f64 count of the 13 tables of a tyre (carfile docstring: 2D(k) = 2k, 3D(a, b) = a + b + a b)."""
    d1, d2, d3, d4, d5, d6, d7 = d

    def g(a, b):
        return a + b + a * b
    return [2 * d3, 2 * d4, 2 * d4, g(d1, d2), g(d6, d7), 2 * d6, 2 * d4, g(d1, d2), 2 * d3, 2 * d6, 2 * d5, 2 * d5, 2 * d5]


def tyre_tables(d, t4):
    """The 13 tables of a tyre with table 4 = t4 and every other table arbitrary."""
    tabs = [arb(k, j) for j, k in enumerate(tyre_sizes(d))]
    tabs[4] = np.asarray(t4, float)
    return tabs


d1, d2, d3, d4, d5, d6, d7 = DIMS4
CAM4 = np.array([-4.0, 0.0, 4.0])      # deg
LOAD4 = np.array([1000.0, 3000.0])     # N
GRID4 = np.array([[10.0 * i + j + 1.0 for j in range(d7)] for i in range(d6)])     # row = camber, column = load
sc = np.arange(18, dtype=float) + 0.5
sc[1], sc[5], sc[6], sc[9], sc[11] = -1.0, 2.0e5, 325.0, 0.12, 3.0
tb = tyre_tables(DIMS4, np.concatenate([CAM4, LOAD4, GRID4.ravel()]))
tb[3] = np.concatenate([1000.0 * np.arange(1, d1 + 1), [300.0, 350.0], arb(d1 * d2, 7)])
tb[10] = np.concatenate([np.linspace(150.0, 250.0, d5), 1.5e5 * (1.0 + 0.01 * np.arange(d5))])
dec = _tyre_decode((list(DIMS4), sc, tb))
ok("table 4 camber axis is read in rad (degrees converted)",
   np.allclose(dec["fx_camber_x"], np.radians(CAM4), rtol=0, atol=1e-15))
ok("table 4 load axis is read in N", np.array_equal(dec["fx_camber_load"], LOAD4))
ok("table 4 grid keeps the file orientation (row = camber, column = load)",
   dec["fx_camber"].shape == (d6, d7) and np.array_equal(dec["fx_camber"], GRID4))
ok("table 4 element (camber 4 deg, load 3000 N) reads 22", dec["fx_camber"][2, 1] == 22.0)
short = list(tb)
short[4] = tb[4][:-1]
try:
    _tyre_decode((list(DIMS4), sc, short))
    short_raised = ""
except ValueError as e:
    short_raised = str(e)
ok("a table 4 of the wrong size raises ValueError naming table 4", "table 4" in short_raised)


# ---- synthetic car with physical hardpoints (model and QSS checks) ----------------------------------
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
    rows = [f"{float(k)!r}   {float(s)!r}" for k, s in zip(K, S)]
    write_text(os.path.join(circ, "tr_synth.txt"), "\r\n".join(rows) + "\r\n")
    return root


INSTALL = build_install(os.path.join(TMP.name, "ChassisSim"))
DIMS = (3, 2, 3, 2, 2, 5, 4)      # d6 = 5 cambers (includes 0 deg, the T5 optimum), d7 = 4 loads
TYRE_TEMPS = (300.0, 350.0)
TYRE_BG = 325.0
PRESSURE = 2.1e5
RADIUS = {"fl": 0.300, "fr": 0.305, "rl": 0.310, "rr": 0.315}
DRAG = {"fl": 0.011, "fr": 0.012, "rl": 0.0135, "rr": 0.014}
KZ = {"fl": 1.60e5, "fr": 1.62e5, "rl": 1.40e5, "rr": 1.42e5}
VEL = np.array([0.0, 100.0, 200.0, 400.0, 800.0])
BUMP = np.array([0.0, 900.0, 1800.0, 2700.0, 3600.0])
REB = np.array([0.0, 600.0, 1200.0, 1800.0, 2400.0])
SPRING = {1: (1.23e5, 0.020), 2: (1.31e5, 0.021), 3: (0.97e5, 0.019), 4: (1.02e5, 0.0205),
          5: (9.0e4, 0.0), 6: (8.0e4, 0.0)}
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}
ARB = {19: 2.0e4, 21: 1.5e4}
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: TRACK, 92: TRACK, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,
    100: 0.9, 102: 1000.0, 103: 9000.0,
    107: 14000.0, 108: 0.6, 109: 0.05,
    111: 0.0, 112: 1.0, 113: 1.0,
    55: RC_STATIC, 59: RC_STATIC, 63: RC_STATIC, 67: RC_STATIC,
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,
    390: 350.0,
}
EXPLICIT.update({7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.010, 12: 0.010})
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
AERO_X = 0.005 * np.arange(7)
AERO_Z = 0.002 + 0.005 * np.arange(7)
MAPS = {114: (4, 4), 116: (4, 4)}


def tyre_record(i, corner, camber_deg):
    """Tyre record (tag 13): T4 = 1 + 0.03 camber[deg] + 1e-5 load[N], T5 = 1 - 0.01 camber[deg]^2 (peak at 0 deg),
    slip curve peak 0.8 (under the 0.9 lateral cap, so the lateral and longitudinal curves are one curve)."""
    d1, d2, d3, _, d5, d6, d7 = DIMS
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(DIMS))]
    load = np.linspace(1000.0, 25000.0, d1)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, arb(d1 * d2, 20 + i)])
    tabs[0] = np.concatenate([np.linspace(0.0, 2.0, d3), np.array([0.0, 0.8, 0.8])[:d3]])
    cam = np.linspace(-6.0, 6.0, d6)
    tabs[5] = np.concatenate([cam, 1.0 - 0.01 * cam ** 2])
    loads4 = 1000.0 * np.arange(1, d7 + 1)
    tabs[4] = np.concatenate([cam, loads4, (1.0 + 0.03 * cam[:, None] + 1e-5 * loads4[None, :]).ravel()])
    px = np.linspace(150.0, 250.0, d5)
    tabs[10] = np.concatenate([px, KZ[corner] * (1.0 + 0.01 * np.arange(d5))])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = 0.0, camber_deg, 2.0, RADIUS[corner], PRESSURE, TYRE_BG
    sc[7], sc[8], sc[9], sc[11], sc[15] = DRAG[corner], 0.11, 0.12, 3.0, 1.05
    return 13, 80 + i, (list(DIMS), sc, tabs)


def build_records(camber_deg=0.0, hp_left=HP_LEFT):
    hp_right = hp_left * np.array([1.0, -1.0, 1.0])
    R = []

    def add(tag, cid, v):
        R.append((tag, cid, v))
    for cid, v in EXPLICIT.items():
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
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    for cid in (127, 131, 135):
        add(9, cid, (AERO_X, AERO_Z, arb(49, cid)))
    for cid, (nx, nz) in MAPS.items():
        add(9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid)))
    for cid in range(31, 36):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    for cid, rows in ((49, hp_left), (50, hp_right), (51, hp_left), (52, hp_right)):
        add(10, cid, (0, np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])))
    for i, c in enumerate(CORNERS):
        add(*tyre_record(i, c, camber_deg))
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


def write_car(name, camber_deg=0.0, hp_left=HP_LEFT):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(camber_deg, hp_left), path)
    return path


def build_model(car_path):
    """MLTP on the synthetic install with a fake IPOPT (no solve). Returns ctx with ctx.m23."""
    return quiet(MLTP, car=f"chassissim:{car_path}", circuit=f"chassissim:{CIRC}", chassissim_dir=INSTALL,
                 physics="corrected", save=False, plot=False, linear_solver="mumps")


def qss_ctx(car_path):
    """MLTP_screen on the synthetic install (no solve)."""
    return quiet(MLTP_screen, car=f"chassissim:{car_path}", circuit=f"chassissim:{CIRC}", chassissim_dir=INSTALL,
                 physics="corrected", save=False, verbose=False)


def state(ctx, vx=20.0, vy=0.0, sx=0.0, zs=0.0, theta=0.0, phi=0.0):
    """Physical state on a straight: speed vx, lateral speed vy, wheel slip sx, heave, pitch and roll. Tyre states at W0 / kt."""
    vp, cs = ctx.vp, ctx.cs
    x = np.zeros(23)
    x[0], x[1] = vx, vy
    x[5] = x[6] = (1.0 + sx) * vx / cs.tyre.fl["radius"]
    x[7] = x[8] = (1.0 + sx) * vx / cs.tyre.rl["radius"]
    x[9], x[11], x[13] = zs, theta, phi
    x[19:23] = [getattr(vp, f"W{c}0") / vp.kt for c in CORNERS]
    return x


def evaluate(ctx, names, x):
    """Values of the named model outputs (attributes of ctx.m23) at the physical state x, straight line."""
    m = ctx.m23
    f = ca.Function("f", [m.x, m.u, m.pv], [ca.vertcat(*[ca.SX(getattr(m, n)) for n in names])])
    out = np.array(f(np.asarray(x, float) / m.x_s, np.zeros(m.nu), 0.0), float).ravel()
    return dict(zip(names, out))


def smooth_clamp(v, lo, hi, e):
    """The smooth clamp of vehModel_cs.poly_sx (_clamp): v kept inside [lo, hi], offset ~e^2 / (4 distance) inside."""
    def smax(a, b):
        return 0.5 * (a + b + np.sqrt((a - b) ** 2 + e * e))

    def smin(a, b):
        return 0.5 * (a + b - np.sqrt((a - b) ** 2 + e * e))
    return smin(smax(v, lo + e), hi - e)


CS_CORNERS = {"fl": "f", "fr": "f", "rl": "r", "rr": "r"}
names = [f"{p}_{c}" for c in CORNERS for p in ("xs", "dynamic_camber", "fx", "fy", "sa", "sx", "fz")]

print("[cs_kinematics] model: dynamic camber and the Fx camber factor (fake IPOPT, no solve)")
BASE = write_car("base.car", camber_deg=0.0)
ctx = build_model(BASE)
curves = suspension_curves(ctx.cs)
cs = ctx.cs
x_roll = state(ctx, zs=0.004, theta=0.01, phi=0.02)
v = evaluate(ctx, names, x_roll)
for c in CORNERS:
    kc = curves.f if CS_CORNERS[c] == "f" else curves.r
    static = getattr(cs.tyre, c)["camber"]
    xs = v[f"xs_{c}"]
    roll = x_roll[13] if c[1] == "l" else -x_roll[13]      # camber from the body roll (vehModel_cs.py)
    expect = static + kc.camber(smooth_clamp(xs, kc.lo, kc.hi, 2e-3)) + roll     # poly_sx keeps xs inside the range
    if c[1] == "r":
        expect = -expect                 # the channel of a right corner is mirrored (vehModel_cs.py)
    ok(f"dynamic camber channel {c} = static + camber poly(xs) + roll camber at the roll state "
       f"(|err| {abs(v[f'dynamic_camber_{c}'] - expect):.2e} rad)",
       abs(v[f"dynamic_camber_{c}"] - expect) < 1e-12)
ok(f"roll moves the front dynamic cambers apart ({np.degrees(v['dynamic_camber_fl']):.3f} vs "
   f"{np.degrees(v['dynamic_camber_fr']):.3f} deg)", abs(v["dynamic_camber_fl"] - v["dynamic_camber_fr"]) > 1e-3)

# Fx / Fy at one slip vector: the factor T4(dynamic camber, load) times the ratio of the normalised slips
x_slip = state(ctx, vx=20.0, vy=0.5, sx=0.05, zs=0.004, theta=0.01, phi=0.02)
w = evaluate(ctx, names, x_slip)
worst = 0.0
for c in CORNERS:
    fze = 0.5 * (w[f"fz_{c}"] + np.sqrt(w[f"fz_{c}"] ** 2 + 40.0 ** 2))     # the model's load floor (smooth max, e = 40 N)
    cam = w[f"dynamic_camber_{c}"] * (-1.0 if c[1] == "r" else 1.0)     # undo the mirror of the channel
    t4 = 1.0 + 0.03 * np.degrees(cam) + 1e-5 * fze
    tyre = getattr(cs.tyre, c)
    ratio = t4 * (w[f"sx_{c}"] / tyre["peak_slip_ratio"]) / (w[f"sa_{c}"] / tyre["peak_slip_angle"])
    worst = max(worst, abs(w[f"fx_{c}"] / w[f"fy_{c}"] - ratio) / abs(ratio))
ok(f"Fx / Fy at one slip vector = T4 at the dynamic camber and load, normalised slips (max rel err {worst:.1e})",
   worst < 1e-6)

print("[cs_kinematics] model: camber factor of table 5 enters Fy")
CAM0 = build_model(write_car("cam0.car", camber_deg=0.0))
CAM3 = build_model(write_car("cam3.car", camber_deg=-3.0))
x_static = state(CAM0, vx=20.0, vy=0.5)
fy0 = evaluate(CAM0, ["fy_fl", "dynamic_camber_fl"], x_static)
fy3 = evaluate(CAM3, ["fy_fl", "dynamic_camber_fl"], x_static)
ratio_t5 = fy3["fy_fl"] / fy0["fy_fl"]
ok(f"a camber of -3 deg (away from the table 5 optimum at 0 deg) lowers Fy by {100 * (1 - ratio_t5):.2f} % "
   f"(expected 9 %)", ratio_t5 < 1.0 and abs(ratio_t5 - 0.91) < 0.005)

print("[cs_kinematics] QSS: camber gain changes the envelope")
ctx_base = qss_ctx(BASE)
env_a = build_envelope(ctx_base)
gain_a = suspension_curves(ctx_base.cs).f.camber.deriv()(0.0)
LOW = write_car("low.car", camber_deg=0.0, hp_left=HP_LOW)
ctx_low = qss_ctx(LOW)
gain_b = suspension_curves(ctx_low.cs).f.camber.deriv()(0.0)
env_b = build_envelope(ctx_low)
ok(f"the two cars have different front camber gains ({np.degrees(gain_a) * 1e-3:.4f} vs "
   f"{np.degrees(gain_b) * 1e-3:.4f} deg/mm)", abs(gain_a - gain_b) > 0.05 * abs(gain_a))
ok(f"build_envelope on the synthetic car runs and gives finite ay_max (peak {env_a.ay_max.max() / G:.3f} g)",
   bool(np.all(np.isfinite(env_a.ay_max))))
d_ay = np.max(np.abs(env_a.ay_max - env_b.ay_max)) / np.max(np.abs(env_a.ay_max))
ok(f"ay_max changes when the camber gain changes (max rel change {d_ay:.2e})", d_ay > 1e-6)

print(f"[cs_kinematics] synthetic: {PASS} checks passed")


print("[cs_kinematics] vendor GP3 2013 (Eastern Creek)")
CS_DIR = os.environ.get("CHASSISSIM_DIR", "")
MODELS = os.path.normpath(os.path.join(CS_DIR, "..", "Models")) if CS_DIR else ""
GP3 = os.path.join(MODELS, "GP3 2013", "Eastern Creek, AU")
CAR_FILE = os.path.join(GP3, "def_GP3 2013.car")
LOG_FILE = os.path.join(GP3, "gp3_baseline_easterncreek.ld")
if not CS_DIR or not os.path.isfile(CAR_FILE) or not os.path.isfile(LOG_FILE):
    print("  SKIP: CHASSISSIM_DIR unset or the GP3 2013 files are missing")
    print(f"[cs_kinematics] {PASS} checks passed (vendor skipped)")
    sys.exit(0)

car = load_car(CAR_FILE)
gc = car.geom
vc = suspension_curves(car)
ch_f, ch_r = vc.f.rc_h(0.0) + vc.f.zg, vc.r.rc_h(0.0) + vc.r.zg     # chassis-frame static RC at w = 0
ok("GP3 static RC: chassis frame equals the stored cids 55 / 59 (front) and 63 / 67 (rear), 1e-9",
   abs(ch_f - gc.rc_height_fl) < 1e-9 and abs(ch_f - gc.rc_height_fr) < 1e-9
   and abs(ch_r - gc.rc_height_rl) < 1e-9 and abs(ch_r - gc.rc_height_rr) < 1e-9)
ok(f"GP3 above-ground static RC: front {vc.f.rc_h(0.0) * 1e3:.2f} mm (18.05), rear {vc.r.rc_h(0.0) * 1e3:.2f} mm (41.41), 0.1 mm",
   abs(vc.f.rc_h(0.0) * 1e3 - 18.05) < 0.1 and abs(vc.r.rc_h(0.0) * 1e3 - 41.41) < 0.1)

LOG_MAP = {
    "s": (("Lap Distance",), "m", 1.0),
    "rc_f": (("RC front",), "mm", 1e-3), "rc_r": (("RC rear",), "mm", 1e-3),
    "zfl": (("Susp Position - Front L",), "mm", 1e-3), "zfr": (("Susp Position - Front R",), "mm", 1e-3),
    "zrl": (("Susp Position - Rear L",), "mm", 1e-3), "zrr": (("Susp Position - Rear R",), "mm", 1e-3),
}
log = read_ld(LOG_FILE, channel_map=LOG_MAP)
lap1 = log["lap"] == 1
for axle, k, zl, zr, rc in (("front", vc.f, log["zfl"], log["zfr"], log["rc_f"]),
                            ("rear", vc.r, log["zrl"], log["zrr"], log["rc_r"])):
    wv = 0.5 * (zl + zr)
    sel = lap1 & (np.abs(wv) <= 0.05)
    err = k.rc_h(wv[sel]) - rc[sel]
    med = float(np.median(err))
    ok(f"GP3 {axle}: median above-ground RC (model at mean travel minus log) {med * 1e3:+.2f} mm, "
       f"within 7 mm over {int(sel.sum())} lap-1 samples", abs(med) < 7e-3)

gain_rear = np.degrees(vc.r.camber.deriv()(0.0)) * 1e-3
ok(f"GP3 rear camber gain {gain_rear:.5f} deg per wheel mm within 5 % of 0.0238",
   gain_rear > 0 and abs(gain_rear - 0.0238) <= 0.05 * 0.0238)

for c, cam_deg, want in (("fl", 3.8, 0.5668), ("rl", 2.5, 0.8125)):
    tyre = getattr(car.tyre, c)
    ok(f"GP3 {c} tyre static camber is {cam_deg} deg", abs(np.degrees(tyre["camber"]) - cam_deg) < 1e-6)
    axle_share = car.mass.front_share if c[0] == "f" else 1.0 - car.mass.front_share
    load = 0.5 * axle_share * car.mass.total * G
    t4 = RegularGridInterpolator((tyre["fx_camber_x"], tyre["fx_camber_load"]), tyre["fx_camber"])
    got = float(t4([[tyre["camber"], load]])[0])
    ok(f"GP3 {c} table 4 at {cam_deg} deg = {got:.5f} (stored {want}, 1e-4)", abs(got - want) < 1e-4)

wgrid = np.linspace(-0.05, 0.05, 201)
for axle, k in (("front", vc.f), ("rear", vc.r)):
    rc_v, cam_v = k.rc_h(wgrid), k.camber(wgrid)
    ok(f"GP3 {axle}: curves finite over +-50 mm travel", bool(np.all(np.isfinite(rc_v)) and np.all(np.isfinite(cam_v))))
    step_rc = float(np.max(np.abs(np.diff(rc_v))))
    step_cam = float(np.max(np.abs(np.diff(cam_v))))
    ok(f"GP3 {axle}: curves smooth (largest step per 0.5 mm: RC {step_rc * 1e3:.3f} mm, camber "
       f"{np.degrees(step_cam):.4f} deg)", step_rc < 1e-3 and np.degrees(step_cam) < 0.02)

print(f"[cs_kinematics] vendor: {PASS} checks passed in total")
