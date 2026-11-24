"""ChassisSim phase 2 task 6 and lanes K1-K3: suspension kinematics, the hardpoint gate, the loader, the anti-dive pitch arm.

Covers functions/chassissim.front_view, axle_kinematics, manual_axle, suspension_curves, _hardpoint_gate and check_car,
the tyre table 4 read in _tyre_decode, the dynamic camber and the Fx camber factor of vehModel_cs, the live rear axle,
the anti-dive pitch arm of vehModel_cs (side-view heights, cids 55, 56, 63 and 64), and the QSS envelope of two cars.

The ground is z = 0 of the hardpoint frame. The datum pose puts the contact patch on the ground. w0 is the datum ride
height less the static ride height of the axle, and the curves use the wheel travel from the static pose. The synthetic
cars have all side-view heights at 0 unless a check sets them.

The synthetic checks use physical hardpoints (left side: x rearward, y left +, z up) written as the car file holds them.
The model checks build a synthetic car with MLTP (a fake IPOPT, no solve) and MLTP_screen (no solve).
The vendor section (last) runs only when CHASSISSIM_DIR names the ChassisSim install folder. It reads the GP3 2013 car and
its logs, the GT3 car (found by glob) and shipped cars of the install folder. It prints SKIP otherwise. Nothing is written
into the vendor folders.

    python tests/test_cs_kinematics.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_kinematics.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import collections
import contextlib
import glob
import io
import os
import sys
import tempfile
import warnings

import casadi as ca
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

import functions.transcription as T
import vehModel_cs
from functions import carfile
from functions.chassissim import (_tyre_decode, axle_kinematics, bump_steer, check_car, front_view, load_car,
                                  list_cars, manual_axle, suspension_curves)
from functions.ggv import build_envelope
from functions.telemetry import read_ld
from MLTP import MLTP
from MLTP_screen import MLTP_screen

PASS = 0
G = 9.81
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair. An over or omit cid is a file cid.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"
CS_DIR = os.environ.get("CHASSISSIM_DIR", "")
MODELS = os.path.normpath(os.path.join(CS_DIR, "..", "Models")) if CS_DIR else ""
F_SW_HAND = 100.0       # N, the switch width of the anti-dive tanh, written here as the hand value
CAP = {}                # the last NLP of the fake solver: nlp dict and the start point


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
    CAP["nlp"] = nlp

    def solver(**k):
        CAP["x0"] = np.asarray(k["x0"], float).ravel()
        return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()), "lam_x": np.zeros(nlp["x"].numel())}
    solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
    if info is not None:
        info["linear_solver"] = "fake"
    return solver


T._make_solver = fake


def raised_text(fn):
    """The message of the ValueError that fn raises, or '' when it does not raise."""
    try:
        fn()
    except ValueError as e:
        return str(e)
    return ""


def tanh_count(fn):
    """Number of tanh instructions of a casadi Function."""
    return sum(1 for i in range(fn.n_instructions()) if fn.instruction_id(i) == ca.OP_TANH)


def plane_z(p0, p1, p2, x, y):
    """Height at (x, y) of the plane through three points (rows x, y, z). Solved here, not with functions/chassissim."""
    A = np.array([[p[0], p[1], 1.0] for p in (p0, p1, p2)])
    a, b, c = np.linalg.solve(A, np.array([p[2] for p in (p0, p1, p2)]))
    return a * x + b * y + c


# ---- synthetic geometry (physical) -------------------------------------------------------------------
# Left side, car frame: x rearward, y left +, z up. Rows: lower inner front, lower inner aft, lower outer,
# upper inner front, upper inner aft, upper outer. The right side is the mirror in y.
HP_LEFT = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.68, 0.11],
                    [-0.15, 0.30, 0.28], [0.15, 0.30, 0.28], [0.0, 0.64, 0.33]])
HP_LOW = HP_LEFT.copy()
HP_LOW[5, 2] = 0.25                  # upper outer lower: another camber gain
HP_RIGHT = HP_LEFT * np.array([1.0, -1.0, 1.0])
HP_OFF = HP_LEFT.copy()              # outer points off the axle plane x = 0: their z is the height of the arm plane
HP_OFF[2, 0], HP_OFF[5, 0] = 0.06, -0.05
HP_OFF[0, 2] = 0.12                  # tilts the lower arm plane in x (the two inner points differ in z)
HP_EQ = HP_LEFT.copy()               # the two lower inner points with the same x
HP_EQ[1, 0] = HP_EQ[0, 0]
TRACK = 1.5

# parallel, equal-length horizontal arms: the upper outer straight above the lower outer
HP_PARALLEL = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.65, 0.10],
                        [-0.15, 0.25, 0.30], [0.15, 0.25, 0.30], [0.0, 0.65, 0.30]])
HP_UNSOLV = HP_LEFT.copy()
HP_UNSOLV[2] = [0.0, 0.27, 0.10]     # lower arm 2 cm long: no solution at 50 mm travel


def swing_gain(hp, track):
    """Camber gain [rad/m] at the datum pose, from the instant centre of the two arm lines: the upright turns about the
    instant centre IC, so dtheta/dz = 1 / (lateral distance from the patch to IC). The datum pose puts the patch on the
    ground (z = 0). The pose is solved here from the hardpoints (the upright rigid, the upper outer joint on its circle)."""
    li, lo = hp[:2, 1:].mean(0), hp[2, 1:]
    ui, uo = hp[3:5, 1:].mean(0), hp[5, 1:]
    L1, L2, Lu = np.hypot(*(lo - li)), np.hypot(*(uo - ui)), np.hypot(*(uo - lo))
    th0 = np.arctan2(*(lo - li)[::-1])

    def pose(dl):
        LO = li + L1 * np.array([np.cos(th0 + dl), np.sin(th0 + dl)])
        UO = min(circle_pts(ui, L2, LO, Lu), key=lambda p: np.hypot(*(p - uo)))
        u = UO - LO
        da = np.arctan2(-u[0], u[1]) - np.arctan2(-(uo - lo)[0], (uo - lo)[1])
        R = np.array([[np.cos(da), -np.sin(da)], [np.sin(da), np.cos(da)]])
        return LO, UO, LO + R @ (np.array([track / 2, 0.0]) - lo)
    LO, UO, cp = pose(brentq(lambda d: pose(d)[2][1], -0.5, 0.5))
    s, _ = np.linalg.solve(np.array([LO - li, -(UO - ui)]).T, ui - li)
    ic = li + s * (LO - li)
    return 1.0 / abs(cp[0] - ic[0])


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
    """Camber gain dcamber/dw [rad/m] at w = 0, by a central difference of the upright angle. The upright is rigid,
    the contact patch is fixed to it, and w = patch height - zg (bump positive; zg = 0 for w0 = 0). Computed from the
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


print("[cs_kinematics] synthetic kinematics: parallel arms, camber gain, no solution")
PAR = axle_kinematics(HP_PARALLEL, TRACK, 0.0, 0.0, name="parallel axle")
wp = np.linspace(-0.03, 0.03, 61)
ok(f"parallel equal arms: camber change over +-30 mm below 1e-9 rad (max {np.max(np.abs(PAR.camber(wp))):.2e})",
   bool(np.max(np.abs(PAR.camber(wp))) < 1e-9))
ok(f"parallel equal arms: roll centre on the ground at the datum pose (|{PAR.rc_datum:.2e}| m)",
   abs(PAR.rc_datum) < 1e-9)
ok(f"parallel equal arms: roll centre above the patch is 0 at w = 0 (|{PAR.rc_h(0.0):.2e}| m)",
   abs(PAR.rc_h(0.0)) < 1e-9)

SYN = axle_kinematics(HP_LEFT, TRACK, 0.0, 0.0, name="synthetic axle")
gain_fd = fd_camber_gain(HP_LEFT, TRACK, 0.0)
gain_fit = SYN.camber.deriv()(0.0)
ok(f"synthetic set: camber gain at w = 0 {np.degrees(gain_fit) * 1e-3:.5f} deg/mm equals the finite difference "
   f"{np.degrees(gain_fd) * 1e-3:.5f} (within 2 %)", abs(gain_fit - gain_fd) <= 0.02 * abs(gain_fd))
gain_sw = swing_gain(HP_LEFT, TRACK)
ok(f"swing arm: camber gain {np.degrees(gain_sw) * 1e-3:.5f} deg/mm equals 1 / lateral lever to the instant centre "
   f"(|err| {abs(abs(gain_fd) - gain_sw):.1e} rad/m, 0.1 %)", abs(abs(gain_fd) - gain_sw) <= 1e-3 * gain_sw)
ok(f"swing arm: the curve slope {abs(gain_fit):.5f} rad/m is within 2 % of the swing gain",
   abs(abs(gain_fit) - gain_sw) <= 0.02 * gain_sw)
ok(f"synthetic set: roll centre height above the patch at w = 0 is {SYN.rc_h(0.0) * 1e3:.2f} mm and the datum "
   f"roll centre is {SYN.rc_datum * 1e3:.2f} mm above the ground",
   abs(SYN.rc_h(0.0) - SYN.rc_datum) < 1e-9 and 0.02 < SYN.rc_datum < 0.04)

print("[cs_kinematics] synthetic kinematics: no solution, equal inner x")
try:
    axle_kinematics(HP_UNSOLV, TRACK, 0.0, 0.0, name="front axle")
    raised = ""
except ValueError as e:
    raised = str(e)
ok("unsolvable geometry raises ValueError naming the front axle", "front axle" in raised)
ok("equal x of the inner pair raises ValueError naming the cids", "cids 49, 50" in raised_text(
    lambda: axle_kinematics(HP_EQ, TRACK, 0.0, 0.0, name="front axle (cids 49, 50)")))

print("[cs_kinematics] loader: outer points on the arm plane (front_view)")
li, lo, ui, uo = front_view(HP_OFF, 0.0, "front axle")
z_arm = plane_z(HP_OFF[0], HP_OFF[1], HP_OFF[2], 0.0, HP_OFF[2, 1])
ok(f"outer lower point z is the arm plane at the axle plane ({lo[1] * 1e3:.4f} mm, plane {z_arm * 1e3:.4f} mm)",
   abs(lo[1] - z_arm) < 1e-12 and lo[0] == HP_OFF[2, 1])
ok(f"outer lower point z differs from the stored z by {abs(z_arm - HP_OFF[2, 2]) * 1e3:.1f} mm (not raw)",
   abs(z_arm - HP_OFF[2, 2]) > 1e-3)
z_up = plane_z(HP_OFF[3], HP_OFF[4], HP_OFF[5], 0.0, HP_OFF[5, 1])
ok(f"outer upper point z is the upper arm plane at the axle plane (|err| {abs(uo[1] - z_up):.1e} m)",
   abs(uo[1] - z_up) < 1e-12)
ok("inner pairs are cut at the axle plane: midpoints of the two pairs (exact)",
   np.allclose(li, [0.25, 0.11], atol=1e-12) and np.allclose(ui, [0.30, 0.28], atol=1e-12))

print("[cs_kinematics] tyre table 4 decode")
DIMS4 = (3, 2, 3, 2, 2, 3, 2)     # d1..d7: table 4 has d6 = 3 cambers and d7 = 2 loads


def arb(n, s):
    """n arbitrary finite numbers, different for each seed s."""
    return np.sin(np.arange(n) * 0.9 + s) * 7.3 - 0.61


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
ok("a table 4 of the wrong size raises ValueError naming table 4",
   "table 4" in raised_text(lambda: _tyre_decode((list(DIMS4), sc, short))))


# ---- synthetic car with physical hardpoints (model, gate and QSS checks) ----------------------------
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
          5: (0.0, 0.0), 6: (0.0, 0.0)}
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}
ARB = {19: 2.0e4, 21: 1.5e4}
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: TRACK, 92: TRACK, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,
    100: 0.9, 102: 1000.0, 103: 9000.0,
    107: 14000.0, 108: 0.6, 109: 0.05,
    111: 0.0, 112: 1.0, 113: 1.0,
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,
    390: 350.0,
}
EXPLICIT.update({7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.0, 12: 0.0})   # third gaps 0: no third element
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})
# The datum ride heights are the mean static ride heights of each axle, so w0 = 0. The side-view heights (cids 55, 56,
# 59, 60, 63, 64, 67 and 68, anti-dive and anti-squat) are 0: the synthetic car has no pitch arm.
EXPLICIT.update({142: 0.5 * (EXPLICIT[117] + EXPLICIT[118]) * 1e-3, 144: 0.5 * (EXPLICIT[119] + EXPLICIT[120]) * 1e-3})
EXPLICIT.update({c: 0.0 for c in (55, 56, 59, 60, 63, 64, 67, 68)})
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


def hp_block(rows):
    """The value of a hardpoint block (tag 10): the type byte is set by the caller; the points as x, y, z columns."""
    return np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])


def build_records(camber_deg=0.0, hp_left=HP_LEFT):
    hp_right = hp_left * np.array([1.0, -1.0, 1.0])
    R = []

    def add(tag, cid, v):
        cid = FILE_CID.get(cid, cid)
        R.append((tag, cid, v))
    for cid, v in EXPLICIT.items():
        add(1, cid, v)
    for cid in (53, 61, 69):
        add(6, cid, 0)                                   # geometry mode 0 (53, 61), no live axle (69)
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
        add(10, cid, (0, hp_block(rows)))
    for i, c in enumerate(CORNERS):
        add(*tyre_record(i, c, camber_deg))
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


def change(records, edits):
    """Records with the value of each (tag, cid) in edits replaced by the new value. The other records stay."""
    lookup = {(t, c): v for t, c, v in edits}
    return [(t, c, lookup.get((t, c), v)) for t, c, v in records]


def write_variant(name, records):
    path = os.path.join(TMP.name, name)
    carfile.write(records, path)
    return path


def write_car(name, camber_deg=0.0, hp_left=HP_LEFT):
    return write_variant(name, build_records(camber_deg, hp_left))


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


def evaluate(ctx, names, x, u=None):
    """Values of the named model outputs (attributes of ctx.m23) at the physical state x and the inputs u (scaled)."""
    m = ctx.m23
    f = ca.Function("f", [m.x, m.u, m.pv], [ca.vertcat(*[ca.SX(getattr(m, n)) for n in names])])
    u = np.zeros(m.nu) if u is None else np.asarray(u, float)
    out = np.array(f(np.asarray(x, float) / m.x_s, u, 0.0), float).ravel()
    return dict(zip(names, out))


def dx_at(ctx, x, delta_n=0.0):
    """The state derivative m.dx at the physical state x and steer input delta_n (other inputs zero)."""
    m = ctx.m23
    f = ca.Function("f", [m.x, m.u, m.pv], [m.dx])
    return np.array(f(np.asarray(x, float) / m.x_s, np.array([0.0, 0.0, delta_n]), 0.0), float).ravel()


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
ok("the synthetic base car passes the hardpoint gate and check_car", raised_text(lambda: check_car(load_car(BASE))) == "")
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
    cam = w[f"dynamic_camber_{c}"]
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

print("[cs_kinematics] manual roll centre (mode 1): the curves of manual_axle")
MAN = manual_axle(0.12, 0.003)
ok(f"manual axle: roll centre slope in travel is -1 exactly ({MAN.rc_h.deriv()(0.0):+.1f})", MAN.rc_h.deriv()(0.0) == -1.0)
ok(f"manual axle: roll centre at w = 0 is the stored height plus the shift ({MAN.rc_h(0.0) * 1e3:.1f} mm, 123.0)",
   abs(MAN.rc_h(0.0) - 0.123) < 1e-12)
ok("manual axle: camber is the static camber only (zero at every travel)",
   bool(np.all(MAN.camber(np.linspace(-0.05, 0.05, 11)) == 0.0)))

print("[cs_kinematics] hardpoint gate: refusals name the cid (synthetic records, no solve)")
GATE_CASES = (
    ("a live axle on the front axle", [(10, 49, (2, hp_block(HP_RIGHT))), (10, 50, (2, hp_block(HP_LEFT)))], "cid 49"),
    ("a live axle on the rear axle without cid 69", [(10, 51, (2, hp_block(HP_RIGHT))), (10, 52, (2, hp_block(HP_LEFT)))], "cid 51"),
    ("cid 69 set without a live axle", [(6, 69, 1)], "cid 69"),
    ("two hardpoint types on the front axle", [(10, 50, (1, hp_block(HP_LEFT[:4])))], "cid 49"),
    ("geometry mode 2 on the front axle", [(6, 53, 2)], "cid 53"),
    ("geometry mode 3 on the rear axle", [(6, 61, 3)], "cid 61"),
    ("hardpoint type 5", [(10, 49, (5, hp_block(HP_RIGHT)))], "cid 49"),
    ("a strut with 6 points", [(10, 49, (1, hp_block(HP_RIGHT)))], "cid 49"),
    ("a live axle in geometry mode 0", [(6, 69, 1), (10, 51, (2, hp_block(HP_RIGHT))), (10, 52, (2, hp_block(HP_LEFT)))], "cid 51"),
)
for i, (label, edits, cid_text) in enumerate(GATE_CASES):
    path = write_variant(f"gate_{i}.car", change(build_records(), edits))
    msg = raised_text(lambda p=path: check_car(load_car(p)))
    ok(f"gate refuses {label}: the message names {cid_text}", cid_text in msg)

print("[cs_kinematics] anti-dive pitch arm: pitch term of the tyre forces (fake IPOPT, no solve)")
HEIGHTS = {55: 0.07, 56: 0.02, 63: 0.10, 64: 0.02}          # front brake, front drive, rear brake, rear drive (m)
CAR_H = write_variant("heights.car", change(build_records(), [(1, c, v) for c, v in HEIGHTS.items()]))
CAR_EQ = write_variant("heights_equal.car", change(build_records(), [(1, 55, 0.05), (1, 56, 0.05), (1, 63, 0.18), (1, 64, 0.18)]))
ctx_h = build_model(CAR_H)
ctx_z = build_model(BASE)
ctx_e = build_model(CAR_EQ)
m_h, m_z = ctx_h.m23, ctx_z.m23
fn_pitch = ca.Function("p", [m_h.x, m_h.u, m_h.pv], [m_h.dx, m_h.sf, m_h.fx_fl, m_h.fx_fr, m_h.fx_rl, m_h.fx_rr])
fn_zero = ca.Function("z", [m_z.x, m_z.u, m_z.pv], [m_z.dx, m_z.sf, m_z.fx_fl, m_z.fx_fr, m_z.fx_rl, m_z.fx_rr])
Iy = float(ctx_h.vp.I_y)
ok(f"the switch width of the anti-dive blend is the hand value {F_SW_HAND:.0f} N", vehModel_cs.F_SW == F_SW_HAND)
worst_pitch = 0.0
for label, sx in (("brake", -0.3), ("drive", 0.2), ("coast", 0.0)):
    x = state(ctx_h, vx=50.0, sx=sx)
    oh = [np.asarray(o, float).ravel() for o in fn_pitch(x / m_h.x_s, np.zeros(m_h.nu), 0.0)]
    oz = [np.asarray(o, float).ravel() for o in fn_zero(x / m_z.x_s, np.zeros(m_z.nu), 0.0)]
    dth_h, dth_z = oh[0][12] * m_h.x_s[12], oz[0][12] * m_z.x_s[12]
    sf_v = float(oh[1][0])
    fx = {c: float(oh[2 + i][0]) for i, c in enumerate(("fl", "fr", "rl", "rr"))}
    h_f = HEIGHTS[55] + (HEIGHTS[56] - HEIGHTS[55]) * 0.5 * (1 + np.tanh((fx["fl"] + fx["fr"]) / F_SW_HAND))
    h_r = HEIGHTS[63] + (HEIGHTS[64] - HEIGHTS[63]) * 0.5 * (1 + np.tanh((fx["rl"] + fx["rr"]) / F_SW_HAND))
    hand = (fx["fl"] + fx["fr"]) * h_f * sf_v / Iy + (fx["rl"] + fx["rr"]) * h_r * sf_v / Iy
    scale = max(abs(hand), 1e-6)            # rad/s^2: the coast term is 1e-20, so the relative test has a floor
    rel = abs((dth_h - dth_z) - hand) / scale
    worst_pitch = max(worst_pitch, rel)
    ok(f"{label} (slip {sx:+.1f}): dthetadot with the heights minus without = sum fx h sf / I_y "
       f"({dth_h - dth_z:+.6e} vs {hand:+.6e} rad/s^2, rel {rel:.1e})", rel <= 1e-6)
ok(f"pitch term: the three states agree with the hand sum (worst rel {worst_pitch:.1e})", worst_pitch <= 1e-6)
m_e = ctx_e.m23
# With all side-view heights 0 the pitch arm term adds nothing. No pinned graph hash: the float bits of the geometry
# fit differ between BLAS kernels, so a value hash differs between platforms. Two checks inside one run instead.
# 1. Structure: equal heights other than 0 add one subtraction to the pitch arm of each corner, and nothing else.
fn_e = ca.Function("e", [m_e.x, m_e.u, m_e.pv], [m_e.dx, m_e.sf, m_e.fx_fl, m_e.fx_fr, m_e.fx_rl, m_e.fx_rr])
ops_z, ops_e = (collections.Counter(f.instruction_id(i) for i in range(f.n_instructions())) for f in (fn_zero, fn_e))
d_ops = {k: ops_e[k] - ops_z[k] for k in set(ops_e) | set(ops_z) if ops_e[k] != ops_z[k] and k != ca.OP_CONST}
ok(f"heights 0 leave no arm node: equal heights add 4 subtractions and no other operation (added {d_ops})",
   d_ops == {ca.OP_SUB: 4})
# 2. Numbers: at random states the graph with heights 0 is the graph with equal heights less the hand arm term.
rng_k3 = np.random.default_rng(3)
worst_row, worst_arm = 0.0, 0.0
for _ in range(20):
    x = state(ctx_z, vx=rng_k3.uniform(20.0, 70.0), vy=rng_k3.uniform(-1.0, 1.0), sx=rng_k3.uniform(-0.1, 0.1),
              zs=rng_k3.uniform(-5e-3, 5e-3), theta=rng_k3.uniform(-5e-3, 5e-3), phi=rng_k3.uniform(-5e-3, 5e-3))
    x[2] = rng_k3.uniform(-0.3, 0.3)
    u = rng_k3.uniform(-0.2, 0.2, m_z.nu)
    u[-1] = 0.0                              # no steer: m.fx_* is the tyre-frame force, and the arm takes the car-frame one
    oe = [np.asarray(o, float).ravel() for o in fn_e(x / m_e.x_s, u, 0.0)]
    oz = [np.asarray(o, float).ravel() for o in fn_zero(x / m_z.x_s, u, 0.0)]
    hand = ((oe[2][0] + oe[3][0]) * 0.05 + (oe[4][0] + oe[5][0]) * 0.18) * oe[1][0] / Iy
    rest = np.delete(np.arange(23), 12)
    worst_row = max(worst_row, float(np.max(np.abs(oe[0][rest] - oz[0][rest]) / np.maximum(np.abs(oz[0][rest]), 1.0))))
    dth_e, dth_z = oe[0][12] * m_e.x_s[12], oz[0][12] * m_z.x_s[12]
    worst_arm = max(worst_arm, abs(dth_e - hand - dth_z) / max(abs(dth_z), abs(hand), 1.0))
ok(f"heights 0, 20 random states: every row but dthetadot equals the equal-heights car (worst rel {worst_row:.1e})",
   worst_row <= 1e-12)
ok(f"heights 0, 20 random states: dthetadot is the equal-heights value less sum fx h sf / I_y (worst rel {worst_arm:.1e})",
   worst_arm <= 1e-12)
t_z = tanh_count(ca.Function("dx", [m_z.x, m_z.u, m_z.pv], [m_z.dx]))
t_h = tanh_count(ca.Function("dx", [m_h.x, m_h.u, m_h.pv], [m_h.dx]))
t_e = tanh_count(ca.Function("dx", [m_e.x, m_e.u, m_e.pv], [m_e.dx]))
ok(f"equal heights on both axles add no tanh node ({t_e} tanh nodes, zero heights {t_z})", t_e == t_z)
ok(f"unequal heights add the two switch blends ({t_h} tanh nodes, zero heights {t_z})", t_h == t_z + 2)


print("[cs_kinematics] K1 hand formula and sign")


def with_castor(records, castor_deg):
    """Records with the tyre castor scalar (tag 13, scalar 2, degrees) of every corner set to castor_deg."""
    out = []
    for tag, cid, v in records:
        if tag == 13:
            dims, sc, tabs = v
            sc = np.array(sc, float)
            sc[2] = castor_deg
            v = (dims, sc, tabs)
        out.append((tag, cid, v))
    return out


def k1_hand(castor, hp):
    """K1 = tan c / sqrt(1 + tan^2 c + tan^2 sigma). sigma: front view of the lower -> upper ball joint. hp has 6 rows
    (double wishbone: rows 2 and 5) or 4 rows (strut: rows 2 and 3)."""
    tc = np.tan(castor)
    top = hp[5] if len(hp) == 6 else hp[3]
    ts = (top[1] - hp[2, 1]) / (top[2] - hp[2, 2])
    return tc / np.sqrt(1.0 + tc * tc + ts * ts)


print("[cs_kinematics] steer camber: K1 from castor and kingpin, dynamic camber from delta (fake IPOPT, no solve)")
DELTA = 0.05                              # rad, a left turn
DELTA_N = DELTA / np.radians(35.0)        # the model: delta = 35 deg x delta_n
DYN = ("delta", "dynamic_camber_fl", "dynamic_camber_fr", "dynamic_camber_rl", "dynamic_camber_rr")
K_C2 = build_model(BASE)                  # castor 2 deg (build_records default), camber 0
K1 = suspension_curves(K_C2.cs).k1
hp_car = np.asarray(K_C2.cs.hardpoints.front_l, float)
ok("the synthetic front hardpoints read back as written (1e-12)", np.allclose(hp_car, HP_LEFT, rtol=0, atol=1e-12))
ok(f"K1 = {K1:.6f} equals the hand formula from castor 2 deg and the hardpoints (rel 1e-12)",
   abs(K1 - k1_hand(K_C2.cs.tyre.fl["castor"], hp_car)) <= 1e-12 * abs(K1))
x_st = state(K_C2, vx=20.0, zs=0.004, theta=0.01, phi=0.02)
d0 = evaluate(K_C2, DYN, x_st, [0.0, 0.0, 0.0])
d1 = evaluate(K_C2, DYN, x_st, [0.0, 0.0, DELTA_N])
ok(f"the steer input gives delta = {d1['delta']:.6f} rad", abs(d1["delta"] - DELTA) < 1e-12)
dfr = d1["dynamic_camber_fr"] - d0["dynamic_camber_fr"]
dfl = d1["dynamic_camber_fl"] - d0["dynamic_camber_fl"]
ok(f"front right channel moves by +K1 delta ({dfr:.6e} vs {K1 * DELTA:.6e}, rel 1e-9)",
   abs(dfr - K1 * DELTA) <= 1e-9 * abs(K1 * DELTA))
ok(f"front left channel moves by -K1 delta ({dfl:.6e} vs {-K1 * DELTA:.6e}, rel 1e-9)",
   abs(dfl + K1 * DELTA) <= 1e-9 * abs(K1 * DELTA))
ok(f"rear channels do not move with delta ({d1['dynamic_camber_rl'] - d0['dynamic_camber_rl']:.1e}, "
   f"{d1['dynamic_camber_rr'] - d0['dynamic_camber_rr']:.1e})",
   abs(d1["dynamic_camber_rl"] - d0["dynamic_camber_rl"]) < 1e-12
   and abs(d1["dynamic_camber_rr"] - d0["dynamic_camber_rr"]) < 1e-12)

K0_PATH = write_variant("castor0.car", with_castor(build_records(camber_deg=0.0), 0.0))
K0 = build_model(K0_PATH)
ok("castor 0 gives K1 = 0 exactly", suspension_curves(K0.cs).k1 == 0.0)
e00, e01 = evaluate(K0, DYN, x_st, [0.0, 0.0, 0.0]), evaluate(K0, DYN, x_st, [0.0, 0.0, DELTA_N])
ok("castor 0: the front cambers do not move with delta (no term, 1e-15)",
   abs(e01["dynamic_camber_fl"] - e00["dynamic_camber_fl"]) < 1e-15
   and abs(e01["dynamic_camber_fr"] - e00["dynamic_camber_fr"]) < 1e-15)
RNG = np.random.default_rng(6)


def rand_state(ctx):
    """A random straight-line state: speed 12-30 m/s, small heave, pitch, roll and unsprung travel."""
    x = state(ctx, vx=RNG.uniform(12.0, 30.0), vy=RNG.uniform(-0.5, 0.5), sx=RNG.uniform(-0.03, 0.03),
              zs=RNG.uniform(-0.01, 0.01), theta=RNG.uniform(-0.01, 0.01), phi=RNG.uniform(-0.01, 0.01))
    x[15:19] = RNG.uniform(-1e-3, 1e-3, 4)
    return x


dx_0 = dx_1 = 0.0
for _ in range(4):
    x = rand_state(K_C2)
    a0, b0 = dx_at(K_C2, x, 0.0), dx_at(K0, x, 0.0)
    a1, b1 = dx_at(K_C2, x, DELTA_N), dx_at(K0, x, DELTA_N)
    dx_0 = max(dx_0, np.max(np.abs(a0 - b0)) / np.max(np.abs(b0)))
    dx_1 = max(dx_1, np.max(np.abs(a1 - b1)) / np.max(np.abs(b1)))
ok(f"castor 2 and castor 0 cars: dx equal at delta 0 at random states (max rel {dx_0:.1e}): K1 enters only through delta",
   dx_0 <= 1e-12)
ok(f"castor 2 and castor 0 cars: dx differ at delta 0.05 (max rel {dx_1:.1e}, float noise is 0)", dx_1 > 1e-10)

print("[cs_kinematics] steer camber: QSS apex camber and lap")


# The QSS needs tyre peaks above zero: the default tables here give no lateral capacity. Peak = mu 1 x load.
def with_peaks(records):
    out = []
    for tag, cid, v in records:
        if tag == 13:
            dims, sc, tabs = v
            tabs = list(tabs)
            load = np.linspace(1000.0, 25000.0, dims[0])
            tabs[3] = np.concatenate([load, TYRE_TEMPS, np.outer(load, np.ones(dims[1])).ravel()])
            v = (dims, sc, tabs)
        out.append((tag, cid, v))
    return out


QK20 = write_variant("qss_castor20.car", with_peaks(with_castor(build_records(camber_deg=0.0), 20.0)))
QK0 = write_variant("qss_castor0.car", with_peaks(with_castor(build_records(camber_deg=0.0), 0.0)))
Q2, Q0 = qss_ctx(QK20), qss_ctx(QK0)
K1_Q = suspension_curves(Q2.cs).k1
ok(f"QSS car: K1 = {K1_Q:.4f} at castor 20 deg, and 0 at castor 0", K1_Q > 0 and suspension_curves(Q0.cs).k1 == 0.0)
env2, env0 = build_envelope(Q2), build_envelope(Q0)
# ponytail: the apex camber -K1 L ay / v^2 is internal to build_envelope_cs (no output carries it), so the check
# compares the envelopes. On this car the front axle binds near 40 m/s only (20, 30, 50 m/s are rear limited),
# so the one speed with a change is checked. A hand value needs the apex camber as an output.
i2 = int(np.argmin(np.abs(env2.v_grid - 40.0)))
i0 = int(np.argmin(np.abs(env0.v_grid - 40.0)))
d40 = float(env2.ay_max[i2] - env0.ay_max[i0])
ok(f"QSS ay_max at 40 m/s moves with K1 ({d40:+.4e} m/s^2, castor 20 minus castor 0)", abs(d40) > 1e-6)
lap2, lap0 = float(Q2.data.lap_time), float(Q0.data.lap_time)
ok(f"QSS lap of the castor 20 car differs from castor 0 ({lap2:.5f} vs {lap0:.5f} s)", abs(lap2 - lap0) > 1e-4)

print("[cs_kinematics] bump steer: toe from the table at the wheel travel (fake IPOPT, no solve)")
TOE_X = np.linspace(-0.02, 0.02, 11)      # m, the bump steer grid of cids 74-77
BUMP_CIDS = {"fr": 74, "fl": 75, "rr": 76, "rl": 77}


def with_bump(records, tables):
    """Records with the bump steer tables (tag 7) of the corners in tables {corner: toe [rad]}; others zero."""
    extra = [(7, BUMP_CIDS[c], (TOE_X, np.asarray(tables.get(c, np.zeros(TOE_X.size)), float))) for c in BUMP_CIDS]
    return records[:-1] + extra + records[-1:]     # the end record stays last


BUMP_FL_PATH = write_variant("bump_fl.car", with_bump(build_records(camber_deg=0.0), {"fl": 0.2 * TOE_X}))
BUMP_FL = build_model(BUMP_FL_PATH)
names_b = ["xs_fl", "xs_fr", "xs_rl", "xs_rr", "sa_fl", "sa_fr", "sa_rl", "sa_rr"]
# The model's table is a sampled spline (vehModel_cs._curve), so a linear table is met to about 3e-6 relative.
# Past the table end the clamp holds the end value to about 4e-5 m of travel (2e-5 rad of toe).
for zs in (0.01, -0.01, 0.05, -0.05):
    vb = evaluate(BUMP_FL, names_b, state(BUMP_FL, zs=zs))
    v0 = evaluate(K_C2, names_b, state(K_C2, zs=zs))
    dfl_ = vb["sa_fl"] - v0["sa_fl"]
    dfr_ = vb["sa_fr"] - v0["sa_fr"]
    xl, xr = vb["xs_fl"], vb["xs_fr"]
    want_l = 0.2 * float(np.clip(xl, TOE_X[0], TOE_X[-1]))      # the table held at its ends
    want_r = -0.2 * float(np.clip(xr, TOE_X[0], TOE_X[-1]))     # right side: mirror copy, minus sign (static toe)
    inside = abs(xl) <= TOE_X[-1]
    tol_l = 1e-5 * abs(want_l) if inside else 2e-5
    ok(f"xs {xl * 1e3:+.1f} mm: FL slip angle moves by the table ({dfl_:+.7f} vs {want_l:+.7f} rad, "
       f"tol {tol_l:.0e} rad)", abs(dfl_ - want_l) <= tol_l)
    ok(f"xs {xr * 1e3:+.1f} mm: FR slip angle moves by minus the table (mirror of FL) "
       f"({dfr_:+.7f} vs {want_r:+.7f} rad, tol {tol_l:.0e} rad)", abs(dfr_ - want_r) <= tol_l)
    ok(f"xs {xl * 1e3:+.1f} mm: rear slip angles do not move (no rear table)",
       abs(vb["sa_rl"] - v0["sa_rl"]) < 1e-12 and abs(vb["sa_rr"] - v0["sa_rr"]) < 1e-12)
ok("the toe checks reach -10 mm and +10 mm of travel (inside the grid) and beyond -20 mm and +20 mm",
   abs(abs(evaluate(BUMP_FL, names_b, state(BUMP_FL, zs=0.01))["xs_fl"]) - 0.01) < 2e-4
   and abs(evaluate(BUMP_FL, names_b, state(BUMP_FL, zs=0.05))["xs_fl"]) > 0.02)

BUMP4 = {c: (i + 1) * 1e-3 * TOE_X / 0.02 for i, c in enumerate(BUMP_CIDS)}   # a distinct line per corner, 1-4 mm
car4 = load_car(write_variant("bump4.car", with_bump(build_records(camber_deg=0.0), BUMP4)))
ok("four distinct bump steer tables decode by cid: 74 FR, 75 FL, 76 RR, 77 RL",
   all(np.array_equal(bump_steer(car4, c)["y"], BUMP4[c]) for c in CORNERS))

bad_x = [(t, c, (v[0][::-1], v[1]) if t == 7 and c == 75 else v) for t, c, v in with_bump(build_records(camber_deg=0.0), {"fl": 0.2 * TOE_X})]
bad_car = load_car(write_variant("bump_bad_x.car", bad_x))
try:
    bump_steer(bad_car, "fl")
    bad_msg = ""
except ValueError as e:
    bad_msg = str(e)
ok("a live bump table with decreasing travel raises ValueError naming corner fl and cid 75",
   "cid 75" in bad_msg and "fl" in bad_msg)

ZERO = build_model(write_variant("bump_zero.car", with_bump(build_records(camber_deg=0.0), {})))
worst_z = 0.0
for _ in range(3):
    x = rand_state(K_C2)
    a, b = dx_at(K_C2, x, DELTA_N), dx_at(ZERO, x, DELTA_N)
    worst_z = max(worst_z, np.max(np.abs(a - b)) / np.max(np.abs(a)))
ok(f"zero bump steer tables add no node: dx equal to the car without tables (max rel {worst_z:.1e})", worst_z <= 1e-12)

QB_PATH = write_variant("qss_bump_fl.car", with_peaks(with_bump(build_records(camber_deg=0.0), {"fl": 0.2 * TOE_X})))
QB0_PATH = write_variant("qss_no_bump.car", with_peaks(build_records(camber_deg=0.0)))
lap_b = float(qss_ctx(QB_PATH).data.lap_time)
lap_nb = float(qss_ctx(QB0_PATH).data.lap_time)
ok(f"the QSS ignores bump steer: lap {lap_b:.9f} s equals the car without tables {lap_nb:.9f} s",
   abs(lap_b - lap_nb) <= 1e-12 * lap_nb)

print(f"[cs_kinematics] synthetic: {PASS} checks passed")


print("[cs_kinematics] vendor: GP3 2013 (Eastern Creek), GT3, strut car, live-axle car, mode-1 car, stored heights")
GP3_CAR = _bootstrap.vendor_car(CS_DIR) if CS_DIR else ""       # the GP3 2013 base car, picked by content
GP3_LOG = os.path.join(MODELS, "GP3 2013", "Eastern Creek, AU", "gp3_baseline_easterncreek.ld")
if not CS_DIR or not os.path.isfile(GP3_CAR) or not os.path.isfile(GP3_LOG):
    print("  SKIP: CHASSISSIM_DIR unset or the GP3 2013 files are missing")
    print(f"[cs_kinematics] {PASS} checks passed (vendor skipped)")
    sys.exit(0)


CARS = {p: load_car(p) for p in list_cars(CS_DIR)}     # every shipped car file, by path


def only(pred, what):
    """The path of the one shipped car whose record satisfies pred (the check asserts that there is one)."""
    hits = [p for p, c in CARS.items() if pred(c)]
    ok(f"exactly one shipped car is {what} ({len(hits)} found)", len(hits) == 1)
    return hits[0]


def qss_builds(path):
    """True when the QSS of the car builds (a car the model refuses raises ValueError)."""
    try:
        quiet(MLTP_screen, car=f"chassissim:{path}", circuit="chassissim:Eastern Creek", chassissim_dir=CS_DIR,
              physics="corrected", save=False, verbose=False)
        return True
    except ValueError:
        return False


STRUT = only(lambda c: c.hardpoints.front_l_type == 1, "a strut car (front type byte 1)")
LIVE = only(lambda c: c.susp.live_axle == 1 and c.hardpoints.rear_r_type == 2 and c.hardpoints.rear_l_type == 2,
            "a live-axle car (cid 69 = 1, rear type byte 2)")
MODE1 = only(lambda c: c.geom.mode_f == 1 and c.geom.mode_r == 1 and qss_builds(c.path) and all(
                  np.unique(t[k]).size >= 2 for t in (c.tyre.fl, c.tyre.fr, c.tyre.rl, c.tyre.rr)
                  for k in ("camber_x", "fx_camber_x")),
             "a mode-1 car whose QSS builds with two or more camber points on every tyre table")


def mm(v):
    return 1e3 * float(v)


LOG_MAP = {
    "s": (("Lap Distance",), "m", 1.0),
    "rc_f": (("RC front",), "mm", 1e-3), "rc_r": (("RC rear",), "mm", 1e-3),
    "zfl": (("Susp Position - Front L",), "mm", 1e-3), "zfr": (("Susp Position - Front R",), "mm", 1e-3),
    "zrl": (("Susp Position - Rear L",), "mm", 1e-3), "zrr": (("Susp Position - Rear R",), "mm", 1e-3),
    "cfl": (("FL Camber",), "deg", 1.0), "cfr": (("FR Camber",), "deg", 1.0),
    "crl": (("RL Camber",), "deg", 1.0), "crr": (("RR Camber",), "deg", 1.0),
    "st": (("Steered Angle",), "deg", 1.0),
}


def rc_medians(car_path, log_path):
    """Median of (model RC at the mean travel - log RC) per axle, lap 1, |travel| <= 50 mm (the vendor method)."""
    vc_ = suspension_curves(load_car(car_path))
    log = read_ld(log_path, channel_map=LOG_MAP)
    lap1 = log["lap"] == 1
    out = {}
    for ax, k, zl, zr, rc in (("front", vc_.f, log["zfl"], log["zfr"], log["rc_f"]),
                              ("rear", vc_.r, log["zrl"], log["zrr"], log["rc_r"])):
        wv = 0.5 * (zl + zr)
        sel = lap1 & (np.abs(wv) <= 0.05)
        out[ax] = float(np.median(k.rc_h(wv[sel]) - rc[sel]))
    return out


def camber_basis(own, oth, cam, steer=None):
    """Least squares of camber on 1, the mean travel, own minus other travel (and steer for the front): the basis
    of the log camber fit. Returns the coefficients."""
    wm, d = 0.5 * (own + oth), own - oth
    cols = [np.ones_like(own), wm, d] + ([steer] if steer is not None else [])
    return np.linalg.lstsq(np.c_[tuple(cols)], cam, rcond=None)[0]


gp3 = load_car(GP3_CAR)
gc = gp3.geom
vc = suspension_curves(gp3)
ok(f"GP3 rc_datum rear equals cid 62 ({mm(vc.r.rc_datum):.4f} mm, 1e-3 mm)", abs(vc.r.rc_datum - gc.rc_datum0_rr) < 1e-6)
ok(f"GP3 rc_datum front is {mm(vc.f.rc_datum):.4f} mm (the stored cid 54 is stale: {mm(gc.rc_datum0_fr):.3f} mm)",
   abs(mm(vc.f.rc_datum) - 10.452) < 1e-3)
ok(f"GP3 static roll centre above the patch: front {mm(vc.f.rc_h(0.0)):.3f} mm (10.452), rear "
   f"{mm(vc.r.rc_h(0.0)):.3f} mm (46.365), 1e-2 mm",
   abs(mm(vc.f.rc_h(0.0)) - 10.452) < 0.01 and abs(mm(vc.r.rc_h(0.0)) - 46.365) < 0.01)
ok(f"GP3 w0: front {mm(vc.f.w0):+.2f} mm, rear {mm(vc.r.w0):+.2f} mm (datum less static ride height)",
   abs(vc.f.w0) < 1e-9 and abs(mm(vc.r.w0) - 6.96) < 0.01)

base_med = rc_medians(GP3_CAR, GP3_LOG)
ok(f"GP3 log median RC front {mm(base_med['front']):+.3f} mm (-1.91, 0.02 mm), gate 4 mm",
   abs(mm(base_med["front"]) + 1.91) < 0.02 and abs(mm(base_med["front"])) < 4.0)
ok(f"GP3 log median RC rear {mm(base_med['rear']):+.3f} mm (+2.50, 0.02 mm), gate 4 mm",
   abs(mm(base_med["rear"]) - 2.50) < 0.02 and abs(mm(base_med["rear"])) < 4.0)
for tag, car_f, log_f in (("17 mm", "4b2861dc", "gp3_rideheight17mm_easterncreek.ld"),      # car files by sha8
                          ("20 mm", "9cf342bb", "gp3_rideheight20mm_easterncreek.ld")):
    med = rc_medians(_bootstrap.vendor_car(CS_DIR, car_f), os.path.join(os.path.dirname(GP3_CAR), log_f))
    ok(f"GP3 ride height {tag}: front median {mm(med['front']):+.3f} mm within 0.1 mm of the default "
       f"({mm(base_med['front']):+.3f})", abs(mm(med["front"]) - mm(base_med["front"])) < 0.1)

LOG = read_ld(GP3_LOG, channel_map=LOG_MAP)
s_ = {k: LOG[k][LOG["lap"] == 1] for k in ("zfl", "zfr", "zrl", "zrr", "st", "cfl", "cfr", "crl", "crr")}
mm_ = {k: v * 1e3 for k, v in s_.items() if k.startswith("z")}          # mm
log_rl = camber_basis(mm_["zrl"], mm_["zrr"], s_["crl"])
log_rr = camber_basis(mm_["zrr"], mm_["zrl"], s_["crr"])
log_fl = camber_basis(mm_["zfl"], mm_["zfr"], s_["cfl"], s_["st"])
log_fr = camber_basis(mm_["zfr"], mm_["zfl"], s_["cfr"], -s_["st"])
ok(f"GP3 rear camber intercept at heave 0: model {np.degrees(gp3.tyre.rl['camber']) + np.degrees(vc.r.camber(0.0)):.4f} "
   f"deg against the log {log_rl[0]:.4f} deg (0.02 deg)",
   abs(np.degrees(gp3.tyre.rl["camber"]) + np.degrees(vc.r.camber(0.0)) - log_rl[0]) < 0.02)
gain_r = np.degrees(vc.r.camber.deriv()(0.0)) * 1e-3          # deg per wheel mm
ok(f"GP3 rear camber gain at w = 0 {gain_r:.5f} deg/mm (regression pin 0.022436, 0.5 %)",
   abs(gain_r - 0.022436) <= 0.005 * 0.022436)
own_r = {"rl": mm_["zrl"] * 1e-3, "rr": mm_["zrr"] * 1e-3}
mod_rl = camber_basis(mm_["zrl"], mm_["zrr"], np.degrees(gp3.tyre.rl["camber"]) + np.degrees(vc.r.camber(own_r["rl"])))
mod_rr = camber_basis(mm_["zrr"], mm_["zrl"], np.degrees(gp3.tyre.rr["camber"]) + np.degrees(vc.r.camber(own_r["rr"])))
mod_r, log_r = 0.5 * (mod_rl[1] + mod_rr[1]), 0.5 * (log_rl[1] + log_rr[1])
ok(f"GP3 rear camber gain on the log basis: model {mod_r:.5f} against the log {log_r:.5f} deg/mm (within 7 %)",
   abs(mod_r / log_r - 1) <= 0.07)
mod_fl = camber_basis(mm_["zfl"], mm_["zfr"], np.degrees(gp3.tyre.fl["camber"]) + np.degrees(vc.f.camber(mm_["zfl"] * 1e-3)), s_["st"])
mod_fr = camber_basis(mm_["zfr"], mm_["zfl"], np.degrees(gp3.tyre.fr["camber"]) + np.degrees(vc.f.camber(mm_["zfr"] * 1e-3)), -s_["st"])
mod_f, log_f = 0.5 * (mod_fl[1] + mod_fr[1]), 0.5 * (log_fl[1] + log_fr[1])
ok(f"GP3 front camber gain on the log basis: model {mod_f:+.5f} against the log {log_f:+.5f} deg/mm (within 5 %)",
   abs(mod_f / log_f - 1) <= 0.05)
for c, cam_deg, want in (("fl", 3.8, 0.5668), ("rl", 2.5, 0.8125)):
    tyre = getattr(gp3.tyre, c)
    ok(f"GP3 {c} tyre static camber is {cam_deg} deg", abs(np.degrees(tyre["camber"]) - cam_deg) < 1e-6)
    axle_share = gp3.mass.front_share if c[0] == "f" else 1.0 - gp3.mass.front_share
    load = 0.5 * axle_share * gp3.mass.total * G
    t4 = RegularGridInterpolator((tyre["fx_camber_x"], tyre["fx_camber_load"]), tyre["fx_camber"])
    got = float(t4([[tyre["camber"], load]])[0])
    ok(f"GP3 {c} table 4 at {cam_deg} deg = {got:.5f} (stored {want}, 1e-4)", abs(got - want) < 1e-4)
ok(f"GP3 front steer camber K1 {vc.k1:.4f} within 1e-3 of 0.1517", abs(vc.k1 - 0.1517) < 1e-3)
ok("GP3 rear tyres have castor 0, so the rear K1 is 0 (tan 0 = 0)",
   abs(gp3.tyre.rl["castor"]) < 1e-12 and abs(gp3.tyre.rr["castor"]) < 1e-12 and np.tan(gp3.tyre.rl["castor"]) == 0.0)

wgrid = np.linspace(-0.05, 0.05, 201)
for axle, k in (("front", vc.f), ("rear", vc.r)):
    rc_v, cam_v = k.rc_h(wgrid), k.camber(wgrid)
    ok(f"GP3 {axle}: curves finite over +-50 mm travel", bool(np.all(np.isfinite(rc_v)) and np.all(np.isfinite(cam_v))))
    step_rc = float(np.max(np.abs(np.diff(rc_v))))
    step_cam = float(np.max(np.abs(np.diff(cam_v))))
    ok(f"GP3 {axle}: curves smooth (largest step per 0.5 mm: RC {step_rc * 1e3:.3f} mm, camber "
       f"{np.degrees(step_cam):.4f} deg)", step_rc < 1e-3 and np.degrees(step_cam) < 0.02)

# GT3 (found by glob): the datum is the stored one on the front axle; the rear datum is 0.38 mm off the stored cid 62
gt3_cars = glob.glob(os.path.join(glob.escape(MODELS), "GT3", "*", "*.car"))
ok("one GT3 car file by glob GT3/*/*.car", len(gt3_cars) == 1)
gt3_logs = glob.glob(os.path.join(glob.escape(os.path.dirname(gt3_cars[0])), "*.ld"))
ok("the GT3 car folder holds one .ld log", len(gt3_logs) == 1)
gt3 = load_car(gt3_cars[0])
gv = suspension_curves(gt3)
ok(f"GT3 rc_datum front equals cid 54 ({mm(gv.f.rc_datum):.4f} mm, 1e-3 mm)",
   abs(gv.f.rc_datum - gt3.geom.rc_datum0_fr) < 1e-6)
ok(f"GT3 rc_datum rear is {mm(gv.r.rc_datum):.4f} mm, {mm(gv.r.rc_datum - gt3.geom.rc_datum0_rr):+.3f} mm off cid 62",
   abs(mm(gv.r.rc_datum - gt3.geom.rc_datum0_rr) - 0.380) < 0.005)
gt3_log = read_ld(gt3_logs[0], channel_map={**LOG_MAP, "roll": (("Roll angle",), "deg", 1.0)})
low_roll = (gt3_log["lap"] == 1) & (np.abs(gt3_log["roll"]) < 0.15)
for axle, k, zl, zr, rc in (("front", gv.f, gt3_log["zfl"], gt3_log["zfr"], gt3_log["rc_f"]),
                            ("rear", gv.r, gt3_log["zrl"], gt3_log["zrr"], gt3_log["rc_r"])):
    wv = 0.5 * (zl + zr)
    sel = low_roll & (np.abs(wv) <= 0.05)
    med = float(np.median(k.rc_h(wv[sel]) - rc[sel]))
    ok(f"GT3 {axle}: median above-ground RC vs the log at low roll {med * 1e3:+.2f} mm over {int(sel.sum())} "
       "lap-1 samples, within 7 mm", int(sel.sum()) > 0 and abs(med) < 7e-3)

# Stored front-view heights at the datum pose (cids 54 and 62): every shipped mode-0, type-0 axle, by content
heights = []                                    # (axle, path, mm off the stored height)
for p, c in CARS.items():
    vk = suspension_curves(c)
    if c.geom.mode_f == 0 and c.hardpoints.front_l_type == 0 and c.hardpoints.front_r_type == 0:
        heights.append(("front", p, mm(vk.f.rc_datum - c.geom.rc_datum0_fr)))
    if c.geom.mode_r == 0 and c.hardpoints.rear_l_type == 0 and c.hardpoints.rear_r_type == 0:
        heights.append(("rear", p, mm(vk.r.rc_datum - c.geom.rc_datum0_rr)))
near = [h for h in heights if abs(h[2]) <= 0.01]
ok(f"shipped mode-0, type-0 axles: {len(near)} of {len(heights)} equal cids 54 / 62 within 0.01 mm (pin 24 of 51)",
   len(heights) == 51 and len(near) == 24)
st_k, st_g = suspension_curves(CARS[STRUT]), CARS[STRUT].geom
st_df, st_dr = mm(st_k.f.rc_datum - st_g.rc_datum0_fr), mm(st_k.r.rc_datum - st_g.rc_datum0_rr)
ok(f"the strut car: front and rear roll centre at the datum equal cids 54 / 62 within 0.2 mm "
   f"({st_df:+.3f} / {st_dr:+.3f} mm)", abs(st_df) < 0.2 and abs(st_dr) < 0.2)
gt3_front = mm(gv.f.rc_datum - gt3.geom.rc_datum0_fr)
ok(f"GT3 front roll centre at the datum equals cid 54 ({gt3_front:+.4f} mm)", abs(gt3_front) < 0.01)

print("[cs_kinematics] vendor: K1 type byte, the strut car hand case")
wrx = load_car(STRUT)
vw = suspension_curves(wrx)
hpw = wrx.hardpoints
ok(f"strut car: type byte 1 on all four blocks and shape (4, 3) (types {hpw.front_r_type} {hpw.front_l_type} "
   f"{hpw.rear_r_type} {hpw.rear_l_type})",
   all(t == 1 for t in (hpw.front_r_type, hpw.front_l_type, hpw.rear_r_type, hpw.rear_l_type))
   and np.shape(hpw.front_l) == (4, 3) and np.shape(hpw.rear_r) == (4, 3))
ok(f"strut car: roll centre at the datum {mm(vw.f.rc_datum):.4f} mm (hand 102.9279, 1e-3 mm)",
   abs(mm(vw.f.rc_datum) - 102.9279) < 1e-3)
cam_p, cam_m = np.degrees(vw.f.camber(0.02)), np.degrees(vw.f.camber(-0.02))
ok(f"strut car: camber change at +20 mm {cam_p:+.4f} deg (+0.4480) and at -20 mm {cam_m:+.4f} deg (-0.5158), 5e-4 deg",
   abs(cam_p - 0.4480) < 5e-4 and abs(cam_m + 0.5158) < 5e-4)
k1_w = k1_hand(wrx.tyre.fl["castor"], np.asarray(hpw.front_l, float))
ok(f"strut car: K1 {vw.k1:.6f} equals the hand formula ({k1_w:.6f}) and 0.269227", abs(vw.k1 - k1_w) < 1e-9
   and abs(vw.k1 - 0.269227) < 1e-6)

v8 = load_car(LIVE)
vv8 = suspension_curves(v8)
hp8 = v8.hardpoints
ok(f"live-axle car: live rear axle (cid 69 = 1), rear type byte 2 on both corners, shape (6, 3) "
   f"(types {hp8.rear_r_type} {hp8.rear_l_type}, live {v8.susp.live_axle})",
   v8.susp.live_axle == 1 and hp8.rear_r_type == 2 and hp8.rear_l_type == 2
   and np.shape(hp8.rear_l) == (6, 3) and vv8.r.live and vv8.r.manual)
ok(f"live-axle car: the rear axle is mode 1 (manual roll centre) and live; the front is a double wishbone "
   f"(roll centre {mm(vv8.r.rc_h(0.0)):.1f} mm)",
   hp8.front_l_type == 0 and v8.geom.mode_r == 1 and abs(mm(vv8.r.rc_h(0.0)) - 235.0) < 1e-3)

gt1 = load_car(MODE1)
vg1 = suspension_curves(gt1)
ok(f"mode-1 car: both axles mode 1, roll centres {mm(vg1.f.rc_h(0.0)):.3f} / {mm(vg1.r.rc_h(0.0)):.3f} mm (50.0 / 105.0), "
   "slope -1 exact, camber static only",
   gt1.geom.mode_f == 1 and gt1.geom.mode_r == 1 and abs(mm(vg1.f.rc_h(0.0)) - 50.0) < 1e-3
   and abs(mm(vg1.r.rc_h(0.0)) - 105.0) < 1e-3 and abs(vg1.f.rc_h.deriv()(0.0) + 1.0) < 1e-12
   and abs(vg1.r.rc_h.deriv()(0.0) + 1.0) < 1e-12 and vg1.f.manual and vg1.r.manual)
ok(f"mode-1 car: K1 is 0 on the mode-1 front axle ({vg1.k1})", vg1.k1 == 0.0)

print("[cs_kinematics] vendor: NLP model of the strut car, the live-axle car and the mode-1 car on Sturn (fake IPOPT, no solve)")


def build_vendor(car_path):
    """MLTP on Sturn with the car and a fake IPOPT (no solve). Returns ctx with ctx.m23."""
    return quiet(MLTP, car=f"chassissim:{car_path}", circuit="Sturn", chassissim_dir=CS_DIR, physics="corrected",
                 save=False, plot=False, linear_solver="mumps")


def seed_check():
    """Size of the last NLP and whether g is finite at the start point of the fake solver."""
    nlp, x0 = CAP["nlp"], CAP["x0"]
    g_seed = np.array(ca.Function("g", [nlp["x"]], [nlp["g"]])(x0)).ravel()
    return nlp["x"].numel(), nlp["g"].numel(), bool(np.all(np.isfinite(g_seed)))


for name, path in (("strut car", STRUT), ("live-axle car", LIVE), ("mode-1 car", MODE1)):
    c_ = build_vendor(path)
    n_w, n_g, fin = seed_check()
    ok(f"{name} on Sturn: nx {c_.m23.nx}, n_w {n_w}, n_g {n_g} (1736 / 1794), g finite at the seed: {fin}",
       c_.m23.nx == 23 and n_w == 1736 and n_g == 1794 and fin)
    tc = tanh_count(ca.Function("dx", [c_.m23.x, c_.m23.u, c_.m23.pv], [c_.m23.dx]))
    ok(f"{name}: no tanh node in the model (equal or zero side-view heights add none): {tc}", tc == 0)
    if name == "mode-1 car":
        x_roll = state(c_, zs=0.01, phi=0.01)
        dyn = evaluate(c_, ["dynamic_camber_fl", "dynamic_camber_fr", "dynamic_camber_rl", "dynamic_camber_rr"], x_roll)
        worst = max(abs(dyn[f"dynamic_camber_{c}"] - getattr(c_.cs.tyre, c)["camber"]) for c in CORNERS)
        ok(f"mode-1 car NLP: a body roll of 0.01 rad and a heave of 10 mm change no camber (max |err| {worst:.1e} rad)",
           worst < 1e-12)

v8_ctx = build_vendor(LIVE)
xa = state(v8_ctx, vx=20.0)
xb = xa.copy()
xb[22] += 0.01                     # the right rear hub 10 mm lower (zt_rr up)
xc = xa.copy()
xc[21] += 0.01                     # the left rear hub 10 mm lower
xd = xa.copy()
xd[21] += 0.01
xd[22] += 0.01                     # both rear hubs 10 mm lower
cam_names = ["dynamic_camber_rl", "dynamic_camber_rr"]
ca0, cb, cc_, cd_ = (evaluate(v8_ctx, cam_names, x) for x in (xa, xb, xc, xd))
d_rl_b = np.degrees(cb["dynamic_camber_rl"] - ca0["dynamic_camber_rl"])
d_rr_b = np.degrees(cb["dynamic_camber_rr"] - ca0["dynamic_camber_rr"])
ok(f"live-axle car: right rear hub 10 mm lower gives rl {d_rl_b:+.5f} and rr {d_rr_b:+.5f} deg "
   "(+0.35877 / -0.35877, 1e-4 deg)", abs(d_rl_b - 0.35877) < 1e-4 and abs(d_rr_b + 0.35877) < 1e-4)
d_rl_c = np.degrees(cc_["dynamic_camber_rl"] - ca0["dynamic_camber_rl"])
d_rr_c = np.degrees(cc_["dynamic_camber_rr"] - ca0["dynamic_camber_rr"])
ok(f"live-axle car: left rear hub 10 mm lower gives the opposite change ({d_rl_c:+.5f} / {d_rr_c:+.5f} deg)",
   abs(d_rl_c + 0.35877) < 1e-4 and abs(d_rr_c - 0.35877) < 1e-4)
ok(f"live-axle car: both hubs 10 mm lower give no change ({np.degrees(cd_['dynamic_camber_rl'] - ca0['dynamic_camber_rl']):.1e}, "
   f"{np.degrees(cd_['dynamic_camber_rr'] - ca0['dynamic_camber_rr']):.1e} deg)",
   abs(cd_["dynamic_camber_rl"] - ca0["dynamic_camber_rl"]) < 1e-12
   and abs(cd_["dynamic_camber_rr"] - ca0["dynamic_camber_rr"]) < 1e-12)

print(f"[cs_kinematics] vendor: {PASS} checks passed in total")
