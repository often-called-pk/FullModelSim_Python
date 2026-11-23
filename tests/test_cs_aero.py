"""ChassisSim phase 2 task 8: aero map scale, aero corrections and a single DRS (functions/chassissim.py,
vehModel_cs.py, functions/ggv.py, userOpts.py, MLTP.py).

Checks: the CLA and CDA ride-height maps are scaled to the car scalars (cids 121, 122); the sideslip tables
(cids 128, 132, 136) act on vy / sqrt(vx^2 + vy^2) and clamp to +-0.4; the speed tables (cids 416-418, enables
413-415) act on vx; one DRS zone (cids 336-341) gates CLA, CDA and AB; the steps run in the order sideslip, DRS,
speed; the unsupported features raise ValueError; a car with no correction builds the same graph as a car
without the cids; the QSS envelope reads the speed and sideslip corrections at zero sideslip and ignores DRS.
The synthetic cars are built with MLTP and a fake IPOPT (no solve). The vendor section (last) runs only when
CHASSISSIM_DIR names the ChassisSim install folder.

    python tests/test_cs_aero.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_aero.py
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
from scipy.interpolate import PchipInterpolator

import functions.transcription as T
from functions import carfile
from functions.chassissim import load_car
from functions.ggv import build_envelope_cs
from MLTP import MLTP
from vehModel_cs import SIDESLIP_SIGN, _surf

PASS = 0
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair. An over or omit cid is a file cid.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}
AERO_OUT = ("f_lift", "f_drag", "f_lift_fl", "rh_f", "rh_r")
REL = 1e-9            # the brief's gate for the aero formulas
ZERO_TABLE = (np.array([-0.5, 0.5]), np.zeros(2))                           # a tag 7 table, all zero
ZERO_MAP = (np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.zeros(4))        # a tag 9 map, all zero


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


# ---- synthetic car: copied from tests/test_cs_diff.py (physical hardpoints, rc heights, zero bump steer,
# flat tyre table 4, rear differential type 1) ------------------------------------------------------------
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
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,          # ride heights [mm]: fl, fr, rl, rr
    100: 0.9, 102: 1000.0, 103: 9000.0,
    107: 14000.0, 108: 0.6, 109: 0.05,
    111: 0.0, 112: 1.0, 113: 1.0,                         # rear differential type 1 (open)
    55: RC_STATIC, 59: RC_STATIC, 63: RC_STATIC, 67: RC_STATIC,
    390: 350.0,
}
EXPLICIT.update({7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.0, 12: 0.0})   # third gaps 0: no third element
EXPLICIT.update({142: 0.5 * (EXPLICIT[117] + EXPLICIT[118]) * 1e-3,
                 144: 0.5 * (EXPLICIT[119] + EXPLICIT[120]) * 1e-3, 56: 0.0, 60: 0.0, 64: 0.0, 68: 0.0})
EXPLICIT.update({c: 0.0 for c in (55, 59, 63, 67) if c not in EXPLICIT})
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
MAP_X = np.array([-10000.0, -5000.0, -3333.0, -1667.0, 0.0, 1667.0, 3333.0, 5000.0, 6667.0, 8333.0, 10000.0])
MAP_Z = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
MAP_DATA = np.where((MAP_X > 0)[:, None] & (MAP_Z >= 0.1)[None, :], 0.1 * MAP_X[:, None], 0.0)

# ---- aero inputs of this file -----------------------------------------------------------------------------
AX = np.linspace(0.0, 0.04, 9)                  # rear ride height axis [m] of the CLA, CDA and AB maps
AZ = np.linspace(0.0, 0.04, 9)                  # front ride height axis [m]
GRID = np.arange(9) / 8.0
RAW_CLA = 1.0 + 0.5 * GRID[:, None] ** 2 + 0.5 * GRID[None, :]          # raw, peak 2.0 at the top corner
RAW_CDA = 0.5 + 0.5 * GRID[:, None] * GRID[None, :]                     # raw, peak 1.0
RAW_AB = 0.42 + 0.06 * GRID[:, None] - 0.03 * GRID[None, :]             # balance map, never scaled
FLAT_ONES = np.ones((9, 9))
CLA_SCALAR, CDA_SCALAR, RHO = 3.0, 0.6, 1.2
SS_X = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])                            # sideslip axis [rad]
SS_CLA = np.array([0.6, 0.15, 0.1, 0.12, 0.6])                          # 0.6 at +-0.5 clamps to 0.4
SS_CDA = np.array([0.3, 0.05, -0.02, 0.04, 0.2])
SS_AB = np.array([0.05, 0.01, 0.0, -0.02, -0.1])
SP_V = np.array([0.0, 20.0, 40.0, 80.0])                                # speed axis [m/s]
SP_CLA = np.array([1.0, 1.1, 1.2, 1.3])
SP_CDA = np.array([1.0, 0.9, 0.85, 0.8])
SP_AB = np.array([0.0, 0.01, 0.02, 0.03])


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


def aero_defaults():
    """Aero cids of the car: the maps, the scalars, the zero corrections, DRS and multi-DRS off, the refused
    features all zero. A caller changes cids with the over argument of build_records."""
    a = {121: (1, CLA_SCALAR), 122: (1, CDA_SCALAR), 123: (1, 0.0), 125: (1, RHO),
         127: (9, (AX, AZ, RAW_CLA.ravel())), 131: (9, (AX, AZ, RAW_CDA.ravel())),
         135: (9, (AX, AZ, RAW_AB.ravel())),
         128: (7, ZERO_TABLE), 132: (7, ZERO_TABLE), 136: (7, ZERO_TABLE),
         413: (6, 0), 414: (6, 0), 415: (6, 0),
         416: (7, ZERO_TABLE), 417: (7, ZERO_TABLE), 418: (7, ZERO_TABLE),
         139: (7, ZERO_TABLE), 870: (7, ZERO_TABLE), 871: (1, 0.0),
         336: (6, 0), 337: (1, 0.0), 338: (1, 0.0), 339: (1, 0.0), 340: (1, 0.0), 341: (1, 0.0),
         366: (6, 0), 367: (6, 0), 368: (6, 0), 369: (6, 0), 370: (6, 0), 371: (6, 0)}
    a.update({c: (6, 0) for c in (434, 435, 436)})
    a.update({c: (7, ZERO_TABLE) for c in (437, 438, 439)})
    a.update({c: (9, ZERO_MAP) for c in range(452, 464)})
    a.update({c: (9, ZERO_MAP) for c in (872, 873, 874, 875)})
    return a


def build_records(over=None, omit=()):
    """Records of the synthetic car. over maps a cid to (tag, value) and replaces the default. Cids in omit are
    left out of the file (the model then reads them as absent)."""
    R = []

    def add(tag, cid, v):
        cid = FILE_CID.get(cid, cid)
        if v is not None and cid not in omit:
            R.append((tag, cid, v))
    vals = dict(EXPLICIT)
    for cid, v in vals.items():
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
    for cid in range(204, 208):
        add(6, cid, 1)
    for cid in (74, 75, 76, 77):
        add(7, cid, (BUMP_X, np.zeros(len(BUMP_X))))      # bump steer, all zero
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    add(9, 114, (0.5 * np.arange(4), 1.0 * np.arange(4) + 0.1, arb(16, 114)))
    add(9, 116, (MAP_X, MAP_Z, MAP_DATA.ravel()))         # C order: the reader reshapes to (nx, nz)
    for cid in range(31, 36):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    for cid, rows in ((49, HP_LEFT), (50, HP_RIGHT), (51, HP_LEFT), (52, HP_RIGHT)):
        add(10, cid, (0, np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])))
    for i, c in enumerate(CORNERS):
        add(*tyre_record(i, c))
    aero = aero_defaults()
    aero.update(over or {})
    for cid, (tag, v) in aero.items():
        add(tag, cid, v)
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


def write_car(name, over=None, omit=()):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(over, omit), path)
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


def build(car_path, chassis_dir=INSTALL, circuit=CIRC, **kw):
    """MLTP on a car with a fake IPOPT (no solve). Returns ctx with ctx.m23, ctx.cs and ctx.track."""
    return quiet(MLTP, car=f"chassissim:{car_path}", circuit=f"chassissim:{circuit}", chassissim_dir=chassis_dir,
                 physics="corrected", save=False, plot=False, linear_solver="mumps", **kw)


def evaluator(ctx):
    """Return ev(x, gate) -> dict of AERO_OUT at the physical state x. gate is the DRS gate of pv (ignored when
    the car has no DRS row). Also the scaled dx function as dx(x, u, gate)."""
    m = ctx.m23
    f = ca.Function("aero", [m.x, m.u, m.pv], [ca.vertcat(*[ca.SX(getattr(m, n)) for n in AERO_OUT])])
    fdx = ca.Function("dx", [m.x, m.u, m.pv], [m.dx])
    npv = int(m.pv.numel())

    def pv_of(gate):
        return [0.0] if npv == 1 else [0.0, float(gate)]

    def ev(x, gate=0.0):
        o = np.array(f(np.asarray(x, float) / m.x_s, np.zeros(m.u.numel()), pv_of(gate)), float).ravel()
        return dict(zip(AERO_OUT, o))

    def dx(x, u, gate=0.0):
        return np.array(fdx(np.asarray(x, float) / m.x_s, np.asarray(u, float), pv_of(gate)), float).ravel()
    return ev, dx


def state(ctx, vx=20.0, vy=0.0, zs=0.0, theta=0.0):
    """Physical state on a straight at speed vx, sideways speed vy, body heave zs [m] and pitch theta [rad].
    Tyre states at W0 / kt."""
    vp, cs = ctx.vp, ctx.cs
    Rr = cs.tyre.rl["radius"]
    x = np.zeros(23)
    x[0], x[1] = vx, vy
    x[5] = x[6] = vx / cs.tyre.fl["radius"]
    x[7] = x[8] = vx / Rr
    x[9], x[11] = zs, theta
    x[19:23] = [getattr(vp, f"W{c}0") / vp.kt for c in CORNERS]
    return x


def raw_at(raw, rh_r, rh_f):
    """The model's map interpolation (vehModel_cs._surf) of a raw map at a ride height pair."""
    return float(np.array(_surf({"x": AX, "z": AZ, "data": raw})(rh_r, rh_f)).ravel()[0])


def table_at(t, x):
    """A tag 7 table (x, y) as the model reads it: PCHIP, the end value held outside the table."""
    tx, ty = t
    return float(PchipInterpolator(tx, ty)(np.clip(x, tx[0], tx[-1])))


def sideslip(vx, vy):
    """sin of the sideslip at the CG, the variable of the sideslip tables. The sign is the model's SIDESLIP_SIGN."""
    return SIDESLIP_SIGN * vy / np.sqrt(vx * vx + vy * vy)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


print("[cs_aero] build the synthetic cars (fake IPOPT, no solve)")
BASE_PATH = write_car("base.car")
BASE = build(BASE_PATH)
ev_b, dx_b = evaluator(BASE)
RHO_B = BASE.cs.aero.rho
ok("base car builds with the default aero cids (CLA scalar 3.0, CDA scalar 0.6)",
   BASE.cs.aero.cla == CLA_SCALAR and BASE.cs.aero.cda == CDA_SCALAR)


def base_terms(o, cla_scale=1.5, cda_scale=0.6, offset=0.0):
    """Expected CLA, CDA and AB of the base car at the ride heights of o (output dict). The scales are the
    record scales of the base car: raw peak 2.0 x 1.5 = 3.0 (CLA), raw peak 1.0 x 0.6 = 0.6 (CDA)."""
    cla = cla_scale * raw_at(RAW_CLA, o["rh_r"], o["rh_f"])
    cda = cda_scale * raw_at(RAW_CDA, o["rh_r"], o["rh_f"])
    ab = raw_at(RAW_AB, o["rh_r"], o["rh_f"]) + offset
    return cla, cda, ab


def model_coeffs(o, vx):
    q = 0.5 * RHO_B * vx * vx
    return o["f_lift"] / q, o["f_drag"] / q, 2.0 * o["f_lift_fl"] / o["f_lift"]


print("[cs_aero] 1. map scale")
# (a) the record holds the scaled map (peak = scalar) and the model lift is q x 1.5 x raw map
cla_rec = BASE.cs.aero.cla_map["data"]
cda_rec = BASE.cs.aero.cda_map["data"]
ok(f"CLA record = 1.5 x raw map (scalar 3.0 / peak 2.0), max err {np.max(np.abs(cla_rec - 1.5 * RAW_CLA)):.1e}",
   np.allclose(cla_rec, 1.5 * RAW_CLA, rtol=REL, atol=0.0))
ok(f"CDA record = 0.6 x raw map (scalar 0.6 / peak 1.0), max err {np.max(np.abs(cda_rec - 0.6 * RAW_CDA)):.1e}",
   np.allclose(cda_rec, 0.6 * RAW_CDA, rtol=REL, atol=0.0))
ok("AB record = raw balance map (not scaled)", np.array_equal(BASE.cs.aero.balance_map["data"], RAW_AB))
for vx in (15.0, 25.0):
    for zs, th in ((0.0, 0.0), (0.004, 0.01)):
        o = ev_b(state(BASE, vx=vx, zs=zs, theta=th))
        cla_m, cda_m, _ = model_coeffs(o, vx)
        cla_e, cda_e, _ = base_terms(o, cla_scale=1.5, cda_scale=0.6)
        worst = max(rel(cla_m, cla_e), rel(cda_m, cda_e))
        ok(f"model CLA and CDA = q x map x scale at vx {vx}, zs {zs} (worst rel {worst:.1e})", worst <= REL)

# (b) F3-like map of all 1.0: CLA = the scalar everywhere
F3 = build(write_car("f3.car", over={127: (9, (AX, AZ, FLAT_ONES.ravel())), 121: (1, 2.5)}))
ev3, _ = evaluator(F3)
ok("all-1.0 CLA map with scalar 2.5: record data is 2.5 everywhere",
   np.allclose(F3.cs.aero.cla_map["data"], 2.5, rtol=REL, atol=0.0))
worst = 0.0
for zs in (0.0, 0.006, -0.004):
    o = ev3(state(F3, vx=20.0, zs=zs))
    worst = max(worst, rel(o["f_lift"] / (0.5 * RHO_B * 400.0), 2.5))
ok(f"all-1.0 map: CLA equals the scalar 2.5 at three heights (worst rel {worst:.1e})", worst <= REL)

# (c) a zero CLA map with cid 121 = 0 builds and gives no lift
Z0 = build(write_car("zero_cla.car", over={127: (9, (AX, AZ, np.zeros(81))), 121: (1, 0.0)}))
ev_z0, _ = evaluator(Z0)
o = ev_z0(state(Z0, vx=20.0))
ok("zero CLA map with cid 121 = 0 builds, lift is exactly 0", o["f_lift"] == 0.0 and o["f_drag"] > 0)

# (d) a zero map or a zero scalar gives a zero force, CLA and CDA separately (ChassisSim: no refusal). A missing
#     scalar and a negative peak still raise ValueError.
def build_msg(name, over=None, omit=()):
    try:
        quiet(build, write_car(name, over=over, omit=omit))
    except ValueError as e:
        return str(e)
    return ""


ZERO_CLA = {127: (9, (AX, AZ, np.zeros(81)))}
ZERO_CDA = {131: (9, (AX, AZ, np.zeros(81)))}
for label, name, over in (("zero CLA map, cid 121 = 1.0", "zero_cla_1.car", {**ZERO_CLA, 121: (1, 1.0)}),
                          ("CLA scalar 0, base map", "scalar_cla_0.car", {121: (1, 0.0)}),
                          ("zero CDA map, cid 122 = 0.6", "zero_cda_map.car", ZERO_CDA),
                          ("CDA scalar 0, base map", "scalar_cda_0.car", {122: (1, 0.0)})):
    Zc = build(write_car(name, over=over))
    ev_zc, _ = evaluator(Zc)
    o = ev_zc(state(Zc, vx=20.0))
    if "CLA" in label:
        ok(f"{label} builds, lift is exactly 0 (drag {o['f_drag']:.3f} N stays > 0)", o["f_lift"] == 0.0 and o["f_drag"] > 0)
    else:
        ok(f"{label} builds, drag is exactly 0 (lift {o['f_lift']:.3f} N stays > 0)", o["f_drag"] == 0.0 and o["f_lift"] > 0)
msg = build_msg("no_scalar.car", omit=(121,))
ok(f"missing CLA scalar (cid 121) still raises ValueError ({msg[:50]!r})", "scalar" in msg)
msg = build_msg("neg_peak.car", over={127: (9, (AX, AZ, -np.ones(81)))})
ok(f"CLA map with a negative peak still raises ValueError ({msg[:50]!r})", "negative peak" in msg)

# (e) the AB offset is added to the balance map, nothing else
OFF = build(write_car("offset.car", over={123: (1, 0.1)}))
ev_o, _ = evaluator(OFF)
worst = 0.0
for vx in (15.0, 30.0):
    o = ev_o(state(OFF, vx=vx))
    _, _, ab_m = model_coeffs(o, vx)
    _, _, ab_e = base_terms(o, offset=0.1)
    worst = max(worst, abs(ab_m - ab_e))
ok(f"AB = balance map + offset 0.1 (worst abs err {worst:.1e}), the map itself unchanged",
   worst <= REL and np.array_equal(OFF.cs.aero.balance_map["data"], RAW_AB))

print("[cs_aero] 2. sideslip")
SS = build(write_car("ss.car", over={128: (7, (SS_X, SS_CLA)), 132: (7, (SS_X, SS_CDA)), 136: (7, (SS_X, SS_AB))}))
ev_s, _ = evaluator(SS)
VX = 20.0
for label, vy in (("vy 0", 0.0), ("interior vy 3", 3.0), ("interior vy -4", -4.0), ("clamp vy +", 0.57735 * VX),
                  ("clamp vy -", -0.57735 * VX)):
    o = ev_s(state(SS, vx=VX, vy=vy))
    cla_m, cda_m, ab_m = model_coeffs(o, VX)
    sb = sideslip(VX, vy)
    fc = [min(max(table_at((SS_X, y), sb), -0.4), 0.4) for y in (SS_CLA, SS_CDA, SS_AB)]
    cla_e, cda_e, ab_e = base_terms(o)
    cla_e, cda_e, ab_e = cla_e * (1 + fc[0]), cda_e * (1 + fc[1]), ab_e + fc[2]
    tol_clamp = 2e-3 if abs(table_at((SS_X, SS_CLA), sb)) > 0.4 else 1e-5
    ok(f"sideslip {label}: sin beta {sb:+.4f}, f_CLA {fc[0]:+.5f}, CLA worst rel "
       f"{max(rel(cla_m, cla_e), rel(cda_m, cda_e)):.1e}",
       rel(cla_m, cla_e) <= tol_clamp and rel(cda_m, cda_e) <= tol_clamp and abs(ab_m - ab_e) <= tol_clamp)
o = ev_s(state(SS, vx=VX, vy=0.0))
ok("vy = 0 gives f(0) = 0.1 on CLA (model factor 1.1 within 1e-5)",
   abs(model_coeffs(o, VX)[0] / base_terms(o)[0] - 1.1) <= 1e-5)
o = ev_s(state(SS, vx=VX, vy=0.57735 * VX))
ratio = model_coeffs(o, VX)[0] / base_terms(o)[0]
ok(f"table value 0.6 at sin beta 0.5 clamps to 0.4 (model factor {ratio:.5f}, np.clip 1.4)",
   abs(ratio - 1.4) <= 2e-3)

# odd table f(x) = 0.5 x on CLA only: vy +3 and -3 give opposite CLA changes, and the sign of the change at
# vy +3 is SIDESLIP_SIGN (the ChassisSim lateral axis is right positive, the model left positive)
SS_ODD = build(write_car("ss_odd.car", over={128: (7, (SS_X, 0.5 * SS_X))}))
ev_odd, _ = evaluator(SS_ODD)


def odd_factor(vy):
    o = ev_odd(state(SS_ODD, vx=VX, vy=vy))
    return model_coeffs(o, VX)[0] / base_terms(o)[0]


fp, fm = odd_factor(3.0), odd_factor(-3.0)
sb_p = sideslip(VX, 3.0)
ok(f"odd table: CLA change at vy +3 {fp - 1:+.6f} and at vy -3 {fm - 1:+.6f} are opposite",
   (fp - 1) * (fm - 1) < 0 and abs((fp - 1) + (fm - 1)) <= 1e-5)
ok(f"odd table: CLA change at vy +3 has the sign of SIDESLIP_SIGN ({SIDESLIP_SIGN:+.0f})",
   np.sign(fp - 1) == SIDESLIP_SIGN)
ok(f"odd table: CLA change at vy +3 equals 0.5 x sin beta {sb_p:+.5f} (err {abs((fp - 1) - 0.5 * sb_p):.1e})",
   abs((fp - 1) - 0.5 * sb_p) <= 1e-5)

print("[cs_aero] 3. speed")
SP = build(write_car("sp.car", over={413: (6, 1), 414: (6, 1), 415: (6, 1),
                                     416: (7, (SP_V, SP_CLA)), 417: (7, (SP_V, SP_CDA)), 418: (7, (SP_V, SP_AB))}))
ev_p, _ = evaluator(SP)
worst_c = worst_d = worst_a = 0.0
for vx in (10.0, 20.0, 33.0, 55.0):
    o = ev_p(state(SP, vx=vx))
    cla_m, cda_m, ab_m = model_coeffs(o, vx)
    cla_e, cda_e, ab_e = base_terms(o)
    worst_c = max(worst_c, rel(cla_m, cla_e * table_at((SP_V, SP_CLA), vx)))
    worst_d = max(worst_d, rel(cda_m, cda_e * table_at((SP_V, SP_CDA), vx)))
    worst_a = max(worst_a, abs(ab_m - (ab_e + table_at((SP_V, SP_AB), vx))))
ok(f"speed enables 1: CLA x r(vx) (worst rel {worst_c:.1e})", worst_c <= 1e-6)
ok(f"speed enables 1: CDA x r(vx) (worst rel {worst_d:.1e})", worst_d <= 1e-6)
ok(f"speed enables 1: AB + o(vx) (worst abs {worst_a:.1e})", worst_a <= 1e-6)
SPOFF = build(write_car("sp_off.car", over={413: (6, 0), 414: (6, 0), 415: (6, 0),
                                            416: (7, (SP_V, SP_CLA)), 417: (7, (SP_V, SP_CDA)),
                                            418: (7, (SP_V, SP_AB))}))
ev_po, _ = evaluator(SPOFF)
worst = 0.0
for vx in (10.0, 33.0, 55.0):
    a, b = ev_po(state(SPOFF, vx=vx)), ev_b(state(BASE, vx=vx))
    worst = max(worst, max(abs(a[k] - b[k]) / max(abs(b[k]), 1e-300) for k in ("f_lift", "f_drag", "f_lift_fl")))
ok(f"speed enables 0 with non-zero tables: unchanged vs base (worst rel {worst:.1e})", worst <= REL)

print("[cs_aero] 4. DRS")
DRS = build(write_car("drs.car", over={336: (6, 1), 337: (1, 0.2), 338: (1, 0.1), 339: (1, 0.15),
                                       340: (1, 200.0), 341: (1, 400.0)}))
ev_d, _ = evaluator(DRS)
ok("DRS car: pv has 2 rows (kappa, gate)", int(DRS.m23.pv.numel()) == 2)
ok("no-DRS car: pv has 1 row (kappa)", int(BASE.m23.pv.numel()) == 1)
s = DRS.track.s
g = np.asarray(DRS.track.drs, float)
ok(f"DRS gate ~1 in the middle of 200-400 m (at 300 m: {np.interp(300.0, s, g):.6f})",
   np.interp(300.0, s, g) > 0.999)
ok(f"DRS gate ~0 outside 200-400 m (at 100 m: {np.interp(100.0, s, g):.1e}, at 450 m: {np.interp(450.0, s, g):.1e})",
   np.interp(100.0, s, g) < 1e-3 and np.interp(450.0, s, g) < 1e-3)
DRW = build(write_car("drs_wrap.car", over={336: (6, 1), 337: (1, 0.2), 338: (1, 0.1), 339: (1, 0.15),
                                            340: (1, 500.0), 341: (1, 100.0)}))
sw, gw = DRW.track.s, np.asarray(DRW.track.drs, float)
ok(f"wrapped zone 500 -> 100 m: gate ~1 at 550 m and 50 m, ~0 at 300 m "
   f"({np.interp(550.0, sw, gw):.6f}, {np.interp(50.0, sw, gw):.6f}, {np.interp(300.0, sw, gw):.1e})",
   np.interp(550.0, sw, gw) > 0.999 and np.interp(50.0, sw, gw) > 0.999 and np.interp(300.0, sw, gw) < 1e-3)
ev_dw, _ = evaluator(DRW)
ok("wrapped zone car: pv has 2 rows", int(DRW.m23.pv.numel()) == 2)

msg = ""
try:
    build(write_car("drs_empty.car", over={336: (6, 1), 337: (1, 0.2), 338: (1, 0.1), 339: (1, 0.15),
                                           340: (1, 200.0), 341: (1, 200.0)}))
except ValueError as e:
    msg = str(e)
ok(f"DRS zone start == end raises ValueError ({msg[:60]!r})", "empty" in msg)

VXD = 25.0
o0 = ev_d(state(DRS, vx=VXD), gate=0.0)
ob = ev_b(state(BASE, vx=VXD))
worst = max(abs(o0[k] - ob[k]) / max(abs(ob[k]), 1e-300) for k in ("f_lift", "f_drag", "f_lift_fl"))
ok(f"DRS gate 0 gives no change vs the base car (worst rel {worst:.1e})", worst <= REL)
o1 = ev_d(state(DRS, vx=VXD), gate=1.0)
cla, cda, ab = base_terms(o1)
cla_n = cla - (0.2 + 0.1)
cda_n = cda - 0.15
ab_n = (cla * ab - 0.2) / cla_n
cla_m, cda_m, ab_m = model_coeffs(o1, VXD)
ok(f"DRS gate 1: CLA_new = CLA - (dCLA_f + dCLA_r) (rel {rel(cla_m, cla_n):.1e})", rel(cla_m, cla_n) <= REL)
ok(f"DRS gate 1: CDA_new = CDA - dCDA (rel {rel(cda_m, cda_n):.1e})", rel(cda_m, cda_n) <= REL)
ok(f"DRS gate 1: AB = (CLA AB - dCLA_f) / CLA_new (rel {rel(ab_m, ab_n):.1e})", rel(ab_m, ab_n) <= REL)
o_half = ev_d(state(DRS, vx=VXD), gate=0.5)
cla_h, _, _ = model_coeffs(o_half, VXD)
ok(f"DRS gate 0.5 gives half the CLA step (CLA {cla_h:.6f}, linear mix {0.5 * (cla + cla_n):.6f})",
   rel(cla_h, 0.5 * (cla + cla_n)) <= REL)

print("[cs_aero] 5. order: sideslip, DRS, speed")
ORD = build(write_car("order.car", over={128: (7, (SS_X, SS_CLA)), 132: (7, (SS_X, SS_CDA)), 136: (7, (SS_X, SS_AB)),
                                         336: (6, 1), 337: (1, 0.2), 338: (1, 0.1), 339: (1, 0.15),
                                         340: (1, 200.0), 341: (1, 400.0),
                                         413: (6, 1), 414: (6, 1), 415: (6, 1),
                                         416: (7, (SP_V, SP_CLA)), 417: (7, (SP_V, SP_CDA)),
                                         418: (7, (SP_V, SP_AB))}))
ev_or, _ = evaluator(ORD)
VXO, VYO = 25.0, 4.0
o = ev_or(state(ORD, vx=VXO, vy=VYO), gate=1.0)
cla, cda, ab = base_terms(o)
sb = sideslip(VXO, VYO)
fc = [min(max(table_at((SS_X, y), sb), -0.4), 0.4) for y in (SS_CLA, SS_CDA, SS_AB)]
cla1, cda1, ab1 = cla * (1 + fc[0]), cda * (1 + fc[1]), ab + fc[2]          # sideslip
cla2, cda2 = cla1 - (0.2 + 0.1), cda1 - 0.15                                  # DRS
ab2 = (cla1 * ab1 - 0.2) / cla2
cla3, cda3 = cla2 * table_at((SP_V, SP_CLA), VXO), cda2 * table_at((SP_V, SP_CDA), VXO)   # speed
ab3 = ab2 + table_at((SP_V, SP_AB), VXO)
cla_m, cda_m, ab_m = model_coeffs(o, VXO)
worst = max(rel(cla_m, cla3), rel(cda_m, cda3), abs(ab_m - ab3) / max(abs(ab3), 1e-300))
ok(f"sideslip, DRS, speed in that order: hand formula (worst rel {worst:.1e})", worst <= 1e-6)

print("[cs_aero] 6. refusals")
REFUSE = (
    ("steer enable 434 = 1", {434: (6, 1)}, "steer correction"),
    ("roll split table 139 non-zero", {139: (7, (SS_X, SS_CLA))}, "roll split"),
    ("2-D map 455 non-zero", {455: (9, (AX, AZ, RAW_CLA.ravel()))}, "2-D steer or state map"),
    ("wake map 872 non-zero", {872: (9, (AX, AZ, RAW_CLA.ravel()))}, "wake map"),
    ("multi-DRS 366 = 1", {366: (6, 1)}, "multiple DRS"),
)
for label, over, key in REFUSE:
    msg = ""
    try:
        build(write_car(f"refuse_{key[:6].replace(' ', '_')}.car", over=over))
    except ValueError as e:
        msg = str(e)
    ok(f"{label}: ValueError naming '{key}' ({msg[:60]!r})", key in msg)

print("[cs_aero] 7. skip rule: no correction, DRS off, same graph as the cids left out")
SKIP_KEYS = (128, 132, 136, 139, 413, 414, 415, 416, 417, 418, 434, 435, 436, 437, 438, 439, *range(452, 464),
             870, 872, 873, 874, 875, 336, 337, 338, 339, 340, 341, 366, 367, 368, 369, 370, 371)
ZC = build(write_car("skip_zero.car"))
ZO = build(write_car("skip_omit.car", omit=SKIP_KEYS))
ev_zc, dx_zc = evaluator(ZC)
ev_zo, dx_zo = evaluator(ZO)
rng = np.random.default_rng(8)
worst_abs = 0.0
same_size = int(ZC.m23.nx) == int(ZO.m23.nx) and int(ZC.m23.pv.numel()) == int(ZO.m23.pv.numel()) == 1
for _ in range(6):
    x = state(ZC, vx=rng.uniform(12.0, 45.0), vy=rng.uniform(-2.0, 2.0), zs=rng.uniform(-0.005, 0.005),
              theta=rng.uniform(-0.01, 0.01))
    x[2:5] = rng.uniform(-0.05, 0.05, 3)
    x[10:15] = rng.uniform(-0.2, 0.2, 5)
    u = rng.uniform(-0.3, 0.3, 3)
    a, b = dx_zc(x, u), dx_zo(x, u)
    worst_abs = max(worst_abs, float(np.max(np.abs(a - b))) if np.all(np.isfinite(a)) else np.inf)
ok(f"no correction, DRS off: dx of the car with zero tables == dx without the cids (max abs {worst_abs:.1e})",
   same_size and worst_abs == 0.0)

print("[cs_aero] 8. QSS: corrections and DRS in build_envelope_cs")
env_b = build_envelope_cs(BASE)
env_sp = build_envelope_cs(SP)
i40 = int(np.argmin(np.abs(env_b.v_grid - 40.0)))
ok(f"speed correction changes ay_max at 40 m/s (base {env_b.ay_max[i40]:.4f}, corrected {env_sp.ay_max[i40]:.4f})",
   abs(env_sp.ay_max[i40] - env_b.ay_max[i40]) > 1e-3 * abs(env_b.ay_max[i40]))
# sideslip at zero sideslip only: a table with f(0) = 0.1 gives the same QSS as a flat 0.1 table and as a 1.1 scalar
SS_ONLY = build(write_car("ss_only.car", over={128: (7, (SS_X, SS_CLA))}))
SS_FLAT = build(write_car("ss_flat.car", over={128: (7, (SS_X, np.full(5, 0.1)))}))
SS_SCALE = build(write_car("ss_scale.car", over={121: (1, CLA_SCALAR * 1.1)}))
e_ss = build_envelope_cs(SS_ONLY)
e_fl = build_envelope_cs(SS_FLAT)
e_sc = build_envelope_cs(SS_SCALE)
w1 = float(np.max(np.abs(e_ss.ay_max - e_fl.ay_max) / np.maximum(np.abs(e_fl.ay_max), 1e-12)))
w2 = float(np.max(np.abs(e_ss.ay_max - e_sc.ay_max) / np.maximum(np.abs(e_sc.ay_max), 1e-12)))
ok(f"sideslip in the QSS reads x = 0 only: table == flat 0.1 table on ay_max (worst rel {w1:.1e})", w1 <= 1e-9)
ok(f"sideslip in the QSS: f(0) = 0.1 equals CLA scalar x 1.1 on ay_max (worst rel {w2:.1e})", w2 <= 1e-9)
# DRS does not change the QSS
e_drs = build_envelope_cs(DRS)
ok("DRS does not change the QSS: ay_max, Fx_trac and Fx_brk equal the base car",
   np.array_equal(e_drs.ay_max, env_b.ay_max) and np.array_equal(e_drs.Fx_trac, env_b.Fx_trac)
   and np.array_equal(e_drs.Fx_brk, env_b.Fx_brk))
# speed correction at QSS: a flat ratio 1.2 on CLA equals the scalar x 1.2 (the QSS evaluates each grid speed)
SP_FLAT = build(write_car("sp_flat.car", over={413: (6, 1), 416: (7, (SP_V, np.full(4, 1.2)))}))
SC12 = build(write_car("sc12.car", over={121: (1, CLA_SCALAR * 1.2)}))
ef, es = build_envelope_cs(SP_FLAT), build_envelope_cs(SC12)
w3 = float(np.max(np.abs(ef.ay_max - es.ay_max) / np.maximum(np.abs(es.ay_max), 1e-12)))
ok(f"speed correction in the QSS: flat ratio 1.2 equals scalar x 1.2 on ay_max (worst rel {w3:.1e})", w3 <= 1e-9)

print("[cs_aero] 9. refine refuses a DRS car")
msg = ""
try:
    build(write_car("drs_refine.car", over={336: (6, 1), 337: (1, 0.2), 338: (1, 0.1), 339: (1, 0.15),
                                            340: (1, 200.0), 341: (1, 400.0)}), refine=True)
except ValueError as e:
    msg = str(e)
ok(f"refine with a DRS car raises ValueError ({msg[:60]!r})", "DRS" in msg and "refine" in msg)

print(f"{PASS} checks passed (synthetic)")


# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs_dir):
    print("[cs_aero] vendor GP3 2013 Eastern Creek")
    car_path = Path(cs_dir).parent / "Models" / "GP3 2013" / "Eastern Creek, AU" / "def_GP3 2013.car"
    ok("GP3 2013 Eastern Creek car file found", car_path.is_file())
    cs = load_car(str(car_path))
    cla_peak = float(np.max(cs.aero.cla_map["data"]))
    cda_peak = float(np.max(cs.aero.cda_map["data"]))
    ok(f"GP3 CLA peak in the record = 3.48 (read {cla_peak:.6f})", abs(cla_peak - 3.48) < 1e-3)
    ok(f"GP3 CDA peak in the record = 0.96 (read {cda_peak:.6f})", abs(cda_peak - 0.96) < 1e-3)
    ctx = build(car_path, chassis_dir=cs_dir, circuit="Eastern Creek")
    ok("GP3 builds with no refusal (aero_refusals silent)", ctx.cs is not None and ctx.m23 is not None)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
print(f"{PASS} checks passed")
