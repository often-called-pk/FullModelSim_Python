"""ChassisSim car record of functions/chassissim.py (ChassisSim phase 2, task 1).

The synthetic checks write a car file with carfile.encode from arbitrary values. The cids and their record tags
are written here as literal numbers, not taken from CAR_FIELDS, so a wrong cid in CAR_FIELDS fails. The checks
cover the SI conversions, the None rule, the tyre decode (tables 4 and 5 included), the hardpoint order, the
corner order of the file (FR, FL, RR, RL, with a distinct value on each corner), the left-takes-right copy rule,
the bump steer tables and their guard in suspension_curves, the strict car_vp and its
vehParams totals (matlab and corrected physics). The vendor section runs only when CHASSISSIM_DIR names the
ChassisSim install folder. It reads the GP3 2013 Eastern Creek car file and checks relations only, no copied
values. It writes nothing into the vendor folders.

    python tests/test_cs_car.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_car.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from functions import carfile
from functions.chassissim import CAR_FIELDS, CS_JW, car_vp, load_car, suspension_curves
from functions.context import Ctx
from Powertrain import Powertrain
from vehParams import vehParams

PASS = 0
RPM_T = 2 * np.pi / 60    # rpm -> rad/s, written here on purpose (not imported from chassissim)
DEG_T = np.pi / 180
LBF_IN = 175.126835      # N/m per lbf/in
PHYSICS = ("matlab", "corrected")
TMP = tempfile.TemporaryDirectory()
CORNERS = ("fl", "fr", "rl", "rr")
FILE = ("fr", "fl", "rr", "rl")     # corner order of every four-record group in the file, written here on purpose


def ok(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASS += 1


def arb(n, s):
    """n arbitrary finite numbers, different for each seed s."""
    return np.sin(np.arange(n) * 0.9 + s) * 7.3 - 0.61


# ---- synthetic car: literal cids and tags ------------------------------------------------------------
# Tags as the file holds them: 1 scalar, 2 array, 6 byte, 7 curve, 9 map, 10 hardpoints,
# 11 spring-like (preload, rate, x, y), 12 damper, 13 tyre.
TYRE = {  # dims (d1..d7, d2 = 2 temperatures), radius m, toe deg, camber deg, castor deg, drag, kz N/m
    "fl": dict(dims=(3, 2, 2, 2, 2, 2, 2), radius=0.300, toe=0.5, camber=-1.0, castor=2.0, drag=0.011, kz=1.60e5),
    "fr": dict(dims=(2, 2, 3, 2, 2, 3, 2), radius=0.305, toe=-0.25, camber=-1.5, castor=2.5, drag=0.012, kz=1.62e5),
    "rl": dict(dims=(4, 2, 2, 3, 2, 2, 2), radius=0.310, toe=0.1, camber=-0.5, castor=1.5, drag=0.0135, kz=1.40e5),
    "rr": dict(dims=(2, 2, 2, 2, 3, 2, 3), radius=0.315, toe=0.0, camber=-0.75, castor=1.0, drag=0.014, kz=1.42e5),
}
TYRE_TEMPS = (300.0, 350.0)    # the two temperature columns of the load grid, K
TYRE_BG = 325.0                # tyre background temperature (scalar 6), between the two columns
TRACK_TEMP = 350.0             # cid 390, equal to the second column: the wrong column would show it
PRESSURE = 2.1e5               # Pa (scalar 5); the kz table reads it in kPa
VEL = np.array([-400.0, -100.0, 0.0, 100.0, 400.0])   # damper velocity, mm/s
BUMP = np.array([0.0, 900.0, 1800.0, 2700.0, 3600.0])  # N
REB = np.array([3600.0, 2400.0, 1200.0, 0.0, 0.0])     # N, positive as the docstring says
SPRING = {1: (1.23e5, 0.020), 2: (1.31e5, 0.021), 3: (0.97e5, 0.019), 4: (1.02e5, 0.0205),
          5: (9.0e4, 0.0), 6: (8.0e4, 0.0)}           # cid -> (rate N/m, preload m)
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}   # cid -> (rate, preload m)
ARB = {19: 2.0e4, 21: 1.5e4}                          # cid -> rate N/m
HARD = (49, 50, 51, 52)                                   # front_r, front_l, rear_r, rear_l
# left-side hardpoints (car frame: x rearward, y left +, z up), rows: lower inner front, lower inner aft,
# lower outer, upper inner front, upper inner aft, upper outer. The right side is the mirror in y.
HP_LEFT = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.68, 0.11],
                    [-0.15, 0.30, 0.28], [0.15, 0.30, 0.28], [0.0, 0.64, 0.33]])
HP_RIGHT = HP_LEFT * np.array([1.0, -1.0, 1.0])
GAP_FLAG = {204: 1, 205: 0, 206: 0, 207: 1, 208: 1, 209: 1}    # cid -> rubber ground-gap flag (distinct left and right)
DAMP = {cid: 1.0 + 0.1 * (cid - 31) for cid in range(31, 37)}   # cid -> factor of the damper bump force (distinct per record)
BUMP_X = np.linspace(-0.02, 0.02, 11)                     # m, wheel travel grid of the bump steer tables (74-77)
AERO_X = np.linspace(0.0, 0.4, 5)                         # axis of the task 8 aero tables and maps (zero values)
CS_TOTAL, CS_UF, CS_UR = 700.0, 25.0, 35.0
CS_FRONT = 0.44
EXPLICIT = {  # tag 1 cids with a literal value
    84: CS_TOTAL, 85: CS_UF, 86: CS_UR, 87: 60.0, 88: 280.0, 89: 600.0, 96: CS_FRONT,
    90: 3.1, 91: 1.52, 92: 1.49, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,            # ride heights fr, fl, rr, rl, mm
    100: 0.9, 102: 1000.0, 103: 9000.0,                     # efficiency, idle and limit rpm
    107: 14000.0, 108: 0.6, 109: 0.05,                      # brake force N, front bias, delay s
    111: 0.0, 112: 1.0, 113: 0.0,                           # torque split (0: userOpts needs it), diff types
    55: 0.050, 59: 0.052, 63: 0.061, 67: 0.063,            # roll-centre heights fr, fl, rr, rl, m
    54: 0.045, 58: 0.047, 62: 0.058, 66: 0.060,            # first values of the same blocks, m
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,                # CLA, CDA, offset, rho
    390: TRACK_TEMP,
}
EXPLICIT.update({c: 0.5 + 0.01 * c for c in range(7, 13)})            # suspension gaps, no unit
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})        # DRS
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
MAPS = {127: ((4, 3), "cla"), 131: ((3, 2), "cda"), 135: ((5, 2), "balance"),
        114: ((3, 2), "diff_f"), 116: ((2, 4), "diff_r")}          # cid -> ((nx, nz), name)


def tyre_sizes(d):
    """f64 count of the 13 tables of a tyre (carfile docstring: 2D(k) = 2k, 3D(a, b) = a + b + a b)."""
    d1, d2, d3, d4, d5, d6, d7 = d

    def g(a, b):
        return a + b + a * b
    return [2 * d3, 2 * d4, 2 * d4, g(d1, d2), g(d6, d7), 2 * d6, 2 * d4, g(d1, d2), 2 * d3, 2 * d6, 2 * d5, 2 * d5, 2 * d5]


def tyre_record(i, corner):
    """Tyre record (tag 13) with a two-column load grid and a kPa pressure axis. Return (record, reference)."""
    t = TYRE[corner]
    dims = t["dims"]
    d1, d2, d3, _, d5, d6, _ = dims
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(dims))]
    A, B = arb(d1, 20 + i), arb(d1, 30 + i)
    load = 1000.0 * np.arange(1, d1 + 1)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, np.stack([A, B], axis=1).ravel()])   # load, temperatures, grid (row = load)
    px = np.linspace(150.0, 250.0, d5)                       # kPa
    py = t["kz"] * (1.0 + 0.01 * np.arange(d5))             # N/m
    tabs[10] = np.concatenate([px, py])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = t["toe"], t["camber"], t["castor"], t["radius"], PRESSURE, TYRE_BG
    sc[7], sc[8], sc[9], sc[11], sc[15] = t["drag"], 0.11 + 0.01 * i, 0.12 + 0.01 * i, 0.7 + 0.1 * i, 1.05 + 0.1 * i
    ref = dict(A=A, B=B, load=load, slip_x=tabs[0][:d3], slip_y=tabs[0][d3:], camber_raw=tabs[5][:d6],
               camber_y=tabs[5][d6:], kz=float(np.interp(PRESSURE * 1e-3, px, py)), sc=sc, dims=dims)
    return (13, 80 + i, (list(dims), sc, tabs)), ref


def build_records(omit=(), over=None):
    """Literal records of the synthetic car. over maps a cid to its replacement value. Return (records, tyre
    references by corner)."""
    R, ref = [], {}
    over = over or {}

    def add(tag, cid, v):
        if cid not in omit:
            R.append((tag, cid, over.get(cid, v)))
    for cid, v in EXPLICIT.items():
        add(1, cid, v)
    for cid, (rate, pre) in SPRING.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.05]), np.array([0.0, 500.0])))
    for cid, (rate, pre) in RUBBER.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.02]), np.array([0.0, 800.0])))
    for cid, rate in ARB.items():
        add(11, cid, (1, 0.0, rate, np.array([0.0, 0.1]), np.array([0.0, 300.0])))
    for cid in (*range(23, 31), 170, 171):
        add(2, cid, arb(5, cid))                                          # motion ratio coefficients (bars: 27 FR, 170 FL, 28 RR, 171 RL)
    for cid in range(366, 372):
        add(6, cid, 10 + cid - 366)                                       # one byte
    for cid in range(204, 210):
        add(6, cid, GAP_FLAG[cid])                                        # rubber ground-gap flags (fr, fl, rr, rl, third f/r), one byte
    for cid in (638, 639, *range(657, 661)):
        add(6, cid, 0)                                                    # hysteresis flags (third f/r, then fr, fl, rr, rl), off as shipped
    for cid in range(661, 665):
        add(4, cid, np.zeros(33))                                         # hysteresis curves, 3 x 11 (x, loading, unloading), zero as shipped
    for cid in (74, 75, 76, 77):
        add(7, cid, (BUMP_X, np.zeros(len(BUMP_X))))                      # bump steer, all zero as shipped
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    for cid, ((nx, nz), _) in MAPS.items():
        add(9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid)))
    add(6, 655, 0)                                                        # stability index correction off, as shipped
    add(9, 656, (np.array([-50.0, 0.0, 50.0]), np.array([10.0, 30.0, 60.0]), np.zeros(9)))   # stability map, all zero
    # Task 8 aero corrections, refused features and DRS-free speed tables: 1-D tables (tag 7) zero in y,
    # enables (tag 6) 0, 2-D maps (tag 9) zero in data. Axes are non-degenerate; only the values are zero.
    for cid in (128, 132, 136, 139, 416, 417, 418, 437, 438, 439):
        add(7, cid, (AERO_X, np.zeros(len(AERO_X))))
    for cid in (413, 414, 415, 434, 435, 436):
        add(6, cid, 0)
    for cid in (*range(452, 464), 872, 873, 874, 875):
        add(9, cid, (AERO_X, AERO_X, np.zeros(len(AERO_X) ** 2)))
    # Powertrain and hybrid records of P1 to P4, off as shipped. Tags as the shipped files hold them: 1 scalar,
    # 6 flag or byte, 7 engine-brake table (x in rpm, y in N m), 8 text, 9 map (x in rpm, z throttle, data).
    for cid in (104, 105, 106, 179, 180, 181, 182, 183, 465, 836, 871):
        add(1, cid, 0.0)
    add(1, 160, 0.5)                                                     # brake left share, even split
    add(7, 161, (ENGINE_RPM, np.zeros(len(ENGINE_RPM))))                  # engine-brake table, zero in y
    for cid in (177, 178, 464, 466, 535, 573, 589, 590, 870, 876, 877, 878, 879, 419, 423, 426, 428, 520, 571, 598, 649):
        add(6, cid, 0)                                                   # flags, off
    for cid in (897, 906):
        add(8, cid, b"")                                                 # text, empty
    for cid in (467, 468):
        add(9, cid, (ENGINE_RPM, np.array([0.0, 1.0]), np.zeros(2 * len(ENGINE_RPM))))   # deploy map, zero data
    for cid in range(31, 37):                                            # dampers fr, fl, rr, rl, third f (35), third r (36)
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP * DAMP[cid], REB])))
    for cid, rows in zip(HARD, (HP_RIGHT, HP_LEFT, HP_RIGHT, HP_LEFT)):
        add(10, cid, (0, np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])))   # tag 10: x, y, z columns
    for i, c in enumerate(FILE):
        rec, ref[c] = tyre_record(i, c)
        add(*rec[:2], rec[2])
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return [r for r in R], ref


def write_car(name, omit=(), over=None):
    recs, ref = build_records(omit, over)
    path = os.path.join(TMP.name, name)
    carfile.write(recs, path)
    return path, ref


def through_vehparams(vp_overrides, physics):
    ctx = Ctx()
    ctx.physics = physics
    Powertrain(ctx)
    vehParams(ctx, vp_overrides=vp_overrides)
    return ctx.vp


def raises_naming(fn, field):
    try:
        fn()
    except ValueError as e:
        return field in str(e)
    return False


print("[cs_car] synthetic car")
SYN, REF = write_car("synthetic.car")
car = load_car(SYN)
written = {c for _, c, _ in build_records()[0] if c != carfile.END_CID}
ok("the synthetic car holds every CAR_FIELDS cid and no other",
   written == set(CAR_FIELDS))

# literal cids: the value of each cid must land in its own field
print("  -- literal cids")
ok("ride heights 117-120 read as m (17, 18.5, 20, 21.5 mm)",
   all(abs(getattr(car.geom, f"ride_height_{c}") - v * 1e-3) < 1e-12
       for c, v in zip(FILE, (17.0, 18.5, 20.0, 21.5))))
ok("brake force 107 N, bias 108, delay 109 s",
   abs(car.brake.force - 14000.0) < 1e-9 and abs(car.brake.front_bias - 0.6) < 1e-12
   and abs(car.brake.delay - 0.05) < 1e-12)
ok("idle rpm 102 and limit rpm 103 read as rad/s",
   abs(car.engine.rpm_idle - 1000.0 * RPM_T) < 1e-12 and abs(car.engine.rpm_limit - 9000.0 * RPM_T) < 1e-12)
ok("aero CLA 121, CDA 122, offset 123, rho 125 as stored",
   abs(car.aero.cla - 3.5) < 1e-12 and abs(car.aero.cda - 0.95) < 1e-12
   and abs(car.aero.offset - 0.1) < 1e-12 and abs(car.aero.rho - 1.2) < 1e-12)
ok("mass 84-86, 96 and geometry 90-93 read as stored",
   abs(car.mass.total - CS_TOTAL) < 1e-9 and abs(car.mass.unsprung_front - CS_UF) < 1e-9
   and abs(car.mass.unsprung_rear - CS_UR) < 1e-9 and abs(car.mass.front_share - CS_FRONT) < 1e-9
   and abs(car.geom.wheelbase - 3.1) < 1e-12 and abs(car.geom.cg_height - 0.33) < 1e-12
   and abs(car.geom.track_front - 1.52) < 1e-12 and abs(car.geom.track_rear - 1.49) < 1e-12)
ok("inertias 87-89 read as stored", abs(car.mass.Ix - 60.0) < 1e-9 and abs(car.mass.Iy - 280.0) < 1e-9
   and abs(car.mass.Iz - 600.0) < 1e-9)
ok("track temperature 390 read as stored", abs(car.env.track_temp - TRACK_TEMP) < 1e-12)
ok("engine efficiency 100, torque split 111 (0), diff types 112 and 113 as stored",
   abs(car.engine.efficiency - 0.9) < 1e-12 and car.engine.torque_split == 0.0
   and car.diff.front_type == 1.0 and car.diff.rear_type == 0.0)
ok("roll-centre heights 55, 59, 63, 67 (fr, fl, rr, rl) read in m as stored",
   abs(car.geom.rc_height_fr - 0.050) < 1e-12 and abs(car.geom.rc_height_fl - 0.052) < 1e-12
   and abs(car.geom.rc_height_rr - 0.061) < 1e-12 and abs(car.geom.rc_height_rl - 0.063) < 1e-12)
ok("RC heights at the hardpoint datum 54, 58, 62, 66 (fr, fl, rr, rl) read in m as stored",
   abs(car.geom.rc_datum0_fr - 0.045) < 1e-12 and abs(car.geom.rc_datum0_fl - 0.047) < 1e-12
   and abs(car.geom.rc_datum0_rr - 0.058) < 1e-12 and abs(car.geom.rc_datum0_rl - 0.060) < 1e-12)

# groups: one decoded value per group
print("  -- groups")
ok("springs 1-4 (fr, fl, rr, rl): rate in N/m and preload in m, and the third springs",
   all(abs(getattr(car.susp, f"spring_{c}")["rate"] - SPRING[i + 1][0]) < 1e-6
       and abs(getattr(car.susp, f"spring_{c}")["preload"] - SPRING[i + 1][1]) < 1e-12
       for i, c in enumerate(FILE))
   and abs(car.susp.spring_third_f["rate"] - SPRING[5][0]) < 1e-6
   and abs(car.susp.spring_third_r["rate"] - SPRING[6][0]) < 1e-6)
ok("rubbers 13-16 (fr, fl, rr, rl) and 17-18: rate, preload, x and y as stored",
   all(abs(getattr(car.susp, f"rubber_{c}")["rate"] - RUBBER[13 + i][0]) < 1e-6
       and abs(getattr(car.susp, f"rubber_{c}")["preload"] - RUBBER[13 + i][1]) < 1e-12
       and np.allclose(getattr(car.susp, f"rubber_{c}")["x"], [0.0, 0.02])
       and np.allclose(getattr(car.susp, f"rubber_{c}")["y"], [0.0, 800.0])
       for i, c in enumerate(FILE))
   and abs(car.susp.rubber_third_f["rate"] - RUBBER[17][0]) < 1e-6
   and abs(car.susp.rubber_third_r["rate"] - RUBBER[18][0]) < 1e-6)
ok("ARBs 19 and 21 rate in N/m as stored",
   abs(car.susp.arb_f["rate"] - ARB[19]) < 1e-6 and abs(car.susp.arb_r["rate"] - ARB[21]) < 1e-6)
ok("motion ratio coefficients 23-26 (fr, fl, rr, rl) and 29-30 read as stored arrays of 5",
   all(np.allclose(getattr(car.susp, f"mr_{c}"), arb(5, 23 + i)) for i, c in enumerate(FILE))
   and np.allclose(car.susp.mr_third_f, arb(5, 29)) and np.allclose(car.susp.mr_third_r, arb(5, 30)))
ok("bar motion ratios: 27 FR, 170 FL, 28 RR, 171 RL",
   all(np.allclose(getattr(car.susp, f"mr_arb_{c}"), arb(5, cid), rtol=0, atol=0)
       for c, cid in (("fr", 27), ("fl", 170), ("rr", 28), ("rl", 171))))
ok("gaps 7-12 read as stored",
   all(abs(getattr(car.susp, f"gap_{n}") - EXPLICIT[7 + i]) < 1e-12
       for i, n in enumerate(FILE + ("third_f", "third_r"))))
ok("rubber ground-gap flags 204-207 (fr, fl, rr, rl) and 208-209 (third f, third r) read as stored",
   all(getattr(car.susp, f"gap_flag_{c}") == GAP_FLAG[204 + i] for i, c in enumerate(FILE + ("third_f", "third_r"))))
ok("damper velocity in mm/s read as m/s, bump and rebound in N as stored (rebound positive)",
   all(np.allclose(getattr(car.susp, f"damper_{c}")["velocity"], VEL * 1e-3, rtol=0, atol=1e-12)
       and np.allclose(getattr(car.susp, f"damper_{c}")["bump"], BUMP * DAMP[31 + i], rtol=0, atol=0)
       and np.allclose(getattr(car.susp, f"damper_{c}")["rebound"], REB, rtol=0, atol=0)
       for i, c in enumerate(FILE))      # dampers 31-34: fr, fl, rr, rl
   and np.allclose(car.susp.damper_third_f["velocity"], VEL * 1e-3, rtol=0, atol=1e-12)
   and np.allclose(car.susp.damper_third_r["bump"], BUMP * DAMP[36], rtol=0, atol=0)
   and np.allclose(car.susp.damper_third_r["rebound"], REB, rtol=0, atol=0))
ok("hardpoint blocks 49-52 (front_r, front_l, rear_r, rear_l) have shape (6, 3), and row i holds point i",
   all(getattr(car.hardpoints, n).shape == (6, 3) for n in ("front_l", "front_r", "rear_l", "rear_r"))
   and np.allclose(car.hardpoints.front_l, HP_LEFT, rtol=0, atol=0)
   and np.allclose(car.hardpoints.front_r, HP_RIGHT, rtol=0, atol=0)
   and np.allclose(car.hardpoints.rear_l, HP_LEFT, rtol=0, atol=0)
   and np.allclose(car.hardpoints.rear_r, HP_RIGHT, rtol=0, atol=0))
ok("aero maps have shape (nx, nz) and hold the stored data (cla 4x3, cda 3x2, balance 5x2)",
   car.aero.cla_map["data"].shape == (4, 3) and car.aero.cda_map["data"].shape == (3, 2)
   and car.aero.balance_map["data"].shape == (5, 2)
   and np.allclose(car.aero.cla_map["data"], arb(12, 127).reshape(4, 3) * (3.5 / arb(12, 127).max()))   # scaled: map / peak x 3.5
   and np.allclose(car.aero.balance_map["data"], arb(10, 135).reshape(5, 2)))
ok("diff maps 114 (3 x 2) and 116 (2 x 4) have their shape and data",
   car.diff.front_map["data"].shape == (3, 2) and car.diff.rear_map["data"].shape == (2, 4)
   and np.allclose(car.diff.front_map["data"], arb(6, 114).reshape(3, 2))
   and np.allclose(car.diff.rear_map["data"], arb(8, 116).reshape(2, 4)))
ok("DRS 336 and multi-DRS 366 (one byte) as stored",
   abs(car.aero.drs_336 - (0.2 + 0.001 * 336)) < 1e-12 and car.aero.multi_drs_366 == 10)
ok("engine torque curve: rpm axis in rad/s, torque in N m",
   np.allclose(car.engine.torque_curve["x"], ENGINE_RPM * RPM_T, rtol=0, atol=1e-12)
   and np.allclose(car.engine.torque_curve["y"], ENGINE_NM, rtol=0, atol=0))
ok("gears read as stored (6)", len(car.engine.gears) == 6 and np.allclose(car.engine.gears, GEARS, rtol=0, atol=0))
ok("P1 cids read as stored: cut delay 104, brake left share 160 (0.5), regen front share 871 (0)",
   car.engine.cut_delay == 0.0 and car.brake.left_share == 0.5 and car.hybrid.regen_front_share == 0.0)
ok("P1 flags and text read as stored: 571 and 897 off (0 and empty bytes), 870 flag 0",
   car.refused.c571 == 0 and car.refused.c897 == b"" and car.hybrid.regen_front_on == 0)
ok("hybrid deploy maps 467 (front) and 468 (rear): rpm axis in rad/s (3000 rpm -> 314.159), throttle z, zero data",
   abs(car.hybrid.map_front["x"][1] - 314.159) < 1e-3 and abs(car.hybrid.map_rear["x"][2] - 6000 * RPM_T) < 1e-9
   and np.allclose(car.hybrid.map_front["z"], [0.0, 1.0], rtol=0, atol=0)
   and car.hybrid.map_rear["data"].shape == (len(ENGINE_RPM), 2) and not np.any(car.hybrid.map_rear["data"]))
ok("engine-brake table 161: rpm axis in rad/s, zero y",
   np.allclose(car.engine.brake_table["x"], ENGINE_RPM * RPM_T, rtol=0, atol=1e-12)
   and not np.any(car.engine.brake_table["y"]))

# tyres: every decoded field of one tyre record
print("  -- tyres")
for c in CORNERS:
    r, t = REF[c], TYRE[c]
    tyre = getattr(car.tyre, c)
    ok(f"tyre {c}: radius, toe and camber (deg to rad), castor, pressure, temp, rolling drag",
       abs(tyre["radius"] - t["radius"]) < 1e-12 and abs(tyre["toe"] - t["toe"] * DEG_T) < 1e-12
       and abs(tyre["camber"] - t["camber"] * DEG_T) < 1e-12 and abs(tyre["castor"] - t["castor"] * DEG_T) < 1e-12
       and abs(tyre["pressure"] - PRESSURE) < 1e-6 and abs(tyre["temp"] - TYRE_BG) < 1e-12
       and abs(tyre["rolling_drag"] - t["drag"]) < 1e-12)
    ok(f"tyre {c}: peak slip angle and ratio, grip, optimum camber (deg to rad)",
       abs(tyre["peak_slip_angle"] - r["sc"][8]) < 1e-12 and abs(tyre["peak_slip_ratio"] - r["sc"][9]) < 1e-12
       and abs(tyre["grip"] - r["sc"][15]) < 1e-12 and abs(tyre["optimum_camber"] - r["sc"][11] * DEG_T) < 1e-12)
    ok(f"tyre {c}: slip_x, slip_y and camber_x (deg to rad), camber_y as stored",
       np.allclose(tyre["slip_x"], r["slip_x"], rtol=0, atol=0) and np.allclose(tyre["slip_y"], r["slip_y"], rtol=0, atol=0)
       and np.allclose(tyre["camber_x"], r["camber_raw"] * DEG_T, rtol=0, atol=1e-15)
       and np.allclose(tyre["camber_y"], r["camber_y"], rtol=0, atol=0))
    ok(f"tyre {c}: load axis, one fy_peak per load at scalar 6 (midpoint of 300 and 350 K)",
       np.allclose(tyre["load"], r["load"]) and tyre["fy_peak"].shape == (len(r["load"]),)
       and np.allclose(tyre["fy_peak"], 0.5 * (r["A"] + r["B"]), rtol=0, atol=1e-12))
    ok(f"tyre {c}: kz read at the pressure in kPa (interp at 210 kPa)",
       abs(tyre["kz"] - r["kz"]) < 1e-6 * r["kz"])
ok("track temperature 350 K (cid 390) is not the interpolation column",
   not np.allclose(car.tyre.rl["fy_peak"], REF["rl"]["B"], rtol=0, atol=1e-6))

# shapes of the tyre tables count: slip and camber lengths follow the dims
ok("tyre tables follow their dims (rr: slip_x 2, camber_y 3-dim table)",
   len(car.tyre.rr["slip_x"]) == TYRE["rr"]["dims"][2] and len(car.tyre.rr["camber_x"]) == TYRE["rr"]["dims"][5])

# file identity and the None rule
print("  -- file identity and missing cids")
ok("sha8 is the first 8 hex digits of the sha256 of the file bytes",
   car.sha8 == hashlib.sha256(Path(SYN).read_bytes()).hexdigest()[:8])
ok("path is absolute and equals the path given", os.path.isabs(car.path) and car.path == os.path.abspath(SYN))
here = os.getcwd()
os.chdir(TMP.name)
try:
    rel = load_car("synthetic.car")
finally:
    os.chdir(here)
ok("a relative path loads and stores the absolute path", os.path.isabs(rel.path) and os.path.samefile(rel.path, SYN))
no_cda, _ = write_car("no_cda.car", omit=(122, 131, 390))     # the CDA map goes with its scalar (task 8 rule)
cn = load_car(no_cda)
ok("a cid the file lacks gives None (aero.cda, aero.cda_map, env.track_temp), the others still read",
   cn.aero.cda is None and cn.aero.cda_map is None and cn.env.track_temp is None and abs(cn.aero.cla - 3.5) < 1e-12)
cda_no_scalar, _ = write_car("cda_no_scalar.car", omit=(122,))
ok("a CDA map without its scalar 122 raises ValueError naming the scalar",
   raises_naming(lambda: load_car(cda_no_scalar), "missing scalar 'cda'"))

print("  -- bump steer, tyre table 4, suspension curves")
ok("bump steer tables 74-77 (fr, fl, rr, rl) hold the travel grid x and a zero y (synthetic car)",
   all(np.allclose(getattr(car.susp, f"bump_steer_{c}")["x"], BUMP_X, rtol=0, atol=0)
       and not np.any(getattr(car.susp, f"bump_steer_{c}")["y"]) for c in CORNERS))
BUMP_Y = {c: (i + 1) * 1e-3 * BUMP_X / 0.02 for i, c in enumerate(CORNERS)}   # a distinct line per corner, 1-4 mm


def bump_over(ys):
    """Replacement records of cids 74-77 (FR, FL, RR, RL): the toe change ys[corner], zero for a corner not in ys."""
    return {74 + i: (BUMP_X, ys.get(c, np.zeros(len(BUMP_X)))) for i, c in enumerate(FILE)}


bump = load_car(write_car("bump_all.car", over=bump_over(BUMP_Y))[0])
ok("bump steer tables decode per corner: x is the travel grid, y the stored toe change (distinct per corner)",
   all(np.allclose(getattr(bump.susp, f"bump_steer_{c}")["x"], BUMP_X, rtol=0, atol=0)
       and np.allclose(getattr(bump.susp, f"bump_steer_{c}")["y"], BUMP_Y[c], rtol=0, atol=0) for c in CORNERS))
bump_fl = load_car(write_car("bump_fl.car", over=bump_over({"fl": BUMP_Y["fl"]}))[0])
ok("a car with one non-zero bump steer table loads", np.any(bump_fl.susp.bump_steer_fl["y"]))
ok("suspension_curves accepts a live bump steer table (the refusal was removed in task 14)",
   suspension_curves(bump_fl) is not None)
curves = suspension_curves(load_car(SYN))
ok("the synthetic car with zero bump steer gives front and rear curves (physical hardpoints)",
   np.isfinite(curves.f.zg) and np.isfinite(curves.r.zg) and curves.f.lo < 0 < curves.f.hi)
t4_rec = tyre_record(1, "fl")[0]            # file cid 81 = the front left tyre
t4_dims, t4_sc, t4_tabs = t4_rec[2]
t4_tabs = list(t4_tabs)
t4_tabs[4] = np.concatenate([[-2.0, 2.0], [1000.0, 3000.0], [0.9, 0.8, 1.1, 1.2]])   # cambers deg, loads N, grid rows = camber
t4_car = load_car(write_car("tyre_t4.car", over={81: (t4_dims, t4_sc, t4_tabs)})[0])
ok("tyre table 4: camber axis read in rad (-2, 2 deg), load axis in N as stored",
   np.allclose(t4_car.tyre.fl["fx_camber_x"], np.deg2rad([-2.0, 2.0]), rtol=0, atol=1e-12)
   and np.allclose(t4_car.tyre.fl["fx_camber_load"], [1000.0, 3000.0], rtol=0, atol=0))
ok("tyre table 4: grid row = camber, column = load ([[0.9, 0.8], [1.1, 1.2]] as stored)",
   np.allclose(t4_car.tyre.fl["fx_camber"], [[0.9, 0.8], [1.1, 1.2]], rtol=0, atol=0))

# left takes right: a left record that is absent or all zero takes the record of its right sibling
print("  -- left takes right")
Z11 = (0, 0.0, 0.0, np.zeros(2), np.zeros(2))                 # an all-zero spring-like record
Z12 = (0, np.zeros(6), np.zeros(15))                          # an all-zero damper record
lr = load_car(write_car("left_empty.car", omit=(2, 26, 83, 170),
                        over={8: 0.0, 14: Z11, 32: Z12, 171: np.zeros(5)})[0])
ok("an absent left record takes the right one: spring 2 <- 1, motion ratio 26 <- 25, tyre 83 <- 82, bar ratio 170 <- 27",
   lr.susp.spring_fl["rate"] == car.susp.spring_fr["rate"] and lr.susp.spring_fl["preload"] == car.susp.spring_fr["preload"]
   and np.array_equal(lr.susp.mr_rl, car.susp.mr_rr) and lr.tyre.rl["radius"] == car.tyre.rr["radius"]
   and np.array_equal(lr.tyre.rl["fy_peak"], car.tyre.rr["fy_peak"]) and np.array_equal(lr.susp.mr_arb_fl, car.susp.mr_arb_fr))
ok("an all-zero left record takes the right one: rubber 14 <- 13, damper 32 <- 31, bar ratio 171 <- 28",
   lr.susp.rubber_fl["rate"] == car.susp.rubber_fr["rate"]
   and np.array_equal(lr.susp.damper_fl["bump"], car.susp.damper_fr["bump"])
   and np.array_equal(lr.susp.mr_arb_rl, car.susp.mr_arb_rr))
ok("a left rubber gap of 0 takes the right gap and its flag (8 <- 7, 205 <- 204)",
   lr.susp.gap_fl == car.susp.gap_fr and lr.susp.gap_flag_fl == GAP_FLAG[204] != GAP_FLAG[205])
ok("the records of the other corners stay as stored (spring rr and rl, gap rl, damper rl)",
   lr.susp.spring_rr["rate"] == SPRING[3][0] and lr.susp.spring_rl["rate"] == SPRING[4][0]
   and lr.susp.gap_rl == EXPLICIT[10] and lr.susp.gap_flag_rl == GAP_FLAG[207]
   and np.array_equal(lr.susp.damper_rl["bump"], BUMP * DAMP[34]))
rl_ = load_car(write_car("right_empty.car", omit=(1, 27), over={7: 0.0, 33: Z12})[0])
ok("an empty right record takes nothing from the left: spring 1 and bar ratio 27 None, gap 7 zero, damper 33 zero",
   rl_.susp.spring_fr is None and rl_.susp.mr_arb_fr is None and rl_.susp.gap_fr == 0.0
   and not np.any(rl_.susp.damper_rr["bump"]) and rl_.susp.spring_fl["rate"] == SPRING[2][0]
   and rl_.susp.gap_fl == EXPLICIT[8] and np.array_equal(rl_.susp.mr_arb_fl, arb(5, 170)))

# car_vp: strict, one totals set
print("  -- car_vp")
vp = car_vp(car)
ok("car_vp keys: mb, md, muf, mur, wB, l, hcg, I_x, I_y, I_z, t, Rw, Jw, f, kt",
   set(vp) == {"mb", "md", "muf", "mur", "wB", "l", "hcg", "I_x", "I_y", "I_z", "t", "Rw", "Jw", "f", "kt"})
ok("car_vp mb + md + muf + mur equals the car mass (700 kg), md 0",
   abs(vp["mb"] + vp["md"] + vp["muf"] + vp["mur"] - CS_TOTAL) < 1e-9 and vp["md"] == 0.0
   and vp["muf"] == CS_UF and vp["mur"] == CS_UR)
expect_wb = (CS_FRONT * CS_TOTAL - CS_UF) / (CS_TOTAL - CS_UF - CS_UR)
ok(f"car_vp wB = (front share x total - muf) / mb = {expect_wb:.6f}", abs(vp["wB"] - expect_wb) < 1e-12)
ok("car_vp wheelbase, CG height, inertias as stored",
   abs(vp["l"] - 3.1) < 1e-12 and abs(vp["hcg"] - 0.33) < 1e-12
   and abs(vp["I_x"] - 60.0) < 1e-12 and abs(vp["I_y"] - 280.0) < 1e-12 and abs(vp["I_z"] - 600.0) < 1e-12)
ok("car_vp t is the mean track (1.505 m)", abs(vp["t"] - (1.52 + 1.49) / 2) < 1e-12)
ok("car_vp Rw is the rear tyre radius (0.310 m)", abs(vp["Rw"] - TYRE["rl"]["radius"]) < 1e-12)
ok("car_vp Jw is CS_JW (the car file holds no wheel inertia)", vp["Jw"] == CS_JW)
ok("car_vp f is the rear tyre rolling drag (0.0135)", abs(vp["f"] - TYRE["rl"]["drag"]) < 1e-12)
ok("car_vp kt is the mean of the front and rear tyre kz",
   abs(vp["kt"] - (car.tyre.fl["kz"] + car.tyre.rl["kz"]) / 2) < 1e-6)
no_mt = write_car("no_mass_total.car", omit=(84,))[0]
ok("car_vp raises ValueError naming mass.total when the car lacks it",
   raises_naming(lambda: car_vp(load_car(no_mt)), "mass.total"))
no_wb = write_car("no_wheelbase.car", omit=(90,))[0]
ok("car_vp raises ValueError naming geom.wheelbase when the car lacks it",
   raises_naming(lambda: car_vp(load_car(no_wb)), "geom.wheelbase"))

print("  -- car_vp through vehParams")
for physics in PHYSICS:
    v = through_vehparams(vp, physics)
    ok(f"vehParams total mass equals the car mass ({physics}): m = {v.m:.9g} kg",
       abs(v.m - CS_TOTAL) < 1e-9 and abs(v.ms - (CS_TOTAL - CS_UF - CS_UR)) < 1e-9)
    share = (v.Wfl0 + v.Wfr0) / (v.g * v.m)
    ok(f"static front axle share through vehParams equals the car front share 0.44 ({physics}): {share:.9f}",
       abs(share - CS_FRONT) < 1e-9)
    ok(f"vehParams takes Jw, f and kt ({physics})",
       v.Jw == CS_JW and abs(v.f - TYRE["rl"]["drag"]) < 1e-12 and abs(v.kt - vp["kt"]) < 1e-6)


# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs):
    car_path = cs.parent / "Models" / "GP3 2013" / "Eastern Creek, AU" / "def_GP3 2013.car"
    print("[cs_car] vendor GP3 2013 Eastern Creek")
    ok("GP3 2013 Eastern Creek car file found", car_path.is_file())
    c = load_car(str(car_path))

    missing = [name for name, _ in CAR_FIELDS.values()
               if getattr(getattr(c, name.split(".")[0]), name.split(".")[1]) is None]
    ok(f"no consumed field of the GP3 car is None (missing: {missing})", not missing)

    def rel(a, b):
        return abs(a - b) / abs(b)
    ok("spring rates: fl and fr 1200 lbf/in, rl and rr 800 lbf/in, in N/m (1e-3 relative)",
       all(rel(getattr(c.susp, f"spring_{k}")["rate"], lbf * LBF_IN) < 1e-3
           for k, lbf in (("fl", 1200.0), ("fr", 1200.0), ("rl", 800.0), ("rr", 800.0))))
    fr_rc, rr_rc = c.geom.rc_height_fl, c.geom.rc_height_rl
    ok(f"roll-centre heights: front {fr_rc:.4f} m < rear {rr_rc:.4f} m, both in (0, 0.2)",
       0.0 < fr_rc < rr_rc < 0.2)
    ok(f"6 gears ({len(c.engine.gears)} read)", len(c.engine.gears) == 6)
    ok("aero maps are 7 x 5 (cla, cda, balance)",
       all(getattr(c.aero, m)["data"].shape == (7, 5) for m in ("cla_map", "cda_map", "balance_map")))
    ok("rubber ground-gap flags are all 1 (GP3)",
       all(getattr(c.susp, f"gap_flag_{k}") == 1 for k in CORNERS))
    ok("damper velocities are in m/s, below 1 m/s",
       all(float(np.max(np.abs(getattr(c.susp, f"damper_{k}")["velocity"]))) < 1.0 for k in CORNERS))

    vpc = car_vp(c)
    for physics in PHYSICS:
        v = through_vehparams(vpc, physics)
        ok(f"car_vp through vehParams reproduces the car mass ({physics}): m = {v.m:.6g} kg",
           abs(v.m - c.mass.total) <= 1e-9 * c.mass.total)
        share = (v.Wfl0 + v.Wfr0) / (v.g * v.m)
        ok(f"static front axle share through vehParams equals the car front share ({physics}): "
           f"{share:.6f} vs {c.mass.front_share:.6f}",
           abs(share - c.mass.front_share) < 1e-6)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(Path(CS))
print(f"{PASS} checks passed")
