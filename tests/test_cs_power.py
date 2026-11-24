"""ChassisSim car powertrain refusals, the wake gate, the atmospheric torque factor (ChassisSim phase 2c, P1).

A synthetic car file (arbitrary values, written with carfile.write; the cids and record tags are literal numbers
here) and a synthetic ChassisSim folder layout go through load_car, power_refusals, aero_refusals, userOpts,
vehModel and MLTP (fake IPOPT, no solve). Checks:
  1. each refusal cid that is on raises ValueError naming the cid; off, absent and NUL text load. The brake ramp
     (571 on) needs one valid ramp time (572 above 0), else it raises naming cid 572,
  2. the wake gate: maps 872 to 875 live or flags 876 to 879 not 0 raise; cid 870 alone loads,
  3. the atmospheric factor of Atmos_Cond.ini: the hand value, the file name rule, the defaults, the errors,
  4. the knob (setup tag, data['car_atmos']), the NLP graph (factor 1.0 changes nothing) and the brake ramp
     bound (571 = 1 with 572 = 0.2: the lower rate bound of the brake on N rows is -1 / 0.2),
  5. the shipped car files (SKIP without CHASSISSIM_DIR): all 31 load and pass the checks, the factor is 1.0 for each.
Nothing is written into the vendor folders.

    python tests/test_cs_power.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_power.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import hashlib
import io
import itertools
import math
import os
import tempfile
import warnings
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.io import loadmat

import functions.transcription as T
from functions import carfile, chassissim as C
from functions.context import Ctx
from MLTP import MLTP
from MLTP_screen import MLTP_screen
from userOpts import knob_tag, setup_knobs, userOpts
from vehModel import vehModel

PASS = 0
TMP = tempfile.TemporaryDirectory()
COUNT = itertools.count()
CORNERS = ("fl", "fr", "rl", "rr")
CIRC = "Synth Ring"
INI_ON = ("[Preferences]\nEng Corr=1\n[Current Conditions]\nAtm Press=0.95\nWater Press=0.03\nTemperature=35\n"
          "[Engine Conditions]\nAtm Press=1.013\nWater Press=0\nTemperature=15\n")
# Hand value of INI_ON: sqrt(T_eng / T_cur) x (P_cur - Pw_cur) / (P_eng - Pw_eng), temperatures in K, pressures in bar.
HAND = math.sqrt((15 + 273.15) / (35 + 273.15)) * (0.95 - 0.03) / (1.013 - 0.0)
HOT = INI_ON.replace("Temperature=35", "Temperature=80")   # a second value, for the decoy and the mangled-name check


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


def rel(a, b):
    return abs(a - b) / abs(b)


CAP = {}     # the last call of the fake solver: its keyword arguments (lbg, ubg, ...)


def fake(ca_, nlp, opts, info=None):
    """Stand-in for functions.transcription._make_solver: returns the start point, no iterations."""
    def solver(**k):
        CAP.update(k=dict(k))
        return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()), "lam_x": np.zeros(nlp["x"].numel())}
    solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
    if info is not None:
        info["linear_solver"] = "fake"
    return solver


T._make_solver = fake


def dget(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else getattr(d, k, default)


# ---- synthetic car: literal cids and tags (as in tests/test_vehmodel_cs.py) ---------------------------
# Tags: 1 scalar, 2 array, 6 byte, 7 curve, 8 text, 9 map, 10 hardpoints, 11 spring-like, 12 damper, 13 tyre.
TYRE = {  # dims (d1..d7), radius m, toe deg, camber deg, castor deg, drag, kz N/m
    "fl": dict(dims=(3, 2, 3, 2, 2, 5, 4), radius=0.300, toe=0.0, camber=-1.0, castor=2.0, drag=0.011, kz=1.60e5),
    "fr": dict(dims=(2, 2, 3, 2, 2, 5, 4), radius=0.305, toe=0.0, camber=-1.5, castor=2.5, drag=0.012, kz=1.62e5),
    "rl": dict(dims=(4, 2, 3, 3, 2, 5, 4), radius=0.310, toe=0.0, camber=-0.5, castor=1.5, drag=0.0135, kz=1.40e5),
    "rr": dict(dims=(2, 2, 3, 2, 3, 5, 4), radius=0.315, toe=0.0, camber=-0.75, castor=1.0, drag=0.014, kz=1.42e5),
}
TYRE_TEMPS = (300.0, 350.0)
TYRE_BG = 325.0
PRESSURE = 2.1e5
VEL = np.array([0.0, 100.0, 200.0, 400.0, 800.0])        # mm/s, from 0 up
BUMP = np.array([0.0, 900.0, 1800.0, 2700.0, 3600.0])    # N, zero at zero velocity
REB = np.array([0.0, 600.0, 1200.0, 1800.0, 2400.0])     # N, positive as the file stores it
SPRING = {1: (1.23e5, 0.020), 2: (1.31e5, 0.021), 3: (0.97e5, 0.019), 4: (1.02e5, 0.0205),
          5: (0.0, 0.0), 6: (0.0, 0.0)}
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}
ARB = {19: 2.0e4, 21: 1.5e4}
HP_LEFT = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.68, 0.11],
                    [-0.15, 0.30, 0.28], [0.15, 0.30, 0.28], [0.0, 0.64, 0.33]])
HP_RIGHT = HP_LEFT * np.array([1.0, -1.0, 1.0])
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: 1.52, 92: 1.49, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,
    100: 0.9, 102: 1000.0, 103: 9000.0,
    107: 14000.0, 108: 0.6, 109: 0.05,
    111: 0.0, 112: 1.0, 113: 1.0,
    55: 0.03, 59: 0.03, 63: 0.03, 67: 0.03,
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,
    390: 350.0,
    7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.0, 12: 0.0,   # rubber ground gaps, m
}
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})        # DRS
EXPLICIT.update({142: 0.5 * (EXPLICIT[117] + EXPLICIT[118]) * 1e-3,
                 144: 0.5 * (EXPLICIT[119] + EXPLICIT[120]) * 1e-3, 56: 0.0, 60: 0.0, 64: 0.0, 68: 0.0})
EXPLICIT.update({c: 0.0 for c in (55, 59, 63, 67) if c not in EXPLICIT})
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
AERO_X = 0.005 * np.arange(7)
AERO_Z = 0.002 + 0.005 * np.arange(7)
MAPS = {114: ((4, 4), "diff_f"), 116: ((4, 4), "diff_r")}
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}


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
    d1, d2, d3, _, d5, d6, d7 = dims
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(dims))]
    A, B = arb(d1, 20 + i), arb(d1, 30 + i)
    load = np.linspace(1000.0, 25000.0, d1)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, np.stack([A, B], axis=1).ravel()])
    tabs[0][:d3] = np.linspace(0.0, 2.0, d3)
    tabs[0][d3:] = np.array([0.0, 0.97, 0.97])[:d3]
    off = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    tabs[5][:d6] = t["camber"] + off
    tabs[5][d6:] = 1.0 - 0.4 * (off / 4.0) ** 2
    tabs[4] = np.concatenate([np.linspace(-3.0, 3.0, d6), 1000.0 * np.arange(1, d7 + 1), np.ones(d6 * d7)])
    px = np.linspace(150.0, 250.0, d5)
    py = t["kz"] * (1.0 + 0.01 * np.arange(d5))
    tabs[10] = np.concatenate([px, py])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = t["toe"], t["camber"], t["castor"], t["radius"], PRESSURE, TYRE_BG
    sc[7], sc[8], sc[9], sc[11], sc[15] = t["drag"], 0.11 + 0.01 * i, 0.12 + 0.01 * i, 0.7 + 0.1 * i, 1.05 + 0.1 * i
    return 13, 80 + i, (list(dims), sc, tabs)


def build_records():
    """The synthetic car without the end record. Each cid goes through FILE_CID (left and right swapped)."""
    R = []

    def add(tag, cid, v):
        R.append((tag, FILE_CID.get(cid, cid), v))
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
    for cid, ((nx, nz), _) in MAPS.items():
        add(9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid)))
    for cid in range(31, 35):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    add(12, 35, (1, arb(6, 35), np.array([])))
    for cid, rows in ((49, HP_LEFT), (50, HP_RIGHT), (51, HP_LEFT), (52, HP_RIGHT)):
        add(10, cid, (0, np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])))
    for i, c in enumerate(CORNERS):
        add(*tyre_record(i, c))
    return R


def write_car(extra=()):
    """Write the synthetic car with the extra (tag, cid, value) records before the end record. Return its path."""
    R = build_records() + list(extra) + [(carfile.END_TAG, carfile.END_CID, None)]
    path = os.path.join(TMP.name, f"car_{next(COUNT)}.car")
    carfile.write(R, path)
    return path


def write_text(path, text):
    data = text if isinstance(text, bytes) else text.encode("latin-1")
    with open(path, "wb") as f:
        f.write(data)


def build_install(root):
    """A ChassisSim install folder with one 600 m circuit (no Atmos file)."""
    s = np.linspace(0.0, 600.0, 121)
    k = (2.0 * np.pi / 600.0) * (1.0 + 0.5 * np.sin(2.0 * np.pi * s / 600.0))
    ini = ("[Total Circ prop]\r\nTotal circuits = 1\r\n\r\n"
           f"[Circuit0]\r\nCircuit name = {CIRC}\r\nCircuit file name = tr_synth.txt\r\nBase Grip factor = 1.0\r\n"
           "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n")
    os.makedirs(os.path.join(root, "Circuits"))
    write_text(os.path.join(root, "Circ_prop.ini"), ini)
    rows = [f"{float(a)!r}   {float(b)!r}" for a, b in zip(k, s)]
    write_text(os.path.join(root, "Circuits", "tr_synth.txt"), "\r\n".join(rows) + "\r\n")
    return root


def layout(ini=None, decoy=None):
    """A ChassisSim layout in a new folder: the install folder ChassisSim_v3_41, its parent holds the file
    ChassisSim_v3_41Atmos_Cond.ini (the name ChassisSim reads), and the install folder holds the decoy file.
    Return the install folder."""
    root = os.path.join(TMP.name, f"layout_{next(COUNT)}")
    inst = build_install(os.path.join(root, "ChassisSim_v3_41"))
    if ini is not None:
        write_text(os.path.join(root, "ChassisSim_v3_41Atmos_Cond.ini"), ini)
    if decoy is not None:
        write_text(os.path.join(inst, "Atmos_Cond.ini"), decoy)
    return inst


def power_msg(car_path, inst):
    """'silent' or the ValueError message of load_car + power_refusals + aero_refusals."""
    try:
        cs = C.load_car(car_path, inst)
        C.power_refusals(cs)
        C.aero_refusals(cs)
        return "silent"
    except ValueError as e:
        return str(e)


def raises_cid(msg, cid):
    return f"(cid {cid})" in msg


INST = build_install(os.path.join(TMP.name, "ChassisSim_v3_41"))   # a plain install for the car refusals

print("[cs_power] refusals by cid")
# flags: on (1 and 2) raise naming the cid, off (0) loads
for cid in (598, 649, 520, 419, 423, 426, 428):
    for v in (1, 2):
        msg = power_msg(write_car([(6, cid, v)]), INST)
        ok(f"cid {cid} = {v} raises ValueError naming the cid", msg != "silent" and raises_cid(msg, cid))
    ok(f"cid {cid} = 0 loads", power_msg(write_car([(6, cid, 0)]), INST) == "silent")
ok("the flags absent load", power_msg(write_car(), INST) == "silent")

print("[cs_power] brake ramp: 571 needs a valid 572")
BAD_572 = (("absent", []), ("NaN", [(1, 572, float("nan"))]), ("0", [(1, 572, 0.0)]), ("-1", [(1, 572, -1.0)]),
           ("inf", [(1, 572, float("inf"))]), ("two values", [(2, 572, np.array([0.1, 0.2]))]))
for v in (1, 2):
    for name, extra in BAD_572:
        msg = power_msg(write_car([(6, 571, v)] + extra), INST)
        ok(f"ramp 571 = {v} with 572 {name} raises ValueError naming cid 572 (no Python None)",
           msg != "silent" and "cid 572" in msg and "None" not in msg)
ok("ramp 571 = 1 with 572 = 0.1 loads", power_msg(write_car([(6, 571, 1), (1, 572, 0.1)]), INST) == "silent")
for name, extra in (("absent", []), ("NaN", [(1, 572, float("nan"))]), ("-1", [(1, 572, -1.0)])):
    ok(f"ramp 571 = 0 with 572 {name} loads", power_msg(write_car([(6, 571, 0)] + extra), INST) == "silent")
    ok(f"ramp 571 absent with 572 {name} loads", power_msg(write_car(extra), INST) == "silent")
HYB_VALID = [(6, 177, 1), (6, 535, 1), (6, 590, 1), (1, 179, 60000.0), (1, 180, 60000.0), (1, 181, 400000.0),
             (1, 182, 1.0), (1, 183, 1.0), (1, 836, 0.0)]
ok("a car with 177 = 1 and valid hybrid values loads (cids 177, 535, 573 are no refusal)",
   power_msg(write_car(HYB_VALID), INST) == "silent")

print("[cs_power] brake split, tyre-limit split and paddle gates")
ok("brake left share 160 = 0.6 raises naming cid 160",
   raises_cid(power_msg(write_car([(1, 160, 0.6)]), INST), 160))
ok("brake left share 160 = 0.5 loads", power_msg(write_car([(1, 160, 0.5)]), INST) == "silent")
ok("cid 853 = 1 (brake split by the tyre limit) raises naming cid 853",
   raises_cid(power_msg(write_car([(6, 853, 1)]), INST), 853))
ok("cid 853 = 0 loads", power_msg(write_car([(6, 853, 0)]), INST) == "silent")
no177 = power_msg(write_car([(6, 573, 1), (6, 589, 1)]), INST)
ok("paddle 573 = 589 = 1 with 177 off raises naming cid 573 and cid 177",
   "573" in no177 and raises_cid(no177, 177))
PADDLE = [r for r in HYB_VALID if r[1] != 182] + [(6, 573, 1), (6, 589, 1)]
ok("paddle on with regen scale 182 = 0.8 raises naming cid 182",
   raises_cid(power_msg(write_car(PADDLE + [(1, 182, 0.8)]), INST), 182))
ok("paddle on with regen scale 182 = 1 loads", power_msg(write_car(PADDLE + [(1, 182, 1.0)]), INST) == "silent")

print("[cs_power] text and engine-brake table")
for cid in (897, 906):
    ok(f"cid {cid} with a file name raises naming the cid",
       raises_cid(power_msg(write_car([(8, cid, b"pack.dll\x00")]), INST), cid))
    for name, v in (("NUL only", b"\x00"), ("blanks, CRLF and NUL", b"  \r\n\x00"), ("a tab", b"\t"), ("empty", b"")):
        ok(f"cid {cid} with {name} loads", power_msg(write_car([(8, cid, v)]), INST) == "silent")
ok("cid 161 with a live y raises naming the cid",
   raises_cid(power_msg(write_car([(7, 161, (ENGINE_RPM, np.array([0.0, 0.0, 5.0, 0.0])))]), INST), 161))
ok("cid 161 with an axis only (y zero) loads",
   power_msg(write_car([(7, 161, (ENGINE_RPM * 1.1, np.zeros(4)))]), INST) == "silent")
ok("cid 161 absent loads", power_msg(write_car(), INST) == "silent")

print("[cs_power] wake gate and cid 870")
inst_msg = power_msg(write_car([(9, 872, (AERO_X, AERO_X, np.ones(49)))]), INST)
ok("wake map 872 live raises (aero_refusals)", "872" in inst_msg and "wake map" in inst_msg)
ok("wake map 875 live raises (aero_refusals)",
   "875" in power_msg(write_car([(9, 875, (AERO_X, AERO_X, np.ones(49)))]), INST))
ok("wake maps 872 to 875 all zero load",
   power_msg(write_car([(9, c, (AERO_X, AERO_X, np.zeros(49))) for c in range(872, 876)]), INST) == "silent")
for cid in (876, 877, 878, 879):
    ok(f"wake flag {cid} = 1 raises naming the cid", raises_cid(power_msg(write_car([(6, cid, 1)]), INST), cid))
    ok(f"wake flag {cid} = 0 loads", power_msg(write_car([(6, cid, 0)]), INST) == "silent")
cs870 = C.load_car(write_car([(6, 870, 1)]), INST)
ok("cid 870 = 1 alone loads (the regen front enable, not a refusal)",
   power_msg(cs870.path, INST) == "silent" and cs870.hybrid.regen_front_on == 1)

print("[cs_power] atmospheric factor")
ok("default: only the current temperature key gives sqrt(288.15 / 308.15)",
   rel(C._atmos(layout(ini="[Preferences]\nEng Corr=1\n[Current Conditions]\nTemperature=35\n")),
       0.9670039006596417) < 1e-15)
ok("the file at the mangled path gives the hand factor (rel 1e-15)", rel(C._atmos(layout(ini=INI_ON)), HAND) < 1e-15)
ok("a decoy Atmos_Cond.ini in the install folder is ignored (decoy only: 1.0)",
   C._atmos(layout(decoy=HOT)) == 1.0)
ok("with the decoy and the mangled file, the mangled file wins",
   rel(C._atmos(layout(ini=INI_ON, decoy=HOT)), HAND) < 1e-15)
ok("a trailing separator on the folder gives the same factor",
   rel(C._atmos(layout(ini=INI_ON) + os.sep), HAND) < 1e-15)
ok("CRLF, a BOM, spaces, a comment and an inline comment parse to the hand factor",
   rel(C._atmos(layout(ini=b"\xef\xbb\xbf" + "; note\r\n[Preferences]\r\nEng Corr = 1 ; on\r\n"
                       "[Current Conditions]\r\nAtm Press = 0.95\r\nWater Press = 0.03 # bar\r\nTemperature = 35\r\n"
                       "[Engine Conditions]\r\nAtm Press = 1.013\r\nWater Press = 0\r\nTemperature = 15\r\n".encode())),
       HAND) < 1e-15)
ok("Eng Corr 0 gives 1.0", C._atmos(layout(ini=INI_ON.replace("Eng Corr=1", "Eng Corr=0"))) == 1.0)
ok("Eng Corr 2 gives the same factor as Eng Corr 1 (on)",
   rel(C._atmos(layout(ini=INI_ON.replace("Eng Corr=1", "Eng Corr=2"))), HAND) < 1e-15)
if not os.environ.get("CHASSISSIM_DIR"):
    ok("no folder argument and no CHASSISSIM_DIR gives 1.0", C._atmos(None) == 1.0)
for name, text in (("a malformed temperature", INI_ON.replace("Temperature=35", "Temperature=abc")),
                   ("a nan temperature", INI_ON.replace("Temperature=35", "Temperature=nan")),
                   ("no section header", "Eng Corr=1\n")):
    try:
        C._atmos(layout(ini=text))
        ok(f"{name} raises ValueError naming the file", False)
    except ValueError as e:
        ok(f"{name} raises ValueError naming the file", "Atmos_Cond.ini" in str(e))
missing = os.path.join(TMP.name, "no_such_install")
try:
    C._atmos(missing)
    ok("a missing folder raises ValueError naming the folder", False)
except ValueError as e:
    ok("a missing folder raises ValueError naming the folder", "no_such_install" in str(e))

print("[cs_power] knob, setup tag and NLP graph")
INST_OFF = layout(ini=INI_ON.replace("Eng Corr=1", "Eng Corr=0"))
INST_ON = layout(ini=INI_ON)
INST_UNITY = layout(ini=INI_ON.replace("Atm Press=0.95", "Atm Press=1.013").replace("Water Press=0.03", "Water Press=0")
                    .replace("Temperature=35", "Temperature=15"))      # equal pressures and temperatures: 1.0
CAR = write_car()


def graph_hash(inst):
    """Hash of (dx, sf) of the 23-state model of the car, and the atmospheric factor it reads."""
    ctx = Ctx()
    quiet(userOpts, ctx, circuit=f"chassissim:{CIRC}", car=f"chassissim:{CAR}", chassissim_dir=inst,
          physics="corrected", linear_solver="mumps")
    quiet(vehModel, ctx)
    m = ctx.m23
    fn = ca.Function("d", [m.x, m.u, m.pv], [m.dx, m.sf])
    return hashlib.sha256(fn.serialize().encode()).hexdigest(), ctx.cs.engine.atmos


h_plain, a_plain = graph_hash(INST)
h_off, a_off = graph_hash(INST_OFF)
h_unity, a_unity = graph_hash(INST_UNITY)
h_on, a_on = graph_hash(INST_ON)
ok("factor 1.0 without the ini (plain) and with Eng Corr 0", a_plain == 1.0 and a_off == 1.0)
ok("unity factor (equal conditions) gives exactly 1.0", a_unity == 1.0)
ok("the plain car graph equals the Eng Corr 0 graph", h_off == h_plain)
ok("the unity graph equals the plain graph", h_unity == h_plain)
ok("Eng Corr 1 changes the graph (the factor reaches the NLP)", h_on != h_plain and rel(a_on, HAND) < 1e-15)


def fields(inst):
    return dict(car=f"chassissim:{CAR}", chassissim_dir=inst)


k_plain, k_off, k_on = setup_knobs(fields(INST)), setup_knobs(fields(INST_OFF)), setup_knobs(fields(INST_ON))
ok("plain and Eng Corr 0 cars have no car_atmos knob", "car_atmos" not in k_plain and "car_atmos" not in k_off)
ok("the Eng Corr car has the car_atmos knob at the hand factor",
   k_on.get("car_atmos") is not None and rel(k_on["car_atmos"], HAND) < 1e-9)
ok("the Eng Corr car has a setup tag different from the plain car",
   knob_tag(k_on) != knob_tag(k_plain) and knob_tag(k_off) == knob_tag(k_plain))

res = os.path.join(TMP.name, "results")
os.makedirs(res)
opts = dict(circuit=f"chassissim:{CIRC}", car=f"chassissim:{CAR}", physics="corrected", save=True, plot=False,
            linear_solver="mumps", results_dir=res)
c_plain = quiet(MLTP, chassissim_dir=INST, **opts)
c_on = quiet(MLTP, chassissim_dir=INST_ON, **opts)
ok("the plain MLTP run has no data['car_atmos']", dget(c_plain.data, "car_atmos") is None)
ok("the Eng Corr MLTP run has data['car_atmos'] at the hand factor",
   dget(c_on.data, "car_atmos") is not None and rel(dget(c_on.data, "car_atmos"), HAND) < 1e-9)
ok("the two MLTP runs save under different names", c_plain.out_path != c_on.out_path
   and knob_tag(k_on) in os.path.basename(c_on.out_path))
mat_on = loadmat(c_on.out_path, simplify_cells=True)["data"]
mat_plain = loadmat(c_plain.out_path, simplify_cells=True)["data"]
ok("the saved .mat holds car_atmos (Eng Corr) and not the plain run",
   "car_atmos" in mat_on and rel(float(mat_on["car_atmos"]), HAND) < 1e-9 and "car_atmos" not in mat_plain)
q_on = quiet(MLTP_screen, circuit=opts["circuit"], car=opts["car"], physics="corrected", save=True, verbose=False,
             results_dir=res, chassissim_dir=INST_ON)
ok("the QSS screen of the Eng Corr car also stores data['car_atmos']",
   dget(q_on.data, "car_atmos") is not None and rel(dget(q_on.data, "car_atmos"), HAND) < 1e-9)

# (g) continue_from='auto' (B5): the plain default of the car seeds the Eng Corr run, which carries the car_atmos knob.
# The default file holds no car_atmos knob, so default_source accepts it. Only the car_file tag is in its name.
def solved(ca_, nlp, opts, info=None):
    """The fake solver with the status of a converged solve: continue_from takes only such a default."""
    sv = fake(ca_, nlp, opts, info)
    sv.stats = lambda: {"iter_count": 0, "return_status": "Solve_Succeeded"}
    return sv


res_g = os.path.join(TMP.name, "results_g")
os.makedirs(res_g)
g_opts = dict(opts, results_dir=res_g)
T._make_solver = solved
try:
    quiet(MLTP, chassissim_dir=INST, **g_opts)                 # the plain default, status Solve_Succeeded
    c_cont = quiet(MLTP, chassissim_dir=INST_ON, continue_from="auto", **g_opts)
finally:
    T._make_solver = fake
ok(f"(g) continue_from auto with car_atmos: the plain default seeds full+duals (got {c_cont.elapsed['warm_start']})",
   c_cont.elapsed["warm_start"] == "full+duals")
ok("(g) the source is the plain default file, the car_atmos run is saved under its own tag",
   c_cont.data.provenance["run"]["source"] == os.path.basename(c_plain.out_path)
   and knob_tag(k_on) in os.path.basename(c_cont.out_path))

print("[cs_power] brake ramp in the NLP")


def ramp_build(extra):
    """MLTP on the synthetic car with the extra records, fake IPOPT. Return (ctx, lbg, ubg, graph hash, N)."""
    res = quiet(MLTP, chassissim_dir=INST, circuit=f"chassissim:{CIRC}", car=f"chassissim:{write_car(extra)}",
                physics="corrected", save=False, plot=False, linear_solver="mumps")
    mm = res.m23
    h = hashlib.sha256(ca.Function("d", [mm.x, mm.u, mm.pv], [mm.dx, mm.sf]).serialize().encode()).hexdigest()
    k = CAP["k"]
    return res, np.ravel(np.asarray(k["lbg"], float)), np.ravel(np.asarray(k["ubg"], float)), h, int(dget(res.data, "N"))


R0, lbg0, ubg0, h0, N = ramp_build([])                                     # no 571 and no 572: the reference
R1, lbg1, ubg1, h1, _ = ramp_build([(6, 571, 1), (1, 572, 0.2)])
R2, lbg2, _, _, _ = ramp_build([(6, 571, 1), (1, 572, 0.01)])
R3, lbg3, ubg3, h3, _ = ramp_build([(6, 571, 0), (1, 572, 0.2)])
nu = R1.m23.nu
lb0 = np.ravel(R0.m23.duk_lb)
lb1 = np.ravel(R1.m23.duk_lb)
ok(f"571 = 1 with 572 = 0.2: the brake rate lower bound is -1 / 0.2 = -5.0 (got {lb1[1]:.6g})",
   lb1[1] == -1.0 / 0.2)
ok(f"the brake rate upper bound does not change ({np.ravel(R1.m23.duk_ub)[1]:.6g})",
   np.array_equal(np.ravel(R1.m23.duk_ub), np.ravel(R0.m23.duk_ub)))
ok("the other input rate bounds do not change", np.array_equal(np.delete(lb1, 1), np.delete(lb0, 1)))
diff = np.flatnonzero(lbg1 != lbg0)
ok(f"exactly N = {N} lower bound rows change, one per interval, stride nu = {nu}",
   diff.size == N and np.all(np.diff(diff) == nu) and np.all(lbg1[diff] == -1.0 / 0.2)
   and np.all(lbg0[diff] == lb0[1]))
ok("no upper bound row changes (0 ubg rows)", np.array_equal(ubg1, ubg0))
ok("the ramp changes the bounds only: the graph (dx, sf) equals the car without the cids", h1 == h0)
ok("572 = 0.01 keeps the 571 = 0 bound (a looser ramp does not loosen it): all lower bounds equal",
   np.array_equal(lbg2, lbg0) and lb0[1] == np.ravel(R2.m23.duk_lb)[1])
ok("571 = 0 with 572 = 0.2 gives the graph of the car without the cids", h3 == h0)
ok("571 = 0 with 572 = 0.2 gives the NLP bounds of the car without the cids",
   np.array_equal(lbg3, lbg0) and np.array_equal(ubg3, ubg0) and np.array_equal(np.ravel(R3.m23.duk_lb), lb0))


# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs_dir):
    """Every shipped car file: load_car, check_car, aero_refusals and power_refusals. Counts only."""
    print("[cs_power] vendor car files")
    root = Path(cs_dir).parent.parent          # the ChassisSimTechnologies tree, one level above the install folder
    cars = sorted(root.rglob("*.car"))
    ok("the shipped car files are 31", len(cars) == 31)
    loaded, refused, factors = 0, 0, []
    for p in cars:
        try:
            cs = C.load_car(str(p), cs_dir)
        except ValueError:
            refused += 1
            continue
        factors.append(cs.engine.atmos)
        try:
            C.check_car(cs)
            C.aero_refusals(cs)
            C.power_refusals(cs)
            loaded += 1
        except ValueError:
            refused += 1
    # Absent gap flags read as 0 (R3), so every shipped car loads and passes the checks.
    ok(f"all 31 shipped car files load and pass the checks ({loaded} load, {refused} refuse)",
       loaded == 31 and refused == 0)
    ok("the atmospheric factor is 1.0 for every car that loads", bool(factors) and all(f == 1.0 for f in factors))


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
print(f"{PASS} checks passed")
