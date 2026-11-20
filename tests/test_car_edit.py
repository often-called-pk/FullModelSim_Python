"""Car setup editing of the ChassisSim car (phase 5, lane P5-B): app/car_edit.py and the car mode of the Setup tab.

Sections: (a) CarEdit on a synthetic car: the D9 gate, no-edit bytes, edits by record, two edits of one record,
an int field. (b) The left copy: an absent left record, a right-only edit, a left edit, the gap flag, the left gap 0
refusal. (c) validate: the range, the hybrid refusal, no temp file. (d) materialize and save: reuse, refusals, the
ChassisSim folder guard. (e) Labels and ranges. (f) The window (offscreen Qt): car mode, highlight, badge, reset,
car change, preset load, Run with edits, Save, back on the 4EM car, the locale of ScientificField. Vendor section
(SKIP without CHASSISSIM_DIR): every shipped car passes the D9 gate with its own bytes and no range error; the GP3
car edited keeps both values in the Run file; a GUI-built qss run with the FR spring at +20 % gives the lap and
leaves the source file unchanged.

    python tests/test_car_edit.py
    CHASSISSIM_DIR="<ChassisSim install folder>" python tests/test_car_edit.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import atexit
import contextlib
import copy
import glob
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np

from functions import carfile, chassissim

# A temp home for everything the GUI reads or writes. Set it before Qt starts (see app.paths).
TMP = types.SimpleNamespace(name=tempfile.mkdtemp(prefix="car_edit_"))
atexit.register(shutil.rmtree, TMP.name, True)
HOME = os.path.join(TMP.name, "home")
os.makedirs(os.path.join(HOME, ".config"))
os.environ["HOME"] = HOME
os.environ["XDG_CONFIG_HOME"] = os.path.join(HOME, ".config")
os.environ["FULLMODELSIM_DOCUMENTS"] = os.path.join(HOME, "Documents")   # Windows: Qt does not read HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
VENDOR = os.environ.pop("CHASSISSIM_DIR", None)      # the non-vendor sections run with no environment folder
VENDOR = VENDOR if VENDOR and os.path.isdir(VENDOR) else None
PRESETS = os.path.join(TMP.name, "presets")          # the user presets folder. The cars go to presets/cars
CARS = os.path.join(PRESETS, "cars")

from app import car_edit                              # noqa: E402  (after the environment is set)
from app.car_edit import CarEdit, FIELDS, in_chassissim  # noqa: E402

FAILS = []
COUNT = [0]


def ok(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {info}" if info and not cond else ""))
    if cond:
        COUNT[0] += 1
    else:
        FAILS.append(name)


def sha(path):
    return hashlib.sha256(blob(path)).hexdigest()


def error_text(fn):
    """The ValueError text of fn, or None when fn runs."""
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    return None


def blob(path):
    with open(path, "rb") as fh:
        return fh.read()


def tree(root):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs)


def model_of(ce):
    """load_car of the run file of ce (no edits: the source file)."""
    return chassissim.load_car(ce.materialize(os.path.join(TMP.name, "runs")))


# ---- synthetic car builder, copied from tests/test_cs_wiring.py (lines 45-51, 54, 93-114, 116-264) ----
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair. An over or omit cid is a file cid.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}
G = 9.81
CIRC = "Synth Ring"
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
    "fl": dict(dims=(3, 2, 3, 2, 2, 4, 4), radius=0.300, toe=0.0, camber=-1.0, castor=2.0, drag=0.011, kz=1.60e5),
    "fr": dict(dims=(2, 2, 3, 2, 2, 4, 4), radius=0.305, toe=0.0, camber=-1.5, castor=2.5, drag=0.012, kz=1.62e5),
    "rl": dict(dims=(4, 2, 3, 3, 2, 4, 4), radius=0.310, toe=0.0, camber=-0.5, castor=1.5, drag=0.0135, kz=1.40e5),
    "rr": dict(dims=(2, 2, 3, 2, 3, 4, 4), radius=0.315, toe=0.0, camber=-0.75, castor=1.0, drag=0.014, kz=1.42e5),
}
TYRE_TEMPS = (300.0, 350.0)
TYRE_BG = 325.0
PRESSURE = 2.1e5
VEL = np.array([0.0, 100.0, 200.0, 400.0, 800.0])        # mm/s, from 0 up (the damper curve starts at the origin)
BUMP = np.array([0.0, 900.0, 1800.0, 2700.0, 3600.0])    # N, zero at zero velocity
REB = np.array([0.0, 600.0, 1200.0, 1800.0, 2400.0])     # N, positive as the file stores it
SPRING = {1: (1.23e5, 0.020), 2: (1.31e5, 0.021), 3: (0.97e5, 0.019), 4: (1.02e5, 0.0205),
          5: (0.0, 0.0), 6: (0.0, 0.0)}
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}
ARB = {19: 2.0e4, 21: 1.5e4}
HARD = (49, 50, 51, 52)                  # front left, front right, rear left, rear right
# Left-side hardpoints (x rearward, y left +, z up, m), rows: lower inner front, lower inner aft, lower outer,
# upper inner front, upper inner aft, upper outer. The right side is y mirrored. Front and rear are the same.
HP_LEFT = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.68, 0.11],
                    [-0.15, 0.30, 0.28], [0.15, 0.30, 0.28], [0.0, 0.64, 0.33]])
HP_RIGHT = HP_LEFT * np.array([1.0, -1.0, 1.0])
BUMP_STEER_X = np.linspace(-0.02, 0.02, 11)   # m of wheel travel. The toe change is zero (cids 74-77)
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: 1.52, 92: 1.49, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,            # ride heights, mm
    100: 0.9, 102: 1000.0, 103: 9000.0,                     # efficiency, idle and limit rpm
    107: 14000.0, 108: 0.6, 109: 0.05,                      # brake force N, front bias, delay s
    111: 0.0, 112: 1.0, 113: 1.0,                           # torque split (0 = rear drive), diff types (1 = open)
    55: 0.03, 59: 0.03, 63: 0.03, 67: 0.03,                # roll-centre heights per corner, m
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,                # CLA, CDA, offset, rho
    390: 350.0,
}
GAP = {7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.0, 12: 0.0}   # rubber ground gaps, m. Third gaps 0 (no third element)
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
    d1, d2, d3, _, d5, d6, d7 = dims
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(dims))]
    A, B = arb(d1, 20 + i), arb(d1, 30 + i)
    load = np.linspace(1000.0, 25000.0, d1)              # N, up to 25 kN (the load floor is tested at 20 kN)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, np.stack([A, B], axis=1).ravel()])
    tabs[0][:d3] = np.linspace(0.0, 2.0, d3)                 # slip abscissae: from 0, increasing
    tabs[0][d3:] = np.array([0.0, 0.97, 0.97])[:d3]          # slip values: from 0, flat after the knee at 1
    tabs[5][:d6] = np.linspace(-3.0, 3.0, d6)                # camber abscissae (deg), increasing
    tabs[4] = np.concatenate([np.linspace(-3.0, 3.0, d6), np.linspace(0.0, 25000.0, d7), np.ones(d6 * d7)])  # flat 1.0 grid
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
    for right, left in MIRROR:      # file cids: the right wheel is the lower one
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
        cid = FILE_CID.get(cid, cid)
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
    for cid in range(204, 210):
        add(6, cid, 1)                                   # rubber gap flags (fl..rr, third f/r): 1 = ground gap
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    for cid in (127, 131, 135):
        add(9, cid, (AERO_X, AERO_Z, np.zeros(49) if mirror else arb(49, cid)))   # the mirror car has no aero: no lift at the static point
    for cid, ((nx, nz), _) in MAPS.items():
        add(9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid)))
    for cid in range(31, 36):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    for cid, hp in zip(HARD, (HP_LEFT, HP_RIGHT, HP_LEFT, HP_RIGHT)):
        add(10, cid, (0, hp.T.ravel()))                  # column-major (6, 3): the 3 coordinate rows of 6 points
    for cid in range(74, 78):
        add(7, cid, (BUMP_STEER_X, np.zeros(BUMP_STEER_X.size)))
    for i, c in enumerate(CORNERS):
        add(*tyre_record(i, c, (toes or {}).get(c)))
    if mirror:
        R = mirror_car(R, toe)
    if over:
        R = [(t, c, over[c]) if c in over else (t, c, v) for t, c, v in R]
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


HYBRID = [(6, 177, 1), (1, 180, 50000.0), (1, 181, 1.0e6), (1, 183, 1.0)]    # hybrid master on, deploy scale, energy cap, map scale


def car_file(name, drop=(), extra=()):
    """A builder car without the records of the cids in drop, with the extra records before the end record."""
    recs = [r for r in build_records() if r[1] not in drop]
    recs = recs[:-1] + list(extra) + recs[-1:]
    path = os.path.join(TMP.name, name)
    carfile.write(recs, path)
    return path


def cid_order(data):
    return [c for _, c, _ in carfile.decode_bytes(data)]


# ---------------------------------------------------------------- (a) CarEdit on a synthetic car
print("(a) CarEdit on a synthetic car: the gate, no-edit bytes, edits by record")
base = car_file("base.car")
src = blob(base)

crafted = os.path.join(TMP.name, "trailing.car")
with open(crafted, "wb") as fh:
    fh.write(src + b"\x00\x00")
ok("D9 gate: data after the end marker refuses the file", (error_text(lambda: CarEdit(crafted)) or "").find("end marker") >= 0)

real_encode = carfile.encode
carfile.encode = lambda recs: real_encode(recs) + b"\x00"       # a writer that does not give the source bytes
try:
    msg = error_text(lambda: CarEdit(base))
finally:
    carfile.encode = real_encode
ok("D9 gate: a file that does not encode back to its bytes is refused, nothing is edited",
   msg is not None and "does not encode back" in msg, msg)

ce = CarEdit(base)
ok("no edits: data() gives the source bytes", ce.data() == src)
ok("no edits: materialize gives the source path", ce.materialize(os.path.join(TMP.name, "unused")) == base)
ok("no edits: validate passes", ce.validate() is None, ce.validate())

ce = CarEdit(base)
ce.set((1, 2), 150000.0)       # FR spring rate
ce.set((108, None), 0.55)      # front brake bias
out = ce.data()
r0, r1 = carfile.decode(base), carfile.decode_bytes(out)
ok("two edits keep the record order", [r[:2] for r in r0] == [r[:2] for r in r1])
diff = sorted(a[1] for a, b in zip(r0, r1) if carfile.encode([a]) != carfile.encode([b]))
ok("only the FR spring record and the brake bias record change", diff == [1, 108], diff)
ok("the other records are bitwise equal", all(carfile.encode([a]) == carfile.encode([b])
                                             for a, b in zip(r0, r1) if a[1] not in (1, 108)))
saved = os.path.join(TMP.name, "fr_bias.car")
ce.save(saved)
ok("save writes the edited bytes", blob(saved) == out)
lc = chassissim.load_car(saved)
ok("the saved file reads the edited FR rate and bias",
   lc.susp.spring_fr["rate"] == 150000.0 and lc.brake.front_bias == 0.55,
   (lc.susp.spring_fr["rate"], lc.brake.front_bias))

ce = CarEdit(base)
ce.set((1, 2), 1.5e5)
ce.set((1, 1), 0.004)           # the same record: spring rate and preload
rec = {c: v for _, c, v in ce.new_records()}
ok("spring rate and preload of one record both survive", rec[1][2] == 1.5e5 and rec[1][1] == 0.004, rec[1][:3])

ce = CarEdit(base)
ce.set((13, 2), 2.5e5)          # the same bump rubber: rate and preload
ce.set((13, 1), 0.002)
rec = {c: v for _, c, v in ce.new_records()}
ok("rubber rate and preload of one record both survive", rec[13][2] == 2.5e5 and rec[13][1] == 0.002, rec[13][:3])

ce = CarEdit(base)
ce.set((80, 0), 0.25)           # FR tyre toe
ce.set((80, 1), -0.5)           # FR tyre camber, the same tyre record
rec = {c: v for _, c, v in ce.new_records()}
ok("two tyre slots of one corner both survive", rec[80][1][0] == 0.25 and rec[80][1][1] == -0.5,
   (rec[80][1][0], rec[80][1][1]))

ce = CarEdit(base)
ce.set((204, None), 0.0)        # a gap flag: an int field
rec = {c: v for _, c, v in ce.new_records()}
ok("an int field stays int", type(rec[204]) is int and rec[204] == 0, type(rec[204]))

# ---------------------------------------------------------------- (b) the left copy
print("(b) the left copy: an absent left record, edits to the pair, the gap flag")
nofl = car_file("no_fl_spring.car", drop=(2,))          # the file lacks the FL spring: ChassisSim copies FR
ce = CarEdit(nofl)
ok("the copy map names cid 1 for the absent cid 2", ce.copies.get(2) == 1 and 2 not in ce.by_cid, ce.copies)
rows = {k: v for k, _, v in ce.rows()}
ok("an absent left record shows the right value", rows[(2, 2)] == rows[(1, 2)] == 131000.0, rows.get((2, 2)))

ce = CarEdit(nofl)
ce.set((1, 2), 2.5e5)           # a right-only edit
recs = {c: v for _, c, v in carfile.decode_bytes(ce.data())}
ok("a right-only edit writes the ORIGINAL right value into the left record", recs[2][2] == 131000.0, recs[2][2])
ok("the right record holds the edit", recs[1][2] == 2.5e5)
lc = model_of(ce)
rows = {k: v for k, _, v in ce.rows()}
ok("the model FL rate equals the shown FL rate", lc.susp.spring_fl["rate"] == rows[(2, 2)] == 131000.0,
   (lc.susp.spring_fl["rate"], rows[(2, 2)]))

ce = CarEdit(nofl)
ce.set((2, 2), 1.1e5)           # a left edit
order = cid_order(ce.data())
ok("a left edit inserts cid 2 right after cid 1", order[order.index(1) + 1] == 2, order[:6])
lc = model_of(ce)
ok("a left edit: FL rate 1.1e5, FR rate unchanged", lc.susp.spring_fl["rate"] == 1.1e5
   and lc.susp.spring_fr["rate"] == 131000.0)

gap = car_file("no_fl_gap.car", drop=(8, 205))         # the FL gap and its flag are absent
ce = CarEdit(gap)
ce.set((8, None), 0.0088)     # a left gap edit: the file shows 0.009 here
order = cid_order(ce.data())
ok("a left gap edit inserts cid 8 right after cid 7", order[order.index(7) + 1] == 8, order[:10])
ok("the gap flag goes along: cid 205 right after cid 204", order[order.index(204) + 1] == 205)
flags = {c: v for _, c, v in carfile.decode_bytes(ce.data())}
ok("the copied flag keeps the right flag value", flags[205] == 1, flags.get(205))

ce = CarEdit(base)
ce.set((8, None), 0.0)          # a left gap of 0 with a live right gap
ok("validate refuses a left gap of 0 with a live right gap",
   ce.validate() == "Gap FL: ChassisSim reads a left gap of 0 as the right gap.", ce.validate())

# ---------------------------------------------------------------- (c) validate
print("(c) validate: range, hybrid refusal, no temp file")
hyb = car_file("hybrid.car", extra=HYBRID)
hyb_err = CarEdit(hyb).validate()
ok("the hybrid base car validates", hyb_err is None, hyb_err)

ce = CarEdit(hyb)
ce.set((183, None), -1.0)
ok("a hybrid value out of range names its field",
   ce.validate() == "Deploy scale after the map: -1 is outside the range 0 to 100.", ce.validate())

priv = os.path.join(TMP.name, "tmp_private")
os.makedirs(priv)
saved_tmp, tempfile.tempdir = tempfile.tempdir, priv
try:
    ce = CarEdit(hyb)
    ce.set((180, None), 0.0)                  # 177 on with the deploy scale at 0
    msg = ce.validate()
    range_msg = (lambda c: (c.set((183, None), -1.0), c.validate())[1])(CarEdit(hyb))
finally:
    tempfile.tempdir = saved_tmp
ok("177 on with 180 = 0 is refused, naming cid 180 and not the temp file",
   msg is not None and "cid 177" in msg and "Cid 180 is 0.0" in msg and "fms_car_check_" not in msg
   and "(edited)" in msg, msg)
ok("the range refusal names its field too", range_msg is not None and "Deploy scale after the map" in range_msg)
ok("validate leaves no temp file", os.listdir(priv) == [], os.listdir(priv))

# ---------------------------------------------------------------- (d) materialize and save
print("(d) materialize and save: one file per bytes, the refusals")
folder = os.path.join(TMP.name, "mat")
ce = CarEdit(base)
ce.set((1, 2), 1.4e5)
p1 = ce.materialize(folder)
p2 = ce.materialize(folder)
files = os.listdir(folder)
ok("materialize writes <base>_<sha8>.car once and reuses it",
   files == [f"base_{hashlib.sha256(ce.data()).hexdigest()[:8]}.car"] and p1 == p2 == os.path.join(folder, files[0]),
   files)
mtime = os.stat(p1).st_mtime_ns
ce.materialize(folder)
ok("a second materialize does not rewrite the file", os.stat(p1).st_mtime_ns == mtime)
with open(p1, "wb") as fh:
    fh.write(b"other")
ok("materialize refuses a file of the same name with other bytes", "exists with other content" in
   (error_text(lambda: ce.materialize(folder)) or ""))

exists = os.path.join(TMP.name, "exists.car")
with open(exists, "wb") as fh:
    fh.write(b"keep")
try:
    ce.save(exists)
    refused = False
except FileExistsError:
    refused = True
ok("save raises FileExistsError for an existing path", refused and blob(exists) == b"keep")

fake = os.path.join(TMP.name, "fake_install", "ChassisSim")
os.makedirs(os.path.join(fake, "Circuits"))
os.makedirs(os.path.join(TMP.name, "fake_install", "Models", "GP"))
ok("the guard refuses the Circuits folder of the ChassisSim folder", in_chassissim(os.path.join(fake, "Circuits", "x.car"), fake))
ok("the guard refuses the ChassisSim folder itself", in_chassissim(os.path.join(fake, "x.car"), fake))
ok("the guard refuses the Models sibling and its folders",
   in_chassissim(os.path.join(TMP.name, "fake_install", "Models", "GP", "x.car"), fake)
   and in_chassissim(os.path.join(TMP.name, "fake_install", "Models", "x.car"), fake))
ok("the guard allows a folder beside the ChassisSim folder",
   not in_chassissim(os.path.join(TMP.name, "x.car"), fake)
   and not in_chassissim(fake + "_copy", fake) and not in_chassissim(os.path.join(fake + "_copy", "x.car"), fake))
ok("the guard allows any folder when no ChassisSim folder is given", not in_chassissim(os.path.join(fake, "x.car"), None))
link = os.path.join(TMP.name, "link_to_cs")
os.symlink(fake, link)
ok("the guard follows a symlink to the ChassisSim folder", in_chassissim(os.path.join(link, "Circuits", "x.car"), fake))

# ---------------------------------------------------------------- (e) labels and ranges
print("(e) labels, units and ranges")
ok("cid 121 is the lift area in m^2", FIELDS[(121, None)].label == "Lift area CL x A" and FIELDS[(121, None)].unit == "m^2")
ok("cid 122 is the drag area in m^2", FIELDS[(122, None)].label == "Drag area CD x A" and FIELDS[(122, None)].unit == "m^2")
for name, car in (("the synthetic car", base), ("the hybrid car", hyb)):
    rows = CarEdit(car).rows()
    outside = [(f.label, v) for _, f, v in rows if not f.lo <= v <= f.hi]
    ok(f"{name}: {len(rows)} rows, none outside its range", rows and not outside, outside[:4])

# ---------------------------------------------------------------- (f) the window
print("(f) the window (offscreen Qt): car mode, edits, Run, Save, the locale")
from PySide6.QtCore import QLocale                                       # noqa: E402
from PySide6.QtGui import QValidator                                     # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog                 # noqa: E402
from app import paths, results                                          # noqa: E402
from app.mainwindow import MainWindow                                   # noqa: E402
from app.widgets import ScientificField                                 # noqa: E402
import headless_solve                                                   # noqa: E402

paths.user_presets_dir = lambda: PRESETS            # the window reads presets/cars from here (app.paths.user_cars_dir)
APP = QApplication.instance() or QApplication([])


CFGS = []                                  # the cfg files of the runs: removed at the end


def run_capture(h):
    """Press Run with the solve runner replaced by a list. Return the cfg path, or None when no run started."""
    started = []
    real = h._runner.start
    h._runner.start = started.append
    try:
        h._on_run()
    finally:
        h._runner.start = real
    if started:
        CFGS.append(started[0])
    return started[0] if started else None


def read_cfg(path):
    with open(path) as fh:
        return json.load(fh)


def fresh_settings():
    s = paths.settings()
    s.clear()
    s.sync()


REAL_OPEN, REAL_SAVE = QFileDialog.getOpenFileName, QFileDialog.getSaveFileName
fresh_settings()
OUT = os.path.join(TMP.name, "out")
os.makedirs(OUT)
w = MainWindow()
w.output_dir.setText(OUT)
w.chassissim_dir.setText(INSTALL)
w._refresh_chassissim()
w.plot_cb.setChecked(False)
w._select_car(base)
w.circuit.setCurrentText(chassissim.PREFIX + CIRC)
w.opt_combos["mode"].setCurrentText("qss")
F = w._car_fields
ok("car mode: every 4EM section is hidden", bool(w._sections) and all(s.isHidden() for s in w._sections.values()))
ok("car mode: the car sections show every row of the file", len(F) > 0 and len(F) == len(CarEdit(base).rows()), len(F))

F[(1, 2)].setValue(1.5e5)                  # FR spring rate
sec = w._car_secs["Springs and bars"][0]
ok("an edited field is highlighted", "b30000" in F[(1, 2)]._edit.styleSheet())
ok("the section badge counts the edit", "(1 changed)" in sec._btn.text(), sec._btn.text())
w._reset_all_params()
ok("Reset all clears the edits and shows the file values",
   w._car_edit.edits == {} and F[(1, 2)].value() == w._car_edit.file_value((1, 2)), w._car_edit.edits)
ok("the badge is back to no count", "changed)" not in sec._btn.text(), sec._btn.text())

F[(1, 2)].setValue(1.5e5)
cfg = run_capture(w)
cfg_1 = read_cfg(cfg) if cfg else {}
car_run = cfg_1.get("car", "")[len(chassissim.PREFIX):]
ok("Run with edits starts one run", cfg is not None)
ok("Run with edits: the cfg car is a file of presets/cars", os.path.dirname(car_run) == CARS and os.path.isfile(car_run),
   car_run)
ok("Run with edits: vp_overrides is empty", cfg_1.get("vp_overrides") == {}, cfg_1.get("vp_overrides"))
ok("Run with edits: the run file holds the edited FR rate",
   os.path.isfile(car_run) and chassissim.load_car(car_run).susp.spring_fr["rate"] == 1.5e5)
cfg = run_capture(w)
cfg_2 = read_cfg(cfg) if cfg else {}
ok("a second Run reuses the file", cfg_2.get("car") == cfg_1.get("car") and os.listdir(CARS) == [os.path.basename(car_run)],
   os.listdir(CARS))

F[(8, None)].setValue(0.0)                 # FL gap 0 with a live FR gap
w.log.clear()
cfg = run_capture(w)
ok("a validation error stops the Run with an [error] line",
   cfg is None and "[error] Gap FL: ChassisSim reads a left gap of 0 as the right gap." in w.log.toPlainText(),
   w.log.toPlainText()[-200:])
w._reset_all_params()

OTHER = car_file("other_car.car", drop=(2,))
F[(1, 2)].setValue(1.5e5)
w.log.clear()
w._select_car(OTHER)
ok("a car change drops the pending edits with one log line",
   w._car_edit.base == "other_car" and w._car_edit.edits == {} and w.log.toPlainText().count("drops the setup edits") == 1,
   w.log.toPlainText())

w._select_car(base)
F = w._car_fields
F[(1, 2)].setValue(1.5e5)
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (base, ""))
w.log.clear()
w._load_preset()
F = w._car_fields
ok("Load preset of the selected car drops the edits and shows the file values",
   w._car_edit.edits == {} and F[(1, 2)].value() == w._car_edit.file_value((1, 2)) and "drops the setup edits" in w.log.toPlainText(),
   w.log.toPlainText())

SAVED = os.path.join(TMP.name, "saved", "edit_one.car")
os.makedirs(os.path.dirname(SAVED))
F[(1, 2)].setValue(1.5e5)
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (SAVED, ""))
w.log.clear()
w._save_preset()
first = blob(SAVED) if os.path.isfile(SAVED) else b""
ok("Save in car mode writes the edited car to a new file",
   first == w._car_edit.data() and "[car saved]" in w.log.toPlainText(), w.log.toPlainText())
F[(1, 2)].setValue(1.6e5)
w.log.clear()
w._save_preset()
ok("Save refuses an existing name and keeps its bytes",
   "exists. Choose a new name" in w.log.toPlainText() and blob(SAVED) == first)

for target, label in ((os.path.join(INSTALL, "Circuits", "guard.car"), "the Circuits folder"),
                      (os.path.join(TMP.name, "Models", "guard.car"), "the Models sibling")):
    QFileDialog.getSaveFileName = staticmethod(lambda *a, t=target, **k: (t, ""))
    w.log.clear()
    w._save_preset()
    ok(f"Save refuses a file in {label} of the ChassisSim folder",
       "Do not save into the ChassisSim folders" in w.log.toPlainText() and not os.path.exists(target),
       w.log.toPlainText())
QFileDialog.getOpenFileName, QFileDialog.getSaveFileName = REAL_OPEN, REAL_SAVE

w.car.setCurrentIndex(0)
F = w._car_fields
ok("back on the 4EM car: 130 rows, every 4EM section shown, no car rows",
   len(w.vp_spins) == 130 and len(w._sections) == 14 and not any(s.isHidden() for s in w._sections.values())
   and F == {}, (len(w.vp_spins), len(w._sections)))

prev_locale = QLocale()
QLocale.setDefault(QLocale(QLocale.German, QLocale.Germany))
try:
    sf = ScientificField()
    st_dot = sf._edit.validator().validate("0.55", 0)[0]
    st_comma = sf._edit.validator().validate("0,55", 0)[0]
    sf._edit.setText("0.55")
    val = sf.value()
finally:
    QLocale.setDefault(prev_locale)
ok("ScientificField accepts 0.55 in a comma-decimal locale", st_dot == QValidator.Acceptable and val == 0.55)
ok("ScientificField refuses 0,55 (the point is the decimal mark)", st_comma != QValidator.Acceptable)

# ---------------------------------------------------------------- vendor section
print("vendor section (CHASSISSIM_DIR)")
if VENDOR is None:
    print("  SKIP: CHASSISSIM_DIR not set (vendor section)")
else:
    models = os.path.join(os.path.dirname(VENDOR), "Models")
    cars = sorted({os.path.abspath(p) for p in glob.glob(os.path.join(VENDOR, "*.car"))
                   + glob.glob(os.path.join(models, "**", "*.car"), recursive=True)})
    before = tree(VENDOR) + tree(models)
    ok(f"vendor: the car files are found ({len(cars)})", len(cars) >= 1)
    gate_bad, bytes_bad, range_bad = [], [], []
    for p in cars:
        try:
            ce = CarEdit(p)
        except ValueError as exc:
            gate_bad.append((os.path.basename(p), str(exc)[:80]))
            continue
        if ce.data() != blob(p) or ce.materialize(os.path.join(TMP.name, "unused")) != p:
            bytes_bad.append(os.path.basename(p))
        range_bad += [(os.path.basename(p), f.label, v) for _, f, v in ce.rows() if not f.lo <= v <= f.hi]
    ok("vendor: every car passes the D9 gate", not gate_bad, gate_bad[:3])
    ok("vendor: every car gives its source bytes and its source path with no edits", not bytes_bad, bytes_bad[:3])
    ok("vendor: no shipped row value is outside its range", not range_bad, range_bad[:3])

    gp3 = sorted(glob.glob(os.path.join(models, "GP3*", "Eastern*", "def*.car")))
    ok("vendor: the GP3 car of Eastern Creek is found", len(gp3) >= 1)
    src = gp3[0]
    h0 = sha(src)
    ce = CarEdit(src)
    fr = ce.file_value((1, 2))
    ce.set((1, 2), fr * 1.2)
    ce.set((1, 1), 0.004)
    run_file = ce.materialize(os.path.join(TMP.name, "vendor_runs"))
    lc = chassissim.load_car(run_file, VENDOR)
    ok("vendor: GP3 FR rate x1.2 and preload 0.004 both in the Run file",
       abs(lc.susp.spring_fr["rate"] - fr * 1.2) < 1e-6 and lc.susp.spring_fr["preload"] == 0.004,
       (lc.susp.spring_fr["rate"], lc.susp.spring_fr["preload"]))
    ok("vendor: the GP3 edit validates", ce.validate(VENDOR) is None, ce.validate(VENDOR))

    fresh_settings()
    hv = MainWindow()
    hv.chassissim_dir.setText(VENDOR)
    hv._refresh_chassissim()
    hv.plot_cb.setChecked(False)
    hv.circuit.setCurrentText(chassissim.PREFIX + "Eastern Creek")
    hv._select_car(src)
    hv.opt_combos["mode"].setCurrentText("qss")
    hv.tol.setValue(1e-3)
    hv.output_dir.setText(os.path.join(TMP.name, "vendor_out"))
    hv._car_fields[(1, 2)].setValue(fr * 1.2)
    cfg_v = run_capture(hv)
    cfg_vd = read_cfg(cfg_v) if cfg_v else {}
    with contextlib.redirect_stdout(io.StringIO()):
        code_v = headless_solve.main([cfg_v]) if cfg_v else 1
    hv._on_finished(1)
    mat_v, _ = results.result_paths(hv._active_rc)
    lap_v = results.parse_summary(mat_v)["lap_time_s"]
    ok(f"vendor: GUI qss run of GP3 with the FR spring at +20 %: lap {lap_v:.5f} s",
       code_v == 0 and abs(lap_v - 74.0695) < 1e-3, lap_v)
    ok("vendor: the run uses the edited file, not the source",
       cfg_vd.get("car", "").startswith(chassissim.PREFIX + CARS) and sha(src) == h0)
    ok("vendor: the source file sha256 is unchanged", sha(src) == h0)
    ok("vendor: the vendor folders keep their file list", tree(VENDOR) + tree(models) == before)

for p in CFGS:
    if os.path.exists(p):
        os.remove(p)
print(f"\n{COUNT[0]} checks passed")
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL test_car_edit TESTS PASSED")
