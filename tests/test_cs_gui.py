"""GUI and car-selection tests for the ChassisSim car and circuit lists (phase 5, lane P5-A).

Sections: (a) chassissim.list_circuits / list_cars on synthetic folders, (b) the desktop window
(offscreen Qt): folder field, circuit and car combos, car coupling, the captured cfg, presets,
restarts, (c) a GUI-built qss cfg of a synthetic car through headless_solve.main, (d) the refusals of
userOpts and MLTP for a car, and a vendor section (SKIP without CHASSISSIM_DIR) with the GP3 car at
Eastern Creek. Every GUI state lives in a temp home: settings, presets, output and the Documents
folder. Synthetic car files come from the car builder of tests/test_cs_wiring.py (copied, not imported).

    python tests/test_cs_gui.py
    CHASSISSIM_DIR="<ChassisSim install folder>" python tests/test_cs_gui.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import atexit
import contextlib
import copy
import glob
import io
import json
import os
import re
import shutil
import tempfile
import types
import warnings

import numpy as np

from functions import carfile

# A temp home for everything the GUI reads or writes. Set it before Qt starts (see app.paths).
TMP = types.SimpleNamespace(name=tempfile.mkdtemp(prefix="cs_gui_"))
atexit.register(shutil.rmtree, TMP.name, True)
HOME = os.path.join(TMP.name, "home")
os.makedirs(os.path.join(HOME, ".config"))
os.environ["HOME"] = HOME
os.environ["XDG_CONFIG_HOME"] = os.path.join(HOME, ".config")
os.environ["FULLMODELSIM_DOCUMENTS"] = os.path.join(HOME, "Documents")   # Windows: Qt does not read HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
VENDOR = os.environ.pop("CHASSISSIM_DIR", None)      # the non-vendor sections run with no environment folder
VENDOR = VENDOR if VENDOR and os.path.isdir(VENDOR) else None
PRESETS = os.path.join(TMP.name, "presets")          # the user presets folder (presets/cars holds saved cars)
os.makedirs(os.path.join(PRESETS, "cars"))

PASS = 0


def ok(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASS += 1


def set_env(value):
    if value is None:
        os.environ.pop("CHASSISSIM_DIR", None)
    else:
        os.environ["CHASSISSIM_DIR"] = value


from PySide6.QtWidgets import QApplication, QFileDialog
import app.mainwindow as mw
from app.mainwindow import MainWindow, CIRCUITS, CAR_4EM
from app import paths, results
import headless_solve
from functions.chassissim import PREFIX, list_cars, list_circuits, load_circuit

paths.user_presets_dir = lambda: PRESETS            # the window reads presets/cars from here (app.paths.user_cars_dir)
APP = QApplication.instance() or QApplication([])
ok("Documents and the presets folder sit in the temp home", paths.default_output_dir().startswith(TMP.name)
   and paths.user_presets_dir().startswith(TMP.name))


def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="latin-1", newline="") as f:
        f.write(text)


def curv_text():
    return "\r\n".join(f"{0.01 * (1 + i)!r} {10.0 * i!r}" for i in range(5)) + "\r\n"   # 'k s' rows, s increasing


def cs_tree(root, ini=None, files=()):
    """A ChassisSim-like folder: Circuits/, the optional Circ_prop.ini, and the listed files (tr_* get a curvature)."""
    os.makedirs(os.path.join(root, "Circuits"), exist_ok=True)
    if ini is not None:
        put(os.path.join(root, "Circ_prop.ini"), ini)
    for rel in files:
        put(os.path.join(root, rel), curv_text() if os.path.basename(rel).lower().startswith("tr_") else "car\r\n")
    return root


def run_capture(h):
    """Press Run with the solve runner replaced by a list. Return the cfg path, or None when the run did not start."""
    started = []
    real = h._runner.start
    h._runner.start = started.append
    try:
        h._on_run()
    finally:
        h._runner.start = real
    return started[0] if started else None


def fresh_settings():
    s = paths.settings()
    s.clear()
    s.sync()

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
EXPLICIT.update({142: 0.5 * (EXPLICIT[117] + EXPLICIT[118]) * 1e-3,
                 144: 0.5 * (EXPLICIT[119] + EXPLICIT[120]) * 1e-3, 56: 0.0, 60: 0.0, 64: 0.0, 68: 0.0})
EXPLICIT.update({c: 0.0 for c in (55, 59, 63, 67) if c not in EXPLICIT})
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
    for cid in (53, 61, 69):
        add(6, cid, 0)                                   # geometry mode 0 (53, 61), no live axle (69)
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


def write_car(name, arb_front=None, mirror=False, toe=0.0, toes=None, over=None):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(arb_front, mirror, toe, toes, over), path)
    return path


def raised(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------- (a) the lists on synthetic folders
print("(a) list_circuits and list_cars on synthetic folders")
LIST = os.path.join(TMP.name, "lists")
LIST_INI = ("[Total Circ prop]\r\nTotal circuits = 3\r\n\r\n"
            "[Circuit0]\r\nCircuit name = Ring A\r\nCircuit file name = tr_ringA.txt\r\nBase Grip factor = 1.0\r\n"
            "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n"
            "[Circuit1]\r\nCircuit name = Ring B\r\nCircuit file name = tr_ringB.txt\r\nBase Grip factor = 1.0\r\n"
            "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n"
            "[Circuit2]\r\nCircuit name = Ring D\r\nCircuit file name = tr_ringd_gone.txt\r\nBase Grip factor = 1.0\r\n"
            "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n")

# Ring A has its curvature file. Ring B and Ring D have none, so they are not circuits. tr_orphan1.txt has no
# block, so it is a circuit. tr_Ring D.txt has no block, but its name equals the block name Ring D, so it is dropped.
circ_root = cs_tree(os.path.join(LIST, "circ"), LIST_INI,
                    ["Circuits/tr_ringA.txt", "Circuits/tr_orphan1.txt", "Circuits/tr_Ring D.txt"])
ok("an ini block with its curvature file and an orphan tr file: the listed names",
   list_circuits(circ_root) == ["Ring A", "orphan1"])
ok("every listed name loads with load_circuit",
   all(load_circuit(n, circ_root).s.size > 2 for n in list_circuits(circ_root)))
ok("an ini block without its curvature file is not listed",
   "Ring B" not in list_circuits(circ_root) and "Ring D" not in list_circuits(circ_root))
ok("... and load_circuit raises FileNotFoundError for it", raised(lambda: load_circuit("Ring B", circ_root), FileNotFoundError))

missing = os.path.join(LIST, "no such folder")
ok("a missing folder lists no circuits and no cars", list_circuits(missing) == [] and list_cars(missing) == [])

# A typo of the folder name: the parent has a Models folder with a car, but the typed folder does not exist.
typo_parent = os.path.join(LIST, "typo")
put(os.path.join(typo_parent, "Models", "GP", "x.car"), "car\r\n")
ok("a typo folder lists 0 cars, though its parent has Models",
   list_cars(os.path.join(typo_parent, "ChasSim")) == [] and len(list_cars(os.path.join(typo_parent, "Models", "GP"))) == 1)

# Brackets in a folder name: the pattern must not read [v3] as a character class (it would match "CS v").
v3 = os.path.join(LIST, "v3")
put(os.path.join(v3, "CS [v3]", "x.car"), "car\r\n")
put(os.path.join(v3, "CS v", "y.car"), "car\r\n")
ok("a folder name with [v3] lists its own car only",
   list_cars(os.path.join(v3, "CS [v3]")) == [os.path.abspath(os.path.join(v3, "CS [v3]", "x.car"))])

# A file where the folder should be: no circuits, no cars, no error.
as_file = os.path.join(LIST, "file_folder.car")
put(as_file, "car\r\n")
ok("a file given as the folder lists nothing", list_cars(as_file) == [] and list_circuits(as_file) == [])

# Source order: install folder, then Models (recursive), then the extra folder. Each group by file name.
game = os.path.join(LIST, "game")
cs_dir = os.path.join(game, "CS")
for name in ("b.car", "a.car"):
    put(os.path.join(cs_dir, name), "car\r\n")
put(os.path.join(game, "Models", "d.car"), "car\r\n")
put(os.path.join(game, "Models", "Sub", "c.car"), "car\r\n")
extra = os.path.join(LIST, "extra")
for name in ("e.car", "a.car"):
    put(os.path.join(extra, name), "car\r\n")
want = [os.path.join(cs_dir, "a.car"), os.path.join(cs_dir, "b.car"),
        os.path.join(game, "Models", "Sub", "c.car"), os.path.join(game, "Models", "d.car"),
        os.path.join(extra, "a.car"), os.path.join(extra, "e.car")]
ok("cars: install folder, Models (recursive), extra folder, each by file name",
   list_cars(cs_dir, extra) == [os.path.abspath(p) for p in want])

# ---------------------------------------------------------------- (b) the desktop window
print("(b) the window: folder field, combos, coupling, cfg, presets, restarts")
set_env(None)
GUI = os.path.join(TMP.name, "gui")
GUI_CS = cs_tree(os.path.join(GUI, "Install CS"), LIST_INI,
                 ["Circuits/tr_ringA.txt", "Circuits/tr_orphan1.txt", "Circuits/tr_Ring D.txt",
                  "def.car", "car_a.car"])
CAR_A = os.path.join(GUI_CS, "car_a.car")
CAR_DEF_SUB = os.path.join(GUI, "Models", "Sub", "def.car")
put(CAR_DEF_SUB, "car\r\n")
OUT = os.path.join(TMP.name, "out")
os.makedirs(OUT)
CIRC_ITEMS = CIRCUITS + [PREFIX + "Ring A", PREFIX + "orphan1"]
ABS_A = os.path.abspath(CAR_A)


def circuit_items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


# (b1) the folder field is saved on Run and restored. The circuit combo lists the ChassisSim circuits
fresh_settings()
h = MainWindow()
h.chassissim_dir.setText(GUI_CS)
h._refresh_chassissim()
h.output_dir.setText(OUT)
ok("a run starts one solve (captured, no process)", run_capture(h) is not None)
h._on_finished(1)
ok("the ChassisSim folder is saved on Run", paths.settings().value("chassissim_dir") == GUI_CS)
h2 = MainWindow()
ok("a new window restores the folder field", h2.chassissim_dir.text() == GUI_CS)
ok("the circuit combo = CIRCUITS + the listed chassissim:<name> entries", circuit_items(h2.circuit) == CIRC_ITEMS)

# (b2) an empty field: the environment variable is the folder
fresh_settings()
set_env(GUI_CS)
h3 = MainWindow()
ok("an empty field takes CHASSISSIM_DIR for the circuits", circuit_items(h3.circuit) == CIRC_ITEMS)
set_env(None)

# (b3) car labels: a repeated file name gets the parent folder name (install folder first, then Models)
fresh_settings()
paths.settings().setValue("chassissim_dir", GUI_CS)
h = MainWindow()
labels = [h.car.itemText(i) for i in range(1, h.car.count())]
ok("car labels: the file name, and the parent folder for a repeated name",
   labels == ["car_a.car", "def.car (Install CS)", "def.car (Sub)"] and h.car.itemText(0) == CAR_4EM)
ok("the car items carry the absolute paths",
   h.car.findData(ABS_A) >= 0 and h.car.findData(os.path.abspath(CAR_DEF_SUB)) >= 0)

# (b4) a car sets and locks the controls. The 4EM car restores them
h.opt_combos["physics"].setCurrentText("matlab")
h.aero.setCurrentText("Active")
h.em4.setChecked(False)
h.atd.setChecked(True)
h.homotopy.setChecked(True)
h.tyre.setCurrentText("PureSlip")
h.tyre_set.setCurrentText("CopyB")
h.opt_combos["start_state"].setCurrentText("rolling")
h._select_car(CAR_A)
st = h.opt_combos["start_state"]
ok("a car sets physics corrected, aero Static, 4 Motors off, ATD off, Homotopy off",
   h.opt_combos["physics"].currentText() == "corrected" and h.aero.currentText() == "Static"
   and not h.em4.isChecked() and not h.atd.isChecked() and not h.homotopy.isChecked())
ok("... and disables each of them",
   not h.opt_combos["physics"].isEnabled() and not h.aero.isEnabled() and not h.em4.isEnabled()
   and not h.atd.isEnabled() and not h.homotopy.isEnabled())
ok("a car sets Tyre model CombinedSlip and Tyre set MF205, both disabled",
   h.tyre.currentText() == "CombinedSlip" and not h.tyre.isEnabled()
   and h.tyre_set.currentText() == "MF205" and not h.tyre_set.isEnabled())
ok("a car moves the start state off rolling to auto and disables the rolling item",
   st.currentText() == "auto" and not st.model().item(2).isEnabled())
h.car.setCurrentIndex(0)
ok("the 4EM car restores physics, aero, ATD, Homotopy, tyre and tyre set",
   h.opt_combos["physics"].currentText() == "matlab" and h.aero.currentText() == "Active"
   and h.atd.isChecked() and h.homotopy.isChecked() and h.tyre.currentText() == "PureSlip"
   and h.tyre_set.currentText() == "CopyB" and h.tyre.isEnabled() and h.atd.isEnabled())
ok("... the rolling start comes back, and the Setup sections are enabled again",
   st.currentText() == "rolling" and st.model().item(2).isEnabled()
   and all(s.isEnabled() for s in h._sections.values()))

# (b5) a car with a changed mb row: the captured cfg has no vp_overrides and the car path. 4EM keeps the row
h.car.setCurrentIndex(h.car.findData(ABS_A))
mb0 = h.vp_spins["mb"].value()
h.vp_spins["mb"].setValue(mb0 + 100.0)
h.output_dir.setText(OUT)
cfg_path = run_capture(h)
cfg_car = json.load(open(cfg_path))
h._on_finished(1)
ok("car + changed mb row: the cfg has no vp_overrides", cfg_car["vp_overrides"] == {})
ok("car + changed mb row: the cfg car is chassissim:<absolute path>", cfg_car["car"] == PREFIX + ABS_A)
ok("car + changed mb row: the table keeps the changed value", h.vp_spins["mb"].value() == mb0 + 100.0)
h.car.setCurrentIndex(0)
cfg_path = run_capture(h)
cfg_4em = json.load(open(cfg_path))
h._on_finished(1)
ok("the 4EM car sends the changed mb row", cfg_4em["vp_overrides"].get("mb") == mb0 + 100.0 and cfg_4em["car"] is None)

# (b6) Load preset with a .car selects that car
real_dialog = QFileDialog.getOpenFileName
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (os.path.abspath(CAR_DEF_SUB), ""))
try:
    h._load_preset()
finally:
    QFileDialog.getOpenFileName = real_dialog
ok("Load preset with a .car selects that car", h.car.currentData() == os.path.abspath(CAR_DEF_SUB))

# (b7) a browsed car outside the folders is kept on restart
outside = os.path.join(TMP.name, "outside", "browsed.car")
put(outside, "car\r\n")
h._select_car(outside)
h.output_dir.setText(OUT)
run_capture(h)
h._on_finished(1)
h4 = MainWindow()
ok("a restart with a browsed car outside the folders keeps it", h4.car.currentData() == os.path.abspath(outside))

# (b8) a saved car that no longer exists: one [car] log line and the 4EM car
fresh_settings()
paths.settings().setValue("chassissim_dir", GUI_CS)
paths.settings().setValue("car", os.path.join(TMP.name, "gone", "vanished.car"))
h5 = MainWindow()
log5 = h5.log.toPlainText()
ok("a saved car that no longer exists gives the 4EM car", h5.car.currentData() is None)
ok("... and one [car] log line, no [error]", log5.count("[car]") == 1 and "[error]" not in log5)

# (b9) a circuit that leaves the list goes back to Sturn with one [circuit] line
fresh_settings()
paths.settings().setValue("chassissim_dir", GUI_CS)
h6 = MainWindow()
h6.circuit.setCurrentText(PREFIX + "Ring A")
h6.log.clear()
h6.chassissim_dir.setText(os.path.join(TMP.name, "gone_folder"))
h6._refresh_chassissim()
log6 = h6.log.toPlainText()
ok("a circuit that leaves the list falls back to the first circuit, one [circuit] line",
   h6.circuit.currentText() == CIRCUITS[0] and log6.count("[circuit]") == 1)

# (b10) a bad Circ_prop.ini: the window starts with one [error] line, and still lists the cars
BAD_FORMS = {
    "a": "garbage line\n[Circuit1]\nCircuit name = X\n",
    "b": "[Circuit1]\nCircuit name = X\nno equals sign here\n",
}
for form, text in BAD_FORMS.items():
    for route in ("saved", "env"):
        bad = cs_tree(os.path.join(TMP.name, f"bad_{form}_{route}", "CS"), text, ["one.car"])
        fresh_settings()
        if route == "saved":
            paths.settings().setValue("chassissim_dir", bad)
            set_env(None)
        else:
            set_env(bad)
        hb = MainWindow()
        logb = hb.log.toPlainText()
        ok(f"bad ini form {form}, folder {route}: one [error] line, and the car is listed",
           logb.count("[error]") == 1 and hb.car.count() == 2 and hb.car.itemText(1) == "one.car")
        set_env(None)


# ---------------------------------------------------------------- (c) a GUI-built qss cfg of a synthetic car
print("(c) qss run of a synthetic car through headless_solve.main")
fresh_settings()
set_env(None)
hc = MainWindow()
hc.chassissim_dir.setText(INSTALL)
hc._refresh_chassissim()
BASE = write_car("gui_base.car")                 # the synthetic car of section 3 of tests/test_cs_wiring.py
hc._select_car(BASE)
hc.circuit.setCurrentText(PREFIX + CIRC)
ok("the synthetic circuit is listed and selected", hc.circuit.currentText() == PREFIX + CIRC)
hc.opt_combos["mode"].setCurrentText("qss")
QSS_OUT = os.path.join(TMP.name, "qss_out")
hc.output_dir.setText(QSS_OUT)
cfg_qss = run_capture(hc)
ok("the qss run starts with one cfg", cfg_qss is not None)
with contextlib.redirect_stdout(io.StringIO()):
    code_qss = headless_solve.main([cfg_qss])
hc._on_finished(1)
mat_qss, pdir_qss = results.result_paths(hc._active_rc)
res_files = os.listdir(os.path.dirname(mat_qss))
ok("headless_solve.main exits 0 on the GUI cfg of the car", code_qss == 0)
ok("the file that result_paths names is the only file in Results",
   os.path.isfile(mat_qss) and res_files == [os.path.basename(mat_qss)])
# the car runs on the synthetic ring, which joins at the line: start_state auto resolves to periodic (lane S)
ok("its name holds the car part, the physics part, the periodic start, the setup tag and _qss",
   re.search(r"_car_gui_base_physCorrected_startPeriodic_[0-9a-f]{8}_qss\.mat$", os.path.basename(mat_qss)) is not None)
ok("the qss plot folder is empty", pdir_qss == "")
lap_qss = results.parse_summary(mat_qss)["lap_time_s"]
ok("the file holds a finite qss lap value", lap_qss == lap_qss and lap_qss > 0.0)

# (c2) the NLP cfg of the car (lane S). start_state auto sends no key and names the periodic stem. An explicit
# matlab sends the key and names the plain stem (no start suffix)
hc.opt_combos["mode"].setCurrentText("nlp")
hc.opt_combos["start_state"].setCurrentText("auto")
with open(run_capture(hc)) as fh:
    cfg_auto = json.load(fh)
kw_auto = headless_solve.build_solve_kwargs(cfg_auto, TMP.name)
ok("auto: the solve kwargs hold no start_state key", "start_state" not in kw_auto)
mat_auto, _ = results.result_paths(hc._active_rc)
ok("auto: result_paths names the periodic start (the synthetic ring joins at the line)",
   "_startPeriodic_" in os.path.basename(mat_auto))
hc.opt_combos["start_state"].setCurrentText("matlab")
with open(run_capture(hc)) as fh:
    cfg_mat = json.load(fh)
kw_mat = headless_solve.build_solve_kwargs(cfg_mat, TMP.name)
ok("matlab: the solve kwargs carry start_state matlab", kw_mat.get("start_state") == "matlab")
mat_m, _ = results.result_paths(hc._active_rc)
ok("matlab: result_paths gives no start suffix", "_start" not in os.path.basename(mat_m))

# ---------------------------------------------------------------- (d) refusals with a car
print("(d) refusals: userOpts and MLTP with a car, and the 4EM car accepts the same values")
import MLTP as MM                                 # the module (the class is MLTP.MLTP)
from functions.context import Ctx
from userOpts import userOpts


def refusal(fn):
    """The ValueError text of fn, or None when fn runs."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn()
    except ValueError as exc:
        return str(exc)
    return None


car_kw = dict(circuit=PREFIX + CIRC, car=PREFIX + BASE, chassissim_dir=INSTALL, physics="corrected", ATD="Off")
m_tyre = refusal(lambda: userOpts(Ctx(), tyre_set="CopyB", **car_kw))
ok("userOpts with a car refuses tyre_set CopyB", m_tyre is not None and "CopyB" in m_tyre)
m_cam = refusal(lambda: userOpts(Ctx(), CamberGain="Linear", **car_kw))
ok("userOpts with a car refuses CamberGain Linear", m_cam is not None and "Linear" in m_cam)
ok("the 4EM car accepts tyre_set CopyB and CamberGain Linear",
   refusal(lambda: userOpts(Ctx(), circuit="Sturn", tyre_set="CopyB")) is None
   and refusal(lambda: userOpts(Ctx(), circuit="Sturn", CamberGain="Linear")) is None)

calls = []


def guard(*args, **kwargs):
    calls.append(1)
    raise RuntimeError("guard: a build was reached")


saved_veh, saved_build = MM.vehModel, MM.build_and_solve_nlp
MM.vehModel = guard
MM.build_and_solve_nlp = guard
try:
    m_pure = refusal(lambda: MM.MLTP(save=False, plot=False, linear_solver="mumps", TyreModel="PureSlip", **car_kw))
    ok("MLTP with a car refuses TyreModel PureSlip before any build",
       m_pure is not None and "PureSlip" in m_pure and calls == [])
    reached = False
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MM.MLTP(save=False, plot=False, linear_solver="mumps", TyreModel="CombinedSlip", **car_kw)
    except RuntimeError as exc:
        reached = "guard" in str(exc)
    ok("the same car with CombinedSlip reaches the build guard (the guard can fail)", reached and calls == [1])
finally:
    MM.vehModel, MM.build_and_solve_nlp = saved_veh, saved_build

# ---------------------------------------------------------------- vendor section
print("vendor section (CHASSISSIM_DIR)")
if VENDOR is None:
    print("  SKIP: CHASSISSIM_DIR not set (vendor section)")
else:
    v_names = list_circuits(VENDOR)
    v_cars = list_cars(VENDOR)
    ok(f"vendor: list_circuits gives 16 names ({len(v_names)})", len(v_names) == 16)
    ok("vendor: no Motoland and no Sebring ESM circuit", not any("Motoland" in n or "Sebring ESM" in n for n in v_names))
    ok(f"vendor: list_cars gives 29 files ({len(v_cars)})", len(v_cars) == 29)
    gp3 = sorted(glob.glob(os.path.join(os.path.dirname(VENDOR), "Models", "GP3*", "Eastern*", "def*.car")))
    ok("vendor: the GP3 car of Eastern Creek is found by a folder glob", len(gp3) >= 1)
    fresh_settings()
    set_env(None)
    hv = MainWindow()
    hv.chassissim_dir.setText(VENDOR)
    hv._refresh_chassissim()
    hv.circuit.setCurrentText(PREFIX + "Eastern Creek")
    hv._select_car(gp3[0])
    hv.opt_combos["mode"].setCurrentText("qss")
    hv.tol.setValue(1e-3)
    VENDOR_OUT = os.path.join(TMP.name, "vendor_out")
    hv.output_dir.setText(VENDOR_OUT)
    cfg_v = run_capture(hv)
    with contextlib.redirect_stdout(io.StringIO()):
        code_v = headless_solve.main([cfg_v])
    hv._on_finished(1)
    mat_v, _ = results.result_paths(hv._active_rc)
    lap_v = results.parse_summary(mat_v)["lap_time_s"]
    ok(f"vendor: GP3 car at Eastern Creek, GUI qss cfg, a flying lap (QSS march starts at the end speed) "
       f"to 73.71639144080545 s: lap {lap_v!r} s", code_v == 0 and abs(lap_v - 73.71639144080545) < 1e-9)
    ok("vendor: the written file equals result_paths(rc)",
       os.path.isfile(mat_v) and os.listdir(os.path.dirname(mat_v)) == [os.path.basename(mat_v)])

print(f"\n{PASS} checks passed")
print("ALL test_cs_gui TESTS PASSED")
