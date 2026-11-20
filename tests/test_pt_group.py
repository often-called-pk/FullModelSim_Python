"""Powertrain group of the Setup tab for the 4EM car (phase 5, lane P5-C). Offscreen Qt, qss only, no IPOPT.

Sections: (a) defaults, (b) Tmax edits and their tag, (c) exact W for the kW rows, (d) physics matlab and
corrected, (e) car mode hides the group, (f) the one-place rule with More options, (g) presets and the Expert
path, (h) enabled spins, highlight, badge and Reset, (i) the Preg tooltip, then a QSS run through
headless_solve on Sturn with physics corrected. The car is synthetic (the builder of tests/test_cs_gui.py,
copied). Every GUI state lives in a temp home.

    python tests/test_pt_group.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import atexit
import contextlib
import copy
import io
import json
import math
import os
import shutil
import tempfile
import types
from unittest import mock

import numpy as np

from functions import carfile

TMP = types.SimpleNamespace(name=tempfile.mkdtemp(prefix="pt_group_"))
atexit.register(shutil.rmtree, TMP.name, True)
HOME = os.path.join(TMP.name, "home")
os.makedirs(os.path.join(HOME, ".config"))
os.environ["HOME"] = HOME
os.environ["XDG_CONFIG_HOME"] = os.path.join(HOME, ".config")
os.environ["FULLMODELSIM_DOCUMENTS"] = os.path.join(HOME, "Documents")   # Windows: Qt does not read HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PRESETS = os.path.join(TMP.name, "presets")          # the user presets folder
os.makedirs(PRESETS)
OUT = os.path.join(TMP.name, "out")                  # the output folder of every run

PASS = 0


def ok(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASS += 1


from PySide6.QtWidgets import QApplication
import app.mainwindow as mw
from app.mainwindow import MainWindow
from app import paths, results
import headless_solve
from userOpts import setup_knobs, knob_tag

APP = QApplication.instance() or QApplication([])
paths.settings().clear()
paths.settings().sync()
paths.user_presets_dir = lambda: PRESETS              # never the real Documents

PT_PHYSICS_TIP = "The powertrain keys need physics = corrected."
PREG_TIP = ("Off: the motor does not regenerate. On: the motor regenerates up to this power. "
            "0 kW has the same effect as Off.")
PT_ROWS_KEYS = ("gear", "Tmax", "Pmax", "Preg", "Ppack")
ONE_PLACE = "[error] Set the powertrain keys in one place: the Powertrain group or More options.\n"


def new_window(physics="corrected"):
    h = MainWindow()
    h.opt_combos["physics"].setCurrentText(physics)
    h.output_dir.setText(OUT)
    return h


def capture(h):
    """Press Run with the solve runner replaced by a list. The cfg dict, or None when the run did not start."""
    started = []
    real = h._runner.start
    h._runner.start = started.append
    try:
        h._on_run()
    finally:
        h._runner.start = real
    if not started:
        return None
    with open(started[0]) as fh:
        cfg = json.load(fh)
    h._remove_cfg()
    return cfg


def knob(vp):
    """The setup tag of a run with these vp_overrides (corrected physics, Sturn, every other default)."""
    return knob_tag(setup_knobs({"circuit": "Sturn", "physics": "corrected", "vp_overrides": vp}))


def save_to(path, h):
    with mock.patch.object(mw.QFileDialog, "getSaveFileName", return_value=(path, "")):
        h._save_preset()


def load_from(path, h):
    with mock.patch.object(mw.QFileDialog, "getOpenFileName", return_value=(path, "")):
        h._load_preset()


# ---- synthetic car builder, copied from tests/test_cs_gui.py ----
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair. An over or omit cid is a file cid.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}
G = 9.81
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


def write_car(name, arb_front=None, mirror=False, toe=0.0, toes=None, over=None):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(arb_front, mirror, toe, toes, over), path)
    return path



# ---------------------------------------------------------------- (a) defaults
print("(a) defaults equal the model")
from Powertrain import Powertrain
from vehParams import vehParams
from functions.context import Ctx

ctx = Ctx()
Powertrain(ctx)
vehParams(ctx)
ok("Powertrain: Tmax 602 N m per motor", ctx.pt.Tmax == 602.0)
ok("Powertrain: Pmax 456 kW per motor", ctx.pt.Pmax == 456000.0)
ok("vp.gear after Powertrain and vehParams is the derived value",
   math.isclose(ctx.vp.gear, 7.2519716776170124, rel_tol=1e-12))

h = new_window()
ok("GUI gear shows the model gear (4 decimals)", abs(h.pt_spins["gear"].value() - 7.2519716776170124) < 5e-5)
ok("GUI Tmax 602 and Pmax 456 kW", h.pt_spins["Tmax"].value() == 602.0 and h.pt_spins["Pmax"].value() == 456.0)
ok("the limit boxes start off", not h.pt_checks["Preg"].isChecked() and not h.pt_checks["Ppack"].isChecked())
cfg = capture(h)
ok("an untouched group sends no vp_overrides", cfg is not None and cfg["vp_overrides"] == {})
ok("an untouched group keeps the setup tag empty (physics corrected)", h.collect_runconfig().setup_tag() == "")

# ---------------------------------------------------------------- (b) Tmax 500
print("(b) Tmax 500 sends one key, and More options gives the same tag")
h = new_window()
h.pt_spins["Tmax"].setValue(500.0)
cfg = capture(h)
ok("Tmax 500 sends {Tmax: 500.0} and nothing else", cfg["vp_overrides"] == {"Tmax": 500.0})
tag_group = h.collect_runconfig().setup_tag()
ok("the group edit gives a setup tag", tag_group != "" and tag_group == knob({"Tmax": 500.0}))
h.pt_spins["Tmax"].setValue(602.0)
h.more_json.setPlainText('{"pt_overrides": {"Tmax": 500.0}}')
ok("More options pt_overrides gives the same tag as the group", h.collect_runconfig().setup_tag() == tag_group)
ok("More options sends the same vp_overrides", capture(h)["vp_overrides"] == {"Tmax": 500.0})
h.more_json.setPlainText("")
ok("a value set back to its default sends nothing and tags nothing",
   capture(h)["vp_overrides"] == {} and h.collect_runconfig().setup_tag() == "")

# ---------------------------------------------------------------- (c) exact W
print("(c) kW rows send exact W")
for key, kw, want in (("Pmax", 16.1, 16100.0), ("Preg", 16.1, 16100.0), ("Ppack", 16.1, 16100.0),
                      ("Pmax", 2.01, 2010.0), ("Preg", 2.01, 2010.0), ("Ppack", 2.01, 2010.0)):
    h = new_window()
    if key in h.pt_checks:
        h.pt_checks[key].setChecked(True)
    h.pt_spins[key].setValue(kw)
    cfg = capture(h)
    ok(f"{key} {kw} kW sends {want} W exactly", cfg is not None and cfg["vp_overrides"] == {key: want})
    ok(f"{key} {kw} kW: the setup tag is the tag of {want} W",
       h.collect_runconfig().setup_tag() == knob({key: want}))
h = new_window()
h.pt_checks["Preg"].setChecked(True)
h.pt_spins["Preg"].setValue(0.0)
ok("Preg on at 0 kW sends 0.0", capture(h)["vp_overrides"] == {"Preg": 0.0})
h = new_window()
h.pt_spins["Ppack"].setValue(300.0)
ok("Ppack spin set with its box off sends nothing", capture(h)["vp_overrides"] == {})

# ---------------------------------------------------------------- (d) physics
print("(d) physics matlab disables the group, corrected sends the edits again")
h = new_window("matlab")
h.pt_spins["Tmax"].setValue(500.0)
h.pt_checks["Preg"].setChecked(True)
h.pt_spins["Preg"].setValue(50.0)
ok("matlab disables the section", not h._pt_sec.isEnabled())
ok("matlab section tooltip gives the physics reason", h._pt_sec.toolTip() == PT_PHYSICS_TIP)
for k in PT_ROWS_KEYS:
    ok(f"matlab {k}: spin and label tooltips end with the physics reason",
       h.pt_spins[k].toolTip().endswith(PT_PHYSICS_TIP) and h.pt_labels[k].toolTip() == h.pt_spins[k].toolTip())
cfg = capture(h)
ok("matlab sends no vp_overrides, even after edits", cfg is not None and cfg["vp_overrides"] == {})
ok("matlab: pt_overrides is None", h.pt_overrides() is None)
h.opt_combos["physics"].setCurrentText("corrected")
ok("corrected enables the section", h._pt_sec.isEnabled() and h._pt_sec.toolTip() == "")
ok("corrected removes the physics reason from the row tooltips",
   all(PT_PHYSICS_TIP not in h.pt_spins[k].toolTip() and PT_PHYSICS_TIP not in h.pt_labels[k].toolTip()
       for k in PT_ROWS_KEYS))
ok("corrected sends the edits again", capture(h)["vp_overrides"] == {"Tmax": 500.0, "Preg": 50000.0})

# ---------------------------------------------------------------- (e) car mode
print("(e) car mode hides the group, the 4EM car shows it again")
CAR = write_car("pt_car.car")                           # the synthetic car of tests/test_cs_gui.py
h = new_window()
h.pt_spins["Tmax"].setValue(500.0)
h._select_car(CAR)
ok("car mode hides the section", h._pt_sec.isHidden())
ok("car mode: pt_overrides is None", h.pt_overrides() is None)
cfg = capture(h)
ok("car mode sends no powertrain key", cfg is not None and cfg["vp_overrides"] == {})
h.car.setCurrentIndex(0)
ok("the 4EM car shows the section again", not h._pt_sec.isHidden())
ok("the 4EM car sends the group edit again", capture(h)["vp_overrides"] == {"Tmax": 500.0})

# ---------------------------------------------------------------- (f) one place: group or More options
print("(f) the group and More options pt_overrides cannot both be set")
h = new_window()
h.more_json.setPlainText('{"pt_overrides": {"Tmax": 450.0}}')
h.pt_spins["Tmax"].setValue(400.0)
h.log.clear()
ok("both set: Run starts nothing", capture(h) is None)
ok("both set: the one-place error is logged", ONE_PLACE in h.log.toPlainText())
h.pt_spins["Tmax"].setValue(602.0)
h.log.clear()
cfg = capture(h)
ok("More options alone (group at default) runs with its value",
   cfg is not None and cfg["vp_overrides"] == {"Tmax": 450.0} and "[error]" not in h.log.toPlainText())

# ---------------------------------------------------------------- (g) presets and the Expert path
print("(g) presets: changed values only, full snapshot on load, unknown keys ignored, Expert runs once")
h = new_window()
default_preset = os.path.join(PRESETS, "default.json")
save_to(default_preset, h)
with open(default_preset) as fh:
    d0 = json.load(fh)
ok("a default group saves no pt_overrides key", "pt_overrides" not in d0)
PF = os.path.join(PRESETS, "group.json")
h.pt_spins["Tmax"].setValue(450.0)
h.pt_checks["Preg"].setChecked(True)
h.pt_spins["Preg"].setValue(16.1)
h.pt_spins["gear"].setValue(9.0)
save_to(PF, h)
with open(PF) as fh:
    d1 = json.load(fh)
ok("the preset stores the changed and switched-on values under pt_overrides",
   d1.get("pt_overrides") == {"gear": 9.0, "Tmax": 450.0, "Preg": 16100.0})
h.pt_spins["Tmax"].setValue(333.0)
h.pt_checks["Preg"].setChecked(False)
h.pt_spins["gear"].setValue(7.0)
load_from(PF, h)
ok("Load restores the group", h.pt_overrides() == {"gear": 9.0, "Tmax": 450.0, "Preg": 16100.0})
ok("Load restores the gear of the preset", abs(h.pt_spins["gear"].value() - 9.0) < 1e-9)
h.pt_spins["Tmax"].setValue(333.0)
load_from(default_preset, h)
ok("Load of a default-group preset gives the default group, gear included",
   h.pt_overrides() is None and abs(h.pt_spins["gear"].value() - 7.252) < 1e-9)
old = os.path.join(PRESETS, "old.json")
with open(old, "w") as fh:
    json.dump({k: v for k, v in d1.items() if k != "pt_overrides"}, fh)
h.pt_spins["Tmax"].setValue(333.0)
h.pt_checks["Ppack"].setChecked(True)
load_from(old, h)
ok("an old preset without the key gives the default group",
   h.pt_overrides() is None and not h.pt_checks["Ppack"].isChecked())
unk = os.path.join(PRESETS, "unknown.json")
with open(unk, "w") as fh:
    json.dump({**d0, "pt_overrides": {"Foo": 1.0, "Tmax": 500.0}}, fh)
h.log.clear()
load_from(unk, h)
ok("an unknown key under pt_overrides counts as ignored", "ignored 1" in h.log.toPlainText())
ok("the known key of that preset still applies", h.pt_overrides() == {"Tmax": 500.0})

h = new_window()
h.pt_spins["Tmax"].setValue(450.0)
h.pt_checks["Preg"].setChecked(True)
h.pt_spins["Preg"].setValue(16.1)
EX = os.path.join(PRESETS, "expert.json")
save_to(EX, h)                                          # 130 vp keys (defaults) and pt_overrides
h2 = new_window()
h2.expert.setText(EX)
h2._load_expert_into_widgets(EX)
ok("an Expert preset sets the group", h2.pt_overrides() == {"Tmax": 450.0, "Preg": 16100.0})
h2.log.clear()
cfg = capture(h2)
ok("an Expert preset with group edits runs once with the group values, no [error]",
   cfg is not None and cfg["vp_overrides"] == {"Tmax": 450.0, "Preg": 16100.0}
   and "[error]" not in h2.log.toPlainText())
ok("its setup tag is the tag of those values",
   h2.collect_runconfig().setup_tag() == knob({"Tmax": 450.0, "Preg": 16100.0}))
part = os.path.join(PRESETS, "partial.json")
with open(part, "w") as fh:
    json.dump({"mb": 2000.0}, fh)
h2.expert.setText(part)
h2._load_expert_into_widgets(part)
ok("a partial Expert file without the key keeps the group", h2.pt_overrides() == {"Tmax": 450.0, "Preg": 16100.0})

# ---------------------------------------------------------------- (h) enabled spins, highlight, badge, Reset
print("(h) limit spins follow their boxes, changed rows are highlighted and counted, Reset clears")
h = new_window()
ok("Preg and Ppack spins are disabled with the boxes off",
   not h.pt_spins["Preg"].isEnabled() and not h.pt_spins["Ppack"].isEnabled())
ok("gear, Tmax and Pmax spins are enabled", all(h.pt_spins[k].isEnabled() for k in ("gear", "Tmax", "Pmax")))
h.pt_checks["Preg"].setChecked(True)
ok("the Preg spin is enabled with its box on, Ppack stays off",
   h.pt_spins["Preg"].isEnabled() and not h.pt_spins["Ppack"].isEnabled())
h.pt_checks["Preg"].setChecked(False)
ok("the Preg spin is disabled again when its box goes off", not h.pt_spins["Preg"].isEnabled())
h.pt_spins["Tmax"].setValue(500.0)
ok("a changed Tmax spin is highlighted", "b30000" in h.pt_spins["Tmax"].styleSheet())
ok("an unchanged Pmax spin is not highlighted", h.pt_spins["Pmax"].styleSheet() == "")
ok("the badge counts one changed row", h._pt_sec._changed == 1 and "(1 changed)" in h._pt_sec._btn.text())
h.pt_checks["Ppack"].setChecked(True)
ok("a switched-on limit row is highlighted and counted",
   "b30000" in h.pt_spins["Ppack"].styleSheet() and h._pt_sec._changed == 2)
h._pt_sec._reset.click()
ok("the section Reset clears the group",
   h.pt_overrides() is None and h._pt_sec._changed == 0 and not h.pt_checks["Ppack"].isChecked()
   and h.pt_spins["Tmax"].value() == 602.0 and h.pt_spins["Tmax"].styleSheet() == "")
h.pt_spins["Pmax"].setValue(228.0)
h._reset_all_params()
ok("Reset all clears the group", h.pt_overrides() is None and h.pt_spins["Pmax"].value() == 456.0
   and h._pt_sec._changed == 0)

# ---------------------------------------------------------------- (i) tooltips
print("(i) the Preg tooltip")
h = new_window()
ok("the Preg spin and label carry the ruled text", h.pt_spins["Preg"].toolTip() == PREG_TIP
   and h.pt_labels["Preg"].toolTip() == PREG_TIP)

# ---------------------------------------------------------------- QSS through headless_solve (Sturn, corrected)
print("qss runs through headless_solve on Sturn, physics corrected (no IPOPT)")


def qss_lap(label, edit):
    h = new_window()
    h.circuit.setCurrentText("Sturn")
    h.opt_combos["mode"].setCurrentText("qss")
    edit(h)
    started = []
    real = h._runner.start
    h._runner.start = started.append
    try:
        h._on_run()
    finally:
        h._runner.start = real
    ok(f"{label}: the GUI starts one cfg", len(started) == 1)
    cfg_path = started[0]
    with open(cfg_path) as fh:
        vo = json.load(fh)["vp_overrides"]
    rc = h.collect_runconfig()
    with contextlib.redirect_stdout(io.StringIO()):
        code = headless_solve.main([cfg_path])
    os.remove(cfg_path)
    mat, _ = results.result_paths(rc)
    ok(f"{label}: headless_solve exits 0 and the result file exists", code == 0 and os.path.isfile(mat))
    lap = results.parse_summary(mat)["lap_time_s"]
    ok(f"{label}: the lap is finite", lap == lap and lap > 0.0)
    print(f"      {label}: vp_overrides {vo}, lap {lap:.5f} s")
    return lap


def no_edit(h):
    pass


def ppack300(h):
    h.pt_checks["Ppack"].setChecked(True)
    h.pt_spins["Ppack"].setValue(300.0)


def tmax400(h):
    h.pt_spins["Tmax"].setValue(400.0)


def preg50(h):
    h.pt_checks["Preg"].setChecked(True)
    h.pt_spins["Preg"].setValue(50.0)


lap_base = qss_lap("default group", no_edit)
lap_pack = qss_lap("Ppack 300 kW", ppack300)
lap_tmax = qss_lap("Tmax 400", tmax400)
lap_preg = qss_lap("Preg 50 kW", preg50)
ok("Ppack 300 kW is slower than no Ppack", lap_pack > lap_base)
ok("Tmax 400 is slower than the default Tmax", lap_tmax > lap_base)

print(f"ALL OK: {PASS} checks")
