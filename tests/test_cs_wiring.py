"""Car source wiring (ChassisSim phase 2, task 4): tests/test_cs_wiring.py.

A synthetic car file and a synthetic ChassisSim install (copied from tests/test_vehmodel_cs.py) go through
MLTP(car=..., circuit='chassissim:...') with a fake IPOPT (no solve). Checks: the refusals of a car, the stems and
the knob car_file, the saved car result (name, data.car), the warm start (full+duals from the car result, cold from
a 4EM result), the ladder seeds (seed_m23, quasi_static_states 'car'), limit_prices (NaN motor prices) and
MLTP_screen (_qss stem). The GP3 2013 car on Eastern Creek runs when CHASSISSIM_DIR names the install (else SKIP).
Nothing is written into the vendor folders; scratch files go to a temp dir.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_wiring.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_wiring.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import copy
import io
import os
import tempfile
import warnings
from pathlib import Path

import casadi as ca
import numpy as np

import functions.transcription as T
from functions import carfile
from functions import ladder as LD
from functions.context import Ctx
from functions.importfile import load_solution, result_stem
from functions.sensitivity import limit_prices
from functions.transcription import discretise
from MLTP import MLTP
from MLTP_initial import MLTP_initial
from MLTP_paramOptim import optimise_design
from MLTP_screen import MLTP_screen
from setup_sweep import setup_sweep
from userOpts import knob_tag, setup_knobs, userOpts
from vehModel import vehModel

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


def refusal(fn, *a, **kw):
    """(True, message) when fn raises ValueError, else (False, '')."""
    try:
        quiet(fn, *a, **kw)
    except ValueError as e:
        return True, str(e)
    return False, ""


def captured(fn, *a, **kw):
    """(return value, stdout text) of fn, warnings swallowed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fn(*a, **kw)
    return out, buf.getvalue()


def build_loud(car_path, **kw):
    """build() without its own stdout capture, so captured() sees MLTP's messages (same defaults)."""
    opts = dict(car=f"chassissim:{car_path}", circuit=f"chassissim:{CIRC}", chassissim_dir=INSTALL,
                physics="corrected", save=False, plot=False, linear_solver="mumps")
    opts.update(kw)
    return MLTP(**opts)


RES = os.path.join(TMP.name, "results")
QSS = os.path.join(TMP.name, "qss")
BASE = write_car("wiring_base.car")
CHASSIS = f"chassissim:{CIRC}"
CAR = f"chassissim:{BASE}"

print("[cs_wiring] 1. refusals")
for label, kw, must in [
    ("start_state='rolling'", dict(start_state="rolling"), "does not accept start_state='rolling'"),
    ("state_scaling='typical'", dict(state_scaling="typical"), "does not accept state_scaling='typical'"),
    ("atd_tbr=2.0", dict(atd_tbr=2.0), "does not accept atd_tbr=2.0"),
    ("between_knots='linear'", dict(between_knots="linear"), "does not accept between_knots='linear'"),
    ("collocation='radau'", dict(collocation="radau"), "does not accept collocation='radau'"),
    ("physics='matlab'", dict(physics="matlab"), "needs physics='corrected'"),
]:
    hit, msg = refusal(build, BASE, **kw)
    ok(f"car with {label} raises ValueError", hit and must in msg)

ci, _ = captured(MLTP_initial, circuit="Sturn", save=True, results_dir=RES)      # 4EM 7-state init (fake IPOPT)
hit, msg = refusal(build, BASE, warm_start=ci.out_path)
ok("car with a 7-state init warm_start raises ValueError", hit and "cannot start from a 7-state init" in msg)
hit, msg = refusal(MLTP_initial, circuit=CHASSIS, chassissim_dir=INSTALL, car=CAR, physics="corrected", save=False)
ok("MLTP_initial with car= raises ValueError", hit and "A ChassisSim car has none" in msg)
hit, msg = refusal(optimise_design, [("brkB", 0.0, 1.0)], "wiring", circuit=CHASSIS, chassissim_dir=INSTALL,
                   car=CAR, physics="corrected", save=False)
ok("optimise_design with car= raises ValueError", hit and "does not support a ChassisSim car" in msg)
hit, msg = refusal(setup_sweep, [("brkB", 0.0, 1.0)], n_samples=4, circuit=CHASSIS, car=CAR,
                   chassissim_dir=INSTALL, physics="corrected", results_root=os.path.join(TMP.name, "sweeps"),
                   confirm=False, plot=False, verbose=False)
ok("setup_sweep with a car base raises ValueError", hit and "does not support a ChassisSim car" in msg)

print("[cs_wiring] 2. stems and knobs")
CAR_ID = "/home/x/Cars/def_GP3 2013.car#0a1b2c3d"
ok("4EM Sturn default stem unchanged", result_stem("Sturn", "Static_ATDOn_EM4Off") == "Sturn_Static_ATDOn_EM4Off")
ok("4EM stem with car='4em' and physics corrected unchanged",
   result_stem("Sturn", "Static_ATDOn_EM4Off", physics="corrected", car="4em")
   == "Sturn_Static_ATDOn_EM4Off_physCorrected")
ok("car stem: _car_<slug> after the tyre set, before _physCorrected",
   result_stem("chassissim:Eastern Creek", "Static_ATDOff_EM4Off", tyre_set="CopyB", physics="corrected", car=CAR_ID)
   == "cs_Eastern_Creek_Static_ATDOff_EM4Off_CopyB_car_def_GP3_2013_physCorrected")
ok("no car: no car_file knob", "car_file" not in setup_knobs({"car": None}))

A = os.path.join(TMP.name, "knob_a.car")
carfile.write(build_records(), A)


def car_knob(p):
    return setup_knobs({"car": f"chassissim:{p}"})["car_file"]


def sha_of(knob):
    return knob.split("#")[1]


knob_a = car_knob(A)
tag_a = knob_tag(setup_knobs({"car": f"chassissim:{A}"}))
os.remove(A)                                             # carfile.write creates the file (exclusive)
carfile.write(build_records(arb_front=0.0), A)           # same path, other bytes
knob_a2 = car_knob(A)
tag_a2 = knob_tag(setup_knobs({"car": f"chassissim:{A}"}))
ok(f"car tag changes when the car file bytes change (sha8 {sha_of(knob_a)} -> {sha_of(knob_a2)})",
   sha_of(knob_a) != sha_of(knob_a2) and bool(tag_a) and bool(tag_a2) and tag_a != tag_a2)

MOVED = os.path.join(TMP.name, "moved", "knob_a.car")
os.makedirs(os.path.dirname(MOVED), exist_ok=True)
carfile.write(build_records(arb_front=0.0), MOVED)       # same bytes, another folder
knob_m = car_knob(MOVED)
tag_m = knob_tag(setup_knobs({"car": f"chassissim:{MOVED}"}))
ok("same bytes in another folder: sha8 equal, tag differs (car_file holds the absolute path, so a move "
   "changes the tag)", sha_of(knob_m) == sha_of(knob_a2) and knob_m != knob_a2 and tag_m != tag_a2)

print("[cs_wiring] 3. saved car result and warm starts")
c_car = quiet(build, BASE, save=True, results_dir=RES)
path_car = c_car.out_path
ok(f"car result stem holds '_car_' ({os.path.basename(path_car)})",
   "_car_" in os.path.basename(path_car) and os.path.isfile(path_car))
ok("data.car holds the car id (path#sha8) in the saved file", str(load_solution(path_car).car) == c_car.car)
c_warm = quiet(build, BASE, warm_start=path_car)
ok(f"warm start from the car result: full+duals (got {c_warm.data.nlp['warm_start']})",
   c_warm.data.nlp["warm_start"] == "full+duals")

c_4em = quiet(MLTP, circuit="Sturn", save=True, results_dir=RES, plot=False, linear_solver="mumps",
              physics="corrected")
c_x, txt = captured(build_loud, BASE, warm_start=c_4em.out_path)
ok(f"warm start from the 4EM result: cold (got {c_x.data.nlp['warm_start']}) and the reason is printed",
   c_x.data.nlp["warm_start"] == "cold" and "do not seed each other" in txt)

print("[cs_wiring] 4. ladder seeds")
disc = discretise(c_car.track, c_car.OPT_ds, c_car.OPT_d, mesh=c_car.mesh, mesh_opts=c_car.mesh_opts)
prof = LD.qss_profile(c_car)
g = LD.seed_m23(c_car, c_car.m23, disc, prof, c_car.input_keys)
ok(f"seed_m23 for a car: input keys {list(c_car.input_keys)}, u0 has 3 rows", list(c_car.input_keys) ==
   ["T_drive", "T_brake", "delta"] and g["u0"].shape[0] == 3)
ok("seed_m23 car seeds (x0, u0, xc0) are finite", all(np.all(np.isfinite(g[k])) for k in ("x0", "u0", "xc0")))
ok(f"seed_m23 car: T_drive >= 0 and T_brake <= 0 (min {g['u0'][0].min():.3g}, max {g['u0'][1].max():.3g})",
   bool(np.all(g["u0"][0] >= -1e-12)) and bool(np.all(g["u0"][1] <= 1e-12)))
qs = LD.quasi_static_states(c_car.vp, [20.0, 40.0], "car")
zt = np.array([getattr(c_car.vp, f"W{c}0") / c_car.vp.kt for c in CORNERS])
ok("quasi_static_states 'car': finite, zt rows = W0 / kt",
   bool(np.all(np.isfinite(qs))) and np.allclose(qs[14:18, 0], zt, rtol=1e-12))

print("[cs_wiring] 5. limit_prices on the car result")
lp = limit_prices(c_car.data)
ok(f"limit_prices: Tmax and Pmax NaN for the car, no raise (Tmax {lp['Tmax']}, Pmax {lp['Pmax']})",
   bool(np.isnan(lp["Tmax"])) and bool(np.isnan(lp["Pmax"])))

print("[cs_wiring] 6. MLTP_screen with car=")
q = quiet(MLTP_screen, circuit=CHASSIS, car=CAR, chassissim_dir=INSTALL, physics="corrected", save=True,
          results_dir=QSS, verbose=False)
name_q = os.path.basename(q.out_path)
ok(f"QSS stem holds '_car_' and ends '_qss.mat' ({name_q})", "_car_" in name_q and name_q.endswith("_qss.mat"))


# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs_dir):
    print("[cs_wiring] vendor GP3 2013 Eastern Creek")
    car_path = Path(cs_dir).parent / "Models" / "GP3 2013" / "Eastern Creek, AU" / "def_GP3 2013.car"
    ok("GP3 2013 Eastern Creek car file found", car_path.is_file())
    c = build(str(car_path), circuit="chassissim:Eastern Creek", chassissim_dir=cs_dir, save=True,
              results_dir=os.path.join(TMP.name, "gp3"))
    name = os.path.basename(c.out_path)
    ok(f"GP3 result stem holds '_car_def_GP3_2013' ({name})", "_car_def_GP3_2013" in name)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
print(f"{PASS} checks passed")

