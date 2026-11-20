"""Hybrid layer in the NLP (ChassisSim phase 2c, task P4): tests/test_cs_hybrid.py.

A synthetic car (literal cids and tags, written with carfile.write into a temp dir) with the hybrid records
set goes through MLTP with a fake IPOPT (no solve). The checks are:
  a. the gates of chassissim.hybrid_gates and the regen gate of cid 590 (no brake harvest without it),
  b. the NLP size: the new inputs, the rows, the growth of n_g, the two lap rows and their bounds,
  c. hand values of the deploy power, the brake harvest (plain, clamped, front share) and the paddle harvest,
  d. the net-force harvest: a drive that overlaps the brake or the paddle harvests almost nothing,
  e. the lap rows equal a numpy quadrature over the collocation points,
  f. the ValueErrors of the value checks (and the values that load),
  g. the skip rule: cid 177 = 0 gives the base graph and no hybrid rows,
  h. warm starts across the hybrid switch,
  i. sensitivity.nlp_index and limit_prices on a hybrid result,
  j. data['hybrid'] and the channels P_dep and P_harv.
  k. the hybrid layer in the QSS (MLTP_screen, task P5): the skip rule, the harvest against a scalar loop, the store,
     the mode 0 / mode 1 runs, one deploy step by hand (e = s and e != s), the cid edges, the ValueErrors, the NLP seed,
     data['hybrid'] and its .mat round trip, and the 4EM Sturn screen with casadi blocked.
The vendor section (last) runs only when CHASSISSIM_DIR names the ChassisSim install folder (SKIP otherwise).
It checks that every shipped car has cid 177 off, and it builds a hybrid copy of the GP3 car in a temp dir.
Nothing is written into the vendor folders. Results go to a temp dir.

    python tests/test_cs_hybrid.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_hybrid.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import glob
import hashlib
import io
import itertools
import math
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.io import loadmat

import functions.ggv as G
import functions.transcription as T
from functions import carfile, chassissim as C
from functions.context import Ctx
from functions.sensitivity import limit_prices, nlp_index
from MLTP import MLTP
from MLTP_screen import MLTP_screen
from userOpts import userOpts
from vehModel import vehModel
from vehModel_cs import regen_share

PASS = 0
TMP = tempfile.TemporaryDirectory()
RES = os.path.join(TMP.name, "results")
os.makedirs(RES)
COUNT = itertools.count()
CIRC = "Synth Ring"
CARS_DIR = os.path.join(TMP.name, "cars")
os.makedirs(CARS_DIR)
E_S = 1e5                                             # J, the scale of the energy rows (vehModel_cs.E_S)
CLAMP, S_DEP = 60000.0, 60000.0                       # regen clamp and deploy scale of the synthetic hybrid car [W]
KR, KD = 0.8, 0.8                                     # regen scale (182) and deploy scale after the map (183)
HEAD = ["P_dep", "P_harv", "sf", "v_r", "om", "T_drive", "T_brake"]
FLAG_CIDS = {177, 178, 464, 466, 535, 573, 589, 590, 870}
CAP = {}                                              # the last fake solve: nlp, bounds and start point
SEED = {}                                             # optional start point: {"nu": n, "u": f(k, N) -> {input: value}}
# Hybrid records of the synthetic car. The regen gate 590 and the deploy blend 464 / 465 are on.
HYB = {177: 1, 179: CLAMP, 180: S_DEP, 181: 400000.0, 182: KR, 183: KD, 836: 0.0, 535: 1, 590: 1, 464: 1, 465: 0.3}
PAD = {573: 1, 589: 1}                                # paddle regen (second control)
FRONT = {870: 1, 871: 0.55}                           # regen front share (the axle split of the brake)
GL_TAU, GL_W = np.polynomial.legendre.leggauss(3)     # Gauss-Legendre weights on [0, 1] (the B of the Legendre scheme)
GL_B = GL_W / 2.0


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
    return abs(a - b) / max(abs(b), 1e-300)


def seeded(nlp, x):
    """Edit the start point of a fake solve: the inputs of SEED at every knot (scaled values)."""
    nx, nu, d = 23, SEED["nu"], 3
    n_w = x.size
    N = next(n for n in range(1, 2000) if (nx + nu) * (n + 1) + nx * n * d == n_w)
    for k in range(N + 1):
        for j, v in SEED["u"](k, N).items():
            x[nx * (N + 1) + j + nu * k] = v
    return x


def fake(ca_, nlp, opts, info=None):
    """Stand-in for functions.transcription._make_solver: returns the start point, no iterations."""
    def solver(**k):
        x = np.array(k["x0"], float).copy()
        if SEED:
            x = seeded(nlp, x)
        CAP.update(nlp=nlp, k=dict(k), x=x)
        return {"x": np.reshape(x, np.shape(k["x0"])), "f": 0.0,
                "lam_g": np.zeros(nlp["g"].numel()), "lam_x": np.zeros(nlp["x"].numel())}
    solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
    if info is not None:
        info["linear_solver"] = "fake"
    return solver


T._make_solver = fake


# ---- synthetic ChassisSim install: one circuit, 600 m, one lap of turning ----------------------------
INI = ("[Total Circ prop]\r\nTotal circuits = 1\r\n\r\n"
       f"[Circuit0]\r\nCircuit name = {CIRC}\r\nCircuit file name = tr_synth.txt\r\nBase Grip factor = 1.0\r\n"
       "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n")


def write_text(path, text):
    data = text if isinstance(text, bytes) else text.encode("latin-1")
    with open(path, "wb") as f:
        f.write(data)


def build_install(root):
    s = np.linspace(0.0, 600.0, 121)
    k = (2.0 * np.pi / 600.0) * (1.0 + 0.5 * np.sin(2.0 * np.pi * s / 600.0))
    os.makedirs(os.path.join(root, "Circuits"))
    write_text(os.path.join(root, "Circ_prop.ini"), INI)
    rows = [f"{float(a)!r}   {float(b)!r}" for a, b in zip(k, s)]
    write_text(os.path.join(root, "Circuits", "tr_synth.txt"), "\r\n".join(rows) + "\r\n")
    return root


INST = build_install(os.path.join(TMP.name, "ChassisSim_v3_41"))


# ---- synthetic car: literal cids and tags (as in tests/test_cs_power.py) -----------------------------
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
    """The synthetic car without the end record and without any hybrid record. Left and right cids are swapped."""
    R = []

    def add(tag, cid, v):
        R.append((tag, FILE_CID.get(cid, cid), v))
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
    for cid, ((nx, nz), _) in MAPS.items():
        add(9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid)))
    for cid in range(31, 35):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    add(12, 35, (1, arb(6, 35), np.array([])))
    for cid, rows in ((49, HP_LEFT), (50, HP_RIGHT), (51, HP_LEFT), (52, HP_RIGHT)):
        add(10, cid, (0, np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])))
    for i, c in enumerate(("fl", "fr", "rl", "rr")):
        add(*tyre_record(i, c))
    return R


def car_path(cids=None, maps=None, name=None):
    """Write the synthetic car with the hybrid records (cid -> value) and the maps (cid -> (x, z, data)).
    The records of the same cid are replaced, not repeated. Return the path."""
    cids, maps = dict(cids or {}), dict(maps or {})
    body = [r for r in build_records() if r[1] not in cids and r[1] not in maps]
    extra = [(6 if c in FLAG_CIDS else 1, c, v) for c, v in cids.items()] + [(9, c, v) for c, v in maps.items()]
    path = os.path.join(CARS_DIR, name or f"car_{next(COUNT)}.car")
    carfile.write(body + extra + [(carfile.END_TAG, carfile.END_CID, None)], path)
    return path


def const_map(v):
    """A 2 x 2 map with one value everywhere: x in rpm (engine speed), z throttle / 100, data = factor."""
    return (np.array([1000.0, 9000.0]), np.array([0.0, 1.0]), np.full(4, float(v)))


def build(path, **kw):
    """MLTP on the car with a fake IPOPT (no solve). Returns ctx (ctx.m23, ctx.cs, ctx.data) with ctx.cap, the
    fake solve record, and ctx.car_path."""
    kw.setdefault("save", False)
    c = quiet(MLTP, car=f"chassissim:{path}", circuit=f"chassissim:{CIRC}", chassissim_dir=INST, physics="corrected",
              plot=False, linear_solver="mumps", results_dir=RES, **kw)
    c.cap, c.car_path = dict(CAP), path
    c.data = dict(vars(c.data))                       # the result as a dict, as data.nlp is
    return c


def msg_of(path):
    """'silent' or the ValueError message of load_car, power_refusals and aero_refusals."""
    try:
        cs = C.load_car(path, INST)
        C.power_refusals(cs)
        C.aero_refusals(cs)
        return "silent"
    except ValueError as e:
        return str(e)


def graph_hash(path):
    """Hash of (dx, sf) of the 23-state model of the car."""
    ctx = Ctx()
    quiet(userOpts, ctx, circuit=f"chassissim:{CIRC}", car=f"chassissim:{path}", chassissim_dir=INST,
          physics="corrected", linear_solver="mumps")
    quiet(vehModel, ctx)
    m = ctx.m23
    fn = ca.Function("d", [m.x, m.u, m.pv], [m.dx, m.sf])
    return hashlib.sha256(fn.serialize().encode()).hexdigest()


# ---- evaluation of the hybrid outputs at a state -------------------------------------------------------
def make_ev(m):
    """Function of the scaled state, the scaled inputs and kappa: the hybrid outputs, the power rows and the lap rows."""
    ex = dict(m.veh_extra)
    outs = [ex["P_dep"], ex["P_harv"], m.sf, m.v_r, m.om_engine, m.T_drive, m.T_brake]
    outs += [e for _, e, _, _ in getattr(m, "hyb_rows", [])]
    outs += [e for e, _, _ in getattr(m, "lap_rows", [])]
    return ca.Function("ev", [m.x, m.u, m.pv], [ca.SX(o) for o in outs])


def at(c, xp, up):
    """Named outputs at the physical state xp (rad/s, m/s) and the scaled inputs up."""
    m = c.m23
    f = make_ev(m)
    xn = np.reshape(np.asarray(xp, float) / np.asarray(m.x_s, float), (-1, 1))
    out = [float(v) for v in f(xn, np.reshape(np.asarray(up, float), (-1, 1)), 0.0)]
    names = HEAD + [n for n, *_ in getattr(m, "hyb_rows", [])] + [f"lap{i}" for i in range(len(getattr(m, "lap_rows", [])))]
    return dict(zip(names, out))


def state(c, vx, spin=0.03):
    """Straight state at vx: the front wheels roll free, the rear wheels spin by spin (v_r = (1 + spin) vx)."""
    x = np.zeros(23)
    x[0] = vx
    x[5] = x[6] = vx / c.cs.tyre.fl["radius"]
    x[7] = x[8] = (1.0 + spin) * vx / c.cs.tyre.rl["radius"]
    return x


def quad(data, key):
    """Lap integral of data['vehicle_full'][key] * sf over the collocation points of the Legendre grid (numpy).
    Each interval gives dsk * sum_j B_j (value at point j), as the objective does. The knots carry no weight."""
    d, N = int(data["OPT_d"]), int(data["N"])
    s_full = np.asarray(data["s_full"], float).ravel()
    dsk = np.diff(s_full[::d + 1])
    sf = np.asarray(data["vehicle_full"]["sf"], float).ravel()
    cols = np.setdiff1d(np.arange(sf.size), np.arange(N + 1) * (d + 1))
    q = (np.asarray(data["vehicle_full"][key], float).ravel() * sf)[cols].reshape(N, d)
    return float(np.sum(dsk * (q @ GL_B)))


def seed_inputs(k, N):
    """Start point with two halves: deploy and paddle in the first half, brake in the second half."""
    if k < N // 2:
        return {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.3, 4: 0.5}
    return {0: 0.0, 1: -0.2, 2: 0.0, 3: 0.0, 4: 0.0}


# ---- checks ----------------------------------------------------------------------------------------------
print("[cs_hybrid] a. gates")
ok("177 off: no layer (all gates off, no input) with 535, 590, 573 and 589 on",
   C.hybrid_gates(C.load_car(car_path({177: 0, 535: 1, 590: 1, 573: 1, 589: 1}), INST)) == (False, False, False)
   and C.hybrid_inputs(C.load_car(car_path({177: 0, 535: 1, 590: 1, 573: 1, 589: 1}), INST)) == [])
cs_no590 = C.load_car(car_path({**HYB, 590: 0}), INST)
ok("535 = 1 with 590 = 0: deploy on, brake regen off", C.hybrid_gates(cs_no590) == (True, False, False))
cs_no589 = C.load_car(car_path({**HYB, **{573: 1, 589: 0}}), INST)
ok("573 = 1 with 589 = 0: deploy on, paddle off, one input (F_dep)",
   C.hybrid_gates(cs_no589) == (True, True, False) and C.hybrid_inputs(cs_no589) == ["F_dep"])

c_no590 = build(car_path({**HYB, 590: 0}))
o = at(c_no590, state(c_no590, 40.0), [0.0, -0.5, 0.0, 0.0])
ok(f"590 = 0: P_harv is 0 at a braking state ({o['P_harv']!r} W)", o["P_harv"] == 0.0)

print("[cs_hybrid] b. NLP size, rows, bounds")
c_base = build(car_path())
c_h = build(car_path(HYB))
c_p = build(car_path({**HYB, **PAD}))
nb, nh_, npd = c_base.m23.nu, c_h.m23.nu, c_p.m23.nu
ok(f"base car: nu 3, inputs T_drive, T_brake, delta (got {nb})", nb == 3 and c_base.input_keys == ["T_drive", "T_brake", "delta"])
ok(f"hybrid car: nu 4, the deploy force after delta (got {nh_}, keys {c_h.input_keys})",
   nh_ == 4 and c_h.input_keys[-1] == "F_dep")
ok(f"hybrid car with paddle: nu 5, F_pad after F_dep (got {npd}, keys {c_p.input_keys})",
   npd == 5 and c_p.input_keys[-2:] == ["F_dep", "F_pad"])
ok("the deploy inputs share the drive scale and run from 0 to 1",
   np.allclose(c_p.m23.u_s[3:], c_p.m23.u_s[0]) and np.all(c_p.m23.u_min[3:] == 0) and np.all(c_p.m23.u_max[3:] == 1))
rows_h = set(c_h.data["constraints"])
rows_p = set(c_p.data["constraints"])
ok("hybrid rows: deploy_pw and deploy_brk, no paddle rows", {"deploy_pw", "deploy_brk"} <= rows_h
   and not rows_h & {"paddle_pw", "paddle_drv"})
ok("paddle rows: paddle_pw and paddle_drv join the hybrid rows", {"deploy_pw", "deploy_brk", "paddle_pw", "paddle_drv"} <= rows_p)
N = int(c_h.data["N"])
ng_b, ng_h, ng_p = (int(c.data["nlp"]["structure"]["n_g"]) for c in (c_base, c_h, c_p))
ok(f"n_g grows by 2 (N+1) + N + 2 with the deploy layer (got {ng_h - ng_b}, expect {2 * (N + 1) + N + 2})",
   ng_h - ng_b == 2 * (N + 1) + N + 2)
ok(f"n_g grows by 2 (N+1) + N with the paddle (got {ng_p - ng_h}, expect {2 * (N + 1) + N})",
   ng_p - ng_h == 2 * (N + 1) + N)
ok("n_lap 2 in the structure of the deploy car, 2 with the paddle too",
   int(c_h.data["nlp"]["structure"]["n_lap"]) == 2 and int(c_p.data["nlp"]["structure"]["n_lap"]) == 2)
lbg, ubg = (np.asarray(c_h.cap["k"][n]).ravel() for n in ("lbg", "ubg"))
ok(f"lap rows: lb -inf, ub 836 / E_S = 0 (E_dep - E_harv) and ub (181 + 836) / E_S = 4 (E_dep)",
   np.all(np.isneginf(lbg[-2:])) and np.allclose(ubg[-2:], [0.0, (HYB[181] + HYB[836]) / E_S]))

print("[cs_hybrid] c. hand values")
m_h = c_h.m23
F_ref, B = float(m_h.u_s[0]), float(m_h.u_s[1])
bias = float(c_h.cs.brake.front_bias)
VX = 40.0
v_r, v_f = 1.03 * VX, VX                             # the rear wheels spin 3 %
x_s = state(c_h, VX)
P_dep_hand = 0.4 * F_ref * (0.7 * v_r + 0.3 * v_f)  # blend 465 = 0.3: front share of the deploy force
o = at(c_h, x_s, [0.0, 0.0, 0.0, 0.4])
ok(f"P_dep = F_dep v_mix with v_mix = 0.7 v_r + 0.3 v_f (rel {rel(o['P_dep'], P_dep_hand):.1e})",
   rel(o["P_dep"], P_dep_hand) < 1e-9)
ok("deploy_pw = (S kd - P_dep) / S with no map (S kd = 48 kW)",
   abs(o["deploy_pw"] - (S_DEP * KD - P_dep_hand) / S_DEP) < 1e-12)

c_map = build(car_path({**HYB, 466: 1}, maps={467: const_map(0.6), 468: const_map(0.2)}))
o = at(c_map, state(c_map, VX), [0.0, 0.0, 0.0, 0.4])
fac = 0.7 * 0.2 + 0.3 * 0.6                           # the blend of the two maps, 1 - s rear + s front
ok("deploy map: cap = S kd x (blend of the maps) (rel check on deploy_pw)",
   rel(o["deploy_pw"], (S_DEP * KD * fac - P_dep_hand) / S_DEP) < 1e-9)
c_zero = build(car_path({**HYB, 466: 1}, maps={467: const_map(0.0), 468: const_map(0.0)}))
o = at(c_zero, state(c_zero, VX), [0.0, 0.0, 0.0, 0.4])
ok("zero deploy maps: no deploy, deploy_pw = -P_dep / S", rel(o["deploy_pw"], -P_dep_hand / S_DEP) < 1e-9)

o = at(c_h, x_s, [0.0, -0.1, 0.0, 0.0])
B_net = 0.1 * B
hb_hand = KR * min(CLAMP, (1.0 - bias) * B_net * v_r)
ok(f"brake harvest, rear only, unclamped: P_harv = 0.8 min(clamp, (1 - bias) B v_r) "
   f"(got {o['P_harv']:.3f}, hand {hb_hand:.3f} W)", abs(o["P_harv"] - hb_hand) < 1e-3 * CLAMP)
o = at(c_h, x_s, [0.0, -0.5, 0.0, 0.0])
hb_hand = KR * min(CLAMP, (1.0 - bias) * 0.5 * B * v_r)
ok(f"brake harvest at the clamp: P_harv {o['P_harv']:.1f} W against 0.8 clamp = {hb_hand:.1f} W "
   "(within the smoothing 1e-3 clamp)", abs(o["P_harv"] - hb_hand) < KR * 1e-3 * CLAMP)

c_front = build(car_path({**HYB, **FRONT}))
o = at(c_front, state(c_front, VX), [0.0, -0.1, 0.0, 0.0])
k_f = min(bias / 0.55, (1.0 - bias) / 0.45)
v_mix55 = 0.45 * v_r + 0.55 * v_f
hb_hand = KR * min(CLAMP, k_f * B_net * v_mix55)
ok(f"brake harvest with the front share 0.55 (k {k_f:.4f}, v_mix 0.45 v_r + 0.55 v_f): "
   f"P_harv {o['P_harv']:.1f} against {hb_hand:.1f} W", abs(o["P_harv"] - hb_hand) < 1e-3 * CLAMP)

o = at(c_p, state(c_p, VX), [0.0, -0.1, 0.0, 0.0, 0.1])
F_pad = 0.1 * F_ref
hb0 = min(CLAMP, (1.0 - bias) * B_net * v_r)
hp_hand = KR * v_r * F_pad
ok(f"paddle: P_harv = P_hb + P_hp = {o['P_harv']:.1f} against {KR * hb0 + hp_hand:.1f} W",
   abs(o["P_harv"] - (KR * hb0 + hp_hand)) < 1e-3 * CLAMP)
row_hand = (CLAMP - hb0 - F_pad * v_r) / CLAMP
ok(f"paddle_pw = (clamp - P_hb0 - F_pad v) / clamp (got {o['paddle_pw']:.6f}, hand {row_hand:.6f})",
   abs(o["paddle_pw"] - row_hand) < 1e-3)
o = at(c_p, state(c_p, VX), [0.0, 0.0, 0.0, 0.0, 0.1])
ok(f"paddle alone: P_harv = 0.8 v_r F_pad = {KR * v_r * F_pad:.1f} W (got {o['P_harv']:.1f})",
   abs(o["P_harv"] - KR * v_r * F_pad) < 1e-3 * CLAMP)

print("[cs_hybrid] d. net-force harvest")
c_b = np.sqrt(1e-3 * F_ref * B)                       # drive = brake on the BrTh_1 bound
c_p_ = np.sqrt(1e-3) * F_ref                          # drive = paddle on the paddle_drv bound
VX_L = 63.26
x_l = state(c_p, VX_L)
o = at(c_p, x_l, [c_b / F_ref, -c_b / B, 0.0, 0.0, 0.0])
ok(f"drive = brake (BrTh_1 at its bound): |P_harv| {abs(o['P_harv']):.3f} W < 1 % of F v {c_b * VX_L:.0f} W",
   abs(o["P_harv"]) < 0.01 * c_b * VX_L)
o = at(c_p, x_l, [c_p_ / F_ref, 0.0, 0.0, 0.0, c_p_ / F_ref])
ok(f"drive = paddle (paddle_drv at its bound): |P_harv| {abs(o['P_harv']):.3f} W < 1 % of F v {c_p_ * VX_L:.0f} W",
   abs(o["P_harv"]) < 0.01 * c_p_ * VX_L)
o = at(c_p, x_l, [0.2, 0.0, 0.0, 0.0, 0.0])
ok(f"drive only: |P_harv| {abs(o['P_harv']):.3f} W < 1 W", abs(o["P_harv"]) < 1.0)
o = at(c_p, x_l, [0.0, 0.0, 0.0, 0.0, 0.0])
ok(f"coasting: |P_harv| {abs(o['P_harv']):.4f} W < 1 W", abs(o["P_harv"]) < 1.0)

print("[cs_hybrid] e. lap rows against a numpy quadrature")
SEED.update(nu=5, u=seed_inputs)
c_q = build(car_path({**HYB, **PAD}))
SEED.clear()
nlp, x_q = c_q.cap["nlp"], c_q.cap["x"]
g = np.array(ca.Function("g", [nlp["x"]], [nlp["g"]])(x_q)).ravel()
lap_rows = nlp_index(c_q.data["nlp"]["structure"])["lap"]
E_dep_q, E_harv_q = quad(c_q.data, "P_dep"), quad(c_q.data, "P_harv")
ok(f"the start point has deploy and harvest energy (E_dep {E_dep_q / 1e3:.1f} kJ, E_harv {E_harv_q / 1e3:.1f} kJ)",
   E_dep_q > 1e3 and E_harv_q > 1e3)
ok(f"lap row 2 (E_dep / E_S) equals the quadrature (rel {rel(g[lap_rows[1]] * E_S, E_dep_q):.1e})",
   rel(g[lap_rows[1]] * E_S, E_dep_q) < 1e-9)
ok(f"lap row 1 ((E_dep - E_harv) / E_S) equals the quadrature (rel {rel(g[lap_rows[0]] * E_S, E_dep_q - E_harv_q):.1e})",
   rel(g[lap_rows[0]] * E_S, E_dep_q - E_harv_q) < 1e-9)

print("[cs_hybrid] f. value checks")
NAN = float("nan")
nan_map = (np.array([1000.0, 9000.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0, NAN, 1.0]))
for name, cids, maps, cid in (
        ("182 = 1.5", {**HYB, 182: 1.5}, None, 182),
        ("182 = NaN", {**HYB, 182: NAN}, None, 182),
        ("183 = -1", {**HYB, 183: -1.0}, None, 183),
        ("836 = -1", {**HYB, 836: -1.0}, None, 836),
        ("180 = 0", {**HYB, 180: 0.0}, None, 180),
        ("NaN deploy map 468 with 466 = 1", {**HYB, 466: 1}, {468: nan_map}, 468)):
    msg = msg_of(car_path(cids, maps))
    ok(f"{name} raises ValueError naming cid {cid}", msg != "silent" and str(cid) in msg)
ok("183 = 2 loads (a deploy scale above 1 is allowed)", msg_of(car_path({**HYB, 183: 2.0})) == "silent")
ok("182 = 1 and 836 = 0 load (the bounds themselves)", msg_of(car_path(HYB)) == "silent")

print("[cs_hybrid] g. skip rule")
ON_OFF = {177: 0, 535: 1, 590: 1, 573: 1, 589: 1, 179: CLAMP, 180: S_DEP, 181: 4e5, 182: 1.0, 183: 1.0, 836: 0.0,
          464: 1, 465: 0.3, 466: 1}
c_off = build(car_path(ON_OFF))
ok("177 = 0: nu 3, no lap rows (n_lap absent or 0), no hybrid rows",
   c_off.m23.nu == 3 and int(c_off.data["nlp"]["structure"].get("n_lap", 0)) == 0
   and not getattr(c_off.m23, "lap_rows", None) and not getattr(c_off.m23, "hyb_rows", None))
ok("177 = 0: no data['hybrid'], no deploy or paddle row", "hybrid" not in c_off.data
   and not {"deploy_pw", "deploy_brk", "paddle_pw", "paddle_drv"} & set(c_off.data["constraints"]))
ok("177 = 0 keeps the base graph (dx, sf hash equal to the car without hybrid records)",
   graph_hash(car_path(ON_OFF)) == graph_hash(car_path()))

print("[cs_hybrid] h. warm starts")
P_H, P_B = car_path(HYB), car_path()
c_hs = build(P_H, save=True)
c_bs = build(P_B, save=True)
w_hh = build(P_H, save=False, warm_start=c_hs.out_path)
w_hb = build(P_B, save=False, warm_start=c_hs.out_path)
w_bh = build(P_H, save=False, warm_start=c_bs.out_path)
ok(f"hybrid -> the same hybrid: full+duals (got {w_hh.data['nlp']['warm_start']})",
   w_hh.data["nlp"]["warm_start"] == "full+duals")
ok(f"hybrid -> base: full-interp (got {w_hb.data['nlp']['warm_start']})", w_hb.data["nlp"]["warm_start"] == "full-interp")
ok(f"base -> hybrid: full-interp (got {w_bh.data['nlp']['warm_start']})", w_bh.data["nlp"]["warm_start"] == "full-interp")

print("[cs_hybrid] i. sensitivity on a hybrid result")
ix = nlp_index(c_hs.data["nlp"]["structure"])
ok("nlp_index: the lap rows are the last two rows of g", list(ix["lap"]) == [ix["n_g"] - 2, ix["n_g"] - 1])
lp = limit_prices(c_hs.data)
ok(f"limit_prices runs on the hybrid result (keys {sorted(lp)[:3]}...)", isinstance(lp, dict) and "Tmax" in lp)

print("[cs_hybrid] j. data['hybrid'] and the channels")
hyb = c_q.data["hybrid"]
ok("data['hybrid'] holds E_dep and E_harv in J, and the cids 177, 590 and 589",
   all(k in hyb for k in ("E_dep", "E_harv", "cid177", "cid590", "cid589")) and hyb["cid590"] == 1.0
   and hyb["cid589"] == 1.0)
ok(f"data['hybrid'] energies equal the quadrature (E_dep rel {rel(hyb['E_dep'], E_dep_q):.1e}, "
   f"E_harv rel {rel(hyb['E_harv'], E_harv_q):.1e})",
   rel(hyb["E_dep"], E_dep_q) < 1e-9 and rel(hyb["E_harv"], E_harv_q) < 1e-9)
ok("data['hybrid'] is absent without the layer", "hybrid" not in c_base.data)
vf, vk = c_q.data["vehicle_full"], c_q.data["vehicle"]
d1 = int(c_q.data["OPT_d"]) + 1
ok("the channels P_dep and P_harv are in vehicle_full and their knot slice is in vehicle (bitwise)",
   "P_dep" in vf and "P_harv" in vf and np.array_equal(np.asarray(vk["P_dep"]), np.asarray(vf["P_dep"])[::d1])
   and np.array_equal(np.asarray(vk["P_harv"]), np.asarray(vf["P_harv"])[::d1]))

# ---- QSS layer (P5): the hybrid layer in the lap estimate of MLTP_screen -----------------------------
VI_QSS = 60.0                                         # MLTP_screen default vi [m/s]
BASE_LAP = {}


def qss(path, circ=CIRC, install=None, **kw):
    """MLTP_screen on a car file (the synthetic install by default). Returns ctx."""
    opts = dict(car=f"chassissim:{path}", circuit=f"chassissim:{circ}", chassissim_dir=install or INST,
                physics="corrected", save=False, verbose=False)
    opts.update(kw)
    return quiet(MLTP_screen, **opts)


def qss_shape(recs, peak=1.0):
    """The shape of the QSS car records (as test_cs_qss.shape): positive tyre peaks, a slip curve from 0 to 1, camber
    factor 1 and flat aero maps. Only the values change, the tables keep their shapes."""
    out = []
    for t, cid, v in recs:
        if t == 9 and cid in (127, 131, 135):              # CLA, CDA and balance maps: flat
            x, z, d = v
            v = (x, z, np.full(len(d), {127: 2.5, 131: 0.95, 135: 0.45}[cid]))
        elif t == 13:
            dims, sc, tabs = v
            tabs = [np.array(a, float) for a in tabs]
            d1, d2, d3, _, _, d6, _ = (int(d) for d in dims)
            load = tabs[3][:d1]
            grid = np.outer(load, np.linspace(1.0, 1.1, d2)) * 1.2 * peak
            tabs[3] = np.concatenate([load, tabs[3][d1:d1 + d2], grid.ravel()])
            tabs[0][d3:] = np.linspace(0.0, 1.0, d3)
            tabs[5][d6:] = 1.0
            v = (dims, sc, tabs)
        out.append((t, cid, v))
    return out


def qss_car_path(cids=None):
    """Write the synthetic car in the QSS shape with the cid overrides (cid -> value), flags tagged 6. Return the path."""
    cids = dict(cids or {})
    body = [r for r in qss_shape(build_records()) if r[1] not in cids]
    extra = [(6 if c in FLAG_CIDS else 1, c, v) for c, v in cids.items()]
    path = os.path.join(CARS_DIR, f"qss_{next(COUNT)}.car")
    carfile.write(body + extra + [(carfile.END_TAG, carfile.END_CID, None)], path)
    return path


def hy_car(over=None):
    """QSS-shaped synthetic car with the hybrid records HYB and the cid overrides over (cid -> value)."""
    return qss_car_path({**HYB, **(over or {})})


def base_lap():
    """QSS lap of the synthetic car with cid 177 off (the layer skipped)."""
    if "lap" not in BASE_LAP:
        BASE_LAP["lap"] = float(qss(qss_car_path({177: 0})).data.lap_time)
    return BASE_LAP["lap"]


def raised(fn, *a, **kw):
    """Return the ValueError message of fn(*a, **kw), or None when it does not raise."""
    try:
        fn(*a, **kw)
    except ValueError as exc:
        return str(exc)
    return None


def interp_at(env, y, v):
    return float(np.interp(v, env.v_grid, y))


def hand_accel(env, hy, v, kap, e, s):
    """Return (a, fd) of the deploy at speed v and curvature kap, written from the law: the engine force comes
    first (tyre or gear envelope), then the deploy force P_dep / v fills the room of each axle."""
    res = env.c_res0 + env.c_res2 * v * v
    r = v * v * kap / max(interp_at(env, env.ay_max, v), 1e-9)
    el = math.sqrt(max(1.0 - r * r, 0.0))
    a_pl = min((interp_at(env, env.Fx_trac, v) * el - res) / env.M, (interp_at(env, env.F_drive_tab, v) - res) / env.M_eff)
    fe = a_pl * env.M_eff + res
    cv = env.M_eff / env.M
    rooms = []
    if s > 0.0:
        rooms.append((interp_at(env, hy["cap_f"], v) * cv * el - e * fe) / s - (cv - 1.0) * res)
    if s < 1.0:
        rooms.append((interp_at(env, hy["cap_r"], v) * cv * el - (1.0 - e) * fe) / (1.0 - s) - (cv - 1.0) * res)
    fd = max(min([interp_at(env, hy["P_dep"], v) / max(v, 1e-3)] + rooms), 0.0)
    return a_pl + fd / env.M_eff, fd


def hand_next(env, hy, v, ka, kb, apex_b, ds):
    """Heun step of the deploy pass in w = v^2, from v (curvature ka) to the next station (curvature kb, apex apex_b)."""
    a1 = hand_accel(env, hy, v, ka, hy["e"], hy["s"])[0]
    vp = min(math.sqrt(v * v + 2.0 * ds * a1), apex_b)
    a2 = hand_accel(env, hy, vp, kb, hy["e"], hy["s"])[0]
    return min(math.sqrt(v * v + ds * (a1 + a2)), apex_b)


def last_accel_run(env, s, k):
    """First segment and last segment + 1 of the last zone (forward pass binds) of the profile without deploy."""
    saved, env.hybrid = env.hybrid, None
    try:
        q = G.march(env, s, k, VI_QSS)
    finally:
        env.hybrid = saved
    up = np.nonzero(q["v_fwd"][:-1] <= q["v_bwd"][:-1] + 1e-9)[0]
    cut = np.nonzero(np.diff(up) > 1)[0]
    lo = up[cut[-1] + 1] if cut.size else up[0]
    return int(lo), int(up[-1]) + 1


def deploy_steps(q, label, e_want, s_want):
    """Check the split and the deploy share of the car, then the Heun step of three deploy stations by hand."""
    env, hy = q.envelope, q.envelope.hybrid
    ok(f"{label}: split e {hy['e']:g} and deploy front share s {hy['s']:.6f} as set",
       abs(hy["e"] - e_want) < 1e-12 and abs(hy["s"] - s_want) < 1e-6)
    s, kk = np.asarray(q.track.s, float), np.asarray(q.track.k, float)
    p = G.march(env, s, kk, VI_QSS)
    lo, hi = last_accel_run(env, s, kk)
    kabs = np.abs(p["k"])
    picks = []                                          # the first three stations of the zone where the hand deploy is above 1 N
    for i in range(lo + 5, hi - 1, 5):
        if hand_accel(env, hy, p["v_fwd"][i], kabs[i], hy["e"], hy["s"])[1] > 1.0:
            picks.append(i)
        if len(picks) == 3:
            break
    ok(f"{label}: {len(picks)} stations of the zone deploy by hand (fd above 1 N)", len(picks) >= 2)
    for i in picks:
        fd = hand_accel(env, hy, p["v_fwd"][i], kabs[i], hy["e"], hy["s"])[1]
        vn = hand_next(env, hy, p["v_fwd"][i], kabs[i], kabs[i + 1], p["v_apex"][i + 1], p["ds"])
        ok(f"{label}: station {i} (v {p['v_fwd'][i]:.4f} m/s, hand fd {fd:.2f} N) Heun step equals the march "
           f"(rel {rel(vn, p['v_fwd'][i + 1]):.1e})", rel(vn, p["v_fwd"][i + 1]) < 1e-12)


print("[cs_hybrid] k. the hybrid layer in the QSS (MLTP_screen)")
# (a) skip rule: cid 177 = 0 gives the lap of the car without the hybrid cids, and no record
q_none = qss(qss_car_path({}))
q_off = qss(qss_car_path({177: 0, 535: 1, 590: 1, 573: 1, 589: 1, 178: 1, 181: 1000.0}))
ok(f"177 = 0 (other hybrid cids on): the QSS lap equals the car without them ({q_off.data.lap_time!r})",
   q_off.data.lap_time == q_none.data.lap_time and "hybrid" not in vars(q_off.data) and q_off.envelope.hybrid is None)

# (b) harvest against a scalar loop over the braking segments, store, deploy energy and the lap
q0 = qss(hy_car({178: 0}))
env0, cs0 = q0.envelope, q0.cs
hy0, h0 = env0.hybrid, q0.data.hybrid
s0, k0 = np.asarray(q0.track.s, float), np.asarray(q0.track.k, float)
p0 = G.march(env0, s0, k0, VI_QSS)
k_hand = regen_share(float(cs0.brake.front_bias), 0.0)
v, vf, vb, ds = p0["v"], p0["v_fwd"], p0["v_bwd"], p0["ds"]
E_hand, n_brk = 0.0, 0
for i in range(v.size - 1):
    vi_, vj = v[i], v[i + 1]
    if not (vj < vi_ - 1e-9 and vb[i] <= vf[i] + 1e-9):
        continue
    vm, dt = 0.5 * (vi_ + vj), 2.0 * ds / (vi_ + vj)
    B = max(env0.M_eff * (vi_ * vi_ - vj * vj) / (2.0 * ds) - (env0.c_res0 + env0.c_res2 * vm * vm), 0.0)
    E_hand += min(CLAMP, k_hand * B * vm) * KR * dt
    n_brk += 1
ok(f"E_harv equals the scalar loop over {n_brk} braking segments ({E_hand:.4f} J, rel {rel(h0['E_harv'], E_hand):.1e})",
   n_brk > 0 and rel(h0["E_harv"], E_hand) < 1e-9 and abs(hy0["k"] - k_hand) < 1e-12
   and hy0["k_regen"] == KR and hy0["clamp"] == CLAMP)
ok(f"E_store = min(E_harv, cid 181) + cid 836 ({h0['E_store']:.3f} J, within 1 J)",
   abs(h0["E_store"] - min(h0["E_harv"], 400000.0)) < 1.0)
ok(f"E_dep <= E_store + 1 J ({h0['E_dep']:.3f} J)", h0["E_dep"] <= h0["E_store"] + 1.0)
ok(f"mode 0: the lap below the cid-177-off lap ({q0.data.lap_time:.5f} s, base {base_lap():.5f} s)",
   q0.data.lap_time < base_lap())

# (c) mode 1 runs, zones, sweeps
q1 = qss(hy_car({178: 1}))
h1 = q1.data.hybrid
ok(f"mode 0 and mode 1 both run: zones {h0['zones']} and {h1['zones']} (> 0), sweeps {h0['sweeps']} and "
   f"{h1['sweeps']} (<= 5), modes {h0['mode']} and {h1['mode']}",
   h0["zones"] > 0 and h1["zones"] > 0 and max(h0["sweeps"], h1["sweeps"]) <= 5 and (h0["mode"], h1["mode"]) == (0, 1))
ok(f"mode 1: store fixed point (E_store {h1['E_store']:.3f} J vs min(E_harv, 181) {min(h1['E_harv'], 400000.0):.3f} J) "
   f"and lap {q1.data.lap_time:.5f} s below the base lap", abs(h1["E_store"] - min(h1["E_harv"], 400000.0)) < 1.0
   and q1.data.lap_time < base_lap())

# (d) one deploy step by hand, e = s and e != s
deploy_steps(qss(hy_car({111: 0.5, 464: 0, 178: 0})), "e = s = 0.5", 0.5, 0.5)
deploy_steps(qss(hy_car({111: 0.5, 178: 0})), "e = 0.5, s = 0.3", 0.5, 0.3)

# (e) cid edges
q_1k = qss(hy_car({178: 1, 181: 1000.0}))
ok(f"181 = 1000 J caps the store (E_store {q_1k.data.hybrid['E_store']:.6f} J, E_dep {q_1k.data.hybrid['E_dep']:.6f} J)",
   abs(q_1k.data.hybrid["E_store"] - 1000.0) < 1e-6 and q_1k.data.hybrid["E_dep"] <= 1001.0)
q_50k = qss(hy_car({178: 1, 836: 50000.0}))
ok(f"836 = 50000 J adds to the store (E_store {q_50k.data.hybrid['E_store']:.3f} J)",
   abs(q_50k.data.hybrid["E_store"] - (min(q_50k.data.hybrid["E_harv"], 400000.0) + 50000.0)) < 1.0)
q_z = qss(hy_car({178: 0, 181: 0.0}))
ok(f"181 = 0: no store, no deploy, the base lap ({q_z.data.lap_time:.9f} s)",
   q_z.data.hybrid["E_store"] == 0.0 and q_z.data.hybrid["E_dep"] == 0.0 and abs(q_z.data.lap_time - base_lap()) < 1e-9)
q_nh = qss(hy_car({178: 0, 590: 0}))
ok(f"590 = 0 (no paddle): E_harv 0 and the base lap ({q_nh.data.lap_time:.9f} s)",
   q_nh.data.hybrid["E_harv"] == 0.0 and abs(q_nh.data.lap_time - base_lap()) < 1e-9)

# (f) ValueErrors name the cid; a car with 177 off is silent
m178 = raised(qss, hy_car({178: 2}))
ok(f"178 = 2 with 177 on: ValueError naming cid 178 ({m178!r:.70})", m178 is not None and "cid 178" in m178)
m836 = raised(qss, hy_car({836: 0.5}))
ok(f"836 = 0.5 with 177 on: ValueError naming cid 836 ({m836!r:.70})", m836 is not None and "cid 836" in m836)
ok("178 = 2 and 836 = 0.5 with 177 off: silent", raised(qss, qss_car_path({177: 0, 178: 2, 836: 0.5})) is None)
ok("836 = 1.0 with 177 on: no error", raised(qss, hy_car({836: 1.0})) is None)

# (g) NLP seed of a hybrid car (fake IPOPT, qss23 ladder): drive_env and the F_dep seed
SEED.clear()
cb = build(hy_car({178: 0}), ladder="qss23")
nlp, x0 = cb.cap["nlp"], np.asarray(cb.cap["x"], float)
ix = nlp_index(cb.data["nlp"]["structure"])
gv = np.array(ca.Function("g", [nlp["x"]], [nlp["g"]])(x0)).ravel()
drv = gv[ix["h"][0]]                                   # row 0 of the path rows: drive_env (lower bound 0)
ok(f"drive_env holds at the start point (min {drv.min():.2e}, no row below -1e-6)", drv.min() >= -1e-6)
jF = cb.input_keys.index("F_dep")
d_c = int(cb.data["OPT_d"])
s_knot = np.asarray(cb.data["s_full"], float).ravel()[:: d_c + 1]
v_k = np.interp(s_knot, p0["s"], p0["v"])
F_k = env0.M * np.interp(s_knot, p0["s"], p0["ax"]) + env0.c_res0 + env0.c_res2 * v_k * v_k
F_env = np.interp(v_k, env0.v_grid, env0.F_drive_tab)
exp_fd = np.clip(F_k - F_env, 0.0, np.interp(v_k, env0.v_grid, hy0["P_dep"]) / np.maximum(v_k, 1e-3))
seed_fd = x0[ix["U"][jF]] * float(cb.m23.u_s[jF])
seed_fd = seed_fd.ravel()
err = float(np.max(np.abs(seed_fd - exp_fd)))
n_pos = int(np.sum(exp_fd > 1.0))
ok(f"F_dep seed equals F - F_env clipped to the deploy cap at the knots (max err {err:.1e} N)",
   err <= 1e-6 * max(1.0, float(np.max(exp_fd))))
ok(f"F_dep is seeded where the QSS deploys ({n_pos} knots above 1 N, all carry the seed)",
   n_pos > 0 and bool(np.all(seed_fd[exp_fd > 1.0] > 0.0)))

# (h) data['hybrid'] and the .mat round trip
mat_dir = os.path.join(TMP.name, "qss_mat")
os.makedirs(mat_dir)
qs = qss(hy_car({178: 1}), save=True, results_dir=mat_dir)
KEYS = ("E_harv", "E_store", "E_dep", "mode", "zones", "sweeps")
rt = loadmat(qs.out_path, squeeze_me=True, struct_as_record=False)["data"].hybrid
ok("data['hybrid'] holds E_harv, E_store, E_dep (J), mode, zones and sweeps", set(KEYS) <= set(qs.data.hybrid))
ok("data['hybrid'] survives the .mat round trip exactly",
   all(float(getattr(rt, k)) == float(qs.data.hybrid[k]) for k in KEYS))

# 4EM Sturn screen with casadi blocked (a child process)
code = ('import sys\nsys.modules["casadi"] = None\nfrom MLTP_screen import MLTP_screen\n'
        'ctx = MLTP_screen(circuit="Sturn", save=False, verbose=False)\n'
        'print("LAP %r" % float(ctx.data.lap_time))\n')
r = subprocess.run([sys.executable, "-c", code], cwd=os.getcwd(), capture_output=True, text=True, timeout=600)
lap4 = float(r.stdout.split("LAP ")[1].split()[0]) if "LAP " in r.stdout else float("nan")
ok(f"4EM Sturn screen with casadi blocked: {lap4!r} (18.743926140324547)",
   r.returncode == 0 and abs(lap4 - 18.743926140324547) < 1e-9)

# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs_dir):
    print("[cs_hybrid] vendor: shipped cars and a hybrid GP3 copy")
    root = Path(cs_dir).parent.parent                 # the ChassisSimTechnologies tree, one level above the install folder
    cars = sorted(root.rglob("*.car"))
    on = []
    for p in cars:
        recs = carfile.decode(str(p))
        on += [c for t, c, v in recs if c == 177 and v not in (None, 0)]
    ok(f"every shipped car file ({len(cars)}) has cid 177 off (no hybrid input)", not on and len(cars) > 0)
    models = os.path.join(os.path.dirname(cs_dir), "Models")
    gp3 = sorted(glob.glob(os.path.join(glob.escape(models), "GP3*", "*Eastern*", "def*.car")))
    ok("GP3 car file found by folder glob", bool(gp3))
    body = [r for r in carfile.decode(gp3[0]) if r[1] not in (177, 535, 590) and r[1] != carfile.END_CID]
    path = os.path.join(CARS_DIR, "gp3_hybrid.car")
    carfile.write(body + [(6, 177, 1), (6, 535, 1), (6, 590, 1), (carfile.END_TAG, carfile.END_CID, None)], path)
    c = quiet(MLTP, car=f"chassissim:{path}", circuit="chassissim:Eastern Creek", chassissim_dir=cs_dir,
              physics="corrected", save=False, plot=False, linear_solver="mumps", results_dir=RES)
    ok(f"the hybrid GP3 copy builds with nu 4 (got {c.m23.nu}, keys {c.input_keys})",
       c.m23.nu == 4 and c.input_keys[-1] == "F_dep")
    ok(f"the deploy front share is 0.2 of the file (got {c.cs.hybrid.blend}, float32 in the file)",
       abs(c.cs.hybrid.blend - 0.2) < 1e-6)
    m = c.m23
    x = state(c, 40.0)
    o = at(c, x, [0.0, 0.0, 0.0, 0.3])
    v_r_, v_f_ = x[7] * c.cs.tyre.rl["radius"], x[5] * c.cs.tyre.fl["radius"]
    share = float(c.cs.hybrid.blend)                  # the front share as the file holds it (float32)
    hand = 0.3 * float(m.u_s[0]) * ((1.0 - share) * v_r_ + share * v_f_)
    ok(f"the GP3 deploy power is F_dep (0.8 v_r + 0.2 v_f) (rel {rel(o['P_dep'], hand):.1e})",
       rel(o["P_dep"], hand) < 1e-9)


def vendor_qss(cs_dir):
    print("[cs_hybrid] vendor QSS: GP3, GT3 and Formula 2 laps unchanged, a hybrid GP3 copy")
    models = Path(cs_dir).parent / "Models"
    gp3 = sorted(models.glob("GP3*/*Eastern*/def*.car"))
    gt3 = [p for p in sorted(models.glob("GT3/*/*.car")) if any(p.parent.glob("*.ld"))]
    f2 = sorted(models.glob("Formula 2/*/*.car"))
    ok("GP3, GT3 and Formula 2 car files found by folder globs", bool(gp3) and bool(gt3) and bool(f2))
    base = float(qss(gp3[0], circ="Eastern Creek", install=cs_dir).data.lap_time)
    for label, path, circ, want in (("GP3", gp3[0], "Eastern Creek", 74.05494989625016),
                                    ("GT3", gt3[0], "Adelaide", 73.05236704024435),
                                    ("Formula 2", f2[0], "Adelaide", 59.5239241129469)):
        lap = float(qss(path, circ=circ, install=cs_dir).data.lap_time) if label != "GP3" else base
        ok(f"{label} QSS lap at {circ} unchanged: {lap!r} (tol 1e-9)", abs(lap - want) < 1e-9)
    recs = carfile.decode(str(gp3[0]))
    for mode in (0, 1):
        body = [r for r in recs if r[1] not in (177, 178, 535, 590) and r[1] != carfile.END_CID]
        path = os.path.join(CARS_DIR, f"gp3_hyb_m{mode}.car")
        carfile.write(body + [(6, 177, 1), (6, 178, mode), (6, 535, 1), (6, 590, 1),
                              (carfile.END_TAG, carfile.END_CID, None)], path)
        c = qss(path, circ="Eastern Creek", install=cs_dir)
        h = c.data.hybrid
        gain = base - float(c.data.lap_time)
        ok(f"hybrid GP3 copy, mode {mode}: gain {gain:.4f} s in [0.40, 0.65] (E_harv {h['E_harv']:.0f} J, "
           f"E_dep {h['E_dep']:.0f} J, zones {h['zones']}, sweeps {h['sweeps']})",
           0.40 <= gain <= 0.65 and h["E_dep"] <= h["E_store"] + 1.0)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
    vendor_qss(CS)
print(f"{PASS} checks passed")
