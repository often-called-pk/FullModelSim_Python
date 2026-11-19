"""ChassisSim phase 2 task 11: bump rubber gap of the other shipped cars (vehModel_cs.rubber_gap, _rubber).

Gap flag 1 (the stored gap is measured on the ground) and flag 0 (the stored gap is measured from the static
compression: effective gap = stored gap - (F0 / (mr0 k) - preload)). A stored gap of 0 means no rubber at the
corner, with no node in the NLP. A negative effective gap is valid. Third springs (cids 5, 6) and third rubbers
(cids 17, 18, gap flags 208, 209) build (ChassisSim phase 2b, task 12). The checks read the synthetic car records with load_car, build the car with MLTP and a fake IPOPT
(no solve), and spy on the rubber helper that the QSS envelope calls. The vendor section (last) runs only when
CHASSISSIM_DIR names the ChassisSim install folder. It reads the GP3 2013, F2, GT3 and F3 car files and writes
nothing into the vendor folders. Scratch files go to a temp dir.

    python tests/test_cs_rubber.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_rubber.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import glob
import io
import os
import tempfile
import warnings
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

import functions.transcription as T
import vehModel_cs as V
from functions import carfile
from functions.chassissim import check_car, load_car
from MLTP import MLTP
from MLTP_screen import MLTP_screen

PASS = 0
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair. An over or omit cid is a file cid.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}
G = 9.81
REFUSE_FLAG = "The car file holds a bump rubber gap flag (cids 204 to 207) other than 0 or 1."


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


# ---- synthetic car (copied from tests/test_cs_diff.py, third spring rates 0) ----------------------------
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
DIMS = (3, 2, 3, 2, 2, 5, 4)
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
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,
    100: 0.9, 102: 1000.0, 103: 9000.0,
    107: 14000.0, 108: 0.6, 109: 0.05,
    111: 0.0, 112: 1.0, 113: 0.0,
    55: RC_STATIC, 59: RC_STATIC, 63: RC_STATIC, 67: RC_STATIC,
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,
    390: 350.0,
}
EXPLICIT.update({7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.0, 12: 0.0})   # third gaps 0: no third element
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
AERO_X = 0.005 * np.arange(7)
AERO_Z = 0.002 + 0.005 * np.arange(7)
MAP_X = np.array([-10000.0, -5000.0, -3333.0, -1667.0, 0.0, 1667.0, 3333.0, 5000.0, 6667.0, 8333.0, 10000.0])
MAP_Z = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
MAP_DATA = np.where((MAP_X > 0)[:, None] & (MAP_Z >= 0.1)[None, :], 0.1 * MAP_X[:, None], 0.0)


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
    grid = (1.4 * load[:, None] + 100.0 * np.arange(d2)[None, :]).ravel()
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


def build_records(over=None, omit=()):
    """Records of the synthetic car. over maps a cid to a new value (tag kept); cids in omit are left out."""
    R = []

    def add(tag, cid, v):
        cid = FILE_CID.get(cid, cid)
        if v is not None:
            R.append((tag, cid, v))
    vals = dict(EXPLICIT)
    vals[113] = 1.0                                       # open rear differential
    for cid, v in vals.items():
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
    for cid in range(204, 210):
        add(6, cid, 1)                                    # rubber gap flags (fl..rr, third f/r): 1 = ground gap
    for cid in (74, 75, 76, 77):
        add(7, cid, (BUMP_X, np.zeros(len(BUMP_X))))
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    add(9, 114, (0.5 * np.arange(4), 1.0 * np.arange(4) + 0.1, arb(16, 114)))
    add(9, 116, (MAP_X, MAP_Z, MAP_DATA.ravel()))
    for cid in (127, 131, 135):
        add(9, cid, (AERO_X, AERO_Z, arb(49, cid)))
    for cid in range(31, 36):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    for cid, rows in ((49, HP_LEFT), (50, HP_RIGHT), (51, HP_LEFT), (52, HP_RIGHT)):
        add(10, cid, (0, np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])))
    for i, c in enumerate(CORNERS):
        add(*tyre_record(i, c))
    over = over or {}
    R = [(t, c, over[c]) if c in over else (t, c, v) for t, c, v in R if c not in omit]
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


def build(car_path, circuit=CIRC, chassis_dir=INSTALL):
    """MLTP on a car with a fake IPOPT (no solve). Returns ctx with ctx.m23, ctx.cs and ctx.vp."""
    return quiet(MLTP, car=f"chassissim:{car_path}", circuit=f"chassissim:{circuit}", chassissim_dir=chassis_dir,
                 physics="corrected", save=False, plot=False, linear_solver="mumps")


def refusal(fn, *a, **kw):
    """(True, message) when fn raises ValueError, else (False, '')."""
    try:
        quiet(fn, *a, **kw)
    except ValueError as e:
        return True, str(e)
    return False, ""


# ---- helpers: the hand formulas, read from the record and vp --------------------------------------------
def static_x(ctx, v=30.0):
    """The static point of the car: straight line at v, tyre states at W0 / kt, zs = 0."""
    vp, cs = ctx.vp, ctx.cs
    x = np.zeros(23)
    x[0] = v
    x[5:7] = v / cs.tyre.fl["radius"]
    x[7:9] = v / cs.tyre.rl["radius"]
    x[19:23] = [getattr(vp, f"W{c}0") / vp.kt for c in CORNERS]
    return x


def hand_eff_gap(cs, vp, c):
    """Effective gap of corner c [m] from the record and vp. None for a stored gap of 0.
    Flag 1: the stored gap. Flag 0: the stored gap - (F0 / (mr0 k) - preload), F0 = m_c x g (vp), mr0 = mr(0)."""
    gap = float(getattr(cs.susp, f"gap_{c}"))
    if gap == 0.0:
        return None
    if getattr(cs.susp, f"gap_flag_{c}") == 1:
        return gap
    F0 = getattr(vp, f"m_c_{c}") * vp.g
    mr0 = float(np.asarray(getattr(cs.susp, f"mr_{c}"), float)[0])
    sp = getattr(cs.susp, f"spring_{c}")
    return gap - (F0 / (mr0 * float(sp["rate"])) - float(sp["preload"]))


def hand_rubber(cs, c, gap_eff, w):
    """Model bump rubber force at strut travel w [m] (wheel force): mr(w) x table(comp), with
    comp = W softplus((sm(w) - gap_eff) / W), W = the model onset width. The table is the 2-point rubber table of
    the synthetic car: linear from (0, 0), extended on its end slopes."""
    mrc = np.asarray(getattr(cs.susp, f"mr_{c}"), float)
    sm = sum(v / (i + 1) * w ** (i + 1) for i, v in enumerate(mrc))
    mr = sum(v * w ** i for i, v in enumerate(mrc))
    comp = V.W_RUBBER * np.log1p(np.exp((sm - gap_eff) / V.W_RUBBER))
    rub = getattr(cs.susp, f"rubber_{c}")
    x, y = V._clean(np.r_[0.0, np.asarray(rub["x"], float)], np.r_[0.0, np.asarray(rub["y"], float)])
    s_lo, s_hi = (y[1] - y[0]) / (x[1] - x[0]), (y[-1] - y[-2]) / (x[-1] - x[-2])
    table = np.where(comp < x[0], y[0] + s_lo * (comp - x[0]),
                     np.where(comp > x[-1], y[-1] + s_hi * (comp - x[-1]), np.interp(comp, x, y)))
    return mr * table


def model_rubber(ctx, c, w):
    """The model's rubber force at strut travel w (the term vehModel_cs adds in the NLP), numeric."""
    x = ca.SX.sym("x")
    return float(ca.Function("R", [x], [V._rubber(ctx.cs, c, ctx.vp)(x)])(w))


def dx_at(ctx, x):
    """Time rates of the 23 states at the physical state x (normalised as in tests/test_vehmodel_cs.py)."""
    m = ctx.m23
    f = ca.Function("f", [m.x, m.u, m.pv], [m.dx])
    return np.asarray(f.call([np.asarray(x, float) / m.x_s, np.zeros(m.nu), 0.0])[0], float).ravel()


def tyre_loads(ctx, x):
    m = ctx.m23
    fz = ca.Function("c", [m.x, m.u, m.pv], [m.fz_fl, m.fz_fr, m.fz_rl, m.fz_rr])
    return np.array([float(v) for v in fz.call([np.asarray(x, float) / m.x_s, np.zeros(m.nu), 0.0])])


def corner_weights(cs):
    """Static corner weights of the car [N]: the front share of the total on the front axle, half per corner."""
    front = cs.mass.front_share * cs.mass.total * G / 2
    rear = (1.0 - cs.mass.front_share) * cs.mass.total * G / 2
    return np.array([front, front, rear, rear])


# ==== 1. flag 1: the effective gap is the stored gap ==========================================================
print("[cs_rubber] 1. flag 1: rubber_gap is the stored gap")
CS_FLAG1 = load_car(write_car("flag1.car"))
ok("synthetic car: all four gap flags are 1",
   all(getattr(CS_FLAG1.susp, f"gap_flag_{c}") == 1 for c in CORNERS))
stored = {c: float(getattr(CS_FLAG1.susp, f"gap_{c}")) for c in CORNERS}
got = {c: V.rubber_gap(CS_FLAG1, c) for c in CORNERS}
ok(f"flag 1: rubber_gap equals the stored gap at all corners "
   f"(fl {got['fl']*1e3:.2f}, fr {got['fr']*1e3:.2f}, rl {got['rl']*1e3:.2f}, rr {got['rr']*1e3:.2f} mm)",
   all(got[c] == stored[c] for c in CORNERS))

# ==== 2. flag 0: effective gap = stored gap - (F0 / (mr0 k) - preload) ======================================
print("[cs_rubber] 2. flag 0: effective gap from the static compression")
# front left: preload 0, so the static compression (about 12 mm) sits above the 1 mm stored gap (negative effective gap)
# file cids: the front left gap is cid 8 and the front left spring is cid 2
OVER_FLAG0 = {204: 0, 205: 0, 206: 0, 207: 0, 8: 0.001, 7: 0.05, 9: 0.05, 10: 0.05,
             2: (1, 0.0, 1.23e5, np.array([0.0, 0.05]), np.array([0.0, 500.0]))}
CAR_FLAG0 = write_car("flag0.car", OVER_FLAG0)
CTX_FLAG0 = build(CAR_FLAG0)
cs0, vp0 = CTX_FLAG0.cs, CTX_FLAG0.vp
ok("flag 0 car builds with all four gap flags 0", all(getattr(cs0.susp, f"gap_flag_{c}") == 0 for c in CORNERS))
for c in CORNERS:
    hand = hand_eff_gap(cs0, vp0, c)
    model = V.rubber_gap(cs0, c, vp0)
    ok(f"flag 0 {c}: effective gap {model*1e3:.3f} mm = stored {float(getattr(cs0.susp, f'gap_{c}'))*1e3:.1f} "
       f"- (F0/(mr0 k) - preload), hand {hand*1e3:.3f} mm",
       model is not None and hand is not None and abs(model - hand) < 1e-12 and abs(model - getattr(cs0.susp, f"gap_{c}")) > 1e-4)

# Independent of the code's formula: a hard-coded number from the fixture values, by hand.
# Front left of the flag 0 fixture: stored gap 1.0 mm (cid 8), preload 0 (cid 2 override), k = 1.23e5 N/m (cid 2).
# Mass: total 700 kg (cid 84), unsprung front 25 kg (cid 85), rear 35 kg (cid 86), front share 0.44 (cid 96).
#   sprung mass mb = 700 - 25 - 35 = 640 kg
#   front share wB = (0.44 x 700 - 25) / 640 = 0.4421875
#   sprung corner mass m_c = 640 x 0.4421875 / 2 = 141.5 kg
#   F0 = m_c x g = 141.5 x 9.81 = 1388.115 N
#   mr0 = 1 + 0.05 sin(23) = 0.9576890 (the front left record, file cid 24, zero travel)
#   static deflection F0 / (mr0 k) = 1388.115 / (0.9576890 x 1.23e5) = 0.0117841 m
#   effective gap = 0.001 - 0.0117841 = -0.0107841 m
FLAG0_FL_EFF_HAND = -0.010784084
ok(f"flag 0 fl effective gap equals the hand value -10.784084 mm (model {V.rubber_gap(cs0, 'fl', vp0)*1e3:.6f} mm)",
   abs(V.rubber_gap(cs0, "fl", vp0) - FLAG0_FL_EFF_HAND) < 1e-7)

# Second independent check: static deflection from the car record alone (no vp), then
# effective gap + static deflection = stored gap + preload, for every corner.
mass = cs0.mass
mb_rec = mass.total - mass.unsprung_front - mass.unsprung_rear
wB_rec = (mass.front_share * mass.total - mass.unsprung_front) / mb_rec
for c in CORNERS:
    share = wB_rec if c[0] == "f" else 1.0 - wB_rec
    F0_rec = (mb_rec * share / 2) * G
    mr0_rec = float(np.asarray(getattr(cs0.susp, f"mr_{c}"), float)[0])
    sp = getattr(cs0.susp, f"spring_{c}")
    defl = F0_rec / (mr0_rec * float(sp["rate"]))
    eff = V.rubber_gap(cs0, c, vp0)
    stored, pre = float(getattr(cs0.susp, f"gap_{c}")), float(sp["preload"])
    ok(f"flag 0 {c}: effective gap + static deflection ({defl*1e3:.3f} mm) = stored gap + preload "
       f"({(stored + pre)*1e3:.3f} mm)", defl > 0.0 and abs((eff + defl) - (stored + pre)) < 1e-9)

# ==== 3. stored gap 0: no rubber, no node ====================================================================
print("[cs_rubber] 3. stored gap 0: no rubber term and no node")
CTX_GAP0 = build(write_car("gap0.car", {7: 0.0}))
CTX_GAP0_NOTAB = build(write_car("gap0_notab.car", {7: 0.0}, omit=(13,)))   # the front right rubber table left out
# The corner is a right one: a left gap of 0 takes the gap of its right sibling (load_car), so it keeps its rubber.
ok("stored gap 0 (fr): rubber_gap is None", V.rubber_gap(CTX_GAP0.cs, "fr", CTX_GAP0.vp) is None)
ok("stored gap 0 (fr): the front left corner keeps its own gap and rubber",
   V.rubber_gap(CTX_GAP0.cs, "fl", CTX_GAP0.vp) == 0.008)
rub_zero = V._rubber(CTX_GAP0.cs, "fr", CTX_GAP0.vp)
travels = [-0.05, -0.01, 0.0, 0.003, 0.02, 0.08, 0.15]
ok("stored gap 0 (fr): the rubber force is exactly 0.0 at 7 travels from -50 to +150 mm",
   all(isinstance(rub_zero(t), float) and rub_zero(t) == 0.0 for t in travels))
x_g = static_x(CTX_GAP0, 30.0)
x_g[9] = -0.01
d_with = dx_at(CTX_GAP0, x_g)
d_without = dx_at(CTX_GAP0_NOTAB, x_g)
diff_dx = float(np.max(np.abs(d_with - d_without)))
ok(f"stored gap 0 (fr): dx equals a car without the rubber table (max diff {diff_dx:.2e})",
   diff_dx <= 1e-12 * max(1.0, float(np.max(np.abs(d_without)))))

# ==== 4. negative effective gap: the rubber is loaded at zero travel =========================================
print("[cs_rubber] 4. negative effective gap")
gap_fl_eff = V.rubber_gap(cs0, "fl", vp0)
ok(f"flag 0 fl effective gap is negative ({gap_fl_eff*1e3:.2f} mm)", gap_fl_eff < 0.0)
f_zero = model_rubber(CTX_FLAG0, "fl", 0.0)
f_hand = float(hand_rubber(cs0, "fl", gap_fl_eff, 0.0))
ok(f"negative effective gap: rubber force at zero travel {f_zero:.2f} N > 0 and equal to the hand value {f_hand:.2f} N",
   f_zero > 0.0 and abs(f_zero - f_hand) <= 1e-6 * abs(f_hand))

# ==== 5. model rubber force = mr(w) x table(sm(w) - gap_eff) =================================================
print("[cs_rubber] 5. rubber force against the hand formula")
gap_fr_eff = V.rubber_gap(cs0, "fr", vp0)
worst, peak = 0.0, 0.0
for w in np.linspace(gap_fr_eff - 0.005, gap_fr_eff + 0.02, 6):
    model = model_rubber(CTX_FLAG0, "fr", float(w))
    hand = float(hand_rubber(cs0, "fr", gap_fr_eff, float(w)))
    worst, peak = max(worst, abs(model - hand)), max(peak, abs(hand))
# the table is a B-spline on ~200 nodes (vehModel_cs._curve), so the error is judged against the peak force
ok(f"fr rubber force at 6 travels equals mr(w) x table(sm(w) - gap_eff): max error {worst:.2e} N of peak "
   f"{peak:.1f} N (onset width {V.W_RUBBER*1e3:.1f} mm)", worst <= 1e-5 * max(peak, 1.0))

# ==== 6. QSS axle force uses the same effective gap =========================================================
print("[cs_rubber] 6. QSS envelope")
calls = []
_orig_rubber = V._rubber


def _spy_rubber(cs, c, vp=None):
    calls.append((c, vp))
    return _orig_rubber(cs, c, vp)


V._rubber = _spy_rubber                                  # build_envelope_cs imports _rubber at call time
try:
    QSS0 = quiet(MLTP_screen, car=f"chassissim:{CAR_FLAG0}", circuit=f"chassissim:{CIRC}",
                 chassissim_dir=INSTALL, physics="corrected", save=False, verbose=False)
finally:
    V._rubber = _orig_rubber
q_corners = sorted({c for c, vp in calls if vp is not None})
ok(f"QSS envelope builds the rubber of all four corners with vp ({q_corners})",
   q_corners == sorted(CORNERS) and getattr(QSS0, "envelope", None) is not None)
qvp = QSS0.vp
q_eff = hand_eff_gap(cs0, qvp, "fr")
w_q = q_eff + 0.005
xq = ca.SX.sym("x")
q_model = float(ca.Function("R", [xq], [_orig_rubber(cs0, "fr", qvp)(xq)])(w_q))
q_hand = float(hand_rubber(cs0, "fr", q_eff, w_q))
ok(f"QSS axle rubber at one travel (fr, {w_q*1e3:.2f} mm): {q_model:.4f} N = hand {q_hand:.4f} N with the "
   f"effective gap",
   abs(q_model - q_hand) <= 1e-6 * abs(q_hand))

# ==== 7. refusals: flag 2; third spring and third rubber build (task 12) ====================================
print("[cs_rubber] 7. refusals and third-element builds")
hit, msg = refusal(build, write_car("flag2.car", {204: 2}))
ok(f"gap flag 2 raises ValueError with the STE text: {msg!r}", hit and msg == REFUSE_FLAG)
CTX_T5 = build(write_car("third5.car", {5: (1, 0.0, 9.0e4, np.array([0.0, 0.05]), np.array([0.0, 500.0]))}))
ok(f"F3-like: third spring rate 9e4 (cid 5) builds (nx {CTX_T5.m23.nx}) with a front third element",
   CTX_T5.m23.nx == 23 and V.third_element(CTX_T5.cs, "f", CTX_T5.vp) is not None)
CTX_T6 = build(write_car("third_rubber_r.car", {11: 0.027, 12: 0.027, 208: 0, 209: 0,
                                               18: (0, 0.0, 4.0e4, np.array([0.0, 0.02]), np.array([0.0, 800.0]))}))
ok(f"F2-like: rear third rubber, flag 0, gap 27 mm (cid 12), builds (nx {CTX_T6.m23.nx}) with a rear third element",
   CTX_T6.m23.nx == 23 and V.third_element(CTX_T6.cs, "r", CTX_T6.vp) is not None
   and CTX_T6.cs.susp.gap_flag_third_r == 0 and abs(CTX_T6.cs.susp.gap_third_r - 0.027) < 1e-12)
hit, msg = refusal(build, write_car("flag0_rate0.car", {204: 0, 205: 0, 206: 0, 207: 0,
                                                       2: (1, 0.02, 0.0, np.array([0.0, 0.05]), np.array([0.0, 500.0]))}))
ok(f"flag 0 with a zero spring rate (fl) raises ValueError naming the rate: {msg!r}",
   hit and "has no rate" in msg and "corner fl" in msg)


# ==== 8. bump rubber hysteresis (ChassisSim phase 2b, task 15) ===============================================
print("[cs_rubber] 8. bump rubber hysteresis")
HYST_IDX = np.arange(21)
HYST_X = 0.001 * HYST_IDX          # m, compression after the gap: 0 to 20 mm
HYST_LOAD = 5.0e5 * HYST_X         # N, loading curve: 10 kN at 20 mm


def loop_curve(lo_idx, hi_idx):
    """Return (x, loading, unloading): unloading = 0.5 x loading on grid points lo_idx to hi_idx, else equal."""
    fac = np.where((HYST_IDX >= lo_idx) & (HYST_IDX <= hi_idx), 0.5, 1.0)
    return HYST_X, HYST_LOAD, HYST_LOAD * fac


LOOP = loop_curve(4, 8)            # loop from 4 to 8 mm (the forces meet again at 9 mm)
SCALE = np.r_[[2.0, 0.3, 0.1, 0.5, 0.02], [3.0] * 4, [0.02, 0.2, 0.002, 0.02, 0.002, 0.02], [0.002] * 8]


HYST_ORDER = ("fr", "fl", "rr", "rl")   # corner order of cids 657 to 660 (flags) and 661 to 664 (curves)


def hyst_car(name, curves, flags, extra=()):
    """Synthetic car with the hysteresis records. curves maps a corner to (x, loading, unloading), flags maps a
    corner to its enable flag. A corner that is not named holds flag 0 and no curve. extra holds more records."""
    R = build_records()
    end = R.pop()
    R += list(extra)
    for i, c in enumerate(HYST_ORDER):
        R.append((6, 657 + i, flags.get(c, 0)))
    for i, c in enumerate(HYST_ORDER):
        if c in curves:
            x, lo, up = curves[c]
            R.append((4, 661 + i, np.concatenate([x, lo, up])))
    R.append(end)
    path = os.path.join(TMP.name, name)
    carfile.write(R, path)
    return path


def rubber_sx(cs, vp, c, w, xsdot=None):
    """Model bump rubber force [N] of corner c at strut travel w [m]. xsdot None gives the loading form (the QSS)."""
    x = ca.SX.sym("x")
    if xsdot is None:
        return float(ca.Function("R", [x], [V._rubber(cs, c, vp)(x)])(w))
    v = ca.SX.sym("v")
    return float(ca.Function("R", [x, v], [V._rubber(cs, c, vp)(x, v)])(w, xsdot))


def hyst_hand(cs, c, w, curve):
    """Hand values at strut travel w for a flag-1 corner c with the hysteresis curve (x, loading, unloading).
    Return (compression after the gap, loading force, unloading force, mr). The loading force is mr x PCHIP of the
    loading column at the compression. The unloading force is the loading force - mr x PCHIP of
    max(loading - unloading, 0), as the model builds it. The curve starts at the origin."""
    gap = float(getattr(cs.susp, f"gap_{c}"))
    mrc = np.asarray(getattr(cs.susp, f"mr_{c}"), float)
    sm = sum(v / (i + 1) * w ** (i + 1) for i, v in enumerate(mrc))
    mr = float(sum(v * w ** i for i, v in enumerate(mrc)))
    comp = float(V.W_RUBBER * np.log1p(np.exp((sm - gap) / V.W_RUBBER)))
    x, lo, up = curve
    Lh = float(PchipInterpolator(x, lo)(comp))
    return (comp, mr * Lh, mr * (Lh - float(PchipInterpolator(x, np.maximum(lo - up, 0.0))(comp))), mr)


# 8.2 model force inside the loop, with the sign of xsdot
PATH_H1 = hyst_car("hyst_loop.car", {"fl": LOOP}, {"fl": 1})
CTX_H1 = build(PATH_H1)
cs_h, vp_h = CTX_H1.cs, CTX_H1.vp
w_h = brentq(lambda w: hyst_hand(cs_h, "fl", w, LOOP)[0] - 0.006, 0.0, 0.05)
comp_h, L, U, mr_h = hyst_hand(cs_h, "fl", w_h, LOOP)
ok(f"compression at the test travel {comp_h*1e3:.4f} mm, inside the loop, and loading > unloading > 0 "
   f"({L:.2f} N > {U:.2f} N)", abs(comp_h - 0.006) < 1e-9 and L > U > 0.0)
F_load = rubber_sx(cs_h, vp_h, "fl", w_h)
F_pos = rubber_sx(cs_h, vp_h, "fl", w_h, 1.0)
F_neg = rubber_sx(cs_h, vp_h, "fl", w_h, -1.0)
F_zero = rubber_sx(cs_h, vp_h, "fl", w_h, 0.0)
ok(f"no xsdot gives the loading force: {F_load:.4f} N vs hand {L:.4f} N",
   abs(F_load - L) <= 1e-6 * abs(L))
ok(f"xsdot > 0 (compressing) gives the loading force: {F_pos:.4f} N vs hand {L:.4f} N",
   abs(F_pos - L) <= 1e-6 * abs(L))
# The loop table (loading - unloading) has steps at 4 mm and 9 mm. Its B-spline (vehModel_cs._curve) rings a
# little next to a step, so the unloading and rest checks use 1e-5. The loading table agrees to 2e-7.
ok(f"xsdot < 0 (extending) gives the unloading force L - D: {F_neg:.4f} N vs hand {U:.4f} N",
   abs(F_neg - U) <= 1e-5 * abs(U))
S_REST = 0.5 * (1 - np.tanh(2.0))       # blend weight at rest: 1.8 % of the loop
ok(f"xsdot = 0 gives the loading force less 1.8 % of the loop: {F_zero:.4f} N vs hand {L - (L - U) * S_REST:.4f} N "
   f"(loading {L:.4f} N)",
   abs(F_zero - (L - (L - U) * S_REST)) <= 1e-5 * abs(L) and 0.0 < L - F_zero < 0.02 * (L - U))

# 8.3 blend weight around the switch speed: s = 0.5 at xsdot = -V0_HYST / mr, 0.5 (1 -+ tanh(1)) at V_HYST / mr on each side
ok(f"V0_HYST = 2 V_HYST ({V.V0_HYST} m/s, {V.V_HYST} m/s)", V.V0_HYST == 2 * V.V_HYST)
weight = lambda v: (rubber_sx(cs_h, vp_h, "fl", w_h, v / mr_h) - F_load) / (F_neg - F_load)
s_mid, s_plus, s_minus = weight(-V.V0_HYST), weight(-V.V0_HYST + V.V_HYST), weight(-V.V0_HYST - V.V_HYST)
ok(f"blend at the damper speed -V0_HYST: s {s_mid:.12f} = 0.5 (rel 1e-9)", abs(s_mid - 0.5) <= 1e-9 * 0.5)
ok(f"blend at -V0_HYST + V_HYST: s {s_plus:.12f} = 0.5 (1 - tanh 1) (rel 1e-9)",
   abs(s_plus - 0.5 * (1 - np.tanh(1.0))) <= 1e-9 * 0.5 * (1 - np.tanh(1.0)))
ok(f"blend at -V0_HYST - V_HYST: s {s_minus:.12f} = 0.5 (1 + tanh 1) (rel 1e-9)",
   abs(s_minus - 0.5 * (1 + np.tanh(1.0))) <= 1e-9 * 0.5 * (1 + np.tanh(1.0)))

# 8.4 flag 0 on the same car: the plain rubber, and dx equal to a car without the curves
CTX_H0 = build(hyst_car("hyst_fl0.car", {"fl": LOOP}, {"fl": 0}))
CTX_HN = build(hyst_car("hyst_nocurve.car", {}, {}))
plain = hand_rubber(CTX_H0.cs, "fl", float(CTX_H0.cs.susp.gap_fl), w_h)
peak = max(abs(plain), 1.0)
F0_pos = rubber_sx(CTX_H0.cs, CTX_H0.vp, "fl", w_h, 1.0)
F0_neg = rubber_sx(CTX_H0.cs, CTX_H0.vp, "fl", w_h, -1.0)
ok(f"flag 0: the plain rubber at both signs of xsdot ({F0_pos:.4f}, {F0_neg:.4f} N) vs hand {plain:.4f} N, "
   f"and not the curve (loading {L:.1f} N)",
   abs(F0_pos - plain) <= 1e-5 * peak and abs(F0_neg - plain) <= 1e-5 * peak and abs(plain - L) > 1e-3 * abs(L))
rng = np.random.default_rng(15)
states = [static_x(CTX_HN, 30.0) + rng.uniform(-1.0, 1.0, 23) * SCALE for _ in range(6)]
diff_dx = max(float(np.max(np.abs(dx_at(CTX_H0, x) - dx_at(CTX_HN, x)))) for x in states)
size_dx = max(float(np.max(np.abs(dx_at(CTX_HN, x)))) for x in states)
ok(f"flag 0: dx equals a car without the curves at 6 random states (max diff {diff_dx:.2e} of {size_dx:.2e})",
   diff_dx <= 1e-12 * max(1.0, size_dx))

# 8.5 QSS: the envelope calls the rubber with no xsdot, so it gets the loading curve
calls = []
_plain_rubber = V._rubber


def _spy_hyst(cs, c, vp=None):
    f = _plain_rubber(cs, c, vp)

    def g(xs, xsdot=None):
        if c == "fl":
            calls.append(xsdot is None)
        return f(xs) if xsdot is None else f(xs, xsdot)
    return g


V._rubber = _spy_hyst
try:
    QSS_H = quiet(MLTP_screen, car=f"chassissim:{PATH_H1}", circuit=f"chassissim:{CIRC}",
                  chassissim_dir=INSTALL, physics="corrected", save=False, verbose=False)
finally:
    V._rubber = _plain_rubber
F_qss = rubber_sx(cs_h, QSS_H.vp, "fl", w_h)
ok(f"QSS: the fl rubber is called {len(calls)} times, all with no xsdot", len(calls) > 0 and all(calls))
ok(f"QSS: the axle rubber force at the test travel is the loading curve: {F_qss:.4f} N vs hand {L:.4f} N",
   abs(F_qss - L) <= 1e-6 * abs(L))

# 8.6 refusals
recs = build_records()
recs.insert(-1, (4, 657, np.zeros(33)))             # flag cid 657 (fr) with tag 4 (the curve tag)
bad_tag = os.path.join(TMP.name, "hyst_bad_tag.car")
carfile.write(recs, bad_tag)
hit, msg = refusal(load_car, bad_tag)
ok(f"cid 657 with tag 4 raises ValueError naming the record and the tag: {msg[:70]!r}",
   hit and "rubber hysteresis record 657 with tag 4" in msg)
hit, msg = refusal(check_car, load_car(hyst_car("hyst_flag1_nocurve.car", {}, {"fl": 1})))
ok(f"flag 1 without its curve: check_car raises naming susp.hyst_fl: {msg[:70]!r}",
   hit and "susp.hyst_fl" in msg)
E3 = np.array([])
X11 = 0.002 * np.arange(11)
L11 = 5.0e5 * X11
BAD = {"empty": (E3, E3, E3),
       "zero (uninitialised)": (np.zeros(11), np.zeros(11), np.zeros(11)),
       "tiny (uninitialised)": (np.full(11, 1e-304), np.full(11, 1e-304), np.zeros(11)),
       "NaN": (X11, np.where(np.arange(11) == 5, np.nan, L11), 0.5 * L11),
       "descending x": (X11[::-1], L11[::-1], 0.5 * L11[::-1])}
for i, (what, curve) in enumerate(BAD.items()):
    hit, msg = refusal(load_car, hyst_car(f"hyst_bad_{i}.car", {"rr": curve}, {"rr": 1}))
    ok(f"flag 1 with a curve that is {what}: load_car raises naming corner rr and cid 663",
       hit and "corner rr (cid 663)" in msg and "cannot use" in msg)
ok("the same curves with flag 0 load",
   all(load_car(hyst_car(f"hyst_bad0_{i}.car", {"rr": c}, {})).susp.hyst_rr is not None for i, c in enumerate(BAD.values())))
hit, msg = refusal(load_car, hyst_car("hyst_flag2.car", {"fr": LOOP}, {"fr": 2}))
ok(f"flag 2 raises ValueError naming corner fr and cid 657: {msg[-70:]!r}",
   hit and "corner fr (cid 657) other than 0 or 1" in msg)
for cid in (638, 639):
    hit, msg = refusal(load_car, hyst_car(f"hyst_third_{cid}.car", {}, {}, extra=[(6, cid, 1)]))
    ok(f"third-spring hysteresis flag cid {cid} = 1 raises ValueError naming the cid",
       hit and f"third-spring hysteresis flag (cid {cid})" in msg)
ok("third-spring hysteresis flags 0 load",
   load_car(hyst_car("hyst_third_0.car", {}, {}, extra=[(6, 638, 0), (6, 639, 0)])).susp.hyst_flag_third_r == 0)

# 8.7 the hysteresis term only removes energy: dF/dxsdot >= 0. Curve C1 has steps in the loop, curve C4 a loop
# that closes at the table end. A curve with the two columns of C1 swapped (unloading above loading) raises.
X_C1 = np.linspace(0.0, 0.018, 10)
L_C1 = 1.0e6 * X_C1
C1 = (X_C1, L_C1, L_C1 * (1 - 0.4 * ((X_C1 >= 0.004) & (X_C1 <= 0.012))))
X_C4 = np.linspace(0.0, 0.02, 11)
L_C4 = 5.0e7 * X_C4 ** 2
C4 = (X_C4, L_C4, L_C4 * (1 - 0.35 * np.sin(np.pi * X_C4 / 0.02)))
for name, curve in (("C1", C1), ("C4", C4)):
    car = load_car(hyst_car(f"hyst_pass_{name}.car", {"fl": curve}, {"fl": 1}))
    xw, xv = ca.SX.sym("xw"), ca.SX.sym("xv")
    Fr = V._rubber(car, "fl")(xw, xv)
    w_lo = brentq(lambda w: hyst_hand(car, "fl", w, curve)[0] - 1e-5, -0.05, 0.05)
    w_hi = brentq(lambda w: hyst_hand(car, "fl", w, curve)[0] - (curve[0][-1] + 0.006), 0.0, 0.2)
    Wg, Vg = np.meshgrid(np.linspace(w_lo, w_hi, 2001), np.r_[-1.0, -0.2, np.linspace(-0.08, 0.04, 61), 0.2, 1.0])
    out = ca.Function("d", [xw, xv], [ca.jacobian(Fr, xv), Fr]).map(Wg.size)(Wg.reshape(1, -1), Vg.reshape(1, -1))
    dF, Fv = np.asarray(out[0]).reshape(Wg.shape), np.asarray(out[1]).reshape(Wg.shape)
    ok(f"{name}: min dF/dxsdot {dF.min():.3e} N s/m >= 0 on {Wg.size} points (compression up to 6 mm past the "
       f"table end), max {dF.max():.3e}; extending force - compressing force at most {np.max(Fv[0] - Fv[-1]):.2e} N",
       dF.min() >= 0.0 and dF.max() > 1e3 and np.max(Fv[0] - Fv[-1]) <= 0.0)
hit, msg = refusal(load_car, hyst_car("hyst_swapped.car", {"rl": (C1[0], C1[2], C1[1])}, {"rl": 1}))
ok(f"C1 with unloading above loading raises ValueError naming corner rl and cid 664: {msg[-80:]!r}",
   hit and "corner rl (cid 664)" in msg and "unloading force above the loading force" in msg)


# ==== vendor ================================================================================================
def find_car(folder, pattern):
    """First car file of folder that matches pattern (sorted), or a missing path when none does."""
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    return Path(hits[0]) if hits else Path(folder) / "missing.car"


def vendor(cs_dir):
    models = Path(cs_dir).parent / "Models"
    print("[cs_rubber] vendor: GP3 2013, F2, GT3, F3")
    gp3 = find_car(models / "GP3 2013" / "Eastern Creek, AU", "def*.car")
    f2 = find_car(models / "Formula 2" / "Adelaide, AU", "*.car")
    gt3 = find_car(models / "GT3" / "Adelaide, AU", "*.car")
    f3 = find_car(models / "Formula 3", "*.car")
    for label, p in (("GP3 2013", gp3), ("Formula 2", f2), ("GT3", gt3), ("Formula 3", f3)):
        ok(f"{label} car file found by folder glob", p.is_file())

    cs_gp3 = load_car(str(gp3))
    g = {c: V.rubber_gap(cs_gp3, c) for c in CORNERS}
    want = {"fl": 0.007, "fr": 0.007, "rl": 0.016, "rr": 0.016}
    ok("GP3 effective gaps (flag 1) 7/7/16/16 mm within 0.5 mm: "
       + " ".join(f"{g[c]*1e3:.2f}" for c in CORNERS),
       all(g[c] is not None and abs(g[c] - want[c]) < 5e-4 for c in CORNERS))

    ctx_f2 = build(str(f2), circuit="Adelaide", chassis_dir=cs_dir)
    e = {c: hand_eff_gap(ctx_f2.cs, ctx_f2.vp, c) for c in CORNERS}
    ok("F2 builds (physics corrected, Adelaide): nx 23 and nu 3", ctx_f2.m23.nx == 23 and ctx_f2.m23.nu == 3)
    ok("F2 effective gaps about 0 / 0 / 16 / 16 mm within 0.5 mm: " + " ".join(f"{e[c]*1e3:.2f}" for c in CORNERS),
       abs(e["fl"]) < 5e-4 and abs(e["fr"]) < 5e-4 and abs(e["rl"] - 0.016) < 5e-4 and abs(e["rr"] - 0.016) < 5e-4)
    wt = corner_weights(ctx_f2.cs)
    ld = tyre_loads(ctx_f2, static_x(ctx_f2))
    err = float(np.max(np.abs(ld - wt) / wt))
    ok(f"F2 static loads equal the corner weights within 0.5 %: max {100 * err:.4f} %", err < 0.005)

    ctx_gt3 = build(str(gt3), circuit="Adelaide", chassis_dir=cs_dir)
    e = {c: V.rubber_gap(ctx_gt3.cs, c, ctx_gt3.vp) for c in CORNERS}
    ok("GT3 effective gaps: front about 37.1 mm within 0.5 mm, rear None: "
       + " ".join("None" if e[c] is None else f"{e[c]*1e3:.2f}" for c in CORNERS),
       e["fl"] is not None and e["fr"] is not None and abs(e["fl"] - 0.0371) < 5e-4
       and abs(e["fr"] - 0.0371) < 5e-4 and e["rl"] is None and e["rr"] is None)
    wt = corner_weights(ctx_gt3.cs)
    ld = tyre_loads(ctx_gt3, static_x(ctx_gt3))
    err = float(np.max(np.abs(ld - wt) / wt))
    ok(f"GT3 static loads equal the corner weights within 0.5 %: max {100 * err:.4f} %", err < 0.005)

    # The F3 car stops at its front axle datum (kinematics), before the third element is read.
    hit, msg = refusal(build, str(f3), circuit="Adelaide", chassis_dir=cs_dir)
    ok(f"F3 raises ValueError naming the front axle ground datum: {msg[:60]!r}",
       hit and "ground datum of the front axle" in msg)


def vendor_hysteresis(cs_dir):
    """Every shipped car in the Models folder loads, and no bump rubber hysteresis flag is on. No file name is printed."""
    files = sorted(glob.glob(str(Path(cs_dir).parent / "Models" / "**" / "*.car"), recursive=True))
    ok(f"the Models folder holds shipped car files ({len(files)} found)", len(files) > 0)
    flags = []
    for p in files:
        car = load_car(p)
        flags += [getattr(car.susp, f"hyst_flag_{c}") for c in CORNERS]
    ok(f"all {len(files)} shipped cars load, every bump rubber hysteresis flag 0 or absent",
       all(f in (0, None) for f in flags))


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
    vendor_hysteresis(CS)
print(f"{PASS} checks passed")
