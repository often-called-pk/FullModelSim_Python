"""ChassisSim phase 2b, task 12: the third (heave) element of the car model and its QSS (tests/test_cs_third.py).

The synthetic car (arbitrary values, written with carfile.write, cids and tags literal here) has a third spring, a
third bump rubber and a third damper on the front axle. The checks compare the model with hand values: the heave
h = sm3(mean wheel travel), each wheel's share 0.5 F3 mr3, the rubber onset (gap flag 1 and flag 0), the gap rules
(0 and at least 0.5 m), the damper sign (bump for a heave velocity up, rebound for one down), the QSS axle table
(F3 mr3 at equal travel) and the no-element case (the same dx as a car without the cids). The models are built with
MLTP and a fake IPOPT (no solve). The vendor section (last) runs only when CHASSISSIM_DIR names the ChassisSim install
folder. It reads the F2 (rear third rubber) and the GP3 2013 car and checks relations only. It writes nothing into the
vendor folders.

    python tests/test_cs_third.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_third.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import copy
import glob
import io
import os
import tempfile
import warnings
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import PchipInterpolator

import functions.transcription as T
import vehModel_cs as V
from functions import carfile
from functions.ggv import build_envelope_cs
from MLTP import MLTP

PASS = 0
TOL_SPLINE = 1e-5             # relative: the rubber table is a B-spline on ~200 nodes (vehModel_cs._curve)
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair. An over or omit cid is a file cid.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}
G = 9.81
W = 1.0e-3                     # m, rubber onset width (softplus); literal here, the hand value of the onset
K3 = 9.0e4                     # N/m, third spring rate of the ALL car (cid 5)
C3 = (1.0, 0.3)                # third motion-ratio coefficients (cid 29): mr3(w) = 1 + 0.3 w, sm3(w) = w + 0.15 w^2
GAP3 = 0.02                    # m, third rubber gap of the ALL car (cid 11, flag 208 = 1)
SLOPE = 4.0e4                  # N/m, third rubber table (cid 17): 0 N at 0 m, 800 N at 0.02 m (linear)
VEL3 = np.array([0.0, 100.0, 400.0])     # third damper velocity, mm/s (cid 35)
BUMP3 = np.array([0.0, 900.0, 3600.0])   # N
REB3 = np.array([0.0, 600.0, 2400.0])    # N, positive as in the record
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
HP_LEFT = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.68, 0.11],
                    [-0.15, 0.30, 0.28], [0.15, 0.30, 0.28], [0.0, 0.64, 0.33]])
HP_RIGHT = HP_LEFT * np.array([1.0, -1.0, 1.0])
SPRING = {1: (1.23e5, 0.020), 2: (1.31e5, 0.021), 3: (0.97e5, 0.019), 4: (1.02e5, 0.0205)}   # corners: rate, preload
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}
ARB = {19: 2.0e4, 21: 1.5e4}
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: 1.5, 92: 1.5, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,
    100: 0.9, 102: 1000.0, 103: 9000.0,
    107: 14000.0, 108: 0.6, 109: 0.05,
    111: 0.0, 112: 1.0, 113: 1.0,
    55: 0.03, 59: 0.03, 63: 0.03, 67: 0.03,
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,
    390: 350.0,
    7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095,        # corner ground gaps (flag 1)
    11: 0.0, 12: 0.0,                                 # third gaps: 0, no third rubber unless over
}
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})
EXPLICIT.update({142: 0.5 * (EXPLICIT[117] + EXPLICIT[118]) * 1e-3,
                 144: 0.5 * (EXPLICIT[119] + EXPLICIT[120]) * 1e-3, 56: 0.0, 60: 0.0, 64: 0.0, 68: 0.0})
EXPLICIT.update({c: 0.0 for c in (55, 59, 63, 67) if c not in EXPLICIT})
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
AERO_X = 0.005 * np.arange(7)
AERO_Z = 0.002 + 0.005 * np.arange(7)
MAP_X = np.array([-10000.0, -5000.0, -3333.0, -1667.0, 0.0, 1667.0, 3333.0, 5000.0, 6667.0, 8333.0, 10000.0])
MAP_Z = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
MAP_DATA = np.where((MAP_X > 0)[:, None] & (MAP_Z >= 0.1)[None, :], 0.1 * MAP_X[:, None], 0.0)


def ok(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASS += 1


def quiet(fn, *a, **kw):
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
    return np.sin(np.arange(n) * 0.9 + s) * 7.3 - 0.61


def tyre_sizes(d):
    d1, d2, d3, d4, d5, d6, d7 = d

    def g(a, b):
        return a + b + a * b
    return [2 * d3, 2 * d4, 2 * d4, g(d1, d2), g(d6, d7), 2 * d6, 2 * d4, g(d1, d2), 2 * d3, 2 * d6, 2 * d5, 2 * d5, 2 * d5]


def tyre_record(i, corner):
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


def table_rec(flag, pre, rate, xs, ys):
    """Gap or rubber record (tag 11): flag, preload, rate, x, y."""
    return (flag, pre, rate, np.asarray(xs, float), np.asarray(ys, float))


def build_records(over=None, omit=()):
    """Records of the synthetic car as {cid: (tag, value)}. Default: no third element (third rates 0, third gaps 0,
    third damper table empty). over replaces or adds cids (value as (tag, value)); omit drops cids."""
    rec = {}

    def put(tag, cid, v):
        cid = FILE_CID.get(cid, cid)
        rec[cid] = (tag, v)
    for cid, v in EXPLICIT.items():
        put(1, cid, v)
    for cid in (53, 61, 69):
        put(6, cid, 0)                                   # geometry mode 0 (53, 61), no live axle (69)
    for cid, (rate, pre) in SPRING.items():
        put(11, cid, table_rec(1, pre, rate, [0.0, 0.05], [0.0, 500.0]))
    put(11, 5, table_rec(1, 0.0, 0.0, [0.0, 0.05], [0.0, 500.0]))             # third spring f: rate 0
    put(11, 6, table_rec(1, 0.0, 0.0, [0.0, 0.05], [0.0, 500.0]))             # third spring r: rate 0
    for cid, (rate, pre) in RUBBER.items():
        put(11, cid, table_rec(1, pre, rate, [0.0, 0.02], [0.0, 800.0]))
    for cid, rate in ARB.items():
        put(11, cid, table_rec(1, 0.0, rate, [0.0, 0.1], [0.0, 300.0]))
    for cid in range(23, 31):
        put(2, cid, np.array([1.0 + 0.05 * np.sin(cid), 0.0, 0.0, 0.0, 0.0]))
    put(2, 29, np.array([C3[0], C3[1], 0.0, 0.0, 0.0]))                       # third motion ratio f
    for cid in range(366, 372):
        put(6, cid, 10 + cid - 366)
    for cid in range(204, 210):
        put(6, cid, 1)                                                        # rubber gap flags (corners, third f/r)
    put(7, 101, (ENGINE_RPM, ENGINE_NM))
    put(2, 110, GEARS)
    for cid in (74, 75, 76, 77):
        put(7, cid, (BUMP_X, np.zeros(len(BUMP_X))))
    put(9, 114, (0.5 * np.arange(4), 1.0 * np.arange(4) + 0.1, arb(16, 114)))
    put(9, 116, (MAP_X, MAP_Z, MAP_DATA.ravel()))
    for cid in (127, 131, 135):
        put(9, cid, (AERO_X, AERO_Z, arb(49, cid)))
    for cid in range(31, 35):
        put(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    put(12, 35, (1, arb(6, 35), np.array([])))                                # third front damper: empty table
    for cid, rows in ((49, HP_LEFT), (50, HP_RIGHT), (51, HP_LEFT), (52, HP_RIGHT)):
        put(10, cid, (0, np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])))
    for i, c in enumerate(CORNERS):
        _, cid, v = tyre_record(i, c)
        put(13, cid, v)
    for cid, tv in (over or {}).items():
        rec[cid] = tv
    for cid in omit:
        rec.pop(cid, None)
    out = [(t, c, v) for c, (t, v) in sorted(rec.items())]
    out.append((carfile.END_TAG, carfile.END_CID, None))
    return out


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


def build(car_path, chassis_dir=INSTALL, circuit=CIRC):
    """MLTP on the car with a fake IPOPT (no solve). Returns ctx with ctx.m23, ctx.cs and ctx.vp."""
    return quiet(MLTP, car=f"chassissim:{car_path}", circuit=f"chassissim:{circuit}", chassissim_dir=chassis_dir,
                 physics="corrected", save=False, plot=False, linear_solver="mumps")


# ---- car variants ---------------------------------------------------------------------------------------------
# ALL: third spring 9e4 N/m, third rubber flag 1 at a 20 mm gap (linear table, 4e4 N/m), third damper, mr3 = (1, 0.3)
ALL_OVER = {
    5: (11, table_rec(1, 0.0, K3, [0.0, 0.05], [0.0, 500.0])),
    11: (1, GAP3),
    17: (11, table_rec(1, 0.0, 2.0e5, [0.0, GAP3], [0.0, SLOPE * GAP3])),
    35: (12, (1, arb(6, 35), np.concatenate([VEL3, BUMP3, REB3]))),
}
# NONE: the third rubber, damper and flag cids left out (the third spring stays at rate 0: check_car needs cids 5, 6)
NONE_OMIT = (11, 12, 17, 18, 35, 36, 208, 209)
# ZERO: the third cids present with zero values (rates 0, gaps 0, empty damper tables); no third element
ZERO_OVER = {
    35: (12, (1, arb(6, 35), np.array([]))),
    36: (12, (1, arb(6, 36), np.array([]))),
}
# RUB0: flag-0 third rubber at a 20 mm stored gap, rate-based law (4e4 N/m), no third spring, no damper
RUB0_OVER = {
    11: (1, GAP3),
    208: (6, 0),
    17: (11, table_rec(0, 0.0, SLOPE, [0.0, GAP3], [0.0, SLOPE * GAP3])),
    35: (12, (1, arb(6, 35), np.array([]))),
}


def sm3(w):
    """Third spring travel of the heave law: sm3(w) = w + 0.15 w^2 (hand, from C3)."""
    return sum(c / (i + 1) * w ** (i + 1) for i, c in enumerate(C3))


def mr3(w):
    """Third motion ratio mr3(w) = 1 + 0.3 w (hand, from C3)."""
    return sum(c * w ** i for i, c in enumerate(C3))


def soft(z):
    """W log(1 + exp(z / W)): the softplus onset of the rubber, written from the formula."""
    return W * np.log1p(np.exp(z / W))


def rub_hand(h, gap=GAP3):
    """Flag 1 rubber force [N] at heave h: linear table of slope SLOPE on the softplus compression."""
    return SLOPE * soft(h - gap)


def F3_hand(h):
    """Third element force [N] at heave h for the ALL car: spring plus rubber. No damper at steady state."""
    return K3 * h + rub_hand(h)


# ---- state and evaluation helpers --------------------------------------------------------------------------------
def static_x(ctx, v=30.0):
    """Straight line at v, tyre states at W0 / kt, zs = 0 (every corner travel 0)."""
    vp, cs = ctx.vp, ctx.cs
    x = np.zeros(23)
    x[0] = v
    x[5:7] = v / cs.tyre.fl["radius"]
    x[7:9] = v / cs.tyre.rl["radius"]
    x[19:23] = [getattr(vp, f"W{c}0") / vp.kt for c in CORNERS]
    return x


def state_zs(ctx, zs, phi=0.0):
    x = static_x(ctx)
    x[9] = zs
    x[13] = phi
    return x


def fzs4(ctx, x):
    """Suspension force of the four corners [N] at the physical state x."""
    m = ctx.m23
    f = ca.Function("Z", [m.x, m.u, m.pv], [m.fzs_fl, m.fzs_fr, m.fzs_rl, m.fzs_rr])
    return np.array([float(v) for v in f.call([np.asarray(x, float) / m.x_s, np.zeros(m.nu), 0.0])])


def dx_at(ctx, x):
    m = ctx.m23
    f = ca.Function("f", [m.x, m.u, m.pv], [m.dx])
    return np.asarray(f.call([np.asarray(x, float) / m.x_s, np.zeros(m.nu), 0.0])[0], float).ravel()


def sx_eval(fn, v):
    """Value of a casadi-compatible scalar function fn at the float v."""
    s = ca.SX.sym("s")
    return float(ca.Function("F", [s], [fn(s)])(v))


def qss_axle_tables(ctx):
    """The axle force functions ('axle') that build_envelope_cs builds: front first, then rear."""
    orig = ca.Function
    got = []

    def spy(name, *a, **k):
        f = orig(name, *a, **k)
        if name == "axle":
            got.append(f)
        return f
    ca.Function = spy
    try:
        quiet(build_envelope_cs, ctx)
    finally:
        ca.Function = orig
    return got


def axle_at(fn, xs):
    xs = np.asarray(xs, float)
    return np.asarray(fn.map(xs.size)(xs.reshape(1, -1))).ravel()


def coeff_sm(c):
    c = np.asarray(c, float)
    return lambda w: sum(v / (i + 1) * w ** (i + 1) for i, v in enumerate(c))


def static_heave(cs, axle, corners, sm):
    """Heave of the third element at static [m], from the car record and the sprung corner loads only:
    sm of the mean wheel travel (F0 / (mr0 k) - preload) / mr0 over the two corners of the axle."""
    ms = cs.mass
    mb = ms.total - ms.unsprung_front - ms.unsprung_rear
    wB = (ms.front_share * ms.total - ms.unsprung_front) / mb
    share = wB if axle == "f" else 1.0 - wB
    d0 = []
    for c in corners:
        F0 = mb * share / 2 * G
        mr0 = float(np.asarray(getattr(cs.susp, f"mr_{c}"), float)[0])
        k = getattr(cs.susp, f"spring_{c}")
        d0.append((F0 / (mr0 * float(k["rate"])) - float(k["preload"])) / mr0)
    return sm(0.5 * (d0[0] + d0[1]))


# ==== 0. builds ==================================================================================================
print("[cs_third] 0. builds")
CTX_ALL = build(write_car("all.car", ALL_OVER))
CTX_NONE = build(write_car("none.car", omit=NONE_OMIT))
CTX_ZERO = build(write_car("zero.car", ZERO_OVER))
CTX_RUB0 = build(write_car("rub0.car", RUB0_OVER))
ok("ALL, NONE, ZERO and RUB0 synthetic cars build with 23 states and 3 inputs",
   all(c.m23.nx == 23 and c.m23.nu == 3 for c in (CTX_ALL, CTX_NONE, CTX_ZERO, CTX_RUB0)))
CS_ALL, VP_ALL = CTX_ALL.cs, CTX_ALL.vp
ok("the ALL car holds a third rate 9e4, a third gap 20 mm with flag 1, and a live third damper",
   CS_ALL.susp.spring_third_f["rate"] == K3 and abs(CS_ALL.susp.gap_third_f - GAP3) < 1e-12
   and CS_ALL.susp.gap_flag_third_f == 1 and CS_ALL.susp.damper_third_f is not None)
ok("the NONE car has no third element on either axle",
   V.third_element(CTX_NONE.cs, "f", CTX_NONE.vp) is None and V.third_element(CTX_NONE.cs, "r", CTX_NONE.vp) is None)

# ==== 1. heave = sm3(mean wheel travel) ==========================================================================
print("[cs_third] 1. heave is sm3 of the mean wheel travel")
sm3_m, mr3_m, f3_sp, f3_dm = V.third_element(CS_ALL, "f", VP_ALL)
ok(f"sm3(0.03) = {sm3(0.03):.6f} m (hand w + 0.15 w^2) through third_element",
   abs(float(sm3_m(0.03)) - sm3(0.03)) < 1e-12 and abs(sm3(0.03) - 0.030135) < 1e-12)
# state A': zs -30 mm and roll 2 mrad: xs(fl) 28.5 mm, xs(fr) 31.5 mm, the mean is 30 mm
xA = state_zs(CTX_ALL, -0.03, phi=0.002)
dA = fzs4(CTX_ALL, xA) - fzs4(CTX_NONE, xA)
e1 = 0.5 * F3_hand(sm3(0.03)) * mr3(0.03)
ok(f"heave from the mean travel (fl 28.5 mm, fr 31.5 mm): each front wheel gains {e1:.4f} N = 0.5 F3 mr3 "
   f"(model {dA[0]:.4f}, {dA[1]:.4f})",
   abs(dA[0] - e1) <= TOL_SPLINE * abs(e1) and abs(dA[1] - e1) <= TOL_SPLINE * abs(e1))

# ==== 2. each wheel gets 0.5 F3 mr3 ==============================================================================
print("[cs_third] 2. wheel share 0.5 F3 mr3")
xB = state_zs(CTX_ALL, -0.03)
dB = fzs4(CTX_ALL, xB) - fzs4(CTX_NONE, xB)
e2 = 0.5 * F3_hand(sm3(0.03)) * mr3(0.03)
ok(f"spring {K3 * sm3(0.03):.3f} N + rubber {rub_hand(sm3(0.03)):.3f} N = F3 {F3_hand(sm3(0.03)):.3f} N at heave "
   f"{sm3(0.03) * 1e3:.3f} mm; each front wheel gains {e2:.3f} N (model {dB[0]:.3f}, {dB[1]:.3f})",
   abs(dB[0] - e2) <= TOL_SPLINE * abs(e2) and abs(dB[1] - e2) <= TOL_SPLINE * abs(e2))
ok("the rear wheels gain nothing (no rear third element)", abs(dB[2]) < 1e-12 and abs(dB[3]) < 1e-12)

# ==== 3. rubber onset: flag 1 = gap, flag 0 = gap - static heave ==================================================
print("[cs_third] 3. third rubber onset")
CS_RATE0 = copy.deepcopy(CS_ALL)
CS_RATE0.susp.spring_third_f["rate"] = 0.0
f_on = V.third_element(CS_RATE0, "f", VP_ALL)[2]
for h, label in ((GAP3, "at the gap"), (GAP3 + 0.005, "5 mm past the gap")):
    model, hand = sx_eval(f_on, h), rub_hand(h)
    ok(f"flag 1, rubber {label} (heave {h * 1e3:.1f} mm): {model:.4f} N = hand {hand:.4f} N",
       abs(model - hand) <= TOL_SPLINE * max(abs(hand), 1.0))

CS0, VP0 = CTX_RUB0.cs, CTX_RUB0.vp
hs = static_heave(CS0, "f", ("fl", "fr"), sm3)
eff0 = GAP3 - hs
ok(f"flag 0: static heave {hs * 1e3:.3f} mm from the record (negative: the front preload 20 mm is above the "
   f"static travel), effective gap {eff0 * 1e3:.3f} mm (stored 20 mm)",
   abs(hs - (-0.009461)) < 1e-5 and eff0 > GAP3)
f0_sp = V.third_element(CS0, "f", VP0)[2]
for h in (eff0, eff0 + 0.005):
    model, hand = sx_eval(f0_sp, h), SLOPE * soft(h - eff0)
    ok(f"flag 0, heave {h * 1e3:.3f} mm: {model:.4f} N = hand {hand:.4f} N (rate law on gap minus static heave)",
       abs(model - hand) <= TOL_SPLINE * max(abs(hand), 1.0))
ok("flag 0 differs from flag 1 at the same heave (the static heave moves the gap)",
   abs(sx_eval(f0_sp, GAP3 + 0.005) - rub_hand(GAP3 + 0.005)) > 10.0)

# ==== 4. gap 0 and gap >= 0.5 m: no rubber ======================================================================
print("[cs_third] 4. gap rules")


def variant(gap, rate=0.0):
    """ALL car with the third gap, the third rate set and no third damper."""
    cs = copy.deepcopy(CS_ALL)
    cs.susp.gap_third_f = gap
    cs.susp.spring_third_f["rate"] = rate
    cs.susp.damper_third_f = None
    return cs


ok("gap 0, no rate, no damper: third_element is None",
   V.third_element(variant(0.0), "f", VP_ALL) is None)
ok("gap 0.5 m, no rate, no damper: third_element is None (no rubber at 0.5 m or more)",
   V.third_element(variant(0.5), "f", VP_ALL) is None)
ok("gap 0.4 m, no rate, no damper: third_element holds the rubber",
   V.third_element(variant(0.4), "f", VP_ALL) is not None)
for gap in (0.0, 0.5):
    fz = V.third_element(variant(gap, rate=K3), "f", VP_ALL)[2]
    h = GAP3 + 0.005
    ok(f"gap {gap:g} m with rate 9e4: the third force is the spring alone ({K3 * h:.3f} N at 25 mm)",
       abs(sx_eval(fz, h) - K3 * h) < 1e-9)

# ==== 5. damper: bump for a heave velocity up, rebound down ======================================================
print("[cs_third] 5. damper sign by the heave velocity")
for wu, hand, label in ((0.1, 0.5 * BUMP3[1], "bump"), (-0.1, -0.5 * REB3[1], "rebound")):
    xv = static_x(CTX_ALL)
    xv[15] = wu
    xv[16] = wu
    d = fzs4(CTX_ALL, xv) - fzs4(CTX_NONE, xv)
    ok(f"heave rate {wu:+.1f} m/s: each front wheel gains {d[0]:.3f} N = 0.5 x {label} at 100 mm/s ({hand:.1f} N)",
       abs(d[0] - hand) <= 1e-4 * abs(hand) and abs(d[1] - hand) <= 1e-4 * abs(hand))
fdm = f3_dm
ok(f"the damper table: +0.1 m/s gives bump {sx_eval(fdm, 0.1):.2f} N, -0.1 m/s gives -rebound "
   f"{sx_eval(fdm, -0.1):.2f} N",
   abs(sx_eval(fdm, 0.1) - BUMP3[1]) < 1e-3 * BUMP3[1] and abs(sx_eval(fdm, -0.1) + REB3[1]) < 1e-3 * REB3[1])

# ==== 6. QSS axle table: F3 mr3 at equal travel ==================================================================
print("[cs_third] 6. QSS axle table")
tabs_all = qss_axle_tables(CTX_ALL)
tabs_none = qss_axle_tables(CTX_NONE)
xs_q = np.array([0.005, 0.02, 0.03])
d_f = axle_at(tabs_all[0], xs_q) - axle_at(tabs_none[0], xs_q)
d_r = axle_at(tabs_all[1], xs_q) - axle_at(tabs_none[1], xs_q)
e_q = np.array([F3_hand(sm3(x)) * mr3(x) for x in xs_q])
ok(f"QSS front axle gains F3 mr3 at equal travel: {np.round(d_f, 3).tolist()} N = hand {np.round(e_q, 3).tolist()} N",
   bool(np.allclose(d_f, e_q, rtol=TOL_SPLINE, atol=1e-6)))
ok("QSS rear axle gains nothing (no rear third element)", float(np.max(np.abs(d_r))) < 1e-9)

# ==== 7. no third element: the same dx as a car without the cids ================================================
print("[cs_third] 7. no third element")
xN = state_zs(CTX_ZERO, -0.01, phi=0.001)
dz = dx_at(CTX_ZERO, xN) - dx_at(CTX_NONE, xN)
scale = max(1.0, float(np.max(np.abs(dx_at(CTX_NONE, xN)))))
ok(f"zero third cids (rates 0, gaps 0, empty damper): dx equals the car without them (max diff {np.max(np.abs(dz)):.1e})",
   float(np.max(np.abs(dz))) <= 1e-12 * scale)
ok("zero third cids: no third element on either axle",
   V.third_element(CTX_ZERO.cs, "f", CTX_ZERO.vp) is None
   and V.third_element(CTX_ZERO.cs, "r", CTX_ZERO.vp) is None)


# ==== vendor =====================================================================================================
def vendor(cs_dir):
    models = Path(cs_dir).parent / "Models"
    print("[cs_third] vendor: F2 (rear third rubber, QSS), GP3 2013 (no third element)")
    f2 = sorted(glob.glob(os.path.join(str(models), "Formula 2", "*", "*.car")))
    gp3 = sorted(glob.glob(os.path.join(str(models), "GP3*", "*", "def*.car")))
    ok("Formula 2 and GP3 2013 car files found by folder globs", bool(f2) and bool(gp3))

    c2 = build(f2[0], chassis_dir=cs_dir, circuit="Adelaide")
    cs2, vp2, su = c2.cs, c2.vp, c2.cs.susp
    ok(f"Formula 2 builds with a rear third element: nx {c2.m23.nx}",
       c2.m23.nx == 23 and V.third_element(cs2, "r", vp2) is not None)

    eff = float(su.gap_third_r) - static_heave(cs2, "r", ("rl", "rr"), coeff_sm(su.mr_third_r))
    ok(f"F2 rear third rubber effective gap {eff * 1e3:.3f} mm, 13.6 mm within 0.5 mm (stored "
       f"{float(su.gap_third_r) * 1e3:.1f} mm, gap flag {su.gap_flag_third_r})",
       su.gap_flag_third_r == 0 and abs(eff - 0.0136) < 5e-4)

    # the table as vehModel_cs builds it: (0, 0) first, then one point on the last chord appended past the end
    x_t = np.r_[0.0, np.asarray(su.rubber_third_r["x"], float)]
    y_t = np.r_[0.0, np.asarray(su.rubber_third_r["y"], float)]
    x_t, y_t = np.r_[x_t, x_t[-1] + (x_t[-1] - x_t[-2])], np.r_[y_t, y_t[-1] + (y_t[-1] - y_t[-2])]
    P_t = PchipInterpolator(x_t, y_t)
    f_r = V.third_element(cs2, "r", vp2)[2]
    for dh in (0.001, 0.004, 0.0075):          # inside the table: compression up to 8.5 mm of 9.14 mm
        h = eff + dh
        model, hand = sx_eval(f_r, h), float(P_t(W * np.log1p(np.exp(dh / W))))
        ok(f"F2 rear third force {dh * 1e3:.1f} mm past the effective gap: {model:.3f} N = hand {hand:.3f} N",
           abs(model - hand) < 0.1)

    env = build_envelope_cs(c2)
    v = env.v_grid
    Fra = env.Fz_rear - (vp2.Wrl0 + vp2.Wrr0)
    rh_r0 = 0.5 * (cs2.geom.ride_height_rl + cs2.geom.ride_height_rr)
    kt_r = cs2.tyre.rl["kz"] + cs2.tyre.rr["kz"]
    hv = rh_r0 - env.info["rh_r"] - Fra / kt_r      # rear heave travel of the envelope (the fixed point of the QSS)
    for speed in (40.0, 50.0, 70.0):
        hh = float(np.interp(speed, v, hv))
        model = sx_eval(f_r, hh)
        hand = float(P_t(W * np.log1p(np.exp((hh - eff) / W))))
        if speed == 40.0:
            ok(f"F2 QSS rear third force at 40 m/s is below 1 N: {model:.3f} N (heave {hh * 1e3:.2f} mm)",
               abs(model) < 1.0 and abs(hand) < 1.0)
        elif speed == 50.0:
            ok(f"F2 QSS rear third force at 50 m/s: {model:.3f} N = hand {hand:.3f} N (heave {hh * 1e3:.2f} mm)",
               abs(model - hand) < 0.1 and model > 0.0)
        else:
            ok(f"F2 QSS rear third force at 70 m/s is above 1000 N: {model:.1f} N = hand {hand:.1f} N "
               f"(heave {hh * 1e3:.2f} mm)", model > 1000.0 and abs(model - hand) < 0.1)

    c3 = build(gp3[0], chassis_dir=cs_dir, circuit="Eastern Creek")
    ok(f"GP3 2013 builds unchanged: nx {c3.m23.nx}, no third element on either axle",
       c3.m23.nx == 23 and V.third_element(c3.cs, "f", c3.vp) is None and V.third_element(c3.cs, "r", c3.vp) is None)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
print(f"{PASS} checks passed")
