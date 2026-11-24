"""ChassisSim camber polynomial and the build rules of five shipped cars (plan 2026-11-23-cs2c-nobuild).

Lane F: functions/chassissim.fit_camber picks the degree of the Fy camber factor table, and vehModel_cs.camber_factor
evaluates the polynomial between a smooth floor 0.5 and a smooth cap (largest table value + 0.05), with no clamp of
the camber at the table ends. Lane N: a car without the gap flags reads them as 0 (R3), a
zero corner spring with a live third spring builds with the static heave (R2), and an empty corner damper with a live
third damper gives zero force (R5). A mixed axle (one corner rate 0), a third motion ratio of 0 and an empty corner
damper on an axle without a third damper raise ValueError. The synthetic car is copied from tests/test_cs_rubber.py. The checks read the car
records with load_car and build the car with MLTP and a fake IPOPT (no solve). The vendor section (last) runs only when
CHASSISSIM_DIR names the ChassisSim install folder: it loads, builds and runs the QSS of every shipped car file it finds.
Scratch files go to a temp dir.

    python tests/test_cs_camberfit.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_camberfit.py
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

import functions.transcription as T
import vehModel_cs as V
from functions import carfile
from functions.chassissim import check_car, fit_camber, load_car
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



# ---- 1. fit_camber: the degree pick ---------------------------------------------------------------------
print("[cs_camberfit] 1. fit_camber: degree pick, tie rule, degenerate tables")
xq = np.linspace(-0.1, 0.1, 9)
c_q = fit_camber(xq, 1.0 - 2.0 * xq + 3.0 * xq ** 2)
ok(f"exact quadratic: degree 2 with coefficients 3, -2, 1 (highest power first): {np.round(c_q, 9).tolist()}",
   c_q.size == 3 and np.allclose(c_q, [3.0, -2.0, 1.0], rtol=0.0, atol=1e-9))
xc = np.linspace(-0.12, 0.12, 25)
cubic = 1.0 + 4.0 * xc - 30.0 * xc ** 2 + 200.0 * xc ** 3
c_cub = fit_camber(xc, cubic)
ok(f"exact cubic: degree 3, not 4 (tie rule): {c_cub.size - 1}", c_cub.size == 4)
c_cn = fit_camber(xc, cubic + 2e-3 * np.sin(97.0 * xc))
ok(f"noisy cubic (sine noise 2e-3, 25 points): degree 3: {c_cn.size - 1}", c_cn.size == 4)
c_flat = fit_camber([0.1, 0.1, 0.1], [0.7, 0.8, 0.9])
ok(f"flat table (all cambers equal): degree 0, the mean 0.8: {c_flat.tolist()}",
   c_flat.size == 1 and abs(c_flat[0] - 0.8) < 1e-12)
c_one = fit_camber([0.05], [0.9])
ok(f"one point: degree 0, the value 0.9: {c_one.tolist()}", c_one.size == 1 and abs(c_one[0] - 0.9) < 1e-12)
c_two = fit_camber([0.0, 0.1], [1.0, 2.0])
ok(f"two distinct cambers: degree at most 1, the line 10 x + 1: {np.round(c_two, 9).tolist()}",
   c_two.size == 2 and np.allclose(c_two, [10.0, 1.0], rtol=0.0, atol=1e-9))

hit, msg = refusal(fit_camber, [], [])
GUARD = "The Fy camber table (table 5) of a tyre needs as many finite x values as y values."
ok("empty table: ValueError with the table 5 text", hit and msg == GUARD)
hit, msg = refusal(fit_camber, [0.0, 0.1, 0.2], [1.0, 0.9])
ok("3 x values and 2 y values: the same ValueError", hit and msg == GUARD)
hit, msg = refusal(fit_camber, [0.0, 0.1], [1.0, np.nan])
hit2, msg2 = refusal(fit_camber, [0.0, np.inf], [1.0, 0.9])
ok("a NaN y value or an infinite x value: the same ValueError", hit and msg == GUARD and hit2 and msg2 == GUARD)

# ---- 2. camber_factor: smooth floor 0.5 and cap y_max + 0.05, no clamp of the camber -------------------
print("[cs_camberfit] 2. camber_factor: floor, cap, no clamp, types")
COEF = [-2.0, 0.5, 1.0]                                    # poly(c) = -2 c^2 + 0.5 c + 1, c in rad
W = V.W_CAMBER


def poly(c):
    return np.polyval(COEF, c)


cc = np.linspace(-0.1, 0.1, 201)
YM = float(poly(cc).max())                                 # the largest table value
CAP = YM + V.CAMBER_CAP_MARGIN
ok(f"the cap margin is 0.05: {V.CAMBER_CAP_MARGIN}", V.CAMBER_CAP_MARGIN == 0.05)
err = np.abs(V.camber_factor(COEF, cc, YM) - poly(cc))
bound = W * np.exp(-(poly(cc) - 0.5) / W) + W * np.exp(-(CAP - poly(cc)) / W)
ok(f"on the table: |factor - poly| within W exp(-d_floor / W) + W exp(-d_cap / W) (max {err.max():.2e})",
   bool(np.all(err <= 1.01 * bound + 1e-15)))
ok(f"on the table (poly from {poly(cc).min():.3f} to {YM:.3f}): |factor - poly| < 1e-6 (max {err.max():.2e})",
   bool(err.max() < 1e-6))
f_low = V.camber_factor([-3.0], 0.0, 1.0)
ok(f"poly far below 0.5 (-3): factor {f_low:.7f} within 1e-9 of 0.5", abs(f_low - 0.5) < 1e-9)
f_end = V.camber_factor(COEF, 0.2, YM)
ok(f"past the table end (0.2 rad, table ends at 0.1): factor {f_end:.7f} = poly {poly(0.2):.7f} within 1e-6, "
   f"not the end value {poly(0.1):.7f}", abs(f_end - poly(0.2)) < 1e-6 and abs(f_end - poly(0.1)) > 5e-3)
x4 = np.linspace(-0.07, 0.07, 15)                          # a degree 4 table: the fit grows without limit past it
y4 = 1.0 - 60.0 * x4 ** 2 + 9000.0 * x4 ** 4
c4 = fit_camber(x4, y4)
far = np.array([-1.0, -0.5, -0.26, -0.17, 0.17, 0.26, 0.5, 1.0])
p_far, f_far = np.polyval(c4, far), V.camber_factor(c4, far, float(y4.max()))
ok(f"degree 4 table (fit degree {c4.size - 1}): far past the table the polynomial reaches {p_far.max():.1f}, the "
   f"factor stays at or below y_max + 0.05 = {y4.max() + 0.05:.4f} (max {f_far.max():.9f})",
   c4.size == 5 and p_far.max() > 100.0 and bool(np.all(f_far <= y4.max() + 0.05 + 1e-9))
   and bool(np.all(np.isfinite(f_far))) and abs(f_far.max() - (y4.max() + 0.05)) < 1e-9)
xf = np.linspace(x4[0], x4[-1], 2001)
in4 = np.polyval(c4, xf) >= 0.525
dev4 = np.abs(V.camber_factor(c4, xf, float(y4.max())) - np.polyval(c4, xf))[in4].max()
ok(f"degree 4 table, inside the table with poly >= 0.525: |factor - poly| < 1e-6 (max {dev4:.2e})", dev4 < 1e-6)
c_sx4 = ca.SX.sym("c")
fn4 = ca.Function("cf4", [c_sx4], [V.camber_factor(c4, c_sx4, float(y4.max())), ca.jacobian(V.camber_factor(c4, c_sx4, float(y4.max())), c_sx4)])
v_sx, d_sx = (float(e) for e in fn4(1.0))
ok(f"casadi SX at 1 rad (poly {np.polyval(c4, 1.0):.0f}): value {v_sx:.9f} at the cap and a finite slope {d_sx:.1e}",
   abs(v_sx - (y4.max() + 0.05)) < 1e-9 and np.isfinite(d_sx))
f_float = V.camber_factor(COEF, 0.1, YM)
ok(f"float in, float out: {f_float:.9f}", isinstance(f_float, float) and abs(f_float - poly(0.1)) < 1e-6)
f_arr = V.camber_factor(COEF, np.array([0.1, 0.2]), YM)
ok("ndarray in, ndarray out, element-wise", isinstance(f_arr, np.ndarray) and f_arr.shape == (2,)
   and np.allclose(f_arr, poly(np.array([0.1, 0.2])), rtol=0.0, atol=1e-6))
c_sx = ca.SX.sym("c")
fn_sx = ca.Function("cf", [c_sx], [V.camber_factor(COEF, c_sx, YM)])
ok("casadi SX: the Function gives the float value (1e-12)", abs(float(fn_sx(0.1)) - f_float) < 1e-12)
T_fl = load_car(write_car("ymax.car")).tyre.fl
ok(f"the tyre record holds camber_ymax = max(table 5 y): {T_fl['camber_ymax']}",
   T_fl["camber_ymax"] == float(np.max(T_fl["camber_y"])))

# ---- 3. R3: absent gap flags read as 0 -------------------------------------------------------------------
print("[cs_camberfit] 3. R3: a car without the gap flags loads with flags 0")
p_nof = write_car("no_gap_flags.car", omit=tuple(range(204, 210)))
su_nof = load_car(p_nof).susp
flags = [getattr(su_nof, f"gap_flag_{n}") for n in ("fl", "fr", "rl", "rr", "third_f", "third_r")]
ok(f"no gap flag records: the six flags read {flags}, all 0", all(f == 0 for f in flags))
su_fl = load_car(write_car("with_gap_flags.car")).susp
ok(f"with the records: the flags read 1 (fl {su_fl.gap_flag_fl}, third_r {su_fl.gap_flag_third_r})",
   su_fl.gap_flag_fl == 1 and su_fl.gap_flag_third_r == 1)
ctx_nof = build(p_nof)
ok(f"the car without flags builds: nx {ctx_nof.m23.nx}, nu {ctx_nof.m23.nu}",
   ctx_nof.m23.nx == 23 and ctx_nof.m23.nu == 3)

# ---- 4. R2: zero corner spring with a live third spring ------------------------------------------------
print("[cs_camberfit] 4. R2: zero corner rate, third spring carries the axle load")


def spring_rec(pre, rate):
    return (1, pre, rate, np.array([0.0, 0.05]), np.array([0.0, 500.0]))


# file cids: 1 = FR, 2 = FL (the file stores the right wheel first); 5 = third spring front
R2_OVER = {1: spring_rec(0.021, 0.0), 2: spring_rec(0.020, 0.0), 5: spring_rec(0.019, 9.0e4)}
p_r2 = write_car("zero_corner_third.car", over=R2_OVER)
ctx_r2 = build(p_r2)
ok(f"front corner rates 0, third spring 9e4 N/m: the car builds, nx {ctx_r2.m23.nx} and nu {ctx_r2.m23.nu}",
   ctx_r2.m23.nx == 23 and ctx_r2.m23.nu == 3)
su_r2, vp_r2 = ctx_r2.cs.susp, ctx_r2.vp
d_code, mr0_code = V._static_defl(su_r2, vp_r2, "fl", "rubber gap")
mr3 = float(np.atleast_1d(su_r2.mr_third_f)[0])
k3, pre3 = float(su_r2.spring_third_f["rate"]), float(su_r2.spring_third_f["preload"])
mr0 = float(np.atleast_1d(su_r2.mr_fl)[0])
h_s = (vp_r2.m_c_fl + vp_r2.m_c_fr) * G / (mr3 * k3) - pre3
d_hand = mr0 * h_s / mr3
ok(f"static heave h_s {h_s * 1e3:.4f} mm = (m_fl + m_fr) g / (mr3 k3) - preload3; d {d_code * 1e3:.4f} mm = "
   f"mr0 h_s / mr3 {d_hand * 1e3:.4f} mm", abs(d_code - d_hand) < 1e-12 and abs(mr0_code - mr0) < 1e-12)
p_r2_none = write_car("zero_corner_nothird.car", over={1: spring_rec(0.021, 0.0), 2: spring_rec(0.020, 0.0)})
hit, msg = refusal(V._static_defl, load_car(p_r2_none).susp, vp_r2, "fl", "rubber gap")
ok(f"corner rate 0 and no third spring: the old text is raised: {msg[:70]!r}",
   hit and msg == "The spring of corner fl has no rate. The static deflection for the rubber gap needs one.")

MIXED = ("Axle front: one corner spring has no rate. The model needs both corner rates 0 (the third spring carries "
         "the axle) or both above 0.")
su_mix = load_car(write_car("mixed_axle.car", over={1: spring_rec(0.021, 0.0), 5: spring_rec(0.019, 9.0e4)})).susp
hit, msg = refusal(V._static_defl, su_mix, vp_r2, "fr", "rubber gap")
hit2, msg2 = refusal(V._static_defl, su_mix, vp_r2, "fl", "rubber gap")
ok(f"mixed front axle (one corner rate 0, the other above 0): both corners raise: {msg[:50]!r}",
   hit and msg == MIXED and hit2 and msg2 == MIXED)
su_mr0 = load_car(write_car("zero_mr3.car", over={**R2_OVER, 29: np.zeros(5)})).susp
hit, msg = refusal(V._static_defl, su_mr0, vp_r2, "fl", "rubber gap")
ok(f"R2 car with a third motion ratio of 0: ValueError that names cid 29: {msg[:60]!r}",
   hit and "cid 29" in msg and msg.startswith("Axle front: the third motion ratio"))
hit, msg = refusal(check_car, load_car(write_car("no_mr3.car", over=R2_OVER, omit=(29,))))
ok(f"R2 car without the cid 29 record: check_car lists susp.mr_third_f: {msg[-70:]!r}", hit and "susp.mr_third_f" in msg)

# ---- 5. R5: empty corner damper with a live third damper ------------------------------------------------
print("[cs_camberfit] 5. R5: empty corner damper with a live third damper")
ZERO = np.zeros(BUMP.size)
EMPTY_TABLE = np.concatenate([VEL, ZERO, ZERO])
DAMPER_FL = 32                                             # file cid of the front left damper (logical 31)
p_r5 = write_car("empty_fl_damper.car", over={DAMPER_FL: (1, arb(6, 31), EMPTY_TABLE)})
ctx_r5 = build(p_r5)
ok(f"empty front left damper, live third damper: the car builds, nx {ctx_r5.m23.nx}",
   ctx_r5.m23.nx == 23)
dm_zero = {"velocity": VEL, "bump": ZERO, "rebound": ZERO}
F_zero = V._damper_curve(dm_zero, "corner fl", True)
v_probe = np.array([-0.3, 0.0, 0.2, 1.0])
ok(f"empty corner table with the third damper live: zero force at {v_probe.tolist()} m/s",
   bool(np.all(F_zero(v_probe) == 0.0)))
hit, msg = refusal(V._damper_curve, dm_zero, "corner fl")
ok(f"empty corner table without the third damper: the old text is raised: {msg[:60]!r}",
   hit and msg == "The damper of corner fl has no table. The car model needs one.")
p_r5_none = write_car("empty_fl_damper_nothird.car", over={DAMPER_FL: (1, arb(6, 31), EMPTY_TABLE),
                                                           35: (1, arb(6, 35), EMPTY_TABLE)})
hit, msg = refusal(build, p_r5_none)
ok(f"empty front left damper and an empty third damper: the build raises the old text: {msg[:60]!r}",
   hit and msg == "The damper of corner fl has no table. The car model needs one.")

DAMPER_RL = 34                                             # file cid of the rear left damper (logical 33)
p_r5_rl = write_car("empty_fl_rl_damper.car", over={DAMPER_FL: (1, arb(6, 31), EMPTY_TABLE),
                                                    DAMPER_RL: (1, arb(6, 31), EMPTY_TABLE)})
hit, msg = refusal(build, p_r5_rl)
ok(f"the same empty table on the front left and the rear left corner, no rear third damper: the rear corner "
   f"raises (the cache key holds the axle rule): {msg[:60]!r}",
   hit and msg == "The damper of corner rl has no table. The car model needs one.")
hit, msg = refusal(build, write_car("empty_fl_damper_mr3_0.car", over={DAMPER_FL: (1, arb(6, 31), EMPTY_TABLE),
                                                                       29: np.zeros(5)}))
ok(f"empty front left damper, live third damper and a third motion ratio of 0: ValueError that names cid 29: "
   f"{msg[:60]!r}", hit and "cid 29" in msg)

# ---- 6. vendor: every shipped car loads, builds and runs the QSS ----------------------------------------
VENDOR_CIRC = "Eastern Creek"


def vendor(cs_dir):
    print("[cs_camberfit] 6. vendor: every shipped car loads, builds and runs the QSS")
    files = sorted(str(p) for p in Path(cs_dir).parent.rglob("*.car"))     # the ChassisSim tree: install, help, Models
    ok(f"shipped car files in the ChassisSim tree: {len(files)} (no names printed)", len(files) == 31)
    n_load = n_build = n_qss = 0
    poly_ok, cap_ok, dev = True, True, 0.0
    for p in files:
        try:
            cs = load_car(p, cs_dir)
            check_car(cs)
            n_load += 1
        except ValueError:
            continue
        poly_ok = poly_ok and all(np.all(np.isfinite(getattr(cs.tyre, c)["camber_poly"])) for c in CORNERS)
        for c in CORNERS:
            Tc = getattr(cs.tyre, c)
            xs = np.linspace(Tc["camber_x"].min(), Tc["camber_x"].max(), 2001)
            pv, ym = np.polyval(Tc["camber_poly"], xs), Tc["camber_ymax"]
            m = (pv >= 0.525) & (pv <= ym)
            if m.any():
                dev = max(dev, float(np.abs(V.camber_factor(Tc["camber_poly"], xs, ym) - pv)[m].max()))
            fw = V.camber_factor(Tc["camber_poly"], np.linspace(-0.5, 0.5, 2001), ym)
            cap_ok = cap_ok and bool(np.all(fw <= ym + 0.05 + 1e-9) and np.all(fw >= 0.5))
        try:
            quiet(MLTP, car=f"chassissim:{p}", circuit=f"chassissim:{VENDOR_CIRC}", chassissim_dir=cs_dir,
                  physics="corrected", save=False, plot=False, linear_solver="mumps")
            n_build += 1
        except ValueError:
            continue
        try:
            lap = float(quiet(MLTP_screen, car=f"chassissim:{p}", circuit=f"chassissim:{VENDOR_CIRC}",
                              chassissim_dir=cs_dir, physics="corrected", save=False, verbose=False).data.lap_time)
            n_qss += int(np.isfinite(lap))
        except ValueError:
            continue
    ok(f"load_car and check_car: {n_load} of {len(files)}", n_load == len(files))
    ok(f"camber polynomial finite on every tyre of every car", poly_ok)
    ok(f"inside every table 5, where the polynomial is from 0.525 to y_max: |factor - poly| < 1e-6 (max {dev:.2e})",
       dev < 1e-6)
    ok(f"every tyre: the factor is from 0.5 to y_max + 0.05 at cambers up to 0.5 rad", cap_ok)
    ok(f"MLTP build with no solve: {n_build} of {len(files)}", n_build == len(files))
    ok(f"QSS lap finite at {VENDOR_CIRC}: {n_qss} of {len(files)}", n_qss == len(files))


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set (vendor section: every shipped car loads, builds and runs the QSS)")
else:
    vendor(CS)
print(f"{PASS} checks passed")
