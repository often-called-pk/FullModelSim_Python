"""ChassisSim phase 2b, task 16: the stability index of a ChassisSim car (tests/test_cs_stbi.py).

The index is stbi = 100 ((lf Cf - lr Cr) - (lf^2 Cf + lr^2 Cr) / V) / (c (lf + lr)). The axle stiffnesses C come from
a central difference (step pi / 18000 rad) of the uncapped lateral law. The operating point of the difference is the
combined slip, held at or below s_c. s_c is the first normalised slip where the slip curve reaches
min(its peak, 0.9): the start of the capped plateau. The synthetic car follows tests/test_cs_third.py,
with a normalised slip table that peaks at 1, so the 0.9 cap bites. The checks write their own hand values for the
tyre law, the axle sums and the index. They cover the formula (numpy, SX, the floor), the channel at a light-slip
state, the clipped operating point past s_c, the sign with a rear peak x 1.5, the QSS apex correction (flag and
map, MLTP_screen), and a fake-IPOPT build. The vendor section (last) runs only when CHASSISSIM_DIR names the
ChassisSim install folder. It builds the GP3 2013 car. The seeded check rebuilds the car from a stored fixed-line
result. It runs only when CS_STBI_SEED_DIR names the folder that holds the result file (SKIP when unset, or when
the folder has no such file). The test writes nothing into the vendor folders.

    python tests/test_cs_stbi.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_stbi.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" CS_STBI_SEED_DIR="<folder of the GP3 fixed-line .mat>" python tests/test_cs_stbi.py
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
from functions.chassissim import load_car
from functions.ggv import _V_FLOOR, build_envelope_cs, march
from MLTP import MLTP
from MLTP_screen import MLTP_screen
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

PASS = 0
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair. An over or omit cid is a file cid.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}
D_ALPHA = np.pi / 18000.0      # the step of the central difference, written here (the model has its own copy)
W_SC = 1e-3                    # width of the smooth minimum at s_c, written here (the model has its own copy)
GP3_STBI_LOG = -8.83           # % (the GP3 log median StabIND, lap 1)
SLIP_X = np.linspace(0.0, 2.0, 5)
SLIP_Y = np.array([0.0, 0.95, 1.0, 1.0, 1.0])   # normalised slip curve, peak 1: the 0.9 lateral cap bites
# s_c by hand: the odd PCHIP of the table reaches 0.9 (the cap, below the peak 1) between the first two points.
FN = PchipInterpolator(np.r_[-SLIP_X[:0:-1], SLIP_X], np.r_[-SLIP_Y[:0:-1], SLIP_Y])
S_C = float(brentq(lambda v: FN(v) - 0.9, SLIP_X[0], SLIP_X[1]))
DIMS = (3, 2, 5, 2, 2, 5, 4)
TYRE_TEMPS = (300.0, 350.0)
TYRE_BG = 325.0
PRESSURE = 2.1e5
PEAK_SLIP_ANGLE = 0.11         # rad, scalar 8 of the tyre record
PEAK_SLIP_RATIO = 0.12         # scalar 9
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
SPRING = {1: (1.23e5, 0.020), 2: (1.31e5, 0.021), 3: (0.97e5, 0.019), 4: (1.02e5, 0.0205)}
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
    11: 0.0, 12: 0.0,                                 # third gaps: 0, no third rubber
}
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
AERO_X = 0.005 * np.arange(7)
AERO_Z = 0.002 + 0.005 * np.arange(7)
MAP_X = np.array([-10000.0, -5000.0, -3333.0, -1667.0, 0.0, 1667.0, 3333.0, 5000.0, 6667.0, 8333.0, 10000.0])
MAP_Z = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
MAP_DATA = np.where((MAP_X > 0)[:, None] & (MAP_Z >= 0.1)[None, :], 0.1 * MAP_X[:, None], 0.0)
INI = ("[Total Circ prop]\r\nTotal circuits = 1\r\n\r\n"
       f"[Circuit0]\r\nCircuit name = {CIRC}\r\nCircuit file name = tr_synth.txt\r\nBase Grip factor = 1.0\r\n"
       "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n")
S_TRACK = np.linspace(0.0, 600.0, 121)
K_TRACK = (2.0 * np.pi / 600.0) * (1.0 + 0.5 * np.sin(2.0 * np.pi * S_TRACK / 600.0))


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


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ---- synthetic car (as tests/test_cs_third.py, slip table peak 1) -------------------------------------------
def arb(n, s):
    return np.sin(np.arange(n) * 0.9 + s) * 7.3 - 0.61


def tyre_sizes(d):
    d1, d2, d3, d4, d5, d6, d7 = d

    def g(a, b):
        return a + b + a * b
    return [2 * d3, 2 * d4, 2 * d4, g(d1, d2), g(d6, d7), 2 * d6, 2 * d4, g(d1, d2), 2 * d3, 2 * d6, 2 * d5, 2 * d5, 2 * d5]


def tyre_record(i, corner, scale=1.0):
    """Tyre record (tag 13). scale multiplies the lateral peak grid (the rear peak x 1.5 check)."""
    d1, d2, d3, _, d5, d6, d7 = DIMS
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(DIMS))]
    load = np.linspace(1000.0, 25000.0, d1)
    grid = scale * (1.4 * load[:, None] + 100.0 * np.arange(d2)[None, :]).ravel()
    tabs[3] = np.concatenate([load, TYRE_TEMPS, grid])
    tabs[0] = np.concatenate([SLIP_X, SLIP_Y])
    cam = np.linspace(-6.0, 6.0, d6)
    tabs[5] = np.concatenate([cam, 1.0 - 0.01 * cam ** 2])
    loads4 = 1000.0 * np.arange(1, d7 + 1)
    tabs[4] = np.concatenate([cam, loads4, np.ones(d6 * d7)])
    px = np.linspace(150.0, 250.0, d5)
    tabs[10] = np.concatenate([px, KZ[corner] * (1.0 + 0.01 * np.arange(d5))])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = 0.0, 0.0, 2.0, RADIUS[corner], PRESSURE, TYRE_BG
    sc[7], sc[8], sc[9], sc[11], sc[15] = DRAG[corner], PEAK_SLIP_ANGLE, PEAK_SLIP_RATIO, 3.0, 1.05
    return 13, 80 + i, (list(DIMS), sc, tabs)


def table_rec(flag, pre, rate, xs, ys):
    return (flag, pre, rate, np.asarray(xs, float), np.asarray(ys, float))


def build_records(rear_scale=1.0, over=None):
    """Records of the synthetic car as a list of (tag, cid, value). over replaces or adds cids as (tag, value)."""
    rec = {}

    def put(tag, cid, v):
        cid = FILE_CID.get(cid, cid)
        rec[cid] = (tag, v)
    for cid, v in EXPLICIT.items():
        put(1, cid, v)
    for cid, (rate, pre) in SPRING.items():
        put(11, cid, table_rec(1, pre, rate, [0.0, 0.05], [0.0, 500.0]))
    put(11, 5, table_rec(1, 0.0, 0.0, [0.0, 0.05], [0.0, 500.0]))
    put(11, 6, table_rec(1, 0.0, 0.0, [0.0, 0.05], [0.0, 500.0]))
    for cid, (rate, pre) in RUBBER.items():
        put(11, cid, table_rec(1, pre, rate, [0.0, 0.02], [0.0, 800.0]))
    for cid, rate in ARB.items():
        put(11, cid, table_rec(1, 0.0, rate, [0.0, 0.1], [0.0, 300.0]))
    for cid in range(23, 31):
        put(2, cid, np.array([1.0 + 0.05 * np.sin(cid), 0.0, 0.0, 0.0, 0.0]))
    put(2, 29, np.array([1.0, 0.3, 0.0, 0.0, 0.0]))
    for cid in range(366, 372):
        put(6, cid, 10 + cid - 366)
    for cid in range(204, 210):
        put(6, cid, 1)
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
    put(12, 35, (1, arb(6, 35), np.array([])))
    for cid, rows in ((49, HP_LEFT), (50, HP_RIGHT), (51, HP_LEFT), (52, HP_RIGHT)):
        put(10, cid, (0, np.concatenate([rows[:, 0], rows[:, 1], rows[:, 2]])))
    for i, c in enumerate(CORNERS):
        _, cid, v = tyre_record(i, c, rear_scale if c[0] == "r" else 1.0)
        put(13, cid, v)
    for cid, tv in (over or {}).items():
        rec[cid] = tv
    out = [(t, c, v) for c, (t, v) in sorted(rec.items())]
    out.append((carfile.END_TAG, carfile.END_CID, None))
    return out


def write_car(name, rear_scale=1.0, over=None):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(rear_scale, over), path)
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


def build(car_path, chassis_dir=INSTALL, circuit=CIRC, **kw):
    """MLTP on the car with a fake IPOPT (no solve). Returns ctx with ctx.m23, ctx.cs, ctx.vp and ctx.data."""
    return quiet(MLTP, car=f"chassissim:{car_path}", circuit=f"chassissim:{circuit}", chassissim_dir=chassis_dir,
                 physics="corrected", save=False, plot=False, linear_solver="mumps", **kw)


# ---- channel evaluation -------------------------------------------------------------------------------------
def make_ev(m):
    """Function of the scaled states, inputs and kappa: stbi, then sa, sx, fz, dynamic camber per corner."""
    outs = [dict(m.veh_extra)["stbi"]]
    for c in CORNERS:
        outs += [getattr(m, f"sa_{c}"), getattr(m, f"sx_{c}"), getattr(m, f"fz_{c}"), getattr(m, f"dynamic_camber_{c}")]
    return ca.Function("ev", [m.x, m.u, m.pv], outs)


def x_phys(vx, vy, r, Om):
    x = np.zeros(23)
    x[0], x[1], x[2] = vx, vy, r
    x[5:9] = Om
    return x


def ev_at(m, ev, xp, up=None):
    """Evaluate the channel at a physical state (scaled by x_s). Inputs default to 0 (no drive, no brake, no steer)."""
    xn = np.asarray(xp, float) / np.asarray(m.x_s, float)
    un = np.zeros(m.nu) if up is None else np.asarray(up, float)
    res = ev(np.reshape(xn, (-1, 1)), np.reshape(un, (-1, 1)), 0.0)
    return [float(o) for o in res]


def zero_slip_state(m, ev, vx, vy, r):
    """Physical state with zero slip ratio at all four wheels (sx is linear in each wheel speed: two evaluations)."""
    sx0 = np.array(ev_at(m, ev, x_phys(vx, vy, r, np.zeros(4)))[2::4])
    sx1 = np.array(ev_at(m, ev, x_phys(vx, vy, r, np.ones(4)))[2::4])
    return x_phys(vx, vy, r, -sx0 / (sx1 - sx0))


# ---- hand values ---------------------------------------------------------------------------------------------
def hand_sa(c, vx, vy, r, ctx):
    """Slip angle of a corner by hand (delta 0, toe 0, the car has no toe)."""
    lf, lr = ctx.vp.l_f, ctx.vp.l_r
    tf, tr = ctx.cs.geom.track_front, ctx.cs.geom.track_rear
    if c == "fl":
        return -np.arctan((lf * r + vy) / (vx - r * tf / 2))
    if c == "fr":
        return -np.arctan((lf * r + vy) / (vx + r * tf / 2))
    if c == "rl":
        return np.arctan((lr * r - vy) / (vx - r * tr / 2))
    return np.arctan((lr * r - vy) / (vx + r * tr / 2))


def hand_fns(T_, cap):
    """Return (F'(Fz) function, camber factor function, normalised slip-curve function) as casadi Functions."""
    z = ca.SX.sym("z")
    fp = ca.Function("fp", [z], [V._curve(T_["load"], T_["fy_peak"])(z)])
    cm = ca.Function("cm", [z], [V._curve(T_["camber_x"], T_["camber_y"])(z)])
    hu = ca.Function("hu", [z], [V._slip_curve(T_, cap)(z)])
    return fp, cm, hu


def law(T_, Fp, cam, sa, sx, e, hu, clip):
    """Lateral force [N] of one tyre on the uncapped law hu, with the slip angle moved by e. The model law, written here.
    clip=True holds the combined slip at or below S_C first (smooth minimum of width W_SC), then adds e."""
    san, ssn = sa / T_["peak_slip_angle"], sx / T_["peak_slip_ratio"]
    sn = np.sqrt(san * san + ssn * ssn + 1e-6)
    q = 0.5 * (sn + S_C - np.sqrt((sn - S_C) ** 2 + W_SC ** 2)) / sn if clip else 1.0
    san, ssn = q * san + e / T_["peak_slip_angle"], q * ssn
    return Fp * cam * float(hu(np.sqrt(san * san + ssn * ssn + 1e-6))) * san


def corner_scale(ctx, c, out):
    """Return (F' x grip, camber factor) of a corner at the channel outputs out (fz and dynamic camber)."""
    k = CORNERS.index(c)
    T_ = getattr(ctx.cs.tyre, c)
    fp, cm, _ = hand_fns(T_, None)
    return (float(fp(out[3 + 4 * k])) * float(getattr(ctx, "grip", 1.0)) * T_["grip"],
            float(cm(out[4 + 4 * k])))


def hand_C(ctx, vx, vy, r, out, axle, clip=True):
    """Axle cornering stiffness [N/rad] by hand: central difference (step D_ALPHA) of the uncapped lateral law, both
    tyres. clip=True is the rule of the index (operating point at or below S_C). clip=False takes the actual slip.
    out: the channel outputs at the state (sx, fz and dynamic camber)."""
    C = 0.0
    for c in (("fl", "fr") if axle == "f" else ("rl", "rr")):
        T_ = getattr(ctx.cs.tyre, c)
        Fp, cam = corner_scale(ctx, c, out)
        hu = hand_fns(T_, None)[2]
        sa, sx = hand_sa(c, vx, vy, r, ctx), out[2 + 4 * CORNERS.index(c)]
        C += (law(T_, Fp, cam, sa, sx, D_ALPHA, hu, clip) - law(T_, Fp, cam, sa, sx, -D_ALPHA, hu, clip)) / (2 * D_ALPHA)
    return C



def hand_stbi(lf, lr, Cf, Cr, Vx, floor):
    """The index by hand: wheelbase divisor, floored sum Cf + Cr (smooth, width floor)."""
    c = Cf + Cr
    c = 0.5 * (c + floor + np.sqrt((c - floor) ** 2 + floor ** 2))
    return 100.0 * (((lf * Cf - lr * Cr) - (lf ** 2 * Cf + lr ** 2 * Cr) / Vx) / (c * (lf + lr)))


def hand_index(ctx, vx, vy, r, out, clip=True):
    lf, lr = ctx.vp.l_f, ctx.vp.l_r
    Cf = hand_C(ctx, vx, vy, r, out, "f", clip)
    Cr = hand_C(ctx, vx, vy, r, out, "r", clip)
    return hand_stbi(lf, lr, Cf, Cr, vx, 2e-2 * ctx.vp.m * ctx.vp.g), Cf, Cr


def frac_of_peak(ctx, c, out, vx, vy, r):
    """Fraction of the lateral capacity of a corner at its slip (uncapped normalised curve x sn)."""
    T_ = getattr(ctx.cs.tyre, c)
    k = CORNERS.index(c)
    _, _, hu = hand_fns(T_, None)
    san = hand_sa(c, vx, vy, r, ctx) / T_["peak_slip_angle"]
    sn = np.sqrt(san * san + (out[2 + 4 * k] / T_["peak_slip_ratio"]) ** 2 + 1e-6)
    return float(hu(sn)) * sn, sn


# ---- the checks ----------------------------------------------------------------------------------------------
print("[cs_stbi] 1. the index formula (numpy, SX, floor)")
LF, LR, VX_F, FLOOR = 1.5, 1.4, 20.0, 137.0
for Cf, Cr in ((8.0e4, 9.0e4), (2.5e4, -1.0e4)):
    ref = hand_stbi(LF, LR, Cf, Cr, VX_F, FLOOR)
    ok(f"numpy index at Cf {Cf:.1e}, Cr {Cr:.1e} equals the hand formula (rel 1e-12): {ref:.6f} %",
       rel(V.stbi_index(LF, LR, Cf, Cr, VX_F, FLOOR), ref) < 1e-12)
sCf, sCr, sV = ca.SX.sym("Cf"), ca.SX.sym("Cr"), ca.SX.sym("V")
fidx = ca.Function("idx", [sCf, sCr, sV], [V.stbi_index(LF, LR, sCf, sCr, sV, FLOOR)])
ok("SX index equals the numpy index at a normal state (rel 1e-12)",
   rel(float(fidx(8.0e4, 9.0e4, VX_F)), V.stbi_index(LF, LR, 8.0e4, 9.0e4, VX_F, FLOOR)) < 1e-12)
for Cf, Cr in ((3.0e4, -3.0e4), (-2.0, -3.0)):
    val = V.stbi_index(LF, LR, Cf, Cr, VX_F, FLOOR)
    ref = hand_stbi(LF, LR, Cf, Cr, VX_F, FLOOR)
    ok(f"Cf + Cr = {Cf + Cr:g}: finite and equal to the hand floored formula ({val:.4f} %)",
       bool(np.isfinite(val)) and rel(val, ref) < 1e-12)
    ok(f"Cf + Cr = {Cf + Cr:g}: the SX value is finite and equals the numpy value",
       bool(np.isfinite(float(fidx(Cf, Cr, VX_F)))) and rel(float(fidx(Cf, Cr, VX_F)), val) < 1e-12)

print("[cs_stbi] building the synthetic car (fake IPOPT)")
BASE = write_car("base.car")
ctx = build(BASE)
m = ctx.m23
ev = make_ev(m)
LIGHT = (20.0, -0.5, 0.0)                      # vx, vy, r: small slip, delta 0
x_light = zero_slip_state(m, ev, *LIGHT)
out_light = ev_at(m, ev, x_light)

print("[cs_stbi] 2. channel against the hand central difference (light slip)")
for c in CORNERS:
    frac, sn = frac_of_peak(ctx, c, out_light, *LIGHT)
    ok(f"light slip: {c} at normalised slip {sn:.3f}, force {frac:.3f} of peak (below 0.7)", frac < 0.7)
hand_val, Cf_h, Cr_h = hand_index(ctx, *LIGHT, out_light)
model_val = out_light[0]
ok(f"channel = hand index: {model_val:.6f} % vs {hand_val:.6f} % (rel 1e-6), Cf {Cf_h:.4e}, Cr {Cr_h:.4e} N/rad",
   rel(model_val, hand_val) < 1e-6)

print("[cs_stbi] 3. past s_c: the slope of the uncapped curve at s_c, not at the actual slip")
pa = PEAK_SLIP_ANGLE
ok(f"s_c of the model slip curve {V._slip_curve(ctx.cs.tyre.rl).s_c:.6f} equals the hand root {S_C:.6f} (rel 1e-9)",
   rel(V._slip_curve(ctx.cs.tyre.rl).s_c, S_C) < 1e-9)
# Rear slip angles 0.85 x peak (past s_c), front 0.2 x peak (below s_c): yaw rate and lateral speed by hand.
R_RISE = LIGHT[0] * (np.tan(0.85 * pa) - np.tan(0.2 * pa)) / (ctx.vp.l_f + ctx.vp.l_r)
RISE = (LIGHT[0], ctx.vp.l_r * R_RISE - LIGHT[0] * np.tan(0.85 * pa), R_RISE)
x_rise = zero_slip_state(m, ev, *RISE)
out_rise = ev_at(m, ev, x_rise)
frac_r, sn_r = frac_of_peak(ctx, "rl", out_rise, *RISE)
ok(f"rear normalised slip {sn_r:.3f} in [0.75, 0.95], past s_c {S_C:.3f}, force {frac_r:.3f} of peak in [0.9, 1.0]",
   0.75 <= sn_r <= 0.95 and sn_r > S_C + 0.1 and 0.9 <= frac_r <= 1.0)
hand_clip, _, Cr_clip = hand_index(ctx, *RISE, out_rise)
hand_act, _, Cr_act = hand_index(ctx, *RISE, out_rise, clip=False)
ok(f"channel = hand index at the clipped point: {out_rise[0]:.6f} % vs {hand_clip:.6f} % (rel 1e-6)",
   rel(out_rise[0], hand_clip) < 1e-6)
# C by hand at s_c: F' x camber factor x fn'(s_c) / peak slip angle, fn' from the PCHIP of the table (written above).
Cr_sc = sum(np.prod(corner_scale(ctx, c, out_rise)) for c in ("rl", "rr")) * float(FN.derivative()(S_C)) / pa
ok(f"rear C {Cr_clip:.4e} N/rad = F' x fn'(s_c) / peak slip angle {Cr_sc:.4e} (rel 1e-2)", rel(Cr_clip, Cr_sc) < 1e-2)
ok(f"rear C at the actual slip {Cr_act:.4e} N/rad is below 0.5 of the C at s_c {Cr_clip:.4e}", 0.0 < Cr_act < 0.5 * Cr_clip)
ok(f"index at the actual slip {hand_act:.4f} % is not the channel {out_rise[0]:.4f} % (differs by more than 10 points)",
   abs(hand_act - out_rise[0]) > 10.0)

print("[cs_stbi] 4. sign: rear peak x 1.5 gives a more negative index")
ctx_r = build(write_car("rear15.car", rear_scale=1.5))
ev_r = make_ev(ctx_r.m23)
out_r = ev_at(ctx_r.m23, ev_r, zero_slip_state(ctx_r.m23, ev_r, *LIGHT))
ok(f"rear peak x 1.5: index {out_r[0]:.6f} % < base {model_val:.6f} %",
   out_r[0] < model_val)

print("[cs_stbi] 5. QSS apex correction (cids 655 and 656)")
ZERO_MAP = (np.array([-40.0, 0.0, 40.0]), np.array([10.0, 30.0, 60.0]))


def map_over(flag, value):
    x, z = ZERO_MAP
    return {655: (6, flag), 656: (9, (x, z, np.full(x.size * z.size, value)))}


def qenv(path):
    """Return (envelope, s, k) of the car: build_envelope_cs on a fake-IPOPT build."""
    c = build(path)
    return build_envelope_cs(c), c.track.s, c.track.k


env_b, s_b, k_b = qenv(BASE)
mb = march(env_b, s_b, k_b, 60.0)
ok("base car (no cids 655 and 656): no apex factor", env_b.apex_factor is None)
env_off, _, _ = qenv(write_car("off_live.car", over=map_over(0, 0.9)))
ok("655 = 0 with a live map: apex factor None and lap bit-equal to the base car",
   env_off.apex_factor is None and float(march(env_off, s_b, k_b, 60.0)["lap_time"]) == mb["lap_time"])
env_zero, _, _ = qenv(write_car("on_zero.car", over=map_over(1, 0.0)))
ok("655 = 1 with a zero map: apex factor None and lap bit-equal to the base car",
   env_zero.apex_factor is None and float(march(env_zero, s_b, k_b, 60.0)["lap_time"]) == mb["lap_time"])
env_one, _, _ = qenv(write_car("on_one.car", over=map_over(1, 1.0)))
lap_one = float(march(env_one, s_b, k_b, 60.0)["lap_time"])
ok(f"655 = 1 with a map of 1.0: apex factor set, lap {lap_one:.9f} s equal to the base {mb['lap_time']:.9f} s (rtol 1e-12)",
   env_one.apex_factor is not None and rel(lap_one, mb["lap_time"]) < 1e-12)
env_09, _, _ = qenv(write_car("on_09.car", over=map_over(1, 0.9)))
m9 = march(env_09, s_b, k_b, 60.0)
mask = (np.abs(mb["k"]) > 0.0) & (mb["v_apex"] >= _V_FLOOR / 0.9) & (mb["v_apex"] < env_b.v_cap)
ok(f"655 = 1 with a map of 0.9: apex speeds x0.9 on {int(mask.sum())} curved stations (rtol 1e-12)",
   bool(mask.any()) and bool(np.allclose(m9["v_apex"][mask], 0.9 * mb["v_apex"][mask], rtol=1e-12, atol=0.0)))
ok(f"655 = 1 with a map of 0.9: slower lap {m9['lap_time']:.4f} s > base {mb['lap_time']:.4f} s",
   m9["lap_time"] > mb["lap_time"])
k_flat = np.full_like(s_b, 1e-7)               # so little curvature that every base apex speed is the speed cap
mb_f, m9_f = march(env_b, s_b, k_flat, 60.0), march(env_09, s_b, k_flat, 60.0)
ok(f"map of 0.9, base apex at the speed cap {env_b.v_cap:.2f} m/s: the apex speed stays (no factor)",
   bool(np.all(mb_f["v_apex"] == env_b.v_cap)) and bool(np.array_equal(m9_f["v_apex"], mb_f["v_apex"])))
scr = quiet(MLTP_screen, circuit=f"chassissim:{CIRC}", car=f"chassissim:{os.path.join(TMP.name, 'on_09.car')}",
            chassissim_dir=INSTALL, physics="corrected", save=True, results_dir=os.path.join(TMP.name, "qss"), verbose=False)
ok(f"MLTP_screen on the map of 0.9 car saves a file, lap {scr.data.lap_time:.9f} s equal to the march lap",
   os.path.isfile(scr.out_path) and scr.data.lap_time == m9["lap_time"] and "apex_factor" not in scr.data.envelope)
BAD_AXES = (np.array([50.0, 0.0, -50.0]), np.array([10.0, 30.0, 60.0]), np.full(9, 0.9))
try:
    load_car(write_car("bad_axis.car", over={655: (6, 1), 656: (9, BAD_AXES)}))
    raised = ""
except ValueError as e:
    raised = str(e)
ok("655 = 1 and a live map whose x axis does not increase raises ValueError ('axes that do not increase strictly')",
   "axes that do not increase strictly" in raised)
ok("655 = 0 with the same map loads (the map is not used)",
   load_car(write_car("bad_axis_off.car", over={655: (6, 0), 656: (9, BAD_AXES)})).stability.on == 0)

print("[cs_stbi] 6. fake-IPOPT build of the synthetic car: the channel in data.vehicle")
stbi_vec = np.asarray(ctx.data.vehicle["stbi"], float)
ok(f"data.vehicle['stbi'] holds {stbi_vec.size} finite values ({np.min(stbi_vec):.2f} to {np.max(stbi_vec):.2f} %)",
   stbi_vec.size > 0 and bool(np.all(np.isfinite(stbi_vec))))

# ---- vendor section ------------------------------------------------------------------------------------------
def vendor(cs_dir):
    models = Path(cs_dir).parent / "Models"
    print("[cs_stbi] vendor: GP3 2013 car, channel and seeded rebuild")
    gp3 = sorted(glob.glob(os.path.join(str(models), "GP3*", "*", "def*.car")))
    ok("GP3 2013 car file found by folder glob", bool(gp3))
    c3 = build(gp3[0], chassis_dir=cs_dir, circuit="Eastern Creek")
    v3 = np.asarray(c3.data.vehicle["stbi"], float)
    ok(f"GP3 2013 builds with the channel, {v3.size} finite values", bool(np.all(np.isfinite(v3))))

    seed_dir = os.environ.get("CS_STBI_SEED_DIR")
    seeds = glob.glob(os.path.join(seed_dir, "*car_def*nbCustom*.mat")) if seed_dir else []
    if not seeds:
        print("  SKIP: CS_STBI_SEED_DIR not set, or no stored GP3 fixed-line result in it")
        return
    seed = max(seeds, key=os.path.getmtime)
    cs_seed = quiet(MLTP, car=f"chassissim:{gp3[0]}", circuit="chassissim:Eastern Creek", chassissim_dir=cs_dir,
                    physics="corrected", save=False, plot=False, linear_solver="mumps", n_bounds=(-0.05, 0.05),
                    warm_start=seed)
    med = float(np.median(np.asarray(cs_seed.data.vehicle["stbi"], float)))
    ok(f"seeded GP3 rebuild: median stbi {med:.2f} % within 2 points of the log {GP3_STBI_LOG} %",
       abs(med - GP3_STBI_LOG) < 2.0)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
print(f"{PASS} checks passed")
