"""ChassisSim phase 2c P3: the gear shift cut as a time-matched envelope band (vehModel_cs, functions/ggv).

A car file with the cut fields (cids 104 to 106) goes through build_envelope_cs and MLTP with a fake IPOPT (no
solve). The synthetic car is written with carfile.write in a temp folder; the cids and record tags are literal
numbers here. Checks:
  1. shift_bumps law: v_g = rpm_limit R / ratio_g for each gear below the top one, one band per gear with F > R,
     W = max((1 - minF) F t / m_eff, 3 m/s), and the crossing time loss of the band equals tau = (1 - minF) F t /
     (F - R) (own quadrature over own bump),
  2. delay 0 gives no band and the same graph (dx, sf) and path rows as a car without cut fields, minF 1 gives no
     band, minF outside [0, 1] with delay > 0 raises ValueError, minF 0 gives bands with c > 0,
  3. the QSS drive table = raw gear envelope x (1 - sum c w), and the NLP F_env at the same speeds matches it,
  4. the throttle channel of a fake-IPOPT build stays in [0, 1].
Vendor section (SKIP without CHASSISSIM_DIR): the GT3 car (found by folder glob) has 4 bands at the upshift speeds
with no band at the fifth upshift, and its Adelaide QSS lap rises by 0.10 to 0.17 s against the same car with
delay 0. The GP3 and F2 cars have no band and keep their QSS laps.

    python tests/test_cs_shift.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_shift.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import glob
import hashlib
import io
import os
import tempfile
import warnings
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.integrate import quad

import functions.transcription as T
from functions import carfile
from functions.ggv import build_envelope_cs
from MLTP import MLTP
from MLTP_screen import MLTP_screen
import vehModel_cs as V

PASS = 0
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}
G = 9.81
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"
CUT = {104: 0.1, 105: 0.5, 106: 0.01}      # delay [s], minimum torque factor, ramp [s] (the GT3 values, arbitrary here)


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
    rows = [f"{float(k)!r}   {float(s)!r}" for k, s in zip(K, S)]
    write_text(os.path.join(circ, "tr_synth.txt"), "\r\n".join(rows) + "\r\n")
    return root


INSTALL = build_install(os.path.join(TMP.name, "ChassisSim"))


# ---- synthetic car: literal cids and tags (as in tests/test_vehmodel_cs.py) ----------------------------
# Tags: 1 scalar, 2 array, 6 byte, 7 curve, 9 map, 10 hardpoints, 11 spring-like, 12 damper, 13 tyre.
TYRE = {
    "fl": dict(dims=(3, 2, 3, 2, 2, 5, 4), radius=0.300, toe=0.0, camber=-1.0, castor=2.0, drag=0.011, kz=1.60e5),
    "fr": dict(dims=(2, 2, 3, 2, 2, 5, 4), radius=0.305, toe=0.0, camber=-1.5, castor=2.5, drag=0.012, kz=1.62e5),
    "rl": dict(dims=(4, 2, 3, 3, 2, 5, 4), radius=0.310, toe=0.0, camber=-0.5, castor=1.5, drag=0.0135, kz=1.40e5),
    "rr": dict(dims=(2, 2, 3, 2, 3, 5, 4), radius=0.315, toe=0.0, camber=-0.75, castor=1.0, drag=0.014, kz=1.42e5),
}
TYRE_TEMPS = (300.0, 350.0)
TYRE_BG = 325.0
PRESSURE = 2.1e5
VEL = np.array([0.0, 100.0, 200.0, 400.0, 800.0])
BUMP = np.array([0.0, 900.0, 1800.0, 2700.0, 3600.0])
REB = np.array([0.0, 600.0, 1200.0, 1800.0, 2400.0])
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
    100: 0.9, 102: 1000.0, 103: 9000.0,                     # efficiency, idle and limit rpm
    107: 14000.0, 108: 0.6, 109: 0.05,                      # brake force N, front bias, brake delay s
    111: 0.0, 112: 1.0, 113: 1.0,                           # torque split (0 = rear drive), diff types (1 = open)
    55: 0.03, 59: 0.03, 63: 0.03, 67: 0.03,                # roll-centre heights per corner, m
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,                # CLA, CDA, offset, rho
    390: 350.0,
}
GAP = {7: 0.008, 8: 0.009, 9: 0.0085, 10: 0.0095, 11: 0.0, 12: 0.0}
EXPLICIT.update(GAP)
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})
EXPLICIT.update({142: 0.5 * (EXPLICIT[117] + EXPLICIT[118]) * 1e-3,
                 144: 0.5 * (EXPLICIT[119] + EXPLICIT[120]) * 1e-3, 56: 0.0, 60: 0.0, 64: 0.0, 68: 0.0})
EXPLICIT.update({c: 0.0 for c in (55, 59, 63, 67) if c not in EXPLICIT})
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
AERO_X = 0.005 * np.arange(7)
AERO_Z = 0.002 + 0.005 * np.arange(7)
MAPS = {114: ((4, 4), "diff_f"), 116: ((4, 4), "diff_r")}


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


def build_records(cut=None):
    """The car records. cut: extra or replaced explicit cids (the shift cut fields 104 to 106)."""
    R = []

    def add(tag, cid, v):
        cid = FILE_CID.get(cid, cid)
        R.append((tag, cid, v))
    ex = dict(EXPLICIT)
    ex.update(cut or {})
    for cid, v in ex.items():
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
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


def write_car(name, cut=None):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(cut), path)
    return path


def build(car_path, **kw):
    """MLTP on the synthetic install with a fake IPOPT (no solve)."""
    opts = dict(car=f"chassissim:{car_path}", circuit=f"chassissim:{CIRC}", chassissim_dir=INSTALL,
                physics="corrected", save=False, plot=False, linear_solver="mumps")
    opts.update(kw)
    return quiet(MLTP, **opts)


def static_x(ctx, v):
    """The static point at speed v on a straight: zs = theta = phi = 0, wheels at the rolling speed."""
    vp, cs = ctx.vp, ctx.cs
    x = np.zeros(23)
    x[0] = v
    x[5:7] = v / cs.tyre.fl["radius"]
    x[7:9] = v / cs.tyre.rl["radius"]
    x[19:23] = [getattr(vp, f"W{c}0") / vp.kt for c in CORNERS]
    return x


def graph_hash(c_):
    """Hash of the model (dx, sf) and of the path rows (h) of a built car."""
    mm = c_.m23
    d = ca.Function("d", [mm.x, mm.u, mm.pv], [mm.dx, mm.sf]).serialize()
    _, h, _, _ = V.cs_path_constraints(ca, mm, c_)
    p = ca.Function("h", [mm.x, mm.u, mm.pv], [h]).serialize()
    return hashlib.sha256(d.encode()).hexdigest(), hashlib.sha256(p.encode()).hexdigest()


def bump(v, vg, W):
    """Independent copy of the band shape: about 1 on [vg, vg + W], tanh edges of width W / 10."""
    return 0.25 * (1.0 + np.tanh((v - vg) / (0.1 * W))) * (1.0 - np.tanh((v - vg - W) / (0.1 * W)))


# ---- 1. the band law on the cut car ----------------------------------------------------------------
print("[cs_shift] 1. band law on the cut car (delay 0.1, minF 0.5, ramp 0.01)")
ctx_c = build(write_car("cut.car", CUT))
cs_c = ctx_c.cs
e = cs_c.engine
env_c = quiet(build_envelope_cs, ctx_c)
bands = env_c.info["shift_bands"]
R_t = cs_c.tyre.rl["radius"]
m_eff, c0, c2 = env_c.M_eff, env_c.c_res0, env_c.c_res2
t_d = e.cut_delay + 0.5 * e.cut_ramp
x_s = ca.SX.sym("v")
F_raw = ca.Function("F", [x_s], [V._env_raw(cs_c, R_t, x_s)[0]])
top = float(np.min(e.gears))
below = [float(g) for g in np.sort(np.asarray(e.gears, float))[::-1] if g != top]
expect = []
for g in below:
    vg = e.rpm_limit * R_t / g
    F, Rg = float(F_raw(vg)), max(c0 + c2 * vg * vg, 0.0)
    if F > Rg:
        expect.append((vg, F, Rg))
ok(f"one band per gear below the top with F > R: {len(bands)} bands, expected {len(expect)} "
   f"(gears below the top {len(below)})", len(bands) == len(expect) and len(bands) >= 1)
# The model's own resistance fit is above the drive force at every upshift on this car (c2 < 0). The resistance
# in the band law is max(c0 + c2 v_g^2, 0) (a resistance never pushes), so the no-band rule is checked with a flat
# resistance R = 1000 N: gears with F <= R get no band.
R_flat = 1000.0
b_flat = V.shift_bumps(cs_c, R_t, m_eff, (R_flat, 0.0))
exp_flat = [e.rpm_limit * R_t / g for g in below if float(F_raw(e.rpm_limit * R_t / g)) > R_flat]
ok(f"flat R 1000 N: bands only where F > R ({len(b_flat)} of {len(below)} upshifts, expected {len(exp_flat)})",
   len(b_flat) == len(exp_flat) == 2 and all(abs(vg - ve) <= 1e-12 * ve for (vg, _, _), ve in zip(b_flat, exp_flat)))
# A resistance fit with a negative slope (-8e4 N per (m/s)^2, no constant) must not push the depth past 1 / max(w)
# or the envelope factor below 0: the clamp keeps R_g >= 0 at every upshift.
b_neg = V.shift_bumps(cs_c, R_t, m_eff, (-8e4, 0.0))
vfine = np.linspace(0.0, 120.0, 24001)
fac_neg = 1.0 - sum(c * bump(vfine, vg, W) for vg, W, c in b_neg)
wmax_ok = all(c * float(np.max(bump(np.linspace(vg - W, vg + 2.0 * W, 6001), vg, W))) < 1.0 for vg, W, c in b_neg)
ok(f"resistance fit (-8e4, 0): {len(b_neg)} bands with c < 1 / w_max (all bands: {wmax_ok}), "
   f"depths {[round(c, 4) for _, _, c in b_neg]}", len(b_neg) >= 1 and wmax_ok)
ok(f"resistance fit (-8e4, 0): the envelope factor stays above 0 at every speed (min {fac_neg.min():.4f})",
   float(fac_neg.min()) > 0.0)
for (vg, W, c), (vg_e, F, Rg) in zip(bands, expect):
    ok(f"v_g = rpm_limit R / g at {vg:.3f} m/s", abs(vg - vg_e) <= 1e-12 * vg_e)
    W_e = max((1.0 - e.cut_min) * F * t_d / m_eff, 3.0)
    ok(f"W at v_g {vg:.3f}: {W:.4f} m = max((1 - minF) F t / m_eff, 3)", abs(W - W_e) <= 1e-12 * W_e)
    tau = (1.0 - e.cut_min) * F * t_d / (F - Rg)
    lo, hi = vg - W, vg + 2.0 * W
    loss = m_eff * quad(lambda v: 1.0 / (F * (1.0 - c * bump(v, vg, W)) - Rg) - 1.0 / (F - Rg), lo, hi,
                        points=[vg, vg + W], limit=500, epsabs=0.0, epsrel=1e-11)[0]
    ok(f"crossing time loss at v_g {vg:.3f} = tau = {1e3 * tau:.2f} ms (rel {abs(loss / tau - 1):.1e}, c {c:.5f})",
       abs(loss / tau - 1.0) < 1e-6 and 0.0 < c < 1.0)

# ---- 2. delay, minF and the refusals ----------------------------------------------------------------
print("[cs_shift] 2. delay 0, minF 1, minF outside [0, 1], minF 0")
ctx_base = build(write_car("base.car"))
hb = graph_hash(ctx_base)
ctx_d0 = build(write_car("delay0.car", {104: 0.0, 105: 0.5, 106: 0.01}))
ok("delay 0: no band (shift_bands empty)", quiet(build_envelope_cs, ctx_d0).info["shift_bands"] == [])
ok("delay 0: the (dx, sf) and path-row graph hashes equal those of the car without cut fields",
   graph_hash(ctx_d0) == hb)
ctx_m1 = build(write_car("minf1.car", {104: 0.1, 105: 1.0, 106: 0.01}))
ok("minF 1.0 (delay 0.1): no band", quiet(build_envelope_cs, ctx_m1).info["shift_bands"] == [])
ok("minF 1.2 with delay > 0: ValueError", raises(ValueError, build, write_car("minf12.car", {104: 0.1, 105: 1.2, 106: 0.01})))
ok("minF -0.1 with delay > 0: ValueError", raises(ValueError, build, write_car("minfm.car", {104: 0.1, 105: -0.1, 106: 0.01})))
ctx_m0 = build(write_car("minf0.car", {104: 0.1, 105: 0.0, 106: 0.01}))
b0 = quiet(build_envelope_cs, ctx_m0).info["shift_bands"]
ok(f"minF 0: bands with c > 0 ({len(b0)} bands, c {[round(c, 4) for _, _, c in b0]})",
   len(b0) >= 1 and all(c > 0.0 for _, _, c in b0))

# ---- 3. QSS drive table = raw x (1 - sum c w), and the NLP F_env at the same speeds --------------------
print("[cs_shift] 3. QSS drive table against the raw envelope and the NLP F_env")
v_grid = np.asarray(env_c.v_grid, float)
idx = set(np.linspace(0, v_grid.size - 1, 9).astype(int).tolist())
for vg, W, c in bands:
    idx.add(int(np.argmin(np.abs(v_grid - (vg + 0.5 * W)))))
idx = sorted(idx)
m = ctx_c.m23
F_nlp = ca.Function("Fe", [m.x, m.u, m.pv], [m.F_env])
worst_q, worst_n, dip_seen = 0.0, 0.0, False
for i in idx:
    v = float(v_grid[i])
    raw = float(F_raw(v))
    fac = 1.0 - sum(c * bump(v, vg, W) for vg, W, c in bands)
    qss = float(env_c.F_drive_tab[i])
    worst_q = max(worst_q, abs(qss - raw * fac) / abs(raw * fac))
    nlp = float(F_nlp.call([static_x(ctx_c, v) / m.x_s, np.zeros(m.nu), 0.0])[0])
    worst_n = max(worst_n, abs(nlp - qss) / abs(qss))
    if fac < 1.0 - 1e-6:
        dip_seen = True
ok(f"QSS drive table = raw x (1 - sum c w) at {len(idx)} speeds: max rel {worst_q:.1e} (bound 1e-9)", worst_q < 1e-9)
ok(f"NLP F_env equals the QSS drive table at the same speeds: max rel {worst_n:.1e} (bound 1e-9)", worst_n < 1e-9)
ok("the table dips below the raw envelope at a band", dip_seen)

# ---- 4. throttle channel of a fake-IPOPT build -----------------------------------------------------
print("[cs_shift] 4. throttle channel in [0, 1]")
thr = np.asarray(ctx_c.data.vehicle["throttle"], float)
ok(f"throttle finite and in [0, 1] on {thr.size} knots: min {thr.min():.4f}, max {thr.max():.4f}",
   thr.size > 0 and bool(np.all(np.isfinite(thr))) and thr.min() >= 0.0 and thr.max() <= 1.0)


# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs_dir):
    print("[cs_shift] vendor: GT3 bands and Adelaide lap, GP3 and F2 unchanged")
    models = glob.escape(str(Path(cs_dir).parent / "Models"))
    gt3 = sorted(glob.glob(os.path.join(models, "GT3", "*", "*.car")))
    ok("one GT3 car file by glob GT3/*/*.car", len(gt3) == 1)
    kw = dict(circuit="chassissim:Adelaide", chassissim_dir=cs_dir, physics="corrected", save=False, verbose=False)
    c_cut = quiet(MLTP_screen, car=f"chassissim:{gt3[0]}", **kw)
    b = c_cut.envelope.info["shift_bands"]
    vg_ref = [41.88, 50.76, 59.44, 68.80]
    vgs = [vg for vg, _, _ in b]
    ok(f"GT3: 4 bands at the upshift speeds {[round(float(v), 2) for v in vgs]} (ref {vg_ref})",
       len(vgs) == 4 and all(abs(a - r) < 0.02 for a, r in zip(vgs, vg_ref)))
    ok("GT3: no band at the fifth upshift (80.41 m/s)", all(abs(v - 80.411) > 0.5 for v in vgs))
    nocut = carfile.decode(gt3[0])
    nocut = [(t, c, 0.0 if c == 104 else v) for t, c, v in nocut]
    p0 = os.path.join(TMP.name, "gt3_delay0.car")
    carfile.write(nocut, p0)
    c_0 = quiet(MLTP_screen, car=f"chassissim:{p0}", **kw)
    d_lap = float(c_cut.data.lap_time) - float(c_0.data.lap_time)
    ok(f"GT3 Adelaide QSS lap rises 0.10 to 0.17 s with the cut: {d_lap:+.4f} s "
       f"(time-domain reference +0.127 s)", 0.10 <= d_lap <= 0.17)
    gp3 = sorted(glob.glob(os.path.join(models, "GP3*", "*Eastern*", "def*.car")))
    ok("GP3 2013 Eastern Creek car found by glob", len(gp3) == 1)
    c_gp3 = quiet(MLTP_screen, car=f"chassissim:{gp3[0]}", circuit="chassissim:Eastern Creek", chassissim_dir=cs_dir,
                  physics="corrected", save=False, verbose=False)
    lap_gp3 = float(c_gp3.data.lap_time)
    ok(f"GP3 no band and QSS lap moved with the roll-centre fix (ground z = 0) to 74.12461931861725 s: "
       f"{lap_gp3:.10f} s",
       c_gp3.envelope.info["shift_bands"] == [] and abs(lap_gp3 - 74.12461931861725) < 1e-6)
    f2 = sorted(glob.glob(os.path.join(models, "Formula 2", "*", "*.car")))
    ok("F2 car found by glob", len(f2) == 1)
    c_f2 = quiet(MLTP_screen, car=f"chassissim:{f2[0]}", **kw)
    lap_f2 = float(c_f2.data.lap_time)
    ok(f"F2 no band and QSS lap moved with the roll-centre fix (ground z = 0) to 59.52389545490832 s: "
       f"{lap_f2:.10f} s",
       c_f2.envelope.info["shift_bands"] == [] and abs(lap_f2 - 59.52389545490832) < 1e-6)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set (vendor section: GT3 bands and lap, GP3 and F2 laps)")
else:
    vendor(CS)
print(f"{PASS} checks passed")
