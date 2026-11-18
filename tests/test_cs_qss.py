"""ChassisSim phase 2 T5: the QSS envelope of a ChassisSim car (functions/ggv.build_envelope_cs).

A synthetic car file (arbitrary values made positive where the QSS needs them, written with carfile.write; the cids
and record tags are literal numbers here) on a synthetic ChassisSim install with one short circuit goes through
MLTP_screen(car=..., physics='corrected', save=False). Checks: the envelope comes from build_envelope_cs and carries
the gear envelope (4EM: none), the drive force equals a numpy recompute of the gear envelope, the braking force stays
under the car brake force (cid 107), ay_max and the downforce, the lap (finite, lower with a stronger tyre table,
higher with more drag), and the 4EM default QSS lap of Sturn. The GP3 2013 car on Eastern Creek runs when
CHASSISSIM_DIR names the install (else SKIP). Nothing is written into the vendor folders.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_qss.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_qss.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import tempfile
import warnings
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import PchipInterpolator

from functions import carfile
from functions.ggv import build_envelope
from MLTP_screen import MLTP_screen
from vehModel_cs import _gear_envelope

PASS = 0
CORNERS = ("fl", "fr", "rl", "rr")
G = 9.81
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"
B_GEAR = 80.0          # vehModel_cs.B_GEAR, the sharpness of the gear maximum
BRAKE_FORCE = 14000.0  # cid 107 of the synthetic car


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


# ---- synthetic car: literal cids and tags (as in tests/test_vehmodel_cs.py) ---------------------------
# Tags: 1 scalar, 2 array, 6 byte, 7 curve, 9 map, 10 hardpoints, 11 spring-like, 12 damper, 13 tyre.
TYRE = {  # dims (d1..d7), radius m, toe deg, camber, castor, drag, kz N/m
    "fl": dict(dims=(3, 2, 2, 2, 2, 2, 2), radius=0.300, toe=0.0, camber=-1.0, castor=2.0, drag=0.011, kz=1.60e5),
    "fr": dict(dims=(2, 2, 3, 2, 2, 3, 2), radius=0.305, toe=0.0, camber=-1.5, castor=2.5, drag=0.012, kz=1.62e5),
    "rl": dict(dims=(4, 2, 2, 3, 2, 2, 2), radius=0.310, toe=0.0, camber=-0.5, castor=1.5, drag=0.0135, kz=1.40e5),
    "rr": dict(dims=(2, 2, 2, 2, 3, 2, 3), radius=0.315, toe=0.0, camber=-0.75, castor=1.0, drag=0.014, kz=1.42e5),
}
TYRE_TEMPS = (300.0, 350.0)
TYRE_BG = 325.0
PRESSURE = 2.1e5
VEL = np.array([0.0, 100.0, 200.0, 400.0, 800.0])        # mm/s, from 0 up
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
    107: BRAKE_FORCE, 108: 0.6, 109: 0.05,                  # brake force N, front bias, delay s
    111: 0.3, 112: 1.0, 113: 0.0,                           # torque split, diff types
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,                # CLA, CDA, offset, rho
    390: 350.0,
}
EXPLICIT.update({c: 0.5 + 0.01 * c for c in range(7, 13)})            # suspension gaps
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})        # DRS
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
MAPS = {127: ((4, 4), "cla"), 131: ((4, 4), "cda"), 135: ((5, 4), "balance"),   # casadi bspline: >= 4 per axis
        114: ((4, 4), "diff_f"), 116: ((4, 4), "diff_r")}


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
    d1, d2, d3, _, d5, d6, _ = dims
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(dims))]
    A, B = arb(d1, 20 + i), arb(d1, 30 + i)
    load = 1000.0 * np.arange(1, d1 + 1)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, np.stack([A, B], axis=1).ravel()])
    tabs[0][:d3] = np.linspace(0.1, 1.5, d3)                 # slip abscissae, increasing
    tabs[5][:d6] = np.linspace(-3.0, 3.0, d6)                # camber abscissae (deg), increasing
    px = np.linspace(150.0, 250.0, d5)
    py = t["kz"] * (1.0 + 0.01 * np.arange(d5))
    tabs[10] = np.concatenate([px, py])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = t["toe"], t["camber"], t["castor"], t["radius"], PRESSURE, TYRE_BG
    sc[7], sc[8], sc[9], sc[11], sc[15] = t["drag"], 0.11 + 0.01 * i, 0.12 + 0.01 * i, 0.7 + 0.1 * i, 1.05 + 0.1 * i
    return 13, 80 + i, (list(dims), sc, tabs)


def shape(R, cda_scale=1.0, peak_scale=1.0):
    """Make the arbitrary record values physical for the QSS: positive peak grid (fy_peak = 1.2 x load x a
    temperature factor, times peak_scale), slip and camber factors 1, a positive CLA / CDA map (CDA x cda_scale)
    and a balance of 0.45. Only the values change, the tables keep their shapes."""
    out = []
    for t, cid, v in R:
        if t == 9 and cid == 127:                       # cla map: downforce [m2]
            x, z, d = v
            v = (x, z, np.full(len(d), 3.5))
        elif t == 9 and cid == 131:                     # cda map: drag coefficient x cda_scale
            x, z, d = v
            v = (x, z, np.full(len(d), 0.95 * cda_scale))
        elif t == 9 and cid == 135:                     # balance map: front share of the downforce
            x, z, d = v
            v = (x, z, np.full(len(d), 0.45))
        elif t == 13:
            dims, sc, tabs = v
            tabs = [np.array(a, float) for a in tabs]
            d1, d2, d3, _, _, d6, _ = (int(d) for d in dims)
            load = tabs[3][:d1]
            grid = np.outer(load, np.linspace(1.0, 1.1, d2)) * 1.2 * peak_scale
            tabs[3] = np.concatenate([load, tabs[3][d1:d1 + d2], grid.ravel()])
            tabs[0][d3:] = 1.0                          # normalised slip curve peak 1
            tabs[5][d6:] = 1.0                          # camber factor 1
            v = (dims, sc, tabs)
        out.append((t, cid, v))
    return out


def build_records(cda_scale=1.0, peak_scale=1.0):
    R = []

    def add(tag, cid, v):
        R.append((tag, cid, v))
    for cid, v in EXPLICIT.items():
        add(1, cid, v)
    for cid, (rate, pre) in SPRING.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.05]), np.array([0.0, 500.0])))
    for cid, (rate, pre) in RUBBER.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.02]), np.array([0.0, 800.0])))
    for cid, rate in ARB.items():
        add(11, cid, (1, 0.0, rate, np.array([0.0, 0.1]), np.array([0.0, 300.0])))
    for cid in range(23, 31):
        add(2, cid, arb(5, cid))                                          # motion ratio coefficients
    for cid in range(366, 372):
        add(6, cid, 10 + cid - 366)
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    for cid, ((nx, nz), _) in MAPS.items():
        add(9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid)))
    for cid in range(31, 36):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    for cid in HARD:
        add(10, cid, (0, arb(18, cid)))
    for i, c in enumerate(CORNERS):
        t, cid, v = tyre_record(i, c)
        add(t, cid, v)
    R = shape(R, cda_scale, peak_scale)
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


def write_car(name, cda_scale=1.0, peak_scale=1.0):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(cda_scale, peak_scale), path)
    return path


def qss(car_path=None, **kw):
    """MLTP_screen on the synthetic install (car_path None: the 4EM default Sturn). Returns ctx."""
    if car_path is None:
        return quiet(MLTP_screen, circuit="Sturn", save=False, verbose=False, **kw)
    opts = dict(car=f"chassissim:{car_path}", circuit=f"chassissim:{CIRC}", chassissim_dir=INSTALL,
                physics="corrected", save=False, verbose=False)
    opts.update(kw)
    return quiet(MLTP_screen, **opts)


def drive_recompute(cs, v):
    """Numpy recompute of the gear envelope [N] at wheel speeds v [m/s]: the smooth maximum over the gears of
    eff x torque(engine speed) x ratio / R, idle clip, 2 % tanh window at the rpm limit, log-sum-exp with B_GEAR."""
    e = cs.engine
    Rr = cs.tyre.rl["radius"]
    x, y = np.asarray(e.torque_curve["x"], float), np.asarray(e.torque_curve["y"], float)
    te = PchipInterpolator(x, y)
    lo, hi = float(x.min()), float(x.max())
    F_ref = float(e.efficiency * te(np.linspace(e.rpm_idle, e.rpm_limit, 400)).max() * np.max(e.gears) / Rr)
    dw = 0.01 * e.rpm_limit
    fi = []
    for Gr in np.asarray(e.gears, float):
        w = v * Gr / Rr
        win = 0.5 * (1.0 + np.tanh((e.rpm_limit - w) / dw))
        fi.append(e.efficiency * te(np.clip(np.maximum(w, e.rpm_idle), lo, hi)) * Gr / Rr * win)
    return F_ref / B_GEAR * np.log(sum(np.exp(B_GEAR * f / F_ref) for f in fi))


print("[cs_qss] synthetic car: envelope source and drive table")
BASE = write_car("base.car")
ctx = qss(BASE)
env = ctx.envelope
ok("envelope comes from build_envelope_cs (load_model 'car', info car)",
   env.load_model == "car" and bool(env.info.get("car")) and getattr(ctx, "cs", None) is not None)
ok("F_drive_tab is set for the car (length = v_grid)",
   env.F_drive_tab is not None and np.size(env.F_drive_tab) == np.size(env.v_grid))
env_direct = build_envelope(ctx)
ok("build_envelope(ctx) on the same ctx gives the same drive table",
   env_direct.F_drive_tab is not None and np.array_equal(env_direct.F_drive_tab, env.F_drive_tab))
ctx4 = qss(None, Electric_4Motors="On")
ok("4EM ctx: F_drive_tab is None", ctx4.envelope.F_drive_tab is None)

idx = [0, 10, 20, env.v_grid.size - 1]
rel = []
for i in idx:
    v = env.v_grid[i]
    ref = float(drive_recompute(ctx.cs, np.array([v]))[0])
    rel.append(abs(env.F_drive_tab[i] - ref) / abs(ref))
ok(f"F_drive_tab equals the numpy gear envelope recompute within 1e-6 relative at 4 speeds (max {max(rel):.2e})",
   max(rel) < 1e-6)

# the casadi gear envelope of vehModel_cs (the NLP's own function). The QSS uses a hard idle clip and PCHIP, the
# NLP a 5 rpm smooth clip, a 200-point spline and a smooth clamp at the ends, so they differ at the two ends of the
# speed range (about 1e-3 near idle and at the top speed). Between 25 and 265 m/s they agree to 1e-4.
dev_all, dev_mid = [], []
for v, Fd in zip(env.v_grid, env.F_drive_tab):
    F_nlp = float(_gear_envelope(ctx.cs, ctx.cs.tyre.rl["radius"], ca.DM(float(v)))[0])
    dev_all.append(abs(Fd - F_nlp) / abs(F_nlp))
    if 25.0 <= v <= 265.0:
        dev_mid.append(dev_all[-1])
print(f"  [info] casadi vehModel_cs gear envelope vs F_drive_tab: max rel {max(dev_all):.2e} over the grid, "
      f"{max(dev_mid):.2e} for 25..265 m/s")
ok("F_drive_tab within 1e-4 of the NLP casadi gear envelope for 25..265 m/s", max(dev_mid) < 1e-4)

print("[cs_qss] envelope pieces")
F_brk = float(env.F_brk_trq)
ok(f"brake force equals cid 107 ({BRAKE_FORCE:.0f} N): {F_brk:.1f} N", abs(F_brk - BRAKE_FORCE) < 1e-6)
res = env.F_res(env.v_grid)
ok("ax_brk * M_eff - resistance stays under the car brake force at every speed",
   bool(np.all(env.ax_brk * env.M_eff - res <= F_brk * (1 + 1e-9) + 1e-6)))
ok("Fx_brk (tyre) is finite and positive on the grid", bool(np.all(np.isfinite(env.Fx_brk)) and np.all(env.Fx_brk > 0)))
ok(f"ay_max positive on the grid (min {env.ay_max.min() / G:.2f} g)", bool(np.all(env.ay_max > 0)))
ok(f"ay_max grows with speed with the car downforce: {env.ay_max[0] / G:.2f} g at {env.v_grid[0]:.1f} m/s, "
   f"{env.ay_max[-1] / G:.2f} g at {env.v_grid[-1]:.1f} m/s",
   bool(env.F_lift[-1] > 0 and env.ay_max[-1] > env.ay_max[0]))
ok("aero downforce positive at the top speed (F_lift)", bool(env.F_lift[-1] > 0))

print("[cs_qss] lap")
lap = float(ctx.data.lap_time)
ok(f"car QSS lap finite and positive: {lap:.3f} s", bool(np.isfinite(lap) and lap > 0))

DBL = write_car("peak2.car", peak_scale=2.0)
lap_peak = float(qss(DBL).data.lap_time)
ok(f"doubling the tyre peak table lowers the QSS lap: {lap:.3f} s -> {lap_peak:.3f} s", lap_peak < lap)

CDA = write_car("cda15.car", cda_scale=1.5)
lap_cda = float(qss(CDA).data.lap_time)
ok(f"raising CDA (drag map x1.5) raises the QSS lap: {lap:.3f} s -> {lap_cda:.3f} s", lap_cda > lap)

print("[cs_qss] 4EM default QSS lap (unchanged)")
sturn = float(qss().data.lap_time)
ok(f"4EM Sturn QSS lap equals 18.74393 s to 1e-5 s: {sturn:.6f} s", abs(sturn - 18.74393) < 1e-5)


# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs_dir):
    print("[cs_qss] vendor GP3 2013 Eastern Creek")
    car_path = Path(cs_dir).parent / "Models" / "GP3 2013" / "Eastern Creek, AU" / "def_GP3 2013.car"
    ok("GP3 2013 Eastern Creek car file found", car_path.is_file())
    c = quiet(MLTP_screen, car=f"chassissim:{car_path}", circuit="chassissim:Eastern Creek", chassissim_dir=cs_dir,
              physics="corrected", save=False, verbose=False)
    gl = float(c.data.lap_time)
    ok("GP3 envelope carries the drive table", c.envelope.F_drive_tab is not None)
    ok(f"GP3 QSS lap finite and within 15 % of 76.742 s: {gl:.3f} s "
       f"({100 * (gl / 76.742 - 1):+.1f} %)", bool(np.isfinite(gl)) and abs(gl / 76.742 - 1) < 0.15)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
print(f"{PASS} checks passed")
