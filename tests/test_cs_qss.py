"""ChassisSim phase 2 T5: the QSS envelope of a ChassisSim car (functions/ggv.build_envelope_cs).

A synthetic car file (arbitrary values made positive where the QSS needs them, written with carfile.write; the cids
and record tags are literal numbers here) on a synthetic ChassisSim install with one short circuit goes through
MLTP_screen(car=..., physics='corrected', save=False). Checks: the envelope comes from build_envelope_cs and carries
the gear envelope (4EM: none), the drive force equals a numpy recompute of the gear envelope, the braking force stays
under the car brake force (cid 107), ay_max and the downforce, the lap (finite, lower with a stronger tyre table,
higher with more drag), and the 4EM default QSS lap of Sturn. The GP3 2013 car on Eastern Creek runs when
CHASSISSIM_DIR names the install (else SKIP). Nothing is written into the vendor folders.

    python tests/test_cs_qss.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" python tests/test_cs_qss.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import tempfile
import warnings
from pathlib import Path

import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from functions import carfile
from functions.chassissim import list_cars, load_car
from functions.ggv import build_envelope
from MLTP_screen import MLTP_screen
import vehModel_cs as VCS
from vehModel_cs import B_GEAR, W_RUBBER, _surf
from functions.chassissim import suspension_curves

PASS = 0
CORNERS = ("fl", "fr", "rl", "rr")
# The value lists below name the corners in the order fl, fr, rl, rr. The car file stores the right wheel first
# (FR, FL, RR, RL), so the builder swaps the two cids of each left / right pair. An over or omit cid is a file cid.
FILE_CID = {a: b for p in ((1, 2), (3, 4), (7, 8), (9, 10), (13, 14), (15, 16), (23, 24), (25, 26), (31, 32), (33, 34),
                           (49, 50), (51, 52), (54, 58), (55, 59), (62, 66), (63, 67), (80, 81), (82, 83),
                           (117, 118), (119, 120), (204, 205), (206, 207)) for a, b in (p, p[::-1])}
G = 9.81
TMP = tempfile.TemporaryDirectory()
CIRC = "Synth Ring"
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
          5: (0.0, 0.0), 6: (0.0, 0.0)}
RUBBER = {cid: (2.0e5 + 1e3 * cid, 0.001 * (cid - 12)) for cid in range(13, 19)}
ARB = {19: 2.0e4, 21: 1.5e4}
HARD = (49, 50, 51, 52)                      # front left, front right, rear left, rear right
# left-side hardpoints in the car frame (x rearward, y left +, z up), rows: lower inner front, lower inner aft,
# lower outer, upper inner front, upper inner aft, upper outer. The rear uses the same rows; the right side mirrors y.
HP_LEFT = np.array([[-0.15, 0.25, 0.10], [0.15, 0.25, 0.10], [0.0, 0.68, 0.11],
                    [-0.15, 0.30, 0.28], [0.15, 0.30, 0.28], [0.0, 0.64, 0.33]])
HP_RIGHT = HP_LEFT * np.array([1.0, -1.0, 1.0])
RC_HEIGHT = 0.03                             # m, roll-centre heights (cids 55, 59, 63, 67)
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: 1.52, 92: 1.49, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,            # ride heights, mm
    100: 0.9, 102: 1000.0, 103: 9000.0,                     # efficiency, idle and limit rpm
    107: BRAKE_FORCE, 108: 0.6, 109: 0.05,                  # brake force N, front bias, delay s
    111: 0.0, 112: 1.0, 113: 0.0,                           # torque split, diff types
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,                # CLA, CDA, offset, rho
    390: 350.0,
}
EXPLICIT.update({c: 0.003 + 0.0005 * (c - 7) for c in range(7, 11)})  # ground gaps at the dampers, 3-4.5 mm
EXPLICIT.update({c: 0.5 + 0.01 * c for c in range(11, 13)})          # third-element gaps
EXPLICIT.update({c: 0.2 + 0.001 * c for c in range(336, 342)})        # DRS
EXPLICIT.update({142: 0.5 * (EXPLICIT[117] + EXPLICIT[118]) * 1e-3,
                 144: 0.5 * (EXPLICIT[119] + EXPLICIT[120]) * 1e-3, 56: 0.0, 60: 0.0, 64: 0.0, 68: 0.0})
EXPLICIT.update({c: 0.0 for c in (55, 59, 63, 67) if c not in EXPLICIT})
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
    dims = tuple(t["dims"][:5]) + (4, 4)        # d6 cambers and d7 loads of tables 4 and 5: >= 4 for the bspline
    d1, d2, d3, _, d5, d6, d7 = dims
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(dims))]
    A, B = arb(d1, 20 + i), arb(d1, 30 + i)
    load = 1000.0 * np.arange(1, d1 + 1)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, np.stack([A, B], axis=1).ravel()])
    tabs[4] = np.concatenate([np.linspace(-3.0, 3.0, d6), 1000.0 * np.arange(1, d7 + 1), np.ones(d6 * d7)])  # Fx camber factor 1
    tabs[0][:d3] = np.linspace(0.0, 1.5, d3)                 # slip abscissae from 0, increasing
    tabs[5][:d6] = np.linspace(-3.0, 3.0, d6)                # camber abscissae (deg), increasing
    px = np.linspace(150.0, 250.0, d5)
    py = t["kz"] * (1.0 + 0.01 * np.arange(d5))
    tabs[10] = np.concatenate([px, py])
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[2], sc[4], sc[5], sc[6] = t["toe"], t["camber"], t["castor"], t["radius"], PRESSURE, TYRE_BG
    sc[7], sc[8], sc[9], sc[11], sc[15] = t["drag"], 0.11 + 0.01 * i, 0.12 + 0.01 * i, 0.7 + 0.1 * i, 1.05 + 0.1 * i
    return 13, 80 + i, (list(dims), sc, tabs)


def shape(R, peak_scale=1.0):
    """Make the arbitrary record values physical for the QSS: positive peak grid (fy_peak = 1.2 x load x a
    temperature factor, times peak_scale), slip and camber factors 1, a positive CLA / CDA map (flat, the
    record scales it to its scalar) and a balance of 0.45. Only the values change, the tables keep their shapes."""
    out = []
    for t, cid, v in R:
        if t == 9 and cid == 127:                       # cla map: downforce coefficient (2.5 keeps the ride heights up at 70 m/s)
            x, z, d = v
            v = (x, z, np.full(len(d), 2.5))
        elif t == 9 and cid == 131:                     # cda map: flat, its scalar (cid 122) sets the drag
            x, z, d = v
            v = (x, z, np.full(len(d), 0.95))
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
            tabs[0][d3:] = np.linspace(0.0, 1.0, d3)    # normalised slip curve from (0, 0), peak 1
            tabs[5][d6:] = 1.0                          # camber factor 1
            v = (dims, sc, tabs)
        out.append((t, cid, v))
    return out


def build_records(cda_scalar=0.95, peak_scale=1.0, mr_fl=None, bias=None):
    R = []

    def add(tag, cid, v):
        cid = FILE_CID.get(cid, cid)
        R.append((tag, cid, v))
    for cid, v in EXPLICIT.items():
        add(1, cid, bias if (bias is not None and cid == 108) else cda_scalar if cid == 122 else v)
    for cid in (55, 59, 63, 67):                                       # roll-centre heights [m], middle values
        v = RC_HEIGHT
        add(1, cid, v)
    for cid in (204, 205, 206, 207):                                  # rubber gaps are ground gaps (flag 1)
        add(6, cid, 1)
    for cid in (53, 61, 69):
        add(6, cid, 0)                                   # geometry mode 0 (53, 61), no live axle (69)
    for cid, (rate, pre) in SPRING.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.05]), np.array([0.0, 500.0])))
    for cid, (rate, pre) in RUBBER.items():
        add(11, cid, (1, pre, rate, np.array([0.0, 0.02]), np.array([0.0, 800.0])))
    for cid, rate in ARB.items():
        add(11, cid, (1, 0.0, rate, np.array([0.0, 0.1]), np.array([0.0, 300.0])))
    for cid in range(23, 31):
        v = arb(5, cid)
        if mr_fl is not None and cid == 23:                              # front left motion ratio polynomial
            v = np.array(mr_fl, float)
        add(2, cid, v)                                                    # motion ratio coefficients
    for cid in range(366, 372):
        add(6, cid, 10 + cid - 366)
    add(7, 101, (ENGINE_RPM, ENGINE_NM))
    add(2, 110, GEARS)
    for cid, ((nx, nz), _) in MAPS.items():
        add(9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid)))
    for cid in range(31, 36):
        add(12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB])))
    for cid, hp in zip(HARD, (HP_LEFT, HP_RIGHT, HP_LEFT, HP_RIGHT)):
        add(10, cid, (0, hp.ravel(order="F")))      # the file holds the (6, 3) block column-major
    for i, c in enumerate(CORNERS):
        t, cid, v = tyre_record(i, c)
        add(t, cid, v)
    R = shape(R, peak_scale)
    R.append((carfile.END_TAG, carfile.END_CID, None))
    return R


def write_car(name, cda_scalar=0.95, peak_scale=1.0, mr_fl=None, bias=None):
    path = os.path.join(TMP.name, name)
    carfile.write(build_records(cda_scalar, peak_scale, mr_fl, bias), path)
    return path


def qss(car_path=None, **kw):
    """MLTP_screen on the synthetic install (car_path None: the 4EM default Sturn). Returns ctx."""
    if car_path is None:
        return quiet(MLTP_screen, circuit="Sturn", save=False, verbose=False, **kw)
    opts = dict(car=f"chassissim:{car_path}", circuit=f"chassissim:{CIRC}", chassissim_dir=INSTALL,
                physics="corrected", save=False, verbose=False)
    opts.update(kw)
    return quiet(MLTP_screen, **opts)


def hand_corner(cs_, vp_, c, x):
    cf = np.atleast_1d(getattr(cs_.susp, f"mr_{c}")).astype(float)
    k = getattr(cs_.susp, f"spring_{c}")["rate"]
    sm = sum(v / (i + 1) * x ** (i + 1) for i, v in enumerate(cf))
    mr = sum(v * x ** i for i, v in enumerate(cf))
    F0, mr0 = getattr(vp_, f"m_c_{c}") * vp_.g, cf[0]
    rub = getattr(cs_.susp, f"rubber_{c}")
    slope = (rub["y"][-1] - rub["y"][0]) / (rub["x"][-1] - rub["x"][0])
    gap = getattr(cs_.susp, f"gap_{c}")
    rubber = mr * slope * W_RUBBER * np.logaddexp(0.0, (sm - gap) / W_RUBBER)
    return (F0 / mr0 + k * sm) * mr - F0 + rubber


def heave_travel(cs_, vp_, pair, F_axle, xg=np.linspace(0.0, 0.15, 6001)):
    """Strut travel [m] of an axle whose spring + rubber load (hand_corner, the MR law) equals F_axle [N]."""
    F = sum(hand_corner(cs_, vp_, c, xg) for c in pair)
    return float(np.interp(F_axle, F, xg))


def ay_balance(ctx, cap, v0=2.0):
    """Independent lateral limit [m/s^2] of the car from its tables: yaw-balanced axles, the corner peaks of the
    fy_peak table (load PCHIP, camber 1, grip, the peak of the slip curve capped at cap), lateral load transfer
    from the springs, bars and tyres in series (roll centres from the car file). Static loads at v0 (aero ~0)."""
    cs, vp, geo = ctx.cs, ctx.vp, ctx.cs.geom
    ty = {c: getattr(cs.tyre, c) for c in CORNERS}
    grip = getattr(ctx, "grip", 1.0)
    pk = {}
    for c in CORNERS:
        T = ty[c]
        o = np.argsort(T["load"], kind="stable")
        x, y = np.asarray(T["load"], float)[o], np.asarray(T["fy_peak"], float)[o]
        keep = np.r_[True, np.diff(x) > 0]
        p = PchipInterpolator(x[keep], y[keep])
        k = grip * T["grip"] * min(float(np.max(T["slip_y"])), cap)
        pk[c] = (lambda p, k, x: lambda Fz: p(np.clip(Fz, x[0], x[-1])) * k)(p, k, x)
    kw = {c: getattr(cs.susp, f"spring_{c}")["rate"] * float(np.atleast_1d(getattr(cs.susp, f"mr_{c}"))[0]) ** 2
          for c in CORNERS}
    kz = {c: ty[c]["kz"] for c in CORNERS}
    mr_arb = {a: float(np.atleast_1d(getattr(cs.susp, f"mr_arb_{a}r"))[0]) for a in "fr"}   # the right record: 27 front, 28 rear
    t_f, t_r = geo.track_front, geo.track_rear
    # roll-centre height above the contact patches at the heave travel of each axle under the aero load at v0
    # (the QSS: the curves of suspension_curves at the travel of the MR law; the test solves that travel by hand)
    curves = suspension_curves(cs)
    Fa_f = float(ctx.envelope.Fz_front[0] - (vp.Wfl0 + vp.Wfr0))
    Fa_r = float(ctx.envelope.Fz_rear[0] - (vp.Wrl0 + vp.Wrr0))
    hv_f = heave_travel(cs, vp, ("fl", "fr"), Fa_f)
    hv_r = heave_travel(cs, vp, ("rl", "rr"), Fa_r)
    hrc_f = float(curves.f.rc_h(np.clip(hv_f, curves.f.lo, curves.f.hi)))
    hrc_r = float(curves.r.rc_h(np.clip(hv_r, curves.r.lo, curves.r.hi)))
    Kr = {}
    for a, (l, r), t in (("f", ("fl", "fr"), t_f), ("r", ("rl", "rr"), t_r)):
        bar = 0.5 * getattr(cs.susp, f"arb_{a}")["rate"] * mr_arb[a] ** 2
        k_s = (kw[l] + kw[r]) * t * t / 4.0 + bar * t * t
        k_t = (kz[l] + kz[r]) * t * t / 4.0
        Kr[a] = k_s * k_t / (k_s + k_t)
    phi_f = Kr["f"] / (Kr["f"] + Kr["r"])
    L = vp.l_f + vp.l_r
    M, h = float(vp.m), float(vp.hcg)
    sh_f, sh_r = vp.l_r / L, vp.l_f / L
    Fz_f, Fz_r = float(vp.Wfl0 + vp.Wfr0), float(vp.Wrl0 + vp.Wrr0)

    def axle(a, Fz, dF):
        l, r = ("fl", "fr") if a == "f" else ("rl", "rr")
        return pk[l](0.5 * Fz + dF) + pk[r](0.5 * Fz - dF)

    def C(ay):
        Fy_f, Fy_r = M * ay * sh_f, M * ay * sh_r
        Mx = Fy_f * (h - hrc_f) + Fy_r * (h - hrc_r)
        dF_f = np.clip((phi_f * Mx + Fy_f * hrc_f) / t_f, 0.0, 0.5 * Fz_f)
        dF_r = np.clip(((1 - phi_f) * Mx + Fy_r * hrc_r) / t_r, 0.0, 0.5 * Fz_r)
        return min(axle("f", Fz_f, dF_f) / (M * sh_f), axle("r", Fz_r, dF_r) / (M * sh_r))
    lo, hi = 0.0, 2.0 * C(0.0) + 1.0
    for _ in range(200):          # C(ay) - ay is decreasing: bisect its root
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if C(mid) > mid else (lo, mid)
    return 0.5 * (lo + hi)


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

# drive table: the 0.1 m/s grid of a car. Away from a gear crossing the numpy max over the gears of the car engine
# curve (PCHIP, own F_ref, independent of vehModel_cs) is the force, to 1e-4 F_ref. The log-sum-exp adds at most
# F_ref ln(n) / B_GEAR, which is a bound on the crossings only.
def gear_forces(cs, v):
    e = cs.engine
    Rr = cs.tyre.rl["radius"]
    x, y = np.asarray(e.torque_curve["x"], float), np.asarray(e.torque_curve["y"], float)
    te = PchipInterpolator(x, y)
    F_ref = float(e.efficiency * te(np.linspace(e.rpm_idle, e.rpm_limit, 400)).max() * np.max(e.gears) / Rr)
    dw = 0.01 * e.rpm_limit
    fg = []
    for Gr in np.asarray(e.gears, float):
        w = v * Gr / Rr
        win = 0.5 * (1.0 + np.tanh((e.rpm_limit - w) / dw))
        fg.append(e.efficiency * te(np.clip(np.maximum(w, e.rpm_idle), x.min(), x.max())) * Gr / Rr * win)
    top2 = np.sort(np.vstack(fg), axis=0)[-2:]
    return top2[1], top2[1] - top2[0], F_ref, len(fg)


dstep = np.diff(env.v_grid)
ok(f"car QSS grid is 0.1 m/s steps at every step (min {dstep.min():.5f}, max {dstep.max():.5f}, "
   f"{env.v_grid.size} nodes)", bool(np.allclose(dstep, 0.1, rtol=0.0, atol=1e-3)))
ok(f"F_drive_tab has one value per node (length {env.F_drive_tab.size})",
   env.F_drive_tab is not None and env.F_drive_tab.size == env.v_grid.size)
top, gap, F_ref, n_g = gear_forces(ctx.cs, env.v_grid)
ft = env.F_drive_tab
away = (gap > 0.1 * F_ref) & (env.v_grid >= 25.0) & (env.v_grid <= 265.0)
dev = np.abs(ft[away] - top[away])
ok(f"F_drive_tab equals the max over the gears to 1e-4 F_ref away from the gear crossings "
   f"({away.sum()} nodes, max dev {dev.max():.3f} N, F_ref {F_ref:.0f} N)",
   bool(away.sum() > 100 and dev.max() < 1e-4 * F_ref))
ok(f"F_drive_tab is positive over the grid (min {ft.min():.1f} N)", bool(np.all(ft > 0)))
ok("n_v does not apply to a car: build_envelope(ctx, n_v=5) keeps the 0.1 m/s grid",
   quiet(build_envelope, ctx, n_v=5).v_grid.size == env.v_grid.size)
quiet(build_envelope, ctx)                  # the first call per car samples the tyre functions (cached on the record)
t0 = time.perf_counter()
quiet(build_envelope, ctx)                  # the timed call is a repeat call
dt = time.perf_counter() - t0
ok(f"build_envelope_cs on the synthetic car takes less than 0.5 s ({1e3 * dt:.0f} ms)", dt < 0.5)

print("[cs_qss] envelope pieces")
# 0.9 lateral cap: the lateral peak of each corner is min(peak, LAT_CAP) x the table. The test recomputes the
# yaw-balanced lateral limit at 2 m/s from the car tables (ay_balance, cap read from the code) and compares it with
# ay_max, both with the cap and with the cap read as 1.0 (no cap, build_envelope_cs reads LAT_CAP at call time).
ay_cap_ref = ay_balance(ctx, VCS.LAT_CAP)
VCS.LAT_CAP = 1.0
try:
    env_free = quiet(build_envelope, ctx)
finally:
    VCS.LAT_CAP = 0.9
ay_free_ref = ay_balance(ctx, 1.0)
e_cap = abs(env.ay_max[0] - ay_cap_ref) / ay_cap_ref
e_free = abs(env_free.ay_max[0] - ay_free_ref) / ay_free_ref
ok(f"ay_max with the 0.9 cap equals the lateral limit from the car tables at {env.v_grid[0]:.1f} m/s "
   f"({env.ay_max[0] / G:.3f} g vs {ay_cap_ref / G:.3f} g, {100 * e_cap:.3f} %)", e_cap < 0.01)
ok(f"ay_max without the cap equals the same limit at cap 1.0 ({env_free.ay_max[0] / G:.3f} g vs "
   f"{ay_free_ref / G:.3f} g, {100 * e_free:.3f} %)", e_free < 0.01)
ok(f"the cap lowers the lateral limit ({env.ay_max[0] / G:.3f} g < {env_free.ay_max[0] / G:.3f} g)",
   bool(env.ay_max[0] < env_free.ay_max[0]))
ok("the cap leaves the longitudinal capacity alone (Fx_trac and Fx_brk unchanged)",
   bool(np.allclose(env_free.Fx_trac, env.Fx_trac, rtol=1e-12, atol=0.0)
        and np.allclose(env_free.Fx_brk, env.Fx_brk, rtol=1e-12, atol=0.0)))
# quasi-static ride heights with the ground-gap rubbers on: finite over the grid. At 70 m/s the value is documented,
# not gated: the model has no ground contact (ChassisSim clamps the aero map), so a car that bottoms out at 70 m/s
# can show a negative height (ruling round 2).
rh_f, rh_r = env.info["rh_f"], env.info["rh_r"]
ok("quasi-static ride heights finite over the grid (rubber at the damper)",
   bool(np.all(np.isfinite(rh_f)) and np.all(np.isfinite(rh_r))))
rf70, rr70 = np.interp(70.0, env.v_grid, rh_f), np.interp(70.0, env.v_grid, rh_r)
print(f"  [doc] quasi-static ride height at 70 m/s: front {1e3 * rf70:.2f} mm, rear {1e3 * rr70:.2f} mm "
      "(no ground contact in the model; a negative value is a bottomed-out car, not gated)")
ok("quasi-static ride heights fall monotonically with speed up to the top speed",
   bool(np.all(np.diff(rh_f) <= 1e-12) and np.all(np.diff(rh_r) <= 1e-12)))

F_brk = float(env.F_brk_trq)
ok(f"brake check: F_brk_trq equals cid 107 ({BRAKE_FORCE:.0f} N): {F_brk:.1f} N", abs(F_brk - BRAKE_FORCE) < 1e-6)
for bias in (0.0, 1.0):          # the brake bracket with one axle only (cid 108 = 0 or 1)
    cb = qss(write_car(f"bias{int(bias)}.car", bias=bias))
    fb = cb.envelope.Fx_brk
    ok(f"brake bias {bias:.0f}: Fx_brk finite and > 0 at every speed (min {fb.min():.1f} N)",
       bool(np.all(np.isfinite(fb)) and np.all(fb > 0)))
ok("Fx_brk (tyre) is finite and positive on the grid",
   bool(np.all(np.isfinite(env.Fx_brk)) and np.all(env.Fx_brk > 0)))
ok(f"ay_max positive on the grid (min {env.ay_max.min() / G:.2f} g)", bool(np.all(env.ay_max > 0)))
ok(f"ay_max grows with speed with the car downforce: {env.ay_max[0] / G:.2f} g at {env.v_grid[0]:.1f} m/s, "
   f"{env.ay_max[-1] / G:.2f} g at {env.v_grid[-1]:.1f} m/s", bool(env.F_lift[-1] > 0 and env.ay_max[-1] > env.ay_max[0]))
ok("aero downforce positive at the top speed (F_lift)", bool(env.F_lift[-1] > 0))
# downforce: the lift at each node is q x CLA of the car map at the quasi-static ride heights of that node (the
# vehModel_cs surface, the same function as the NLP)
rho = ctx.cs.aero.rho if ctx.cs.aero.rho else 1.225
q = 0.5 * rho * env.v_grid ** 2
cla = _surf(ctx.cs.aero.cla_map)
cla_at = np.array([float(cla(r, f)) for r, f in zip(env.info["rh_r"], env.info["rh_f"])])
lift_ref = q * cla_at
e_lift = np.max(np.abs(env.F_lift - lift_ref) / np.maximum(np.abs(lift_ref), 1.0))
ok(f"downforce: F_lift = q x CLA(rh_f, rh_r) at the fixed-point ride heights (max rel {e_lift:.1e})",
   bool(e_lift < 1e-6))

# motion ratio law (e): a car whose front left corner has c1 = 0.9, c2 = 0.5 (sm = c1 x + c2 x^2 / 2, mr = c1 + c2 x).
# The QSS axle force table is the sum over the corners of (F0/mr0 + k sm) mr - F0 + rubber, with the rubber
# mr x slope x W softplus((sm - gap) / W) (2-point table, linear). The table is read back at the travel that the
# fixed point implies: travel = rh0 - rh - F_axle / kt, and the hand force there must equal the axle load of the QSS.
MR_CAR = write_car("mr.car", mr_fl=[0.9, 0.5, 0.0, 0.0, 0.0])
ctx_mr = qss(MR_CAR)
env_mr, cs_m, vp_m, geo_m = ctx_mr.envelope, ctx_mr.cs, ctx_mr.vp, ctx_mr.cs.geom
kt = {"f": cs_m.tyre.fl["kz"] + cs_m.tyre.fr["kz"], "r": cs_m.tyre.rl["kz"] + cs_m.tyre.rr["kz"]}
rh0 = {"f": 0.5 * (geo_m.ride_height_fl + geo_m.ride_height_fr), "r": 0.5 * (geo_m.ride_height_rl + geo_m.ride_height_rr)}
Fa = {"f": env_mr.Fz_front - (vp_m.Wfl0 + vp_m.Wfr0), "r": env_mr.Fz_rear - (vp_m.Wrl0 + vp_m.Wrr0)}
rh = {"f": env_mr.info["rh_f"], "r": env_mr.info["rh_r"]}
pair = {"f": ("fl", "fr"), "r": ("rl", "rr")}
nodes = [int(np.argmin(np.abs(env_mr.v_grid - v))) for v in (30.0, 70.0)] + [env_mr.v_grid.size - 1]
for a in ("f", "r"):
    for i in nodes:
        tr = rh0[a] - rh[a][i] - Fa[a][i] / kt[a]
        hand = sum(hand_corner(cs_m, vp_m, c, tr) for c in pair[a])
        rel = abs(hand - Fa[a][i]) / abs(Fa[a][i])
        ok(f"MR law ({a} axle, {env_mr.v_grid[i]:.1f} m/s): QSS axle load {Fa[a][i]:.1f} N vs hand formula at travel "
           f"{1e3 * tr:.2f} mm {hand:.1f} N ({100 * rel:.3f} %)", bool(rel < 0.005))

# aero maps 50 mm below their grid (f): the clamp holds the edge value, not zero
for name, tab in (("cla", ctx.cs.aero.cla_map), ("cda", ctx.cs.aero.cda_map), ("balance", ctx.cs.aero.balance_map)):
    f_ = _surf(tab)
    xm, z0 = 0.5 * (tab["x"][0] + tab["x"][-1]), tab["z"][0]
    below, edge = float(f_(xm, z0 - 0.05)), float(f_(xm, z0))
    ok(f"{name} map 50 mm below the grid holds the edge value ({below:.4g} vs edge {edge:.4g})",
       bool(below != 0.0 and abs(below - edge) <= 1e-3 * abs(edge)))

# brake bias cars (a): one axle only
print("[cs_qss] lap")
lap = float(ctx.data.lap_time)
ok(f"car QSS lap finite and positive: {lap:.3f} s", bool(np.isfinite(lap) and lap > 0))

DBL = write_car("peak2.car", peak_scale=2.0)
lap_peak = float(qss(DBL).data.lap_time)
ok(f"doubling the tyre peak table lowers the QSS lap: {lap:.3f} s -> {lap_peak:.3f} s", lap_peak < lap)

CDA = write_car("cda15.car", cda_scalar=0.95 * 1.5)     # the CDA scalar (cid 122) x1.5: the map is normalised to it
lap_cda = float(qss(CDA).data.lap_time)
ok(f"raising CDA (drag map x1.5) raises the QSS lap: {lap:.3f} s -> {lap_cda:.3f} s", lap_cda > lap)

print("[cs_qss] 4EM default QSS lap (unchanged)")
sturn = float(qss().data.lap_time)
ok(f"4EM Sturn QSS lap equals 18.74393 s to 1e-5 s: {sturn:.6f} s", abs(sturn - 18.74393) < 1e-5)


# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs_dir):
    print("[cs_qss] vendor GP3 2013 Eastern Creek")
    car_path = Path(_bootstrap.vendor_car(cs_dir))       # the GP3 2013 base car, picked by content
    ok("GP3 2013 Eastern Creek car file found", car_path.is_file())
    c = quiet(MLTP_screen, car=f"chassissim:{car_path}", circuit="chassissim:Eastern Creek", chassissim_dir=cs_dir,
              physics="corrected", save=False, verbose=False)
    gl = float(c.data.lap_time)
    ok("GP3 envelope carries the drive table", c.envelope.F_drive_tab is not None)
    ok(f"GP3 QSS lap finite and within 15 % of 76.742 s: {gl:.3f} s "
       f"({100 * (gl / 76.742 - 1):+.1f} %)", bool(np.isfinite(gl)) and abs(gl / 76.742 - 1) < 0.15)
    # The three cars that build now (strut, live axle, mode-1 roll centres), picked by content from every shipped
    # car. Pins measured after the roll-centre fix.
    cars = {p: load_car(p) for p in list_cars(cs_dir)}

    def pick(pred, what):
        hits = [p for p, c in cars.items() if pred(c)]
        ok(f"exactly one shipped car is {what} ({len(hits)} found)", len(hits) == 1)
        return hits[0]

    def lap_at(path):
        return float(quiet(MLTP_screen, car=f"chassissim:{path}", circuit="chassissim:Eastern Creek",
                           chassissim_dir=cs_dir, physics="corrected", save=False, verbose=False).data.lap_time)

    def builds(path):
        try:
            lap_at(path)
            return True
        except ValueError:
            return False

    strut = pick(lambda c: c.hardpoints.front_l_type == 1, "a strut car (front type byte 1)")
    live = pick(lambda c: c.susp.live_axle == 1 and c.hardpoints.rear_r_type == 2 and c.hardpoints.rear_l_type == 2,
                "a live-axle car (cid 69 = 1, rear type byte 2)")
    mode1 = pick(lambda c: c.geom.mode_f == 1 and c.geom.mode_r == 1 and builds(c.path) and all(
                 np.unique(t[k]).size >= 2 for t in (c.tyre.fl, c.tyre.fr, c.tyre.rl, c.tyre.rr)
                 for k in ("camber_x", "fx_camber_x")),
                 "a mode-1 car whose QSS builds with two or more camber points on every tyre table")
    for label, path, pin, why in (
            ("strut car", strut, 106.19322212836927, "builds now: MacPherson struts (type 1) on both axles"),
            ("live-axle car", live, 86.90939692648801, "builds now: rear live axle (type 2, cid 69 = 1)"),
            ("mode-1 car", mode1, 82.81056085360748, "builds now: mode-1 roll centres on both axles")):
        lap = lap_at(path)
        ok(f"{label} flying-lap QSS lap (march starts at the end speed) at Eastern Creek {lap!r} s, pin {pin}, "
           f"tol 1e-6 ({why})", abs(lap - pin) < 1e-6)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(CS)
print(f"{PASS} checks passed")
