"""ChassisSim car record of functions/chassissim.py (ChassisSim phase 2, task 1).

The synthetic checks build a car file with carfile.encode from arbitrary values (every cid of CAR_FIELDS, with the
record tag that its cid needs), load it with load_car and check the SI conversions, the None rule, the tyre
temperature column, the hardpoint and aero shapes and the car_vp totals through vehParams. The vendor section runs
only when CHASSISSIM_DIR names the ChassisSim install folder. It reads the GP3 2013 Eastern Creek car file and checks
relations only, no copied values. It writes nothing into the vendor folders.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_car.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_cs_car.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from functions import carfile
from functions.chassissim import CAR_FIELDS, car_vp, load_car
from functions.context import Ctx
from Powertrain import Powertrain
from vehParams import vehParams

PASS = 0
RPM_T = 2 * np.pi / 60    # rpm -> rad/s, written here on purpose (not imported from chassissim)
DEG_T = np.pi / 180
LBF_IN = 175.126835      # N/m per lbf/in
PHYSICS = ("matlab", "corrected")
TMP = tempfile.TemporaryDirectory()


def ok(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASS += 1


def arb(n, s):
    """n arbitrary finite numbers, different for each seed s."""
    return np.sin(np.arange(n) * 0.9 + s) * 7.3 - 0.61


# ---- synthetic car ---------------------------------------------------------------------------------
CORNERS = ("fl", "fr", "rl", "rr")
RADIUS = {"fl": 0.300, "fr": 0.305, "rl": 0.310, "rr": 0.315}   # m, tyre scalar 4
TOE = {"fl": 0.5, "fr": -0.25, "rl": 0.1, "rr": 0.0}           # deg, tyre scalar 0
CAMBER = {"fl": -1.0, "fr": -1.5, "rl": -0.5, "rr": -0.75}     # deg, tyre scalar 1
TYRE_DIMS = {  # distinct per corner; d2 = 2 temperatures in every tyre
    "fl": (3, 2, 2, 2, 2, 2, 2),
    "fr": (2, 2, 3, 2, 2, 3, 2),
    "rl": (4, 2, 2, 3, 2, 2, 2),
    "rr": (2, 2, 2, 2, 3, 2, 3),
}
TYRE_TEMPS = (300.0, 350.0)     # the two temperature columns of the load grid, K
TYRE_BG = 325.0                 # tyre background temperature (scalar 6), between the two columns
TRACK_TEMP = 350.0              # cid 390, equal to the second column: the wrong column would show it
VEL = np.array([-400.0, -100.0, 0.0, 100.0, 400.0])   # damper velocity, mm/s
BUMP = np.array([0.0, 900.0, 1800.0, 2700.0, 3600.0])
REB = np.array([-3600.0, -2400.0, -1200.0, 0.0, 0.0])
SPRING = {1: (1.23e5, 200.0), 2: (1.31e5, 210.0), 3: (0.97e5, 190.0), 4: (1.02e5, 205.0),
          5: (9.0e4, 0.0), 6: (8.0e4, 0.0)}             # cid -> (rate N/m, preload N)
DAMPER = (31, 32, 33, 34, 35)
HARD = (49, 50, 51, 52)
TYRE_CID = {80: "fl", 81: "fr", 82: "rl", 83: "rr"}
AERO_MAP = {127: ("cla", (4, 3)), 131: ("cda", (3, 2)), 135: ("balance", (5, 2))}   # cid -> (name, (nx, nz))
ENGINE_RPM = np.array([1000.0, 3000.0, 6000.0, 9000.0])
ENGINE_NM = np.array([200.0, 250.0, 240.0, 150.0])
GEARS = np.array([3.2, 2.5, 1.9, 1.5, 1.25, 1.05])
# explicit tag-1 values (mass kg, lengths m, ride heights mm, rpm, brakes, aero); all others 0.2 + 0.011 cid
EXPLICIT = {
    84: 700.0, 85: 25.0, 86: 35.0, 87: 60.0, 88: 280.0, 89: 600.0, 96: 0.44,
    90: 3.1, 91: 1.52, 92: 1.49, 93: 0.33,
    117: 17.0, 118: 18.5, 119: 20.0, 120: 21.5,
    102: 1000.0, 103: 9000.0,
    107: 14000.0, 108: 0.6, 109: 0.05,
    100: 0.9, 111: 0.3,
    121: 3.5, 122: 0.95, 123: 0.1, 125: 1.2,
    390: TRACK_TEMP,
}


def tyre_sizes(d):
    """f64 count of the 13 tables of a tyre (carfile docstring: 2D(k) = 2k, 3D(a, b) = a + b + a b)."""
    d1, d2, d3, d4, d5, d6, d7 = d

    def g(a, b):
        return a + b + a * b
    return [2 * d3, 2 * d4, 2 * d4, g(d1, d2), g(d6, d7), 2 * d6, 2 * d4, g(d1, d2), 2 * d3, 2 * d6, 2 * d5, 2 * d5, 2 * d5]


def tyre_record(i, corner):
    """Tyre record (tag 13) with a load grid of two temperature columns. Return (record, A, B) with A, B per load."""
    dims = TYRE_DIMS[corner]
    d1 = dims[0]
    tabs = [arb(k, 10 * i + j) for j, k in enumerate(tyre_sizes(dims))]
    A, B = arb(d1, 20 + i), arb(d1, 30 + i)
    load = 1000.0 * np.arange(1, d1 + 1)
    tabs[3] = np.concatenate([load, TYRE_TEMPS, np.stack([A, B], axis=1).ravel()])   # load, temperatures, grid (row = load)
    sc = arb(18, 40 + i)
    sc[0], sc[1], sc[4], sc[5], sc[6] = TOE[corner], CAMBER[corner], RADIUS[corner], 2.1e5, TYRE_BG
    return (13, 80 + i, (list(dims), sc, tabs)), A, B


def synthetic_records(omit=()):
    """One record of every CAR_FIELDS cid (except omit), with the tag of its cid. Return (records, tyre A/B by corner)."""
    recs, ab = [], {}
    for cid in CAR_FIELDS:
        if cid in omit:
            continue
        if cid in SPRING_CIDS:
            rate, pre = SPRING[cid]
            recs.append((11, cid, (1, pre, rate, np.array([0.0, 0.05]), np.array([0.0, 500.0]))))
        elif cid in DAMPER:
            recs.append((12, cid, (1, arb(6, cid), np.concatenate([VEL, BUMP, REB]))))
        elif cid in HARD:
            recs.append((10, cid, (0, arb(18, cid))))
        elif cid in TYRE_CID:
            rec, A, B = tyre_record(cid - 80, TYRE_CID[cid])
            recs.append(rec)
            ab[TYRE_CID[cid]] = (A, B)
        elif cid in AERO_MAP:
            _, (nx, nz) = AERO_MAP[cid]
            recs.append((9, cid, (0.5 * np.arange(nx), 1.0 * np.arange(nz) + 0.1, arb(nx * nz, cid))))
        elif cid == 101:
            recs.append((7, cid, (ENGINE_RPM, ENGINE_NM)))
        elif cid == 110:
            recs.append((2, cid, GEARS))
        else:
            recs.append((1, cid, EXPLICIT.get(cid, 0.2 + 0.011 * cid)))
    recs.append((carfile.END_TAG, carfile.END_CID, None))
    return recs, ab


SPRING_CIDS = tuple(SPRING)


def write_car(name, omit=()):
    recs, ab = synthetic_records(omit)
    path = os.path.join(TMP.name, name)
    carfile.write(recs, path)
    return path, ab


def through_vehparams(vp_overrides, physics):
    ctx = Ctx()
    ctx.physics = physics
    Powertrain(ctx)
    vehParams(ctx, vp_overrides=vp_overrides)
    return ctx.vp


print("[cs_car] synthetic car")
SYN, SYN_AB = write_car("synthetic.car")
car = load_car(SYN)

# SI conversions
print("  -- SI conversions")
ok("ride heights stored in mm read as m (17, 18.5, 20, 21.5 mm)",
   all(abs(getattr(car.geom, f"ride_height_{c}") - EXPLICIT[117 + i] * 1e-3) < 1e-12
       for i, c in enumerate(CORNERS)))
ok("damper velocity stored in mm/s read as m/s (fl, fr, rl, rr, third)",
   all(np.allclose(getattr(car.susp, f"damper_{c}")["velocity"], VEL * 1e-3, rtol=0, atol=1e-12)
       for c in CORNERS) and np.allclose(car.susp.damper_third_f["velocity"], VEL * 1e-3, rtol=0, atol=1e-12))
ok("damper bump and rebound forces stay in N",
   np.allclose(car.susp.damper_fl["bump"], BUMP) and np.allclose(car.susp.damper_fl["rebound"], REB))
ok("engine rpm axis read as rad/s",
   np.allclose(car.engine.torque_curve["x"], ENGINE_RPM * RPM_T, rtol=0, atol=1e-12))
ok("engine torque read in N m, unchanged",
   np.allclose(car.engine.torque_curve["y"], ENGINE_NM, rtol=0, atol=0))
ok("idle and limit rpm read as rad/s",
   abs(car.engine.rpm_idle - 1000.0 * RPM_T) < 1e-12 and abs(car.engine.rpm_limit - 9000.0 * RPM_T) < 1e-12)
ok("tyre toe and camber stored in deg read in rad (fl)",
   abs(car.tyre.fl["toe"] - TOE["fl"] * DEG_T) < 1e-12 and abs(car.tyre.fl["camber"] - CAMBER["fl"] * DEG_T) < 1e-12)
ok("tyre radius read in m as stored (rl)", abs(car.tyre.rl["radius"] - RADIUS["rl"]) < 1e-12)
ok("spring rate and preload read as stored (N/m, N)",
   all(abs(getattr(car.susp, f"spring_{c}")["rate"] - SPRING[i + 1][0]) < 1e-6
       and abs(getattr(car.susp, f"spring_{c}")["preload"] - SPRING[i + 1][1]) < 1e-9
       for i, c in enumerate(CORNERS)))
ok("gear ratios read as stored (6 gears)", np.allclose(car.engine.gears, GEARS, rtol=0, atol=0) and len(car.engine.gears) == 6)

# tyre load curve at the background temperature (scalar 6)
print("  -- tyre temperature column")
A, B = SYN_AB["rl"]
ok("tyre background temperature is scalar 6 (325 K)", abs(car.tyre.rl["temp"] - TYRE_BG) < 1e-12)
ok("rl load axis read as stored, one fy_peak per load", np.allclose(car.tyre.rl["load"], 1000.0 * np.arange(1, 5))
   and car.tyre.rl["fy_peak"].shape == (4,))
ok("fy_peak interpolates the 300 K and 350 K columns at scalar 6 (325 K), not the track temperature (cid 390 = 350 K)",
   np.allclose(car.tyre.rl["fy_peak"], 0.5 * (A + B), rtol=0, atol=1e-12)
   and not np.allclose(car.tyre.rl["fy_peak"], B, rtol=0, atol=1e-6))

# shapes
print("  -- shapes")
ok("hardpoint block has shape (6, 3) (front_l)", car.hardpoints.front_l.shape == (6, 3))
ok("aero maps have shape (nx, nz): cla (4, 3), cda (3, 2), balance (5, 2)",
   car.aero.cla_map["data"].shape == (4, 3) and car.aero.cda_map["data"].shape == (3, 2)
   and car.aero.balance_map["data"].shape == (5, 2))

# file identity
print("  -- file identity")
ok("sha8 is the first 8 hex digits of the sha256 of the file bytes",
   car.sha8 == hashlib.sha256(Path(SYN).read_bytes()).hexdigest()[:8])
ok("path is absolute and equals the path given", os.path.isabs(car.path) and car.path == os.path.abspath(SYN))
here = os.getcwd()
os.chdir(TMP.name)
try:
    rel = load_car("synthetic.car")
finally:
    os.chdir(here)
ok("a relative path loads and stores the absolute path", os.path.isabs(rel.path) and os.path.samefile(rel.path, SYN))

# None rule
print("  -- missing cids")
no_cda, _ = write_car("no_cda.car", omit=(122, 390))
cn = load_car(no_cda)
ok("a cid the file lacks gives None (aero.cda, env.track_temp)", cn.aero.cda is None and cn.env.track_temp is None)
ok("the other fields of that car are still read", abs(cn.aero.cla - 3.5) < 1e-12)

# car_vp
print("  -- car_vp")
vp = car_vp(car)
ok("car_vp mb + md + muf + mur equals the car mass (700 kg)",
   abs(vp["mb"] + vp["md"] + vp["muf"] + vp["mur"] - 700.0) < 1e-9 and vp["md"] == 0.0
   and vp["muf"] == 25.0 and vp["mur"] == 35.0)
ok("car_vp track is the mean of the two tracks (1.505 m)", abs(vp["t"] - (1.52 + 1.49) / 2) < 1e-12)
ok("car_vp Rw is the rear left tyre radius (0.310 m)", abs(vp["Rw"] - RADIUS["rl"]) < 1e-12)
ok("car_vp wheelbase, CG height and inertias read as stored",
   abs(vp["l"] - 3.1) < 1e-12 and abs(vp["hcg"] - 0.33) < 1e-12
   and abs(vp["I_x"] - 60.0) < 1e-12 and abs(vp["I_y"] - 280.0) < 1e-12 and abs(vp["I_z"] - 600.0) < 1e-12)
ok("car_vp front share key is present in (0, 1)", 0.0 < vp["wB"] < 1.0)

no_mass, _ = write_car("no_mass.car", omit=(84, 85, 86))
vp_nm = car_vp(load_car(no_mass))
ok("a car without the mass cids gives no mass keys (mb, md, muf, mur)",
   not any(k in vp_nm for k in ("mb", "md", "muf", "mur")) and "l" in vp_nm)
no_share, _ = write_car("no_share.car", omit=(96,))
ok("a car without the front share cid gives no wB key", "wB" not in car_vp(load_car(no_share)))

# through vehParams: total mass and front share
print("  -- car_vp through vehParams")
for physics in PHYSICS:
    v = through_vehparams(vp, physics)
    ok(f"vehParams total mass equals the car mass ({physics}): m = {v.m:.9g} kg",
       abs(v.m - 700.0) < 1e-9 and abs(v.ms - (700.0 - 25.0 - 35.0)) < 1e-9)
for physics in PHYSICS:
    v = through_vehparams(vp, physics)
    share = (v.Wfl0 + v.Wfr0) / (v.g * v.m)
    ok(f"static front axle share through vehParams equals the car front share 0.44 ({physics}): {share:.6f}",
       abs(share - car.mass.front_share) < 1e-6)


# ---- vendor -----------------------------------------------------------------------------------------
def vendor(cs):
    car_path = cs.parent / "Models" / "GP3 2013" / "Eastern Creek, AU" / "def_GP3 2013.car"
    print("[cs_car] vendor GP3 2013 Eastern Creek")
    ok("GP3 2013 Eastern Creek car file found", car_path.is_file())
    c = load_car(str(car_path))

    missing = [name for name, _ in CAR_FIELDS.values()
               if getattr(getattr(c, name.split(".")[0]), name.split(".")[1]) is None]
    ok(f"no consumed field of the GP3 car is None (missing: {missing})", not missing)

    def rel(a, b):
        return abs(a - b) / abs(b)
    ok("spring rates: fl and fr 1200 lbf/in, rl and rr 800 lbf/in, in N/m (1e-3 relative)",
       all(rel(getattr(c.susp, f"spring_{k}")["rate"], lbf * LBF_IN) < 1e-3
           for k, lbf in (("fl", 1200.0), ("fr", 1200.0), ("rl", 800.0), ("rr", 800.0))))
    ok(f"6 gears ({len(c.engine.gears)} read)", len(c.engine.gears) == 6)
    ok("aero maps are 7 x 5 (cla, cda, balance)",
       all(getattr(c.aero, m)["data"].shape == (7, 5) for m in ("cla_map", "cda_map", "balance_map")))
    ok("damper velocities are in m/s, below 1 m/s",
       all(float(np.max(np.abs(getattr(c.susp, f"damper_{k}")["velocity"]))) < 1.0 for k in CORNERS))

    vpc = car_vp(c)
    for physics in PHYSICS:
        v = through_vehparams(vpc, physics)
        ok(f"car_vp through vehParams reproduces the car mass ({physics}): m = {v.m:.6g} kg",
           abs(v.m - c.mass.total) <= 1e-9 * c.mass.total)
    for physics in PHYSICS:
        v = through_vehparams(vpc, physics)
        share = (v.Wfl0 + v.Wfr0) / (v.g * v.m)
        ok(f"static front axle share through vehParams equals the car front share ({physics}): "
           f"{share:.6f} vs {c.mass.front_share:.6f}",
           abs(share - c.mass.front_share) < 1e-6)


CS = os.environ.get("CHASSISSIM_DIR")
if not CS or not os.path.isdir(CS):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(Path(CS))
print(f"{PASS} checks passed")
