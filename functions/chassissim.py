"""ChassisSim v3.41 input files: readers and the circuit source.

The install folder comes from the argument or from the environment variable CHASSISSIM_DIR.
A circuit name for userOpts has the form "chassissim:<Circuit name>".
The readers use only the standard library and numpy.
"""
import configparser
import hashlib
import os
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq

from functions import carfile

PREFIX = "chassissim:"
ENC = "latin-1"


def install_dir(chassissim_dir=None):
    """Return the install folder: the argument first, then CHASSISSIM_DIR."""
    p = chassissim_dir or os.environ.get("CHASSISSIM_DIR")
    if not p:
        raise ValueError("Set CHASSISSIM_DIR to the ChassisSim install folder, or pass chassissim_dir.")
    if not os.path.isdir(p):
        raise ValueError(f"The folder '{p}' does not exist.")
    return p


def _rows(path, ncol=None):
    """Read the non-blank lines as lists of float. ncol checks the column count."""
    out = []
    with open(path, encoding=ENC) as f:
        for ln in f:
            t = ln.split()
            if not t:
                continue
            if ncol is not None and len(t) != ncol:
                raise ValueError(f"{path}: a row has {len(t)} columns. The file needs {ncol}.")
            out.append([float(x) for x in t])
    return out


def _increasing(path, s):
    if len(s) < 3 or not np.all(np.diff(s) > 0):
        raise ValueError(f"{path}: the file needs 3 or more rows, and the distance must increase from row to row.")


def read_curvature(path):
    """Read a curvature file (columns: curvature 1/m, distance m). Return (s, k) as stored (right turn positive)."""
    a = np.array(_rows(path, 2), dtype=float).reshape(-1, 2)
    _increasing(path, a[:, 1])
    return a[:, 1], a[:, 0]


def read_circ_prop(path):
    """Read Circ_prop.ini. Return one dict per [CircuitN] block (keys as written, values str)."""
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    with open(path, encoding=ENC) as f:
        cp.read_file(f)
    return [dict(cp[sec]) for sec in cp.sections() if sec.startswith("Circuit") and sec[7:].isdigit()]


def read_alt_camber(path):
    """Read an alt_rcamb file. Return (base_alt m, s m, pitch rad, camber rad)."""
    rows = _rows(path)
    if not rows or len(rows[0]) != 1 or any(len(r) != 3 for r in rows[1:]):
        raise ValueError(f"{path}: line 1 must hold the base altitude only. The other rows need 3 columns.")
    base = rows[0][0]
    a = np.array(rows[1:], dtype=float).reshape(-1, 3)
    _increasing(path, a[:, 0])
    return base, a[:, 0], a[:, 1], a[:, 2]


def read_bump(path):
    """Read a bump profile (columns: FL FR RL RR road height mm, distance m). Return (s m, h m with shape (n, 4))."""
    a = np.array(_rows(path, 5), dtype=float).reshape(-1, 5)
    _increasing(path, a[:, 4])
    return a[:, 4], a[:, :4] * 1e-3


def _find(folder, name):
    """Return the path of name in folder, ignoring case, or None."""
    if not name or not os.path.isdir(folder):
        return None
    want = name.lower()
    for f in os.listdir(folder):
        if f.lower() == want:
            return os.path.join(folder, f)
    return None


def load_circuit(name, cs_dir):
    """Load a circuit by its ChassisSim name. Return SimpleNamespace(s, k, grip, bump, alt, curvature).

    k uses the model sign (left turn positive), so it is the file value with the sign flipped.
    bump and alt are file paths, or None when the circuit has no such file or the flag is off.
    """
    circ = os.path.join(os.path.abspath(cs_dir), "Circuits")
    ini = os.path.join(cs_dir, "Circ_prop.ini")
    blocks = read_circ_prop(ini) if os.path.isfile(ini) else []
    name = name.strip()
    blk = next((b for b in blocks if b.get("Circuit name", "").strip().lower() == name.lower()), None)
    if blk is not None:
        tr = _find(circ, blk.get("Circuit file name", ""))
        if tr is None:
            raise FileNotFoundError(f"The curvature file '{blk.get('Circuit file name')}' of circuit '{name}' is not in '{circ}'.")

        def flagged(key, flag):
            f = _find(circ, blk.get(key, ""))
            return f if blk.get(flag, "0").strip() == "1" and f and blk.get(key) != "default.dat" else None

        grip = float(blk.get("Base Grip factor", 1.0))
        bump, alt = flagged("Circuit bp name", "Circuit bp flag"), flagged("Circuit alt rcamb name", "Circuit alt rcamb flag")
    else:
        # A circuit without an ini block: its curvature file is tr_<name>.txt and no block lists that file.
        listed = {b.get("Circuit file name", "").strip().lower() for b in blocks}
        files = os.listdir(circ) if os.path.isdir(circ) else []
        orphans = {f[3:-4]: f for f in files
                   if f.lower().startswith("tr_") and f.lower().endswith(".txt") and f.lower() not in listed}
        hit = next((f for o, f in orphans.items() if o.lower() == name.lower()), None)
        if hit is None:
            known = sorted({b.get("Circuit name", "").strip() for b in blocks} | set(orphans))
            raise ValueError(f"ChassisSim has no circuit '{name}'. The known circuits are: {', '.join(known)}.")
        tr, grip = os.path.join(circ, hit), 1.0
        bump, alt = _find(circ, f"bp_{name}.dat"), _find(circ, f"alt_rcamb_{name}.txt")
    s, k = read_curvature(tr)
    return SimpleNamespace(s=s, k=-k, grip=grip, bump=bump, alt=alt, curvature=os.path.abspath(tr))


# Car record: the consumed fields of a ChassisSim car file, in SI.
RPM = 2 * np.pi / 60
DEG = np.pi / 180
MM = 1e-3
_C4 = ("fl", "fr", "rl", "rr")


def _corners(group, base, prefix=""):
    """Return the CAR_FIELDS entries of four corner records. base is the cid of the front left one."""
    return {base + i: (f"{group}.{prefix}{c}", 1.0) for i, c in enumerate(_C4)}


# cid -> (group.name, factor to SI). A tuple factor has one entry per part of the decoded record:
#   tag 7 (x, y)   tag 9 (x, z, data)   tag 11 (preload, rate, x, y)   tag 12 (unused, force)
# The file stores springs, bars and dampers in SI, so no field needs lbf/in. It stores a ride height in mm
# and a damper velocity in mm/s.
# ponytail: skipped the cids that the sheet leaves unnamed (54, 56, 58, 60, 62, 64, 66, 68, 94, 95, 97, 126, 150-153).
# Add one when a task needs it.
CAR_FIELDS = {
    84: ("mass.total", 1.0), 85: ("mass.unsprung_front", 1.0), 86: ("mass.unsprung_rear", 1.0),
    87: ("mass.Ix", 1.0), 88: ("mass.Iy", 1.0), 89: ("mass.Iz", 1.0), 96: ("mass.front_share", 1.0),
    90: ("geom.wheelbase", 1.0), 91: ("geom.track_front", 1.0), 92: ("geom.track_rear", 1.0),
    93: ("geom.cg_height", 1.0),
    # The middle value of each stored geometry triple (cids 54-56, 58-60, 62-64, 66-68) is the roll-centre height.
    55: ("geom.rc_height_fl", 1.0), 59: ("geom.rc_height_fr", 1.0),
    63: ("geom.rc_height_rl", 1.0), 67: ("geom.rc_height_rr", 1.0),
    **{117 + i: (f"geom.ride_height_{c}", MM) for i, c in enumerate(_C4)},
    **_corners("susp", 1, "spring_"), 5: ("susp.spring_third_f", 1.0), 6: ("susp.spring_third_r", 1.0),
    **{7 + i: (f"susp.gap_{n}", 1.0) for i, n in enumerate(("fl", "fr", "rl", "rr", "third_f", "third_r"))},
    **{204 + i: (f"susp.gap_flag_{c}", 1.0) for i, c in enumerate(_C4)},     # 1 = the gap is measured on the ground
    **{74 + i: (f"susp.bump_steer_{c}", 1.0) for i, c in enumerate(_C4)},    # tag 7: wheel travel [m], toe change
    **_corners("susp", 13, "rubber_"), 17: ("susp.rubber_third_f", 1.0), 18: ("susp.rubber_third_r", 1.0),
    19: ("susp.arb_f", 1.0), 21: ("susp.arb_r", 1.0),
    **_corners("susp", 23, "mr_"), 27: ("susp.mr_arb_f", 1.0), 28: ("susp.mr_arb_r", 1.0),
    29: ("susp.mr_third_f", 1.0), 30: ("susp.mr_third_r", 1.0),
    **_corners("susp", 31, "damper_"), 35: ("susp.damper_third_f", 1.0),
    **{49 + i: (f"hardpoints.{n}", 1.0) for i, n in enumerate(("front_l", "front_r", "rear_l", "rear_r"))},
    **_corners("tyre", 80),
    121: ("aero.cla", 1.0), 122: ("aero.cda", 1.0), 123: ("aero.offset", 1.0), 125: ("aero.rho", 1.0),
    127: ("aero.cla_map", 1.0), 131: ("aero.cda_map", 1.0), 135: ("aero.balance_map", 1.0),
    **{c: (f"aero.drs_{c}", 1.0) for c in range(336, 342)},
    **{c: (f"aero.multi_drs_{c}", 1.0) for c in range(366, 372)},
    # Corrections of the aero maps (tag 7 tables). Sideslip x is in rad, speed x in m/s. Enables are 0 or 1.
    128: ("aero.ss_cla", 1.0), 132: ("aero.ss_cda", 1.0), 136: ("aero.ss_ab", 1.0),
    **{413 + i: (f"aero.sp_on_{n}", 1.0) for i, n in enumerate(("cla", "cda", "ab"))},
    **{416 + i: (f"aero.sp_{n}", 1.0) for i, n in enumerate(("cla", "cda", "ab"))},
    # Features that the car model refuses when they are live: steer correction, roll split, 2-D maps, wake maps.
    **{c: (f"aero.refused_{c}", 1.0) for c in (139, *range(434, 440), *range(452, 464), 870, *range(872, 876))},
    100: ("engine.efficiency", 1.0), 101: ("engine.torque_curve", (RPM, 1.0)),
    102: ("engine.rpm_idle", RPM), 103: ("engine.rpm_limit", RPM), 110: ("engine.gears", 1.0),
    111: ("engine.torque_split", 1.0),
    112: ("diff.front_type", 1.0), 113: ("diff.rear_type", 1.0), 114: ("diff.front_map", 1.0),
    116: ("diff.rear_map", 1.0),
    107: ("brake.force", 1.0), 108: ("brake.front_bias", 1.0), 109: ("brake.delay", 1.0),
    390: ("env.track_temp", 1.0),
}


def _f(f, i):
    """Return entry i of a factor. A plain number is the factor of entry 0 only."""
    return f[i] if isinstance(f, tuple) else (f if i == 0 else 1.0)


def _tyre_decode(v):
    """Decode a tyre record (tag 13) into a dict in SI. The load table is read at the background temperature of
    the tyre (scalar 6, K), the temperature of a tyre that is up to temperature."""
    dims, sc, tb = v
    sc = np.asarray(sc, float)
    tb = [np.asarray(t, float) for t in tb]
    d1, d2, d3, _, d5, d6, d7 = (int(d) for d in dims)
    t4, d4, n4 = tb[4], d6, d7      # table 4: d6 cambers [deg], d7 loads [N], then the grid (row = camber)
    if len(t4) != d4 + n4 + d4 * n4:
        raise ValueError("The longitudinal camber table (table 4) of the tyre has a size that does not fit its dimensions.")
    z = tb[3][d1:d1 + d2]
    grid = tb[3][d1 + d2:].reshape(d1, d2)
    t = sc[6]
    return {
        "toe": sc[0] * DEG, "camber": sc[1] * DEG, "castor": sc[2] * DEG, "radius": sc[4], "pressure": sc[5],
        "temp": sc[6], "rolling_drag": sc[7], "optimum_camber": sc[11] * DEG, "peak_slip_angle": sc[8], "peak_slip_ratio": sc[9], "grip": sc[15],
        "load": tb[3][:d1], "fy_peak": np.array([np.interp(t, z, r) for r in grid]),
        "slip_x": tb[0][:d3], "slip_y": tb[0][d3:],
        "fx_camber_x": t4[:d4] * DEG, "fx_camber_load": t4[d4:d4 + n4], "fx_camber": t4[d4 + n4:].reshape(d4, n4),
        "camber_x": tb[5][:d6] * DEG, "camber_y": tb[5][d6:],
        "kz": float(np.interp(sc[5] * 1e-3, tb[10][:d5], tb[10][d5:])),
    }


def _conv(tag, v, f):
    """Convert one decoded record to SI. f is the factor entry of CAR_FIELDS."""
    if tag == 1:
        return float(v) * _f(f, 0)
    if tag == 2:
        return np.asarray(v, float) * _f(f, 0)
    if tag == 6:
        return int(v)
    if tag == 7:
        return {"x": np.asarray(v[0], float) * _f(f, 0), "y": np.asarray(v[1], float) * _f(f, 1)}
    if tag == 9:
        x, z, d = v
        return {"x": np.asarray(x, float) * _f(f, 0), "z": np.asarray(z, float) * _f(f, 1),
                "data": np.asarray(d, float).reshape(len(x), len(z)) * _f(f, 2)}
    if tag == 10:
        return np.asarray(v[1], float).reshape(3, -1).T * _f(f, 0)
    if tag == 11:
        return {"flag": int(v[0]), "preload": v[1] * _f(f, 0), "rate": v[2] * _f(f, 1),
                "x": np.asarray(v[3], float) * _f(f, 2), "y": np.asarray(v[4], float) * _f(f, 3)}
    if tag == 12:
        a = np.asarray(v[2], float).reshape(3, -1)
        return {"flag": int(v[0]), "scalars": np.asarray(v[1], float), "velocity": a[0] * MM,
                "bump": a[1] * _f(f, 1), "rebound": a[2] * _f(f, 1)}
    return v


def load_car(path):
    """Read a ChassisSim car file and return its consumed fields in SI.

    The result has the groups mass, geom, tyre, aero, susp, engine, diff, brake, hardpoints and env.
    Each group is a SimpleNamespace. A field that the file lacks is None. The result also holds
    path (absolute) and sha8 (the first 8 hex digits of the sha256 of the file bytes).
    Angles are in rad and lengths in m. The engine axis is in rad/s and the torque is in N m.
    A damper holds the velocity in m/s and the bump and rebound forces in N, both positive.
    Its table is active when it has points. A spring holds its preload in m and its rate in N/m.
    Toe is positive for toe-out. Camber is the negative camber: positive when the top of the wheel leans in.
    Brake force is the total in N, bias is the front share, delay is in s.
    A tyre is a dict. It holds the peak lateral force over the load at its background temperature,
    the slip and camber curves, the peak slip angle and ratio, the radius and the vertical rate.
    A hardpoint block has shape (6, 3), in the car frame of ChassisSim. The rows are lower inner front,
    lower inner aft, lower outer, upper inner front, upper inner aft and upper outer.
    """
    path = os.path.abspath(path)
    with open(path, "rb") as fh:
        sha8 = hashlib.sha256(fh.read()).hexdigest()[:8]
    recs = {c: (t, v) for t, c, v in carfile.decode(path)}
    groups = {}
    for name, _ in CAR_FIELDS.values():
        g, a = name.split(".")
        setattr(groups.setdefault(g, SimpleNamespace()), a, None)
    for cid, (name, f) in CAR_FIELDS.items():
        if cid in recs:
            g, a = name.split(".")
            t, v = recs[cid]
            setattr(groups[g], a, _tyre_decode(v) if t == 13 else _conv(t, v, f))
    a = groups["aero"]
    for mp, sc in (("cla_map", "cla"), ("cda_map", "cda")):
        # ChassisSim scales the CLA and CDA maps to a peak of 1 and multiplies them by the scalar.
        if getattr(a, mp) is None:
            continue
        peak, scal = float(np.max(getattr(a, mp)["data"])), getattr(a, sc)
        if scal is None:
            raise ValueError(f"The car file {path} has the aero map '{mp}' but a missing scalar '{sc}'.")
        if peak < 0:
            raise ValueError(f"The car file {path} has the aero map '{mp}' with a negative peak.")
        if (peak == 0) != (scal == 0):
            raise ValueError(f"The car file {path} has the aero map '{mp}' with a zero peak or a zero scalar '{sc}', "
                             f"but not both. A car without this force needs both at zero.")
        if peak > 0:        # a zero map with a zero scalar stays as it is
            getattr(a, mp)["data"] = getattr(a, mp)["data"] * (scal / peak)
    return SimpleNamespace(path=path, sha8=sha8, **groups)


def table_live(v):
    """Return True when a decoded aero record holds a non-zero value. A table counts by its y or data, not its axis."""
    if v is None:
        return False
    if isinstance(v, dict):
        v = v["y"] if "y" in v else v.get("data", 0.0)
    try:
        return bool(np.any(np.asarray(v, float) != 0))
    except (TypeError, ValueError):
        return any(table_live(e) for e in v)


def aero_refusals(cs):
    """Raise ValueError for an aero feature of the car that the model does not build."""
    a = cs.aero
    live = lambda *names: [n for n in names if table_live(getattr(a, n, None))]
    for what, names in (
            ("the roll split table (cid 139)", ["refused_139"]),
            ("a 2-D steer or state map (cids 452 to 463)", [f"refused_{c}" for c in range(452, 464)]),
            ("a wake map (cids 870, 872 to 875)", [f"refused_{c}" for c in (870, 872, 873, 874, 875)])):
        if live(*names):
            raise ValueError(f"The car file {cs.path} uses {what}. The car model does not build it.")
    if any(getattr(a, f"refused_{c}", None) == 1 for c in (434, 435, 436)):     # ChassisSim ignores a table with its enable at 0
        raise ValueError(f"The car file {cs.path} uses the steer correction (cids 434 to 436). The car model does not build it.")
    if a.multi_drs_366 == 1:
        raise ValueError(f"The car file {cs.path} uses the multiple DRS zones (cid 366). The car model does not build it.")


# ChassisSim has no wheel inertia (its lap simulation has no wheel-spin states). FullModelSim needs one.
# ponytail: one constant for a single-seater wheel, tyre and brake, plus a share of the driveline. Read a value
# from the car file when a ChassisSim version stores one.
CS_JW = 1.0  # kg m^2


def car_vp(car):
    """Return the vehParams primary keys of the shared fields, as a dict for vp_overrides.

    The keys make vehParams derive the car mass, the front weight share, the wheelbase, the CG height,
    the track, the inertias, the wheel radius, the rolling resistance and the tyre rate. mb is the sprung
    mass and md is 0, because the car mass already holds the driver.
    ChassisSim gives each unsprung mass as the sum of the left and right corners, as muf and mur of vehParams.
    wB of vehParams is the front share of the sprung mass. So wB = (front share x total - muf) / mb.
    vehParams holds one track, one tyre rate and one wheel radius: t and kt are the means of the axles,
    and Rw is the rear radius (the driven wheel). The car model reads the axle values from the car record.
    Jw is CS_JW, because the car file holds no wheel inertia.
    Raise ValueError when the car lacks a field, so that no default of another car enters the solve.
    """
    m, g, ty = car.mass, car.geom, car.tyre
    need = {"mass.total": m.total, "mass.unsprung_front": m.unsprung_front, "mass.unsprung_rear": m.unsprung_rear,
            "mass.front_share": m.front_share, "mass.Ix": m.Ix, "mass.Iy": m.Iy, "mass.Iz": m.Iz,
            "geom.wheelbase": g.wheelbase, "geom.cg_height": g.cg_height, "geom.track_front": g.track_front,
            "geom.track_rear": g.track_rear, "tyre.fl": ty.fl, "tyre.rl": ty.rl}
    missing = sorted(k for k, v in need.items() if v is None)
    if missing:
        raise ValueError(f"The car file {car.path} lacks the fields {missing}. A ChassisSim car needs them.")
    mb = m.total - m.unsprung_front - m.unsprung_rear
    return {"mb": mb, "md": 0.0, "muf": m.unsprung_front, "mur": m.unsprung_rear,
            "wB": (m.front_share * m.total - m.unsprung_front) / mb,
            "l": g.wheelbase, "hcg": g.cg_height, "I_x": m.Ix, "I_y": m.Iy, "I_z": m.Iz,
            "t": (g.track_front + g.track_rear) / 2, "Rw": ty.rl["radius"], "Jw": CS_JW,
            "f": ty.rl["rolling_drag"], "kt": (ty.fl["kz"] + ty.rl["kz"]) / 2}


# ---------------------------------------------------------------- suspension kinematics (front view)
TRAVEL = 0.05       # m, wheel travel range of the fitted curves (bump positive)


def _circ_hit(c0, r0, c1, r1, near):
    """Return (p, miss): the intersection of two circles in the plane that lies nearest to the point near, and the
    distance by which the circles miss each other (0 when they intersect). A miss keeps the point on the line of the
    centres, so that a root finder still sees a continuous function. The caller checks miss at the root."""
    d = c1 - c0
    dn = np.hypot(*d)
    a = (r0 * r0 - r1 * r1 + dn * dn) / (2 * dn)
    h = np.sqrt(max(r0 * r0 - a * a, 0.0))
    miss = np.sqrt(max(a * a - r0 * r0, 0.0))
    m = c0 + a * d / dn
    p = [m + s * h * np.array([-d[1], d[0]]) / dn for s in (1.0, -1.0)]
    return (p[0] if np.hypot(*(p[0] - near)) < np.hypot(*(p[1] - near)) else p[1]), miss


def _line_hit(a, b, c, d):
    """Return the intersection of the line a-b with the line c-d (2D)."""
    t = np.linalg.solve(np.array([[b[0] - a[0], c[0] - d[0]], [b[1] - a[1], c[1] - d[1]]]), c - a)
    return a + t[0] * (b - a)


def axle_kinematics(hp, track, rc_static, travel=TRAVEL, n=21, name=""):
    """Solve the front view of one double wishbone and fit the curves. hp has shape (6, 3): the rows of the car
    file (lower inner front, lower inner aft, lower outer, upper inner front, upper inner aft, upper outer).
    The inner points of each wishbone are projected to one pivot (the mean of the two). The upright is rigid and
    the contact patch (y = track / 2) is fixed to it. The wheel moves by w, bump positive.
    The ground height of ChassisSim is a reference z that no file field holds. It is found here: the z of the
    ground at which the static roll centre equals rc_static. The result holds the datum, the curves as poly1d in
    w [m]: rc_h [m] above the contact patches and camber [rad]. The camber is the change of the negative camber
    against the body: positive when the top leans in. The result also holds the range lo, hi [m]."""
    li, lo, ui, uo = hp[:2, 1:].mean(0), hp[2, 1:], hp[3:5, 1:].mean(0), hp[5, 1:]
    L1, L2, Lu = np.hypot(*(lo - li)), np.hypot(*(uo - ui)), np.hypot(*(uo - lo))
    th0 = np.arctan2(*(lo - li)[::-1])
    yc = track / 2

    def pose(dl):
        LO = li + L1 * np.array([np.cos(th0 + dl), np.sin(th0 + dl)])
        UO, miss = _circ_hit(ui, L2, LO, Lu, uo)
        return LO, UO, miss

    def ang(LO, UO):
        u = UO - LO
        return np.arctan2(-u[0], u[1])
    a0 = ang(lo, uo)

    def rc_line(zg, dl):
        """Return (contact patch, travel w, instant centre, camber) of the pose dl."""
        LO, UO, _ = pose(dl)
        da = ang(LO, UO) - a0
        c0 = np.array([yc, zg]) - lo
        cp = LO + np.array([[np.cos(da), -np.sin(da)], [np.sin(da), np.cos(da)]]) @ c0
        u = (LO - li) / L1
        v = (UO - ui) / L2
        if abs(u[0] * v[1] - u[1] * v[0]) < 1e-12:     # parallel wishbones: the instant centre is at infinity
            ic = cp + 1e6 * u       # ponytail: a far point on the arm direction gives the parallel RC line, exact slope
        else:
            ic = _line_hit(li, LO, ui, UO)
        return cp, cp[1] - zg, ic, da

    def height(cp, ic):             # height at y = 0 of the line from the contact patch to the instant centre
        return cp[1] + (0.0 - cp[0]) * (ic[1] - cp[1]) / (ic[0] - cp[0])
    cp, _, ic, _ = rc_line(0.0, 0.0)
    h0 = height(cp, ic)             # linear in zg: h(zg) = zg (1 - k) + k ic_z
    k = (height(*rc_line(1.0, 0.0)[::2]) - h0)
    if abs(k) < 1e-9:
        raise ValueError(f"The stored roll-centre height of the {name} axle cannot set the ground datum, "
                         "because the roll centre of this geometry does not depend on the ground height.")
    zg = (rc_static - h0) / k
    w = np.linspace(-travel, travel, n)
    dls = []
    try:
        for wi in w:
            dls.append(brentq(lambda d: rc_line(zg, d)[1] - wi, -0.6, 0.6))
    except (ValueError, np.linalg.LinAlgError) as e:
        raise ValueError(f"The suspension geometry of the {name} axle has no solution for a wheel travel of "
                         f"+-{travel * 1e3:.0f} mm. Check the hardpoints in the car file.") from e
    if any(pose(d)[2] > 1e-6 for d in dls):        # the wishbones and the upright must close at every root
        raise ValueError(f"The suspension geometry of the {name} axle has no solution for a wheel travel of "
                         f"+-{travel * 1e3:.0f} mm. Check the hardpoints in the car file.")
    rows = [rc_line(zg, d) for d in dls]
    rc_h = np.array([height(r[0], r[2]) - (zg + wi) for r, wi in zip(rows, w)])     # above the contact patch
    cam = np.array([r[3] for r in rows])
    return SimpleNamespace(zg=zg, lo=-travel, hi=travel, rc_h=np.poly1d(np.polyfit(w, rc_h, 6)),
                           camber=np.poly1d(np.polyfit(w, cam - cam[n // 2], 3)))


def suspension_curves(car):
    """Return SimpleNamespace(f, r) of axle_kinematics for a car record. Cached on the record.
    The stored roll-centre height (cids 55, 59, 63, 67) is the chassis-frame height at w = 0: rc_h(0) + zg.
    rc_h itself is the height above the contact patches. A car with a bump steer table (cids 74-77) that is not
    zero raises ValueError, because the model has no bump steer."""
    for c in _C4:
        t = getattr(car.susp, f"bump_steer_{c}", None)
        if t is not None and np.any(t["y"]):
            raise ValueError("The car file holds a bump steer table that is not zero. The model does not support bump steer yet.")
    if getattr(car, "_curves", None) is None:
        g, h = car.geom, car.hardpoints
        need = (g.rc_height_fl, g.rc_height_fr, g.rc_height_rl, g.rc_height_rr, h.front_l, h.rear_l)
        if any(v is None for v in need):
            raise ValueError("The car file lacks the roll-centre heights (cids 55, 59, 63, 67) or the hardpoints (cids 49-52).")
        car._curves = SimpleNamespace(
            f=axle_kinematics(np.asarray(h.front_l, float), g.track_front, 0.5 * (g.rc_height_fl + g.rc_height_fr), name="front"),
            r=axle_kinematics(np.asarray(h.rear_l, float), g.track_rear, 0.5 * (g.rc_height_rl + g.rc_height_rr), name="rear"))
    return car._curves
