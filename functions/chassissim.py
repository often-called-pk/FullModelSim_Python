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
# ponytail: skipped the cids that the sheet leaves unnamed (54-68, 94, 95, 97, 126, 150-153). Add one when a task needs it.
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
    d1, d2, d3, _, d5, d6, _ = (int(d) for d in dims)
    z = tb[3][d1:d1 + d2]
    grid = tb[3][d1 + d2:].reshape(d1, d2)
    t = sc[6]
    return {
        "toe": sc[0] * DEG, "camber": sc[1] * DEG, "castor": sc[2] * DEG, "radius": sc[4], "pressure": sc[5],
        "temp": sc[6], "rolling_drag": sc[7], "optimum_camber": sc[11] * DEG, "peak_slip_angle": sc[8], "peak_slip_ratio": sc[9], "grip": sc[15],
        "load": tb[3][:d1], "fy_peak": np.array([np.interp(t, z, r) for r in grid]),
        "slip_x": tb[0][:d3], "slip_y": tb[0][d3:],
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
    return SimpleNamespace(path=path, sha8=sha8, **groups)


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
