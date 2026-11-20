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
# The order of the four corner records of every group in the file: ChassisSim stores the right wheel first.
_C4 = ("fr", "fl", "rr", "rl")
_GAPS = (*_C4, "third_f", "third_r")


def _corners(group, base, prefix="", f=1.0):
    """Return the CAR_FIELDS entries of four corner records. base is the cid of the FR record.
    ChassisSim stores the right wheel first: FR, FL, RR, RL."""
    return {base + i: (f"{group}.{prefix}{c}", f) for i, c in enumerate(_C4)}


# (right cid, left cid) of the records that ChassisSim copies from right to left when the left one is empty.
# The groups: springs, rubber air gaps, rubber tables, corner motion ratios, dampers, tyres, bar motion ratios.
_SIDES = [(b + i, b + i + 1) for b in (1, 7, 13, 23, 31, 80) for i in (0, 2)] + [(27, 170), (28, 171)]


def table_live(v):
    """Return True when a record holds a value that is not zero, raw from the file or decoded.
    A decoded table counts by its y or data, not its axis. A text value counts as content."""
    if v is None:
        return False
    if isinstance(v, dict):
        v = v["y"] if "y" in v else v.get("data", 0.0)
    if isinstance(v, str):      # a str holds str elements: the loop below would never end
        return True
    try:
        return bool(np.any(np.asarray(v, float) != 0))
    except (TypeError, ValueError):
        return any(table_live(e) for e in v)


# cid -> (group.name, factor to SI). A tuple factor has one entry per part of the decoded record:
#   tag 4 (x, force)   tag 7 (x, y)   tag 9 (x, z, data)   tag 11 (preload, rate, x, y)   tag 12 (unused, force)
# The file stores springs, bars and dampers in SI, so no field needs lbf/in. It stores a ride height in mm
# and a damper velocity in mm/s.
# ponytail: skipped the cids that the sheet leaves unnamed (56, 60, 64, 68, 94, 95, 97, 126, 150-153).
# Add one when a task needs it.
CAR_FIELDS = {
    84: ("mass.total", 1.0), 85: ("mass.unsprung_front", 1.0), 86: ("mass.unsprung_rear", 1.0),
    87: ("mass.Ix", 1.0), 88: ("mass.Iy", 1.0), 89: ("mass.Iz", 1.0), 96: ("mass.front_share", 1.0),
    90: ("geom.wheelbase", 1.0), 91: ("geom.track_front", 1.0), 92: ("geom.track_rear", 1.0),
    93: ("geom.cg_height", 1.0),
    # A stored geometry block is a flag and a triple per corner: FR flag 53, triple 54-56. FL 57, 58-60.
    # RR 61, 62-64. RL 65, 66-68. Cid 69 is a trailing flag. The second value of a triple is the roll-centre height.
    **{55 + 4 * i: (f"geom.rc_height_{c}", 1.0) for i, c in enumerate(_C4)},
    # The first value of a triple (cids 54, 58, 62, 66): the RC height at the hardpoint datum. That is the height
    # with the contact patch on z = 0 of the hardpoint frame.
    **{54 + 4 * i: (f"geom.rc_datum0_{c}", 1.0) for i, c in enumerate(_C4)},
    **_corners("geom", 117, "ride_height_", MM),
    **_corners("susp", 1, "spring_"), 5: ("susp.spring_third_f", 1.0), 6: ("susp.spring_third_r", 1.0),
    **{7 + i: (f"susp.gap_{n}", 1.0) for i, n in enumerate(_GAPS)},
    **{204 + i: (f"susp.gap_flag_{n}", 1.0) for i, n in enumerate(_GAPS)},   # 1 = the gap is measured on the ground
    **_corners("susp", 74, "bump_steer_"),      # tag 7: wheel travel [m], toe change [rad]
    **_corners("susp", 13, "rubber_"), 17: ("susp.rubber_third_f", 1.0), 18: ("susp.rubber_third_r", 1.0),
    19: ("susp.arb_f", 1.0), 21: ("susp.arb_r", 1.0),
    # The bar motion ratio is one record per corner: 27 FR, 170 FL, 28 RR, 171 RL.
    **_corners("susp", 23, "mr_"), 27: ("susp.mr_arb_fr", 1.0), 170: ("susp.mr_arb_fl", 1.0),
    28: ("susp.mr_arb_rr", 1.0), 171: ("susp.mr_arb_rl", 1.0),
    29: ("susp.mr_third_f", 1.0), 30: ("susp.mr_third_r", 1.0),
    **_corners("susp", 31, "damper_"), 35: ("susp.damper_third_f", 1.0), 36: ("susp.damper_third_r", 1.0),
    **{49 + i: (f"hardpoints.{n}", 1.0) for i, n in enumerate(("front_r", "front_l", "rear_r", "rear_l"))},
    **_corners("tyre", 80),
    # Bump rubber hysteresis per corner: the enable flag (tag 6) and the curve (tag 4: x, loading force, unloading force).
    # x is the compression past the rubber air gap [m].
    **_corners("susp", 657, "hyst_flag_"), **_corners("susp", 661, "hyst_"),
    # Third-spring hysteresis flags (tag 6), front and rear. load_car refuses a value other than 0.
    # ponytail: the cids are confirmed. The third spring needs the same law on the heave, with the curves of
    # cids 640 / 641 (not read). Add it when a car sets a flag.
    638: ("susp.hyst_flag_third_f", 1.0), 639: ("susp.hyst_flag_third_r", 1.0),
    121: ("aero.cla", 1.0), 122: ("aero.cda", 1.0), 123: ("aero.offset", 1.0), 125: ("aero.rho", 1.0),
    127: ("aero.cla_map", 1.0), 131: ("aero.cda_map", 1.0), 135: ("aero.balance_map", 1.0),
    **{c: (f"aero.drs_{c}", 1.0) for c in range(336, 342)},
    **{c: (f"aero.multi_drs_{c}", 1.0) for c in range(366, 372)},
    # Corrections of the aero maps (tag 7 tables). Sideslip x is in rad, speed x in m/s. Enables are 0 or 1.
    128: ("aero.ss_cla", 1.0), 132: ("aero.ss_cda", 1.0), 136: ("aero.ss_ab", 1.0),
    **{413 + i: (f"aero.sp_on_{n}", 1.0) for i, n in enumerate(("cla", "cda", "ab"))},
    **{416 + i: (f"aero.sp_{n}", 1.0) for i, n in enumerate(("cla", "cda", "ab"))},
    # Features that the car model refuses when they are live: steer correction, roll split, 2-D maps, wake maps
    # (maps 872 to 875, with the flags refused.c876 to c879).
    **{c: (f"aero.refused_{c}", 1.0) for c in (139, *range(434, 440), *range(452, 464), *range(872, 876))},
    100: ("engine.efficiency", 1.0), 101: ("engine.torque_curve", (RPM, 1.0)),
    102: ("engine.rpm_idle", RPM), 103: ("engine.rpm_limit", RPM), 110: ("engine.gears", 1.0),
    111: ("engine.torque_split", 1.0),
    # Shift cut: delay [s], minimum torque factor, ramp-back time [s] (P3). Engine-brake table (tag 7): refused when live.
    104: ("engine.cut_delay", 1.0), 105: ("engine.cut_min", 1.0), 106: ("engine.cut_ramp", 1.0),
    161: ("engine.brake_table", (RPM, 1.0)),
    # Hybrid layer (P4). 177 master, 178 deploy mode, 179 regen clamp [W], 180 deploy scale [W], 181 energy cap [J],
    # 182 regen scale, 183 deploy scale after the map, 836 free energy offset, 590 brake regen gate, 589 paddle gate.
    # 973 = brake regen blend: ChassisSim cuts the friction brake for the regen only when it is on (absent = off).
    # 870 = 1 sends the regen front share 871 to the front axle (0: rear axle only).
    # Deploy maps 467 (front) and 468 (rear): x engine speed, z throttle / 100, data = fraction of cid 180.
    **{c: (f"hybrid.{n}", 1.0) for c, n in ((177, "on"), (178, "mode"), (179, "clamp_regen"), (180, "scale_deploy"),
                                             (181, "e_cap"), (182, "k_regen"), (183, "k_deploy"), (836, "e_free"),
                                             (590, "gate_brake"), (535, "use_brake_maps"), (573, "use_paddle"),
                                             (589, "gate_paddle"), (973, "blend_brake"),
                                             (464, "blend_on"), (465, "blend"), (466, "map_on"),
                                             (870, "regen_front_on"), (871, "regen_front_share"))},
    467: ("hybrid.map_front", (RPM, 1.0, 1.0)), 468: ("hybrid.map_rear", (RPM, 1.0, 1.0)),
    # Features that the model refuses when on: brake ramp 571, split recompute 598, push-to-pass 649, torque
    # multiplier maps 520, throttle trims 419 / 423 / 426 / 428, battery plugin 897 / 906, wake map flags 876 to 879.
    # 853 = brake split by the tyre limit of each wheel (ChassisSim then ignores cid 160).
    **{c: (f"refused.c{c}", 1.0) for c in (571, 598, 853, 649, 520, 419, 423, 426, 428, 897, 906, *range(876, 880))},
    112: ("diff.front_type", 1.0), 113: ("diff.rear_type", 1.0), 114: ("diff.front_map", 1.0),
    116: ("diff.rear_map", 1.0),
    107: ("brake.force", 1.0), 108: ("brake.front_bias", 1.0), 109: ("brake.delay", 1.0),
    160: ("brake.left_share", 1.0),
    390: ("env.track_temp", 1.0),
    # Stability index correction of the QSS apex speed: 655 = 1 turns it on, 656 = map (avg stbi, speed) -> factor.
    655: ("stability.on", 1.0), 656: ("stability.map", (1.0, 1.0, 1.0)),
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
    if tag == 4:
        a = np.asarray(v, float).reshape(3, -1)
        return {"x": a[0] * _f(f, 0), "load": a[1] * _f(f, 1), "unload": a[2] * _f(f, 1)}
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


# Record tag that load_car expects for each hysteresis cid: 6 for a flag, 4 for a curve.
_HYST_TAG = {**dict.fromkeys((638, 639, *range(657, 661)), 6), **dict.fromkeys(range(661, 665), 4)}


def _check_hyst(h, c, cid, path):
    """Raise ValueError when the hysteresis curve h of corner c (record cid) cannot give a rubber law."""
    x, lo, up = h["x"], h["load"], h["unload"]
    who = f"The car file {path} has a bump rubber hysteresis curve of corner {c} (cid {cid})"
    if (x.size < 2 or not np.all(np.isfinite(np.r_[x, lo, up])) or np.any(np.diff(x) <= 0)
            or not np.max(lo) > 0):
        raise ValueError(f"{who} that the model cannot use. The curve needs at least 2 points, finite values, "
                         f"an x column that increases strictly and a loading force above 0.")
    if np.any(up > lo + 0.01 * np.max(lo)):
        raise ValueError(f"{who} with an unloading force above the loading force. "
                         f"The limit is 1 % of the peak loading force.")


def file_sha8(path):
    """Return the first 8 hex digits of the sha256 of the file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _atmos(chassissim_dir=None):
    """Return the engine torque factor of the Atmos_Cond.ini file that ChassisSim uses.
    ChassisSim joins the install folder and the file name without a separator. So the file is
    '<install folder name>Atmos_Cond.ini' in the parent of the install folder. A file in the install folder is not read.
    The folder is the argument, else CHASSISSIM_DIR, as install_dir. The factor is 1.0 when neither is set, without
    the file, or with Eng Corr = 0 (the default). A folder that does not exist raises ValueError.
    Factor = sqrt(T_eng / T_cur) x (P_cur - Pw_cur) / (P_eng - Pw_eng), the temperatures in K, the pressures in bar.
    In the two sections, a missing key takes Atm Press 1.013 bar, Water Press 0 bar and Temperature 15 degC.
    Section and key names ignore case. A value that is not a finite number raises ValueError."""
    if not (chassissim_dir or os.environ.get("CHASSISSIM_DIR")):
        return 1.0
    # ChassisSim joins the install folder and the file name without a separator. We read the same path.
    folder = os.path.abspath(install_dir(chassissim_dir))
    f = _find(os.path.dirname(folder), os.path.basename(folder) + "Atmos_Cond.ini")
    if f is None:
        return 1.0
    cp = configparser.ConfigParser(strict=False, interpolation=None, inline_comment_prefixes=(";", "#"))
    key = "the sections"

    def get(sec, k, d):
        nonlocal key
        key = f"the key '{k}' of the section [{sec}]"
        v = float(secs[sec].get(k, d)) if sec in secs else d
        if not np.isfinite(v):
            raise ValueError
        return v
    try:
        with open(f, "rb") as fh:
            cp.read_string(fh.read().removeprefix(b"\xef\xbb\xbf").decode(ENC))
        secs = {s.lower(): cp[s] for s in cp.sections()}
        if get("preferences", "eng corr", 0.0) == 0:       # ChassisSim applies the correction when the flag is not 0
            return 1.0
        (pc, wc, tc), (pe, we, te) = (
            (get(sec, "atm press", 1.013), get(sec, "water press", 0.0), get(sec, "temperature", 15.0) + 273.15)
            for sec in ("current conditions", "engine conditions"))
    except (configparser.Error, ValueError) as ex:
        raise ValueError(f"{f}: cannot read {key}. The file needs sections with lines 'key = finite number'.") from ex
    if min(tc, te) <= 0 or pe - we <= 0 or pc - wc <= 0:
        raise ValueError(f"{f}: the temperatures must be above -273.15 degC and each dry air pressure above 0.")
    return float(np.sqrt(te / tc) * (pc - wc) / (pe - we))


def load_car(path, chassissim_dir=None):
    """Read a ChassisSim car file and return its consumed fields in SI.

    The result has the groups mass, geom, tyre, aero, susp, engine, hybrid, refused, diff, brake, hardpoints, env
    and stability. hybrid holds the records of the hybrid layer. refused holds the flags of features that
    power_refusals and aero_refusals refuse.
    Each group is a SimpleNamespace. A field that the file lacks is None. The result also holds
    path (absolute) and sha8 (the first 8 hex digits of the sha256 of the file bytes).
    Angles are in rad and lengths in m. Every engine-speed axis is in rad/s (the torque curve, the engine-brake
    table and the hybrid deploy maps). The torque is in N m.
    A damper holds the velocity in m/s and the bump and rebound forces in N, both positive.
    Its table is active when it has points. A spring holds its preload in m and its rate in N/m.
    Toe is positive for toe-out. Camber is the negative camber: positive when the top of the wheel leans in.
    Brake force is the total in N, bias is the front share, delay is in s.
    A tyre is a dict. It holds the peak lateral force over the load at its background temperature,
    the slip and camber curves, the peak slip angle and ratio, the radius and the vertical rate.
    A hardpoint block has shape (6, 3), in the car frame of ChassisSim. The rows are lower inner front,
    lower inner aft, lower outer, upper inner front, upper inner aft and upper outer.
    susp.hyst_<c> is the bump rubber hysteresis curve of corner c: a dict with x in m and the load and unload
    forces in N. With the flag susp.hyst_flag_<c> at 1, a curve that cannot give a rubber law raises ValueError.
    The file stores the right wheel first: the lowest cid of four corner records is FR, then FL, RR, RL.
    As ChassisSim does, a left record that is absent or all zero takes the record of its right sibling, when
    that record holds a value. This applies to the springs, the rubber air gaps (the gap flag goes with the gap),
    the rubber tables, the corner and bar motion ratios, the dampers and the tyres.
    engine.atmos is the atmospheric torque factor of the Atmos_Cond.ini file of the ChassisSim folder (the argument,
    else CHASSISSIM_DIR). _atmos gives the path. It is 1.0 when no folder is set, without the file, or with
    Eng Corr = 0.
    """
    path = os.path.abspath(path)
    sha8 = file_sha8(path)
    recs = {c: (t, v) for t, c, v in carfile.decode(path)}
    for r, l in _SIDES:
        if r in recs and table_live(recs[r][1]) and not (l in recs and table_live(recs[l][1])):
            recs[l] = recs[r]
            if 7 <= r <= 10 and r + 197 in recs:        # the flag of a copied rubber gap (cids 204 to 207)
                recs[l + 197] = recs[r + 197]
    groups = {}
    for name, _ in CAR_FIELDS.values():
        g, a = name.split(".")
        setattr(groups.setdefault(g, SimpleNamespace()), a, None)
    for cid, (name, f) in CAR_FIELDS.items():
        if cid in recs:
            g, a = name.split(".")
            t, v = recs[cid]
            want = _HYST_TAG.get(cid, t)
            if t != want:
                raise ValueError(f"The car file {path} holds the rubber hysteresis record {cid} with tag {t}. "
                                 f"The model expects tag {want}.")
            setattr(groups[g], a, _tyre_decode(v) if t == 13 else _conv(t, v, f))
    su = groups["susp"]
    for cid, a in ((638, "f"), (639, "r")):
        if getattr(su, f"hyst_flag_third_{a}") not in (None, 0):
            raise ValueError(f"The car file {path} sets a third-spring hysteresis flag (cid {cid}). "
                             f"The car model has no third-spring hysteresis.")
    for cid_flag, c in zip(range(657, 661), _C4):          # hysteresis on: check the curve
        flag, h = getattr(su, f"hyst_flag_{c}"), getattr(su, f"hyst_{c}")
        if flag not in (None, 0, 1):
            raise ValueError(f"The car file {path} holds a bump rubber hysteresis flag of corner {c} "
                             f"(cid {cid_flag}) other than 0 or 1.")
        if flag == 1 and h is not None:
            _check_hyst(h, c, cid_flag + 4, path)
    sb, mp = groups["stability"], groups["stability"].map
    # The QSS reads the map only with the flag on and a map that is not all zero. Only then the axes must be valid.
    if (sb.on == 1 and mp is not None and np.any(mp["data"] != 0)
            and (np.any(np.diff(mp["x"]) <= 0) or np.any(np.diff(mp["z"]) <= 0))):
        raise ValueError(f"The car file {path} has a stability index map (cid 656) with axes that do not increase strictly.")
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
        # A zero scalar gives a zero force in ChassisSim. A map with a zero peak makes ChassisSim divide by zero (NaN).
        # Here each of the two gives a zero force, for CLA and CDA separately.
        getattr(a, mp)["data"] = getattr(a, mp)["data"] * (scal / peak if peak > 0 else 0.0)
    groups["engine"].atmos = _atmos(chassissim_dir)
    return SimpleNamespace(path=path, sha8=sha8, **groups)


def aero_refusals(cs):
    """Raise ValueError for an aero feature of the car that the model does not build."""
    a = cs.aero
    live = lambda *names: [n for n in names if table_live(getattr(a, n, None))]
    for what, names in (
            ("the roll split table (cid 139)", ["refused_139"]),
            ("a 2-D steer or state map (cids 452 to 463)", [f"refused_{c}" for c in range(452, 464)]),
            ("a wake map (cids 872 to 875)", [f"refused_{c}" for c in range(872, 876)])):
        if live(*names):
            raise ValueError(f"The car file {cs.path} uses {what}. The car model does not build it.")
    for c in range(876, 880):
        if getattr(cs.refused, f"c{c}") not in (None, 0):
            raise ValueError(f"The car file {cs.path} sets a wake map flag (cid {c}). The car model does not build it.")
    if any(getattr(a, f"refused_{c}", None) == 1 for c in (434, 435, 436)):     # ChassisSim ignores a table with its enable at 0
        raise ValueError(f"The car file {cs.path} uses the steer correction (cids 434 to 436). The car model does not build it.")
    if a.multi_drs_366 == 1:
        raise ValueError(f"The car file {cs.path} uses the multiple DRS zones (cid 366). The car model does not build it.")


def _text_live(v):
    """Return True when a text record (tag 8) holds more than NUL bytes and white space."""
    if v is None:
        return False
    b = v if isinstance(v, (bytes, bytearray)) else str(v).encode(ENC)
    return bool(bytes(b).replace(b"\x00", b"").strip())


def flag_on(v):
    """Return True when a flag record is on: present and not 0."""
    return v not in (None, 0)


def hybrid_gates(cs):
    """Return (deploy, brake regen, paddle regen) of the hybrid layer, as ChassisSim gates them.
    Deploy needs cid 177. Brake regen needs 177, 535, 590 and 973. Paddle regen needs 177, 573 and 589."""
    h = cs.hybrid
    dep = flag_on(h.on)
    return (dep, dep and flag_on(h.use_brake_maps) and flag_on(h.gate_brake) and flag_on(h.blend_brake),
            dep and flag_on(h.use_paddle) and flag_on(h.gate_paddle))


def hybrid_inputs(cs):
    """Return the names of the extra inputs of the hybrid layer: [] (cid 177 off), ['F_dep'] or ['F_dep', 'F_pad'].
    F_dep is the deploy force and F_pad the paddle regen force (cids 573 and 589), both in N at the wheels."""
    dep, _, pad = hybrid_gates(cs)
    return (["F_dep"] + (["F_pad"] if pad else [])) if dep else []


def power_refusals(cs):
    """Raise ValueError for a powertrain feature of the car that the model does not build."""
    r, e, h = cs.refused, cs.engine, cs.hybrid
    s = e.torque_split
    if s is not None and not 0 <= s <= 1:       # NaN fails the comparison too
        raise ValueError(f"The car file {cs.path} holds a torque split (cid 111) of {s}. The split must be from 0 to 1.")
    on = flag_on
    dep, brk, pad = hybrid_gates(cs)
    if dep:             # the hybrid layer (cid 177 is the master): its scales must be given
        for c, v, low in ((180, h.scale_deploy, False), (181, h.e_cap, True), (179, h.clamp_regen if brk or pad else 1.0, False)):
            if v is None or not np.isfinite(v) or v < 0 or (v == 0 and not low):
                raise ValueError(f"The car file {cs.path} sets the hybrid layer (cid 177). "
                                 + (f"The file does not hold cid {c}. " if v is None else f"Cid {c} is {v}. ")
                                 + f"The value must be {'0 or more' if low else 'above 0'}.")
        for c, v in ((182, h.k_regen), (183, h.k_deploy), (836, h.e_free)):     # None keeps 1, 1 and 0
            if v is not None and not (np.isfinite(v) and v >= 0):
                raise ValueError(f"The car file {cs.path} holds {v} in cid {c}. The value must be 0 or more.")
        if h.mode not in (None, 0, 1):
            raise ValueError(f"The car file {cs.path} holds a deploy mode (cid 178) of {h.mode}. The mode must be 0 or 1.")
        if h.e_free is not None and 0 < h.e_free < 1:
            raise ValueError(f"The car file {cs.path} holds {h.e_free} in cid 836. ChassisSim reads a value between 0 and 1 "
                             "as an early stop of the deploy. The model does not build that stop. Use 0, or a value of 1 or more.")
        if h.k_regen is not None and h.k_regen > 1:     # the blend keeps the axle forces, so a scale above 1 makes energy
            raise ValueError(f"The car file {cs.path} holds a regen scale (cid 182) of {h.k_regen}. "
                             "The model harvests the brake force that the car applies. A scale above 1 makes energy.")
        for c, v in ((465, h.blend), (871, h.regen_front_share)):
            if v is not None and not 0 <= v <= 1:
                raise ValueError(f"The car file {cs.path} holds {v} in cid {c}. The share must be from 0 to 1.")
        for c, tab in ((467, h.map_front), (468, h.map_rear)):
            if on(h.map_on) and tab is not None:
                d = np.asarray(tab["data"], float)
                if not (np.all(np.isfinite(d)) and np.all(d >= 0)):
                    raise ValueError(f"The car file {cs.path} holds a deploy map (cid {c}) with a value that is "
                                     "not a number or is below 0. Each map value must be 0 or more.")
    ls = cs.brake.left_share
    if ls is not None and ls != 0.5:
        raise ValueError(f"The car file {cs.path} holds a brake left share (cid 160) of {ls}. "
                         "The car model gives each side 0.5 of the brake force. Use 0.5.")
    if on(h.use_paddle) and on(h.gate_paddle) and not dep:
        raise ValueError(f"The car file {cs.path} sets the paddle regen (cids 573 and 589) with the hybrid layer off "
                         "(cid 177). The car model builds the paddle only in the hybrid layer. Set cid 177, or clear cid 573.")
    if pad and h.k_regen not in (None, 1):
        raise ValueError(f"The car file {cs.path} sets the paddle regen (cids 177, 573 and 589) with a regen scale "
                         f"(cid 182) of {h.k_regen}. ChassisSim scales the paddle force and its energy together. "
                         "The car model does not build that. Use a regen scale of 1.")
    for what, c, hit in (
            ("the brake ramp", 571, on(r.c571)),
            ("the per-step torque split recompute", 598, on(r.c598)),
            ("the brake split by the tyre limit of each wheel", 853, on(r.c853)),
            *(("a battery plugin", c, _text_live(getattr(r, f"c{c}"))) for c in (897, 906)),
            ("push-to-pass", 649, on(r.c649)),
            ("the engine torque multiplier maps", 520, on(r.c520)),
            *(("a throttle trim", c, on(getattr(r, f"c{c}"))) for c in (419, 423, 426, 428)),
            ("a non-zero engine-brake table", 161, table_live(e.brake_table))):
        if hit:
            raise ValueError(f"The car file {cs.path} uses {what} (cid {c}). The car model does not build it.")


def check_car(cs):
    """Raise one ValueError that lists every field the car model or the QSS reads and the car file lacks.
    An aero map correction, a DRS value or a speed table counts only when its enable is 1."""
    C = ("fl", "fr", "rl", "rr")
    for cid, n in zip(range(49, 53), ("front_r", "front_l", "rear_r", "rear_l")):
        hp = getattr(cs.hardpoints, n)
        if hp is not None and len(hp) != 6:
            raise ValueError(f"The car file {cs.path} holds a hardpoint block (cid {cid}) with {len(hp)} points. "
                             f"The car model needs 6 points (two wishbones). A strut suspension is not built yet.")
    need = {f"mass.{n}": getattr(cs.mass, n) for n in ("total", "unsprung_front", "unsprung_rear", "Ix", "Iy", "Iz", "front_share")}
    for grp, obj, names in (
            ("geom", cs.geom, ("wheelbase", "track_front", "track_rear", "cg_height", *(f"rc_height_{c}" for c in C),
                               *(f"ride_height_{c}" for c in C))),
            ("hardpoints", cs.hardpoints, ("front_l", "rear_l")),
            ("susp", cs.susp, (*(f"{p}_{c}" for p in ("spring", "mr", "gap", "damper") for c in C),
                               "arb_f", "arb_r", "mr_arb_fr", "mr_arb_rr", "spring_third_f", "spring_third_r")),
            ("aero", cs.aero, ("cla", "cda", "cla_map", "cda_map", "balance_map")),
            ("engine", cs.engine, ("efficiency", "torque_curve", "rpm_idle", "rpm_limit", "gears")),
            ("diff", cs.diff, ("rear_type",)),
            ("brake", cs.brake, ("force", "front_bias")),
            ("tyre", cs.tyre, C)):
        need.update({f"{grp}.{n}": getattr(obj, n) for n in names})
    h = cs.hybrid                   # the front deploy share is s_dep of vehModel_cs.hybrid_params
    s_dep = flag_on(h.on) and flag_on(h.blend_on) and (h.blend or 0) > 0
    if cs.engine.torque_split or s_dep:     # a front drive or deploy share needs the front differential
        need["diff.front_type"] = cs.diff.front_type
        if (cs.diff.front_type or 0) >= 4:      # ChassisSim clamps the front type at 4, so a higher type reads the map
            need["diff.front_map"] = cs.diff.front_map
    for c in C:                     # the rubber and its flag matter only when the corner has a gap
        if getattr(cs.susp, f"gap_{c}") not in (None, 0):
            need.update({f"susp.{p}_{c}": getattr(cs.susp, f"{p}_{c}") for p in ("gap_flag", "rubber")})
            if getattr(cs.susp, f"hyst_flag_{c}") == 1:
                need[f"susp.hyst_{c}"] = getattr(cs.susp, f"hyst_{c}")
    for a in "fr":                  # a live third rubber needs its flag
        if 0 < (getattr(cs.susp, f"gap_third_{a}") or 0) < 0.5 and getattr(cs.susp, f"rubber_third_{a}") is not None:
            need[f"susp.gap_flag_third_{a}"] = getattr(cs.susp, f"gap_flag_third_{a}")
    a = cs.aero
    if a.drs_336 == 1:
        need.update({f"aero.drs_{c}": getattr(a, f"drs_{c}") for c in range(337, 342)})
    for n in ("cla", "cda", "ab"):
        if getattr(a, f"sp_on_{n}") == 1:
            need[f"aero.sp_{n}"] = getattr(a, f"sp_{n}")
    missing = sorted(k for k, v in need.items() if v is None)
    if missing:
        front = ""
        if any(k.startswith("diff.front") for k in missing):
            front = (" The front axle gets a drive force (torque split, cid 111, or hybrid deploy share, cids 464 and 465)."
                     " It needs the front differential type (cid 112). A type of 4 or more also needs the front map (cid 114).")
        raise ValueError(f"The car file {cs.path} lacks the fields {missing}. A ChassisSim car needs them.{front}")


# vp keys that the car model reads. The car record wins for t, kt, Rw, f and the tyre, aero and engine data.
CAR_VP_KEYS = frozenset({"mb", "md", "muf", "mur", "wB", "l", "hcg", "I_x", "I_y", "I_z", "Jw", "g"})


def vp_refusal(vp_overrides):
    """Raise ValueError for a vp_overrides key that the car model does not read."""
    bad = sorted(set(vp_overrides or ()) - CAR_VP_KEYS)
    if bad:
        raise ValueError(f"A ChassisSim car does not read the vp_overrides keys {bad}. The car file gives them. "
                         f"The car model reads {sorted(CAR_VP_KEYS)}.")


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
    and Rw is the rear radius (the engine speed reference). The car model reads the axle values from the car record.
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


def axle_kinematics(hp, track, rc_static, travel=TRAVEL, n=21, name="", rc_datum0=None, radius=None):
    """Solve the front view of one double wishbone and fit the curves. hp has shape (6, 3): the rows of the car
    file (lower inner front, lower inner aft, lower outer, upper inner front, upper inner aft, upper outer).
    The inner points of each wishbone are projected to one pivot (the mean of the two). The upright is rigid and
    the contact patch (y = track / 2) is fixed to it. The wheel moves by w, bump positive.
    The ground height of ChassisSim is a reference z that no file field holds. It is found here: the z of the
    ground at which the static roll centre equals rc_static. The result holds the datum, the curves as poly1d in
    w [m]: rc_h [m] above the contact patches and camber [rad]. The camber is the change of the negative camber
    against the body: positive when the top leans in. The result also holds the range lo, hi [m].
    With radius, zg must lie in [LO_z - radius, LO_z]: the contact patch is below the lower outer ball joint by at
    most the tyre radius. If the datum from rc_static fails this check, or rc_static is exactly 0.0 (not set),
    zg = 0 is the fallback. The fallback needs
    rc_datum0 (the stored RC height at the hardpoint datum) to equal the roll-centre height of this geometry at
    zg = 0 within 1 mm. Then the geometry agrees with the file, and the stored height rc_static is not usable."""
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
    ok = lambda z: radius is None or lo[1] - radius <= z <= lo[1]
    # ponytail: a stored height of exactly 0.0 counts as not set (one shipped car). A car with a true roll centre
    # at 0.0 of the chassis frame takes the fallback too. Add a flag when a car needs that height.
    zg = None if rc_static == 0.0 else (rc_static - h0) / k
    if (zg is None or not ok(zg)) and rc_datum0 is not None and abs(h0 - rc_datum0) <= 1e-3:
        zg = 0.0
    if zg is None or not ok(zg):
        raise ValueError(f"The ground datum of the {name} axle is not plausible. The contact patch must lie "
                         "below the lower outer ball joint by at most the tyre radius.")
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
    rc_h itself is the height above the contact patches. The result also holds k1: the camber change of a steered
    front wheel per radian of steer."""
    if getattr(car, "_curves", None) is None:
        g, h = car.geom, car.hardpoints
        need = (g.rc_height_fl, g.rc_height_fr, g.rc_height_rl, g.rc_height_rr, h.front_l, h.rear_l)
        if any(v is None for v in need):
            raise ValueError("The car file lacks the roll-centre heights (cids 55, 59, 63, 67) or the hardpoints (cids 49-52).")
        def datum0(a, b):
            v = (getattr(g, f"rc_datum0_{a}", None), getattr(g, f"rc_datum0_{b}", None))
            return None if None in v else 0.5 * (v[0] + v[1])
        ty, hf = car.tyre, np.asarray(h.front_l, float)
        car._curves = SimpleNamespace(
            f=axle_kinematics(hf, g.track_front, 0.5 * (g.rc_height_fl + g.rc_height_fr), name="front",
                              rc_datum0=datum0("fl", "fr"), radius=ty.fl["radius"] if ty.fl else None),
            r=axle_kinematics(np.asarray(h.rear_l, float), g.track_rear, 0.5 * (g.rc_height_rl + g.rc_height_rr), name="rear",
                              rc_datum0=datum0("rl", "rr"), radius=ty.rl["radius"] if ty.rl else None))
        # K1 = x component of the kingpin unit vector: castor c (tyre scalar), inclination sigma (front view of the
        # line from the lower to the upper outer ball joint). No castor or no front tyre gives 0.
        # ponytail: K1 is the camber rate per rad of kingpin rotation. Per rad of road-wheel yaw, small-angle
        # kinematics give tan(castor) (GP3 0.158, +5 % against the log). K1 fits the GP3 log within 0.7 %.
        # The quadratic term of the kingpin inclination is not in the model (about 11 % of the linear term at 8 deg
        # of steer). Add it when a log of a car with more steer shows the curve.
        castor = ty.fl["castor"] if ty.fl else 0.0
        if not np.isfinite(castor):
            raise ValueError(f"The castor scalar of the front left tyre is not finite ({castor}). Correct the car file.")
        d = hf[5, 1:] - hf[2, 1:]
        tc, ts = np.tan(castor), d[0] / d[1]
        car._curves.k1 = float(tc / np.sqrt(1 + tc ** 2 + ts ** 2))
    return car._curves


def bar_ratios(car, a):
    """Return (left, right): the bar motion ratios of the two wheels of axle a ("f" or "r").
    Each is the first coefficient of its record. A car without the left record uses the right one for both."""
    r = float(np.atleast_1d(getattr(car.susp, f"mr_arb_{a}r"))[0])
    l = getattr(car.susp, f"mr_arb_{a}l")
    return (r if l is None else float(np.atleast_1d(l)[0])), r


def bump_steer(car, c):
    """Return the bump steer table {x: travel [m], y: toe [rad]} of corner c, or None when it is zero.
    If one side of the axle is zero and the other is not, the table of the other side serves both sides.
    A table that is not zero needs finite values and an increasing travel with 2 distinct points. Else ValueError."""
    for k in (c, c[0] + ("r" if c[1] == "l" else "l")):
        t = getattr(car.susp, f"bump_steer_{k}", None)
        if not table_live(t):
            continue
        x, y = np.asarray(t["x"], float), np.asarray(t["y"], float)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.diff(x) >= 0) and x[-1] > x[0]):
            cid = 74 + _C4.index(k)
            raise ValueError(f"The bump steer table of corner {k} (cid {cid}) is not valid. It needs finite values "
                             f"and an increasing wheel travel with at least 2 distinct points.")
        return t
    return None
