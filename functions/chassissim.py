"""ChassisSim v3.41 input files: readers and the circuit source.

The install folder comes from the argument or from the environment variable CHASSISSIM_DIR.
A circuit name for userOpts has the form "chassissim:<Circuit name>".
The readers use only the standard library and numpy.
"""
import configparser
import os
from types import SimpleNamespace

import numpy as np

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
                raise ValueError(f"{path}: expected {ncol} columns, found {len(t)}.")
            out.append([float(x) for x in t])
    return out


def _increasing(path, s):
    if len(s) < 3 or np.any(np.diff(s) <= 0):
        raise ValueError(f"{path}: need 3 or more rows with strictly increasing distance.")


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
    if not rows or any(len(r) != 3 for r in rows[1:]):
        raise ValueError(f"{path}: expected a base altitude line, then rows with 3 columns.")
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
    """Load a circuit by its ChassisSim name. Return SimpleNamespace(s, k, grip, bump, alt).

    k uses the model sign (left turn positive), so it is the file value with the sign flipped.
    bump and alt are file paths, or None when the circuit has no such file or the flag is off.
    """
    circ = os.path.join(cs_dir, "Circuits")
    ini = _find(cs_dir, "Circ_prop.ini")
    blocks = read_circ_prop(ini) if ini else []
    want = name.strip().lower()
    blk = next((b for b in blocks if b.get("Circuit name", "").strip().lower() == want), None)
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
        tr = _find(circ, f"tr_{name}.txt")
        if tr is None:
            orphans = [f[3:-4] for f in os.listdir(circ) if f.lower().startswith("tr_") and f.lower().endswith(".txt")]
            known = sorted({b.get("Circuit name", "") for b in blocks} | set(orphans))
            raise ValueError(f"Unknown ChassisSim circuit '{name}'. Known circuits: {', '.join(known)}.")
        grip = 1.0
        bump, alt = _find(circ, f"bp_{name}.dat"), _find(circ, f"alt_rcamb_{name}.txt")
    s, k = read_curvature(tr)
    return SimpleNamespace(s=s, k=-k, grip=grip, bump=bump, alt=alt)
