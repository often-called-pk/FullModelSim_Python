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
