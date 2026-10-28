"""importfile.py - port of Functions/importfile.m

MATLAB original loads a ``-mat`` file and injects every field into the base
workspace via ``assignin('base', ...)``. Python has no base workspace, so this
version returns a dict mapping each saved variable name to a Python value, with
MATLAB structs converted recursively to ``types.SimpleNamespace`` so that field
access keeps the MATLAB dot syntax (e.g. ``data.x_opt``, ``track.s``).

Typical use mirrors the MATLAB ``importfile('Static_MidDF.mat'); data.init = data;``:

    vars = importfile('Data/Initialisation/Static_MidDF.mat')
    ctx.data = vars['data']            # the saved result struct
"""

import re

import numpy as np
import scipy.io as sio
from types import SimpleNamespace

try:                                                    # scipy layout differs by version
    from scipy.io.matlab import mat_struct
except Exception:                                       # pragma: no cover
    from scipy.io.matlab.mio5_params import mat_struct


def mat_to_namespace(obj):
    """Recursively convert loadmat output (mat_struct / object arrays) to
    SimpleNamespace / numpy arrays so attribute access works like MATLAB."""
    if isinstance(obj, mat_struct):
        return SimpleNamespace(**{f: mat_to_namespace(getattr(obj, f))
                                  for f in obj._fieldnames})
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.array([mat_to_namespace(o) for o in obj.ravel()],
                        dtype=object).reshape(obj.shape)
    return obj


def importfile(file_to_read):
    """Load a .mat file and return {var_name: value} with structs as namespaces."""
    raw = sio.loadmat(file_to_read, squeeze_me=True, struct_as_record=False)
    return {k: mat_to_namespace(v) for k, v in raw.items()
            if not k.startswith("__")}


def read_tir(path):
    """Read a tyre property file (.tir). Return {NAME: float}, or {NAME: str} for a text value.
    Names are upper case. A section line such as [MODEL] adds no key. A '$' or '!' starts a comment.
    A D exponent reads as E. The function reads one line at a time."""
    out = {}
    with open(path, encoding="ascii", errors="replace") as fh:
        for line in fh:
            line = re.split(r"[$!]", line, maxsplit=1)[0]
            if "=" not in line:
                continue
            key, val = (s.strip() for s in line.split("=", 1))
            val = val.strip("'\"")
            try:
                out[key.upper()] = float(re.sub("[dD]", "E", val))
            except ValueError:
                out[key.upper()] = val
    return out


def load_solution(path):
    """Reload a saved MLTP/MLTP_initial solution .mat and return the ``data``
    struct as a namespace (data.track.xopt, data.vehicle.fx_fl, data.x_full, ...).
    Works for both full-solve files ({'data': {...}}) and warm-start files
    ({'data': {'init': {...}}}). A full result's primal/dual NLP record comes
    back as data.nlp with 1-D float arrays (w_opt, lam_g, lam_x, x_s, u_s),
    even where loadmat squeezed a length-1 vector to a scalar; the same holds
    for the per-attempt arrays of a refinement record (data.refine: N, lap_time,
    ..., s_knot0, eta), which a single-attempt run would otherwise reload as
    scalars."""
    loaded = importfile(path)
    data = loaded["data"]
    if hasattr(data, "init") and not hasattr(data, "x_opt"):
        return data.init
    nlp = getattr(data, "nlp", None)
    if nlp is not None:
        for key in ("w_opt", "lam_g", "lam_x", "x_s", "u_s"):
            if hasattr(nlp, key):
                setattr(nlp, key, np.atleast_1d(np.asarray(getattr(nlp, key),
                                                           dtype=float)).reshape(-1))
    ref = getattr(data, "refine", None)
    if isinstance(ref, SimpleNamespace):
        from .refine import RECORD_COLUMNS          # numpy only; lazy to keep this module light
        for key in RECORD_COLUMNS + ("s_knot0", "eta"):
            if hasattr(ref, key):
                setattr(ref, key, np.atleast_1d(np.asarray(getattr(ref, key))).reshape(-1))
    return data


def result_stem(circuit, cfg, tyre_set="MF205", mesh_requested="auto", physics="matlab",
                start_state="matlab", n_tag="", setup_tag=""):
    """File stem (no extension) of a saved MLTP result; casadi-free, so MLTP and
    the GUI build the same name.

    Rule: ``<circuit>_<cfg>``, plus ``_<tyre_set>`` when tyre_set != "MF205"
    (e.g. ``_CopyB``, or ``_tir_<name>`` for ``file:<dir>/<name>.tir``: the base name only, so userOpts.setup_knobs
    adds the full path as the knob ``tyre_file``), plus ``_physCorrected`` when physics == "corrected", plus
    ``_start<State>`` when start_state != "matlab" (``_startRolling``,
    ``_startPeriodic``), plus ``_<n_tag>`` when n_tag is not empty (userOpts
    ``ctx.n_tag``: ``_hw3p5`` for n_bounds=3.5, ``_nbCustom`` for a (lo, hi) pair),
    plus ``_mesh<Mesh>`` when mesh_requested is neither None
    nor "auto" (e.g. ``_meshCurvature``; the mesh asked for, not the one "auto"
    resolved to), plus ``_<setup_tag>`` last when setup_tag is not empty
    (userOpts.knob_tag: 8 hex digits for the other setup knobs that differ from
    their defaults, such as vp_overrides, vi or OPT_ds, listed in data.provenance.setup).
    Default runs (MF205, mesh "auto", physics "matlab", start_state "matlab", no
    n_bounds, no other knob) keep the old stem ``<circuit>_<cfg>``. Any other tyre set /
    mesh / physics / start state / lateral band / knob gets its own file instead of
    overwriting the default one."""
    stem = f"{circuit}_{cfg}"
    if tyre_set != "MF205":
        if tyre_set.startswith("file:"):
            tyre_set = "tir_" + re.sub(r"\W+", "_", re.split(r"[\\/]", tyre_set[5:])[-1].rsplit(".", 1)[0])
        stem += f"_{tyre_set}"
    if physics == "corrected":
        stem += "_physCorrected"
    if start_state != "matlab":
        stem += f"_start{start_state.capitalize()}"
    if n_tag:
        stem += f"_{n_tag}"
    if mesh_requested is not None and mesh_requested != "auto":
        stem += f"_mesh{mesh_requested.capitalize()}"
    if setup_tag:
        stem += f"_{setup_tag}"
    return stem
