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


def load_solution(path):
    """Reload a saved MLTP/MLTP_initial solution .mat and return the ``data``
    struct as a namespace (data.track.xopt, data.vehicle.fx_fl, data.x_full, ...).
    Works for both full-solve files ({'data': {...}}) and warm-start files
    ({'data': {'init': {...}}}). A full result's primal/dual NLP record comes
    back as data.nlp with 1-D float arrays (w_opt, lam_g, lam_x, x_s, u_s),
    even where loadmat squeezed a length-1 vector to a scalar."""
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
    return data


def result_stem(circuit, cfg, tyre_set="MF205", mesh_requested="auto"):
    """File stem (no extension) of a saved MLTP result; casadi-free, so MLTP and
    the GUI build the same name.

    Rule: ``<circuit>_<cfg>``, plus ``_<tyre_set>`` when tyre_set != "MF205"
    (e.g. ``_CopyB``), plus ``_mesh<Mesh>`` when mesh_requested is neither None
    nor "auto" (e.g. ``_meshCurvature``; the mesh asked for, not the one "auto"
    resolved to). Default runs (MF205, mesh "auto") keep the old stem
    ``<circuit>_<cfg>``; any other tyre set / mesh gets its own file instead of
    overwriting the default one."""
    stem = f"{circuit}_{cfg}"
    if tyre_set != "MF205":
        stem += f"_{tyre_set}"
    if mesh_requested is not None and mesh_requested != "auto":
        stem += f"_mesh{mesh_requested.capitalize()}"
    return stem
