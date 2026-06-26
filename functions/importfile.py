"""importfile.py - port of Functions/importfile.m

MATLAB injects every .mat field into the base workspace; Python has none, so this
returns {var_name: value} with structs as SimpleNamespace to keep dot syntax
(e.g. data.x_opt, track.s).

    vars = importfile('Data/Initialisation/Static_MidDF.mat')
    ctx.data = vars['data']
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
    """Reload a saved MLTP solution .mat and return the ``data`` namespace.
    Handles full-solve files ({'data': {...}}) and warm-start files
    ({'data': {'init': {...}}})."""
    loaded = importfile(path)
    data = loaded["data"]
    if hasattr(data, "init") and not hasattr(data, "x_opt"):
        return data.init
    return data
