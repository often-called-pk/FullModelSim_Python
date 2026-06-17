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
