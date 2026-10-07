"""Python side of validation gates 1 and 2, what validation/matlab/vexport.m writes for the MATLAB original.
Nothing is solved.

    params(circuit, opt_ds, mesh='uniform') -> dict, same layout as the MATLAB JSON (compare.diff_params);
        the NLP is not even built (only the model, the grid size and the path-constraint rows)
    nlp(circuit, opt_ds, W=None) -> dict lbw ubw lbg ubg w0 F G (compare.nlp_gate); builds the 23-state NLP
        through MLTP at IPOPT max_iter 0 (inside matlab_seed()) and evaluates f and g at the columns of W
    matlab_seed() -> context manager: MATLAB's 23-state start point inside the ``with`` (states 9-22 at OPT_e)

Parity settings (docs/validation_matlab_vs_python.md section 3): MATLAB as shipped = uniform mesh, tol 1e-8,
acceptable_tol 1e-6, MUMPS, monotone mu, Static / ATD On / EM4 Off / CombinedSlip, vi 60, MF205, and the
23-state seed of MLTP.m:229-251 (matlab_seed; production code keeps its quasi-static seed).
"""
import hashlib
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.context import Ctx                       # noqa: E402
from functions.transcription import discretise          # noqa: E402
from MLTP import MLTP, build_path_constraints           # noqa: E402
import MLTP as _mltp                                    # noqa: E402  (the module: the line above binds the function)
from userOpts import userOpts, _REAL_CIRCUITS           # noqa: E402
from vehModel import vehModel                           # noqa: E402

# the parity tier of docs section 3 (the rest of MATLAB as shipped is Python's default or passed by the caller)
PARITY = dict(tol=1e-8, linear_solver="mumps", ipopt_overrides={"acceptable_tol": 1e-6, "mu_strategy": "monotone"},
              circuits_dir=ROOT / "Circuits", data_dir=ROOT / "Data")


@contextmanager
def matlab_seed():
    """MATLAB's as-shipped 23-state start point (MLTP.m:229-251) inside the ``with``: states 9-22 (suspension,
    unsprung, tyre deflection) at OPT_e = 1e-2 instead of Python's quasi-static seed (suspension 0, zt at the static
    deflection); states 5-8 (Om = vx / Rw) are the original's. Swaps the MLTP.quasi_static_states that warmstart_guesses
    looks up at call time and restores it on exit. Validation only: production code keeps its seed."""
    orig = _mltp.quasi_static_states

    def seed(vp, vx):
        out = orig(vp, vx)
        out[4:] = 1e-2                                  # rows 4-17 of the 18 = states 9-22; OPT_e (equal in both codes)
        return out

    _mltp.quasi_static_states = seed
    try:
        yield
    finally:
        _mltp.quasi_static_states = orig


# MATLAB workspace name -> attribute path on ctx (ctx.m23 = the 23-state model, ctx.opts a dict).
# Everything else in the dump is computed in params(): circuit, tire, N, n_w, n_g, h_lb, h_ub, hnames,
# aero_used, track, track_sha256, aero_sha256.
MAP = {
    "AeroConfig": "AeroConfig", "ATD": "ATD", "Electric_4Motors": "Electric_4Motors",
    "TyreModel": "m23.TyreModel", "vi": "vi", "ni": "ni",
    "vp": "vp", "pt": "pt", "mf": "mf", "CG_lin": "cg", "aero": "aero",
    "Xi": "Xi", "Xf": "Xf", "OPT_ds": "OPT_ds", "OPT_d": "OPT_d", "OPT_e": "OPT_e", "OPT_uinter": "OPT_uinter",
    "nx": "m23.nx", "nu": "m23.nu",
    "x_s": "m23.x_s", "u_s": "m23.u_s", "x_min": "m23.x_min", "x_max": "m23.x_max",
    "u_min": "m23.u_min", "u_max": "m23.u_max", "duk_lb": "m23.duk_lb", "duk_ub": "m23.duk_ub",
    "ru": "ru", "rdu": "rdu", "rdu2": "rdu2", "ipopt": "opts.ipopt",
}
TIRE = {"MF205": "MF_205_60R15_V91"}                    # tyre_set -> the `tire` string vehParams.m selects


def _get(obj, path):
    for part in path.split("."):
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    return obj


def _plain(x):
    """Namespaces -> dicts, arrays -> lists, numpy scalars -> Python numbers."""
    if isinstance(x, SimpleNamespace):
        return {k: _plain(v) for k, v in vars(x).items()}
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x.item() if isinstance(x, np.generic) else x


def _aero_used(m, vp):
    """Cl per corner and Cd as the Python model's own force expressions use them (vx = 1 m/s,
    f_lift_i = -q Cl_i, f_drag = q Cd): the counterpart of vp.Cl_fl ... vp.Cd after vehModel.m.
    Static aero only is meaningful: with active-aero inputs (zero here) Python's wings are live, MATLAB's dead."""
    q = 0.5 * vp.rho * vp.A
    f = ca.Function("aero", [m.x, m.u], [m.f_lift_fl, m.f_lift_fr, m.f_lift_rl, m.f_lift_rr, m.f_drag])
    fl, fr, rl, rr, drag = (float(v) for v in f(np.r_[1.0 / m.x_s[0], np.zeros(m.nx - 1)], np.zeros(m.nu)))
    return dict(Cl_fl=-fl / q, Cl_fr=-fr / q, Cl_rl=-rl / q, Cl_rr=-rr / q, Cd=drag / q)


def params(circuit, opt_ds, mesh="uniform"):
    ctx = Ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        userOpts(ctx, circuit=circuit, OPT_ds=opt_ds, mesh=mesh, **PARITY)
    vehModel(ctx)
    m = ctx.m23
    hnames, _, h_lb, h_ub = build_path_constraints(ca, m, ctx.pt)
    N = discretise(ctx.track, ctx.OPT_ds, ctx.OPT_d, mesh=ctx.mesh)["N"]
    d, nh = ctx.OPT_d, len(hnames)

    out = {k: _plain(_get(ctx, v)) for k, v in MAP.items()}
    out.update(
        circuit=ctx.circuit, tire=TIRE[ctx.tyre_set], N=N,
        n_w=(m.nx + m.nu) * (N + 1) + m.nx * N * d,
        n_g=2 * m.nx + (d + 1) * N * m.nx + (N + 1) * nh + N * m.nu,
        h_lb=_plain(h_lb), h_ub=_plain(h_ub), hnames=hnames,
        aero_used=_aero_used(m, ctx.vp),
        track=dict(n=int(ctx.track.s.size), s_end=float(ctx.track.s[-1]),
                   k_abssum=float(np.sum(np.abs(ctx.track.k))), k_sumsq=float(np.sum(ctx.track.k ** 2))),
        track_sha256=_sha256(ROOT / "Circuits" / _REAL_CIRCUITS[circuit]) if circuit in _REAL_CIRCUITS else None,
        aero_sha256=_sha256(ROOT / "Data" / "DATA_AA.mat"))
    return out


def nlp(circuit, opt_ds, W=None):
    """Gate 2: build the 23-state NLP of the parity call through MLTP at IPOPT max_iter 0 (the 7-state init and the
    23-state solve both stop before iteration 1; MLTP_KEEP_NLP=1 keeps f, g and the bounds) and evaluate f and g
    at the columns of W (n_w x k; default: the start point w0 alone). The start point is MATLAB's (matlab_seed).
    Returns dict(lbw, ubw, lbg, ubg, w0 as 1-D arrays, F (k,), G (n_g, k))."""
    os.environ["MLTP_KEEP_NLP"] = "1"
    try:
        with matlab_seed():
            ctx = MLTP(circuit, OPT_ds=opt_ds, mesh="uniform", max_iter=0, save=False, plot=False, warm_start=None,
                       **PARITY)
    finally:
        os.environ.pop("MLTP_KEEP_NLP", None)
    b = ctx.nlp_bounds
    W = b["w0"].reshape(-1, 1) if W is None else np.asarray(W, dtype=float)
    W = W.reshape(W.shape[0], -1)
    if W.shape[0] != b["w0"].size:
        raise ValueError(f"W has {W.shape[0]} rows, the Python NLP has n_w = {b['w0'].size} (N or settings differ)")
    FG = [ctx.nlp_fg(w) for w in W.T]
    return dict(b, F=np.array([float(f) for f, _ in FG]), G=np.column_stack([g.full().ravel() for _, g in FG]))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
