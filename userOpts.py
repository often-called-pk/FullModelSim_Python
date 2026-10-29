"""userOpts.py - direct port of userOpts.m (FullModel_4EM_Suspension_FullTyre)

Defines the user options for the optimisation problem and the collocation method,
after loading the powertrain and vehicle parameters. Everything is written onto
the shared ``ctx``.

Configuration is exposed as function arguments with the same defaults as the
MATLAB file (AeroConfig='Static', ATD='On', Electric_4Motors='Off', circuit='BCN',
vi=60, ni=nan). Edit the call (or the defaults) the way you would edit userOpts.m.

Linear solver: IPOPT defaults to HSL ``ma57`` (Coin-HSL, MC64 auto-scaling) and
transparently falls back to MUMPS, which is bundled inside the casadi wheel and
needs no external library, when no working HSL library is found, so a solve runs
out of the box on any platform.
"""

import os
import functools
import inspect
import json
import numbers
import subprocess
import warnings
import numpy as np
import scipy.io as sio
from types import SimpleNamespace

from Powertrain import Powertrain
from vehParams import vehParams, all_vp_defaults, PRIMARY_KEYS, MF_KEYS, PT_KEYS
from functions.simpleMA import simpleMA
from functions.importfile import mat_to_namespace, n_tag
from functions.mesh import curvature_mesh, mesh_opts_record
from functions.sweep import fingerprint
from functions.track_qa import check_track


# ----------------------------------------------------------------------------- 
# Track definitions
# -----------------------------------------------------------------------------
_REAL_CIRCUITS = {
    "BCN":        "Barcelona_circuit.mat",
    "BCN_S1":     "Barcelona_circuit_s1.mat",
    "BCN_S2":     "Barcelona_circuit_s2.mat",
    "BCN_S3":     "Barcelona_circuit_s3.mat",
    "Jarama":     "Jarama_circuit.mat",
    "Spa":        "Spa_circuit.mat",
    "NBR":        "Nurburgring_circuit.mat",
}


def _synthetic_curvature(circuit):
    """Return the raw curvature array k for the synthetic tracks (pre-smoothing)."""
    pi = np.pi
    if circuit == "Straight":
        return np.zeros(300)
    if circuit == "Hairpin":
        return np.concatenate([np.zeros(75), 0.0975 * np.ones(16), np.zeros(73)])
    if circuit == "Sturn":
        return np.concatenate([np.zeros(100), (pi / 45) * np.ones(20), np.zeros(30),
                               -(pi / 45) * np.ones(20), np.zeros(100)])
    if circuit == "Circle":
        return (2 * pi / 1000) * np.ones(500)
    if circuit == "ZigZag":
        return np.concatenate([np.zeros(75), 0.0975 * np.ones(16), np.zeros(146),
                               -0.0975 * np.ones(16), np.zeros(146),
                               (2 * pi / 1000) * np.ones(250), np.zeros(146),
                               -(2 * pi / 1000) * np.ones(250), np.zeros(146)])
    if circuit == "ZigZagMirror":
        return np.concatenate([np.zeros(75), -0.0975 * np.ones(16), np.zeros(146),
                               0.0975 * np.ones(16), np.zeros(146),
                               -(2 * pi / 1000) * np.ones(250), np.zeros(146),
                               (2 * pi / 1000) * np.ones(250), np.zeros(146)])
    if circuit == "VirtualTrack":
        ramp = np.arange(1.0, 6.0 + 1e-9, 0.05)         # MATLAB 1:0.05:6
        return np.concatenate([np.zeros(150), (pi / 25) * np.ones(20), np.zeros(50),
                               -(pi / 125) * np.ones(90), -(pi / 50) * np.ones(35),
                               (pi / 50) * np.ones(35), -(pi / 500) * ramp,
                               np.zeros(200), -(pi / 20) * np.ones(20), np.zeros(80)])
    return None


def _load_track(circuit, circuits_dir):
    """Build (synthetic) or load (real) the track as a namespace with .s, .k[, .x, .y]."""
    if circuit in _REAL_CIRCUITS:
        path = os.path.join(circuits_dir, _REAL_CIRCUITS[circuit])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Track file '{path}' not found. Place the circuit .mat in "
                f"'{circuits_dir}/'.")
        raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        # The .mat may store the fields directly or under a 'track' struct.
        if "track" in raw:
            track = mat_to_namespace(raw["track"])
        else:
            track = SimpleNamespace(**{k: mat_to_namespace(v)
                                       for k, v in raw.items()
                                       if not k.startswith("__")})
        if not (hasattr(track, "s") and hasattr(track, "k")):
            raise ValueError(f"Track file '{path}' has no s/k field (found: {sorted(vars(track))})")
        track.s = np.asarray(track.s, dtype=float).reshape(-1)
        track.k = np.asarray(track.k, dtype=float).reshape(-1)
        if hasattr(track, "x"):
            track.x = np.asarray(track.x, dtype=float).reshape(-1)
        if hasattr(track, "y"):
            track.y = np.asarray(track.y, dtype=float).reshape(-1)
        _, issues = check_track(track)
        errors = [text for level, text in issues if level == "error"]
        if errors:
            raise ValueError(f"Track '{circuit}' ({path}) fails the track QA: {'; '.join(errors)}. "
                             "Rebuild k from x, y with functions.track_qa.k_from_xy.")
        if issues:
            warnings.warn(f"Track '{circuit}' QA: {'; '.join(text for _, text in issues)}")
        return track

    k = _synthetic_curvature(circuit)
    if k is None:
        raise ValueError(f"Unknown circuit '{circuit}'.")
    k = simpleMA(k, 10, 2)                                # smoothen curvature
    s = np.linspace(0, 2 * len(k), len(k))               # ~2 m spacing assumed
    return SimpleNamespace(s=s, k=k)


# IPOPT "screening" tolerance preset (userOpts(screening=True)): a looser
# stopping rule for ranking / sweep solves, not a guaranteed saving. Measured on
# Sturn (N=18, ma57): iteration counts moved by -41 % to +16 % across five matched
# pairs (196 vs 257 with the 5-coefficient MF205 override, 290 vs 489 with the
# full tyre_set='MF205') for lap-time changes of -4 to +20 ms; iteration counts on
# this NLP swing up to +-40 % under 1-ulp perturbations, so judge it per track.
# Not for final results. Applied over the defaults and under ipopt_overrides.
SCREENING_IPOPT = {
    "tol": 1e-3,
    "acceptable_tol": 1e-2,
    "dual_inf_tol": 1e-2,
    "constr_viol_tol": 1e-3,
    "compl_inf_tol": 1e-3,
}

# mesh='auto' (the userOpts default) picks the collocation knots from the lap
# length L = s[-1] - s[0]: L >= MESH_AUTO_MIN_LENGTH metres gets the
# curvature-weighted knots (BCN-size real circuits), a shorter lap stays uniform
# (the short synthetic tracks; ZigZag / ZigZagMirror, ~2.4 km, are above it).
# 'auto' only redistributes the knots; N stays round(L/OPT_ds). Measured: at
# equal N the lap error vs a fine reference drops (Sturn N=18: +0.72% -> +0.16%);
# the 2.5x BCN speedup (curvature N=103 vs uniform N=155, same lap) needs
# OPT_ds=45 together with the curvature mesh.
MESH_AUTO_MIN_LENGTH = 2000.0

# Accepted userOpts(mesh_opts=...) keys: the curvature_mesh keyword arguments
# (discretise() passes s, k, OPT_ds and the speed profile v, so they are not mesh_opts keys).
_MESH_OPTS_KEYS = tuple(n for n in inspect.signature(curvature_mesh).parameters
                        if n not in ("s", "k", "OPT_ds", "v"))

# userOpts(physics=...): 'matlab' (default) keeps the inherited MATLAB vehicle
# physics bit for bit. 'corrected' is the opt-in set of fixes A1-A6
# (docs/laptime_sim_improvements.md, section A). The check is case-sensitive.
PHYSICS = ("matlab", "corrected")

# userOpts(Steering=..., CamberGain=...): the two vehModel switches that MATLAB hard-codes
# ('NA' and 'Off'). vehModel reads ctx.Steering / ctx.CamberGain when its own argument is None.
STEERING = ("NA", "Ackermann")
CAMBER_GAIN = ("Off", "Linear", "Table")

# userOpts(start_state=...) sets how the lap starts. The check is case-sensitive.
#   'matlab'   (default) the MATLAB start box: vx = vi, vy = 0, heave / pitch / roll = 0,
#              the rest free.
#   'rolling'  r = 0 and a narrow box (ROLL_TOL) around the straight-line coast
#              equilibrium on rows 9-22. MLTP fills the rows from the model.
#   'periodic' free start. The end state and input equal the start (flying lap).
#              Closed circuits only (CLOSED_CIRCUITS).
START_STATES = ("matlab", "rolling", "periodic")
ROLL_TOL = 1e-3  # ponytail: fixed chassis start-box half-width, promote to a kwarg when tuning matters
CLOSED_CIRCUITS = ("BCN", "Jarama", "Spa", "NBR", "Circle")

# userOpts(between_knots=...) sets what the 23-state NLP does between two knots (B3). The check is case-sensitive.
#   'matlab'       (default) MATLAB's arithmetic. The dynamics at the collocation points see U_k + duk * tau.
#                  duk is per metre, so the input is a near hold of U_k. The path rows apply at the knots only.
#   'linear'       the dynamics see the linear input U_k + (U_k+1 - U_k) * tau. Same rows, same variables.
#   'linear+power' 'linear', plus the motor power rows (and the pack row) at the N * OPT_d collocation points.
#                  EM4 under physics='matlab' gets no such row, so there it equals 'linear' (n_pathcol 0).
# The power rows need the linear input. With the near hold they would derate U_k by the speed at the end of the interval.
# The 7-state init keeps the MATLAB arithmetic in every mode.
BETWEEN_KNOTS = ("matlab", "linear", "linear+power")

# userOpts(collocation=...) sets the collocation points of the 23-state NLP (B4). The check is case-sensitive.
#   'legendre' (default) Gauss-Legendre points. The NLP is as MATLAB builds it.
#   'radau'    Radau IIA points. The last point of each interval is the next knot, so s_full repeats every interior
#              knot. The scheme damps the fast modes that Gauss-Legendre lets swing from knot to knot.
#              The NLP has the same variables and rows. With between_knots='linear+power', the point on the next knot
#              gets no power row, so g has N rows fewer per power row.
#              Its order is 2 OPT_d - 1, against 2 OPT_d for Gauss-Legendre. It needs OPT_d >= 2 (ValueError).
# The 7-state init always uses 'legendre'.
COLLOCATION = ("legendre", "radau")

# userOpts(state_scaling=...) sets the scales of the 14 chassis states, rows 9-22 (B7). The check is case-sensitive.
#   'matlab'  (default) scale 1.0 for all 14 rows, as vehModel.m.
#   'typical' TYPICAL_CHASSIS_SCALE: a typical size per row (zs 10 mm, wu 0.3 m/s, zt 20 mm and so on).
#             Each value rounds the RMS size of that state in the stored optima.
#             The physical problem is the same. The start box keeps its physical width.
#             The units of w, g and the multipliers change. A saved w or multiplier of the other setting does not apply.
#             IPOPT applies constr_viol_tol and dual_inf_tol in these units. So the stop rule is up to 100 times
#             tighter on the chassis defects and looser on the dual infeasibility of the chassis states.
# The 7-state init ignores it.
STATE_SCALINGS = ("matlab", "typical")
TYPICAL_CHASSIS_SCALE = np.array([1e-2, 1e-1, 1e-2, 1e-1, 1e-2, 1e-1] + [3e-1] * 4 + [2e-2] * 4)


def _parse_n_bounds(n_bounds, s):
    """userOpts(n_bounds=...) -> (lo, hi, tag): float arrays on the track stations ``s`` [m]
    (n > 0 = left of the centreline) and the result-stem tag. None -> (None, None, "").

      None      the model's own +-4 m bound, nothing per point.
      W         a real number > 0 (not a bool): the band (-W, W), tag "hw<W>" (3.5 -> "hw3p5").
      (lo, hi)  a tuple or list of two entries, each a number or an array of len(ctx.track.s)
                values, tag "nbCustom". The arrays live on ctx.track.s as it is after the
                periodic closing sample (periodic Spa 1402 values, the file 1401).

    The band bounds the CG path n at the knots and the collocation points only (not between
    them), and replaces the model's +-4 m there (it may be wider). lo < hi must hold at every point, so a fixed line
    is a narrow band (n_ref - b, n_ref + b) around it, the correlation mode. That band is
    continuity-feasible when b >= max |continuity mismatch of n_ref| / 4.67 (Legendre d = 3);
    a 3 m, 100 m wavelength line at OPT_ds 30 has a mismatch of 0.078 m, so b >= 0.017 m.
    """
    if n_bounds is None:
        return None, None, ""
    if isinstance(n_bounds, numbers.Real) and not isinstance(n_bounds, (bool, np.bool_)):
        W = float(n_bounds)
        if not (np.isfinite(W) and W > 0):
            raise ValueError(f"n_bounds half-width W must be finite and > 0 m, got {n_bounds!r}")
        return np.full(s.size, -W), np.full(s.size, W), n_tag(W)
    if not (isinstance(n_bounds, (tuple, list)) and len(n_bounds) == 2):
        raise ValueError("n_bounds must be None, a number W > 0 (the band -W..W in m) or a (lo, hi) "
                         f"tuple or list of two entries, got {n_bounds!r:.80}")
    try:
        lo, hi = (np.broadcast_to(np.asarray(v, float), s.shape).copy() for v in n_bounds)
    except (TypeError, ValueError):
        raise ValueError("n_bounds lo and hi must each be a number or an array of "
                         f"len(ctx.track.s) = {s.size} values") from None
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("n_bounds lo and hi must be finite")
    if not (lo < hi).all():
        raise ValueError("n_bounds needs lo < hi at every track point; a fixed line is a narrow "
                         "band (n_ref - b, n_ref + b)")
    return lo, hi, n_tag(n_bounds)


# -----------------------------------------------------------------------------
# Setup identity (E2): which arguments name a result file
# -----------------------------------------------------------------------------
# A setup knob that differs from its default adds 8 hex digits to the result file name (knob_tag) and appears
# in data['provenance']['setup']. Every userOpts argument is a setup knob, so a new argument is hashed at once.
# These arguments are not: the file stem has them already, or they change only how the same NLP is solved
# or where files live (ipopt_overrides goes to provenance['run']). TyreModel (an MLTP argument) is a knob too,
# and so are load_model and ds_fine (MLTP_screen arguments).
_NOT_SETUP = frozenset({"circuit", "AeroConfig", "ATD", "Electric_4Motors", "tyre_set", "physics", "start_state",
                        "mesh", "linear_solver", "hsl_dir", "jit", "cse", "circuits_dir", "data_dir",
                        "ipopt_overrides"})


def _canon(v):
    """Copy of a knob value for the hash: numbers become float (45 and 45.0 give one tag), None leaves a dict."""
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items() if x is not None}
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)):
        return float(v)
    return v


def _is_default(v, d):
    if isinstance(v, numbers.Real) and isinstance(d, numbers.Real):
        return bool((np.isnan(v) and np.isnan(d)) or np.isclose(v, d, rtol=1e-12, atol=0.0))
    return (_canon(v) or None) == (_canon(d) or None)       # None, {} and [] are the same


def setup_knobs(kw):
    """{name: value} of the setup knobs in ``kw`` that differ from their defaults, {} for the default setup.

    ``kw`` holds the keyword arguments of a call: MLTP passes its userOpts kwargs plus vi, ni and TyreModel, the
    GUI passes headless_solve.build_solve_kwargs(cfg), MLTP_screen adds load_model and ds_fine. Only userOpts
    arguments, TyreModel, load_model and ds_fine are read, so save, plot, warm_start, ladder and the like never
    count. vp_overrides keeps the A5 keys and the entries that differ from the default of the tyre set. A (lo, hi)
    n_bounds pair is hashed by value (the stem says only 'nbCustom'). A number W is in the stem (hw<W>), so it is
    not a knob. A file tyre set (C5) adds the knob ``tyre_file``, the absolute path. The stem has the base name
    only, so the tag tells two files with one name apart."""
    defaults = {k: p.default for k, p in inspect.signature(userOpts).parameters.items()
                if p.default is not p.empty and k not in _NOT_SETUP}
    defaults.update(TyreModel="CombinedSlip", load_model="vehModel", ds_fine=1.0)
    out = {}
    for k, d in defaults.items():
        v = kw.get(k, d)
        if k == "vp_overrides":
            base = all_vp_defaults(kw.get("tyre_set", "MF205"))
            v = {a: float(b) for a, b in dict(v or {}).items() if a in PT_KEYS or not _is_default(b, base.get(a))}
            keep = bool(v)
        elif k == "n_bounds":
            keep = isinstance(v, (tuple, list))
            v = fingerprint(_canon(v))[:12] if keep else None
        else:
            keep = not _is_default(v, d)
        if keep:
            out[k] = v
    ts = kw.get("tyre_set", "MF205")
    if isinstance(ts, str) and ts.startswith("file:"):
        out["tyre_file"] = os.path.abspath(ts[5:])
    return out


def knob_tag(knobs):
    """8 hex digits that name a setup, '' for the default setup (a default result keeps its file name)."""
    return fingerprint(_canon(knobs))[:8] if knobs else ""


@functools.lru_cache(maxsize=1)
def code_version():
    """The output of ``git describe --always --dirty`` in the checkout of this file, or 'unknown' without
    git (a frozen app). Never raises. On Windows the child gets no console window (a windowed app)."""
    try:
        r = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=5,
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def provenance_record(knobs, ctx=None, **run):
    """data['provenance'] (savemat-safe): the code version, the setup knobs and how the solve ran (None is dropped).
    run['ipopt_overrides'] is JSON text. Some IPOPT option names have more than 31 characters, and savemat does
    not accept a longer struct field name. With ``ctx`` it adds ``params``: every vehicle, Magic Formula and
    powertrain value of the built model."""
    flat = lambda d: {k: mesh_opts_record(v) if isinstance(v, dict) else v for k, v in d.items() if v is not None}
    if isinstance(run.get("ipopt_overrides"), dict):
        run["ipopt_overrides"] = json.dumps(run["ipopt_overrides"], sort_keys=True, default=repr)
    rec = {"code": code_version(), "setup": flat(knobs), "run": flat(run)}
    if ctx is not None:
        rec["params"] = {**{k: float(getattr(ctx.vp, k)) for k in sorted(PRIMARY_KEYS)},
                         **{k: float(getattr(ctx.mf, k)) for k in sorted(MF_KEYS)},
                         **{k: float(getattr(ctx.pt, k)) for k in ("Pmax", "Tmax", "OMmax", "Vmax", "eff", "Preg", "Ppack")
                            if hasattr(ctx.pt, k)}, "gear": float(ctx.vp.gear)}
    return rec


# ----------------------------------------------------------------------------- 
# Rate-limit / regularisation table (the MATLAB 'c' struct)
# -----------------------------------------------------------------------------
def _build_c():
    c = SimpleNamespace(ub=SimpleNamespace(), lb=SimpleNamespace(),
                        ru=SimpleNamespace(), rdu=SimpleNamespace(),
                        rdu2=SimpleNamespace())

    # Rate limits  (per second): deal(ub, lb)
    rate = {
        "T_motor":    (2e4, -2e4),
        "T_motor_fl": (2e4, -2e4),
        "T_motor_fr": (2e4, -2e4),
        "T_motor_rl": (2e4, -2e4),
        "T_motor_rr": (2e4, -2e4),
        "T_brake":    (1.45e5, -1.45e5),
        "ATD":        (2e4, -2e4),
        "delta":      (0.1, -0.1),
        "FW":         (20.0, -20.0),
        "RW":         (60.0, -60.0),
        "TW":         (24.0, -24.0),
    }
    for k, (ub, lb) in rate.items():
        setattr(c.ub, k, ub)
        setattr(c.lb, k, lb)

    # Regularisation: deal(ru, rdu, rdu2)
    reg = {
        "T_motor":    (0.0, 0.0, 0.0),
        "T_motor_fl": (0.0, 0.0, 0.0),
        "T_motor_fr": (0.0, 0.0, 0.0),
        "T_motor_rl": (0.0, 0.0, 0.0),
        "T_motor_rr": (0.0, 0.0, 0.0),
        "T_brake":    (0.0, 0.0, 0.0),
        "delta":      (0.0, 8.0, 15.0),
        "ATD":        (0.0, 0.0, 1.5),
        "FW":         (0.0, 0.0, 0.3),
        "RW":         (0.0, 0.0, 0.9),
        "TW":         (0.005, 0.0, 0.4),
    }
    for k, (ru, rdu, rdu2) in reg.items():
        setattr(c.ru, k, ru)
        setattr(c.rdu, k, rdu)
        setattr(c.rdu2, k, rdu2)

    return c


def _input_keys(EM4, ATD, ActAero):
    """Ordered list of input channels for a given configuration.

    Reproduces exactly the explicit if/elseif ladder in userOpts.m that builds
    duk_ub/duk_lb/ru/rdu/rdu2. Order: motor(s) -> brake -> [ATD x4] -> aero -> delta.
    """
    keys = (["T_motor_fl", "T_motor_fr", "T_motor_rl", "T_motor_rr"]
            if EM4 == 1 else ["T_motor"])
    keys = keys + ["T_brake"]
    if ATD == 1:
        keys += ["ATD", "ATD", "ATD", "ATD"]
    if ActAero == 1:        # active RW
        keys += ["RW"]
    elif ActAero == 2:      # active FW + RW
        keys += ["FW", "RW"]
    elif ActAero == 3:      # AALB: split FW (x2) + RW + TW
        keys += ["FW", "FW", "RW", "TW"]
    keys += ["delta"]
    return keys


def _col(ns_group, keys):
    """Column vector of attribute values for the given keys."""
    return np.array([[getattr(ns_group, k)] for k in keys], dtype=float)


# ----------------------------------------------------------------------------- 
# Main entry point
# -----------------------------------------------------------------------------
def userOpts(ctx,
             AeroConfig="Static",          # 'Static' | 'Active_RW' | 'Active' | 'AALB'
             ATD="On",                     # 'On' | 'Off'
             Electric_4Motors="Off",       # 'On' | 'Off'
             circuit="BCN",
             vi=60.0,                      # initial velocity [m/s]
             ni=np.nan,                    # initial lateral position [m]
             circuits_dir="Circuits",
             data_dir="Data",
             linear_solver="ma57",         # 'ma57'|'ma97'|'ma27'|'mumps'; ma* uses Coin-HSL
             hsl_dir=None,                 # Coin-HSL bin dir; None -> COINHSL_DIR env / default
             vp_overrides=None,            # dict of vehParams primary/mf overrides (+ PT_KEYS if physics='corrected')
             tyre_set="MF205",             # 'MF205' (default, MATLAB-run lateral set) | 'CopyB' (legacy shipped set, pKy4=0) | 'file:<path to a .tir>'
             OPT_ds=30,                    # collocation step (m)
             OPT_d=3,                      # degree of interpolating polynomials
             OPT_e=1e-2,                   # slack for path constraints / guesses
             mesh="auto",                  # 'auto' | 'uniform' | 'curvature' collocation knots (auto: by lap length)
             mesh_opts=None,               # dict for functions.mesh.curvature_mesh (a, b, speed, ds_min, ...)
             max_iter=6000,                # IPOPT max iterations
             tol=1e-4,                     # IPOPT convergence tolerance
             jit=False,                    # JIT-compile the casadi Functions (needs gcc/clang/cl)
             cse=False,                    # casadi common-subexpression elimination (opt-in)
             screening=False,              # loose IPOPT tolerances (SCREENING_IPOPT) for sweeps / ranking
             ipopt_overrides=None,         # dict of IPOPT options merged in last (any option, per call)
             physics="matlab",             # 'matlab' (default) | 'corrected' (opt-in fixes A1-A6)
             Steering="NA",                # 'NA' | 'Ackermann': front steering geometry (vehModel)
             CamberGain="Off",             # 'Off' | 'Linear' | 'Table': dynamic camber model (vehModel)
             start_state="matlab",         # 'matlab' | 'rolling' | 'periodic': lap start (START_STATES)
             n_bounds=None,                # None | W | (lo, hi): CG-path lateral band [m], see _parse_n_bounds
             between_knots="matlab",       # 'matlab' | 'linear' | 'linear+power': inputs and power rows between the knots (BETWEEN_KNOTS)
             collocation="legendre",       # 'legendre' | 'radau': collocation points of the 23-state NLP (COLLOCATION)
             state_scaling="matlab",       # 'matlab' | 'typical': scales of the 14 chassis states (STATE_SCALINGS)
             atd_tbr=None):                # None | TBR >= 1: torque-bias ratio of both axle differentials (ATD On only)

    # ---- physics flag: set first, so Powertrain / vehParams / the models can read it
    if physics not in PHYSICS:
        raise ValueError(f"physics must be one of {list(PHYSICS)}, got {physics!r}")
    ctx.physics = physics

    # ---- start state: checked before any work, like the physics flag
    if start_state not in START_STATES:
        raise ValueError(f"start_state must be one of {list(START_STATES)}, got {start_state!r}")
    if start_state == "periodic" and circuit not in CLOSED_CIRCUITS:
        raise ValueError(f"start_state='periodic' needs a closed circuit, one of "
                         f"{list(CLOSED_CIRCUITS)}, got {circuit!r}")
    ctx.start_state, ctx.periodic = start_state, start_state == "periodic"

    # ---- between knots (B3): checked before any work. MLTP reads ctx.linear_inputs and ctx.power_at_col.
    if between_knots not in BETWEEN_KNOTS:
        raise ValueError(f"between_knots must be one of {list(BETWEEN_KNOTS)}, got {between_knots!r}")
    ctx.between_knots = between_knots
    ctx.linear_inputs, ctx.power_at_col = between_knots != "matlab", between_knots == "linear+power"

    # ---- collocation scheme (B4): checked before any work. MLTP passes ctx.collocation to discretise().
    if collocation not in COLLOCATION:
        raise ValueError(f"collocation must be one of {list(COLLOCATION)}, got {collocation!r}")
    if collocation == "radau" and OPT_d < 2:        # casadi gives B = [0.5] for the one Radau point tau = 1
        raise ValueError(f"collocation='radau' needs OPT_d >= 2, got OPT_d={OPT_d!r}. With one point, the CasADi "
                         "quadrature weight is 0.5, not 1. Then the lap time is half of the true value.")
    ctx.collocation = collocation

    # ---- state scaling (B7): checked before any work. vehModel reads ctx.chassis_scale (None = MATLAB's 1.0).
    if state_scaling not in STATE_SCALINGS:
        raise ValueError(f"state_scaling must be one of {list(STATE_SCALINGS)}, got {state_scaling!r}")
    ctx.state_scaling = state_scaling
    ctx.chassis_scale = None if state_scaling == "matlab" else TYPICAL_CHASSIS_SCALE

    # ---- vehModel switches (checked here, read by vehModel through ctx)
    if Steering not in STEERING:
        raise ValueError(f"Steering must be one of {list(STEERING)}, got {Steering!r}")
    if CamberGain not in CAMBER_GAIN:
        raise ValueError(f"CamberGain must be one of {list(CAMBER_GAIN)}, got {CamberGain!r}")
    ctx.Steering, ctx.CamberGain = Steering, CamberGain

    # ---- load powertrain and vehicle parameters ---------------------------
    Powertrain(ctx)
    vehParams(ctx, data_dir=data_dir, vp_overrides=vp_overrides, tyre_set=tyre_set)
    vp, pt = ctx.vp, ctx.pt

    # ---- aerodynamic / torque-distribution configuration ------------------
    actaero_map = {"Static": 0, "Active_RW": 1, "Active": 2, "AALB": 3}
    if AeroConfig not in actaero_map:
        raise ValueError(f"Unknown AeroConfig '{AeroConfig}'.")
    vp.ActAero = actaero_map[AeroConfig]

    # conflict guard (ATD and 4 motors cannot both be on) -- matches MATLAB warning
    if ATD == "On" and Electric_4Motors == "On":
        warnings.warn("Electric_4Motors and ATD cannot both be On. "
                      "Setting ATD to Off and ElectricMotors to On.")
        ATD = "Off"
        Electric_4Motors = "On"
    pt.ATD = 1 if ATD == "On" else 0
    pt.EM4 = 1 if Electric_4Motors == "On" else 0

    # ---- torque-bias rows (C4): none unless you give atd_tbr. MLTP.build_path_constraints reads pt.atd_k.
    # k = (TBR - 1) / (TBR + 1) bounds |T_l - T_r| / |T_l + T_r| on each axle.
    ctx.atd_tbr = atd_tbr
    if atd_tbr is not None:
        if pt.ATD != 1:                 # the guard above sets ATD Off when 4 motors are On
            raise ValueError("atd_tbr needs ATD='On' and Electric_4Motors='Off'. It bounds the four ATD inputs.")
        if (isinstance(atd_tbr, (bool, np.bool_)) or not isinstance(atd_tbr, numbers.Real)
                or not (1.0 <= atd_tbr < np.inf)):
            raise ValueError(f"atd_tbr must be None or a finite number >= 1, got {atd_tbr!r}. "
                             "The value 1 means open differentials.")
        pt.atd_k = (atd_tbr - 1.0) / (atd_tbr + 1.0)

    ctx.AeroConfig = AeroConfig
    ctx.ATD = ATD
    ctx.Electric_4Motors = Electric_4Motors
    ctx.circuit = circuit

    # ---- track ------------------------------------------------------------
    ctx.track = _load_track(circuit, circuits_dir)
    tr = ctx.track
    if ctx.periodic and getattr(tr, "x", None) is not None:
        # Spa, NBR: the file stops one step short of the start. Close the lap with a
        # sample at the start point so the periodic rows join the same position.
        gap = float(np.hypot(tr.x[-1] - tr.x[0], tr.y[-1] - tr.y[0]))
        if gap > 1e-3:
            tr.s, tr.k = np.append(tr.s, tr.s[-1] + gap), np.append(tr.k, tr.k[0])
            tr.x, tr.y = np.append(tr.x, tr.x[0]), np.append(tr.y, tr.y[0])

    # ---- lateral bounds: parsed on the final track (after the closing sample above) ----
    n_lo, n_hi, ctx.n_tag = _parse_n_bounds(n_bounds, tr.s)
    ctx.n_bounds = n_bounds
    if n_lo is not None:
        # the space-domain factor sf = (1 - n*kappa)/(...) changes sign where |n*kappa| reaches 1
        nk = np.maximum(np.abs(n_lo), np.abs(n_hi)) * np.abs(np.asarray(tr.k, dtype=float).reshape(-1))
        if np.max(nk) >= 1.0:
            i = int(np.argmax(nk))
            raise ValueError(f"n_bounds reach {max(abs(n_lo[i]), abs(n_hi[i])):.2f} m at s = {float(tr.s[i]):.1f} m, "
                             f"beyond the corner radius 1/|k| = {1.0 / abs(float(tr.k[i])):.2f} m there")
        tr.n_lo, tr.n_hi = n_lo, n_hi

    # ---- boundary conditions ----------------------------------------------
    # state order x = [vx vy r n eps Om_fl Om_fr Om_rl Om_rr
    #                  zs zsdot theta thetadot phi phidot
    #                  wu_fl wu_fr wu_rl wu_rr  zt_fl zt_fr zt_rl zt_rr]   (23)
    zsi = zsdoti = thetai = thetadoti = phii = phidoti = 0.0
    nan = np.nan
    ctx.vi, ctx.ni = vi, ni
    ctx.Xi = np.array([vi, 0.0, nan, ni, nan, nan, nan, nan, nan,
                       zsi, zsdoti, thetai, thetadoti, phii, phidoti,
                       nan, nan, nan, nan, nan, nan, nan, nan])
    ctx.Xi_init = np.array([vi, 0.0, nan, ni, nan, nan, nan])

    vf = nan
    nf = nan
    ctx.Xf = np.array([vf, nan, nan, nf, nan, nan, nan, nan, nan,
                       nan, nan, nan, nan, nan, nan,
                       nan, nan, nan, nan, nan, nan, nan, nan])
    ctx.Xf_init = np.array([vf, nan, nan, nan, nan, nan, nan])

    # start_state variants. Xi_tol holds the half-width of the 23-state start box per state,
    # in scaled units like OPT_e. None means OPT_e for every state. MLTP_initial does not use it.
    ctx.Xi_tol = None
    if start_state == "rolling":
        # straight start: r = 0, wheels roll. Rows 1 and 5-22 are filled from the model by MLTP.
        ctx.Xi[2] = ctx.Xi_init[2] = 0.0
        ctx.Xi_init[5], ctx.Xi_init[6] = vi / vp.Rw_f, vi / vp.Rw_r
        ctx.Xi_tol = np.r_[np.full(9, OPT_e), np.full(14, ROLL_TOL)]
    elif start_state == "periodic":
        # free start (only n if given): the NLP closes the lap with equality rows instead
        ctx.Xi[:] = ctx.Xi_init[:] = nan
        ctx.Xi[3] = ctx.Xi_init[3] = ni
    if ctx.chassis_scale is not None:       # B7: rows 9-22 keep their physical start-box width
        box = np.full(23, OPT_e) if ctx.Xi_tol is None else ctx.Xi_tol
        ctx.Xi_tol = np.r_[box[:9], box[9:] / ctx.chassis_scale]

    # ---- collocation options ----------------------------------------------
    ctx.OPT_ds = OPT_ds         # collocation step (m)
    ctx.OPT_d = OPT_d           # degree of interpolating polynomials
    ctx.OPT_uinter = "linear"   # MATLAB's name for U_k + duk * tau (a near hold, see between_knots). 'constant' holds U_k (not exposed)
    ctx.OPT_e = OPT_e           # slack for path constraints / initial guesses
    # knot placement for discretise(): 'uniform' = N equal steps of ~OPT_ds;
    # 'curvature' = the same N redistributed by functions/mesh.curvature_mesh
    # (denser in corners / at corner entry-exit, sparser on straights);
    # 'auto' = one of those two by lap length (MESH_AUTO_MIN_LENGTH), resolved
    # here because it needs the track loaded above. ctx.mesh is always the
    # concrete 'uniform' / 'curvature' (what discretise, the saved data and
    # MLTP_screen read); ctx.mesh_requested keeps what was asked for.
    if mesh not in ("auto", "uniform", "curvature"):
        raise ValueError(f"mesh must be 'auto', 'uniform' or 'curvature', got {mesh!r}")
    unknown = [key for key in (mesh_opts or {}) if key not in _MESH_OPTS_KEYS]
    if unknown:
        raise ValueError(f"unknown mesh_opts key(s) {unknown}; "
                         f"allowed: {list(_MESH_OPTS_KEYS)}")
    ctx.mesh_requested = mesh
    if mesh == "auto":
        L = float(ctx.track.s[-1] - ctx.track.s[0])
        ctx.mesh = "curvature" if L >= MESH_AUTO_MIN_LENGTH else "uniform"
    else:
        ctx.mesh = mesh
    ctx.mesh_opts = dict(mesh_opts) if mesh_opts else None
    if ctx.mesh_opts and ctx.mesh == "uniform":
        why = (f"lap {L:.0f} m < MESH_AUTO_MIN_LENGTH {MESH_AUTO_MIN_LENGTH:g} m"
               if mesh == "auto" else "mesh='uniform'")
        warnings.warn(f"mesh_opts ignored: mesh resolved to 'uniform' ({why}); "
                      "pass mesh='curvature' to use them.")
        ctx.mesh_opts = None

    # ---- solver options (IPOPT) -------------------------------------------
    # `linear_solver` is configurable (default 'ma57'). An ma* solver uses
    # Coin-HSL, which IPOPT loads at solve time from `hsl_dir` (resolved against
    # COINHSL_DIR / a seeded default in functions/hsl.py); _make_solver probes it
    # and transparently falls back to MUMPS if it is unavailable. 'mumps' (the
    # casadi-bundled solver) is always available and needs no external library.
    ipopt = {
        "max_iter": max_iter,
        "fixed_variable_treatment": "make_constraint",
        "tol": tol,
        "acceptable_tol": 1e-3,
        "mu_init": 1e-1,
        "mu_strategy": "adaptive",
        "bound_push": 1e-2,
        "bound_frac": 1e-2,
        "constr_viol_tol": 1e-4,
        "dual_inf_tol": 1e-4,
        "compl_inf_tol": 1e-4,
        "linear_solver": linear_solver,
        "print_timing_statistics": "yes",
    }
    # screening=True swaps the five tolerances for the looser SCREENING_IPOPT
    # preset (it also replaces the `tol` argument). Applied BEFORE the per-call
    # overrides below, so an explicit ipopt_overrides entry still wins.
    ctx.screening = bool(screening)
    if ctx.screening:
        ipopt.update(SCREENING_IPOPT)
    # Per-call IPOPT overrides win over every default above (and, in MLTP, over
    # the dual warm-start recipe functions.warmstart.warm_start_ipopt_opts, which
    # is applied only to a dual-seeded resolve, never to a cold solve).
    ctx.ipopt_overrides = dict(ipopt_overrides or {})
    ipopt.update(ctx.ipopt_overrides)
    hess = str(ipopt.get("hessian_approximation", "exact")).strip().lower()
    if hess != "exact":
        warnings.warn(f"IPOPT hessian_approximation={hess!r}: the MLTP is tuned for the "
                      "exact Hessian; a quasi-Newton Hessian converges far more slowly on "
                      "this NLP and can change the local optimum found.")
    ctx.opts = {"ipopt": ipopt}
    ctx.opts["_hsl_dir"] = hsl_dir          # consumed + stripped by _make_solver

    # ---- casadi Function options ------------------------------------------
    # Both are opt-in; functions/casadi_opts.fn_opts reads ctx.cse / ctx.jit.
    # cse=True (or MLTP_CSE=1) turns on common-subexpression elimination.
    # jit=True (or MLTP_JIT=1) JIT-compiles the symbolic Functions; it needs
    # gcc/clang/cl on PATH and otherwise warns once and runs without JIT.
    ctx.jit = bool(jit)
    ctx.cse = bool(cse)

    # ---- rate limits and regularisation -----------------------------------
    c = _build_c()
    ctx.c = c
    ctx.rdy = np.array([[0.0], [0.0]])      # first-derivative aux-variable reg
    ctx.rdy2 = np.array([[1.0], [0.0]])     # second-derivative aux-variable reg

    keys = _input_keys(pt.EM4, pt.ATD, vp.ActAero)
    ctx.input_keys = keys
    ctx.duk_ub = _col(c.ub, keys)
    ctx.duk_lb = _col(c.lb, keys)
    ctx.ru = _col(c.ru, keys)
    ctx.rdu = _col(c.rdu, keys)
    ctx.rdu2 = _col(c.rdu2, keys)

    # ---- rate limits / regularisation for the 7-state init program --------
    init_keys = ["T_motor", "T_brake", "delta"]
    ctx.duk_ub_init = _col(c.ub, init_keys)
    ctx.duk_lb_init = _col(c.lb, init_keys)
    ctx.ru_init = _col(c.ru, init_keys)
    ctx.rdu_init = _col(c.rdu, init_keys)
    ctx.rdu2_init = _col(c.rdu2, init_keys)
    ctx.rdy_init = ctx.rdy[0, 0]
    ctx.rdy2_init = ctx.rdy2[0, 0]

    # ---- mesh_opts speed > 0 (B6): the speed term needs the QSS speed. discretise() reads track.v_qss.
    if ctx.mesh_opts and ctx.mesh_opts.get("speed"):
        from functions.ladder import qss_profile
        prof = qss_profile(ctx)
        ctx.track.v_qss = (prof["s"], prof["v"])

    return ctx
