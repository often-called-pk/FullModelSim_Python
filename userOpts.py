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
import inspect
import warnings
import numpy as np
import scipy.io as sio
from types import SimpleNamespace

from Powertrain import Powertrain
from vehParams import vehParams
from functions.simpleMA import simpleMA
from functions.importfile import mat_to_namespace
from functions.mesh import curvature_mesh


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
    "BCNAssetto": "Barcelona_circuit_fromassetto.mat",
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
        track.s = np.asarray(track.s, dtype=float).reshape(-1)
        track.k = np.asarray(track.k, dtype=float).reshape(-1)
        if hasattr(track, "x"):
            track.x = np.asarray(track.x, dtype=float).reshape(-1)
        if hasattr(track, "y"):
            track.y = np.asarray(track.y, dtype=float).reshape(-1)
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
# (s, k and OPT_ds are passed by discretise(), so they are not mesh_opts keys).
_MESH_OPTS_KEYS = tuple(n for n in inspect.signature(curvature_mesh).parameters
                        if n not in ("s", "k", "OPT_ds"))


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
             vp_overrides=None,            # dict of vehParams primary/mf overrides
             tyre_set="MF205",             # 'MF205' (default, MATLAB-run lateral set) | 'CopyB' (legacy shipped set, pKy4=0)
             OPT_ds=30,                    # collocation step (m)
             OPT_d=3,                      # degree of interpolating polynomials
             OPT_e=1e-2,                   # slack for path constraints / guesses
             mesh="auto",                  # 'auto' | 'uniform' | 'curvature' collocation knots (auto: by lap length)
             mesh_opts=None,               # dict for functions.mesh.curvature_mesh (a, b, ds_min, ...)
             max_iter=6000,                # IPOPT max iterations
             tol=1e-4,                     # IPOPT convergence tolerance
             jit=False,                    # JIT-compile the casadi Functions (needs gcc/clang/cl)
             cse=False,                    # casadi common-subexpression elimination (opt-in)
             screening=False,              # loose IPOPT tolerances (SCREENING_IPOPT) for sweeps / ranking
             ipopt_overrides=None):        # dict of IPOPT options merged in last (any option, per call)

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

    ctx.AeroConfig = AeroConfig
    ctx.ATD = ATD
    ctx.Electric_4Motors = Electric_4Motors
    ctx.circuit = circuit

    # ---- track ------------------------------------------------------------
    ctx.track = _load_track(circuit, circuits_dir)

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

    # ---- collocation options ----------------------------------------------
    ctx.OPT_ds = OPT_ds         # collocation step (m)
    ctx.OPT_d = OPT_d           # degree of interpolating polynomials
    ctx.OPT_uinter = "linear"   # 'linear' or 'constant' inputs (not exposed)
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

    return ctx
