"""warmstart.py - primal + dual warm starts for chained MLTP solves.

Roadmap Items 7 (+ the Item 4 enabler): a solved 23-state NLP is saved with its
primal vector and multipliers (data["nlp"]), and a neighbouring problem of
IDENTICAL structure can be re-solved from them in O(10) IPOPT iterations instead
of O(1000). Everything here is casadi-free (unit-tested by test_warmstart.py):

  warm_start_ipopt_opts()  the IPOPT dual warm-start recipe. Applied ONLY to a
                           dual-seeded resolve, never to a cold solve.
  nlp_structure()          the NLP sizes transcription.build_and_solve_nlp
                           produces for a problem (same packing / constraint order).
  structure_mismatch()     why a saved solution can NOT be re-injected verbatim
                           into a new problem ([] = compatible). Sizes, input
                           channels and the collocation grid only: setup / tyre
                           values (vp_overrides) never block re-injection.
  scaling_changed()        x_s / u_s moved with the setup (informational).
  resolve_source()         classify a warm_start argument: 7-state init (file or
                           MLTP_initial ctx) vs full 23-state result (file, MLTP
                           ctx, or its data namespace / dict).
  knots_from_full(), interp_rows(), map_input_rows(), guesses_from_full()
                           primal-only fallback when the structure differs: a
                           previous 23-state result interpolated by physical s.
  nlp_record()             the data["nlp"] dict saved with every result.
"""

import os
import warnings

import numpy as np

# integer fields of the NLP structure record (n_param = number of static design
# parameters appended to w by MLTP_paramOptim; 0 for a plain MLTP solve)
STRUCT_KEYS = ("nx", "nu", "ny", "N", "OPT_d", "n_w", "n_g", "n_param")

# neutral seeds for input channels a source result does not have (as in
# MLTP.warmstart_guesses: equal torque split, wings at 0)
_NEUTRAL_INPUT = {"ATD": 0.25, "FW": 0.0, "RW": 0.0, "TW": 0.0}

# IPOPT return statuses whose multipliers are worth re-injecting
GOOD_STATUS = ("Solve_Succeeded", "Solved_To_Acceptable_Level")


def warm_start_ipopt_opts(mu_init=1e-6, push=1e-10):
    """The standard IPOPT dual warm-start recipe: start from the supplied primal
    AND dual point (warm_start_init_point) and move it only `push` into the
    interior, with a small initial barrier `mu_init`. mu_strategy is left as
    configured. Use only when lam_g0 / lam_x0 are passed to the solver.

    Tuning (Sturn, corrected tyre, MA57, 2026-10-04): IPOPT rebuilds its
    inequality slacks from g(x0), so complementarity at the warm point is
    ~ push x |multiplier| (multipliers reach ~1e5 here): push=1e-8 fails the
    iteration-0 check (compl 1.45e-3 > 1e-4) and re-centres for 18 iterations
    on an identical resolve, push=1e-10 converges at iteration 0 with the same
    lap; on +3 % setup changes 1e-10 needed 11-32 iterations (1e-8: 17-41,
    1e-6: 5-23 but drifts the identical resolve by 2.6e-3 s). Under the default
    mu_strategy='adaptive' mu_init is ignored by IPOPT (identical iterates for
    mu_init 1e-2 and 1e-9); it matters only with mu_strategy='monotone'."""
    push = float(push)
    return {
        "warm_start_init_point": "yes",
        "warm_start_bound_push": push,
        "warm_start_bound_frac": push,
        "warm_start_slack_bound_push": push,
        "warm_start_slack_bound_frac": push,
        "warm_start_mult_bound_push": push,
        "mu_init": float(mu_init),
    }


def nlp_structure(nx, nu, ny, N, OPT_d, nh, n_param=0):
    """Sizes of the NLP that build_and_solve_nlp assembles:
    w = [Xk; Uk; (Yk); Xkj (; P)],
    g = [x0 bounds; xN bounds; (OPT_d+1)*nx defects per interval; nh path
         constraints per knot; nu input-rate constraints per interval]."""
    nx, nu, ny, N, d, nh, n_param = (int(v) for v in (nx, nu, ny, N, OPT_d, nh, n_param))
    n_w = (nx + nu + ny) * (N + 1) + nx * N * d + n_param
    n_g = 2 * nx + (d + 1) * N * nx + nh * (N + 1) + nu * N
    return dict(nx=nx, nu=nu, ny=ny, N=N, OPT_d=d, n_w=n_w, n_g=n_g, n_param=n_param)


def get_field(obj, key, default=None):
    """Field access for dicts (in-memory results) and namespaces (loaded .mat)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    return dict(vars(obj))


def vec(v):
    """1-D float array (loadmat squeezes a length-1 vector to a Python float)."""
    return np.atleast_1d(np.asarray(v, dtype=float)).reshape(-1)


def norm_keys(keys):
    """input_keys as a list of clean str (loadmat returns a blank-padded char
    matrix: 'ATD    ')."""
    if keys is None:
        return None
    if isinstance(keys, str):
        return [keys.strip()]
    return [str(k).strip() for k in np.asarray(keys, dtype=object).reshape(-1)]


def scaling_changed(saved, expected, rtol=1e-12):
    """Names of the variable-scaling vectors ('x_s', 'u_s') that differ between
    two records (compared only when present in both). Informational only: x_s /
    u_s follow setup parameters (vp.Rw, vp.Tbrake_max) BY DESIGN, so a saved
    w_opt / lam stays a good seed in normalised units (a wheel speed tracks
    vx / vx_s, full braking stays T_brake_n = -1) and is re-injected as is."""
    out = []
    for name in ("x_s", "u_s"):
        a, b = get_field(saved, name), get_field(expected, name)
        if a is not None and b is not None:
            a, b = vec(a), vec(b)
            if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=0.0):
                out.append(name)
    return out


def structure_mismatch(saved, expected, s_atol=1e-6):
    """Reasons why `saved` cannot seed `expected` with its exact w / lam vectors.

    Both are dicts with the STRUCT_KEYS integers and optionally `input_keys` and
    `s_full` (the collocation grid: same N on another mesh puts the same w
    entries at other track positions); these are compared only when present in
    both. Setup values (vp / tyre parameters, vp_overrides) are deliberately NOT
    part of the structure: a sweep re-injects across them.
    Returns [] when compatible, else a list of human-readable reasons."""
    if not saved:
        return ["no saved NLP structure"]
    reasons = []
    for k in STRUCT_KEYS:
        dflt = 0 if k == "n_param" else None
        a, b = get_field(saved, k, dflt), get_field(expected, k, dflt)
        try:
            same = a is not None and b is not None and int(a) == int(b)
        except (TypeError, ValueError):
            same = False
        if not same:
            reasons.append(f"{k} {a} -> {b}")
    ka, kb = norm_keys(get_field(saved, "input_keys")), norm_keys(get_field(expected, "input_keys"))
    if ka is not None and kb is not None and ka != kb:
        reasons.append(f"input_keys {ka} -> {kb}")
    sa, sb = get_field(saved, "s_full"), get_field(expected, "s_full")
    if sa is not None and sb is not None:
        sa, sb = vec(sa), vec(sb)
        tol = s_atol * max(1.0, float(np.max(np.abs(sb))) if sb.size else 1.0)
        if sa.shape != sb.shape or not np.allclose(sa, sb, rtol=0.0, atol=tol):
            reasons.append("collocation grid s_full differs")
    return reasons


def is_full_result(data, nx=23):
    """True for a full-model result (data.x_opt with nx rows)."""
    x_opt = get_field(data, "x_opt")
    if x_opt is None:
        return False
    x_opt = np.asarray(x_opt)
    return x_opt.ndim == 2 and x_opt.shape[0] == nx


def resolve_source(warm_start, loader=None, nx=23):
    """Classify a ``warm_start`` argument.

    Accepts a path to an init file (data.init) or to a full result, an MLTP /
    MLTP_initial ctx (uses ctx.data), or a data namespace / dict. Returns
    ("init", data.init) for a 7-state warm start or ("full", data) for a
    previous full-model result (x_opt with ``nx`` rows; may carry data.nlp).
    A path is read with ``loader`` (default functions.importfile.importfile),
    exactly as the legacy init-file branch did."""
    if isinstance(warm_start, (str, os.PathLike)):
        if loader is None:
            from functions.importfile import importfile as loader
        data = loader(os.fspath(warm_start))["data"]
    else:
        data = get_field(warm_start, "data")
        if data is None:                      # already a data namespace / dict
            data = warm_start
    if is_full_result(data, nx):
        return "full", data
    init = get_field(data, "init")
    if init is not None:
        return "init", init
    raise ValueError("warm_start must be an init file / MLTP_initial ctx (data.init) or a "
                     f"full {nx}-state result (data.x_opt with {nx} rows); got {type(warm_start).__name__}")


def knots_from_full(s_full, OPT_d, n_knots=None):
    """Knot positions of a saved grid: s_full = [s_0, d colloc pts, s_1, ..., s_N],
    so the knots are s_full[::OPT_d+1]."""
    knots = vec(s_full)[::int(OPT_d) + 1]
    if n_knots is not None and knots.size != int(n_knots):
        raise ValueError(f"s_full gives {knots.size} knots for OPT_d={OPT_d}, expected {n_knots}")
    return knots


def interp_rows(s_src, M, s_new):
    """Row-wise linear interpolation of M (rows x len(s_src)) onto s_new
    (clamped to the end values outside the source range)."""
    s_src = vec(s_src)
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    if M.shape[1] != s_src.size:
        raise ValueError(f"interp_rows: {M.shape[1]} columns for {s_src.size} grid points")
    if s_src.size > 1 and np.any(np.diff(s_src) <= 0):
        raise ValueError("interp_rows: source grid must be strictly increasing")
    s_new = vec(s_new)
    return np.vstack([np.interp(s_new, s_src, row) for row in M])


def map_input_rows(src_keys, src_rows, new_keys):
    """Rows for ``new_keys`` taken from ``src_rows`` by channel name (the j-th
    occurrence of a repeated key such as ATD maps to the j-th source row of that
    key). Missing channels: a motor torque takes the source's first motor row,
    others get a neutral seed (ATD 0.25, wings / brake / steering 0)."""
    src_keys, new_keys = norm_keys(src_keys) or [], norm_keys(new_keys)
    src_rows = np.atleast_2d(np.asarray(src_rows, dtype=float))
    if src_keys and len(src_keys) != src_rows.shape[0]:
        raise ValueError(f"map_input_rows: {len(src_keys)} keys for {src_rows.shape[0]} rows")
    motor = [i for i, k in enumerate(src_keys) if k.startswith("T_motor")]
    seen, out = {}, []
    for key in new_keys:
        j = seen.get(key, 0)
        seen[key] = j + 1
        idx = [i for i, k in enumerate(src_keys) if k == key]
        if j < len(idx):
            out.append(src_rows[idx[j]])
        elif key.startswith("T_motor") and motor:
            out.append(src_rows[motor[0]])
        else:
            out.append(np.full(src_rows.shape[1], _NEUTRAL_INPUT.get(key, 0.0)))
    return np.vstack(out)


def guesses_from_full(src, s_knot, s_col, input_keys, x_s, u_s):
    """Scaled primal guesses {x0, u0, xc0} for a new grid (s_knot, s_col) from a
    previous full-model result ``src`` (x_opt, u_opt, s_full, OPT_d, x_full,
    input_keys), interpolated by physical s. Works across N / OPT_d / mesh /
    config changes; no multipliers (primal-only)."""
    x_opt = np.asarray(get_field(src, "x_opt"), dtype=float)
    u_opt = np.atleast_2d(np.asarray(get_field(src, "u_opt"), dtype=float))
    s_full = vec(get_field(src, "s_full"))
    d_src = int(get_field(src, "OPT_d"))
    s_knot_src = knots_from_full(s_full, d_src, n_knots=x_opt.shape[1])
    s_knot, s_col = vec(s_knot), vec(s_col)

    x0 = interp_rows(s_knot_src, x_opt, s_knot)
    x_full = get_field(src, "x_full")
    if x_full is not None and np.shape(x_full) == (x_opt.shape[0], s_full.size):
        xc = interp_rows(s_full, x_full, s_col)       # source collocation detail kept
    else:
        d_new = s_col.size // max(s_knot.size - 1, 1)
        xc = np.kron(x0[:, :-1], np.ones((1, d_new)))
    u_knots = interp_rows(s_knot_src, u_opt, s_knot)
    u0 = map_input_rows(get_field(src, "input_keys"), u_knots, input_keys)

    x_s = vec(x_s).reshape(-1, 1)
    u_s = vec(u_s).reshape(-1, 1)
    return {"x0": x0 / x_s, "u0": u0 / u_s, "xc0": xc / x_s}


def plan_full_warm_start(src, expected, use_duals=True, ipopt_overrides=None):
    """Decide how a previous full result ``src`` seeds a new NLP whose
    ``expected`` structure is nlp_structure(...) plus input_keys / s_full
    (and x_s / u_s, used only for the note). A source solved with different
    vp / tyre values (vp_overrides) re-injects as long as the structure matches,
    but not across a different tyre SET: when ``expected`` carries "tyre_set"
    (the target ctx.tyre_set) it must equal the source's data.tyre_set (absent =
    a pre-flip result = 'CopyB'); without the key the check is skipped.
    Returns (warm, mode, note):

      None, "cold"                  the tyre sets differ: ``src`` is no usable
                                    start at all, primal-only interpolation
                                    included (a CopyB seed sat in IPOPT
                                    restoration for 4000+ iterations on an
                                    MF205 target, a cold start needs 489)
      None, "full-interp"           no saved NLP vectors, or the structure
                                    differs: the caller uses s-interpolated
                                    primal guesses only (no duals)
      {x0}, "full-primal"           identical structure, use_duals=False:
                                    exact saved w_opt, cold IPOPT options
      {x0, lam_g0, lam_x0, ipopt},  identical structure: exact primal + duals
        "full+duals"                with the warm-start recipe, then any
                                    per-call ipopt_overrides on top
    """
    tyre = get_field(expected, "tyre_set")
    if tyre is not None:
        have = str(get_field(src, "tyre_set", None) or "CopyB").strip()
        want = str(tyre).strip()
        if have != want:
            return None, "cold", (f"source result used tyre_set {have!r}, target is {want!r}; "
                                  "a different tyre set is not a usable warm start "
                                  "(primal-only included)")
    nlp = get_field(src, "nlp")
    if nlp is None:
        return None, "full-interp", "full result has no saved NLP vectors (data.nlp)"
    saved = dict(as_dict(get_field(nlp, "structure")),
                 input_keys=get_field(src, "input_keys"), s_full=get_field(src, "s_full"),
                 x_s=get_field(nlp, "x_s"), u_s=get_field(nlp, "u_s"))
    why = structure_mismatch(saved, expected)
    if why:
        return None, "full-interp", "NLP structure differs (" + "; ".join(why) + ")"
    rescaled = scaling_changed(saved, expected)
    extra = (f"; {'/'.join(rescaled)} changed with the setup, re-injected in normalised units"
             if rescaled else "")
    warm = {"x0": vec(get_field(nlp, "w_opt"))}
    if not use_duals:
        return warm, "full-primal", "same NLP structure, saved primal only (duals disabled)" + extra
    status = str(get_field(nlp, "return_status", "unknown"))
    if status not in GOOD_STATUS:
        warnings.warn(f"warm start: the source solve ended with {status!r}; "
                      "re-injecting its multipliers anyway", RuntimeWarning, stacklevel=2)
    warm["lam_g0"] = vec(get_field(nlp, "lam_g"))
    warm["lam_x0"] = vec(get_field(nlp, "lam_x"))
    ip = warm_start_ipopt_opts()
    ip.update(dict(ipopt_overrides or {}))
    warm["ipopt"] = ip
    return warm, "full+duals", "same NLP structure, saved primal + duals re-injected" + extra


def nlp_record(res, stats=None, x_s=None, u_s=None, warm_start="cold"):
    """The data["nlp"] dict: what a later solve needs to re-inject this one
    (savemat-safe: arrays, ints, non-empty strings, one nested struct)."""
    stats = stats or {}
    rec = {
        "w_opt": vec(res["w_opt"]),
        "lam_g": vec(res["lam_g"]),
        "lam_x": vec(res["lam_x"]),
        "structure": {k: int(res["structure"][k]) for k in STRUCT_KEYS},
        "sym_type": str(res.get("sym_type") or "SX"),
        "linear_solver": str(res.get("linear_solver") or "unknown"),
        "ipopt_iters": int(stats.get("iter_count", -1)),
        "return_status": str(stats.get("return_status") or "unknown"),
        "warm_start": str(warm_start or "cold"),
    }
    if x_s is not None:
        rec["x_s"] = vec(x_s)
    if u_s is not None:
        rec["u_s"] = vec(u_s)
    return rec
