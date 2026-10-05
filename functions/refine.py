"""refine.py - iterative adaptive h-refinement of the collocation mesh (Roadmap Item 9).

numpy only: casadi enters solely through the model Function ``f_dyn`` handed to
defect_errors (called on numpy arrays), so the knot logic and the loop are
testable without it. MLTP(refine=...) drives it:

    base solve on the userOpts mesh (uniform / curvature / auto, unchanged)
    repeat:  eta = per-interval error indicator of the converged solution
             stop 'tol' if max(eta) <= tol, 'passes' after `passes` passes
             bisect every interval with eta_k > tol (refine_knots, nested knots)
             re-solve on the new knots from the previous solution
             (MLTP.warmstart_refined; one cold retry from the 7-state init)
             accept the pass only if IPOPT converged and its lap is at most
             max_lap_rise slower than the pass it refines (pass_accepted)

The collocation degree OPT_d never changes (h-refinement only). The lap check
exists because the 23-state NLP is path sensitive: a re-solve can converge to a
worse local optimum (Hairpin PureSlip N 11 -> 15: Solve_Succeeded at 11.558 s
against the 11.056 s base, +4.5%, where other seeds on the same knots reached
11.002 and 10.831 s), while every other measured pass (Sturn, BCN, EM4) lowered
the lap.

Error indicator (defect_errors): the integrated defect of the state polynomial
(GPOPS-II style integrated residual), in scaled units. For interval k (length h_k,
nodes tau_0 = 0 and the d Legendre points) take the NLP's own degree-d state
polynomial p_k through [X_k, X_k1..X_kd], the NLP's own in-interval inputs
u_k(t) = U_k + (U_{k+1} - U_k) / h_k * t (transcription.build_and_solve_nlp's
arithmetic, MLTP.m line 345 ``Uk + kron(duk, tau)`` with duk = du/ds: a near hold,
NOT the linear interpolant; U_k for OPT_uinter='constant'; an aux y the same way)
and kappa(s) = interp(track.s, track.k). On n_sub = 2d equal sub-segments of
[0, 1] with n_quad = d+1 Gauss points each (one f_dyn.map call for the lap):

    E_ik  = max_e | p_ik(sigma_e) - X_ik - h_k * int_0^sigma_e f_i(p_k, u_k, [y_k], kappa) dt |
    eta_k = max over the indicator states i of E_ik,          sigma_e = e / n_sub

The NLP enforces p_k' = h_k f at the d collocation points only; E is how far the
polynomial drifts from the ODE in between. It is zero to rounding for a solution
that is a polynomial of degree <= d and O(h^(d+1)) for a smooth one (both pinned by
test_refine.py). The max is taken at the n_sub + 1 sub-segment ends only: on
converged Sturn solutions a dense sampling (48 points) reads up to 15-20% higher,
so tol acts as about 1.2 x tol. The lap-time quadrature defect
dT_k = h_k (B.L(X_kj) - int L(p_k)) is returned as a diagnostic only.

Indicator states: n and eps (``states`` option). Their f_dyn rows,
dn/ds = sf (vx sin eps + vy cos eps) and deps/ds = sf r - kappa, contain no tyre,
wheel or suspension state, while the 23-state model is very stiff: df/dx gives
|lambda| h = 2e4-7e4 at h = 45 m (wheel spin), so the all-state defect is ~1e3 on
EVERY interval for the unsprung / pitch / roll rates (flat, max/min ~4) and still
~1e2 at N=36. It cannot localise and would refine uniformly forever; vx's defect is
polluted by the wheel-spin states the same way. The kinematic defect is what
curvature and steering transients produce: on Sturn N=12 it runs from 9e-4 on the
start straight to 0.109 at the S transition, and bisecting the 8 intervals above
2e-2 took eta_max to 0.023 and the lap from 18.226 to 17.937 s. Rejected
alternatives: explicit re-integration of each interval (stiff Radau: ~130 s and 1e6
right-hand sides per evaluation, intervals fail on open-loop wheel lock under the
held NLP inputs), a stiffness-filtered defect (ill-conditioned by the unstable
open-loop modes) and the derivative jump at the knots (it flags the input steps
at every knot and does not use f_dyn). The rows are found by symbol name
(state_rows), so the indicator works for the 7-state model (f_dyn(x, u, y, pv))
as well as the 23-state one (f_dyn(x, u, pv)).

Tolerance: tol = 1e-2 (default) is 5 cm of lateral offset n (n_s = 5 m) or
10 mrad of heading eps.

Knot update (refine_knots): bisect every interval with eta_k > tol (or
clip(ceil((eta_k/tol)^(1/(d+1))), 2, max_split) equal parts), skipped where the
parts would be shorter than ds_min; above max_N the worst intervals are split
first (the pass is flagged capped). merge=True also joins adjacent intervals not
over tol whose eta are both below merge_ratio * tol (default 2^-(d+1) / 2, 1/32
at d=3) when the union is at most ds_max; the knots a merge frees count towards
max_N in the same pass. Knots are nested (every old knot kept) unless merged; the
end points never move. A pass in which nothing can be split ends the loop, so
merge=True never re-solves only to coarsen.

Records: run_refinement returns a per-attempt log (ctx.refine_log) and
refine_record() its savemat-safe summary (data['refine']).
"""

import math

import numpy as np

from .warmstart import GOOD_STATUS

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
REFINE_DEFAULTS = {
    "passes": 2,             # refinement passes after the base solve
    "tol": 1e-2,             # eta tolerance (scaled: 5 cm of n, 10 mrad of eps)
    "max_N": None,           # interval cap; None -> MAX_N_FACTOR * N0 (base mesh)
    "merge": False,          # join adjacent unsplit intervals far below tol
    "ds_min": None,          # shortest interval [m]; None -> DS_MIN_FRAC * OPT_ds
    "ds_max": None,          # longest merged interval [m]; None -> DS_MAX_FRAC * OPT_ds
    "max_split": 2,          # parts per refined interval (2 = bisection)
    "pass_max_iter": 1000,   # IPOPT max_iter of a refinement pass (min with the configured)
    "states": ("n", "eps"),  # indicator states (symbol stems of m.x, see state_rows)
    "max_lap_rise": 3e-3,    # a pass whose lap is more than this fraction slower than the
                             # pass it refines counts as failed (worse local optimum); inf = off
}
DS_MIN_FRAC = 0.125          # 3 bisections of a uniform mesh, 1 below the curvature-mesh floor
DS_MAX_FRAC = 2.5            # the curvature mesh's own ds_max
MAX_N_FACTOR = 4             # default max_N = 4 * N0
STOP_REASONS = ("tol", "passes", "max_N", "no-split", "solve-failed", "lap-rise", "base-failed")
# per-attempt columns of refine_record (1-D arrays; load_solution keeps them 1-D)
RECORD_COLUMNS = ("pass_no", "N", "lap_time", "eta_max", "eta_mean", "s_argmax", "iters", "wall",
                  "n_split", "n_merged", "capped", "converged", "accepted", "dT_sum")
_STATE_ALIASES = {"r": "yawrate"}   # vehModel names the yaw-rate symbol 'yawrate_n'


def _vec(v):
    return np.atleast_1d(np.asarray(v, dtype=float)).reshape(-1)


def _as_int(v, name, lo):
    msg = f"refine option {name!r} must be an integer >= {lo}, got {v!r}"
    if isinstance(v, (bool, np.bool_)):
        raise ValueError(msg)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not (math.isfinite(f) and f.is_integer() and f >= lo):
        raise ValueError(msg)
    return int(f)


def _as_float(v, name, lo, strict, allow_inf=False):
    rel = ">" if strict else ">="
    msg = f"refine option {name!r} must be a number {rel} {lo:g}, got {v!r}"
    if isinstance(v, (bool, np.bool_)):
        raise ValueError(msg)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if math.isnan(f) or (math.isinf(f) and not allow_inf):
        raise ValueError(msg)
    if (f <= lo) if strict else (f < lo):
        raise ValueError(msg)
    return f


def refine_options(refine, OPT_ds):
    """Validated refinement options, or None when refinement is off.

    refine : None / False -> None (no refinement: MLTP behaves exactly as before);
             True -> REFINE_DEFAULTS; a dict -> those keys over the defaults.
    OPT_ds : the nominal collocation step [m]; ds_min / ds_max default to
             DS_MIN_FRAC / DS_MAX_FRAC times it.
    An unknown key or a bad value raises ValueError (MLTP calls this before any
    solve; the state names need the model, MLTP checks them with state_rows right
    after building it, also before any solve). max_N stays None when not given;
    resolve_max_N() turns it into MAX_N_FACTOR * N0 once the base mesh is known."""
    if refine is None or refine is False:
        return None
    if refine is True:
        given = {}
    elif isinstance(refine, dict):
        given = dict(refine)
    else:
        raise ValueError("refine must be None, False, True or a dict of options "
                         f"({list(REFINE_DEFAULTS)}), got {type(refine).__name__}")
    unknown = sorted(str(k) for k in set(given) - set(REFINE_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown refine option(s) {unknown}; allowed: {list(REFINE_DEFAULTS)}")
    o = dict(REFINE_DEFAULTS, **given)
    OPT_ds = float(OPT_ds)
    o["passes"] = _as_int(o["passes"], "passes", 0)
    o["tol"] = _as_float(o["tol"], "tol", 0.0, strict=True)
    o["max_N"] = None if o["max_N"] is None else _as_int(o["max_N"], "max_N", 1)
    if not isinstance(o["merge"], (bool, np.bool_)):
        raise ValueError(f"refine option 'merge' must be True or False, got {o['merge']!r}")
    o["merge"] = bool(o["merge"])
    o["ds_min"] = (DS_MIN_FRAC * OPT_ds if o["ds_min"] is None
                   else _as_float(o["ds_min"], "ds_min", 0.0, strict=False))
    o["ds_max"] = (DS_MAX_FRAC * OPT_ds if o["ds_max"] is None
                   else _as_float(o["ds_max"], "ds_max", 0.0, strict=True, allow_inf=True))
    if not o["ds_min"] < o["ds_max"]:
        raise ValueError(f"refine options need ds_min < ds_max (got {o['ds_min']:g}, {o['ds_max']:g})")
    o["max_split"] = _as_int(o["max_split"], "max_split", 2)
    o["pass_max_iter"] = _as_int(o["pass_max_iter"], "pass_max_iter", 1)
    o["max_lap_rise"] = _as_float(o["max_lap_rise"], "max_lap_rise", 0.0, strict=False, allow_inf=True)
    states = o["states"]
    if isinstance(states, str):
        states = (states,)
    try:
        states = tuple(str(x).strip() for x in states)
    except TypeError:
        raise ValueError(f"refine option 'states' must be a sequence of state names, got {states!r}") from None
    if not states or not all(states):
        raise ValueError(f"refine option 'states' must name at least one state, got {o['states']!r}")
    o["states"] = states
    return o


def resolve_max_N(opts, N0):
    """The interval cap of a run: opts['max_N'], or MAX_N_FACTOR * N0 when None."""
    m = opts.get("max_N")
    return int(m) if m is not None else MAX_N_FACTOR * int(N0)


# ---------------------------------------------------------------------------
# polynomial and input helpers (the NLP's own representation)
# ---------------------------------------------------------------------------
def lagrange_basis(nodes, t):
    """Lagrange basis on ``nodes``: L[j, e] = l_j(t_e) and dL[j, e] = l_j'(t_e).
    With nodes [0, tau_1..tau_d] (the collocation nodes) dL at tau is
    collocation_coeff's C and L at 1 its D."""
    nodes = _vec(nodes)
    t = _vec(t)
    n = nodes.size
    L = np.ones((n, t.size))
    dL = np.zeros((n, t.size))
    for j in range(n):
        others = [r for r in range(n) if r != j]
        denom = 1.0
        for r in others:
            denom *= nodes[j] - nodes[r]
        for r in others:
            L[j] *= t - nodes[r]
        for r in others:
            term = np.ones(t.size)
            for q in others:
                if q != r:
                    term = term * (t - nodes[q])
            dL[j] += term
        L[j] /= denom
        dL[j] /= denom
    return L, dL


def _grid(disc):
    s_knot = _vec(disc["s_knot"])
    dsk = _vec(disc["dsk"])
    tau = _vec(disc["tau"])
    N = s_knot.size - 1
    if dsk.size != N:
        raise ValueError(f"disc has {s_knot.size} knots but {dsk.size} interval lengths")
    return s_knot, dsk, tau, N


def interval_polynomials(disc, Xk, Xkj):
    """Per-interval node values Z (nx, d+1, N): Z[:, :, k] = [X_k, X_k1..X_kd], the
    data of interval k's degree-d state polynomial p_k(t) = Z[:, :, k] @ l(t),
    t in [0, 1]. Xk (nx, N+1) knot states, Xkj (nx, N*d) collocation states with
    column k*d + j = point j of interval k (the NLP's column order)."""
    _, _, tau, N = _grid(disc)
    d = tau.size
    Xk = np.atleast_2d(np.asarray(Xk, dtype=float))
    Xkj = np.atleast_2d(np.asarray(Xkj, dtype=float))
    nx = Xk.shape[0]
    if Xk.shape != (nx, N + 1) or Xkj.shape != (nx, N * d):
        raise ValueError(f"interval_polynomials: Xk {Xk.shape} / Xkj {Xkj.shape} do not fit "
                         f"N = {N}, OPT_d = {d} (expected ({nx}, {N + 1}) / ({nx}, {N * d}))")
    Z = np.empty((nx, d + 1, N))
    Z[:, 0, :] = Xk[:, :N]
    Z[:, 1:, :] = Xkj.reshape(nx, d, N, order="F")
    return Z


def _locate(s_knot, dsk, s):
    """Interval index (the interval a point belongs to: [s_k, s_{k+1}), the last
    one closed) and normalised position t in [0, 1] of the points s."""
    N = dsk.size
    idx = np.clip(np.searchsorted(s_knot, s, side="right") - 1, 0, N - 1)
    t = (s - s_knot[idx]) / dsk[idx]
    return idx, t


def eval_states(disc, Z, s, x_end=None):
    """States of the piecewise state polynomial Z (interval_polynomials) at the
    points s (any arc lengths on the mesh, in the units Z is in). A point on an
    interior knot s_k returns X_k exactly; ``x_end`` (X_N) is returned at the last
    knot, where p_{N-1}(1) differs from X_N by the continuity defect."""
    s_knot, dsk, tau, N = _grid(disc)
    s = _vec(s)
    nodes = np.concatenate(([0.0], tau))
    idx, t = _locate(s_knot, dsk, s)
    L, _ = lagrange_basis(nodes, t)
    out = np.einsum("ijn,jn->in", Z[:, :, idx], L)
    hit = t == 0.0                                     # on a knot: the knot value itself
    out[:, hit] = Z[:, 0, idx[hit]]
    if x_end is not None:
        at_end = s >= s_knot[-1]
        out[:, at_end] = _vec(x_end)[:, None]
    return out


def _input_affine(V, dsk, OPT_uinter):
    """(a, b) with the NLP's in-interval value a[:, k] + b[:, k] * t, t in [0, 1]:
    'linear' -> a = V_k, b = duk = (V_{k+1} - V_k) / h_k (per metre, used with the
    normalised t: the transcription's arithmetic); 'constant' -> b = 0."""
    V = np.atleast_2d(np.asarray(V, dtype=float))
    N = dsk.size
    if V.shape[1] != N + 1:
        raise ValueError(f"input array has {V.shape[1]} columns for N + 1 = {N + 1} knots")
    a = V[:, :N]
    if OPT_uinter == "linear":
        b = (V[:, 1:] - V[:, :-1]) / dsk[None, :]
    elif OPT_uinter == "constant":
        b = np.zeros_like(a)
    else:
        raise ValueError("OPT_uinter must be 'linear' or 'constant'")
    return a, b


def nlp_inputs(disc, Uk, s, OPT_uinter="linear"):
    """The NLP's own input values at the points s: inside interval k
    U_k + duk * t with duk = (U_{k+1} - U_k) / h_k and t = (s - s_k) / h_k in [0, 1]
    (transcription.build_and_solve_nlp evaluates f_dyn with exactly this at the
    collocation points: MLTP.m's Uk + kron(duk, tau), a near hold, NOT the linear
    interpolant interp_inputs / u_full use); U_k held for 'constant'. Exact U at
    every knot, including U_N at the last one."""
    s_knot, dsk, _, N = _grid(disc)
    U = np.atleast_2d(np.asarray(Uk, dtype=float))
    a, b = _input_affine(U, dsk, OPT_uinter)
    s = _vec(s)
    idx, t = _locate(s_knot, dsk, s)
    out = a[:, idx] + b[:, idx] * t[None, :]
    k = np.minimum(np.searchsorted(s_knot, s, side="left"), N)
    hit = s_knot[k] == s
    out[:, hit] = U[:, k[hit]]
    return out


def _quad_plan(n_sub, n_quad):
    """Composite Gauss-Legendre rule on [0, 1]: sub-segment ends sig (n_sub+1,),
    abscissae tq and weights wq (n_sub*n_quad,), sub-segment by sub-segment."""
    g, w = np.polynomial.legendre.leggauss(int(n_quad))
    g = 0.5 * (g + 1.0)
    w = 0.5 * w
    sig = np.linspace(0.0, 1.0, int(n_sub) + 1)
    tq = np.concatenate([sig[e] + (sig[e + 1] - sig[e]) * g for e in range(int(n_sub))])
    wq = np.concatenate([(sig[e + 1] - sig[e]) * w for e in range(int(n_sub))])
    return sig, tq, wq


def _track_sk(track):
    get = track.get if isinstance(track, dict) else (lambda key: getattr(track, key))
    return _vec(get("s")), _vec(get("k"))


def _np(v):
    return np.asarray(v.full() if hasattr(v, "full") else v, dtype=float)


def defect_errors(f_dyn, disc, track, Xk, Xkj, Uk, Yk=None, OPT_uinter="linear",
                  n_sub=None, n_quad=None):
    """Integrated defect of the converged collocation solution, per state and interval.

    f_dyn   : the model Function the NLP used, (x, u, pv) -> (dx, L) or, with an aux
              variable, (x, u, y, pv) -> (dx, L); called on SCALED values
    disc    : transcription.discretise() output of the solution's mesh
    track   : namespace / dict with s, k (kappa = np.interp(s, track.s, track.k))
    Xk, Xkj : scaled knot (nx, N+1) and collocation (nx, N*d) states (w_opt unpacked
              with unit scales); Uk (nu, N+1) and Yk (ny, N+1) scaled inputs / aux
    n_sub, n_quad : sub-segments of each interval (default 2d) and Gauss points per
              sub-segment (default d+1)
    Returns dict(E (nx, N), dT (N,), n_eval):
      E[i, k] = max over sigma_e = e/n_sub of |p_ik(sigma_e) - X_ik - h_k int_0^sigma_e f_i dt|
      dT[k]   = h_k * (B . L(X_kj) - int_0^1 L(p_k(t)) dt), the lap-time quadrature defect
      n_eval  = number of f_dyn evaluations (one mapped call)."""
    s_knot, dsk, tau, N = _grid(disc)
    d = tau.size
    B = _vec(disc["B"])
    n_sub = 2 * d if n_sub is None else int(n_sub)
    n_quad = d + 1 if n_quad is None else int(n_quad)
    has_aux = Yk is not None
    n_in = f_dyn.n_in()
    if n_in != (4 if has_aux else 3):
        raise ValueError(f"defect_errors: f_dyn takes {n_in} inputs; expected "
                         + ("(x, u, y, pv) with Yk" if has_aux else "(x, u, pv) without Yk"))
    Z = interval_polynomials(disc, Xk, Xkj)
    nx = Z.shape[0]
    Xkj = np.atleast_2d(np.asarray(Xkj, dtype=float))
    s_tr, k_tr = _track_sk(track)

    sig, tq, wq = _quad_plan(n_sub, n_quad)
    nq = tq.size
    nodes = np.concatenate(([0.0], tau))
    Lq, _ = lagrange_basis(nodes, tq)                  # (d+1, nq)
    Ls, _ = lagrange_basis(nodes, sig)                 # (d+1, n_sub+1)

    # quadrature points (column k*nq + q), then the N*d collocation points
    Pq = np.einsum("ijk,jq->iqk", Z, Lq).reshape(nx, nq * N, order="F")
    kk = np.repeat(np.arange(N), nq)
    tt = np.tile(tq, N)
    kc = np.repeat(np.arange(N), d)
    tc = np.tile(tau, N)
    ua, ub = _input_affine(Uk, dsk, OPT_uinter)
    U_all = np.hstack([ua[:, kk] + ub[:, kk] * tt[None, :], ua[:, kc] + ub[:, kc] * tc[None, :]])
    args = [np.hstack([Pq, Xkj]), U_all]
    if has_aux:
        ya, yb = _input_affine(Yk, dsk, OPT_uinter)
        args.append(np.hstack([ya[:, kk] + yb[:, kk] * tt[None, :],
                               ya[:, kc] + yb[:, kc] * tc[None, :]]))
    s_q = s_knot[kk] + dsk[kk] * tt
    k_col = (_vec(disc["k_col"]) if "k_col" in disc
             else np.interp(s_knot[kc] + dsk[kc] * tc, s_tr, k_tr))
    args.append(np.concatenate([np.interp(s_q, s_tr, k_tr), k_col]).reshape(1, -1))
    n_eval = N * (nq + d)
    out = f_dyn.map(n_eval)(*args)
    dx = _np(out[0]).reshape(nx, n_eval)
    Lval = _np(out[1]).reshape(-1)

    # cumulative integral of h f over the sub-segments, per interval
    hf = dx[:, :nq * N].reshape(nx, nq, N, order="F") * dsk[None, None, :]
    I_seg = np.einsum("iegk,eg->iek", hf.reshape(nx, n_sub, n_quad, N), wq.reshape(n_sub, n_quad))
    I_cum = np.concatenate([np.zeros((nx, 1, N)), np.cumsum(I_seg, axis=1)], axis=1)
    p_sig = np.einsum("ijk,je->iek", Z, Ls)            # (nx, n_sub+1, N)
    E = np.max(np.abs(p_sig - Z[:, 0:1, :] - I_cum), axis=1)

    T_quad = dsk * (wq @ Lval[:nq * N].reshape(nq, N, order="F"))
    T_col = dsk * (B @ Lval[nq * N:].reshape(d, N, order="F"))
    return dict(E=E, dT=T_col - T_quad, n_eval=int(n_eval))


def state_rows(m, names=("n", "eps")):
    """Row indices of the states called ``names`` in the model's state vector m.x,
    matched by symbol stem (m.x[i].name() == name + '_n'; 'r' is an alias of the
    yaw-rate symbol 'yawrate_n'). n and eps are rows 3 and 4 in both vehModel and
    vehModel_initial."""
    stems = []
    for i in range(int(m.nx)):
        nm = str(m.x[i].name())
        stems.append(nm[:-2] if nm.endswith("_n") else nm)
    rows = []
    for name in names:
        key = _STATE_ALIASES.get(name, name)
        if key not in stems:
            raise ValueError(f"refine state {name!r} not in the model's states {stems}")
        rows.append(stems.index(key))
    return rows


# ---------------------------------------------------------------------------
# knot update
# ---------------------------------------------------------------------------
def refine_knots(s_knot, eta, tol, ds_min=0.0, ds_max=np.inf, max_N=None, merge=False,
                 merge_ratio=None, max_split=2, order=4):
    """New knot vector from the per-interval indicator ``eta`` (len N).

    Every interval with eta_k > tol is split into
    n_k = clip(ceil((eta_k / tol)^(1/order)), 2, max_split) equal parts (order =
    OPT_d + 1, the defect's convergence order; max_split=2 is plain bisection),
    fewer where the parts would be shorter than ds_min (blocked when not even two
    fit). merge=True also joins adjacent pairs that are not over tol, both with eta
    below merge_ratio * tol (default 2^-order / 2) and a union of at most ds_max;
    left to right, each interval at most once. If N (after the merges) would
    exceed max_N the largest eta are split first and the rest is dropped (capped);
    the knots merges free are part of that budget.
    Returns (s_new, info): s_new keeps every old knot unless merged and never moves
    the end points; info = dict(n_split, n_merged, blocked, capped, split (bool
    mask of the old intervals), parts, merged (bool mask), n_over, N_old, N_new)."""
    s = _vec(s_knot)
    N = s.size - 1
    if N < 1 or not np.all(np.diff(s) > 0):
        raise ValueError("refine_knots: s_knot must be strictly increasing with >= 2 entries")
    eta = _vec(eta)
    if eta.size != N:
        raise ValueError(f"refine_knots: {eta.size} indicator values for {N} intervals")
    h = np.diff(s)
    max_split = int(max_split)
    if max_split < 2:
        raise ValueError("refine_knots: max_split must be >= 2")
    over = eta > tol
    parts = np.ones(N, dtype=int)
    for k in np.flatnonzero(over):
        n_k = int(min(max(math.ceil((eta[k] / tol) ** (1.0 / order)), 2), max_split))
        while n_k >= 2 and h[k] / n_k < ds_min * (1.0 - 1e-9):
            n_k -= 1
        parts[k] = n_k if n_k >= 2 else 1
    blocked = int(np.sum(over & (parts < 2)))

    drop = np.zeros(N + 1, dtype=bool)                 # interior knots removed by a merge
    merged = np.zeros(N, dtype=bool)
    n_merged = 0
    if merge:                                          # never an interval over tol (split or not)
        ratio = 2.0 ** (-order) / 2.0 if merge_ratio is None else float(merge_ratio)
        lim = ratio * tol
        k = 0
        while k < N - 1:
            if (not over[k] and not over[k + 1] and eta[k] < lim and eta[k + 1] < lim
                    and h[k] + h[k + 1] <= ds_max * (1.0 + 1e-12)):
                drop[k + 1] = True
                merged[k] = merged[k + 1] = True
                n_merged += 1
                k += 2
            else:
                k += 1

    capped = False
    n_kept = N - n_merged                              # intervals before the splits
    if max_N is not None and n_kept + int(np.sum(parts - 1)) > int(max_N):
        capped = True
        budget = max(int(max_N) - n_kept, 0)
        kept = np.ones(N, dtype=int)
        for k in np.argsort(-eta, kind="stable"):
            if budget <= 0:
                break
            if parts[k] >= 2:
                n_k = min(int(parts[k]), budget + 1)
                if n_k >= 2:
                    kept[k] = n_k
                    budget -= n_k - 1
        parts = kept
    split = parts >= 2

    out = [s[0]]
    for k in range(N):
        a, b = s[k], s[k + 1]
        if parts[k] >= 2:
            out.extend(a + (b - a) * np.arange(1, parts[k]) / parts[k])
        if not drop[k + 1]:
            out.append(b)
    s_new = np.asarray(out, dtype=float)
    info = dict(n_split=int(np.sum(split)), n_merged=int(n_merged), blocked=blocked,
                capped=bool(capped), split=split, parts=parts, merged=merged,
                n_over=int(np.sum(over)), N_old=int(N), N_new=int(s_new.size - 1))
    return s_new, info


# ---------------------------------------------------------------------------
# refinement loop (casadi-free; MLTP supplies the solve and the indicator)
# ---------------------------------------------------------------------------
def _converged(result):
    return str(result.get("status", "")) in GOOD_STATUS


def lap_rise(result, parent):
    """Relative lap change of a pass against the pass it refines,
    (lap - lap_parent) / lap_parent; NaN when either lap is missing or not finite."""
    try:
        new, old = float(result.get("lap", np.nan)), float(parent.get("lap", np.nan))
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(new) and math.isfinite(old)) or old <= 0.0:
        return float("nan")
    return (new - old) / old


def pass_accepted(result, parent, opts):
    """True when the refinement pass ``result`` may replace ``parent`` (the pass it
    refines): IPOPT converged and its lap is at most opts['max_lap_rise'] (relative)
    slower than the parent's. A slower lap is a worse local optimum of the
    path-sensitive NLP, not a mesh effect (module docstring); laps that are not
    known are not judged."""
    if not _converged(result):
        return False
    rise = lap_rise(result, parent)
    return not (math.isfinite(rise) and rise > float(opts.get("max_lap_rise", math.inf)))


def _entry(p, result, info=None):
    """One log row (one solve attempt) from a step / base result dict."""
    s_knot = _vec(result["s_knot"])
    return {
        "pass": int(p),
        "seed": str(result.get("seed") or ("base" if p == 0 else "unknown")),
        "warm_start": str(result.get("warm_start") or ""),
        "N": int(s_knot.size - 1),
        "s_knot": s_knot,
        "lap": float(result.get("lap", np.nan)),
        "iters": int(result.get("iters", -1)),
        "status": str(result.get("status") or "unknown"),
        "converged": _converged(result),
        "accepted": False,              # set by run_refinement for the base / an accepted pass
        "wall": float(result.get("wall", np.nan)),
        "n_split": int(info["n_split"]) if info else 0,
        "n_merged": int(info["n_merged"]) if info else 0,
        "capped": bool(info["capped"]) if info else False,
        "eta": None, "eta_max": np.nan, "eta_mean": np.nan, "s_argmax": np.nan,
        "n_over": -1, "dT_sum": np.nan, "t_eval": np.nan,
    }


def _fill_eta(entry, eta, extras, tol):
    eta = _vec(eta)
    s_knot = entry["s_knot"]
    if eta.size != s_knot.size - 1:
        raise ValueError(f"evaluate returned {eta.size} indicator values for {s_knot.size - 1} intervals")
    k = int(np.argmax(eta))
    entry.update(eta=eta, eta_max=float(eta[k]), eta_mean=float(np.mean(eta)),
                 s_argmax=float(0.5 * (s_knot[k] + s_knot[k + 1])),
                 n_over=int(np.sum(eta > tol)))
    extras = extras or {}
    if extras.get("dT") is not None:
        entry["dT_sum"] = float(np.sum(extras["dT"]))
    if extras.get("t_eval") is not None:
        entry["t_eval"] = float(extras["t_eval"])


def run_refinement(first, step, evaluate, opts, order=4, report=None):
    """Adaptive h-refinement loop around a solver.

    first    : result dict of the base solve: status (IPOPT return status), s_knot,
               and optionally iters, wall, lap, seed, warm_start, payload (anything
               step / evaluate need; never copied into the log)
    step     : step(result, s_new) -> result dict of a solve on the knots s_new
               seeded from ``result`` (s_knot defaults to s_new). A step may list
               attempts that preceded it and were not accepted (pass_accepted; e.g. a
               reseed before a cold retry) as result['attempts'] (result dicts):
               they are logged too.
    evaluate : evaluate(result) -> (eta (N,), extras dict with optional dT, t_eval)
    opts     : refine_options() output (tol, passes, max_N, merge, ds_min, ds_max,
               max_split, max_lap_rise); order = OPT_d + 1 (refine_knots)
    report   : optional callable(str) for progress lines
    Returns (final, log, stop_reason): final = the last ACCEPTED result (the base
    when no pass was accepted), log = one dict per solve attempt (pass, seed, N,
    s_knot, lap, iters, status, converged, accepted, wall, n_split, n_merged,
    capped, eta, eta_max, eta_mean, s_argmax, n_over, dT_sum, t_eval; eta None and
    the eta_* NaN where not evaluated), stop_reason in STOP_REASONS:
      tol          every eta <= tol on the final mesh
      passes       `passes` refinement passes done
      max_N        nothing over tol can be split because of max_N
      no-split     nothing over tol can be split (ds_min)
      solve-failed a refinement pass did not converge (previous result kept)
      lap-rise     a refinement pass converged to a lap more than max_lap_rise
                   slower than the pass it refines (previous result kept)
      base-failed  the base solve did not converge (no refinement)"""
    say = report or (lambda msg: None)
    tol = float(opts["tol"])
    passes = int(opts["passes"])
    cur = first
    N0 = _vec(first["s_knot"]).size - 1
    max_N = resolve_max_N(opts, N0)
    entry = _entry(0, cur)
    entry["accepted"] = entry["converged"]
    log = [entry]
    if not _converged(cur):
        say(f"base solve ended {entry['status']}: no refinement")
        return cur, log, "base-failed"
    p = 0
    while True:
        eta, extras = evaluate(cur)
        _fill_eta(entry, eta, extras, tol)
        say(f"pass {p} ({entry['seed']}): N={entry['N']}, lap {entry['lap']:.4f} s, "
            f"eta max {entry['eta_max']:.3g} at s={entry['s_argmax']:.0f} m, mean {entry['eta_mean']:.3g}, "
            f"{entry['n_over']} interval(s) over tol {tol:g}")
        if entry["eta_max"] <= tol:
            reason = "tol"
            break
        if p >= passes:
            reason = "passes"
            break
        s_old = _vec(cur["s_knot"])
        s_new, info = refine_knots(s_old, eta, tol, ds_min=opts.get("ds_min", 0.0),
                                   ds_max=opts.get("ds_max", np.inf), max_N=max_N,
                                   merge=bool(opts.get("merge", False)),
                                   max_split=int(opts.get("max_split", 2)), order=order)
        if info["n_split"] == 0:        # nothing to refine (a merge-only pass is not re-solved)
            reason = "max_N" if info["capped"] else "no-split"
            say(f"no interval over tol can be split ({info['blocked']} blocked by ds_min"
                + (f", max_N = {max_N} reached" if info["capped"] else "") + ")")
            break
        say(f"pass {p + 1}: N {info['N_old']} -> {info['N_new']} ({info['n_split']} split, "
            f"{info['n_merged']} merged" + (", capped by max_N" if info["capped"] else "")
            + (f", {info['blocked']} blocked by ds_min" if info["blocked"] else "") + ")")
        res = step(cur, s_new)
        res.setdefault("s_knot", s_new)
        p += 1
        for att in res.get("attempts") or []:
            att.setdefault("s_knot", s_new)
            log.append(_entry(p, att, info))
        entry = _entry(p, res, info)
        log.append(entry)
        say(f"pass {p}: {entry['seed']} solve {entry['status']} in {entry['iters']} iterations, "
            f"{entry['wall']:.1f} s")
        if not _converged(res):
            reason = "solve-failed"
            break
        if not pass_accepted(res, cur, opts):
            reason = "lap-rise"
            say(f"pass {p}: lap {entry['lap']:.4f} s is {100.0 * lap_rise(res, cur):.2f}% slower than "
                f"pass {p - 1}'s {float(cur.get('lap', np.nan)):.4f} s (max_lap_rise "
                f"{100.0 * float(opts.get('max_lap_rise', np.inf)):g}%): a worse local optimum, "
                "rejected; previous result kept")
            break
        entry["accepted"] = True
        cur = res
    say(f"stop: {reason}")
    return cur, log, reason


def refine_record(log, opts, stop_reason, base_mesh):
    """savemat-safe summary of a refinement run (data['refine']): the options
    (max_N resolved), the base mesh, the stop reason, one array entry per solve
    attempt (RECORD_COLUMNS: pass_no, N, lap_time, eta_max, eta_mean, s_argmax,
    iters, wall, n_split, n_merged, capped, converged, accepted, dT_sum; NaN where
    not evaluated; load_solution keeps them 1-D for a single attempt), status /
    seed joined with ', ', the base knots s_knot0 and the final (last accepted) eta.
    converged = IPOPT converged; accepted = the attempt became the current result
    (a converged pass rejected for its lap has converged 1, accepted 0)."""
    log = list(log)
    if not log:
        raise ValueError("refine_record: empty log")
    o = {}
    for key, val in dict(opts).items():
        if key == "max_N":
            val = resolve_max_N(opts, log[0]["N"])
        if key == "states":
            val = ",".join(str(x) for x in val)
        elif isinstance(val, (bool, np.bool_)):
            val = int(val)
        elif isinstance(val, (int, np.integer)):
            val = int(val)
        elif isinstance(val, (float, np.floating)):
            val = float(val)
        elif val is None:
            continue
        else:
            val = str(val)
        o[str(key)] = val

    def col(key, dtype):
        return np.array([e.get(key, np.nan) if dtype is float else e.get(key, -1) for e in log],
                        dtype=dtype)

    done = [e for e in log if e.get("accepted") and e.get("eta") is not None]
    final_eta = _vec(done[-1]["eta"]) if done else np.array([np.nan])
    return {
        "opts": o,
        "base_mesh": str(base_mesh or "unknown"),
        "stop_reason": str(stop_reason),
        "n_attempts": len(log),
        "pass_no": col("pass", int),
        "N": col("N", int),
        "lap_time": col("lap", float),
        "eta_max": col("eta_max", float),
        "eta_mean": col("eta_mean", float),
        "s_argmax": col("s_argmax", float),
        "iters": col("iters", int),
        "wall": col("wall", float),
        "n_split": col("n_split", int),
        "n_merged": col("n_merged", int),
        "capped": np.array([int(bool(e.get("capped"))) for e in log], dtype=int),
        "converged": np.array([int(bool(e.get("converged"))) for e in log], dtype=int),
        "accepted": np.array([int(bool(e.get("accepted"))) for e in log], dtype=int),
        "dT_sum": col("dT_sum", float),
        "status": ", ".join(str(e.get("status") or "unknown") for e in log),
        "seed": ", ".join(str(e.get("seed") or "unknown") for e in log),
        "s_knot0": _vec(log[0]["s_knot"]),
        "eta": final_eta,
    }
