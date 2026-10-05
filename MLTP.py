"""MLTP.py - direct port of MLTP.m

Solves the full Minimum Lap Time Problem with the 23-state model:

    userOpts -> vehModel -> build OCP (objective + config-dependent path
    constraints) -> warm start (the cold-start ladder of functions/ladder.py:
    constant or QSS seed -> 7-state MLTP_initial, or the QSS profile directly;
    an init .mat; or a previous full result) -> direct-collocation transcription
    -> IPOPT (ma57 by default, MUMPS fallback) -> postprocess -> save .mat

The optimal solution is written to Results/<circuit>_<config>.mat (a non-default
tyre set / mesh request appends _<tyre_set> / _mesh<Mesh>, see
functions.importfile.result_stem; an adaptively refined solve, refine=..., saves
as _meshAdaptive). The saved
struct is self-contained (states, inputs, time, the cartesian racing line,
boundaries, per-tyre forces, lap time) so the racing line can be redrawn without
re-solving. Plotly figures are produced by plotSDI.py.
"""

import os
import time
import warnings
import numpy as np
import scipy.io as sio
from types import SimpleNamespace

import casadi as ca

from functions.casadi_opts import fn_opts
from functions.context import Ctx
from functions.importfile import importfile, result_stem
from functions.ladder import (quasi_static_states, resolve_ladder, qss_profile, seed_m23,
                              friction_overrides, homotopy_schedule, ladder_record,
                              homotopy_record, MA57_PRE_ALLOC)
from functions.mesh import solution_knots, mesh_opts_record
from functions.refine import (refine_options, run_refinement, defect_errors, state_rows,
                              interval_polynomials, eval_states, nlp_inputs, refine_record,
                              pass_accepted, lap_rise)
from functions.warmstart import (resolve_source, guesses_from_full, nlp_structure,
                                 plan_full_warm_start, nlp_record, GOOD_STATUS)
from functions.transcription import (discretise, build_and_solve_nlp,
                                      unpack_solution, reconstruct_x_full,
                                      interp_inputs, compute_time,
                                      reconstruct_track)
from userOpts import userOpts
from vehModel import vehModel
from MLTP_initial import MLTP_initial


def _interp_to(grid_new, row, grid_old=None):
    """Linear interpolation of an init signal onto the points ``grid_new``.
    ``grid_old`` = the signal's own abscissae (physical arc length of its
    knots); None keeps the legacy assumption of a uniform grid on [0, 1]."""
    row = np.asarray(row, dtype=float).reshape(-1)
    if grid_old is None:
        grid_old = np.linspace(0.0, 1.0, row.size)
    return np.interp(grid_new, grid_old, row)


def build_path_constraints(ca, m, pt, TyreModel=None):
    """Config-dependent path constraints, ported from MLTP.m's ``switch TyreModel``.
    Returns (hnames, h_expr, h_lb, h_ub). Shared by MLTP and the optim variants.

    ``TyreModel`` defaults to the one the model was built with (m.TyreModel):
      'CombinedSlip' (default, the only case MATLAB's MLTP runs): powertrain rows
          only, motor_power, motor_rpm, BrTh_1 (+ ATD_eq with ATD On) or the 12
          per-motor rows with EM4 (nh = 3 / 4 / 12); the combined-slip Magic
          Formula itself bounds the tyre forces.
      'PureSlip': the four friction circles rho_lim_fl..rr in [0, 1] first, then
          the same rows (nh = 7 / 8 / 16)."""
    if TyreModel is None:
        TyreModel = getattr(m, "TyreModel", "CombinedSlip")
    if TyreModel not in ("CombinedSlip", "PureSlip"):
        raise ValueError(f"TyreModel must be 'CombinedSlip' or 'PureSlip', got {TyreModel!r}")

    if pt.EM4 == 0:
        BrTh_1 = (m.T_motor_n * m.T_brake_n) / 1e-3
        motor_power = (pt.Pmax - m.Om_motor * m.T_motor) / pt.Pmax
        motor_rpm = (pt.OMmax - m.Om_motor) / pt.OMmax
        hnames = ["motor_power", "motor_rpm", "BrTh_1"]
        h = [motor_power, motor_rpm, BrTh_1]
        h_lb = [0, 0, -1.0]
        h_ub = [1, 1, 1.0]
        if pt.ATD == 1:
            hnames.append("ATD_eq")
            h.append(1 - (m.ATD_FL + m.ATD_FR + m.ATD_RL + m.ATD_RR))
            h_lb.append(-1e-3)
            h_ub.append(1e-3)
    else:
        BrTh_fl = (m.T_motor_fl_n * m.T_brake_n) / 1e-3
        BrTh_fr = (m.T_motor_fr_n * m.T_brake_n) / 1e-3
        BrTh_rl = (m.T_motor_rl_n * m.T_brake_n) / 1e-3
        BrTh_rr = (m.T_motor_rr_n * m.T_brake_n) / 1e-3
        mp_fl = (pt.Pmax - m.Om_motor_fl * m.T_motor_fl) / pt.Pmax
        mp_fr = (pt.Pmax - m.Om_motor_fr * m.T_motor_fr) / pt.Pmax
        mp_rl = (pt.Pmax - m.Om_motor_rl * m.T_motor_rl) / pt.Pmax
        mp_rr = (pt.Pmax - m.Om_motor_rr * m.T_motor_rr) / pt.Pmax
        mr_fl = (pt.OMmax - m.Om_motor_fl) / pt.OMmax
        mr_fr = (pt.OMmax - m.Om_motor_fr) / pt.OMmax
        mr_rl = (pt.OMmax - m.Om_motor_rl) / pt.OMmax
        mr_rr = (pt.OMmax - m.Om_motor_rr) / pt.OMmax
        hnames = ["motor_power_fl", "motor_power_fr", "motor_power_rl", "motor_power_rr",
                  "motor_rpm_fl", "motor_rpm_fr", "motor_rpm_rl", "motor_rpm_rr",
                  "BrTh_fl", "BrTh_fr", "BrTh_rl", "BrTh_rr"]
        h = [mp_fl, mp_fr, mp_rl, mp_rr, mr_fl, mr_fr, mr_rl, mr_rr,
             BrTh_fl, BrTh_fr, BrTh_rl, BrTh_rr]
        h_lb = [0, 0, 0, 0, 0, 0, 0, 0, -1.0, -1, -1, -1]
        h_ub = [1, 1, 1, 1, 1, 1, 1, 1, 1.0, 1, 1, 1]

    if TyreModel == "PureSlip":                 # friction circles, ahead of the rest
        def rho(fx, fy, mux, muy, fz):
            return ca.sqrt((fx / (mux * fz))**2 + (fy / (muy * fz))**2)
        hnames = ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr"] + hnames
        h = [rho(m.fx_fl, m.fy_fl, m.mu_fl_x, m.mu_fl_y, m.fz_fl),
             rho(m.fx_fr, m.fy_fr, m.mu_fr_x, m.mu_fr_y, m.fz_fr),
             rho(m.fx_rl, m.fy_rl, m.mu_rl_x, m.mu_rl_y, m.fz_rl),
             rho(m.fx_rr, m.fy_rr, m.mu_rr_x, m.mu_rr_y, m.fz_rr)] + h
        h_lb = [0, 0, 0, 0] + h_lb
        h_ub = [1, 1, 1, 1] + h_ub

    h = ca.vertcat(*h)
    assert len(hnames) == h.shape[0], "Number of path constraints not consistent"
    return hnames, h, np.array(h_lb, dtype=float), np.array(h_ub, dtype=float)


# quasi_static_states (rows 5-22 seeded from vx: rolling wheels, suspension at rest,
# static tyre deflections) lives in functions/ladder.py and is imported above, so
# MLTP.quasi_static_states is the object warmstart_guesses, warmstart_refined and
# ladder.seed_m23 share.


def warmstart_guesses(ctx, m, init_x, init_u, s_knot, s_knot_init=None):
    """Warm-start guesses for the 23-state NLP, interpolated from the 7-state
    data.init plus neutral/static seeds for the suspension and tyre states.

    The init signals are interpolated by PHYSICAL arc length, so either grid may
    be non-uniform (curvature mesh) and the two may have different N:
      s_knot      : target knots, disc["s_knot"] (N = len(s_knot) - 1)
      s_knot_init : knots of the init solution, solution_knots(init) =
                    init.s_full[::OPT_d+1]; None -> uniform over the target span.
    On identical grids the knot guesses equal the init values exactly. A scalar
    ``s_knot`` is the legacy call (N, both grids taken as uniform on [0, 1])."""
    vp = ctx.vp
    n_src = np.shape(init_x)[1]
    if np.ndim(s_knot) == 0:                      # legacy: warmstart_guesses(..., N)
        N = int(s_knot)
        grid, grid_src = np.linspace(0.0, 1.0, N + 1), None
    else:
        grid = np.asarray(s_knot, dtype=float).reshape(-1)
        N = grid.size - 1
        grid_src = (np.linspace(grid[0], grid[-1], n_src) if s_knot_init is None
                    else np.asarray(s_knot_init, dtype=float).reshape(-1))
        if grid_src.size != n_src:
            raise ValueError(f"s_knot_init has {grid_src.size} points for {n_src} init knots")

    def _at(row):
        return _interp_to(grid, row, grid_src)

    vx_0 = _at(init_x[0]); vy_0 = _at(init_x[1])
    r_0 = _at(init_x[2]); n_0 = _at(init_x[3])
    eps_0 = _at(init_x[4])
    # Om = vx / Rw, suspension 0, zt = W0 / kt
    x0_phys = np.vstack([vx_0, vy_0, r_0, n_0, eps_0, quasi_static_states(vp, vx_0)])
    x0 = x0_phys / m.x_s[:, None]

    T_brake_0 = _at(init_u[1])
    delta_0 = _at(init_u[2])
    T_motor_0 = _at(init_u[0])
    u0_rows = []
    for key in ctx.input_keys:
        if key.startswith("T_motor"):
            u0_rows.append(T_motor_0)
        elif key == "T_brake":
            u0_rows.append(T_brake_0)
        elif key == "ATD":
            u0_rows.append(0.25 * np.ones(N + 1))
        elif key in ("FW", "RW", "TW"):
            u0_rows.append(np.zeros(N + 1))
        elif key == "delta":
            u0_rows.append(delta_0)
    u0 = np.vstack(u0_rows) / m.u_s[:, None]
    xc0 = np.kron(x0[:, :-1], np.ones((1, ctx.OPT_d)))
    return {"x0": x0, "u0": u0, "xc0": xc0}


def warmstart_guesses_full(ctx, m, src, disc):
    """Primal warm-start guesses for the 23-state NLP from a previous 23-state
    result ``src`` (loaded data, or ctx.data of an earlier MLTP call). Valid
    across N / OPT_d / mesh / config changes: knot states and inputs are
    interpolated by physical s (source knots = s_full[::OPT_d+1]) onto the new
    knots, collocation states from the source x_full onto the new collocation
    points, inputs matched by channel name. No duals."""
    return guesses_from_full(src, disc["s_knot"], disc["s_col"], ctx.input_keys,
                             m.x_s, m.u_s)


def warmstart_full(ctx, m, src, disc, nh, use_duals=True):
    """Warm start from a previous full 23-state result: (guesses, warm, mode).
    Identical NLP structure (sizes, input_keys, collocation grid; setup / tyre
    values may differ, that is the sweep case) -> warm carries the saved w_opt
    (+ lam_g/lam_x and the IPOPT warm-start recipe unless use_duals=False);
    otherwise warm=None and the s-interpolated guesses are the (primal-only)
    start. A source solved with another tyre set is no start at all:
    (None, None, "cold"), the caller falls back to the 7-state init."""
    expected = dict(nlp_structure(m.nx, m.nu, m.ny, disc["N"], ctx.OPT_d, nh),
                    input_keys=list(ctx.input_keys), s_full=disc["s_full"],
                    x_s=m.x_s, u_s=m.u_s, tyre_set=getattr(ctx, "tyre_set", "MF205"))
    warm, mode, note = plan_full_warm_start(src, expected, use_duals=use_duals,
                                            ipopt_overrides=getattr(ctx, "ipopt_overrides", None))
    if mode == "cold":
        print(f"[MLTP] warm start ignored: {note}; running the 7-state init")
        return None, None, mode
    print(f"[MLTP] warm start from a full result: {note} -> {mode}")
    return warmstart_guesses_full(ctx, m, src, disc), warm, mode


def warmstart_refined(ctx, m, prev_disc, prev_sol, disc_new):
    """Primal seeds {x0, u0, xc0} (scaled) for a refined mesh ``disc_new`` from the
    previous pass's converged solution ``prev_sol`` = (Xk, Uk, Xkj), scaled (w_opt
    unpacked with unit scales), on ``prev_disc`` (the 'reseed' of MLTP(refine=...)):

      vx, vy, r, n, eps   the previous state polynomials at the new knots and
                          collocation points (exact at the kept knots of a nested mesh)
      inputs              the previous NLP's own in-interval arithmetic at the new
                          knots (functions.refine.nlp_inputs; exact at the kept knots,
                          so ATD / aero rows carry over)
      Om, suspension, zt  re-seeded quasi-statically from vx, as warmstart_guesses
                          seeds the 7-state init (Om = vx/Rw, 0, W0/kt)

    Primal only. Interpolating all 23 states instead (guesses_from_full,
    'full-interp') leaves stiff-state defects of order |lambda| h delta: IPOPT
    had not converged after 1885-2964 iterations on Sturn N 12 -> 20 / 18 -> 36.
    This reseed converges in ~90-500 iterations on most default-config Sturn / BCN
    passes, but often not elsewhere, coarse meshes included (runs differing by
    1e-10 relative seed perturbations): Sturn EM4 / AALB N 12 -> 18 hit 1000
    iterations in 2 of 3 runs; Hairpin PureSlip N 11 -> 15 ended
    Restoration_Failed in 2 of 4 (the other 2 converged to optima 4.5-5.2% slower)
    and N 15 -> 17 hit 1000 in 2 of 2; Sturn N 18 -> 36 took 171 iterations from
    one seed and did not converge in 1000 from another. The outcome is chaotic (a
    1e-10 seed change moved N 18 -> 24 from 87 to 269 iterations) and the
    alternatives tried (linear input knots, all-state polynomial carry-over) were
    no better, hence MLTP's pass_max_iter cap, cold retry, lap check and
    keep-the-last-accepted-pass rule."""
    Xk, Uk, Xkj = prev_sol
    Z = interval_polynomials(prev_disc, Xk, Xkj)
    x0 = eval_states(prev_disc, Z, disc_new["s_knot"], x_end=np.asarray(Xk)[:, -1])
    xc0 = eval_states(prev_disc, Z, disc_new["s_col"])
    x_s = np.asarray(m.x_s, dtype=float).reshape(-1)
    for X in (x0, xc0):
        X[5:] = quasi_static_states(ctx.vp, X[0] * x_s[0]) / x_s[5:, None]
    u0 = nlp_inputs(prev_disc, Uk, disc_new["s_knot"], ctx.OPT_uinter)
    return {"x0": x0, "u0": u0, "xc0": xc0}


def MLTP(circuit="Sturn", vi=60.0, ni=np.nan, warm_start=None,
         AeroConfig="Static", ATD="On", Electric_4Motors="Off", TyreModel="CombinedSlip",
         save=True, plot=True, results_dir="Results", plots_dir="Plots",
         warm_start_duals=True, refine=None, ladder="auto", homotopy=None, **useropts_kwargs):
    """Solve the full 23-state MLTP. ``warm_start`` may be None (solve the 7-state
    init first), a path to an init file (data.init), a path to a previous full
    result .mat, or the ctx / ctx.data of an earlier MLTP() call (chain without
    disk I/O). A full result of identical NLP structure is re-injected with its
    primal AND dual solution (IPOPT warm start; set ``warm_start_duals=False``
    for primal only); a structurally different one is interpolated by s. A full
    result solved with another tyre set (no tyre_set field = CopyB) is ignored:
    the 7-state init is solved instead and the warm start is recorded as 'cold'.

    ``refine`` (default None: one solve on the userOpts mesh, nothing else changes)
    turns on adaptive h-refinement of the collocation mesh (functions/refine.py):
    True for the defaults or a dict with any of passes (2), tol (1e-2), max_N
    (4 x the base N), merge (False), ds_min (0.125 OPT_ds), ds_max (2.5 OPT_ds),
    max_split (2), pass_max_iter (1000), states (('n', 'eps')) and max_lap_rise
    (3e-3); an unknown key, a bad value or a state name the model does not have
    raises ValueError before any solve. After the base solve each pass computes
    the integrated defect of the n / eps state polynomials per interval, bisects
    the intervals above tol (nested knots, OPT_d fixed) and re-solves from the
    previous solution (warmstart_refined; primal only, IPOPT max_iter capped at
    pass_max_iter). A pass that does not converge, or converges to a lap more
    than max_lap_rise (relative) slower than the pass it refines (a worse local
    optimum), is retried once from the 7-state init; if the retry fails either
    test too, the last accepted pass is kept (stop 'solve-failed' / 'lap-rise').
    Each pass can thus cost up to 2 x pass_max_iter iterations. Stops when every
    interval is below tol, after `passes` passes, or when nothing over tol can be
    split (max_N, ds_min). The per-attempt log is ctx.refine_log and
    data['refine'] (functions.refine.refine_record); once a refinement pass was
    accepted, data.mesh = data.mesh_requested = ctx.mesh_requested = 'adaptive',
    so the result saves as <stem>_meshAdaptive, and data.nlp.warm_start is
    'refine' ('init7' after a cold retry).

    ``ladder`` (functions/ladder.py) picks the fidelity ladder that builds every cold
    start (warm_start=None, a full result refused for its tyre set, refine's cold
    retry): 'auto' (default, = ladder.AUTO_LADDER = 'legacy'), 'legacy' (constant
    guesses -> 7-state init -> 23-state), 'qss7' (QSS speed profile -> 7-state init
    -> 23-state) or 'qss23' (QSS profile straight into the 23-state NLP, no 7-state
    solve); anything else raises ValueError before any solve, as does an explicit
    'qss7' / 'qss23' together with a 7-state init warm_start (the init replaces the
    lower rungs). A non-legacy ladder also sets IPOPT's ma57_pre_alloc to
    ladder.MA57_PRE_ALLOC unless ipopt_overrides give one. data['ladder'] records
    the ladder and what its lower rungs cost (ladder.ladder_record). ``homotopy``
    (default None: off) is a tyre-friction continuation for a cold solve that fails:
    a schedule of friction scales ending in 1.0, e.g. (1.2, 1.1, 1.0) (True = that
    default), each step an MLTP solve with pDx1, pDx2, pDy1, pDy2 scaled
    (ladder.friction_overrides on top of the caller's vp_overrides); step 1 starts
    cold through the ladder (or from warm_start), every later step from the previous
    step's result (full + duals); refine, save and plot apply to the last step only,
    which is exactly the caller's problem. data['homotopy'] records each step."""
    t0 = time.time()
    elapsed = {}
    hom_steps = useropts_kwargs.pop("_homotopy_steps", None)    # set by a homotopy's last step

    # ---- setup + config ---------------------------------------------------
    ctx = Ctx()
    userOpts(ctx, circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig,
             ATD=ATD, Electric_4Motors=Electric_4Motors, **useropts_kwargs)
    vp, pt = ctx.vp, ctx.pt
    ropts = refine_options(refine, ctx.OPT_ds)  # None = no refinement (ValueError if invalid)
    lname, chain = resolve_ladder(ladder)       # cold-start ladder (ValueError if unknown)
    sched = None if homotopy is None or homotopy is False else homotopy_schedule(homotopy)
    if lname != "legacy":               # QSS rungs: MA57 reallocation failed twice from them
        ctx.opts["ipopt"].setdefault("ma57_pre_alloc", MA57_PRE_ALLOC)

    # ---- warm start: data.init (7-state) or a previous full result -------
    src_full = None                     # previous 23-state result, if given
    init = None
    if warm_start is None:
        ws_mode = "cold"                # built below by the ladder
    else:
        ws_kind, ws_src = resolve_source(warm_start, loader=importfile)
        if ws_kind == "init":
            if ladder not in (None, "auto", "legacy"):
                raise ValueError(f"ladder={ladder!r} with a 7-state init warm_start: the init "
                                 "replaces the lower rungs; pass ladder='auto' or warm_start=None")
            ws_mode = "init7"
            init = ws_src
        else:
            ws_mode = "full"
            src_full = ws_src
    elapsed["init"] = time.time() - t0

    # ---- friction homotopy (homotopy=...): one solve per scale, the last = this problem
    if sched is not None:
        if ropts is not None:           # refine's state names need the model: check before step 1
            state_rows(vehModel(ctx, TyreModel=TyreModel).m23, ropts["states"])
        base_ov = useropts_kwargs.get("vp_overrides")
        steps, prev, c = [], warm_start, None
        for i, scale in enumerate(sched):
            last = i == len(sched) - 1
            kw = dict(useropts_kwargs, vp_overrides=friction_overrides(ctx.mf, scale, base_ov))
            if last:
                kw["_homotopy_steps"] = list(steps)
            t1 = time.time()
            c = MLTP(circuit=circuit, vi=vi, ni=ni, warm_start=prev, AeroConfig=AeroConfig,
                     ATD=ATD, Electric_4Motors=Electric_4Motors, TyreModel=TyreModel,
                     save=save and last, plot=plot and last, results_dir=results_dir,
                     plots_dir=plots_dir, warm_start_duals=warm_start_duals,
                     refine=refine if last else None, ladder=ladder, homotopy=None, **kw)
            print(f"[MLTP] homotopy step {i + 1}/{len(sched)}: friction x{scale:g} -> "
                  f"{c.elapsed['ipopt_iters']} iterations [{c.solve_stats.get('return_status', '?')}], "
                  f"lap {float(c.data.lap_time):.4f} s ({time.time() - t1:.1f} s)")
            steps.append(dict(scale=scale, iters=c.elapsed["ipopt_iters"],
                              status=str(c.solve_stats.get("return_status", "unknown")),
                              lap=float(c.data.lap_time), wall=time.time() - t1,
                              warm_start=c.elapsed.get("warm_start", "?"),
                              m7_iters=(getattr(c.data, "ladder", None) or {}).get("m7_iters", -1)))
            prev = c
        return c

    # ---- full model -------------------------------------------------------
    vehModel(ctx, TyreModel=TyreModel)
    m = ctx.m23
    # refine's indicator rows (a bad state name raises ValueError here, before any solve)
    ind_rows = None if ropts is None else state_rows(m, ropts["states"])

    # ---- OCP: dynamics + objective ---------------------------------------
    L = m.sf
    f_dyn = ca.Function("f_dyn", [m.x, m.u, m.pv], [m.dx, L], ["x", "u", "pv"], ["dx", "L"],
                        fn_opts(ctx))
    f_sf = ca.Function("sf", [m.x, m.kappa], [m.sf], ["x", "kappa"], ["sf"], fn_opts(ctx))

    # ---- OCP: path constraints (powertrain; + friction circles if PureSlip) ----
    hnames, h, h_lb, h_ub = build_path_constraints(ca, m, pt)
    h_eq = ca.Function("h_eq", [m.x, m.u, m.pv], [h], ["x", "u", "pv"], ["h"], fn_opts(ctx))

    # ---- discretisation ---------------------------------------------------
    disc = discretise(ctx.track, ctx.OPT_ds, ctx.OPT_d,
                      mesh=getattr(ctx, "mesh", "uniform"),
                      mesh_opts=getattr(ctx, "mesh_opts", None))
    N = disc["N"]

    # ---- warm-start guesses ----------------------------------------------
    ladder_rec = ladder_record(lname, chain)    # rungs below m23 run by this call
    elapsed["ladder"] = lname
    prof = None                                 # QSS profile of a qss -> m23 ladder (cached)

    def _init_guesses(disc_):
        """Cold-start guesses on disc_ from the ladder's rung below m23: the QSS
        profile (ladder.seed_m23; computed once) or the 7-state init interpolated by
        arc length (solved here, once, seeded by chain[0], unless one was passed in)."""
        nonlocal init, prof
        if init is None and chain[-2] == "qss":
            t_q = time.time()
            if prof is None:
                prof = qss_profile(ctx)
                ladder_rec["qss_lap_s"] = float(prof["lap_time"])
            g = seed_m23(ctx, m, disc_, prof, ctx.input_keys)
            dt = time.time() - t_q
            if not np.isfinite(ladder_rec["qss_wall_s"]):
                ladder_rec["qss_wall_s"] = dt
            elapsed["init"] += dt
            return g
        if init is None:
            t_init = time.time()
            ctx_init = MLTP_initial(circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig,
                                    ATD=ATD, Electric_4Motors=Electric_4Motors, save=False,
                                    seed=chain[0], **useropts_kwargs)
            init = ctx_init.data.init
            dt = time.time() - t_init
            elapsed["init"] += dt
            el7 = getattr(ctx_init, "elapsed", None) or {}
            q7 = getattr(ctx_init, "qss_profile", None)
            if q7 is not None:
                ladder_rec["qss_lap_s"] = float(q7["lap_time"])
                ladder_rec["qss_wall_s"] = float(el7.get("qss", np.nan))
            ladder_rec.update(m7_iters=int(el7.get("ipopt_iters", -1)),
                              m7_status=str(el7.get("return_status", "unknown")),
                              m7_lap_s=float(getattr(init, "lap_time", np.nan)),
                              m7_wall_s=dt - float(el7.get("qss", 0.0)))
            if el7 and ladder_rec["m7_status"] not in GOOD_STATUS:
                print(f"[MLTP] note: the 7-state init ended {ladder_rec['m7_status']} after "
                      f"{ladder_rec['m7_iters']} iterations; continuing from it (data['ladder'])")
        init_x = np.asarray(init.x_opt, dtype=float)
        init_u = np.asarray(init.u_opt, dtype=float)
        return warmstart_guesses(ctx, m, init_x, init_u, disc_["s_knot"],
                                 solution_knots(init, init_x.shape[1]))

    warm = None                         # primal/dual seeds from a full result
    if src_full is not None:
        guesses, warm, ws_mode = warmstart_full(ctx, m, src_full, disc, len(hnames),
                                                use_duals=warm_start_duals)
        if ws_mode == "cold":
            src_full = None
    if src_full is None:                # cold start through the ladder (or the given init)
        guesses = _init_guesses(disc)

    reg = {"ru": ctx.ru.reshape(-1), "rdu": ctx.rdu.reshape(-1), "rdu2": ctx.rdu2.reshape(-1)}

    # ---- build + solve NLP -----------------------------------------------
    # input-rate bounds: m.duk_* = ctx.duk_* / u_s (the NLP bounds normalised rates)
    def _solve(disc_, guesses_, warm_, opts_):
        return build_and_solve_nlp(
            ca, m, f_dyn, f_sf, h_eq, h_lb, h_ub, disc_, guesses_, reg,
            m.duk_lb, m.duk_ub, ctx.Xi, ctx.Xf,
            ctx.OPT_d, ctx.OPT_uinter, ctx.OPT_e, opts_, warm=warm_)

    res = _solve(disc, guesses, warm, ctx.opts)
    sol = res["sol"]
    ctx.solve_stats = res["solver"].stats()
    elapsed["solve"] = time.time() - t0 - elapsed["init"]
    winfo = res["warm_info"]
    if warm is not None:                # what the transcription actually used
        ws_mode = ("full+duals" if winfo["duals"] else
                   "full-primal" if winfo["x0"] else "full-interp")
    elapsed["ipopt_iters"] = int(ctx.solve_stats.get("iter_count", -1))
    elapsed["warm_start"] = ws_mode
    elapsed["duals"] = bool(winfo["duals"])

    # ---- adaptive mesh refinement (refine=...; functions/refine.py) --------
    if ropts is not None:
        t_ref = time.time()
        init_ref = elapsed["init"]      # a cold retry's lazy init is booked to 'init' only
        elapsed["solve_base"] = elapsed["solve"]
        ip = dict(ctx.opts.get("ipopt", {}))
        ip["max_iter"] = min(int(ip.get("max_iter", ropts["pass_max_iter"])), ropts["pass_max_iter"])
        pass_opts = dict(ctx.opts, ipopt=ip)    # cold IPOPT options, max_iter capped
        unit_x, unit_u = np.ones(m.nx), np.ones(m.nu)

        def _pass(res_, disc_, seed, mode, wall):
            """Result dict of one solve for run_refinement (payload: what the next
            step / evaluate / the postprocessing need)."""
            stats = res_["solver"].stats()
            x_n, u_n, _, xc_n = unpack_solution(res_["w_opt"], m.nx, m.nu, m.ny, disc_["N"],
                                                ctx.OPT_d, unit_x, unit_u, None)
            lap = float(compute_time(ca, f_sf, disc_, xc_n, unit_x)[-1])
            return dict(status=str(stats.get("return_status", "unknown")),
                        iters=int(stats.get("iter_count", -1)), wall=float(wall), lap=lap,
                        seed=seed, warm_start=mode, s_knot=disc_["s_knot"],
                        payload=dict(res=res_, disc=disc_, stats=stats, scaled=(x_n, u_n, xc_n)))

        def _step(cur, s_new):
            prev = cur["payload"]
            disc_new = discretise(ctx.track, ctx.OPT_ds, ctx.OPT_d, s_knot=s_new)
            t1 = time.time()
            g = warmstart_refined(ctx, m, prev["disc"], prev["scaled"], disc_new)
            out = _pass(_solve(disc_new, g, None, pass_opts), disc_new, "reseed", "refine",
                        time.time() - t1)
            if pass_accepted(out, cur, ropts):
                return out
            why = (f"ended {out['status']} after {out['iters']} iterations"
                   if out["status"] not in GOOD_STATUS else
                   f"converged to a lap {100.0 * lap_rise(out, cur):.2f}% slower than the pass it refines "
                   f"({out['lap']:.4f} s vs {cur['lap']:.4f} s: a worse local optimum)")
            print(f"[MLTP] refine: the reseeded N={disc_new['N']} solve {why}; retrying once from the "
                  "7-state init")
            out["payload"] = None
            g0 = _init_guesses(disc_new)        # its own time goes to elapsed['init']
            t1 = time.time()
            retry = _pass(_solve(disc_new, g0, None, pass_opts), disc_new, "cold", "init7",
                          time.time() - t1)
            retry["attempts"] = [out]
            return retry

        def _evaluate(r):
            pl = r["payload"]
            x_n, u_n, xc_n = pl["scaled"]
            t1 = time.perf_counter()
            de = defect_errors(f_dyn, pl["disc"], ctx.track, x_n, xc_n, u_n, None, ctx.OPT_uinter)
            return de["E"][ind_rows].max(axis=0), dict(dT=de["dT"], t_eval=time.perf_counter() - t1)

        first = _pass(res, disc, "base", ws_mode, elapsed["solve"])
        final, refine_log, refine_stop = run_refinement(
            first, _step, _evaluate, ropts, order=ctx.OPT_d + 1,
            report=lambda msg: print(f"[MLTP] refine {msg}"))
        accepted = [e for e in refine_log if e["accepted"]]
        refine_passes = int(accepted[-1]["pass"]) if accepted else 0
        if final is not first:          # postprocess the last accepted refined pass
            pl = final["payload"]
            res, disc = pl["res"], pl["disc"]
            N = disc["N"]
            sol = res["sol"]
            ctx.solve_stats = pl["stats"]
            winfo = res["warm_info"]
            ws_mode = final["warm_start"]
            elapsed["ipopt_iters"] = int(ctx.solve_stats.get("iter_count", -1))
            elapsed["warm_start"] = ws_mode
            elapsed["duals"] = bool(winfo["duals"])
        elapsed["solve"] = time.time() - t0 - elapsed["init"]
        elapsed["refine"] = time.time() - t_ref - (elapsed["init"] - init_ref)
        elapsed["ipopt_iters_total"] = int(sum(max(e["iters"], 0) for e in refine_log))
        elapsed["refine_passes"] = refine_passes
        elapsed["refine_stop"] = refine_stop
        ctx.refine_log = refine_log

    if res["structure"] != nlp_structure(m.nx, m.nu, m.ny, N, ctx.OPT_d, len(hnames)):
        warnings.warn("functions.warmstart.nlp_structure is out of sync with the "
                      f"transcription ({res['structure']}): dual re-injection is unreliable")

    # ---- postprocess: collect + reconstruct ------------------------------
    w_opt = np.array(sol["x"]).reshape(-1)
    x_opt, u_opt, _, xc_opt = unpack_solution(
        w_opt, m.nx, m.nu, m.ny, N, ctx.OPT_d, m.x_s, m.u_s, None)
    x_full = reconstruct_x_full(x_opt, xc_opt, m.nx, N, ctx.OPT_d)
    u_full = interp_inputs(u_opt, disc["s_knot"], disc["s_full"], ctx.OPT_uinter)
    t_opt = compute_time(ca, f_sf, disc, xc_opt, m.x_s)

    n_width = 2.0 * (m.x_s[3] * m.x_max[3])
    track = reconstruct_track(ctx.track, disc["s_full"], disc["k_full"], x_full[3, :], n_width)

    # ---- vehicle channels at the knot points -----------------------------
    veh_syms = [
        ("f_drag", m.f_drag), ("f_lift", m.f_lift),
        ("f_lift_fl", m.f_lift_fl), ("f_lift_fr", m.f_lift_fr),
        ("f_lift_rl", m.f_lift_rl), ("f_lift_rr", m.f_lift_rr),
        ("fx_fl", m.fx_fl), ("fy_fl", m.fy_fl), ("fz_fl", m.fz_fl),
        ("fx_fr", m.fx_fr), ("fy_fr", m.fy_fr), ("fz_fr", m.fz_fr),
        ("fx_rl", m.fx_rl), ("fy_rl", m.fy_rl), ("fz_rl", m.fz_rl),
        ("fx_rr", m.fx_rr), ("fy_rr", m.fy_rr), ("fz_rr", m.fz_rr),
        ("sa_fl", m.sa_fl), ("sa_fr", m.sa_fr), ("sa_rl", m.sa_rl), ("sa_rr", m.sa_rr),
        ("sx_fl", m.sx_fl), ("sx_fr", m.sx_fr), ("sx_rl", m.sx_rl), ("sx_rr", m.sx_rr),
        ("dynamic_camber_fl", m.dynamic_camber_fl), ("dynamic_camber_fr", m.dynamic_camber_fr),
        ("dynamic_camber_rl", m.dynamic_camber_rl), ("dynamic_camber_rr", m.dynamic_camber_rr),
        ("zs", m.zs), ("theta", m.theta), ("phi", m.phi),
        ("T_fl", m.T_fl), ("T_fr", m.T_fr), ("T_rl", m.T_rl), ("T_rr", m.T_rr),
        ("Lon_acc", m.Lon_acc), ("Lat_acc", m.Lat_acc),
        # --- added: per-tyre friction coefficients (needed for the friction circle) ---
        ("mu_fl_x", m.mu_fl_x), ("mu_fl_y", m.mu_fl_y),
        ("mu_fr_x", m.mu_fr_x), ("mu_fr_y", m.mu_fr_y),
        ("mu_rl_x", m.mu_rl_x), ("mu_rl_y", m.mu_rl_y),
        ("mu_rr_x", m.mu_rr_x), ("mu_rr_y", m.mu_rr_y),
    ]
    if pt.EM4 == 0:
        veh_syms += [("Om_motor", m.Om_motor), ("P_motor", m.P_motor)]
    else:
        veh_syms += [("P_motor_fl", m.P_motor_fl), ("P_motor_fr", m.P_motor_fr),
                     ("P_motor_rl", m.P_motor_rl), ("P_motor_rr", m.P_motor_rr),
                     ("Om_motor_fl", m.Om_motor_fl), ("Om_motor_fr", m.Om_motor_fr),
                     ("Om_motor_rl", m.Om_motor_rl), ("Om_motor_rr", m.Om_motor_rr)]
    labels = [k for k, _ in veh_syms]
    f_veh = ca.Function("f_veh", [m.x, m.u, m.pv], [s for _, s in veh_syms], fn_opts(ctx))
    vv = f_veh(x_opt / m.x_s[:, None], u_opt / m.u_s[:, None], disc["pv_knot"])
    vehicle = {labels[i]: np.array(vv[i]).reshape(-1) for i in range(len(labels))}

    # energy [kWh]
    if pt.EM4 == 0:
        P = vehicle["P_motor"]
    else:
        P = (vehicle["P_motor_fl"] + vehicle["P_motor_fr"]
             + vehicle["P_motor_rl"] + vehicle["P_motor_rr"])
    E = np.zeros(N + 1)
    for i in range(N):
        E[i + 1] = E[i] + 0.5 * (P[i] + P[i + 1]) * (t_opt[i + 1] - t_opt[i]) * 2.7778e-4 / pt.eff
    vehicle["E_motor"] = E

    # constraint values along the lap
    hval = np.array(h_eq(x_opt / m.x_s[:, None], u_opt / m.u_s[:, None], disc["pv_knot"]).full())
    constraints = {hnames[i]: hval[i, :] for i in range(len(hnames))}

    # ---- assemble data + save --------------------------------------------
    data = {
        "s_full": disc["s_full"], "k_full": disc["k_full"],
        "x_opt": x_opt, "u_opt": u_opt, "xc_opt": xc_opt, "x_full": x_full, "u_full": u_full,
        "t_opt": t_opt, "lap_time": float(t_opt[-1]),
        "track": {k: v for k, v in track.items()}, "vehicle": vehicle,
        "constraints": constraints, "input_keys": list(ctx.input_keys),
        "N": N, "OPT_ds": ctx.OPT_ds, "OPT_d": ctx.OPT_d, "circuit": circuit,
        "AeroConfig": AeroConfig, "ATD": ctx.ATD, "EM4": ctx.Electric_4Motors,
        # collocation mesh and tyre set this solution was computed with
        # (mesh_opts: savemat-safe copy, {} when none were given)
        "mesh": getattr(ctx, "mesh", "uniform"),
        "mesh_requested": getattr(ctx, "mesh_requested", "auto"),
        "mesh_opts": mesh_opts_record(getattr(ctx, "mesh_opts", None)),
        "tyre_set": getattr(ctx, "tyre_set", "MF205"),
        "mf_overrides": list(getattr(ctx, "mf_overrides", [])),
        # primal + dual NLP solution, so a later solve can be warm-started
        # from it (MLTP(warm_start=<this .mat or ctx>)); w_opt is scaled
        "nlp": nlp_record(res, ctx.solve_stats, m.x_s, m.u_s, ws_mode),
    }
    data["ladder"] = dict(ladder_rec)   # cold-start ladder + what its lower rungs cost (metadata)
    if hom_steps is not None:           # last step of MLTP(homotopy=...): the record of every step
        data["homotopy"] = homotopy_record(list(hom_steps) + [dict(
            scale=1.0, iters=elapsed["ipopt_iters"],
            status=str(ctx.solve_stats.get("return_status", "unknown")), lap=float(t_opt[-1]),
            wall=time.time() - t0, warm_start=ws_mode, m7_iters=ladder_rec["m7_iters"])])
        elapsed["homotopy"] = data["homotopy"]
    if ropts is not None:               # adaptive mesh refinement record
        data["refine"] = refine_record(refine_log, ropts, refine_stop,
                                       getattr(ctx, "mesh", "uniform"))
        if refine_passes > 0:           # the solution lives on the refined mesh
            ctx.mesh_requested = "adaptive"
            data["mesh"] = data["mesh_requested"] = "adaptive"
    ctx.data = SimpleNamespace(**data)

    if save:
        os.makedirs(results_dir, exist_ok=True)
        cfg = f"{AeroConfig}_ATD{ctx.ATD}_EM4{ctx.Electric_4Motors}"
        stem = result_stem(circuit, cfg, getattr(ctx, "tyre_set", "MF205"),
                           getattr(ctx, "mesh_requested", "auto"))
        out_path = os.path.join(results_dir, f"{stem}.mat")
        sio.savemat(out_path, {"data": data}, do_compression=True)
        print(f"Saved optimal solution -> {out_path}")

    summary = (f"[MLTP] circuit={circuit}  config={AeroConfig}/ATD={ctx.ATD}/EM4={ctx.Electric_4Motors}  "
               f"N={N}  lap time = {t_opt[-1]:.3f} s  (init {elapsed['init']:.1f}s, solve {elapsed['solve']:.1f}s)  "
               f"IPOPT iters={elapsed['ipopt_iters']} [{ctx.solve_stats.get('return_status', '?')}]  "
               f"warm start={ws_mode}  duals={'yes' if elapsed['duals'] else 'no'}")
    if lname != "legacy":
        summary += f"  ladder={lname}"
    if ropts is not None:
        chain_N = "->".join(str(e["N"]) for e in accepted) or str(N)
        chain_lap = "->".join(f"{e['lap']:.3f}" for e in accepted) or "n/a"
        chain_eta = "->".join(f"{e['eta_max']:.3f}" for e in accepted) or "n/a"
        summary += (f"  refine: N {chain_N}, lap {chain_lap} s, eta_max {chain_eta}, stop={refine_stop}"
                    f" (IPOPT iters total {elapsed['ipopt_iters_total']})")
    print(summary)

    if plot:
        try:
            from plotSDI import plotSDI
            plotSDI(ctx, save_dir=plots_dir)
        except Exception as exc:    # plotSDI may not exist yet / plotly missing
            print(f"[MLTP] plotting skipped: {exc}")

    ctx.elapsed = elapsed
    return ctx


if __name__ == "__main__":
    MLTP(circuit="Sturn", vi=60.0)
