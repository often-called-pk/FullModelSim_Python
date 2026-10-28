"""MLTP_paramOptim.py - direct port of MLTP_paramOptim.m

Co-optimises a set of static *design parameters* together with the racing line:
the parameters are promoted from fixed values (in ctx.vp) to scalar decision
variables that are constant over the lap, appended once to the NLP decision
vector with their own bounds, and solved simultaneously with the trajectory.

Default promoted parameters (Static aero config): brake balance ``brkB``, torque
distribution ``Tdist``, and the four wing angles ``alpha_FL/FR/RW/TW``. (The roll-
stiffness fraction ``ksD`` is intentionally excluded: in the 23-state model the
lumped roll-stiffness block is inactive, so it has no effect on the dynamics.)

The shared solve core ``optimise_design`` is reused by MLTP_TyreOptim.py.

Warm start: besides the 7-state init (solved here, or an init file / MLTP_initial
ctx), ``warm_start`` takes a full 23-state result (a result .mat with data.nlp, or
the ctx / ctx.data of an MLTP() call). If its NLP is this one without the appended
parameter block, the solve starts from its primal AND dual solution with the
parameters at their current vp values (see optimise_design).
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
from functions.importfile import load_solution, result_stem
from functions.mesh import solution_knots, mesh_opts_record
from functions.warmstart import (nlp_record, get_field, is_full_result, nlp_structure,
                                 plan_design_warm_start)
from functions.transcription import (discretise, build_and_solve_nlp,
                                      unpack_solution, reconstruct_x_full,
                                      compute_time, reconstruct_track)
from userOpts import userOpts, setup_knobs, knob_tag, provenance_record
from vehModel import vehModel
from vehParams import MF_KEYS
from MLTP import build_path_constraints, warmstart_guesses, warmstart_guesses_full
from MLTP_initial import MLTP_initial


def optimise_design(param_specs, tag, circuit="Sturn", vi=60.0, ni=np.nan,
                    warm_start=None, AeroConfig="Static", ATD="On",
                    Electric_4Motors="Off", save=True, results_dir="Results",
                    warm_start_duals=True, Fz0_follows_shift=False, build_only=False, **useropts_kwargs):
    """Solve the MLTP with the given design parameters promoted to decision
    variables. ``param_specs`` is a list of (vp_field_name, lower, upper).
    A name can also be an MF key of ctx.mf (for example LMUY or LKY).
    Returns ctx with ctx.data (including ctx.data.optimal_params) and
    ctx.elapsed (init / solve seconds, ipopt_iters, warm_start mode, duals).

    ``warm_start`` (the mode is printed, kept in ctx.elapsed["warm_start"] and
    saved in data["nlp"]["warm_start"], as in MLTP):
      None            solve the 7-state init first and interpolate it ('cold')
      7-state init    an init .mat (data.init) or an MLTP_initial ctx / data
                      ('init7'), interpolated by arc length as before
      full result     a 23-state result .mat (data.nlp) or the ctx / ctx.data
                      of an MLTP() call. If its NLP is this one without the
                      appended P block (same N, OPT_d, nx, nu, ny, path rows,
                      input_keys, collocation grid; setup values may differ),
                      the solve starts from [w_opt; P0], P0 = the current vp
                      values of the promoted fields, with its multipliers
                      (lam_g as saved, lam_x with zero bound multipliers for P)
                      under IPOPT's warm-start recipe ('full+duals';
                      warm_start_duals=False seeds the primal only,
                      'full-primal'). Another structure (N, mesh, config, or a
                      co-optimisation result with its own P block) is
                      interpolated by arc length, primal only ('full-interp').
                      A result solved with another tyre set is ignored and the
                      7-state init is solved instead ('cold').

    ``start_state`` (a userOpts kwarg) works as in MLTP, except that 'rolling' raises
    ValueError: its start equilibrium depends on the design parameters. data.start_state and
    data.Xi record the start box. ``n_bounds`` (a userOpts kwarg) works as in MLTP: the result
    saves as <stem>_hw<W> or <stem>_nbCustom, with data.n_tag. ``between_knots`` (a userOpts kwarg)
    other than 'matlab' raises ValueError: this NLP keeps MATLAB's arithmetic between the knots.
    A source result of another between_knots seeds the primal only. ``collocation`` (a userOpts kwarg)
    works as in MLTP: the NLP and the result use the Radau grid, and data.collocation holds 'radau'.
    A source result of another scheme seeds the primal only.
    Any other setup knob that differs from its default (vp_overrides, vi, OPT_ds and more, see userOpts.setup_knobs)
    appends _<8 hex digits> to the stem and appears in data.provenance.setup, as in MLTP. The promoted parameters and their
    bounds (param_specs, Fz0_shift_bounds) are not part of the name. ctx.out_path is the file that save=True wrote.

    ``Fz0_follows_shift=True`` (only MLTP_TyreOptim passes it) multiplies vp.Fz0 by Fz0_shift
    before it builds the model, as MLTP_TyreOptim.m:128 does. MLTP_paramOptim and the setup_sweep
    finish keep Fz0 fixed, as MLTP.m and MLTP_paramOptim.m do.

    ``build_only=True`` (only functions/sensitivity.setup_gradients passes it) returns the dict of
    build_and_solve_nlp(build_only=True) and does not solve. It raises ValueError if ``warm_start`` is None
    or a full result that this NLP does not accept with its multipliers ('full+duals'). A 7-state init gives
    the dict without lam_g0 and lam_x0."""
    if build_only and warm_start is None:       # a cold start would solve the 7-state init with IPOPT
        raise ValueError("build_only needs a full result as warm_start")
    t0 = time.time()
    elapsed = {}
    ctx = Ctx()
    userOpts(ctx, circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig,
             ATD=ATD, Electric_4Motors=Electric_4Motors, **useropts_kwargs)
    knobs = setup_knobs(dict(useropts_kwargs, vi=vi, ni=ni))
    setup_tag = knob_tag(knobs)
    if ctx.start_state == "rolling":
        raise ValueError("optimise_design: start_state='rolling' is not supported, the start "
                         "equilibrium depends on the design parameters; use 'matlab' or 'periodic'")
    if ctx.between_knots != "matlab":
        raise ValueError(f"optimise_design does not support between_knots={ctx.between_knots!r}. The "
                         "co-optimisation NLP keeps MATLAB's arithmetic between the knots. Use between_knots='matlab'.")
    vp, pt = ctx.vp, ctx.pt

    # ---- promote design parameters: replace vp floats with SX symbols -----
    P_syms, p_lb, p_ub, p_x0, p_names = [], [], [], [], []
    for field, lb, ub in param_specs:
        holder = ctx.mf if field in MF_KEYS else vp
        x0_val = float(getattr(holder, field))     # current value = warm-start guess
        sym = ca.SX.sym(field)
        setattr(holder, field, sym)                # model now sees it symbolically
        P_syms.append(sym); p_lb.append(lb); p_ub.append(ub)
        p_x0.append(x0_val); p_names.append(field)
    P = ca.vertcat(*P_syms)
    nP = len(p_names)
    if Fz0_follows_shift:       # MLTP_TyreOptim.m:128: vp.Fz0 = 4905*vp.Fz0_shift before vehModel
        vp.Fz0 = vp.Fz0 * vp.Fz0_shift

    def _init7():
        """The 7-state init solve (warm_start=None, or a refused full result)."""
        ctx_init = MLTP_initial(circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig,
                                ATD=ATD, Electric_4Motors=Electric_4Motors, save=False,
                                **useropts_kwargs)
        return ctx_init.data.init

    # ---- warm start: data.init (7-state) or a previous full result -------
    src_full = None                     # previous 23-state result, if given
    if warm_start is None:
        ws_mode = "cold"
        init = _init7()
    else:
        if isinstance(warm_start, (str, os.PathLike)):
            src = init = load_solution(os.fspath(warm_start))   # data, or data.init
        else:
            src = get_field(warm_start, "data", warm_start)
            init = get_field(src, "init", src)
        if is_full_result(src):         # data.x_opt with 23 rows (MLTP result)
            ws_mode, src_full, init = "full", src, None
        else:
            ws_mode = "init7"
            x_in = get_field(init, "x_opt")
            if np.ndim(x_in) != 2 or np.shape(x_in)[0] != 7:
                raise ValueError("optimise_design needs a 7-state init (an init file or an "
                                 "MLTP_initial ctx: init.x_opt with 7 rows) or a full 23-state "
                                 "result (a result file or an MLTP ctx: data.x_opt with 23 rows)")
    elapsed["init"] = time.time() - t0

    # ---- full model (symbolic in the promoted parameters) -----------------
    vehModel(ctx)
    m = ctx.m23
    n_per = (m.nx + m.nu) if ctx.periodic else 0

    L = m.sf
    f_dyn = ca.Function("f_dyn", [m.x, m.u, m.pv, P], [m.dx, L], ["x", "u", "pv", "P"], ["dx", "L"],
                        fn_opts(ctx))
    f_sf = ca.Function("sf", [m.x, m.kappa], [m.sf], ["x", "kappa"], ["sf"], fn_opts(ctx))
    hnames, h, h_lb, h_ub = build_path_constraints(ca, m, pt)
    h_eq = ca.Function("h_eq", [m.x, m.u, m.pv, P], [h], ["x", "u", "pv", "P"], ["h"],
                       fn_opts(ctx))

    disc = discretise(ctx.track, ctx.OPT_ds, ctx.OPT_d,
                      mesh=getattr(ctx, "mesh", "uniform"),
                      mesh_opts=getattr(ctx, "mesh_opts", None), scheme=ctx.collocation)
    N = disc["N"]
    nh = len(hnames)

    # ---- warm-start guesses ----------------------------------------------
    warm = None                         # primal/dual seeds from a full result
    if src_full is not None:
        # the source must be this NLP minus the P block; the saved vectors are
        # then extended by P0 = the current vp values (bound multipliers 0)
        expected = dict(nlp_structure(m.nx, m.nu, m.ny, N, ctx.OPT_d, nh, n_param=nP,
                                      n_periodic=n_per),
                        input_keys=list(ctx.input_keys), s_full=disc["s_full"],
                        x_s=m.x_s, u_s=m.u_s, tyre_set=getattr(ctx, "tyre_set", "MF205"),
                        start_state=ctx.start_state, between_knots=ctx.between_knots,
                        collocation=ctx.collocation)
        warm, ws_mode, note = plan_design_warm_start(
            src_full, expected, p_x0, use_duals=warm_start_duals,
            ipopt_overrides=getattr(ctx, "ipopt_overrides", None))
        if build_only and ws_mode != "full+duals":
            raise ValueError(f"the stored result does not fit this NLP: {note}")
        if ws_mode == "cold":
            print(f"[{tag}] warm start ignored: {note}; running the 7-state init")
            src_full = None
            t_init = time.time()
            init = _init7()
            elapsed["init"] += time.time() - t_init
        else:
            print(f"[{tag}] warm start from a full result: {note} -> {ws_mode}")
            guesses = warmstart_guesses_full(ctx, m, src_full, disc)
    if src_full is None:
        # init interpolated by arc length (its grid may differ, e.g. a loaded
        # init file solved with other options)
        init_x = np.asarray(get_field(init, "x_opt"), dtype=float)
        init_u = np.asarray(get_field(init, "u_opt"), dtype=float)
        guesses = warmstart_guesses(ctx, m, init_x, init_u, disc["s_knot"],
                                    solution_knots(init, init_x.shape[1]))
    reg = {"ru": ctx.ru.reshape(-1), "rdu": ctx.rdu.reshape(-1), "rdu2": ctx.rdu2.reshape(-1)}

    param = {"sym": P, "lb": np.array(p_lb), "ub": np.array(p_ub), "x0": np.array(p_x0)}
    res = build_and_solve_nlp(
        ca, m, f_dyn, f_sf, h_eq, h_lb, h_ub, disc, guesses, reg,
        m.duk_lb, m.duk_ub, ctx.Xi, ctx.Xf,     # rate bounds / u_s (normalised)
        ctx.OPT_d, ctx.OPT_uinter, ctx.OPT_e, ctx.opts, param=param, warm=warm,
        Xi_tol=ctx.Xi_tol, periodic=ctx.periodic, build_only=build_only)
    if build_only:
        return res
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
    if res["structure"] != nlp_structure(m.nx, m.nu, m.ny, N, ctx.OPT_d, nh, n_param=nP,
                                         n_periodic=n_per):
        warnings.warn("functions.warmstart.nlp_structure is out of sync with the "
                      f"transcription ({res['structure']}): dual re-injection is unreliable")

    # ---- collect + reconstruct -------------------------------------------
    w_opt = np.array(sol["x"]).reshape(-1)
    P_opt = w_opt[-nP:]
    x_opt, u_opt, _, xc_opt = unpack_solution(
        w_opt, m.nx, m.nu, m.ny, N, ctx.OPT_d, m.x_s, m.u_s, None)
    x_full = reconstruct_x_full(x_opt, xc_opt, m.nx, N, ctx.OPT_d)
    t_opt = compute_time(ca, f_sf, disc, xc_opt, m.x_s)
    n_width = 2.0 * (m.x_s[3] * m.x_max[3])
    track = reconstruct_track(ctx.track, disc["s_full"], disc["k_full"], x_full[3, :], n_width)

    optimal_params = {p_names[i]: float(P_opt[i]) for i in range(nP)}
    data = {
        "s_full": disc["s_full"], "k_full": disc["k_full"],
        "x_opt": x_opt, "u_opt": u_opt, "xc_opt": xc_opt, "x_full": x_full,
        "t_opt": t_opt, "lap_time": float(t_opt[-1]),
        "track": {k: v for k, v in track.items()}, "input_keys": list(ctx.input_keys),
        "optimal_params": optimal_params, "N": N, "OPT_ds": ctx.OPT_ds, "OPT_d": ctx.OPT_d,
        "circuit": circuit, "AeroConfig": AeroConfig, "ATD": ctx.ATD, "EM4": ctx.Electric_4Motors,
        "mesh": getattr(ctx, "mesh", "uniform"),
        "mesh_requested": getattr(ctx, "mesh_requested", "auto"),
        "mesh_opts": mesh_opts_record(getattr(ctx, "mesh_opts", None)),
        "tyre_set": getattr(ctx, "tyre_set", "MF205"),
        "mf_overrides": list(getattr(ctx, "mf_overrides", [])),
        "physics": getattr(ctx, "physics", "matlab"),
        "start_state": ctx.start_state, "Xi": np.array(ctx.Xi, float),   # start box (NaN = free)
        # primal + dual NLP solution (w ends with the nP design parameters, so
        # structure.n_param = nP: a later solve re-injects it only by
        # interpolation) and the warm-start mode this solve used
        "nlp": nlp_record(res, ctx.solve_stats, m.x_s, m.u_s, ws_mode),
        # no params: the promoted vp fields are symbols here
        "provenance": provenance_record(knobs, linear_solver=ctx.opts["ipopt"].get("linear_solver"),
                                        ipopt_overrides=ctx.ipopt_overrides),
    }
    if ctx.n_tag:                       # userOpts(n_bounds=...): the stem tag of this band
        data["n_tag"] = ctx.n_tag
    if ctx.collocation != "legendre":   # B4: the scheme of the grid in s_full. A result without the field is 'legendre'.
        data["collocation"] = ctx.collocation
    if setup_tag:                       # the other setup knobs: the last part of the stem (E2)
        data["setup_tag"] = setup_tag
    ctx.data = SimpleNamespace(**data)

    if save:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, result_stem(
            circuit, tag, getattr(ctx, "tyre_set", "MF205"),
            getattr(ctx, "mesh_requested", "auto"),
            getattr(ctx, "physics", "matlab"), ctx.start_state,
            n_tag=ctx.n_tag, setup_tag=setup_tag) + ".mat")
        sio.savemat(out_path, {"data": data}, do_compression=True)
        ctx.out_path = out_path
        print(f"Saved -> {out_path}")

    summary = ", ".join(f"{k}={v:.4f}" for k, v in optimal_params.items())
    print(f"[{tag}] circuit={circuit}  N={N}  lap time = {t_opt[-1]:.3f} s  "
          f"(init {elapsed['init']:.1f}s, solve {elapsed['solve']:.1f}s)  "
          f"IPOPT iters={elapsed['ipopt_iters']} [{ctx.solve_stats.get('return_status', '?')}]  "
          f"warm start={ws_mode}  duals={'yes' if elapsed['duals'] else 'no'}   optimal: {summary}")
    ctx.elapsed = elapsed
    return ctx


def MLTP_paramOptim(circuit="Sturn", vi=60.0, ni=np.nan, params=None,
                    AeroConfig="Static", ATD="On", Electric_4Motors="Off",
                    warm_start=None, save=True, results_dir="Results",
                    warm_start_duals=True, **useropts_kwargs):
    """Co-optimise design parameters with the lap. ``params`` overrides the
    default (field, lower, upper) list. ``warm_start`` / ``warm_start_duals``:
    see optimise_design (a 7-state init, or a full MLTP result whose primal and
    dual solution is re-injected with the parameters at their vp values)."""
    if params is None:
        params = [
            ("brkB", 0.0, 1.0),        # brake balance (front fraction)
            ("Tdist", 0.0, 1.0),       # torque distribution (rear fraction)
            ("alpha_FL", 0.0, 10.0),   # front-left wing angle  [deg]
            ("alpha_FR", 0.0, 10.0),   # front-right wing angle [deg]
            ("alpha_RW", 0.0, 30.0),   # rear wing angle        [deg]
            ("alpha_TW", -12.0, 12.0), # rear wing tilt         [deg]
        ]
    return optimise_design(params, "paramOptim", circuit=circuit, vi=vi, ni=ni,
                           warm_start=warm_start, AeroConfig=AeroConfig, ATD=ATD,
                           Electric_4Motors=Electric_4Motors, save=save,
                           results_dir=results_dir, warm_start_duals=warm_start_duals,
                           **useropts_kwargs)


if __name__ == "__main__":
    MLTP_paramOptim(circuit="Sturn", vi=60.0)
