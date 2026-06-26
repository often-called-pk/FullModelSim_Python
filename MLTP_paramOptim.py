"""Port of MLTP_paramOptim.m - co-optimise static design parameters with the racing line.

Promotes chosen ctx.vp fields from fixed values to constant-over-lap decision
variables, appended to the NLP vector with their own bounds and solved jointly
with the trajectory. Default params: brkB, Tdist, alpha_FL/FR/RW/TW.

ksD is intentionally excluded: the 23-state model's lumped roll-stiffness block
is inactive, so it has no effect on the dynamics.

Shared core optimise_design is reused by MLTP_TyreOptim.py.
"""

import os
import numpy as np
import scipy.io as sio
from types import SimpleNamespace

import casadi as ca

from functions.context import Ctx
from functions.importfile import importfile
from functions.transcription import (discretise, build_and_solve_nlp,
                                      unpack_solution, reconstruct_x_full,
                                      compute_time, reconstruct_track)
from userOpts import userOpts
from vehModel import vehModel
from MLTP import build_path_constraints, warmstart_guesses
from MLTP_initial import MLTP_initial


def optimise_design(param_specs, tag, circuit="Sturn", vi=60.0, ni=np.nan,
                    warm_start=None, AeroConfig="Static", ATD="On",
                    Electric_4Motors="Off", save=True, results_dir="Results",
                    **useropts_kwargs):
    """Solve the MLTP with the given design parameters promoted to decision
    variables. ``param_specs`` is a list of (vp_field_name, lower, upper).
    Returns ctx with ctx.data (including ctx.data.optimal_params)."""
    ctx = Ctx()
    userOpts(ctx, circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig,
             ATD=ATD, Electric_4Motors=Electric_4Motors, **useropts_kwargs)
    vp, pt = ctx.vp, ctx.pt

    # ---- promote design parameters: replace vp floats with SX symbols -----
    P_syms, p_lb, p_ub, p_x0, p_names = [], [], [], [], []
    for field, lb, ub in param_specs:
        x0_val = float(getattr(vp, field))         # current value = warm-start guess
        sym = ca.SX.sym(field)
        setattr(vp, field, sym)                    # model now sees it symbolically
        P_syms.append(sym); p_lb.append(lb); p_ub.append(ub)
        p_x0.append(x0_val); p_names.append(field)
    P = ca.vertcat(*P_syms)
    nP = len(p_names)

    # ---- warm start -------------------------------------------------------
    if warm_start is None:
        ctx_init = MLTP_initial(circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig,
                                ATD=ATD, Electric_4Motors=Electric_4Motors, save=False)
        init = ctx_init.data.init
    else:
        init = importfile(warm_start)["data"].init
    init_x = np.asarray(init.x_opt, dtype=float)
    init_u = np.asarray(init.u_opt, dtype=float)

    # ---- full model (symbolic in the promoted parameters) -----------------
    vehModel(ctx)
    m = ctx.m23

    L = m.sf
    f_dyn = ca.Function("f_dyn", [m.x, m.u, m.pv, P], [m.dx, L], ["x", "u", "pv", "P"], ["dx", "L"])
    f_sf = ca.Function("sf", [m.x, m.kappa], [m.sf], ["x", "kappa"], ["sf"])
    hnames, h, h_lb, h_ub = build_path_constraints(ca, m, pt)
    h_eq = ca.Function("h_eq", [m.x, m.u, m.pv, P], [h], ["x", "u", "pv", "P"], ["h"])

    disc = discretise(ctx.track, ctx.OPT_ds, ctx.OPT_d)
    N = disc["N"]
    guesses = warmstart_guesses(ctx, m, init_x, init_u, N)
    reg = {"ru": ctx.ru.reshape(-1), "rdu": ctx.rdu.reshape(-1), "rdu2": ctx.rdu2.reshape(-1)}

    param = {"sym": P, "lb": np.array(p_lb), "ub": np.array(p_ub), "x0": np.array(p_x0)}
    res = build_and_solve_nlp(
        ca, m, f_dyn, f_sf, h_eq, h_lb, h_ub, disc, guesses, reg,
        ctx.duk_lb, ctx.duk_ub, ctx.Xi, ctx.Xf,
        ctx.OPT_d, ctx.OPT_uinter, ctx.OPT_e, ctx.opts, param=param)
    sol = res["sol"]

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
    }
    ctx.data = SimpleNamespace(**data)

    if save:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{circuit}_{tag}.mat")
        sio.savemat(out_path, {"data": data}, do_compression=True)
        print(f"Saved -> {out_path}")

    summary = ", ".join(f"{k}={v:.4f}" for k, v in optimal_params.items())
    print(f"[{tag}] circuit={circuit}  N={N}  lap time = {t_opt[-1]:.3f} s   optimal: {summary}")
    return ctx


def MLTP_paramOptim(circuit="Sturn", vi=60.0, ni=np.nan, params=None,
                    AeroConfig="Static", ATD="On", Electric_4Motors="Off",
                    warm_start=None, save=True, results_dir="Results", **useropts_kwargs):
    """Co-optimise design parameters with the lap. ``params`` overrides the
    default (field, lower, upper) list."""
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
                           results_dir=results_dir, **useropts_kwargs)


if __name__ == "__main__":
    MLTP_paramOptim(circuit="Sturn", vi=60.0)
