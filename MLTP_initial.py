"""MLTP_initial.py - direct port of MLTP_initial.m

Solves the simplified 7-state Minimum Lap Time Problem to produce a warm start
for the full model. Pipeline:

    userOpts -> vehModel_initial -> build OCP (objective + path constraints)
    -> direct-collocation transcription -> IPOPT -> postprocess -> save .mat

The optimal solution is written to Results/init_<circuit>.mat as a struct
``data.init`` that MLTP.py reloads (via importfile) to seed the 23-state solve.
"""

import os
import time
import numpy as np
import scipy.io as sio

import casadi as ca

from functions.context import Ctx
from userOpts import userOpts
from vehModel_initial import vehModel_initial
from functions.transcription import (discretise, build_and_solve_nlp,
                                      unpack_solution, reconstruct_x_full,
                                      interp_inputs, compute_time,
                                      reconstruct_track)


def _ns_to_dict(track):
    """Convert a track namespace (or dict) to a plain dict for savemat."""
    if isinstance(track, dict):
        return track
    return {k: v for k, v in vars(track).items()}


def MLTP_initial(circuit="Sturn", vi=60.0, ni=np.nan, save=True,
                 results_dir="Results", **useropts_kwargs):
    t0 = time.time()
    elapsed = {}

    # ---- setup ------------------------------------------------------------
    ctx = Ctx()
    userOpts(ctx, circuit=circuit, vi=vi, ni=ni, **useropts_kwargs)
    vehModel_initial(ctx)
    m = ctx.m7
    vp, pt = ctx.vp, ctx.pt
    OPT_e = ctx.OPT_e

    # ---- OCP: dynamics + objective ---------------------------------------
    L = m.sf                                   # objective integrand (lap time)
    f_dyn = ca.Function("f_dyn", [m.x, m.u, m.y, m.pv], [m.dx, L],
                        ["x", "u", "y", "pv"], ["dx", "L"])
    f_sf = ca.Function("sf", [m.x, m.kappa], [m.sf], ["x", "kappa"], ["sf"])

    # ---- OCP: path constraints (nh = 6) ----------------------------------
    BrTh = m.Tdrive_n * m.Tbrake_n
    ltx_eq = ((m.fx_f * ca.cos(m.delta) - m.fy_f * ca.sin(m.delta)
               + m.fx_r + m.f_drag) * vp.hcg / vp.l - m.ltx) / m.ltx_s
    motor_power = (m.Om_motor * m.Tdrive - pt.Pmax) / \
                  (m.Tdrive_s * ((m.Om_f_s + m.Om_r_s) / 2) * vp.gear)
    motor_rpm = m.Om_motor - pt.OMmax
    mu_lim_f = m.mu_f**2 - (m.fx_f**2 + m.fy_f**2) / m.fz_f**2
    mu_lim_r = m.mu_r**2 - (m.fx_r**2 + m.fy_r**2) / m.fz_r**2

    hnames = ["mu_lim_f", "mu_lim_r", "motor_power", "motor_rpm", "ltx_eq", "BrTh"]
    h = ca.vertcat(mu_lim_f, mu_lim_r, motor_power, motor_rpm, ltx_eq, BrTh)
    h_lb = np.array([0.0, 0.0, -np.inf, -np.inf, -OPT_e, 0.0])
    h_ub = np.array([np.inf, np.inf, 0.0, 0.0, OPT_e, np.inf])
    assert len(hnames) == h.shape[0], "Number of path constraints not consistent"
    h_eq = ca.Function("h_eq", [m.x, m.u, m.y, m.pv], [h], ["x", "u", "y", "pv"], ["h"])

    # ---- discretisation ---------------------------------------------------
    disc = discretise(ctx.track, ctx.OPT_ds, ctx.OPT_d)
    N = disc["N"]
    elapsed["setup"] = time.time() - t0

    # ---- initial guesses (constant) --------------------------------------
    vx_0 = vi * np.ones(N + 1)
    vy_0 = OPT_e * np.ones(N + 1)
    r_0 = OPT_e * np.ones(N + 1)
    n_0 = OPT_e * np.ones(N + 1)
    eps_0 = np.zeros(N + 1)
    Om_f_0 = vx_0 / vp.Rw
    Om_r_0 = vx_0 / vp.Rw
    Tdrive_0 = 0.85 * pt.Tmax * np.ones(N + 1)
    Tbrake_0 = np.zeros(N + 1)
    delta_0 = np.zeros(N + 1)
    ltx_0 = np.zeros((m.ny, N + 1))

    x0 = np.vstack([vx_0, vy_0, r_0, n_0, eps_0, Om_f_0, Om_r_0]) / m.x_s[:, None]
    u0 = np.vstack([Tdrive_0, Tbrake_0, delta_0]) / m.u_s[:, None]
    y0 = ltx_0 / m.y_s[:, None]
    xc0 = np.kron(x0[:, :-1], np.ones((1, ctx.OPT_d)))     # nx x (N*d)
    guesses = {"x0": x0, "u0": u0, "y0": y0, "xc0": xc0}

    reg = {"ru": ctx.ru_init.reshape(-1),
           "rdu": ctx.rdu_init.reshape(-1),
           "rdu2": ctx.rdu2_init.reshape(-1),
           "rdy": np.array([ctx.rdy_init]),
           "rdy2": np.array([ctx.rdy2_init])}

    # ---- build + solve NLP -----------------------------------------------
    res = build_and_solve_nlp(
        ca, m, f_dyn, f_sf, h_eq, h_lb, h_ub, disc, guesses, reg,
        m.duk_lb_init, m.duk_ub_init, ctx.Xi_init, ctx.Xf_init,
        ctx.OPT_d, ctx.OPT_uinter, ctx.OPT_e, ctx.opts)
    sol = res["sol"]
    elapsed["solve"] = time.time() - t0 - elapsed["setup"]

    # ---- postprocess: collect + reconstruct ------------------------------
    w_opt = np.array(sol["x"]).reshape(-1)
    x_opt, u_opt, y_opt, xc_opt = unpack_solution(
        w_opt, m.nx, m.nu, m.ny, N, ctx.OPT_d, m.x_s, m.u_s, m.y_s)
    x_full = reconstruct_x_full(x_opt, xc_opt, m.nx, N, ctx.OPT_d)
    u_full = interp_inputs(u_opt, disc["s_knot"], disc["s_full"], ctx.OPT_uinter)
    y_full = interp_inputs(y_opt, disc["s_knot"], disc["s_full"], "linear")
    t_opt = compute_time(ca, f_sf, disc, xc_opt, m.x_s)

    n_width = 2.0 * (m.x_s[3] * m.x_max[3])     # physical track width (2 * n_max)
    track = reconstruct_track(ctx.track, disc["s_full"], disc["k_full"],
                              x_full[3, :], n_width)

    # ---- postprocess: vehicle channels at the knot points ----------------
    f_veh = ca.Function(
        "f_veh", [m.x, m.u, m.y, m.pv],
        [m.f_drag, m.f_lift, m.fx_f, m.fy_f, m.fz_f, m.fx_r, m.fy_r, m.fz_r,
         m.sa_f, m.sa_r, m.sx_f, m.sx_r, m.T_f, m.T_r, m.Om_motor, m.P_motor,
         m.mu_f, m.mu_r])
    vv = f_veh(x_opt / m.x_s[:, None], u_opt / m.u_s[:, None],
               y_opt / m.y_s[:, None], disc["pv_knot"])
    vlabels = ["f_drag", "f_lift", "fx_f", "fy_f", "fz_f", "fx_r", "fy_r", "fz_r",
               "sa_f", "sa_r", "sx_f", "sx_r", "T_f", "T_r", "Om_motor", "P_motor",
               "mu_f", "mu_r"]
    vehicle = {lbl: np.array(vv[i]).reshape(-1) for i, lbl in enumerate(vlabels)}

    # path-constraint values along the lap
    hval = np.array(h_eq(x_opt / m.x_s[:, None], u_opt / m.u_s[:, None],
                         y_opt / m.y_s[:, None], disc["pv_knot"]).full())
    constraints = {hnames[i]: hval[i, :] for i in range(len(hnames))}

    # ---- assemble data.init ----------------------------------------------
    init = {
        "s_full": disc["s_full"], "k_full": disc["k_full"],
        "x_opt": x_opt, "u_opt": u_opt, "y_opt": y_opt, "xc_opt": xc_opt,
        "x_full": x_full, "u_full": u_full, "y_full": y_full,
        "t_opt": t_opt, "lap_time": float(t_opt[-1]),
        "track": _ns_to_dict(track), "vehicle": vehicle, "constraints": constraints,
        "N": N, "OPT_ds": ctx.OPT_ds, "OPT_d": ctx.OPT_d, "circuit": circuit,
    }
    from types import SimpleNamespace
    ctx.data = SimpleNamespace(init=SimpleNamespace(**init))

    if save:
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"init_{circuit}.mat")
        sio.savemat(save_path, {"data": {"init": init}}, do_compression=True)
        print(f"Saved warm start -> {save_path}")

    print(f"[MLTP_initial] circuit={circuit}  N={N}  "
          f"lap time = {t_opt[-1]:.3f} s  "
          f"(setup {elapsed['setup']:.1f}s, solve {elapsed['solve']:.1f}s)")
    return ctx


if __name__ == "__main__":
    MLTP_initial(circuit="Sturn", vi=60.0)
