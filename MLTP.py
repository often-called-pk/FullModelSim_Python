"""MLTP.py - direct port of MLTP.m

Solves the full Minimum Lap Time Problem with the 23-state model:

    userOpts -> warm start (MLTP_initial or importfile) -> vehModel
    -> build OCP (objective + config-dependent path constraints)
    -> direct-collocation transcription -> IPOPT (MUMPS) -> postprocess -> save .mat

The optimal solution is written to Results/<circuit>_<config>.mat. The saved
struct is self-contained (states, inputs, time, the cartesian racing line,
boundaries, per-tyre forces, lap time) so the racing line can be redrawn without
re-solving. Plotly figures are produced by plotSDI.py.
"""

import os
import time
import numpy as np
import scipy.io as sio
from types import SimpleNamespace

import casadi as ca

from functions.context import Ctx
from functions.importfile import importfile
from functions.transcription import (discretise, build_and_solve_nlp,
                                      unpack_solution, reconstruct_x_full,
                                      interp_inputs, compute_time,
                                      reconstruct_track)
from userOpts import userOpts
from vehModel import vehModel
from MLTP_initial import MLTP_initial


def _interp_to(grid_new, row):
    """Linear interpolation of an init signal (uniform grid) onto N+1 points."""
    row = np.asarray(row, dtype=float).reshape(-1)
    grid_old = np.linspace(0.0, 1.0, row.size)
    return np.interp(grid_new, grid_old, row)


def build_path_constraints(ca, m, pt):
    """Config-dependent path constraints (friction circles + powertrain limits).
    Returns (hnames, h_expr, h_lb, h_ub). Shared by MLTP and the optim variants."""
    def rho(fx, fy, mux, muy, fz):
        return ca.sqrt((fx / (mux * fz))**2 + (fy / (muy * fz))**2)
    rho_lim_fl = rho(m.fx_fl, m.fy_fl, m.mu_fl_x, m.mu_fl_y, m.fz_fl)
    rho_lim_fr = rho(m.fx_fr, m.fy_fr, m.mu_fr_x, m.mu_fr_y, m.fz_fr)
    rho_lim_rl = rho(m.fx_rl, m.fy_rl, m.mu_rl_x, m.mu_rl_y, m.fz_rl)
    rho_lim_rr = rho(m.fx_rr, m.fy_rr, m.mu_rr_x, m.mu_rr_y, m.fz_rr)

    if pt.EM4 == 0:
        BrTh_1 = (m.T_motor_n * m.T_brake_n) / 1e-3
        motor_power = (pt.Pmax - m.Om_motor * m.T_motor) / pt.Pmax
        motor_rpm = (pt.OMmax - m.Om_motor) / pt.OMmax
        if pt.ATD == 0:
            hnames = ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr",
                      "motor_power", "motor_rpm", "BrTh_1"]
            h = ca.vertcat(rho_lim_fl, rho_lim_fr, rho_lim_rl, rho_lim_rr,
                           motor_power, motor_rpm, BrTh_1)
            h_lb = np.array([0, 0, 0, 0, 0, 0, -1.0])
            h_ub = np.array([1, 1, 1, 1, 1, 1, 1.0])
        else:
            ATD_eq = 1 - (m.ATD_FL + m.ATD_FR + m.ATD_RL + m.ATD_RR)
            hnames = ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr",
                      "motor_power", "motor_rpm", "BrTh_1", "ATD_eq"]
            h = ca.vertcat(rho_lim_fl, rho_lim_fr, rho_lim_rl, rho_lim_rr,
                           motor_power, motor_rpm, BrTh_1, ATD_eq)
            h_lb = np.array([0, 0, 0, 0, 0, 0, -1.0, -1e-3])
            h_ub = np.array([1, 1, 1, 1, 1, 1, 1.0, 1e-3])
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
        hnames = ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr",
                  "motor_power_fl", "motor_power_fr", "motor_power_rl", "motor_power_rr",
                  "motor_rpm_fl", "motor_rpm_fr", "motor_rpm_rl", "motor_rpm_rr",
                  "BrTh_fl", "BrTh_fr", "BrTh_rl", "BrTh_rr"]
        h = ca.vertcat(rho_lim_fl, rho_lim_fr, rho_lim_rl, rho_lim_rr,
                       mp_fl, mp_fr, mp_rl, mp_rr, mr_fl, mr_fr, mr_rl, mr_rr,
                       BrTh_fl, BrTh_fr, BrTh_rl, BrTh_rr)
        h_lb = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, -1, -1, -1])
        h_ub = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0, 1, 1, 1])
    assert len(hnames) == h.shape[0], "Number of path constraints not consistent"
    return hnames, h, h_lb, h_ub


def warmstart_guesses(ctx, m, init_x, init_u, N):
    """Warm-start guesses for the 23-state NLP, interpolated from the 7-state
    data.init plus neutral/static seeds for the suspension and tyre states."""
    vp = ctx.vp
    grid = np.linspace(0.0, 1.0, N + 1)
    vx_0 = _interp_to(grid, init_x[0]); vy_0 = _interp_to(grid, init_x[1])
    r_0 = _interp_to(grid, init_x[2]); n_0 = _interp_to(grid, init_x[3])
    eps_0 = _interp_to(grid, init_x[4])
    Om_fl_0 = vx_0 / vp.Rw_f; Om_fr_0 = vx_0 / vp.Rw_f
    Om_rl_0 = vx_0 / vp.Rw_r; Om_rr_0 = vx_0 / vp.Rw_r
    z0 = np.zeros(N + 1)
    zt_fl_0 = (vp.Wfl0 / vp.kt) * np.ones(N + 1)
    zt_fr_0 = (vp.Wfr0 / vp.kt) * np.ones(N + 1)
    zt_rl_0 = (vp.Wrl0 / vp.kt) * np.ones(N + 1)
    zt_rr_0 = (vp.Wrr0 / vp.kt) * np.ones(N + 1)
    x0_phys = np.vstack([vx_0, vy_0, r_0, n_0, eps_0, Om_fl_0, Om_fr_0, Om_rl_0, Om_rr_0,
                         z0, z0, z0, z0, z0, z0, z0, z0, z0, z0,
                         zt_fl_0, zt_fr_0, zt_rl_0, zt_rr_0])
    x0 = x0_phys / m.x_s[:, None]

    T_brake_0 = _interp_to(grid, init_u[1])
    delta_0 = _interp_to(grid, init_u[2])
    T_motor_0 = _interp_to(grid, init_u[0])
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


def MLTP(circuit="Sturn", vi=60.0, ni=np.nan, warm_start=None,
         AeroConfig="Static", ATD="On", Electric_4Motors="Off", TyreModel="CombinedSlip",
         save=True, plot=True, results_dir="Results", **useropts_kwargs):
    t0 = time.time()
    elapsed = {}

    # ---- setup + config ---------------------------------------------------
    ctx = Ctx()
    userOpts(ctx, circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig,
             ATD=ATD, Electric_4Motors=Electric_4Motors, **useropts_kwargs)
    vp, pt = ctx.vp, ctx.pt

    # ---- warm start (data.init) ------------------------------------------
    if warm_start is None:
        ctx_init = MLTP_initial(circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig,
                                ATD=ATD, Electric_4Motors=Electric_4Motors, save=False)
        init = ctx_init.data.init
        init_x = np.asarray(init.x_opt, dtype=float)
        init_u = np.asarray(init.u_opt, dtype=float)
    else:
        loaded = importfile(warm_start)
        init = loaded["data"].init
        init_x = np.asarray(init.x_opt, dtype=float)
        init_u = np.asarray(init.u_opt, dtype=float)
    elapsed["init"] = time.time() - t0

    # ---- full model -------------------------------------------------------
    vehModel(ctx, TyreModel=TyreModel)
    m = ctx.m23

    # ---- OCP: dynamics + objective ---------------------------------------
    L = m.sf
    f_dyn = ca.Function("f_dyn", [m.x, m.u, m.pv], [m.dx, L], ["x", "u", "pv"], ["dx", "L"])
    f_sf = ca.Function("sf", [m.x, m.kappa], [m.sf], ["x", "kappa"], ["sf"])

    # ---- OCP: path constraints (friction circle + powertrain) ------------
    hnames, h, h_lb, h_ub = build_path_constraints(ca, m, pt)
    h_eq = ca.Function("h_eq", [m.x, m.u, m.pv], [h], ["x", "u", "pv"], ["h"])

    # ---- discretisation ---------------------------------------------------
    disc = discretise(ctx.track, ctx.OPT_ds, ctx.OPT_d)
    N = disc["N"]

    # ---- warm-start guesses ----------------------------------------------
    guesses = warmstart_guesses(ctx, m, init_x, init_u, N)

    reg = {"ru": ctx.ru.reshape(-1), "rdu": ctx.rdu.reshape(-1), "rdu2": ctx.rdu2.reshape(-1)}

    # ---- build + solve NLP -----------------------------------------------
    res = build_and_solve_nlp(
        ca, m, f_dyn, f_sf, h_eq, h_lb, h_ub, disc, guesses, reg,
        ctx.duk_lb, ctx.duk_ub, ctx.Xi, ctx.Xf,
        ctx.OPT_d, ctx.OPT_uinter, ctx.OPT_e, ctx.opts)
    sol = res["sol"]
    elapsed["solve"] = time.time() - t0 - elapsed["init"]

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
    ]
    if pt.EM4 == 0:
        veh_syms += [("Om_motor", m.Om_motor), ("P_motor", m.P_motor)]
    else:
        veh_syms += [("P_motor_fl", m.P_motor_fl), ("P_motor_fr", m.P_motor_fr),
                     ("P_motor_rl", m.P_motor_rl), ("P_motor_rr", m.P_motor_rr),
                     ("Om_motor_fl", m.Om_motor_fl), ("Om_motor_fr", m.Om_motor_fr),
                     ("Om_motor_rl", m.Om_motor_rl), ("Om_motor_rr", m.Om_motor_rr)]
    labels = [k for k, _ in veh_syms]
    f_veh = ca.Function("f_veh", [m.x, m.u, m.pv], [s for _, s in veh_syms])
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
    }
    ctx.data = SimpleNamespace(**data)

    if save:
        os.makedirs(results_dir, exist_ok=True)
        cfg = f"{AeroConfig}_ATD{ctx.ATD}_EM4{ctx.Electric_4Motors}"
        out_path = os.path.join(results_dir, f"{circuit}_{cfg}.mat")
        sio.savemat(out_path, {"data": data}, do_compression=True)
        print(f"Saved optimal solution -> {out_path}")

    print(f"[MLTP] circuit={circuit}  config={AeroConfig}/ATD={ctx.ATD}/EM4={ctx.Electric_4Motors}  "
          f"N={N}  lap time = {t_opt[-1]:.3f} s  (init {elapsed['init']:.1f}s, solve {elapsed['solve']:.1f}s)")

    if plot:
        try:
            from plotSDI import plotSDI
            plotSDI(ctx)
        except Exception as exc:    # plotSDI may not exist yet / plotly missing
            print(f"[MLTP] plotting skipped: {exc}")

    return ctx


if __name__ == "__main__":
    MLTP(circuit="Sturn", vi=60.0)
