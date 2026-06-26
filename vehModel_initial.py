"""vehModel_initial.py - direct port of vehModel_initial.m

Builds the simplified 7-state single-track ("bicycle") vehicle model used to
generate a warm start for the full model. All symbolic work uses CasADi.

States  (7): vx, vy, r, n, eps, Om_f, Om_r
Inputs  (3): Tdrive, Tbrake, delta
Aux     (1): ltx        (longitudinal load transfer)

Symbols, scalings, limits, dynamics, and intermediate force/torque expressions
are stored on ``ctx.m7`` (used by MLTP_initial.py for constraints + post-processing).
"""

import numpy as np
import casadi as ca
from types import SimpleNamespace


def vehModel_initial(ctx):
    vp, pt = ctx.vp, ctx.pt
    OPT_e = ctx.OPT_e
    SX = ca.SX

    # ===================== model (A): states ==============================
    nx = 7

    vx_n = SX.sym("vx_n");      vx_s = 100.0;            vx = vx_s * vx_n
    vy_n = SX.sym("vy_n");      vy_s = 10.0;             vy = vy_s * vy_n
    r_n = SX.sym("yawrate_n");  r_s = 1.0;               r = r_s * r_n
    n_n = SX.sym("n_n");        n_s = 5.0;               n = n_s * n_n
    eps_n = SX.sym("eps_n");    eps_s = 1.0;             eps = eps_s * eps_n
    Om_f_n = SX.sym("Om_f_n");  Om_f_s = vx_s / vp.Rw;   Om_f = Om_f_s * Om_f_n
    Om_r_n = SX.sym("Om_r_n");  Om_r_s = vx_s / vp.Rw;   Om_r = Om_r_s * Om_r_n

    # state limits (scaled)
    vx_lim = np.array([OPT_e, pt.Vmax]) / vx_s
    vy_lim = np.array([-10, 10]) / vy_s
    r_lim = np.array([-np.pi / 2, np.pi / 2]) / r_s
    n_lim = np.array([-4, 4]) / n_s
    eps_lim = np.array([-np.pi / 4, np.pi / 4]) / eps_s
    Om_f_lim = np.array([0, pt.Vmax / vp.Rw]) / Om_f_s
    Om_r_lim = np.array([0, pt.Vmax / vp.Rw]) / Om_r_s

    x_s = np.array([vx_s, vy_s, r_s, n_s, eps_s, Om_f_s, Om_r_s])
    x = ca.vertcat(vx_n, vy_n, r_n, n_n, eps_n, Om_f_n, Om_r_n)
    x_lim = np.vstack([vx_lim, vy_lim, r_lim, n_lim, eps_lim, Om_f_lim, Om_r_lim])
    x_min = x_lim[:, 0]
    x_max = x_lim[:, 1]
    assert nx == x.shape[0], "Number of states is not consistent"

    # ===================== model (A): inputs ==============================
    nu = 3
    Tdrive_n = SX.sym("T_drive_n"); Tdrive_s = pt.Tmax;        Tdrive = Tdrive_s * Tdrive_n
    Tbrake_n = SX.sym("T_brake_n"); Tbrake_s = vp.Tbrake_max;  Tbrake = Tbrake_s * Tbrake_n
    delta_n = SX.sym("delta_n");    delta_s = np.pi / 8;       delta = delta_s * delta_n

    Tdrive_lim = np.array([0, pt.Tmax]) / Tdrive_s
    Tbrake_lim = np.array([-vp.Tbrake_max, 0]) / Tbrake_s
    delta_lim = np.array([-np.pi / 4, np.pi / 4]) / delta_s

    u_s = np.array([Tdrive_s, Tbrake_s, delta_s])
    u = ca.vertcat(Tdrive_n, Tbrake_n, delta_n)
    u_lim = np.vstack([Tdrive_lim, Tbrake_lim, delta_lim])
    u_min = u_lim[:, 0]
    u_max = u_lim[:, 1]
    assert nu == u.shape[0], "Number of inputs is not consistent"

    # rate-of-input limits for the init program, scaled by u_s
    duk_ub_init = ctx.duk_ub_init.reshape(-1) / u_s
    duk_lb_init = ctx.duk_lb_init.reshape(-1) / u_s

    # ===================== model (A): aux variable ========================
    ny = 1
    ltx_n = SX.sym("loadTransferX_n")
    ltx_s = vp.m * vp.g * vp.hcg / vp.l
    ltx = ltx_s * ltx_n

    ltx_lim = np.array([-vp.m * vp.g, vp.m * vp.g]) / ltx_s
    y_s = np.array([ltx_s])
    y = ca.vertcat(ltx_n)
    y_lim = np.vstack([ltx_lim])
    y_min = y_lim[:, 0]
    y_max = y_lim[:, 1]
    assert ny == y.shape[0], "Number of aux. variables is not consistent"

    # variable parameter: curvature
    kappa = SX.sym("kappa")     # kappa > 0 for left turns
    pv = ca.vertcat(kappa)

    # ===================== model (B): equations ===========================
    # aerodynamic forces [N] - initialise with max aero settings
    Cd_max = float(np.max(np.atleast_1d(vp.Cd)))
    Cl_max = float(np.max(np.atleast_1d(vp.Cl)))
    f_drag = 0.5 * vp.rho * vp.A * vx**2 * Cd_max
    f_lift = 0.5 * vp.rho * vp.A * vx**2 * Cl_max

    # slip angles [rad]
    sa_f = delta - ca.atan((vp.l_f * r + vy) / vx)
    sa_r = ca.atan((vp.l_r * r - vy) / vx)

    # slip ratio
    v_f = ca.sqrt((vy + vp.l_f * r)**2 + vx**2)
    v_r = ca.sqrt((vy - vp.l_r * r)**2 + vx**2)
    v_fx = v_f * ca.cos(sa_f)
    v_rx = v_r * ca.cos(sa_r)
    sx_f = (vp.Rw * Om_f - v_fx) / v_fx
    sx_r = (vp.Rw * Om_r - v_rx) / v_rx

    # vertical tyre forces [N] (long. load transfer: accel +, brake -)
    fz_f = (vp.Wfl0 + vp.Wfr0) - ltx + f_lift * 0.45
    fz_r = (vp.Wrl0 + vp.Wrr0) + ltx + f_lift * 0.55

    f_roll_f = vp.f * fz_f
    f_roll_r = vp.f * fz_r

    # friction coefficient (2x nominal because tyres are combined)
    mu_f = vp.tyre.mu + vp.tyre.pD2 * (fz_f - (2 * vp.Fz0)) / (2 * vp.Fz0)
    mu_r = vp.tyre.mu + vp.tyre.pD2 * (fz_r - (2 * vp.Fz0)) / (2 * vp.Fz0)

    # longitudinal tyre forces (Magic Formula)
    Dx_f = mu_f * fz_f
    Dx_r = mu_r * fz_r
    bx, cx, ex = vp.tyre.bx, vp.tyre.cx, vp.tyre.ex
    fx_f = Dx_f * ca.sin(cx * ca.atan(bx * sx_f - ex * (bx * sx_f - ca.atan(bx * sx_f))))
    fx_r = Dx_r * ca.sin(cx * ca.atan(bx * sx_r - ex * (bx * sx_r - ca.atan(bx * sx_r))))

    # lateral tyre forces (Magic Formula)
    Dy_f = mu_f * fz_f
    Dy_r = mu_r * fz_r
    by, cy, ey = vp.tyre.by, vp.tyre.cy, vp.tyre.ey
    fy_f = Dy_f * ca.sin(cy * ca.atan(by * sa_f - ey * (by * sa_f - ca.atan(by * sa_f))))
    fy_r = Dy_r * ca.sin(cy * ca.atan(by * sa_r - ey * (by * sa_r - ca.atan(by * sa_r))))

    # wheel torque
    T_f = vp.gear * Tdrive * (1 - vp.Tdist) + 2 * Tbrake * vp.brkB
    T_r = vp.gear * Tdrive * vp.Tdist + 2 * Tbrake * (1 - vp.brkB)

    Om_motor = (Om_f) / 2 * vp.gear + (Om_r) / 2 * vp.gear
    P_motor = Tdrive * Om_motor

    # change of independent variable
    sf = (1 - n * kappa) / (vx * ca.cos(eps) - vy * ca.sin(eps))

    # ===================== model (B): state derivatives ===================
    # NOTE the 0.85 scaling factors on the lateral forces, exactly as in MATLAB.
    dvx = (fx_r + fx_f * ca.cos(delta) - 0.85 * fy_f * ca.sin(delta)
           + vp.m * vy * r - f_drag - f_roll_f - f_roll_r) * sf / vp.m
    dvy = (0.85 * fy_r + 0.85 * fy_f * ca.cos(delta) + fx_f * ca.sin(delta)
           - vp.m * vx * r) * sf / vp.m
    dr = (0.85 * fy_f * ca.cos(delta) * vp.l_f + fx_f * ca.sin(delta) * vp.l_f
          - 0.85 * fy_r * vp.l_r) * sf / vp.I_z
    dn = (vx * ca.sin(eps) + vy * ca.cos(eps)) * sf
    deps = sf * r - kappa
    dOm_f = sf * (T_f - fx_f * vp.Rw) / vp.Jw
    dOm_r = sf * (T_r - fx_r * vp.Rw) / vp.Jw

    dx = ca.vertcat(dvx, dvy, dr, dn, deps, dOm_f, dOm_r) / ca.DM(x_s)

    # ---- expose everything on ctx.m7 -------------------------------------
    m = SimpleNamespace()
    # symbols & sizes
    m.nx, m.nu, m.ny = nx, nu, ny
    m.x, m.u, m.y, m.pv, m.kappa = x, u, y, pv, kappa
    m.dx, m.sf = dx, sf
    m.x_s, m.u_s, m.y_s = x_s, u_s, y_s
    m.x_min, m.x_max = x_min, x_max
    m.u_min, m.u_max = u_min, u_max
    m.y_min, m.y_max = y_min, y_max
    m.duk_ub_init, m.duk_lb_init = duk_ub_init, duk_lb_init
    # scalings used by constraints
    m.Tdrive_s, m.Om_f_s, m.Om_r_s, m.ltx_s = Tdrive_s, Om_f_s, Om_r_s, ltx_s
    # raw signals
    m.vx, m.vy, m.r, m.n, m.eps = vx, vy, r, n, eps
    m.Om_f, m.Om_r = Om_f, Om_r
    m.Tdrive, m.Tbrake, m.delta = Tdrive, Tbrake, delta
    m.Tdrive_n, m.Tbrake_n, m.delta_n = Tdrive_n, Tbrake_n, delta_n
    m.ltx = ltx
    # forces / torques / aux for constraints & postprocessing
    m.f_drag, m.f_lift = f_drag, f_lift
    m.sa_f, m.sa_r, m.sx_f, m.sx_r = sa_f, sa_r, sx_f, sx_r
    m.fz_f, m.fz_r = fz_f, fz_r
    m.mu_f, m.mu_r = mu_f, mu_r
    m.fx_f, m.fx_r, m.fy_f, m.fy_r = fx_f, fx_r, fy_f, fy_r
    m.T_f, m.T_r, m.Om_motor, m.P_motor = T_f, T_r, Om_motor, P_motor
    ctx.m7 = m
    return ctx
