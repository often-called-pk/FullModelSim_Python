"""vehModel.py - direct port of vehModel.m (FullModel_4EM_Suspension_FullTyre)

The full 23-state vehicle model: four wheels, sprung heave/pitch/roll + four
unsprung-mass and four tyre-deflection states, full Pacejka 5.2 Magic Formula
(pure + combined slip, both axes, with camber), active aerodynamics, and the
single-motor / ATD / four-motor (EM4) powertrain branches.

The selected tyre coefficient set is read from ``ctx.mf`` (Copy B, per the user's
confirmation), the DATA_AA aero polynomials from ``ctx.aero``, and the linear
camber-gain coefficients from ``ctx.cg``. Hard-coded model switches in the MATLAB
(``Steering='NA'``, ``CamberGain='Off'``, ``TyreModel='CombinedSlip'``) are exposed
as arguments with the same defaults.

There are NO aux variables (ny = 0): the load transfers are produced by the
suspension/tyre states, not by algebraic decision variables.

Everything needed by MLTP.py (symbols, sizes, scalings, limits, dynamics, and the
intermediate signals for path constraints and post-processing) is stored on
``ctx.m23`` (a namespace).
"""

import numpy as np
import casadi as ca
from types import SimpleNamespace


def _poly(coeffs, x, powers):
    """Evaluate sum_i coeffs[i] * x**powers[i]  (coeffs 0-indexed numpy array)."""
    out = 0
    for i, p in enumerate(powers):
        out = out + float(coeffs[i]) * x**p
    return out


def vehModel(ctx, Steering="NA", CamberGain="Off", TyreModel="CombinedSlip"):
    vp, pt, mf = ctx.vp, ctx.pt, ctx.mf
    aero = ctx.aero
    cg = ctx.cg
    OPT_e = ctx.OPT_e
    SX = ca.SX
    deg2rad = np.pi / 180.0
    if aero is None:
        raise RuntimeError("DATA_AA.mat must be loaded (ctx.aero) to build the full model.")

    # ===================== model (A): states (nx = 23) ====================
    nx = 23

    def state(name, scale):
        s = SX.sym(name)
        return s, scale, scale * s

    vx_n, vx_s, vx = state("vx_n", 100.0)
    vy_n, vy_s, vy = state("vy_n", 10.0)
    r_n, r_s, r = state("yawrate_n", 1.0)
    n_n, n_s, n = state("n_n", 5.0)
    eps_n, eps_s, eps = state("eps_n", 1.0)
    Om_fl_n, Om_fl_s, Om_fl = state("Om_fl_n", vx_s / vp.Rw_f)
    Om_fr_n, Om_fr_s, Om_fr = state("Om_fr_n", vx_s / vp.Rw_f)
    Om_rl_n, Om_rl_s, Om_rl = state("Om_rl_n", vx_s / vp.Rw_r)
    Om_rr_n, Om_rr_s, Om_rr = state("Om_rr_n", vx_s / vp.Rw_r)
    zs_n, zs_s, zs = state("zs_n", 1.0)
    zsdot_n, zsdot_s, zsdot = state("zsdot_n", 1.0)
    theta_n, theta_s, theta = state("theta_n", 1.0)
    thetadot_n, thetadot_s, thetadot = state("thetadot_n", 1.0)
    phi_n, phi_s, phi = state("phi_n", 1.0)
    phidot_n, phidot_s, phidot = state("phidot_n", 1.0)
    wu_fl_n, wu_fl_s, wu_fl = state("wu_fl_n", 1.0)
    wu_fr_n, wu_fr_s, wu_fr = state("wu_fr_n", 1.0)
    wu_rl_n, wu_rl_s, wu_rl = state("wu_rl_n", 1.0)
    wu_rr_n, wu_rr_s, wu_rr = state("wu_rr_n", 1.0)
    zt_fl_n, zt_fl_s, zt_fl = state("zt_fl_n", 1.0)
    zt_fr_n, zt_fr_s, zt_fr = state("zt_fr_n", 1.0)
    zt_rl_n, zt_rl_s, zt_rl = state("zt_rl_n", 1.0)
    zt_rr_n, zt_rr_s, zt_rr = state("zt_rr_n", 1.0)

    # state limits (scaled)
    x_lim = np.array([
        [OPT_e, pt.Vmax], [-10, 10], [-np.pi/2, np.pi/2], [-4, 4], [-np.pi/4, np.pi/4],
        [OPT_e/vp.Rw_f, pt.Vmax/vp.Rw_f], [OPT_e/vp.Rw_f, pt.Vmax/vp.Rw_f],
        [OPT_e/vp.Rw_r, pt.Vmax/vp.Rw_r], [OPT_e/vp.Rw_r, pt.Vmax/vp.Rw_r],
        [-0.1, 0.1], [-10, 10], [-0.1, 0.1], [-10, 10], [-0.09, 0.09], [-10, 10],
        [-1, 1], [-1, 1], [-1, 1], [-1, 1],
        [0, 0.1], [0, 0.1], [0, 0.1], [0, 0.1]], dtype=float)
    x_s = np.array([vx_s, vy_s, r_s, n_s, eps_s, Om_fl_s, Om_fr_s, Om_rl_s, Om_rr_s,
                    zs_s, zsdot_s, theta_s, thetadot_s, phi_s, phidot_s,
                    wu_fl_s, wu_fr_s, wu_rl_s, wu_rr_s, zt_fl_s, zt_fr_s, zt_rl_s, zt_rr_s])
    x_lim = x_lim / x_s[:, None]
    x_min, x_max = x_lim[:, 0], x_lim[:, 1]

    x = ca.vertcat(vx_n, vy_n, r_n, n_n, eps_n, Om_fl_n, Om_fr_n, Om_rl_n, Om_rr_n,
                   zs_n, zsdot_n, theta_n, thetadot_n, phi_n, phidot_n,
                   wu_fl_n, wu_fr_n, wu_rl_n, wu_rr_n, zt_fl_n, zt_fr_n, zt_rl_n, zt_rr_n)
    assert nx == x.shape[0], "Number of states is not consistent"

    # ===================== model (A): inputs (config-dependent) ===========
    # motor torque(s)
    if pt.EM4 == 0:
        T_motor_n = SX.sym("T_motor_n"); T_motor_s = pt.Tmax; T_motor = T_motor_s * T_motor_n
        T_motor_lim = np.array([0, pt.Tmax]) / T_motor_s
    else:
        T_motor_fl_n = SX.sym("T_motor_fl_n"); T_motor_fl_s = pt.Tmax; T_motor_fl = T_motor_fl_s * T_motor_fl_n
        T_motor_fr_n = SX.sym("T_motor_fr_n"); T_motor_fr_s = pt.Tmax; T_motor_fr = T_motor_fr_s * T_motor_fr_n
        T_motor_rl_n = SX.sym("T_motor_rl_n"); T_motor_rl_s = pt.Tmax; T_motor_rl = T_motor_rl_s * T_motor_rl_n
        T_motor_rr_n = SX.sym("T_motor_rr_n"); T_motor_rr_s = pt.Tmax; T_motor_rr = T_motor_rr_s * T_motor_rr_n
        T_motor_lim = np.array([0, pt.Tmax]) / pt.Tmax

    # brake torque
    T_brake_n = SX.sym("T_brake_n"); T_brake_s = vp.Tbrake_max; T_brake = T_brake_s * T_brake_n
    T_brake_lim = np.array([-vp.Tbrake_max, 0]) / T_brake_s

    # steering angle
    delta_max = 35.0 * deg2rad
    delta_n = SX.sym("delta_n"); delta = delta_max * delta_n
    delta_lim = np.array([-1.0, 1.0])

    # ATD
    if pt.ATD == 1:
        ATD_FL_n = SX.sym("ATD_FL_n"); ATD_FL = ATD_FL_n
        ATD_FR_n = SX.sym("ATD_FR_n"); ATD_FR = ATD_FR_n
        ATD_RL_n = SX.sym("ATD_RL_n"); ATD_RL = ATD_RL_n
        ATD_RR_n = SX.sym("ATD_RR_n"); ATD_RR = ATD_RR_n
        ATD_lim = np.array([0.0, 1.0])

    # active aero inputs (override the fixed wing angles where active)
    aFL, aFR, aRW, aTW = vp.alpha_FL, vp.alpha_FR, vp.alpha_RW, vp.alpha_TW
    if vp.ActAero == 1:
        activeAeroRW_n = SX.sym("activeAeroRW_n"); activeAeroRW_s = 30.0
        activeAeroRW = activeAeroRW_s * activeAeroRW_n; activeAeroRW_lim = np.array([0, 30]) / activeAeroRW_s
        aRW = activeAeroRW
    elif vp.ActAero == 2:
        activeAeroFW_n = SX.sym("activeAeroFW_n"); activeAeroFW_s = 10.0
        activeAeroFW = activeAeroFW_s * activeAeroFW_n; activeAeroFW_lim = np.array([0, 10]) / activeAeroFW_s
        activeAeroRW_n = SX.sym("activeAeroRW_n"); activeAeroRW_s = 30.0
        activeAeroRW = activeAeroRW_s * activeAeroRW_n; activeAeroRW_lim = np.array([0, 30]) / activeAeroRW_s
        aFL = aFR = activeAeroFW; aRW = activeAeroRW
    elif vp.ActAero == 3:
        activeAeroFL_n = SX.sym("activeAeroFL_n"); activeAeroFL_s = 10.0
        activeAeroFL = activeAeroFL_s * activeAeroFL_n; activeAeroFL_lim = np.array([0, 10]) / activeAeroFL_s
        activeAeroFR_n = SX.sym("activeAeroFR_n"); activeAeroFR_s = 10.0
        activeAeroFR = activeAeroFR_s * activeAeroFR_n; activeAeroFR_lim = np.array([0, 10]) / activeAeroFR_s
        activeAeroRW_n = SX.sym("activeAeroRW_n"); activeAeroRW_s = 30.0
        activeAeroRW = activeAeroRW_s * activeAeroRW_n; activeAeroRW_lim = np.array([0, 30]) / activeAeroRW_s
        activeAeroTW_n = SX.sym("activeAeroTW_n"); activeAeroTW_s = 12.0
        activeAeroTW = activeAeroTW_s * activeAeroTW_n; activeAeroTW_lim = np.array([-12, 12]) / activeAeroTW_s
        aFL = activeAeroFL; aFR = activeAeroFR; aRW = activeAeroRW; aTW = activeAeroTW

    # assemble u, u_s, u_lim in the MATLAB order: motors, brake, [ATD x4], aero, delta
    u_list = []   # (sym_norm, scale, lim_row)
    if pt.EM4 == 0:
        u_list.append((T_motor_n, T_motor_s, T_motor_lim))
    else:
        u_list += [(T_motor_fl_n, T_motor_fl_s, T_motor_lim), (T_motor_fr_n, T_motor_fr_s, T_motor_lim),
                   (T_motor_rl_n, T_motor_rl_s, T_motor_lim), (T_motor_rr_n, T_motor_rr_s, T_motor_lim)]
    u_list.append((T_brake_n, T_brake_s, T_brake_lim))
    if pt.ATD == 1:
        u_list += [(ATD_FL_n, 1.0, ATD_lim), (ATD_FR_n, 1.0, ATD_lim),
                   (ATD_RL_n, 1.0, ATD_lim), (ATD_RR_n, 1.0, ATD_lim)]
    if vp.ActAero == 1:
        u_list.append((activeAeroRW_n, activeAeroRW_s, activeAeroRW_lim))
    elif vp.ActAero == 2:
        u_list += [(activeAeroFW_n, activeAeroFW_s, activeAeroFW_lim),
                   (activeAeroRW_n, activeAeroRW_s, activeAeroRW_lim)]
    elif vp.ActAero == 3:
        u_list += [(activeAeroFL_n, activeAeroFL_s, activeAeroFL_lim),
                   (activeAeroFR_n, activeAeroFR_s, activeAeroFR_lim),
                   (activeAeroRW_n, activeAeroRW_s, activeAeroRW_lim),
                   (activeAeroTW_n, activeAeroTW_s, activeAeroTW_lim)]
    u_list.append((delta_n, delta_max, delta_lim))

    u = ca.vertcat(*[s for (s, _, _) in u_list])
    u_s = np.array([sc for (_, sc, _) in u_list], dtype=float)
    u_lim = np.vstack([lr for (_, _, lr) in u_list])
    u_min, u_max = u_lim[:, 0], u_lim[:, 1]
    nu = u.shape[0]

    # ===================== variable parameter ==============================
    kappa = SX.sym("kappa")
    pv = ca.vertcat(kappa)

    # ===================== aerodynamic coefficients ========================
    # front wing left/right (linear in angle)
    FW_L_Cl_front = float(aero.FW_L.Cl_front) * aFL
    FW_L_Cl_rear = float(aero.FW_L.Cl_rear) * aFL
    FW_L_Cl_left = FW_L_Cl_front + FW_L_Cl_rear
    FW_R_Cl_front = float(aero.FW_R.Cl_front) * aFR
    FW_R_Cl_rear = float(aero.FW_R.Cl_rear) * aFR
    FW_R_Cl_right = FW_R_Cl_front + FW_R_Cl_rear
    # rear wing (polynomials in angle of attack)
    RW_Cl_front = _poly(np.atleast_1d(aero.RW.Cl_front), aRW, [3, 2, 1])
    RW_Cl_rear = _poly(np.atleast_1d(aero.RW.Cl_rear), aRW, [5, 4, 3, 2, 1])
    RW_Cl_left = 0.5 * (RW_Cl_front + RW_Cl_rear)
    RW_Cl_right = 0.5 * (RW_Cl_front + RW_Cl_rear)
    RW_Cd = _poly(np.atleast_1d(aero.RW.Cd), aRW, [2, 1])
    # rear wing tilt
    TW_Cl_left = _poly(np.atleast_1d(aero.TW.Cl_left), aTW, [2, 1])
    TW_Cl_right = _poly(np.atleast_1d(aero.TW.Cl_right), aTW, [2, 1])
    TW_Cl_rear = TW_Cl_left + TW_Cl_right
    TW_Cs_rear = float(aero.TW.Cs_rear) * aTW

    Cl_front = vp.Cl0_front + FW_L_Cl_front + FW_R_Cl_front + RW_Cl_front
    Cl_rear = vp.Cl0_rear + FW_L_Cl_rear + FW_R_Cl_rear + RW_Cl_rear + TW_Cl_rear
    Cl_left = vp.Cl0_left + FW_L_Cl_left + RW_Cl_left + TW_Cl_left
    Cl_right = vp.Cl0_right + FW_R_Cl_right + RW_Cl_right + TW_Cl_right
    Cd = vp.Cd0 + RW_Cd
    Cs_front = vp.Cs0_front
    Cs_rear = vp.Cs0_rear + TW_Cs_rear

    Cl = Cl_front + Cl_rear
    Cl_fl = Cl_front * (Cl_left / Cl)
    Cl_fr = Cl_front * (Cl_right / Cl)
    Cl_rl = Cl_rear * (Cl_left / Cl)
    Cl_rr = Cl_rear * (Cl_right / Cl)

    rollCoeff = vp.f

    # aerodynamic forces
    f_lift_fl = -0.5 * vp.rho * Cl_fl * vp.A * vx**2
    f_lift_fr = -0.5 * vp.rho * Cl_fr * vp.A * vx**2
    f_lift_rl = -0.5 * vp.rho * Cl_rl * vp.A * vx**2
    f_lift_rr = -0.5 * vp.rho * Cl_rr * vp.A * vx**2
    f_lift = f_lift_fl + f_lift_fr + f_lift_rl + f_lift_rr
    f_drag = 0.5 * vp.rho * vp.A * vx**2 * Cd
    f_side_fl = -0.5 * (0.5 * vp.rho * Cs_front * vp.A * vx**2)
    f_side_fr = -0.5 * (0.5 * vp.rho * Cs_front * vp.A * vx**2)
    f_side_rl = -0.5 * (0.5 * vp.rho * Cs_rear * vp.A * vx**2)
    f_side_rr = -0.5 * (0.5 * vp.rho * Cs_rear * vp.A * vx**2)
    f_side = f_side_fl + f_side_fr + f_side_rl + f_side_rr

    # ===================== suspension kinematics & forces =================
    xsfl_dot = wu_fl - (-vp.l_f * thetadot + (vp.t/2) * phidot + zsdot)
    xsfr_dot = wu_fr - (-vp.l_f * thetadot - (vp.t/2) * phidot + zsdot)
    xsrl_dot = wu_rl - (+vp.l_r * thetadot + (vp.t/2) * phidot + zsdot)
    xsrr_dot = wu_rr - (+vp.l_r * thetadot - (vp.t/2) * phidot + zsdot)

    xs_fl = -(zs - vp.l_f * ca.sin(theta) + (vp.t/2) * ca.sin(phi))
    xs_fr = -(zs - vp.l_f * ca.sin(theta) - (vp.t/2) * ca.sin(phi))
    xs_rl = -(zs + vp.l_r * ca.sin(theta) + (vp.t/2) * ca.sin(phi))
    xs_rr = -(zs + vp.l_r * ca.sin(theta) - (vp.t/2) * ca.sin(phi))

    def _cap(xs):
        return ca.if_else(xs >= 0.075, 0.075, ca.if_else(xs <= -0.05, -0.05, xs))
    xs_fl, xs_fr, xs_rl, xs_rr = _cap(xs_fl), _cap(xs_fr), _cap(xs_rl), _cap(xs_rr)

    fzs_fl = vp.k_fl * (xs_fl + vp.xsi_fl) + vp.c_fl * xsfl_dot + f_lift_fl
    fzs_fr = vp.k_fr * (xs_fr + vp.xsi_fr) + vp.c_fr * xsfr_dot + f_lift_fr
    fzs_rl = vp.k_rl * (xs_rl + vp.xsi_rl) + vp.c_rl * xsrl_dot + f_lift_rl
    fzs_rr = vp.k_rr * (xs_rr + vp.xsi_rr) + vp.c_rr * xsrr_dot + f_lift_rr
    fzs_fl = ca.if_else(xs_fl >= 0.075, 0, fzs_fl)
    fzs_fr = ca.if_else(xs_fr >= 0.075, 0, fzs_fr)
    fzs_rl = ca.if_else(xs_rl >= 0.075, 0, fzs_rl)
    fzs_rr = ca.if_else(xs_rr >= 0.075, 0, fzs_rr)

    fzt_fl = zt_fl * vp.kt
    fzt_fr = zt_fr * vp.kt
    fzt_rl = zt_rl * vp.kt
    fzt_rr = zt_rr * vp.kt
    fz_fl, fz_fr, fz_rl, fz_rr = fzt_fl, fzt_fr, fzt_rl, fzt_rr

    # ===================== tyre slip & loads ==============================
    if Steering == "NA":
        sa_fl = delta - ca.atan((vp.l_f*r + vy) / (vx - r*vp.t/2)) + vp.toe_front_rad
        sa_fr = delta - ca.atan((vp.l_f*r + vy) / (vx + r*vp.t/2)) - vp.toe_front_rad
    elif Steering == "Ackermann":
        delta_fl = ca.atan(vp.l_f / (vp.l_f / ca.tan(delta) - vp.t/2))
        delta_fr = ca.atan(vp.l_f / (vp.l_f / ca.tan(delta) + vp.t/2))
        sa_fl = delta_fl - ca.atan((vp.l_f*r + vy) / (vx - r*vp.t/2)) + vp.toe_rear_rad
        sa_fr = delta_fr - ca.atan((vp.l_f*r + vy) / (vx + r*vp.t/2)) - vp.toe_rear_rad
    else:
        raise ValueError("Steering must be 'NA' or 'Ackermann'")
    sa_rl = ca.atan((vp.l_r*r - vy) / (vx - r*vp.t/2)) + vp.toe_rear_rad
    sa_rr = ca.atan((vp.l_r*r - vy) / (vx + r*vp.t/2)) - vp.toe_front_rad

    v_fl = ca.sqrt((vy + vp.l_f*r)**2 + (vx - r*vp.t/2)**2)
    v_fr = ca.sqrt((vy + vp.l_f*r)**2 + (vx + r*vp.t/2)**2)
    v_rl = ca.sqrt((vy - vp.l_r*r)**2 + (vx - r*vp.t/2)**2)
    v_rr = ca.sqrt((vy - vp.l_r*r)**2 + (vx + r*vp.t/2)**2)
    v_flx = v_fl * ca.cos(sa_fl)
    v_frx = v_fr * ca.cos(sa_fr)
    v_rlx = v_rl * ca.cos(sa_rl)
    v_rrx = v_rr * ca.cos(sa_rr)
    sx_fl = (vp.Rw_f*Om_fl - v_flx) / v_flx
    sx_fr = (vp.Rw_f*Om_fr - v_frx) / v_frx
    sx_rl = (vp.Rw_r*Om_rl - v_rlx) / v_rlx
    sx_rr = (vp.Rw_r*Om_rr - v_rrx) / v_rrx

    dfz_fl = (fz_fl - vp.Fz0) / vp.Fz0
    dfz_fr = (fz_fr - vp.Fz0) / vp.Fz0
    dfz_rl = (fz_rl - vp.Fz0) / vp.Fz0
    dfz_rr = (fz_rr - vp.Fz0) / vp.Fz0
    fz_fl_shift = fz_fl / vp.Fz0_shift
    fz_fr_shift = fz_fr / vp.Fz0_shift
    fz_rl_shift = fz_rl / vp.Fz0_shift
    fz_rr_shift = fz_rr / vp.Fz0_shift

    # ===================== camber gain & dynamic camber ===================
    if CamberGain == "Table":
        zs_vals = vp.CG_h_deg_per_mm_table[0::2]; CG_h_vals = vp.CG_h_deg_per_mm_table[1::2]
        phi_vals = vp.CG_r_deg_per_deg_table[0::2]; CG_r_vals = vp.CG_r_deg_per_deg_table[1::2]
        theta_vals = vp.CG_p_deg_per_deg_table[0::2]; CG_p_vals = vp.CG_p_deg_per_deg_table[1::2]
        CG_h = _linear_interp(ca, zs, zs_vals, CG_h_vals) * (np.pi/180) * 1000
        CG_r = _linear_interp(ca, phi, phi_vals, CG_r_vals)
        CG_p = _linear_interp(ca, theta, theta_vals, CG_p_vals)
    elif CamberGain == "Linear":
        CG_h = cg.CG_h_deg_per_mm_linear * (np.pi/180) * 1000
        CG_r = cg.CG_r_deg_per_deg_linear
        CG_p = cg.CG_p_deg_per_deg_linear
    elif CamberGain == "Off":
        CG_h = CG_r = CG_p = 0
    else:
        raise ValueError("CamberGain must be 'Table', 'Linear' or 'Off'")

    delta_gamma_fl = CG_h * zs + CG_r * phi + CG_p * theta
    delta_gamma_fr = CG_h * zs - CG_r * phi + CG_p * theta
    delta_gamma_rl = CG_h * zs + CG_r * phi - CG_p * theta
    delta_gamma_rr = CG_h * zs - CG_r * phi - CG_p * theta
    dynamic_camber_fl = vp.gamma_fl_rad + delta_gamma_fl
    dynamic_camber_fr = vp.gamma_fr_rad + delta_gamma_fr
    dynamic_camber_rl = vp.gamma_rl_rad + delta_gamma_rl
    dynamic_camber_rr = vp.gamma_rr_rad + delta_gamma_rr

    # ===================== Pacejka 5.2: longitudinal (Fx) =================
    Cx = mf.pCx1
    mu_fl_x = mf.pDx1 + mf.pDx2 * dfz_fl
    mu_fr_x = mf.pDx1 + mf.pDx2 * dfz_fr
    mu_rl_x = mf.pDx1 + mf.pDx2 * dfz_rl
    mu_rr_x = mf.pDx1 + mf.pDx2 * dfz_rr
    Dx_fl = mu_fl_x * fz_fl_shift; Dx_fr = mu_fr_x * fz_fr_shift
    Dx_rl = mu_rl_x * fz_rl_shift; Dx_rr = mu_rr_x * fz_rr_shift

    def _shx(dfz): return mf.pHx1 + mf.pHx2 * dfz
    kx_fl = sx_fl + _shx(dfz_fl); kx_fr = sx_fr + _shx(dfz_fr)
    kx_rl = sx_rl + _shx(dfz_rl); kx_rr = sx_rr + _shx(dfz_rr)

    def _ex(dfz, kx): return (mf.pEx1 + mf.pEx2*dfz + mf.pEx3*dfz**2) * (1 - mf.pEx4*ca.sign(kx))
    Ex_fl = _ex(dfz_fl, kx_fl); Ex_fr = _ex(dfz_fr, kx_fr)
    Ex_rl = _ex(dfz_rl, kx_rl); Ex_rr = _ex(dfz_rr, kx_rr)

    def _kxk(fzs, dfz): return fzs * (mf.pKx1 + mf.pKx2*dfz) * ca.exp(mf.pKx3*dfz)
    Kxk_fl = _kxk(fz_fl_shift, dfz_fl); Kxk_fr = _kxk(fz_fr_shift, dfz_fr)
    Kxk_rl = _kxk(fz_rl_shift, dfz_rl); Kxk_rr = _kxk(fz_rr_shift, dfz_rr)
    Bx_fl = Kxk_fl / (Cx*Dx_fl + vp.eps_x); Bx_fr = Kxk_fr / (Cx*Dx_fr + vp.eps_x)
    Bx_rl = Kxk_rl / (Cx*Dx_rl + vp.eps_x); Bx_rr = Kxk_rr / (Cx*Dx_rr + vp.eps_x)

    def _svx(fzs, dfz): return fzs * (mf.pVx1 + mf.pVx2*dfz)
    Svx_fl = _svx(fz_fl_shift, dfz_fl); Svx_fr = _svx(fz_fr_shift, dfz_fr)
    Svx_rl = _svx(fz_rl_shift, dfz_rl); Svx_rr = _svx(fz_rr_shift, dfz_rr)

    def _fx0(D, B, k, E, Sv): return D * ca.sin(Cx * ca.atan(B*k - E*(B*k - ca.atan(B*k)))) + Sv
    fx_fl_0 = _fx0(Dx_fl, Bx_fl, kx_fl, Ex_fl, Svx_fl)
    fx_fr_0 = _fx0(Dx_fr, Bx_fr, kx_fr, Ex_fr, Svx_fr)
    fx_rl_0 = _fx0(Dx_rl, Bx_rl, kx_rl, Ex_rl, Svx_rl)
    fx_rr_0 = _fx0(Dx_rr, Bx_rr, kx_rr, Ex_rr, Svx_rr)

    # combined-slip longitudinal weighting
    Cxa = mf.rCx1; Shxa = mf.rHx1
    def _exa(dfz): return mf.rEx1 + mf.rEx2*dfz
    def _bxa(gam, sx): return (mf.rBx1 + mf.rBx3*gam**2) * ca.cos(ca.atan(mf.rBx2*sx))
    def _gxa(B, E, aS, B0, E0):
        G0 = ca.cos(Cxa*ca.atan(B0*Shxa - E0*(B0*Shxa - ca.atan(B0*Shxa))))
        return ca.cos(Cxa*ca.atan(B*aS - E*(B*aS - ca.atan(B*aS)))) / G0
    Bxa_fl = _bxa(dynamic_camber_fl, sx_fl); Bxa_fr = _bxa(dynamic_camber_fr, sx_fr)
    Bxa_rl = _bxa(dynamic_camber_rl, sx_rl); Bxa_rr = _bxa(dynamic_camber_rr, sx_rr)
    Exa_fl = _exa(dfz_fl); Exa_fr = _exa(dfz_fr); Exa_rl = _exa(dfz_rl); Exa_rr = _exa(dfz_rr)
    aS_fl = sa_fl + Shxa; aS_fr = sa_fr + Shxa; aS_rl = sa_rl + Shxa; aS_rr = sa_rr + Shxa
    Gxa_fl = _gxa(Bxa_fl, Exa_fl, aS_fl, Bxa_fl, Exa_fl)
    Gxa_fr = _gxa(Bxa_fr, Exa_fr, aS_fr, Bxa_fr, Exa_fr)
    Gxa_rl = _gxa(Bxa_rl, Exa_rl, aS_rl, Bxa_rl, Exa_rl)
    Gxa_rr = _gxa(Bxa_rr, Exa_rr, aS_rr, Bxa_rr, Exa_rr)

    # ===================== Pacejka 5.2: lateral (Fy) ======================
    mu_fl_y = (mf.pDy1 + mf.pDy2*dfz_fl) / (1 + mf.pDy3*dynamic_camber_fl**2)
    mu_fr_y = (mf.pDy1 + mf.pDy2*dfz_fr) / (1 + mf.pDy3*dynamic_camber_fr**2)
    mu_rl_y = (mf.pDy1 + mf.pDy2*dfz_rl) / (1 + mf.pDy3*dynamic_camber_rl**2)
    mu_rr_y = (mf.pDy1 + mf.pDy2*dfz_rr) / (1 + mf.pDy3*dynamic_camber_rr**2)
    Dy_fl = mu_fl_y * fz_fl_shift; Dy_fr = mu_fr_y * fz_fr_shift
    Dy_rl = mu_rl_y * fz_rl_shift; Dy_rr = mu_rr_y * fz_rr_shift
    Cy = mf.pCy1

    def _svyg(fzs, dfz, gam): return fzs * (mf.pVy3 + mf.pVy4*dfz) * gam
    Svy_gamma_fl = _svyg(fz_fl_shift, dfz_fl, dynamic_camber_fl)
    Svy_gamma_fr = _svyg(fz_fr_shift, dfz_fr, dynamic_camber_fr)
    Svy_gamma_rl = _svyg(fz_rl_shift, dfz_rl, dynamic_camber_rl)
    Svy_gamma_rr = _svyg(fz_rr_shift, dfz_rr, dynamic_camber_rr)
    Svy_fl = fz_fl_shift*(mf.pVy1 + mf.pVy2*dfz_fl) + Svy_gamma_fl
    Svy_fr = fz_fr_shift*(mf.pVy1 + mf.pVy2*dfz_fr) + Svy_gamma_fr
    Svy_rl = fz_rl_shift*(mf.pVy1 + mf.pVy2*dfz_rl) + Svy_gamma_rl
    Svy_rr = fz_rr_shift*(mf.pVy1 + mf.pVy2*dfz_rr) + Svy_gamma_rr

    def _kya(fzs, gam):
        return mf.pKy1 * vp.Fz0 * ca.sin(mf.pKy4 * ca.atan(fzs / ((mf.pKy2 + mf.pKy5*gam**2)*vp.Fz0))) / (1 + mf.pKy3*gam**2)
    Kya_fl = _kya(fz_fl_shift, dynamic_camber_fl); Kya_fr = _kya(fz_fr_shift, dynamic_camber_fr)
    Kya_rl = _kya(fz_rl_shift, dynamic_camber_rl); Kya_rr = _kya(fz_rr_shift, dynamic_camber_rr)
    By_fl = Kya_fl / (Cy*Dy_fl + vp.eps_y); By_fr = Kya_fr / (Cy*Dy_fr + vp.eps_y)
    By_rl = Kya_rl / (Cy*Dy_rl + vp.eps_y); By_rr = Kya_rr / (Cy*Dy_rr + vp.eps_y)

    def _kyg0(fzs, dfz): return fzs * (mf.pKy6 + mf.pKy7*dfz)
    Ky_gamma0_fl = _kyg0(fz_fl_shift, dfz_fl); Ky_gamma0_fr = _kyg0(fz_fr_shift, dfz_fr)
    Ky_gamma0_rl = _kyg0(fz_rl_shift, dfz_rl); Ky_gamma0_rr = _kyg0(fz_rr_shift, dfz_rr)

    Shy_fl = (mf.pHy1 + mf.pHy2*dfz_fl) + (Ky_gamma0_fl*dynamic_camber_fl - Svy_gamma_fl)/(Kya_fl + vp.eps_K)
    Shy_fr = (mf.pHy1 + mf.pHy2*dfz_fr) + (Ky_gamma0_fr*dynamic_camber_fr - Svy_gamma_fr)/(Kya_fr + vp.eps_K)
    Shy_rl = (mf.pHy1 + mf.pHy2*dfz_rl) + (Ky_gamma0_rl*dynamic_camber_rl - Svy_gamma_rl)/(Kya_rl + vp.eps_K)
    Shy_rr = (mf.pHy1 + mf.pHy2*dfz_rr) + (Ky_gamma0_rr*dynamic_camber_rr - Svy_gamma_rr)/(Kya_rr + vp.eps_K)
    say_fl = sa_fl + Shy_fl; say_fr = sa_fr + Shy_fr; say_rl = sa_rl + Shy_rl; say_rr = sa_rr + Shy_rr

    def _ey(dfz, gam, say): return (mf.pEy1 + mf.pEy2*dfz) * (1 + mf.pEy5*gam**2 - (mf.pEy3 + mf.pEy4*gam)*ca.sign(say))
    Ey_fl = _ey(dfz_fl, dynamic_camber_fl, say_fl); Ey_fr = _ey(dfz_fr, dynamic_camber_fr, say_fr)
    Ey_rl = _ey(dfz_rl, dynamic_camber_rl, say_rl); Ey_rr = _ey(dfz_rr, dynamic_camber_rr, say_rr)

    def _fy0(D, B, say, E, Sv): return D*ca.sin(Cy*ca.atan(B*say - E*(B*say - ca.atan(B*say)))) + Sv
    Fy0_fl = _fy0(Dy_fl, By_fl, say_fl, Ey_fl, Svy_fl)
    Fy0_fr = _fy0(Dy_fr, By_fr, say_fr, Ey_fr, Svy_fr)
    Fy0_rl = _fy0(Dy_rl, By_rl, say_rl, Ey_rl, Svy_rl)
    Fy0_rr = _fy0(Dy_rr, By_rr, say_rr, Ey_rr, Svy_rr)

    # combined-slip lateral
    Cyk = mf.rCy1
    def _shyk(dfz): return mf.rHy1 + mf.rHy2*dfz
    kappa_S_fl = sx_fl + _shyk(dfz_fl); kappa_S_fr = sx_fr + _shyk(dfz_fr)
    kappa_S_rl = sx_rl + _shyk(dfz_rl); kappa_S_rr = sx_rr + _shyk(dfz_rr)
    def _eyk(dfz): return mf.rEy1 + mf.rEy2*dfz
    Eyk_fl = _eyk(dfz_fl); Eyk_fr = _eyk(dfz_fr); Eyk_rl = _eyk(dfz_rl); Eyk_rr = _eyk(dfz_rr)
    def _dvyk(muy, fzs, dfz, gam, sa): return muy*fzs*(mf.rVy1 + mf.rVy2*dfz + mf.rVy3*gam)*ca.cos(ca.atan(mf.rVy4*sa))
    Dvyk_fl = _dvyk(mu_fl_y, fz_fl_shift, dfz_fl, dynamic_camber_fl, sa_fl)
    Dvyk_fr = _dvyk(mu_fr_y, fz_fr_shift, dfz_fr, dynamic_camber_fr, sa_fr)
    Dvyk_rl = _dvyk(mu_rl_y, fz_rl_shift, dfz_rl, dynamic_camber_rl, sa_rl)
    Dvyk_rr = _dvyk(mu_rr_y, fz_rr_shift, dfz_rr, dynamic_camber_rr, sa_rr)
    def _svyk(Dv, sx): return Dv * ca.sin(mf.rVy5 * ca.atan(mf.rVy6 * sx))
    Svyk_fl = _svyk(Dvyk_fl, sx_fl); Svyk_fr = _svyk(Dvyk_fr, sx_fr)
    Svyk_rl = _svyk(Dvyk_rl, sx_rl); Svyk_rr = _svyk(Dvyk_rr, sx_rr)
    def _byk(gam, sa): return (mf.rBy1 + mf.rBy4*gam**2) * ca.cos(ca.atan(mf.rBy2*(sa - mf.rBy3)))
    Byk_fl = _byk(dynamic_camber_fl, sa_fl); Byk_fr = _byk(dynamic_camber_fr, sa_fr)
    Byk_rl = _byk(dynamic_camber_rl, sa_rl); Byk_rr = _byk(dynamic_camber_rr, sa_rr)
    Shyk_fl = _shyk(dfz_fl); Shyk_fr = _shyk(dfz_fr); Shyk_rl = _shyk(dfz_rl); Shyk_rr = _shyk(dfz_rr)
    def _gyk2(B, E, kS, Sh):
        G0 = ca.cos(Cyk*ca.atan(B*Sh - E*(B*Sh - ca.atan(B*Sh))))
        return ca.cos(Cyk*ca.atan(B*kS - E*(B*kS - ca.atan(B*kS)))) / G0
    Gyk_fl = _gyk2(Byk_fl, Eyk_fl, kappa_S_fl, Shyk_fl)
    Gyk_fr = _gyk2(Byk_fr, Eyk_fr, kappa_S_fr, Shyk_fr)
    Gyk_rl = _gyk2(Byk_rl, Eyk_rl, kappa_S_rl, Shyk_rl)
    Gyk_rr = _gyk2(Byk_rr, Eyk_rr, kappa_S_rr, Shyk_rr)

    # ===================== combine slip ====================================
    if TyreModel == "CombinedSlip":
        fx_fl = fx_fl_0 * Gxa_fl; fx_fr = fx_fr_0 * Gxa_fr
        fx_rl = fx_rl_0 * Gxa_rl; fx_rr = fx_rr_0 * Gxa_rr
        fy_fl = Gyk_fl * Fy0_fl + Svyk_fl; fy_fr = Gyk_fr * Fy0_fr + Svyk_fr
        fy_rl = Gyk_rl * Fy0_rl + Svyk_rl; fy_rr = Gyk_rr * Fy0_rr + Svyk_rr
    elif TyreModel == "PureSlip":
        fx_fl, fx_fr, fx_rl, fx_rr = fx_fl_0, fx_fr_0, fx_rl_0, fx_rr_0
        fy_fl, fy_fr, fy_rl, fy_rr = Fy0_fl, Fy0_fr, Fy0_rl, Fy0_rr
    else:
        raise ValueError("TyreModel must be 'CombinedSlip' or 'PureSlip'")

    # rolling resistance
    f_roll_fl = rollCoeff * fz_fl; f_roll_fr = rollCoeff * fz_fr
    f_roll_rl = rollCoeff * fz_rl; f_roll_rr = rollCoeff * fz_rr
    f_roll = f_roll_fl + f_roll_fr + f_roll_rl + f_roll_rr

    # tyre forces in vehicle frame
    fxg_fl = fx_fl*ca.cos(delta) - fy_fl*ca.sin(delta)
    fxg_fr = fx_fr*ca.cos(delta) - fy_fr*ca.sin(delta)
    fxg_rl = fx_rl; fxg_rr = fx_rr
    fyg_fl = fy_fl*ca.cos(delta) + fx_fl*ca.sin(delta)
    fyg_fr = fy_fr*ca.cos(delta) + fx_fr*ca.sin(delta)
    fyg_rl = fy_rl; fyg_rr = fy_rr

    # jacking forces
    fj_fl = -fyg_fl * (2*vp.hRCf/vp.t)
    fj_fr = +fyg_fr * (2*vp.hRCf/vp.t)
    fj_rl = -fyg_rl * (2*vp.hRCr/vp.t)
    fj_rr = +fyg_rr * (2*vp.hRCr/vp.t)

    # moments about x
    Mx_fl = fyg_fl * (vp.hcg - vp.hRCf); Mx_fr = fyg_fr * (vp.hcg - vp.hRCf)
    Mx_rl = fyg_rl * (vp.hcg - vp.hRCr); Mx_rr = fyg_rr * (vp.hcg - vp.hRCr)

    # instantaneous strut length & moments about y
    ls_fl = vp.lsi_fl - (xs_fl - vp.xsi_fl); ls_fr = vp.lsi_fr - (xs_fr - vp.xsi_fr)
    ls_rl = vp.lsi_rl - (xs_rl - vp.xsi_rl); ls_rr = vp.lsi_rr - (xs_rr - vp.xsi_rr)
    My_fl = -(fxg_fl * (vp.Rw_f - zt_fl + ls_fl)); My_fr = -(fxg_fr * (vp.Rw_f - zt_fr + ls_fr))
    My_rl = -(fxg_rl * (vp.Rw_r - zt_rl + ls_rl)); My_rr = -(fxg_rr * (vp.Rw_r - zt_rr + ls_rr))

    # ===================== powertrain torque split ========================
    if pt.EM4 == 0:
        Om_motor = (Om_fl + Om_fr)/4*vp.gear + (Om_rl + Om_rr)/4*vp.gear
        if pt.ATD == 0:
            T_fl = 0.5*T_motor*vp.gear*(1 - vp.Tdist) + T_brake*vp.brkB
            T_fr = 0.5*T_motor*vp.gear*(1 - vp.Tdist) + T_brake*vp.brkB
            T_rl = 0.5*T_motor*vp.gear*vp.Tdist + T_brake*(1 - vp.brkB)
            T_rr = 0.5*T_motor*vp.gear*vp.Tdist + T_brake*(1 - vp.brkB)
        else:
            T_fl = ATD_FL*T_motor*vp.gear + ATD_FL*T_brake*2
            T_fr = ATD_FR*T_motor*vp.gear + ATD_FR*T_brake*2
            T_rl = ATD_RL*T_motor*vp.gear + ATD_RL*T_brake*2
            T_rr = ATD_RR*T_motor*vp.gear + ATD_RR*T_brake*2
        T_fl = ca.if_else(xs_fl >= 0.075, 0, T_fl)
        T_fr = ca.if_else(xs_fr >= 0.075, 0, T_fr)
        T_rl = ca.if_else(xs_rl >= 0.075, 0, T_rl)
        T_rr = ca.if_else(xs_rr >= 0.075, 0, T_rr)
        P_motor = T_motor * Om_motor
        Om_motor_each = P_motor_each = None
    else:
        Om_motor_fl = Om_fl/vp.gear; Om_motor_fr = Om_fr/vp.gear
        Om_motor_rl = Om_rl/vp.gear; Om_motor_rr = Om_rr/vp.gear
        T_fl = T_motor_fl*vp.gear + T_brake*vp.brkB
        T_fr = T_motor_fr*vp.gear + T_brake*vp.brkB
        T_rl = T_motor_rl*vp.gear + T_brake*(1 - vp.brkB)
        T_rr = T_motor_rr*vp.gear + T_brake*(1 - vp.brkB)
        T_fl = ca.if_else(xs_fl >= 0.075, 0, T_fl)
        T_fr = ca.if_else(xs_fr >= 0.075, 0, T_fr)
        T_rl = ca.if_else(xs_rl >= 0.075, 0, T_rl)
        T_rr = ca.if_else(xs_rr >= 0.075, 0, T_rr)
        P_motor_fl = T_motor_fl*Om_motor_fl; P_motor_fr = T_motor_fr*Om_motor_fr
        P_motor_rl = T_motor_rl*Om_motor_rl; P_motor_rr = T_motor_rr*Om_motor_rr
        Om_motor = P_motor = None

    # ===================== change of variable & derivatives ===============
    sf = (1 - n*kappa) / (vx*ca.cos(eps) - vy*ca.sin(eps))

    dvx = (fxg_fl + fxg_fr + fxg_rl + fxg_rr - f_drag - f_roll + vp.ms*vy*r
           - vp.ms*thetadot*zsdot + vp.ms*vp.g*ca.sin(theta)) * sf / vp.ms
    dvy = (fyg_fl + fyg_fr + fyg_rl + fyg_rr + f_side - vp.ms*vx*r
           - vp.ms*vp.g*ca.sin(phi)*ca.cos(theta)) * sf / vp.ms
    dr = ((fyg_fl + fyg_fr)*vp.l_f - (fyg_rl + fyg_rr)*vp.l_r
          + (-fxg_fl + fxg_fr - fxg_rl + fxg_rr)*(vp.t/2)
          - (f_side_rl + f_side_rr)*vp.l_r + (f_side_fl + f_side_fr)*vp.l_f) * sf / vp.I_z
    dn = (vx*ca.sin(eps) + vy*ca.cos(eps)) * sf
    deps = sf*r - kappa
    dOm_fl = sf * (T_fl - fx_fl*vp.Rw_f) / vp.Jw
    dOm_fr = sf * (T_fr - fx_fr*vp.Rw_f) / vp.Jw
    dOm_rl = sf * (T_rl - fx_rl*vp.Rw_r) / vp.Jw
    dOm_rr = sf * (T_rr - fx_rr*vp.Rw_r) / vp.Jw
    dzs = zsdot * sf
    dzsdot = (-f_lift + fzs_fl + fzs_fr + fzs_rl + fzs_rr + fj_fl + fj_fr + fj_rl + fj_rr
              - vp.ms*vp.g*ca.cos(phi)*ca.cos(theta) - vp.ms*phidot*vy + vp.ms*thetadot*vx) * sf / vp.ms
    dtheta = thetadot * sf
    dthetadot = (My_fr + My_fl + My_rl + My_rr + (fzs_rl + fzs_rr)*vp.l_r
                 - (fzs_fl + fzs_fr)*vp.l_f) * sf / vp.I_y
    dphi = phidot * sf
    dphidot = (Mx_fl + Mx_fr + Mx_rl + Mx_rr + (fzs_fl + fzs_rl - fzs_fr - fzs_rr)*0.5*vp.t) * sf / vp.I_x
    dwu_fl = (zt_fl*vp.kt - vp.mus*vp.g - fj_fl - fzs_fl) * sf / vp.mus
    dwu_fr = (zt_fr*vp.kt - vp.mus*vp.g - fj_fr - fzs_fr) * sf / vp.mus
    dwu_rl = (zt_rl*vp.kt - vp.mus*vp.g - fj_rl - fzs_rl) * sf / vp.mus
    dwu_rr = (zt_rr*vp.kt - vp.mus*vp.g - fj_rr - fzs_rr) * sf / vp.mus
    dzt_fl = -wu_fl * sf; dzt_fr = -wu_fr * sf; dzt_rl = -wu_rl * sf; dzt_rr = -wu_rr * sf

    Lon_acc = dvx / sf
    Lat_acc = dvy / sf

    dx = ca.vertcat(dvx, dvy, dr, dn, deps, dOm_fl, dOm_fr, dOm_rl, dOm_rr,
                    dzs, dzsdot, dtheta, dthetadot, dphi, dphidot,
                    dwu_fl, dwu_fr, dwu_rl, dwu_rr, dzt_fl, dzt_fr, dzt_rl, dzt_rr) / ca.DM(x_s)

    # ===================== expose on ctx.m23 ==============================
    m = SimpleNamespace()
    m.nx, m.nu, m.ny = nx, nu, 0
    m.x, m.u, m.pv, m.kappa = x, u, pv, kappa
    m.dx, m.sf = dx, sf
    m.x_s, m.u_s = x_s, u_s
    m.x_min, m.x_max, m.u_min, m.u_max = x_min, x_max, u_min, u_max
    # raw signals
    m.vx, m.vy, m.r, m.n, m.eps = vx, vy, r, n, eps
    m.zs, m.theta, m.phi = zs, theta, phi
    m.delta, m.T_brake, m.T_brake_n = delta, T_brake, T_brake_n
    # tyre signals
    m.sa_fl, m.sa_fr, m.sa_rl, m.sa_rr = sa_fl, sa_fr, sa_rl, sa_rr
    m.sx_fl, m.sx_fr, m.sx_rl, m.sx_rr = sx_fl, sx_fr, sx_rl, sx_rr
    m.fx_fl, m.fx_fr, m.fx_rl, m.fx_rr = fx_fl, fx_fr, fx_rl, fx_rr
    m.fy_fl, m.fy_fr, m.fy_rl, m.fy_rr = fy_fl, fy_fr, fy_rl, fy_rr
    m.fz_fl, m.fz_fr, m.fz_rl, m.fz_rr = fz_fl, fz_fr, fz_rl, fz_rr
    m.mu_fl_x, m.mu_fr_x, m.mu_rl_x, m.mu_rr_x = mu_fl_x, mu_fr_x, mu_rl_x, mu_rr_x
    m.mu_fl_y, m.mu_fr_y, m.mu_rl_y, m.mu_rr_y = mu_fl_y, mu_fr_y, mu_rl_y, mu_rr_y
    m.dynamic_camber_fl, m.dynamic_camber_fr = dynamic_camber_fl, dynamic_camber_fr
    m.dynamic_camber_rl, m.dynamic_camber_rr = dynamic_camber_rl, dynamic_camber_rr
    m.CG_h, m.CG_r, m.CG_p = CG_h, CG_r, CG_p
    # aero
    m.f_drag, m.f_lift = f_drag, f_lift
    m.f_lift_fl, m.f_lift_fr, m.f_lift_rl, m.f_lift_rr = f_lift_fl, f_lift_fr, f_lift_rl, f_lift_rr
    m.f_side, m.f_side_fl, m.f_side_fr, m.f_side_rl, m.f_side_rr = f_side, f_side_fl, f_side_fr, f_side_rl, f_side_rr
    # torques / powertrain
    m.T_fl, m.T_fr, m.T_rl, m.T_rr = T_fl, T_fr, T_rl, T_rr
    m.Lon_acc, m.Lat_acc = Lon_acc, Lat_acc
    if pt.EM4 == 0:
        m.T_motor, m.T_motor_n = T_motor, T_motor_n
        m.Om_motor, m.P_motor = Om_motor, P_motor
        m.Om_fl, m.Om_fr, m.Om_rl, m.Om_rr = Om_fl, Om_fr, Om_rl, Om_rr
    else:
        m.T_motor_fl, m.T_motor_fr, m.T_motor_rl, m.T_motor_rr = T_motor_fl, T_motor_fr, T_motor_rl, T_motor_rr
        m.T_motor_fl_n, m.T_motor_fr_n = T_motor_fl_n, T_motor_fr_n
        m.T_motor_rl_n, m.T_motor_rr_n = T_motor_rl_n, T_motor_rr_n
        m.Om_motor_fl, m.Om_motor_fr, m.Om_motor_rl, m.Om_motor_rr = Om_motor_fl, Om_motor_fr, Om_motor_rl, Om_motor_rr
        m.P_motor_fl, m.P_motor_fr, m.P_motor_rl, m.P_motor_rr = P_motor_fl, P_motor_fr, P_motor_rl, P_motor_rr
        m.Om_fl, m.Om_fr, m.Om_rl, m.Om_rr = Om_fl, Om_fr, Om_rl, Om_rr
    if pt.ATD == 1:
        m.ATD_FL, m.ATD_FR, m.ATD_RL, m.ATD_RR = ATD_FL, ATD_FR, ATD_RL, ATD_RR
    ctx.m23 = m
    return ctx


def _linear_interp(ca, x, x_data, y_data):
    """Piecewise-linear interpolation as a CasADi expression (port of linear_interp.m)."""
    x_data = np.asarray(x_data, dtype=float).reshape(-1)
    y_data = np.asarray(y_data, dtype=float).reshape(-1)
    nseg = x_data.size
    y = ca.DM(y_data[0])
    for i in range(nseg - 1):
        slope = (y_data[i+1] - y_data[i]) / (x_data[i+1] - x_data[i])
        y = ca.if_else(x < x_data[i], y,
                       ca.if_else(x <= x_data[i+1], y_data[i] + slope*(x - x_data[i]), y))
    slope = (y_data[-1] - y_data[-2]) / (x_data[-1] - x_data[-2])
    y = ca.if_else(x > x_data[-1], y_data[-2] + slope*(x - x_data[-1]), y)
    return y
