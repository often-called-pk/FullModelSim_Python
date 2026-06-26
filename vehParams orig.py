"""vehParams.py - direct port of vehParams.m (FullModel_4EM_Suspension_FullTyre)

Builds the vehicle/tyre/suspension parameter namespace ``ctx.vp`` plus:
  * ``ctx.mf``  : Pacejka 5.2 Magic-Formula coefficients (used by vehModel.py)
  * ``ctx.aero``: DATA_AA aero-coefficient struct (polynomials + nominal coeffs)
  * ``ctx.cg``  : linear camber-gain coefficients

TYRE SET: uses **Copy B** of the MF_205_60R15_V91 lateral block (Fz0 = 4905;
pKy1 = -20.505; pEy1 = 0.15; pKy4 = 0; pKy5 = 0.002; pKy6 = -0.002;
pVy1..4 = 0,0,0,0.08). Other parameters are common to both indexed copies.

WHEEL-RADIUS QUIRK: Powertrain.py sets vp.Rw = 0.3142857 and derives vp.gear
from it; this module overwrites vp.Rw = vp.Rw_f = vp.Rw_r = 0.355 but does NOT
recompute vp.gear (intentional, matches MATLAB; gear stays based on 0.3142857).
"""

import os
import warnings
import numpy as np
import scipy.io as sio
from types import SimpleNamespace

from functions.importfile import mat_to_namespace


def vehParams(ctx, data_dir="Data"):
    if not hasattr(ctx, "vp") or ctx.vp is None:
        ctx.vp = SimpleNamespace()
    vp = ctx.vp

    # ---- vehicle parameter inputs -----------------------------------------
    vp.brkB = 0.6766          # fraction of total brake force to front wheels (-)
    vp.Tdist = 0.7281         # fraction of total torque to rear wheels       (-)
    vp.ksD = 0.4620           # fraction of total roll stiffness, rear axle   (-)

    # ---- aerodynamic input ------------------------------------------------
    vp.alpha_FL = 10.0        # left  front wing AoA  [0 10]  (deg)
    vp.alpha_FR = vp.alpha_FL  # right front wing AoA  [0 10]  (deg)
    vp.alpha_RW = 8.0         # rear wing AoA         [0 30]  (deg)
    vp.alpha_TW = 0.0         # rear wing tilt        [-12 12](deg)

    # ---- constants --------------------------------------------------------
    vp.g = 9.81               # gravitational acceleration (m/s^2)
    vp.rho = 1.204            # air density                (kg/m^3)

    # ---- masses -----------------------------------------------------------
    vp.mb = 1820.0            # sprung mass            (kg)
    vp.md = 75.0              # driver mass            (kg)
    vp.ms = vp.mb + vp.md     # total sprung mass      (kg)
    vp.muf = 90.0             # unsprung mass front    (kg)
    vp.mur = 100.0            # unsprung mass rear     (kg)
    vp.mus = vp.muf + vp.mur  # total unsprung mass    (kg)
    vp.m = vp.ms + vp.muf + vp.mur   # total mass      (kg)

    # ---- dimensions -------------------------------------------------------
    vp.A = 1.95               # reference area (m^2)
    vp.t = 1.8                # track width    (m)
    vp.l = 3.0                # wheelbase      (m)
    vp.wB = 0.5               # COG distribution front (-)
    vp.l_f = vp.l * (1 - vp.wB)   # COG -> front axle (m)
    vp.l_r = vp.l * vp.wB         # COG -> rear  axle (m)

    vp.hcg = 0.5              # COG height (m)
    vp.huf = 0.2968771        # height COG unsprung front (m)
    vp.hur = 0.2968771        # height COG unsprung rear  (m)
    vp.hw = 1.28              # rear wing height (m)
    vp.hRCf = 0.07            # roll centre height front (m)
    vp.hRCr = 0.11            # roll centre height rear  (m)
    vp.hRC = (vp.l_f * vp.hRCr + vp.l_r * vp.hRCf) / vp.l   # RC axis height at COG (m)
    vp.d = vp.hcg - vp.hRC    # COG-to-roll-axis distance (m)
    vp.hride = 0.117          # ride height (m)

    # ---- inertias ---------------------------------------------------------
    vp.I_z = 1960.0           # yaw   (kg*m^2)
    vp.I_y = 1600.0           # pitch (kg*m^2)
    vp.I_x = 1000.0           # roll  (kg*m^2)

    # placeholders (overwritten below by DATA_AA)
    vp.Cd = 0.75
    vp.Cl = 1.45

    # ---- tyre parameters --------------------------------------------------
    # NOTE: overwrites Powertrain's vp.Rw = 0.3142857 (see module docstring).
    vp.Rw = 0.355
    vp.Rw_r = 0.355
    vp.Rw_f = 0.355
    vp.Jw = 0.9 * 2 * 2       # = 3.6  (kg*m^2)
    vp.f = 0.01               # rolling resistance coefficient (-)
    vp.kt = 300000.0          # tyre vertical stiffness (N/m)

    vp.Fz0 = 4905.0           # nominal vertical wheel load (N)   [Copy B]
    vp.Fz0_shift = 1.0        # nominal-load shift (optimised in TyreOptim)

    # ---- suspension -------------------------------------------------------
    vp.k_fl = 75000.0
    vp.k_fr = 75000.0
    vp.k_rl = 80000.0
    vp.k_rr = 80000.0

    vp.zeta_fl = 0.7
    vp.zeta_fr = 0.7
    vp.zeta_rl = 0.7
    vp.zeta_rr = 0.7

    vp.m_eff_f = vp.m * (1 - vp.wB)
    vp.m_eff_r = vp.m * vp.wB

    vp.c_fl = vp.zeta_fl * 2 * np.sqrt(vp.m_eff_f / 2) * vp.k_fl
    vp.c_fr = vp.zeta_fr * 2 * np.sqrt(vp.m_eff_f / 2) * vp.k_fr
    vp.c_rl = vp.zeta_rl * 2 * np.sqrt(vp.m_eff_r / 2) * vp.k_rl
    vp.c_rr = vp.zeta_rr * 2 * np.sqrt(vp.m_eff_r / 2) * vp.k_rr

    # static wheel loads
    vp.Wfl0 = 0.5 * vp.g * (vp.muf + vp.wB * vp.ms)
    vp.Wfr0 = 0.5 * vp.g * (vp.muf + vp.wB * vp.ms)
    vp.Wrl0 = 0.5 * vp.g * (vp.mur + (1 - vp.wB) * vp.ms)
    vp.Wrr0 = 0.5 * vp.g * (vp.mur + (1 - vp.wB) * vp.ms)

    vp.xti_fl = vp.Wfl0 / vp.kt
    vp.xti_fr = vp.Wfr0 / vp.kt
    vp.xti_rl = vp.Wrl0 / vp.kt
    vp.xti_rr = vp.Wrr0 / vp.kt

    vp.xsi_fl = 0.5 * vp.g * vp.ms * vp.l_f / (vp.l * vp.k_fl)
    vp.xsi_fr = 0.5 * vp.g * vp.ms * vp.l_f / (vp.l * vp.k_fr)
    vp.xsi_rl = 0.5 * vp.g * vp.ms * vp.l_r / (vp.l * vp.k_rl)
    vp.xsi_rr = 0.5 * vp.g * vp.ms * vp.l_r / (vp.l * vp.k_rr)

    vp.lsi_fl = vp.hcg - (vp.Rw - vp.xti_fl)
    vp.lsi_fr = vp.hcg - (vp.Rw - vp.xti_fr)
    vp.lsi_rl = vp.hcg - (vp.Rw - vp.xti_rl)
    vp.lsi_rr = vp.hcg - (vp.Rw - vp.xti_rr)

    # ---- brakes -----------------------------------------------------------
    vp.Tbrake_max = 4e3       # max braking torque (Nm)

    # ---- aerodynamics (DATA_AA) ------------------------------------------
    aa_path = os.path.join(data_dir, "DATA_AA.mat")
    if os.path.exists(aa_path):
        raw = sio.loadmat(aa_path, squeeze_me=True, struct_as_record=False)
        ctx.aero = mat_to_namespace(raw["aero"])
        a = ctx.aero
        vp.Cl0_right = float(a.veh.Cl0_right)
        vp.Cl0_left = float(a.veh.Cl0_left)
        vp.Cl0_front = float(a.veh.Cl0_front)
        vp.Cl0_rear = float(a.veh.Cl0_rear)
        vp.Cd0 = float(a.veh.Cd0)
        vp.Cs0_front = float(a.veh.Cs0_front)
        vp.Cs0_rear = float(a.veh.Cs0_rear)
        vp.Cl = vp.Cl0_front + vp.Cl0_rear     # total lift (no actuation)
        vp.Cd = vp.Cd0                          # total drag (no actuation)
    else:
        ctx.aero = None
        warnings.warn(
            f"DATA_AA.mat not found at '{aa_path}'. Aerodynamic coefficients are "
            "unset; place DATA_AA.mat in the Data/ folder before a real run.")

    # ---- Pacejka 5.2 Magic-Formula coefficients (ctx.mf) ------------------
    # MF_205_60R15_V91, lateral block = Copy B (user-confirmed).
    ctx.mf = SimpleNamespace(
        # Longitudinal (Fx)
        pCx1=1.6055, pDx1=1.1703, pDx2=-0.081328, pDx3=0.0,
        pEx1=0.53409, pEx2=-0.019956, pEx3=0.18089, pEx4=0.0,
        pKx1=36.411, pKx2=0.12615, pKx3=0.51289,
        pHx1=0.0, pHx2=0.0, pVx1=0.0, pVx2=0.0,
        # Longitudinal combined slip
        rBx1=18.456, rBx2=16.314, rBx3=0.0,
        rCx1=1.091, rEx1=0.0, rEx2=0.0, rHx1=0.0058715,
        # Lateral (Fy)  -- Copy B
        pCy1=2.1322, pDy1=1.0283, pDy2=-0.16758, pDy3=-1.5821,
        pEy1=0.15, pEy2=-1.8733, pEy3=0.0, pEy4=0.0, pEy5=0.0,
        pKy1=-20.505, pKy2=2.0284, pKy3=0.89994, pKy4=0.0,
        pKy5=0.002, pKy6=-0.002, pKy7=0.0,
        pHy1=0.0031377, pHy2=0.00051596,
        pVy1=0.0, pVy2=0.0, pVy3=0.0, pVy4=0.08,
        # Lateral combined slip
        rBy1=22.003, rBy2=-13.623, rBy3=-0.0093616, rBy4=0.0,
        rCy1=0.98294, rEy1=0.0, rEy2=0.0,
        rHy1=-9.1492e-11, rHy2=0.0,
        rVy1=22.965, rVy2=0.37981, rVy3=1.8552,
        rVy4=0.08767, rVy5=-8.8234e-11, rVy6=0.90374,
    )

    # ---- simplified Pacejka used by the 7-state init model ----------------
    vp.tyre = SimpleNamespace(
        mu=1.41, pD2=-0.6,
        bx=6.0, cx=2.3, ex=0.9,
        by=6.0, cy=2.5, ey=0.5,
    )

    vp.eps_x = 1e-6
    vp.eps_y = 1e-6
    vp.eps_K = 1e-6

    deg2rad = np.pi / 180.0

    # ---- camber (deg; positive = camber-in) -------------------------------
    vp.gamma_fl = 0.0
    vp.gamma_rl = 0.0
    vp.gamma_fr = -vp.gamma_fl
    vp.gamma_rr = -vp.gamma_rl
    vp.gamma_fl_rad = vp.gamma_fl * deg2rad
    vp.gamma_fr_rad = vp.gamma_fr * deg2rad
    vp.gamma_rl_rad = vp.gamma_rl * deg2rad
    vp.gamma_rr_rad = vp.gamma_rr * deg2rad

    # ---- toe (deg; negative = toe-in) -------------------------------------
    vp.toe_front = 0.0
    vp.toe_rear = 0.0
    vp.toe_front_rad = vp.toe_front * deg2rad
    vp.toe_rear_rad = vp.toe_rear * deg2rad

    # ---- camber-gain coefficients (linear option) -------------------------
    ctx.cg = SimpleNamespace(
        CG_h_deg_per_mm_linear=-0.03,    # [deg/mm] heave
        CG_r_deg_per_deg_linear=0.1,     # [deg/deg] roll
        CG_p_deg_per_deg_linear=0.05,    # [deg/deg] pitch
    )

    # camber-gain tables (used only when CamberGain='Table' in vehModel)
    vp.CG_h_deg_per_mm_table = np.array([
        -0.12, 0.038, -0.08, 0.027, -0.04, 0.012, 0.00, 0.000,
        0.02, -0.006, 0.05, -0.015, 0.08, -0.025, 0.10, -0.032])
    vp.CG_r_deg_per_deg_table = np.array([
        -0.105, -0.8, -0.070, -0.75, -0.035, -0.7, 0.000, -0.65,
        0.035, -0.6, 0.070, -0.55, 0.105, -0.5])
    # vp.CG_p_deg_per_deg_table: not captured from source; only needed for the
    # 'Table' camber-gain option (default in vehModel is 'Off').
    vp.CG_p_deg_per_deg_table = None

    return ctx
