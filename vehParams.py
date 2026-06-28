"""vehParams.py - two-phase build of ctx.vp (+ ctx.mf / ctx.aero / ctx.cg).

Phase 1 builds a dict of primary (typed-in) inputs and merges any
``vp_overrides`` into it; phase 2 computes every derived quantity from the
merged primaries, so an override of any primary propagates correctly. With no
overrides the result is byte-for-byte identical to the original hardcoded
values (Copy-B tyre set; Rw overwritten to 0.355 with gear left on 0.3142857).
"""
import os
import warnings
import numpy as np
from types import SimpleNamespace

from functions.importfile import mat_to_namespace


def default_primaries():
    """Primary (leaf) vehicle inputs and their default values."""
    return {
        # balance / aero inputs
        "brkB": 0.6766, "Tdist": 0.7281, "ksD": 0.4620,
        "alpha_FL": 10.0, "alpha_FR": 10.0, "alpha_RW": 8.0, "alpha_TW": 0.0,
        # constants
        "g": 9.81, "rho": 1.204,
        # masses
        "mb": 1820.0, "md": 75.0, "muf": 90.0, "mur": 100.0,
        # dimensions
        "A": 1.95, "t": 1.8, "l": 3.0, "wB": 0.5,
        "hcg": 0.5, "huf": 0.2968771, "hur": 0.2968771, "hw": 1.28,
        "hRCf": 0.07, "hRCr": 0.11, "hride": 0.117,
        # inertias
        "I_z": 1960.0, "I_y": 1600.0, "I_x": 1000.0,
        # aero placeholders (overwritten by DATA_AA when present)
        "Cd": 0.75, "Cl": 1.45,
        # tyre
        "Rw": 0.355, "Jw": 3.6, "f": 0.01, "kt": 300000.0,
        "Fz0": 4905.0, "Fz0_shift": 1.0,
        # suspension
        "k_fl": 75000.0, "k_fr": 75000.0, "k_rl": 80000.0, "k_rr": 80000.0,
        "zeta_fl": 0.7, "zeta_fr": 0.7, "zeta_rl": 0.7, "zeta_rr": 0.7,
        # brakes
        "Tbrake_max": 4000.0,
        # camber / toe
        "gamma_fl": 0.0, "gamma_rl": 0.0, "toe_front": 0.0, "toe_rear": 0.0,
        # numerical epsilons
        "eps_x": 1e-6, "eps_y": 1e-6, "eps_K": 1e-6,
    }


def _default_mf():
    """Pacejka 5.2 coefficients (MF_205_60R15_V91, lateral block = Copy B)."""
    return SimpleNamespace(
        pCx1=1.6055, pDx1=1.1703, pDx2=-0.081328, pDx3=0.0,
        pEx1=0.53409, pEx2=-0.019956, pEx3=0.18089, pEx4=0.0,
        pKx1=36.411, pKx2=0.12615, pKx3=0.51289,
        pHx1=0.0, pHx2=0.0, pVx1=0.0, pVx2=0.0,
        rBx1=18.456, rBx2=16.314, rBx3=0.0,
        rCx1=1.091, rEx1=0.0, rEx2=0.0, rHx1=0.0058715,
        pCy1=2.1322, pDy1=1.0283, pDy2=-0.16758, pDy3=-1.5821,
        pEy1=0.15, pEy2=-1.8733, pEy3=0.0, pEy4=0.0, pEy5=0.0,
        pKy1=-20.505, pKy2=2.0284, pKy3=0.89994, pKy4=0.0,
        pKy5=0.002, pKy6=-0.002, pKy7=0.0,
        pHy1=0.0031377, pHy2=0.00051596,
        pVy1=0.0, pVy2=0.0, pVy3=0.0, pVy4=0.08,
        rBy1=22.003, rBy2=-13.623, rBy3=-0.0093616, rBy4=0.0,
        rCy1=0.98294, rEy1=0.0, rEy2=0.0,
        rHy1=-9.1492e-11, rHy2=0.0,
        rVy1=22.965, rVy2=0.37981, rVy3=1.8552,
        rVy4=0.08767, rVy5=-8.8234e-11, rVy6=0.90374,
    )


PRIMARY_KEYS = frozenset(default_primaries())
MF_KEYS = frozenset(vars(_default_mf()))


def vehParams(ctx, data_dir="Data", vp_overrides=None):
    if not hasattr(ctx, "vp") or ctx.vp is None:
        ctx.vp = SimpleNamespace()
    vp = ctx.vp

    overrides = dict(vp_overrides or {})
    unknown = set(overrides) - PRIMARY_KEYS - MF_KEYS
    if unknown:
        raise ValueError(f"Unknown vehParams override keys: {sorted(unknown)}")
    mf_over = {k: overrides.pop(k) for k in list(overrides) if k in MF_KEYS}

    # ---- phase 1: primaries (+ overrides) ---------------------------------
    p = default_primaries()
    p.update(overrides)
    for k, v in p.items():
        setattr(vp, k, v)

    # ---- phase 2: derived quantities --------------------------------------
    vp.ms = vp.mb + vp.md
    vp.mus = vp.muf + vp.mur
    vp.m = vp.ms + vp.muf + vp.mur

    vp.l_f = vp.l * (1 - vp.wB)
    vp.l_r = vp.l * vp.wB

    vp.hRC = (vp.l_f * vp.hRCr + vp.l_r * vp.hRCf) / vp.l
    vp.d = vp.hcg - vp.hRC

    vp.Rw_r = vp.Rw
    vp.Rw_f = vp.Rw

    vp.m_eff_f = vp.m * (1 - vp.wB)
    vp.m_eff_r = vp.m * vp.wB

    vp.c_fl = vp.zeta_fl * 2 * np.sqrt(vp.m_eff_f / 2) * vp.k_fl
    vp.c_fr = vp.zeta_fr * 2 * np.sqrt(vp.m_eff_f / 2) * vp.k_fr
    vp.c_rl = vp.zeta_rl * 2 * np.sqrt(vp.m_eff_r / 2) * vp.k_rl
    vp.c_rr = vp.zeta_rr * 2 * np.sqrt(vp.m_eff_r / 2) * vp.k_rr

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

    # ---- aerodynamics (DATA_AA overwrites Cd/Cl when present) -------------
    aa_path = os.path.join(data_dir, "DATA_AA.mat")
    if os.path.exists(aa_path):
        import scipy.io as sio
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
        vp.Cl = vp.Cl0_front + vp.Cl0_rear
        vp.Cd = vp.Cd0
    else:
        ctx.aero = None
        warnings.warn(
            f"DATA_AA.mat not found at '{aa_path}'. Aerodynamic coefficients are "
            "unset; place DATA_AA.mat in the Data/ folder before a real run.")

    # ---- Pacejka 5.2 coefficients (+ mf overrides) ------------------------
    ctx.mf = _default_mf()
    for k, v in mf_over.items():
        setattr(ctx.mf, k, v)

    # ---- simplified Pacejka used by the 7-state init model ----------------
    vp.tyre = SimpleNamespace(
        mu=1.41, pD2=-0.6,
        bx=6.0, cx=2.3, ex=0.9,
        by=6.0, cy=2.5, ey=0.5,
    )

    deg2rad = np.pi / 180.0

    # ---- camber (mirrored across the axle) --------------------------------
    vp.gamma_fr = -vp.gamma_fl
    vp.gamma_rr = -vp.gamma_rl
    vp.gamma_fl_rad = vp.gamma_fl * deg2rad
    vp.gamma_fr_rad = vp.gamma_fr * deg2rad
    vp.gamma_rl_rad = vp.gamma_rl * deg2rad
    vp.gamma_rr_rad = vp.gamma_rr * deg2rad

    # ---- toe --------------------------------------------------------------
    vp.toe_front_rad = vp.toe_front * deg2rad
    vp.toe_rear_rad = vp.toe_rear * deg2rad

    # ---- camber-gain coefficients (linear option) -------------------------
    ctx.cg = SimpleNamespace(
        CG_h_deg_per_mm_linear=-0.03,
        CG_r_deg_per_deg_linear=0.1,
        CG_p_deg_per_deg_linear=0.05,
    )

    vp.CG_h_deg_per_mm_table = np.array([
        -0.12, 0.038, -0.08, 0.027, -0.04, 0.012, 0.00, 0.000,
        0.02, -0.006, 0.05, -0.015, 0.08, -0.025, 0.10, -0.032])
    vp.CG_r_deg_per_deg_table = np.array([
        -0.105, -0.8, -0.070, -0.75, -0.035, -0.7, 0.000, -0.65,
        0.035, -0.6, 0.070, -0.55, 0.105, -0.5])
    vp.CG_p_deg_per_deg_table = None

    return ctx
