"""ggv.py - quasi-steady-state (QSS) g-g-v envelope + speed march (screening tier).

The fast *screen* tier that sits alongside the IPOPT NLP (roadmap item 5). It
replaces the optimal-control problem by the classic point-mass lap simulation,
split into two separable stages so a parameter sweep only rebuilds the envelope:

    env  = build_envelope(ctx)            per-speed limits from ctx.vp/aero/pt/mf
    prof = march(env, s, k, vi)           apex speeds + forward/backward pass, t = sum ds/v

The result is an ESTIMATE, never an optimum: the line is fixed to the
centreline (n = 0, the track width is not used), and the transient states of
the 23-state model (suspension, wheel spin, yaw/side-slip dynamics) are absent.

numpy-only at runtime: this module never imports casadi.

Modelling choices (mirroring vehModel.py / MLTP.build_path_constraints)
-----------------------------------------------------------------------
Aero      Cl_front/rear/left/right and Cd are built from ctx.aero exactly as in
          vehModel.py (same polynomials, wing angles in degrees). F_lift =
          -0.5*rho*A*Cl*v^2 (positive = downforce), F_drag = 0.5*rho*A*Cd*v^2.
          Active aero (vp.ActAero > 0) is treated as STATIC at the nominal
          angles vp.alpha_FL/FR/RW/TW. The aero side force (Cs) is ignored
          (zero for the shipped static set: Cs0 = 0, alpha_TW = 0).
Loads     load_model='vehModel' (default) reproduces the quasi-steady vertical
          equilibrium of the 23-state model (checked against a CasADi
          rootfinder in test_screen.py):
            * total tyre load = vp.ms*g + 4*vp.mus*g + F_lift: vehModel's
              unsprung equation subtracts vp.mus*g (the TOTAL unsprung mass) at
              every corner, so the static sum is ~26.0 kN, not m*g = 20.5 kN;
            * the aero load reaches the axles in the CG ratio l_r/L : l_f/L
              (the pitch balance acts on fzs, which already contains
              f_lift_i, so the Cl front/rear split only changes body attitude);
            * translational mass M = vp.ms (dvx/dvy divide by vp.ms).
          load_model='nominal' is the textbook alternative: M = vp.m, static
          corner loads vp.Wfl0..Wrr0, aero split by Cl_front/Cl_rear.
Transfer  First order, from the vehModel moment balances. Longitudinal:
          dFz = Fx_tyre*h_x/L (h_x = the model's My lever arm Rw - zt + ls at
          static, ~0.555 m; 'nominal' uses vp.hcg). Lateral, per axle:
          dFz = [phi*sum(Mx) + Fy_axle*hRC_axle]/t, with roll-stiffness share
          phi = (k_fl+k_fr)/sum(k) and sum(Mx) = sum Fy_axle*(hcg - hRC_axle)
          (elastic + jacking terms of vehModel). Axle lateral forces follow yaw
          balance, Fy_f = M*ay*l_r/L. Combined (ax + ay) transfer is ignored.
Tyres     Peak friction from the Pacejka 5.2 set in ctx.mf, same expressions
          and scaling as vehModel.py:
            mu_x = pDx1 + pDx2*dfz,   mu_y = (pDy1 + pDy2*dfz)/(1 + pDy3*gamma^2),
            dfz  = (Fz - Fz0)/Fz0,     peak force D = mu*Fz/Fz0_shift,
          gamma = static camber (CamberGain='Off'). The slip shape (B, C, E)
          and vertical shifts Sv are ignored. NOTE: with the legacy
          tyre_set='CopyB' (mf.pKy4 = 0) vehModel's cornering stiffness, and so
          its fy, is identically zero (only for that set; the default MF205 set
          has nonzero Kya); the screen uses the intended mu_y peak.
Drive     EM4=0: wheel force <= min(Tmax*gear/Rw, Pmax/v), none above the motor
          rpm limit v = OMmax*Rw/gear. EM4=1: four motors, each <=
          min(Tmax*gear/Rw, Pmax*gear^2/v) -- vehModel's per-motor speed is
          Om_wheel/gear (sic), mirrored here. Tyre traction: all four corners
          when ATD=1 or EM4=1 (free distribution); fixed rear share vp.Tdist
          when ATD=0 and EM4=0. No regenerative braking (T_motor >= 0).
Brakes    Total wheel torque <= 2*vp.Tbrake_max. Split free with ATD (EM4=0),
          fixed front share vp.brkB otherwise.
Inertia   The torque/power/brake-torque limits act on M + 4*Jw/Rw^2 (wheel
          spin-up); the tyre limits act on M.
Resist.   F_res = F_drag + vp.f*sum(Fz) (rolling resistance on aero-loaded tyres).
Combined  Friction ellipse on the TYRE part only:
          Fx_avail = Fx_tyre(v)*sqrt(1 - (ay/ay_max(v))^2); the powertrain and
          brake-torque limits are not reduced by ay.
Speed     v <= pt.Vmax (the NLP's vx state bound).
"""

import math
from dataclasses import dataclass, field

import numpy as np

LOAD_MODELS = ("vehModel", "nominal")
_V_FLOOR = 0.1          # [m/s] numerical floor for the march (never reached on a sane track)
_BISECT_ITERS = 30      # bracket shrinks by 2^-30 ~ 1e-9
_AY_FLOOR = 1e-6


# ============================================================================
# Envelope container
# ============================================================================
@dataclass
class Envelope:
    """Quasi-steady performance envelope on a speed grid.

    Tables (same length as v_grid) hold the speed-dependent tyre limits; the
    powertrain/brake-torque limits and the resistance are analytic in v and kept
    as scalars so the march evaluates them exactly.
    """
    v_grid: np.ndarray            # [m/s]   speed grid (increasing)
    ay_max: np.ndarray            # [m/s^2] pure-lateral limit (tyres, lateral load transfer)
    Fx_trac: np.ndarray           # [N]     pure-longitudinal tyre traction capacity
    Fx_brk: np.ndarray            # [N]     pure-longitudinal tyre braking capacity
    ax_acc: np.ndarray            # [m/s^2] pure-longitudinal acceleration limit (net of resistance)
    ax_brk: np.ndarray            # [m/s^2] pure-longitudinal deceleration limit (> 0, incl. resistance)
    M: float                      # [kg]    translational mass (tyre-limited phases)
    M_eff: float                  # [kg]    M + 4*Jw/Rw^2 (torque/power/brake-torque phases)
    c_res0: float = 0.0           # [N]     resistance F_res(v) = c_res0 + c_res2*v^2
    c_res2: float = 0.0           # [N s^2/m^2]
    F_mot_trq: float = np.inf     # [N]     torque-limited drive force at the wheels
    P_mot: float = np.inf         # [W]     power limit at the wheels (drive force <= P_mot/v)
    v_rpm: float = np.inf         # [m/s]   motor rpm limit (no drive force above it)
    F_brk_trq: float = np.inf     # [N]     brake-torque-limited braking force
    v_cap: float = np.inf         # [m/s]   speed cap (pt.Vmax)
    load_model: str = "synthetic"
    F_lift: np.ndarray = None     # [N]     aero downforce on v_grid (diagnostics)
    F_drag: np.ndarray = None     # [N]     aero drag on v_grid (diagnostics)
    Fz_front: np.ndarray = None   # [N]     front-axle load, static + aero (no transfer)
    Fz_rear: np.ndarray = None    # [N]     rear-axle load, static + aero (no transfer)
    info: dict = field(default_factory=dict)

    def F_res(self, v):
        """Aero drag + rolling resistance [N]."""
        v = np.asarray(v, dtype=float)
        return self.c_res0 + self.c_res2 * v * v

    def F_mot(self, v):
        """Powertrain drive-force limit at the wheels [N]."""
        v = np.asarray(v, dtype=float)
        with np.errstate(divide="ignore"):
            f = np.minimum(self.F_mot_trq, self.P_mot / np.maximum(v, 1e-12))
        return np.where(v < self.v_rpm, f, 0.0)

    def to_dict(self):
        """Plain dict for scipy.io.savemat (no None values)."""
        out = {}
        for key, val in vars(self).items():
            if val is None:
                continue
            if key == "info":
                out[key] = {k: (v if v is not None else "") for k, v in val.items()}
            elif isinstance(val, np.ndarray):
                out[key] = val
            elif isinstance(val, str):
                out[key] = val
            else:
                out[key] = float(val)
        return out

    @classmethod
    def from_constants(cls, ay_max, ax_trac, ax_brk, v_cap, M=1.0, M_eff=None,
                       F_mot_trq=np.inf, P_mot=np.inf, v_rpm=np.inf,
                       F_brk_trq=np.inf, c_res0=0.0, c_res2=0.0,
                       v_grid=None):
        """Speed-independent synthetic envelope (tests / sanity checks).

        ay_max [m/s^2], ax_trac / ax_brk [m/s^2] are the pure tyre limits; the
        optional drive/brake/resistance terms are forces [N] like the real one.
        """
        if v_grid is None:
            v_grid = np.linspace(0.0, float(v_cap), 41)
        v_grid = np.asarray(v_grid, dtype=float)
        ones = np.ones_like(v_grid)
        M_eff = float(M if M_eff is None else M_eff)
        env = cls(v_grid=v_grid, ay_max=float(ay_max) * ones,
                  Fx_trac=float(ax_trac) * M * ones, Fx_brk=float(ax_brk) * M * ones,
                  ax_acc=ones, ax_brk=ones, M=float(M), M_eff=M_eff,
                  c_res0=float(c_res0), c_res2=float(c_res2),
                  F_mot_trq=float(F_mot_trq), P_mot=float(P_mot), v_rpm=float(v_rpm),
                  F_brk_trq=float(F_brk_trq), v_cap=float(v_cap))
        res = env.F_res(v_grid)
        env.ax_acc = np.minimum((env.Fx_trac - res) / env.M, (env.F_mot(v_grid) - res) / env.M_eff)
        env.ax_brk = np.minimum((env.Fx_brk + res) / env.M, (env.F_brk_trq + res) / env.M_eff)
        return env


# ============================================================================
# Building blocks (each mirrors a piece of vehModel.py)
# ============================================================================
def _poly(coeffs, x, powers):
    """sum_i coeffs[i] * x**powers[i]  (numpy twin of vehModel._poly)."""
    c = np.atleast_1d(np.asarray(coeffs, dtype=float)).reshape(-1)
    return sum(float(c[i]) * x ** p for i, p in enumerate(powers))


def aero_coefficients(vp, aero):
    """Aero coefficients at the static wing angles, exactly as vehModel.py builds
    them (lines 'aerodynamic coefficients'). Returns a dict with Cl_front,
    Cl_rear, Cl_left, Cl_right, Cl, Cd, the per-corner Cl_fl..Cl_rr and
    Cs_front/Cs_rear. Wing angles are vp.alpha_* in degrees (active aero is
    evaluated at these nominal angles)."""
    if aero is None:
        raise RuntimeError("DATA_AA.mat must be loaded (ctx.aero) to build the QSS envelope.")
    aFL, aFR = float(vp.alpha_FL), float(vp.alpha_FR)
    aRW, aTW = float(vp.alpha_RW), float(vp.alpha_TW)

    FW_L_Cl_front = float(aero.FW_L.Cl_front) * aFL
    FW_L_Cl_rear = float(aero.FW_L.Cl_rear) * aFL
    FW_L_Cl_left = FW_L_Cl_front + FW_L_Cl_rear
    FW_R_Cl_front = float(aero.FW_R.Cl_front) * aFR
    FW_R_Cl_rear = float(aero.FW_R.Cl_rear) * aFR
    FW_R_Cl_right = FW_R_Cl_front + FW_R_Cl_rear
    RW_Cl_front = _poly(aero.RW.Cl_front, aRW, [3, 2, 1])
    RW_Cl_rear = _poly(aero.RW.Cl_rear, aRW, [5, 4, 3, 2, 1])
    RW_Cl_side = 0.5 * (RW_Cl_front + RW_Cl_rear)          # RW_Cl_left == RW_Cl_right
    RW_Cd = _poly(aero.RW.Cd, aRW, [2, 1])
    TW_Cl_left = _poly(aero.TW.Cl_left, aTW, [2, 1])
    TW_Cl_right = _poly(aero.TW.Cl_right, aTW, [2, 1])
    TW_Cl_rear = TW_Cl_left + TW_Cl_right
    TW_Cs_rear = float(aero.TW.Cs_rear) * aTW

    Cl_front = vp.Cl0_front + FW_L_Cl_front + FW_R_Cl_front + RW_Cl_front
    Cl_rear = vp.Cl0_rear + FW_L_Cl_rear + FW_R_Cl_rear + RW_Cl_rear + TW_Cl_rear
    Cl_left = vp.Cl0_left + FW_L_Cl_left + RW_Cl_side + TW_Cl_left
    Cl_right = vp.Cl0_right + FW_R_Cl_right + RW_Cl_side + TW_Cl_right
    Cl = Cl_front + Cl_rear
    return dict(Cl_front=Cl_front, Cl_rear=Cl_rear, Cl_left=Cl_left, Cl_right=Cl_right,
                Cl=Cl, Cd=vp.Cd0 + RW_Cd,
                Cl_fl=Cl_front * (Cl_left / Cl), Cl_fr=Cl_front * (Cl_right / Cl),
                Cl_rl=Cl_rear * (Cl_left / Cl), Cl_rr=Cl_rear * (Cl_right / Cl),
                Cs_front=vp.Cs0_front, Cs_rear=vp.Cs0_rear + TW_Cs_rear)


def peak_mu_x(Fz, mf, Fz0):
    """Peak longitudinal friction coefficient = vehModel's mu_*_x (MF 5.2 Dx/Fz):
    pDx1 + pDx2*dfz with dfz = (Fz - Fz0)/Fz0 (no camber term in vehModel)."""
    dfz = (np.asarray(Fz, dtype=float) - Fz0) / Fz0
    return mf.pDx1 + mf.pDx2 * dfz


def peak_mu_y(Fz, mf, Fz0, gamma=0.0):
    """Peak lateral friction coefficient = vehModel's mu_*_y (MF 5.2 Dy/Fz):
    (pDy1 + pDy2*dfz) / (1 + pDy3*gamma^2), gamma = camber [rad]."""
    dfz = (np.asarray(Fz, dtype=float) - Fz0) / Fz0
    return (mf.pDy1 + mf.pDy2 * dfz) / (1.0 + mf.pDy3 * np.asarray(gamma, dtype=float) ** 2)


def _peak_force(mu, Fz, Fz0_shift):
    """MF peak force D = mu*Fz/Fz0_shift, clipped at 0 (no 'negative grip')."""
    return np.maximum(mu * Fz / Fz0_shift, 0.0)


def lateral_load_transfer(vp, Fy_f, Fy_r):
    """Quasi-steady lateral load transfer (dFz_f, dFz_r) [N] per axle = half the
    outer-minus-inner corner load difference, for axle lateral forces Fy_f, Fy_r.
    Exactly the vehModel roll balance: the elastic moment
    sum(Fy_axle*(hcg - hRC_axle)) is shared by roll stiffness
    phi_f = (k_fl+k_fr)/sum(k), plus the jacking (roll-centre) term Fy_axle*hRC/t."""
    k_f, k_r = vp.k_fl + vp.k_fr, vp.k_rl + vp.k_rr
    phi_f = k_f / (k_f + k_r)
    Mx = Fy_f * (vp.hcg - vp.hRCf) + Fy_r * (vp.hcg - vp.hRCr)
    return (phi_f * Mx + Fy_f * vp.hRCf) / vp.t, ((1.0 - phi_f) * Mx + Fy_r * vp.hRCr) / vp.t


def _bisect_fixed_point(fun, lo, hi, iters=_BISECT_ITERS):
    """Vectorised bisection for fun(x) = x on [lo, hi], with fun(x) - x
    decreasing: the x where the capacity fun(x) just meets the demand x."""
    lo = np.asarray(lo, dtype=float).copy()
    hi = np.asarray(hi, dtype=float).copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = fun(mid) >= mid
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


def _load_basis(vp, aero_c, load_model):
    """Mass, static axle loads, aero front fraction and pitch lever arm."""
    g, L = vp.g, vp.l_f + vp.l_r
    if load_model == "vehModel":
        M = vp.ms
        Fz_f0 = 2.0 * vp.mus * g + vp.ms * g * vp.l_r / L
        Fz_r0 = 2.0 * vp.mus * g + vp.ms * g * vp.l_f / L
        aero_front = vp.l_r / L
        # vehModel pitch lever arm  Rw - zt + ls,  ls = lsi - (xs - xsi),
        # evaluated at the static equilibrium (springs carry ms*g in the CG ratio)
        S_f, S_r = 0.5 * vp.ms * g * vp.l_r / L, 0.5 * vp.ms * g * vp.l_f / L
        arms = []
        for S, k, xsi, lsi, Rw, Fz in ((S_f, vp.k_fl, vp.xsi_fl, vp.lsi_fl, vp.Rw_f, Fz_f0 / 2),
                                       (S_f, vp.k_fr, vp.xsi_fr, vp.lsi_fr, vp.Rw_f, Fz_f0 / 2),
                                       (S_r, vp.k_rl, vp.xsi_rl, vp.lsi_rl, vp.Rw_r, Fz_r0 / 2),
                                       (S_r, vp.k_rr, vp.xsi_rr, vp.lsi_rr, vp.Rw_r, Fz_r0 / 2)):
            xs = S / k - xsi
            arms.append(Rw - Fz / vp.kt + lsi - xs + xsi)
        h_x = float(np.mean(arms))
    elif load_model == "nominal":
        M = vp.m
        Fz_f0 = vp.Wfl0 + vp.Wfr0
        Fz_r0 = vp.Wrl0 + vp.Wrr0
        aero_front = aero_c["Cl_front"] / aero_c["Cl"]
        h_x = vp.hcg
    else:
        raise ValueError(f"load_model must be one of {LOAD_MODELS}, got {load_model!r}")
    return float(M), float(Fz_f0), float(Fz_r0), float(aero_front), h_x


# ============================================================================
# Stage (a): the envelope
# ============================================================================
def build_envelope(ctx, v_grid=None, load_model="vehModel", n_v=40):
    """Precompute the quasi-steady limits on a speed grid.

    ctx must carry vp, pt (with EM4/ATD, i.e. after userOpts), mf and aero.
    v_grid defaults to n_v uniform points on [2, pt.Vmax] m/s. See the module
    docstring for every modelling choice. Cost ~5 ms (numpy, vectorised in v).
    """
    vp, pt, mf = ctx.vp, ctx.pt, ctx.mf
    aero_c = aero_coefficients(vp, getattr(ctx, "aero", None))
    EM4 = int(getattr(pt, "EM4", 0))
    ATD = int(getattr(pt, "ATD", 0))
    v_cap = float(pt.Vmax)
    if v_grid is None:
        v_grid = np.linspace(2.0, v_cap, int(n_v))
    v = np.sort(np.asarray(v_grid, dtype=float).reshape(-1))

    L = vp.l_f + vp.l_r
    M, Fz_f0, Fz_r0, aero_front, h_x = _load_basis(vp, aero_c, load_model)

    # ---- aero (vehModel sign convention: Cl < 0 is downforce) ----------------
    q = 0.5 * vp.rho * vp.A * v * v
    F_lift = -q * aero_c["Cl"]
    F_drag = q * aero_c["Cd"]
    Fz_f = Fz_f0 + aero_front * F_lift                      # axle loads, no transfer
    Fz_r = Fz_r0 + (1.0 - aero_front) * F_lift

    # ---- tyre peak forces per corner ----------------------------------------
    Fz0, shift = vp.Fz0, vp.Fz0_shift
    gam_f, gam_r = vp.gamma_fl_rad, vp.gamma_rl_rad         # mirrored camber -> same gamma^2

    def Dx(Fz):
        return _peak_force(peak_mu_x(Fz, mf, Fz0), Fz, shift)

    def Dy(Fz, gam):
        return _peak_force(peak_mu_y(Fz, mf, Fz0, gam), Fz, shift)

    # ---- lateral: yaw-balanced axles + first-order lateral load transfer -----
    sh_f, sh_r = vp.l_r / L, vp.l_f / L                     # Fy_f = M ay l_r/L, Fy_r = M ay l_f/L

    def axle_lat(Fz_axle, dF, gam):
        dF = np.clip(dF, 0.0, 0.5 * Fz_axle)
        return Dy(0.5 * Fz_axle + dF, gam) + Dy(0.5 * Fz_axle - dF, gam)

    def lat_capacity(ay):
        dF_f, dF_r = lateral_load_transfer(vp, M * ay * sh_f, M * ay * sh_r)
        return np.minimum(axle_lat(Fz_f, dF_f, gam_f) / (M * sh_f),
                          axle_lat(Fz_r, dF_r, gam_r) / (M * sh_r))

    ay0 = lat_capacity(np.zeros_like(v))
    ay_max = _bisect_fixed_point(lat_capacity, np.zeros_like(v), 2.0 * ay0 + 1.0)

    # ---- longitudinal: tyre capacities with first-order pitch transfer -------
    def loads(F, sign):       # sign=+1 traction (load to the rear), -1 braking
        dF = sign * F * h_x / L
        f = np.clip(Fz_f - dF, 0.0, Fz_f + Fz_r)
        r = np.clip(Fz_r + dF, 0.0, Fz_f + Fz_r)
        return f, r

    def axle_lon(F_axle):
        return 2.0 * Dx(0.5 * F_axle)

    def split_cap(cap_f, cap_r, front_share):
        with np.errstate(divide="ignore"):
            cf = cap_f / front_share if front_share > 0 else np.inf
            cr = cap_r / (1.0 - front_share) if front_share < 1 else np.inf
        return np.minimum(cf, cr)

    drive_all = (ATD == 1) or (EM4 == 1)
    brake_free = (ATD == 1) and (EM4 == 0)

    def trac_capacity(F):
        f, r = loads(F, +1.0)
        if drive_all:
            return axle_lon(f) + axle_lon(r)
        return split_cap(axle_lon(f), axle_lon(r), 1.0 - vp.Tdist)

    def brk_capacity(F):
        f, r = loads(F, -1.0)
        if brake_free:
            return axle_lon(f) + axle_lon(r)
        return split_cap(axle_lon(f), axle_lon(r), vp.brkB)

    hi_x = 2.0 * (axle_lon(Fz_f) + axle_lon(Fz_r)) + 1.0
    Fx_trac = _bisect_fixed_point(trac_capacity, np.zeros_like(v), hi_x)
    Fx_brk = _bisect_fixed_point(brk_capacity, np.zeros_like(v), hi_x)

    # ---- powertrain / brakes (vehModel + build_path_constraints) -------------
    Rw, gear = vp.Rw, vp.gear
    if EM4 == 1:     # four motors; vehModel: Om_motor_i = Om_i/gear, P_i = T_motor_i*Om_motor_i
        F_mot_trq = 4.0 * pt.Tmax * gear / Rw
        P_mot = 4.0 * pt.Pmax * gear ** 2
        v_rpm = pt.OMmax * gear * Rw
    else:            # one motor;   vehModel: Om_motor = mean(Om_i)*gear
        F_mot_trq = pt.Tmax * gear / Rw
        P_mot = pt.Pmax
        v_rpm = pt.OMmax * Rw / gear
    F_brk_trq = 2.0 * vp.Tbrake_max / Rw
    M_eff = M + 4.0 * vp.Jw / Rw ** 2
    Fz_tot0 = Fz_f0 + Fz_r0
    c_res0 = vp.f * Fz_tot0
    c_res2 = 0.5 * vp.rho * vp.A * (aero_c["Cd"] - vp.f * aero_c["Cl"])

    env = Envelope(v_grid=v, ay_max=ay_max, Fx_trac=Fx_trac, Fx_brk=Fx_brk,
                   ax_acc=np.zeros_like(v), ax_brk=np.zeros_like(v),
                   M=M, M_eff=float(M_eff), c_res0=float(c_res0), c_res2=float(c_res2),
                   F_mot_trq=float(F_mot_trq), P_mot=float(P_mot), v_rpm=float(v_rpm),
                   F_brk_trq=float(F_brk_trq), v_cap=v_cap, load_model=load_model,
                   F_lift=F_lift, F_drag=F_drag, Fz_front=Fz_f, Fz_rear=Fz_r)
    res = env.F_res(v)
    env.ax_acc = np.minimum((Fx_trac - res) / M, (env.F_mot(v) - res) / env.M_eff)
    env.ax_brk = np.minimum((Fx_brk + res) / M, (F_brk_trq + res) / env.M_eff)
    env.info = dict(EM4=EM4, ATD=ATD, ActAero=int(getattr(vp, "ActAero", 0)),
                    active_aero_as_static=int(getattr(vp, "ActAero", 0) > 0),
                    drive="all-wheel" if drive_all else "split Tdist",
                    brake="free split" if brake_free else "fixed brkB",
                    Cl=float(aero_c["Cl"]), Cd=float(aero_c["Cd"]),
                    aero_front=aero_front, h_x=h_x, Fz_static=Fz_tot0)
    return env


# ============================================================================
# Stage (b): the speed march
# ============================================================================
def apex_speeds(env, kabs, iters=60, tol=1e-9):
    """Largest v with v^2*|k| <= ay_max(v), capped at env.v_cap (vectorised
    fixed-point v <- sqrt(ay_max(v)/|k|) iterated down from the cap)."""
    kabs = np.abs(np.asarray(kabs, dtype=float))
    v = np.full(kabs.shape, env.v_cap, dtype=float)
    idx = np.nonzero(kabs > 0.0)[0]
    if idx.size:
        kk = kabs[idx]
        vv = np.full(idx.size, env.v_cap)
        for _ in range(iters):
            ay = np.maximum(np.interp(vv, env.v_grid, env.ay_max), 0.0)
            vn = np.minimum(env.v_cap, np.sqrt(ay / kk))
            done = np.max(np.abs(vn - vv)) < tol
            vv = vn
            if done:
                break
        v[idx] = vv
    return np.maximum(v, _V_FLOOR)          # degenerate setups (ay_max <= 0) stay marchable


def _lut(env):
    """Uniform-grid lookup tables (plain lists) for the scalar march loops."""
    vg = env.v_grid
    tabs = (env.ay_max, env.Fx_trac, env.Fx_brk)
    dv = np.diff(vg)
    if vg.size < 2 or not np.allclose(dv, dv[0], rtol=1e-9, atol=1e-12):
        vu = np.linspace(vg[0], vg[-1] if vg.size > 1 else vg[0] + 1.0, max(4 * vg.size, 2))
        tabs = tuple(np.interp(vu, vg, tb) for tb in tabs)
        vg = vu
    inv_dv = 1.0 / (vg[1] - vg[0]) if vg.size > 1 else 0.0
    out = [float(vg[0]), inv_dv, vg.size - 1]
    for tb in tabs:
        tb = np.asarray(tb, dtype=float)
        out += [tb.tolist(), np.append(np.diff(tb), 0.0).tolist()]
    return out


def _make_accel(env, lut, brake):
    """Scalar a(v, |k|) [m/s^2] for the march: the drive acceleration
    (brake=False; may be < 0 when drag wins) or the available deceleration
    (brake=True; > 0). Tyre part on the friction ellipse, powertrain or
    brake-torque part unreduced by ay; resistance included in both."""
    vlo, inv_dv, jmax, AY, dAY, FX, dFX, FB, dFB = lut
    c0, c2 = env.c_res0, env.c_res2
    invM, invMe = 1.0 / env.M, 1.0 / env.M_eff
    Ft, P, vrpm, Fb = env.F_mot_trq, env.P_mot, env.v_rpm, env.F_brk_trq
    T, dT = (FB, dFB) if brake else (FX, dFX)
    sqrt = math.sqrt

    def accel(v, kab):
        x = (v - vlo) * inv_dv
        if x <= 0.0:
            j, f = 0, 0.0
        else:
            j = int(x)
            if j >= jmax:
                j, f = jmax, 0.0
            else:
                f = x - j
        vv = v * v
        ay_lim = AY[j] + f * dAY[j]
        r = vv * kab / (ay_lim if ay_lim > _AY_FLOOR else _AY_FLOOR)
        e = 1.0 - r * r
        e = sqrt(e) if e > 0.0 else 0.0
        res = c0 + c2 * vv
        if brake:
            a = ((T[j] + f * dT[j]) * e + res) * invM           # tyre (ellipse)
            a2 = (Fb + res) * invMe                             # brake torque
        else:
            a = ((T[j] + f * dT[j]) * e - res) * invM           # tyre (ellipse)
            fm = (P / v if v * Ft > P else Ft) if v < vrpm else 0.0
            a2 = (fm - res) * invMe                             # motor torque / power / rpm
        return a2 if a2 < a else a

    return accel


def _heun_pass(v_start, kl, vl, ds, accel, reverse):
    """One march pass in w = v^2 (dw/ds = 2a) with Heun's predictor-corrector
    (2nd order in ds), capped at the apex speeds vl. reverse=True marches from
    the end with the deceleration (braking pass)."""
    n = len(kl)
    out = [0.0] * n
    sqrt, wmin, two_ds = math.sqrt, _V_FLOOR ** 2, 2.0 * ds
    if reverse:
        idx, step, i0 = range(n - 1, 0, -1), -1, n - 1
    else:
        idx, step, i0 = range(n - 1), 1, 0
    v = out[i0] = v_start
    for i in idx:
        j = i + step
        w = v * v
        k1 = accel(v, kl[i])
        wp = w + two_ds * k1
        vp = sqrt(wp) if wp > wmin else _V_FLOOR
        if vp > vl[j]:
            vp = vl[j]
        w = w + ds * (k1 + accel(vp, kl[j]))
        v = sqrt(w) if w > wmin else _V_FLOOR
        if v > vl[j]:
            v = vl[j]
        out[j] = v
    return out


def march(env, s, k, vi, ds_fine=None, s_out=None):
    """Forward/backward QSS speed march on the centreline (n = 0).

    s, k   : track arc length [m] and curvature [1/m] (kappa > 0 = left turn)
    vi     : initial speed [m/s] (the NLP's Xi; the final speed is free)
    ds_fine: march step [m] (default 1 m; the track is resampled uniformly)
    s_out  : optional stations (e.g. discretise()['s_full']) to resample onto

    Returns a dict: s, k, v, v_apex, v_fwd, v_bwd, t, ax, ay, lap_time, ds,
    vi_feasible (+ s_out, v_out, t_out, ax_out, ay_out, v_apex_out if s_out).
    Both passes integrate w = v^2 with Heun's method (lap-time error ~2e-4 at
    ds = 1 m); the segment time 2*ds/(v_i + v_i+1) is exact for a constant
    acceleration within the step.
    """
    s = np.asarray(s, dtype=float).reshape(-1)
    k = np.asarray(k, dtype=float).reshape(-1)
    ds_fine = 1.0 if ds_fine is None else float(ds_fine)
    n = max(int(round((s[-1] - s[0]) / ds_fine)), 1) + 1
    s_f = np.linspace(s[0], s[-1], n)
    ds = (s[-1] - s[0]) / (n - 1)
    k_f = np.interp(s_f, s, k)
    kabs = np.abs(k_f)

    v_apex = apex_speeds(env, kabs)
    lut = _lut(env)
    kl, vapl = kabs.tolist(), v_apex.tolist()
    v0 = min(float(vi), vapl[0])
    v_fwd = np.array(_heun_pass(v0, kl, vapl, ds, _make_accel(env, lut, False), False))
    v_bwd = np.array(_heun_pass(vapl[-1], kl, vapl, ds, _make_accel(env, lut, True), True))
    v = np.minimum(v_fwd, v_bwd)

    t = np.concatenate(([0.0], np.cumsum(2.0 * ds / (v[:-1] + v[1:]))))
    ax = 0.5 * np.gradient(v * v, s_f)
    ay = v * v * k_f
    out = dict(s=s_f, k=k_f, v=v, v_apex=v_apex, v_fwd=v_fwd, v_bwd=v_bwd, t=t,
               ax=ax, ay=ay, lap_time=float(t[-1]), ds=ds,
               vi_feasible=bool(v[0] >= float(vi) - 1e-9))
    if s_out is not None:
        so = np.asarray(s_out, dtype=float).reshape(-1)
        out.update(s_out=so, v_out=np.interp(so, s_f, v), t_out=np.interp(so, s_f, t),
                   ax_out=np.interp(so, s_f, ax), ay_out=np.interp(so, s_f, ay),
                   v_apex_out=np.interp(so, s_f, v_apex))
    return out
