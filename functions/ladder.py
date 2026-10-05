"""ladder.py - multi-fidelity solve ladder (roadmap Item 10): tier registry + seed builders.

A cold 23-state solve climbs a ladder of model fidelities, each rung seeding the
next (cross-tier warm starting):

    const  constant guesses (vx = vi, T_drive 0.85 Tmax, zero steer): MLTP_initial's
           legacy block, no model at all
    qss    rung 0: the quasi-steady g-g-v march of functions/ggv.py on the centreline,
           v(s), ax(s), ay(s), t(s) in 10-65 ms (qss_profile)
    m7     rung 1: the 7-state bicycle NLP (MLTP_initial / vehModel_initial)
    m23    rung 2: the 23-state NLP (MLTP / vehModel)

LADDERS names the chains MLTP(ladder=...) accepts (seed tier first, 'm23' last):
'legacy' = const -> m7 -> m23 (the default, AUTO_LADDER), 'qss7' = qss -> m7 -> m23
(MLTP_initial(seed='qss')), 'qss23' = qss -> m23 (seed_m23, no 7-state solve).
Rung 1 -> rung 2 is MLTP.warmstart_guesses, unchanged.

Seed rules (all built in physical units, divided by the target's x_s / u_s / y_s,
clipped to its bounds, and knot 0 / knot N clipped into build_and_solve_nlp's
boundary box from Xi / Xf, NaN = free; states are evaluated at the knots AND at the
collocation points, s_col order = the Xkj column order):

  longitudinal (both)  the tyre force that holds the QSS acceleration,
                       F = M ax + c0 + c2 v^2, and the wheel torque including the
                       wheel spin-up, T_w = Rw F + n_w Jw ax / Rw (lon_model,
                       wheel_torque); T_w >= 0 drives (per motor T_w / (gear n),
                       clipped to [0, Tmax], one motor also to its power limit),
                       T_w < 0 brakes (T_brake = T_w / 2, clipped to [-Tbrake_max, 0]),
                       so T_motor * T_brake = 0 at every knot (split_torque)
  m7 ('steady')        r = v k; the rear slip sa_r inverts the 7-state lateral Magic
                       Formula for Fy_r = M v^2 k l_f / (0.85 L); sideslip
                       beta = l_r k - sa_r; vx = v cos(beta), vy = v sin(beta),
                       eps = -beta (no n drift, path speed v); n = 0; Om = vx / Rw;
                       delta = L k + sa_f - sa_r; ltx from the ltx_eq row with zero
                       steer force (seed_m7)
  m23 ('neutral')      vx = v, vy = r = n = OPT_e, eps = 0, the 18 chassis rows
                       quasi-static (quasi_static_states); motor / brake torques from
                       the longitudinal rule, ATD 0.25, wings 0, delta 0: QSS supplies
                       the speed and torque profile, the NLP builds the turn and the
                       line together (seed_m23)

Rejected by measurement (Sturn / BCN, ma57, MF205): a turning 23-state seed
(r = v k, delta = L k: 474 iterations to the opposite-line branch, the stiff yaw and
wheel-spin modes start far from equilibrium at corner entry), a Newton projection of
the fast states onto their quasi-steady manifold (m23 929 iterations, m7 infeasible),
a zero-sideslip / neutral m7 seed, the 'nominal' load basis, kron collocation seeds.

Homotopy hook (MLTP(homotopy=...)): a continuation on a global tyre-friction scale,
friction_overrides scales pDx1, pDx2, pDy1, pDy2 (exact mu scaling) and
homotopy_schedule validates the scale schedule. OPT_e is no lever for the 23-state
NLP: it only sets the Xi / Xf box and the vx / Om lower bounds there.

numpy only: this module never imports casadi (functions.ggv, also numpy only, is
imported inside qss_profile / lon_model), so tests can pass any namespace with
nx, nu, ny, x_s, u_s, (y_s), x_min, x_max, u_min, u_max, (y_min, y_max) as ``m``.
"""

import math

import numpy as np

from functions.warmstart import _NEUTRAL_INPUT

TIERS = {
    "const": "constant guesses: vx = vi, vy = r = n = OPT_e, Om = vx/Rw, T_drive 0.85 Tmax, "
             "no brake, zero steer (MLTP_initial's legacy block)",
    "qss": "quasi-steady g-g-v march on the centreline (functions/ggv.py): v(s), ax(s), "
           "ay(s), t(s) in milliseconds",
    "m7": "7-state bicycle NLP (MLTP_initial / vehModel_initial), lumped Magic Formula",
    "m23": "23-state NLP (MLTP / vehModel): suspension, four wheels, Pacejka 5.2",
}
LADDERS = {
    "legacy": ("const", "m7", "m23"),
    "qss7": ("qss", "m7", "m23"),
    "qss23": ("qss", "m23"),
}
SEEDS = ("const", "qss")                # MLTP_initial(seed=...): the rung below m7
AUTO_LADDER = "legacy"                  # what ladder='auto' / None resolves to (owner call)
# MA57 work-space factor on the QSS rungs: two QSS-seeded solves ended 'Not enough
# memory' (Insufficient_Memory) right after 'Reallocating memory for MA57'; 3.0 removed
# every reallocation (numerically neutral in 2 of 2 bit-for-bit checks, ~3x the MA57
# factor memory: BCN ~60 -> 180 MB). Set with setdefault, so ipopt_overrides win.
MA57_PRE_ALLOC = 3.0
HOMOTOPY_DEFAULT = (1.2, 1.1, 1.0)      # MLTP(homotopy=True): 0.1 steps (Sturn EM4)
FRICTION_COEFFS = ("pDx1", "pDx2", "pDy1", "pDy2")


def resolve_ladder(ladder="auto"):
    """(name, chain) for a ladder argument: None / 'auto' -> AUTO_LADDER, else a key of
    LADDERS. Any other value raises ValueError (before any solve)."""
    name = AUTO_LADDER if ladder is None or ladder == "auto" else ladder
    if not isinstance(name, str) or name not in LADDERS:
        raise ValueError(f"ladder must be 'auto' or one of {sorted(LADDERS)}, got {ladder!r}")
    return name, LADDERS[name]


def qss_profile(ctx, load_model="vehModel", ds_fine=1.0):
    """Rung 0: functions.ggv.build_envelope(ctx, load_model) + march(env, ctx.track.s,
    ctx.track.k, ctx.vi, ds_fine) on the fine grid. Returns the march dict (s, k, v,
    ax, ay, t, lap_time, ...) plus 'env'. ctx must be past userOpts (pt.EM4 / pt.ATD,
    ctx.aero). Sample it at any station with np.interp on prof['s']."""
    from functions.ggv import build_envelope, march
    env = build_envelope(ctx, load_model=load_model)
    prof = march(env, ctx.track.s, ctx.track.k, float(ctx.vi), ds_fine=ds_fine)
    prof["env"] = env
    return prof


def quasi_static_states(vp, vx):
    """Physical seeds of the 18 states after eps (rows 5-22 of the 23-state vector:
    Om_fl, Om_fr, Om_rl, Om_rr, the ten suspension / unsprung states, zt_fl..zt_rr)
    for the speeds ``vx`` [m/s]: rolling wheels (Om = vx / Rw_f, vx / Rw_r),
    suspension at rest (0) and the static tyre deflections W0 / kt. Shared by
    MLTP.warmstart_guesses (7-state init), MLTP.warmstart_refined (mesh refinement)
    and seed_m23 (QSS profile); MLTP re-exports it."""
    vx = np.asarray(vx, dtype=float).reshape(-1)
    n = vx.size
    z0 = np.zeros(n)
    return np.vstack([vx / vp.Rw_f, vx / vp.Rw_f, vx / vp.Rw_r, vx / vp.Rw_r,
                      z0, z0, z0, z0, z0, z0, z0, z0, z0, z0,
                      (vp.Wfl0 / vp.kt) * np.ones(n), (vp.Wfr0 / vp.kt) * np.ones(n),
                      (vp.Wrl0 / vp.kt) * np.ones(n), (vp.Wrr0 / vp.kt) * np.ones(n)])


def lon_model(ctx, tier):
    """Longitudinal balance of a target model, dict(M, n_wheels, c0, c2, c_drag):
    the tyre force that holds an acceleration ax at speed v is M ax + c0 + c2 v^2,
    with n_wheels wheel inertias Jw to spin up and c_drag v^2 the aero drag.

      'm7'   vehModel_initial: M = vp.m, two wheels, c0 = f (Wfl0+Wfr0+Wrl0+Wrr0),
             c2 = q (Cd_max + f Cl_max), q = rho A / 2: its own drag and its aero
             LIFT (f_lift = q Cl v^2 adds to the axle loads; DATA_AA Cl < 0)
      'm23'  vehModel at the static wing angles: M = vp.ms, four wheels,
             c0 = f (ms + 4 mus) g, c2 = q (Cd - f Cl); equal to the QSS envelope's
             M / c_res0 / c_res2 with load_model='vehModel'"""
    vp = ctx.vp
    q = 0.5 * vp.rho * vp.A
    if tier == "m7":
        Cd = float(np.max(np.atleast_1d(vp.Cd)))
        Cl = float(np.max(np.atleast_1d(vp.Cl)))
        W = vp.Wfl0 + vp.Wfr0 + vp.Wrl0 + vp.Wrr0
        return dict(M=float(vp.m), n_wheels=2, c0=float(vp.f * W),
                    c2=float(q * (Cd + vp.f * Cl)), c_drag=float(q * Cd))
    if tier == "m23":
        from functions.ggv import aero_coefficients, _load_basis
        ac = aero_coefficients(vp, getattr(ctx, "aero", None))
        M, Fz_f0, Fz_r0, _, _ = _load_basis(vp, ac, "vehModel")
        return dict(M=float(M), n_wheels=4, c0=float(vp.f * (Fz_f0 + Fz_r0)),
                    c2=float(q * (ac["Cd"] - vp.f * ac["Cl"])), c_drag=float(q * ac["Cd"]))
    raise ValueError(f"lon_model: tier must be 'm7' or 'm23', got {tier!r}")


def wheel_torque(lon, vp, v, ax):
    """(T_w, F): the total wheel torque [Nm] T_w = Rw F + n_wheels Jw ax / Rw that
    holds the acceleration ax [m/s^2] at speed v [m/s] (tyre force plus wheel
    spin-up), and the tyre force F = M ax + c0 + c2 v^2 [N] (``lon`` from lon_model)."""
    v = np.asarray(v, dtype=float)
    ax = np.asarray(ax, dtype=float)
    F = lon["M"] * ax + lon["c0"] + lon["c2"] * v * v
    return vp.Rw * F + lon["n_wheels"] * vp.Jw * ax / vp.Rw, F


def split_torque(T_w, v, vp, pt, n_motors=1):
    """(T_motor per motor, T_brake) [Nm] for a total wheel torque T_w. Both models put
    gear * sum(T_motor) + 2 T_brake at the wheels (any ATD / Tdist / brkB split).
    T_w >= 0: T_motor = T_w / (gear n_motors) clipped to [0, Tmax] and, for a single
    motor, to the power limit Pmax Rw / (gear v); T_brake = 0. T_w < 0: T_brake =
    T_w / 2 clipped to [-Tbrake_max, 0]; T_motor = 0. So T_motor * T_brake = 0
    everywhere, and both are non-decreasing in T_w."""
    T_w = np.asarray(T_w, dtype=float)
    v = np.maximum(np.asarray(v, dtype=float), 1e-3)
    drive = np.maximum(T_w, 0.0) / vp.gear / int(n_motors)
    if int(n_motors) == 1:
        drive = np.minimum(drive, pt.Pmax * vp.Rw / (vp.gear * v))
    T_motor = np.clip(drive, 0.0, pt.Tmax)
    T_brake = np.clip(np.minimum(T_w, 0.0) / 2.0, -vp.Tbrake_max, 0.0)
    return T_motor, T_brake


def _mf_inverse(D, B, C, E, F, sa_max=0.25, n=2001):
    """Slip angle [rad] with D sin(C atan(B x - E (B x - atan(B x)))) = F on the rising
    branch of the curve (|F| beyond the peak is capped at the peak slip)."""
    xs = np.linspace(0.0, sa_max, n)
    Bx = B * xs
    f = np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))
    ipk = int(np.argmax(f))
    xs, f = xs[:ipk + 1], f[:ipk + 1]
    F = np.asarray(F, dtype=float)
    ratio = np.clip(np.abs(F) / max(float(D), 1e-9), 0.0, f[-1])
    return np.sign(F) * np.interp(ratio, f, xs)


def _profile_at(prof, s):
    """QSS speed and acceleration at the stations s (linear in the fine grid)."""
    s = np.asarray(s, dtype=float).reshape(-1)
    return np.interp(s, prof["s"], prof["v"]), np.interp(s, prof["s"], prof["ax"])


def _vec(v):
    return np.asarray(v, dtype=float).reshape(-1)


def _scaled(phys, scale, lo, hi):
    """phys / scale, clipped row-wise to [lo, hi] (scaled bounds)."""
    return np.clip(np.asarray(phys, dtype=float) / _vec(scale)[:, None],
                   _vec(lo)[:, None], _vec(hi)[:, None])


def boundary_box(x_min, x_max, X, x_s, OPT_e):
    """The scaled box build_and_solve_nlp puts on knot 0 (X = Xi) or knot N (X = Xf):
    [fmax(x_min, X/x_s - OPT_e), fmin(x_max, X/x_s + OPT_e)], NaN entries = free."""
    X, x_s = _vec(X), _vec(x_s)
    return (np.fmax(_vec(x_min), X / x_s - OPT_e), np.fmin(_vec(x_max), X / x_s + OPT_e))


def _clip_ends(x0, m, Xi, Xf, OPT_e, rederive):
    """Clip knot 0 into the Xi box and knot N into the Xf box (in place), then
    re-derive the rows that follow from vx (wheel speeds, chassis) at a clipped end:
    ``rederive(j)`` rewrites the dependent rows of column j from x0[0, j]."""
    for j, X in ((0, Xi), (x0.shape[1] - 1, Xf)):
        if X is None:
            continue
        lo, hi = boundary_box(m.x_min, m.x_max, X, m.x_s, OPT_e)
        col = np.clip(x0[:, j], lo, hi)
        if not np.array_equal(col, x0[:, j]):
            x0[:, j] = col
            rederive(j)
    return x0


def seed_m7(ctx, m, disc, prof):
    """Scaled 7-state guesses {x0, u0, y0, xc0} from the QSS profile ``prof``
    (qss_profile) on the grid ``disc`` (transcription.discretise), the 'steady' rule:

      r = v k; the rear slip sa_r inverts the 7-state lateral Magic Formula (by, cy,
      ey, D = mu_r (Wrl0 + Wrr0)) on its rising branch for Fy_r = M v^2 k l_f / (0.85 L);
      sideslip beta = l_r k - sa_r; vx = v cos(beta), vy = v sin(beta), eps = -beta
      (dn/ds = 0, path speed v); n = 0; Om_f = Om_r = vx / Rw. Same rule at the knots
      and at the collocation points (profile values at s_col).
      Knot inputs: T_drive / T_brake from the longitudinal rule (lon_model 'm7');
      delta = L k + sa_f - sa_r, sa_f for Fy_f = M v^2 k l_r / (0.85 L) at the static
      front load. Aux: ltx = (F + q Cd_max v^2) hcg / l (the ltx_eq row, zero steer force).

    The QSS envelope mirrors vehModel (ay_max ~12.5 m/s^2 at low speed) while the
    7-state model's own limit is ~11.5 m/s^2 (0.85 lateral factor, aero lift), so the
    seed is a few % too fast in corners; the 7-state NLP repairs that."""
    vp, pt = ctx.vp, ctx.pt
    L = vp.l_f + vp.l_r
    lon = lon_model(ctx, "m7")
    ty = vp.tyre

    def peak(fz):                       # 7-state peak force mu * fz at a static axle load
        return (ty.mu + ty.pD2 * (fz - 2.0 * vp.Fz0) / (2.0 * vp.Fz0)) * fz

    D_f, D_r = peak(vp.Wfl0 + vp.Wfr0), peak(vp.Wrl0 + vp.Wrr0)

    def rule(s, k):
        v, ax = _profile_at(prof, s)
        k = _vec(k)
        ay = v * v * k                                  # yaw-balanced axle forces / 0.85
        sa_f = _mf_inverse(D_f, ty.by, ty.cy, ty.ey, lon["M"] * ay * vp.l_r / L / 0.85)
        sa_r = _mf_inverse(D_r, ty.by, ty.cy, ty.ey, lon["M"] * ay * vp.l_f / L / 0.85)
        beta = vp.l_r * k - sa_r
        vx = v * np.cos(beta)
        X = np.vstack([vx, v * np.sin(beta), v * k, np.zeros_like(v), -beta,
                       vx / vp.Rw, vx / vp.Rw])
        return X, v, ax, k, sa_f, sa_r

    Xk, v, ax, k, sa_f, sa_r = rule(disc["s_knot"], disc["k_knot"])
    Xc = rule(disc["s_col"], disc["k_col"])[0]
    T_w, F = wheel_torque(lon, vp, v, ax)
    T_drive, T_brake = split_torque(T_w, v, vp, pt, n_motors=1)
    delta = L * k + sa_f - sa_r
    ltx = (F + lon["c_drag"] * v * v) * vp.hcg / vp.l

    x0 = _scaled(Xk, m.x_s, m.x_min, m.x_max)
    xc0 = _scaled(Xc, m.x_s, m.x_min, m.x_max)
    x_s = _vec(m.x_s)

    def rederive(j):                    # Om_f = Om_r = vx / Rw at a clipped end knot
        om = x0[0, j] * x_s[0] / vp.Rw
        x0[5:7, j] = np.clip(om / x_s[5:7], _vec(m.x_min)[5:7], _vec(m.x_max)[5:7])

    _clip_ends(x0, m, getattr(ctx, "Xi_init", None), getattr(ctx, "Xf_init", None),
               ctx.OPT_e, rederive)
    u0 = _scaled(np.vstack([T_drive, T_brake, delta]), m.u_s, m.u_min, m.u_max)
    y0 = _scaled(ltx.reshape(1, -1), m.y_s, m.y_min, m.y_max)
    return {"x0": x0, "u0": u0, "y0": y0, "xc0": xc0}


def seed_m23(ctx, m, disc, prof, input_keys):
    """Scaled 23-state guesses {x0, u0, xc0} straight from the QSS profile ``prof``
    (ladder 'qss23'), the 'neutral lateral' rule: QSS supplies the speed and torque
    profile, the lateral part is the constant seed's (the NLP builds the turn and the
    line together).

      states   vx = v; vy = r = n = OPT_e and eps = 0 (physical units); rows 5-22 =
               quasi_static_states(vp, vx); the same at the knots and the
               collocation points
      inputs   by ``input_keys`` (ctx.input_keys): T_motor, or T_motor_fl..rr with
               EM4 (each the total / 4), and T_brake from the longitudinal rule
               (lon_model 'm23'); ATD 0.25 each (they sum to 1, ATD_eq holds); FW / RW /
               TW 0 (warmstart._NEUTRAL_INPUT, as warmstart_guesses); delta 0"""
    vp, pt = ctx.vp, ctx.pt
    e = float(ctx.OPT_e)
    lon = lon_model(ctx, "m23")

    def rule(s):
        v, ax = _profile_at(prof, s)
        E = np.full(v.size, e)
        return np.vstack([v, E, E, E, np.zeros(v.size), quasi_static_states(vp, v)]), v, ax

    Xk, v, ax = rule(disc["s_knot"])
    Xc = rule(disc["s_col"])[0]
    keys = [str(key).strip() for key in input_keys]
    n_mot = sum(1 for key in keys if key.startswith("T_motor"))
    T_w, _ = wheel_torque(lon, vp, v, ax)
    T_motor, T_brake = split_torque(T_w, v, vp, pt, n_motors=max(n_mot, 1))
    n = v.size
    rows = []
    for key in keys:
        if key.startswith("T_motor"):
            rows.append(T_motor)
        elif key == "T_brake":
            rows.append(T_brake)
        elif key in _NEUTRAL_INPUT:
            rows.append(np.full(n, float(_NEUTRAL_INPUT[key])))
        elif key == "delta":
            rows.append(np.zeros(n))
        else:
            raise ValueError(f"seed_m23: unknown input channel {key!r}")
    if len(rows) != int(m.nu):
        raise ValueError(f"seed_m23: {len(rows)} input channels for a model with nu = {m.nu}")

    x0 = _scaled(Xk, m.x_s, m.x_min, m.x_max)
    xc0 = _scaled(Xc, m.x_s, m.x_min, m.x_max)
    x_s = _vec(m.x_s)

    def rederive(j):                    # chassis rows quasi-static at a clipped end knot
        rows_ = quasi_static_states(vp, [x0[0, j] * x_s[0]])[:, 0]
        x0[5:, j] = np.clip(rows_ / x_s[5:], _vec(m.x_min)[5:], _vec(m.x_max)[5:])

    _clip_ends(x0, m, getattr(ctx, "Xi", None), getattr(ctx, "Xf", None), e, rederive)
    u0 = _scaled(np.vstack(rows), m.u_s, m.u_min, m.u_max)
    return {"x0": x0, "u0": u0, "xc0": xc0}


def friction_overrides(mf, scale, base=None):
    """vp_overrides for a tyre-friction continuation step: ``base`` (the caller's
    vp_overrides) first, then pDx1, pDx2, pDy1, pDy2 set to ``scale`` times their
    effective value (base's if it overrides one, else mf's). That scales vehModel's
    peak mu_x = pDx1 + pDx2 dfz and mu_y = (pDy1 + pDy2 dfz) / (1 + pDy3 gamma^2)
    exactly (the MF205 set overrides none of the four). scale 1.0 returns base
    unchanged (None stays None): the caller's own problem."""
    scale = float(scale)
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"friction_overrides: scale must be a finite number > 0, got {scale!r}")
    if scale == 1.0:
        return None if base is None else dict(base)
    out = dict(base or {})
    for name in FRICTION_COEFFS:
        out[name] = scale * float(out.get(name, getattr(mf, name)))
    return out


def homotopy_schedule(h):
    """Validated friction-scale schedule for MLTP(homotopy=...): a tuple of >= 2
    finite floats > 0 whose last entry is 1.0 (the caller's own problem), e.g.
    (1.2, 1.1, 1.0). True gives HOMOTOPY_DEFAULT. Anything else raises ValueError."""
    if h is True:
        return HOMOTOPY_DEFAULT
    if isinstance(h, (str, bytes, dict)) or not hasattr(h, "__iter__"):
        raise ValueError(f"homotopy must be a sequence of friction scales ending in 1.0, got {h!r}")
    out = []
    for x in h:
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)):
            raise ValueError(f"homotopy: scales must be numbers, got {x!r}")
        x = float(x)
        if not (math.isfinite(x) and x > 0.0):
            raise ValueError(f"homotopy: scales must be finite and > 0, got {x!r}")
        out.append(x)
    if len(out) < 2:
        raise ValueError(f"homotopy: need at least 2 scales (the last 1.0), got {len(out)}")
    if out[-1] != 1.0:
        raise ValueError(f"homotopy: the last scale must be 1.0 (the caller's problem), got {out[-1]!r}")
    return tuple(out)


def ladder_record(name, chain, qss_lap_s=np.nan, qss_wall_s=np.nan, m7_iters=-1,
                  m7_status="none", m7_lap_s=np.nan, m7_wall_s=np.nan):
    """data['ladder'] of an MLTP result (savemat-safe): the ladder name, its chain
    joined with '->', and what the lower rungs cost in this call. A rung that did not
    run keeps -1 / 'none' / NaN ('none' rather than '': loadmat returns '' as an empty
    array)."""
    return {
        "name": str(name),
        "chain": "->".join(str(t) for t in chain),
        "qss_lap_s": float(qss_lap_s),
        "qss_wall_s": float(qss_wall_s),
        "m7_iters": int(m7_iters),
        "m7_status": str(m7_status or "none"),
        "m7_lap_s": float(m7_lap_s),
        "m7_wall_s": float(m7_wall_s),
    }


def homotopy_record(steps):
    """data['homotopy'] (savemat-safe) from one dict per continuation step with keys
    scale, iters, status, lap, wall (+ optional warm_start, m7_iters): arrays scales,
    iters, lap_s, wall_s, m7_iters and the statuses / warm-start modes joined by ', '."""
    steps = list(steps)
    if not steps:
        raise ValueError("homotopy_record: no steps")
    return {
        "scales": np.array([float(st["scale"]) for st in steps]),
        "iters": np.array([int(st["iters"]) for st in steps], dtype=int),
        "status": ", ".join(str(st["status"]) for st in steps),
        "lap_s": np.array([float(st["lap"]) for st in steps]),
        "wall_s": np.array([float(st["wall"]) for st in steps]),
        "warm_start": ", ".join(str(st.get("warm_start", "?")) for st in steps),
        "m7_iters": np.array([int(st.get("m7_iters", -1)) for st in steps], dtype=int),
    }
