"""correlate.py - fit knobs of the QSS tier to a logged lap (numpy and scipy).

E6 is run_stage with its ledger. E7a is grip_fit. E8a is drag_fit. E8b is lift_fit.
A log is a dict of SI arrays on one time base, with the model signs (read_monster and read_csv give one).
It holds t [s], vx [m/s], ax and ay [m/s^2] (ay > 0 in a left turn).
A stage can need more channels. Tm is the total motor torque [Nm]. throttle is the torque in units of Tmax.
brake is 0 or 1. xs_fl, xs_fr, xs_rl and xs_rr are the damper compressions [m].
ctx is a Ctx after userOpts, for the car as it ran. A fit restores every knob of ctx that it moves.
A fit gives an effective value of the QSS, not a property of the tyre or the car.
This module imports no casadi. Only nlp_forecast needs it, through setup_gradients.
"""
import os
import time

import numpy as np
from scipy.optimize import brentq

from functions.context import Ctx
from functions.ggv import aero_coefficients, build_envelope, march
from functions.ladder import lon_model
from functions.sweep import array_sha, read_json, write_json_atomic
from userOpts import userOpts
from vehParams import MF_KEYS

# ponytail: a fit keeps the whole log in memory (6.4 MB for 30 laps at 50 Hz). Stream or decimate above 1e7 samples.
STAGES = ("grip", "drag", "downforce")      # the order of Canopy: apex speed, end-of-straight speed, fast corners
# A lap cannot separate these names from LMUY or LMUX (docs/tyre_scaling_track_method.md, tables 4 and 5).
NEVER = {"LFZO": "collinear with LMUY on one circuit",
         "Fz0_shift": "acts in the same direction as LMUX and LMUY in the QSS",
         "LKX": "the QSS ignores slip stiffness", "LKY": "the QSS ignores slip stiffness"}
GAMMA_MAX = 15.0                            # Brun et al. 2001: critical values 5 to 20


def _set(ctx, name, val):
    setattr(ctx.mf if name in MF_KEYS else ctx.vp, name, val)


def _get(ctx, name):
    return getattr(ctx.mf if name in MF_KEYS else ctx.vp, name)


def _lerp(e, tab, x):
    return np.interp(x, e.v_grid, tab)


def _lap(ctx):
    return march(build_envelope(ctx), ctx.track.s, ctx.track.k, ctx.vi)["lap_time"]


def _decel(e, v, a):
    """Return (tyre part, brake torque part) of the QSS braking limit [m/s^2] at speed v and lateral acceleration a."""
    uy = np.clip(a / _lerp(e, e.ay_max, v), 0.0, 1.0)
    return ((_lerp(e, e.Fx_brk, v) * np.sqrt(1 - uy ** 2) + e.F_res(v)) / e.M, (e.F_brk(v) + e.F_res(v)) / e.M_eff)


def gamma(S):
    """Return the collinearity index 1 / sqrt(lambda_min) of S'S, for S (n_obs, n_par).
    The function scales each column of S to norm 1. The index is 1 for orthogonal columns.
    A zero column or a rank loss gives inf. Above GAMMA_MAX the data cannot tell the columns apart."""
    S = np.atleast_2d(np.asarray(S, float))
    nrm = np.linalg.norm(S, axis=0)
    if not np.all(nrm > 1e-12 * max(nrm.max(), 1e-300)):
        return np.inf
    Z = S / nrm
    lam = np.linalg.eigvalsh(Z.T @ Z)[0]
    return np.inf if lam <= 1e-14 else float(1.0 / np.sqrt(lam))


def identifiable(ctx, names, obs, sigma=0.005, rel=0.02, inflate=1.0):
    """Test if the log can tell the knobs in names apart. Return dict(gamma, se, never, ok).
    obs(ctx) gives the model values at the samples of the log, as a 1-D array.
    Example: the QSS peak ay at the speeds of the limit samples.
    The columns of the test are d ln obs / d ln p (central difference, step rel).
    se[name] is inflate * sigma * sqrt(inv(S'S)), in ln units. It is the standard error for independent noise sigma
    on every sample, so it is a lower bound. For neighbouring samples of one event,
    set inflate to sqrt(samples per event).
    never lists the names that are in NEVER. ok is True when gamma <= GAMMA_MAX and never is empty."""
    base = np.asarray(obs(ctx), float)
    cols = []
    for n in names:
        p0 = _get(ctx, n)
        try:
            _set(ctx, n, p0 * (1 + rel))
            hi = np.asarray(obs(ctx), float)
            _set(ctx, n, p0 * (1 - rel))
            lo = np.asarray(obs(ctx), float)
        finally:
            _set(ctx, n, p0)
        cols.append((hi - lo) / (np.log1p(rel) - np.log1p(-rel)) / base)
    S = np.column_stack(cols)
    g = gamma(S)
    se = inflate * sigma * np.sqrt(np.diag(np.linalg.inv(S.T @ S))) if np.isfinite(g) else np.full(len(names), np.inf)
    never = [n for n in names if n in NEVER]
    return dict(gamma=g, se=dict(zip(names, map(float, se))), never=never, ok=bool(g <= GAMMA_MAX and not never))


def check_ax(log, win=0.5, floor=0.3):
    """Return the correlation of ax with the moving average (win [s]) of d vx / dt.
    Raise ValueError when it is below floor. The usual causes are a wrong ax sign, a wrong channel or a time lag.
    Return nan when the log has no t or no ax, when fewer than 20 samples are usable, or when the speed is constant."""
    if "t" not in log or "ax" not in log:
        return np.nan
    t, v, ax = (np.asarray(log[k], float) for k in ("t", "vx", "ax"))
    w = max(1, min(int(round(win / np.median(np.diff(t)))), t.size))     # np.convolve keeps the log length
    dv = np.gradient(np.convolve(v, np.ones(w) / w, mode="same"), t)
    ok = np.isfinite(dv) & np.isfinite(ax)
    ok[:w], ok[-w:] = False, False                                      # the average is wrong at both ends
    if ok.sum() < 20 or np.std(dv[ok]) < 1e-6:
        return np.nan
    c = float(np.corrcoef(ax[ok], dv[ok])[0, 1])
    if c < floor:
        raise ValueError(f"ax does not follow d vx / dt (correlation {c:.2f}). "
                         "Examine the sign, the channel and the time lag of ax.")
    return c


def grip_fit(log, ctx, q=0.98, u_min=0.6, ax_tol=3.0, smooth=0.25, lam=(0.3, 2.0)):
    """Fit LMUY and LMUX of ctx to the near-limit samples of a log (E7a). Return a dict with these keys.
    LMUY, LMUX: the fitted values. They are nan if the log does not identify the knob. why_y or why_x gives the reason.
    n_y, n_x: the number of near-limit cornering samples, and of tyre-limited trail braking samples.
    n_events: the number of corners in the limit set (a gap of more than 0.5 s starts a new corner).
    rms_before, rms_after: the rms of ln ay on the limit set, before and after the LMUY fit.
    vt: the speeds of the limit set. vb, ab: the speeds and |ay| of the trail braking set.
    The fit selects samples on vx, ax and ay averaged over smooth [s], and takes their values from the raw samples.
    So it does not select noise peaks. lam is the bracket of the root search.
    LMUY: the samples have |ax| < ax_tol and |ay| / ay_max(vx) > u_min. The top 1 - q share of them is the limit set.
    The fit divides out the friction ellipse (the tyre part of ax). LMUY is the root of mean ln|ay| = mean ln ay_max.
    LMUX: the samples brake (ax < -ax_tol) and corner at |ay| / ay_max > u_min.
    The tyre braking limit of the model must be below 0.97 times its brake torque limit for them.
    LMUX is the root of mean ln(-ax) = mean ln(tyre braking limit), on the top 1 - q share by -ax / limit.
    The fit does not use a raw sample with ax >= 0 (a spike) in this share.
    In a straight line the brake torque limits the QSS (11.4 to 12.8 m/s^2 for the default car, against 15.5 to 17.6
    for the tyre). So only trail braking shows the tyre. Fewer than 20 such samples give nan."""
    check_ax(log)
    v0, ax0, ay0 = (np.asarray(log[k], float) for k in ("vx", "ax", "ay"))
    dt = float(np.median(np.diff(log["t"])))
    w = max(1, min(int(round(smooth / dt)), v0.size))                   # np.convolve keeps the log length
    v, ax, ay = (np.convolve(x, np.ones(w) / w, mode="same") for x in (v0, ax0, ay0)) if w > 1 else (v0, ax0, ay0)
    a, a0 = np.abs(ay), np.abs(ay0)
    ly0, lx0 = ctx.mf.LMUY, ctx.mf.LMUX
    out = dict(LMUY=np.nan, LMUX=np.nan, n_y=0, n_x=0, n_events=0, why_y="", why_x="", rms_before=np.nan,
               rms_after=np.nan, vt=np.empty(0), vb=np.empty(0), ab=np.empty(0))

    def env_at(ly=ly0, lx=lx0):
        ctx.mf.LMUY, ctx.mf.LMUX = ly, lx
        return build_envelope(ctx)

    try:
        env = env_at()
        ok = (v > env.v_grid[0]) & (v < env.v_grid[-1])
        Ft = env.M * ax + env.F_res(v)                                  # the tyre force that the car used
        Fcap = np.where(Ft >= 0, _lerp(env, env.Fx_trac, v), _lerp(env, env.Fx_brk, v))
        ell = np.sqrt(np.clip(1.0 - (Ft / Fcap) ** 2, 1e-6, 1.0))
        u = a / (_lerp(env, env.ay_max, v) * ell)
        sel = ok & (np.abs(ax) < ax_tol) & (u > u_min)
        out["n_y"] = int(sel.sum())
        if sel.sum() < 20:
            out["why_y"] = "fewer than 20 near-limit cornering samples"
        else:
            top = sel & (u >= np.quantile(u[sel], min(q, 1.0 - 5.0 / sel.sum())))
            vt, lt = v0[top], np.log(a0[top] / ell[top])
            gap = max(1, int(round(0.5 / dt)))
            out["vt"], out["n_events"] = vt, 1 + int(np.count_nonzero(np.diff(np.flatnonzero(top)) > gap))

            def res(x):
                e = env_at(ly=x)
                return lt - np.log(_lerp(e, e.ay_max, vt))

            if np.mean(res(lam[0])) * np.mean(res(lam[1])) < 0:
                out["LMUY"] = brentq(lambda x: np.mean(res(x)), *lam, xtol=1e-6)
                out["rms_before"], out["rms_after"] = (float(np.sqrt(np.mean(res(x) ** 2))) for x in (ly0, out["LMUY"]))
            else:
                out["why_y"] = "no root: the log is not at the limit, or the model is far off"
        ly = out["LMUY"] if np.isfinite(out["LMUY"]) else ly0
        lx = lx0
        for _ in range(2):                                              # the tyre-limited set depends on LMUX
            out["LMUX"] = np.nan                                        # a pass that stops gives no value
            e = env_at(ly=ly, lx=lx)
            tyre, trq = _decel(e, v, a)
            selx = ok & (ax < -ax_tol) & (a / _lerp(e, e.ay_max, v) > u_min) & (tyre < 0.97 * trq)
            out["n_x"] = int(selx.sum())
            if selx.sum() < 20:
                out["why_x"] = "fewer than 20 tyre-limited trail-braking samples (the brake torque binds first)"
                break
            rho = -ax / np.minimum(tyre, trq)
            topx = selx & (rho >= np.quantile(rho[selx], min(q, 1.0 - 5.0 / selx.sum()))) & (ax0 < 0)
            vb, db, ab = v0[topx], -ax0[topx], a0[topx]
            out["vb"], out["ab"] = vb, ab

            def fx(x):
                return np.mean(np.log(db)) - np.mean(np.log(_decel(env_at(ly=ly, lx=x), vb, ab)[0]))

            if not fx(lam[0]) * fx(lam[1]) < 0:                         # also stops on nan
                out["why_x"] = "no root"
                break
            lx = out["LMUX"] = brentq(fx, *lam, xtol=1e-6)
    finally:
        ctx.mf.LMUY, ctx.mf.LMUX = ly0, lx0
    return out


def grip_from_lap(ctx, ratio, lam=(0.5, 1.5)):
    """Return s with lap_QSS(s) / lap_QSS(1) = ratio. LMUX and LMUY both change by the factor s.
    Return nan if s is outside lam. ratio is the measured lap time divided by the simulated lap time of the same setup.
    The QSS gives only the elasticity. Its gap to the NLP (+4 % Sturn, +11 % BCN) comes from the centreline path.
    This gap drops out. s holds the line, the width, the driver margin and the model gap. It is not a tyre property."""
    lx0, ly0 = ctx.mf.LMUX, ctx.mf.LMUY
    t1 = _lap(ctx)

    def f(s):
        ctx.mf.LMUX, ctx.mf.LMUY = lx0 * s, ly0 * s
        return _lap(ctx) / t1 - ratio

    try:
        return brentq(f, *lam, xtol=1e-6) if f(lam[0]) * f(lam[1]) < 0 else np.nan
    finally:
        ctx.mf.LMUX, ctx.mf.LMUY = lx0, ly0


def drag_fit(log, ctx, eta=1.0, win=1.0, ay_tol=0.5, v_min=20.0, coast=False):
    """Fit Cd_scale and the rolling coefficient f of ctx to the straights of a log (E8a). Return a dict.
    A window of win [s] counts when all its samples are straight. Straight means |ay| < ay_tol, no brake and vx > v_min.
    If the log has no brake, a sample with d vx / dt below -3 m/s^2 (average over 0.5 s) counts as braking.
    The default car coasts at -1.4 m/s^2 at 75 m/s. Light braking above -3 m/s^2 then counts as a coast.
    Each window gives:
        eta gear/Rw int Tm v dt - int M_eff ax v dt = c0 int v dt + c2 int v^3 dt.
    Here c0 = f sum(Fz0) and c2 = q (Cd - f Cl). Also q = rho A / 2 and M_eff = M + 4 Jw / Rw^2.
    functions.ladder.lon_model gives M and c0.
    Without ax in the log, the kinetic term M_eff v^2 / 2 comes from vx alone. This is noisier.
    Tm is the sum over the motors. When the log has no Tm, the fit uses throttle * Tmax (times 4 with four motors).
    eta is the wheel torque divided by the logged motor torque. An error of 5 % in eta moves Cd_scale by 14 %.
    So with an uncalibrated torque channel, set coast=True. The fit then keeps only the windows with Tm = 0.
    Return dict(Cd_scale, f, se_Cd_scale, gamma, n, ok). ok is True when gamma <= GAMMA_MAX and n >= 8 windows."""
    check_ax(log)
    vp = ctx.vp
    t, v, ay = (np.asarray(log[k], float) for k in ("t", "vx", "ay"))
    n_mot = 4 if ctx.pt.EM4 else 1                                      # throttle is the mean of the motors, Tm the sum
    Tm = np.asarray(log["Tm"], float) if "Tm" in log else n_mot * ctx.pt.Tmax * np.asarray(log["throttle"], float)
    ax = np.asarray(log["ax"], float) if "ax" in log else np.zeros_like(v)
    fin = np.isfinite(v) & np.isfinite(ay) & np.isfinite(Tm) & np.isfinite(ax)      # a dropout is not a straight sample
    v, ay, Tm, ax = (np.where(fin, x, 0.0) for x in (v, ay, Tm, ax))
    if "brake" in log:
        brake = np.asarray(log["brake"], float) != 0
    else:                                   # ponytail: one constant -3 m/s^2 threshold. Log brake for light braking.
        w = max(1, min(int(round(0.5 / np.median(np.diff(t)))), t.size))
        brake = np.convolve(np.gradient(v, t), np.ones(w) / w, mode="same") < -3.0
    straight = fin & (np.abs(ay) < ay_tol) & ~brake & (v > v_min)
    if coast:
        straight &= np.abs(Tm) < 1e-6
    lon = lon_model(ctx, "m23")
    M_eff = lon["M"] + lon["n_wheels"] * vp.Jw / vp.Rw ** 2

    def cum(y):
        return np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))]

    P, s, v3 = cum(eta * vp.gear * Tm * v / vp.Rw), cum(v), cum(v ** 3)
    KE = cum(M_eff * ax * v) if "ax" in log else 0.5 * M_eff * v ** 2
    bad = np.r_[0, np.cumsum(~straight)]
    e = np.searchsorted(t, np.arange(t[0], t[-1], win))
    i0, i1 = e[:-1], e[1:] - 1
    g = (i1 > i0) & (bad[i1 + 1] - bad[i0] == 0)                        # the window has no sample that is not straight
    i0, i1 = i0[g], i1[g]
    dt = t[i1] - t[i0]
    y = ((P[i1] - P[i0]) - (KE[i1] - KE[i0])) / dt
    A = np.column_stack([(s[i1] - s[i0]) / dt, (v3[i1] - v3[i0]) / dt])
    out = dict(Cd_scale=np.nan, f=np.nan, se_Cd_scale=np.nan, gamma=np.inf, n=int(len(y)), ok=False)
    if len(y) < 3:
        return out
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    q = 0.5 * vp.rho * vp.A
    ac = aero_coefficients(vp, ctx.aero)
    Cd1 = ac["Cd"] / vp.Cd_scale                                        # Cd at Cd_scale = 1
    out.update(Cd_scale=float((c[1] / q + vp.f * ac["Cl"]) / Cd1), f=float(c[0] * vp.f / lon["c0"]), gamma=gamma(A))
    if np.isfinite(out["gamma"]):
        out["se_Cd_scale"] = float(np.sqrt(np.linalg.inv(A.T @ A)[1, 1] * (r @ r) / max(len(y) - 2, 1)) / q / Cd1)
    out["ok"] = bool(out["gamma"] <= GAMMA_MAX and out["n"] >= 8)
    return out


def lift_fit(log, ctx, v_min=15.0, ay_tol=0.5, ax_tol=1.0):
    """Fit Cl_scale of ctx to the damper compression against speed squared (E8b). physics='corrected' only.
    For each axle, on the straight steady samples: mean(xs left, xs right) = c + a v^2 + b ax.
    In the corrected model each spring carries its own downforce. So a = q |Cl_wheel| / k,
    and Cl_scale = a / a(Cl_scale = 1).
    The fit ignores the drag pitch moment (2 to 4 % of a).
    The matlab model gives a front travel of the opposite sign (-2.6 mm at 60 m/s, against +10.0 mm corrected).
    So physics='matlab' raises ValueError.
    Return dict(front, rear, mean, gamma_front, gamma_rear)."""
    if getattr(ctx, "physics", "matlab") != "corrected":
        raise ValueError("lift_fit needs physics='corrected'. "
                         "The matlab model does not load its springs with downforce.")
    vp = ctx.vp
    v, ax, ay = (np.asarray(log[k], float) for k in ("vx", "ax", "ay"))
    xs = {w: np.asarray(log[f"xs_{w}"], float) for w in ("fl", "fr", "rl", "rr")}
    sel = (v > v_min) & (np.abs(ay) < ay_tol) & (np.abs(ax) < ax_tol)
    for x in xs.values():
        sel &= np.isfinite(x)                                           # a dropout is not a steady sample
    ac = aero_coefficients(vp, ctx.aero)
    q = 0.5 * vp.rho * vp.A
    out = {}
    for axle, wheels in (("front", ("fl", "fr")), ("rear", ("rl", "rr"))):
        y = np.mean([xs[w] for w in wheels], axis=0)[sel]
        A = np.column_stack([np.ones(sel.sum()), v[sel] ** 2, ax[sel]])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        a1 = q * np.mean([abs(ac[f"Cl_{w}"]) / getattr(vp, f"k_{w}") for w in wheels]) / vp.Cl_scale
        out[axle], out[f"gamma_{axle}"] = float(c[1] / a1), gamma(A)
    out["mean"] = 0.5 * (out["front"] + out["rear"])
    return out


def _frozen(ledger, stage):
    """Return the apply values of the earlier stages of the ledger.
    For each earlier stage, the last record that is not stale gives the values, also when its apply is empty."""
    last = {}
    for rec in (read_json(ledger)["stages"] if ledger and os.path.exists(ledger) else []):
        if rec["stage"] in STAGES[:STAGES.index(stage)] and not rec.get("stale"):
            last[rec["stage"]] = rec["apply"]
    return {k: v for a in last.values() for k, v in a.items()}


def ledger_add(path, rec):
    """Append the record rec of a stage to the JSON ledger at path. Return the ledger.
    One ledger holds one car. The circuit, physics and tyre_set must match the first record, else ValueError.
    A fitted value has a meaning only with the model that it came from.
    A stored stage that comes after rec in STAGES becomes stale, because rec can change the values that it used."""
    key = [rec["circuit"], rec["physics"], rec["tyre_set"]]
    led = read_json(path) if os.path.exists(path) else dict(version=1, key=key, stages=[])
    if led["key"] != key:
        raise ValueError(f"{path} holds {led['key']}, not {key}. Use another ledger file.")
    for old in led["stages"]:
        if old["stage"] in STAGES[STAGES.index(rec["stage"]) + 1:]:
            old["stale"] = True
    led["stages"].append(rec)
    write_json_atomic(path, led)
    return led


def run_stage(stage, log, circuit, ledger=None, link=True, lap=None, tags=None, **useropts_kwargs):
    """Fit one stage of STAGES to a log with the QSS (E6). Return the record.
    If ledger is a path, append the record to the ledger.
    The apply values of the earlier stages of the ledger (not stale) go into vp_overrides.
    They replace the same keys that you pass. useropts_kwargs go to userOpts (vp_overrides, physics, tyre_set ...).
    tags is a dict for the owner (driver, session ...).
    Grip stage: grip_fit on the accelerations of the log, and grip_from_lap on lap = (measured lap, simulated lap).
    The accelerations win. If they do not identify LMUY, or if the gate refuses their fit, the lap ratio sets LMUY
    and LMUX. log can be None if you pass lap.
    With link=True, LMUX takes the value of LMUY (one effective grip), and the gate tests LMUY alone.
    With link=False, LMUX keeps its fitted value, and the gate tests the two knobs.
    If the log does not identify LMUX, LMUX keeps its base value.
    The record has stage, method, fit, apply, identifiable (gamma, se, never, ok), detail,
    and qss_ratio = lap(apply) / lap(base).
    apply holds vp_overrides for the QSS only. A fit that the gate refuses (identifiable["ok"] False) gives no values.
    The full model keeps 1 until a study gives its own elasticity."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    kw = dict(useropts_kwargs)
    base = {**(kw.pop("vp_overrides", None) or {}), **_frozen(ledger, stage)}
    ctx = Ctx()
    userOpts(ctx, circuit=circuit, vp_overrides=base, **kw)
    t0 = time.perf_counter()
    method, apply, idn = None, {}, None
    if stage == "grip":
        f = grip_fit(log, ctx) if log is not None and "ay" in log else None
        s = grip_from_lap(ctx, lap[0] / lap[1]) if lap is not None else np.nan
        ly, lx = (f["LMUY"], f["LMUX"]) if f else (np.nan, np.nan)
        fit = dict(LMUY=ly, LMUX=lx, s_lap=s)
        detail = {k: f[k] for k in ("n_y", "n_x", "n_events", "why_y", "why_x", "rms_before", "rms_after")} if f else {}
        if np.isfinite(ly):                                             # LMUX alone gives no grip level
            names = ["LMUY"] if link or not np.isfinite(lx) else ["LMUY", "LMUX"]     # one linked knob, or two

            def obs(c):
                e = build_envelope(c)
                top = _lerp(e, e.ay_max, f["vt"])
                return np.r_[top, _decel(e, f["vb"], f["ab"])[0]] if "LMUX" in names else top

            idn = identifiable(ctx, names, obs, inflate=float(np.sqrt(len(f["vt"]) / max(f["n_events"], 1))))
            if idn["ok"]:
                method, apply = "accel", (dict(LMUY=ly, LMUX=ly) if link else dict(zip(names, (ly, lx))))
        if not apply and np.isfinite(s):
            method, apply = "lap", dict(LMUY=ctx.mf.LMUY * s, LMUX=ctx.mf.LMUX * s)
    elif stage == "drag":
        f = drag_fit(log, ctx)
        fit, idn = dict(Cd_scale=f["Cd_scale"]), dict(gamma=f["gamma"], ok=f["ok"])
        detail = {k: f[k] for k in ("f", "se_Cd_scale", "n", "ok")}
    else:
        f = lift_fit(log, ctx)
        g = max(f["gamma_front"], f["gamma_rear"])
        fit, idn, detail = dict(Cl_scale=f["mean"]), dict(gamma=g, ok=g <= GAMMA_MAX), f
    if stage != "grip" and idn["ok"]:                                   # the grip stage gates its accel fit above
        apply = dict(fit)
    ratio = None
    if apply:
        t_base = _lap(ctx)
        for k, val in apply.items():
            _set(ctx, k, val)                                           # ctx is local: no need to restore it
        ratio = _lap(ctx) / t_base
    vx = None if log is None else np.asarray(log["vx"], float)
    rec = dict(stage=stage, method=method, circuit=circuit, physics=ctx.physics, tyre_set=ctx.tyre_set,
               base_overrides=base, tags=tags or {}, log_sha=None if vx is None else array_sha(vx),
               n_samples=0 if vx is None else int(len(vx)), fit=fit, apply=apply, identifiable=idn, detail=detail,
               qss_ratio=ratio, wall_s=time.perf_counter() - t0)
    if ledger:
        ledger_add(ledger, rec)
    return rec


def nlp_forecast(data, fit, **kw):
    """Return the first-order change [s] of the NLP objective of a stored full result. No solve.
    fit maps each knob to its new value. The change is the sum over the knobs of
    dJ/dp (setup_gradients) times (new value - old value).
    The old value is the one in kw['vp_overrides'], or 1.0. kw are the options of setup_gradients."""
    from functions.sensitivity import setup_gradients
    g = setup_gradients(data, list(fit), **kw)
    old = kw.get("vp_overrides") or {}
    return float(sum(g[k] * (v - old.get(k, 1.0)) for k, v in fit.items()))
