"""vehModel_cs.py - the car model of a whole ChassisSim car (first target: GP3 2013).

The model has the 23 states and the channel names of vehModel. It has three inputs:
T_drive (total rear drive force [N]), T_brake (total brake force [N], <= 0) and delta.
The chassis equations come from vehModel._chassis_corrected. The car record ctx.cs
(functions.chassissim.load_car) gives the tyre, aero, suspension and powertrain data.
Every table becomes a C2 curve once, in numpy. The NLP sees only smooth expressions:
no if_else, fabs, fmax or fmin. ctx.physics is 'corrected' (userOpts checks it).

Sign rules are those of vehModel. xs > 0 is compression. A lift value is a downforce (> 0).
Toe > 0 is toe-out, as in vehModel. The camber of the record is the negative camber.
The axle values (tyre rate, track, radius) come from the car record. vp holds the means of the two axles,
except Rw and f, which are the rear values.
"""
import numpy as np
import casadi as ca
from types import SimpleNamespace
from scipy.interpolate import PchipInterpolator

from vehModel import _chassis_corrected
from functions.chassissim import suspension_curves, table_live

C4 = ("fl", "fr", "rl", "rr")
W_RUBBER = 1e-3     # m, width of the bump rubber onset (softplus)
FZ_FLOOR = 20.0     # N, scale of the tyre load floor (smooth maximum of the load and 0)
LAT_CAP = 0.9       # the lateral force of a wheel stays under this share of its pure lateral capacity
B_GEAR = 80.0       # sharpness of the smooth maximum over the gears
E_D, E_D2, P0_DET = 0.02, 0.01, 100.0   # diff: wheel speed width [m/s], map floor [m/s], Detroit gate force [N]
# ponytail: one spool stiffness [N s/m] for every car. It holds dv near 0.01 m/s at a 2 g corner (the geometric r t is 0.7 m/s). Make it a car field if a car needs another.
K_SPOOL = 5e4
# The sideslip table x is the lateral velocity over the speed. On the GP3 log the ChassisSim sideslip channel runs
# against the vy of this model (corr -0.968 along the fixed line, opposite signs at every point above 1.5 g), so the
# sign is -1. ponytail: assumes the tables use the logged sideslip; the binary did not show the frame sign.
SIDESLIP_SIGN = -1.0
S_MAX = 4.0         # largest normalised slip of the slip curve table


# ---------------------------------------------------------------- smooth tools
def _smax(a, b, e):
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + e * e))


def _smin(a, b, e):
    return 0.5 * (a + b - ca.sqrt((a - b) ** 2 + e * e))


def _clamp(v, lo, hi, e):
    """Keep v smoothly inside [lo + e, hi - e], so that it stays inside the grid [lo, hi] of a curve.
    The error is below e within e of an edge, and tiny farther in."""
    return _smin(_smax(v, lo + e, e), hi - e, e)


def _sp(z):
    """log(1 + exp(z)). The callers keep |z| below 700."""
    return ca.log1p(ca.exp(z))


def _clean(x, y):
    """Return the table sorted by x, without repeated abscissae."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    keep = np.r_[True, np.diff(x) > 0]
    return x[keep], y[keep]


def _curve(x, y, ext=0.0, n=200):
    """Return f(v): a C2 curve through (x, y). PCHIP samples feed a cubic B-spline.
    Inside the table the node step is at most half the smallest table spacing, with n nodes at least.
    Every table abscissa and the origin are nodes, so the knees of the table stay.
    ext > 0 extends both ends on the end slope by ext x range, on a coarse grid. Else the end values hold.
    v is clamped inside the grid, because a casadi B-spline returns 0 outside its grid."""
    x, y = _clean(x, y)                      # sorted, repeated abscissae dropped
    p = PchipInterpolator(x, y)
    lo, hi = x[0], x[-1]
    e = ext * (hi - lo)
    # Grid spacing h = 0.95 x the target step, so the gaps stay <= the step after the node filter below.
    # ponytail: at most 5000 nodes in the table range, so one close pair of points cannot blow up the grid
    h = max(0.95 * min((hi - lo) / (n - 1), 0.5 * np.min(np.diff(x))), (hi - lo) / 5000)
    step = h / 0.95
    xs = np.linspace(lo, hi, int(np.ceil((hi - lo) / h)) + 1)
    if e > 0:
        xs = np.r_[np.linspace(lo - e, lo, 20)[:-1], xs, np.linspace(hi, hi + e, 20)[1:]]
    nodes = np.unique(np.r_[x, [0.0] if lo - e < 0 < hi + e else []])
    i = np.clip(np.searchsorted(nodes, xs), 1, nodes.size - 1)
    near = np.minimum(np.abs(xs - nodes[i - 1]), np.abs(xs - nodes[i]))
    xs = np.union1d(xs[near > 0.05 * step], nodes)
    d = p.derivative()
    ys = np.where(xs < lo, y[0] + d(lo) * (xs - lo), np.where(xs > hi, y[-1] + d(hi) * (xs - hi), p(np.clip(xs, lo, hi))))
    s_lo, s_hi = float(d(lo)), float(d(hi))
    lo, hi = lo - e, hi + e
    f = ca.interpolant("curve", "bspline", [xs], ys)
    w = 1e-3 * (hi - lo)
    if ext > 0:              # beyond the grid the curve goes on along its end slopes (a stop keeps its stiffness)
        return lambda v: f(_clamp(v, lo, hi, w)) + s_hi * _smax(v - hi, 0, w) - s_lo * _smax(lo - v, 0, w)
    return lambda v: f(_clamp(v, lo, hi, w))


def _surf(tab, e=5e-4):
    """Return f(a, b) for a car map {x, z, data}: a on the x axis, b on the z axis. Both are clamped to the grid.
    e is the clamp width of both axes, or a pair (ex, ez) with one width per axis."""
    x, z = np.asarray(tab["x"], float), np.asarray(tab["z"], float)
    ex, ez = e if isinstance(e, tuple) else (e, e)
    f = ca.interpolant("surf", "bspline", [x, z], np.asarray(tab["data"], float).ravel(order="F"))
    return lambda a, b: f(ca.vertcat(_clamp(a, x[0], x[-1], ex), _clamp(b, z[0], z[-1], ez)))


def _lock_map(tab, kx=4, kz=20):
    """Return f(P, dv) for a diff map {x, z, data}: the force moved to the slower wheel [N].
    The force axis gets kx and the speed axis kz PCHIP steps per interval. The step of a map sits on the speed axis.
    The code refines the data one axis after the other. PCHIP keeps the data monotone, so a step gives no ringing.
    Both axes are clamped to the grid, as ChassisSim does. The value stays inside the range of the data."""
    x, z, d = (np.asarray(tab[a], float) for a in ("x", "z", "data"))
    fx = np.unique(np.concatenate([np.linspace(a, b, kx + 1) for a, b in zip(x[:-1], x[1:])]))
    fz = np.unique(np.concatenate([np.linspace(a, b, kz + 1) for a, b in zip(z[:-1], z[1:])]))
    d = PchipInterpolator(z, d, axis=1)(fz)
    d = PchipInterpolator(x, d, axis=0)(fx)
    f = _surf({"x": fx, "z": fz, "data": d}, (1.0, 1e-3))
    lo, hi = float(np.min(tab["data"])), float(np.max(tab["data"]))
    return lambda P, dv: _smin(_smax(f(P, dv), lo, 0.1), hi, 0.1)     # the B-spline may ring by a few N at a kink


def _slip_curve(T, cap=None):
    """Return h(v) = fn(v)/v for the normalised slip table of a tyre. fn is odd, so h is even and positive.
    The table gets two more points on a falling or flat last slope, then fn holds. cap clips fn (the lateral cap).
    Past S_MAX the force holds at fn x F', because c = min(v, S_MAX) and h = fn(c) / v x c. h(0) = fn'(0)."""
    x, y = np.asarray(T["slip_x"], float), np.asarray(T["slip_y"], float)
    if np.any(np.diff(x) < 0):
        raise ValueError("The slip table of the tyre must have increasing x.")
    keep = np.r_[True, np.diff(x) > 0]      # exact duplicates go
    x, y = x[keep], y[keep]
    if x.size < 2 or x[0] != 0 or y[0] != 0:
        raise ValueError("The slip table of the tyre must start at (0, 0) and have 2 or more points with increasing x.")
    sl, dx = min((y[-1] - y[-2]) / (x[-1] - x[-2]), 0.0), x[-1] - x[-2]
    x = np.r_[x, x[-1] + dx, x[-1] + 2 * dx]
    y = np.r_[y, np.maximum(y[-1] + sl * dx, 0.05), np.maximum(y[-1] + 2 * sl * dx, 0.05)]
    if cap is not None:
        y = np.minimum(y, cap)
    p = PchipInterpolator(np.r_[-x[:0:-1], x], np.r_[-y[:0:-1], y])
    sg = np.linspace(0.0, S_MAX, 201)
    fn = p(np.minimum(sg, x[-1]))
    h = np.empty_like(sg)
    h[0], h[1:] = float(p.derivative()(0.0)), fn[1:] / sg[1:]
    f = ca.interpolant("slip", "bspline", [sg], h)

    def hv(v):                  # v >= 1e-3 in the model. c stays inside (0, S_MAX), so inside the grid.
        c = _smin(v, S_MAX, 4e-3)
        return f(c) * c / v
    return hv


# ---------------------------------------------------------------- car numbers (numpy)
def drive_scale(cs, R):
    """Largest drive force of any gear [N]: the scale of T_drive."""
    e = cs.engine
    w = np.linspace(e.rpm_idle, e.rpm_limit, 400)
    te = PchipInterpolator(e.torque_curve["x"], e.torque_curve["y"])(w)
    return float(e.efficiency * te.max() * np.max(e.gears) / R)


def _mr(c):
    """Return (sm, mr) for the motion-ratio polynomial c (c1..c5): spring travel sm(x) and ratio mr(x) = sm'(x)."""
    c = [float(v) for v in np.atleast_1d(c)]
    return (lambda x: sum(v / (i + 1) * x ** (i + 1) for i, v in enumerate(c) if v != 0.0),
            lambda x: sum(v * x ** i for i, v in enumerate(c) if v != 0.0))


def _gear_envelope(cs, R, v):
    """Return (F_env, om_engine) for the wheel speed v [m/s] (SX or float).
    F_env is the smooth maximum over the gears of the drive force: efficiency x torque(engine speed) x ratio / R.
    Below idle the clutch holds the engine at idle. Above the limit a tanh window (2% of the limit) fades the torque.
    The log-sum-exp overestimates the true maximum. The error is at most F_ref ln(n) / B_GEAR, where n gears give
    nearly equal force: 0.9 % of F_ref for n = 2 at a gear crossing.
    om_engine is the engine speed of the gears, weighted by their share of F_env. It holds at idle under the clutch."""
    e = cs.engine
    te = _curve(e.torque_curve["x"], e.torque_curve["y"])
    F_ref = drive_scale(cs, R)
    dw = 0.01 * e.rpm_limit
    f, w_e = [], []
    for G in np.asarray(e.gears, float):
        w = v * G / R
        win = 0.5 * (1 + ca.tanh((e.rpm_limit - w) / dw))
        f.append(e.efficiency * te(_smax(w, e.rpm_idle, 5.0)) * G / R * win)
        w_e.append(w)
    F_env = F_ref / B_GEAR * ca.log(sum(ca.exp(B_GEAR * fi / F_ref) for fi in f))
    om = sum(ca.exp(B_GEAR * (fi - F_env) / F_ref) * _smax(wi, e.rpm_idle, 5.0) for fi, wi in zip(f, w_e))
    return F_env, om


def gear_envelope_numeric(cs, R, v):
    """Return F_env [N] at the wheel speeds v [m/s] (numpy array).
    It evaluates the function of the NLP, so the QSS and the NLP share one source."""
    x = ca.SX.sym("v")
    fn = ca.Function("F_env", [x], [_gear_envelope(cs, R, x)[0]]).map(int(np.size(v)))
    return np.asarray(fn(np.asarray(v, float).reshape(1, -1))).reshape(np.shape(v))


def rubber_gap(cs, c, vp=None):
    """Return the effective bump rubber gap of corner c at the damper [m], or None when the corner has no rubber.
    Flag 1: the stored gap. Flag 0: the stored gap minus the static spring compression at the damper,
    F0 / (mr0 k) - preload, with F0 from vp (needed for flag 0 only).
    The result is negative when the rubber is loaded at static. A stored gap of 0 means no rubber."""
    gap = float(getattr(cs.susp, f"gap_{c}"))
    if gap == 0.0:
        return None
    if getattr(cs.susp, f"gap_flag_{c}") == 0:
        if vp is None:
            raise ValueError(f"The rubber gap of corner {c} needs the vehicle parameters (gap flag 0).")
        mr0 = float(np.atleast_1d(getattr(cs.susp, f"mr_{c}"))[0])
        sp = getattr(cs.susp, f"spring_{c}")
        if not sp["rate"]:
            raise ValueError(f"The spring of corner {c} has no rate. The static deflection for the rubber gap needs one.")
        gap -= getattr(vp, f"m_c_{c}") * vp.g / (mr0 * sp["rate"]) - sp["preload"]
    return gap


def _rubber(cs, c, vp=None):
    """Return F(xs): the bump rubber force on the wheel [N] at the strut travel xs [m, SX].
    The rubber and its gap sit at the damper. The compression is sm(xs) - gap, with sm the spring travel of the
    motion-ratio law (gap from rubber_gap), and the wheel force is mr(xs) x the rubber force. The onset is smooth
    (softplus, width W_RUBBER). A table (flag 1: compression in m, force in N) gives the law. Else the rate does.
    A corner without rubber gives 0."""
    gap = rubber_gap(cs, c, vp)
    if gap is None:
        return lambda xs: 0.0
    rub = getattr(cs.susp, f"rubber_{c}")
    sm, mr = _mr(getattr(cs.susp, f"mr_{c}"))
    law = (_curve(np.r_[0.0, rub["x"]], np.r_[0.0, rub["y"]], ext=1.0) if rub["flag"] == 1
           else (lambda comp: rub["rate"] * comp))
    return lambda xs: mr(xs) * law(W_RUBBER * _sp((sm(xs) - gap) / W_RUBBER))


def poly_sx(p, x, lo, hi):
    """Evaluate the poly1d p at x (SX), with x kept smoothly inside the fitted range [lo, hi] (Horner)."""
    x = _clamp(x, lo, hi, 2e-3)
    y = 0.0
    for a in p.coeffs:
        y = y * x + float(a)
    return y


# ---------------------------------------------------------------- the model
def vehModel_cs(ctx):
    vp, cs = ctx.vp, ctx.cs
    SX = ca.SX
    OPT_e = ctx.OPT_e
    Steering = getattr(ctx, "Steering", "NA")
    grip = getattr(ctx, "grip", 1.0)
    ty = {c: getattr(cs.tyre, c) for c in C4}
    Rw = {"f": ty["fl"]["radius"], "r": ty["rl"]["radius"]}
    t_f, t_r = cs.geom.track_front, cs.geom.track_rear
    kt = {c: ty[c]["kz"] for c in C4}           # tyre rate per corner
    kt_ref = vp.kt                              # the zt states hold the tyre load / kt_ref
    deg2rad = np.pi / 180.0
    curves = suspension_curves(cs)
    eng = cs.engine
    vmax = eng.rpm_limit * Rw["r"] / float(np.min(eng.gears))

    # ===================== states (23, order and names of vehModel) ==========
    def state(name, scale):
        s = SX.sym(name)
        return s, scale, scale * s

    names = ["vx", "vy", "yawrate", "n", "eps", "Om_fl", "Om_fr", "Om_rl", "Om_rr", "zs", "zsdot", "theta", "thetadot",
             "phi", "phidot", "wu_fl", "wu_fr", "wu_rl", "wu_rr", "zt_fl", "zt_fr", "zt_rl", "zt_rr"]
    scales = [100.0, 10.0, 1.0, 5.0, 1.0, 100.0 / Rw["f"], 100.0 / Rw["f"], 100.0 / Rw["r"], 100.0 / Rw["r"]] + [1.0] * 14
    sym = [state(f"{n}_n", s) for n, s in zip(names, scales)]
    (vx_n, _, vx), (vy_n, _, vy), (r_n, _, r), (n_n, _, n), (eps_n, _, eps) = sym[:5]
    (Om_fl, Om_fr, Om_rl, Om_rr) = (q[2] for q in sym[5:9])
    (zs, zsdot, theta, thetadot, phi, phidot) = (q[2] for q in sym[9:15])
    (wu_fl, wu_fr, wu_rl, wu_rr) = (q[2] for q in sym[15:19])
    (zt_fl, zt_fr, zt_rl, zt_rr) = (q[2] for q in sym[19:23])
    x_s = np.array(scales)
    x_lim = np.array([
        [OPT_e, vmax], [-10, 10], [-np.pi/2, np.pi/2], [-4, 4], [-np.pi/4, np.pi/4],
        [OPT_e/Rw["f"], vmax/Rw["f"]], [OPT_e/Rw["f"], vmax/Rw["f"]],
        [OPT_e/Rw["r"], vmax/Rw["r"]], [OPT_e/Rw["r"], vmax/Rw["r"]],
        [-0.1, 0.1], [-10, 10], [-0.1, 0.1], [-10, 10], [-0.09, 0.09], [-10, 10],
        [-1, 1], [-1, 1], [-1, 1], [-1, 1],
        [0, 0.1], [0, 0.1], [0, 0.1], [0, 0.1]], dtype=float)
    x_lim = x_lim * (1.0 / x_s[:, None])
    x_min, x_max = x_lim[:, 0], x_lim[:, 1]
    x = ca.vertcat(*[q[0] for q in sym])
    nx = x.shape[0]
    zt = dict(zip(C4, (zt_fl, zt_fr, zt_rl, zt_rr)))
    wu = dict(zip(C4, (wu_fl, wu_fr, wu_rl, wu_rr)))

    # ===================== inputs ===========================================
    F_ref = drive_scale(cs, Rw["r"])
    F_brk = cs.brake.force
    T_drive_n, T_brake_n, delta_n = SX.sym("T_drive_n"), SX.sym("T_brake_n"), SX.sym("delta_n")
    T_drive, T_brake, delta = F_ref * T_drive_n, F_brk * T_brake_n, 35.0 * deg2rad * delta_n
    u = ca.vertcat(T_drive_n, T_brake_n, delta_n)
    u_s = np.array([F_ref, F_brk, np.pi / 8])
    u_lim = np.array([[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]])
    u_min, u_max = u_lim[:, 0], u_lim[:, 1]
    nu = u.shape[0]
    duk_ub = np.asarray(ctx.duk_ub, float).reshape(-1) / u_s
    duk_lb = np.asarray(ctx.duk_lb, float).reshape(-1) / u_s
    assert nu == duk_ub.size == duk_lb.size, "Rate limits (ctx.input_keys) do not match the inputs"

    kappa = SX.sym("kappa")
    drs = getattr(ctx.track, "drs", None) is not None       # userOpts builds the gate when the car has a DRS
    gate = SX.sym("drs_gate") if drs else None
    pv = ca.vertcat(kappa, gate) if drs else ca.vertcat(kappa)

    # ===================== suspension =======================================
    # xs = 0 is the static state: the spring force then equals the static sprung corner load.
    d_tyre = {c: zt[c] * kt_ref / kt[c] for c in C4}                  # real tyre deflection
    d0 = {c: getattr(vp, f"W{c}0") / kt[c] for c in C4}
    xs = {"fl": -(zs - vp.l_f * theta + (t_f / 2) * phi) - (d_tyre["fl"] - d0["fl"]),
          "fr": -(zs - vp.l_f * theta - (t_f / 2) * phi) - (d_tyre["fr"] - d0["fr"]),
          "rl": -(zs + vp.l_r * theta + (t_r / 2) * phi) - (d_tyre["rl"] - d0["rl"]),
          "rr": -(zs + vp.l_r * theta - (t_r / 2) * phi) - (d_tyre["rr"] - d0["rr"])}
    xsdot = {"fl": wu_fl - (-vp.l_f * thetadot + (t_f / 2) * phidot + zsdot),
             "fr": wu_fr - (-vp.l_f * thetadot - (t_f / 2) * phidot + zsdot),
             "rl": wu_rl - (+vp.l_r * thetadot + (t_r / 2) * phidot + zsdot),
             "rr": wu_rr - (+vp.l_r * thetadot - (t_r / 2) * phidot + zsdot)}
    dampers = {}
    fzs = {}
    for c in C4:
        sp = getattr(cs.susp, f"spring_{c}")
        sm, mr = _mr(getattr(cs.susp, f"mr_{c}"))
        mr0 = float(np.atleast_1d(getattr(cs.susp, f"mr_{c}"))[0])
        F0 = getattr(vp, f"m_c_{c}") * vp.g             # static sprung corner load
        spring = (F0 / mr0 + sp["rate"] * sm(xs[c])) * mr(xs[c])
        rubber = _rubber(cs, c, vp)(xs[c])
        dm = getattr(cs.susp, f"damper_{c}")
        v, bump, reb = np.asarray(dm["velocity"], float), np.asarray(dm["bump"], float), np.asarray(dm["rebound"], float)
        if v.size < 2 or not (np.any(bump) or np.any(reb)):
            raise ValueError(f"The damper of corner {c} has no table. The car model needs one.")
        if v[0] != 0 or np.any(np.diff(v) <= 0):
            raise ValueError(f"The damper table of corner {c} must start at velocity 0 and increase.")
        key = (v.tobytes(), bump.tobytes(), reb.tobytes())
        if key not in dampers:
            dampers[key] = _curve(np.r_[-v[:0:-1], v], np.r_[-reb[:0:-1], bump], ext=2.0)
        fd = mr(xs[c]) * dampers[key](mr(xs[c]) * xsdot[c])
        fzs[c] = spring + fd + rubber
    # Anti-roll bars. A ChassisSim bar rate acts like a wheel rate in roll (axle roll stiffness k t^2 / 2), so each
    # wheel gets k/2 (xs_l - xs_r). The GP3 roll in the ChassisSim log (front damper difference 1.9 mm at 2.5 g)
    # fits this form. The C3 form k (xs_l - xs_r) of the 4EM car gives half that roll.
    for ax_, (a, b) in (("f", ("fl", "fr")), ("r", ("rl", "rr"))):
        bar = 0.5 * getattr(cs.susp, f"arb_{ax_}")["rate"] * float(np.atleast_1d(getattr(cs.susp, f"mr_arb_{ax_}"))[0]) ** 2
        if bar:
            fzs[a], fzs[b] = fzs[a] + bar * (xs[a] - xs[b]), fzs[b] - bar * (xs[a] - xs[b])
    fz = {c: zt[c] * kt_ref for c in C4}

    # ===================== aero =============================================
    rh_f = 0.5 * (cs.geom.ride_height_fl + cs.geom.ride_height_fr) + zs - vp.l_f * theta
    rh_r = 0.5 * (cs.geom.ride_height_rl + cs.geom.ride_height_rr) + zs + vp.l_r * theta
    rho = cs.aero.rho if cs.aero.rho else 1.225
    q = 0.5 * rho * vx ** 2
    ae = cs.aero
    cla, cda = _surf(ae.cla_map)(rh_r, rh_f), _surf(ae.cda_map)(rh_r, rh_f)
    AB = _surf(ae.balance_map)(rh_r, rh_f) + (ae.offset or 0.0)
    tab = lambda t, x: _curve(t["x"], t["y"])(x)
    if any(table_live(t) for t in (ae.ss_cla, ae.ss_cda, ae.ss_ab)):    # sideslip: sin of the angle at the CG
        sb = SIDESLIP_SIGN * vy / ca.sqrt(vx ** 2 + vy ** 2)
        fc = lambda t: _clamp(tab(t, sb), -0.4, 0.4, 1e-3) if table_live(t) else 0.0
        cla, cda, AB = cla * (1 + fc(ae.ss_cla)), cda * (1 + fc(ae.ss_cda)), AB + fc(ae.ss_ab)
    if drs:                         # a DRS zone: the deltas of the car file, weighted by the gate
        cla_n = cla - gate * (ae.drs_337 + ae.drs_338)
        # ponytail: smooth floor of 1e-3 m^2 on the divisor, so a delta equal to the whole CLA cannot divide by zero
        AB, cla, cda = (cla * AB - gate * ae.drs_337) / _smax(cla_n, 1e-3, 1e-6), cla_n, cda - gate * ae.drs_339
    # speed correction: a ratio of CLA and CDA, an offset of AB, each when its enable is 1
    if ae.sp_on_cla == 1:
        cla = cla * tab(ae.sp_cla, vx)
    if ae.sp_on_cda == 1:
        cda = cda * tab(ae.sp_cda, vx)
    if ae.sp_on_ab == 1:
        AB = AB + tab(ae.sp_ab, vx)
    L, D = q * cla, q * cda
    f_lift_fl = f_lift_fr = 0.5 * AB * L
    f_lift_rl = f_lift_rr = 0.5 * (1 - AB) * L
    f_lift, f_drag = L, D

    # ===================== tyre slip (the geometry of vehModel) =============
    # toe > 0 = toe-out, as in vehModel (a left wheel adds it to its slip angle, a right wheel subtracts it)
    if Steering == "NA":
        sa_fl = delta - ca.atan((vp.l_f*r + vy) / (vx - r*t_f/2)) + ty["fl"]["toe"]
        sa_fr = delta - ca.atan((vp.l_f*r + vy) / (vx + r*t_f/2)) - ty["fr"]["toe"]
    elif Steering == "Ackermann":
        delta_fl = ca.atan(vp.l_f / (vp.l_f / ca.tan(delta) - t_f/2))
        delta_fr = ca.atan(vp.l_f / (vp.l_f / ca.tan(delta) + t_f/2))
        sa_fl = delta_fl - ca.atan((vp.l_f*r + vy) / (vx - r*t_f/2)) + ty["fl"]["toe"]
        sa_fr = delta_fr - ca.atan((vp.l_f*r + vy) / (vx + r*t_f/2)) - ty["fr"]["toe"]
    else:
        raise ValueError("Steering must be 'NA' or 'Ackermann'")
    sa_rl = ca.atan((vp.l_r*r - vy) / (vx - r*t_r/2)) + ty["rl"]["toe"]
    sa_rr = ca.atan((vp.l_r*r - vy) / (vx + r*t_r/2)) - ty["rr"]["toe"]
    v_fl = ca.sqrt((vy + vp.l_f*r)**2 + (vx - r*t_f/2)**2)
    v_fr = ca.sqrt((vy + vp.l_f*r)**2 + (vx + r*t_f/2)**2)
    v_rl = ca.sqrt((vy - vp.l_r*r)**2 + (vx - r*t_r/2)**2)
    v_rr = ca.sqrt((vy - vp.l_r*r)**2 + (vx + r*t_r/2)**2)
    sa = {"fl": sa_fl, "fr": sa_fr, "rl": sa_rl, "rr": sa_rr}
    vxw = {"fl": v_fl*ca.cos(sa_fl), "fr": v_fr*ca.cos(sa_fr), "rl": v_rl*ca.cos(sa_rl), "rr": v_rr*ca.cos(sa_rr)}
    Om = {"fl": Om_fl, "fr": Om_fr, "rl": Om_rl, "rr": Om_rr}
    sx = {c: (Rw[c[0]] * Om[c] - vxw[c]) / vxw[c] for c in C4}

    # ===================== tyre forces ======================================
    fx, fy, mu_x, mu_y, dcam, Cx = {}, {}, {}, {}, {}, {}
    for c in C4:
        T = ty[c]
        fze = _smax(fz[c], 0.0, 2 * FZ_FLOOR)                                   # load floor
        Fp = _curve(T["load"], T["fy_peak"])(fze) * (grip * T["grip"])
        kc = curves.f if c[0] == "f" else curves.r      # dynamic camber: static + the curve at the wheel travel
        # Camber against the ground = static + the curve at the wheel travel (against the body) + the body roll.
        # phi > 0 is the right side down (xs_fr > xs_fl). A left wheel then leans its top inward (camber +phi),
        # a right wheel outward (-phi). So the outer wheel of a turn loses negative camber and the inner one gains it.
        dyn = T["camber"] + poly_sx(kc.camber, xs[c], kc.lo, kc.hi) + (phi if c[1] == "l" else -phi)
        cam = _curve(T["camber_x"], T["camber_y"])(dyn)
        dcam[c] = dyn
        t4 = _surf({"x": T["fx_camber_x"], "z": T["fx_camber_load"], "data": T["fx_camber"]}, (5e-4, 5.0))(dyn, fze)   # Fx peak / Fy peak
        sa_n, ss_n = sa[c] / T["peak_slip_angle"], sx[c] / T["peak_slip_ratio"]
        sn = ca.sqrt(sa_n ** 2 + ss_n ** 2 + 1e-6)
        fy[c] = Fp * cam * _slip_curve(T, LAT_CAP)(sn) * sa_n         # the lateral force stays under 0.9 of its capacity
        fx[c] = Fp * cam * t4 * _slip_curve(T)(sn) * ss_n
        mu_x[c], mu_y[c] = Fp * cam * t4 / fze, Fp * cam / fze
        Cx[c] = Fp * cam * t4                                       # the Fx peak of the corner [N]
    f_roll_c = {c: ty[c]["rolling_drag"] * fz[c] for c in C4}
    f_roll = sum(f_roll_c.values())

    fxg = {"fl": fx["fl"]*ca.cos(delta) - fy["fl"]*ca.sin(delta), "fr": fx["fr"]*ca.cos(delta) - fy["fr"]*ca.sin(delta),
           "rl": fx["rl"], "rr": fx["rr"]}
    fyg = {"fl": fy["fl"]*ca.cos(delta) + fx["fl"]*ca.sin(delta), "fr": fy["fr"]*ca.cos(delta) + fx["fr"]*ca.sin(delta),
           "rl": fy["rl"], "rr": fy["rr"]}
    # roll centres: the curve of the axle at the mean wheel travel (rc_h(0) = the stored height minus the ground datum zg: the height above the contact patches). Jacking and Mx as vehModel.
    hRC = {a: poly_sx(k.rc_h, 0.5 * (xs[a + "l"] + xs[a + "r"]), k.lo, k.hi) for a, k in (("f", curves.f), ("r", curves.r))}
    fj = {"fl": -fyg["fl"] * (2*hRC["f"]/t_f), "fr": +fyg["fr"] * (2*hRC["f"]/t_f),
          "rl": -fyg["rl"] * (2*hRC["r"]/t_r), "rr": +fyg["rr"] * (2*hRC["r"]/t_r)}
    Mx = {c: fyg[c] * (vp.hcg - hRC[c[0]]) for c in C4}
    My = {c: -(fxg[c] * (vp.hcg - (d_tyre[c] - d0[c]) - xs[c])) for c in C4}

    # ===================== powertrain =======================================
    bias = cs.brake.front_bias
    # Rear differential. P = T_drive is the drive force at the wheels. The faster wheel loses T_x to the slower one.
    E_DET = F_ref / (2 * K_SPOOL)       # ponytail: Detroit tanh width, so its slope at dv = 0 never exceeds the spool K at full drive
    P, dv, dt = T_drive, Rw["r"] * (Om_rl - Om_rr), cs.diff.rear_type
    if dt == 1:
        T_rl, T_rr = T_drive * Rw["r"] / 2, T_drive * Rw["r"] / 2                               # open (the graph of the car before the diff types)
    elif dt == 2:                                                           # super: the split follows the Fx peaks
        T_rl, T_rr = (P * Cx[c] / (Cx["rl"] + Cx["rr"]) * Rw["r"] for c in ("rl", "rr"))
    else:
        if dt == 3:
            T_x = K_SPOOL * dv                                              # spool
        elif dt == 4:
            if cs.diff.rear_map is None:
                raise ValueError("The car has a custom rear differential but no differential map.")
            T_x = ca.tanh(dv / E_D) * _lock_map(cs.diff.rear_map)(P, ca.sqrt(dv ** 2 + E_D2 ** 2))
        elif dt == 5:                                                       # Detroit: locked under drive
            T_x = 0.5 * P * 0.5 * (1 + ca.tanh(P / P0_DET)) * ca.tanh(dv / E_DET)
        else:
            raise ValueError(f"The rear differential type {dt} is not known. Use 1 to 5.")
        T_rl, T_rr = (P / 2 - T_x) * Rw["r"], (P / 2 + T_x) * Rw["r"]
    T_w = {"fl": T_brake * bias / 2 * Rw["f"], "fr": T_brake * bias / 2 * Rw["f"],
           "rl": T_rl + T_brake * (1 - bias) / 2 * Rw["r"],
           "rr": T_rr + T_brake * (1 - bias) / 2 * Rw["r"]}
    v_r = 0.5 * (Om_rl + Om_rr) * Rw["r"]
    F_env, om_eng = _gear_envelope(cs, Rw["r"], v_r)

    # ===================== chassis ==========================================
    cc = lambda d: tuple(d[c] for c in C4)
    sf, d = _chassis_corrected(
        vp, kappa, (vx, vy, r, n, eps), (zsdot, thetadot, phidot), cc(wu), cc(zt), cc(fxg), cc(fyg), cc(fx), cc(T_w),
        cc(fzs), cc(fj), cc(Mx), cc(My), (f_lift_fl, f_lift_fr, f_lift_rl, f_lift_rr), (0.0, 0.0, 0.0, 0.0),
        f_drag, f_roll, f_lift, 0.0, 0.0, t_axle=(t_f, t_r))
    d = list(d)
    for i, c in enumerate(C4):      # zt holds F/kt_ref, so its rate is the deflection rate x kt/kt_ref
        d[19 + i] = -wu[c] * sf * kt[c] / kt_ref
    Lon_acc, Lat_acc = d[0] / sf, d[1] / sf
    ax, ay = Lon_acc - vy*r, Lat_acc + vx*r
    dx = ca.vertcat(*d) / ca.DM(x_s)

    # ===================== expose on ctx.m23 ================================
    m = SimpleNamespace()
    m.nx, m.nu, m.ny = nx, nu, 0
    m.x, m.u, m.pv, m.kappa = x, u, pv, kappa
    m.dx, m.sf = dx, sf
    m.x_s, m.u_s = x_s, u_s
    m.x_min, m.x_max, m.u_min, m.u_max = x_min, x_max, u_min, u_max
    m.duk_lb, m.duk_ub = duk_lb, duk_ub
    m.TyreModel = "CombinedSlip"
    m.vx, m.vy, m.r, m.n, m.eps = vx, vy, r, n, eps
    m.zs, m.theta, m.phi = zs, theta, phi
    m.delta, m.T_brake, m.T_brake_n, m.T_drive, m.T_drive_n = delta, T_brake, T_brake_n, T_drive, T_drive_n
    for c in C4:
        for nm, dd in (("sa", sa), ("sx", sx), ("fx", fx), ("fy", fy), ("fz", fz), ("xs", xs), ("xsdot", xsdot),
                       ("fzs", fzs), ("T", T_w), ("Om", Om)):
            setattr(m, f"{nm}_{c}", dd[c])
        setattr(m, f"mu_{c}_x", mu_x[c])
        setattr(m, f"mu_{c}_y", mu_y[c])
        setattr(m, f"f_lift_{c}", {"fl": f_lift_fl, "fr": f_lift_fr, "rl": f_lift_rl, "rr": f_lift_rr}[c])
        setattr(m, f"dynamic_camber_{c}", dcam[c] * (-1.0 if c[1] == "r" else 1.0))
    m.f_roll, m.f_drag, m.f_lift = f_roll, f_drag, f_lift
    m.Lon_acc, m.Lat_acc, m.ax, m.ay = Lon_acc, Lat_acc, ax, ay
    m.F_env, m.om_engine, m.v_r, m.rh_f, m.rh_r = F_env, om_eng, v_r, rh_f, rh_r
    m.veh_extra = [("om_gear", om_eng), ("throttle", T_drive / F_env), ("F_env", F_env)]
    ctx.m23 = m
    return ctx


def cs_path_constraints(ca, m, ctx):
    """Return (hnames, h, h_lb, h_ub) of the car. drive_env: the drive force stays under the gear envelope.
    BrTh_1: no drive and brake together (as vehModel)."""
    hnames = ["drive_env", "BrTh_1"]
    h = ca.vertcat((m.F_env - m.T_drive) / m.u_s[0], (m.T_drive_n * m.T_brake_n) / 1e-3)
    return hnames, h, np.array([0.0, -1.0]), np.array([np.inf, 1.0])


def cs_seed_inputs(ctx, m, prof, s):
    """Scaled seed of the three inputs at the stations s [m] from the QSS profile: the tyre force that holds the
    profile acceleration (mass and resistance of the car envelope prof['env']) goes to T_drive or T_brake. delta is 0."""
    env = prof["env"]
    s = np.asarray(s, float).reshape(-1)
    v, ax = np.interp(s, prof["s"], prof["v"]), np.interp(s, prof["s"], prof["ax"])
    F = env.M * ax + env.F_res(v)
    u = np.vstack([np.maximum(F, 0.0), np.minimum(F, 0.0), np.zeros_like(v)]) / np.asarray(m.u_s, float)[:, None]
    return np.clip(u, np.asarray(m.u_min)[:, None], np.asarray(m.u_max)[:, None])
