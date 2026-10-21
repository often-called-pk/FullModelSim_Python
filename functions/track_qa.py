"""Track QA and curvature from x, y (numpy + scipy, no casadi).

check_track() tests a loaded track (s, k and optional x, y) and never changes it.
userOpts._load_track runs it on every real circuit. k_from_xy() builds a valid k
from a centreline polyline: use it to rebuild a track file that fails the QA.
write_track() builds a track file from a polyline and checks it. xy_from_gps() converts GPS fixes to a polyline.
"""
import numpy as np
import scipy.io as sio
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from types import SimpleNamespace

TWO_PI = 2.0 * np.pi
TURN_WARN, TURN_ERROR = 0.005, 0.25   # laps: |turning of k - turning of x,y|
SIGN_WARN, SIGN_ERROR = 0.98, 0.90    # share of curved points where sign(k) == sign(k of x,y)
K_CURVED = 2e-3                       # 1/m: a point is curved above this |k| of x,y (R < 500 m)
GAP_WARN = 0.005                      # closed lap: end gap of the path integrated from k, share of L
DS_WARN = 25.0                        # m: largest step of s
K_WARN = 0.2                          # 1/m: |k| above this is R < 5 m
WGS84_A, WGS84_E2 = 6378137.0, 0.00669437999014   # semi-major axis [m], first eccentricity squared


def _cumtrapz(f, s):
    return np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(s))))


def check_track(track):
    """Check a track namespace (s, k, optional x, y). Never changes it.

    Returns (metrics, issues). issues is a list of (level, text), level 'error' (the
    track is not usable) or 'warn'. An input error returns at once, with the metrics
    found so far. Without x, y only the s step and |k| checks run.
    """
    s = np.asarray(track.s, float).ravel()
    k = np.asarray(track.k, float).ravel()
    if s.size != k.size or s.size < 3:
        return {}, [("error", f"s and k need the same length >= 3 (got {s.size}, {k.size})")]
    if not (np.isfinite(s).all() and np.isfinite(k).all()):
        return {}, [("error", "s or k holds NaN or inf")]
    ds = np.diff(s)
    if not (ds > 0).all():
        return {}, [("error", f"s is not strictly increasing ({int((ds <= 0).sum())} steps <= 0)")]
    L = float(s[-1] - s[0])
    psi = _cumtrapz(k, s)
    met = dict(n=s.size, L=L, ds_min=float(ds.min()), ds_max=float(ds.max()),
               turn_k=float(psi[-1] / TWO_PI), k_max=float(np.abs(k).max()))
    issues = []
    x, y = getattr(track, "x", None), getattr(track, "y", None)
    # ponytail: no turning reference for an s,k-only file, add one when such a file is registered
    if x is not None and y is not None:
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        if x.size != s.size or y.size != s.size:
            return met, [("error", f"x, y length {x.size}, {y.size} != s length {s.size}")]
        th = np.unwrap(np.arctan2(np.gradient(y, s), np.gradient(x, s)))
        gap_xy = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        closed = bool(gap_xy <= 3.0 * float(np.median(ds)))
        turn_xy = float((th[-1] - th[0]) / TWO_PI)
        ref = round(turn_xy) if closed else turn_xy
        kxy = np.gradient(th, s)
        cur = np.abs(kxy) > K_CURVED
        agree = float(np.mean(np.sign(k[cur]) == np.sign(kxy[cur]))) if cur.any() else 1.0
        err = met["turn_k"] - ref
        met.update(closed=closed, gap_xy=gap_xy, turn_xy=turn_xy, sign_agree=agree, turn_err=float(err))
        if closed:
            met["gap_k"] = float(np.hypot(_cumtrapz(np.cos(psi), s)[-1], _cumtrapz(np.sin(psi), s)[-1]))
        if agree < SIGN_WARN:
            issues.append(("error" if agree < SIGN_ERROR else "warn",
                           f"sign of k agrees with the x,y heading at {agree:.1%} of the curved points"))
        if abs(err) > TURN_WARN:
            issues.append(("error" if abs(err) > TURN_ERROR else "warn",
                           f"k turns {met['turn_k']:+.4f} laps, x,y {ref:+.4f} ({err * 360:+.1f} deg)"))
        if closed and met["gap_k"] > GAP_WARN * L:
            issues.append(("warn", f"the path integrated from k ends {met['gap_k']:.1f} m from its start "
                                   f"({met['gap_k'] / L:.2%} of the lap)"))
    if met["ds_max"] > DS_WARN:
        issues.append(("warn", f"largest s step {met['ds_max']:.1f} m at s = {s[int(np.argmax(ds))]:.1f} m"))
    if met["k_max"] > K_WARN:
        i = int(np.argmax(np.abs(k)))
        issues.append(("warn", f"|k| {abs(k[i]):.3f} 1/m (R {1 / abs(k[i]):.1f} m) at s = {s[i]:.1f} m"))
    return met, issues


def k_from_xy(x, y, closed=True, ds=1.0, window=5, passes=2, max_chord=30.0):
    """Curvature from a centreline polyline, heading-conserving. Returns (s, x, y, k).

    Chords longer than max_chord are split first, so a sparse straight stays straight.
    A chord-length cubic spline (periodic if closed) gives the tangent heading on a
    uniform ds grid. k is its central difference (circular if closed), smoothed by
    `passes` centred moving averages of `window` samples. Closed: the last point
    equals the first, and trapz(k, s) is 2 pi times the winding number to rounding.
    """
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if closed:
        if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-9: x, y = x[:-1], y[:-1]
        x, y = np.append(x, x[0]), np.append(y, y[0])
    seg = np.hypot(np.diff(x), np.diff(y)); c = np.concatenate(([0.0], np.cumsum(seg)))
    n_sub = np.maximum(np.ceil(seg / max_chord).astype(int), 1)          # split long chords only
    q = np.concatenate([c[i] + seg[i] * np.arange(n_sub[i]) / n_sub[i] for i in range(seg.size)] + [c[-1:]])
    x, y = np.interp(q, c, x), np.interp(q, c, y)
    bc = "periodic" if closed else "not-a-knot"
    cx, cy = CubicSpline(q, x, bc_type=bc), CubicSpline(q, y, bc_type=bc)
    s = np.linspace(0.0, q[-1], max(int(round(q[-1] / ds)), 3) + 1); h = s[1] - s[0]
    if closed:
        th = np.arctan2(cy(s[:-1], 1), cx(s[:-1], 1))
        k = np.angle(np.exp(1j * (np.roll(th, -1) - np.roll(th, 1)))) / (2 * h)   # circular, telescopes
    else:
        k = np.gradient(np.unwrap(np.arctan2(cy(s, 1), cx(s, 1))), h)
    for _ in range(passes):                                               # centred MA
        k = np.convolve(np.pad(k, (window // 2, window - 1 - window // 2), mode="wrap" if closed else "edge"),
                        np.ones(window) / window, mode="valid")
    if closed: k = np.append(k, k[0])                                     # last point == first point
    return s, cx(s), cy(s), k


def xy_from_gps(lat, lon):
    """East, north [m] of WGS84 lat, lon [deg] on the tangent plane at the mean position.

    East is +x and north is +y. A left turn then runs counter-clockwise, so k > 0 as in the model.
    """
    # ponytail: tangent plane, relative error 1e-4 at 0.8 km from the mean, use a real projection beyond about 5 km
    lat, lon = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))
    p0 = lat.mean()
    w = np.sqrt(1.0 - WGS84_E2 * np.sin(p0) ** 2)
    n = WGS84_A / w                                   # prime-vertical radius of curvature
    return n * np.cos(p0) * (lon - lon.mean()), n * (1.0 - WGS84_E2) / w ** 2 * (lat - p0)


def smooth_xy(x, y, length, closed=True, ds=1.0):
    """Smooth a polyline: resample it to ds [m], then apply a cubic Savitzky-Golay filter over `length` [m].

    Use it for GPS noise above about 0.05 m. A closed result has no repeated end point.
    """
    if closed:
        x, y = np.append(x, x[0]), np.append(y, y[0])
    c = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    u = np.linspace(0.0, c[-1], max(int(round(c[-1] / ds)), 3) + 1)[: -1 if closed else None]
    w, mode = 2 * int(length / ds / 2) + 1, "wrap" if closed else "interp"
    return tuple(savgol_filter(np.interp(u, c, a), w, 3, mode=mode) for a in (x, y))


def write_track(path, x, y, closed=True, smooth=0.0, **kw):
    """Build s, k, x, y from a polyline [m], run the track QA and write the .mat file. Return (metrics, issues).

    kw goes to k_from_xy (ds, window, passes, max_chord). window and passes set the smoothing of k.
    smooth > 0 first smooths the polyline over that length [m] (smooth_xy).
    Raise ValueError, and write nothing, in these cases:
    - The QA finds an error.
    - A closed lap has a gap between its start and its end.
    - A closed lap does not turn +-1 laps.
    To use the file as a circuit, add it to userOpts._REAL_CIRCUITS and app/mainwindow.CIRCUITS by hand.
    """
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    keep = np.r_[True, np.hypot(np.diff(x), np.diff(y)) > 1e-3]       # a logger repeats each GPS fix between updates
    x, y = x[keep], y[keep]
    if closed:
        # a lap that overshoots its start has a closing chord that points backward: trim the tail until it points forward
        # ponytail: the first two chords set the start heading, untested for a lap that starts inside a hairpin
        while x.size > 3 and (x[0] - x[-1]) * (x[2] - x[0]) + (y[0] - y[-1]) * (y[2] - y[0]) <= 0:
            x, y = x[:-1], y[:-1]
        if x.size <= 3:                       # no later fix lies behind the start. 3 fixes always pass the gap test
            raise ValueError("The line does not come back to its start, so this is not a closed lap. "
                             "Pass closed=False for an open track.")
        gap = np.hypot(x[-1] - x[0], y[-1] - y[0])
        if gap > 3.0 * np.hypot(np.diff(x), np.diff(y)).max():
            raise ValueError(f"The start and end are {gap:.1f} m apart, so this is not a closed lap. "
                             "Pass closed=False for an open track.")
    if smooth:
        x, y = smooth_xy(x, y, smooth, closed, kw.get("ds", 1.0))
    s, x, y, k = k_from_xy(x, y, closed, **kw)
    met, issues = check_track(SimpleNamespace(s=s, k=k, x=x, y=y))
    errs = [t for lvl, t in issues if lvl == "error"]
    if closed and abs(abs(met["turn_k"]) - 1.0) > 1e-6:
        errs.append(f"the line turns {met['turn_k']:+.6f} laps, not +-1, so it crosses itself or holds more than one lap")
    if errs:
        raise ValueError("The track fails the track QA:\n- " + "\n- ".join(errs))
    sio.savemat(path, {"s": s, "k": k, "x": x, "y": y}, do_compression=True)
    return met, issues
