"""mesh.py - curvature-weighted non-uniform collocation mesh (numpy only).

Roadmap Item 3 (single pass, no re-solve). The N+1 collocation knots are placed
by EQUIDISTRIBUTING a monitor function of the track curvature,

    M(s) = 1 + a * |k(s)| / k_ref + b * |dk/ds(s)| / dk_ref,

so that every mesh interval carries the same integral of M (de Boor's
equidistribution principle; Betts, "Practical Methods for Optimal Control",
mesh refinement; Patterson & Rao, GPOPS-II). The local spacing is therefore
~ c / M(s): dense in corners (|k|) and at corner entry / exit (|dk/ds|, where
the braking, turn-in and traction transients live), sparse on the straights.

Interval count. N defaults to the uniform mesh's round(L / OPT_ds), so the NLP
has the same size and the knots are only REDISTRIBUTED (mean spacing = L / N
~ OPT_ds). Pass N explicitly (or a larger OPT_ds) to spend the gained accuracy
on a smaller NLP instead.

Normalisation. k_ref / dk_ref are the `pct`-th percentiles (default 90th,
length-weighted) of the smoothed |k| and |dk/ds| of the track itself, so a and b
are dimensionless and track-independent: a corner as tight as the track's
typical tight corners gets M ~ 1 + a, i.e. ~(1 + a)x the straight-line knot
density before the spacing bounds act. A signal that is zero up to rounding drops
its term, and M == 1 reproduces the uniform mesh. That covers a straight, dk/ds on
a constant-radius circle (a smoothed constant is constant only to ~1e-16, so its
gradient is pure rounding noise, which must not be rescaled to O(1)) and a track
whose curvature is noise around zero. "Zero up to rounding" is judged against the
track itself: if |k| held at its peak over the whole lap would turn the car by less
than _NEGLIGIBLE (1e-6) rad the track is straight and both terms are dropped; if
dk/ds held at its peak over the whole lap would change k by less than _NEGLIGIBLE
of its peak the curvature is constant and the dk/ds term is dropped. A percentile
at or below its noise floor falls back to the max, as for exact zeros, so noise on
the straights of a track with one real corner does not rescale the corner. Noise
ABOVE these floors is data, not rounding: on a track whose corners cover less than
(100 - pct) percent of its length the percentile is then set by that noise and the
corners can saturate at ds_min.

Smoothing. k is resampled onto a uniform fine grid and smoothed with a centred,
edge-normalised moving average `smooth_window` metres wide (default OPT_ds)
before it is differentiated (real-track curvature is noisy, and the mesh cannot
resolve features shorter than its own spacing). |k| and |dk/ds| are smoothed
again with the same window, so the spacing changes gradually and the refinement
starts about half a window ahead of each corner (mesh_stats reports the largest
neighbouring-interval ratio; a wider window lowers it). functions/simpleMA.py
is deliberately not reused: it is a causal filter followed by a circular shift,
which wraps the end of the lap into its start, whereas the mesh needs an
open-ended (non-periodic) filter.

Defaults a = b = 1 (corners ~2x, corner entry / exit up to ~3x the straight
knot density). Chosen with an a-priori check against a fine (N=36) Sturn
reference solution re-represented on candidate meshes (knot + Legendre points):
at N = 18 / 12 / 9 the curvature settings tried (a, b in 1..3, window 1-2 x
OPT_ds) typically cut the max speed and lateral-offset representation errors
2-3x versus the uniform mesh, with no consistent winner among them, so the
gentlest one is the default: on Barcelona at OPT_ds = 30 it keeps the
straights at <= ~1.65 x OPT_ds (a = b = 2 stretches them to ~2.3 x) and the
neighbouring-interval ratio below ~2.7. The monitor is geometric only: it does
not see brake points, which sit upstream of the corners on the straights;
ds_max bounds the interval that can contain one.

Spacing bounds. ds_min <= dsk <= ds_max (defaults 0.25*OPT_ds and 2.5*OPT_ds)
hold exactly: the monitor is clipped to [c/ds_max, c/ds_min], c being the
per-interval monitor integral. Clipping changes c, so the clip levels and c are
re-normalised together; the bisection below returns the converged limit of the
"clip, re-normalise, repeat" iteration. The ds_max cap keeps a long straight
from swallowing a brake point in one huge interval; ds_min keeps one very tight
(or noisy) corner from draining the whole knot budget.
"""

import numpy as np

A_DEFAULT = 1.0          # weight of |k| / k_ref       (corner body)
B_DEFAULT = 1.0          # weight of |dk/ds| / dk_ref  (corner entry / exit)
PCT_DEFAULT = 90.0       # percentile that sets k_ref and dk_ref
DS_MIN_FRAC = 0.25       # default ds_min = DS_MIN_FRAC * OPT_ds
DS_MAX_FRAC = 2.5        # default ds_max = DS_MAX_FRAC * OPT_ds
_MAX_GRID = 200001       # cap on the fine monitor grid size
_NEGLIGIBLE = 1e-6       # rounding-noise level: turning in rad over the lap (|k|), or
                         # curvature change as a fraction of peak |k| over the lap (dk/ds)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _track_arrays(s, k):
    s = np.asarray(s, dtype=float).reshape(-1)
    k = np.asarray(k, dtype=float).reshape(-1)
    if s.size != k.size or s.size < 2:
        raise ValueError("s and k must be 1-D arrays of the same length (>= 2).")
    if np.any(np.diff(s) < 0) or not s[-1] > s[0]:
        raise ValueError("track s must be increasing with s[-1] > s[0].")
    return s, k


def _box_smooth(x, n):
    """Centred moving average over n samples (n odd), renormalised near the
    ends: no zero padding and no wrap-around."""
    n = int(n)
    if n <= 1 or x.size < 3:
        return x.copy()
    n = min(n, x.size if x.size % 2 == 1 else x.size - 1)
    ker = np.ones(n)
    num = np.convolve(x, ker, mode="same")
    den = np.convolve(np.ones_like(x), ker, mode="same")
    return num / den


def _ref(x, pct, floor=0.0):
    """Scale of a non-negative signal: its pct-th percentile; if that is at or below
    ``floor`` (zero or rounding noise), its max; if even that is at or below
    ``floor``, 0.0 (the signal is numerically zero and its monitor term is dropped)."""
    r = float(np.percentile(x, pct))
    if not r > floor:
        r = float(np.max(x))
    return r if r > floor else 0.0


def default_n_intervals(s, OPT_ds):
    """Interval count of the uniform mesh, round(L / OPT_ds) (at least 1)."""
    s = np.asarray(s, dtype=float).reshape(-1)
    return max(1, int(round((s[-1] - s[0]) / OPT_ds)))


# ----------------------------------------------------------------------------
# monitor function
# ----------------------------------------------------------------------------
def monitor(s, k, OPT_ds, a=A_DEFAULT, b=B_DEFAULT, smooth_window=None,
            pct=PCT_DEFAULT, grid_ds=None):
    """Monitor M(s) = 1 + a|k|/k_ref + b|dk/ds|/dk_ref on a uniform fine grid.

    s, k          : track arc length [m] (increasing) and curvature [1/m]
    OPT_ds        : nominal collocation step [m] (sets the default smoothing)
    smooth_window : moving-average width [m]; None -> OPT_ds, 0 -> no smoothing
    pct           : percentile used for k_ref / dk_ref
    grid_ds       : fine-grid spacing [m]; None -> min(median track ds, OPT_ds/10)
    Returns (sg, M, info) with info = dict(k_ref, dk_ref, n_window, grid_ds, K, DK);
    k_ref / dk_ref are 0.0 for a term that is dropped (numerically zero signal).
    """
    s, k = _track_arrays(s, k)
    L = s[-1] - s[0]
    if grid_ds is None:
        d = np.diff(s)
        d = d[d > 0]
        grid_ds = min(float(np.median(d)) if d.size else L, OPT_ds / 10.0)
    n_g = int(np.ceil(L / float(grid_ds))) + 1
    n_g = int(min(max(n_g, 3), _MAX_GRID))
    sg = np.linspace(s[0], s[-1], n_g)
    h = sg[1] - sg[0]
    kg = np.interp(sg, s, k)

    w = float(OPT_ds) if smooth_window is None else float(smooth_window)
    n_w = max(1, int(round(w / h)))
    if n_w % 2 == 0:
        n_w += 1                                   # centred window: odd length

    K = _box_smooth(np.abs(kg), n_w)               # |k| (smoothed)
    dk = np.gradient(_box_smooth(kg, n_w), sg)     # differentiate the SMOOTHED k
    DK = _box_smooth(np.abs(dk), n_w)              # |dk/ds| (smoothed)

    # Rounding-noise floors, judged against the track itself (module docstring,
    # "Normalisation"): a lap-long turn under _NEGLIGIBLE rad is a straight, so dk/ds
    # is noise too; a lap-long change of k under _NEGLIGIBLE of its peak is constant
    # curvature. Without them _ref would rescale float noise to O(1).
    k_ref = _ref(K, pct, _NEGLIGIBLE / L)
    dk_ref = _ref(DK, pct, _NEGLIGIBLE * float(np.max(K)) / L) if k_ref > 0.0 else 0.0
    M = np.ones_like(sg)
    if a and k_ref > 0.0:
        M = M + a * K / k_ref
    if b and dk_ref > 0.0:
        M = M + b * DK / dk_ref
    return sg, M, dict(k_ref=k_ref, dk_ref=dk_ref, n_window=n_w, grid_ds=h, K=K, DK=DK)


# ----------------------------------------------------------------------------
# equidistribution with spacing bounds
# ----------------------------------------------------------------------------
def _equidistribute(sg, M, N, ds_min, ds_max):
    """Knots s_0 < ... < s_N with equal integrals of clip(M, c/ds_max, c/ds_min)."""
    h = np.diff(sg)
    lo_f = 1.0 / ds_max if np.isfinite(ds_max) else 0.0     # clip floor  = c * lo_f
    hi_f = 1.0 / ds_min if ds_min > 0.0 else np.inf         # clip ceiling = c * hi_f

    def clipped(c):
        return np.clip(M, c * lo_f, c * hi_f)

    def total(Mc):                                          # trapezoid integral
        return float(np.sum(0.5 * (Mc[:-1] + Mc[1:]) * h))

    c = total(M) / N                                        # unclipped solution
    if np.any(M < c * lo_f) or np.any(M > c * hi_f):
        # g(c) = int clip(M/c, lo_f, hi_f) ds is non-increasing in c, from L*hi_f
        # (c -> 0) to L*lo_f (c -> inf); solve g(c) = N by bisection in log c
        c_lo = float(np.min(M)) / hi_f if np.isfinite(hi_f) else c * 1e-12
        c_hi = float(np.max(M)) / lo_f if lo_f > 0.0 else c * 1e12
        for _ in range(200):
            c_mid = np.sqrt(c_lo * c_hi)
            if total(clipped(c_mid)) / c_mid > N:
                c_lo = c_mid
            else:
                c_hi = c_mid
            if c_hi - c_lo <= 1e-15 * c_hi:
                break
        c = 0.5 * (c_lo + c_hi)
    Mc = clipped(c)

    cum = np.concatenate(([0.0], np.cumsum(0.5 * (Mc[:-1] + Mc[1:]) * h)))
    targets = np.linspace(0.0, cum[-1], N + 1)
    s_knot = np.interp(targets, cum, sg)
    s_knot[0], s_knot[-1] = sg[0], sg[-1]
    if not np.all(np.diff(s_knot) > 0):                     # cannot happen for M > 0
        raise RuntimeError("curvature_mesh produced a non-monotone knot vector.")
    return s_knot


def curvature_mesh(s, k, OPT_ds, a=A_DEFAULT, b=B_DEFAULT, ds_min=None, ds_max=None,
                   smooth_window=None, N=None, pct=PCT_DEFAULT, grid_ds=None):
    """Curvature-weighted collocation knots s_knot (length N+1).

    s, k          : track arc length [m] and curvature [1/m]
    OPT_ds        : nominal step [m]; N defaults to round(L / OPT_ds), the uniform
                    mesh's count, so the mean spacing is L / N ~ OPT_ds
    a, b          : weights of |k|/k_ref and |dk/ds|/dk_ref (dimensionless)
    ds_min/ds_max : spacing bounds [m]; None -> 0.25*OPT_ds / 2.5*OPT_ds
    smooth_window : smoothing width [m] (None -> OPT_ds; 0 disables smoothing)
    N             : explicit interval count (overrides round(L / OPT_ds))
    pct, grid_ds  : see monitor()
    Returns a strictly increasing array with s_knot[0] = s[0], s_knot[-1] = s[-1].
    """
    s, k = _track_arrays(s, k)
    L = s[-1] - s[0]
    N = default_n_intervals(s, OPT_ds) if N is None else int(N)
    if N < 1:
        raise ValueError("curvature_mesh needs N >= 1 intervals.")
    ds_min = DS_MIN_FRAC * float(OPT_ds) if ds_min is None else float(ds_min)
    ds_max = DS_MAX_FRAC * float(OPT_ds) if ds_max is None else float(ds_max)
    h_mean = L / N
    if not (0.0 <= ds_min < ds_max):
        raise ValueError(f"need 0 <= ds_min < ds_max (got {ds_min}, {ds_max}).")
    tol = 1e-9 * h_mean
    if ds_min > h_mean + tol or ds_max < h_mean - tol:
        raise ValueError(
            f"spacing bounds [{ds_min:g}, {ds_max:g}] m exclude the mean spacing "
            f"L/N = {h_mean:g} m (N = {N}); widen ds_min/ds_max or change N.")
    if N == 1:
        return np.array([s[0], s[-1]])
    sg, M, _ = monitor(s, k, OPT_ds, a=a, b=b, smooth_window=smooth_window,
                       pct=pct, grid_ds=grid_ds)
    return _equidistribute(sg, M, N, ds_min, ds_max)


# ----------------------------------------------------------------------------
# diagnostics + warm-start helpers
# ----------------------------------------------------------------------------
def mesh_stats(s_knot):
    """Summary of a knot vector: N, length and the interval-size distribution
    (min / mean / max / std / p10 / p50 / p90 [m]) plus the max ratio of
    neighbouring interval sizes (mesh grading)."""
    s_knot = np.asarray(s_knot, dtype=float).reshape(-1)
    dsk = np.diff(s_knot)
    if dsk.size > 1:
        r = dsk[1:] / dsk[:-1]
        grading = float(np.max(np.maximum(r, 1.0 / r)))
    else:
        grading = 1.0
    return dict(N=int(dsk.size), L=float(s_knot[-1] - s_knot[0]),
                ds_min=float(dsk.min()), ds_mean=float(dsk.mean()),
                ds_max=float(dsk.max()), ds_std=float(dsk.std()),
                ds_p10=float(np.percentile(dsk, 10)),
                ds_p50=float(np.percentile(dsk, 50)),
                ds_p90=float(np.percentile(dsk, 90)),
                max_adjacent_ratio=grading)


def mesh_opts_record(mesh_opts):
    """savemat-safe copy of a ``mesh_opts`` dict, for the config fields saved with
    a result (data.mesh_opts). Entries whose value is None (curvature_mesh's
    "use the default") are dropped, numbers become int / float, anything else
    str; None / {} give {} (saved as an empty struct, reloaded as an empty
    namespace). scipy.io.savemat raises TypeError on a None value, which would
    lose a finished solve at save time."""
    out = {}
    for key, val in dict(mesh_opts or {}).items():
        if val is None:
            continue
        if isinstance(val, (bool, np.bool_, int, np.integer)):
            val = int(val)
        elif isinstance(val, (float, np.floating)):
            val = float(val)
        else:
            val = str(val)
        out[str(key)] = val
    return out


def solution_knots(sol, n_pts=None):
    """Arc-length knots of an MLTP / MLTP_initial solution, s_full[::OPT_d+1].

    ``sol`` is a data / data.init namespace or dict (in memory or reloaded from
    .mat) carrying s_full and OPT_d. Returns None if either is missing, or if
    the result is not strictly increasing or does not have ``n_pts`` entries
    (the caller then falls back to a uniform grid)."""
    get = sol.get if isinstance(sol, dict) else (lambda key: getattr(sol, key, None))
    s_full = get("s_full")
    OPT_d = get("OPT_d")
    if s_full is None or OPT_d is None:
        return None
    try:
        d = int(np.asarray(OPT_d).reshape(-1)[0])
        s_full = np.asarray(s_full, dtype=float).reshape(-1)
    except (TypeError, ValueError, IndexError):
        return None
    if d < 1 or (s_full.size - 1) % (d + 1) != 0:
        return None
    knots = s_full[::d + 1]
    if n_pts is not None and knots.size != int(n_pts):
        return None
    if knots.size < 2 or not np.all(np.diff(knots) > 0):
        return None
    return knots
