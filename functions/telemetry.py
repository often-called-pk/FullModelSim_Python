"""telemetry.py - a solved lap as logger channels, and logger files as channels (numpy and scipy).

resample() gives the channels at equal steps of distance or time. export_monster() writes a ChassisSim
monster file. export_csv() writes the same table as a CSV with a header row.
read_monster() and read_csv() read a logger file as the same channels. split_laps() cuts a log into laps.
project() adds the track position s and n to a log. overlay_report() writes a Plotly page that compares
a log lap with the simulation. Plotly loads only inside overlay_report().
`data` is the result of MLTP (ctx.data) or of load_solution(). It needs OPT_d, s_full, t_full, x_full and
vehicle_full. A result saved before E1 stage M (2026-10-21) has no vehicle_full: solve it again.
"""
import csv
import itertools
import json
import os
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from .collocation import collocation_points
from .refine import eval_states, interval_polynomials
from .track_qa import xy_from_gps
from .warmstart import get_field, vec

G = 9.80665
# EXACT: the states, the inputs the NLP sees (a near hold, or linear with between_knots) and the channels built from
# them. The cubic through the NLP nodes of one interval follows them (0.002 % of range). LINEAR: the force balance.
# On a 30 m mesh it wiggles between the nodes and a cubic overshoots by up to 60 %. Straight lines join the nodes,
# so the values stay in range.
EXACT = ("delta", "throttle", "om_gear", "xs_fl", "xs_fr", "xs_rl", "xs_rr")
LINEAR = ("ax", "ay")
STATES = ("vx", "vy", "r", "n", "eps")                      # rows 0-4 of x_full


def _cubic(data, rows, s_out):
    """Interpolate rows (n, len(s_full)) per interval through its NLP nodes (the knot and the d collocation
    points). An input jump at a knot stays a jump. The points are those of data.collocation (absent: 'legendre').
    Returns (n, len(s_out))."""
    d = int(data.OPT_d)
    s_full = np.asarray(data.s_full, float)
    knot = np.zeros(s_full.size, bool)
    knot[:: d + 1] = True
    scheme = str(getattr(data, "collocation", "legendre"))
    disc = dict(s_knot=s_full[knot], dsk=np.diff(s_full[knot]), tau=collocation_points(d, scheme))
    return eval_states(disc, interval_polynomials(disc, rows[:, knot], rows[:, ~knot]), s_out, x_end=rows[:, -1])


def stations(data, by, step):
    """Arc lengths of the output rows: every `step` m, or at the times 0, step, 2 step ... (t_full inverted
    on a 0.1 m grid)."""
    s0, s1 = float(data.s_full[0]), float(data.s_full[-1])
    if by == "distance":
        return np.append(np.arange(s0, s1, step), s1)
    if by != "time":
        raise ValueError(f"by must be 'distance' or 'time', got {by!r}")
    s_fine = np.append(np.arange(s0, s1, 0.1), s1)
    t_fine = _cubic(data, np.asarray(data.t_full, float)[None], s_fine)[0]
    return np.interp(np.arange(0.0, t_fine[-1], step), t_fine, s_fine)


def resample(data, by="distance", step=1.0):
    """The channels at the stations, as a dict of arrays: s [m], t [s], vx vy [m/s], r [rad/s], n [m], eps [rad],
    delta [rad], throttle [-], om_gear [rad/s], xs_fl..xs_rr [m], ax ay [m/s^2]."""
    vf = data.vehicle_full
    vf = vf if isinstance(vf, dict) else vars(vf)               # a result loaded from a .mat holds a namespace
    s_full, s_out = np.asarray(data.s_full, float), stations(data, by, step)
    rows = np.vstack([np.asarray(data.t_full, float), np.asarray(data.x_full, float)[:5]]
                     + [np.asarray(vf[n], float)[None] for n in EXACT])
    out = dict(s=s_out, **dict(zip(("t",) + STATES + EXACT, _cubic(data, rows, s_out))))
    out.update({n: np.interp(s_out, s_full, np.asarray(vf[n], float)) for n in LINEAR})
    return out


def export_monster(data, path, step=0.02, sign=(-1.0, -1.0)):
    """Write a ChassisSim monster file and return the array written. It has tab-delimited ASCII, no header and a
    row every `step` s (50 Hz). Columns: lap distance [m], rpm, ay [g], ax [g], damper FL FR RL RR [mm],
    steering at the tyre [deg], throttle [%], speed [km/h] (CSim_monster_file_creation.pdf).
    The lap distance is the integral of the speed, as a logger gives it. The throttle is the motor torque in % of
    Tmax. A negative torque (regen) gives 0 %. sign = (ay, steer). The model is left positive, and the default
    (-1, -1) makes the file right positive. Five ChassisSim manuals say that import flag 1 takes a right turn as
    positive, so set both flags to 1. The monster-file manual states no sign, and no file went through a ChassisSim
    import yet (docs/logging_spec.md, Table C)."""
    v = resample(data, "time", step)
    dist = np.concatenate(([0.0], np.cumsum(0.5 * (v["vx"][1:] + v["vx"][:-1]) * np.diff(v["t"]))))
    out = np.column_stack([dist, v["om_gear"] * 30.0 / np.pi, sign[0] * v["ay"] / G, v["ax"] / G,
                           *(v[f"xs_{w}"] * 1e3 for w in ("fl", "fr", "rl", "rr")),
                           sign[1] * np.degrees(v["delta"]), np.maximum(v["throttle"], 0.0) * 100.0, v["vx"] * 3.6])
    fmt = ["%.3f", "%.2f", "%.3f", "%.3f"] + ["%.2f"] * 4 + ["%.3f", "%.1f", "%.3f"]
    np.savetxt(path, out, fmt=fmt, delimiter="\t")
    return out


def export_csv(data, path, step=1.0, by="distance"):
    """Write resample(data, by, step) as a CSV with a header row. The columns and units are those of resample.
    Return the dict."""
    v = resample(data, by, step)
    np.savetxt(path, np.column_stack(list(v.values())), delimiter=",", header=",".join(v), comments="", fmt="%.9g")
    return v


def read_monster(path, step=0.02, sign=(-1.0, -1.0)):
    """Read a ChassisSim monster file (the file of export_monster) as a dict of SI channels with model signs.
    The file has no time column, so t = row * step. s is the lap distance column. It is a path length, not the
    centreline s. sign = (ay, steer) as in export_monster: the same pair reads a file back.
    The reader ignores the columns after the 11th (strain). A file with fewer than 11 columns raises ValueError."""
    # ponytail: loads whole file, stream if > 100 MB (a 100 s lap at 50 Hz is 0.4 MB)
    a = np.loadtxt(path, ndmin=2)                                # tabs or spaces
    if a.shape[1] < 11:
        raise ValueError(f"A monster file needs 11 columns. This file has {a.shape[1]}.")
    return dict(t=step * np.arange(len(a)), s=a[:, 0], om_gear=a[:, 1] * np.pi / 30.0, ay=sign[0] * a[:, 2] * G,
                ax=a[:, 3] * G, **{f"xs_{w}": a[:, 4 + i] * 1e-3 for i, w in enumerate(("fl", "fr", "rl", "rr"))},
                delta=np.radians(sign[1] * a[:, 8]), throttle=a[:, 9] * 0.01, vx=a[:, 10] / 3.6)


def read_csv(path, channel_map=None, beacons="Beacon Markers", chunk=1 << 14):
    """Read a logger CSV as a dict of SI channels. The reader streams the file: it parses `chunk` rows at a
    time and keeps only the mapped columns.
    channel_map is {channel: column} or {channel: (column, unit, factor)}, or the path of a JSON file with the
    same content. The channel is factor * column. If the file has a units row, it must show `unit`.
    Without a map, every column keeps its own name (this reads the file of export_csv).
    The header is the first row above the data that has as many cells as the data and holds every mapped column.
    The reader skips the rows above the header (metadata). A units row can follow the header.
    beacons is the metadata row with the lap end times (the "Beacon Markers" row of an AiM file). If the file has
    this row and the map has a channel t, the result gets the channel lap. Lap 0 ends at the first time, lap 1 at
    the second time, and so on. The times have the unit of the time column, so the factor of t applies to them.
    The first data row must lie in the first 1000 rows. The reader raises ValueError for a file without data,
    a missing column, a wrong unit or an empty cell. Only "," separates the cells."""
    if isinstance(channel_map, (str, os.PathLike)):
        with open(channel_map, encoding="utf-8") as fh:
            channel_map = json.load(fh)
    ent = None if channel_map is None else {k: (v, None, 1.0) if isinstance(v, str) else tuple(v)
                                            for k, v in channel_map.items()}
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        pre = []
        for line in itertools.islice(fh, 1000):                  # rows above the data: metadata, header, units
            row = next(csv.reader([line]), [])
            if not any(c.strip() for c in row):
                continue
            try:
                float(row[0])
            except ValueError:
                pre.append(row)
                continue
            break                                                # `line` is the first data row
        else:
            raise ValueError("The first 1000 rows of the file hold no data row.")
        want = set() if ent is None else {c for c, _, _ in ent.values()}
        i = next((i for i, r in enumerate(pre) if len(r) == len(row) and want <= set(r)), None)
        if i is None:
            raise ValueError(f"The file has no header row. A header row has {len(row)} cells and holds the columns "
                             f"{sorted(want)}.")
        head = pre[i]
        nxt = pre[i + 1] if i + 1 < len(pre) else []             # the units row, if one follows the header
        units = dict(zip(head, (c.strip() for c in nxt))) if len(nxt) == len(head) else {}
        ent = ent or {c: (c, None, 1.0) for c in head}
        for c, u, _ in ent.values():
            if u is not None and units and units.get(c) != u:
                raise ValueError(f"Column {c!r} has the unit {units.get(c)!r}. The map says {u!r}.")
        cols = sorted({head.index(c) for c, _, _ in ent.values()})
        parts, body = [], itertools.chain([line], fh)
        while True:
            block = list(itertools.islice(body, chunk))
            if not block:
                break
            parts.append(np.loadtxt(block, delimiter=",", quotechar='"', usecols=cols, ndmin=2))
    a = np.concatenate(parts)
    del parts
    out = {k: f * a[:, cols.index(head.index(c))] for k, (c, _, f) in ent.items()}
    ends = next((r[1:] for r in pre if r[0] == beacons), None)
    if ends and "t" in out:
        out["lap"] = np.searchsorted([ent["t"][2] * float(c) for c in ends if c.strip()], out["t"], side="left")
    return out


def split_laps(t, dist=None, beacon=None, gap=None, drop=500.0):
    """Return the lap number of each sample as an int array. Lap 0 is the part before the first mark.
    A new lap starts where
    - dist drops by more than `drop` m (the lap distance resets, or s wraps at the line),
    - beacon rises (a lap counter or a pulse),
    - t steps by more than `gap` s (a pause in the log)."""
    t = np.asarray(t, float)
    mark = np.zeros(t.size, bool)
    if dist is not None:
        dist = np.asarray(dist, float)
        i = np.flatnonzero(np.isfinite(dist))                  # a dropout (NaN) must not hide a wrap
        mark[i[1:]] |= np.diff(dist[i]) < -drop
    if beacon is not None:
        mark[1:] |= np.diff(np.asarray(beacon, float)) > 0     # np.diff of a bool array also marks the fall
    if gap is not None:
        mark[1:] |= np.diff(t) > gap
    return np.cumsum(mark)


def _foot(tree, xs, ys, ss, p):
    """Foot point of each row of p on the polyline (xs, ys, arc length ss). Return (s, n).
    n > 0 lies left of the travel direction."""
    _, i = tree.query(p)
    a = np.clip(np.stack([i - 1, i]), 0, xs.size - 2)           # the two segments at the nearest vertex
    A = np.stack([xs[a], ys[a]], -1)                            # (2, M, 2) segment starts
    D = np.stack([xs[a + 1], ys[a + 1]], -1) - A
    u = np.clip(((p - A) * D).sum(-1) / (D * D).sum(-1), 0.0, 1.0)
    d2 = ((p - (A + u[..., None] * D)) ** 2).sum(-1)
    k, m = np.argmin(d2, axis=0), np.arange(p.shape[0])
    a, u, D, A = a[k, m], u[k, m], D[k, m], A[k, m]
    n = np.sign(D[:, 0] * (p[:, 1] - A[:, 1]) - D[:, 1] * (p[:, 0] - A[:, 0])) * np.sqrt(d2[k, m])
    return ss[a] + u * (ss[a + 1] - ss[a]), n


def project(log, track):
    """Return the log with s [m] and n [m] added: its position projected on the track centreline.
    The position is x, y [m] in the frame of the track, or lat, lon [deg]. lat, lon go through xy_from_gps on the
    finite fixes of the log. The origin of that frame is the mean of those fixes, so the track must have the same
    origin. To get it, call xy_from_gps on the finite fixes of this whole log. Give one lap of that x, y to
    write_track. Then project this same log. A track from other fixes (one lap only, or another session) has another
    origin. The log is then off by the distance between the two means. On a synthetic BCN session, a 5 m shift
    gives an rms error of 3 to 4 m in s.
    track needs s, x, y (a dict or a namespace). n > 0 lies left of the travel direction, as in the model.
    The centreline is a cubic spline through the track points, sampled every 1 m. A coarse track still projects well.
    A sample without a position (NaN) gives NaN. The function estimates no shift and no rotation between the log
    and the track.
    It warns if the moving samples (vx > 5 m/s) lie a median of more than 8 m from the centreline. A shift of up to
    20 m can stay below that limit."""
    if get_field(track, "x") is None or get_field(track, "y") is None:
        raise ValueError("The track has no x, y. Build them with curv2cart(s, k).")
    s_t, x_t, y_t = (vec(get_field(track, k)) for k in ("s", "x", "y"))
    ss = np.append(np.arange(s_t[0], s_t[-1], 1.0), s_t[-1])
    xs, ys = CubicSpline(s_t, x_t)(ss), CubicSpline(s_t, y_t)(ss)
    tree = cKDTree(np.column_stack([xs, ys]))
    if "x" in log:
        x, y = np.asarray(log["x"], float), np.asarray(log["y"], float)
    else:
        ok = np.isfinite(log["lat"]) & np.isfinite(log["lon"])
        x, y = np.full(ok.size, np.nan), np.full(ok.size, np.nan)
        x[ok], y[ok] = xy_from_gps(np.asarray(log["lat"], float)[ok], np.asarray(log["lon"], float)[ok])
    ok = np.isfinite(x) & np.isfinite(y)
    out = {**log, "s": np.full(ok.size, np.nan), "n": np.full(ok.size, np.nan)}
    idx = np.flatnonzero(ok)
    for i in range(0, idx.size, 1 << 17):                      # chunks keep the memory low for a log of 1e6 samples
        j = idx[i:i + (1 << 17)]
        out["s"][j], out["n"][j] = _foot(tree, xs, ys, ss, np.column_stack([x[j], y[j]]))
    v = np.asarray(log["vx"], float) if "vx" in log else np.full(ok.size, np.inf)
    d = np.abs(out["n"][ok & (v > 5.0)])                       # a car on the track is within a few m of the centreline
    if d.size and np.median(d) > 8.0:
        warnings.warn(f"The moving samples lie a median of {np.median(d):.1f} m from the centreline. "
                      "The track, the frame or the rotation may be wrong.")
    return out


def time_delta(sim, lap):
    """Return (s, dt): the time delta of one log lap against the simulation.
    s are the stations of `sim` (from resample) that the lap covers. lap is a dict with s and t of one lap.
    dt is the time of the log from the first station, minus the simulated time. dt > 0 means the log is slower.
    dt at the end is good to one log sample: there is no interpolation at the line."""
    ok = np.isfinite(lap["s"])
    s, t = np.maximum.accumulate(lap["s"][ok]), lap["t"][ok]     # a GPS step back must not reverse s
    m = (sim["s"] >= s[0]) & (sim["s"] <= s[-1])
    tl = np.interp(sim["s"][m], s, t)
    return sim["s"][m], (tl - tl[0]) - (sim["t"][m] - sim["t"][m][0])


def _lap_pick(log, s0, L, lap):
    """Return (slice, number, table) of the lap to compare. table lists (number, time, whole) of every lap.
    lap=None picks the fastest whole lap. A whole lap visits 48 of 50 equal parts of s."""
    ids = split_laps(log["t"], dist=log["s"], drop=0.5 * L)
    b = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, ids.size]                  # laps are runs of equal ids
    best, table = None, []
    for i0, i1 in zip(b[:-1], b[1:]):
        s, t = log["s"][i0:i1], log["t"][i0:i1]
        full = np.unique(np.floor((s[np.isfinite(s)] - s0) / (L / 50)).clip(0, 49)).size >= 48
        table.append((int(ids[i0]), float(t[-1] - t[0]), bool(full)))
        if ids[i0] == lap or (lap is None and full and (best is None or t[-1] - t[0] < best[0])):
            best = (t[-1] - t[0], slice(i0, i1), int(ids[i0]))
    if best is None:
        raise ValueError("No lap of the log covers the track." if lap is None else f"The log has no lap {lap}.")
    return best[1], best[2], table


def overlay_report(data, log, path, lap=None, step=1.0):
    """Write one Plotly HTML file that overlays a log lap on the simulated lap. All panels use s as the x axis:
    speed, time delta, ax, ay and steer.
    data is the result of MLTP (as telemetry.resample takes it). log is a dict of SI channels with t, s, vx, ax, ay
    and delta: read_csv or read_monster, then project for a log with positions.
    lap is a number from the laps table of the result. split_laps(t, dist=s, drop=L/2) gives these numbers (L is
    the length of the track). None picks the fastest lap that covers the track.
    Return a dict with the keys lap, log_time, sim_time, delta_end, corr, laps and fig (a Plotly figure).
    The key corr holds the correlation of ay and delta of the log with the simulation. A value below 0 means a
    flipped sign in the channel map or in the sign pair of read_monster, and the function warns. The key laps lists
    (number, time, whole) of every lap.
    The figure is the one that the file holds."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if "s" not in log:
        raise ValueError("The log has no s. Call project(log, track) first.")
    sim = resample(data, "distance", step)
    sl, k, table = _lap_pick(log, sim["s"][0], sim["s"][-1] - sim["s"][0], lap)
    lg = {c: np.asarray(v)[sl] for c, v in log.items() if np.size(v) == np.size(log["t"])}
    s, dt = time_delta(sim, lg)
    ok = np.isfinite(lg["s"])
    s_log = np.maximum.accumulate(lg["s"][ok])
    corr = {c: float(np.corrcoef(np.interp(s, s_log, lg[c][ok]), np.interp(s, sim["s"], sim[c]))[0, 1])
            for c in ("ay", "delta") if c in lg}
    T_log, T_sim = float(lg["t"][-1] - lg["t"][0]), float(sim["t"][-1] - sim["t"][0])
    for c, r in corr.items():
        if r < 0:
            warnings.warn(f"{c} of the log runs against the simulation (corr {r:+.2f}). "
                          "Flip its sign in the channel map of read_csv or in the sign pair of read_monster.")
    rows = (("vx", "v [m/s]", 1.0), ("dt", "log - sim [s]", 1.0), ("ax", "ax [m/s^2]", 1.0), ("ay", "ay [m/s^2]", 1.0),
            ("delta", "steer [deg]", 180.0 / np.pi))
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for r, (c, label, f) in enumerate(rows, 1):
        if c == "dt":
            fig.add_trace(go.Scatter(x=s, y=dt, name="log - sim", line=dict(color="#555"), showlegend=False),
                          row=r, col=1)
            fig.add_hline(y=0.0, line=dict(color="#999", dash="dash"), row=r, col=1)
        else:
            if c in lg:
                fig.add_trace(go.Scatter(x=lg["s"], y=f * lg[c], name="log", legendgroup="log", showlegend=r == 1,
                                         line=dict(color="#e24b4a")), row=r, col=1)
            fig.add_trace(go.Scatter(x=sim["s"], y=f * sim[c], name="simulation", legendgroup="sim", showlegend=r == 1,
                                     line=dict(color="#378ADD")), row=r, col=1)
        fig.update_yaxes(title_text=label, row=r, col=1)
    fig.update_xaxes(title_text="s [m]", row=5, col=1)
    fig.update_layout(template="plotly_white", hovermode="x unified", height=1000,
                      title=f"Log lap {k} ({T_log:.3f} s) against the simulation ({T_sim:.3f} s, start state "
                            f"{get_field(data, 'start_state', '?')}), delta at the end {dt[-1]:+.3f} s")
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return dict(lap=k, log_time=T_log, sim_time=T_sim, delta_end=float(dt[-1]), corr=corr, laps=table, fig=fig)
