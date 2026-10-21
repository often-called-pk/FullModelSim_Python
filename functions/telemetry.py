"""telemetry.py - a solved lap as logger channels (numpy only).

resample() gives the channels at equal steps of distance or time. export_monster() writes a ChassisSim
monster file. export_csv() writes the same table as a CSV with a header row.
`data` is the result of MLTP (ctx.data) or of load_solution(). It needs OPT_d, s_full, t_full, x_full and
vehicle_full. A result saved before E1 has no vehicle_full: solve it again.
"""
import numpy as np

from .collocation import collocation_points
from .refine import eval_states, interval_polynomials

G = 9.80665
# EXACT: the states, the near-hold inputs and the channels built from them. The cubic through the NLP nodes of
# one interval follows them (0.002 % of range). LINEAR: the force balance. On a 30 m mesh it wiggles between
# the nodes and a cubic overshoots by up to 60 %. Straight lines join the nodes, so the values stay in range.
EXACT = ("delta", "throttle", "om_gear", "xs_fl", "xs_fr", "xs_rl", "xs_rr")
LINEAR = ("ax", "ay")
STATES = ("vx", "vy", "r", "n", "eps")                      # rows 0-4 of x_full


def _cubic(data, rows, s_out):
    """Interpolate rows (n, len(s_full)) per interval through its NLP nodes (the knot and the d collocation
    points). An input jump at a knot stays a jump. Returns (n, len(s_out))."""
    d = int(data.OPT_d)
    s_full = np.asarray(data.s_full, float)
    knot = np.zeros(s_full.size, bool)
    knot[:: d + 1] = True
    disc = dict(s_knot=s_full[knot], dsk=np.diff(s_full[knot]), tau=collocation_points(d, "legendre"))
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
    Tmax. A negative torque (regen) gives 0 %. sign = (ay, steer). The model is left positive. ChassisSim is
    ASSUMED right positive, because the vendor document does not say."""
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
