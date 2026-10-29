"""E5 log tools of functions/telemetry.py: read_monster, read_csv, split_laps, project, time_delta and overlay_report.

Task 11 of the phase 3 plan. The test uses synthetic data only. A stand-in result and fake logs sit on the BCN and
Sturn centrelines. Every file goes to a temporary directory. The test needs numpy, scipy and plotly. It runs no solve
and needs no Data/DATA_AA.mat. Run it from any directory:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_telemetry_logs.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import json
import os
import pathlib
import tempfile
import tracemalloc
import warnings
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

import functions.telemetry as P
from functions.cartPath import cartPath
from functions.collocation import collocation_points
from functions.curv2cart import curv2cart
from functions.track_qa import WGS84_A, WGS84_E2
from userOpts import _load_track

G = 9.80665
CIRC = os.path.join(_bootstrap.ROOT, "Circuits")

# The AiM map of this test. The code ships no AiM map. Its signs are not verified.
# An entry is (column, unit, factor). A channel equals factor times the column. The units row must show the unit.
AIM_MAP = {"t": ("Time", "s", 1.0), "vx": ("GPS Speed", "km/h", 1 / 3.6), "ax": ("Inline Acc", "g", G),
           "ay": ("Lateral Acc", "g", G), "delta": ("Steering Pos", "deg", np.pi / 180.0),
           "lat": ("GPS Latitude", "deg", 1.0), "lon": ("GPS Longitude", "deg", 1.0)}


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def raises(exc, fn, *a, **kw):
    """The message of the exception when fn raises exc. None when it does not raise."""
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    return None


def warned(fn, *a, **kw):
    """(result, texts of the warnings) of fn(*a, **kw)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [str(x.message) for x in w]


def wrap(e, L):
    """e wrapped to [-L/2, L/2). It removes the jump at the line of a closed track."""
    return ((e + L / 2) % L) - L / 2


# ---------------------------------------------------------------------------------------------- stand-ins
def bcn():
    """The BCN centreline from the parity file. Its s is the chord length."""
    raw = sio.loadmat(os.path.join(CIRC, "Barcelona_circuit.mat"), squeeze_me=True)
    return SimpleNamespace(**{k: np.asarray(raw[k], float) for k in ("s", "k", "x", "y")})


def sturn():
    """The synthetic Sturn centreline. Its curvature comes from userOpts and its x, y from curv2cart."""
    trk = _load_track("Sturn", CIRC)
    x, y = curv2cart(trk.s, trk.k)
    return SimpleNamespace(s=trk.s, k=trk.k, x=x, y=y)


def speed(trk, a_lat=14.0, vmax=70.0, vmin=12.0):
    """A smooth speed profile [m/s] on the track grid. It is the corner limit, smoothed by a moving average."""
    v = np.clip(np.sqrt(a_lat / np.maximum(np.abs(trk.k), 1e-6)), vmin, vmax)
    for _ in range(6):
        v = np.convolve(np.pad(v, 20, mode="edge"), np.ones(41) / 41, mode="valid")[: v.size]
    return v


def fake_result(trk, v, d=3, ds_knot=30.0):
    """A stand-in for a solved lap. It holds the fields that telemetry.resample reads, on 30 m intervals."""
    s0, s1 = trk.s[0], trk.s[-1]
    N = int(round((s1 - s0) / ds_knot))
    sk = np.linspace(s0, s1, N + 1)
    tau = np.asarray(collocation_points(d, "legendre"), float)
    s_full = np.concatenate([sk[i] + np.r_[0.0, tau] * (sk[i + 1] - sk[i]) for i in range(N)] + [[sk[-1]]])
    vf, kf = np.interp(s_full, trk.s, v), np.interp(s_full, trk.s, trk.k)
    fine = np.arange(s0, s1, 0.1)
    vi = np.interp(fine, trk.s, v)
    t_full = np.interp(s_full, fine, np.concatenate(([0.0], np.cumsum(0.1 / (0.5 * (vi[1:] + vi[:-1]))))))
    dv = np.gradient(vf, s_full)
    veh = dict(ax=vf * dv, ay=vf ** 2 * kf, delta=3.0 * kf, throttle=np.clip(dv * vf / 8.0, -1, 1),
               om_gear=vf / 0.355 * 8.0, xs_fl=0.002 * kf * vf, xs_fr=-0.002 * kf * vf, xs_rl=0.001 * dv,
               xs_rr=-0.001 * dv)                      # four different dampers: a swap of two columns shows
    return SimpleNamespace(OPT_d=d, s_full=s_full, t_full=t_full, vehicle_full=veh, lap_time=t_full[-1],
                           start_state="periodic", x_full=np.vstack([vf, 0 * vf, vf * kf, 0 * vf, 0 * vf]))


def lap_on(trk, v, nline, scale=1.0, s_from=None, s_to=None, noise=0.0, t0=0.0, rng=None):
    """One pass over the track from s_from to s_to at scale * v. The line sits nline(s) m to the left.
    The channels are SI values with the model signs."""
    i = (trk.s >= (trk.s[0] if s_from is None else s_from)) & (trk.s <= (trk.s[-1] if s_to is None else s_to))
    s, vv = trk.s[i], scale * v[i]
    x, y = cartPath(trk.x, trk.y, nline(trk.s))
    x, y = x[i], y[i]
    if noise:
        x, y = x + noise * rng.standard_normal(x.size), y + noise * rng.standard_normal(y.size)
    t = t0 + np.concatenate(([0.0], np.cumsum(np.diff(s) / (0.5 * (vv[1:] + vv[:-1])))))
    return dict(t=t, s_true=s, x=x, y=y, vx=vv, ax=vv * np.gradient(vv, s), ay=vv ** 2 * trk.k[i],
                delta=3.0 * trk.k[i])


def session(trk, v, scales=(1.03, 0.98), pit=300, noise=0.3, seed=0):
    """A session with a pit stop, an out-lap (from 0.78 L to the line), flying laps at the scales,
    an in-lap (to 0.55 L) and a second pit stop. The standing samples carry the noise too."""
    rng = np.random.default_rng(seed)
    L = trk.s[-1] - trk.s[0]
    nline = lambda s: 1.0 + np.sin(s / 180.0)
    parts, t0 = [], 0.0

    def stand(x0, y0):
        nonlocal t0
        t = t0 + np.arange(pit) * 0.05
        z = np.zeros(pit)
        parts.append(dict(t=t, s_true=np.full(pit, np.nan), x=x0 + noise * rng.standard_normal(pit),
                          y=y0 + noise * rng.standard_normal(pit), vx=z, ax=z, ay=z, delta=z))
        t0 = t[-1] + 0.05

    stand(trk.x[0] + 25.0, trk.y[0] + 30.0)
    for a, b, sc in [(0.78 * L, L, 0.7)] + [(0.0, L, c) for c in scales] + [(0.0, 0.55 * L, 0.6)]:
        lap = lap_on(trk, v, nline, sc, a, b, noise, t0, rng)
        parts.append(lap)
        t0 = lap["t"][-1] + 0.05
    stand(trk.x[1000] + 20.0, trk.y[1000] + 25.0)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def to_latlon(x, y, lat0=41.57, lon0=2.26):
    """The inverse of the tangent plane of xy_from_gps, at (lat0, lon0)."""
    p0 = np.deg2rad(lat0)
    w = np.sqrt(1.0 - WGS84_E2 * np.sin(p0) ** 2)
    N = WGS84_A / w
    M = N * (1.0 - WGS84_E2) / w ** 2
    return lat0 + np.rad2deg(y / M), lon0 + np.rad2deg(x / (N * np.cos(p0)))


def write_aim_like(path, ses, lat, lon, beacons, nl="\r\n", units_speed="km/h", n_extra=0):
    """Write an AiM-style CSV. It has metadata rows, a blank line, a names row, a units row, a blank line
    and data rows with every field quoted."""
    names = ["Time", "GPS Speed", "GPS Latitude", "GPS Longitude", "Steering Pos", "Lateral Acc", "Inline Acc",
             "Engine RPM", "Temp"] + [f"Other {i}" for i in range(n_extra)]
    units = ["s", units_speed, "deg", "deg", "deg", "g", "g", "rpm", chr(0xB0) + "C"]
    units += [" "] * (len(names) - len(units))
    n = ses["t"].size
    a = np.column_stack([ses["t"], ses["vx"] * 3.6, lat, lon, np.degrees(ses["delta"]), ses["ay"] / G, ses["ax"] / G,
                         3000 + np.arange(n) % 7, 40 + np.arange(n) % 3]
                        + [np.arange(n) % (11 + i) * 1.2345 for i in range(n_extra)])
    fmt = ['"%.4f"', '"%.4f"', '"%.8f"', '"%.8f"'] + ['"%.4f"'] * (len(names) - 4)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for row in (("Format", "AiM CSV File"), ("Session", "Synthetic"), ("Time", "1:35 PM"), ("Sample Rate", "20")):
            fh.write(",".join('"%s"' % c for c in row) + nl)
        fh.write(",".join(['"Beacon Markers"'] + ['"%.3f"' % b for b in beacons]) + nl + nl)
        fh.write(",".join('"%s"' % c for c in names) + nl + ",".join('"%s"' % c for c in units) + nl + nl)
        np.savetxt(fh, a, fmt=",".join(fmt), newline=nl)
    return a


# ---------------------------------------------------------------------------------------------- fixtures
trk_b, trk_s = bcn(), sturn()
L_b = trk_b.s[-1] - trk_b.s[0]
v_b = speed(trk_b)
data = fake_result(trk_b, v_b)

with tempfile.TemporaryDirectory() as tmp:
    # =========================================================================================
    print("7. read_monster: export_monster, then read_monster, gives the channels of resample (file rounding)")
    rt = P.resample(data, "time", 0.02)
    f_mon = os.path.join(tmp, "monster.txt")
    P.export_monster(data, f_mon)
    mon = P.read_monster(f_mon)
    # half the last printed digit of each column (the throttle is a percentage with one decimal)
    tol = dict(vx=0.0005 / 3.6, ax=0.0005 * G, ay=0.0005 * G, delta=np.radians(0.0005), om_gear=0.005 * np.pi / 30,
               throttle=0.0005, xs_fl=5e-6, xs_fr=5e-6, xs_rl=5e-6, xs_rr=5e-6)
    for c, e in tol.items():
        ref = np.maximum(rt[c], 0.0) if c == "throttle" else rt[c]      # the file clips a regen torque to 0 %
        err = np.abs(mon[c] - ref).max()
        ok(f"{c}: max |read - resample| = {err:.2e} <= {e * 1.01:.2e} (half the last digit of the file)",
           err <= e * 1.01)
    ok("t is the 0.02 s grid of the file and agrees with the resampled t to 2 ms",
       np.abs(mon["t"] - rt["t"]).max() < 2e-3 and np.allclose(np.diff(mon["t"]), 0.02))
    ok("s is the distance column: it starts at 0 and ends within 0.5 % of the track length",
       mon["s"][0] == 0.0 and abs(mon["s"][-1] - data.s_full[-1]) < 0.005 * data.s_full[-1])
    f_pos = os.path.join(tmp, "monster_pos.txt")
    P.export_monster(data, f_pos, sign=(1.0, 1.0))
    pos = P.read_monster(f_pos, sign=(1.0, 1.0))
    ok("the other sign pair, on export and on read, gives the same channels",
       all(np.allclose(pos[c], mon[c], rtol=0, atol=1e-12) for c in mon))
    flip = P.read_monster(f_pos)
    ok("a read with the default sign pair flips only ay and delta",
       sorted(c for c in mon if not np.allclose(flip[c], pos[c], rtol=0, atol=1e-12)) == ["ay", "delta"]
       and np.allclose(flip["ay"], -pos["ay"]) and np.allclose(flip["delta"], -pos["delta"]))
    f_sp = os.path.join(tmp, "monster_spaces.txt")
    with open(f_mon) as fh_in, open(f_sp, "w") as fh_out:
        fh_out.write(fh_in.read().replace("\t", "  "))           # the vendor allows tabs or spaces
    ok("a space delimited monster file reads the same as the tab file",
       all(np.array_equal(P.read_monster(f_sp)[c], mon[c]) for c in mon))
    bad = os.path.join(tmp, "bad.txt")
    np.savetxt(bad, np.zeros((3, 4)), delimiter="\t")
    ok("a file with fewer than 11 columns raises", raises(ValueError, P.read_monster, bad) is not None)
    f_strain = os.path.join(tmp, "monster_strain.txt")
    with open(f_mon) as fh_in, open(f_strain, "w") as fh_out:
        fh_out.writelines(ln.rstrip("\n") + "\t1.5\t2.5\t3.5\t4.5\n" for ln in fh_in)      # 4 strain columns [kgf]
    ok("a file with the 4 strain columns of the vendor (15 columns) reads the same channels",
       all(np.array_equal(P.read_monster(f_strain)[c], mon[c]) for c in mon))

    # =========================================================================================
    print("8. read_csv on the export_csv file (no channel map)")
    f_csv = os.path.join(tmp, "own.csv")
    P.export_csv(data, f_csv, 1.0)
    own, ref = P.read_csv(f_csv), P.resample(data, "distance", 1.0)
    ok("the 16 channels come back under their names", list(own) == list(ref) and len(own) == 16)
    ok("every channel equals resample to 1e-8 of its range",
       all(np.abs(own[c] - ref[c]).max() <= 1e-8 * max(np.abs(ref[c]).max(), 1e-9) for c in ref))
    f_bom = os.path.join(tmp, "own_bom.csv")
    with open(f_csv, encoding="utf-8") as fh_in, open(f_bom, "w", encoding="utf-8-sig") as fh_out:
        fh_out.writelines(fh_in)                                  # the BOM sits right before the header row
    bom = P.read_csv(f_bom)
    ok("a BOM before the header row keeps the 16 names and the values",
       list(bom) == list(own) and all(np.array_equal(bom[c], own[c]) for c in own))
    ok("a map with a unit reads a file that has no units row (there is nothing to check)",
       np.array_equal(P.read_csv(f_csv, {"s": ("s", "m", 1.0)})["s"], own["s"]))

    # =========================================================================================
    print("9. read_csv on an AiM-style file (metadata rows, units row, quoted fields, CRLF, UTF-8)")
    ses = {k: v[:123] for k, v in lap_on(trk_b, v_b, lambda s: 0 * s, 1.0, 0.0, 200.0).items()}
    lat, lon = to_latlon(ses["x"], ses["y"])
    beacons = [0.5 * (ses["t"][40] + ses["t"][41]), 0.5 * (ses["t"][90] + ses["t"][91])]   # Beacon Markers row
    f_aim = os.path.join(tmp, "aim.csv")
    write_aim_like(f_aim, ses, lat, lon, beacons)
    log = P.read_csv(f_aim, AIM_MAP)
    ok("the map gives t, vx, ax, ay, delta, lat, lon and lap (from the Beacon Markers row)",
       sorted(log) == sorted(list(AIM_MAP) + ["lap"]))
    ok("lap 0 ends at the first beacon and lap 1 at the second: 41, 50 and 32 samples",
       np.array_equal(np.bincount(log["lap"]), [41, 50, 32])
       and np.array_equal(P.split_laps(log["t"], beacon=log["lap"]), log["lap"]))
    # The times have 3 decimals, so a marker can equal a sample time.
    tb = np.round(ses["t"], 3)
    f_mk = os.path.join(tmp, "aim_marker.csv")
    write_aim_like(f_mk, dict(ses, t=tb), lat, lon, [tb[40], tb[90]])
    mk = P.read_csv(f_mk, AIM_MAP)["lap"]
    ok("a sample at a marker time belongs to the lap that the marker ends (samples 40 and 41)",
       mk[40] == 0 and mk[41] == 1)
    f_ms = os.path.join(tmp, "aim_ms.csv")                      # a logger that writes the time in ms
    write_aim_like(f_ms, dict(ses, t=1e3 * ses["t"]), lat, lon, [1e3 * b for b in beacons])
    ms = P.read_csv(f_ms, dict(AIM_MAP, t=("Time", None, 1e-3)))
    ok("time and beacons in ms, map factor 1e-3: t in s and the same laps as the file in s",
       np.allclose(ms["t"], log["t"], atol=1e-4) and np.array_equal(ms["lap"], log["lap"]))
    ok("beacons=None gives no lap channel", "lap" not in P.read_csv(f_aim, AIM_MAP, beacons=None))
    ok("a map without t gives no lap channel, although the file has the Beacon Markers row",
       list(P.read_csv(f_aim, {"vx": "GPS Speed"})) == ["vx"])
    ok("km/h, g and deg become SI (the rounding of 4 decimals)",
       np.abs(log["vx"] - ses["vx"]).max() < 2e-4 / 3.6 and np.abs(log["ay"] - ses["ay"]).max() < 6e-5 * G
       and np.abs(log["ax"] - ses["ax"]).max() < 6e-5 * G and np.abs(log["delta"] - ses["delta"]).max() < 1e-6
       and np.allclose(log["t"], ses["t"], atol=1e-4))
    ok("lat and lon keep their 8 decimals",
       np.abs(log["lat"] - lat).max() < 1e-8 and np.abs(log["lon"] - lon).max() < 1e-8)
    small = P.read_csv(f_aim, AIM_MAP, chunk=5)
    ok("chunk = 5 rows gives the same arrays as one chunk", all(np.array_equal(small[k], log[k]) for k in log))
    ok("a map with only Time works, although Time is also a metadata key, and its unit check reads the units row "
       "under the names row",
       list(P.read_csv(f_aim, {"t": "Time"})) == ["t", "lap"]
       and np.array_equal(P.read_csv(f_aim, {"t": ("Time", "s", 1.0)})["t"], log["t"]))
    f_json = os.path.join(tmp, "map.json")
    with open(f_json, "w") as fh:
        json.dump({k: list(v) for k, v in AIM_MAP.items()}, fh)
    ok("the map can be a JSON file (a str or a Path)",
       all(np.array_equal(P.read_csv(f_aim, f_json)[k], log[k]) for k in log)
       and all(np.array_equal(P.read_csv(f_aim, pathlib.Path(f_json))[k], log[k]) for k in log))
    f_cp = os.path.join(tmp, "aim_cp1252.csv")
    with open(f_aim, encoding="utf-8", newline="") as fh_in, \
            open(f_cp, "w", encoding="cp1252", newline="") as fh_out:
        fh_out.write(fh_in.read())
    ok("a cp1252 file (the degree sign as one byte) reads the same",
       all(np.array_equal(P.read_csv(f_cp, AIM_MAP)[k], log[k]) for k in log))
    f_lf = os.path.join(tmp, "aim_lf.csv")
    write_aim_like(f_lf, ses, lat, lon, beacons, nl="\n")
    ok("LF line ends read the same", all(np.array_equal(P.read_csv(f_lf, AIM_MAP)[k], log[k]) for k in log))
    f_mph = os.path.join(tmp, "aim_mph.csv")
    write_aim_like(f_mph, ses, lat, lon, beacons, units_speed="mph")
    msg = raises(ValueError, P.read_csv, f_mph, AIM_MAP)
    ok("a units row that disagrees with the map raises and names the column", msg is not None and "GPS Speed" in msg)
    msg = raises(ValueError, P.read_csv, f_aim, {"v": "Nope"})
    ok("a missing column raises and names it", msg is not None and "Nope" in msg)
    f_empty = os.path.join(tmp, "empty.csv")
    with open(f_empty, "w") as fh:
        fh.write("t,v\n")
    ok("a file without data rows raises", raises(ValueError, P.read_csv, f_empty) is not None)
    f_far = os.path.join(tmp, "far_header.csv")
    with open(f_far, "w") as fh:
        fh.write("".join(f"note {i}\n" for i in range(1500)) + "t,v\n0.0,1.0\n1.0,2.0\n")
    msg = raises(ValueError, P.read_csv, f_far)
    ok("1500 text rows, then a header and data: the scan stops at row 1000 and raises",
       msg is not None and "1000" in msg)
    # streaming: the file is read chunk by chunk, so the peak memory stays far below the size of the file
    n_big = 60000
    big = {k: np.resize(v, n_big) for k, v in ses.items()}
    big["t"] = np.arange(n_big) * 0.05
    f_big = os.path.join(tmp, "big.csv")
    write_aim_like(f_big, big, np.resize(lat, n_big), np.resize(lon, n_big), beacons, n_extra=31)   # 40 columns
    size = os.path.getsize(f_big)
    tracemalloc.start()
    try:
        got = P.read_csv(f_big, AIM_MAP, chunk=2000)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    kept = sum(v.nbytes for v in got.values())
    ok(f"{n_big} rows ({size / 1e6:.1f} MB file): peak {peak / 1e6:.1f} MB < half the file, "
       f"kept {kept / 1e6:.1f} MB", got["t"].size == n_big and peak < 0.5 * size and peak < 3.0 * kept)

    # =========================================================================================
    print("10. split_laps: distance reset, beacon column, time gap, NaN dropout")
    t = np.arange(40) * 0.1
    d = np.tile(np.arange(10) * 100.0, 4)                         # the lap distance resets every 10 samples
    ok("a drop of dist by more than drop starts a lap",
       np.array_equal(P.split_laps(t, dist=d), np.repeat([0, 1, 2, 3], 10)))
    dj = d.copy()
    dj[5] = dj[4] - 5.0                                           # a 5 m step back inside a lap: GPS jitter
    ok("a small drop (GPS jitter) does not start a lap", P.split_laps(t, dist=dj).max() == 3)
    b = np.r_[np.zeros(7), np.ones(8), 2 * np.ones(25)]
    ok("a rise of the beacon column starts a lap",
       np.array_equal(P.split_laps(t, beacon=b), np.repeat([0, 1, 2], [7, 8, 25])))
    tg = np.r_[np.arange(20) * 0.1, 50.0 + np.arange(20) * 0.1]
    ok("a time step above gap starts a lap", np.array_equal(P.split_laps(tg, gap=5.0), np.repeat([0, 1], 20)))
    dn = d.copy()
    dn[9:11] = np.nan                                             # the reset falls inside a dropout
    ok("a NaN dropout does not hide the reset", P.split_laps(t, dist=dn).max() == 3)
    ok("no rule gives one lap", P.split_laps(t).max() == 0)
    pulse = np.zeros(40, bool)
    pulse[[10, 20, 30]] = True                                    # a bool pulse: True for one sample at the line
    ok("a bool pulse starts one lap per pulse (its fall starts none)",
       np.array_equal(P.split_laps(t, beacon=pulse), np.repeat([0, 1, 2, 3], 10)))

    # =========================================================================================
    print("11. project: laps offset by n = +-1.5 m (cartPath) on the BCN and Sturn centrelines")
    for name, trk, L in (("BCN", trk_b, L_b), ("Sturn", trk_s, trk_s.s[-1] - trk_s.s[0])):
        for n0 in (1.5, -1.5):
            x, y = cartPath(trk.x, trk.y, np.full(trk.s.size, n0))
            out = P.project({"x": x, "y": y}, trk)
            es = wrap(out["s"] - trk.s, L if name == "BCN" else 1e9)      # Sturn is open: no seam
            en = out["n"] - n0
            ok(f"{name}, n = {n0:+.1f}: |s error| {np.abs(es).max():.3f} m < 0.3, "
               f"|n error| {np.abs(en).max():.3f} m < 0.03", np.abs(es).max() < 0.3 and np.abs(en).max() < 0.03)
    x, y = cartPath(trk_b.x, trk_b.y, np.full(trk_b.s.size, 1.5))
    rng = np.random.default_rng(2)
    out = P.project({"x": x + 0.3 * rng.standard_normal(x.size), "y": y + 0.3 * rng.standard_normal(y.size)}, trk_b)
    rms_s = np.sqrt(np.mean(wrap(out["s"] - trk_b.s, L_b) ** 2))
    ok(f"GPS noise 0.3 m: n mean {np.mean(out['n']):.3f} (true 1.5), rms s error {rms_s:.2f} m",
       abs(np.mean(out["n"]) - 1.5) < 0.05 and rms_s < 0.5)
    sub = np.append(np.arange(0, trk_b.s.size, 7), trk_b.s.size - 1)
    coarse = dict(s=trk_b.s[sub], x=trk_b.x[sub], y=trk_b.y[sub])     # about 7 m apart, like a collocation mesh
    out = P.project({"x": x, "y": y}, coarse)
    ok(f"a 7 m polyline (a dict) still gives |n error| {np.abs(out['n'] - 1.5).max():.2f} m < 0.3",
       np.abs(out["n"] - 1.5).max() < 0.3)
    xn = x.copy()
    xn[5] = np.nan
    out = P.project({"x": xn, "y": y}, trk_b)
    others = np.arange(out["s"].size) != 5
    ok("a NaN position gives NaN s and n, and the other samples stay finite",
       np.isnan(out["s"][5]) and np.isnan(out["n"][5]) and np.isfinite(out["s"][others]).all()
       and np.isfinite(out["n"][others]).all())
    msg = raises(ValueError, P.project, {"x": x[:9], "y": y[:9]}, SimpleNamespace(s=trk_b.s, k=trk_b.k))
    ok("a track without x, y raises and names curv2cart", msg is not None and "curv2cart" in msg)
    log_in = {"x": x, "y": y}
    out = P.project(log_in, trk_b)
    ok("the input log is not changed, and the result has s and n", "s" not in log_in and {"s", "n"} <= set(out))

    ses = session(trk_b, v_b)
    lat, lon = to_latlon(ses["x"], ses["y"])
    glog = {"lat": lat, "lon": lon, "t": ses["t"], "vx": ses["vx"], "ax": ses["ax"], "ay": ses["ay"],
            "delta": ses["delta"]}
    fin = np.isfinite(ses["s_true"])                              # the lap samples (standing ones have no true s)
    # xy_from_gps puts the origin at the mean of the fixes. The log frame is the track frame minus that mean.
    mx, my = ses["x"].mean(), ses["y"].mean()
    track_gps = SimpleNamespace(s=trk_b.s, x=trk_b.x - mx, y=trk_b.y - my)
    proj = P.project(glog, track_gps)
    es = wrap(proj["s"][fin] - ses["s_true"][fin], L_b)
    ok(f"lat, lon in the track frame: s error rms {np.sqrt(np.mean(es ** 2)):.2f} m (< 1.5), "
       f"max {np.abs(es).max():.2f} m (< 4)", np.sqrt(np.mean(es ** 2)) < 1.5 and np.abs(es).max() < 4.0)
    ok("the flying laps stay within 3 m of the centreline (the standing samples do not count)",
       np.abs(proj["n"][fin][ses["vx"][fin] > 20]).max() < 3.0)
    _, w_good = warned(P.project, glog, track_gps)
    ok("a log in the right frame gives no centreline warning", not [m for m in w_good if "centreline" in m])
    ref_lap = lap_on(trk_b, v_b, lambda s: 0 * s)
    cx, cy = np.mean(ref_lap["x"]), np.mean(ref_lap["y"])
    track_other = SimpleNamespace(s=trk_b.s, x=trk_b.x - cx, y=trk_b.y - cy)     # built from another lap
    _, w_off = warned(P.project, glog, track_other)
    ok(f"a track {np.hypot(mx - cx, my - cy):.0f} m off the log frame warns once with the median distance",
       len([m for m in w_off if "median" in m]) == 1)
    th = np.deg2rad(5.0)
    dx, dy = ses["x"] - mx, ses["y"] - my
    lat_r, lon_r = to_latlon(np.cos(th) * dx - np.sin(th) * dy, np.sin(th) * dx + np.cos(th) * dy)
    _, w_rot = warned(P.project, dict(glog, lat=lat_r, lon=lon_r), track_gps)
    ok("a log rotated by 5 deg about its mean (not estimated) warns once with the rotation hint",
       len([m for m in w_rot if "rotation" in m]) == 1)

    # =========================================================================================
    print("12. time delta, lap pick and the overlay report")
    sim = P.resample(data, "distance", 1.0)
    ids = P.split_laps(proj["t"], dist=proj["s"], drop=0.5 * L_b)
    ok("the session splits into 4 laps at the wrap of s (pit and out-lap, 2 flying, in-lap and pit)",
       ids.max() == 3)
    f_html = os.path.join(tmp, "overlay.html")
    res, w_ok = warned(P.overlay_report, data, proj, f_html)
    ok(f"the fastest whole lap is lap {res['lap']} (the 1.03 scale lap: {data.t_full[-1] / 1.03:.2f} s)",
       res["lap"] == 1)
    ok("the lap table: laps 1 and 2 are whole, the out-lap and the in-lap are not",
       [whole for _, _, whole in res["laps"]] == [False, True, True, False])
    # Each sample gets its part of s, from 0 to 49. Lap 2 without part 25 covers 49 parts.
    part = np.floor((proj["s"] - sim["s"][0]) / ((sim["s"][-1] - sim["s"][0]) / 50)).clip(0, 49)
    lap49 = {c: np.asarray(v)[(ids == 2) & (part != 25)] for c, v in proj.items()}
    msg = raises(ValueError, P.overlay_report, data, lap49, os.path.join(tmp, "lap49.html"))
    ok("a lap of 49 of its 50 parts of s counts as whole (the rule is 48), so overlay_report accepts it",
       msg is None)
    lg = {c: v[ids == res["lap"]] for c, v in proj.items()}
    s_d, dt = P.time_delta(sim, lg)
    t_sim = np.interp(s_d, sim["s"], sim["t"])
    an = (1 / 1.03 - 1.0) * (t_sim - t_sim[0])
    ok(f"dt(s) follows (1/1.03 - 1) t_sim within {np.abs(dt - an).max():.3f} s (< 0.1 s), "
       f"ends at {dt[-1]:+.3f} s", np.abs(dt - an).max() < 0.1 and abs(dt[-1] - an[-1]) < 0.05)
    self_lap = {"s": sim["s"].copy(), "t": sim["t"].copy()}
    ok("a log equal to the simulation has dt = 0", np.abs(P.time_delta(sim, self_lap)[1]).max() < 1e-9)
    # The stations lie 30 m apart.
    st30 = dict(s=np.arange(0.0, 290.0, 30.0), t=np.arange(0.0, 290.0, 30.0) / 20.0)
    j = np.arange(300.0)
    # The log has a sample every 1 m at 20 m/s. Its s steps back 5 m at 124 m.
    dip = dict(s=np.where(j <= 124, j, j - 6.0), t=0.05 * j)
    s_dip, dt_dip = P.time_delta(st30, dip)
    ok("a 5 m step back of s inside a lap: the station at 120 m keeps dt 0, the station at 150 m has 0.3 s",
       np.abs(dt_dip[s_dip == 120.0]).max() < 1e-9 and np.abs(dt_dip[s_dip == 150.0] - 0.3).max() < 1e-9)
    ok("overlay_report gives the lap times and the delta",
       abs(res["log_time"] - data.t_full[-1] / 1.03) < 0.1 and abs(res["sim_time"] - data.t_full[-1]) < 1e-9
       and abs(res["delta_end"] - an[-1]) < 0.05)
    titles = [a.title.text for a in res["fig"].select_yaxes()]
    ok("the figure has 5 panels (speed, time delta, ax, ay, steer) and 9 traces (4 pairs and the delta)",
       titles == ["v [m/s]", "log - sim [s]", "ax [m/s^2]", "ay [m/s^2]", "steer [deg]"] and len(res["fig"].data) == 9)
    with open(f_html) as fh:
        html_text = fh.read()
    ok("the HTML file holds the figure and plotly.js (it opens offline)",
       os.path.getsize(f_html) > 1e6 and "Plotly" in html_text)
    ok("ay and steer of the log agree in sign with the simulation: corr > 0.9, no sign warning",
       res["corr"]["ay"] > 0.9 and res["corr"]["delta"] > 0.9 and not [m for m in w_ok if "runs against" in m])
    flipped = dict(proj, ay=-proj["ay"], delta=-proj["delta"])
    res2, w_flip = warned(P.overlay_report, data, flipped, os.path.join(tmp, "flipped.html"))
    ok("a flipped ay and steer give corr < -0.9 and one warning each",
       res2["corr"]["ay"] < -0.9 and res2["corr"]["delta"] < -0.9
       and len([m for m in w_flip if "runs against" in m]) == 2)
    ok("lap = 2 compares the other flying lap",
       warned(P.overlay_report, data, proj, os.path.join(tmp, "lap2.html"), lap=2)[0]["lap"] == 2)
    # A GPS dropout: 40 NaN fixes in lap 1. The frame origin is the mean of the finite fixes.
    gap_i = np.flatnonzero(ids == 1)[1000:1040]
    gn = dict(glog, lat=glog["lat"].copy(), lon=glog["lon"].copy())
    gn["lat"][gap_i] = gn["lon"][gap_i] = np.nan
    fix = np.isfinite(gn["lat"])
    pn = P.project(gn, SimpleNamespace(s=trk_b.s, x=trk_b.x - ses["x"][fix].mean(), y=trk_b.y - ses["y"][fix].mean()))
    ok("NaN fixes give NaN s and n there, and every other sample keeps a finite s and n",
       np.isnan(pn["s"][~fix]).all() and np.isnan(pn["n"][~fix]).all()
       and np.isfinite(pn["s"][fix]).all() and np.isfinite(pn["n"][fix]).all())
    rn = warned(P.overlay_report, data, pn, os.path.join(tmp, "dropout.html"))[0]
    ok(f"the lap with the dropout: lap {rn['lap']}, delta at the end {rn['delta_end']:+.3f} s, "
       f"corr {min(rn['corr'].values()):.3f}",
       rn["lap"] == 1 and abs(rn["delta_end"] - an[-1]) < 0.05 and min(rn["corr"].values()) > 0.9)
    thin = {c: v[::70] for c, v in proj.items()}                 # samples 70 m apart: the wrap drops by L - 70 m
    ok("samples 70 m apart (each of the 50 parts of s keeps samples) still give lap 1",
       warned(P.overlay_report, data, thin, os.path.join(tmp, "thin.html"))[0]["lap"] == 1)
    msg = raises(ValueError, P.overlay_report, data, glog, f_html)
    ok("a log without s raises and says project", msg is not None and "project" in msg)
    ok("a log without a complete lap raises",
       raises(ValueError, P.overlay_report, data, {k: v[:400] for k, v in proj.items()}, f_html) is not None)
    mon_log = P.read_monster(f_mon)
    res3 = P.overlay_report(data, mon_log, os.path.join(tmp, "monster.html"))
    ok(f"the monster file of the simulation overlaid on the simulation: delta {res3['delta_end']:+.4f} s, "
       f"corr {res3['corr']['ay']:.4f}", abs(res3["delta_end"]) < 0.01 and res3["corr"]["ay"] > 0.999)

print("\nALL telemetry log TESTS PASSED")
