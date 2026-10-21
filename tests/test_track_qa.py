"""Track QA gate and rebuilt circuit data (Phase 2 plan, Task 2).

Covers functions/track_qa.py (check_track, k_from_xy, xy_from_gps, write_track), the QA hook in
userOpts._load_track, the rebuilt Jarama and BCN sector files, the removal of
BCNAssetto, the QSS lap pins and the GPS track builder. No NLP solve. Sections 11 and 14 need
Data/DATA_AA.mat and print SKIP without it. Run from any directory:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_track_qa.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import shutil
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

from functions.context import Ctx
from functions.track_qa import check_track, k_from_xy, write_track, xy_from_gps
from MLTP_screen import MLTP_screen
from userOpts import _load_track, userOpts

TWO_PI = 2.0 * np.pi
R = 100.0                                   # circle radius [m] for the synthetic checks
CIRC = os.path.join(_bootstrap.ROOT, "Circuits")


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


WARNS = []


def quiet(fn, *a, **kw):
    """Call fn with stdout swallowed. Warning texts go to WARNS, not to the screen."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            return fn(*a, **kw)
        finally:
            WARNS[:] = [str(x.message) for x in w]


def circle(k_scale=1.0, n=2001):
    """Circle of radius R with k = k_scale / R. The last sample repeats the first."""
    t = np.linspace(0.0, TWO_PI, n)
    return SimpleNamespace(s=R * t, k=np.full(n, k_scale / R), x=R * np.cos(t), y=R * np.sin(t))


def dist_to_polygon(px, py, qx, qy):
    """Distance from each point (qx, qy) to the closed polygon (px, py)."""
    dx = np.roll(px, -1) - px               # segment i runs from vertex i to vertex i + 1
    dy = np.roll(py, -1) - py
    seg2 = dx * dx + dy * dy
    qx = np.asarray(qx, float)[:, None]
    qy = np.asarray(qy, float)[:, None]
    u = np.clip(((qx - px) * dx + (qy - py) * dy) / seg2, 0.0, 1.0)
    return np.hypot(qx - (px + u * dx), qy - (py + u * dy)).min(axis=1)


# 1-3. check_track on a circle with k = 1/R, -1/R and 1.03/R
met, iss = check_track(circle())
ok("1 circle k = 1/R: no issues", iss == [])
ok("1 circle: closed is True", met["closed"] is True)
ok("1 circle: k turns 1 lap", abs(met["turn_k"] - 1) < 1e-9)
ok("1 circle: sign agrees at every curved point", met["sign_agree"] == 1.0)
ok("1 circle: path from k closes", met["gap_k"] < 1e-6)

met, iss = check_track(circle(-1.0))
ok("2 circle k = -1/R: sign and turning errors",
   iss == [("error", "sign of k agrees with the x,y heading at 0.0% of the curved points"),
           ("error", "k turns -1.0000 laps, x,y +1.0000 (-720.0 deg)")])

met, iss = check_track(circle(1.03))
ok("3 circle k = 1.03/R: turning and gap warnings",
   iss == [("warn", "k turns +1.0300 laps, x,y +1.0000 (+10.8 deg)"),
           ("warn", "the path integrated from k ends 18.3 m from its start (2.91% of the lap)")])

# 4. input errors come back at once, one per call
base = circle()
s_dup = base.s.copy()
s_dup[5] = s_dup[4]
ok("4 zero s step: one error",
   check_track(SimpleNamespace(s=s_dup, k=base.k, x=base.x, y=base.y))[1]
   == [("error", "s is not strictly increasing (1 steps <= 0)")])
k_nan = base.k.copy()
k_nan[3] = np.nan
ok("4 NaN in k: one error",
   check_track(SimpleNamespace(s=base.s, k=k_nan, x=base.x, y=base.y))[1]
   == [("error", "s or k holds NaN or inf")])
ok("4 k one sample short: length error",
   check_track(SimpleNamespace(s=base.s, k=base.k[:-1], x=base.x, y=base.y))[1]
   == [("error", "s and k need the same length >= 3 (got 2001, 2000)")])

# 5. k_from_xy on circles, an open arc and a stadium with long chords
t37 = np.linspace(0.0, TWO_PI, 38)[:-1]     # 37 vertices; k_from_xy closes the polygon
s37, x37, y37, k37 = k_from_xy(R * np.cos(t37), R * np.sin(t37), closed=True)
ok("5 37-vertex circle: turning 1 lap (1e-12)", abs(np.trapezoid(k37, s37) / TWO_PI - 1) < 1e-12)
ok("5 37-vertex circle: median k = 1/R (2e-3)", abs(np.median(k37) * R - 1) < 2e-3)
t2001 = np.linspace(0.0, TWO_PI, 2001)
_, _, _, k2001 = k_from_xy(R * np.cos(t2001), R * np.sin(t2001), closed=True)
ok("5 2001-point circle: max |k R - 1| < 1e-5", np.abs(k2001 * R - 1).max() < 1e-5)
s_cw, _, _, k_cw = k_from_xy(R * np.cos(t37), -R * np.sin(t37), closed=True)
ok("5 clockwise circle: turning -1 lap (1e-12)", abs(np.trapezoid(k_cw, s_cw) / TWO_PI + 1) < 1e-12)
tq = np.linspace(0.0, np.pi / 2, 200)
s_arc, _, _, k_arc = k_from_xy(R * np.cos(tq), R * np.sin(tq), closed=False)
ok("5 open quarter arc: turning 0.25 lap (1e-6)",
   abs(np.trapezoid(k_arc, s_arc) / TWO_PI - 0.25) < 1e-6)
ok("5 closed output: x[0] == x[-1]", x37[0] == x37[-1])
ok("5 closed output: y[0] == y[-1]", y37[0] == y37[-1])
ok("5 closed output: k[0] == k[-1]", k37[0] == k37[-1])
ok("5 closed output: s steps uniform (ptp < 1e-9)", np.ptp(np.diff(s37)) < 1e-9)
a1 = np.deg2rad(np.arange(-90, 91, 10))
a2 = np.deg2rad(np.arange(90, 271, 10))
px = np.r_[300 + 50 * np.cos(a1), 50 * np.cos(a2)]
py = np.r_[50 * np.sin(a1), 50 * np.sin(a2)]
_, xs, ys, _ = k_from_xy(px, py, closed=True)
ok("5 stadium with two 300 m chords: rebuilt points within 1 m of the polygon",
   dist_to_polygon(px, py, xs, ys).max() < 1.0)

# 6. Jarama, rebuilt from its x,y
t_jar = quiet(_load_track, "Jarama", "Circuits")
ok("6 Jarama: no QA warning", not any(w.startswith("Track 'Jarama' QA") for w in WARNS))
ok("6 Jarama: k turns 1 lap (1e-9)", abs(np.trapezoid(t_jar.k, t_jar.s) / TWO_PI - 1) < 1e-9)
ok("6 Jarama: max |k| < 0.1 1/m", np.abs(t_jar.k).max() < 0.1)
ok("6 Jarama: 3908 samples", t_jar.s.size == 3908)
ok("6 Jarama: 3906 < s[-1] < 3908", 3906 < t_jar.s[-1] < 3908)

# 7. BCN: one QA warning, k left as in the file
t_bcn = quiet(_load_track, "BCN", "Circuits")
bcn_qa = [w for w in WARNS if w.startswith("Track 'BCN' QA")]
ok("7 BCN: exactly one QA warning", len(bcn_qa) == 1)
bcn_text = bcn_qa[0] if bcn_qa else ""
ok("7 BCN warning: k turns -1.0230 laps", "turns -1.0230 laps" in bcn_text)
ok("7 BCN warning: sign agreement 96.5%", "96.5%" in bcn_text)
ok("7 BCN warning: path gap 156.1 m", "156.1 m" in bcn_text)
bcn_file = sio.loadmat(os.path.join(CIRC, "Barcelona_circuit.mat"), squeeze_me=True)
ok("7 BCN k equals the file (array_equal)",
   np.array_equal(t_bcn.k, np.asarray(bcn_file["k"], float).reshape(-1)))

# 8. other registered circuits: no QA warning, sector sizes, raw BCN spike
SECTOR_N = {"BCN_S1": 1599, "BCN_S2": 1702, "BCN_S3": 1354}
for c in ("NBR", "Spa", "BCN_S1", "BCN_S2", "BCN_S3"):
    t = quiet(_load_track, c, "Circuits")
    ok(f"8 {c}: no QA warning", not any("QA" in w for w in WARNS))
    if c in SECTOR_N:
        ok(f"8 {c}: {SECTOR_N[c]} samples", t.s.size == SECTOR_N[c])
raw = sio.loadmat(os.path.join(CIRC, "Barcelona_circuit_raw.mat"), squeeze_me=True)
raw_track = SimpleNamespace(**{n: np.asarray(raw[n], float).reshape(-1) for n in ("s", "k", "x", "y")})
ok("8 raw BCN (S3 spike source): only the |k| spike",
   check_track(raw_track)[1] == [("warn", "|k| 0.250 1/m (R 4.0 m) at s = 4433.2 m")])

# 9. error paths, in a temp dir removed at exit
with tempfile.TemporaryDirectory(prefix="track_qa_") as tmp:
    d_abs = os.path.join(tmp, "abs_k")
    os.mkdir(d_abs)
    nbr = sio.loadmat(os.path.join(CIRC, "Nurburgring_circuit.mat"), squeeze_me=True)
    sio.savemat(os.path.join(d_abs, "Nurburgring_circuit.mat"),
                {"s": nbr["s"], "x": nbr["x"], "y": nbr["y"], "k": np.abs(np.asarray(nbr["k"], float))})
    kind, err = "", ""
    try:
        quiet(_load_track, "NBR", d_abs)
    except Exception as e:
        kind, err = type(e).__name__, str(e)
    ok("9 NBR with k = |k|: raises ValueError", kind == "ValueError")
    ok("9 NBR with k = |k|: message names the sign check", "sign of k" in err)
    ok("9 NBR with k = |k|: message points to k_from_xy", "k_from_xy" in err)

    d_sec = os.path.join(tmp, "no_s")
    os.mkdir(d_sec)
    shutil.copyfile(os.path.join(CIRC, "Barcelona_circuit_fromassetto.mat"),
                    os.path.join(d_sec, "Nurburgring_circuit.mat"))
    kind, msg = "", ""
    try:
        quiet(_load_track, "NBR", d_sec)
    except Exception as e:
        kind, msg = type(e).__name__, str(e)
    ok("9 NBR file with S_sector / k_sector only: ValueError, not AttributeError", kind == "ValueError")
    ok("9 NBR file with S_sector / k_sector only: message says has no s", "has no s" in msg)

# 10. BCNAssetto is out of the registry
err = None
try:
    quiet(userOpts, Ctx(), circuit="BCNAssetto")
except ValueError as e:
    err = str(e)
ok("10 BCNAssetto: 'Unknown circuit' error", err == "Unknown circuit 'BCNAssetto'.")

# 11. QSS lap pins on the rebuilt tracks (numpy only, no solve)
DATA_AA = os.path.exists(os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat"))
if not DATA_AA:
    print("  [SKIP] 11 QSS pins: Data/DATA_AA.mat missing")
else:
    for c, want in (("Jarama", 119.836), ("BCN_S1", 35.242), ("BCN_S2", 49.400), ("BCN_S3", 50.078)):
        lap = quiet(MLTP_screen, circuit=c, save=False, verbose=False).data.lap_time
        ok(f"11 QSS {c}: lap {lap:.3f} s within 0.1% of {want:.3f} s", abs(lap - want) <= 1e-3 * want)

# 12. GPS projection and GPS circles: WGS84 tangent plane, east +x, north +y
LAT0, LON0 = 41.57, 2.2611                       # Barcelona, as in the other GPS checks
WGS84_A, WGS84_E2 = 6378137.0, 0.00669437999014  # the test's own constants: the ECEF oracle below
PHI0 = np.deg2rad(LAT0)
W0 = np.sqrt(1.0 - WGS84_E2 * np.sin(PHI0) ** 2)
N0 = WGS84_A / W0                                # prime-vertical radius at LAT0
M0 = N0 * (1.0 - WGS84_E2) / W0 ** 2             # meridian radius at LAT0


def ecef(lat, lon):
    """WGS84 ECEF position [m], 3 x N, of lat, lon [deg]."""
    a, b = np.deg2rad(lat), np.deg2rad(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(a) ** 2)
    return np.stack([n * np.cos(a) * np.cos(b), n * np.cos(a) * np.sin(b), n * (1.0 - WGS84_E2) * np.sin(a)])


def gps_lap(sign=+1, hold=1, extra=0, drop=0.0, sigma=0.0, seed=0, radius=R):
    """GPS fixes [deg] of a circle of radius [m], fixes 3..5 m apart, clockwise for sign -1.
    hold repeats each fix, extra repeats the first fixes at the end, drop cuts the lap
    short (share of the lap), sigma adds white noise [m] east and north."""
    g = np.random.default_rng(seed)
    t = [0.0]
    while t[-1] < TWO_PI * radius * (1.0 - drop) - 5.0:
        t.append(t[-1] + g.uniform(3.0, 5.0))
    t = np.array(t) / radius
    x, y = radius * np.cos(t), sign * radius * np.sin(t)
    if extra:
        x, y = np.append(x, x[:extra]), np.append(y, y[:extra])
    x, y = np.repeat(x, hold), np.repeat(y, hold)
    if sigma:
        x, y = x + sigma * g.standard_normal(x.size), y + sigma * g.standard_normal(x.size)
    return LAT0 + np.rad2deg(y / M0), LON0 + np.rad2deg(x / (N0 * np.cos(PHI0)))


def build(lat, lon, **kw):
    """write_track on GPS fixes in a temp dir. Returns (metrics, issues, dict of s, k, x, y read back)."""
    with tempfile.TemporaryDirectory(prefix="track_qa_") as tmp:
        p = os.path.join(tmp, "gps.mat")
        met, iss = write_track(p, *xy_from_gps(lat, lon), **kw)
        r = sio.loadmat(p, squeeze_me=True)
    return met, iss, {n: np.asarray(r[n], float) for n in "skxy"}


x, y = xy_from_gps([LAT0, LAT0, LAT0 + 0.001], [LON0, LON0 + 0.001, LON0])
ok("12 projection: 0.001 deg east = 83.41 m (+-0.05)", abs(x[1] - x[0] - 83.41) < 0.05)
ok("12 projection: 0.001 deg north = 111.06 m (+-0.05)", abs(y[2] - y[0] - 111.06) < 0.05)
ok("12 projection: east is +x, north is +y",
   x[1] > x[0] and y[2] > y[0] and abs(y[1] - y[0]) < 1e-9 and abs(x[2] - x[0]) < 1e-9)
rng = np.random.default_rng(1)
la, lo = LAT0 + rng.uniform(-0.007, 0.007, 300), LON0 + rng.uniform(-0.01, 0.01, 300)
xg, yg = xy_from_gps(la, lo)
i, j = rng.integers(0, 300, 1000), rng.integers(0, 300, 1000)
pairs = i != j
E = ecef(la, lo)
chord = np.linalg.norm(E[:, i[pairs]] - E[:, j[pairs]], axis=0)
rel = np.abs(np.hypot(xg[i[pairs]] - xg[j[pairs]], yg[i[pairs]] - yg[j[pairs]]) - chord) / chord
ok(f"12 projection: pair distances match ECEF chords in a box of about 1.6 km (max rel {rel.max():.1e} < 2e-4)",
   rel.max() < 2e-4)
for sign, name in ((+1, "counter-clockwise"), (-1, "clockwise")):
    met, _, t = build(*gps_lap(sign))
    ok(f"12 GPS circle {name}: turning {sign:+d} lap (1e-9)", abs(met["turn_k"] - sign) < 1e-9)
    ok(f"12 GPS circle {name}: length within 1e-3 of 2 pi R", abs(t["s"][-1] / (TWO_PI * R) - 1) < 1e-3)
    ok(f"12 GPS circle {name}: median k R within 2e-3 of {sign:+d}", abs(np.median(t["k"]) * R - sign) < 2e-3)

# 13. closure guards: overlap and repeated fixes give one lap. A gap or a figure of eight is refused.
# A 55% gap leaves no fix behind the start: the overshoot trim then removes all but 3 fixes.
# closed=False writes an open arc, also with smoothing.
met, _, _ = build(*gps_lap(+1, extra=3))
ok("13 overlap: last 3 fixes repeat the first 3: turning +1 (not +2)", abs(met["turn_k"] - 1) < 1e-9)
met, _, _ = build(*gps_lap(+1, hold=5))
ok("13 duplicates: every fix repeated 5 times: turning +1", abs(met["turn_k"] - 1) < 1e-9)
for drop in (0.3, 0.55):
    try:
        build(*gps_lap(+1, drop=drop))
        ok(f"13 {drop:.0%} gap: ValueError raised", False)
    except ValueError as e:
        ok(f"13 {drop:.0%} gap: ValueError 'not a closed lap'", "not a closed lap" in str(e))
with tempfile.TemporaryDirectory(prefix="track_qa_") as tmp:
    met, iss = write_track(os.path.join(tmp, "arc.mat"), R * np.cos(tq), R * np.sin(tq), closed=False, smooth=20.0)
ok("13 open quarter arc, closed=False, smooth 20 m: turning +0.25 lap (1e-4), no QA issue",
   abs(met["turn_k"] - 0.25) < 1e-4 and iss == [])
try:
    with tempfile.TemporaryDirectory(prefix="track_qa_") as tmp:
        t8 = np.arange(0.0, TWO_PI, 0.04)
        write_track(os.path.join(tmp, "eight.mat"), 100 * np.sin(t8), 50 * np.sin(2 * t8))
    ok("13 figure of eight: ValueError raised", False)
except ValueError as e:
    ok("13 figure of eight: ValueError 'crosses itself'", "crosses itself" in str(e))

# 14. BCN rebuilt from its own x,y: turning, QA, closure and the QSS lap (DATA_AA guard).
# x, y stay N x 1 columns, as loadmat gives them. Window 301 smooths k so far that the QA sign check fails.
r_bcn = sio.loadmat(os.path.join(CIRC, "Barcelona_circuit.mat"))
with tempfile.TemporaryDirectory(prefix="track_qa_") as tmp:
    met, iss = write_track(os.path.join(tmp, "Barcelona_circuit.mat"), r_bcn["x"], r_bcn["y"], True)
    ok(f"14 BCN rebuilt: turning -1 (1e-9): {met['turn_k']:.12f}", abs(met["turn_k"] + 1) < 1e-9)
    ok("14 BCN rebuilt: no QA issue", iss == [])
    ok(f"14 BCN rebuilt: path from k ends {met['gap_k']:.2f} m from its start (< 1 m)", met["gap_k"] < 1.0)
    if DATA_AA:
        lap = quiet(MLTP_screen, circuit="BCN", circuits_dir=tmp, save=False, verbose=False).data.lap_time
        ok(f"14 BCN rebuilt: QSS lap {lap:.3f} s within 0.1% of 134.768 s", abs(lap - 134.768) <= 1e-3 * 134.768)
    else:
        print("  [SKIP] 14 BCN rebuilt QSS lap: Data/DATA_AA.mat missing")
    p_bad = os.path.join(tmp, "window301.mat")
    try:
        write_track(p_bad, r_bcn["x"], r_bcn["y"], window=301)
        ok("14 BCN with window 301: ValueError raised", False)
    except ValueError as e:
        ok("14 BCN with window 301: ValueError 'track QA', no file written",
           "track QA" in str(e) and not os.path.exists(p_bad))

# 15. noisy GPS circles (0.05 m white noise) with 40 m smoothing. R = 20 m catches a smoothing
# length that is too long (160 m), R = 100 m does not.
for radius in (100.0, 20.0):
    met, iss, t = build(*gps_lap(+1, sigma=0.05, seed=2, radius=radius), smooth=40.0)
    kR = np.median(t["k"]) * radius
    ok(f"15 noisy circle R {radius:g} m, smooth 40 m: median k R within 3% of +1 ({kR:.4f})", abs(kR - 1) < 0.03)

print("\nALL track QA TESTS PASSED" + ("" if DATA_AA else " (sections 11 and 14 skipped)"))
