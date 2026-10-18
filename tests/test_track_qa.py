"""Track QA gate and rebuilt circuit data (Phase 2 plan, Task 2).

Covers functions/track_qa.py (check_track, k_from_xy), the QA hook in
userOpts._load_track, the rebuilt Jarama and BCN sector files, the removal of
BCNAssetto and the QSS lap pins. No NLP solve. Section 11 needs
Data/DATA_AA.mat and prints SKIP without it. Run from any directory:

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
from functions.track_qa import check_track, k_from_xy
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

print("\nALL track QA TESTS PASSED" + ("" if DATA_AA else " (section 11 skipped)"))
