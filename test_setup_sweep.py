"""setup_sweep.py, functions/sweep.py and MLTP_screen.screen_batch (plain script, no pytest).
Run from the repo root:

    venv\\Scripts\\python.exe test_setup_sweep.py

  1. functions.sweep units: validate_specs, sample_box, select_shortlist, bridge_points,
     rank_metrics on synthetic data, noise floor / unresolved pairs, fingerprint, writers
  2. screen_batch == screen_sweep, exact ==: 16 seeded Sobol setups over 10 fields in 4
     configs on Sturn (+ BCN when its .mat exists), the MLTP_screen baseline, an invalid row
     isolated; workers=2 with chunk 1 / 7 in a `python -c` child (Windows spawn re-imports
     the main script in every worker, so pools never start from this file)
  3. casadi blocked: this file re-runs itself in a child with sys.modules['casadi'] = None
     (sections 1-2 in-process, the imports of functions.sweep / MLTP_screen / setup_sweep,
     4 rows screened, a confirm=False sweep), then stops before section 4
  4. field classification (casadi, no solve): nlp_signature determinism, NLP-inert / live
     per config, scale_changed, promotion_safe, QSS-blind, an all-dropped call
  5. NLP mini-sweep (Sturn, ~1 min, in a `python -c` child): brkB dropped, the hub check,
     the baseline delta, the accepted-row fields, files and schemas, a resume that runs no
     task, a second run from hub.mat (1 worker, HSL not pinned) with identical laps and
     w_sha, and the SweepErrors (another OPT_ds with that base, another seed, same name)

No NLP lap time or iteration count is pinned: only properties (warm-start modes, statuses,
0 iterations on the hub check, equality between runs, file schemas).
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)                                     # userOpts reads Circuits/ and Data/ relatively
NO_CASADI = os.environ.get("TEST_SETUP_SWEEP_NO_CASADI") == "1"
if NO_CASADI:
    sys.modules["casadi"] = None                   # any 'import casadi' now raises

import functions.sweep as S
from MLTP_screen import MLTP_screen, screen_sweep, screen_batch


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def raises(exc, fn, *args, **kw):
    """The message of the ``exc`` that fn(...) raises ('raised' if empty), else None."""
    try:
        fn(*args, **kw)
    except exc as e:
        return str(e) or "raised"
    return None


def caught(fn, *args, **kw):
    """(result, [warning texts]) of fn(...)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec]


TMP_ROOT = os.environ.get("CLAUDE_JOB_DIR_TMP") or tempfile.gettempdir()
os.makedirs(TMP_ROOT, exist_ok=True)
TMP = tempfile.mkdtemp(prefix="sweep_test_", dir=TMP_ROOT)
PY = sys.executable

# =============================================================================
print("1. functions.sweep units" + (" (casadi blocked)" if NO_CASADI else ""))
specs, dropped = S.validate_specs([("alpha_RW", 4, 16), ("pDy1", 0.97, 1.09), ("mb", 1700, 1900.0)])
ok("valid specs kept in order as floats, a Pacejka key is sweepable",
   specs == [("alpha_RW", 4.0, 16.0), ("pDy1", 0.97, 1.09), ("mb", 1700.0, 1900.0)] and dropped == {}
   and all(isinstance(v, float) for s in specs for v in s[1:]))
(sp, dr), w = caught(S.validate_specs, [("alpha_RW", 4, 16), ("foo", 0, 1), ("ms", 1, 2)])
ok("unknown 'foo' and derived 'ms' warned (UserWarning) and dropped",
   [s[0] for s in sp] == ["alpha_RW"] and set(dr) == {"foo", "ms"}
   and sum("foo" in x or "'ms'" in x for x in w) == 2)
BAD_SPECS = [([("mb", 1900, 1700)], "lower > upper"), ([("mb", 1800, 1800)], "lower == upper"),
             ([("mb", float("nan"), 1900)], "NaN bound"), ([("mb", 1700, float("inf"))], "inf bound"),
             ([("mb", 1700, 1900), ("mb", 1750, 1850)], "duplicate field"),
             ([(3, 0, 1)], "non-str field"), ([("mb", 1700)], "not a triple"),
             (["mb"], "a bare string item"), ([("mb", "a", "b")], "non-numeric bounds")]
for bad, why in BAD_SPECS:
    ok(f"ValueError: {why}", raises(ValueError, S.validate_specs, bad) is not None)

sp3 = [("alpha_RW", 4.0, 16.0), ("hcg", 0.45, 0.55), ("mb", 1730.0, 1910.0)]
U1, X1 = S.sample_box(sp3, 64, seed=0)
U2, X2 = S.sample_box(sp3, 64, seed=0)
U3, X3 = S.sample_box(sp3, 64, seed=1)
lo, hi = np.array([s[1] for s in sp3]), np.array([s[2] for s in sp3])
ok("sample_box: same seed -> identical design, another seed -> another design",
   np.array_equal(U1, U2) and np.array_equal(X1, X2) and not np.array_equal(X1, X3))
ok("sample_box: 64 x 3, U in [0, 1), X = lower + U (upper - lower) inside the box",
   X1.shape == (64, 3) and U1.min() >= 0 and U1.max() < 1 and np.all(X1 >= lo) and np.all(X1 <= hi)
   and np.allclose(X1, lo + U1 * (hi - lo), rtol=0, atol=1e-12))
(U4, X4), w = caught(S.sample_box, sp3, 100, seed=0)
ok("n=100 (not 2**m): one balance warning, exactly 100 samples inside the box",
   X4.shape == (100, 3) and np.all(X4 >= lo) and np.all(X4 <= hi)
   and sum("power of 2" in x for x in w) == 1 and len(w) == 1)
U5, X5 = S.sample_box(sp3, 50, sampler="lhs", seed=0)
ok("lhs: every column stratified (one point per 1/n bin)",
   all(sorted(np.floor(U5[:, j] * 50).astype(int)) == list(range(50)) for j in range(3)))
ok("unknown sampler / n=0 raise ValueError",
   raises(ValueError, S.sample_box, sp3, 8, sampler="grid") is not None
   and raises(ValueError, S.sample_box, sp3, 0) is not None)

q = np.array([10.0, 3.0, 1.0, 2.0, 2.0, np.nan, 5.0, 4.0, 6.0, 7.0, 8.0])
okm = np.array([True] * 11)
okm[9] = False                                    # an invalid row (finite lap) is never selected
sl = S.select_shortlist(q, okm, top_k=3, n_probes=2)
ok("shortlist: baseline first, top by (lap, row_id) with ties in row order",
   sl[:4] == [(0, "baseline"), (2, "top"), (3, "top"), (4, "top")])
rest = [1, 7, 6, 8, 10]                            # remaining valid rows by QSS lap (NaN / invalid out)
pos = [math.floor(j / 2 * (len(rest) - 1) + 0.5) for j in (1, 2)]
ok(f"probes at QSS-rank quantiles {pos} of the remainder {rest}",
   sl[4:] == [(rest[p], "probe") for p in pos] == [(6, "probe"), (10, "probe")])
sl2 = S.select_shortlist(q, okm, top_k=6, n_probes=5)
ok("5 probes asked of a 2-row remainder [8, 10]: positions 0,0,1,1,1 de-duplicated",
   [k for _, k in sl2].count("top") == 6 and [r for r, k in sl2 if k == "probe"] == [8, 10])
ok("top_k=0, n_probes=0 -> baseline only", S.select_shortlist(q, okm, 0, 0) == [(0, "baseline")])

b, t = {"alpha_RW": 8.0, "hcg": 0.5}, {"alpha_RW": 10.123, "hcg": 0.4567}
pts = S.bridge_points(b, t, 4)
ok("bridge_points: k points, the last is the target exactly, the others on the segment",
   len(pts) == 4 and pts[-1] == t and all(isinstance(v, float) for v in pts[-1].values())
   and all(abs(pts[j]["alpha_RW"] - (8.0 + (j + 1) / 4 * 2.123)) < 1e-12 for j in range(3)))
ok("bridge_points k=1 -> [target]; k=0 raises", S.bridge_points(b, t, 1) == [t]
   and raises(ValueError, S.bridge_points, b, t, 0) is not None)

from scipy.stats import spearmanr, kendalltau
rs = np.random.RandomState(3)
Q, Y = rs.rand(40, 9), rs.rand(40, 9)
Y[:, 3] = Y[:, 4]                                  # ties
Q[:, 1] = Q[:, 2]
vr, vt = S.spearman_rows(Q, Y), S.kendall_b_rows(Q, Y)
ok("vectorised Spearman / Kendall tau-b == scipy (ties included)",
   np.allclose(vr, [spearmanr(a, c).statistic for a, c in zip(Q, Y)], atol=1e-12)
   and np.allclose(vt, [kendalltau(a, c).statistic for a, c in zip(Q, Y)], atol=1e-12))


def mrows(qd, yd, kinds=None, branch=None):
    kinds = kinds or (["baseline"] + ["top"] * 3 + ["probe"] * (len(qd) - 4))
    return [dict(row_id=i, kind=kinds[i], qss_lap=18.7 + qd[i], nlp_lap=18.0 + yd[i],
                 branch_ok=True if branch is None else branch[i]) for i in range(len(qd))]


qd = [0.0, -0.009, -0.007, -0.004, 0.002, 0.005, 0.008, 0.011]
m = S.rank_metrics(mrows(qd, [0.8 * x for x in qd]), 1e-4, top_ids=[1, 2, 3], seed=0)
ok("rank_metrics, monotone data: rho = tau = 1, concordance 1, regret 0, trusted",
   m["n"] == 8 and abs(m["all"]["spearman_rho"] - 1) < 1e-12 and abs(m["all"]["kendall_tau_b"] - 1) < 1e-12
   and m["concordance"]["value"] == 1.0 and m["top1_regret_s"] == 0.0 and m["screen_trusted"]
   and m["nlp_best_in_qss_top_k"])
ok("slope / intercept of NLP delta on QSS delta, sign agreement 1",
   abs(m["slope"] - 0.8) < 1e-9 and abs(m["intercept_s"]) < 1e-12 and m["sign_agreement"]["value"] == 1.0)
ok("bootstrap CIs (n >= 6): seeded, two ordered numbers inside [-1, 1]",
   m["all"]["spearman_ci95"] is not None and -1 <= m["all"]["spearman_ci95"][0] <= m["all"]["spearman_ci95"][1] <= 1
   and S.rank_metrics(mrows(qd, [0.8 * x for x in qd]), 1e-4, [1, 2, 3], seed=0)["all"] == m["all"])
m = S.rank_metrics(mrows(qd, [-x for x in qd]), 1e-4, top_ids=[1, 2, 3])
ok("anti-monotone data: rho = tau = -1, concordance 0, regret > 0, not trusted",
   abs(m["all"]["spearman_rho"] + 1) < 1e-12 and abs(m["all"]["kendall_tau_b"] + 1) < 1e-12
   and m["concordance"]["value"] == 0.0 and m["top1_regret_s"] > 0 and not m["screen_trusted"]
   and m["qss_best_row"] == 1 and m["nlp_best_row"] == 7 and not m["nlp_best_in_qss_top_k"])
yd = [0.0, -0.009, -0.009, -0.004, 0.002, 0.002, 0.008, 0.011]
m = S.rank_metrics(mrows(qd, yd), 1e-4, top_ids=[1, 2, 3])
ok("tau-b with NLP ties == scipy kendalltau",
   abs(m["all"]["kendall_tau_b"] - kendalltau(qd, yd).statistic) < 1e-12)
ok("pairs within the floor are not resolved (2 tied pairs of 28 dropped)",
   m["concordance"]["n_pairs"] == 26 and m["concordance"]["value"] == 1.0)
m = S.rank_metrics(mrows(qd, [0.8 * x for x in qd]), 0.0025, top_ids=[1, 2, 3])
ok("a larger floor drops more pairs and sign checks",
   m["concordance"]["n_pairs"] < 28 and m["sign_agreement"]["n"] < 7)
m = S.rank_metrics(mrows(qd[:5], qd[:5], kinds=["baseline", "top", "top", "top", "probe"]), 1e-4, [1, 2, 3])
ok("n < 6: no bootstrap CI", m["all"]["spearman_ci95"] is None and m["all"]["kendall_ci95"] is None)
br = [True, True, False, True, True, None, True, True]
m = S.rank_metrics(mrows(qd, [0.8 * x for x in qd], branch=br), 1e-4, [1, 2, 3])
ok("branch_ok block uses only branch_ok rows; within_shortlist = baseline + top",
   m["branch_ok"]["n"] == 6 and m["within_shortlist"]["n"] == 4
   and m["within_shortlist"]["label"] == "within shortlist (range restricted)")
rows_nf = [dict(branch_ok=True, rt_residual_s=-3e-4), dict(branch_ok=True, rt_residual_s=1e-5),
           dict(branch_ok=False, rt_residual_s=0.02), dict(branch_ok=True, rt_residual_s=None)]
ok("noise_floor = max(resolve_s, |r| over branch_ok rows)",
   S.noise_floor(rows_nf, 1e-4) == 3e-4 and S.noise_floor(rows_nf[1:2], 1e-4) == 1e-4)
up = S.unresolved_pairs({0: 18.0, 1: 18.00005, 2: 18.00012, 5: float("nan")}, 1e-4)
ok("unresolved_pairs within the floor (not transitive), nan laps left out",
   up == {0: [1], 1: [0, 2], 2: [1]})

f1 = S.fingerprint({"b": [1, 2.5], "a": {"y": np.float64(0.1), "x": (1, 2)}})
f2 = S.fingerprint({"a": {"x": [1, 2], "y": 0.1}, "b": [np.int64(1), 2.5]})
ok("fingerprint: key order / tuple / numpy scalar invariant, value sensitive",
   f1 == f2 and len(f1) == 64 and S.fingerprint({"a": 1}) != S.fingerprint({"a": 2}))
jp = os.path.join(TMP, "w", "x.json")
S.write_json_atomic(jp, {"a": float("nan"), "b": 0.1 + 0.2, "c": np.float64(1.5), "d": (1, 2)})
back = S.read_json(jp)
ok("write_json_atomic: strict JSON (nan -> null), exact floats, no temp file left",
   back == {"a": None, "b": 0.1 + 0.2, "c": 1.5, "d": [1, 2]} and os.listdir(os.path.dirname(jp)) == ["x.json"])
cp = os.path.join(TMP, "w", "x.csv")
S.write_csv_atomic(cp, ["i", "v", "flag", "lst", "none"],
                   [dict(i=1, v=0.1 + 0.2, flag=True, lst=[3, 4], none=None), dict(i=2, v=float("nan"))])
rows = S.read_csv(cp)
ok("write_csv_atomic: repr floats round-trip, nan / None empty, bools, ';' lists",
   float(rows[0]["v"]) == 0.1 + 0.2 and rows[0]["flag"] == "True" and rows[0]["lst"] == "3;4"
   and rows[0]["none"] == "" and rows[1]["v"] == "" and math.isnan(S.num(rows[1]["v"])))
ok("auto_workers in [1, n_tasks]", S.auto_workers(1) == 1 and 1 <= S.auto_workers(64, rss_mb=260) <= 64)
with S.blas_single_thread():
    inside = [os.environ.get(k) for k in S.BLAS_VARS]
ok("blas_single_thread sets the BLAS variables to '1' inside the block only",
   inside == ["1", "1", "1"])

# =============================================================================
print("2. screen_batch == screen_sweep (exact ==)")
F10 = [("mb", 1638.0, 2002.0), ("alpha_RW", 4.0, 12.0), ("hcg", 0.45, 0.55),
       ("Tbrake_max", 3000.0, 5000.0), ("Rw", 0.33, 0.38), ("k_fl", 60000.0, 90000.0),
       ("pDy1", 0.97, 1.09), ("gamma_fl", -3.0, 0.0), ("brkB", 0.55, 0.75), ("Tdist", 0.6, 0.85)]
_, X16 = S.sample_box(F10, 16, seed=7)
OVS = [{f: float(v) for (f, _, _), v in zip(F10, x)} for x in X16]
CONFIGS = {"default": {}, "ATD Off": dict(ATD="Off"), "EM4 On": dict(ATD="Off", Electric_4Motors="On"),
           "AALB": dict(AeroConfig="AALB")}
ref = {}
for label, cfg in CONFIGS.items():
    a = [r["lap_time"] for r in screen_sweep("Sturn", OVS, **cfg)]
    bb = screen_batch("Sturn", OVS, **cfg)
    ref[label] = bb["lap_time"]
    ok(f"Sturn {label}: 16 laps == screen_sweep, all 'ok'",
       list(bb["lap_time"]) == a and bb["status"] == ["ok"] * 16 and bb["error"] == [None] * 16)
HAVE_BCN = os.path.exists(os.path.join(HERE, "Circuits", "Barcelona_circuit.mat"))
if HAVE_BCN:
    a = [r["lap_time"] for r in screen_sweep("BCN", OVS)]
    ok("BCN default: 16 laps == screen_sweep", list(screen_batch("BCN", OVS)["lap_time"]) == a)
else:
    print("  [SKIP] Circuits/Barcelona_circuit.mat missing")
lap0 = screen_batch("Sturn", [{}])["lap_time"][0]
ok("baseline row == MLTP_screen(save=False).data.lap_time",
   lap0 == MLTP_screen("Sturn", save=False, verbose=False).data.lap_time)
base_ov = {"mb": 1850.0}
bb = screen_batch("Sturn", OVS[:3] + [{"alpha_RW": 9.0}], vp_overrides=base_ov)
a = [r["lap_time"] for r in screen_sweep("Sturn", OVS[:3] + [{"alpha_RW": 9.0}], vp_overrides=base_ov)]
ok("a base vp_overrides is merged under every row, as in screen_sweep", list(bb["lap_time"]) == a)
bad = OVS[:4] + [{"mb": -100.0}] + OVS[4:6]
bb = screen_batch("Sturn", bad)
ok("mb=-100 is 'invalid' (nan lap, error text), the other rows unchanged",
   bb["status"][4] == "invalid" and math.isnan(bb["lap_time"][4]) and "floor" in bb["error"][4]
   and list(bb["lap_time"][:4]) == list(ref["default"][:4])
   and list(bb["lap_time"][5:]) == list(ref["default"][4:6])
   and bb["status"].count("ok") == 6 and not bb["vi_feasible"][4])
bb = screen_batch("Sturn", [{"Rw": 0.0}, {}])
ok("a row that raises is 'invalid' with the exception text", bb["status"] == ["invalid", "ok"]
   and bb["error"][0].startswith("ZeroDivisionError") and bb["lap_time"][1] == lap0)
ok("workers / chunk validation", raises(ValueError, screen_batch, "Sturn", [{}], workers=0) is not None
   and raises(ValueError, screen_batch, "Sturn", [{}], chunk=0) is not None)

if not NO_CASADI:
    code = r"""
import json, sys
from functions.sweep import sample_box
from MLTP_screen import screen_batch
F10 = json.loads(sys.argv[1])
_, X16 = sample_box([tuple(s) for s in F10], 16, seed=7)
ovs = [{f: float(v) for (f, _, _), v in zip(F10, x)} for x in X16]
out = {}
for chunk in (1, 7):
    r = screen_batch("Sturn", ovs + [{"mb": -100.0}], workers=2, chunk=chunk)
    out[str(chunk)] = [list(map(float, r["lap_time"][:16])), r["status"]]
print("RESULT " + json.dumps(out))
"""
    r = subprocess.run([PY, "-c", code, json.dumps(F10)], cwd=HERE, capture_output=True, text=True,
                       timeout=600)
    res = json.loads(r.stdout.split("RESULT ", 1)[1]) if "RESULT " in r.stdout else None
    if res is None:
        print(r.stdout[-2000:], r.stderr[-3000:])
    ok("workers=2, chunk 1 and 7 (spawn pool, `python -c` child): laps == in-process, bitwise",
       res is not None and all(res[c][0] == list(ref["default"]) for c in ("1", "7"))
       and all(res[c][1] == ["ok"] * 16 + ["invalid"] for c in ("1", "7")))

# =============================================================================
if NO_CASADI:
    print("3. casadi blocked: setup_sweep imports, QSS-only sweep")
    import setup_sweep as SS
    ok("functions.sweep, MLTP_screen and setup_sweep import without casadi",
       sys.modules["casadi"] is None and hasattr(SS, "setup_sweep") and SS._casadi_version() is None)
    four = screen_batch("Sturn", OVS[:4])
    print("FOUR " + json.dumps(list(map(float, four["lap_time"]))))
    ok("4 rows screened", four["status"] == ["ok"] * 4)
    ok("confirm=True raises SweepError without casadi",
       raises(SS.SweepError, SS.setup_sweep, [("alpha_RW", 4, 16)], 4, results_root=TMP,
              name="noca", verbose=False) is not None)
    rq, w = caught(SS.setup_sweep, [("alpha_RW", 4, 16), ("mb", 1730, 1910)], 8, confirm=False,
                   top_k=2, n_probes=1, results_root=TMP, name="qss_only", verbose=False)
    ok("confirm=False sweep: QSS-only shortlist and best, warned about the skipped NLP probe",
       rq.n_tasks_run == 0 and rq.shortlist[0] == (0, "baseline") and len(rq.shortlist) == 4
       and rq.best["nlp_lap_s"] is None and any("not checked for NLP inertness" in x for x in w)
       and os.path.isfile(os.path.join(TMP, "qss_only", "samples.csv")))
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nCASADI-FREE SECTIONS PASSED")
    sys.exit(0)

print("3. casadi blocked (this file re-run in a child with sys.modules['casadi'] = None)")
r = subprocess.run([PY, os.path.abspath(__file__)], cwd=HERE, capture_output=True, text=True,
                   timeout=900, env=dict(os.environ, TEST_SETUP_SWEEP_NO_CASADI="1"))
four = (json.loads(r.stdout.split("FOUR ", 1)[1].splitlines()[0]) if "FOUR " in r.stdout else None)
if r.returncode != 0:
    print(r.stdout[-3000:], r.stderr[-3000:])
ok("sections 1-3 pass with casadi blocked (child exit 0)",
   r.returncode == 0 and "CASADI-FREE SECTIONS PASSED" in r.stdout)
ok("the 4 rows screened without casadi == this process", four == list(map(float, ref["default"][:4])))

# =============================================================================
print("4. field classification (casadi, no solve)")
from functions.context import Ctx
from userOpts import userOpts
import setup_sweep as SS
c1, c2 = userOpts(Ctx(), circuit="Sturn"), userOpts(Ctx(), circuit="Sturn")
s1, s2 = SS.nlp_signature(c1), SS.nlp_signature(c2)
ok("nlp_signature deterministic across builds (graph, scale)", s1 == s2 and len(s1[0]) == len(s1[1]) == 64)
FS = [("brkB", 0.6, 0.7), ("Tdist", 0.6, 0.8), ("Cd", 0.7, 0.8), ("mb", 1700, 1900),
      ("alpha_RW", 4, 12), ("Tbrake_max", 3000, 5000), ("Rw", 0.33, 0.37), ("zeta_fl", 0.6, 0.8),
      ("hcg", 0.45, 0.55), ("gamma_fl", -2.0, 0.0)]
live = {}
for label, cfg in (("default", {}), ("ATD Off", dict(ATD="Off")),
                   ("EM4 On", dict(ATD="Off", Electric_4Motors="On"))):
    fl = SS.classify_fields(FS, "Sturn", **cfg)
    live[label] = {f: fl[f]["nlp_live"] for f in fl}
    if label == "default":
        FL = fl
ok("default (ATD On): brkB, Tdist, Cd NLP-inert; mb, alpha_RW live",
   [live["default"][f] for f in ("brkB", "Tdist", "Cd", "mb", "alpha_RW")] == [False, False, False, True, True])
ok("ATD Off: brkB, Tdist live; Cd inert",
   [live["ATD Off"][f] for f in ("brkB", "Tdist", "Cd")] == [True, True, False])
ok("EM4 On: Tdist inert, brkB live, Cd inert",
   [live["EM4 On"][f] for f in ("brkB", "Tdist", "Cd")] == [True, False, False])
ok("scale_changed for Tbrake_max and Rw only",
   {f for f in FL if FL[f]["scale_changed"]} == {"Tbrake_max", "Rw"})
ok("promotion_safe: alpha_RW yes; mb, hcg, gamma_fl, zeta_fl, Tbrake_max no",
   FL["alpha_RW"]["promotion_safe"] and not any(FL[f]["promotion_safe"] for f in
                                                ("mb", "hcg", "gamma_fl", "zeta_fl", "Tbrake_max")))
ok("phase2_free: alpha_RW, Tbrake_max yes; mb, Rw, zeta_fl no",
   FL["alpha_RW"]["phase2_free"] and FL["Tbrake_max"]["phase2_free"]
   and not any(FL[f]["phase2_free"] for f in ("mb", "Rw", "zeta_fl")))
ok("zeta_fl is QSS-blind (one QSS lap at base / lower / upper), mb is not",
   FL["zeta_fl"]["qss_blind"] and not FL["mb"]["qss_blind"] and FL["brkB"]["qss_blind"] is None)
rb, w = caught(SS.setup_sweep, [("brkB", 0.6, 0.7), ("Cd", 0.7, 0.8), ("foo", 0, 1)], 8,
               results_root=TMP, name="all_dropped", verbose=False)
ok("all fields dropped: a baseline-only SweepResult, warnings, no NLP task",
   rb.n_tasks_run == 0 and rb.shortlist == [(0, "baseline")] and rb.confirmed == []
   and set(rb.dropped) == {"brkB", "Cd", "foo"} and rb.best["vp_overrides"] == {}
   and any("baseline-only" in x for x in w) and any("ignored by the 23-state NLP" in x for x in w)
   and sorted(os.listdir(os.path.join(TMP, "all_dropped"))) == ["plan.json", "summary.json"])

# =============================================================================
print("5. NLP mini-sweep (Sturn, `python -c` child; ~1 min)")
code = r"""
import json, os, sys, time, warnings, filecmp
from setup_sweep import setup_sweep, SweepError
from functions.warmstart import GOOD_STATUS
OUT = sys.argv[1]
specs = [("alpha_RW", 7, 9), ("hcg", 0.49, 0.51), ("brkB", 0.6, 0.7)]
kw = dict(n_samples=8, circuit="Sturn", results_root=OUT, top_k=2, n_probes=1, verbose=False)
res = {}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r1 = setup_sweep(specs, name="mini", workers=2, plot=True, **kw)
res["warn1"] = [str(x.message) for x in w]
res["r1"] = dict(dropped=r1.dropped, hub=r1.hub, confirmed=r1.confirmed, n_tasks_run=r1.n_tasks_run,
                 shortlist=r1.shortlist, files=r1.files, metrics_keys=sorted(r1.metrics),
                 best=r1.best, throughput=r1.throughput)
d1 = os.path.join(OUT, "mini")
csv1 = open(os.path.join(d1, "confirmed.csv"), "rb").read()
t = time.perf_counter()
r2 = setup_sweep(specs, name="mini", workers=2, plot=True, **kw)
res["resume"] = dict(n_tasks_run=r2.n_tasks_run, wall=time.perf_counter() - t,
                     csv_same=open(os.path.join(d1, "confirmed.csv"), "rb").read() == csv1)
hub = os.path.join(d1, "hub.mat")
r3 = setup_sweep(specs, name="mini_b", workers=1, pin_hsl=False, roundtrip="none", base=hub,
                 plot=False, **kw)
res["r3"] = dict(confirmed=r3.confirmed, hub=r3.hub, n_tasks_run=r3.n_tasks_run)
for label, extra in (("opt_ds", dict(name="mini_c", base=hub, OPT_ds=60)),
                     ("seed", dict(name="mini", seed=1))):
    t = time.perf_counter()
    try:
        setup_sweep(specs, **{**kw, **extra})
        res[label] = None
    except SweepError as e:
        res[label] = dict(msg=str(e), wall=time.perf_counter() - t,
                          folder=os.path.isdir(os.path.join(OUT, extra["name"])))
print("RESULT " + json.dumps(res))
"""
out5 = os.path.join(TMP, "nlp")
r = subprocess.run([PY, "-c", code, out5], cwd=HERE, capture_output=True, text=True, timeout=1500)
R = json.loads(r.stdout.split("RESULT ", 1)[1]) if "RESULT " in r.stdout else None
if R is None:
    print(r.stdout[-4000:], r.stderr[-4000:])
ok("the child ran the sweeps", R is not None)
from functions.warmstart import GOOD_STATUS
r1, d1 = R["r1"], os.path.join(out5, "mini")
ok("brkB dropped as NLP-inert (warned), alpha_RW and hcg kept",
   list(r1["dropped"]) == ["brkB"] and any("brkB is ignored" in x for x in R["warn1"]))
chk = r1["hub"]["check"]
ok("hub: solved cold in a worker, then checked full+duals, a good status, 0 iterations",
   r1["hub"]["source"] == "cold" and r1["hub"]["cold"]["status"] in GOOD_STATUS
   and chk["mode"] == "full+duals" and chk["status"] in GOOD_STATUS and chk["iters"] == 0
   and r1["hub"]["check_ok"] and not r1["hub"]["rehubbed"])
conf = {c["row_id"]: c for c in r1["confirmed"]}
ok("shortlist = baseline + 2 top + 1 probe, all confirmed", len(r1["shortlist"]) == 4
   and sorted(conf) == sorted(s[0] for s in r1["shortlist"]))
ok("baseline row: path 'hub', nlp_delta_s == 0.0 exactly, lap == hub lap",
   conf[0]["path"] == "hub" and conf[0]["nlp_delta_s"] == 0.0 and conf[0]["nlp_lap_s"] == r1["hub"]["lap_s"])
acc = [c for c in r1["confirmed"] if c["row_id"] != 0]
ok("accepted rows: full+duals, the hub's linear solver, a star / bridge path, a nlp_rank",
   all(c["status"] == "accepted" and c["warm_start_mode"] == "full+duals"
       and c["linear_solver"] == r1["hub"]["linear_solver"] and c["path"] in ("star", "bridge2", "bridge4")
       and c["nlp_rank"] for c in acc))
ok("roundtrip='all': every accepted row has rt_residual_s, delta_rev_s and branch_ok",
   all(c["rt_residual_s"] is not None and c["delta_rev_s"] is not None and c["branch_ok"] is not None
       for c in acc))
rec = {rid: json.load(open(os.path.join(d1, "rows", f"{rid}.json"))) for rid in conf if rid}
ok("rows/<id>.json: forward hop accepted with a good IPOPT status, round trip recorded, repro",
   all(x["ipopt_status"] in GOOD_STATUS and x["fingerprint"] == json.load(open(os.path.join(d1, "plan.json")))["fingerprint"]
       and x["rt_ok"] is not None and "from MLTP import MLTP" in x["repro"] and "warm_start" in x["repro"]
       and all(h["mode"] == "full+duals" for h in x["hops"]) for x in rec.values()))
ok("files: plan, samples, hub.mat/json, rows/<id>.mat|json, logs, confirmed, summary, report",
   all(os.path.isfile(os.path.join(d1, f)) for f in ("plan.json", "samples.csv", "hub.mat", "hub.json",
                                                      "confirmed.csv", "summary.json", "report.html",
                                                      "logs/hub_cold.log", "logs/hub_check.log"))
   and all(os.path.isfile(os.path.join(d1, "rows", f"{rid}.{e}")) for rid in rec for e in ("json", "mat")))
smp = S.read_csv(os.path.join(d1, "samples.csv"))
ok("samples.csv schema: 9 rows (row 0 = baseline), u_1/u_2, fields, QSS columns, selected",
   len(smp) == 9 and smp[0]["kind"] == "baseline" and smp[0]["selected"] == "baseline"
   and list(smp[0])[:6] == ["row_id", "kind", "u_1", "u_2", "alpha_RW", "hcg"]
   and all(k in smp[0] for k in ("qss_lap_s", "qss_delta_s", "qss_rank", "status", "vi_feasible", "selected"))
   and float(smp[0]["alpha_RW"]) == 8.0 and float(smp[0]["hcg"]) == 0.5)
cc = S.read_csv(os.path.join(d1, "confirmed.csv"))
ok("confirmed.csv schema and ranking by NLP lap",
   list(cc[0])[:5] == ["nlp_rank", "row_id", "kind", "alpha_RW", "hcg"]
   and all(k in cc[0] for k in ("nlp_lap_s", "nlp_delta_s", "delta_rev_s", "rt_residual_s", "branch_ok",
                                "unresolved_with", "iters", "warm_start_mode", "path", "w_sha", "repro"))
   and [S.num(x["nlp_lap_s"]) for x in cc if x["nlp_rank"]]
   == sorted(S.num(x["nlp_lap_s"]) for x in cc if x["nlp_rank"]))
sm = json.load(open(os.path.join(d1, "summary.json")))
ok("summary.json: metrics, noise floor, best (vp_overrides), timings, throughput",
   all(k in sm for k in ("plan", "fields", "warnings", "hub", "shortlist", "metrics", "noise_floor_s", "best",
                         "failures", "timings", "throughput", "files"))
   and all(k in r1["metrics_keys"] for k in ("all", "branch_ok", "within_shortlist", "concordance",
                                                "top1_regret_s", "nlp_best_in_qss_top_k", "slope",
                                                "sign_agreement", "screen_trusted"))
   and set(r1["best"]["vp_overrides"]) <= {"alpha_RW", "hcg"} and r1["throughput"]["nlp_workers"] == 2)
ok("resume: 0 NLP tasks, confirmed.csv byte-identical",
   R["resume"]["n_tasks_run"] == 0 and R["resume"]["csv_same"])
r3 = R["r3"]
c3 = {c["row_id"]: c for c in r3["confirmed"]}
ok("second run (base=hub.mat, workers=1, HSL not pinned): hub check 0 iterations, same rows",
   r3["hub"]["source"] == "file" and r3["hub"]["check"]["iters"] == 0 and sorted(c3) == sorted(conf))
ok("... identical nlp_lap_s and w_sha for every row (bitwise)",
   all(c3[k]["nlp_lap_s"] == conf[k]["nlp_lap_s"] and c3[k]["w_sha"] == conf[k]["w_sha"] for k in conf))
ok("OPT_ds=60 with that base: SweepError in the pre-check, no folder, no task",
   R["opt_ds"] is not None and "does not match" in R["opt_ds"]["msg"] and not R["opt_ds"]["folder"])
ok("another seed under the same name: SweepError (a sweep folder is never overwritten)",
   R["seed"] is not None and "another sweep" in R["seed"]["msg"])

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL SETUP SWEEP TESTS PASSED")
