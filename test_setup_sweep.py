"""setup_sweep.py, functions/sweep.py and MLTP_screen.screen_batch (plain script, no pytest).
Run from the repo root:

    venv\\Scripts\\python.exe test_setup_sweep.py

  1. functions.sweep units: validate_specs, sample_box, select_shortlist, bridge_points,
     rank_metrics on synthetic data (lap brackets, the trust rule: minimum n, an
     unconfirmed QSS-best row, the exact small-n Spearman p), noise floor / brackets /
     unresolved pairs, fingerprint, writers
  1b. setup_sweep internals with stand-ins (casadi-free, no solve): the hub candidate pick,
     _reject, _load_row, _rt_wave, _task_confirm with a stand-in hop (star, bridge ladder,
     solver mismatch, a better baseline branch), the argument ValueErrors (no folder),
     _resolve_base / _precheck_base, the finish stage with a stand-in runner (reuse rules,
     skips), the report on a stub sweep (an OFF-BRANCH winner, a failed QSS-best row:
     best provenance, unresolved brackets, warnings, failures), the inline runner, and the
     code hash (covers the import closure; an edit to functions/simpleMA.py changes it)
  2. screen_batch == screen_sweep, exact ==: 16 seeded Sobol setups over 10 fields in 4
     configs on Sturn (+ BCN when its .mat exists), the MLTP_screen baseline, an invalid row
     isolated, warnings captured; workers=2 with chunk 1 / 7 in a `python -c` child (Windows
     spawn re-imports the main script in every worker, so pools never start from this file)
  3. casadi blocked: this file re-runs itself in a child with sys.modules['casadi'] = None
     (sections 1-2 in-process, the imports of functions.sweep / MLTP_screen / setup_sweep,
     4 rows screened, a confirm=False sweep), then stops before section 3b
  3b. _Runner recovery (`python -c` child, stand-in tasks in spawn pools): a worker that dies
     once, a later wave after an earlier break, a poison task among innocent ones, a worker
     that died while idle before the next submit
  4. field classification (casadi, no solve): nlp_signature determinism, NLP-inert / live
     per config (AALB included), scale_changed, promotion_safe, QSS-blind, an all-dropped call
  5. NLP mini-sweep (Sturn, ~1 min, in a `python -c` child): brkB dropped, the hub (one
     cold candidate on the default base) and its check, the baseline delta, the accepted-row
     fields, best's provenance, files and schemas, a resume that runs no task, a second run
     from hub.mat (1 worker, HSL not pinned) with identical laps and w_sha, and the
     SweepErrors (another OPT_ds with that base, another seed, same name)

No NLP lap time or iteration count is pinned: only properties (warm-start modes, statuses,
0 iterations on the hub check, equality between runs, file schemas); the stand-in sections
use synthetic laps.
"""
import ast
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
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
H = 18.4                       # an OFF-BRANCH best row: round trip -9.994 ms, runner-up 1.786 ms behind
laps = {0: H, 1: H - 0.02373, 2: H - 0.021944}
brk = {0: S.lap_bracket(H, 0.0), 1: S.lap_bracket(laps[1], -0.009994), 2: S.lap_bracket(laps[2], 2e-5)}
ok("lap_bracket: (lap, lap - rt_residual) ordered; the point lap without a converged round trip",
   brk[1][0] == laps[1] and abs(brk[1][1] - (laps[1] + 0.009994)) < 1e-12 and brk[0] == (H, H)
   and S.lap_bracket(H, None) == (H, H) and S.lap_bracket(H, float("nan")) == (H, H))
ok("bracket_gap: 0 when the intervals overlap, else their distance",
   S.bracket_gap((1, 3), (2, 5)) == 0.0 and S.bracket_gap((1, 2), (2.5, 3)) == 0.5
   and S.bracket_gap((4, 5), (1, 2)) == 2.0)
ok("the off-branch row is unresolved against the row inside its bracket (the point rule resolved it)",
   S.unresolved_pairs(laps, 1e-4, brk) == {0: [], 1: [2], 2: [1]}
   and S.unresolved_pairs(laps, 1e-4) == {0: [], 1: [], 2: []}
   and S.noise_floor([dict(branch_ok=False, rt_residual_s=-0.009994), dict(branch_ok=True, rt_residual_s=2e-5)],
                     1e-4) == 1e-4)
rows_b = mrows(qd, [0.8 * x for x in qd])
rows_b[2].update(nlp_lo=rows_b[2]["nlp_lap"] - 0.004, nlp_hi=rows_b[2]["nlp_lap"])   # reaches row 1
rows_b[4].update(nlp_lo=rows_b[4]["nlp_lap"] - 0.002, nlp_hi=rows_b[4]["nlp_lap"])   # reaches the baseline
m = S.rank_metrics(rows_b, 1e-4, top_ids=[1, 2, 3])
ok("rank_metrics with brackets: pairs inside a bracket are unresolved (concordance 26 of 28, sign 6 of 7)",
   m["concordance"]["n_pairs"] == 26 and m["concordance"]["value"] == 1.0 and m["sign_agreement"]["n"] == 6
   and m["screen_trusted"] and m["trust_reasons"] == [])
m = S.rank_metrics(mrows(qd[:3], qd[:3], kinds=["baseline", "top", "top"]), 1e-4, [1, 2])
ok("n = 3 in perfect agreement: rho = 1 but not trusted (fewer than 6 rows); exact Spearman p = 1/3",
   abs(m["all"]["spearman_rho"] - 1) < 1e-12 and not m["screen_trusted"] and S.MIN_TRUST_N == 6
   and any("fewer than 6" in s for s in m["trust_reasons"]) and abs(m["all"]["spearman_p"] - 1 / 3) < 1e-12
   and m["all"]["spearman_p_method"] == "exact")
qx, yx = [1, 2, 2, 4, 5], [2.0, 1.0, 4.0, 3.0, 5.0]
r_obs = abs(spearmanr(qx, yx).statistic)
ok("exact Spearman p == brute force over the 5! orders (ties included)",
   abs(S.spearman_exact_p(qx, yx) - np.mean([abs(spearmanr(qx, p).statistic) >= r_obs - 1e-12
                                              for p in itertools.permutations(yx)])) < 1e-12)
m = S.rank_metrics(mrows(qd, [0.8 * x for x in qd]), 1e-4, [1, 2, 3],
                   unconfirmed=[dict(row_id=9, kind="top", qss_lap=18.7 - 0.02)])
ok("the QSS-best row unconfirmed: regret unknown (nan), not trusted, the row counted",
   m["qss_best_row"] == 9 and m["qss_best_confirmed"] is False and math.isnan(m["top1_regret_s"])
   and not m["screen_trusted"] and m["n_unconfirmed"] == 1 and m["unconfirmed_rows"] == [9]
   and any("row 9" in s for s in m["trust_reasons"]))
m = S.rank_metrics(mrows(qd, [0.8 * x for x in qd]), 1e-4, [1, 2, 3],
                   unconfirmed=[dict(row_id=9, kind="probe", qss_lap=18.7 + 0.5)])
ok("an unconfirmed row that is not the QSS best leaves the verdict alone",
   m["screen_trusted"] and m["qss_best_row"] == 1 and m["qss_best_confirmed"] and m["n_unconfirmed"] == 1)

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
print("1b. setup_sweep internals with stand-ins" + (" (casadi blocked)" if NO_CASADI else ""))
import setup_sweep as SS
from functions.context import Ctx
from functions.transcription import discretise
from functions.warmstart import nlp_structure
from userOpts import userOpts

cands = {"cold": dict(ok=True, lap_s=18.418836, iters=1980), "via_default": dict(ok=True, lap_s=17.951937, iters=5)}
name, msg = SS._pick_hub(cands, 1e-3)
ok("_pick_hub: the faster converged candidate, warned when the candidates disagree (466.899 ms)",
   name == "via_default" and msg is not None and "466.899 ms" in msg and "'cold' 18.418836 s in 1980 it" in msg)
ok("_pick_hub: within tol -> the faster, no warning; a failed candidate skipped; none -> (None, None)",
   SS._pick_hub({"cold": dict(ok=True, lap_s=18.0086, iters=179),
                 "via_default": dict(ok=True, lap_s=18.0087, iters=0)}, 1e-3) == ("cold", None)
   and SS._pick_hub({"cold": dict(ok=False, reason="error: x"),
                     "via_default": dict(ok=True, lap_s=18.2, iters=4)}, 1e-3) == ("via_default", None)
   and SS._pick_hub({"cold": dict(ok=False)}, 1e-3) == (None, None))

hop = dict(step="star", linear_solver="ma57", mode="full+duals", status="Solve_Succeeded", iters=3)
ok("_reject: accepted, else solver_mismatch before mode before status",
   SS._reject(hop, "ma57") is None
   and SS._reject(dict(hop, linear_solver="mumps", mode="cold"), "ma57")[0] == "solver_mismatch"
   and SS._reject(dict(hop, mode="full-interp", status="Maximum_Iterations_Exceeded"), "ma57")[0] == "mode"
   and SS._reject(dict(hop, status="Maximum_Iterations_Exceeded"), "ma57")[0] == "status")

LR = os.path.join(TMP, "load_row", "rows")
os.makedirs(LR, exist_ok=True)
jp5, vals5 = os.path.join(LR, "5.json"), {"alpha_RW": 9.5, "hcg": 0.48}


def put5(**kw):
    S.write_json_atomic(jp5, dict(dict(row_id=5, fingerprint="fp", status="accepted", overrides=dict(vals5)), **kw))


put5()
no_mat = SS._load_row(jp5, "fp", vals5)
open(os.path.join(LR, "5.mat"), "wb").close()
ok("_load_row: an accepted row is reused only with its .mat, the same fingerprint and the same values",
   no_mat is None and SS._load_row(jp5, "fp", vals5)["row_id"] == 5 and SS._load_row(jp5, "other", vals5) is None
   and SS._load_row(jp5, "fp", dict(vals5, hcg=0.49)) is None and SS._load_row(jp5, "fp", {"alpha_RW": 9.5}) is None)
again = []
for st in ("crashed", "code_changed", "error", "failed", "solver_mismatch"):
    put5(status=st)
    again.append(SS._load_row(jp5, "fp", vals5) is None)
with open(os.path.join(LR, "7.json"), "w") as fh:
    fh.write("{")
ok("_load_row: crashed / code_changed / error run again, failed / solver_mismatch are kept; "
   "a missing or unreadable record -> None",
   again == [True, True, True, False, False] and SS._load_row(os.path.join(LR, "6.json"), "fp", vals5) is None
   and SS._load_row(os.path.join(LR, "7.json"), "fp", vals5) is None)

RT = {1: dict(status="accepted", kind="top", nlp_lap_s=17.90, path="star", iters=5),
      2: dict(status="accepted", kind="top", nlp_lap_s=17.95, path="star", iters=8),
      3: dict(status="accepted", kind="top", nlp_lap_s=17.97, path="star", iters=3),
      4: dict(status="accepted", kind="probe", nlp_lap_s=18.20, path="star", iters=10),
      5: dict(status="accepted", kind="probe", nlp_lap_s=18.30, path="bridge2", iters=20),
      6: dict(status="accepted", kind="probe", nlp_lap_s=18.10, path="star", iters=60),
      7: dict(status="failed", kind="top"),
      8: dict(status="accepted", kind="top", nlp_lap_s=17.80, path="star", iters=2, rt_ok=True),
      9: dict(status="accepted", kind="finish", nlp_lap_s=17.70, path="star", iters=2)}
ok("_rt_wave: 'all' = accepted rows without a round trip; 'top' = the 3 best + bridged + > 50 it; 'none'",
   SS._rt_wave(RT, "all") == [1, 2, 3, 4, 5, 6] and SS._rt_wave(RT, "top") == [1, 2, 5, 6]
   and SS._rt_wave(RT, "none") == [])


class _FakeC:                                   # what _task_confirm reads of an MLTP ctx
    def __init__(self, lap):
        self.data = types.SimpleNamespace(lap_time=lap, nlp={"w_opt": np.zeros(2)})


def run_confirm(tag, fail=(), mismatch=(), bad_mode=(), rt_lap=18.0004):
    """_task_confirm (hub lap 18.0, row alpha_RW 8 -> 10) with SS._hop replaced by a stand-in:
    steps starting with an entry of ``fail`` end Maximum_Iterations_Exceeded, ``mismatch``
    on MUMPS, ``bad_mode`` as full-interp; forward hops give 17.95 s, the round trip rt_lap."""
    d = os.path.join(TMP, "confirm", tag)
    os.makedirs(d, exist_ok=True)
    task = dict(id="row_7", kind="confirm", row_id=7, row_kind="top", row={"alpha_RW": 10.0}, rt_only=False,
                roundtrip=True, row_mat=os.path.join(d, "7.mat"), rt_path=os.path.join(d, "7_rt.mat"),
                circuit="Sturn", vi=60.0, ni=float("nan"), AeroConfig="Static", ATD="On",
                Electric_4Motors="Off", TyreModel="CombinedSlip", base_ov={}, useropts={},
                hop_ipopt={"max_iter": 250}, cold_ipopt={}, fingerprint="f", model_hash="m",
                tol_branch_s=1e-3, bridge_steps=[2, 4], hub_vals={"alpha_RW": 8.0},
                hub_path=os.path.join(d, "hub.mat"), hub_lap=18.0, hub_linear_solver="ma57")
    calls = []

    def fake_hop(task_, warm, overrides, step):
        calls.append(step)
        hit = lambda keys: any(step.startswith(k) for k in keys)   # noqa: E731
        lap = rt_lap if step == "roundtrip" else 17.95
        return _FakeC(lap), dict(step=step, overrides=dict(overrides), lap=lap, iters=4,
                                 status="Maximum_Iterations_Exceeded" if hit(fail) else "Solve_Succeeded",
                                 mode="full-interp" if hit(bad_mode) else "full+duals",
                                 linear_solver="mumps" if hit(mismatch) else "ma57", wall=0.01, w_sha="w")

    old, SS._hop = SS._hop, fake_hop
    try:
        return SS._task_confirm(task), calls, task
    finally:
        SS._hop = old


rec, calls, task = run_confirm("star")
ok("_task_confirm, star: accepted, its .mat written, round trip within tol -> branch_ok, delta_rev, a repro",
   rec["status"] == "accepted" and rec["path"] == "star" and calls == ["star", "roundtrip"]
   and os.path.isfile(task["row_mat"]) and rec["branch_ok"] is True and abs(rec["rt_residual_s"] - 4e-4) < 1e-12
   and abs(rec["delta_rev_s"] - (17.95 - 18.0004)) < 1e-12 and abs(rec["nlp_delta_s"] + 0.05) < 1e-12
   and rec["repro"].startswith("from MLTP import MLTP") and "warm_start=r'" in rec["repro"])
rec, calls, _ = run_confirm("bridge2", fail=("star",))
ok("_task_confirm: a star hop that does not converge is bridged in 2 steps (never cold), repro chains them",
   rec["status"] == "accepted" and rec["path"] == "bridge2"
   and calls == ["star", "bridge2 1/2", "bridge2 2/2", "roundtrip"] and "reduce(" in rec["repro"])
rec, calls, _ = run_confirm("bridge4", fail=("star", "bridge2 2/2"))
ok("_task_confirm: bridge2 failing -> bridge4",
   rec["status"] == "accepted" and rec["path"] == "bridge4"
   and calls == ["star", "bridge2 1/2", "bridge2 2/2"] + [f"bridge4 {j}/4" for j in range(1, 5)] + ["roundtrip"])
rec, calls, task = run_confirm("fail", fail=("star", "bridge"))
ok("_task_confirm: star and both bridges failing -> 'failed' with every reason, no .mat, no round trip",
   rec["status"] == "failed" and rec["path"] is None and "star:" in rec["reason"] and "bridge2 1/2" in rec["reason"]
   and "bridge4 1/4" in rec["reason"] and "roundtrip" not in calls and not os.path.isfile(task["row_mat"]))
rec, calls, _ = run_confirm("mismatch", mismatch=("star",))
ok("_task_confirm: another linear solver -> 'solver_mismatch', never bridged",
   rec["status"] == "solver_mismatch" and calls == ["star"])
rec, calls, _ = run_confirm("mode", fail=("star",), bad_mode=("bridge2 1/2",))
ok("_task_confirm: a bridge step that is not full+duals stops the ladder ('failed')",
   rec["status"] == "failed" and calls == ["star", "bridge2 1/2"] and "not full+duals" in rec["reason"])
rec, calls, task = run_confirm("better", rt_lap=17.99)
ok("_task_confirm: a round trip below the hub lap by > tol saves the better baseline branch, OFF-BRANCH",
   rec["status"] == "accepted" and rec["branch_ok"] is False and abs(rec["rt_residual_s"] + 0.01) < 1e-12
   and rec["rt_better_baseline_file"] == task["rt_path"] and os.path.isfile(task["rt_path"]))

BADKW = [dict(roundtrip="bogus"), dict(TyreModel="Pure"), dict(sampler="grid"), dict(n_samples=0),
         dict(n_samples=2.5), dict(top_k=-1), dict(n_probes=-2), dict(bridge_steps=(1,)), dict(workers=-1),
         dict(workers=1.5), dict(screen_workers=0), dict(max_warm_iter=0)]
bad_ok = []
for i, bad in enumerate(BADKW):
    kw = dict(dict(n_samples=8), **bad)
    msg = raises(ValueError, SS.setup_sweep, [("alpha_RW", 4, 16)], results_root=TMP, name=f"bad_{i}",
                 verbose=False, **kw)
    bad_ok.append(msg is not None and not os.path.exists(os.path.join(TMP, f"bad_{i}")))
ok(f"setup_sweep: {len(BADKW)} bad arguments (sampler included) raise ValueError before any folder exists",
   all(bad_ok))

bc = userOpts(Ctx(), circuit="Sturn")
disc = discretise(bc.track, bc.OPT_ds, bc.OPT_d, mesh=bc.mesh, mesh_opts=bc.mesh_opts)
st4 = nlp_structure(23, len(bc.input_keys), 0, disc["N"], bc.OPT_d, 4)
good = dict(x_opt=np.zeros((23, disc["N"] + 1)), input_keys=list(bc.input_keys), s_full=disc["s_full"],
            tyre_set="MF205", circuit="Sturn", nlp=dict(structure=st4, w_opt=np.zeros(3)))
ok("_resolve_base: a missing file and a non-result raise SweepError; an in-memory full result is a 'ctx'",
   raises(SS.SweepError, SS._resolve_base, os.path.join(TMP, "nope.mat")) is not None
   and raises(SS.SweepError, SS._resolve_base, types.SimpleNamespace(data=dict(x_opt=np.zeros((7, 3))))) is not None
   and SS._resolve_base(types.SimpleNamespace(data=good))[2] == "ctx")
ok("_precheck_base accepts a base of this NLP's structure", raises(SS.SweepError, SS._precheck_base, good, bc, disc, 4,
                                                                    "Sturn", "<good>") is None)
BAD_BASES = [(dict(good, x_opt=np.zeros((7, 3))), "not a full 23-state result"),
             ({k: v for k, v in good.items() if k != "nlp"}, "no saved NLP vectors"),
             (dict(good, tyre_set="CopyB"), "tyre_set 'CopyB'"), (dict(good, circuit="BCN"), "circuit 'BCN'"),
             (dict(good, nlp=dict(structure=dict(st4, N=st4["N"] + 1), w_opt=np.zeros(3))), "N "),
             (dict(good, input_keys=list(bc.input_keys)[::-1]), "input_keys")]
pre_ok = []
for src, why in BAD_BASES:
    msg = raises(SS.SweepError, SS._precheck_base, src, bc, disc, 4, "Sturn", "<x>")
    pre_ok.append(msg is not None and why in msg and "does not match" in msg)
ok("_precheck_base rejects: not a full result, no data.nlp, another tyre_set / circuit / N / input_keys",
   pre_ok == [True] * len(BAD_BASES))


def stub_sweep(tag, records, qss, top_ids, hub_lap, **extra):
    """A _Sweep at the report / finish stage (one field alpha_RW, base 8), nothing solved."""
    out = os.path.join(TMP, "stub", tag)
    os.makedirs(os.path.join(out, "rows"), exist_ok=True)
    sw = SS._Sweep(dict(n_samples=len(qss) - 1, circuit="Sturn", name=tag, vi=60.0, ni=float("nan"),
                        AeroConfig="Static", ATD="On", Electric_4Motors="Off", TyreModel="CombinedSlip",
                        sampler="sobol", seed=0, top_k=len(top_ids), n_probes=0, base=None, confirm=True,
                        roundtrip="all", max_warm_iter=250, bridge_steps=(2, 4), tol_branch_s=1e-3,
                        resolve_s=1e-4, finish=False, workers=1, screen_workers=1, pin_hsl=False,
                        load_model="vehModel", ds_fine=1.0, results_root=os.path.dirname(out), plot=False,
                        resume=True, verbose=False))
    order = sorted(range(len(qss)), key=lambda i: (qss[i], i))
    sw.__dict__.update(
        n=len(qss) - 1, eATD="On", eEM4="Off", base_ov={}, kw={}, hop_ip={"max_iter": 250}, cold_ip={},
        fp="f" * 64, mhash="m" * 64, fields=["alpha_RW"], kept=[("alpha_RW", 4.0, 16.0)],
        base_vals={"alpha_RW": 8.0}, out_dir=out, plan={}, flags={}, dropped={}, records=dict(records),
        samples=[dict(row_id=i, qss_lap_s=q, qss_rank=order.index(i) + 1) for i, q in enumerate(qss)],
        top_ids=list(top_ids), shortlist=[(0, "baseline")] + [(i, "top") for i in top_ids],
        qss=dict(reused=True, wall_s=None, workers=0), P=1, resumed=False,
        hub=dict(lap_s=hub_lap, check=dict(iters=0, wall_s=1.0, mode="full+duals", linear_solver="ma57",
                                           w_sha="h")),
        paths={k: os.path.join(out, v) for k, v in dict(
            hub="hub.mat", rows="rows", logs="logs", finish="finish", confirmed="confirmed.csv",
            summary="summary.json", report="report.html", plan="plan.json", samples="samples.csv").items()})
    sw.__dict__.update(extra)
    return sw


def acc_rec(tag, rid, lap, rt, hub_lap, alpha):
    """rows/<id>.json of an accepted row whose round trip ended rt seconds from the hub."""
    return dict(row_id=rid, kind="top", overrides={"alpha_RW": alpha}, status="accepted", nlp_lap_s=lap,
                iters=5, path="star", warm_start_mode="full+duals", linear_solver="ma57", w_sha="w",
                result_file=os.path.join(TMP, "stub", tag, "rows", f"{rid}.mat"), repro=f"repro {rid}",
                reason=None, wall_s=1.0, rt_ok=True, rt_residual_s=rt, branch_ok=abs(rt) <= 1e-3,
                rt_lap_s=hub_lap + rt, delta_rev_s=lap - (hub_lap + rt), rt_iters=3)


recs = {1: acc_rec("offb", 1, H - 0.02373, -0.009994, H, 10.0), 2: acc_rec("offb", 2, H - 0.021944, 2e-5, H, 11.0),
        3: dict(row_id=3, kind="top", overrides={"alpha_RW": 12.0}, status="failed", wall_s=9.0,
                reason="star: Maximum_Iterations_Exceeded after 250 iterations"),
        4: dict(acc_rec("offb", 4, H + 0.05, 0.004, H, 5.0), kind="probe")}
sw = stub_sweep("offb", recs, qss=[18.70, 18.69, 18.68, 18.60, 18.75], top_ids=[3, 2, 1], hub_lap=H)
res, w = caught(sw.finalise)
b = res.best
ok("report: best = the OFF-BRANCH row 1 with its flag, residual, delta bracket, result file and repro",
   b["row_id"] == 1 and b["branch_ok"] is False and abs(b["rt_residual_s"] + 0.009994) < 1e-12
   and abs(b["delta_bracket_s"][0] + 0.02373) < 1e-9 and abs(b["delta_bracket_s"][1] + 0.013736) < 1e-9
   and os.path.normpath(b["result_file"]) == os.path.normpath(recs[1]["result_file"]) and b["repro"] == "repro 1"
   and b["vp_overrides"] == {"alpha_RW": 10.0})
ok("report: row 2 sits inside row 1's bracket -> unresolved both ways, although 1.8 ms apart",
   b["unresolved_with"] == [2] and {r["row_id"]: r["unresolved_with"] for r in res.confirmed}[2] == [1])
ok("report: warned about the OFF-BRANCH winner, 2 of 3 rows off-branch, the failed confirmation and "
   "the untrusted screen",
   any("best row 1 is OFF-BRANCH" in x for x in w) and any("2 of 3 round-tripped rows are OFF-BRANCH" in x for x in w)
   and any("1 of 4 confirmation(s) did not succeed" in x for x in w)
   and any("not trusted" in x and "fewer than 6" in x and "QSS-best row 3" in x for x in w))
ok("report: the failed QSS-best row makes the regret unknown; failures on the result and in summary.json",
   res.metrics["qss_best_row"] == 3 and not res.metrics["qss_best_confirmed"]
   and math.isnan(res.metrics["top1_regret_s"]) and not res.metrics["screen_trusted"]
   and [f["row_id"] for f in res.failures] == [3] and res.failures[0]["status"] == "failed"
   and S.read_json(sw.paths["summary"])["failures"] == res.failures
   and abs(S.read_json(sw.paths["summary"])["rt_residual_max_s"] - 0.009994) < 1e-12
   and S.read_json(sw.paths["summary"])["best"]["repro"] == "repro 1")
recs = {1: acc_rec("base", 1, H + 0.01, 2e-4, H, 10.0), 2: acc_rec("base", 2, H + 0.02, -1e-4, H, 11.0)}
sw = stub_sweep("base", recs, qss=[18.70, 18.69, 18.68], top_ids=[2, 1], hub_lap=H)
res, w = caught(sw.finalise)
ok("report: the baseline as best is on the hub's branch, its result file is hub.mat, repro re-solves the hub",
   res.best["row_id"] == 0 and res.best["branch_ok"] is True and res.best["delta_s"] == 0.0
   and res.best["result_file"].endswith("hub.mat") and res.best["repro"].startswith("from MLTP import MLTP")
   and "hub.mat" in res.best["repro"] and not any("OFF-BRANCH" in x for x in w) and res.failures == [])


class FakeRunner:                               # the _Runner interface, canned results
    def __init__(self, results):
        self.results, self.submitted, self.inline, self.n_submitted = results, [], False, 0

    def submit(self, task):
        self.submitted.append(task)

    def collect(self, ids, on_done=None):
        out = {t: self.results[t] for t in ids}
        for t in ids if on_done else ():
            on_done(next(x for x in self.submitted if x["id"] == t), out[t])
        return out


recs = {1: acc_rec("fin", 1, H - 0.02, 2e-4, H, 10.0)}
eligible = [["alpha_RW", 4.0, 16.0]]
sw = stub_sweep("fin", recs, qss=[18.70, 18.69], top_ids=[1], hub_lap=H,
                runner=FakeRunner({"finish": dict(status="skipped", reason="stand-in: warm start rejected")}))
fjson = os.path.join(sw.paths["finish"], "finish.json")
S.write_json_atomic(fjson, dict(fingerprint=sw.fp, winner_row=1, eligible=eligible, design_status="crashed",
                                status="crashed"))
fin = dict(status="skipped", fingerprint=sw.fp, eligible=eligible, excluded={})    # as run_finish passes it
_, w = caught(sw._finish_solve, fin, eligible)
ok("finish: a crashed finish.json is not reused, optimise_design is submitted again (stand-in runner)",
   [t["id"] for t in sw.runner.submitted] == ["finish"] and sw.runner.submitted[0]["winner"] == {"alpha_RW": 10.0}
   and fin["design_status"] == "skipped" and S.read_json(fjson)["design_status"] == "skipped"
   and any("finish skipped" in x for x in w))
sw.runner = FakeRunner({})
fin = dict(status="skipped", fingerprint=sw.fp, eligible=eligible, excluded={})
sw._finish_solve(fin, eligible)
ok("finish: a skipped / failed finish.json of this sweep is reused, nothing submitted",
   sw.runner.submitted == [] and fin["design_status"] == "skipped")
sw.TyreModel = "PureSlip"
sw.flags = {"alpha_RW": dict(promotion_safe=True, primary=True, phase2_free=True, scale_changed=False)}
_, w = caught(sw.run_finish)
ok("finish with TyreModel PureSlip: skipped with its reason (optimise_design is CombinedSlip only)",
   sw.fin["status"] == "skipped" and "CombinedSlip" in sw.fin["reason"] and any("finish skipped" in x for x in w))
sw.TyreModel = "CombinedSlip"
sw.flags = {"alpha_RW": dict(promotion_safe=False, primary=False, phase2_free=True, scale_changed=False)}
_, w = caught(sw.run_finish)
ok("finish without a promotion-safe field: skipped, the field excluded with its reason",
   sw.fin["reason"] == "no promotion-safe field" and "Pacejka" in sw.fin["excluded"]["alpha_RW"]
   and any("not promotable" in x for x in w))

def screen_workers_picked(n, row_s, cold, n_running):
    """The worker count _Sweep.screen gives the n-row batch (screen_workers='auto'), with
    screen_batch replaced by a recorder whose baseline row takes row_s seconds."""
    sw = stub_sweep(f"scr_{n}_{cold}_{n_running}", {}, qss=[18.7, 18.7], top_ids=[], hub_lap=H)
    U = (np.arange(n, dtype=float).reshape(-1, 1) + 0.5) / n
    sw.__dict__.update(n=n, stored=None, U=U, u0=np.array([1 / 3]), screen_workers="auto", top_k=2, n_probes=1,
                       hub_cold_running=cold, P=2, row_vals=[{"alpha_RW": 8.0}] + [{"alpha_RW": 4 + 12 * u} for u in U[:, 0]],
                       runner=None if n_running is None else types.SimpleNamespace(inline=False,
                                                                                   n_running=lambda: n_running))
    sw.row_ovs = [{}] + sw.row_vals[1:]
    calls = []

    def recorder(circuit, ovs, workers=1, **kw):
        calls.append(workers)
        k = len(ovs)
        return dict(lap_time=18.7 + np.arange(k) * 1e-3, status=["ok"] * k, vi_feasible=np.ones(k, bool),
                    error=[None] * k, warnings=[], wall_s=row_s * k)

    old, SS.screen_batch = SS.screen_batch, recorder
    try:
        sw.screen()
    finally:
        SS.screen_batch = old
    return calls[1]


aw = lambda n: S.auto_workers(math.ceil(n / 256))           # noqa: E731
ok("screen_workers='auto': serial while it hides behind a cold hub (or is short); past that a pool on the "
   "cores the NLP tasks leave",
   screen_workers_picked(1024, 0.01, True, 1) == 1 and screen_workers_picked(200, 0.01, False, None) == 1
   and screen_workers_picked(4096, 0.01, True, 1) == max(1, aw(4096) - 1)
   and screen_workers_picked(4096, 0.01, True, 2) == max(1, aw(4096) - 2)
   and screen_workers_picked(1024, 0.01, False, None) == aw(1024))

seen, got = [], []
rn = SS._Runner(0, dict(root=HERE, circuit="Sturn"), inline=True, say=lambda m: None)
rn._init_fn, rn._run_fn = (lambda spec: seen.append("init")), (lambda t: dict(status="done", id=t["id"]))
rn.submit(dict(id="a"))
rn.submit(dict(id="b"))
out = rn.collect(["a", "b"], on_done=lambda t, x: got.append(t["id"]))
rn.collect(["a"], on_done=lambda t, x: got.append("again"))
ok("inline runner (workers=0): the initializer once, results in order, on_done once per task",
   seen == ["init"] and [out[k]["status"] for k in "ab"] == ["done", "done"] and got == ["a", "b"])


def import_closure(mods, root=HERE):
    """Repo files a plain import of ``mods`` can reach (every import statement, nested and
    relative ones included, plus the package __init__ of a submodule)."""
    def resolve(mod):
        for cand in (os.path.join(root, *mod.split(".")) + ".py", os.path.join(root, *mod.split("."), "__init__.py")):
            if os.path.isfile(cand):
                return os.path.relpath(cand, root).replace("\\", "/")
        return None

    seen_f, todo = set(), [resolve(m) for m in mods]
    while todo:
        f = todo.pop()
        if f is None or f in seen_f:
            continue
        seen_f.add(f)
        pkg = os.path.dirname(f).replace("/", ".")
        with open(os.path.join(root, f), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level <= 1:
                base = node.module if node.level == 0 else ".".join(x for x in (pkg, node.module) if x)
                names = [base] + [f"{base}.{a.name}" for a in node.names] if base else []
            for nm in names:
                parts = nm.split(".")
                todo += [resolve(".".join(parts[:i])) for i in range(1, len(parts) + 1)]
    return seen_f


closure = import_closure(["MLTP", "MLTP_initial", "MLTP_screen", "MLTP_paramOptim", "setup_sweep"])
ok("the code hash covers the import closure of the solve entry points (minus the plot-only files)",
   {"functions/simpleMA.py", "functions/importfile.py", "setup_sweep.py"} <= closure
   and closure - set(SS._HASH_EXEMPT) <= set(SS._CODE_FILES)
   and all(os.path.isfile(os.path.join(HERE, f)) for f in SS._CODE_FILES))
hroots = [os.path.join(TMP, "hash1"), os.path.join(TMP, "hash2")]   # copies: immune to concurrent edits
for src, dst in ((HERE, hroots[0]), (hroots[0], hroots[1])):
    for f in SS._CODE_FILES:
        os.makedirs(os.path.dirname(os.path.join(dst, f)), exist_ok=True)
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
h1, h2 = (S.model_hash(SS._CODE_FILES, root=d) for d in hroots)
with open(os.path.join(hroots[1], "functions", "simpleMA.py"), "a", encoding="utf-8") as fh:
    fh.write("\n# edited\n")
ok("model_hash: location independent (a byte copy hashes alike); an edit to functions/simpleMA.py changes it",
   h1 == h2 and S.model_hash(SS._CODE_FILES, root=hroots[1]) != h1)

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
bb, w = caught(screen_batch, "Sturn", [{}, {"mb": 1800.0}], ATD="On", Electric_4Motors="On")
ok("warnings are captured per row: the userOpts ATD + 4-motor guard once in bb['warnings'], none leaks",
   len(bb["warnings"]) == 1 and "cannot both be On" in bb["warnings"][0] and w == []
   and bb["status"] == ["ok", "ok"])

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
print("3b. _Runner recovery (stand-in tasks in spawn pools, `python -c` child)")
STANDIN = '''"""Stand-in task functions for test_setup_sweep.py section 3b (written to a temp dir)."""
import os
import threading
import time


def init(spec):
    pass


def run(task):
    die = task.get("die")
    if die == "always" or (die == "once" and not os.path.exists(task["marker"])):
        open(task["marker"], "w").close()
        os._exit(3)                                   # a native crash / OOM kill stand-in
    if task.get("die_idle"):
        threading.Timer(0.3, lambda: os._exit(4)).start()
    time.sleep(task.get("sleep", 0.0))
    return dict(status="done", pid=os.getpid())
'''
SDIR = os.path.join(TMP, "standin")
os.makedirs(SDIR, exist_ok=True)
with open(os.path.join(SDIR, "sweep_standin.py"), "w", encoding="utf-8") as fh:
    fh.write(STANDIN)
code = r"""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
import setup_sweep as SS
import sweep_standin as F
D, out, lines = sys.argv[1], {}, []
r = SS._Runner(2, dict(root=os.getcwd(), circuit="Sturn"), inline=False, say=lines.append)
r._init_fn, r._run_fn = F.init, F.run


def wave(name, tasks):
    for t in tasks:
        r.submit(dict(t, marker=os.path.join(D, t["id"] + ".marker")))
    out[name] = {k: v["status"] for k, v in r.collect([t["id"] for t in tasks]).items()}


wave("w1", [dict(id="a", die="once"), dict(id="b", sleep=0.5)])
wave("w2", [dict(id="x", die="once")] + [dict(id=f"y{i}", sleep=0.2) for i in range(6)])
wave("w3", [dict(id="p", die="always")] + [dict(id=f"q{i}", sleep=0.2) for i in range(3)])
wave("w4", [dict(id="k", die_idle=True)])
time.sleep(2.0)                                       # the worker that ran k dies while idle
try:
    wave("w5", [dict(id="z")])
except Exception as exc:
    out["w5"] = f"raised {type(exc).__name__}: {exc}"
out["breaks"], out["lines"] = r.n_breaks, lines
r.close()
print("RESULT " + json.dumps(out))
"""
r = subprocess.run([PY, "-c", code, SDIR], cwd=HERE, capture_output=True, text=True, timeout=600)
RR = json.loads(r.stdout.split("RESULT ", 1)[1]) if "RESULT " in r.stdout else None
if RR is None:
    print(r.stdout[-3000:], r.stderr[-3000:])
ok("the runner child ran", RR is not None)
ok("a worker that dies once: its task and its neighbour re-run one per process and finish",
   RR["w1"] == {"a": "done", "b": "done"})
ok("a later wave after that break: x dies once, all seven finish (no run-wide recovery budget)",
   RR["w2"] == {"x": "done", **{f"y{i}": "done" for i in range(6)}})
ok("a poison task: 'crashed' once it died alone too; its three neighbours finish",
   RR["w3"] == {"p": "crashed", **{f"q{i}": "done" for i in range(3)}})
ok("a worker that died while idle: the next submit does not raise, its task finishes",
   RR["w4"] == {"k": "done"} and RR["w5"] == {"z": "done"})
ok("four shared-pool breaks handled; every one that held tasks reported",
   RR["breaks"] == 4 and sum("one per fresh worker process" in x for x in RR["lines"]) >= 3)

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
PO = [("brkB", 0.0, 1.0), ("Tdist", 0.0, 1.0), ("alpha_FL", 0.0, 10.0), ("alpha_FR", 0.0, 10.0),
      ("alpha_RW", 0.0, 30.0), ("alpha_TW", -12.0, 12.0)]           # MLTP_paramOptim's defaults
fa = SS.classify_fields(PO, "Sturn", AeroConfig="AALB")
ok("AALB (ATD On): all six default paramOptim fields are NLP-inert (a sweep would drop them)",
   sorted(fa) == sorted(f for f, *_ in PO) and all(fa[f]["nlp_live"] is False for f in fa))
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
                 best=r1.best, throughput=r1.throughput, failures=r1.failures,
                 trusted=r1.metrics["screen_trusted"], n_metric=r1.metrics["n"])
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
ok("hub on the default base: one candidate ('cold', no continuation needed), kept as hub.mat",
   list(r1["hub"]["candidates"]) == ["cold"] and r1["hub"]["chosen"] == "cold"
   and r1["hub"]["candidates"]["cold"]["ok"] and os.path.isfile(os.path.join(d1, "hub_candidates", "cold.mat")))
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
bst = r1["best"]
ok("best carries its branch flag, residual, delta bracket, result file and repro; failures on the result",
   all(k in bst for k in ("branch_ok", "rt_ok", "rt_residual_s", "delta_bracket_s", "result_file", "repro"))
   and os.path.isfile(bst["result_file"]) and "from MLTP import MLTP" in bst["repro"]
   and bst["delta_bracket_s"][0] <= bst["delta_s"] <= bst["delta_bracket_s"][1]
   and r1["failures"] == [] and sm["best"] == bst)
ok("4 confirmed rows are too few to trust the screen: screen_trusted False and warned",
   r1["n_metric"] == 4 and r1["trusted"] is False
   and any("not trusted" in x and "fewer than 6" in x for x in R["warn1"]))
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
