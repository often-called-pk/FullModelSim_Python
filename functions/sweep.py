"""sweep.py - casadi-free helpers of the setup sweep (setup_sweep.py, roadmap Item 8).

Pure numpy / scipy / stdlib code; this module never imports casadi, so the
sampling, shortlist and statistics layers are testable without the NLP stack.

  validate_specs()    check [(field, lower, upper), ...]: malformed items raise
                      ValueError, fields vp_overrides would reject are dropped
                      with a UserWarning
  sample_box()        scrambled Sobol' (or Latin hypercube) design: unit-cube
                      points U and box points X
  select_shortlist()  baseline + QSS top-k + probes at QSS-rank quantiles of the rest
  bridge_points()     intermediate setups on the segment hub -> row (the private
                      homotopy of one confirmation)
  rank_metrics()      QSS-vs-NLP agreement: Spearman / Kendall tau-b with seeded
                      bootstrap CIs, resolved-pair concordance, regret, slope ...,
                      and whether the screen ranking can be trusted
  noise_floor(), lap_bracket(), bracket_gap(), unresolved_pairs()
  fingerprint(), model_hash(), sha256_file(), array_sha()
  free_ram_mb(), auto_workers(), blas_single_thread()
  replace_retry(), write_text_atomic(), write_json_atomic(), read_json(),
  write_csv_atomic(), read_csv(), num(), dedup()
"""

import contextlib
import csv
import ctypes
import hashlib
import inspect
import io
import json
import math
import os
import sys
import time
import warnings

import numpy as np

SAMPLERS = ("sobol", "lhs")
BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
UNKNOWN_REASON = "unknown (vp_overrides would raise)"
MIN_TRUST_N = 6            # fewer confirmed rows never give screen_trusted (rho = 1 by chance: 1/n!)
SPEARMAN_EXACT_MAX_N = 8   # exact permutation p-value of Spearman's rho up to this n (8! orders)
BOOT_CHUNK_MAX = 256       # bootstrap resamples per chunk, at most ...
BOOT_CHUNK_ELEMS = 2_000_000   # ... and (resamples x row pairs) at most this many (~65 MB peak at any n)
REPLACE_TRIES = 8          # os.replace attempts per atomic write (Windows: a target open elsewhere)
REPLACE_WAIT_S = 0.05      # pause after the first failed attempt; doubles each time ...
REPLACE_MAX_WAIT_S = 0.5   # ... up to this (8 attempts wait 2.25 s in all)


# ============================================================================
# 1. Parameter specs, design, shortlist, bridges
# ============================================================================
def validate_specs(param_specs, known=None):
    """Check optimise_design-style specs [(field, lower, upper), ...].

    Returns (specs, dropped): specs = [(field, float(lower), float(upper))] in
    input order; dropped = {field: reason}. Raises ValueError for a spec error
    (an item that is not a triple, a non-str field, a duplicate field, a
    non-finite bound, lower >= upper). A field that vp_overrides would reject
    (not in vehParams.PRIMARY_KEYS | MF_KEYS, e.g. a typo or the derived 'ms')
    gets a UserWarning and is dropped. Pacejka (mf) keys are sweepable.
    ``known`` overrides the accepted key set (default: read from vehParams)."""
    if known is None:
        from vehParams import PRIMARY_KEYS, MF_KEYS
        known = PRIMARY_KEYS | MF_KEYS
    if isinstance(param_specs, (str, bytes, dict)) or not hasattr(param_specs, "__iter__"):
        raise ValueError("param_specs must be a list of (field, lower, upper) triples")
    parsed, seen = [], set()
    for item in param_specs:
        if isinstance(item, (str, bytes)):
            raise ValueError(f"param_specs item {item!r} is not a (field, lower, upper) triple")
        try:
            field, lo, hi = item
        except (TypeError, ValueError):
            raise ValueError(f"param_specs item {item!r} is not a (field, lower, upper) triple") from None
        if not isinstance(field, str) or not field:
            raise ValueError(f"param_specs field {field!r} must be a non-empty str")
        if field in seen:
            raise ValueError(f"param_specs lists {field!r} twice")
        seen.add(field)
        if isinstance(lo, bool) or isinstance(hi, bool):
            raise ValueError(f"{field}: bounds must be numbers, got ({lo!r}, {hi!r})")
        try:
            lo_f, hi_f = float(lo), float(hi)
        except (TypeError, ValueError):
            raise ValueError(f"{field}: bounds must be numbers, got ({lo!r}, {hi!r})") from None
        if not (math.isfinite(lo_f) and math.isfinite(hi_f)):
            raise ValueError(f"{field}: non-finite bound ({lo!r}, {hi!r})")
        if not lo_f < hi_f:
            raise ValueError(f"{field}: lower {lo_f!r} must be < upper {hi_f!r}")
        parsed.append((field, lo_f, hi_f))
    specs, dropped = [], {}
    for field, lo_f, hi_f in parsed:
        if field not in known:
            warnings.warn(f"setup_sweep: {field!r} is not a vehParams primary or Pacejka "
                          f"coefficient ({UNKNOWN_REASON}); dropped", UserWarning, stacklevel=2)
            dropped[field] = UNKNOWN_REASON
            continue
        specs.append((field, lo_f, hi_f))
    return specs, dropped


def _qmc_engine(cls, d, rng, **kw):
    """``cls(d, **kw)`` (a scipy.stats.qmc engine) driven by the numpy Generator
    ``rng``. The generator keyword is ``rng`` since scipy 1.15 (SPEC 7) and ``seed``
    before, so it is picked from the constructor's signature: requirements.txt
    allows scipy >= 1.10."""
    key = "rng" if "rng" in inspect.signature(cls).parameters else "seed"
    return cls(d, **{key: rng}, **kw)


def sample_box(specs, n, sampler="sobol", seed=0):
    """Space-filling design over the box of ``specs``: returns (U, X), both
    (n, d): U in the unit cube, X = qmc.scale(U, lower, upper).

    sampler='sobol': qmc.Sobol(d, scramble=True) driven by default_rng(seed);
    random_base2(m) when n == 2**m, else a UserWarning (the balance properties
    need a power of 2) and random(n). sampler='lhs': qmc.LatinHypercube(d)
    driven by default_rng(seed), random(n) (the generator keyword is ``rng`` or,
    before scipy 1.15, ``seed``: _qmc_engine). The same (specs, n, sampler, seed)
    gives the same design (the stored samples.csv of a sweep stays authoritative).
    seed=None draws OS entropy, i.e. a different design on every call (setup_sweep
    replaces it by a drawn, stored seed)."""
    from scipy.stats import qmc
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
    if isinstance(n, bool) or int(n) != n or int(n) < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n!r}")
    n, d = int(n), len(specs)
    if d < 1:
        raise ValueError("sample_box needs at least one field")
    lo = np.array([float(s[1]) for s in specs])
    hi = np.array([float(s[2]) for s in specs])
    rng = np.random.default_rng(seed)
    if sampler == "sobol":
        eng = _qmc_engine(qmc.Sobol, d, rng, scramble=True)
        m = n.bit_length() - 1
        if n == 1 << m:
            U = eng.random_base2(m)
        else:
            warnings.warn(f"sample_box: n_samples={n} is not a power of 2, so the Sobol' design "
                          f"loses its balance properties (drawn with random(n)); use e.g. "
                          f"{1 << m} or {1 << (m + 1)}", UserWarning, stacklevel=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)    # scipy's own balance warning
                U = eng.random(n)
    else:
        U = _qmc_engine(qmc.LatinHypercube, d, rng).random(n)
    return U, qmc.scale(U, lo, hi)


def select_shortlist(qss_lap, ok, top_k, n_probes):
    """Rows to confirm with the NLP: [(row_id, kind), ...].

    Row 0 (the baseline) is always first ('baseline'). The valid sample rows
    1..n (ok and a finite lap) are sorted stably by (qss_lap, row_id); 'top' =
    the first top_k; 'probe' = the remainder R at positions
    floor(j/n_probes*(len(R)-1) + 0.5), j = 1..n_probes (QSS-rank quantiles, so
    the confirmed set spans the QSS range), clipped and de-duplicated."""
    q = np.asarray(qss_lap, dtype=float).reshape(-1)
    okb = np.asarray(ok, dtype=bool).reshape(-1)
    if okb.size != q.size:
        raise ValueError("select_shortlist: qss_lap and ok differ in length")
    top_k, n_probes = max(int(top_k), 0), max(int(n_probes), 0)
    valid = [i for i in range(1, q.size) if okb[i] and math.isfinite(q[i])]
    order = sorted(valid, key=lambda i: (q[i], i))
    out = [(0, "baseline")] + [(i, "top") for i in order[:top_k]]
    rest = order[top_k:]
    if rest and n_probes:
        pos = []
        for j in range(1, n_probes + 1):
            p = int(math.floor(j / n_probes * (len(rest) - 1) + 0.5))
            p = min(max(p, 0), len(rest) - 1)
            if p not in pos:
                pos.append(p)
        out += [(rest[p], "probe") for p in pos]
    return out


def bridge_points(base_vals, target_vals, k):
    """k intermediate setups on the straight segment base -> target, at
    t = j/k for j = 1..k (dicts over target_vals' keys). The last point is
    target_vals itself (same values, no recomputation)."""
    k = int(k)
    if k < 1:
        raise ValueError(f"bridge_points: k must be >= 1, got {k}")
    missing = [f for f in target_vals if f not in base_vals]
    if missing:
        raise ValueError(f"bridge_points: no base value for {missing}")
    pts = []
    for j in range(1, k):
        t = j / k
        pts.append({f: float(base_vals[f]) + t * (float(target_vals[f]) - float(base_vals[f]))
                    for f in target_vals})
    pts.append(dict(target_vals))
    return pts


# ============================================================================
# 2. Statistics: screen (QSS) vs NLP ranking
# ============================================================================
def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def noise_floor(rows, resolve_s=1e-4):
    """The on-branch continuation noise: max(resolve_s, max |rt_residual_s| over rows
    with branch_ok). Round-trip residuals measure how far a warm hop and its way back
    disagree; two NLP laps closer than the floor cannot be ordered. A row whose round
    trip landed on another branch (branch_ok False) is deliberately not folded into
    this one number (a single 0.1 s outlier would blur every pair of the sweep): it
    carries its own lap bracket (lap_bracket), which unresolved_pairs and
    rank_metrics apply on top of the floor."""
    res = [abs(float(r["rt_residual_s"])) for r in rows
           if r.get("branch_ok") is True and _finite(r.get("rt_residual_s"))]
    return max([float(resolve_s)] + res)


def lap_bracket(lap, rt_residual_s=None):
    """(lo, hi) bracket of a confirmed NLP lap: the forward lap and, when the row's
    round trip back to the base setup converged, the same lap re-referenced to the
    baseline branch that round trip found (lap - rt_residual_s, i.e. hub lap +
    delta_rev). A row without a converged round trip, and the baseline itself, is the
    point (lap, lap)."""
    lap = float(lap)
    if _finite(rt_residual_s):
        other = lap - float(rt_residual_s)
        return (min(lap, other), max(lap, other))
    return (lap, lap)


def bracket_gap(a, b):
    """Distance between the closed intervals a = (lo, hi) and b (0 when they overlap)."""
    return max(0.0, max(float(a[0]), float(b[0])) - min(float(a[1]), float(b[1])))


def unresolved_pairs(laps, floor, brackets=None):
    """{row_id: sorted ids that cannot be ordered against this row} for laps =
    {row_id: lap} (finite laps only). Without ``brackets``: the laps are within
    ``floor``. With brackets = {row_id: (lo, hi)} (lap_bracket; a missing id is its
    point lap): the brackets are within ``floor`` of each other, so a row whose round
    trip found another branch is unresolved against every row inside its bracket."""
    items = [(rid, float(v)) for rid, v in laps.items() if _finite(v)]
    br = {rid: (tuple(brackets[rid]) if brackets and rid in brackets else (v, v)) for rid, v in items}
    out = {}
    for rid, v in items:
        out[rid] = sorted(o for o, _ in items if o != rid and bracket_gap(br[rid], br[o]) <= floor)
    return out


def _avg_ranks(A):
    """Average ranks (1-based, ties averaged) along the last axis."""
    from scipy.stats import rankdata
    return rankdata(A, axis=-1)


def spearman_rows(Q, Y):
    """Spearman rho for each row pair of Q, Y (B x n): Pearson on average ranks
    (scipy.stats.spearmanr's definition); NaN where a row is constant."""
    rq = _avg_ranks(np.atleast_2d(Q))
    ry = _avg_ranks(np.atleast_2d(Y))
    rq = rq - rq.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rq * rq).sum(axis=1) * (ry * ry).sum(axis=1))
    num = (rq * ry).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return np.clip(r, -1.0, 1.0)


def kendall_b_rows(Q, Y):
    """Kendall tau-b for each row pair of Q, Y (B x n), O(n^2) per row:
    (C - D) / sqrt((n0 - n1)(n0 - n2)) with n1 / n2 the pairs tied in Q / Y
    (scipy.stats.kendalltau's default variant); NaN where undefined."""
    Q, Y = np.atleast_2d(np.asarray(Q, float)), np.atleast_2d(np.asarray(Y, float))
    n = Q.shape[1]
    iu, ju = np.triu_indices(n, 1)
    dq = np.sign(Q[:, iu] - Q[:, ju])
    dy = np.sign(Y[:, iu] - Y[:, ju])
    n0 = iu.size
    S = (dq * dy).sum(axis=1)
    untied_q = (n0 - (dq == 0).sum(axis=1)).astype(float)
    untied_y = (n0 - (dy == 0).sum(axis=1)).astype(float)
    den = np.sqrt(untied_q * untied_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(den > 0, S / np.where(den > 0, den, 1.0), np.nan)
    return np.clip(t, -1.0, 1.0)


def spearman_exact_p(q, y):
    """Two-sided exact permutation p-value of Spearman's rho: the fraction of the n!
    orderings of y whose |rho| reaches the observed one (scipy's t approximation says
    p = 0 for n = 3 and rho = 1; the exact value is 1/3). NaN when rho is undefined."""
    from itertools import permutations
    q, y = np.asarray(q, float), np.asarray(y, float)
    obs = spearman_rows(q, y)[0]
    if not np.isfinite(obs):
        return float("nan")
    perms = np.array(list(permutations(range(q.size))))
    rho = spearman_rows(np.broadcast_to(q, perms.shape), y[perms])
    return float(np.mean(np.abs(rho) >= abs(obs) - 1e-12))


def _boot_chunk(n):
    """Bootstrap resamples per chunk for n rows. kendall_b_rows holds a few
    (resamples x n (n - 1) / 2) float arrays, so a fixed chunk of 256 needs
    O(256 n^2) memory (about 1 GB at n = 500); the chunk is the largest number of
    resamples (at most BOOT_CHUNK_MAX, at least 1) that keeps that product within
    BOOT_CHUNK_ELEMS."""
    pairs = max(n * (n - 1) // 2, 1)
    return int(max(1, min(BOOT_CHUNK_MAX, BOOT_CHUNK_ELEMS // pairs)))


def _corr_block(q, y, seed, n_boot, label=None):
    """Spearman rho / Kendall tau-b with p-values (Spearman exact by permutation for
    n <= SPEARMAN_EXACT_MAX_N, else scipy's), plus percentile bootstrap 95% CIs
    (``n_boot`` resamples of the rows, seeded; skipped when n < 6) for one subset.
    The resamples are processed in chunks of _boot_chunk(n), which bounds the memory
    (the CIs do not depend on the chunk size)."""
    from scipy.stats import spearmanr, kendalltau
    q, y = np.asarray(q, float), np.asarray(y, float)
    n = int(q.size)
    out = dict(n=n, spearman_rho=float("nan"), spearman_p=float("nan"), spearman_ci95=None,
               kendall_tau_b=float("nan"), kendall_p=float("nan"), kendall_ci95=None)
    if label:
        out["label"] = label
    if n >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # constant input -> nan
            r = spearmanr(q, y)
            t = kendalltau(q, y)
        out.update(spearman_rho=float(r.statistic), spearman_p=float(r.pvalue),
                   kendall_tau_b=float(t.statistic), kendall_p=float(t.pvalue))
        if n <= SPEARMAN_EXACT_MAX_N and math.isfinite(out["spearman_rho"]):
            out.update(spearman_p=spearman_exact_p(q, y), spearman_p_method="exact")
    if n >= 6 and n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(int(n_boot), n))
        rho, tau, step = [], [], _boot_chunk(n)
        for a in range(0, idx.shape[0], step):           # chunks keep the n^2 arrays small
            sl = idx[a:a + step]
            rho.append(spearman_rows(q[sl], y[sl]))
            tau.append(kendall_b_rows(q[sl], y[sl]))
        for key, vals in (("spearman_ci95", np.concatenate(rho)), ("kendall_ci95", np.concatenate(tau))):
            vals = vals[np.isfinite(vals)]
            if vals.size >= 10:
                out[key] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        out["n_boot"] = int(n_boot)
    return out


def rank_metrics(rows, floor_s, top_ids, seed=0, n_boot=2000, unconfirmed=None,
                 min_n=MIN_TRUST_N):
    """Agreement between the QSS screen and the NLP on the confirmed rows.

    rows    : dicts with row_id, kind ('baseline' | 'top' | 'probe'), qss_lap,
              nlp_lap, branch_ok (True / False / None) and optionally nlp_lo /
              nlp_hi, the row's lap bracket (lap_bracket(); default the point
              nlp_lap); finish rows must be left out by the caller. Deltas are
              taken against the 'baseline' row.
    floor_s : the NLP noise floor (noise_floor()): two rows whose brackets are
              within it are unresolved.
    top_ids : the QSS top-k row ids (for nlp_best_in_qss_top_k).
    unconfirmed : the shortlisted rows without an accepted NLP lap (dicts with
              row_id, kind, qss_lap): they cannot be ranked, but when one of them is
              the QSS-best row the top-1 regret is unknown (it is never re-defined
              over the survivors).
    min_n   : the fewest confirmed rows that can give screen_trusted (rho = 1 occurs
              by chance with probability 1/n!: 1/6 at n = 3).
    Returns a dict: n; 'all', 'branch_ok' and 'within_shortlist' (baseline +
    top, range restricted) blocks with Spearman rho / Kendall tau-b, p-values
    and bootstrap 95% CIs; concordance over resolved pairs (brackets farther
    apart than the floor); qss_best_row (over the confirmed and unconfirmed
    rows), qss_best_confirmed, top1_regret_s (NLP lap of the QSS-best row minus
    the NLP-best lap; NaN when the QSS-best row is unconfirmed);
    nlp_best_in_qss_top_k; slope / intercept_s of NLP delta on QSS delta; sign
    agreement where a row's bracket is farther than the floor from the
    baseline's; n_unconfirmed / unconfirmed_rows; screen_trusted (n >= min_n,
    the QSS-best row confirmed, rho >= 0.9, concordance >= 0.9 and regret <=
    floor) and trust_reasons, the criteria that failed."""
    rows = [r for r in rows if _finite(r.get("qss_lap")) and _finite(r.get("nlp_lap"))]
    rows = sorted(rows, key=lambda r: r["row_id"])
    ids = [r["row_id"] for r in rows]
    q = np.array([float(r["qss_lap"]) for r in rows])
    y = np.array([float(r["nlp_lap"]) for r in rows])
    br = [(float(r["nlp_lo"]), float(r["nlp_hi"])) if _finite(r.get("nlp_lo")) and _finite(r.get("nlp_hi"))
          else (y[i], y[i]) for i, r in enumerate(rows)]
    unconf = sorted([r for r in (unconfirmed or []) if _finite(r.get("qss_lap"))],
                    key=lambda r: r["row_id"])
    floor_s = float(floor_s)
    out = dict(n=len(rows), floor_s=floor_s, row_ids=ids, n_unconfirmed=len(unconfirmed or []),
               unconfirmed_rows=sorted(r["row_id"] for r in (unconfirmed or [])))
    out["all"] = _corr_block(q, y, seed, n_boot)
    sel = [i for i, r in enumerate(rows) if r.get("branch_ok") is True]
    out["branch_ok"] = _corr_block(q[sel], y[sel], seed, n_boot)
    sel = [i for i, r in enumerate(rows) if r.get("kind") in ("baseline", "top")]
    out["within_shortlist"] = _corr_block(q[sel], y[sel], seed, n_boot,
                                          label="within shortlist (range restricted)")
    # resolved-pair concordance (disjoint brackets order the laps as the brackets)
    n_pairs = n_conc = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if bracket_gap(br[i], br[j]) > floor_s:
                n_pairs += 1
                n_conc += int((q[i] - q[j]) * (y[i] - y[j]) > 0)
    out["concordance"] = dict(value=(n_conc / n_pairs) if n_pairs else float("nan"),
                              n_pairs=n_pairs, n_concordant=n_conc)
    cand = [(q[i], ids[i], True) for i in range(len(rows))] + [
        (float(r["qss_lap"]), r["row_id"], False) for r in unconf]
    if cand:
        _, qss_best, confirmed = min(cand, key=lambda c: (c[0], c[1]))
        out["qss_best_row"], out["qss_best_confirmed"] = qss_best, confirmed
    else:
        out.update(qss_best_row=None, qss_best_confirmed=False)
    if rows:
        i_y = min(range(len(rows)), key=lambda i: (y[i], ids[i]))
        out["nlp_best_row"] = ids[i_y]
        out["top1_regret_s"] = (float(y[ids.index(out["qss_best_row"])] - y[i_y])
                                if out["qss_best_confirmed"] else float("nan"))
        out["nlp_best_in_qss_top_k"] = bool(ids[i_y] in set(top_ids))
    else:
        out.update(nlp_best_row=None, top1_regret_s=float("nan"), nlp_best_in_qss_top_k=False)
    base = [i for i, r in enumerate(rows) if r.get("kind") == "baseline"]
    slope = intercept = float("nan")
    agree = dict(value=float("nan"), n=0)
    if base:
        b = base[0]
        qd, yd = q - q[b], y - y[b]
        if np.unique(qd).size >= 2:
            slope, intercept = (float(v) for v in np.polyfit(qd, yd, 1))
        m = [i for i in range(len(rows)) if i != b and bracket_gap(br[i], br[b]) > floor_s]
        if m:
            agree = dict(value=float(np.mean([np.sign(qd[i]) == np.sign(yd[i]) for i in m])), n=len(m))
    out["slope"], out["intercept_s"], out["sign_agreement"] = slope, intercept, agree
    rho, conc, reg = out["all"]["spearman_rho"], out["concordance"]["value"], out["top1_regret_s"]
    why = []
    if len(rows) < int(min_n):
        why.append(f"only {len(rows)} confirmed row(s), fewer than {int(min_n)}")
    if out["qss_best_row"] is not None and not out["qss_best_confirmed"]:
        why.append(f"the QSS-best row {out['qss_best_row']} has no confirmed NLP lap, so the "
                   "top-1 regret is unknown")
    elif not (_finite(reg) and reg <= floor_s):
        why.append(f"top-1 regret {1e3 * reg:.3f} ms > noise floor {1e3 * floor_s:.3f} ms")
    if not (_finite(rho) and rho >= 0.9):
        why.append(f"Spearman {rho:.3f} < 0.9")
    if not (_finite(conc) and conc >= 0.9):
        why.append(f"concordance {conc:.3f} < 0.9 ({n_pairs} resolved pairs)")
    out["screen_trusted"] = not why
    out["trust_reasons"] = why
    out["screen_trusted_rule"] = (f"n >= {int(min_n)} and the QSS-best shortlisted row confirmed and "
                                  "spearman_rho >= 0.9 and concordance >= 0.9 and top1_regret_s <= floor_s")
    return out


# ============================================================================
# 3. Identity: fingerprints and code hashes
# ============================================================================
def _plain(obj):
    """JSON-ready copy: numpy scalars / arrays, tuples, sets and namespaces
    become Python scalars, lists and dicts (keys as str)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_plain(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "__dict__") and type(obj).__name__ in ("SimpleNamespace", "Ctx"):
        return _plain(vars(obj))
    return obj


def fingerprint(obj):
    """sha256 hex of json.dumps(obj, sort_keys=True, default=repr,
    separators=(',', ':')) after numpy / tuple normalisation, so dict key
    order never matters."""
    text = json.dumps(_plain(obj), sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    """sha256 hex of the float64 bytes of ``a`` (e.g. an NLP w_opt)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


def model_hash(paths, root=None):
    """sha256 over (relative path, file bytes) of ``paths`` in order (a missing
    file hashes as a marker), i.e. the code and data a solve depends on."""
    h = hashlib.sha256()
    for p in paths:
        full = p if root is None or os.path.isabs(p) else os.path.join(root, p)
        h.update(str(p).replace("\\", "/").encode("utf-8") + b"\0")
        try:
            with open(full, "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"<missing>")
        h.update(b"\0")
    return h.hexdigest()


# ============================================================================
# 4. Worker sizing and process environment
# ============================================================================
class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_ram_mb():
    """Available physical memory [MiB] (Windows GlobalMemoryStatusEx; None
    elsewhere or on failure, since the venv has no psutil)."""
    if not sys.platform.startswith("win"):
        return None
    try:
        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullAvailPhys / 2.0 ** 20
    except Exception:
        return None


def auto_workers(n_tasks, rss_mb=None):
    """Worker count for a process pool: max(1, min(n_tasks, cpu_count//2 - 1,
    floor(0.7 * free RAM / rss_mb) when both are known)). cpu_count//2 skips
    SMT siblings (MA57 / the march are compute bound); one core stays free."""
    cap = [max(int(n_tasks), 1), (os.cpu_count() or 2) // 2 - 1]
    free = free_ram_mb()
    if rss_mb and free:
        cap.append(int(math.floor(0.7 * free / float(rss_mb))))
    return max(1, min(cap))


@contextlib.contextmanager
def blas_single_thread():
    """Set OPENBLAS / OMP / MKL _NUM_THREADS = '1' for child processes created
    inside the block (they read them before numpy / casadi load, so results do
    not depend on the worker count); the previous values are restored."""
    old = {k: os.environ.get(k) for k in BLAS_VARS}
    try:
        for k in BLAS_VARS:
            os.environ[k] = "1"
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ============================================================================
# 5. Atomic writers / readers
# ============================================================================
def dedup(items):
    """Unique items in first-seen order."""
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _json_clean(obj):
    """_plain() with non-finite floats as None (strict JSON)."""
    obj = _plain(obj)
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def replace_retry(src, dst, tries=None, wait_s=None, max_wait_s=None, discard=False):
    """os.replace(src, dst) that survives a target another program holds open for a
    moment. On Windows os.replace raises PermissionError (WinError 5 / 32) while a
    viewer, an editor, a spreadsheet or a virus scanner has ``dst`` open; such a lock
    is usually short, so a failed attempt is retried: ``tries`` attempts in all
    (default REPLACE_TRIES), pausing wait_s, 2 wait_s, ... between them (default
    REPLACE_WAIT_S, capped at REPLACE_MAX_WAIT_S). When the last attempt fails too, a
    PermissionError that names ``dst`` and what to do (chained to the original) is
    raised. Any other error is raised at once, not retried. ``discard`` removes ``src``
    when an error leaves this function (a temporary file; otherwise it is left for the
    caller)."""
    tries = max(REPLACE_TRIES if tries is None else int(tries), 1)
    wait_s = REPLACE_WAIT_S if wait_s is None else float(wait_s)
    cap = REPLACE_MAX_WAIT_S if max_wait_s is None else float(max_wait_s)
    try:
        for k in range(tries):
            try:
                os.replace(src, dst)
                return
            except PermissionError as exc:
                if k < tries - 1:
                    time.sleep(min(wait_s * 2.0 ** k, cap))
                    continue
                raise PermissionError(
                    exc.errno, f"cannot replace {dst}: {exc.strerror or exc} after {tries} attempts. On "
                               "Windows another program has the file open: close the viewer, editor or "
                               "spreadsheet that shows it and run again (a sweep call with resume=True "
                               "continues from the rows on disk); elsewhere check the folder "
                               "permissions") from exc
    except BaseException:
        if discard:
            with contextlib.suppress(OSError):
                os.remove(src)
        raise


def _replace_into(path, data, mode):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": ""})) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    replace_retry(tmp, path, discard=True)


def write_text_atomic(path, text):
    """Write a text file via a temporary file + os.replace."""
    _replace_into(path, text, "w")


def write_json_atomic(path, obj):
    """Write ``obj`` as strict JSON (NaN / inf -> null; floats via repr, so they
    round-trip exactly) to a temporary file and os.replace it into ``path``."""
    text = json.dumps(_json_clean(obj), indent=1, allow_nan=False, default=repr)
    _replace_into(path, text + "\n", "w")


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return repr(v) if math.isfinite(v) else ""
    if isinstance(v, (list, tuple)):
        return ";".join(_cell(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(_json_clean(v), sort_keys=True, separators=(",", ":"))
    return str(v)


def write_csv_atomic(path, header, rows):
    """stdlib csv (no pandas): ``rows`` are dicts keyed by ``header`` (or
    sequences); floats via repr (exact round trip), NaN / None as an empty
    cell, bools as True / False, lists ';'-joined. Atomic (tmp + os.replace)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(header))
    for r in rows:
        vals = [r.get(h) for h in header] if isinstance(r, dict) else list(r)
        w.writerow([_cell(v) for v in vals])
    _replace_into(path, buf.getvalue(), "w")


def read_csv(path):
    """List of {column: str} dicts."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def num(s):
    """Float from a CSV / JSON cell ('' / None -> nan)."""
    if s is None or s == "":
        return float("nan")
    return float(s)
