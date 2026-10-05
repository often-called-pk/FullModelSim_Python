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
                      bootstrap CIs, resolved-pair concordance, regret, slope ...
  noise_floor(), unresolved_pairs()
  fingerprint(), model_hash(), sha256_file(), array_sha()
  free_ram_mb(), auto_workers(), blas_single_thread()
  write_text_atomic(), write_json_atomic(), read_json(), write_csv_atomic(), read_csv(),
  num(), dedup()
"""

import contextlib
import csv
import ctypes
import hashlib
import io
import json
import math
import os
import sys
import warnings

import numpy as np

SAMPLERS = ("sobol", "lhs")
BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
UNKNOWN_REASON = "unknown (vp_overrides would raise)"


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


def sample_box(specs, n, sampler="sobol", seed=0):
    """Space-filling design over the box of ``specs``: returns (U, X), both
    (n, d): U in the unit cube, X = qmc.scale(U, lower, upper).

    sampler='sobol': qmc.Sobol(d, scramble=True, rng=default_rng(seed));
    random_base2(m) when n == 2**m, else a UserWarning (the balance properties
    need a power of 2) and random(n). sampler='lhs': qmc.LatinHypercube(d,
    rng=default_rng(seed)).random(n). The same (specs, n, sampler, seed) gives
    the same design (the stored samples.csv of a sweep stays authoritative)."""
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
        eng = qmc.Sobol(d, scramble=True, rng=rng)
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
        U = qmc.LatinHypercube(d, rng=rng).random(n)
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
    """max(resolve_s, max |rt_residual_s| over rows with branch_ok): two NLP laps
    closer than this cannot be ordered (round-trip residuals measure how far a
    warm hop and its way back disagree, i.e. the continuation noise)."""
    res = [abs(float(r["rt_residual_s"])) for r in rows
           if r.get("branch_ok") is True and _finite(r.get("rt_residual_s"))]
    return max([float(resolve_s)] + res)


def unresolved_pairs(laps, floor):
    """{row_id: sorted ids whose lap is within ``floor`` of this row's} for
    laps = {row_id: lap} (finite laps only)."""
    items = [(rid, float(v)) for rid, v in laps.items() if _finite(v)]
    out = {}
    for rid, v in items:
        out[rid] = sorted(o for o, w in items if o != rid and abs(w - v) <= floor)
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


def _corr_block(q, y, seed, n_boot, label=None):
    """Spearman rho / Kendall tau-b with scipy p-values, plus percentile
    bootstrap 95% CIs (``n_boot`` resamples of the rows, seeded; skipped when
    n < 6) for one subset."""
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
    if n >= 6 and n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(int(n_boot), n))
        rho, tau = [], []
        for a in range(0, idx.shape[0], 256):            # chunks keep the n^2 arrays small
            sl = idx[a:a + 256]
            rho.append(spearman_rows(q[sl], y[sl]))
            tau.append(kendall_b_rows(q[sl], y[sl]))
        for key, vals in (("spearman_ci95", np.concatenate(rho)), ("kendall_ci95", np.concatenate(tau))):
            vals = vals[np.isfinite(vals)]
            if vals.size >= 10:
                out[key] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        out["n_boot"] = int(n_boot)
    return out


def rank_metrics(rows, floor_s, top_ids, seed=0, n_boot=2000):
    """Agreement between the QSS screen and the NLP on the confirmed rows.

    rows    : dicts with row_id, kind ('baseline' | 'top' | 'probe'), qss_lap,
              nlp_lap and branch_ok (True / False / None); finish rows must be
              left out by the caller. Deltas are taken against the 'baseline' row.
    floor_s : the NLP noise floor (noise_floor()): pairs / deltas closer than
              this count as unresolved.
    top_ids : the QSS top-k row ids (for nlp_best_in_qss_top_k).
    Returns a dict: n; 'all', 'branch_ok' and 'within_shortlist' (baseline +
    top, range restricted) blocks with Spearman rho / Kendall tau-b, p-values
    and bootstrap 95% CIs; concordance over resolved pairs (|dNLP| > floor);
    top1_regret_s (NLP lap of the QSS-best row minus the NLP-best lap);
    nlp_best_in_qss_top_k; slope / intercept_s of NLP delta on QSS delta;
    sign agreement where |NLP delta| > floor; screen_trusted (rho >= 0.9 and
    concordance >= 0.9 and regret <= floor)."""
    rows = [r for r in rows if _finite(r.get("qss_lap")) and _finite(r.get("nlp_lap"))]
    rows = sorted(rows, key=lambda r: r["row_id"])
    ids = [r["row_id"] for r in rows]
    q = np.array([float(r["qss_lap"]) for r in rows])
    y = np.array([float(r["nlp_lap"]) for r in rows])
    floor_s = float(floor_s)
    out = dict(n=len(rows), floor_s=floor_s, row_ids=ids)
    out["all"] = _corr_block(q, y, seed, n_boot)
    sel = [i for i, r in enumerate(rows) if r.get("branch_ok") is True]
    out["branch_ok"] = _corr_block(q[sel], y[sel], seed, n_boot)
    sel = [i for i, r in enumerate(rows) if r.get("kind") in ("baseline", "top")]
    out["within_shortlist"] = _corr_block(q[sel], y[sel], seed, n_boot,
                                          label="within shortlist (range restricted)")
    # resolved-pair concordance
    n_pairs = n_conc = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dy = y[i] - y[j]
            if abs(dy) > floor_s:
                n_pairs += 1
                n_conc += int((q[i] - q[j]) * dy > 0)
    out["concordance"] = dict(value=(n_conc / n_pairs) if n_pairs else float("nan"),
                              n_pairs=n_pairs, n_concordant=n_conc)
    if rows:
        i_q = min(range(len(rows)), key=lambda i: (q[i], ids[i]))
        i_y = min(range(len(rows)), key=lambda i: (y[i], ids[i]))
        out["qss_best_row"], out["nlp_best_row"] = ids[i_q], ids[i_y]
        out["top1_regret_s"] = float(y[i_q] - y[i_y])
        out["nlp_best_in_qss_top_k"] = bool(ids[i_y] in set(top_ids))
    else:
        out.update(qss_best_row=None, nlp_best_row=None, top1_regret_s=float("nan"),
                   nlp_best_in_qss_top_k=False)
    base = [i for i, r in enumerate(rows) if r.get("kind") == "baseline"]
    slope = intercept = float("nan")
    agree = dict(value=float("nan"), n=0)
    if base:
        qd, yd = q - q[base[0]], y - y[base[0]]
        if np.unique(qd).size >= 2:
            slope, intercept = (float(v) for v in np.polyfit(qd, yd, 1))
        m = [i for i in range(len(rows)) if i != base[0] and abs(yd[i]) > floor_s]
        if m:
            agree = dict(value=float(np.mean([np.sign(qd[i]) == np.sign(yd[i]) for i in m])), n=len(m))
    out["slope"], out["intercept_s"], out["sign_agreement"] = slope, intercept, agree
    rho, conc, reg = out["all"]["spearman_rho"], out["concordance"]["value"], out["top1_regret_s"]
    out["screen_trusted"] = bool(_finite(rho) and _finite(conc) and _finite(reg)
                                 and rho >= 0.9 and conc >= 0.9 and reg <= floor_s)
    out["screen_trusted_rule"] = "spearman_rho >= 0.9 and concordance >= 0.9 and top1_regret_s <= floor_s"
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


def _replace_into(path, data, mode):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": ""})) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


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
