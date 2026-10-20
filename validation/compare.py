"""Gate 1 diff: the MATLAB parameter dump (validation/matlab/vexport.m) against the Python one
(validation/py_export.params). Exact for ints and strings, floats at rel 1e-12 (NaN equals NaN).

    diff_params(mat, py, info=None) -> [(key, detail), ...]   one hit per differing leaf, dotted keys

IPOPT options the MATLAB file leaves unset are compared with the IPOPT defaults; options only Python sets
and that do not change the iterates (print_timing_statistics, HSL tuning) go to ``info`` (a list), not hits.

    python validation/compare.py MATLAB_PARAMS.json --circuit BCN --opt-ds 10      # prints the hits

Gate 2 (NLP functions, docs/validation_matlab_vs_python.md section 5):

    nlp_gate(mat_path, py) -> dict   the MATLAB NLP export (vexport.m) against py_export.nlp(...)
    python validation/compare.py nlp MATLAB_NLP.mat --circuit BCN --opt-ds 10 [--bound-tol REL] [--save OUT.npz]

The bounds match exactly (--bound-tol 0, the default: vehModel.py:88 multiplies by 1/x_s as vehModel.m:130-138 does)
and the start point w0 is exact under py_export.matlab_seed(); see tests/test_matlab_parity.py.

Run matrix (docs section 7-9; files <track>_mat_ship_r<k>, <track>_py_{par,prod,prod8}_r<k> .mat + .json from
matlab_batch.py solve and run_py.py in one directory D):

    python validation/compare.py runs --track BCN --dir D
        markdown: runs, parity gap per rep pair (pass: rel <= 1e-4 of the MATLAB lap), RMS on a 1 m grid against the
        reference (mat_ship r1 until a reference exists), production lap error (band 0.5 %), speed table (median (min,
        max) per tag); the numbers also go to D/<track>_compare.json
    python validation/compare.py xcheck --track BCN --mat <mat run>.mat --py <py run>.mat
        each code's w* evaluated in the other code's NLP (Python build, then one MATLAB export; no solve); pass:
        max bound/constraint violation <= 1e-6 (normalised NLP units) and objective rel <= 1e-9 against the producing
        code's f; D/<track>_xcheck_<pair>.json, D = folder of the mat run
    python validation/compare.py reference --track BCN --dir D
        lowest-lap converged mat_ship / py_par run whose xcheck passed -> D/reference/<track>/summary.json + its .mat
"""
import argparse
import gc
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
RTOL = 1e-12
NLP_TOL = 1e-10
IPOPT_DEFAULTS = {"linear_solver": "mumps", "mu_strategy": "monotone"}      # of the options userOpts.m leaves unset
HARMLESS = ("print_timing_statistics", "ma27_", "ma57_", "ma86_", "ma97_", "hsllib")


def _flat(x):
    if isinstance(x, (list, tuple)):
        return [leaf for v in x for leaf in _flat(v)]
    return [x]


def _leaf_same(a, b):
    if a is None or b is None or isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        return a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b or abs(a - b) <= RTOL * max(abs(a), abs(b))


def _same(a, b):
    fa, fb = _flat(a), _flat(b)
    return len(fa) == len(fb) and all(_leaf_same(u, v) for u, v in zip(fa, fb))


def _short(x, n=4):
    f = _flat(x)
    return repr(x) if len(f) <= n else f"[{len(f)} values: {', '.join(map(repr, f[:n]))}, ...]"


def _walk(path, a, b, hits):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            sub = f"{path}.{k}" if path else k
            if k not in b:
                hits.append((sub, f"only in matlab: {_short(a[k])}"))
            elif k not in a:
                hits.append((sub, f"only in python: {_short(b[k])}"))
            else:
                _walk(sub, a[k], b[k], hits)
    elif isinstance(a, dict) or isinstance(b, dict) or not _same(a, b):
        hits.append((path, f"matlab={_short(a)} python={_short(b)}"))


def _ipopt(m, p, hits, info):
    for k in sorted(set(m) | set(p)):
        if k in m and k in p:
            _walk(f"ipopt.{k}", m[k], p[k], hits)
        elif k in m:
            hits.append((f"ipopt.{k}", f"only in matlab: {m[k]!r}"))
        elif k in IPOPT_DEFAULTS:
            if not _same(IPOPT_DEFAULTS[k], p[k]):
                hits.append((f"ipopt.{k}", f"matlab unset (IPOPT default {IPOPT_DEFAULTS[k]!r}) python={p[k]!r}"))
        elif k.startswith(HARMLESS):
            if info is not None:
                info.append((f"ipopt.{k}", f"only python: {p[k]!r}"))
        else:
            hits.append((f"ipopt.{k}", f"only in python: {p[k]!r}"))


def diff_params(mat, py, info=None):
    hits = []
    _walk("", {k: v for k, v in mat.items() if k != "ipopt"}, {k: v for k, v in py.items() if k != "ipopt"}, hits)
    _ipopt(mat.get("ipopt", {}), py.get("ipopt", {}), hits, info)
    return hits


def _scaled(a, b, floor=1.0):
    """max |a - b| / max(floor, |a|) over the elements (floor 0: relative). Equal values, infs included, count as
    0; different shapes or NaN positions that differ give inf."""
    a, b = np.atleast_1d(np.asarray(a, dtype=float)), np.atleast_1d(np.asarray(b, dtype=float))
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return math.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(a == b, 0.0, np.abs(a - b) / np.maximum(floor, np.abs(a)))
    d[np.isnan(a)] = 0.0
    return float(d.max()) if d.size else 0.0


def nlp_gate(mat_path, py, bound_tol=0.0):
    """Gate 2: the MATLAB NLP export at ``mat_path`` (lbw ubw lbg ubg w0 W W_names F G) against ``py``, the dict of
    py_export.nlp(circuit, opt_ds, W=<that W>). Returns
        bounds   {lbw, ubw, lbg, ubg: max |a-b| / |a|}   0.0 = exact (equal infs are equal), inf if the sizes differ
        columns  [{name, dF, dG}, ...]                   per column of W: max |a-b| / max(1, |a|) over F and over g
                                                         (a = MATLAB); NaN positions must match, else inf
        w0       max |w0_py - w0_mat|                    info, inf if the sizes differ
        n_w, n_g (matlab, python) sizes                  info
        ok       every bound <= bound_tol (0: exact, as docs section 5) and every dF, dG <= NLP_TOL (1e-10)"""
    mat = sio.loadmat(mat_path, squeeze_me=True)
    F, G = np.atleast_1d(mat["F"]), np.asarray(mat["G"], dtype=float).reshape(np.size(mat["lbg"]), -1)
    w0_mat, w0_py = np.ravel(mat["w0"]), np.ravel(py["w0"])
    out = dict(
        bounds={k: _scaled(mat[k], py[k], floor=0.0) for k in ("lbw", "ubw", "lbg", "ubg")},
        columns=[dict(name=str(n), dF=_scaled(F[j], py["F"][j]), dG=_scaled(G[:, j], py["G"][:, j]))
                 for j, n in enumerate(np.atleast_1d(mat["W_names"]))],
        w0=float(np.max(np.abs(w0_mat - w0_py))) if w0_mat.size == w0_py.size else math.inf,
        n_w=(np.size(mat["lbw"]), np.size(py["lbw"])), n_g=(np.size(mat["lbg"]), np.size(py["lbg"])))
    out["ok"] = (all(v <= bound_tol for v in out["bounds"].values())
                 and all(max(c["dF"], c["dG"]) <= NLP_TOL for c in out["columns"]))
    return out


def nlp_main(argv):
    ap = argparse.ArgumentParser(prog="compare.py nlp", description="gate 2: MATLAB NLP export against the Python NLP")
    ap.add_argument("matlab_mat")
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--opt-ds", type=float, required=True)
    ap.add_argument("--bound-tol", type=float, default=0.0, help="relative tolerance on the bounds (default 0: exact)")
    ap.add_argument("--save", help="also write the Python evaluation (lbw ubw lbg ubg w0 F G) to this .npz")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from validation.py_export import nlp
    t0 = time.perf_counter()
    py = nlp(a.circuit, a.opt_ds, W=sio.loadmat(a.matlab_mat, squeeze_me=True)["W"])
    wall = time.perf_counter() - t0
    if a.save:
        np.savez_compressed(a.save, **py)
    g = nlp_gate(a.matlab_mat, py, a.bound_tol)
    print(f"n_w matlab {g['n_w'][0]} python {g['n_w'][1]}, n_g matlab {g['n_g'][0]} python {g['n_g'][1]}")
    print("bounds, max |a-b|/|a| (0 = exact): " + ", ".join(f"{k} {v:.3e}" for k, v in g["bounds"].items()))
    print(f"max |w0_py - w0_mat| = {g['w0']:.3e} (info)")
    for c in g["columns"]:
        verdict = "PASS" if max(c["dF"], c["dG"]) <= NLP_TOL else "FAIL"
        print(f"column {c['name']:8s} dF {c['dF']:.3e}  dG {c['dG']:.3e}  {verdict}")
    print(f"python build + evaluation {wall:.1f} s;  gate 2 {a.circuit} OPT_ds {a.opt_ds:g} "
          f"(bound tol {a.bound_tol:g}): {'PASS' if g['ok'] else 'FAIL'}")
    return 0 if g["ok"] else 1


# ============================================================================
# Run matrix (docs sections 7-9): runs, xcheck, reference
# ============================================================================
TAGS = ("mat_ship", "py_par", "py_prod", "py_prod8")
GOOD_STATUS = ("Solve_Succeeded", "Solved_To_Acceptable_Level")        # IPOPT's successful returns
PARITY_REL = 1e-4                   # doc section 8: lap |dT| <= 0.01 % of the lap
PROD_BAND = 5e-3                    # production lap error, sanity band 0.5 %
RMS_FLAG = {"vx": 0.1, "n": 0.05}   # m/s, m: flagged, not failed
VIOL_TOL, OBJ_TOL = 1e-6, 1e-9      # cross-feasibility (xcheck): violation in normalised NLP units, objective rel


def _g(obj, *path):
    """obj[path[0]][path[1]]... or None where a key or index is missing."""
    for p in path:
        try:
            obj = obj[p]
        except (KeyError, IndexError, TypeError):
            return None
    return obj


def _dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                          encoding="utf-8")


def _fmt(v, spec=".8g"):                   # 8 digits: a lap of 116.44076 s resolves 0.01 % and the 1e-7 s below it
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def _table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def load_runs(track, d):
    """{tag: {rep: run}} of the files in ``d`` (names: matlab_batch.py solve, run_py.py). run = dict(tag, rep, stem, json,
    mat (path or None for a run that failed), ok, status, iters, lap, f, N, OPT_ds)."""
    runs = {}
    for tag in TAGS:
        for jf in sorted(Path(d).glob(f"{track}_{tag}_r*.json")):
            m = re.fullmatch(rf"{re.escape(track)}_{tag}_r(\d+)", jf.stem)
            if not m:
                continue
            J = json.loads(jf.read_text(encoding="utf-8"))
            st, mat = J.get("stats") or {}, jf.with_suffix(".mat")
            runs.setdefault(tag, {})[int(m.group(1))] = dict(
                tag=tag, rep=int(m.group(1)), stem=jf.stem, json=J, mat=mat if mat.exists() else None, ok=bool(J.get("ok")),
                status=st.get("return_status") or ("killed" if J.get("timed_out") else "FAILED"),
                iters=st.get("iter_count"), lap=J.get("lap"), f=J.get("f"), N=J.get("N"), OPT_ds=J.get("OPT_ds"))
    return runs


def _poly(sk, xk, xc, s):
    """One state at the arc lengths ``s``: per interval the Lagrange polynomial through the knot value xk and the d Legendre
    collocation values xc (columns d*k..d*k+d-1 of interval k), i.e. the NLP's own state polynomial, not a chord."""
    N = sk.size - 1
    d = xc.size // N
    nodes = np.r_[0.0, (np.polynomial.legendre.leggauss(d)[0] + 1.0) / 2.0]
    i = np.clip(np.searchsorted(sk, s, side="right") - 1, 0, N - 1)
    t = (s - sk[i]) / (sk[i + 1] - sk[i])
    vals = np.column_stack([xk[i]] + [xc[d * i + j] for j in range(d)])
    out = np.zeros(s.size)
    for a in range(d + 1):
        w = np.ones(s.size)
        for b in range(d + 1):
            if b != a:
                w *= (t - nodes[b]) / (nodes[a] - nodes[b])
        out += vals[:, a] * w
    return out


def _rms(mat_a, mat_b):
    """RMS difference of run a against run b on a 1 m grid over their common s range: vx [m/s], n [m], delta_n = steering row
    of u_opt / (pi/8) (the normalised steering, both codes), T_motor and T_brake [Nm] (rows 0 and 1 of u_opt, EM4 Off).
    States follow the collocation polynomials, inputs are linear between knots (OPT_uinter 'linear')."""
    A, B = (sio.loadmat(p, squeeze_me=True) for p in (mat_a, mat_b))
    sa, sb = (np.atleast_1d(M["s_knot"]).astype(float) for M in (A, B))
    s = np.arange(np.ceil(max(sa[0], sb[0])), np.floor(min(sa[-1], sb[-1])) + 0.5, 1.0)

    def prof(M, sk):
        x, xc, u = (np.atleast_2d(M[k]).astype(float) for k in ("x_opt", "xc_opt", "u_opt"))
        row = lambda r: np.interp(s, sk, u[r])
        return dict(vx=_poly(sk, x[0], xc[0], s), n=_poly(sk, x[3], xc[3], s), delta_n=row(-1) / (np.pi / 8),
                    T_motor=row(0), T_brake=row(1))

    pa, pb = prof(A, sa), prof(B, sb)
    return {k: float(np.sqrt(np.mean((pa[k] - pb[k]) ** 2))) for k in pa}


def _reference_run(track, d, runs):
    """What profiles and production laps are scored against: the bundle D/reference/<track> once `compare.py reference`
    wrote it, else mat_ship r1 (the name says which). dict(name, stem, mat, lap) or None."""
    summ = Path(d) / "reference" / track / "summary.json"
    if summ.exists():
        S = json.loads(summ.read_text(encoding="utf-8"))
        return dict(name=f"the reference bundle (source {S['source_run']})", stem=S["source_run"],
                    mat=summ.parent / S["source_mat"], lap=S["lap"])
    r = runs.get("mat_ship", {}).get(1)
    if r and r["mat"]:
        return dict(name=f"{r['stem']} (no reference bundle yet)", stem=r["stem"], mat=r["mat"], lap=r["lap"])
    return None


def _spi(J):
    st = J.get("stats") or {}
    return st["t_wall_total"] / st["iter_count"] if st.get("t_wall_total") is not None and st.get("iter_count") else None


def _fn_share(J):
    st = J.get("stats") or {}
    w = st.get("t_wall_total")
    return sum(v for k, v in st.items() if k.startswith("t_wall_nlp_")) / w if w else None


def _init_wall(J):
    e = _g(J, "init", "elapsedTime")
    return e[1] + e[2] if e and len(e) >= 3 else None


def _gb(v):
    return None if v is None else v / 2 ** 30


# label, MATLAB getter (mat_ship JSON), Python getter (py_par, py_prod, py_prod8 JSON); None = the code has no such timer
SPEED = (
    ("external wall [s]", lambda J: J.get("wall_external_s"), lambda J: J.get("wall_external_s")),
    ("startup / import [s]", lambda J: J.get("startup_s"), lambda J: _g(J, "timers", "import_s")),
    ("child wall, start to result [s]", None, lambda J: _g(J, "timers", "child_wall_s")),
    ("model build, vehModel [s]", None, lambda J: _g(J, "timers", "model_build_s")),
    ("NLP build [s]", None, lambda J: _g(J, "timers", "nlp_build_s")),
    ("transcription: model + NLP build [s]", lambda J: _g(J, "elapsedTime", 1), lambda J: _g(J, "timers", "transcription_s")),
    ("solver call, MATLAB elapsedTime(3) [s]", lambda J: _g(J, "elapsedTime", 2), None),
    ("7-state init iterations", lambda J: _g(J, "init", "stats", "iter_count"), lambda J: _g(J, "timers", "init7_iters")),
    ("7-state init IPOPT wall [s]", lambda J: _g(J, "init", "stats", "t_wall_total"), lambda J: _g(J, "timers", "init7_ipopt_s")),
    ("7-state init wall [s]", _init_wall, lambda J: _g(J, "timers", "init7_wall_s")),
    ("23-state IPOPT iterations", lambda J: _g(J, "stats", "iter_count"), lambda J: _g(J, "stats", "iter_count")),
    ("23-state IPOPT wall [s]", lambda J: _g(J, "stats", "t_wall_total"), lambda J: _g(J, "stats", "t_wall_total")),
    ("IPOPT s per iteration", _spi, _spi),
    ("function-evaluation share of IPOPT wall", _fn_share, _fn_share),
    ("peak working set [GB]", lambda J: _gb(J.get("peak_ws_bytes")), lambda J: _gb(J.get("peak_ws_bytes"))),
    ("peak commit [GB]", None, lambda J: _gb(J.get("peak_commit_bytes"))),
)


def _stat(vals):
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return dict(n=len(v), median=float(np.median(v)), min=min(v), max=max(v), values=v) if v else None


def _cell(s):
    if s is None:
        return "-"
    return _fmt(s["median"], ".4g") if s["n"] == 1 else f"{_fmt(s['median'], '.4g')} ({_fmt(s['min'], '.4g')}, {_fmt(s['max'], '.4g')})"


def runs_main(argv):
    ap = argparse.ArgumentParser(prog="compare.py runs", description="markdown report of the run matrix of one track")
    ap.add_argument("--track", required=True)
    ap.add_argument("--dir", required=True, help="folder with the <track>_{mat_ship,py_par,py_prod,py_prod8}_r<k> .mat + .json files")
    a = ap.parse_args(argv)
    d, T = Path(a.dir), a.track
    runs = load_runs(T, d)
    if not runs:
        print(f"no {T} runs in {d}")
        return 1
    every = [r for tag in TAGS for _, r in sorted(runs.get(tag, {}).items())]
    ref = _reference_run(T, d, runs)
    out = dict(track=T, dir=str(d), reference=ref and dict(name=ref["name"], stem=ref["stem"], lap=ref["lap"]))

    print(f"## {T}: runs in {d}\n")
    print(_table(("run", "status", "lap [s]", "objective f", "IPOPT iterations", "N", "OPT_ds", "IPOPT wall [s]"),
                 [(r["stem"], r["status"], _fmt(r["lap"]), _fmt(r["f"]), _fmt(r["iters"]), _fmt(r["N"]), _fmt(r["OPT_ds"]),
                   _fmt(_g(r["json"], "stats", "t_wall_total"), ".5g")) for r in every]))
    out["runs"] = [dict(run=r["stem"], tag=r["tag"], rep=r["rep"], ok=r["ok"], status=r["status"], lap=r["lap"], f=r["f"],
                        iters=r["iters"], N=r["N"], OPT_ds=r["OPT_ds"], ipopt_wall_s=_g(r["json"], "stats", "t_wall_total"))
                   for r in every]
    for r in (r for r in every if not r["ok"]):
        print(f"\nFAILED {r['stem']}: {((r['json'].get('error') or '').strip() or 'no error text').splitlines()[-1]}")

    print("\n### Parity: mat_ship vs py_par per rep pair (pass: |dT| / MATLAB lap <= 1e-4)\n")
    out["parity"], rows = [], []
    for k in sorted(set(runs.get("mat_ship", {})) & set(runs.get("py_par", {}))):
        m, p = runs["mat_ship"][k], runs["py_par"][k]
        if m["lap"] is None or p["lap"] is None:
            rows.append((k, "-", "-", "-", "-", "-", "n/a (a run failed)"))
            continue
        gap = p["lap"] - m["lap"]
        rel, ok = abs(gap) / m["lap"], abs(gap) / m["lap"] <= PARITY_REL
        out["parity"].append(dict(rep=k, lap_mat=m["lap"], lap_py=p["lap"], gap_s=gap, abs_s=abs(gap), rel=rel, status_mat=m["status"],
                                  status_py=p["status"], **{"pass": ok}))
        rows.append((k, _fmt(m["lap"]), _fmt(p["lap"]), _fmt(abs(gap), ".3e"), _fmt(100 * rel, ".4f") + " %",
                     f"{m['status']} / {p['status']}", "PASS" if ok else "FAIL"))
    print(_table(("rep", "lap mat [s]", "lap py [s]", "abs [s]", "rel", "status mat / py", "parity"), rows)
          if rows else "no rep pair (needs mat_ship and py_par of the same rep)")

    print(f"\n### Profile RMS on a 1 m grid against {ref['name'] if ref else 'no reference'} (flag: vx > 0.1 m/s, n > 0.05 m)\n")
    out["rms"], rows = [], []
    for r in every:
        if ref is None or r["mat"] is None or r["stem"] == ref["stem"]:
            continue
        e = _rms(r["mat"], ref["mat"])
        flags = [f"{k} > {RMS_FLAG[k]:g}" for k in RMS_FLAG if e[k] > RMS_FLAG[k]]
        out["rms"].append(dict(run=r["stem"], **e, flags=flags))
        rows.append((r["stem"], *(_fmt(e[k], ".4g") for k in ("vx", "n", "delta_n", "T_motor", "T_brake")), ", ".join(flags) or "ok"))
    print(_table(("run", "vx [m/s]", "n [m]", "delta_n [-]", "T_motor [Nm]", "T_brake [Nm]", "flags"), rows)
          if rows else "nothing to compare (needs a reference and another run with a .mat)")

    print(f"\n### Production lap error against {ref['name'] if ref else 'no reference'} (signed, band 0.5 %)\n")
    out["production"], rows = [], []
    for r in (x for x in every if x["tag"] in ("py_prod", "py_prod8")):       # `every` is ordered by tag, then rep
        if ref is None or r["lap"] is None:
            continue
        err = r["lap"] - ref["lap"]
        flag = abs(err) / ref["lap"] > PROD_BAND
        out["production"].append(dict(run=r["stem"], lap=r["lap"], err_s=err, err_pct=100 * err / ref["lap"], flag=flag))
        rows.append((r["stem"], _fmt(r["lap"]), f"{err:+.4f}", f"{100 * err / ref['lap']:+.3f} %", "FLAG" if flag else "ok"))
    print(_table(("run", "lap [s]", "error [s]", "error", "band 0.5 %"), rows) if rows else "no py_prod / py_prod8 run with a lap")

    print("\n### Speed: median (min, max) over the reps\n")
    out["speed"] = {}
    for t in TAGS:
        col = 1 if t == "mat_ship" else 2                           # getter column of SPEED
        js = [r["json"] for r in runs.get(t, {}).values() if r["ok"]]
        out["speed"][t] = {s[0]: _stat([s[col](J) for J in js]) if s[col] else None for s in SPEED}
    print(_table(("metric", *(f"{t} (n={sum(r['ok'] for r in runs.get(t, {}).values())})" for t in TAGS)),
                 [(label, *(_cell(out["speed"][t][label]) for t in TAGS)) for label, *_ in SPEED]))
    _dump(d / f"{T}_compare.json", out)
    print(f"\nwrote {d / f'{T}_compare.json'}")
    return 0


def _check(w, F, G, bounds, f_ref):
    """One evaluation of an NLP at the point w: max bound and constraint violation in the NLP's normalised units (lbw/ubw,
    lbg/ubg of that NLP) and the objective against the f the producing code reported; inf where g is not finite."""
    lbw, ubw, lbg, ubg = bounds
    if not np.all(np.isfinite(G)):
        bv = gv = math.inf
    else:
        bv = float(max(0.0, np.max(lbw - w), np.max(w - ubw)))
        gv = float(max(0.0, np.max(lbg - G), np.max(G - ubg)))
    rel = abs(F - f_ref) / abs(f_ref)
    return dict(bound_violation=bv, constraint_violation=gv, violation=max(bv, gv), f_eval=float(F), f_reported=f_ref, f_rel=rel,
                **{"pass": max(bv, gv) <= VIOL_TOL and rel <= OBJ_TOL})


def _say(label, c):
    print(f"{label}: bound viol {c['bound_violation']:.3e}, constraint viol {c['constraint_violation']:.3e}, f {c['f_eval']:.12g} "
          f"vs reported {c['f_reported']:.12g} (rel {c['f_rel']:.2e}) -> {'PASS' if c['pass'] else 'FAIL'}")


def xcheck_main(argv):
    ap = argparse.ArgumentParser(prog="compare.py xcheck", description="each code's w* evaluated in the other code's NLP (no solve)")
    ap.add_argument("--track", required=True)
    ap.add_argument("--mat", required=True, help="MATLAB run, <track>_mat_<tier>_r<k>.mat")
    ap.add_argument("--py", required=True, help="Python run, <track>_py_<tier>_r<k>.mat")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds for the MATLAB export")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from validation.matlab_batch import export
    from validation.py_export import nlp

    M, P = (sio.loadmat(p, squeeze_me=True) for p in (a.mat, a.py))
    w_mat, w_py = (np.ravel(X["w_opt"]).astype(float) for X in (M, P))
    f_mat, f_py, ds = float(M["f"]), float(P["f"]), float(M["OPT_ds"])
    if float(P["OPT_ds"]) != ds or w_mat.size != w_py.size:
        print(f"FAIL: not the same NLP (OPT_ds {ds:g} vs {float(P['OPT_ds']):g}, n_w {w_mat.size} vs {w_py.size})")
        return 1
    stems = [Path(p).stem for p in (a.mat, a.py)]
    tags = [re.fullmatch(r".+_(?:mat|py)_([a-z]+_r\d+)", s) for s in stems]
    pair = "-".join(t.group(1) if t else s for t, s in zip(tags, stems))
    d = Path(a.mat).resolve().parent
    work = d / f"xcheck_{pair}"
    work.mkdir(exist_ok=True)
    print(f"xcheck {a.track}: {stems[0]} vs {stems[1]} (OPT_ds {ds:g}, n_w {w_mat.size})")

    t0 = time.perf_counter()                                       # one build at a time: Python, then MATLAB
    py = nlp(a.track, ds, W=np.column_stack([w_mat, w_py]))
    t_py = time.perf_counter() - t0
    gc.collect()
    pb = tuple(py[k] for k in ("lbw", "ubw", "lbg", "ubg"))
    py_at_mat = _check(w_mat, py["F"][0], py["G"][:, 0], pb, f_mat)
    py_at_py = _check(w_py, py["F"][1], py["G"][:, 1], pb, f_py)

    sio.savemat(work / "xcheck_w.mat", {"w_mat": w_mat.reshape(-1, 1), "w_py": w_py.reshape(-1, 1)})
    t0 = time.perf_counter()
    try:
        _, nlp_file, _ = export(a.track, ds, work, w=work / "xcheck_w.mat", timeout_s=a.timeout)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    t_mat = time.perf_counter() - t0
    X = sio.loadmat(nlp_file, squeeze_me=True)
    names = [str(n) for n in np.atleast_1d(X["W_names"])]
    F, G = np.atleast_1d(X["F"]), np.asarray(X["G"], dtype=float).reshape(np.size(X["lbg"]), -1)
    mb = tuple(np.ravel(X[k]).astype(float) for k in ("lbw", "ubw", "lbg", "ubg"))
    mat_at_py = _check(w_py, F[names.index("w_py")], G[:, names.index("w_py")], mb, f_py)
    mat_at_mat = _check(w_mat, F[names.index("w_mat")], G[:, names.index("w_mat")], mb, f_mat)

    _say("Python NLP at the MATLAB w*", py_at_mat)
    _say("MATLAB NLP at the Python w*", mat_at_py)
    _say("  self-check, Python NLP at the Python w*", py_at_py)
    _say("  self-check, MATLAB NLP at the MATLAB w*", mat_at_mat)
    ok = py_at_mat["pass"] and mat_at_py["pass"]
    _dump(d / f"{a.track}_xcheck_{pair}.json", dict(
        track=a.track, mat_run=stems[0], py_run=stems[1], opt_ds=ds, n_w=int(w_mat.size), n_g=int(np.size(pb[2])),
        tol=dict(violation=VIOL_TOL, objective=OBJ_TOL), py_nlp_at_mat_w=py_at_mat, mat_nlp_at_py_w=mat_at_py,
        self_check=dict(py_nlp_at_py_w=py_at_py, mat_nlp_at_mat_w=mat_at_mat), wall_s=dict(python=t_py, matlab=t_mat),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), **{"pass": ok}))
    print(f"Python build + evaluation {t_py:.0f} s, MATLAB export {t_mat:.0f} s;  xcheck {pair}: {'PASS' if ok else 'FAIL'}"
          f"\n{d / f'{a.track}_xcheck_{pair}.json'}")
    return 0 if ok else 1


def reference_main(argv):
    ap = argparse.ArgumentParser(prog="compare.py reference", description="pick the reference solution of a track")
    ap.add_argument("--track", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    d, T = Path(a.dir), a.track
    runs = load_runs(T, d)
    passed = {}                                                     # run stem -> xcheck file it passed in
    for xf in sorted(d.glob(f"{T}_xcheck_*.json")):
        X = json.loads(xf.read_text(encoding="utf-8"))
        if X.get("pass"):
            passed.update({X["mat_run"]: xf.name, X["py_run"]: xf.name})
    cand = [r for tag in ("mat_ship", "py_par") for r in runs.get(tag, {}).values()
            if r["ok"] and r["mat"] and r["status"] in GOOD_STATUS and r["lap"] is not None]
    print(_table(("run", "status", "lap [s]", "xcheck passed"),
                 [(r["stem"], r["status"], _fmt(r["lap"]), passed.get(r["stem"], "no")) for r in cand])
          if cand else f"no converged mat_ship / py_par run of {T} in {d}")
    best = min((r for r in cand if r["stem"] in passed), key=lambda r: r["lap"], default=None)
    if best is None:
        print("\nno reference: a converged run needs a passed xcheck (compare.py xcheck)")
        return 1
    J = best["json"]
    pj = sorted((ROOT / "tests" / "data").glob(f"matlab_params_{T}_ds*.json"),
                key=lambda p: (p.stem != f"matlab_params_{T}_ds{_fmt(best['OPT_ds'], 'g')}", p.name))
    P = json.loads(pj[0].read_text(encoding="utf-8")) if pj else {}
    dst = d / "reference" / T
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(best["mat"], dst / best["mat"].name)
    S = dict(track=T, lap=best["lap"], f=best["f"], N=best["N"], OPT_ds=best["OPT_ds"], return_status=best["status"],
             iter_count=best["iters"], source_run=best["stem"], source_mat=best["mat"].name, code=J.get("code"), tier=J.get("tier"),
             rep=J.get("rep"), xcheck=passed[best["stem"]],
             settings={k: J[k] for k in ("ipopt", "settings", "args", "runner_caps", "linear_solver", "mesh", "matlab_version",
                                         "casadi_version", "python_version") if k in J},
             track_sha256=P.get("track_sha256"), aero_sha256=P.get("aero_sha256"), params_json=pj[0].name if pj else None,
             candidates=[dict(run=r["stem"], lap=r["lap"], status=r["status"], xcheck=passed.get(r["stem"])) for r in cand],
             created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _dump(dst / "summary.json", S)
    print(f"\nreference {T}: {best['stem']} lap {best['lap']:.6f} s ({best['status']}), {dst / 'summary.json'}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    sub = {"nlp": nlp_main, "runs": runs_main, "xcheck": xcheck_main, "reference": reference_main}
    if argv[:1] and argv[0] in sub:
        return sub[argv[0]](argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("matlab_json")
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--opt-ds", type=float, required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from validation.py_export import params
    mat = json.loads(Path(a.matlab_json).read_text())
    info = []
    hits = diff_params(mat, params(a.circuit, a.opt_ds), info)
    for key, detail in hits:
        print(f"HIT  {key}: {detail}")
    for key, detail in info:
        print(f"info {key}: {detail}")
    print(f"{len(hits)} hit(s), {len(info)} info")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
