"""setup_sweep.py - setup sweep: a QSS screen, then a star of warm-started 23-state
confirmations (roadmap Item 8).

    from setup_sweep import setup_sweep
    res = setup_sweep([("alpha_RW", 4, 16), ("hcg", 0.45, 0.55), ("mb", 1730, 1910)],
                      n_samples=64, circuit="Sturn", top_k=4, n_probes=4)
    res.best["vp_overrides"]          # ready for MLTP(vp_overrides=...)

Stages (everything is written into Results/sweeps/<name>/, nothing else is touched;
every MLTP / optimise_design call runs with save=False, plot=False):

  fields    validate the (field, lower, upper) specs (optimise_design's format); drop
            unknown keys and fields the 23-state NLP ignores in this configuration
            (a graph-signature probe of the CasADi model, no solve); flag fields that
            move the NLP scaling, fields safe to promote in optimise_design, and fields
            the QSS screen cannot see
  design    row 0 = the baseline setup (vp_overrides), rows 1..n = a scrambled Sobol'
            (or Latin hypercube) design over the box             -> samples.csv
  hub       the baseline NLP solution every confirmation starts from: solved cold in a
            worker (base=None) or taken from a full result (base=<.mat> / ctx), then
            checked: a warm re-solve with duals at the base setup must be 'full+duals',
            converge and take 0 IPOPT iterations                  -> hub.mat, hub.json
  screen    QSS lap of every row (MLTP_screen.screen_batch, bitwise screen_sweep),
            run while the hub solves
  shortlist baseline + the QSS top_k + n_probes rows at QSS-rank quantiles of the rest
  confirm   STAR topology: each shortlisted row is ONE warm hop (primal + duals) from
            the hub, MLTP(warm_start=hub.mat, vp_overrides=base | row). A hop that does
            not converge is retried along a private bridge (2, then 4 equal steps from
            the hub), never cold; a round trip back to the base setup measures the
            branch                                  -> rows/<id>.json|mat, logs/<id>.log
  report    ranking with a noise floor (unresolved pairs), QSS-vs-NLP rank metrics
            (Spearman / Kendall tau-b + bootstrap CIs, resolved-pair concordance,
            regret, slope), the best setup    -> confirmed.csv, summary.json, report.html
  finish    (finish=True) optimise_design over the promotion-safe fields, warm-started
            from the winner's row; its p* is re-confirmed through the star  -> finish/

Why a star: cold solves of this NLP land on local optima about 0.01 s apart, more than
many setup effects, so no cold lap may enter a ranking. A chain of warm starts avoids
cold scatter but makes every row depend on the order (a chain and its reverse differed
by up to 1.9 ms). A star row is a pure function of (hub bytes, setup, options): bitwise
reproducible across worker counts, order independent, resumable and parallel. Far rows
can still hop to another branch, hence the round trips and the noise floor; keep the
box within continuation range.

The QSS screen is an ESTIMATE (fixed centreline): about +4% on Sturn and +11% on BCN
against the NLP, with deltas over-stated about 1.2-1.5x. The confirmed NLP deltas are
the numbers to use; summary.json says whether the screen ranking could be trusted.

Parallelism: NLP tasks run in a spawn ProcessPoolExecutor (BLAS pinned to 1 thread,
Coin-HSL kept loaded per worker), the screen in-process or in its own pool. Call
setup_sweep from `python -c`, from a function, or under `if __name__ == "__main__":`
(Windows spawn re-imports the main script in every worker); never run the repo as a
package with -m. casadi is imported lazily: `import setup_sweep` works without it, and
so does setup_sweep(..., confirm=False).
"""

import contextlib
import dataclasses
import datetime
import hashlib
import importlib
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np

from functions import sweep as S
from functions.context import Ctx
from functions.importfile import importfile, result_stem
from functions.transcription import discretise
from functions.warmstart import (GOOD_STATUS, nlp_structure, structure_mismatch, get_field,
                                 as_dict, is_full_result, vec)
from MLTP_screen import screen_batch
from userOpts import userOpts
from vehParams import vehParams, PRIMARY_KEYS

_REPO = os.path.dirname(os.path.abspath(__file__))

# code whose bytes define a confirmation (plan fingerprint; checked again in every worker)
_CODE_FILES = ("MLTP.py", "MLTP_initial.py", "MLTP_screen.py", "MLTP_paramOptim.py",
               "vehModel.py", "vehModel_initial.py", "vehParams.py", "userOpts.py",
               "Powertrain.py", "functions/transcription.py", "functions/warmstart.py",
               "functions/ggv.py", "functions/mesh.py", "functions/collocation.py",
               "functions/casadi_opts.py", "functions/hsl.py")

ROUNDTRIP_MODES = ("all", "top", "none")
_RETRY = ("crashed", "code_changed", "error")   # row statuses a resume runs again
_QUIET = {"print_level": 0, "sb": "yes"}         # silence IPOPT; never changes its path
_RT_FIELDS = ("rt_lap_s", "rt_iters", "rt_status", "rt_mode", "rt_wall_s", "rt_ok", "rt_reason",
              "rt_residual_s", "branch_ok", "rt_better_baseline_file")
_CSV_TAIL = ["qss_lap_s", "qss_rank", "qss_delta_s", "nlp_lap_s", "nlp_delta_s", "delta_rev_s",
             "rt_residual_s", "branch_ok", "unresolved_with", "iters", "rt_iters", "wall_s",
             "warm_start_mode", "path", "status", "linear_solver", "w_sha", "result_file",
             "repro", "reason"]

_W = {}                                           # worker-process globals (_worker_init)


class SweepError(RuntimeError):
    """A sweep that cannot run as asked: a base that does not match this NLP, a folder
    that holds another sweep, a hub that fails its check, casadi missing, ..."""


@dataclasses.dataclass
class SweepResult:
    """What setup_sweep returns (summary.json holds the same)."""
    name: str
    out_dir: str
    fingerprint: str
    fields: dict = dataclasses.field(repr=False)
    dropped: dict = dataclasses.field(default_factory=dict)
    samples: list = dataclasses.field(default_factory=list, repr=False)
    shortlist: list = dataclasses.field(default_factory=list)
    confirmed: list = dataclasses.field(default_factory=list, repr=False)
    metrics: dict = dataclasses.field(default_factory=dict, repr=False)
    best: dict = dataclasses.field(default_factory=dict)
    finish: dict = dataclasses.field(default_factory=dict, repr=False)
    hub: dict = dataclasses.field(default_factory=dict, repr=False)
    warnings: list = dataclasses.field(default_factory=list, repr=False)
    timings: dict = dataclasses.field(default_factory=dict, repr=False)
    throughput: dict = dataclasses.field(default_factory=dict, repr=False)
    files: dict = dataclasses.field(default_factory=dict, repr=False)
    n_tasks_run: int = 0


# =============================================================================
# small helpers (parent and worker)
# =============================================================================
def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _model_hash(circuit, circuits_dir="Circuits", data_dir="Data"):
    """Hash of the code (next to this file) and the data (relative to the working
    directory) a confirmation depends on; location independent."""
    from userOpts import _REAL_CIRCUITS
    data = [os.path.join(data_dir, "DATA_AA.mat")]
    if circuit in _REAL_CIRCUITS:
        data.append(os.path.join(circuits_dir, _REAL_CIRCUITS[circuit]))
    return S.fingerprint([S.model_hash(_CODE_FILES, root=_REPO),
                          S.model_hash(data, root=os.getcwd())])


def _casadi_version():
    try:
        import casadi
        return str(casadi.__version__)
    except Exception:
        return None


def _git_info():
    """Read-only: HEAD and the number of dirty files (None outside a git checkout)."""
    out = {"head": None, "dirty_files": None}
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0:
            out["head"] = r.stdout.strip()
            r = subprocess.run(["git", "status", "--porcelain"], cwd=_REPO, capture_output=True,
                               text=True, timeout=10)
            if r.returncode == 0:
                out["dirty_files"] = len([ln for ln in r.stdout.splitlines() if ln.strip()])
    except Exception:
        pass
    return out


def _to_mat(obj):
    """savemat-ready copy of a result (namespaces from a loaded .mat -> dicts)."""
    if isinstance(obj, dict):
        return {k: _to_mat(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, (np.ndarray, type)):
        return {k: _to_mat(v) for k, v in vars(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_mat(v) for v in obj]
    return obj


def _savemat_atomic(path, data):
    """{'data': data} -> path (MLTP's own result format) via a temporary file."""
    import scipy.io as sio
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        sio.savemat(fh, {"data": _to_mat(data)}, do_compression=True)
    os.replace(tmp, path)


def _copy_atomic(src, dst):
    d = os.path.dirname(os.path.abspath(dst))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(dst)}.{os.getpid()}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _field_value(ctx, field):
    """Current value of a sweepable field: ctx.vp (vehParams primary) or ctx.mf."""
    return float(getattr(ctx.vp, field) if field in PRIMARY_KEYS else getattr(ctx.mf, field))


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# =============================================================================
# field classification (parent; the NLP probe needs casadi and never solves)
# =============================================================================
def _nlp_probe(ctx, TyreModel="CombinedSlip"):
    """(graph_sha, scale_sha, nh) of the NLP that MLTP builds for ``ctx``."""
    import casadi as ca
    from vehModel import vehModel
    from MLTP import build_path_constraints
    vehModel(ctx, TyreModel=TyreModel)
    m = ctx.m23
    hnames, h, h_lb, h_ub = build_path_constraints(ca, m, ctx.pt)
    f = ca.Function("sig", [m.x, m.u, m.pv], [m.dx, m.sf, h])
    graph = hashlib.sha256(f.serialize().encode("utf-8")).hexdigest()
    arr = b"".join(np.ascontiguousarray(np.asarray(a, dtype=np.float64).reshape(-1)).tobytes()
                   for a in (m.x_s, m.u_s, m.x_min, m.x_max, m.u_min, m.u_max,
                             m.duk_lb, m.duk_ub, h_lb, h_ub))
    return graph, hashlib.sha256(arr).hexdigest(), len(hnames)


def nlp_signature(ctx, TyreModel="CombinedSlip"):
    """(graph_sha, scale_sha) of the 23-state NLP MLTP would build for ``ctx`` (a
    userOpts ctx; vehModel is built on it, nothing is solved):

      graph_sha  sha256 of ca.Function('sig', [x, u, pv], [dx, sf, h]).serialize(): the
                 dynamics, the lap-time integrand and the path constraints
                 (MLTP.build_path_constraints) with every vehicle constant baked in
      scale_sha  sha256 of the float64 bytes of x_s, u_s, x_min, x_max, u_min, u_max,
                 duk_lb, duk_ub, h_lb, h_ub (scaling and bounds)

    Two setups with equal signatures give the same NLP, so a field whose change leaves
    both equal is ignored by the NLP (brkB / Tdist with ATD On, Cd always). Needs
    casadi; deterministic across builds (~15 ms on Sturn)."""
    graph, scale, _ = _nlp_probe(ctx, TyreModel)
    return graph, scale


def _vp_snapshot(base_ov, extra, data_dir, tyre_set):
    c = Ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vehParams(c, data_dir=data_dir, vp_overrides={**base_ov, **extra}, tyre_set=tyre_set)
    d = dict(vars(c.vp))
    d["tyre"] = dict(vars(d["tyre"]))
    return d


def _same(a, b):
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(_same(a[k], b[k]) for k in a))
    if a is None or b is None:
        return a is b
    return np.array_equal(np.asarray(a), np.asarray(b))


def _classify(specs, base_ov, cfg, kw, TyreModel, load_model, ds_fine, nlp):
    """({field: flags}, probe info {nh, timings}); see classify_fields. ``cfg`` =
    dict(circuit, vi, ni, AeroConfig, ATD, Electric_4Motors)."""
    t0 = time.perf_counter()
    data_dir, tyre_set = kw.get("data_dir", "Data"), kw.get("tyre_set", "MF205")

    def octx(extra):
        c = Ctx()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            userOpts(c, vp_overrides={**base_ov, **extra}, **cfg, **kw)
        return c

    base = octx({})
    info = dict(nh=None, base_graph_sha=None, base_scale_sha=None, t_nlp_probe_ms=[])
    if nlp:
        t1 = time.perf_counter()
        info["base_graph_sha"], info["base_scale_sha"], info["nh"] = _nlp_probe(octx({}), TyreModel)
        info["t_nlp_probe_ms"].append(1e3 * (time.perf_counter() - t1))
    snap0 = _vp_snapshot(base_ov, {}, data_dir, tyre_set)
    flags = {}
    for f, lo, hi in specs:
        b = _field_value(base, f)
        v = hi if hi != b else lo
        fl = dict(lower=lo, upper=hi, base=b, probe_value=v, primary=f in PRIMARY_KEYS,
                  nlp_live=None, graph_changed=None, scale_changed=None)
        if nlp:
            t1 = time.perf_counter()
            g, s, _ = _nlp_probe(octx({f: v}), TyreModel)
            fl["t_nlp_probe_ms"] = 1e3 * (time.perf_counter() - t1)
            info["t_nlp_probe_ms"].append(fl["t_nlp_probe_ms"])
            fl["graph_changed"] = g != info["base_graph_sha"]
            fl["scale_changed"] = s != info["base_scale_sha"]
            fl["nlp_live"] = bool(fl["graph_changed"] or fl["scale_changed"])
        snap = _vp_snapshot(base_ov, {f: v}, data_dir, tyre_set)
        changed = sorted(k for k in set(snap) | set(snap0)
                         if k != f and (k not in snap or k not in snap0 or not _same(snap[k], snap0[k])))
        fl["phase2_free"] = not changed
        fl["phase2_changed"] = changed[:12]
        fl["promotion_safe"] = bool(fl["primary"] and fl["phase2_free"] and fl["nlp_live"] is True
                                    and not fl["scale_changed"])
        flags[f] = fl
    # QSS-blind: one lap at base, base|{f: lower} and base|{f: upper}
    live = [f for f, *_ in specs if flags[f]["nlp_live"] is not False]
    rows = [{}]
    for f in live:
        rows += [{f: flags[f]["lower"]}, {f: flags[f]["upper"]}]
    qb = screen_batch(cfg["circuit"], rows, vi=cfg["vi"], AeroConfig=cfg["AeroConfig"],
                      ATD=cfg["ATD"], Electric_4Motors=cfg["Electric_4Motors"],
                      load_model=load_model, ds_fine=ds_fine, workers=1,
                      vp_overrides=base_ov, **kw)
    lap = qb["lap_time"]
    for j, f in enumerate(live):
        trio = [float(lap[0]), float(lap[1 + 2 * j]), float(lap[2 + 2 * j])]
        flags[f]["qss_laps"] = trio
        flags[f]["qss_blind"] = bool(all(math.isfinite(x) for x in trio) and trio[0] == trio[1] == trio[2])
    for f, *_ in specs:
        flags[f].setdefault("qss_blind", None)
    probe = info["t_nlp_probe_ms"]
    info.update(t_total_ms=1e3 * (time.perf_counter() - t0),
                t_nlp_probe_mean_ms=float(np.mean(probe)) if probe else None,
                qss_ms_per_row=1e3 * qb["wall_s"] / max(len(rows), 1))
    return flags, info


def classify_fields(param_specs, circuit="Sturn", *, vi=60.0, ni=np.nan, AeroConfig="Static",
                    ATD="On", Electric_4Motors="Off", TyreModel="CombinedSlip",
                    load_model="vehModel", ds_fine=1.0, nlp=None, **useropts_kwargs):
    """How a sweep treats each field of ``param_specs`` [(field, lower, upper)] at the
    base setup (vp_overrides in **useropts_kwargs). Nothing is solved. Returns
    {field: flags} for the valid, known fields (unknown ones are warned about and left
    out; malformed specs raise ValueError):

      nlp_live        the 23-state NLP changes when the field moves from its base value to
                      the probe value (the upper bound, or the lower one if the base sits
                      on it), i.e. nlp_signature differs. False = NLP-inert (a sweep drops
                      it); None when casadi is unavailable or nlp=False.
      graph_changed / scale_changed   which part of the signature moved (scale: x_s / u_s
                      / bounds follow the field, e.g. Rw, Tbrake_max)
      phase2_free     vehParams derives nothing else from it (setattr after vehParams, as
                      optimise_design does, gives the same vp)
      promotion_safe  safe to co-optimise in optimise_design: a vehParams primary,
                      phase2_free, NLP-live and not scale_changed
      qss_blind       the QSS screen gives one lap at base, lower and upper (it cannot rank
                      the field, e.g. damping, inertias, toe)
    plus lower / upper / base / probe_value / primary / qss_laps / t_nlp_probe_ms."""
    specs, _ = S.validate_specs(param_specs)
    if nlp is None:
        nlp = _casadi_version() is not None
    kw = dict(useropts_kwargs)
    base_ov = dict(kw.pop("vp_overrides", None) or {})
    cfg = dict(circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig, ATD=ATD,
               Electric_4Motors=Electric_4Motors)
    flags, _ = _classify(specs, base_ov, cfg, kw, TyreModel, load_model, ds_fine, bool(nlp))
    return flags


# =============================================================================
# worker side (module level: the spawn pool pickles these by reference)
# =============================================================================
def _worker_init(spec):
    """Pool initializer: pin BLAS, chdir to the sweep's working directory, hash the
    model files (a task whose plan hash differs returns 'code_changed'), keep the
    Coin-HSL library loaded (pin_hsl: IPOPT then stops re-loading it on every solve)
    and import MLTP, so the hash describes the code this worker runs."""
    _W.clear()
    try:
        for k in S.BLAS_VARS:
            os.environ[k] = "1"
        os.chdir(spec["root"])
        if spec["root"] not in sys.path:
            sys.path.insert(0, spec["root"])
        _W["model_hash"] = _model_hash(spec["circuit"], spec.get("circuits_dir", "Circuits"),
                                       spec.get("data_dir", "Data"))
        _W["pinned"] = None
        if spec.get("pin_hsl") and str(spec.get("linear_solver", "ma57")).startswith("ma"):
            import ctypes
            from functions.hsl import resolve_hsl_dir, hsllib_path, register_hsl_dll_dir
            d = resolve_hsl_dir(spec.get("hsl_dir"))
            lib = hsllib_path(d)
            if lib:
                register_hsl_dll_dir(d)
                _W["pinned"] = ctypes.CDLL(lib)
                _W["pinned_path"] = lib
        import MLTP  # noqa: F401  (casadi and the model code, loaded once per worker)
    except Exception as exc:
        _W["init_error"] = f"{type(exc).__name__}: {exc}"


def _mltp(task, warm, overrides, ipopt):
    from MLTP import MLTP
    return MLTP(circuit=task["circuit"], vi=task["vi"], ni=task["ni"], warm_start=warm,
                AeroConfig=task["AeroConfig"], ATD=task["ATD"],
                Electric_4Motors=task["Electric_4Motors"], TyreModel=task["TyreModel"],
                save=False, plot=False, warm_start_duals=True,
                vp_overrides={**task["base_ov"], **overrides}, ipopt_overrides=ipopt,
                **task["useropts"])


def _hop_info(c, t0, step, overrides):
    nlp = c.data.nlp
    return dict(step=step, overrides=dict(overrides), lap=float(c.data.lap_time),
                iters=int(c.elapsed.get("ipopt_iters", -1)),
                status=str(c.solve_stats.get("return_status", "unknown")),
                mode=str(c.elapsed.get("warm_start")), linear_solver=str(nlp.get("linear_solver")),
                wall=time.perf_counter() - t0, w_sha=S.array_sha(nlp["w_opt"]))


def _hop(task, warm, overrides, step):
    """One warm MLTP solve at base_ov | overrides from ``warm`` (a path or a ctx)."""
    t0 = time.perf_counter()
    c = _mltp(task, warm, overrides, task["hop_ipopt"])
    return c, _hop_info(c, t0, step, overrides)


def _reject(h, solver):
    """None if a warm hop is accepted, else (kind, reason): 'solver_mismatch' (not the
    hub's linear solver, e.g. an HSL fallback to MUMPS: another IPOPT path), 'mode'
    (not a full+duals re-injection), 'status' (no convergence: bridge it)."""
    if h["linear_solver"] != solver:
        return ("solver_mismatch", f"{h['step']}: linear solver {h['linear_solver']} != hub's {solver}")
    if h["mode"] != "full+duals":
        return ("mode", f"{h['step']}: warm start {h['mode']}, not full+duals")
    if h["status"] not in GOOD_STATUS:
        return ("status", f"{h['step']}: {h['status']} after {h['iters']} iterations")
    return None


def _repro(task, steps):
    """One-line python call that reproduces a confirmed row from the hub file."""
    kw = dict(circuit=task["circuit"], vi=task["vi"], AeroConfig=task["AeroConfig"],
              ATD=task["ATD"], Electric_4Motors=task["Electric_4Motors"],
              TyreModel=task["TyreModel"], save=False, plot=False, warm_start_duals=True,
              ipopt_overrides=task["hop_ipopt"], **task["useropts"])
    ni = task["ni"]
    args = (f"ni={ni!r}, " if _fin(ni) else "ni=float('nan'), ") + ", ".join(
        f"{k}={v!r}" for k, v in kw.items())
    tail = "  # cwd = the repo root, OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1"
    ovs = [{**task["base_ov"], **s} for s in steps]
    hub = task["hub_path"]
    if len(ovs) == 1:
        return f"from MLTP import MLTP; c = MLTP(warm_start=r'{hub}', vp_overrides={ovs[0]!r}, {args})" + tail
    return (f"from functools import reduce; from MLTP import MLTP; c = reduce(lambda w, ov: "
            f"MLTP(warm_start=w, vp_overrides=ov, {args}), {ovs!r}, r'{hub}')" + tail)


def _round_trip(task, warm, hops):
    """Hop from ``warm`` (the forward solution: ctx or file) back to the base setup."""
    c, h = _hop(task, warm, {}, "roundtrip")
    hops.append(h)
    why = _reject(h, task["hub_linear_solver"])
    hub_lap = task["hub_lap"]
    out = dict(rt_lap_s=h["lap"], rt_iters=h["iters"], rt_status=h["status"], rt_mode=h["mode"],
               rt_wall_s=h["wall"], rt_ok=why is None, rt_reason=None if why is None else why[1],
               rt_residual_s=float("nan"), branch_ok=False, rt_better_baseline_file=None)
    if why is None:
        r = h["lap"] - hub_lap
        out["rt_residual_s"] = r
        out["branch_ok"] = bool(abs(r) <= task["tol_branch_s"])
        if h["lap"] < hub_lap - task["tol_branch_s"]:
            _savemat_atomic(task["rt_path"], vars(c.data))
            out["rt_better_baseline_file"] = task["rt_path"]
    return out


def _task_hub_cold(task):
    t0 = time.perf_counter()
    c = _mltp(task, None, {}, task["cold_ipopt"])
    h = _hop_info(c, t0, "hub_cold", {})
    h["init_s"] = float(c.elapsed.get("init", float("nan")))
    _savemat_atomic(task["hub_path"], vars(c.data))
    return dict(status="done", hop=h)


def _task_hub_check(task):
    c, h = _hop(task, task["hub_path"], {}, task["id"])
    if h["iters"] > 0 and _reject(h, h["linear_solver"]) is None:
        _savemat_atomic(task["rehub_path"], vars(c.data))     # the parent may promote it
    return dict(status="done", hop=h)


def _task_confirm(task):
    """A star confirmation: the forward hop hub -> row (bridges if it fails), its
    rows/<id>.mat, and the round trip back to the base setup (task['roundtrip']).
    rt_only: just the round trip, from rows/<id>.mat (second wave)."""
    rid, row = task["row_id"], dict(task["row"])
    solver, hops = task["hub_linear_solver"], []
    rec = dict(row_id=rid, kind=task["row_kind"], overrides=row,
               vp_overrides={**task["base_ov"], **row}, fingerprint=task["fingerprint"],
               model_hash=task["model_hash"], pid=os.getpid())
    if task.get("rt_only"):
        rec.update(_round_trip(task, task["row_mat"], hops), hops=hops, status="rt_done")
        return rec
    c, h = _hop(task, task["hub_path"], row, "star")
    hops.append(h)
    why = _reject(h, solver)
    path, final_c, final, steps = "star", c, h, [row]
    if why is not None and why[0] == "status":
        reasons, final_c = [why[1]], None
        for k in task["bridge_steps"]:
            prev, ok = task["hub_path"], True
            pts = S.bridge_points(task["hub_vals"], row, k)
            for j, p in enumerate(pts, 1):
                cj, hj = _hop(task, prev, p, f"bridge{k} {j}/{k}")
                hops.append(hj)
                wj = _reject(hj, solver)
                if wj is not None:
                    ok, why = False, wj
                    reasons.append(wj[1])
                    break
                prev = cj
            if ok:
                path, final_c, final, steps, why = f"bridge{k}", cj, hj, pts, None
                break
            if why[0] != "status":
                break
        if why is not None:
            why = (why[0], "; ".join(reasons))
    rec.update(hops=hops, iters_total=int(sum(max(x["iters"], 0) for x in hops)))
    if why is not None:
        rec.update(status="solver_mismatch" if why[0] == "solver_mismatch" else "failed",
                   reason=why[1], path=None)
        return rec
    path_hops = hops[-len(steps):]
    rec.update(status="accepted", reason=None, path=path, nlp_lap_s=final["lap"],
               nlp_delta_s=final["lap"] - task["hub_lap"],
               iters=int(sum(max(x["iters"], 0) for x in path_hops)),
               ipopt_status=final["status"], warm_start_mode=final["mode"],
               linear_solver=final["linear_solver"], w_sha=final["w_sha"],
               result_file=task["row_mat"], repro=_repro(task, steps))
    _savemat_atomic(task["row_mat"], vars(final_c.data))
    if task.get("roundtrip"):
        rec.update(_round_trip(task, final_c, hops))
        rec["delta_rev_s"] = (final["lap"] - rec["rt_lap_s"]) if rec["rt_ok"] else None
    return rec


def _task_finish(task):
    from MLTP_paramOptim import optimise_design
    t0 = time.perf_counter()
    try:
        c = optimise_design([tuple(s) for s in task["specs"]], "sweep_finish",
                            circuit=task["circuit"], vi=task["vi"], ni=task["ni"],
                            warm_start=task["warm_path"], AeroConfig=task["AeroConfig"],
                            ATD=task["ATD"], Electric_4Motors=task["Electric_4Motors"],
                            save=False, warm_start_duals=True,
                            vp_overrides={**task["base_ov"], **task["winner"]},
                            ipopt_overrides=task["cold_ipopt"], **task["useropts"])
    except ValueError as exc:
        return dict(status="skipped", reason=f"optimise_design rejected the warm start: {exc}")
    st = str(c.solve_stats.get("return_status", "unknown"))
    p_star = {k: float(v) for k, v in dict(c.data.optimal_params).items()}
    at_bound = {}
    for f, lo, hi in task["specs"]:
        tol = 1e-5 * (hi - lo)
        at_bound[f] = dict(lower=bool(p_star[f] - lo <= tol), upper=bool(hi - p_star[f] <= tol))
    _savemat_atomic(task["design_mat"], vars(c.data))
    return dict(status="done" if st in GOOD_STATUS else "failed",
                reason=None if st in GOOD_STATUS else f"optimise_design ended {st}",
                opt_lap_s=float(c.data.lap_time), iters=int(c.elapsed.get("ipopt_iters", -1)),
                warm_start_mode=str(c.elapsed.get("warm_start")), ipopt_status=st,
                linear_solver=str(c.data.nlp.get("linear_solver")), p_star=p_star,
                at_bound=at_bound, wall_s=time.perf_counter() - t0,
                result_file=task["design_mat"])


_TASKS = {"hub_cold": _task_hub_cold, "hub_check": _task_hub_check,
          "confirm": _task_confirm, "finish": _task_finish}


def _run_task(task):
    """Run one NLP task in a pool worker (or in-process for workers=0): stdout goes to
    logs/<id>.log, warnings are captured, the plan's model hash is checked first."""
    t0 = time.perf_counter()
    out = dict(task_id=task["id"], pid=os.getpid())
    if "init_error" in _W:
        out.update(status="error", reason=f"worker initialisation failed: {_W['init_error']}")
        return out
    if _W.get("model_hash") != task["model_hash"]:
        out.update(status="code_changed",
                   reason="the model files changed after the sweep was planned (worker hash "
                          f"{str(_W.get('model_hash'))[:8]} != plan {task['model_hash'][:8]})")
        return out
    os.makedirs(os.path.dirname(task["log"]), exist_ok=True)
    with open(task["log"], "w", encoding="utf-8") as fh, contextlib.redirect_stdout(fh), \
            warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        print(f"[setup_sweep] task {task['id']} ({task['kind']}) pid {os.getpid()} {_now()}")
        try:
            out.update(_TASKS[task["kind"]](task))
        except Exception as exc:
            out.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            traceback.print_exc(file=fh)
        print(f"[setup_sweep] task {task['id']} finished: {out.get('status')} "
              f"in {time.perf_counter() - t0:.2f} s")
    out["warnings"] = S.dedup(f"{w.category.__name__}: {w.message}" for w in rec)
    out["task_wall_s"] = time.perf_counter() - t0
    out["hsl_pinned"] = _W.get("pinned") is not None
    return out


# =============================================================================
# parent side: task runner
# =============================================================================
class _Runner:
    """Runs task dicts in a spawn ProcessPoolExecutor (``P`` workers, _worker_init), or
    in this process when inline (workers=0, debugging). After a BrokenProcessPool the
    pool is re-created once and the unfinished tasks are resubmitted; tasks still
    unfinished when a second pool breaks are recorded 'crashed'."""

    def __init__(self, P, spec, inline, say):
        self.P, self.spec, self.inline, self.say = int(P), dict(spec), bool(inline), say
        self.tasks, self.futs, self.results, self.delivered = {}, {}, {}, set()
        self.recreated, self.n_submitted, self.pool = False, 0, None
        mod = importlib.import_module("setup_sweep")    # pickled by module name, not __main__
        self._init_fn, self._run_fn = mod._worker_init, mod._run_task
        if self.inline:
            _worker_init(self.spec)
        else:
            self.pool = self._new_pool()

    def _new_pool(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        return ProcessPoolExecutor(max_workers=self.P, mp_context=mp.get_context("spawn"),
                                   initializer=self._init_fn, initargs=(self.spec,))

    def submit(self, task):
        tid = task["id"]
        self.tasks[tid] = task
        self.results.pop(tid, None)
        self.delivered.discard(tid)
        self.n_submitted += 1
        if self.inline:
            self.results[tid] = _run_task(task)
        elif self.pool is None:
            self.results[tid] = dict(status="crashed", reason="no process pool left after two breaks")
        else:
            self.futs[tid] = self.pool.submit(self._run_fn, task)

    def _recover(self):
        unfinished = [t for t in self.futs if t not in self.results]
        try:
            self.pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        if self.recreated:
            for t in unfinished:
                self.results[t] = dict(status="crashed", reason="a worker process died in two "
                                       "process pools (BrokenProcessPool); not retried")
                self.futs.pop(t, None)
            self.pool = None
            return
        self.recreated = True
        self.say(f"[sweep] a worker process died (BrokenProcessPool): re-creating the pool once "
                 f"and resubmitting {len(unfinished)} unfinished task(s)")
        self.pool = self._new_pool()
        for t in unfinished:
            self.futs[t] = self.pool.submit(self._run_fn, self.tasks[t])

    def collect(self, ids, on_done=None):
        """Wait for the tasks ``ids``; on_done(task, result) once per task, as it ends."""
        from concurrent.futures import wait, FIRST_COMPLETED
        from concurrent.futures.process import BrokenProcessPool
        ids = list(ids)

        def deliver(t):
            if t not in self.delivered:
                self.delivered.add(t)
                if on_done is not None:
                    on_done(self.tasks[t], self.results[t])

        for t in ids:
            if t in self.results:
                deliver(t)
        while True:
            pending = {self.futs[t]: t for t in ids if t not in self.results and t in self.futs}
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            broken = False
            for fut in done:
                t = pending[fut]
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    broken = True
                    continue
                except Exception as exc:
                    res = dict(status="error", reason=f"{type(exc).__name__}: {exc}")
                self.results[t] = res
                self.futs.pop(t, None)
                deliver(t)
            if broken:
                self._recover()
                for t in ids:
                    if t in self.results:
                        deliver(t)
        return {t: self.results[t] for t in ids}

    def close(self, cancel=False):
        if self.pool is not None:
            try:
                self.pool.shutdown(wait=not cancel, cancel_futures=cancel)
            except Exception:
                pass
            self.pool = None


# =============================================================================
# parent side: base / hub helpers
# =============================================================================
def _resolve_base(base):
    """(data or None, identity for the fingerprint, kind, path) of a ``base`` argument."""
    if base is None:
        return None, None, None, None
    if isinstance(base, (str, os.PathLike)):
        p = os.path.abspath(os.fspath(base))
        if not os.path.isfile(p):
            raise SweepError(f"base file {p} does not exist")
        try:
            data = importfile(p)["data"]
        except Exception as exc:
            raise SweepError(f"base file {p} is not a readable result .mat ({exc})") from None
        return data, {"kind": "file", "sha256": S.sha256_file(p)}, "file", p
    data = get_field(base, "data", None)
    if data is None:
        data = base
    nlp = get_field(data, "nlp")
    if not is_full_result(data) or nlp is None or get_field(nlp, "w_opt") is None:
        raise SweepError("base must be None, a full 23-state result .mat with data.nlp, or the "
                         "ctx / ctx.data of an MLTP() call")
    return data, {"kind": "ctx", "w_sha": S.array_sha(vec(get_field(nlp, "w_opt")))}, "ctx", None


def _precheck_base(src, base_ctx, disc, nh, circuit, label):
    """casadi-free: the supplied base must be a solution of THIS NLP's structure (else
    MLTP would quietly fall back to interpolation or a minutes-long cold solve)."""
    reasons = []
    if not is_full_result(src):
        reasons.append("not a full 23-state result (data.x_opt with 23 rows)")
    nlp = get_field(src, "nlp")
    if nlp is None:
        reasons.append("no saved NLP vectors (data.nlp)")
    if not reasons:
        expected = dict(nlp_structure(23, len(base_ctx.input_keys), 0, disc["N"], base_ctx.OPT_d, nh),
                        input_keys=list(base_ctx.input_keys), s_full=disc["s_full"])
        saved = dict(as_dict(get_field(nlp, "structure")), input_keys=get_field(src, "input_keys"),
                     s_full=get_field(src, "s_full"))
        reasons += structure_mismatch(saved, expected)
    have = str(get_field(src, "tyre_set", None) or "CopyB").strip()
    if have != str(base_ctx.tyre_set):
        reasons.append(f"tyre_set {have!r} != {base_ctx.tyre_set!r}")
    c = get_field(src, "circuit", None)
    if c is not None and str(c).strip() != circuit:
        reasons.append(f"circuit {str(c).strip()!r} != {circuit!r}")
    if reasons:
        raise SweepError(f"base {label} does not match this sweep's NLP ({'; '.join(reasons)}); "
                         "pass a result solved with the same circuit, config, OPT_ds / OPT_d / "
                         "mesh and tyre_set, or base=None for a cold hub")


def _load_row(path, fp, vals):
    """A row record of an earlier run of this sweep, if it can be reused."""
    if not os.path.isfile(path):
        return None
    try:
        r = S.read_json(path)
    except Exception:
        return None
    if r.get("fingerprint") != fp or r.get("status") in _RETRY:
        return None
    ov = r.get("overrides") or {}
    if set(ov) != set(vals) or any(float(ov[f]) != float(vals[f]) for f in vals):
        return None
    if r.get("status") == "accepted" and not os.path.isfile(
            os.path.join(os.path.dirname(path), f"{r['row_id']}.mat")):
        return None
    return r


def _rt_wave(records, roundtrip):
    """Rows still missing a round trip: 'top' -> the 3 best accepted rows plus rows with a
    bridge path or > 50 forward iterations; 'all' -> every accepted row without one."""
    acc = {rid: r for rid, r in records.items()
           if r.get("status") == "accepted" and r.get("kind") != "finish"}
    need = {rid for rid, r in acc.items() if r.get("rt_ok") is None}
    if roundtrip == "all":
        return sorted(need)
    if roundtrip == "top":
        best = sorted(acc, key=lambda rid: (acc[rid]["nlp_lap_s"], rid))[:3]
        sus = [rid for rid, r in acc.items()
               if str(r.get("path") or "").startswith("bridge") or (r.get("iters") or 0) > 50]
        return sorted(set(best + sus) & need)
    return []


# =============================================================================
# the sweep
# =============================================================================
class _Sweep:
    """State of one setup_sweep call (see setup_sweep for the arguments)."""

    def __init__(self, a):
        self.__dict__.update(a)
        self.t_start = time.perf_counter()
        self.timings, self.wlist, self.files = {}, [], {}
        self.runner, self.P, self.hub, self.fin = None, 0, None, None
        self.records, self.new_rows = {}, set()

    # ---- messages ---------------------------------------------------------------
    def say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def warn(self, msg, category=UserWarning):
        self.wlist.append(str(msg))
        warnings.warn(str(msg), category, stacklevel=4)

    def rewarn(self, records):
        for w in records:
            self.warn(str(w.message), w.category)

    # ---- 0-4: arguments, base, fields, fingerprint, folder ------------------------
    def prepare(self, param_specs, useropts_kwargs):
        if self.roundtrip not in ROUNDTRIP_MODES:
            raise ValueError(f"roundtrip must be one of {ROUNDTRIP_MODES}, got {self.roundtrip!r}")
        if self.TyreModel not in ("CombinedSlip", "PureSlip"):
            raise ValueError(f"TyreModel must be 'CombinedSlip' or 'PureSlip', got {self.TyreModel!r}")
        n = self.n_samples
        if isinstance(n, bool) or int(n) != n or int(n) < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n!r}")
        self.n = int(n)
        self.top_k, self.n_probes = int(self.top_k), int(self.n_probes)
        if self.top_k < 0 or self.n_probes < 0:
            raise ValueError("top_k and n_probes must be >= 0")
        self.bridge_steps = tuple(int(k) for k in self.bridge_steps)
        if any(k < 2 for k in self.bridge_steps):
            raise ValueError(f"bridge_steps must be integers >= 2, got {self.bridge_steps}")
        w = self.workers
        if not (w == "auto" or (isinstance(w, int) and not isinstance(w, bool) and w >= 0)):
            raise ValueError(f"workers must be 'auto' or an int >= 0, got {w!r}")
        w = self.screen_workers
        if not (w == "auto" or (isinstance(w, int) and not isinstance(w, bool) and w >= 1)):
            raise ValueError(f"screen_workers must be 'auto' or an int >= 1, got {w!r}")
        if int(self.max_warm_iter) < 1:
            raise ValueError("max_warm_iter must be >= 1")
        self.max_warm_iter = int(self.max_warm_iter)
        kw = dict(useropts_kwargs)
        self.base_ov = dict(kw.pop("vp_overrides", None) or {})
        user_ip = dict(kw.pop("ipopt_overrides", None) or {})
        self.kw = kw
        if kw.get("screening"):
            self.warn("screening=True is forwarded to every NLP solve, but its loose tolerances "
                      "move laps by -4..+20 ms, too much to rank ms-size setup effects")
        self.hop_ip = {**user_ip, **_QUIET, "max_iter": self.max_warm_iter}
        self.cold_ip = {**user_ip, **_QUIET}
        self.kw_fp = dict(kw, ipopt_overrides=user_ip) if user_ip else dict(kw)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            specs, self.dropped = S.validate_specs(param_specs)
        self.rewarn(rec)

        # 1. base ctx (bad base vp_overrides raise here)
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            bc = userOpts(Ctx(), circuit=self.circuit, vi=self.vi, ni=self.ni,
                          AeroConfig=self.AeroConfig, ATD=self.ATD,
                          Electric_4Motors=self.Electric_4Motors, vp_overrides=self.base_ov,
                          ipopt_overrides=user_ip or None, **kw)
        self.rewarn(rec)
        self.base_ctx = bc
        self.eATD, self.eEM4 = bc.ATD, bc.Electric_4Motors
        self.config = f"{self.AeroConfig}_ATD{self.eATD}_EM4{self.eEM4}"
        self.stem = result_stem(self.circuit, self.config, bc.tyre_set, bc.mesh_requested)
        self.cfg = dict(circuit=self.circuit, vi=self.vi, ni=self.ni, AeroConfig=self.AeroConfig,
                        ATD=self.eATD, Electric_4Motors=self.eEM4)
        self.disc = discretise(bc.track, bc.OPT_ds, bc.OPT_d, mesh=getattr(bc, "mesh", "uniform"),
                               mesh_opts=getattr(bc, "mesh_opts", None))
        self.timings["setup_s"] = time.perf_counter() - t0

        # 2. field classification
        self.ca_version = _casadi_version()
        if self.ca_version is None:
            if self.confirm:
                raise SweepError("casadi is not importable: the NLP confirmations need it "
                                 "(pass confirm=False for a QSS-only screen)")
            self.warn("casadi is not importable: fields are not checked for NLP inertness "
                      "(confirm=False)")
        t0 = time.perf_counter()
        self.flags, self.cinfo = _classify(specs, self.base_ov, self.cfg, kw, self.TyreModel,
                                           self.load_model, self.ds_fine, self.ca_version is not None)
        self.timings["classify_s"] = time.perf_counter() - t0
        for f, fl in self.flags.items():
            if fl["nlp_live"] is False:
                self.warn(f"{f} is ignored by the 23-state NLP in {self.config} "
                          f"(TyreModel {self.TyreModel}); dropped")
                self.dropped[f] = f"NLP-inert in {self.config}"
        self.kept = [s for s in specs if s[0] not in self.dropped]
        self.fields = [s[0] for s in self.kept]
        for f in self.fields:
            fl = self.flags[f]
            if fl["qss_blind"]:
                self.warn(f"the QSS screen cannot rank {f} (one QSS lap at {fl['lower']:g}, "
                          f"{fl['base']:g} and {fl['upper']:g}); kept, the NLP sees it")
        if self.fields and all(self.flags[f]["qss_blind"] for f in self.fields):
            self.warn("every field is QSS-blind: the shortlist degenerates to the Sobol' order "
                      "and the rank metrics are undefined")
        self.base_vals = {f: self.flags[f]["base"] for f in self.fields}

        # 3. supplied base: casadi-free pre-check, before any folder or task
        self.base_src, base_ident, self.base_kind, self.base_path = _resolve_base(self.base)
        if self.confirm and self.base_src is not None and self.fields:
            _precheck_base(self.base_src, bc, self.disc, self.cinfo["nh"], self.circuit,
                           self.base_path or f"<{type(self.base).__name__}>")

        # 4. fingerprint, folder, plan
        import scipy
        self.mhash = _model_hash(self.circuit, kw.get("circuits_dir", "Circuits"),
                                 kw.get("data_dir", "Data"))
        fp_in = dict(specs=self.kept, dropped=self.dropped, n_samples=self.n, sampler=self.sampler,
                     seed=self.seed, circuit=self.circuit, vi=self.vi, ni=self.ni,
                     config=[self.AeroConfig, self.eATD, self.eEM4], TyreModel=self.TyreModel,
                     base_ov=self.base_ov, useropts=self.kw_fp, load_model=self.load_model,
                     ds_fine=self.ds_fine, max_warm_iter=self.max_warm_iter,
                     bridge_steps=list(self.bridge_steps), base=base_ident, model_hash=self.mhash,
                     versions=dict(numpy=np.__version__, scipy=scipy.__version__,
                                   casadi=self.ca_version))
        self.fp = fp = S.fingerprint(fp_in)
        self.name = self.name or f"{self.stem}_{fp[:8]}"
        self.out_dir = out = os.path.abspath(os.path.join(self.results_root, self.name))
        self.paths = dict(plan=os.path.join(out, "plan.json"), samples=os.path.join(out, "samples.csv"),
                          hub=os.path.join(out, "hub.mat"), hub_json=os.path.join(out, "hub.json"),
                          rehub=os.path.join(out, "hub_rehub.mat"), rows=os.path.join(out, "rows"),
                          logs=os.path.join(out, "logs"), finish=os.path.join(out, "finish"),
                          confirmed=os.path.join(out, "confirmed.csv"),
                          summary=os.path.join(out, "summary.json"),
                          report=os.path.join(out, "report.html"))
        self.resumed = False
        if os.path.isfile(self.paths["plan"]):
            old = S.read_json(self.paths["plan"])
            if old.get("fingerprint") != fp:
                raise SweepError(f"{out} holds another sweep (fingerprint "
                                 f"{str(old.get('fingerprint'))[:8]}, this call {fp[:8]}); a sweep "
                                 "folder is never overwritten: pick another name or results_root")
            if not self.resume:
                raise SweepError(f"{out} already holds this sweep; pass resume=True to continue it")
            self.resumed = True
            self.plan = old
            self.plan.setdefault("runs", []).append(_now())
        else:
            if os.path.isdir(out) and os.listdir(out):
                raise SweepError(f"{out} exists and is not a sweep folder (no plan.json)")
            os.makedirs(out, exist_ok=True)
            self.plan = self._plan(param_specs, fp_in, scipy.__version__)
        S.write_json_atomic(self.paths["plan"], self.plan)
        self.files["plan"] = self.paths["plan"]
        self.say(f"[sweep] {self.name}: {len(self.fields)} field(s) {self.fields}, {self.n} samples "
                 f"on {self.circuit} {self.config} (fingerprint {fp[:8]}"
                 f"{', resumed' if self.resumed else ''}) -> {out}")
        if self.dropped:
            self.say(f"[sweep] dropped: {self.dropped}")

    def _plan(self, param_specs, fp_in, scipy_version):
        bc = self.base_ctx
        call = dict(param_specs=[list(s) if isinstance(s, (list, tuple)) else repr(s) for s in param_specs],
                    n_samples=self.n, circuit=self.circuit, vi=self.vi, ni=self.ni,
                    AeroConfig=self.AeroConfig, ATD=self.ATD, Electric_4Motors=self.Electric_4Motors,
                    TyreModel=self.TyreModel, sampler=self.sampler, seed=self.seed, top_k=self.top_k,
                    n_probes=self.n_probes,
                    base=(self.base_path if self.base_kind == "file" else
                          None if self.base is None else f"<{type(self.base).__name__}>"),
                    confirm=bool(self.confirm), roundtrip=self.roundtrip,
                    max_warm_iter=self.max_warm_iter, bridge_steps=list(self.bridge_steps),
                    tol_branch_s=self.tol_branch_s, resolve_s=self.resolve_s, finish=bool(self.finish),
                    workers=self.workers, screen_workers=self.screen_workers,
                    pin_hsl=bool(self.pin_hsl), load_model=self.load_model, ds_fine=self.ds_fine,
                    results_root=self.results_root, plot=bool(self.plot), resume=bool(self.resume),
                    useropts_kwargs=dict(self.kw_fp, vp_overrides=self.base_ov))
        return dict(fingerprint=self.fp, name=self.name, out_dir=self.out_dir, stem=self.stem,
                    created=_now(), runs=[_now()], call=call,
                    effective_config=dict(AeroConfig=self.AeroConfig, ATD=self.eATD,
                                          Electric_4Motors=self.eEM4, TyreModel=self.TyreModel,
                                          tyre_set=bc.tyre_set, mesh=bc.mesh, OPT_ds=bc.OPT_ds,
                                          OPT_d=bc.OPT_d, N=int(self.disc["N"])),
                    specs=self.kept, dropped=self.dropped, fields=self.flags,
                    classification=self.cinfo, fingerprint_inputs=fp_in, model_hash=self.mhash,
                    env=dict(python=sys.version.split()[0], numpy=np.__version__,
                             scipy=scipy_version, casadi=self.ca_version, platform=sys.platform,
                             cpu_count=os.cpu_count(), free_ram_mb=S.free_ram_mb(),
                             cwd=os.getcwd(), repo=_REPO, git=_git_info()))

    # ---- 5: design -----------------------------------------------------------------
    def design(self):
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            U, X = S.sample_box(self.kept, self.n, self.sampler, self.seed)
        self.rewarn(rec)
        lo = np.array([s[1] for s in self.kept])
        hi = np.array([s[2] for s in self.kept])
        self.u0 = (np.array([self.base_vals[f] for f in self.fields]) - lo) / (hi - lo)
        outside = [f"{f}={self.base_vals[f]:g}" for f, u in zip(self.fields, self.u0) if u < 0 or u > 1]
        if outside:
            self.warn(f"the baseline setup lies outside the box for {', '.join(outside)}")
        self.stored = None
        if self.resumed and os.path.isfile(self.paths["samples"]):
            st = S.read_csv(self.paths["samples"])
            try:
                Xs = np.array([[float(r[f]) for f in self.fields] for r in st[1:]])
                Us = np.array([[float(r[f"u_{j + 1}"]) for j in range(len(self.fields))] for r in st[1:]])
                same = Xs.shape == X.shape and np.array_equal(Xs, X) and np.array_equal(Us, U)
            except (KeyError, ValueError):
                same = False
            if not same:
                raise SweepError(f"{self.paths['samples']} does not match the regenerated design of "
                                 "this sweep; the stored design is authoritative, refusing to mix them")
            X, U, self.stored = Xs, Us, st
        self.U, self.X = U, X
        self.row_vals = [dict(self.base_vals)] + [{f: float(x) for f, x in zip(self.fields, X[i])}
                                                  for i in range(self.n)]
        self.row_ovs = [{}] + self.row_vals[1:]
        self.timings["design_s"] = time.perf_counter() - t0

    # ---- 6: NLP pool + hub start ---------------------------------------------------
    def task_base(self):
        return dict(circuit=self.circuit, vi=self.vi, ni=self.ni, AeroConfig=self.AeroConfig,
                    ATD=self.eATD, Electric_4Motors=self.eEM4, TyreModel=self.TyreModel,
                    base_ov=self.base_ov, useropts=self.kw, hop_ipopt=self.hop_ip,
                    cold_ipopt=self.cold_ip, fingerprint=self.fp, model_hash=self.mhash,
                    tol_branch_s=float(self.tol_branch_s), bridge_steps=list(self.bridge_steps),
                    hub_vals=dict(self.base_vals), hub_path=self.paths["hub"],
                    hub_lap=None if not self.hub or not self.hub.get("check_ok") else float(self.hub["lap_s"]),
                    hub_linear_solver=None if not self.hub else self.hub.get("linear_solver"))

    def start_hub(self):
        """Create the NLP pool and submit the hub task first, so it runs during the screen."""
        self.t_hub0 = time.perf_counter()
        self.hub_cold_running = False
        if not self.confirm:
            return
        p = self.paths
        if self.resumed and os.path.isfile(p["hub_json"]) and os.path.isfile(p["hub"]):
            hj = S.read_json(p["hub_json"])
            if (hj.get("fingerprint") == self.fp and hj.get("check_ok")
                    and hj.get("sha256") == S.sha256_file(p["hub"])):
                self.hub = hj
                self.say(f"[sweep] hub reused from {p['hub']} (lap {hj['lap_s']:.6f} s)")
        bc = self.base_ctx
        n_w = nlp_structure(23, len(bc.input_keys), 0, self.disc["N"], bc.OPT_d, self.cinfo["nh"])["n_w"]
        self.rss_mb = 130.0 + 0.072 * n_w
        n_conf = max(1, min(self.n, self.top_k + self.n_probes))
        self.P = (S.auto_workers(n_conf, rss_mb=self.rss_mb) if self.workers == "auto"
                  else int(self.workers))
        spec = dict(root=os.getcwd(), circuit=self.circuit,
                    circuits_dir=self.kw.get("circuits_dir", "Circuits"),
                    data_dir=self.kw.get("data_dir", "Data"), pin_hsl=bool(self.pin_hsl),
                    linear_solver=self.kw.get("linear_solver", "ma57"), hsl_dir=self.kw.get("hsl_dir"))
        self.runner = _Runner(max(self.P, 1), spec, inline=(self.P == 0), say=self.say)
        if self.hub is not None:
            return
        self.hub = dict(fingerprint=self.fp, source=self.base_kind or "cold",
                        source_path=self.base_path, cold=None, check=None, rehubbed=False,
                        check_ok=False)
        tb = self.task_base()
        if self.base_kind is None:
            self.runner.submit(dict(tb, id="hub_cold", kind="hub_cold",
                                    log=os.path.join(p["logs"], "hub_cold.log")))
            self.hub_cold_running = not self.runner.inline
            self.say(f"[sweep] hub: cold solve submitted ({self.P or 'in-process'} NLP worker(s), "
                     f"~{self.rss_mb:.0f} MB each); screening meanwhile")
            return
        if self.base_kind == "file":
            _copy_atomic(self.base_path, p["hub"])
        else:
            _savemat_atomic(p["hub"], self.base_src)
        self.runner.submit(dict(tb, id="hub_check", kind="hub_check", rehub_path=p["rehub"],
                                log=os.path.join(p["logs"], "hub_check.log")))
        self.say(f"[sweep] hub: the {self.base_kind} base is hub.mat; check submitted")

    # ---- 7-8: screen, shortlist, samples.csv ---------------------------------------
    def screen(self):
        t0 = time.perf_counter()
        n = self.n
        st = self.stored
        if st is not None and all(k in st[0] for k in ("qss_lap_s", "status", "vi_feasible")):
            q = dict(lap_time=np.array([S.num(r["qss_lap_s"]) for r in st]),
                     status=[r["status"] for r in st],
                     vi_feasible=np.array([r["vi_feasible"] == "True" for r in st]),
                     error=[r.get("error") or None for r in st], warnings=[], workers=0,
                     wall_s=None, reused=True)
            self.say(f"[sweep] screen: {n + 1} QSS laps reused from samples.csv")
        else:
            scr = dict(vi=self.vi, AeroConfig=self.AeroConfig, ATD=self.eATD,
                       Electric_4Motors=self.eEM4, load_model=self.load_model,
                       ds_fine=self.ds_fine, vp_overrides=self.base_ov, **self.kw)
            r0 = screen_batch(self.circuit, self.row_ovs[:1], workers=1, **scr)
            if self.screen_workers == "auto":
                sw = (1 if (self.hub_cold_running or n * r0["wall_s"] < 3.0)
                      else S.auto_workers(math.ceil(n / 256)))
            else:
                sw = int(self.screen_workers)
            rr = screen_batch(self.circuit, self.row_ovs[1:], workers=sw, **scr)
            q = dict(lap_time=np.concatenate([r0["lap_time"], rr["lap_time"]]),
                     status=r0["status"] + rr["status"],
                     vi_feasible=np.concatenate([r0["vi_feasible"], rr["vi_feasible"]]),
                     error=r0["error"] + rr["error"],
                     warnings=S.dedup(r0["warnings"] + rr["warnings"]), workers=sw, reused=False)
            q["wall_s"] = time.perf_counter() - t0
            for msg in q["warnings"]:
                self.warn(f"QSS screen: {msg}")
            n_bad = sum(s != "ok" for s in q["status"])
            self.say(f"[sweep] screen: {n + 1} setups in {q['wall_s']:.2f} s "
                     f"({(n + 1) / max(q['wall_s'], 1e-9):.0f} setups/s, {sw} worker(s)), "
                     f"{n_bad} invalid")
            if q["status"][0] != "ok":
                self.warn(f"the baseline setup fails the QSS screen ({q['error'][0]}); "
                          "QSS deltas are undefined")
        self.qss = q
        self.timings["screen_s"] = time.perf_counter() - t0
        lap = q["lap_time"]
        self.lap = lap
        self.ok = np.array([s == "ok" for s in q["status"]])
        order = sorted([i for i in range(n + 1) if self.ok[i]], key=lambda i: (lap[i], i))
        qrank = {i: r + 1 for r, i in enumerate(order)}
        self.shortlist = S.select_shortlist(lap, self.ok, self.top_k, self.n_probes)
        self.kind_of = dict(self.shortlist)
        self.top_ids = [i for i, k in self.shortlist if k == "top"]
        d = len(self.fields)
        header = (["row_id", "kind"] + [f"u_{j + 1}" for j in range(d)] + self.fields
                  + ["qss_lap_s", "qss_delta_s", "qss_rank", "status", "vi_feasible", "selected", "error"])
        self.samples = []
        for i in range(n + 1):
            u = self.u0 if i == 0 else self.U[i - 1]
            self.samples.append(dict(
                row_id=i, kind="baseline" if i == 0 else "sample",
                **{f"u_{j + 1}": float(u[j]) for j in range(d)}, **self.row_vals[i],
                qss_lap_s=float(lap[i]), qss_delta_s=float(lap[i] - lap[0]),
                qss_rank=qrank.get(i), status=q["status"][i], vi_feasible=bool(q["vi_feasible"][i]),
                selected=self.kind_of.get(i, ""), error=q["error"][i]))
        S.write_csv_atomic(self.paths["samples"], header, self.samples)
        self.files["samples"] = self.paths["samples"]
        self.say(f"[sweep] shortlist: baseline + top {self.top_ids} + probes "
                 f"{[i for i, k in self.shortlist if k == 'probe']}")

    # ---- 9: hub cold -> check (-> re-hub once) --------------------------------------
    def finish_hub(self):
        p, hub = self.paths, self.hub
        if not hub.get("check_ok"):
            if "hub_cold" in self.runner.tasks:
                res = self.runner.collect(["hub_cold"])["hub_cold"]
                h = res.get("hop") or {}
                if res.get("status") != "done" or h.get("status") not in GOOD_STATUS:
                    raise SweepError(f"the cold hub solve failed ({res.get('status')}: "
                                     f"{res.get('reason') or h.get('status')}); see "
                                     f"{os.path.join(p['logs'], 'hub_cold.log')}")
                hub["cold"] = dict(iters=h["iters"], wall_s=h["wall"], status=h["status"],
                                   lap_s=h["lap"], init_s=h.get("init_s"), mode=h["mode"],
                                   linear_solver=h["linear_solver"], w_sha=h["w_sha"])
                self.say(f"[sweep] hub: cold solve {h['lap']:.6f} s, {h['iters']} iterations, "
                         f"{h['wall']:.1f} s ({h['linear_solver']})")
                self.runner.submit(dict(self.task_base(), id="hub_check", kind="hub_check",
                                        rehub_path=p["rehub"],
                                        log=os.path.join(p["logs"], "hub_check.log")))
            for attempt, tid in ((1, "hub_check"), (2, "hub_check2")):
                res = self.runner.collect([tid])[tid]
                h = res.get("hop") or {}
                log_path = os.path.join(p["logs"], f"{tid}.log")
                if (res.get("status") != "done" or h.get("mode") != "full+duals"
                        or h.get("status") not in GOOD_STATUS):
                    raise SweepError(f"hub check failed: warm start mode {h.get('mode')!r}, IPOPT "
                                     f"{h.get('status') or res.get('reason')!r} (needs 'full+duals' "
                                     f"and a status in {GOOD_STATUS}); see {log_path}")
                hub["check" if attempt == 1 else "rehub_check"] = dict(
                    iters=h["iters"], wall_s=h["wall"], status=h["status"], mode=h["mode"],
                    lap_s=h["lap"], linear_solver=h["linear_solver"], w_sha=h["w_sha"],
                    warnings=res.get("warnings", []), log=log_path)
                if h["iters"] == 0:
                    break
                if attempt == 2:
                    raise SweepError(f"the re-hubbed base still needs {h['iters']} IPOPT iterations "
                                     f"at the base setup; see {log_path}")
                self.warn(f"base is not a solution of this NLP at the base setup ({h['iters']} "
                          f"iterations to {h['lap']:.6f} s); re-hubbed")
                os.replace(p["rehub"], p["hub"])
                hub["rehubbed"] = True
                self.runner.submit(dict(self.task_base(), id="hub_check2", kind="hub_check",
                                        rehub_path=p["rehub"],
                                        log=os.path.join(p["logs"], "hub_check2.log")))
            final = hub.get("rehub_check") or hub["check"]
            from functions.importfile import load_solution
            hub.update(check_ok=True, lap_s=final["lap_s"], linear_solver=final["linear_solver"],
                       check_iters=final["iters"], sha256=S.sha256_file(p["hub"]),
                       w_sha=S.array_sha(load_solution(p["hub"]).nlp.w_opt), model_hash=self.mhash)
            S.write_json_atomic(p["hub_json"], hub)
            self.hub = S.read_json(p["hub_json"])     # the record exactly as a resume reads it
            self.say(f"[sweep] hub check: {final['mode']}, {final['status']}, {final['iters']} "
                     f"iterations, lap {final['lap_s']:.6f} s ({final['wall_s']:.1f} s)")
        self.timings["hub_s"] = time.perf_counter() - self.t_hub0
        self.files.update(hub=p["hub"], hub_json=p["hub_json"])

    # ---- 10: confirmations ----------------------------------------------------------
    def confirm_task(self, rid, kind, row, roundtrip, rt_only=False):
        r = self.paths["rows"]
        return dict(self.task_base(), id=f"row_{rid}{'_rt' if rt_only else ''}", kind="confirm",
                    row_id=rid, row_kind=kind, row=dict(row), roundtrip=bool(roundtrip),
                    rt_only=bool(rt_only), row_mat=os.path.join(r, f"{rid}.mat"),
                    rt_path=os.path.join(r, f"{rid}_rt.mat"),
                    log=os.path.join(self.paths["logs"], f"{rid}{'_rt' if rt_only else ''}.log"))

    def on_row(self, task, res):
        """Write rows/<id>.json from a task result (round-trip-only results are merged
        into the forward record) and print one progress line."""
        rid = task["row_id"]
        path = os.path.join(self.paths["rows"], f"{rid}.json")
        if task.get("rt_only"):
            rec = S.read_json(path)
            if res.get("status") == "rt_done":
                rec.update({k: res.get(k) for k in _RT_FIELDS})
                rec["delta_rev_s"] = (rec["nlp_lap_s"] - res["rt_lap_s"]) if res.get("rt_ok") else None
                rec["hops"] = list(rec.get("hops") or []) + list(res.get("hops") or [])
                rec["rt_wave"] = True
            else:                       # the round-trip task itself died: retried on resume
                rec["rt_reason"] = f"{res.get('status')}: {res.get('reason')}"
            rec["warnings"] = S.dedup(list(rec.get("warnings") or []) + list(res.get("warnings") or []))
            rec["wall_s"] = (rec.get("wall_s") or 0.0) + (res.get("task_wall_s") or 0.0)
        elif res.get("status") in ("accepted", "failed", "solver_mismatch"):
            rec = dict(res)
            rec["wall_s"] = res.get("task_wall_s")
        else:                           # crashed / code_changed / error: retried on resume
            rec = dict(row_id=rid, kind=task["row_kind"], overrides=dict(task["row"]),
                       fingerprint=self.fp, model_hash=self.mhash, status=res.get("status", "error"),
                       reason=res.get("reason"), warnings=res.get("warnings", []),
                       wall_s=res.get("task_wall_s"))
        rec["kind"] = task["row_kind"]
        S.write_json_atomic(path, rec)
        self.records[rid] = S.read_json(path)   # the record exactly as a resume reads it
        self.new_rows.add(rid)
        r = self.records[rid]
        if r.get("rt_better_baseline_file"):
            self.warn(f"row {rid}: its round trip found a better baseline branch ({r['rt_lap_s']:.6f} s "
                      f"< hub {self.hub['lap_s']:.6f} s); consider base=r'{r['rt_better_baseline_file']}'")
        self._row_line(r)

    def _row_line(self, r):
        rid = r["row_id"]
        if r.get("status") != "accepted":
            self.say(f"[sweep]   row {rid:>4} ({r.get('kind')}): {r.get('status')} - {r.get('reason')}")
            return
        rt = ""
        if r.get("rt_ok") is not None:
            rt = (f", round trip {1e3 * r['rt_residual_s']:+.3f} ms "
                  f"{'ok' if r.get('branch_ok') else 'OFF-BRANCH'}" if r.get("rt_ok")
                  else f", round trip failed ({r.get('rt_reason')})")
        q = f"QSS {self.lap[rid]:.4f} s, " if rid < len(self.lap) else ""
        wall = r.get("wall_s")
        self.say(f"[sweep]   row {rid:>4} ({r.get('kind')}): {q}NLP {r['nlp_lap_s']:.6f} s "
                 f"({1e3 * (r['nlp_lap_s'] - self.hub['lap_s']):+.3f} ms), {r.get('iters')} it, "
                 f"{wall if wall is not None else float('nan'):.1f} s, {r.get('warm_start_mode')} "
                 f"{r.get('path')}{rt}")

    def run_confirms(self):
        t0 = time.perf_counter()
        todo = []
        for rid, kind in self.shortlist:
            if rid == 0:
                continue
            prev = (_load_row(os.path.join(self.paths["rows"], f"{rid}.json"), self.fp, self.row_vals[rid])
                    if self.resumed else None)
            if prev is not None:                 # its shortlist kind follows this call's top_k
                self.records[rid] = dict(prev, kind=kind)
            else:
                todo.append((rid, kind))
        todo.sort(key=lambda t: (-float(np.linalg.norm(self.U[t[0] - 1] - self.u0)), t[0]))
        if self.records:
            self.say(f"[sweep] confirm: {len(self.records)} row(s) reused from rows/")
        if todo:
            self.say(f"[sweep] confirm: {len(todo)} warm hop(s) from the hub "
                     f"({self.P or 'in-process'} worker(s), longest hops first)")
        for rid, kind in todo:
            self.runner.submit(self.confirm_task(rid, kind, self.row_ovs[rid], self.roundtrip == "all"))
        self.runner.collect([f"row_{rid}" for rid, _ in todo], on_done=self.on_row)
        wave = _rt_wave(self.records, self.roundtrip)
        if wave:
            self.say(f"[sweep] round trips (second wave): rows {wave}")
            for rid in wave:
                self.runner.submit(self.confirm_task(rid, self.records[rid]["kind"], self.row_ovs[rid],
                                                     True, rt_only=True))
            self.runner.collect([f"row_{rid}_rt" for rid in wave], on_done=self.on_row)
        self.timings["confirm_s"] = time.perf_counter() - t0

    # ---- 11: finish --------------------------------------------------------------------
    def run_finish(self):
        t0 = time.perf_counter()
        fin = dict(status="skipped", fingerprint=self.fp)
        eligible = [list(s) for s in self.kept if self.flags[s[0]]["promotion_safe"]]
        excluded = {}
        for f in self.fields:
            fl = self.flags[f]
            if fl["promotion_safe"]:
                continue
            why = ("not a vehParams primary (a Pacejka coefficient; optimise_design promotes vp "
                   "fields)" if not fl["primary"] else
                   "vehParams derives other quantities from it (phase 2), so promoting it would "
                   "leave them stale" if not fl["phase2_free"] else
                   "it moves the NLP scaling x_s / u_s, which a symbolic parameter breaks"
                   if fl["scale_changed"] else "not NLP-live")
            excluded[f] = why
            self.warn(f"finish: {f} is not promotable ({why}); excluded")
        fin.update(eligible=eligible, excluded=excluded)
        self.fin = fin
        if self.TyreModel != "CombinedSlip":
            fin["reason"] = "optimise_design builds the CombinedSlip NLP only"
            self.warn(f"finish skipped: {fin['reason']}")
        elif not eligible:
            fin["reason"] = "no promotion-safe field"
            self.warn("finish skipped: no promotion-safe field")
        else:
            self._finish_solve(fin, eligible)
        self.timings["finish_s"] = time.perf_counter() - t0

    def _finish_solve(self, fin, eligible):
        p = self.paths
        hub_lap = float(self.hub["lap_s"])
        acc = {rid: r for rid, r in self.records.items()
               if r.get("status") == "accepted" and r.get("kind") != "finish"}
        _, winner = min([(hub_lap, 0)] + [(r["nlp_lap_s"], rid) for rid, r in acc.items()])
        wvals = (dict(self.base_vals) if winner == 0
                 else {f: float(acc[winner]["overrides"][f]) for f in self.fields})
        warm = p["hub"] if winner == 0 else os.path.join(p["rows"], f"{winner}.mat")
        fjson = os.path.join(p["finish"], "finish.json")
        prev = S.read_json(fjson) if os.path.isfile(fjson) else None
        if (prev and prev.get("fingerprint") == self.fp and prev.get("winner_row") == winner
                and prev.get("eligible") == eligible and prev.get("design_status") is not None):
            fin.clear()
            fin.update(prev)
            self.say(f"[sweep] finish: reused {fjson}")
        else:
            self.say(f"[sweep] finish: optimise_design over {[s[0] for s in eligible]} "
                     f"warm-started from row {winner}")
            self.runner.submit(dict(self.task_base(), id="finish", kind="finish", specs=eligible,
                                    winner=wvals, warm_path=warm,
                                    design_mat=os.path.join(p["finish"], "design.mat"),
                                    log=os.path.join(p["logs"], "finish.log")))
            res = self.runner.collect(["finish"])["finish"]
            fin.update(winner_row=winner, winner=wvals, warm_path=warm,
                       design_status=res.get("status"), reason=res.get("reason"))
            fin.update({k: res.get(k) for k in ("opt_lap_s", "iters", "warm_start_mode", "ipopt_status",
                                                "linear_solver", "p_star", "at_bound", "wall_s",
                                                "result_file")})
            fin["status"] = res.get("status")
            if fin["status"] == "skipped":
                self.warn(f"finish skipped: {res.get('reason')}")
            elif fin["status"] != "done":
                self.warn(f"finish: optimise_design failed ({res.get('status')}: {res.get('reason')})")
            S.write_json_atomic(fjson, fin)
        self.files["finish"] = fjson
        if fin.get("status") != "done":
            return
        rid = self.n + 1
        row = {**fin["winner"], **fin["p_star"]}
        prev_row = _load_row(os.path.join(p["rows"], f"{rid}.json"), self.fp, row)
        if prev_row is None:
            self.runner.submit(self.confirm_task(rid, "finish", row, self.roundtrip != "none"))
            self.runner.collect([f"row_{rid}"], on_done=self.on_row)
        else:
            self.records[rid] = prev_row
        scr = screen_batch(self.circuit, [row], vi=self.vi, AeroConfig=self.AeroConfig, ATD=self.eATD,
                           Electric_4Motors=self.eEM4, load_model=self.load_model,
                           ds_fine=self.ds_fine, workers=1, vp_overrides=self.base_ov, **self.kw)
        fin.update(confirm_row=rid, qss_lap_s=float(scr["lap_time"][0]),
                   confirm_status=self.records[rid].get("status"),
                   confirm_nlp_lap_s=self.records[rid].get("nlp_lap_s"))
        S.write_json_atomic(fjson, fin)
        self.fin = S.read_json(fjson)
        self.say(f"[sweep] finish: p* {fin['p_star']} (at bound {fin['at_bound']}); optimise_design "
                 f"{fin['opt_lap_s']:.6f} s in {fin['iters']} it ({fin['warm_start_mode']}); star "
                 f"re-confirmation row {rid}: {fin['confirm_status']}")

    # ---- 12: report -------------------------------------------------------------------
    def ranked_rows(self):
        """confirmed.csv rows: accepted metric rows ranked by (nlp_lap, row_id), then the
        finish row and the failures (with their reasons)."""
        hub = self.hub
        hub_lap = float(hub["lap_s"])
        chk = hub.get("rehub_check") or hub["check"]
        recs = {0: dict(row_id=0, kind="baseline", overrides=dict(self.base_vals), status="accepted",
                        nlp_lap_s=hub_lap, delta_rev_s=0.0, rt_residual_s=0.0, branch_ok=True,
                        rt_ok=True, iters=chk["iters"], rt_iters=None, wall_s=chk["wall_s"],
                        warm_start_mode=chk["mode"], path="hub", linear_solver=chk["linear_solver"],
                        w_sha=chk["w_sha"], result_file=self.paths["hub"], repro=None, reason=None)}
        recs.update(self.records)
        metric = [rid for rid, r in recs.items()
                  if r.get("status") == "accepted" and r.get("kind") != "finish"]
        floor = S.noise_floor([recs[rid] for rid in metric], self.resolve_s)
        unres = S.unresolved_pairs({rid: recs[rid]["nlp_lap_s"] for rid in metric}, floor)
        ranked = sorted(metric, key=lambda rid: (recs[rid]["nlp_lap_s"], rid))
        rest = sorted([rid for rid in recs if rid not in metric],
                      key=lambda rid: (recs[rid].get("kind") != "finish", rid))
        q0 = self.samples[0]["qss_lap_s"]
        fin_row = (self.fin or {}).get("confirm_row")
        rows = []
        for k, rid in enumerate(ranked + rest, 1):
            r = recs[rid]
            acc = r.get("status") == "accepted"
            if rid < len(self.samples):
                qlap, qrank = self.samples[rid]["qss_lap_s"], self.samples[rid]["qss_rank"]
            else:
                qlap, qrank = (self.fin.get("qss_lap_s") if rid == fin_row else None), None
            ov = r.get("overrides") or {}
            rf = r.get("result_file")
            rows.append(dict(
                nlp_rank=k if rid in ranked else None, row_id=rid, kind=r.get("kind"),
                **{f: ov.get(f) for f in self.fields},
                qss_lap_s=qlap, qss_rank=qrank,
                qss_delta_s=(qlap - q0) if _fin(qlap) and _fin(q0) else None,
                nlp_lap_s=r.get("nlp_lap_s") if acc else None,
                nlp_delta_s=(r["nlp_lap_s"] - hub_lap) if acc else None,
                delta_rev_s=r.get("delta_rev_s") if acc else None,
                rt_residual_s=r.get("rt_residual_s") if acc else None,
                # unknown (empty) unless a round trip converged
                branch_ok=r.get("branch_ok") if (acc and r.get("rt_ok")) else None,
                rt_ok=r.get("rt_ok") if acc else None,
                unresolved_with=unres.get(rid, []) if rid in ranked else [],
                iters=r.get("iters"), rt_iters=r.get("rt_iters"), wall_s=r.get("wall_s"),
                warm_start_mode=r.get("warm_start_mode"), path=r.get("path"), status=r.get("status"),
                linear_solver=r.get("linear_solver"), w_sha=r.get("w_sha"),
                result_file=os.path.relpath(rf, self.out_dir) if rf and os.path.isabs(rf) else rf,
                repro=r.get("repro"), reason=r.get("reason"),
                overrides=dict(ov)))
        return rows, floor

    def report_html(self, table, floor):
        import plotly.graph_objects as go
        acc = [r for r in table if r["status"] == "accepted" and r["qss_delta_s"] is not None]
        fig1 = go.Figure()
        for kind, color in (("baseline", "#444444"), ("top", "#1f77b4"), ("probe", "#ff7f0e"),
                            ("finish", "#2ca02c")):
            pts = [r for r in acc if r["kind"] == kind]
            if not pts:
                continue
            up, dn = [], []
            for r in pts:              # bracket [delta_rev, delta_fwd] around the forward delta
                d = 0.0 if r["delta_rev_s"] is None else 1e3 * (r["delta_rev_s"] - r["nlp_delta_s"])
                up.append(max(d, 0.0))
                dn.append(max(-d, 0.0))
            fig1.add_trace(go.Scatter(
                x=[1e3 * r["qss_delta_s"] for r in pts], y=[1e3 * r["nlp_delta_s"] for r in pts],
                mode="markers+text", name=kind, text=[str(r["row_id"]) for r in pts],
                textposition="top center", marker=dict(size=10, color=color),
                error_y=dict(type="data", array=up, arrayminus=dn, visible=True)))
        if acc:
            v = [1e3 * r["qss_delta_s"] for r in acc] + [1e3 * r["nlp_delta_s"] for r in acc] + [0.0]
            fig1.add_trace(go.Scatter(x=[min(v), max(v)], y=[min(v), max(v)], mode="lines",
                                      name="y = x", line=dict(dash="dash", color="#999999")))
        m = (self.metrics or {}).get("all", {})
        rho = m.get("spearman_rho")
        fig1.update_layout(
            title=(f"{self.name}: NLP vs QSS lap delta to the baseline (bars: [delta_rev, delta_fwd]); "
                   f"Spearman {rho if rho is not None else float('nan'):.3f}, "
                   f"noise floor {1e3 * floor:.3f} ms"),
            xaxis_title="QSS delta [ms]", yaxis_title="NLP delta [ms]", template="plotly_white")
        code = {"": 0, "probe": 1, "top": 2, "baseline": 3}
        laps = [s["qss_lap_s"] for s in self.samples]
        good = [v for v in laps if _fin(v)]
        laps = [v if _fin(v) else (max(good) if good else 0.0) for v in laps]
        dims = [dict(label=f, values=[s[f] for s in self.samples], range=[lo, hi])
                for f, lo, hi in self.kept]
        dims.append(dict(label="QSS lap [s]", values=laps))
        dims.append(dict(label="shortlist", values=[code.get(s["selected"], 0) for s in self.samples],
                         tickvals=[0, 1, 2, 3], ticktext=["sample", "probe", "top", "baseline"],
                         constraintrange=[0.5, 3.5]))
        fig2 = go.Figure(go.Parcoords(
            line=dict(color=laps, colorscale="Viridis", showscale=True, colorbar=dict(title="QSS lap [s]")),
            dimensions=dims, unselected=dict(line=dict(opacity=0.12))))
        fig2.update_layout(title=f"{len(self.samples)} setups (row 0 = baseline) coloured by QSS lap; "
                                 "the shortlist is highlighted", template="plotly_white")
        S.write_text_atomic(self.paths["report"],
                            "<html><head><meta charset='utf-8'><title>" + self.name + "</title></head><body>"
                            + fig1.to_html(full_html=False, include_plotlyjs=True)
                            + fig2.to_html(full_html=False, include_plotlyjs=False) + "</body></html>\n")
        self.files["report"] = self.paths["report"]

    def print_table(self, table):
        def num(v, fmt, scale=None):
            return format(v if scale is None else scale * v, fmt) if _fin(v) else "-"

        self.say("")
        self.say(f"{'rank':>4} {'row':>4} {'kind':<8} {'QSS lap':>9} {'NLP lap':>10} {'NLP d ms':>9} "
                 f"{'QSS d ms':>9} {'iters':>5} {'wall s':>6}  {'mode':<10} {'path':<8} branch")
        for r in table:
            head = (f"{num(r['nlp_rank'], 'd') if r['nlp_rank'] else '-':>4} {r['row_id']:>4} "
                    f"{str(r['kind']):<8} {num(r['qss_lap_s'], '9.4f'):>9} ")
            if r["status"] != "accepted":
                self.say(head + f"{'':>10} {'':>9} {num(r['qss_delta_s'], '+9.3f', 1e3):>9} "
                                f"{'':>5} {num(r['wall_s'], '6.1f'):>6}  {r['status']}: {r['reason']}")
                continue
            br = ("rt-fail" if r.get("rt_ok") is False else "-" if r["branch_ok"] is None
                  else "ok" if r["branch_ok"] else "OFF")
            if r["unresolved_with"]:
                br += " ~" + ",".join(map(str, r["unresolved_with"]))
            self.say(head + f"{num(r['nlp_lap_s'], '10.6f'):>10} {num(r['nlp_delta_s'], '+9.3f', 1e3):>9} "
                            f"{num(r['qss_delta_s'], '+9.3f', 1e3):>9} {num(r['iters'], '5d'):>5} "
                            f"{num(r['wall_s'], '6.1f'):>6}  {str(r['warm_start_mode']):<10} "
                            f"{str(r['path']):<8} {br}")
        self.say("")

    def throughput(self):
        q = self.qss
        n_rows = self.n + 1
        hops = [h for rid in self.new_rows for h in (self.records.get(rid, {}).get("hops") or [])]
        conf_s = self.timings.get("confirm_s") or 0.0
        n_new = sum(1 for rid in self.new_rows
                    if self.records.get(rid, {}).get("status") == "accepted"
                    and self.records[rid].get("kind") != "finish")
        ran = self.runner.n_submitted if self.runner else 0
        return dict(
            screen_setups=0 if q.get("reused") else n_rows, screen_wall_s=q.get("wall_s"),
            screen_setups_per_s=None if q.get("reused") else n_rows / max(q["wall_s"], 1e-12),
            screen_workers=q.get("workers"), nlp_workers=self.P, nlp_tasks_run=ran,
            hops=len(hops),
            hop_wall_mean_s=float(np.mean([h["wall"] for h in hops])) if hops else None,
            hop_wall_median_s=float(np.median([h["wall"] for h in hops])) if hops else None,
            hop_iters_median=float(np.median([h["iters"] for h in hops])) if hops else None,
            confirmations=n_new,
            confirmations_per_hour=(3600.0 * n_new / conf_s) if (n_new and conf_s > 0) else None,
            hsl_pinned=bool(self.pin_hsl), rss_mb_per_worker_estimate=getattr(self, "rss_mb", None))

    def _history(self):
        """Timings / throughput of the earlier calls of a resumed sweep (each call
        rewrites summary.json; a resume that runs nothing keeps the original numbers)."""
        if not (self.resumed and os.path.isfile(self.paths["summary"])):
            return []
        try:
            old = S.read_json(self.paths["summary"])
        except Exception:
            return []
        return list(old.get("history") or []) + [dict(
            run=len(old.get("history") or []) + 1, timings=old.get("timings"),
            throughput=old.get("throughput"), n_tasks_run=old.get("n_tasks_run"))]

    def finalise(self):
        t0 = time.perf_counter()
        self.metrics, floor, table = {}, None, []
        if not self.confirm:
            ids = [i for i in range(self.n + 1) if self.ok[i]]
            bid = min(ids, key=lambda i: (self.lap[i], i)) if ids else 0
            best = dict(vp_overrides={**self.base_ov, **self.row_ovs[bid]}, nlp_lap_s=None, delta_s=None,
                        row_id=bid, qss_lap_s=float(self.lap[bid]), basis="QSS only (confirm=False)")
        else:
            table, floor = self.ranked_rows()
            mrows = [dict(row_id=r["row_id"], kind=r["kind"], qss_lap=r["qss_lap_s"],
                          nlp_lap=r["nlp_lap_s"], branch_ok=r["branch_ok"]) for r in table
                     if r["status"] == "accepted" and r["kind"] in ("baseline", "top", "probe")]
            self.metrics = S.rank_metrics(mrows, floor, self.top_ids, seed=self.seed)
            b = next(r for r in table if r["nlp_rank"] == 1)
            best = dict(vp_overrides={**self.base_ov, **(b["overrides"] if b["row_id"] else {})},
                        nlp_lap_s=b["nlp_lap_s"], delta_s=b["nlp_delta_s"], row_id=b["row_id"],
                        kind=b["kind"], unresolved_with=b["unresolved_with"])
            m = self.metrics
            if m["n"] >= 3 and not m["screen_trusted"]:
                self.warn(f"the QSS screen ranking is not trusted on this box (Spearman "
                          f"{m['all']['spearman_rho']:.3f}, concordance {m['concordance']['value']:.3f}, "
                          f"top-1 regret {1e3 * m['top1_regret_s']:.3f} ms vs floor {1e3 * floor:.3f} ms): "
                          "raise top_k / n_probes, or Item 11 (free-line QSS)")
            S.write_csv_atomic(self.paths["confirmed"], ["nlp_rank", "row_id", "kind"] + self.fields + _CSV_TAIL,
                               table)
            self.files.update(confirmed=self.paths["confirmed"], rows=self.paths["rows"],
                              logs=self.paths["logs"])
        failures = [dict(row_id=r["row_id"], kind=r["kind"], status=r["status"], reason=r["reason"])
                    for r in table if r["status"] != "accepted"]
        self.timings["report_s"] = time.perf_counter() - t0
        if self.plot:
            try:
                self.report_html(table, floor or 0.0) if self.confirm else None
            except Exception as exc:
                self.warn(f"report.html skipped: {type(exc).__name__}: {exc}")
        self.timings["total_s"] = time.perf_counter() - self.t_start
        thr = self.throughput()
        self.files["summary"] = self.paths["summary"]
        summary = dict(fingerprint=self.fp, name=self.name, out_dir=self.out_dir, plan=self.plan,
                       fields=self.flags, dropped=self.dropped, warnings=self.wlist, hub=self.hub,
                       shortlist=[list(s) for s in self.shortlist], metrics=self.metrics,
                       noise_floor_s=floor, best=best, finish=self.fin, failures=failures,
                       timings=self.timings, throughput=thr, n_tasks_run=thr["nlp_tasks_run"],
                       files=self.files, history=self._history())
        S.write_json_atomic(self.paths["summary"], summary)
        if self.confirm:
            self.print_table(table)
            m = self.metrics
            self.say(f"[sweep] best: row {best['row_id']} ({best['kind']}) NLP {best['nlp_lap_s']:.6f} s, "
                     f"{1e3 * best['delta_s']:+.3f} ms vs the baseline (noise floor {1e3 * floor:.3f} ms); "
                     f"Spearman {m['all']['spearman_rho']:.3f}, Kendall {m['all']['kendall_tau_b']:.3f}, "
                     f"concordance {m['concordance']['value']:.3f}, screen_trusted {m['screen_trusted']}")
        self.say(f"[sweep] done in {self.timings['total_s']:.1f} s ({thr['nlp_tasks_run']} NLP task(s)) "
                 f"-> {self.out_dir}")
        return SweepResult(name=self.name, out_dir=self.out_dir, fingerprint=self.fp, fields=self.flags,
                           dropped=self.dropped, samples=self.samples, shortlist=self.shortlist,
                           confirmed=table, metrics=self.metrics, best=best, finish=self.fin or {},
                           hub=self.hub or {}, warnings=list(self.wlist), timings=self.timings,
                           throughput=thr, files=dict(self.files), n_tasks_run=thr["nlp_tasks_run"])

    def baseline_only(self):
        """No field left after validation / classification: plan + summary, no NLP."""
        self.warn("no field left to sweep (all dropped): returning a baseline-only result, no NLP solve")
        scr = screen_batch(self.circuit, [{}], vi=self.vi, AeroConfig=self.AeroConfig, ATD=self.eATD,
                           Electric_4Motors=self.eEM4, load_model=self.load_model, ds_fine=self.ds_fine,
                           workers=1, vp_overrides=self.base_ov, **self.kw)
        sample = dict(row_id=0, kind="baseline", qss_lap_s=float(scr["lap_time"][0]),
                      status=scr["status"][0], qss_delta_s=0.0)
        best = dict(vp_overrides=dict(self.base_ov), nlp_lap_s=None, delta_s=None, row_id=0,
                    qss_lap_s=sample["qss_lap_s"], basis="baseline only")
        self.timings["total_s"] = time.perf_counter() - self.t_start
        self.files["summary"] = self.paths["summary"]
        summary = dict(fingerprint=self.fp, name=self.name, out_dir=self.out_dir, plan=self.plan,
                       fields=self.flags, dropped=self.dropped, warnings=self.wlist, hub=None,
                       shortlist=[[0, "baseline"]], metrics={}, noise_floor_s=None, best=best,
                       finish=None, failures=[], timings=self.timings, throughput={},
                       samples=[sample], baseline_only=True, n_tasks_run=0, files=self.files)
        S.write_json_atomic(self.paths["summary"], summary)
        self.say(f"[sweep] baseline only: QSS lap {sample['qss_lap_s']:.4f} s -> {self.paths['summary']}")
        return SweepResult(name=self.name, out_dir=self.out_dir, fingerprint=self.fp, fields=self.flags,
                           dropped=self.dropped, samples=[sample], shortlist=[(0, "baseline")],
                           best=best, warnings=list(self.wlist), timings=self.timings,
                           files=dict(self.files), n_tasks_run=0)


def setup_sweep(param_specs, n_samples=256, circuit="Sturn", *, name=None, vi=60.0, ni=np.nan,
                AeroConfig="Static", ATD="On", Electric_4Motors="Off", TyreModel="CombinedSlip",
                sampler="sobol", seed=0, top_k=8, n_probes=4, base=None, confirm=True,
                roundtrip="all", max_warm_iter=250, bridge_steps=(2, 4), tol_branch_s=1e-3,
                resolve_s=1e-4, finish=False, workers="auto", screen_workers="auto", pin_hsl=True,
                load_model="vehModel", ds_fine=1.0, results_root=os.path.join("Results", "sweeps"),
                plot=True, resume=True, verbose=True, **useropts_kwargs):
    """Screen ``n_samples`` setups with the QSS model, then confirm a shortlist with
    warm-started 23-state NLP solves (a star from one validated hub). Returns a
    SweepResult; res.best['vp_overrides'] is the NLP-best confirmed setup.

    param_specs     [(field, lower, upper), ...] (optimise_design's format): vehParams
                    primaries or Pacejka coefficients. Unknown fields and fields the NLP
                    ignores in this configuration are dropped with a warning; malformed
                    specs raise ValueError.
    n_samples       design size (a power of 2 for Sobol'); row 0, the baseline, is extra.
    base            None (a cold hub is solved first), a full 23-state result .mat with
                    data.nlp (a previous sweep's hub.mat, Results/<stem>.mat) or an MLTP
                    ctx / ctx.data. It must be this NLP (circuit, config, OPT_ds / OPT_d /
                    mesh, tyre_set), else SweepError before any solve; if it needs IPOPT
                    iterations at the base setup it is replaced by that re-solve (warned).
    top_k, n_probes the shortlist: the QSS top_k plus n_probes rows at QSS-rank
                    quantiles of the rest (keeps the rank metrics from range restriction).
    roundtrip       'all' (every confirmed row hops back to the base setup: residual,
                    bracket, branch flag), 'top' (the 3 best rows and suspicious ones: a
                    bridge path or > 50 forward iterations) or 'none'.
    max_warm_iter   IPOPT max_iter of a warm hop (~ one cold solve); a hop that needs more
                    goes to the bridge ladder ``bridge_steps`` (never a cold solve).
    tol_branch_s    round-trip residual within which a row is on the hub's branch.
    resolve_s       minimum noise floor: NLP laps closer than the floor are unresolved.
    finish          co-optimise the promotion-safe fields with optimise_design from the
                    NLP-best row, then re-confirm p* through the star (kind 'finish').
    workers         NLP processes: 'auto' (cores and free RAM) or an int; 0 runs the NLP
                    tasks in this process (debugging only, not bitwise comparable).
    screen_workers  'auto' or an int, the processes of MLTP_screen.screen_batch.
    pin_hsl         keep the Coin-HSL library loaded in every worker (faster hops).
    name, results_root  output folder results_root/name (default '<stem>_<fp[:8]>'); the
                    folder of the same sweep resumes (only missing rows run), a folder of
                    another sweep raises SweepError (never overwritten).
    confirm=False   QSS screen + shortlist only (no NLP, casadi not needed).
    **useropts_kwargs  forwarded unchanged to userOpts / screen_batch / MLTP /
                    optimise_design (tyre_set, OPT_ds, mesh, linear_solver, ipopt_overrides,
                    ...); vp_overrides is the BASE setup merged under every row."""
    sw = _Sweep(dict(n_samples=n_samples, circuit=circuit, name=name, vi=vi, ni=ni,
                     AeroConfig=AeroConfig, ATD=ATD, Electric_4Motors=Electric_4Motors,
                     TyreModel=TyreModel, sampler=sampler, seed=seed, top_k=top_k, n_probes=n_probes,
                     base=base, confirm=confirm, roundtrip=roundtrip, max_warm_iter=max_warm_iter,
                     bridge_steps=bridge_steps, tol_branch_s=tol_branch_s, resolve_s=resolve_s,
                     finish=finish, workers=workers, screen_workers=screen_workers, pin_hsl=pin_hsl,
                     load_model=load_model, ds_fine=ds_fine, results_root=results_root, plot=plot,
                     resume=resume, verbose=verbose))
    sw.prepare(param_specs, useropts_kwargs)
    if not sw.fields:
        return sw.baseline_only()
    sw.design()
    with S.blas_single_thread():          # inherited by every worker spawned in here
        try:
            sw.start_hub()
            sw.screen()
            if sw.confirm:
                sw.finish_hub()
                sw.run_confirms()
                if sw.finish:
                    sw.run_finish()
            if sw.runner is not None:
                sw.runner.close()
        except BaseException:             # KeyboardInterrupt included: finished rows stay on disk
            if sw.runner is not None:
                sw.runner.close(cancel=True)
            raise
    return sw.finalise()


if __name__ == "__main__":
    setup_sweep([("alpha_RW", 4, 16), ("hcg", 0.45, 0.55), ("mb", 1730, 1910),
                 ("Tbrake_max", 3000, 5000)],
                n_samples=64, circuit="Sturn", top_k=4, n_probes=4, seed=0)
