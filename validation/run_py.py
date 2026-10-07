"""Python side of the validation run matrix (docs/validation_matlab_vs_python.md sections 7, 9, 10): one MLTP solve in a
child process, timed from outside. The counterpart of ``matlab_batch.py solve``.

    python validation/run_py.py --track BCN --tier par|prod --rep 1 --out Results/validation [--opt-ds 20] [--max-iter N] [--timeout 2700]

tier ``par``  = the parity call of doc section 3 (py_export.PARITY: uniform mesh, OPT_ds 10, tol 1e-8, MUMPS, monotone
                mu, acceptable_tol 1e-6) inside py_export.matlab_seed() (MATLAB's 23-state start point);
tier ``prod`` = Python defaults, MLTP(track) with nothing set (OPT_ds 30, mesh auto, tol 1e-4, ma57, adaptive mu).
Both add the runner cap IPOPT max_wall_time 1800 s and run with save=False, plot=False (the results go to --out).
``--opt-ds`` is for the failure-policy retry (doc section 10, 20 m), ``--max-iter`` for the smoke test only
(0 = everything is built, nothing is solved).

The parent (standard library only) starts this script again with ``--child``, kills the whole process tree at
``--timeout`` (taskkill /T) and times the call from outside. Output, next to each other in --out:

    <track>_py_<tier>_r<rep>.mat    the field names of the MATLAB result (validation/matlab/vsolve.m) where they exist:
                                    w_opt lam_g lam_x f x_opt u_opt xc_opt t_opt lap s_knot k_knot N OPT_ds n_w n_g
                                    stats iterations init ipopt peak_ws_bytes casadi_version; Python only: timers
                                    peak_commit_bytes elapsed python_version. w is scaled (order MLTP.m:371), the
                                    others physical. init = the 7-state solve (stats iter_count return_status
                                    t_wall_total, wall_s, lap).
    <track>_py_<tier>_r<rep>.json   the scalars of the same + settings, linear solver actually used, return status,
                                    versions, then from the parent wall_external_s, rc, timed_out, git sha. A run that
                                    failed or was killed has ok false, an error text and no .mat (the child writes the
                                    JSON of an exception, the parent that of a kill).
    <track>_py_<tier>_r<rep>.log    stdout and stderr of the child (MLTP prints, IPOPT iterations).
Exit code 0 = result files written (IPOPT status is in the JSON), 1 = failed or killed.

timers (JSON, seconds): import_s, model_build_s (vehModel), nlp_build_s (transcription t_build), transcription_s
(everything of the 23-state call before IPOPT: model, Functions, guesses, NLP build), init7_wall_s / init7_iters /
init7_ipopt_s (the 7-state MLTP_initial call, its iterations and IPOPT wall), ipopt_wall_s / iters / s_per_iter /
fn_eval_share (23-state solver.stats(): t_wall_total, sum(t_wall_nlp_*) / t_wall_total), mltp_wall_s (the MLTP call),
child_wall_s (child start to just before the result files are written). Peak working set and peak commit come from
GetProcessMemoryInfo of the child itself. MLTP_* environment variables are removed from the child (defaults measured).
Skipped: polling the child's memory from the parent, add when a killed run's footprint is needed.
"""
import argparse
import contextlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WALL_CAP = 1800                                   # IPOPT max_wall_time of every solve: runner cap, doc section 10
STATS = re.compile(r"^(iter_count|return_status|success|t_wall_total|t_proc_total|t_wall_nlp_.*|n_call_nlp_.*)$")


def _clean(x):
    """JSON-safe copy: numpy to Python, Paths to str, non-finite floats to null."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "tolist"):                      # numpy arrays and scalars
        return _clean(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def _memory():
    """(peak working set, peak commit) in bytes of this process, (None, None) off Windows."""
    if os.name != "nt":
        return None, None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):                  # PROCESS_MEMORY_COUNTERS
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None, None
    return int(pmc.PeakWorkingSetSize), int(pmc.PeakPagefileUsage)


def child(a):
    """The solve itself (module docstring). Returns the exit code."""
    t_start = time.perf_counter()
    stem = f"{a.track}_py_{a.tier}_r{a.rep}"
    mat_file, json_file = Path(a.out).resolve() / f"{stem}.mat", Path(a.out).resolve() / f"{stem}.json"
    os.chdir(ROOT)                                # the production tier reads Circuits/ and Data/ relative to the root
    sys.path.insert(0, str(ROOT))
    try:
        t0 = time.perf_counter()
        import casadi
        import numpy as np
        import scipy
        import scipy.io as sio
        from MLTP import MLTP
        if a.tier == "par":
            from validation.py_export import PARITY, matlab_seed
        import_s = time.perf_counter() - t0

        if a.tier == "par":                       # doc section 3, parity call
            kw = dict(PARITY, OPT_ds=a.opt_ds or 10, mesh="uniform", warm_start=None)
            kw["ipopt_overrides"] = {**PARITY["ipopt_overrides"], "max_wall_time": WALL_CAP}
            seed = matlab_seed()
        else:                                     # Python defaults
            kw = dict(ipopt_overrides={"max_wall_time": WALL_CAP})
            if a.opt_ds:
                kw["OPT_ds"] = a.opt_ds
            seed = contextlib.nullcontext()
        if a.max_iter is not None:
            kw["max_iter"] = a.max_iter

        t1 = time.perf_counter()
        with seed:
            ctx = MLTP(a.track, save=False, plot=False, **kw)
        mltp_wall_s = time.perf_counter() - t1

        d, el, st, lad = ctx.data, ctx.elapsed, ctx.solve_stats, ctx.data.ladder
        nlp, k = d.nlp, d.OPT_d + 1
        stats = {key: v for key, v in st.items() if STATS.match(key)}
        trace = {key: np.asarray(v, dtype=float).ravel() for key, v in st.get("iterations", {}).items()}
        wall, iters = stats.get("t_wall_total", float("nan")), stats.get("iter_count", -1)
        fn_eval = sum(v for key, v in stats.items() if key.startswith("t_wall_nlp_"))
        timers = dict(
            import_s=import_s, model_build_s=el.get("model_build"), nlp_build_s=el.get("nlp_build"),
            transcription_s=el["solve"] - wall,
            init7_wall_s=lad["m7_wall_s"], init7_iters=el.get("init_iters"), init7_ipopt_s=el.get("init_ipopt"),
            ipopt_wall_s=wall, iters=iters, s_per_iter=wall / iters if iters > 0 else None,
            fn_eval_share=fn_eval / wall if wall > 0 else None, mltp_wall_s=mltp_wall_s)
        init = dict(stats=dict(iter_count=lad["m7_iters"], return_status=lad["m7_status"],
                               t_wall_total=el.get("init_ipopt", float("nan"))),
                    wall_s=lad["m7_wall_s"], lap=lad["m7_lap_s"])
        elapsed = {key: v for key, v in el.items() if isinstance(v, (str, int, float, bool))}
        scal = dict(N=d.N, OPT_ds=d.OPT_ds, n_w=nlp["structure"]["n_w"], n_g=nlp["structure"]["n_g"], f=el["f_opt"],
                    lap=d.lap_time)

        timers["child_wall_s"] = time.perf_counter() - t_start
        ws, commit = _memory()
        mat = dict(w_opt=nlp["w_opt"], lam_g=nlp["lam_g"], lam_x=nlp["lam_x"], x_opt=d.x_opt, u_opt=d.u_opt,
                   xc_opt=d.xc_opt, t_opt=d.t_opt, s_knot=d.s_full[::k], k_knot=d.k_full[::k], stats=stats,
                   iterations=trace, init=init, ipopt=ctx.opts["ipopt"], timers=timers, elapsed=elapsed,
                   peak_ws_bytes=ws, peak_commit_bytes=commit, casadi_version=casadi.__version__,
                   python_version=sys.version.split()[0], **scal)
        sio.savemat(mat_file, _clean_for_mat(mat), do_compression=True)
        rec = dict(ok=True, circuit=a.track, **scal, return_status=stats.get("return_status"), stats=stats, init=init,
                   linear_solver=nlp["linear_solver"], mesh=d.mesh, mesh_requested=d.mesh_requested,
                   warm_start=el["warm_start"], ladder=lad["name"], ipopt=ctx.opts["ipopt"], timers=timers,
                   settings=kw, peak_ws_bytes=ws, peak_commit_bytes=commit,
                   inf_pr=float(trace["inf_pr"][-1]) if trace.get("inf_pr", np.empty(0)).size else None,
                   inf_du=float(trace["inf_du"][-1]) if trace.get("inf_du", np.empty(0)).size else None,
                   casadi_version=casadi.__version__, numpy_version=np.__version__, scipy_version=scipy.__version__,
                   python_version=sys.version.split()[0])
        json_file.write_text(json.dumps(_clean(rec), indent=1), encoding="utf-8")
        print(f"run_py: wrote {mat_file} (lap {d.lap_time:.6f} s, {iters} iterations, {stats.get('return_status')})")
        return 0
    except BaseException:
        ws, commit = _memory()
        rec = dict(ok=False, error=traceback.format_exc(), peak_ws_bytes=ws, peak_commit_bytes=commit,
                   child_wall_s=time.perf_counter() - t_start)
        json_file.write_text(json.dumps(_clean(rec), indent=1), encoding="utf-8")
        traceback.print_exc()
        return 1


def _clean_for_mat(x):
    """savemat cannot store None: drop those keys (recursively)."""
    return {k: _clean_for_mat(v) if isinstance(v, dict) else v for k, v in x.items() if v is not None}


def _git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def parent(a):
    """Run the child under the timeout, add the external wall and the provenance to its JSON."""
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.track}_py_{a.tier}_r{a.rep}"
    mat_file, json_file, log_file = (out / f"{stem}{ext}" for ext in (".mat", ".json", ".log"))
    for f in (mat_file, json_file, log_file):
        f.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--track", a.track, "--tier", a.tier,
           "--rep", str(a.rep), "--out", str(out)]
    if a.opt_ds is not None:
        cmd += ["--opt-ds", repr(a.opt_ds)]
    if a.max_iter is not None:
        cmd += ["--max-iter", str(a.max_iter)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLTP_")}     # measure the defaults
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    started, t0 = datetime.now(timezone.utc), time.perf_counter()
    with open(log_file, "wb") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            p.wait()
            rc = -9
    wall = time.perf_counter() - t0
    ok = rc == 0 and mat_file.exists() and json_file.exists()
    rec = json.loads(json_file.read_text(encoding="utf-8")) if json_file.exists() else {}
    rec.update(circuit=a.track, code="py", tier=a.tier, rep=a.rep, ok=ok, started_utc=started.isoformat(timespec="seconds"),
               wall_external_s=wall, rc=rc, timed_out=rc == -9, git_sha=_git("rev-parse", "HEAD"),
               git_dirty=bool(_git("status", "--porcelain")),
               args=dict(opt_ds=a.opt_ds, max_iter=a.max_iter, timeout_s=a.timeout), runner_caps=dict(max_wall_time=WALL_CAP))
    tail = log_file.read_bytes().decode("utf-8", "replace")[-2500:]
    if not ok:
        rec.setdefault("error", f"killed at the {a.timeout} s timeout" if rc == -9 else f"child exited rc {rc} without result files")
        rec["log_tail"] = tail
        print(tail, file=sys.stderr)
    json_file.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    st, tm = rec.get("stats", {}), rec.get("timers", {})
    head = f"rc {rc}{' (killed at the timeout)' if rc == -9 else ''}, Python wall {wall:.0f} s"
    if ok:
        print(f"ok: {head} (import {tm['import_s']:.1f} s), lap {rec['lap']}, {st['iter_count']} iterations, "
              f"{st['return_status']}\n{json_file}")
    else:
        print(f"FAILED: {head}: {(rec['error'].strip() or 'no error text').splitlines()[-1]}\n{json_file}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--track", required=True, help="BCN, NBR, Sturn, ...")
    ap.add_argument("--tier", required=True, choices=("par", "prod"), help="par = parity call, prod = Python defaults")
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opt-ds", type=float, help="OPT_ds override (par default 10, prod default 30; the failure-policy retry uses 20)")
    ap.add_argument("--max-iter", type=int, help="smoke test only: IPOPT max_iter override (0 = build, no solve)")
    ap.add_argument("--timeout", type=int, default=2700, help="seconds, kills the child's whole process tree")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    return child(a) if a.child else parent(a)


if __name__ == "__main__":
    sys.exit(main())
