# MPI Sweep Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run many independent `MLTP()` solves (a sweep/DOE) across MPI ranks on an HPC node, with master/worker dispatch, a CSV case file, and a results manifest — without touching the model/solver core.

**Architecture:** A thin `run_sweep.py` MPI shell (manager on rank 0, workers on ranks 1+) drives one `MLTP()` solve per case. All decision/format logic lives in a pure, MPI-free, casadi-free module `functions/sweep.py` so it is unit-testable without either dependency. One core per rank; threads pinned to 1.

**Tech Stack:** Python 3, `mpi4py` (HPC-only, not in `requirements.txt`), the existing CasADi/IPOPT `MLTP()` entry point, `csv` from the stdlib. Tests are plain scripts (repo convention — no pytest).

## Global Constraints

- **Zero changes** to `MLTP.py`, `vehModel.py`, `functions/transcription.py`, or any model/solver code. The driver only *calls* `MLTP()`.
- **Tests are plain scripts** (CLAUDE.md): assertions at module top level, run via `python test_sweep.py`; the first failing assert aborts the file. No pytest/unittest. Match the existing `ok(name, cond)` helper style.
- **Tests must not import `casadi` or `mpi4py`** — inject a stub solve function and a fake clock.
- **`mpi4py` stays out of `requirements.txt`** (would break local non-MPI installs). It is an HPC-only, documented dependency.
- **Per-case output dir** is `Results/<sweep_name>/case_<case_id>/` — required because `MLTP()`'s own `<circuit>_<config>.mat` filename omits `vi`, so two cases differing only by `vi` would otherwise collide.
- **Threads pinned to 1**: set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` (and friends) to `"1"` via `setdefault` *before* importing casadi/MLTP.
- **Result-extraction field names** (verbatim from the codebase): `ctx.solve_stats` is a dict with `"return_status"` and `"iter_count"`; `ctx.elapsed` is a dict with `"init"` and `"solve"`; `ctx.data.lap_time` is a float.
- **Accepted CSV columns** (allow-list): `circuit, vi, ni, warm_start, AeroConfig, ATD, Electric_4Motors, TyreModel, linear_solver` (plus an optional `case_id`). `linear_solver` reaches `MLTP` via `**useropts_kwargs`. Driver-controlled kwargs (`save`, `plot`, `results_dir`) are NOT user columns.
- **Resume**: skip only `status=ok` rows; `error` rows are retried on re-submission.

## File Structure

- **Create** `functions/sweep.py` — pure helpers (no mpi4py, no casadi): CSV parse, kwarg mapping, output paths, manifest formatting, resume filtering, per-case runner.
- **Create** `run_sweep.py` — thin MPI shell: arg parsing, thread pinning, manager/worker loop, serial fallback, lazy `MLTP` import.
- **Create** `test_sweep.py` — plain-script tests over `functions/sweep.py` + `run_sweep.py`'s serial path, with a stub solve fn.
- **Create** `cases.csv` — a small worked example DOE.
- **Create** `docs/mpi_sweep.md` — how to install mpi4py against the cluster MPI, launch, resume.
- **Modify** none in the model/solver core.

---

### Task 1: Pure CSV parsing + kwarg mapping (`functions/sweep.py`)

**Files:**
- Create: `functions/sweep.py`
- Test: `test_sweep.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `ACCEPTED_COLUMNS: tuple[str, ...]`
  - `read_cases(path: str) -> list[dict]` — each dict has string values plus a string `"case_id"`.
  - `case_kwargs(case: dict) -> dict` — MLTP kwargs; `vi`/`ni` floats; `plot=False`; raises `ValueError` on unknown column.

- [ ] **Step 1: Write the failing test** — create `test_sweep.py`:

```python
"""Tests for the MPI sweep driver's pure logic (functions/sweep.py) and
run_sweep.py's serial path. No casadi, no mpi4py — a stub solve fn is injected.
Plain-script style (CLAUDE.md): run with `python test_sweep.py`."""
import os, sys, csv, tempfile
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import sweep

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

def _write_csv(rows, header):
    fd, path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header); w.writeheader()
        for r in rows:
            w.writerow(r)
    return path

print("read_cases + case_kwargs")
hdr = ["case_id", "circuit", "vi", "ATD", "Electric_4Motors", "linear_solver"]
path = _write_csv([
    {"case_id": "0", "circuit": "BCN", "vi": "40", "ATD": "On",
     "Electric_4Motors": "Off", "linear_solver": "ma57"},
    {"case_id": "1", "circuit": "Spa", "vi": "60", "ATD": "",
     "Electric_4Motors": "On", "linear_solver": ""},   # blanks -> dropped
], hdr)
cases = sweep.read_cases(path)
ok("two cases parsed", len(cases) == 2)
ok("case_id preserved", cases[0]["case_id"] == "0")
ok("blank ATD dropped", "ATD" not in cases[1])
ok("blank linear_solver dropped", "linear_solver" not in cases[1])

kw = sweep.case_kwargs(cases[0])
ok("vi coerced to float", isinstance(kw["vi"], float) and kw["vi"] == 40.0)
ok("plot forced False", kw["plot"] is False)
ok("case_id not a kwarg", "case_id" not in kw)
ok("circuit passed through", kw["circuit"] == "BCN")

# missing case_id column -> row index used
path2 = _write_csv([{"circuit": "BCN", "vi": "50"}], ["circuit", "vi"])
ok("auto case_id from row index", sweep.read_cases(path2)[0]["case_id"] == "0")

# unknown column rejected
raised = False
try:
    sweep.case_kwargs({"case_id": "9", "bogus": "x"})
except ValueError:
    raised = True
ok("unknown column raises ValueError", raised)
os.remove(path); os.remove(path2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_sweep.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'functions.sweep'` (file not created yet).

- [ ] **Step 3: Write minimal implementation** — create `functions/sweep.py`:

```python
"""Pure (MPI-free, casadi-free) helpers for the MLTP sweep driver.

run_sweep.py is a thin MPI shell over these functions; everything that needs a
decision or formatting lives here so it can be unit-tested without MPI or
CasADi. The only injected dependency is the per-case solve function (the real
MLTP in production, a stub in tests).
"""
import csv
import os

# Columns a user may set in cases.csv: explicit MLTP() params they may vary,
# plus userOpts passthroughs (linear_solver reaches MLTP via **useropts_kwargs).
ACCEPTED_COLUMNS = (
    "circuit", "vi", "ni", "warm_start", "AeroConfig", "ATD",
    "Electric_4Motors", "TyreModel", "linear_solver",
)
_FLOAT_COLUMNS = ("vi", "ni")


def read_cases(path):
    """Parse cases.csv into a list of case dicts.

    Blank cells are dropped (so the MLTP/userOpts default applies). Each case
    gets a string 'case_id': the 'case_id' column if present, else the
    zero-based row index.
    """
    cases = []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            case = {k: v.strip() for k, v in row.items()
                    if k is not None and v is not None and v.strip() != ""}
            case["case_id"] = case.get("case_id", str(i))
            cases.append(case)
    return cases


def case_kwargs(case):
    """Map a case dict to MLTP() kwargs.

    Validates columns against ACCEPTED_COLUMNS, coerces float columns, drops
    case_id, and forces plot=False. Raises ValueError on an unknown column.
    """
    unknown = set(case) - set(ACCEPTED_COLUMNS) - {"case_id"}
    if unknown:
        raise ValueError(
            f"case {case.get('case_id', '?')}: unknown column(s) "
            f"{sorted(unknown)}; accepted: {sorted(ACCEPTED_COLUMNS)}")
    kwargs = {}
    for k, v in case.items():
        if k == "case_id":
            continue
        kwargs[k] = float(v) if k in _FLOAT_COLUMNS else v
    kwargs["plot"] = False
    return kwargs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_sweep.py`
Expected: PASS — all `read_cases + case_kwargs` lines print `[PASS]`.

- [ ] **Step 5: Commit**

```bash
git add functions/sweep.py test_sweep.py
git commit -m "feat(sweep): CSV case parsing + MLTP kwarg mapping"
```

---

### Task 2: Output paths + manifest formatting (`functions/sweep.py`)

**Files:**
- Modify: `functions/sweep.py`
- Test: `test_sweep.py`

**Interfaces:**
- Consumes: `ACCEPTED_COLUMNS` (Task 1).
- Produces:
  - `output_dir_for(sweep_name: str, case_id: str) -> str` → `Results/<sweep_name>/case_<case_id>`
  - `OUTCOME_FIELDS: tuple[str, ...]` = `("status","return_status","iter_count","lap_time","init_s","solve_s","wall_s","out_path")`
  - `manifest_fieldnames(cases: list[dict]) -> list[str]` — input cols (`case_id` first, then `ACCEPTED_COLUMNS` order, only those present) then `OUTCOME_FIELDS`.
  - `manifest_row(case: dict, result: dict) -> dict` — merged inputs+outcome.

- [ ] **Step 1: Write the failing test** — append to `test_sweep.py`:

```python
print("output_dir_for + manifest formatting")
ok("vi-only-differing cases get distinct dirs",
   sweep.output_dir_for("run1", "0") != sweep.output_dir_for("run1", "1"))
ok("output dir shape",
   sweep.output_dir_for("run1", "3").replace("\\", "/")
   == "Results/run1/case_3")

cs = [
    {"case_id": "0", "circuit": "BCN", "vi": "40"},
    {"case_id": "1", "circuit": "Spa", "vi": "60", "ATD": "On"},
]
fields = sweep.manifest_fieldnames(cs)
ok("case_id is first field", fields[0] == "case_id")
ok("input cols before outcome", fields.index("circuit") < fields.index("status"))
ok("ATD column included once", fields.count("ATD") == 1)
ok("outcome fields present at end",
   fields[-len(sweep.OUTCOME_FIELDS):] == list(sweep.OUTCOME_FIELDS))

row = sweep.manifest_row(cs[0], {"status": "ok", "lap_time": 12.3})
ok("manifest_row merges inputs", row["circuit"] == "BCN")
ok("manifest_row merges outcome", row["status"] == "ok" and row["lap_time"] == 12.3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_sweep.py`
Expected: FAIL — `AttributeError: module 'functions.sweep' has no attribute 'output_dir_for'`.

- [ ] **Step 3: Write minimal implementation** — append to `functions/sweep.py`:

```python
# Manifest outcome columns, in order, appended after the input columns.
OUTCOME_FIELDS = ("status", "return_status", "iter_count",
                  "lap_time", "init_s", "solve_s", "wall_s", "out_path")


def output_dir_for(sweep_name, case_id):
    """Per-case results dir: Results/<sweep_name>/case_<case_id>/.

    Unique per case even when two cases differ only by vi (which MLTP's own
    <circuit>_<config>.mat filename omits).
    """
    return os.path.join("Results", sweep_name, f"case_{case_id}")


def input_fieldnames(cases):
    """Ordered input columns across all cases: case_id first, then
    ACCEPTED_COLUMNS in declared order, keeping only those that appear."""
    present = set()
    for c in cases:
        present.update(c)
    return ["case_id"] + [c for c in ACCEPTED_COLUMNS if c in present]


def manifest_fieldnames(cases):
    """Full manifest header: input columns then outcome columns."""
    return input_fieldnames(cases) + list(OUTCOME_FIELDS)


def manifest_row(case, result):
    """Flatten a case's inputs and its outcome into one manifest row dict."""
    row = dict(case)
    row.update(result)
    return row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_sweep.py`
Expected: PASS — `output_dir_for + manifest formatting` lines all `[PASS]`.

- [ ] **Step 5: Commit**

```bash
git add functions/sweep.py test_sweep.py
git commit -m "feat(sweep): per-case output dirs + manifest field formatting"
```

---

### Task 3: Resume filtering (`functions/sweep.py`)

**Files:**
- Modify: `functions/sweep.py`
- Test: `test_sweep.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `completed_ids(manifest_path: str) -> set[str]` — case_ids with `status=ok`; missing file → empty set.
  - `pending_cases(cases: list[dict], manifest_path: str, resume: bool = True) -> list[dict]` — drops completed when `resume`; else returns all.

- [ ] **Step 1: Write the failing test** — append to `test_sweep.py`:

```python
print("resume filtering")
fd, manifest = tempfile.mkstemp(suffix=".csv"); os.close(fd)
with open(manifest, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["case_id", "status"]); w.writeheader()
    w.writerow({"case_id": "0", "status": "ok"})
    w.writerow({"case_id": "1", "status": "error"})
ok("completed_ids returns only ok rows",
   sweep.completed_ids(manifest) == {"0"})
ok("completed_ids on missing file is empty",
   sweep.completed_ids(manifest + ".nope") == set())

allcases = [{"case_id": c} for c in ("0", "1", "2")]
pend = sweep.pending_cases(allcases, manifest, resume=True)
ok("resume drops ok, keeps error + unseen",
   {c["case_id"] for c in pend} == {"1", "2"})
ok("resume=False keeps all",
   len(sweep.pending_cases(allcases, manifest, resume=False)) == 3)
os.remove(manifest)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_sweep.py`
Expected: FAIL — `AttributeError: module 'functions.sweep' has no attribute 'completed_ids'`.

- [ ] **Step 3: Write minimal implementation** — append to `functions/sweep.py`:

```python
_DONE_STATUSES = ("ok",)


def completed_ids(manifest_path):
    """case_ids whose manifest row has status 'ok'. Missing file -> empty set.
    Only successful cases are skipped on resume; 'error' rows are retried."""
    if not os.path.isfile(manifest_path):
        return set()
    done = set()
    with open(manifest_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") in _DONE_STATUSES:
                done.add(row.get("case_id"))
    return done


def pending_cases(cases, manifest_path, resume=True):
    """Cases still to run: drop already-completed (status ok) when resume is
    True; otherwise return all cases unchanged."""
    if not resume:
        return list(cases)
    done = completed_ids(manifest_path)
    return [c for c in cases if c["case_id"] not in done]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_sweep.py`
Expected: PASS — `resume filtering` lines all `[PASS]`.

- [ ] **Step 5: Commit**

```bash
git add functions/sweep.py test_sweep.py
git commit -m "feat(sweep): resume filtering via manifest (skip completed)"
```

---

### Task 4: Per-case runner with failure isolation (`functions/sweep.py`)

**Files:**
- Modify: `functions/sweep.py`
- Test: `test_sweep.py`

**Interfaces:**
- Consumes: `case_kwargs`, `output_dir_for` (Tasks 1–2).
- Produces:
  - `run_case(case: dict, sweep_name: str, solve_fn, clock) -> dict` — outcome dict with all `OUTCOME_FIELDS`. `solve_fn(results_dir=..., **kwargs) -> ctx`. `clock()` returns seconds. Catches any exception → `status="error"`.

Result extraction reads (verbatim from the codebase): `ctx.solve_stats["return_status"]`, `ctx.solve_stats["iter_count"]`, `ctx.elapsed["init"]`, `ctx.elapsed["solve"]`, `ctx.data.lap_time`.

- [ ] **Step 1: Write the failing test** — append to `test_sweep.py`:

```python
print("run_case (stubbed solve + fake clock)")

class _FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self):
        self.t += 1.5      # each call advances 1.5s -> wall_s = 1.5
        return self.t

def _good_solve(results_dir, **kwargs):
    return SimpleNamespace(
        data=SimpleNamespace(lap_time=42.0),
        solve_stats={"return_status": "Solve_Succeeded", "iter_count": 37},
        elapsed={"init": 1.0, "solve": 9.0})

def _bad_solve(results_dir, **kwargs):
    raise RuntimeError("diverged")

good = sweep.run_case({"case_id": "0", "circuit": "BCN", "vi": "40"},
                      "run1", _good_solve, _FakeClock())
ok("status ok on success", good["status"] == "ok")
ok("return_status captured", good["return_status"] == "Solve_Succeeded")
ok("iter_count captured", good["iter_count"] == 37)
ok("lap_time captured", good["lap_time"] == 42.0)
ok("init_s captured", good["init_s"] == 1.0)
ok("solve_s captured", good["solve_s"] == 9.0)
ok("wall_s measured", good["wall_s"] == 1.5)
ok("out_path is the case dir",
   good["out_path"].replace("\\", "/") == "Results/run1/case_0")

bad = sweep.run_case({"case_id": "1", "circuit": "Spa"},
                     "run1", _bad_solve, _FakeClock())
ok("error isolated to the case", bad["status"] == "error")
ok("exception type recorded", bad["return_status"] == "RuntimeError")
ok("error row still has out_path",
   bad["out_path"].replace("\\", "/") == "Results/run1/case_1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_sweep.py`
Expected: FAIL — `AttributeError: module 'functions.sweep' has no attribute 'run_case'`.

- [ ] **Step 3: Write minimal implementation** — append to `functions/sweep.py`:

```python
def _result_from_ctx(ctx, out_path, wall_s):
    """Build a manifest outcome dict from a returned MLTP ctx."""
    st = getattr(ctx, "solve_stats", {}) or {}
    elapsed = getattr(ctx, "elapsed", {}) or {}
    return {
        "status": "ok",
        "return_status": st.get("return_status", "?"),
        "iter_count": st.get("iter_count", "?"),
        "lap_time": float(ctx.data.lap_time),
        "init_s": float(elapsed.get("init", float("nan"))),
        "solve_s": float(elapsed.get("solve", float("nan"))),
        "wall_s": round(wall_s, 3),
        "out_path": out_path,
    }


def run_case(case, sweep_name, solve_fn, clock):
    """Run one case through solve_fn; return a manifest outcome dict.

    solve_fn(results_dir=..., **case_kwargs(case)) -> ctx (MLTP-like). Wrapped
    in try/except so a crash is isolated to this case (status='error'); a
    non-converged solve returns normally and is recorded via return_status.
    `clock` is a no-arg callable returning seconds (time.perf_counter in prod,
    a fake in tests) so wall time is deterministically testable.
    """
    out_dir = output_dir_for(sweep_name, case["case_id"])
    t0 = clock()
    try:
        ctx = solve_fn(results_dir=out_dir, **case_kwargs(case))
        return _result_from_ctx(ctx, out_dir, clock() - t0)
    except Exception as exc:
        return {
            "status": "error",
            "return_status": type(exc).__name__,
            "iter_count": "",
            "lap_time": "",
            "init_s": "",
            "solve_s": "",
            "wall_s": round(clock() - t0, 3),
            "out_path": out_dir,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_sweep.py`
Expected: PASS — `run_case (stubbed solve + fake clock)` lines all `[PASS]`.

- [ ] **Step 5: Commit**

```bash
git add functions/sweep.py test_sweep.py
git commit -m "feat(sweep): per-case runner with failure isolation + ctx extraction"
```

---

### Task 5: MPI driver shell + serial-path test (`run_sweep.py`)

**Files:**
- Create: `run_sweep.py`
- Test: `test_sweep.py`

**Interfaces:**
- Consumes: `read_cases`, `pending_cases`, `manifest_fieldnames`, `manifest_row`, `run_case` (Tasks 1–4).
- Produces (module-level, for tests/ops):
  - `_parse_args(argv) -> Namespace` (`.cases`, `.name`, `.no_resume`)
  - `_sweep_name(args) -> str`
  - `_solve(**kwargs) -> ctx` (lazy `from MLTP import MLTP`)
  - `_open_manifest(path, fieldnames) -> (fh, writer)` (writes header iff new)
  - `_run_serial(cases, sweep_name, manifest_path) -> None`
  - `_run_manager(comm, cases, sweep_name, manifest_path)`, `_run_worker(comm, sweep_name)`, `main(argv=None)`

- [ ] **Step 1: Write the failing test** — append to `test_sweep.py`:

```python
print("run_sweep serial path (stubbed _solve)")
import run_sweep

# Stub out the real (casadi) solve. _run_serial -> run_case references the
# module global _solve, so rebinding it here takes effect.
_calls = {"n": 0}
def _stub_solve(results_dir, **kwargs):
    _calls["n"] += 1
    if kwargs.get("circuit") == "BOOM":
        raise RuntimeError("kaboom")
    return SimpleNamespace(
        data=SimpleNamespace(lap_time=10.0 + _calls["n"]),
        solve_stats={"return_status": "Solve_Succeeded", "iter_count": 5},
        elapsed={"init": 0.5, "solve": 2.0})
run_sweep._solve = _stub_solve

tmpdir = tempfile.mkdtemp()
manifest = os.path.join(tmpdir, "manifest.csv")
serial_cases = [
    {"case_id": "0", "circuit": "BCN", "vi": "40"},
    {"case_id": "1", "circuit": "BOOM", "vi": "60"},   # raises -> error row
]
run_sweep._run_serial(serial_cases, "unit", manifest)

with open(manifest, newline="") as fh:
    rows = list(csv.DictReader(fh))
ok("manifest has one row per case", len(rows) == 2)
ok("first case ok", rows[0]["status"] == "ok" and rows[0]["lap_time"] == "11.0")
ok("second case error-isolated", rows[1]["status"] == "error")
ok("manifest header has outcome cols",
   "return_status" in rows[0] and "wall_s" in rows[0])

# arg parsing + sweep name default
args = run_sweep._parse_args(["cases.csv"])
ok("default sweep name is csv stem", run_sweep._sweep_name(args) == "cases")
args2 = run_sweep._parse_args(["x/foo.csv", "--name", "doe1"])
ok("explicit --name wins", run_sweep._sweep_name(args2) == "doe1")
ok("--no-resume flag parsed", run_sweep._parse_args(["c.csv", "--no-resume"]).no_resume)

import shutil; shutil.rmtree(tmpdir)
print("ALL SWEEP TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_sweep.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'run_sweep'`.

- [ ] **Step 3: Write minimal implementation** — create `run_sweep.py`:

```python
#!/usr/bin/env python
"""MPI master/worker driver: run a sweep of independent MLTP() solves.

Launch one rank per core, e.g.:
    srun python run_sweep.py cases.csv
    mpirun -np 32 python run_sweep.py cases.csv

Rank 0 is the manager (hands cases to workers, writes the manifest); ranks 1+
are workers (each runs one MLTP solve at a time). With a single rank it falls
back to running every case serially in-process, so it works without mpirun too.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pin every numerical library to one thread BEFORE importing casadi/MLTP, so N
# ranks use N cores without oversubscription. setdefault lets the env override.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from functions.sweep import (read_cases, pending_cases, manifest_fieldnames,
                             manifest_row, run_case)

_TAG_WORK = 2     # manager -> worker: here is a case
_TAG_STOP = 3     # manager -> worker: no more work, exit


def _parse_args(argv):
    p = argparse.ArgumentParser(description="MPI sweep of MLTP solves")
    p.add_argument("cases", help="path to the cases CSV")
    p.add_argument("--name", default=None,
                   help="sweep name (default: cases-file stem)")
    p.add_argument("--no-resume", action="store_true",
                   help="re-run cases even if already completed")
    return p.parse_args(argv)


def _sweep_name(args):
    return args.name or os.path.splitext(os.path.basename(args.cases))[0]


def _manifest_path(sweep_name):
    return os.path.join("Results", sweep_name, "manifest.csv")


def _solve(**kwargs):
    """Real per-case solve. MLTP is imported lazily so only workers pay the
    casadi import, and the pure module/tests never need it."""
    from MLTP import MLTP
    return MLTP(**kwargs)


def _open_manifest(path, fieldnames):
    """Open the manifest for appending; write the header iff the file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.isfile(path)
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="",
                            extrasaction="ignore")
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def _run_serial(cases, sweep_name, manifest_path):
    """Run every case in this process (1-rank fallback / debugging)."""
    fh, writer = _open_manifest(manifest_path, manifest_fieldnames(cases))
    try:
        for case in cases:
            result = run_case(case, sweep_name, _solve, time.perf_counter)
            writer.writerow(manifest_row(case, result))
            fh.flush()
            print(f"[serial] case {case['case_id']}: {result['status']} "
                  f"{result.get('return_status', '')}")
    finally:
        fh.close()


def _run_manager(comm, cases, sweep_name, manifest_path):
    """Rank 0: dispatch cases to workers, write each returned result."""
    from mpi4py import MPI
    fh, writer = _open_manifest(manifest_path, manifest_fieldnames(cases))
    queue = list(cases)
    stopped = 0
    n_workers = comm.Get_size() - 1
    try:
        while stopped < n_workers:
            status = MPI.Status()
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                            status=status)
            src = status.Get_source()
            if msg is not None:                      # a completed (case, result)
                case, result = msg
                writer.writerow(manifest_row(case, result))
                fh.flush()
                print(f"[mgr] case {case['case_id']}: {result['status']} "
                      f"{result.get('return_status', '')}")
            if queue:
                comm.send(queue.pop(0), dest=src, tag=_TAG_WORK)
            else:
                comm.send(None, dest=src, tag=_TAG_STOP)
                stopped += 1
    finally:
        fh.close()


def _run_worker(comm, sweep_name):
    """Rank >=1: ask for work, solve, report, until told to stop."""
    from mpi4py import MPI
    comm.send(None, dest=0, tag=_TAG_WORK)           # initial "ready"
    while True:
        status = MPI.Status()
        case = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == _TAG_STOP:
            break
        result = run_case(case, sweep_name, _solve, time.perf_counter)
        comm.send((case, result), dest=0, tag=_TAG_WORK)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    sweep_name = _sweep_name(args)
    manifest_path = _manifest_path(sweep_name)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size, rank = comm.Get_size(), comm.Get_rank()
    except Exception:
        comm, size, rank = None, 1, 0

    if size == 1:
        cases = pending_cases(read_cases(args.cases), manifest_path,
                              resume=not args.no_resume)
        print(f"[serial] {len(cases)} case(s) to run (sweep '{sweep_name}')")
        _run_serial(cases, sweep_name, manifest_path)
    elif rank == 0:
        cases = pending_cases(read_cases(args.cases), manifest_path,
                              resume=not args.no_resume)
        print(f"[mgr] {len(cases)} case(s) across {size - 1} worker(s)")
        _run_manager(comm, cases, sweep_name, manifest_path)
    else:
        _run_worker(comm, sweep_name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_sweep.py`
Expected: PASS — ends with `ALL SWEEP TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add run_sweep.py test_sweep.py
git commit -m "feat(sweep): MPI master/worker driver with serial fallback"
```

---

### Task 6: Example `cases.csv` + HPC docs

**Files:**
- Create: `cases.csv`
- Create: `docs/mpi_sweep.md`

**Interfaces:**
- Consumes: `read_cases` (Task 1) for the verification step.
- Produces: a runnable example DOE and operator docs. No code.

- [ ] **Step 1: Create the example `cases.csv`**

```csv
case_id,circuit,vi,ATD,Electric_4Motors,AeroConfig,TyreModel,linear_solver
0,BCN,40,On,Off,Static,CombinedSlip,ma57
1,BCN,60,On,Off,Static,CombinedSlip,ma57
2,BCN,80,On,Off,Static,CombinedSlip,ma57
3,Spa,60,Off,On,Static,CombinedSlip,ma57
4,Jarama,60,On,Off,Static,CombinedSlip,ma57
```

- [ ] **Step 2: Create `docs/mpi_sweep.md`**

```markdown
# Running an MLTP sweep with MPI

`run_sweep.py` distributes many independent `MLTP()` solves across MPI ranks
(master/worker). IPOPT itself is not MPI-parallel, so this parallelises the
*sweep*, not a single solve. One core per rank; threads are pinned to 1.

## Define the sweep

Edit `cases.csv` — one row per solve. Columns map to `MLTP()` kwargs; blanks
fall back to defaults. Accepted columns: `case_id, circuit, vi, ni, warm_start,
AeroConfig, ATD, Electric_4Motors, TyreModel, linear_solver`. Keep
`linear_solver=ma57` (single-threaded, fast factorisation) for one-core ranks.

Do not change the column set partway through a sweep you intend to resume — the
manifest header is written once from the first run's columns.

## Install mpi4py against the cluster MPI

Do NOT rely on a generic wheel — build it against the node's MPI:

    module load openmpi        # or mpich, per your cluster
    pip install --no-binary mpi4py mpi4py

`mpi4py` is intentionally not in `requirements.txt` (HPC-only).

## Launch

    srun python run_sweep.py cases.csv          # Slurm
    mpirun -np 32 python run_sweep.py cases.csv  # generic MPI

Rank 0 coordinates; ranks 1+ solve. Run on N ranks to keep N-1 solves busy.
Single-rank (no mpirun) runs every case serially — handy for a login-node check.

Set `COINHSL_DIR` so `ma57` loads on the node (otherwise it falls back to MUMPS).

## Outputs

- `Results/<name>/case_<id>/<circuit>_<config>.mat` — per-case solution.
- `Results/<name>/manifest.csv` — one row per case: inputs + `status`,
  `return_status`, `iter_count`, `lap_time`, `init_s`, `solve_s`, `wall_s`,
  `out_path`. `<name>` is the CSV stem or `--name`.

## Resume

Re-submitting the same command skips cases already marked `status=ok` in the
manifest (so a wall-time-killed job continues). `error` rows are retried. Pass
`--no-resume` to force a full re-run.
```

- [ ] **Step 3: Verify the example parses (casadi-free)**

Run: `python -c "from functions.sweep import read_cases, case_kwargs; cs=read_cases('cases.csv'); print(len(cs), case_kwargs(cs[0]))"`
Expected: prints `5 {...'circuit': 'BCN'...'vi': 40.0...'plot': False}` (5 cases, first case's kwargs).

- [ ] **Step 4: Run the full test suite once more**

Run: `python test_sweep.py`
Expected: PASS — ends with `ALL SWEEP TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add cases.csv docs/mpi_sweep.md
git commit -m "docs(sweep): example cases.csv + HPC launch/resume guide"
```

---

## Self-Review

**1. Spec coverage:**
- Master/worker dynamic dispatch → Task 5 (`_run_manager`/`_run_worker`). ✔
- External CSV case file → Tasks 1, 6. ✔
- One core per rank, threads pinned → Task 5 (thread-env `setdefault`). ✔
- Zero changes to model/solver core → no task modifies them; `_solve` only calls `MLTP()`. ✔
- Per-case unique output dir (vi collision) → Task 2 `output_dir_for`, tested. ✔
- Incremental manifest + flush → Task 5 `_run_serial`/`_run_manager`. ✔
- Resume (skip ok, retry error) → Task 3 + Task 5 wiring. ✔
- Failure isolation → Task 4 `run_case` try/except, tested. ✔
- Non-convergence via return_status → Task 4 `_result_from_ctx`. ✔
- `size==1` serial fallback → Task 5 `main`. ✔
- mpi4py-on-HPC + launch notes, not in requirements → Task 6. ✔
- Tests without MPI/CasADi via stub → Tasks 1–5 all use stubs/temp files. ✔

**2. Placeholder scan:** No TBD/TODO; every code and test step shows full content. ✔

**3. Type consistency:** `run_case(case, sweep_name, solve_fn, clock)` signature identical in Task 4 definition, its test, and both Task 5 call sites. `solve_fn(results_dir=..., **kwargs)` matches `_solve(**kwargs)` and the stubs (which accept `results_dir`). `manifest_fieldnames`/`manifest_row`/`pending_cases`/`read_cases` names consistent across Tasks 1–5. Outcome dict keys equal `OUTCOME_FIELDS` in both `run_case` branches. ✔
```
