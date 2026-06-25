# MPI Sweep Driver — Design

**Date:** 2026-06-25
**Branch:** `coin-hsl`
**Status:** Approved (design); pending implementation plan
**Author:** Prashant Kumar (with Claude Code)

## Problem

`MLTP.py` solves a single minimum-lap-time NLP via CasADi → IPOPT. On an HPC
cluster the user wants to exploit MPI, but IPOPT's algorithm is **not
MPI-aware**: only the linear-solver factorisation could be distributed, and the
CasADi wheel ships a *sequential* MUMPS plus shared-memory-only HSL solvers.
MPI-parallelising a *single* solve is therefore not viable without rebuilding
IPOPT + MUMPS + MPI from source — far more than "minimum changes," and of
dubious benefit at this problem size.

The genuinely parallel workload here is a **sweep / DOE**: many *independent*
`MLTP()` solves over circuits, initial speeds (`vi`), and config switches
(`ATD` / `Electric_4Motors` / `AeroConfig` / `TyreModel`). These share no state
and are embarrassingly parallel — one solve per MPI rank, perfect cluster
utilisation, and the numerical core (`MLTP.py`, `vehModel.py`,
`transcription.py`) does not change at all.

## Goals

- Run a **sweep of independent `MLTP()` solves** across MPI ranks on the HPC node.
- **Master/worker (dynamic) dispatch** so uneven solve times (some cases hit
  `max_iter=6000` and run long) don't leave ranks idle.
- **External CSV case file** so the DOE can be edited and version-controlled
  without touching code.
- **One core per rank (pure MPI):** highest case throughput; threading pinned to
  1 to avoid oversubscription.
- **Zero changes** to `MLTP.py`, `vehModel.py`, `transcription.py`.
- **Robust on HPC:** failure isolation per case, incremental manifest so a
  preempted/wall-time-killed job keeps completed results, and resume on
  re-submission.
- **Testable** without MPI or CasADi installed.

## Non-Goals

- MPI-parallelising a single IPOPT solve (not viable; see Problem).
- Hybrid OpenMP threading per rank (rejected in favour of 1-core-per-rank
  throughput; can be revisited later without touching this design's interfaces).
- Sharing the 7-state warm start across cases (each case stays self-contained;
  `MLTP()` runs its own `MLTP_initial` internally). YAGNI.
- Adding `mpi4py` to `requirements.txt` (would break local non-MPI installs).

## Architecture

Two new files plus one CSV; nothing in the existing model code changes.

```
run_sweep.py        # thin driver: MPI manager/worker loop, sets thread env, calls MLTP() per case
functions/sweep.py  # pure logic (no mpi4py, no casadi): parse + dispatch helpers
cases.csv           # the DOE — user-edited
```

The split keeps all decision logic in a casadi-free, MPI-free module so it can be
unit-tested with the repo's existing plain-script test style. `run_sweep.py` is a
thin shell: MPI plumbing + the per-case `MLTP()` call.

### `functions/sweep.py` (pure, no MPI/CasADi)

Well-bounded, independently testable functions:

- `read_cases(path) -> list[dict]` — parse `cases.csv` into a list of case
  dicts. Adds a `case_id` (from the column if present, else the row index).
  Blank cells are dropped so the corresponding `MLTP()` default applies.
- `case_kwargs(case) -> dict` — map a case dict to `MLTP()` kwargs. Coerces
  types (`vi`/`ni` → float, `case_id` excluded) and forces `plot=False`. Column
  names are validated against an **accepted-column allow-list** = the explicit
  `MLTP()` params the user may set (`circuit, vi, ni, warm_start, AeroConfig,
  ATD, Electric_4Motors, TyreModel`) **plus known `userOpts` passthroughs**
  (notably `linear_solver`, which reaches `MLTP` via `**useropts_kwargs`). A
  column outside this set raises a clear error at parse time. Driver-controlled
  kwargs (`save`, `plot`, `results_dir`) are not user-settable columns.
- `output_dir_for(sweep_name, case_id) -> str` — `Results/<sweep_name>/case_<id>/`.
  Guarantees uniqueness across cases that differ only by `vi` (which MLTP's own
  `<circuit>_<config>.mat` filename omits). `sweep_name` defaults to the
  `cases.csv` filename stem (e.g. `cases` → `Results/cases/...`) and is
  overridable with a `--name` CLI argument.
- `manifest_fieldnames(cases) -> list[str]` — column order for the manifest.
- `manifest_row(case, result) -> dict` — flatten inputs + outcome
  (`status`, `return_status`, `lap_time`, `init_s`, `solve_s`, `wall_s`,
  `out_path`) into one CSV row.
- `completed_ids(manifest_path) -> set` — case_ids whose manifest row has
  `status=ok`, for resume. Only successful cases are skipped; `error` rows are
  retried on re-submission (so transient HPC failures get another chance).

### `run_sweep.py` (thin MPI shell)

1. **Startup (all ranks):** set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
   `MKL_NUM_THREADS` = `"1"` *before* importing casadi/MLTP, so each rank's solve
   is single-threaded and N ranks use N cores without oversubscription.
2. **Manager (rank 0):** build the case queue from `read_cases()`, minus
   `completed_ids()` unless `--no-resume`. Manager-worker protocol: each worker
   sends a "ready"/result message; manager replies with the next `case_id` or a
   stop sentinel when the queue is empty. On each returned result, append a row
   to `manifest.csv` and flush immediately.
3. **Worker (rank ≥ 1):** loop — request work; on a `case_id`, build kwargs via
   `case_kwargs()`, call `MLTP(**kwargs, results_dir=output_dir_for(...))`
   inside try/except, build a result summary, send it back; on the stop
   sentinel, exit.
4. **`size == 1` fallback:** no workers available (login node / debug) → run all
   cases serially in-process using the same per-case execution path, writing the
   same manifest. The driver always runs, with or without `mpirun`.

### Per-case execution & result summary

`MLTP()` returns `ctx`; the summary captures:

- `status`: `ok` if the call returned; `error` if it raised (with the message).
- `return_status`: IPOPT status from the solve stats on `ctx` (records a
  non-converged `Maximum_Iterations_Exceeded` without treating it as a crash).
- `lap_time`: `ctx.data.lap_time`.
- `init_s`, `solve_s`: from `ctx.elapsed`.
- `wall_s`: measured around the call.
- `out_path`: the per-case results directory / file.

A raised exception is isolated to that case: it's recorded as `status=error` and
the worker continues with the next case.

## Data Flow

```
cases.csv ─read_cases→ [case dicts] ──(manager queue, minus completed)──┐
                                                                        │ MPI dispatch
                              ┌──────────── worker rank ────────────────┘
                              │ case_kwargs → MLTP(**kwargs, results_dir=…, plot=False)
                              │ → ctx → result summary ──MPI──▶ manager
                              ▼
        Results/<sweep_name>/case_<id>/<circuit>_<config>.mat   (per-case output)
        Results/<sweep_name>/manifest.csv                       (appended per result)
```

## Error Handling

- **Solve crash:** try/except around `MLTP()`; `status=error` + message in the
  manifest; worker proceeds. One bad case never kills the sweep.
- **Non-convergence:** not an exception — captured via `return_status`.
- **Bad CSV row / unknown kwarg:** `case_kwargs()` validates against the
  `MLTP()` parameter allow-list and raises a clear error at parse time (fail
  fast, before any MPI work starts).
- **Job preemption / wall-time kill:** manifest is appended-and-flushed per
  result, so completed cases survive. Re-submitting resumes via
  `completed_ids()`.
- **Launched on 1 rank:** serial fallback, no MPI calls that would deadlock.

## HPC Deployment Notes (documented, not pinned)

- **Install mpi4py against the cluster MPI**, not a generic wheel:
  `module load <openmpi|mpich>` then `pip install --no-binary mpi4py mpi4py`.
- **Launch:** `srun python run_sweep.py cases.csv` (Slurm) or
  `mpirun -np <N> python run_sweep.py cases.csv`.
- **Coin-HSL:** set `COINHSL_DIR` (or rely on the seeded default) so `ma57`
  loads on the node; otherwise the existing fallback uses MUMPS.
- `mpi4py` stays out of `requirements.txt` (HPC-only dependency).

## Testing

`test_sweep.py` — plain-script style matching the repo's other `test_*.py`
(assertions at module top level, no test runner), exercising `functions/sweep.py`
with a **stub solve function**, so it needs neither MPI nor CasADi:

- `read_cases()` round-trips a sample CSV; blanks dropped; `case_id` assigned.
- `case_kwargs()` coerces types, forces `plot=False`, rejects unknown kwargs.
- `output_dir_for()` yields distinct dirs for two cases differing only by `vi`.
- `manifest_row()` flattens inputs + outcome into the expected columns.
- `completed_ids()` parses an existing manifest and the manager's skip logic
  excludes those ids (tested via the pure queue-building helper, not MPI).

The MPI manager/worker loop itself is validated via the `size == 1` serial path
using the stub solve function (no real solver, no ranks spawned).

## Files

- **New:** `run_sweep.py`, `functions/sweep.py`, `cases.csv` (example), `test_sweep.py`.
- **Changed:** none in the model/solver core. (Optional: a short README/docs note
  on launching the sweep.)
