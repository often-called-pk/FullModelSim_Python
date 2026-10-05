# Coin-HSL Linear Solver Integration — Design

**Date:** 2026-06-23
**Branch:** `coin-hsl`
**Status:** Approved (design); pending implementation plan
**Author:** Prashant Kumar (with Claude Code)

## Problem

The full 23-state MLTP solve (`MLTP.py` → `vehModel` → IPOPT) is too slow to
converge. IPOPT currently uses **MUMPS**, the linear solver bundled inside the
CasADi wheel. For a problem of this size the symmetric-indefinite KKT
factorisation dominates wall time, and HSL solvers (MA57/MA97) are typically
several times faster than MUMPS on it.

The codebase actively *prevents* HSL use today: `_make_solver()` in
`functions/transcription.py` forces `linear_solver="mumps"` and strips any
`hsllib`/`ma*` request, as a defensive guard against a missing HSL DLL crashing
a solve. The goal is to **enable Coin-HSL as a configurable, faster linear
solver while preserving that "never crash on a missing DLL" guarantee.**

## Goals

- Make IPOPT's `linear_solver` **configurable** (default `ma57`), reachable from
  `userOpts`, applied through the single existing chokepoint.
- Use the user's existing **prebuilt** Coin-HSL DLL with no build/compile step.
- **Graceful fallback** to MUMPS when HSL is unavailable or incompatible, so a
  solve never hard-fails on DLL issues.
- **Portable** configuration (no machine-specific path committed to the repo).
- **Profile** the 23-state solve to confirm the linear solver is the real
  bottleneck and quantify the speed-up (mumps vs ma57 vs ma97).

## Non-goals

- Rebuilding or recompiling IPOPT, CasADi, or HSL (not needed — see Feasibility).
- Changing the NLP formulation, collocation scheme, or vehicle models.
- Acquiring/building HSL (user already has a license + prebuilt DLL).
- Adding `ma86` to defaults (segfaulted in a no-DLL loader probe; see Risks).

## Feasibility — proven end-to-end before design

All of the following were verified empirically against this exact environment
(CasADi **3.7.2**, Windows 11, in-repo `venv`):

1. **CasADi's bundled IPOPT already has the HSL runtime loader.** Requesting
   `linear_solver="ma57"` with `hsllib="libhsl.dll"` raised IPOPT's own
   `IpLibraryLoader.cpp` error (`DYNAMIC_LIBRARY_FAILURE ... Error 126 while
   loading DLL libhsl.dll`), i.e. IPOPT tried to `LoadLibrary` an external HSL
   DLL and only failed because none was on the path. **No IPOPT rebuild needed.**

2. **The user's DLL is ABI-matched.** It is the Julia `CoinHSL_jll` build
   `CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5`, located at
   `D:\Software Installs\CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin\`.
   Same MinGW / `libgfortran5` ABI as CasADi's bundled IPOPT runtime
   (`libgfortran-5.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`).

3. **The bin folder is self-contained.** It ships its own
   `libcoinhsl.dll`, `libhsl.dll` (IPOPT-compatible shim), `libopenblas.dll`,
   `libmetis.dll`, `libgfortran-5.dll`, `libquadmath-0.dll`, `libgomp-1.dll`,
   `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`, `libstdc++-6.dll`,
   `libatomic-1.dll`, `libssp-0.dll`.

4. **The exact loading recipe works.** With
   `os.add_dll_directory(bin)` + prepend `bin` to `PATH` + `hsllib` = full path
   to `libhsl.dll`, a trivial NLP solved successfully on **ma57, ma27, ma97**
   (and mumps as control), all `Solve_Succeeded`. This recipe is what the
   implementation will encode.

## Decisions

- **(A1) Env-var-first config resolver.** HSL location resolves in order:
  `COINHSL_DIR` environment variable → explicit `hsl_dir` kwarg/default in
  `userOpts` → known seeded default (the user's current path) → `None`.
  Portable across machines/clones; nothing machine-specific is *required* in
  committed code.
- **(B1) One-time pre-flight probe with MUMPS fallback.** Because IPOPT loads
  HSL lazily at *solve* time (not at `nlpsol` construction), a missing/broken
  DLL cannot be caught by a try/except around construction. Instead, before the
  real solve, run a trivial 1-variable NLP with the requested HSL solver +
  `hsllib`. On `DYNAMIC_LIBRARY_FAILURE`/`Invalid_Option`/exception, log a clear
  warning and transparently downgrade `opts` to `mumps`. Result cached per
  `(solver, hsllib)` so it costs ~1 ms once per process.

## Architecture

The whole solve path funnels through one chokepoint, so the change is localised:

```
userOpts(ctx, linear_solver=, hsl_dir=)        # builds ctx.opts["ipopt"]
   -> MLTP / MLTP_initial / MLTP_paramOptim / MLTP_TyreOptim
   -> build_and_solve_nlp(..., ctx.opts)        # functions/transcription.py
   -> _make_solver(ca, nlp, opts)               # THE chokepoint
   -> functions/hsl.apply_linear_solver(...)    # NEW: register dir + probe + fallback
   -> ca.nlpsol("solver", "ipopt", nlp, opts)
```

### Component 1 — `functions/hsl.py` (new, single responsibility)

Isolated, independently testable helper. Public surface:

- `resolve_hsl_dir(explicit=None) -> str | None`
  Resolution order: `COINHSL_DIR` env → `explicit` → seeded default → `None`.
  Returns the directory that should contain `libhsl.dll`/`libcoinhsl.dll` (the
  `bin/` folder). **casadi-free, pure → unit-testable.**

- `hsllib_path(hsl_dir) -> str | None`
  Joins `hsl_dir` + `libhsl.dll` (preferred IPOPT shim), falling back to
  `libcoinhsl.dll`; returns `None` if neither exists. **casadi-free.**

- `register_hsl_dll_dir(hsl_dir) -> None`
  Idempotent. On Windows: `os.add_dll_directory(hsl_dir)` **and** prepend
  `hsl_dir` to `os.environ["PATH"]`. No-op if dir missing or non-Windows.
  Tracks already-registered dirs to stay idempotent across repeated solves.

- `probe_linear_solver(solver_name, hsllib) -> bool`
  Builds + solves a trivial NLP with the given solver/`hsllib`. Returns whether
  it reached an HSL-backed `Solve_Succeeded`/normal status (vs. library-load
  failure). Cached per `(solver_name, hsllib)`. **casadi-dependent.**

- `apply_linear_solver(opts, *, linear_solver, hsl_dir) -> dict`
  Orchestrator. If `linear_solver` is MUMPS (or non-`ma*`), return opts
  unchanged. If `ma*`: resolve dir → register → set `hsllib` → probe; on success
  return opts with HSL wired; on failure return opts rewritten to `mumps`
  (strip `hsllib`) with a `warnings.warn`/log line. The opts-rewriting and
  resolution logic (everything except the probe) is **casadi-free**.

### Component 2 — `userOpts.py`

- Add kwargs: `linear_solver="ma57"`, `hsl_dir=None`.
- In the IPOPT options dict, `"linear_solver"` becomes the kwarg value
  (replacing the hardcoded `"mumps"`).
- `userOpts` carries only the user's *intent*, not resolved paths: it sets
  `ctx.opts["ipopt"]["linear_solver"] = linear_solver` and stashes the raw
  `hsl_dir` kwarg under a private key `ctx.opts["_hsl_dir"] = hsl_dir` (a
  CasADi-invisible hint, **not** an IPOPT option). All path resolution and
  `hsllib` wiring is **delegated to `_make_solver`/`functions/hsl.py`** — no
  path logic is duplicated in `userOpts`.
- Replace the stale "MUMPS only / HSL not used" comment block.

### Component 3 — `functions/transcription.py::_make_solver`

Replace the strip-to-MUMPS body with:

```text
opts = deepcopy(opts)
hsl_dir = resolve_hsl_dir(explicit=opts.pop("_hsl_dir", None))  # consume the hint
ip = opts.setdefault("ipopt", {})
if ip.get("linear_solver", "mumps").startswith("ma"):
    opts = apply_linear_solver(opts, linear_solver=ip["linear_solver"],
                               hsl_dir=hsl_dir)   # registers dir + probes + may downgrade to mumps
return ca.nlpsol("solver", "ipopt", nlp, opts)
```

The private `_hsl_dir` hint is always popped (so it never reaches CasADi),
regardless of which solver is selected.

Net behaviour: HSL when available and working; MUMPS otherwise — preserving the
original "never crash on a missing DLL" guarantee. Single chokepoint ⇒ all four
MLTP entry points benefit with no per-script edits.

### Component 4 — `bench_linear_solver.py` (new, repo root)

Profiling deliverable. Solves the real 23-state MLTP across the available
solvers (`mumps`, `ma57`, `ma97`) on a chosen circuit (default a small synthetic
one, e.g. `Sturn`, for fast turnaround) and tabulates per solver:

- return status, **iteration count**, **total wall time**,
- **linear-solver time vs. function-eval time split** — from
  `solver.stats()` `t_wall_*` callback timings plus IPOPT's
  `print_timing_statistics` (already `"yes"` in `userOpts`).

Output: a console table (and optionally a small `.mat`/CSV under `Results/`).
This both confirms the linear solver is the bottleneck and quantifies the win.

### Component 5 — Tests & docs

- `test_hsl.py` (plain-script style, matching the other `test_*.py`): casadi-free
  assertions on `resolve_hsl_dir` precedence (env > explicit > default > None),
  `hsllib_path`, and `apply_linear_solver` opts-rewriting/fallback (simulate
  "no dir" → opts downgraded to mumps). A probe-based check that auto-skips when
  no DLL is present.
- Docs: update the comment block in `requirements.txt`, the comment block in
  `userOpts.py`, and the `_make_solver` "forces MUMPS / strips HSL" gotcha note
  in `CLAUDE.md` (behaviour changes). Add a short setup note documenting the
  `COINHSL_DIR` env var and the default path.

## Data flow

`linear_solver`/`hsl_dir` (user intent, via `userOpts` kwargs) → `ctx.opts` →
`build_and_solve_nlp` → `_make_solver` → `apply_linear_solver` (resolve →
register DLL dir → probe → keep-HSL or fallback-to-MUMPS) → `nlpsol` →
solve. Profiling reads back through `solver.stats()`.

## Error handling

- **Missing/empty `hsl_dir`** → `resolve_hsl_dir` returns `None` →
  `apply_linear_solver` falls back to MUMPS with a warning. No crash.
- **DLL present but broken/incompatible** → probe fails → fallback to MUMPS with
  a warning. No mid-solve crash.
- **Non-Windows / unusual platform** → `register_hsl_dll_dir` no-ops the
  Windows-specific bits; probe still gates correctness.
- **User explicitly requests `mumps`** → untouched, original path.

## Testing strategy

1. `test_hsl.py` casadi-free unit assertions (resolution + fallback rewriting).
2. Smoke: trivial NLP via `apply_linear_solver` with the real DLL → ma57 solves
   (already verified manually; encode as a skip-if-absent test).
3. Integration: one real 23-state `MLTP(circuit="Sturn")` solve with `ma57`
   reaches `Solve_Succeeded`.
4. `bench_linear_solver.py` produces the mumps-vs-ma57(-vs-ma97) comparison.

## Risks & mitigations

| Risk | Status / mitigation |
|------|--------------------|
| IPOPT can't load external HSL | **Proven false** — loader present, DLL loads, solves. |
| ABI mismatch (Fortran/BLAS runtime) | **Proven false** — MinGW/libgfortran5 match; self-contained bin. |
| Machine-specific path committed | A1 env-var resolver; default is a *fallback*, not required. |
| Mid-solve crash on bad DLL | B1 pre-flight probe + MUMPS fallback. |
| `ma86` instability | Segfaulted in a no-DLL probe → excluded from defaults; ma57/ma97 validated. |
| Repeated solves re-register dir / re-probe | Idempotent registration + cached probe. |

## Acceptance criteria

- `MLTP(circuit=..., linear_solver="ma57")` runs the 23-state solve on HSL and
  reaches `Solve_Succeeded`.
- With no DLL / `COINHSL_DIR` unset and no default present, the same call falls
  back to MUMPS with a warning and still solves (no crash).
- `bench_linear_solver.py` reports iteration count, total wall time, and the
  linear-solver vs function-eval split for each available solver, demonstrating
  the ma57 speed-up over mumps on the 23-state problem.
- `linear_solver` is selectable (`ma57`/`ma97`/`mumps`) from `userOpts`.
- `test_hsl.py` passes; existing `test_*.py` unaffected.
