# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python port of a MATLAB **Minimum-Lap-Time-Problem (MLTP)** framework for a 4-motor
electric race car (`FullModel_4EM_Suspension_FullTyre`). It poses the racing line as an
optimal-control problem in the **space (track arc-length) domain**, transcribes it with
**direct Legendre collocation**, and solves the resulting NLP with **IPOPT** (shipped inside
the CasADi wheel; linear solver HSL **ma57** by default, **MUMPS** as the automatic fallback).
There is no server: the deliverables are top-level scripts you run, `.mat` result files and
Plotly HTML figures, plus a PySide6 desktop app (`app/`, `headless_solve.py`, `build/`) that
drives the same solve.

## Environment & commands

Everything assumes the **repo root as working directory**: all paths are relative
(`Circuits/`, `Data/`, `Results/`, `Plots/`). Use the in-repo venv:

```powershell
venv\Scripts\Activate.ps1        # PowerShell; or call venv\Scripts\python.exe directly
pip install -r requirements.txt  # casadi, numpy, scipy, plotly, kaleido (optional), PySide6, pyinstaller
```

**Running a solve.** The scripts have **no argparse/sys.argv**; their `__main__` block calls
the function with hardcoded args (`circuit='Sturn'`, `vi=60.0`). To change circuit/config,
either edit the `__main__` call or import and call with kwargs. Extra kwargs of `MLTP()` and
`MLTP_screen()` are forwarded to `userOpts` (`vp_overrides`, `tyre_set`, `OPT_ds`, `mesh`,
`screening`, `linear_solver`, `ipopt_overrides`, ...):

```powershell
python MLTP_initial.py                                   # 7-state warm-start solve
python MLTP.py                                           # full 23-state solve
python MLTP_screen.py                                    # QSS lap-time estimate (no NLP)
python -c "from MLTP import MLTP; MLTP(circuit='BCN', AeroConfig='Static', ATD='On', Electric_4Motors='Off', TyreModel='CombinedSlip')"
python -c "from MLTP_paramOptim import MLTP_paramOptim; MLTP_paramOptim(circuit='Sturn')"
```

**Using Coin-HSL (HSL linear solver).** The solve defaults to IPOPT's HSL `ma57`. Provide a
Coin-HSL library (the MinGW/`libgfortran5` `CoinHSL_jll` build matches the casadi wheel's ABI)
by setting `COINHSL_DIR` to its `bin/` folder, or rely on the seeded default in
`functions/hsl.py`. The directory is registered on the Windows DLL path and probed once; if HSL
can't load, `_make_solver` falls back to MUMPS with a warning (a solve never crashes on a
missing DLL). `ma57` runs with MC64 auto-scaling (`ma57_automatic_scaling='yes'`, set
automatically in `functions/hsl.py`) so it converges on the stiff 23-state problem, where
unscaled MA57 can stall. Choose the solver per call: `MLTP(circuit='BCN', linear_solver='ma97')`
or `linear_solver='mumps'`. `python bench_linear_solver.py` reports a per-iteration
linear-solver cost (the apples-to-apples metric: MA57 factorises ~4-5x faster per IPOPT
iteration than MUMPS; total wall-clock also depends on the iteration count each takes).

**CasADi options (CSE / JIT / symbol type).** Every symbolic `ca.Function` is built with
`fn_opts(ctx)` (`functions/casadi_opts.py`); all of these are opt-in.
CSE: `MLTP_CSE=1` or `userOpts(cse=True)`; it changes the IPOPT path on the path-sensitive
23-state NLP and gains only ~1% wall time, since MA57 factorisation dominates.
JIT: `userOpts(jit=True)` (e.g. `MLTP(circuit='BCN', jit=True)`) or env `MLTP_JIT=1`; needs
gcc/clang/cl on PATH, else it warns once and builds without JIT.
Symbols: SX by default; `MLTP_SYM_TYPE=MX` (or `build_and_solve_nlp(sym_type='MX')`) builds the
NLP ~15x faster but evaluates the Jacobian/Hessian ~8x slower and takes a different IPOPT path.

**Tests.** There is **no pytest/unittest** and no runner script in the repo: the `test_*.py`
files (28 today) are plain scripts whose assertions run at module top level (no
`if __name__ == '__main__'` block), so the finest selectable unit is a **whole file** (the first
failing assert aborts that file). Run one directly, or loop over all (each exits non-zero on failure):

```powershell
python test_foundation.py        # one file
foreach ($f in Get-ChildItem test_*.py) { "== $($f.Name)"; python $f.Name; if ($LASTEXITCODE) { "FAILED: $($f.Name)" } }
```

- `test_foundation.py`: casadi-free helpers (collocation, geometry, simpleMA, Powertrain constants)
- `test_transcription.py`: `discretise`, `order='F'` pack/unpack, curvature mesh, warm-start interpolation
- `test_save_load.py`: result `.mat` -> `load_solution` round-trip (init nesting, `data.nlp`, config fields)
- `test_params_useropts.py`: `vehParams` (MF205 default, legacy CopyB) + `userOpts` branching and `mesh` (`auto` 2000 m rule) / `screening` / `tyre_set`
- `test_useropts_solveropts.py`: `userOpts` solver/collocation kwargs and defaults
- `test_vp_overrides.py`: `vp_overrides` propagation; unknown keys raise
- `test_warmstart.py`: `functions/warmstart.py` (structure check, interpolation, source planning)
- `test_screen.py`: QSS screen (analytic cases, BCN march, `screen_sweep`; CasADi anchors vs `vehModel` if casadi is present)
- `test_casadi_opts.py`: CSE/JIT opt-in and JIT fallback (needs casadi)
- `test_hsl.py`: HSL path resolution, opts rewriting, MUMPS fallback (needs casadi: it solves toy NLPs through IPOPT; the real-DLL checks run only if Coin-HSL is found)
- `test_mltp_params.py`: `MLTP` signature checks (imports casadi, no solve)
- `test_mltp_warmstart.py`: `warmstart_guesses` / `warmstart_full` on the real 23-state model (row layout, input seeding by channel across configs, non-uniform grids, the warm-start modes; casadi, no solve)
- `test_mltp_constraints.py`: `build_path_constraints` per `TyreModel` / config vs `MLTP.m`, input-rate bounds divided by `u_s`, what `MLTP()` / `MLTP_paramOptim` hand to `build_and_solve_nlp` (captured, no solve), `plotSDI.friction_usage`
- `test_paramoptim_warmstart.py`: `optimise_design(warm_start=...)` from a full 23-state result (`extend_full_start`, the `plan_design_warm_start` modes `full+duals` / `full-primal` / `full-interp` / `cold`, what `MLTP_paramOptim` / `MLTP_TyreOptim` hand to `build_and_solve_nlp` captured by a stand-in, one real solve capped at `max_iter=5`; sections 3-4 need casadi + `Data/DATA_AA.mat`)
- `test_refine.py`: `functions/refine.py` (defect indicator exact for polynomial solutions, O(h^(d+1)), localised; the NLP's input arithmetic pinned by one tiny real solve; `refine_knots`, `run_refinement` stop reasons, options, record round trip) and the `MLTP(refine=...)` wiring with a stand-in solver (casadi + `Data/DATA_AA.mat`, no 23-state solve)
- `test_ladder.py`: `functions/ladder.py` (ladder registry, `homotopy_schedule`, exact-mu `friction_overrides`, the longitudinal torque rules, `qss_profile` = the screen's march, `seed_m7` / `seed_m23` on the real model scales across configs and meshes incl. the Xi-box clip) and the wiring: `MLTP_initial(seed='const')` guesses bit-identical to the legacy constants, `MLTP(ladder=...)` / `MLTP(homotopy=...)` with stand-ins, one real Sturn solve capped at `max_iter=5` (~17 s; sections 4-10 need casadi + `Data/DATA_AA.mat`)
- `test_setup_sweep.py`: `setup_sweep.py`, `functions/sweep.py` and `MLTP_screen.screen_batch` (Sobol/LHS design, shortlist, bridges, rank metrics, `screen_batch == screen_sweep` exactly incl. a worker pool, a casadi-blocked child run, field classification; section 5 is a ~1 min real Sturn mini-sweep in a child process: hub check at 0 iterations, resume, determinism, `SweepError`s)
- `test_vehmodel_matlab.py`: `vehModel.py` vs `vehModel.m` reference values and the inherited model quirks (casadi + `Data/DATA_AA.mat`)
- App/GUI group (`test_headless_config`, `test_runconfig`, `test_vp_params`, `test_presets`, `test_paths`, `test_results`, `test_solve_runner`, `test_spec_includes`, `test_mainwindow`, `test_gui_logic`): cfg.json forwarding, `RunConfig`, vp registry, presets, paths, results parsing, solve dispatch, PyInstaller-spec lint, offscreen Qt window (PySide6)

Only three files run the real 23-state NLP through IPOPT: `test_setup_sweep.py` (section 5, ~1 min),
`test_paramoptim_warmstart.py` (section 4) and `test_ladder.py` (section 9), both capped at
`max_iter=5`; the others use stand-ins, toy NLPs or no solve. `test_runconfig.py` / `test_results.py` write scratch files into
the repo root unless `CLAUDE_JOB_DIR_TMP` is set. Smoke-test the symbolic models (CasADi needed),
and a real solve (~20 s, default MF205 tyre), with:

```powershell
python -c "from functions.context import Ctx; from Powertrain import Powertrain; from vehParams import vehParams; from userOpts import userOpts; from vehModel import vehModel; ctx=Ctx(); Powertrain(ctx); vehParams(ctx); userOpts(ctx); vehModel(ctx); print(ctx.m23.nx, ctx.m23.nu)"
python -c "from MLTP import MLTP; MLTP(circuit='Sturn', save=False, plot=False)"
```

## Architecture (the big picture)

### The `ctx` spine
A single mutable `Ctx` object (a `SimpleNamespace` subclass, `functions/context.py`) replaces
MATLAB's base workspace and is threaded through **every** stage. Each stage **mutates `ctx`
in place** (and also returns it). The canonical driver order is:

```
Powertrain(ctx) -> vehParams(ctx) -> userOpts(ctx) -> vehModel(ctx) | vehModel_initial(ctx) -> MLTP*(...)
```

Key namespaces hung off `ctx`: `ctx.vp` (vehicle params, flat), `ctx.pt` (powertrain ratings +
`EM4`/`ATD` flags), `ctx.mf` (full Pacejka 5.2 coefficients), `ctx.aero` (from `DATA_AA.mat`),
`ctx.track` (s, k, optional x/y), `ctx.opts` (IPOPT options dict), `ctx.Xi`/`ctx.Xf` (boundary
conditions), and the result models `ctx.m7` / `ctx.m23` / `ctx.data`.

### Two-stage solve
1. **`MLTP_initial.py`** builds a simplified **7-state bicycle model** (`vehModel_initial` ->
   `ctx.m7`, nx=7/nu=3/ny=1, lumped Magic-Formula tyre) and solves a reduced OCP to produce a
   warm start, saved as `Results/init_<circuit>.mat` (`data.init`).
2. **`MLTP.py`** builds the **full 23-state model** (`vehModel` -> `ctx.m23`) and solves the real
   problem, saving `Results/<circuit>_<config>.mat`. If `warm_start=None`, `MLTP()` calls
   `MLTP_initial(save=False)` itself and interpolates the 7-state solution onto the 23-state grid
   **by arc length** via `warmstart_guesses()` (the two grids may differ in N and mesh). That is
   the default `ladder='legacy'`; `ladder='qss7'` / `'qss23'` seed from the QSS speed profile
   instead (see Multi-fidelity ladder).

`MLTP.py` also owns the only definition of `build_path_constraints()`, ported from `MLTP.m`'s
`switch TyreModel` (it follows the model's `m.TyreModel`). With the default `'CombinedSlip'` there
are powertrain rows only: `motor_power`, `motor_rpm`, `BrTh_1` (+ `ATD_eq` with ATD On), or 12
per-motor rows with EM4 (nh = 3 / 4 / 12). `'PureSlip'` puts the four friction circles
`rho_lim_*` <= 1 first (nh = 7 / 8 / 16). Without those rows `plotSDI` computes the
friction-circle usage from `data.vehicle`.

### Warm starts and sweeps
`MLTP(warm_start=...)` takes an init `.mat` (7-state `data.init`), a full result `.mat`, or an
in-memory `ctx` / `ctx.data` (chain solves without disk I/O). Every full result stores `data.nlp`
(`w_opt`, `lam_g`, `lam_x`, `structure`, `x_s`/`u_s`, IPOPT status). If the new NLP's **structure**
matches (sizes, `input_keys`, collocation grid `s_full`; setup/tyre values such as `vp_overrides`
may differ, which is the sweep case) the saved primal AND duals are re-injected with IPOPT's
warm-start recipe (`warmstart.warm_start_ipopt_opts()`, used only on a dual-seeded resolve, never
cold; explicit `ipopt_overrides` still win; `warm_start_duals=False` seeds the primal only).
Otherwise (N, `OPT_d`, mesh or config changed) the old result is interpolated onto the new knots by
arc length (primal only). The mode (`cold | init7 | full+duals | full-primal | full-interp`) is
printed and stored in `data.nlp.warm_start`. Measured on Sturn (N=18, ma57; five-coefficient MF205
proxy; pre-fix NLP, see the constraint-set bullet): a cold solve takes 257 iterations / 32.5 s; an
identical resolve with duals 0 iterations / 4.7 s (primal-only 30); +3% `alpha_RW` takes 11 iterations warm vs 364 cold, +3% `mb` 32 vs 327.
Full `tyre_set='MF205'` re-check (also pre-fix), +3% mass: 10 warm, 109 primal-only, 505 cold.
Iteration counts are path dependent. A `MLTP_paramOptim` result has `n_param` > 0, so it seeds by interpolation only.
`optimise_design(warm_start=...)` (`MLTP_paramOptim`, `MLTP_TyreOptim`) takes the same full MLTP result: if it is the design NLP minus
the appended P block (`warmstart.plan_design_warm_start`) it starts from `[w_opt; P0]` (P0 = current vp values), `lam_g`, `[lam_x; 0]`
(`extend_full_start`), else `full-interp` / `cold` as above; mode in `ctx.elapsed` and `data.nlp` (`test_paramoptim_warmstart.py`). Sturn,
default params, ma57: 5 iterations to 18.0080 s from the 179-iteration 18.0086 s result (primal-only 26, same point) vs 220 to 18.2439 s from the 7-state init.

### Multi-fidelity ladder (`functions/ladder.py`)
`MLTP(ladder=...)` builds every cold start (no or a refused `warm_start`, refine's cold retry) from a chain of
tiers: `const` (constant guesses), `qss` (the screen's g-g-v march, `qss_profile`, 10-40 ms), `m7`, `m23`.
`'auto'` (default) = `ladder.AUTO_LADDER` = `'legacy'` (const -> m7 -> m23, the old path); `'qss7'` = qss -> m7
-> m23 (`MLTP_initial(seed='qss')`); `'qss23'` = qss -> m23; anything else, or a QSS ladder with a 7-state init
`warm_start`, raises `ValueError`. Seeds: the QSS speed plus the motor or brake torque that holds its
acceleration; m7 adds a steady turn (r = v k, slips from the inverted Magic Formula, steer), m23 stays
laterally neutral (vy = r = n = `OPT_e`, eps = delta = 0, chassis quasi-static). QSS rungs default
`ma57_pre_alloc` to 3.0 (else `Insufficient_Memory`); `data['ladder']` records each rung's cost.
`homotopy=True` (= `(1.2, 1.1, 1.0)`; default None) solves once per tyre-friction scale (`pDx1, pDx2, pDy1,
pDy2`), step 1 like a plain call, later steps warm with duals; refine, save and plot act on the last step
(`data['homotopy']`). Measured (vi = 60, idle, ma57, MF205; 7-state + 23-state iterations, total wall, lap):

| | `legacy` | `qss7` | `qss23` |
|---|---|---|---|
| Sturn, N=18 | 250 + 179, 20.0 s, 18.0086 s | 255 + 626, 55.5 s, +3 ms (same branch) | 166, 14.9 s, +12 ms (near) |
| BCN, N=155 | 1072 + 247, 305 s, 116.4408 s | 840 + 188, 301 s, +1 ms (same branch) | 313, 292 s, +27 ms (slower) |

With vi at 60 and 1-2 mm/s off (5 Sturn / 3 BCN starts; cold scatter ~0.01 s) the 23-state medians were Sturn
213 / 343 / 221 and BCN 247 / 188 / 234 (legacy / qss7 / qss23); one Sturn start per ladder went wrong (legacy
+1.02 s, qss23 +0.60 s, qss7 `max_iter`) and all qss23 BCN laps were 13-33 ms slow. So `'legacy'` stays the
default: no QSS seed gives the 1.5-3x the roadmap expected (qss7 trims BCN iterations by a fifth, not its wall,
and is fragile on Sturn; qss23 only saves the 7-state solve: -25% wall on Sturn, -4% on BCN). Use `'qss23'` for
a quick screening-grade cold start, `homotopy=True` when a cold solve fails: EM4 Sturn (ATD Off) runs legacy to
`max_iter` 6000, homotopy converges in 250 + 165 + 65 + 70 iterations (35 s, 16.611 s; qss23: 1092, 16.663 s).

### Co-optimization wrappers
- **`MLTP_paramOptim.py`** promotes static design parameters (`vp` fields) to constant-over-lap
  decision variables and solves them **jointly** with the racing line. Its `optimise_design(param_specs, tag, ...)`
  is the shared core (`param_specs` = list of `(vp_field, lower, upper)`). Default params:
  `brkB, Tdist, alpha_FL/FR/RW/TW`.
- **`MLTP_TyreOptim.py`** is a thin wrapper over `optimise_design()` promoting only `Fz0_shift`.

### Fast QSS screening tier (`MLTP_screen.py` + `functions/ggv.py`)
A numpy-only quasi-steady g-g-v lap-time ESTIMATE (7 ms Sturn, 30 ms BCN per setup once `ctx`
exists; Sturn 18.744 s and BCN 129.467 s, i.e. +4.1% / +11.2% vs the default NLP laps of 18.0086 s
and 116.441 s) for ranking setups.
**Not an optimum**: fixed centreline (n = 0, track width unused), point mass, no transients.
`python MLTP_screen.py` or `MLTP_screen(circuit='BCN', vp_overrides={...})` -> `Results/<circuit>_<cfg>_qss.mat`
(`data.fidelity='qss'`, profile on the NLP `s_full` grid + `data.envelope`);
`screen_sweep(circuit, [ov1, ov2, ...])` is the DoE hook. Two separable stages:
`build_envelope(ctx)` (per-speed tyre/aero/powertrain limits, ~5 ms), then `march(env, s, k, vi)`
(apex speeds + Heun forward/backward pass, `ds_fine=1` m). Default `load_model='vehModel'`
reproduces the 23-state model's quasi-steady loads, quirks included (tyre-load sum
`ms*g + 4*mus*g`, aero split by `l_r/L`, mass `ms`); `'nominal'` is the textbook `m*g`/`Wfl0`
basis. Active aero is evaluated at the static wing angles; `test_screen.py` anchors mu, aero and
loads against vehModel. The screen uses the intended `mu_y` peak; the legacy `tyre_set='CopyB'`
zeroes vehModel's cornering stiffness (see the tyre bullet below), so compare the screen only with
NLPs on the default MF205 set.

### The transcription engine: `functions/transcription.py`
Shared by all MLTP scripts. `discretise(track, OPT_ds, OPT_d, mesh='uniform', mesh_opts=None,
s_knot=None)` builds the Legendre collocation grid (defaults from `userOpts.py`: step
`OPT_ds=30` m, degree `OPT_d=3`, `OPT_uinter='linear'`; knot placement: see Collocation mesh);
`build_and_solve_nlp(..., warm=None, sym_type=None)` assembles the CasADi NLP, decision vector
`w = [Xk; Uk; (Yk); Xkj (; P)]` packed **column-major (`order='F'`)**, collocation defect +
endpoint continuity, path constraints, time-domain input-rate limits, `Xi/Xf` boundary bounds
(**`NaN` = free state**), objective `J = Σ Qk·B·dsk` + regularisation, then calls
`nlpsol('ipopt')`. Besides `sol` it returns `w_opt`, `lam_g`, `lam_x`, `structure` (the sizes a
later solve must match to re-inject them) and `warm_info`. `_make_solver()` **selects the
configured `linear_solver`** (HSL `ma*` if a Coin-HSL DLL loads, else `mumps`; see Using
Coin-HSL). After the solve: `unpack_solution`, `reconstruct_x_full`, `interp_inputs`,
`compute_time`, `reconstruct_track`.

### Collocation mesh
`userOpts(mesh='auto' | 'uniform' | 'curvature', mesh_opts={...})` (default `'auto'`; any other
value raises `ValueError`). `'auto'` is resolved in `userOpts` once the track is loaded: to
`'curvature'` when the track length (`track.s[-1] - track.s[0]`) is >= 2000 m (BCN-size circuits),
else `'uniform'` (Sturn and the other short synthetic tracks); an explicit `'uniform'` or
`'curvature'` always overrides. `ctx.mesh` holds the resolved value (what `discretise` receives),
`ctx.mesh_requested` the requested one. `'curvature'` uses `functions/mesh.curvature_mesh`
(numpy only): the same N = round(L/`OPT_ds`) knots (or `mesh_opts={'N': ...}`) are placed by
equidistributing `M = 1 + a|k|/k_ref + b|dk/ds|/dk_ref`, so corners and their entry/exit get
denser knots and straights sparser (spacing within [0.25, 2.5] x `OPT_ds`). `k_ref` / `dk_ref`
are the `pct`-th percentiles (90) of the track's own smoothed |k| and |dk/ds|; a signal that is
zero up to float rounding (a straight, a constant-radius circle's dk/ds, curvature noise around
zero; judged as under 1e-6 rad of turning, or under 1e-6 of peak |k| of curvature change, over
the whole lap) drops its term, so such tracks get the uniform mesh. `mesh_opts` keys:
`a, b, ds_min, ds_max, smooth_window, N, pct, grid_ds` (a = b = 1 by default). `'auto'` only
redistributes the knots and does not change N (still round(L/`OPT_ds`)), so BCN at the default
`OPT_ds=30` gets a curvature mesh with N=155, the same interval count as before. Measured with
the default call (ma57, `tyre_set='MF205'`, pre-fix NLP; with the constraint-set fix in Conventions it
takes 247 iterations, 116.441 s, 253 s of solve), `'auto'` on BCN took 614 iterations, a 116.523 s lap
and 566 s solve against 852 iterations, 117.42 s and 968 s for the documented uniform N=155 run
(five-coefficient proxy tyre), i.e. 1.7x wall at equal N, and its 7-state init now converges
(1072 iterations, Optimal; the uniform-mesh init had ended `Error_In_Step_Computation`). ZigZag
(`'auto'` resolves to curvature, N=79; pre-fix NLP): `Solve_Succeeded`, 481 iterations, 56.216 s;
Jarama and Spa also resolve to curvature under `'auto'` and are unmeasured. The gain at equal N is
also accuracy (Sturn N=18,
five-coefficient MF205 proxy, pre-fix NLP, lap error vs the fine `OPT_ds=15` reference: uniform
+0.72% in 257 iterations, curvature +0.16% in 240). The measured BCN 2.5x speedup compared curvature at
`OPT_ds=45` (N=103: 117.40 s, 497 iterations, 392 s wall) with uniform at `OPT_ds=30` (N=155:
117.42 s, 852 iterations, 968 s wall; both pre-fix, five-coefficient proxy), so to get the
interval reduction pass `OPT_ds=45`
(or larger) together with the curvature mesh, e.g. `MLTP(circuit='BCN', OPT_ds=45)`. Results record
`data.mesh` / `data.mesh_opts`. A new mesh changes `s_full`, so a result saved on another
mesh (e.g. an older uniform-mesh BCN result) seeds by interpolation (no dual re-injection).

**Adaptive mesh refinement (`MLTP(refine=...)`, `functions/refine.py`; off by default).**
`refine=None` is the unchanged single solve. `refine=True` or a dict (`passes` 2, `tol` 1e-2,
`max_N` 4 x N0, `merge` False, `ds_min` 0.125 `OPT_ds`, `ds_max` 2.5 `OPT_ds`, `max_split` 2,
`pass_max_iter` 1000, `states` `('n', 'eps')`; a bad key or value raises `ValueError` before any
solve) solves on the userOpts mesh, then per pass scores every interval by the integrated defect
of the n / eps state polynomials (`defect_errors`: max |p(sigma) - X_k - h int f| over 2d
sub-segments, Gauss quadrature of `f_dyn` with the NLP's own near-hold input arithmetic; `tol`
1e-2 = 5 cm of n, 10 mrad of eps), bisects the intervals above `tol` (nested knots, `OPT_d` fixed)
and re-solves from `MLTP.warmstart_refined` (vx, vy, r, n, eps from the previous polynomials,
inputs from the previous NLP, wheel / suspension / tyre states re-seeded quasi-statically; primal
only, IPOPT `max_iter` capped at `pass_max_iter`). Only n / eps are scored because the model is
stiff (wheel spin |lambda| h ~ 1e4): the all-state defect is ~1e3 on every interval and cannot
localise. A pass that does not converge is retried once from the 7-state init, else the last
converged pass is kept (stop `solve-failed`; the others are `tol`, `passes`, `max_N`,
`no-split`). `ctx.refine_log` / `data.refine` hold one row per solve attempt; once a refined pass
is solved, `data.mesh = data.mesh_requested = 'adaptive'` (saved as `<stem>_meshAdaptive`,
`data.nlp.warm_start = 'refine'`, or `'init7'` after a cold retry). Measured (ma57, MF205; Sturn
lap error vs the uniform `OPT_ds=15` N=36 optimum 17.856 s, which a cold solve does not reach in
6000 iterations): uniform N=18 +0.85% (179 iterations, 20 s); from `OPT_ds=45` (N=12, +2.07%)
the default two passes give N 12 -> 21 -> 27 at +0.007% (625 iterations, ~57 s) and a third pass
meets `tol` at N=30 (-0.09%, 917 iterations, 93 s); `refine=True` on the default N=18 mesh stops
on `tol` after one pass at N=24 (-0.02%, 278 iterations, 33 s). BCN from `OPT_ds=60` (curvature
N=78): N 78 -> 130 -> 139, 116.747 s in 363 s, against 116.441 s in 315 s for the default N=155
solve (no gain). Fine meshes are fragile: 2 of 8 measured passes did not converge (Sturn
N 18 -> 36 bisect-all, which a seed differing only in the last bits had solved in 171
iterations, and N 36 -> 40), each burning ~2 x `pass_max_iter` iterations (5-7 min on Sturn)
before the fallback; laps on one mesh differ by up to ~0.3% between seeds, so judge against a fine
reference, not pass to pass. `refine` is an `MLTP` kwarg only (no GUI widget).

### Configuration: `userOpts.py`
Builds `ctx`: calls `Powertrain`+`vehParams`, loads or **synthesizes** the track, sets `Xi/Xf`,
collocation options, the IPOPT options dict, and the config switches the models branch on
(`vp.ActAero`, `pt.ATD`, `pt.EM4`). Circuit selection: `_REAL_CIRCUITS` maps names to `.mat`
files in `Circuits/` (`BCN` -> `Barcelona_circuit.mat`, the sector splits `BCN_S1`/`BCN_S2`/`BCN_S3`,
plus `Jarama`, `Spa`, `BCNAssetto`); any other name is treated as **synthetic** and its curvature
is generated analytically by `_synthetic_curvature` (`Straight`, `Hairpin`, `Sturn`, `Circle`,
`ZigZag`, `ZigZagMirror`, `VirtualTrack`), no `.mat` needed. Guard: if `ATD` **and**
`Electric_4Motors` are both `On`, it forces `ATD=Off` with a warning.
Other kwargs: `vp_overrides` (vehParams primaries + Pacejka coefficients), `tyre_set` (default
`'MF205'`; `'CopyB'` is the legacy set, see the tyre bullet), `OPT_ds` / `OPT_d` / `OPT_e`, `mesh`
(default `'auto'`, see Collocation mesh) / `mesh_opts`, `max_iter` / `tol`, `linear_solver` /
`hsl_dir`, `cse` / `jit` (opt-in), `screening`, and `ipopt_overrides` (dict of any IPOPT options,
merged last so it beats every default, the screening preset and the warm-start recipe; a non-exact
`hessian_approximation` warns). Kept on `ctx.mesh` (resolved), `ctx.mesh_requested`,
`ctx.mesh_opts`, `ctx.tyre_set`, `ctx.screening`, `ctx.ipopt_overrides`, `ctx.cse`, `ctx.jit`.

### Helpers (`functions/`)
- **Solver-side:** `collocation.py` (Legendre points/coeffs; CasADi-native with a numpy fallback),
  `simpleMA.py` (pre-solve curvature smoothing), `importfile.py` (`.mat` I/O, `mat_to_namespace`,
  `load_solution` warm-start unwrap), `context.py` (`Ctx`), `hsl.py` (Coin-HSL dir, probe, MUMPS
  fallback), `casadi_opts.py` (`fn_opts`: CSE/JIT), `warmstart.py` (dual warm-start recipe,
  NLP-structure check, warm-start source resolution, `nlp_record`), `mesh.py` (`curvature_mesh`,
  `solution_knots`, `mesh_opts_record`), `ggv.py` (QSS envelope + march), `ladder.py` (multi-fidelity
  ladder: tier registry, `resolve_ladder`, QSS seeds `seed_m7` / `seed_m23`, `quasi_static_states`,
  the friction homotopy hook; numpy only, see Multi-fidelity ladder), `refine.py` (adaptive
  mesh refinement: defect indicator, knot bisection, the refinement loop; see Collocation mesh).
- **Post-processing only** (reached via `reconstruct_track` *after* the solve): `curv2cart.py`
  (s,k -> cartesian centreline), `cartPath.py` (lateral offset `n` -> racing line), `trackLimits.py`
  (boundary polylines), `rotatePoint2D.py` (used only inside `curv2cart`).
- **`plotSDI.py`** writes Plotly HTML to `Plots/<circuit>/<config>/`; called from `MLTP()` when `plot=True`.

### Windows app (`app/`, `headless_solve.py`, `build/`)
PySide6 GUI (`python -m app.main`) around the same solve: `MainWindow` collects a `RunConfig`
(`app/runconfig.py`), writes a `cfg.json` and runs `headless_solve.py cfg.json` (frozen: `<exe> --headless
cfg.json`) as a subprocess. `headless_solve.build_solve_kwargs(cfg, resource_root)` maps cfg ->
`MLTP(**kwargs)`: forward a new `userOpts` kwarg there and add the matching `RunConfig` field.
The Advanced tab has a Tyre set combo (`MF205` default, `CopyB`) and a Mesh combo (`auto` default,
`uniform`, `curvature`); `mesh_opts` and `screening` still have no widget. The Setup table's Pacejka
defaults, Reset and changed-value highlighting follow the Tyre set combo: on a switch, the
tyre-set-dependent (lateral) cells still at the old set's default are re-seeded with the new set's
values, while cells the user edited keep their value and are flagged against the new set. The
`brkB`/`Tdist` rows are greyed while ATD is on and 4 Motors is off (inert in the 23-state model; only
the 7-state warm start reads them). The tabs are built Main, Advanced, Setup, Output (the Setup table
seeds from the Tyre set combo) but shown Main, Setup, Output, Advanced. Packaging notes: `build/README.md`.

### I/O directories
`Circuits/` (real track `.mat`), `Data/DATA_AA.mat` (aero coefficients, **required for a real
solve**), `Results/` (`.mat` outputs), `Plots/` (HTML figures).

## Conventions & gotchas (read before editing models)

- **Space-domain ODEs.** Every derivative in `m.dx` is multiplied by the change-of-variable
  `sf = (1 - n·kappa)/(vx·cos(eps) - vy·sin(eps))` (dt/ds). Curvilinear coords: `n` = lateral
  offset from centreline, `eps` = heading error, `kappa` = local curvature passed in per-knot as
  the variable parameter `pv` (`kappa > 0` = left turn). `sf` is also the lap-time integrand `L`.
- **Everything is normalised.** `x`/`u` are scaled symbols; physical values (e.g. `vx = 100·vx_n`)
  are reconstructed inside the equations, and `m.dx` is divided by the scale vector at the end.
  When evaluating optimal values, multiply/divide by `m.x_s`/`m.u_s` (e.g. `x_opt / m.x_s`).
- **Full state vector (nx=23, in order):** `vx, vy, r, n, eps, Om_fl, Om_fr, Om_rl, Om_rr, zs,
  zsdot, theta, thetadot, phi, phidot, wu_fl, wu_fr, wu_rl, wu_rr, zt_fl, zt_fr, zt_rl, zt_rr`.
  The full model has **no aux variables (ny=0)**; the init model has 1.
- **Config-dependent controls.** The control vector is assembled in MATLAB order and its length
  varies: `[1 motor torque if pt.EM4==0 else 4] + T_brake + [4 ATD_* if pt.ATD==1] +
  [0/1/2/4 active-aero inputs per vp.ActAero] + delta (steering, last)`. "4EM" = four per-corner
  motors; "Suspension" = heave/pitch/roll + 4 unsprung + 4 tyre-deflection states; "FullTyre" =
  full Pacejka 5.2 combined-slip on both axes from `ctx.mf`.
- **Two distinct tyre models coexist**, don't conflate: full Pacejka 5.2 `ctx.mf` (~60 coeffs,
  used by `vehModel.py`) vs. the simplified 8-param `vp.tyre` (used **only** by `vehModel_initial.py`).
- **Default tyre set is `tyre_set='MF205'`; the legacy `'CopyB'` set has zero cornering stiffness.**
  `'MF205'` (a `userOpts` / `MLTP` / `MLTP_screen` kwarg, or `vehParams(ctx, tyre_set=...)`) is the lateral
  set MATLAB actually runs, `MF_205_60R15_V91` (nine lateral values, e.g. `pKy4 = 2.0`, `pKy1 = 20.505`).
  `'CopyB'` is the legacy shipped set (Copy B), kept to reproduce old results: its lateral block (pEy1, pKy1,
  pKy4, pKy5, pKy6, pVy1-4) is the MATLAB `Test` case (`vehParams.m` ~279-346), where `pKy4 = 0` makes `Kya`,
  and so the slip-driven lateral tyre force, identically zero in `vehModel` (the NLP corners by drifting);
  the rest of the set matches `MF_205_60R15_V91`. Sturn, N=18, before the NLP constraint-set fix below: Copy-B
  gave a 25.57 s lap in 3209 IPOPT iterations (~250 s), MF205 18.01 s in 489 iterations (~35 s of solve on an
  idle machine); with the fix MF205 takes 179 iterations (18.0086 s, ~16 s of solve); MATLAB gets
  18.008-18.022 s. The benchmark runs in `docs/phase1_findings_2026-10-04.md` used a five-coefficient subset of
  MF205 (`pEy1, pKy1, pKy4, pKy5, pVy1`) as a proxy; it took 257 iterations for 18.016 s (pre-fix NLP), so
  quote 257 only for that proxy. The owner flipped the default from CopyB to MF205 on 2026-10-04 (every
  earlier Python result came from a zero-cornering-stiffness car; reproduce one with `tyre_set='CopyB'`). `vp_overrides` apply on top of
  either set. Results record `data.tyre_set`.
- **Solver facts (measured on Sturn/BCN with ma57).** MA57 factorisation dominates wall time (~90%) and
  NLP function evaluation is only ~10%, so per-evaluation tweaks (CSE, JIT, `f_dyn.map`, tyre tabulation) gain
  a few percent at most; the levers are the IPOPT iteration count (warm start + duals, mesh, tolerances) and
  the number of factorisations. The 23-state NLP is **path-sensitive**: a 1-ulp Hessian change (CSE on) or the
  MX route can land IPOPT on a different local optimum, so validate a solver change by oracle-function
  equivalence or against a fine reference, never by expecting identical iterates or lap times. Keep
  `ma57_automatic_scaling='yes'` (default; without MC64 MA57 stalls, e.g. on `VirtualTrack`). The IPOPT
  defaults are near-best: L-BFGS, monotone mu, `nlp_scaling_method='none'` and MUMPS all measured worse.
  `userOpts(screening=True)` loosens five tolerances (`userOpts.SCREENING_IPOPT`: `tol=1e-3`,
  `acceptable_tol=1e-2`, `dual_inf_tol=1e-2`, `constr_viol_tol=1e-3`, `compl_inf_tol=1e-3`). It is a looser
  stopping rule, not a guaranteed saving: on Sturn five matched pairs moved -41% to +16% in iterations
  (pre-fix NLP, not re-measured since: 196 vs 257 with the five-coefficient MF205 proxy, 290 vs 489 with
  `tyre_set='MF205'`; the other three proxy pairs gave +16%, -7% and +0.3%) for lap changes of -4 to
  +20 ms, and iteration counts on this
  NLP swing up to +-40% under 1-ulp perturbations. Use it for ranking sweeps, not final numbers; it
  replaces the `tol` argument and an explicit `ipopt_overrides` still wins.
- **`vehModel_initial.py` is NOT obsolete**: it's the deliberately reduced warm-start model.
  `vehModel.py` is the canonical full model.
- **Wheel-radius / gear quirk (intentional, do not "fix").** `Powertrain` sets `vp.Rw=0.3142857`
  and derives `vp.gear` from it; `vehParams` then overwrites `vp.Rw=0.355` but deliberately does
  **not** recompute `vp.gear`. `test_params_useropts.py` asserts gear stays based on the old radius.
- **MATLAB-inherited model quirks (audited 2026-10-04, pinned by `test_vehmodel_matlab.py`).** `vehModel.py`
  matches `vehModel.m` bit for bit, so these are kept for parity, not port bugs: EM4 motor speed is
  `Om_wheel/gear` (single motor: `gear*mean(Om)`), so the per-motor power/rpm rows are gear^2 ~ 52.6x looser
  and never bind; every unsprung corner subtracts the total `vp.mus` (static tyre loads sum to `(ms+4*mus)*g`
  = 26.0 kN, not `m*g` = 20.5 kN; planar dynamics use `ms`); the front/rear Cl split does not move steady axle
  loads (aero reaches them in `l_r/l`, only attitude changes); with ATD On `brkB`/`Tdist` are inert. Open Python
  deviation: active-aero inputs are live (dead in `vehModel.m`).
- **23-state NLP constraint set matches `MLTP.m` (fixed 2026-10-04, pinned by `test_mltp_constraints.py`).**
  Two port gaps were closed. (1) The input-rate bounds are divided by `u_s` (`m.duk_lb/ub`, as `vehModel.m`
  L377-379 and the 7-state model do). Before, the physical Nm/s, rad/s and deg/s values bounded the normalised
  rates, which held steering to 0.061 rad/s and left the motor, brake and wing rates 602x, 4000x and 10-30x
  looser than specified. Steering is divided by the model's own scale `delta_max`, giving the documented
  0.1 rad/s; `vehModel.m` divides by a `delta_s = pi/8` leaked from `vehModel_initial.m` (0.156 rad/s as MATLAB
  runs), so matching that instead is a one-line owner call in `vehModel.py`. (2) The four friction-circle rows
  exist only with `TyreModel='PureSlip'`, as in `MLTP.m`. The default Sturn NLP now has MATLAB's size
  (n_w = 1812, n_g = 1904). Default solves after the fix (ma57, MF205): Sturn 179 iterations, 18.00859 s lap
  (before: 489, 18.00934 s; MATLAB 18.008 s); BCN (`'auto'` = curvature, N=155) 247 iterations, 116.441 s,
  253 s of solve (before: 614, 116.523 s, 566 s), with the 0.1 rad/s steering bound active; BCN AALB 241
  iterations, 116.065 s. The other iteration counts, laps and wall times in this file (the
  five-coefficient proxy runs, the screening-preset pairs, the warm-start timings) predate the fix:
  re-measure before quoting them. The default `Results/` baselines were regenerated after the fix
  (Sturn, BCN, BCN AALB, the inits, QSS, TyreOptim; `Sturn_paramOptim` again from the full-solution warm
  start); the full results among them seed a default solve with duals (same NLP structure). The
  `*_CopyB.mat` files are legacy pre-fix results (no `data.nlp`).
- **`Powertrain.py` is not a map.** Despite the name it only stores 5 scalar ratings
  (`Pmax, Tmax, OMmax, Vmax, eff`); `eff=0.9` is used only in the post-solve energy integral. The
  actual power/rpm limits are enforced in `MLTP.py`/`vehModel.py`. `pt.EM4`/`pt.ATD` are set later
  in `userOpts.py`, so code calling `Powertrain(ctx)` without `userOpts` lacks those attributes.
- **Missing `DATA_AA.mat` does not raise** in `vehParams` (it warns and falls back to placeholder
  `Cd`/`Cl`), but `vehModel(ctx)` raises `RuntimeError` if `ctx.aero is None`.
- **Units are SI by convention only (not enforced)**, except some aero AoA and camber/toe fields
  stored in **degrees** (with separate `*_rad` companions). Mixing the deg vs rad field is an easy bug.
- **The root `__init__.py` is stale/broken**: it imports `.importfile`/`.collocation`/`.context`,
  which live under `functions/`, not the root, so importing the repo root as a package raises
  `ImportError`. The working package init is `functions/__init__.py`. Import submodules directly
  (e.g. `from functions.importfile import load_solution`); the tests do.
