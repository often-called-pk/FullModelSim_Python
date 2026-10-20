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
NLP functions: env `MLTP_KEEP_NLP=1` keeps `ctx.nlp_fg` (casadi Function w -> (f, g)) and
`ctx.nlp_bounds` (lbw, ubw, lbg, ubg, w0) after the 23-state build, for the MATLAB parity gate; off by
default because the SX graph is large (peak commit ~8 GB at BCN N=465).

**MATLAB validation (`validation/`, plan and results in `docs/validation_matlab_vs_python.md`).**
`matlab_batch.py` runs patched scratch copies of the MATLAB original through `matlab -batch`
(`export`: parameter dump + NLP numbers at IPOPT max_iter 0; `solve`: MATLAB as shipped, plots cut);
`py_export.py` is the Python side (`PARITY` settings, `matlab_seed()` = MATLAB's 23-state start point);
`run_py.py` runs the Python tiers with split timers; `compare.py` diffs parameters, NLPs and runs.
Outputs go to `Results/validation/` (gitignored); the MATLAB repo is never edited.

**Tests.** There is **no pytest/unittest** and no runner script in the repo: the `tests/test_*.py`
files (41 today) are plain scripts whose assertions run at module top level (no
`if __name__ == '__main__'` block), so the finest selectable unit is a **whole file** (the first
failing assert aborts that file). Each starts with `import _bootstrap` (`tests/_bootstrap.py`: the
repo root goes first on `sys.path` and becomes the working directory), so a test runs from any
directory and its scratch files go to a temp dir, never the repo. Run one directly, or loop over
all (each exits non-zero on failure):

```powershell
python tests\test_foundation.py        # one file
foreach ($f in Get-ChildItem tests\test_*.py) { "== $($f.Name)"; python $f.FullName; if ($LASTEXITCODE) { "FAILED: $($f.Name)" } }
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
- `test_refine.py`: `functions/refine.py` (defect indicator exact for polynomial solutions, O(h^(d+1)), localised; the NLP's input arithmetic pinned by one tiny real solve; `refine_knots` incl. merge under `max_N`, `run_refinement` stop reasons and the lap-rise rule, options, record round trip) and the `MLTP(refine=...)` wiring with a stand-in solver (reseed, indicator and lap against an independent evaluation, cold retry, lap-rise rejection, `save=True`, time accounting, state names checked before any solve; casadi + `Data/DATA_AA.mat`, no 23-state solve)
- `test_ladder.py`: `functions/ladder.py` (ladder registry, `homotopy_schedule`, exact-mu `friction_overrides`, the longitudinal torque rules, `qss_profile` = the screen's march, `seed_m7` / `seed_m23` on the real model scales across configs and meshes incl. the Xi-box clip) and the wiring: `MLTP_initial(seed='const')` guesses bit-identical to the legacy constants, `MLTP(ladder=...)` / `MLTP(homotopy=...)` with stand-ins, one real Sturn solve capped at `max_iter=5` (~17 s; sections 4-10 need casadi + `Data/DATA_AA.mat`)
- `test_setup_sweep.py`: `setup_sweep.py`, `functions/sweep.py` and `MLTP_screen.screen_batch` (Sobol/LHS design, shortlist, bridges, rank metrics, `screen_batch == screen_sweep` exactly incl. a worker pool, a casadi-blocked child run, field classification; section 5 is a ~1 min real Sturn mini-sweep in a child process: hub check at 0 iterations, resume, determinism, `SweepError`s)
- `test_vehmodel_matlab.py`: `vehModel.py` vs `vehModel.m` reference values and the inherited model quirks (casadi + `Data/DATA_AA.mat`)
- `test_matlab_parity.py`: MATLAB parity gates against stored MATLAB exports in `tests/data/` (section 1: parameter dump of Sturn ds30, BCN and NBR ds10, no hit since the CG_p table port of 2026-10-18; section 2: 23-state NLP bounds exact, f and g within 1e-10 at the MATLAB start point, a random point and the MATLAB solution, Sturn N=18 at IPOPT max_iter 0; casadi + `Data/DATA_AA.mat`, no solve)
- `test_physics_flag.py`: `userOpts(physics=...)` values, the `_physCorrected` result stem, `data.physics` in the QSS record
- `test_physics_mass_frame.py`: corrected A1/A3 vs matlab: static equilibrium (shift-free tyre), planar mass `m` vs `ms`, no attitude gravity or Euler terms, QSS Sturn lap +7..13%, corrected seed equilibrium
- `test_physics_suspension_aero.py`: corrected A2/A4: damper formula, `xsdot` = d(xs)/dt, vertical modes (max|eig| < 2000 1/s, a 1-4 Hz body mode), smooth stops, no downforce in `fzs`, pitch sign, QSS Cl split
- `test_physics_em4_kinematics.py`: corrected A5/A6: EM4 `Om_motor = gear*Om` (model and QSS), toe indices, `Steering` / `CamberGain` options, CG_p table, table extrapolation as `linear_interp.m`
- `test_track_qa.py`: `functions/track_qa.py` verdicts (`check_track` turning, closure, sign, spacing; `k_from_xy` on circles, an open arc, a stadium), load gate (error -> ValueError, else one warning), rebuilt Jarama + BCN sectors, BCN k untouched (one warning), BCNAssetto out of the registry, QSS lap pins on the rebuilt tracks
- `test_output_channels.py`: E1: `m.ax = Lon_acc - vy*r`, `m.ay = Lat_acc + vx*r` (both physics, EM4 on/off), `data.vehicle` += `ax, ay, delta, Om_fl..Om_rr`, energy stored in Wh and shown in kWh, `plotSDI.accel_channels` (stored, derived for a pre-E1 result, or (None, None)), stored BCN RMS(vx*r)
- `test_start_state.py`: `userOpts(start_state=...)` boxes (matlab unchanged, rolling `Xi_tol`, periodic closed circuits only + closing sample), start stems, `n_periodic` in the structure record and the warm-start planning, periodic rows + start box in a fake-solver build
- `test_start_wiring.py`: `vehModel.rolling_start` (60 m/s coast equilibrium, variants, bounds, failures); start_state through `MLTP`, `MLTP_initial`, `optimise_design`, the refine passes, `setup_sweep._precheck_base`, the `MLTP_screen` stem; real builds with a fake solver (saved records, warm-start modes, plotSDI folder). Section 12: braking-limit warning (text values, silent at vi 67.5), rolling seed rows 9-22 at the equilibrium, rows 0-8 and inputs = matlab-start seed
- `test_track_width.py`: `userOpts(n_bounds=...)` forms + errors, `n_tag` stems, bounds at knots + collocation points (23- and 7-state), `n_bounds=4.0` == default, empty-start-box and corner-radius ValueErrors, `reconstruct_track` edges, full+duals warm start with a band, narrow-band continuity, periodic Spa arrays, `MLTP_screen` stem
- `test_sensitivity.py`: `functions/sensitivity.py`: `nlp_index` rows of w and g, `limit_prices` (W start-row rule, torque, brake, power, start speed), `lateral_prices`, `costates`, prices of the stored results, KKT check on the rebuilt Sturn NLP, oracle (rebuilt Lagrangian differences == prices)
- `test_tyreoptim_fz0.py`: C6: TyreOptim model at `Fz0_shift = S` == plain model with `Fz0 = 4905*S` (both physics), the coupling at S = 0.5, S = 1 exact, no coupling in `MLTP_paramOptim`
- `test_mf_scaling.py`: C6 part 2: `mf.LMUX` / `mf.LMUY` defaults (1.0, both tyre sets), overrides, unknown key raises. Full model at 0.9 / 0.8 (both physics, camber on, no solve): `mu_*`, the peak forces and the vertical shifts scale. The zero-slip slopes do not change (pHy1 = pHy2 = 0). With camber off and pVy1 = pVy2 = 0, the fy zero crossing (alpha = -SHy) does not move, and Svyk scales. QSS peaks and lap (1.0 = default lap). 7-state build unchanged. Parity export drops both only while both are 1.0 (checked for each factor). Sections 2-5 need casadi + `Data/DATA_AA.mat`
- App/GUI group (`test_headless_config`, `test_runconfig`, `test_vp_params`, `test_presets`, `test_paths`, `test_results`, `test_solve_runner`, `test_spec_includes`, `test_mainwindow`, `test_gui_logic`): cfg.json forwarding, `RunConfig`, vp registry, presets, paths, results parsing, solve dispatch, PyInstaller-spec lint, offscreen Qt window (PySide6)

Only three files run the real 23-state NLP through IPOPT: `test_setup_sweep.py` (section 5, ~1 min),
`test_paramoptim_warmstart.py` (section 4) and `test_ladder.py` (section 9), both capped at
`max_iter=5`; the others use stand-ins, toy NLPs or no solve. Smoke-test the symbolic models
(CasADi needed), and a real solve (~20 s, default MF205 tyre), with:

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
default params, ma57, pi/8 steering (2026-10-08 baseline): 5 iterations to 18.016 s from the 174-iteration 18.016 s result. Pre-pi/8: 5 to
18.0080 s from the 179-iteration 18.0086 s result (primal-only 26, same point) vs 220 to 18.2439 s from the 7-state init. All other numbers
in this subsection are pre-pi/8 (and pre-fix where stated).

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
(`data['homotopy']`). Measured pre-pi/8 (vi = 60, idle, ma57, MF205; 7-state + 23-state iterations, total wall, lap;
with the pi/8 steering scale the legacy default now gives Sturn 174 iterations, 18.016 s and BCN 1072 + 165, 116.444 s):

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
  As `MLTP_TyreOptim.m:128` it also scales the nominal load: `optimise_design(Fz0_follows_shift=True)` sets
  `vp.Fz0 = vp.Fz0*Fz0_shift` before `vehModel` (only this wrapper; `MLTP_paramOptim` and `setup_sweep` keep
  Fz0 fixed, as `MLTP.m` / `MLTP_paramOptim.m`). So S enters the lateral stiffness Kya twice (`fz/S` over
  `pKy2*Fz0`, ported as is). QSS mirror of a TyreOptim setup: `MLTP_screen(vp_overrides={"Fz0_shift": S,
  "Fz0": 4905*S})`.

### Fast QSS screening tier (`MLTP_screen.py` + `functions/ggv.py`)
A numpy-only quasi-steady g-g-v lap-time ESTIMATE (7 ms Sturn, 30 ms BCN per setup once `ctx`
exists; Sturn 18.744 s and BCN 129.467 s, i.e. +4.0% / +11.2% vs the default NLP laps of 18.016 s
and 116.444 s) for ranking setups.
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
takes 247 iterations, 116.441 s, 253 s of solve pre-pi/8, and 165 iterations, 116.444 s, 148 s with the
pi/8 steering scale), `'auto'` on BCN took 614 iterations, a 116.523 s lap
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
`pass_max_iter` 1000, `states` `('n', 'eps')`, `max_lap_rise` 3e-3; a bad key or value, or a state
name the model does not have, raises `ValueError` before any solve) solves on the userOpts mesh,
then per pass scores every interval by the integrated defect of the n / eps state polynomials
(`defect_errors`: max |p(sigma) - X_k - h int f| at the 2d+1 sub-segment ends, Gauss quadrature of
`f_dyn` with the NLP's own near-hold input arithmetic; `tol` 1e-2 = 5 cm of n, 10 mrad of eps; a
dense sampling reads up to 15-20% higher on Sturn solutions, so `tol` acts as ~1.2 x `tol`),
bisects the intervals above `tol` (nested knots, `OPT_d` fixed; with `merge=True` the knots merges
free count towards `max_N`, and a pass with nothing to split ends the run) and re-solves from
`MLTP.warmstart_refined` (vx, vy, r, n, eps from the previous polynomials, inputs from the previous
NLP, wheel / suspension / tyre states re-seeded quasi-statically; primal only, IPOPT `max_iter`
capped at `pass_max_iter`). Only n / eps are scored because the model is stiff (wheel spin
|lambda| h ~ 1e4): the all-state defect is ~1e3 on every interval and cannot localise. A pass that
does not converge, or converges to a lap more than `max_lap_rise` slower than the pass it refines
(a worse local optimum: Hairpin PureSlip N 11 -> 15 converged 4.5% slower than its base where other
seeds on the same knots were 0.5-2% faster), is retried once from the 7-state init, else the last
accepted pass is kept (stop `solve-failed` / `lap-rise`; the others are `tol`, `passes`, `max_N`,
`no-split`), so a pass can cost 2 x `pass_max_iter` iterations (BCN runs ~0.6 s per iteration).
`ctx.refine_log` / `data.refine` hold one row per solve attempt (`converged` = IPOPT status,
`accepted` = kept; `pass_no`, `N`, `lap_time`, `eta_max`, ...); once a refined pass is accepted,
`data.mesh = data.mesh_requested = 'adaptive'` (saved as `<stem>_meshAdaptive`,
`data.nlp.warm_start = 'refine'`, or `'init7'` after a cold retry; no entry rebuilds those knots, so
such a result seeds a later solve by interpolation only). Measured pre-pi/8 (ma57, MF205; every pass of these
runs lowered the lap, so the lap check keeps them; Sturn lap error vs the uniform `OPT_ds=15` N=36
optimum 17.856 s, which a cold solve does not reach in 6000 iterations and which is not itself
mesh-converged): uniform N=18 +0.85% (179 iterations, 20 s); `refine=True` on it stops on `tol`
after one pass at N=24 (-0.02%, 278 iterations, 33 s); from `OPT_ds=45` (N=12, +2.07%) two passes
give N 12 -> 21 -> 27 (+0.007%, 625 iterations, ~57 s) and a third meets `tol` at N=30 (-0.09%, 917
iterations, 93 s). A one-shot `mesh='curvature'` solve at equal N does as well for less: N=18 +0.15%
(109 iterations, 17 s), N=24 -0.10% (171, 33 s), N=27 -0.10% (210, 30 s); `refine=True` on a
curvature N=18 base ends at N=28, -0.001%, after 387 iterations. BCN from `OPT_ds=60` (curvature
N=78): N 78 -> 130 -> 139, 116.747 s in 363 s, against 116.501 s (274 iterations, 290 s) for a
curvature N=139 solve and 116.441 s in 315 s for the default N=155 one. Differences under ~0.1% do
not rank these: reseeds from seeds 1e-10 apart land up to 0.06% apart (one pair took 87 and 269
iterations), a cold and a reseeded solve on one mesh 0.05% apart, and the objective depends on the mesh (the
input-rate regularisation is summed per interval without `dsk`, as in `MLTP.m`: 0.26-0.34% of the
lap, growing with N). The reseed is fragile, coarse meshes included: Sturn N 18 -> 36 and 36 -> 40
did not converge (5-7 min each before the fallback), Sturn EM4 / AALB N 12 -> 18 hit 1000
iterations in 2 of 3 runs from seeds 1e-10 apart, Hairpin PureSlip N 11 -> 15 ended
`Restoration_Failed` in 2 of 4 and 15 -> 17 hit 1000 in 2 of 2; the cold retry rescued EM4 (965 iterations) and
Hairpin 15 -> 17 (231), never Sturn N >= 36. Use refinement to reach N=24-30 meshes that cold
uniform solves do not converge on; for speed or accuracy at a given N prefer `mesh='curvature'`.
`refine` is an `MLTP` kwarg only (no GUI widget).

### Configuration: `userOpts.py`
Builds `ctx`: calls `Powertrain`+`vehParams`, loads or **synthesizes** the track, sets `Xi/Xf`,
collocation options, the IPOPT options dict, and the config switches the models branch on
(`vp.ActAero`, `pt.ATD`, `pt.EM4`). Circuit selection: `_REAL_CIRCUITS` maps names to `.mat`
files in `Circuits/` (`BCN` -> `Barcelona_circuit.mat`, the sector splits `BCN_S1`/`BCN_S2`/`BCN_S3`,
plus `Jarama`, `Spa`, `NBR` (Nurburgring GP, 5139 m)); any other
name is treated as **synthetic** and its curvature
is generated analytically by `_synthetic_curvature` (`Straight`, `Hairpin`, `Sturn`, `Circle`,
`ZigZag`, `ZigZagMirror`, `VirtualTrack`), no `.mat` needed. Guard: if `ATD` **and**
`Electric_4Motors` are both `On`, it forces `ATD=Off` with a warning.
Track QA (2026-10-18, `functions/track_qa.py`): every real track goes through `check_track` at load
(turning, closure, sign agreement with x,y, spacing, monotonic s). An error raises ValueError (no bypass
kwarg), other issues give one warning. BCN warns (turning -1.023 laps): kept as is, the parity gates pin its
k. Jarama (was +78.7 laps of turning) and `BCN_S1/S2/S3` were rebuilt from their own x,y with
`k_from_xy(x, y, closed, ds=1.0, window=5, passes=2, max_chord=30.0)`; the source files stay in git at
ce0752a. BCNAssetto left the registry (no s, x, y; file kept on disk). The GUI circuit list shows NBR.
Other kwargs: `vp_overrides` (vehParams primaries + Pacejka coefficients), `tyre_set` (default
`'MF205'`; `'CopyB'` is the legacy set, see the tyre bullet), `OPT_ds` / `OPT_d` / `OPT_e`, `mesh`
(default `'auto'`, see Collocation mesh) / `mesh_opts`, `max_iter` / `tol`, `linear_solver` /
`hsl_dir`, `cse` / `jit` (opt-in), `screening`, `physics` (`'matlab'` default | `'corrected'`, see the physics
bullet), `Steering` (`'NA'` | `'Ackermann'`) / `CamberGain` (`'Off'` | `'Linear'` | `'Table'`), `start_state` (`'matlab'`
default | `'rolling'` | `'periodic'`, see the start-state bullet), `n_bounds` (None default | half-width W |
`(lo, hi)`, see the track-width bullet), and `ipopt_overrides` (dict of any IPOPT options,
merged last so it beats every default, the screening preset and the warm-start recipe; a non-exact
`hessian_approximation` warns). Kept on `ctx.mesh` (resolved), `ctx.mesh_requested`,
`ctx.mesh_opts`, `ctx.tyre_set`, `ctx.screening`, `ctx.ipopt_overrides`, `ctx.cse`, `ctx.jit`, `ctx.physics`,
`ctx.Steering`, `ctx.CamberGain` (`vehModel(ctx)` reads the last two when its arguments are None),
`ctx.start_state`, `ctx.periodic`, `ctx.Xi_tol`, `ctx.n_bounds`, `ctx.n_tag`, plus `ctx.track.n_lo` / `n_hi`
when a band is set.

### Helpers (`functions/`)
- **Solver-side:** `collocation.py` (Legendre points/coeffs; CasADi-native with a numpy fallback),
  `simpleMA.py` (pre-solve curvature smoothing), `importfile.py` (`.mat` I/O, `mat_to_namespace`,
  `load_solution` warm-start unwrap), `context.py` (`Ctx`), `hsl.py` (Coin-HSL dir, probe, MUMPS
  fallback), `casadi_opts.py` (`fn_opts`: CSE/JIT), `warmstart.py` (dual warm-start recipe,
  NLP-structure check, warm-start source resolution, `nlp_record`), `mesh.py` (`curvature_mesh`,
  `solution_knots`, `mesh_opts_record`), `ggv.py` (QSS envelope + march), `ladder.py` (multi-fidelity
  ladder: tier registry, `resolve_ladder`, QSS seeds `seed_m7` / `seed_m23`, `quasi_static_states`,
  the friction homotopy hook; numpy only, see Multi-fidelity ladder), `refine.py` (adaptive
  mesh refinement: defect indicator, knot bisection, the refinement loop; see Collocation mesh), `track_qa.py` (load-time track QA, `k_from_xy`
  rebuild recipe).
- **Post-solve analysis:** `sensitivity.py` (numpy only, no solve): `limit_prices`, `lateral_prices`,
  `costates` from the stored multipliers of one full result; first order, and J includes the regularisation
  (0.27% of the Sturn lap, 0.12% of BCN).
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
  idle machine); with the fix MF205 takes 179 iterations (18.0086 s, ~16 s of solve; with the pi/8
  steering scale 174 iterations, 18.016 s); MATLAB gets
  18.008-18.022 s. The benchmark runs in `docs/phase1_findings_2026-10-04.md` used a five-coefficient subset of
  MF205 (`pEy1, pKy1, pKy4, pKy5, pVy1`) as a proxy; it took 257 iterations for 18.016 s (pre-fix NLP), so
  quote 257 only for that proxy. The owner flipped the default from CopyB to MF205 on 2026-10-04 (every
  earlier Python result came from a zero-cornering-stiffness car; reproduce one with `tyre_set='CopyB'`). `vp_overrides` apply on top of
  either set. Results record `data.tyre_set`.
- **MF peak-grip scaling factors (C6 part 2, 2026-10-20, pinned by `tests/test_mf_scaling.py`).** `ctx.mf.LMUX` and
  `ctx.mf.LMUY` are the MF 5.2 factors lambda_mux and lambda_muy, float 1.0 in both tyre sets. They are MF keys, so
  `vp_overrides={"LMUX": x, "LMUY": y}` puts them in `ctx.mf`. `vehModel.py` places them as MF 5.2 does (PAC2002
  manual, eqs 20-23, 28, 34-35, 41, 74-75). LMUX scales mu_x and SVx. LMUY scales mu_y and SVy, camber part
  `Svy_gamma` included. Kxk and Kya do not change, so `Bx = Kxk/(Cx*Dx)` and `By = Kya/(Cy*Dy)` keep the slip
  stiffness. LVX and LVY are not added (= 1). The pHy part of SHy does not scale. Its camber part follows
  `Svy_gamma` (MF 6.1 form), so the camber thrust at small slip does not change. `_mf_scale` skips a plain 1.0, so
  the default SX graph stays byte-identical (NLP snapshot: none). A symbolic factor always multiplies. The QSS
  scales `peak_mu_x` / `peak_mu_y` (`functions/ggv.py`). The 7-state model (`vp.tyre`) ignores both, so its init
  seeds a cold solve at grip 1. `validation/py_export.params` drops both while both are 1.0 (no MATLAB
  counterpart). Any other value goes into the export, and gate 1 flags it. `Fz0_shift` and the TyreOptim Fz0
  coupling do not change. GUI: two Setup rows in "Tyre & Wheel". `result_stem` has no override tag, and
  `data.mf_overrides` keeps the key names only. So a factor run with `save=True` (the default of `MLTP` and
  `MLTP_screen`) overwrites the stored default result, for example `Results/Sturn_Static_ATDOn_EM4Off.mat`. Until
  E2 of `docs/laptime_sim_improvements.md` gives unique stems, run factor solves with `save=False` or with their
  own `results_dir`. Sanity run 2026-10-20 (Linux, MUMPS): Sturn at LMUX = LMUY = 0.95, full+duals from the
  stored Sturn result. It took 20 iterations (Solve_Succeeded) to 18.24010 s, +0.22377 s (+1.24%) against
  18.01633 s, in 3.5 s wall (2.2 s NLP build). The QSS gives 18.74393 s at 1.0 (= default) and 18.98036 s at 0.95
  (+1.26%). Track method for the values (candidate verdicts, no human review): `docs/tyre_scaling_track_method.md`.
  Its Table 5 links LMUX and LMUY to one value, set from data in the QSS only, after four gates. The full model
  keeps 1 until study B2 gives the full-model elasticity. Skipped: LKX, LKY, LFZO and the Cd / Cl scales, because,
  per the report, lap times, tyre temperature and pressure cannot identify them. Report Table 5 gives the data
  that each one needs. LKX needs steer, motor speeds, IMU data and hard runs. LKY needs steer, wheel speeds, IMU
  data and a side-slip sensor. LFZO needs a source for a temperature or pressure effect. Never fit LFZO from lap
  times. For the Cd and Cl scales, coast-down runs (Cd) and ride-height or GPS data (Cl) give leads only.
  Skipped: a QSS inversion (stint median lap -> grip scale), add when session data exist. Skipped: promotion
  in `optimise_design`, add when a co-optimisation needs it.
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
- **Physics flag (2026-10-18, pinned by `tests/test_physics_*.py`).** `userOpts(physics='matlab' | 'corrected')`.
  Default `'matlab'` = the MATLAB physics above, bit for bit (serialized SX graph hash checked in 10+ configs, parity
  gates green). `'corrected'` (opt-in) fixes the inherited physics, items A1-A6 of `docs/laptime_sim_improvements.md`:
  A1 per-corner unsprung mass `muf/2`, `mur/2`, planar equations over `m`, lever-rule static loads; A2 damper
  `c = 2*zeta*sqrt(k*m_corner)`, spring to the wheel (small-angle `xs` incl. wheel travel, so `xsdot` is exact), smooth
  bump/rebound stops (`vehModel.BUMP_GAP`, `REBOUND_GAP`, `K_STOP`, `W_STOP`), no torque cutoff; A3 no body-attitude
  gravity or Euler terms on a flat road; A4 downforce through the springs plus aero pitch/roll moments; A5 EM4
  `Om_motor = Om*gear`; A6 toe indices. All as python `if` branches. The QSS envelope (`_load_basis(..., physics)`,
  EM4 limits) and the ladder seeds (`quasi_static_states(vp, vx, physics)`) follow the flag. Results get the
  `_physCorrected` stem and `data.physics`. Corrected vertical modes: heave 1.9 Hz zeta 0.53, roll 3.0 Hz zeta 0.57,
  pitch 5.2 Hz zeta 0.35, max|eig| 191 1/s (matlab 2.7e4). Measured (Linux, MUMPS, cold): Sturn 18.019 -> 20.024 s
  (+11.1%, 152 -> 218 iterations), BCN 116.444 -> 130.035 s (+11.7%, 207 -> 179 iterations); QSS forecast +10.0% /
  +11.1%. Watch: the body sits 9.4 mm low at 60 m/s under corrected (12.8 mm at 70 m/s), and the matlab start box
  holds zs to +-OPT_e = 10 mm. The old claim "vi above ~62 m/s makes the start infeasible" is wrong: corrected Sturn
  at vi = 70 m/s converges (404 iterations, 20.18902 s) with zs(0) on the -10 mm box edge, so the car starts off its
  equilibrium. Under corrected, two local frontier solves on Sturn (2 seeds, no global proof) put the highest coast
  rolling vx(0) at about 68.5 m/s. Their rows 1-22 stay in the vi = 70 rolling box. So rolling at vi = 70 m/s is
  infeasible as posed (see the start-state bullet). Not in the GUI / cfg.json yet. Skipped:
  per-motor ratings, gear as a setup field, regen, drag pitch moment. `Steering` / `CamberGain` are now `userOpts`
  kwargs (both physics); the CG_p table is ported, and `_linear_interp` extrapolates as `linear_interp.m`.
- **Start state (2026-10-18, pinned by `tests/test_start_state.py`, `tests/test_start_wiring.py`).**
  `userOpts(start_state='matlab' | 'rolling' | 'periodic')`, one kwarg, modes exclusive. `'matlab'` (default) = MATLAB
  boxes, NLP byte-identical (sha256 snapshot of 7 builds). `'rolling'`: start box centred on the model's straight-line
  coast equilibrium at vi (`vehModel.rolling_start(ctx, m, vx)` -> (X, U): Newton with lateral trim, delta seeded at
  1e-6 for Ackermann; not `ladder.quasi_static_states`). r(0) = 0 +- OPT_e, vy and rows 5-22 from the equilibrium, rows 9-22
  within +-`ROLL_TOL` = 1e-3 (`ctx.Xi_tol`), vx box vi +- OPT_e*100 kept. Warns on a curved start. `optimise_design`
  refuses it (ValueError: the equilibrium depends on the design). `'periodic'`: rows X(N) = X(0), U(N) = U(0) on all
  states and inputs, last in g (`n_periodic` in the structure record), start fully free. Closed circuits only
  (`userOpts.CLOSED_CIRCUITS`: BCN, Jarama, Spa, NBR, Circle); Spa and NBR get a 5 m closing sample. Stems
  `_startRolling` / `_startPeriodic` (after `_physCorrected`, before `_mesh`; init and `_qss` files too),
  `data['start_state']`, `data['Xi']`. A seed from another start state gives full-interp (no duals). Caveats: the
  synthetic Circle k turns 0.988 laps (periodic Circle = wiring check only); periodic BCN runs on the parity k, which
  does not close (-1.023 laps). Measured 2026-10-18 (Linux, MUMPS; lap changes vs the stored 18.01633 / 116.44407 s):
  default cold Sturn 152 iterations, 18.01857 s (Phase 1 to every digit); rolling Sturn seeded from the stored default
  (full-interp) 95, 18.08490 s (+0.069 s, first order +0.045); rolling BCN seeded 118, 116.72927 s (+0.285, first
  order +0.197); periodic Circle cold 65, 21.71527 s, periodic residual 0; periodic BCN seeded 74, 116.91133 s (+0.467,
  forecast ~+0.3). Rolling start rows end on their box edges (rows 9-22 at +-1e-3, Om at +-2.8 rad/s).
  **B1 cause and fix (2026-10-19, pinned by `tests/test_start_wiring.py` section 12).** Corrected Sturn rolling at
  vi = 70 m/s is infeasible as posed, not a solver or transcription bug. Two local frontier solves (max vx(0), rows
  1-22 in the rolling box, 2 seeds, no global proof) put vx(0) at 68.50 / 68.53 m/s. The vx box edge is
  69 m/s. With only the vx(0) box widened to [59, 71], the cold solve converges (348 iterations, vx(0) 65.475
  inside). The corrected matlab-start solve at vi = 70 converges only pre-braked, off the equilibrium (rear wheels
  locked, wu on the bounds). Rolling now warns ("braking limit") when the vx box lower edge `vi - Xi_tol[0]*x_s[0]`
  (vi - 1 m/s) is above the QSS limit `qss_profile(ctx)["v_bwd"][0]`. Sturn centreline limit: corrected 66.87 m/s,
  matlab 70.79 m/s. The solve still runs. The QSS limit is ~1.6 m/s below the frontier, so the warning can fire for
  a feasible start. Corrected Sturn vi 67.9-69.5 m/s warns (box edge <= 68.5), a feasible point is likely,
  untested. At the default vi = 60 the warning also fires on other tracks (fake build, not solved). Margin = box
  edge 59 m/s minus the QSS limit (matlab / corrected). Jarama +0.6 / +8.5 m/s, BCN_S2 +4.9 / +9.0,
  Circle +11.1 / +17.5 (curved start). Corrected only: Hairpin +1.7, ZigZag and ZigZagMirror +1.8. Silent: Sturn, Spa, NBR, BCN.
  Only Sturn has a slack measurement (1.6 m/s). The old cold fragility
  (530 iterations, 18.61105 s) came from the seed. Its quasi-static rows 9-22 (rows 9-18 = 0, zt = W0/kt, no aero)
  miss the pinned start. Mechanism not isolated (likely the slow matlab chassis modes, tau 8-35 s). `_init_guesses` now
  seeds rows 9-22 at `Xi / x_s` at every knot and collocation point (7-state init, init7 warm start, qss23). Matlab and
  periodic seeds unchanged. Measured 2026-10-19 (Linux, MUMPS, cold): rolling Sturn 149 iterations, 18.06169 s
  (+0.045 s vs stored, first order +0.045), 13 s wall (was 93 s). That is 0.023 s below the seeded rolling Sturn.
  Corrected vi = 60 rolling: 220 iterations, 20.03744 s (was 270, 20.02972 s: lap +0.008 s, another local optimum).
  Default cold Sturn and seeded rolling
  Sturn unchanged to every digit. Not built: the rows 9-22 override in `warmstart_refined` (refine reseed), add when
  a rolling run uses refine. No solve covers qss23 + rolling. Rejected: a knot-0 clip of the seed (no gain, moves the
  default seed bytes).
- **Track width (2026-10-18, pinned by `tests/test_track_width.py`).** `userOpts(n_bounds=W | (lo, hi))`: bounds on
  n (CG path, m), scalars or arrays on `ctx.track.s` (after the periodic closing sample), applied at knots and
  collocation points only. None = model default +-4 m (`n_bounds=4.0` builds the same NLP). `lo < hi` everywhere:
  equal bounds fix an interval's n polynomial, so its continuity row turns constant. Fixed line for correlation =
  narrow band `(n_ref - b, n_ref + b)`, continuity-feasible when b >= max mismatch / 4.67 (Legendre d = 3).
  ValueError for a band wider than 1/|k| at any point (sf changes sign) and, inside `build_and_solve_nlp`, for an
  empty start / end box (default calls too: |ni| > 4.05 m, or vi more than 1 m/s above Vmax). Stems `_hw<W>`
  (`f'{W:g}'`, 6 significant digits) / `_nbCustom`, `data['n_tag']`; `reconstruct_track` draws the edges at the band.
  No real width data in the repo yet. Stored prices (`limit_prices`): W -0.2025 s/m Sturn, -1.8611 s/m BCN. Price
  check 2026-10-18 (Linux, MUMPS, full+duals from the stored default): Sturn W 4.2 m: 6 iterations, 17.97682 s, dJ
  -0.04014 s vs first order -0.04050 (0.9%), second order -0.03993 (0.5%); BCN W 4.1 m: 17 iterations, 116.26122 s,
  dJ -0.18407 s vs -0.18612 (1.1%), -0.18401 (0.03%).
- **Output channels (E1, 2026-10-18, pinned by `tests/test_output_channels.py`).** `data.vehicle` `Lon_acc` /
  `Lat_acc` = body-velocity derivatives (dvx/dt, dvy/dt), not accelerometer values. Accelerometer equivalents
  `ax = Lon_acc - vy*r`, `ay = Lat_acc + vx*r` (`m.ax`, `m.ay`, both physics, not in `dx`); stored BCN: RMS(vx*r)
  10.6 m/s^2, peak |ay| 1.45 g. Also `delta` (physical steer angle, rad, 35 deg * delta_n; the `u_opt` steering row
  keeps MATLAB's pi/8 scale) and `Om_fl..Om_rr` (rad/s, as `MLTP.m:584-587`). `E_motor` stays Wh in data (`P_motor`
  W); the GUI and plotSDI show kWh (MATLAB stores kW, `MLTP.m:592`, so its energy is kWh). g-g plots read
  `plotSDI.accel_channels` (ax / ay derived from `x_opt` for results saved before E1). EM4 energy is valid only with
  `physics='corrected'`: matlab EM4 `Om_motor = Om/gear`, so P and E come out gear^2 = 52.6x low.
- **23-state NLP constraint set matches `MLTP.m` (fixed 2026-10-04, pinned by `test_mltp_constraints.py`).**
  Two port gaps were closed. (1) The input-rate bounds are divided by `u_s` (`m.duk_lb/ub`, as `vehModel.m`
  L377-379 and the 7-state model do). Before, the physical Nm/s, rad/s and deg/s values bounded the normalised
  rates, which held steering to 0.061 rad/s and left the motor, brake and wing rates 602x, 4000x and 10-30x
  looser than specified. Steering scale (owner call 2026-10-07): `vehModel.py` uses `u_s = pi/8` for the
  steering input exactly as `vehModel.m` runs it (its `delta_s` leaks from `vehModel_initial.m`), while both
  keep `delta = 35 deg * delta_n`. So the rate bound is 0.1/(pi/8) per second on `delta_n`, 0.1556 rad/s
  physical, and the steering initial guess, the `u_opt` steering row and the plotSDI trace carry MATLAB's
  pi/8 scale (9/14 of the physical angle; the physical angle is `data.vehicle['delta']` since E1). The default baselines
  were re-measured with it on 2026-10-08 (end of this bullet); every other lap, iteration count and wall
  time in this file was measured with the earlier 0.1 rad/s bound (`delta_max` scale): pre-pi/8.
  (2) The four friction-circle rows
  exist only with `TyreModel='PureSlip'`, as in `MLTP.m`. The default Sturn NLP now has MATLAB's size
  (n_w = 1812, n_g = 1904). Default solves after the fix (ma57, MF205): Sturn 179 iterations, 18.00859 s lap
  (before: 489, 18.00934 s; MATLAB 18.008 s); BCN (`'auto'` = curvature, N=155) 247 iterations, 116.441 s,
  253 s of solve (before: 614, 116.523 s, 566 s), with the 0.1 rad/s steering bound active; BCN AALB 241
  iterations, 116.065 s. The other iteration counts, laps and wall times in this file (the
  five-coefficient proxy runs, the screening-preset pairs, the warm-start timings) predate the fix:
  re-measure before quoting them. The default `Results/` baselines were regenerated after the fix
  (Sturn, BCN, BCN AALB, the inits, QSS, TyreOptim; `Sturn_paramOptim` again from the full-solution warm
  start); the full results among them seed a default solve with duals (same NLP structure). The
  `*_CopyB.mat` files are legacy pre-fix results (no `data.nlp`). Regenerated again on 2026-10-08 for the
  pi/8 steering scale (ma57, MF205, default calls): Sturn 174 iterations, 18.01633 s; BCN 1072 + 165,
  116.44407 s; BCN AALB 169, 116.05468 s; `Sturn_paramOptim` 5 from the Sturn result (full+duals),
  18.01610 s; `Sturn_TyreOptim` then 172, 15.75002 s, Fz0_shift 0.5 (bound), without the Fz0 coupling (C6
  port gap). Regenerated 2026-10-18 with the coupling (Linux, MUMPS): `MLTP_TyreOptim(circuit="Sturn",
  warm_start="Results/Sturn_Static_ATDOn_EM4Off.mat", linear_solver="mumps")` (full+duals at P0 = 1):
  51 iterations, 17.49913 s, Fz0_shift 0.787 (interior). A reseed at Fz0_shift 0.53 ends on a worse optimum
  (21.832 s, Fz0_shift 0.622, 2798 iterations). The inits and QSS files are unaffected (content identical, kept).
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
