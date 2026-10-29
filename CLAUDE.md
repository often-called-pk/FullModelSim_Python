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
files (55 today) are plain scripts whose assertions run at module top level (no
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
- `test_track_qa.py`: `functions/track_qa.py` verdicts (`check_track` turning, closure, sign, spacing; `k_from_xy` on circles, an open arc, a stadium), load gate (error -> ValueError, else one warning), rebuilt Jarama + BCN sectors, BCN k untouched (one warning), BCNAssetto out of the registry, QSS lap pins on the rebuilt tracks. Sections 12-15: the D1 builder. `xy_from_gps` axes, GPS circles turn +-1. Repeated GPS points and an overlap give one lap. A gap and a figure of eight raise before any write. BCN rebuilt from its own x,y. Noisy GPS circles with `smooth`
- `test_output_channels.py`: E1: `m.ax = Lon_acc - vy*r`, `m.ay = Lat_acc + vx*r` (both physics, EM4 on/off), `data.vehicle` += `ax, ay, delta, Om_fl..Om_rr`, energy stored in Wh and shown in kWh, `plotSDI.accel_channels` (stored, derived for a pre-E1 result, or (None, None)), stored BCN RMS(vx*r)
- `test_start_state.py`: `userOpts(start_state=...)` boxes (matlab unchanged, rolling `Xi_tol`, periodic closed circuits only + closing sample), start stems, `n_periodic` in the structure record and the warm-start planning, periodic rows + start box in a fake-solver build
- `test_start_wiring.py`: `vehModel.rolling_start` (60 m/s coast equilibrium, variants, bounds, failures); start_state through `MLTP`, `MLTP_initial`, `optimise_design`, the refine passes, `setup_sweep._precheck_base`, the `MLTP_screen` stem; real builds with a fake solver (saved records, warm-start modes, plotSDI folder). Section 12: braking-limit warning (text values, silent at vi 67.5), rolling seed rows 9-22 at the equilibrium, rows 0-8 and inputs = matlab-start seed
- `test_track_width.py`: `userOpts(n_bounds=...)` forms + errors, `n_tag` stems, bounds at knots + collocation points (23- and 7-state), `n_bounds=4.0` == default, empty-start-box and corner-radius ValueErrors, `reconstruct_track` edges, full+duals warm start with a band, narrow-band continuity, periodic Spa arrays, `MLTP_screen` stem
- `test_sensitivity.py`: `functions/sensitivity.py`: `nlp_index` rows of w and g, `limit_prices` (W start-row rule, torque, brake, power, start speed, NaN under regen), `lateral_prices`, `costates`, prices of the stored results, KKT check on the rebuilt Sturn NLP, oracle (rebuilt Lagrangian differences == prices). Section 8: `setup_gradients`, no solve. brkB = Tdist = 0.0 exact with ATD on, wing signs, central difference of the rebuilt Lagrangian (rel 1e-6). ValueError for derived fields, `Tbrake_max`, mismatched options and a missing `data.nlp`
- `test_tyreoptim_fz0.py`: C6: TyreOptim model at `Fz0_shift = S` == plain model with `Fz0 = 4905*S` (both physics), the coupling at S = 0.5, S = 1 exact, no coupling in `MLTP_paramOptim`
- `test_mf_scaling.py`: C6: the five MF scaling factors `LMUX`, `LMUY`, `LFZO`, `LKX`, `LKY` (defaults 1.0, both tyre sets), overrides, unknown key raises. Full model at LMUX / LMUY 0.9 / 0.8 (both physics, camber on, no solve): `mu_*`, the peak forces and the vertical shifts scale. The zero-slip slopes do not change (pHy1 = pHy2 = 0). With camber off and pVy1 = pVy2 = 0, the fy zero crossing (alpha = -SHy) does not move, and Svyk scales. Section 2b: LFZO 0.9 == the override `Fz0 = 0.9*4905` (mu, fx, fy), LKX / LKY 1.2 scale the zero-slip slope by 1.2 and move no peak. QSS peaks and lap (1.0 = default lap, LFZO lap == Fz0 override lap, LKX / LKY leave it). 7-state build unchanged (all five). Parity export drops all five only while all are 1.0. Sections 2-5 need casadi + `Data/DATA_AA.mat`
- `test_aero_scale.py`: E3 `Cd_scale` / `Cl_scale`: defaults, registry, `Cd` / `Cl` placeholders inert with DATA_AA. f_drag and the five lifts scale, the side force does not (both physics). QSS == vehModel, parity export joint filter. Section 5, A4: the corrected drag pitch moment (closed form at 20 / 60 m/s, none under matlab, `Cd_scale` on it). QSS axle shift == coast equilibrium, QSS lap moves < 0.01%. Casadi + `Data/DATA_AA.mat`, no solve
- `test_power_a5.py`: A5 `vp_overrides` keys `gear, Tmax, Pmax, Preg, Ppack`. Defaults change nothing. Keys refused under matlab and by value. Regen bounds, pack row last in [0, inf] (n_g + N + 1). QSS regen anchored to vehModel wheel torques, QSS pack cap. Energy motor by motor on a stand-in solve, NLP sizes with a fake IPOPT. Casadi + `Data/DATA_AA.mat`, no IPOPT
- `test_telemetry.py`: E1 stage M. `time_at_points` exact. `resample` keeps input jumps, ax / ay stay inside the node range. Monster file (11 tab columns, no header, distance = integral of speed) and CSV. Stored Sturn point, fake IPOPT: `vehicle[k] == vehicle_full[k][::4]` and `t_full[::4] == t_opt` bitwise. ax / ay at the nodes == state-polynomial kinematics (pins `nlp_inputs`). EM4 and corrected stubs, `.mat` round trip
- `test_provenance.py`: E2. `userOpts.setup_knobs` and `knob_tag`: the default setup is empty, a value equal to the default is no knob, and 45 == 45.0. The tyre set decides the defaults, and A5 keys, n_bounds, `load_model` and `ds_fine` count. Keys that never count, and every userOpts argument classified by rule. `result_stem` puts the tag last. `code_version` and `provenance_record`: savemat round trip, `ipopt_overrides` as JSON text, an option name over 31 characters. GUI tag == MLTP tag. Fake IPOPT: the names and fields of MLTP, MLTP_initial, MLTP_screen and optimise_design. A knob run leaves the default file untouched
- `test_continue_from.py`: B5. `warmstart.default_source` gates: missing, knobs, unconverged, legacy, acceptable. Argument errors before any build. Fake IPOPT: LMUY 0.95, n_bounds 4.2 and vi 65 with ni 0.5 give full+duals. `warm_start_duals=False` gives full-primal. Another start_state needs its own default. OPT_ds 45, PureSlip and a default without `data.tyre_set` raise with the reason, and so does a knob file at the default name. A hop that does not converge warns. `continue_from=None` reads nothing
- `test_between_knots.py`: B3. Values and errors, `nlp_structure` and the planner guard, `refine.nlp_inputs` / `_input_affine`, `nlp_index` pathcol rows, rpm cap gear*Vmax/Rw = 0.885 OMmax. Fake-solver Sturn: only the defect rows differ. The defect rows match a numpy f_dyn (1e-9), and the tail rows match (Pmax - T Om_motor)/Pmax (1e-12). EM4, regen, pack and periodic rows. Warm-start modes, saved fields, Pmax NaN and refusals. The refine reseed and indicator follow the flag (sections 6-9 need casadi + `Data/DATA_AA.mat`)
- `test_radau.py`: B4. Radau points, D, B and stability. `discretise(scheme='radau')` gives exact duplicates, including a 1-ulp case. `reconstruct_track` gives one answer per station, and `guesses_from_full` takes a Radau source. The telemetry nodes follow `data.collocation`. userOpts errors, and the builder ValueError for collocation path rows without linear inputs. Fake-solver Sturn: sizes equal Legendre, N zero steps, end-point input, continuity nnz 2 vs 5. Warm starts Radau -> Radau full+duals, Radau -> Legendre full-interp. Refine keeps the scheme
- `test_mesh_report.py`: B6. `mesh.mesh_report` text. The speed term: speed 0 bit for bit, ValueError without v, more knots in braking, a constant speed drops it. The userOpts `track.v_qss` hook. `vehModel.vertical_eigs` vs AD (both physics). MLTP prints 3 lines and survives a failing report (fake IPOPT)
- `test_state_scaling.py`: B7. Boxes for default, typical, rolling and periodic (physical width kept), and ValueError. The planner refuses duals across scalings. build_only equivalence: f equal, g' = R g, J' = R J C, bounds. The IPOPT options of 'typical' equal the default ones (the tol bug of 6451820). Fake-IPOPT records and `.mat` round trip
- `test_atd_tbr.py`: C4. Defaults, k and ValueErrors. Row names, bounds and values on hand splits, and the wheel-torque identity. Sizes 1904 / 1942 / 1980. The planner compares path-row names (PureSlip 8 rows vs TBR 8 rows: full-interp). QSS lap unchanged, fake-IPOPT records
- `test_arb.py`: C3. Registry and defaults, and a plain 0 = default graph. Force sign and shape (both physics), roll stiffness -k t^2 / I_x. Steady transfer vs QSS, and the QSS lap. `classify_fields`, RunConfig and the parity export. The zero-solve gradient on the stored Sturn optimum
- `test_tir.py`: C5. Reader: sections, comments, quotes, D exponent, CRLF. `tyre_set='file:'` round trip of MF205 (graph hash equal), FNOMIN, overrides win. Errors and warnings, and no `.tir` tracked by git. `mf_pressure` floats and the MF 6.1 factors at +10%. Full-model sites: SVx / SVy unmoved, camber stiffness, INFLPRES as a symbol. QSS lap at +10% == lap at the mu factors. Two same-name files get two tags. Parity export
- `test_telemetry_logs.py`: E5. `read_monster` round trip and sign pair. `read_csv`: map forms, JSON map, units-row check, metadata rows, beacons, streaming peak. `split_laps`. `project` on BCN / Sturn: |s err| < 0.3 m, |n err| < 0.03 m, centreline guard. `time_delta`. `overlay_report`: lap pick, panels, sign warning. Synthetic data, no solve, no `Data/DATA_AA.mat`
- `test_correlate.py`: E6-E8. `gamma`, and `identifiable` (it refuses LFZO and Fz0_shift). `grip_fit` on QSS logs: LMUY within 0.01, LMUX within 0.03 noise free. `check_ax` and NaN dropouts. `drag_fit`: exact with ax, torque-scale error. `lift_fit`: matlab refused, corrected equilibria. `run_stage` + ledger: frozen stages, stale marks. Import without casadi. `nlp_forecast` vs V9
- App/GUI group (`test_headless_config`, `test_runconfig`, `test_vp_params`, `test_presets`, `test_paths`, `test_results`, `test_solve_runner`, `test_spec_includes`, `test_mainwindow`, `test_gui_logic`): cfg.json forwarding, `RunConfig`, vp registry, presets, paths, results parsing, solve dispatch, PyInstaller-spec lint, offscreen Qt window (PySide6). F2: `FORWARD` sends a key only when set and not default, and `mode='qss'` drops the NLP-only keys and runs `MLTP_screen`. `RunConfig` folds `pt_overrides` into `vp_overrides`, and `setup_tag()` == the MLTP tag. `result_paths`, `freeze_support` first in `__main__`, the Advanced-tab combos and the More options box. 4 Motors ticks Homotopy

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
   problem, saving `Results/<circuit>_<config>.mat`. A non-default setup adds stem parts and a setup tag (see the
   result-names bullet). If `warm_start=None`, `MLTP()` calls
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
arc length (primal only). `warmstart.WARM_KNOBS` (`start_state`, `between_knots`, `collocation`, `state_scaling`, absent =
default) also refuse the duals when they differ (full-interp). So do the path-row names (`hnames` against the keys of
`data.constraints`). The mode (`cold | init7 | full+duals | full-primal | full-interp`) is
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

**`MLTP(continue_from='auto')` (B5, 2026-10-28, pinned by `tests/test_continue_from.py`).** It seeds one off-default solve from the
stored DEFAULT result `results_dir/<stem>.mat`, with primal and duals (`warm_start_duals=False`: primal only). The stem is the one of
the call without the n_bounds band and without the setup tag. A tyre file keeps the tag of its knob `tyre_file`. Strict, no
fallback: ValueError before the solve. It raises for another value, or with `warm_start` or `homotopy`. It raises for a missing
default, or for knobs in `data.provenance.setup` other than `tyre_file`. It raises for an IPOPT status outside `GOOD_STATUS`. It
raises for a source that does not fit: other N, mesh grid, input channels, path rows, tyre set or a `WARM_KNOBS` value. A default
saved before E2 (no provenance) is trusted by its name. A hop that does not converge warns. The result saves under its own tag,
and `data.provenance.run.source` names the default file. Not checked: the linear solver, the IPOPT options and the code version
of the source. Measured (Linux, MUMPS): LMUX = LMUY = 0.95 takes 20 iterations to 18.24010 s (= V9), 4.1 s wall. `mb` +3% takes
15 iterations to 18.16919 s, 3.9 s wall. Its cold solve takes 500 iterations to 19.17591 s (a slow branch), 79.6 s wall.

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
  is the shared core (`param_specs` = list of `(vp_field, lower, upper)`). Since 2026-10-21 a name can also be an
  MF key of `ctx.mf`, e.g. `LMUY`, `LKY`. Default params: `brkB, Tdist, alpha_FL/FR/RW/TW`.
  `optimise_design(build_only=True)` (only `setup_gradients` passes it) returns the dict of
  `build_and_solve_nlp(build_only=True)`: nlp, w0, bounds, accepted lam_g0 / lam_x0, no solver. It raises
  ValueError for `warm_start=None`, or for a full result whose warm-start mode is not `full+duals`. A 7-state init
  gives the dict without lam_g0 / lam_x0. `optimise_design` refuses `between_knots` other than 'matlab' and takes
  `collocation`, `state_scaling` and `atd_tbr` as `MLTP` does. Its file gets the setup tag of its userOpts knobs, not of
  `param_specs` or `Fz0_shift_bounds`.
- **Setup gradients, zero solves (C2 stage M, 2026-10-21, `tests/test_sensitivity.py` section 8).**
  `functions/sensitivity.setup_gradients(data, params, tol=1e-3, **kw)` -> `{p: dJ*/dp}` [s per unit of p] of a
  stored full result. It promotes each p (vehParams primary or MF key) and rebuilds the NLP at the stored point
  with the stored multipliers. The gradient of `f + lam_g'g` in the P rows is dJ*/dp (envelope theorem). `kw` =
  the solve's options (circuit, vi, AeroConfig, `vp_overrides`, physics, tyre_set ...). ValueError for a derived
  field (`ms`, `gamma_fl_rad`) or a primary that vehParams derives others from (`gamma_fl`, `mb`, `k_fl`).
  ValueError for `Tbrake_max`: it sets the scale `u_s` of `T_brake` (`limit_prices` gives its price). Also
  ValueError for a result that does not fit, or a point that is not an optimum of the rebuilt NLP. First order,
  and J includes the regularisation. Sturn 0.11 s, BCN 1.01 s. Sturn values: alpha_FL -4.60e-4,
  alpha_FR -4.53e-4, alpha_RW +1.90e-4, alpha_TW -7.01e-5, LMUY -3.29, LKY -0.640. brkB = Tdist = 0.0 exact (ATD
  on). 27 params agree with central differences of the rebuilt NLP to 1.8e-6. Check 2026-10-21 (Linux, MUMPS,
  full+duals from the stored Sturn result). LKY 0.9: 6 iterations, 18.09017 s (+0.07384 s), objective +0.07146 s vs first
  order +0.06404 s. LKY 0.95: 3 iterations, objective +0.03379 s vs first order +0.03202 s. The first-order gap
  halves with the step (10.4% -> 5.2%): a second-order term, not a gradient error. `setup_sweep` still promotes no
  MF key (`promotion_safe` needs a primary). A `kw` with `between_knots` other than 'matlab' raises ValueError. A Radau
  result needs `collocation='radau'` in `kw`.
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
NLPs on the default MF205 set. Knobs the screen reads: `LMUX`, `LMUY`, `LFZO`, `Cd_scale`, `Cl_scale`, `k_arb_f` and
`k_arb_r` (roll split, C3). It also reads the MF 6.1 mu pressure factors (`INFLPRES` with `ppx3`, `ppx4`, `ppy3`, `ppy4`,
C5). Under `physics='corrected'` it reads the A4 drag axle shift and the A5 `Ppack` cap / `Preg` regen (`Envelope.F_brk`).
`LKX` / `LKY` and the stiffness pressure terms have no QSS site (peak friction only), so the screen cannot rank them.
The QSS lap ignores the collocation scheme, but `collocation='radau'` needs casadi for the Radau points of the `s_full` grid.

### The transcription engine: `functions/transcription.py`
Shared by all MLTP scripts. `discretise(track, OPT_ds, OPT_d, mesh='uniform', mesh_opts=None,
s_knot=None, scheme='legendre')` builds the collocation grid, Legendre or Radau IIA (B4) (defaults from `userOpts.py`: step
`OPT_ds=30` m, degree `OPT_d=3`, `OPT_uinter='linear'`; knot placement: see Collocation mesh);
`build_and_solve_nlp(..., warm=None, sym_type=None, linear_inputs=False, h_col_rows=None)` (B3) assembles the CasADi NLP,
decision vector
`w = [Xk; Uk; (Yk); Xkj (; P)]` packed **column-major (`order='F'`)**, collocation defect +
endpoint continuity, path constraints, time-domain input-rate limits, `Xi/Xf` boundary bounds
(**`NaN` = free state**), objective `J = Σ Qk·B·dsk` + regularisation, then calls
`nlpsol('ipopt')`. Besides `sol` it returns `w_opt`, `lam_g`, `lam_x`, `structure` (the sizes a
later solve must match to re-inject them) and `warm_info`. `_make_solver()` **selects the
configured `linear_solver`** (HSL `ma*` if a Coin-HSL DLL loads, else `mumps`; see Using
Coin-HSL). After the solve: `unpack_solution`, `reconstruct_x_full`, `interp_inputs`,
`compute_time`, `time_at_points` (time at every knot and collocation point: Lagrange quadrature of sf,
`t_full[::d+1] == t_opt` exactly), `reconstruct_track`. `build_only=True` returns the NLP, w0 and bounds before
the solver exists (no HSL probe, used by `setup_gradients`).

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
`a, b, ds_min, ds_max, smooth_window, N, pct, grid_ds, speed`. By default a = b = 1 and speed = 0 (B6 paragraph). `'auto'` only
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

**Mesh report and speed term (B6, 2026-10-28, pinned by `tests/test_mesh_report.py`).** After the summary, `MLTP` prints
three `[MLTP]` lines (`functions.mesh.mesh_report`). They give the interval durations, the vertical modes and the count of
intervals that last at most half the slowest period. `vehModel.vertical_eigs` takes the modes from central differences of
`f_dyn` at the static coast point (states 9-22, about 1 ms). A failure prints one skip line and never raises. Matlab
physics: 3.38, 3.90 and 4.55 Hz, plus 4 decaying modes (77 us to 37 us). Corrected: 1.90, 3.03 and 5.21 Hz. On Sturn 0 of
18 and on BCN 0 of 155 intervals resolve a mode. `mesh_opts={'speed': w}` adds w |dv/ds| / dv_ref of the QSS speed to the
curvature monitor. The default 0 is off (knots bit for bit). userOpts sets the speed `track.v_qss` from `qss_profile`.
N stays. The knots follow the setup, so `setup_sweep` refuses the term, and a homotopy step or a setup chain with it
seeds the primal only. BCN at w = 1 (cold, MUMPS): 218 iterations, 116.35533 s (-0.076%), so the keep rule held (lap <=
116.385 s, iterations <= 248). At vi = 60.001 m/s: 156 iterations, 116.39807 s, against 277 iterations, 116.44196 s on the
default mesh. The lap on the speed mesh moves 0.043 s between the two starts (default mesh 0.002 s). Its 7-state init at
vi = 60.000 m/s ended with `Error_In_Step_Computation`, and the 23-state solve still converged.

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
`refine` reaches `MLTP` from cfg.json too (F2). The GUI has no widget for it, because the Output tab does not look for
`_meshAdaptive` files.

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
Track builder (D1 stage M, 2026-10-21): `track_qa.write_track(path, x, y, closed=True, smooth=0.0, **kw)` returns
(metrics, issues). It drops repeated GPS points (< 1 mm) and trims an overshoot past the start. Then it runs
`k_from_xy` + `check_track` and writes s, k, x, y. ValueError and no file for a start-end gap (> 3 x the longest
chord) or any QA error. The same for a closed line that does not turn +-1 laps (figure of eight, two laps).
`xy_from_gps(lat, lon)`: WGS84 tangent plane at the mean position, east +x, north +y (ponytail ceiling: real
projection beyond ~5 km). `smooth_xy`: cubic Savitzky-Golay over `smooth` m. GPS noise above ~0.05 m needs it (noisy
circle, 0.05 m, 40 m: median kR 0.9986). No circuit is registered: add the file to `_REAL_CIRCUITS` and
`app/mainwindow.CIRCUITS` by hand. `write_track` overwrites `path`: never pass a pinned file
(`Circuits/Barcelona_circuit.mat`). BCN rebuilt from its own x,y: turning -1.000000000000 laps, no issue, end gap
0.57 m. Its QSS lap is 134.768 s, +4.1% vs the parity file. Smoothing is a fit knob, so BCN keeps its file.
Other kwargs: `vp_overrides` (vehParams primaries + Pacejka coefficients), `tyre_set` (default
`'MF205'`, `'CopyB'` is the legacy set, `'file:<path>'` reads a .tir, see the tyre bullets), `OPT_ds` / `OPT_d` / `OPT_e`, `mesh`
(default `'auto'`, see Collocation mesh) / `mesh_opts`, `max_iter` / `tol`, `linear_solver` /
`hsl_dir`, `cse` / `jit` (opt-in), `screening`, `physics` (`'matlab'` default | `'corrected'`, see the physics
bullet), `Steering` (`'NA'` | `'Ackermann'`) / `CamberGain` (`'Off'` | `'Linear'` | `'Table'`), `start_state` (`'matlab'`
default | `'rolling'` | `'periodic'`, see the start-state bullet), `n_bounds` (None default | half-width W |
`(lo, hi)`, see the track-width bullet), `between_knots` (B3), `collocation` (B4), `state_scaling` (B7), `atd_tbr` (C4, see
their bullets), and `ipopt_overrides` (dict of any IPOPT options,
merged last so it beats every default, the screening preset and the warm-start recipe; a non-exact
`hessian_approximation` warns). Kept on `ctx.mesh` (resolved), `ctx.mesh_requested`,
`ctx.mesh_opts`, `ctx.tyre_set`, `ctx.screening`, `ctx.ipopt_overrides`, `ctx.cse`, `ctx.jit`, `ctx.physics`,
`ctx.Steering`, `ctx.CamberGain` (`vehModel(ctx)` reads the last two when its arguments are None),
`ctx.start_state`, `ctx.periodic`, `ctx.Xi_tol`, `ctx.n_bounds`, `ctx.n_tag`, `ctx.between_knots`, `ctx.linear_inputs`,
`ctx.power_at_col`, `ctx.collocation`, `ctx.state_scaling`, `ctx.chassis_scale`, `ctx.atd_tbr` (and `pt.atd_k` when set), plus
`ctx.track.n_lo` / `n_hi` with a band and `ctx.track.v_qss` with a speed weight in `mesh_opts`. `MLTP` adds
`ctx.out_path` after a save.

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
  rebuild recipe, track builder `write_track` / `xy_from_gps` / `smooth_xy`, see Configuration).
- **Post-solve analysis:** `sensitivity.py` (numpy only, no solve): `limit_prices`, `lateral_prices`,
  `costates` from the stored multipliers of one full result; first order, and J includes the regularisation
  (0.27% of the Sturn lap, 0.12% of BCN). `setup_gradients` is the exception: it imports casadi and
  `MLTP_paramOptim` and rebuilds the NLP (no solve, see Co-optimization wrappers). `telemetry.py` uses numpy and scipy
  (plotly only inside `overlay_report`, casadi for a Radau result). It has `resample(data, by='distance' | 'time', step)`,
  `export_monster` and `export_csv` (see the output-channels bullet). E5 log tools (2026-10-28, pinned by
  `tests/test_telemetry_logs.py`): `read_monster(path, step=0.02, sign=(-1, -1))` (whitespace columns, t = row * step, the
  sign pair of `export_monster`). `read_csv(path, channel_map=None, beacons='Beacon Markers', chunk=1 << 14)` streams in
  chunks and keeps the mapped columns only. Its map is `{channel: column}`, `{channel: [column, unit, factor]}` or a JSON
  file of either. A units row must show the unit of the map. The beacon row gives the channel `lap`. Then
  `split_laps(t, dist, beacon, gap, drop)` and `project(log, track)`: s and n on the centreline, n left positive, no shift
  or rotation estimate. `project` warns when the median |n| of the moving samples is above 8 m. Last, `time_delta(sim, lap)`
  and `overlay_report(data, log, path, lap=None, step=1.0)`: 5 panels on s, and a warning when the `corr` of ay or delta is
  below 0. No AiM channel map ships (signs unverified). `correlate.py` (E6 to E8, 2026-10-28, pinned by
  `tests/test_correlate.py`) uses numpy and scipy, and casadi only in `nlp_forecast`. `STAGES` = grip, drag, downforce.
  `run_stage(stage, log, circuit, ledger=None, link=True, lap=None, tags=None, **useropts_kwargs)` fits one stage with the
  QSS. It appends a record to a JSON ledger. One ledger holds one car (circuit, physics, tyre_set), and a re-run marks the
  later stages stale. `grip_fit` (E7a) fits LMUY and LMUX from ax and ay, and `grip_from_lap` sets LMUX = LMUY from a lap
  ratio. `drag_fit` (E8a) fits Cd_scale and f from a power balance on straights. `lift_fit` (E8b, physics='corrected'
  only) fits Cl_scale from the damper travel. `identifiable` is the gate: `gamma` <= 15, and no name of `NEVER` (LFZO,
  Fz0_shift, LKX, LKY). `check_ax` raises ValueError when ax does not follow d vx/dt. `nlp_forecast` gives the first-order
  NLP change via `setup_gradients`. The fits give effective QSS values and apply to the QSS only. On NLP data, VE7 (Sturn)
  finds an LMUY step of -0.10 within 0.01, but an LMUX step of -0.10 as -0.027. VE8 (BCN): ax at the collocation points
  rings, so `drag_fit` on a straight fails. A whole-lap fit without ax finds +0.315 for a +0.30 step.
- **Post-processing only** (reached via `reconstruct_track` *after* the solve): `curv2cart.py`
  (s,k -> cartesian centreline), `cartPath.py` (lateral offset `n` -> racing line), `trackLimits.py`
  (boundary polylines), `rotatePoint2D.py` (used only inside `curv2cart`).
- **`plotSDI.py`** writes Plotly HTML to `Plots/<circuit>/<config>/`; called from `MLTP()` when `plot=True`.

### Windows app (`app/`, `headless_solve.py`, `build/`)
PySide6 GUI (`python -m app.main`) around the same solve: `MainWindow` collects a `RunConfig`
(`app/runconfig.py`), writes a `cfg.json` and runs `headless_solve.py cfg.json` (frozen: `<exe> --headless
cfg.json`) as a subprocess. `headless_solve.build_solve_kwargs(cfg, resource_root)` maps cfg ->
`MLTP(**kwargs)`, or `MLTP_screen(**kwargs)` for the cfg key `"mode": "qss"`. F2 (2026-10-28): its `FORWARD` table sends a
key only when it is set and not default, so an old cfg makes the old call. Add a new option to `FORWARD` and add the matching
`RunConfig` field.
The Advanced tab has a Tyre set combo (`MF205` default, `CopyB`) and a Mesh combo (`auto` default,
`uniform`, `curvature`); `mesh_opts` and `screening` still have no widget. The Setup table's Pacejka
defaults, Reset and changed-value highlighting follow the Tyre set combo: on a switch, the
tyre-set-dependent (lateral) cells still at the old set's default are re-seeded with the new set's
values, while cells the user edited keep their value and are flagged against the new set. The
`brkB`/`Tdist` rows are greyed while ATD is on and 4 Motors is off (inert in the 23-state model; only
the 7-state warm start reads them). The tabs are built Main, Advanced, Setup, Output (the Setup table
seeds from the Tyre set combo) but shown Main, Setup, Output, Advanced. The Advanced-tab group "Run, model and start" has
the combos Run (`nlp` | `qss`), Physics and Start state. It also has a Homotopy check box and a "More options" JSON box.
The box takes the keys of `mainwindow.MORE_KEYS`: n_bounds, ladder, ipopt_overrides, Steering, CamberGain, pt_overrides,
between_knots, collocation, state_scaling, atd_tbr and continue_from. Choosing 4 Motors ticks Homotopy, and clearing 4 Motors
unticks it. The library default stays None. F2-V1 (BCN, 4 motors, ATD Off, homotopy, MUMPS): 938 + 234 + 98 + 59
iterations, 111.17993 s. `app.results.result_paths(rc)` builds the .mat path and the plot folder from the same stem as the
solve (physics, start state, band, `RunConfig.setup_tag()`, `_qss`). `app/main.py` calls `multiprocessing.freeze_support()`
first (not yet tested in a frozen exe). Packaging notes: `build/README.md`.

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
- **MF scaling factors (C6, 2026-10-20 and 2026-10-21, pinned by `tests/test_mf_scaling.py`).** `ctx.mf.LMUX`,
  `LMUY`, `LFZO`, `LKX` and `LKY` are the MF 5.2 factors lambda_mux, lambda_muy, lambda_Fz0, lambda_Kx and
  lambda_Ky, float 1.0 in both tyre sets. They are MF keys, so `vp_overrides={"LMUX": x, ...}` puts them in
  `ctx.mf`. `vehModel.py` places them as MF 5.2 does (PAC2002 manual, eqs 16, 20-23, 25, 28, 34-35, 37, 41, 74-75).
  LMUX scales mu_x and SVx. LMUY scales mu_y and SVy, camber part `Svy_gamma` included. LMUX and LMUY leave Kxk and
  Kya, so `Bx = Kxk/(Cx*Dx)` and `By = Kya/(Cy*Dy)` keep the slip stiffness. LFZO: `Fz0' = LFZO*Fz0` in dfz and
  Kya. So LFZO = a equals the override `Fz0 = a*4905` exactly in the full model and the QSS (QSS Sturn LFZO 0.9:
  18.89453 s, +0.80%). The 7-state init reads `vp.Fz0`, not LFZO. LKX multiplies Kxk, and LKY multiplies Kya. Bx,
  By and the camber part of SHy follow, and the peaks do not move. At 1.2 the zero-slip slope is x1.2 (rel error
  1.6e-10 / 2.8e-11). LVX and LVY are not added (= 1). The pHy part of SHy does not scale. Its camber part follows
  `Svy_gamma` (MF 6.1 form), so the camber thrust at small slip does not change. `_mf_scale` skips a plain 1.0, so
  the default SX graph stays byte-identical (NLP snapshot: none). A symbolic factor always multiplies. The QSS
  scales `peak_mu_x` / `peak_mu_y` and uses `Fz0*LFZO` (`functions/ggv.py`). It has no LKX / LKY site. The 7-state model
  (`vp.tyre`) ignores all five, so its init seeds a cold solve at the default tyre. `validation/py_export.params`
  drops all five (`MF_SCALES`) while all five are 1.0 (no MATLAB counterpart). Any other value of one exports all
  five, and gate 1 flags them. `Fz0_shift` and the TyreOptim Fz0
  coupling do not change. GUI: five Setup rows in "Tyre & Wheel". `data.mf_overrides` keeps the key names only, and a
  factor run saves under its own setup tag (E2), with the values in `data.provenance.setup`. Sanity run 2026-10-20
  (Linux, MUMPS): Sturn at LMUX = LMUY = 0.95, full+duals from the
  stored Sturn result. It took 20 iterations (Solve_Succeeded) to 18.24010 s, +0.22377 s (+1.24%) against
  18.01633 s, in 3.5 s wall (2.2 s NLP build). The QSS gives 18.74393 s at 1.0 (= default) and 18.98036 s at 0.95
  (+1.26%). Track method for the values (candidate verdicts, no human review): `docs/tyre_scaling_track_method.md`.
  Its Table 5 links LMUX and LMUY to one value, set from data in the QSS only, after four gates. The full model
  keeps 1 until study B2 gives the full-model elasticity. LKX, LKY, LFZO and the Cd / Cl scales exist as knobs at 1
  (2026-10-21). Per the report, lap times, tyre temperature and pressure cannot identify them: keep them at 1
  without the data. Report Table 5 gives the data that each one needs. LKX needs steer, motor speeds, IMU data
  and hard runs. LKY needs steer, wheel speeds, IMU data and a side-slip sensor. LFZO needs a source for a temperature or pressure effect. Never fit LFZO from lap
  times. For the Cd and Cl scales, coast-down runs (Cd) and ride-height or GPS data (Cl) give leads only.
  Skipped: a QSS inversion (stint median lap -> grip scale), add when session data exist. `optimise_design`
  promotes MF keys since 2026-10-21 (see Co-optimization wrappers). No real co-optimisation solve of one yet.
- **Aero scales (E3, 2026-10-21, pinned by `tests/test_aero_scale.py`).** Primaries `vp.Cd_scale`, `vp.Cl_scale`,
  float 1.0, GUI rows in "Aero & Environment". `Cd = (Cd0 + RW_Cd)*Cd_scale`, and all four of `Cl_front`,
  `Cl_rear`, `Cl_left`, `Cl_right` `*= Cl_scale` (front / rear alone cancel in the corner split). Full model and QSS
  (`ggv.aero_coefficients`, so the envelope, `c_res2` and ladder `m23` follow), both physics. The 7-state model
  ignores them. `_mf_scale` skips a plain 1.0 (NLP snapshot: none, QSS default lap bit-identical). Parity export
  drops both while both are 1.0, exports both otherwise. At 1.2 / 0.85: f_drag x1.2, f_lift and the 4 corner lifts
  x0.85 (rel < 1e-12), side force unchanged. QSS Sturn (18.74393 s): Cd_scale 1.2 +0.0084 s, Cl_scale 1.2 -0.0041 s.
  `vp.Cd` / `vp.Cl` stay MATLAB placeholders (GUI label "(placeholder)"): DATA_AA.mat overwrites them, so an override
  acts only without that file, and then only in the 7-state model.
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
  gravity or Euler terms on a flat road; A4 downforce through the springs plus aero pitch/roll moments and the
  rear-wing drag pitch moment; A5 EM4 `Om_motor = Om*gear` and the powertrain keys; A6 toe indices. All as python
  `if` branches. The QSS envelope (`_load_basis(..., physics)`, EM4 limits) and the ladder seeds (`quasi_static_states(vp, vx, physics)`) follow the flag. Results get the
  `_physCorrected` stem and `data.physics`. Corrected vertical modes: heave 1.9 Hz zeta 0.53, roll 3.0 Hz zeta 0.57,
  pitch 5.2 Hz zeta 0.35, max|eig| 191 1/s (matlab 2.7e4). Measured (Linux, MUMPS, cold): Sturn 18.019 -> 20.024 s
  (+11.1%, 152 -> 218 iterations), BCN 116.444 -> 130.035 s (+11.7%, 207 -> 179 iterations); QSS forecast +10.0% /
  +11.1%. These corrected laps predate the drag pitch moment. Re-measured 2026-10-21 (same setup): Sturn 20.02081 s,
  392 iterations. BCN 130.04167 s, 212 iterations (+0.0065 s vs 18 Oct; QSS forecast -0.012%; both too small to
  resolve the moment in a cold NLP).
  Watch: the body sits 9.4 mm low at 60 m/s under corrected (12.8 mm at 70 m/s), and the matlab start box
  holds zs to +-OPT_e = 10 mm. The old claim "vi above ~62 m/s makes the start infeasible" is wrong: corrected Sturn
  at vi = 70 m/s converges (404 iterations, 20.18902 s) with zs(0) on the -10 mm box edge, so the car starts off its
  equilibrium. Under corrected, two local frontier solves on Sturn (2 seeds, no global proof) put the highest coast
  rolling vx(0) at about 68.5 m/s. Their rows 1-22 stay in the vi = 70 rolling box. So rolling at vi = 70 m/s is
  infeasible as posed (see the start-state bullet). The GUI Physics combo and cfg.json reach it (F2). Drag pitch
  moment and the
  A5 powertrain keys: next two bullets. `Steering` / `CamberGain` are now `userOpts`
  kwargs (both physics); the CG_p table is ported, and `_linear_interp` extrapolates as `linear_interp.m`.
- **Drag pitch moment (A4, corrected only, 2026-10-21, pinned by `tests/test_aero_scale.py` section 5).** The
  rear-wing drag acts at `vp.hw` (1.28 m) and the body drag at `hcg` (0.5 m). This is the split of `vehModel.m` and
  `MLTP_paramOptim.m` (no new field). Corrected `dthetadot` gets `-q*RW_Cd*Cd_scale*(hw - hcg)`: a rearward force
  above the CG lifts the nose. The QSS moves `dFz = q*Cd_rw*(hw - hcg)/L` to the rear axle, in both load models
  (`aero_coefficients` returns `Cd_rw`). Coast pitch at 60 m/s 2.2569 -> 2.0886 mrad (`test_start_wiring` pin
  2.0-2.1 mrad). Axle shift 7.77 / 21.58 / 42.29 N at 30 / 50 / 70 m/s, QSS vs coast equilibrium within 3.5 mN.
  Corrected QSS lap Sturn 20.6231 -> 20.6225 s (-0.0029%), BCN 143.8824 -> 143.8648 s (-0.0122%). NLP size
  unchanged, only the corrected graph hashes change (by design). Not built: the side-force roll moment (the side
  force is 0 at the default `alpha_TW = 0`).
- **Powertrain keys (A5, corrected only, 2026-10-21, pinned by `tests/test_power_a5.py`).** `vp_overrides` takes
  `gear, Tmax, Pmax, Preg, Ppack` (`vehParams.PT_KEYS`). They are not primaries: no GUI rows (the More options box takes
  them as `pt_overrides`, F2), no sweep.
  ValueError under `physics='matlab'` or for a value not finite and > 0 (Preg may be 0). Set only when given: `gear`
  -> `vp.gear`, the rest -> `ctx.pt`. Tmax / Pmax = ratings of ONE motor. `Preg` (regen W per motor) sets the torque
  lower bound -Tmax, the `motor_power` ub `1 + Preg/Pmax` (P >= -Preg) and the `BrTh` ub inf. The energy then
  credits `eff*|P|` motor by motor (drive costs `P/eff`). `Ppack` (W, sum of motor powers) adds one path row, last:
  `(Ppack - sum P)/Ppack` in [0, inf], n_g + N + 1. A warm start across Ppack on/off goes full-interp. QSS:
  `P_mot = min(P_mot, Ppack)`, and regen adds `min(F_mot_trq, P_regen/v)` below the rpm limit to the brake-torque
  limit (`P_regen` = Preg x motors). At the defaults no attribute, row, variable or SX node is new. Corrected EM4
  QSS, Sturn 18.99446 s and BCN 136.20374 s. Half ratings (301 Nm, 228 kW): +1.10% / +0.89%. Ppack 456 kW: +4.88%
  / +4.07%. Preg 456 kW: -0.58% / -0.44%. Regen check 2026-10-21 (Linux, MUMPS, full+duals from the corrected cold
  Sturn): single motor, Preg 50 kW. 49 iterations, 19.97510 s (-0.04571 s, -0.23%, QSS -0.42%). Min P_motor
  -50.000 kW at the knots, -50.069 kW at the collocation points (the power row holds at the knots only, B3).
  Energy 1016.4 -> 982.1 Wh. No corrected EM4 solve with the keys yet. `limit_prices` does not price the regen
  bounds or the pack row. It gives NaN for Tmax and Pmax if a motor torque is below -1e-6 x its scale `u_s`
  (regen). A run with a key saves under its own setup tag (E2).
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
  Stage M (2026-10-21, pinned by `tests/test_telemetry.py`): one `f_veh` call on every knot and collocation point.
  It uses the NLP's near-hold inputs (`refine.nlp_inputs`), never `u_full`: linear inputs put ax / ay off by rms
  0.70 / 1.15 m/s^2. `data.vehicle_full` holds every point, `data.vehicle` = its knot slice `[::OPT_d+1]`
  (bitwise), `data.t_full` = `time_at_points`. New channels: `om_gear` (mean wheel speed x `vp.gear`, rad/s, as
  matlab EM4 `Om_motor_*` is off by gear^2) and `throttle` (normalised motor torque, mean of 4 with EM4). Also
  `xs_fl..xs_rr` (damper, m, positive in bump) and `sf` (dt/ds). The 66 old result arrays stay bitwise equal. ax / ay
  at the nodes match the state-polynomial kinematics to 5.2e-5 / 7.4e-7 m/s^2.
  `functions/telemetry.export_monster(data, path, step=0.02, sign=(-1, -1))` writes a ChassisSim monster file
  (vendor PDF): 11 tab columns, no header, 50 Hz. Columns 1-4: distance (integral of speed) m, rpm, ay g, ax g.
  Columns 5-8: dampers FL FR RL RR mm. Columns 9-11: steer at the tyre deg, throttle % (0 under regen), speed
  km/h. Sign default = right positive. Five ChassisSim manuals say that import flag 1 takes a right turn as positive.
  The monster-file PDF does not say it. So set both flags to 1. No ChassisSim import ran yet. `read_monster` takes the
  same pair (E5), and `docs/logging_spec.md` (E4) has the sign table.
  `export_csv(data, path, step=1.0, by='distance')` writes an own CSV with a header row. Its columns: s, t, vx, vy,
  r, n, eps, delta, throttle, om_gear, xs_*, ax, ay. Export times (stored Sturn): 5.5 ms for the monster
  file (901 x 11), 2.3 ms for the CSV (541 x 16). BCN stand-in solve (N 155, not a solved lap): 23 ms for the
  monster file (5953 x 11). A result saved before stage M has no `vehicle_full` (resample raises): solve it again.
  ax, ay and the forces stay mesh limited on the 30 m mesh.
- **Result names and provenance (E2, 2026-10-28, pinned by `tests/test_provenance.py`).** A setup knob that differs from its
  default adds `_<8 hex>` as the LAST part of the stem. `userOpts.setup_knobs(kw)` gives the knobs, and `knob_tag` the first 8
  hex digits of the sha256 of their canonical JSON (`functions.sweep.fingerprint`). Default stems never change. Pin:
  `{'vp_overrides': {'LMUY': 0.95}}` -> `8c76aec0`. Knobs: every userOpts argument outside `userOpts._NOT_SETUP`, plus
  `TyreModel` and the `MLTP_screen` arguments `load_model` and `ds_fine`. So a new userOpts argument is a knob at once.
  `_NOT_SETUP` holds the stem keys (circuit, AeroConfig, ATD, Electric_4Motors, tyre_set, physics, start_state, mesh) and the
  route keys (linear_solver, hsl_dir, jit, cse, circuits_dir, data_dir, ipopt_overrides). `vp_overrides` counts the entries
  that differ from the tyre-set default, plus the A5 keys. A number `n_bounds` is in the stem (`_hw<W>`), a (lo, hi) pair is
  a knob. A file tyre set adds the knob `tyre_file` (absolute path). `data['setup_tag']` exists for a knob run only.
  `data['provenance']` holds `code` (`git describe --always --dirty`, 'unknown' without git), `setup` (the knobs) and `run`.
  `run` holds the warm-start source, linear solver, jit, cse, casadi version and `ipopt_overrides` as JSON text. savemat
  refuses field names over 31 characters. `params` holds every vp primary, MF key and pt rating (about 1.5 kB).
  `optimise_design` stores no `params`. `ctx.out_path` = the file that save=True wrote: read it after a knob run.
  `RunConfig.setup_tag()` gets the same tag from `headless_solve.build_solve_kwargs`. Not in the name: `ipopt_overrides` (a
  looser tol through it can overwrite the default), the `optimise_design` `param_specs` / `Fz0_shift_bounds` and the
  `refine` options (one `_meshAdaptive` stem). The tracked `Results/*.mat` predate E2 (no provenance).
  `validation/run_py.py` runs with `OPT_ds`, `tol` or `max_iter` get a tag.
- **Between knots (B3, 2026-10-28, pinned by `tests/test_between_knots.py`).** `userOpts(between_knots='matlab' | 'linear' |
  'linear+power')`. 'matlab' (default) = MATLAB's U_k + duk*tau with duk per metre. That is a near hold of U_k at the
  collocation points, and the path rows hold at the knots only. 'linear' feeds the dynamics U_k + (U_k+1 - U_k)*tau (same
  rows and variables). 'linear+power' adds the motor power rows and the pack row (`MLTP.colloc_path_rows`) at the N*OPT_d
  collocation points. They come LAST in g, after the periodic rows, point by point. `structure.n_pathcol` counts them: Sturn
  n_g 1904 -> 1958, BCN 16015 -> 16480, corrected EM4 2038 -> 2254. EM4 under matlab gets no rows. No rpm rows: the Om box
  caps the motor speed at gear*Vmax/Rw = 0.885 OMmax. Ponytail: add them for a gear above 8.19 or an Rw below 0.3143 m. The
  7-state init keeps MATLAB's arithmetic. `data['between_knots']` only when not 'matlab'. Another value seeds the primal
  only. `optimise_design`, `setup_gradients` and `setup_sweep` refuse a value other than 'matlab'. `limit_prices` gives NaN
  for Pmax when n_pathcol > 0. Measured (Linux, MUMPS, cold): Sturn 'linear' 143 iterations, 18.05249 s (+0.188%). Sturn
  'linear+power' 125, 18.05521 s (+0.203%). BCN 'linear+power' 117, 116.47793 s (+0.029%, first-order forecast +0.33% to
  +0.45%).
- **Radau collocation (B4, 2026-10-28, pinned by `tests/test_radau.py`).** `userOpts(collocation='legendre' | 'radau')`,
  23-state NLP only (the 7-state init stays Legendre). `discretise(..., scheme=)` takes the Radau IIA points from casadi. There
  is no numpy fallback, so a Radau result or a Radau QSS screen needs casadi. The last point of each interval is the next
  knot, so `s_full` repeats every interior knot (N zero steps). n_w and n_g equal Legendre: each interval keeps the end-point
  copy and its trivial continuity row (Choice A). The copy has free bounds (fc611ac). Equal bounds on the copy and the knot
  gave 3 active constraints on 2 variables: 5325 iterations and a lap +5.3% on Sturn. With 'linear+power', the point on the
  next knot gets no power row, so n_pathcol counts OPT_d - 1 points per interval. `h_col_rows` on a Radau grid needs linear
  inputs (else ValueError). At a Radau end point, `vehicle_full` holds the left limit (input U_k + b_k), and the next entry is
  the knot. `data['collocation']` only for 'radau'. Another scheme seeds the primal only. `setup_gradients` of a Radau result
  needs `collocation='radau'` by hand. Measured (cold, MUMPS): Sturn 278 iterations, 17.99226 s (-0.146%), 19.5 s solve
  against 12.5 s with Legendre. Corrected Sturn 249 iterations, 20.05447 s (+0.168% vs V10), 12.0 s against 75.3 s. Knot
  roughness against the Legendre knots: fz_rl 0.18 (matlab) / 0.44 (corrected), zs 0.12 (matlab). But fx_rl 0.73 (matlab),
  ax 0.67 and zs 1.86 (corrected). Radau damps the undamped body modes of physics='matlab', so parity runs stay on Legendre.
  Skipped: Choice B (drop the copy, KKT -22%), add when a matlab-physics Radau study needs the wall time.
- **State scaling (B7, 2026-10-28, pinned by `tests/test_state_scaling.py`).** `userOpts(state_scaling='matlab' | 'typical')`.
  'typical' = `userOpts.TYPICAL_CHASSIS_SCALE` for rows 9-22: zs, theta, phi 1e-2, their rates 0.1, wu 0.3, zt 2e-2. The
  physical NLP is the same, in other units (f equal, g' = R g, J' = R J C). The start box keeps its physical width
  (`ctx.Xi_tol` rows 9-22 / scale). `ctx.chassis_scale` is None for MATLAB's 1.0. The 7-state init ignores it.
  `data['state_scaling']` only for 'typical'. A result of the other scaling seeds the primal only (no dual map). Stop rule:
  the IPOPT tolerances act on scaled errors. Under 'typical' the largest physical chassis defect is 8-18x smaller, so
  iteration counts across scalings are exploration only. Measured (MUMPS, cold, Sturn): the median of 5 starts is 207
  against 178 iterations. One start per scaling ends on a slow branch (typical vi 60.002: 19.11574 s, default vi 59.998:
  18.70993 s). MUMPS scaling off: 170 against 157. Corrected: 860 against 392. Never the default. Open: the Windows MA57 arm
  (`ipopt_overrides={'ma57_automatic_scaling': 'no'}`), owner. Commit 6451820 removed a bug: the start-box code reused the
  name `tol`, so IPOPT got an array for tol under 'typical'.
- **Torque-bias rows (C4, 2026-10-28, pinned by `tests/test_atd_tbr.py`).** `userOpts(atd_tbr=None | TBR >= 1)`, ATD On and EM4
  Off only (else ValueError). k = (TBR - 1)/(TBR + 1) = `pt.atd_k`. Per axle |ATD_l - ATD_r| <= k (ATD_l + ATD_r). That is
  |T_l - T_r| <= k |T_l + T_r| for drive and brake, as T_i = ATD_i (gear T_motor + 2 T_brake). TBR > 1: rows `diff_F_l`,
  `diff_F_r`, `diff_R_l`, `diff_R_r` in [0, inf] after `ATD_eq` (nh 8). TBR 1 (open differentials): `diff_F` and `diff_R`
  fixed at 0 (nh 6). Sturn n_g 1904 / 1942 (TBR 1) / 1980. The warm start compares the path-row names, so PureSlip with ATD On
  (also nh 8) seeds a TBR 2 NLP with the primal only. `data['atd_tbr']` only when set. Measured (cold, MUMPS): ATD Off against
  On +0.371 s on Sturn (+2.06%), +3.434 s on BCN (+2.95%). Sturn TBR 3 / 2 / 1: 18.02621 / 18.04571 / 18.33339 s (196 / 281 /
  119 iterations). BCN TBR 2: 116.64141 s (186 iterations), both axles reach k. The QSS forecast of ATD Off was +0.022 s: the
  QSS has no yaw moment of a left-right split. Not built: a passive LSD. Add it when the owner says the car has one and TBR 2
  costs 1% or more. `limit_prices` has no price for the diff rows. `setup_sweep._precheck_base` compares nh only.
- **Anti-roll bars (C3, 2026-10-28, pinned by `tests/test_arb.py`).** Primaries `vp.k_arb_f`, `vp.k_arb_r` [N/m at the wheel],
  default 0.0 (MATLAB has none). The left strut gets k (xs_l - xs_r) and the right strut the opposite force, after the MATLAB
  bump cut, both physics. Roll stiffness added = k t^2 per axle. A plain 0 adds no SX node (`vehModel._arb`), a symbol always
  adds the term (`optimise_design`). QSS (`ggv.lateral_load_transfer`): 4 k_arb added to the axle spring sum, exact for matlab
  physics. Under corrected the QSS keeps the spring-only split (error up to 9%, series law skipped). GUI rows in "Suspension
  stiffness" (max 3e5). `ksD` stays, labelled "(unused)". The parity export drops both at 0. QSS Sturn: front 2e4 +0.119%,
  rear 2e4 +0.494%, both -0.026%. QSS BCN over front shares 0.3 to 0.7: -0.10% to +1.20%, optimum 0.52 (default 0.48). NLP
  (`setup_gradients`, Sturn, ATD On): -5.96e-8 / +1.12e-8 s per N/m (front / rear), about 100x below the QSS slope. ATD Off:
  +6.35e-8 / -7.95e-8, so the gap stays. Warm hops (full+duals): 2e3 N/m steps match first order within 4%, and 2 of 3 hops of
  2e4 N/m reach another local optimum. A hop back reaches 1.444 ms of objective below the stored Sturn default. Bars lower the
  roll damping ratio (c from the spring k only: 0.56 -> 0.48 at a 2e4 front bar, corrected).
- **Tyre files and pressure (C5, 2026-10-28, pinned by `tests/test_tir.py`).** `functions/importfile.read_tir(path)` reads a .tir
  one line at a time: upper-case names, `$` / `!` comments, D exponents, text values kept. `tyre_set='file:<path>'` uses it as
  written (the sign convention of the model, MF 5.2 form). ValueError for a missing or non-numeric coefficient or FNOMIN, or
  for units other than meter / newton / radians. Also ValueError for a file set with Kya(Fz0) <= 0: a file in ISO signs has
  pKy1 < 0. FNOMIN -> `vp.Fz0`. An entry that no model line reads gives a warning when it is not neutral. MF keys 64 -> 75: `ppx1..ppx4` and `ppy1..ppy5`
  (0.0), `NOMPRES` = `INFLPRES` = 220000 Pa (placeholders, hot gauge Pa). `vehParams.mf_pressure(mf)` -> 6 factors (plain 1.0
  at dpi 0): mu_x, Kxk, mu_y, Kya, the load peak of Kya, Kyg0. None in SVx, SVy, E or C. The QSS folds the two mu factors into
  its LMUX / LMUY copy. Stem `_tir_<base name>`, plus the knob `tyre_file` (absolute path) in the tag.
  `setup_sweep._model_hash` hashes the file. `.gitignore`: `*.tir`, `Data/tyres/`. Not read from the file: Rw, kt, Jw, f,
  gear. The 7-state init and the friction homotopy ignore the file and the pressure (the init still reads `vp.Fz0` =
  FNOMIN). No TYRESIDE mirror. A relative and an
  absolute spelling of one file differ for the warm start (full-interp). Made-up pp terms at +10% pressure: mu_x x0.9855,
  Kxk x0.971, mu_y x0.9884, QSS Sturn +0.29%, dJ/dINFLPRES +1.508590e-6 s/Pa. Warm hop (full+duals): 3 iterations, dJ
  +0.03343 s against first order +0.03319 s.
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
  in `userOpts.py`, so code calling `Powertrain(ctx)` without `userOpts` lacks those attributes. Under
  `physics='corrected'`, `vp_overrides` can reset `Tmax` / `Pmax` / `vp.gear` and add `pt.Preg` / `pt.Ppack` (A5
  powertrain keys bullet). Without them `ctx.pt` has no `Preg` or `Ppack`.
- **Missing `DATA_AA.mat` does not raise** in `vehParams` (it warns and falls back to placeholder
  `Cd`/`Cl`, which only the 7-state model reads), but `vehModel(ctx)` raises `RuntimeError` if `ctx.aero is None`.
- **Units are SI by convention only (not enforced)**, except some aero AoA and camber/toe fields
  stored in **degrees** (with separate `*_rad` companions). Mixing the deg vs rad field is an easy bug.
- **The root `__init__.py` is stale/broken**: it imports `.importfile`/`.collocation`/`.context`,
  which live under `functions/`, not the root, so importing the repo root as a package raises
  `ImportError`. The working package init is `functions/__init__.py`. Import submodules directly
  (e.g. `from functions.importfile import load_solution`); the tests do.
