# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python port of a MATLAB **Minimum-Lap-Time-Problem (MLTP)** framework for a 4-motor
electric race car (`FullModel_4EM_Suspension_FullTyre`). It poses the racing line as an
optimal-control problem in the **space (track arc-length) domain**, transcribes it with
**direct Legendre collocation**, and solves the resulting NLP with **IPOPT** (shipped inside
the CasADi wheel, using the **MUMPS** linear solver). There is no application/server — the
deliverables are top-level scripts you run, `.mat` result files, and Plotly HTML figures.

## Environment & commands

Everything assumes the **repo root as working directory** — all paths are relative
(`Circuits/`, `Data/`, `Results/`, `Plots/`). Use the in-repo venv:

```powershell
venv\Scripts\Activate.ps1        # PowerShell; or call venv\Scripts\python.exe directly
pip install -r requirements.txt  # casadi, numpy, scipy, plotly, kaleido (kaleido optional)
```

**Running a solve.** The scripts have **no argparse/sys.argv** — their `__main__` block calls
the function with hardcoded args (`circuit='Sturn'`, `vi=60.0`). To change circuit/config,
either edit the `__main__` call or import and call with kwargs:

```powershell
python MLTP_initial.py                                   # 7-state warm-start solve
python MLTP.py                                           # full 23-state solve
python -c "from MLTP import MLTP; MLTP(circuit='BCN', AeroConfig='Static', ATD='On', Electric_4Motors='Off', TyreModel='CombinedSlip')"
python -c "from MLTP_paramOptim import MLTP_paramOptim; MLTP_paramOptim(circuit='Sturn')"
```

**Using Coin-HSL (HSL linear solver).** The solve defaults to IPOPT's HSL `ma57`
solver. Provide a Coin-HSL library (the MinGW/`libgfortran5` `CoinHSL_jll` build
matches the casadi wheel's ABI) by setting `COINHSL_DIR` to its `bin/` folder, or
rely on the seeded default in `functions/hsl.py`. The directory is registered on
the Windows DLL path and probed once; if HSL can't load, `_make_solver` falls
back to MUMPS with a warning (a solve never crashes on a missing DLL). `ma57`
runs with MC64 auto-scaling (`ma57_automatic_scaling`, set automatically in
`functions/hsl.py`) so it converges on the stiff 23-state problem, where unscaled
MA57 can stall. Choose the solver per call: `MLTP(circuit='BCN',
linear_solver='ma97')` or `linear_solver='mumps'`. Benchmark them with `python
bench_linear_solver.py`, which reports a per-iteration linear-solver cost (the
apples-to-apples metric — MA57 factorises ~4–5× faster per IPOPT iteration than
MUMPS; total wall-clock depends on how many iterations each takes on this
nonconvex problem).

**Tests.** There is **no pytest/unittest** — the four `test_*.py` files are plain scripts whose
assertions run at module top level (no `if __name__ == '__main__'` block). Run a file directly;
there is no test runner and the finest selectable unit is a **whole file** (the first failing
assert aborts that file). There is no "run all" command — run each in turn:

```powershell
python test_foundation.py        # casadi-free helpers: collocation, geometry, simpleMA, Powertrain constants
python test_transcription.py     # discretise/pack/unpack round-trip, column-major (order='F') packing
python test_save_load.py         # solution .mat save -> load_solution round-trip (incl. warm-start nesting)
python test_params_useropts.py   # vehParams (Copy-B tyre set) + userOpts config branching
```

These four cover the **casadi-free numerical/config core only** — they do **not** exercise
`vehModel.py` / `MLTP.py` (which need CasADi + IPOPT). Smoke-test the symbolic models with:

```powershell
python -c "from functions.context import Ctx; from Powertrain import Powertrain; from vehParams import vehParams; from userOpts import userOpts; from vehModel import vehModel; ctx=Ctx(); Powertrain(ctx); vehParams(ctx); userOpts(ctx); vehModel(ctx); print(ctx.m23.nx, ctx.m23.nu)"
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
1. **`MLTP_initial.py`** builds a simplified **7-state bicycle model** (`vehModel_initial` →
   `ctx.m7`, nx=7/nu=3/ny=1, lumped Magic-Formula tyre) and solves a reduced OCP to produce a
   warm start, saved as `Results/init_<circuit>.mat` (`data.init`).
2. **`MLTP.py`** builds the **full 23-state model** (`vehModel` → `ctx.m23`) and solves the real
   problem, saving `Results/<circuit>_<config>.mat`. If `warm_start=None`, `MLTP()` calls
   `MLTP_initial(save=False)` itself and interpolates the 7-state solution onto the 23-state grid
   via `warmstart_guesses()`.

`MLTP.py` also owns the only definition of `build_path_constraints()` (config-dependent
friction-circle + powertrain constraints; the count changes with EM4/ATD).

### Co-optimization wrappers
- **`MLTP_paramOptim.py`** promotes static design parameters (`vp` fields) to constant-over-lap
  decision variables and solves them **jointly** with the racing line. Its `optimise_design(param_specs, tag, ...)`
  is the shared core (`param_specs` = list of `(vp_field, lower, upper)`). Default params:
  `brkB, Tdist, alpha_FL/FR/RW/TW`.
- **`MLTP_TyreOptim.py`** is a thin wrapper over `optimise_design()` promoting only `Fz0_shift`.

### The transcription engine — `functions/transcription.py`
Shared by all MLTP scripts. `discretise(track, OPT_ds, OPT_d)` builds the Legendre collocation
grid (defaults from `userOpts.py`: step `OPT_ds=30` m, degree `OPT_d=3`, `OPT_uinter='linear'`);
`build_and_solve_nlp(...)` assembles the CasADi NLP — decision vector
`w = [Xk; Uk; (Yk); Xkj (; P)]` packed **column-major (`order='F'`)**, collocation defect +
endpoint continuity, path constraints, time-domain input-rate limits, `Xi/Xf` boundary bounds
(**`NaN` = free state**), objective `J = Σ Qk·B·dsk` + regularisation — then calls
`nlpsol('ipopt')`. `_make_solver()` **selects the configured `linear_solver`**: it uses HSL
(`ma*`) when a working Coin-HSL DLL is found (via the `COINHSL_DIR` env var or a
seeded default, registered on the Windows DLL path and probed once), otherwise
it transparently falls back to `mumps` — so a solve never crashes on a missing
HSL DLL. Default is `ma57`. After the solve: `unpack_solution`,
`reconstruct_x_full`, `interp_inputs`, `compute_time`, `reconstruct_track`.

### Configuration — `userOpts.py`
Builds `ctx`: calls `Powertrain`+`vehParams`, loads or **synthesizes** the track, sets `Xi/Xf`,
collocation options, the IPOPT options dict, and the config switches the models branch on
(`vp.ActAero`, `pt.ATD`, `pt.EM4`). Circuit selection: `_REAL_CIRCUITS` maps names to `.mat`
files in `Circuits/` (`BCN` → `Barcelona_circuit.mat`, the sector splits `BCN_S1`/`BCN_S2`/`BCN_S3`,
plus `Jarama`, `Spa`, `BCNAssetto`); any other name is treated as **synthetic** and its curvature
is generated analytically by `_synthetic_curvature` (`Straight`, `Hairpin`, `Sturn`, `Circle`,
`ZigZag`, `ZigZagMirror`, `VirtualTrack`) — no `.mat` needed. Guard: if `ATD` **and**
`Electric_4Motors` are both `On`, it forces `ATD=Off` with a warning.

### Helpers (`functions/`)
- **Solver-side:** `collocation.py` (Legendre points/coeffs; CasADi-native with a numpy fallback),
  `simpleMA.py` (pre-solve curvature smoothing), `importfile.py` (`.mat` I/O, `mat_to_namespace`,
  `load_solution` warm-start unwrap), `context.py` (`Ctx`).
- **Post-processing only** (reached via `reconstruct_track` *after* the solve): `curv2cart.py`
  (s,k → cartesian centreline), `cartPath.py` (lateral offset `n` → racing line), `trackLimits.py`
  (boundary polylines), `rotatePoint2D.py` (used only inside `curv2cart`).
- **`plotSDI.py`** writes Plotly HTML to `Plots/<circuit>/<config>/`; called from `MLTP()` when `plot=True`.

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
- **Two distinct tyre models coexist** — don't conflate: full Pacejka 5.2 `ctx.mf` (~60 coeffs,
  used by `vehModel.py`) vs. the simplified 8-param `vp.tyre` (used **only** by `vehModel_initial.py`).
- **`vehModel_initial.py` is NOT obsolete** — it's the deliberately reduced warm-start model.
  `vehModel.py` is the canonical full model.
- **Wheel-radius / gear quirk (intentional, do not "fix").** `Powertrain` sets `vp.Rw=0.3142857`
  and derives `vp.gear` from it; `vehParams` then overwrites `vp.Rw=0.355` but deliberately does
  **not** recompute `vp.gear`. `test_params_useropts.py` asserts gear stays based on the old radius.
- **`Powertrain.py` is not a map.** Despite the name it only stores 5 scalar ratings
  (`Pmax, Tmax, OMmax, Vmax, eff`); `eff=0.9` is used only in the post-solve energy integral. The
  actual power/rpm limits are enforced in `MLTP.py`/`vehModel.py`. `pt.EM4`/`pt.ATD` are set later
  in `userOpts.py`, so code calling `Powertrain(ctx)` without `userOpts` lacks those attributes.
- **Missing `DATA_AA.mat` does not raise** in `vehParams` (it warns and falls back to placeholder
  `Cd`/`Cl`), but `vehModel(ctx)` raises `RuntimeError` if `ctx.aero is None`.
- **Units are SI by convention only (not enforced)** — except some aero AoA and camber/toe fields
  stored in **degrees** (with separate `*_rad` companions). Mixing the deg vs rad field is an easy bug.
- **The root `__init__.py` is stale/broken** — it imports `.importfile`/`.collocation`/`.context`,
  which live under `functions/`, not the root, so importing the repo root as a package raises
  `ImportError`. The working package init is `functions/__init__.py`. Import submodules directly
  (e.g. `from functions.importfile import load_solution`); the tests do.
