# Graph Report - FullModelSim_Python  (2026-10-03)

## Corpus Check
- Large corpus: 88 files · ~2,544,345 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 486 nodes · 1486 edges · 17 communities (10 shown, 7 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 44 edges (avg confidence: 0.9)
- Token cost: 236,288 input · 0 output

## Community Hubs (Navigation)
- MLTP Solve Pipeline
- Windows GUI App Design
- IPOPT & HSL Linear Solver
- App Entry & Packaging
- Full Vehicle & Tyre Model
- GG Diagram Plots
- Solution Plotting (plotSDI)
- Vehicle-Param Editor Config
- Vehicle Parameters & Presets
- User Options & Powertrain
- Results Parsing
- Collapsible Section Widget
- Track Loading & Smoothing

## God Nodes (most connected - your core abstractions)
1. `MainWindow` - 61 edges
2. `CLAUDE.md - Project Guide` - 60 edges
3. `Windows App (GUI + standalone exe) - Implementation Plan` - 59 edges
4. `Windows App (GUI + standalone exe) - Design` - 52 edges
5. `Setup-tab Vehicle-Parameter Editor - Design` - 43 edges
6. `vehModel()` - 42 edges
7. `MLTP()` - 40 edges
8. `README.md - Quick-start and Status` - 39 edges
9. `Setup-tab Vehicle-Parameter Editor - Implementation Plan` - 38 edges
10. `userOpts()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `MainWindow` --implements--> `FullModelSim desktop app (PySide6 GUI)`  [INFERRED]
  app/mainwindow.py → README.md
- `ma86 excluded from defaults` --rationale_for--> `MainWindow`  [INFERRED]
  docs/superpowers/specs/2026-06-23-coin-hsl-linear-solver-design.md → app/mainwindow.py
- `README.md - Quick-start and Status` --references--> `MainWindow`  [INFERRED]
  README.md → app/mainwindow.py
- `README.md - Quick-start and Status` --references--> `parse_summary()`  [INFERRED]
  README.md → app/results.py
- `README.md - Quick-start and Status` --references--> `RunConfig`  [INFERRED]
  README.md → app/runconfig.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **HSL linear-solver resolution chain (userOpts -> _make_solver -> functions/hsl)** — useropts_useropts, functions_transcription_build_and_solve_nlp, functions_transcription_make_solver, functions_hsl_apply_linear_solver, functions_hsl_resolve_hsl_dir, functions_hsl_register_hsl_dll_dir, functions_hsl_hsllib_path, functions_hsl_probe_linear_solver [EXTRACTED 1.00]
- **GUI configuration flow (widgets -> RunConfig -> cfg.json -> headless_solve -> MLTP -> vehParams/userOpts)** — app_mainwindow_mainwindow, app_runconfig_runconfig, headless_solve_build_solve_kwargs, headless_solve_main, mltp_mltp, vehparams_vehparams, useropts_useropts [EXTRACTED 1.00]
- **Canonical ctx driver order (Powertrain -> vehParams -> userOpts -> vehModel|vehModel_initial -> MLTP*)** — powertrain_powertrain, vehparams_vehparams, useropts_useropts, vehmodel_vehmodel, vehmodel_initial_vehmodel_initial, mltp_mltp, mltp_initial_mltp_initial, functions_context_ctx [EXTRACTED 1.00]

## Communities (17 total, 7 thin omitted)

### Community 0 - "MLTP Solve Pipeline"
Cohesion: 0.05
Nodes (34): CLAUDE.md - Project Guide, Design-parameter co-optimisation, Config-dependent control vector, Full Pacejka 5.2 tyre model (ctx.mf), Powertrain ratings (not a map), Simplified 8-parameter tyre (vp.tyre), cartPath(), collocation_coeff() (+26 more)

### Community 1 - "Windows GUI App Design"
Cohesion: 0.06
Nodes (30): MainWindow, _spin(), default_output_dir(), user_presets_dir(), list_plots(), parse_summary(), RunConfig, fmt_sci() (+22 more)

### Community 2 - "IPOPT & HSL Linear Solver"
Cohesion: 0.08
Nodes (33): bench(), _nan_row(), _print_table(), _row(), CasADi, Coin-HSL linear solvers (MA57/MA97/MA27), IPOPT (bundled in the CasADi wheel), MUMPS linear solver (+25 more)

### Community 3 - "App Entry & Packaging"
Cohesion: 0.12
Nodes (17): is_headless(), main(), is_frozen(), resource_path(), resource_root(), solve_command(), SolveRunner, Direct Legendre collocation transcription (+9 more)

### Community 4 - "Full Vehicle & Tyre Model"
Cohesion: 0.07
Nodes (4): Full 23-state vector, _linear_interp(), _poly(), vehModel()

### Community 5 - "GG Diagram Plots"
Cohesion: 0.16
Nodes (13): _add_rings_and_guides(), _apply_theme(), _as_dict(), build_series_from_solution(), _field(), generate_gg_plots(), _get(), _hull_loop() (+5 more)

### Community 7 - "Solution Plotting (plotSDI)"
Cohesion: 0.32
Nodes (13): _arr(), _as_dict(), _get(), _knot_grid(), plot_friction(), plot_inputs(), plot_powertrain(), plot_racing_line() (+5 more)

### Community 9 - "Vehicle Parameters & Presets"
Cohesion: 0.21
Nodes (3): mat_to_namespace(), default_primaries(), vehParams()

### Community 12 - "Results Parsing"
Cohesion: 0.22
Nodes (3): _cfg(), plot_dir(), result_mat_path()

### Community 15 - "Track Loading & Smoothing"
Cohesion: 0.29
Nodes (4): Real circuits (.mat in Circuits/), _matlab_round(), simpleMA(), _load_track()

## Knowledge Gaps
- **1 isolated node(s):** `NumPy`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 186 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MainWindow` connect `Windows GUI App Design` to `IPOPT & HSL Linear Solver`, `App Entry & Packaging`, `Vehicle-Param Editor Config`, `Qt Main Window & Widgets`, `Results Parsing`, `Collapsible Section Widget`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **Why does `vehModel()` connect `Full Vehicle & Tyre Model` to `MLTP Solve Pipeline`, `Windows GUI App Design`, `IPOPT & HSL Linear Solver`, `App Entry & Packaging`, `Vehicle Parameters & Presets`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `CLAUDE.md - Project Guide` connect `MLTP Solve Pipeline` to `Windows GUI App Design`, `IPOPT & HSL Linear Solver`, `App Entry & Packaging`, `Full Vehicle & Tyre Model`, `Solution Plotting (plotSDI)`, `Vehicle Parameters & Presets`, `User Options & Powertrain`, `Track Loading & Smoothing`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `MainWindow` (e.g. with `FullModelSim desktop app (PySide6 GUI)` and `ma86 excluded from defaults`) actually correct?**
  _`MainWindow` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `CLAUDE.md - Project Guide` (e.g. with `apply_linear_solver()` and `resolve_hsl_dir()`) actually correct?**
  _`CLAUDE.md - Project Guide` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Windows App (GUI + standalone exe) - Design` (e.g. with `Windows App (GUI + standalone exe) - Implementation Plan` and `_frozen_bundle_dir()`) actually correct?**
  _`Windows App (GUI + standalone exe) - Design` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `NumPy` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._