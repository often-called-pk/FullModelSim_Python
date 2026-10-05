# Windows App (GUI + standalone exe) — Design

**Date:** 2026-06-24
**Branch:** `windows-app` (off `coin-hsl`)
**Status:** Approved design, pre-implementation

## Goal

Ship the Minimum-Lap-Time-Problem (MLTP) simulator as a user-friendly Windows
desktop application, packaged as a standalone `.exe` that runs on a **clean
Windows machine with no Python installed**. The app lets a non-expert configure
and run a solve, watch its progress, and view results, while still allowing a
power user to override the full vehicle parameter set through an editable config
file.

## Scope decisions (locked)

- **Target:** true standalone — PyInstaller bundles CasADi (+ IPOPT/MUMPS),
  Coin-HSL DLLs, numpy/scipy/plotly, `Circuits/`, `Data/DATA_AA.mat`, and the
  shipped presets.
- **Toolkit:** PySide6 / Qt.
- **Solve execution:** subprocess with a live IPOPT log; Cancel kills the
  process. Crash-isolated from the GUI.
- **Results:** summary panel (numbers parsed from the saved `.mat`) plus a list
  of generated Plotly plots, each opened in the default browser. No embedded
  Chromium.
- **Expert params:** full `vehParams` overrides live in editable text config
  files (JSON); GUI has a "Load config" picker. No in-app 40-field form.
- **Presets:** ship `default.json` (the current baseline car). Users can save
  their own config files.

### Out of scope (v1, YAGNI)

- Parameter-optimization mode (`MLTP_paramOptim` / `MLTP_TyreOptim`).
- Inline plot rendering via QWebEngine.
- In-app expert form / in-app config editor.
- Multi-run queue / batch solving.
- Live plotting during a solve.

## Architecture

### File layout

```
app/                      NEW GUI package
  main.py                 entry: QApplication + MainWindow; detects --headless flag
  mainwindow.py           tabs + Run button + log pane + results panel
  runconfig.py            RunConfig dataclass <-> JSON  (run-config + Tier-1 + expert-config path)
  solve_runner.py         QProcess wrapper: spawn headless solve, stream stdout, Cancel=kill
  results.py              parse saved .mat -> summary numbers; list plot HTMLs
  paths.py                frozen-aware resource + output-dir resolution (sys._MEIPASS)
  presets/
    default.json          baseline car (current vehParams + run defaults)
headless_solve.py         NEW top-level: read config JSON -> run MLTP -> print progress -> exit
build/
  windows-app.spec        PyInstaller spec
```

Existing solver code (`MLTP.py`, `MLTP_initial.py`, `userOpts.py`, `vehParams.py`,
`vehModel*.py`, `functions/`) is untouched except the `vehParams` two-phase
refactor (below) and a frozen-path patch in `functions/hsl.py`.

### One exe, two entry modes

The same frozen exe serves both roles:

- Launched normally → runs the GUI (`MainWindow`).
- Launched with `--headless <cfg.json>` → runs the solve and exits.

The GUI spawns the solve by re-invoking **itself** (`sys.executable` +
`--headless`), so no separate Python interpreter is needed on the target
machine. In a dev checkout (not frozen) the runner spawns
`python headless_solve.py <cfg.json>` instead. `app/main.py` inspects `sys.argv`
for `--headless` and dispatches before constructing the QApplication.

The GUI layer is intentionally thin: all non-Qt logic (config build/serialize,
results parse) lives in plain modules (`runconfig.py`, `results.py`) that import
without a display, so they are unit-testable.

## Configuration flow

```
GUI form  ->  RunConfig  ->  cfg.json  ->  headless_solve.py  ->  MLTP(**run_kwargs, vp_overrides=...)
```

`cfg.json` has three groups:

1. **run-config:** `circuit`, `AeroConfig`, `ATD`, `Electric_4Motors`,
   `TyreModel`, `vi`, `ni`, `linear_solver`, `save`, `plot`, `output_dir`.
2. **Tier-1 tunables:** `brkB`, `Tdist`, `ksD`, `alpha_FL`, `alpha_FR`,
   `alpha_RW`, `alpha_TW`.
3. **vp_overrides:** dict of `vehParams` primary overrides merged from the
   loaded expert config file (if any).

The Tier-1 tunables are themselves `vehParams` primaries, so the GUI folds them
into a single `vp_overrides` dict before serialization. A preset file *is* a full
`vp_overrides` dict; `default.json` carries the current values.

### GUI tabs

- **Main:** circuit, AeroConfig, ATD, Electric_4Motors, TyreModel, vi, ni;
  Run / Cancel; live log pane.
- **Setup (Tier-1):** brkB, Tdist, ksD, four wing angles.
- **Output:** output_dir picker, save/plot toggles, results summary + plot list.
- **Advanced:** linear_solver, warm_start picker, "Load config" (expert file).

### Config-conflict rule (enforced in GUI)

`ATD=On` together with `Electric_4Motors=On` is forbidden. The underlying code
auto-flips `ATD -> Off` with a warning; the GUI must instead **disable** the ATD
control while 4-motors is On, so there is no silent change.

## The `vehParams` two-phase refactor (safety-critical)

**Problem.** `vehParams.py` currently hardcodes primary inputs *and* computes
derived quantities (`ms`, `m`, `l_f`, `l_r`, `c_fl..c_rr`, `m_eff_f/r`,
`Wfl0..Wrr0`, `xti/xsi/lsi`, mirrored camber, static loads) inline from those
hardcoded values. Applying `vp_overrides` *after* `vehParams` runs would leave
every derived quantity stale — silent wrong physics.

**Fix.** Restructure `vehParams` into two phases:

1. Build a `primaries` dict containing every typed-in leaf value, then
   **merge `vp_overrides` into it**.
2. Compute every derived quantity *from the merged dict*.

So overriding any primary (e.g. `mb`) propagates correctly to all dependents
(`ms`, `m`, dampers, static loads) by construction. This is also the mechanism
that makes presets and expert config files correct.

**Contract / guardrails:**

- With no overrides, all current defaults are reproduced **byte-for-byte**,
  including the intentional `Rw=0.355` / `gear`-based-on-0.3142857 quirk and the
  Copy-B tyre set. `test_params_useropts.py` must stay green **unchanged**.
- Override keys are validated against the set of known primaries; an unknown key
  raises, it is not silently ignored.
- The Pacejka `mf` block (~60 coeffs) is overridable via config for experts but
  is **not** present in `default.json`.

**Threading the kwarg.** Add `vp_overrides=None` to `vehParams(ctx, ...)` and
`userOpts(ctx, ...)`. `MLTP` / `MLTP_initial` / the optim wrappers already pass
`**useropts_kwargs` through to `userOpts`, so they accept it without signature
changes.

## Bundling (PyInstaller)

**Data files (collected into the bundle, resolved at runtime via
`sys._MEIPASS`):** `Circuits/*.mat`, `Data/DATA_AA.mat` (required — a real solve
raises without it), `app/presets/*.json`.

**Native libraries:**

- **CasADi:** the wheel ships its compiled DLLs plus IPOPT/MUMPS. The
  PyInstaller hook normally collects them; we verify by treating a frozen MUMPS
  solve as the acceptance floor.
- **Coin-HSL:** bundle the HSL `bin/` DLLs and patch `functions/hsl.py` to also
  resolve the DLL directory from the frozen path (`sys._MEIPASS`) in addition to
  `COINHSL_DIR` / the seeded default. If HSL fails to load, the existing code
  already falls back to MUMPS with a warning, so the exe always solves; HSL is a
  performance bonus, not a hard dependency.

**Writable output (important).** A frozen exe may live in a read-only location
(`Program Files`). The current code writes `Results/` and `Plots/` relative to
the working directory, which would crash there. Resolution:

- Default `output_dir` = `%USERPROFILE%\Documents\FullModelSim\`, created on
  first run.
- The GUI exposes `output_dir` (editable) and passes it through `cfg.json`; the
  headless solve writes results and plots there.
- The results panel and plot-Open buttons read from that directory.

**Build strategy.** `windows-app.spec`: build **one-folder first** (far easier
to debug DLL collection), then one-file for shipping.

## Error handling

- Missing `DATA_AA.mat` in the bundle → headless solve raises a clear message;
  GUI surfaces it in the log pane and a dialog rather than a bare traceback.
- HSL load failure → MUMPS fallback (already implemented), warning shown in log.
- Solve non-convergence / IPOPT failure → non-zero exit from the headless
  process; GUI reports failure and keeps the log for inspection.
- Cancel → `QProcess.kill()`; partial output (if any) is left in `output_dir`.
- Unknown `vp_overrides` key → headless solve raises before solving.

## Testing

Keep the existing convention (plain scripts, asserts at module top level, run a
file directly — no pytest).

- `test_params_useropts.py` — **unchanged**, must stay green (the refactor
  contract).
- `test_runconfig.py` (new) — RunConfig ↔ JSON round-trip; Tier-1 + expert file
  merge into a single `vp_overrides`; unknown key rejected.
- `test_vp_overrides.py` (new) — overriding a primary (e.g. `mb`) recomputes
  derived (`ms`, `m`, dampers); empty overrides reproduce current defaults
  byte-for-byte.

GUI widgets are not auto-tested (no display in CI); the testable logic lives
outside Qt. Manual smoke test = the acceptance floor below.

## Acceptance floor

The built exe, on a clean Windows VM with no Python:

1. Launches the GUI.
2. Runs `Sturn` (synthetic curvature — needs `DATA_AA.mat` but no circuit file)
   end-to-end with the MUMPS linear solver.
3. Streams the IPOPT log live, writes results + plots to the Documents output
   folder, and opens a plot in the browser.

HSL working in the frozen app is a stretch goal on top of this floor.

## v1 summary

Single solve. Run-config + Tier-1 tunables in the GUI; full vehParams via an
editable config file. Live IPOPT log with Cancel. Summary + open-plots-in-browser.
Standalone exe on clean Windows, MUMPS as the guaranteed solver and HSL as a
bonus.
