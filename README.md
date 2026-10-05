# FullModelSim_Python

A Python port of a MATLAB **Minimum-Lap-Time-Problem (MLTP)** framework for a
4-motor electric race car (`FullModel_4EM_Suspension_FullTyre`). It poses the
racing line as an optimal-control problem in the **space (track arc-length)
domain**, transcribes it with **direct Legendre collocation**, and solves the
resulting NLP with **IPOPT** (shipped inside the CasADi wheel).

There are two ways to use it:

1. **Python scripts** — `MLTP.py`, `MLTP_initial.py`, and the co-optimisation
   wrappers; deliverables are `.mat` result files and Plotly HTML figures.
2. **FullModelSim desktop app** — a PySide6 GUI frozen to a standalone Windows
   `.exe` (PyInstaller) that runs on a machine with **no Python installed**.

> Deep architecture/internals live in [`CLAUDE.md`](CLAUDE.md). This README is
> the quick-start and current-status overview.

---

## Status

| Area | State |
|---|---|
| Core MLTP solver (7-state warm start → full 23-state) | ✅ working |
| Co-optimisation (`MLTP_paramOptim`, `MLTP_TyreOptim`) | ✅ working |
| HSL linear solvers (MA57/MA97/MA27) with MUMPS fallback | ✅ working |
| Desktop GUI app (`app/`) | ✅ working |
| Standalone Windows `.exe` (PyInstaller) | ✅ builds & runs; verified end-to-end |
| Coin-HSL inside the frozen exe (`sys._MEIPASS`) | ✅ verified (`ma57`, self-contained) |
| Automated test suite (15 files, casadi-free core) | ✅ 15/15 |

The frozen build has been verified on the dev machine end-to-end (Sturn solve on
`ma57` from the bundled Coin-HSL → `Optimal Solution Found`, results + plots
written). A literal **clean-VM run** (a machine with no Python/casadi/HSL) remains
the gold-standard final acceptance check — see [`build/README.md`](build/README.md).

---

## Setup

Everything assumes the **repo root as working directory** — all paths are
relative (`Circuits/`, `Data/`, `Results/`, `Plots/`). Use the in-repo venv:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1                 # PowerShell; or call venv\Scripts\python.exe directly
pip install -r requirements.txt           # casadi, numpy, scipy, plotly, kaleido(optional), PySide6, pyinstaller
```

`Data/DATA_AA.mat` (aerodynamic coefficients) is **required for a real solve**.

---

## Usage

### A) Run a solve from Python

The scripts have **no argparse/CLI** — their `__main__` block calls the function
with hardcoded args. Either edit the `__main__` call or import and call with
kwargs:

```powershell
python MLTP_initial.py                         # 7-state warm-start solve
python MLTP.py                                 # full 23-state solve (Sturn by default)

# pick circuit/config via kwargs:
python -c "from MLTP import MLTP; MLTP(circuit='BCN', AeroConfig='Static', ATD='On', Electric_4Motors='Off', TyreModel='CombinedSlip')"

# co-optimise design parameters jointly with the racing line:
python -c "from MLTP_paramOptim import MLTP_paramOptim; MLTP_paramOptim(circuit='Sturn')"
```

The two-stage solve: `MLTP()` first builds a reduced 7-state warm start
(`MLTP_initial`) and interpolates it onto the full 23-state grid, then solves the
real problem. Results go to `Results/<circuit>_<config>.mat`; Plotly HTML figures
to `Plots/<circuit>/<config>/`.

**Circuits.** Real tracks live as `.mat` files in `Circuits/` (e.g. `BCN` →
`Barcelona_circuit.mat`, plus the `BCN_S1/S2/S3` sector splits); any other name
(`Straight`, `Hairpin`, `Sturn`, `Circle`, `ZigZag`, `ZigZagMirror`,
`VirtualTrack`) is **synthetic** — its curvature is generated analytically, no
`.mat` needed.

### B) Run the desktop GUI (development)

```powershell
venv\Scripts\python.exe -m app.main
```

The GUI writes a JSON config, spawns the solve as a subprocess, streams the IPOPT
log into the log pane, and lists the resulting plots. GUI runs write results and
plots under `%USERPROFILE%\Documents\FullModelSim`.

### C) Build the standalone Windows exe

```powershell
venv\Scripts\Activate.ps1
$env:COINHSL_DIR = "C:\path\to\coinhsl\bin"   # optional: bundle Coin-HSL for MA57/MA97
venv\Scripts\pyinstaller.exe build\windows-app.spec
```

Output: `dist\FullModelSim\FullModelSim.exe` (onedir). It launches the GUI; run
`FullModelSim.exe --headless cfg.json` to solve and exit (this is what the Run
button spawns). The build is **windowed** by default (no console window); set
`FMS_CONSOLE=1` at build time for a debug build with a console. Full build notes,
packaging rationale, and the acceptance checklist are in
[`build/README.md`](build/README.md).

---

## Linear solver (HSL / MUMPS)

IPOPT ships **inside** the CasADi wheel. The solve **defaults to HSL `ma57`**
(run with `ma57_automatic_scaling` so it converges on the stiff 23-state
problem). HSL solvers need a **Coin-HSL** shared library (the MinGW/`libgfortran5`
`CoinHSL_jll` build matches the casadi wheel's ABI); point the framework at it
with the `COINHSL_DIR` environment variable, or rely on the seeded default in
`functions/hsl.py`. In a frozen exe, the bundled Coin-HSL DLLs are found at the
PyInstaller root (`sys._MEIPASS`).

If no working Coin-HSL library is found, the solve **transparently falls back to
MUMPS** (bundled, needs no external library) — a solve never crashes on a missing
DLL. Choose the solver per call (`MLTP(circuit='BCN', linear_solver='ma97')` or
`'mumps'`), and benchmark them with:

```powershell
python bench_linear_solver.py
```

---

## Tests

There is **no pytest** — the 15 `test_*.py` files are plain scripts whose asserts
run at module top level. Run a file directly; the finest selectable unit is a
whole file. Run the full suite (each in turn):

```powershell
foreach ($f in Get-ChildItem test_*.py) { python $f.Name }
```

These cover the **casadi-free numerical/config core** plus the app's
config/serialisation/results layer. They do **not** run the full symbolic models
(which need CasADi + IPOPT). Smoke-test the symbolic model with:

```powershell
python -c "from functions.context import Ctx; from Powertrain import Powertrain; from vehParams import vehParams; from userOpts import userOpts; from vehModel import vehModel; ctx=Ctx(); Powertrain(ctx); vehParams(ctx); userOpts(ctx); vehModel(ctx); print(ctx.m23.nx, ctx.m23.nu)"
```

---

## Project layout

```
MLTP.py, MLTP_initial.py        # the two-stage solve drivers
MLTP_paramOptim.py, MLTP_TyreOptim.py   # design co-optimisation wrappers
vehModel.py, vehModel_initial.py        # full 23-state / reduced 7-state models
vehParams.py, Powertrain.py, userOpts.py  # config: params, ratings, options + overrides
plotSDI.py, gg_plots.py, bench_linear_solver.py   # plotting + solver benchmark
headless_solve.py               # cfg.json -> MLTP entry (used by the GUI subprocess)

functions/                      # transcription engine, collocation, hsl, .mat I/O, geometry
app/                            # PySide6 GUI: main, mainwindow, runconfig, solve_runner,
                                #   results, paths, presets/default.json
build/                          # PyInstaller spec, runtime hook, version info, build README

Circuits/   Data/DATA_AA.mat    # inputs (real track .mat, aero coefficients)
Results/    Plots/              # outputs (.mat results, Plotly HTML figures)
```

See [`CLAUDE.md`](CLAUDE.md) for the `ctx`-spine architecture, the space-domain
ODE conventions, the full state/control vectors, and editing gotchas.
