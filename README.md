# FullModelSim_Python

FullModelSim_Python is a Python port of a MATLAB **Minimum-Lap-Time-Problem (MLTP)** framework for a
4-motor electric race car (`FullModel_4EM_Suspension_FullTyre`). It poses the racing line as an
optimal-control problem in the **space (track arc-length) domain**. It transcribes the problem with
**direct Legendre collocation** and solves the NLP with **IPOPT**, which ships inside the CasADi wheel.

You can use it in two ways:

1. **Python scripts**: `MLTP.py`, `MLTP_initial.py`, `MLTP_screen.py` and the co-optimisation
   wrappers. They write `.mat` result files and Plotly HTML figures.
2. **The FullModelSim desktop app**: a PySide6 GUI. PyInstaller freezes it into a standalone app for
   Windows and Linux. The app runs on a machine with **no Python installed**.

> [`CLAUDE.md`](CLAUDE.md) describes the architecture and the internals. This README gives the quick
> start and the current status.

---

## Status

| Area | State |
|---|---|
| Core MLTP solver (7-state warm start → full 23-state) | ✅ working |
| Co-optimisation (`MLTP_paramOptim`, `MLTP_TyreOptim`) | ✅ working |
| HSL linear solvers (MA57/MA97/MA27) with MUMPS fallback | ✅ working |
| Desktop GUI app (`app/`) | ✅ working |
| Standalone app (PyInstaller, `build/app.spec`) | Linux: builds, smoke checks pass. Windows: first run in CI |
| Coin-HSL inside the frozen app (`hsl` folder of the bundle) | found with a stand-in library. No run with real Coin-HSL |
| Automated test suite (the files in `tests/`) | ✅ all pass on Linux. Windows: first run in CI |

On the dev machine, the frozen Linux build passes the three smoke checks of the release workflow.
They are a headless QSS lap, a headless NLP with one iteration and an offscreen GUI start. The
Windows build has no run yet, and no CI run exists yet. A run on a clean machine (no
Python, casadi or HSL) is still the final acceptance check, see [`build/README.md`](build/README.md).

---

## Quick start

Do these steps to get a first lap:

1. Install Git.
2. Clone the repository.
3. Open a terminal in the repo root.
4. On Linux, run `./start.sh`. On Windows, run `.\start.bat`.
5. If the script asks to install uv, type `y`.
6. Wait until the GUI opens.
7. In the GUI, click Run.

The result and the plots go to the folder `FullModelSim` in your Documents folder.

---

## Setup

### Requirements

- Git.
- Python 3.10 or later, or [uv](https://docs.astral.sh/uv/).
- Windows x64, or Linux x64 with glibc 2.34 or later (Ubuntu 22.04 or later).

The repo holds `Data/DATA_AA.mat` (aerodynamic coefficients). A real solve needs this file.

Run every command from the repo root. All paths are relative (`Circuits/`, `Data/`, `Results/`,
`Plots/`).

### One-command setup

Run the setup script:

```
./start.sh          # Linux. The command "sh start.sh" also works.
.\start.bat         # Windows, in PowerShell or in cmd
```

The script does these steps:

1. If uv is missing, it asks to install uv. Then it installs uv.
2. It creates `.venv` from `uv.lock`. If the machine does not have Python 3.13, uv downloads it.
3. It checks the Qt system libraries (Linux) and `Data/DATA_AA.mat`.
4. It reports the optional parts: Coin-HSL, the ChassisSim folder (`CHASSISSIM_DIR`) and a C
   compiler. A missing optional part does not stop the setup.
5. It builds the model and computes one QSS lap.
6. It runs the quick tests.
7. It opens the GUI.

The script has these flags:

- `--yes`: the script does not ask questions.
- `--no-gui`: the script does not open the GUI.
- `--no-tests`: the script does not run the quick tests.

### Manual setup

Run these commands:

```
uv sync                                     # create .venv from uv.lock
uv run fullmodelsim                         # start the GUI
uv run python tests/run_tests.py --quick    # run the quick tests
```

If your `.venv` is from before the switch to PySide6-Essentials, `uv sync` can delete Qt files.
Then run `uv sync --reinstall-package pyside6-essentials`. Or delete `.venv` and run the setup
again.

`requirements.txt` is for pip users. uv makes it from `uv.lock`. Do not edit it. To change a
dependency, do these steps:

1. Edit `pyproject.toml`.
2. Run `uv lock`.
3. Run `uv export --no-dev --no-hashes --no-emit-project -o requirements.txt`.

### Download the app

The release workflow attaches `FullModelSim-<tag>-windows.zip` and
`FullModelSim-<tag>-linux.tar.gz` to the GitHub Release of each `v*` tag. To use a download, do
these steps:

1. Extract the archive.
2. On Windows, start `FullModelSim.exe`. On Linux, start `FullModelSim`.

The Linux app needs glibc 2.35 or later and the system libraries libGL, libEGL and libxcb. The
downloads use MUMPS and hold no vendor data. [`build/README.md`](build/README.md) gives the
release steps.

---

## Usage

### A) Run a solve from Python

The scripts have **no command-line arguments**. Their `__main__` block calls the function with
fixed arguments (`circuit='Sturn'`, `vi=60.0`). To change the circuit or the configuration, edit
that call. Or import the function and give keyword arguments.

Run a script:

```
uv run python MLTP_initial.py     # 7-state warm-start solve
uv run python MLTP.py             # full 23-state solve
uv run python MLTP_screen.py      # QSS lap estimate (no NLP)
```

Give keyword arguments:

```
# another circuit and configuration
uv run python -c "from MLTP import MLTP; MLTP(circuit='BCN', AeroConfig='Static', ATD='On', Electric_4Motors='Off', TyreModel='CombinedSlip')"

# design parameters, optimised together with the racing line
uv run python -c "from MLTP_paramOptim import MLTP_paramOptim; MLTP_paramOptim(circuit='Sturn')"

# one setup change, started from the stored default result (B5)
uv run python -c "from MLTP import MLTP; MLTP(circuit='Sturn', continue_from='auto', vp_overrides={'LMUY': 0.95})"

# QSS lap estimate of a setup change
uv run python -c "from MLTP_screen import MLTP_screen; MLTP_screen(circuit='BCN', vp_overrides={'LMUY': 0.95})"
```

`MLTP()` solves in two stages. First it builds a reduced 7-state warm start (`MLTP_initial`) and
interpolates it onto the full 23-state grid. Then it solves the full problem. The result goes to
`Results/<stem>.mat`, and the Plotly HTML figures go to `Plots/<circuit>/<stem without the
circuit>/`. The section "Result files" gives the stem.

`MLTP_screen()` gives a quasi-steady lap estimate in about 1 s, with no IPOPT. The estimate is
not an optimum: it follows the centreline and has no transients. Use it to rank setups. It writes
`Results/<stem>_qss.mat`.

`continue_from='auto'` starts a setup change from the stored default result of the same circuit
and configuration. It uses the primal and dual values of that result. It raises `ValueError`
before the solve in these cases:

- The default result is missing.
- The default result holds setup knobs.
- The default result did not converge.
- The default result does not fit the new NLP.

Solve the default setup first. A cold solve of an off-default setup can end on a slower local
optimum. On Sturn, a 3% mass step continued from the default in 15 iterations. Its cold solve took
500 iterations, and its lap was 1.007 s slower.

**Circuits.** The real tracks are `.mat` files in `Circuits/`: `BCN` (`Barcelona_circuit.mat`), the
sector splits `BCN_S1`, `BCN_S2` and `BCN_S3`, `Jarama`, `Spa` and `NBR` (Nurburgring GP). Each real
track goes through a QA check when it loads. An error in the check stops the load with a
`ValueError`, and a smaller issue gives one warning. BCN gives a warning but stays as it is, because
the MATLAB parity tests use it. Any other name (`Straight`, `Hairpin`, `Sturn`, `Circle`, `ZigZag`,
`ZigZagMirror`, `VirtualTrack`) is **synthetic**. A formula gives its curvature, so it needs no
`.mat` file.

### B) Run a ChassisSim car or circuit

The repo holds no ChassisSim files. To use them, do these steps:

1. Set `CHASSISSIM_DIR` to the ChassisSim install folder. Or give the keyword `chassissim_dir`.
2. For a ChassisSim circuit, give `circuit='chassissim:<name>'`. The name is the `Circuit name`
   in `Circ_prop.ini`.
3. For a ChassisSim car, give `car='chassissim:<path of the .car file>'` and `physics='corrected'`.

For example:

```
uv run python -c "from MLTP import MLTP; MLTP(circuit='chassissim:<name>', car='chassissim:<path>.car', physics='corrected', n_bounds=(-0.05, 0.05))"
```

`n_bounds=(-0.05, 0.05)` holds the car within 5 cm of the centreline, which is the line of the
ChassisSim lap. Without it, the NLP finds its own racing line. A car refuses the options that it
cannot use, for example `start_state='rolling'`. [`CLAUDE.md`](CLAUDE.md) lists these options. The
same keywords work for `MLTP_screen()`.

### C) Run the desktop GUI (development)

Run this command:

```
uv run fullmodelsim
```

The GUI writes a JSON config and starts the solve as a subprocess. It shows the IPOPT log in the
log pane and lists the plots. To run a solve in the GUI, do these steps:

1. On the Main tab, select the circuit and the configuration.
2. Click Run.
3. When the solve ends, read the lap time in the summary. The Output tab lists the plots.

GUI runs write the results and the plots to the output folder of the Output tab. The default is the
folder `FullModelSim` in the Documents folder of the user, on Windows and on Linux. When you click
Run, the GUI saves the folder for the next start. The Advanced tab has a Coin-HSL folder field (see
"Linear solver").

**ChassisSim cars and circuits.** To run a ChassisSim car or circuit in the GUI, do these steps:

1. On the Advanced tab, set the ChassisSim folder. Or set the environment variable `CHASSISSIM_DIR`.
2. On the Main tab, select a circuit `chassissim:<name>`, or keep a FullModelSim circuit.
3. Select a car in the Car combo. Or click Browse, or use Load preset with a `.car` file.
4. Click Run.

A car sets Physics to `corrected`. It locks the controls that a car cannot use: Aero, 4 Motors, ATD,
Homotopy, the rolling start, the tyre model and the tyre set. The 4EM car restores them. The Car combo
lists the `.car` files of the ChassisSim folder, of the `Models` folder beside it and of
`<Documents>/FullModelSim/presets/cars`.

**Edit a car.** With a car selected, the Setup tab shows the values of its `.car` file in the units of
ChassisSim, for example N/m, mm and rpm. Edit a value and click Run. The GUI writes the edited car to
`<Documents>/FullModelSim/presets/cars/<name>_<code>.car` and solves it. The source file does not
change. To keep the edits, click Save preset. The GUI then writes a new `.car` file. It never writes
over a file or into a ChassisSim folder. Tables, curves and maps are read-only. A row that the
FullModelSim solve does not read has a tooltip that says so.

**Powertrain group.** For the 4EM car, the Setup tab has the group `Powertrain (physics corrected)`.
It holds the gear ratio, Tmax and Pmax of one motor, the regen power Preg and the pack power limit Ppack.
Preg and Ppack each have a check box. With Preg off, the motor does not regenerate. With Ppack off, the
pack has no power limit. The group is active only when Physics is `corrected`. The GUI
sends only the values that you change.

The Advanced tab has the group "Run, model and start":

- Run: `nlp` (the solve) or `qss` (a QSS lap estimate in about 1 s, with no IPOPT and no plots).
  The QSS lap is not an optimum.
- Physics: `matlab` (default) or `corrected`.
- Start state: `matlab` (default), `rolling` or `periodic`.
- Homotopy: three solves with the tyre friction scaled by 1.2, 1.1 and 1.0. When you select 4
  Motors, the GUI ticks this box.
- More options (JSON): a JSON object with the keys `n_bounds`, `ladder`, `ipopt_overrides`,
  `Steering`, `CamberGain`, `pt_overrides`, `between_knots`, `collocation`, `state_scaling`,
  `atd_tbr` and `continue_from`.

The Output tab finds the result file with the same name rule as the solve.

### D) Build the standalone app (Windows and Linux)

To build the app, do these steps:

1. If you want MA57 or MA97 in the app, set `COINHSL_DIR` to the Coin-HSL folder.
2. Run `uv run pyinstaller build/app.spec`.

The output is `dist\FullModelSim\FullModelSim.exe` on Windows and `dist/FullModelSim/FullModelSim` on
Linux (onedir). The program starts the GUI. To solve and exit with no GUI, run
`FullModelSim --headless cfg.json`. The Run button of the GUI starts this command. The Windows build
has no console window. For a debug build with a console, set `FMS_CONSOLE=1` at build time.
[`build/README.md`](build/README.md) gives the build notes and the release steps.

---

## Result files

A solve writes `Results/<stem>.mat`. The default setup keeps the stem
`<circuit>_<AeroConfig>_ATD<On|Off>_EM4<On|Off>`, for example
`Results/Sturn_Static_ATDOn_EM4Off.mat`. A choice other than the default adds a part to the
stem, in this order:

1. `_<tyre set>`: for example `_CopyB`, or `_tir_<name>` for `tyre_set='file:<dir>/<name>.tir'`.
2. `_physCorrected` for `physics='corrected'`.
3. `_startRolling` or `_startPeriodic` for a start state other than `'matlab'`.
4. `_hw<W>` or `_nbCustom` for a lateral band (`n_bounds`).
5. `_mesh<Mesh>` for a mesh request other than `'auto'`.
6. `_<8 hex digits>` for any other setup knob that differs from its default, such as
   `vp_overrides`, `vi` or `OPT_ds`.

The 8 hex digits are the first 8 digits of a sha256 hash of the knob values. So a knob run
never overwrites the default result. `data['provenance']` in the file records the code
version (`git describe`) and the knobs. It also records the run (warm start, linear solver,
IPOPT overrides) and every parameter value of the model. After a solve, `ctx.out_path`
holds the path of the file that it wrote. These inputs are not knobs, so two runs that
differ only in them share one file name:

- `ipopt_overrides`.
- The promoted parameters of `optimise_design`: `params` of `MLTP_paramOptim` (one
  `<circuit>_paramOptim.mat`) and `Fz0_shift_bounds` of `MLTP_TyreOptim`.
- The `refine` options of `MLTP`: every refined result gets the one `_meshAdaptive` stem.

The QSS screen writes `<stem>_qss.mat`.

## Defaults that move the lap

The table lists the defaults with the largest effect on the lap. A price is the first-order
change of the lap objective per unit of the knob. The stored default results give it (Sturn
18.01633 s, BCN 116.44407 s). A row with "solves" gives the measured change of the lap.
The objective includes the input regularisation (0.27% of the Sturn lap, 0.12% of the BCN
lap). "NLP" marks a value from `limit_prices`, `setup_gradients` or a solve. "QSS estimate"
marks a value of the QSS screen. On Sturn, the QSS mass slope is 1.1 times the NLP slope.
For other knobs the QSS can be far off. Its anti-roll-bar slope is about 100 times larger
than the NLP slope. Its ATD Off cost is 17 times too small.

| Knob | Default | Price or effect, Sturn / BCN | Source |
|---|---|---|---|
| `physics` | `'matlab'`: the MATLAB physics (items A1 to A6) | `'corrected'`: +11.1% / +11.7% | NLP solves |
| `ATD` | `'On'`: ideal torque vectoring, no loss | `'Off'`: +2.06% / +2.95%. `atd_tbr=2`: +0.15% / +0.17% | NLP solves |
| `n_bounds` (half-width) | None: 4 m on every circuit | -0.2025 / -1.861 s per m | NLP |
| `vi` (start speed) | 60 m/s, box 59 to 61 m/s | -0.0114 / -0.0700 s per m/s | NLP |
| `start_state` | `'matlab'`: free wheel speeds at s = 0 | `'rolling'`: +0.045 s / +0.285 s. `'periodic'`: BCN +0.467 s | NLP solves |
| `Tmax` (motor torque) | 602 Nm | -3.39e-3 / -6.71e-3 s per Nm | NLP |
| `Pmax` (motor power) | 456 kW | -4.78e-7 / -1.30e-5 s per W | NLP |
| `Tbrake_max` | 4000 Nm | -1.40e-4 / -3.73e-4 s per Nm | NLP |
| `LMUX`, `LMUY` (peak grip) | 1, 1 | Sturn -1.06 and -3.29, BCN -5.00 and -26.7 s per unit | NLP |
| `LFZO` (nominal load) | 1 (Fz0 4905 N) | -1.37 / -10.3 s per unit | NLP |
| `LKX`, `LKY` (slip stiffness) | 1, 1 | Sturn -0.192 and -0.640, BCN -1.54 and -4.48 s per unit | NLP |
| `Cd_scale`, `Cl_scale` | 1, 1 | Sturn +0.045 and -0.032, BCN +0.99 and -0.66 s per unit | NLP |
| `mb` (mass) | 1820 kg (car 2085 kg) | +2.8e-3 / +1.9e-2 s per kg | NLP solve (Sturn, a 3% step) / QSS estimate (BCN) |
| `OPT_ds`, `mesh` | 30 m, `'auto'` | BCN +0.20% and NBR -0.04% against a fine reference | NLP solves |
| `between_knots` | `'matlab'`: a near hold of the input between two knots | `'linear+power'`: +0.20% / +0.029% | NLP solves |

The periodic BCN lap is provisional, because the BCN curvature does not close the lap.
`setup_gradients` refuses the mass and the other fields that `vehParams` derives other
values from. For them, use a solve or the QSS. `docs/laptime_sim_improvements.md` has the
runs behind each number.

## Not modelled

Each item has the trigger that makes it worth adding.

- 3D road (grade, bank, vertical curvature): before a drag or braking correlation on NBR or
  Spa, when usable altitude data exists.
- Ride-height aeromap: when ride-height or damper data exists.
- Full-model g-g-v envelope by trim: together with a QSS that has a free racing line.
- Mesh-consistent regularisation: when refinement studies compare objectives across meshes.
- Seven-post rig and bump-grip derating: when road-profile data exists.
- Heading-consistent curvature sampling: when a circuit of the track builder needs it.
- Driver model, thermal tyre, wear and relaxation length: a thermal state when data shows a
  grip drift that load and slip cannot explain.
- Side-force roll moment of the twist wing: when a run tilts the twist wing.
- Passive limited-slip differential: when the owner says that the car has one, and TBR 2
  costs 1% or more against ideal torque vectoring. The measured cost is 0.15% (Sturn) and
  0.17% (BCN).
- Grip against load per axle (E7b): when wheel-load channels exist.
- Tracking objective inside the NLP (E6, step L): when a real log and its projection on the
  track exist.
- Full MF 6.1 tyre equations and a left-right tyre mirror (TYRESIDE): with the first real
  tyre file.
- Damper rate that includes the anti-roll bar: when a bar above 3e4 N/m runs with
  `physics='corrected'`.

---

## Linear solver (HSL / MUMPS)

IPOPT ships **inside** the CasADi wheel. The solve **uses HSL `ma57` by default**, with
`ma57_automatic_scaling`, so that it converges on the stiff 23-state problem. HSL solvers need a
**Coin-HSL** shared library. The MinGW/`libgfortran5` build of `CoinHSL_jll` matches the ABI of the
casadi wheel:

- On Windows, use the `bin/` folder of the MinGW build (`libhsl.dll`).
- On Linux, use the `lib/` folder of the `x86_64-linux-gnu-libgfortran5` build (`libhsl.so`).

The solve uses the first folder of this list that exists:

1. The `COINHSL_DIR` environment variable.
2. The Coin-HSL folder of the GUI (Advanced tab), or the `hsl_dir` keyword.
3. In a frozen app, the `hsl` folder of the bundle.

If no Coin-HSL library works, the solve **uses MUMPS** (bundled, no external library) and gives a
warning. A missing DLL never stops a solve. To select the solver for one call, give
`linear_solver`, for example `MLTP(circuit='BCN', linear_solver='ma97')` or `'mumps'`. To compare
the solvers, run:

```
uv run python bench_linear_solver.py
```

---

## Tests

There is no pytest. The files `tests/test_*.py` are plain scripts. Their asserts run at module top
level, so the smallest unit that you can run is a whole file. Every test starts with
`import _bootstrap` (`tests/_bootstrap.py`). That module puts the repo root first on `sys.path` and
makes it the working directory. So a test runs from any directory, and it writes its scratch files
to a temp directory.

Run the tests:

```
uv run python tests/test_foundation.py        # one file
uv run python tests/run_tests.py --quick      # every file but the four that iterate IPOPT on a vehicle NLP
uv run python tests/run_tests.py --all        # every file
```

The tests cover the numerical and configuration core (collocation, mesh, warm start, QSS screen)
and the configuration, serialisation and results layer of the app. `CLAUDE.md` lists each file and
what it needs (casadi, `Data/DATA_AA.mat`, PySide6). Only `test_setup_sweep.py` (about a minute),
`test_paramoptim_warmstart.py` and `test_ladder.py` (both capped at 5 iterations) solve the real
23-state NLP with IPOPT. `test_refine.py` runs one small real solve. The `--quick` run skips these
four files.

To check the symbolic model, run:

```
uv run python -c "from functions.context import Ctx; from Powertrain import Powertrain; from vehParams import vehParams; from userOpts import userOpts; from vehModel import vehModel; ctx=Ctx(); Powertrain(ctx); vehParams(ctx); userOpts(ctx); vehModel(ctx); print(ctx.m23.nx, ctx.m23.nu)"
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
start.py, start.sh, start.bat   # one-command setup (checks, uv sync, model check, tests, GUI)
pyproject.toml, uv.lock         # dependencies (requirements.txt is generated from uv.lock)
vehModel_cs.py, MLTP_screen.py  # ChassisSim car model, QSS lap estimate

functions/                      # transcription engine, collocation, hsl, .mat I/O, geometry
app/                            # PySide6 GUI: main, mainwindow, runconfig, solve_runner,
                                #   results, paths, presets/default.json
build/                          # PyInstaller spec, runtime hook, version info, build README
tests/                          # plain-script test suite: test_*.py, _bootstrap.py, run_tests.py
.github/workflows/              # CI: tests.yml (branch pushes, pull requests), release.yml (v* tags)

Circuits/   Data/DATA_AA.mat    # inputs (real track .mat, aero coefficients)
Results/    Plots/              # outputs (.mat results, Plotly HTML figures)
```

See [`CLAUDE.md`](CLAUDE.md) for the `ctx`-spine architecture, the space-domain
ODE conventions, the full state/control vectors, and editing gotchas.
