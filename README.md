# FullModelSim_Python

A Python port of a MATLAB **Minimum-Lap-Time-Problem (MLTP)** framework for a 4-motor electric
race car (`FullModel_4EM_Suspension_FullTyre`). It poses the racing line as an optimal-control
problem in the **space (track arc-length) domain**, transcribes it with **direct Legendre
collocation**, and solves the resulting NLP with **IPOPT** (bundled inside the CasADi wheel).

There is no GUI or server — you run top-level scripts that write `.mat` result files and
interactive Plotly HTML figures.

> Working on the code? See [`CLAUDE.md`](CLAUDE.md) for the architecture, conventions, and gotchas.

---

## What it computes

Given a track (real or synthetic) and a vehicle configuration, it finds the **minimum-lap-time
trajectory** — the racing line, speed profile, per-corner motor/brake torques, steering, and the
full suspension/tyre state — subject to friction-circle and powertrain limits. The full vehicle is
a **23-state** model: planar dynamics + heave/pitch/roll suspension + 4 unsprung masses + 4 tyre
deflections, with a full **Pacejka 5.2 combined-slip** tyre on every corner.

A fast **7-state bicycle model** is solved first to warm-start the full problem.

---

## Requirements & install

Use the in-repo virtual environment, with the **repo root as the working directory** (all paths
are relative):

```powershell
venv\Scripts\Activate.ps1          # PowerShell; or call venv\Scripts\python.exe directly
pip install -r requirements.txt    # casadi, numpy, scipy, plotly, kaleido (kaleido optional)
```

- **IPOPT** ships inside the `casadi` wheel — no separate solver install needed.
- **Coin-HSL** (for the default `ma57` linear solver) is optional; without it the solve falls back
  to MUMPS automatically. See [Linear solver](#linear-solver-hsl--mumps).
- **`mpi4py`** is intentionally *not* in `requirements.txt` — it is only needed for the HPC
  [batch sweep](#batch-parameter-sweeps-mpi) and should be built against the cluster MPI.

A real solve also needs `Data/DATA_AA.mat` (aerodynamic coefficients) and, for a real circuit, the
corresponding track `.mat` in the track folder.

---

## Quick start

The MLTP scripts have **no command-line arguments** — their `__main__` block calls the function
with hardcoded defaults (`circuit='Sturn'`, `vi=60.0`). Run a script directly, or import and call
with keyword arguments to change the circuit/configuration:

```powershell
python MLTP_initial.py     # 7-state warm-start solve only
python MLTP.py             # full 23-state solve (auto-runs the 7-state warm start)
```

```powershell
# Full solve on Barcelona with a custom configuration:
python -c "from MLTP import MLTP; MLTP(circuit='BCN', AeroConfig='Static', ATD='On', Electric_4Motors='Off', TyreModel='CombinedSlip')"
```

`MLTP()` returns the mutated `ctx` object and (by default) saves the solution `.mat` and writes
the Plotly figures.

### `MLTP()` options

```python
MLTP(circuit='Sturn', vi=60.0, ni=nan, warm_start=None,
     AeroConfig='Static', ATD='On', Electric_4Motors='Off', TyreModel='CombinedSlip',
     save=True, plot=True, results_dir='Results', **useropts_kwargs)
```

| Argument           | Values / default                                              | Meaning |
|--------------------|--------------------------------------------------------------|---------|
| `circuit`          | a real or synthetic name (see below); `'Sturn'`              | track to solve |
| `vi`               | float, `60.0`                                                | initial velocity [m/s] |
| `ni`               | float or `nan` (free); `nan`                                 | initial lateral position [m] |
| `warm_start`       | `None` or a `.mat` path; `None`                              | `None` → auto 7-state warm start; else load an init `.mat` |
| `AeroConfig`       | `Static` \| `Active_RW` \| `Active` \| `AALB`; `Static`     | active-aero layout (0/1/2/4 aero inputs) |
| `ATD`              | `On` \| `Off`; `On`                                          | active torque distribution (4 corners) |
| `Electric_4Motors` | `On` \| `Off`; `Off`                                         | four per-corner motors vs. a single motor |
| `TyreModel`        | `CombinedSlip` \| `PureSlip`; `CombinedSlip`                 | Pacejka 5.2 mode |
| `save` / `plot`    | bool; `True`                                                 | write the `.mat` / write the HTML figures |
| `results_dir`      | str; `'Results'`                                             | output directory for the `.mat` |
| `linear_solver`*   | `ma57` \| `ma97` \| `ma27` \| `mumps`; `ma57`               | IPOPT linear solver (via `**useropts_kwargs`) |
| `hsl_dir`*         | str or `None`                                                | Coin-HSL `bin/` override (via `**useropts_kwargs`) |

\* `linear_solver`, `hsl_dir`, `circuits_dir`, and `data_dir` are passed through `**useropts_kwargs`
to `userOpts`.

> **Guard:** `ATD='On'` and `Electric_4Motors='On'` cannot both hold — the framework warns and
> forces `ATD='Off'`.

### Circuits

- **Real tracks** (loaded from the track folder): `BCN`, the sector splits `BCN_S1`/`BCN_S2`/
  `BCN_S3`, `Jarama`, `Spa`, `BCNAssetto`.
- **Synthetic tracks** (curvature generated analytically — no `.mat` needed): `Straight`,
  `Hairpin`, `Sturn`, `Circle`, `ZigZag`, `ZigZagMirror`, `VirtualTrack`.

Any unrecognised name raises an error.

---

## How it works (two-stage solve)

1. **`MLTP_initial.py`** builds a reduced **7-state bicycle model** with a lumped Magic-Formula
   tyre and solves a smaller OCP, saving `Results/init_<circuit>.mat`.
2. **`MLTP.py`** builds the **full 23-state model** and solves the real problem. When
   `warm_start=None` it runs stage 1 itself and interpolates that solution onto the 23-state grid.
   The output is saved as `<results_dir>/<circuit>_<AeroConfig>_ATD<On/Off>_EM4<On/Off>.mat`, a
   self-contained struct (states, inputs, time, cartesian racing line, boundaries, per-tyre forces,
   lap time) so the racing line can be redrawn without re-solving.

The transcription (Legendre collocation, default step `OPT_ds=30` m, degree `OPT_d=3`) and the
NLP assembly live in `functions/transcription.py`.

---

## Co-optimization (design + racing line)

Solve for static design parameters **jointly** with the trajectory:

```powershell
# Optimise brake bias, torque distribution, and aero AoAs together with the racing line:
python -c "from MLTP_paramOptim import MLTP_paramOptim; MLTP_paramOptim(circuit='Sturn')"

# Optimise only the tyre nominal-load shift (Fz0_shift):
python -c "from MLTP_TyreOptim import MLTP_TyreOptim; MLTP_TyreOptim(circuit='Sturn')"
```

- `MLTP_paramOptim` promotes `vp` fields (default: `brkB`, `Tdist`, `alpha_FL/FR/RW/TW`) to
  constant-over-lap decision variables; `MLTP_TyreOptim` is a thin wrapper promoting only
  `Fz0_shift`. The optimal values are saved on `ctx.data.optimal_params` and written to
  `<results_dir>/<circuit>_<tag>.mat` (`tag` = `paramOptim` / `TyreOptim`).

---

## Linear solver (HSL / MUMPS)

IPOPT defaults to the HSL **`ma57`** solver (run with MC64 auto-scaling so it converges on the
stiff 23-state problem). To enable HSL, point the framework at a **Coin-HSL** library by setting
`COINHSL_DIR` to its `bin/` folder (the MinGW / `libgfortran5` `CoinHSL_jll` build matches the
CasADi wheel's ABI):

```powershell
setx COINHSL_DIR "D:\path\to\CoinHSL\bin"     # new shells pick it up
```

If no working Coin-HSL library is found, the solve **transparently falls back to MUMPS** (bundled
in the CasADi wheel, no external library) — so a solve never crashes on a missing DLL. Choose a
solver per call with `linear_solver=` (`ma27`/`ma57`/`ma97`/`mumps`). Compare them with:

```powershell
python bench_linear_solver.py     # reports per-iteration linear-solver cost (MA57 vs MUMPS)
```

---

## Batch parameter sweeps (MPI)

`run_sweep.py` runs many **independent** `MLTP()` solves in parallel across MPI ranks (it
parallelises the *sweep*, not a single solve). Define the sweep in a CSV — one row per solve,
columns map to `MLTP()` keyword arguments:

```csv
case_id,circuit,vi,ATD,Electric_4Motors,AeroConfig,TyreModel,linear_solver
0,BCN,40,On,Off,Static,CombinedSlip,ma57
1,BCN,60,On,Off,Static,CombinedSlip,ma57
3,Spa,60,Off,On,Static,CombinedSlip,ma57
```

Accepted columns: `circuit, vi, ni, warm_start, AeroConfig, ATD, Electric_4Motors, TyreModel,
linear_solver` (plus an optional `case_id`). Run it:

```powershell
python run_sweep.py cases.csv                 # single process: runs every case serially
```

```bash
srun python run_sweep.py cases.csv            # Slurm: rank 0 coordinates, ranks 1+ solve
mpirun -np 32 python run_sweep.py cases.csv   # generic MPI
```

With a single rank (or no `mpi4py`) it runs every case serially in-process — handy for a
login-node check. Each case writes `Results/<sweep>/case_<id>/<circuit>_<config>.mat` plus a
run-level `Results/<sweep>/manifest.csv` (one row per case: inputs + `status, return_status,
iter_count, lap_time, init_s, solve_s, wall_s, out_path`). **Resume** is automatic — re-running
skips cases already marked `status=ok` (use `--no-resume` to force a full re-run; `--name` to set
the sweep name). On HPC, build `mpi4py` against the cluster MPI:

```bash
module load openmpi                       # or mpich
pip install --no-binary mpi4py mpi4py
```

See [`docs/mpi_sweep.md`](docs/mpi_sweep.md) for the full HPC launch/resume guide.

---

## Outputs

- **`Results/`** — solution `.mat` files (and sweep manifests).
- **`Plots/<circuit>/<config>/`** — interactive Plotly HTML (racing line, speed, tyre forces,
  friction circle, suspension, powertrain, inputs), written when `plot=True`. The friction-circle
  and g-g diagrams (`friction_circle_gg.html`, `gg_diagram.html`) come from `gg_plots.py`.

---

## Tests

There is no test runner — the `test_*.py` files are plain scripts whose assertions run at module
top level. Run each directly:

```powershell
python test_foundation.py        # collocation, geometry, simpleMA, Powertrain constants
python test_transcription.py     # discretise / pack / unpack round-trip
python test_save_load.py         # .mat save -> load_solution round-trip
python test_params_useropts.py   # vehParams + userOpts config branching
python test_sweep.py             # sweep helpers + run_sweep.py serial path
python test_hsl.py               # Coin-HSL resolve/probe/fallback (one section runs IPOPT)
```

All but `test_hsl.py` are CasADi-free (`test_hsl.py`'s solver section needs CasADi + IPOPT, and its
real-DLL check is skipped when no Coin-HSL library is present).

---

## Repository layout

| Path | What |
|------|------|
| `MLTP.py`               | full 23-state solve (the canonical entry point) |
| `MLTP_initial.py`       | 7-state warm-start solve |
| `MLTP_paramOptim.py` / `MLTP_TyreOptim.py` | design / racing-line co-optimization |
| `run_sweep.py` / `cases.csv` | MPI batch-sweep driver + example case list |
| `vehModel.py` / `vehModel_initial.py` | full / reduced symbolic vehicle models |
| `userOpts.py`           | configuration, track loading, IPOPT options |
| `Powertrain.py` / `vehParams.py` | powertrain ratings / vehicle + tyre + aero parameters |
| `plotSDI.py` / `gg_plots.py` | Plotly figure generation |
| `bench_linear_solver.py`| HSL vs MUMPS benchmark |
| `functions/`            | transcription engine, collocation, HSL support, sweep helpers, `.mat` I/O, post-processing geometry |
| `Circuits/` / `Data/` / `Results/` / `Plots/` | track `.mat` / aero data / outputs / figures |
| `docs/mpi_sweep.md`     | HPC sweep launch/resume guide |

> **Note (Linux/HPC):** the track folder is committed in lowercase as `circuits/`, while the code
> defaults to `circuits_dir='Circuits'`. These resolve identically on case-insensitive
> Windows/macOS but differ on Linux — pass `circuits_dir='circuits'` (or rename) there. The
> `app/`, `build/`, and `dist/` folders are local-only build artifacts, not part of the framework.
