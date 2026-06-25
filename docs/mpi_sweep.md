# Running an MLTP sweep with MPI

`run_sweep.py` distributes many independent `MLTP()` solves across MPI ranks
(master/worker). IPOPT itself is not MPI-parallel, so this parallelises the
*sweep*, not a single solve. One core per rank; threads are pinned to 1.

## Define the sweep

Edit `cases.csv` — one row per solve. Columns map to `MLTP()` kwargs; blanks
fall back to defaults. Accepted columns: `case_id, circuit, vi, ni, warm_start,
AeroConfig, ATD, Electric_4Motors, TyreModel, linear_solver`. Keep
`linear_solver=ma57` (single-threaded, fast factorisation) for one-core ranks.

Do not change the column set partway through a sweep you intend to resume — the
manifest header is written once from the first run's columns.

## Install mpi4py against the cluster MPI

Do NOT rely on a generic wheel — build it against the node's MPI:

    module load openmpi        # or mpich, per your cluster
    pip install --no-binary mpi4py mpi4py

`mpi4py` is intentionally not in `requirements.txt` (HPC-only).

## Launch

    srun python run_sweep.py cases.csv          # Slurm
    mpirun -np 32 python run_sweep.py cases.csv  # generic MPI

Rank 0 coordinates; ranks 1+ solve. Run on N ranks to keep N-1 solves busy.
Single-rank (no mpirun) runs every case serially — handy for a login-node check.

Set `COINHSL_DIR` so `ma57` loads on the node (otherwise it falls back to MUMPS).

## Outputs

- `Results/<name>/case_<id>/<circuit>_<config>.mat` — per-case solution.
- `Results/<name>/manifest.csv` — one row per case: inputs + `status`,
  `return_status`, `iter_count`, `lap_time`, `init_s`, `solve_s`, `wall_s`,
  `out_path`. `<name>` is the CSV stem or `--name`.

## Resume

Re-submitting the same command skips cases already marked `status=ok` in the
manifest (so a wall-time-killed job continues). `error` rows are retried. Pass
`--no-resume` to force a full re-run.
