#!/usr/bin/env python
"""MPI master/worker driver: run a sweep of independent MLTP() solves.

Launch one rank per core, e.g.:
    srun python run_sweep.py cases.csv
    mpirun -np 32 python run_sweep.py cases.csv

Rank 0 is the manager (hands cases to workers, writes the manifest); ranks 1+
are workers (each runs one MLTP solve at a time). With a single rank it falls
back to running every case serially in-process, so it works without mpirun too.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pin every numerical library to one thread BEFORE importing casadi/MLTP, so N
# ranks use N cores without oversubscription. setdefault lets the env override.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from functions.sweep import (read_cases, pending_cases, case_kwargs,
                             manifest_fieldnames, manifest_row, run_case)

_TAG_WORK = 2     # manager -> worker: here is a case
_TAG_STOP = 3     # manager -> worker: no more work, exit


def _parse_args(argv):
    p = argparse.ArgumentParser(description="MPI sweep of MLTP solves")
    p.add_argument("cases", help="path to the cases CSV")
    p.add_argument("--name", default=None,
                   help="sweep name (default: cases-file stem)")
    p.add_argument("--no-resume", action="store_true",
                   help="re-run cases even if already completed")
    return p.parse_args(argv)


def _sweep_name(args):
    return args.name or os.path.splitext(os.path.basename(args.cases))[0]


def _manifest_path(sweep_name):
    return os.path.join("Results", sweep_name, "manifest.csv")


def _solve(**kwargs):
    """Real per-case solve. MLTP is imported lazily so only workers pay the
    casadi import, and the pure module/tests never need it."""
    from MLTP import MLTP
    return MLTP(**kwargs)


def _open_manifest(path, fieldnames):
    """Open the manifest for appending; write the header iff the file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.isfile(path)
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="",
                            extrasaction="ignore")
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def _load_cases(cases_path, manifest_path, resume):
    """Read cases, fail fast on a bad column (spec: validate before any MPI
    work starts), then drop already-completed cases when resuming."""
    cases = read_cases(cases_path)
    for case in cases:
        case_kwargs(case)        # raises ValueError on an unknown column
    return pending_cases(cases, manifest_path, resume=resume)


def _run_serial(cases, sweep_name, manifest_path):
    """Run every case in this process (1-rank fallback / debugging)."""
    fh, writer = _open_manifest(manifest_path, manifest_fieldnames(cases))
    try:
        for case in cases:
            result = run_case(case, sweep_name, _solve, time.perf_counter)
            writer.writerow(manifest_row(case, result))
            fh.flush()
            print(f"[serial] case {case['case_id']}: {result['status']} "
                  f"{result.get('return_status', '')}")
    finally:
        fh.close()


def _run_manager(comm, cases, sweep_name, manifest_path):
    """Rank 0: dispatch cases to workers, write each returned result."""
    from mpi4py import MPI
    fh, writer = _open_manifest(manifest_path, manifest_fieldnames(cases))
    queue = list(cases)
    stopped = 0
    n_workers = comm.Get_size() - 1
    try:
        while stopped < n_workers:
            status = MPI.Status()
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                            status=status)
            src = status.Get_source()
            if msg is not None:                      # a completed (case, result)
                case, result = msg
                writer.writerow(manifest_row(case, result))
                fh.flush()
                print(f"[mgr] case {case['case_id']}: {result['status']} "
                      f"{result.get('return_status', '')}")
            if queue:
                comm.send(queue.pop(0), dest=src, tag=_TAG_WORK)
            else:
                comm.send(None, dest=src, tag=_TAG_STOP)
                stopped += 1
    finally:
        fh.close()


def _run_worker(comm, sweep_name):
    """Rank >=1: ask for work, solve, report, until told to stop."""
    from mpi4py import MPI
    comm.send(None, dest=0, tag=_TAG_WORK)           # initial "ready"
    while True:
        status = MPI.Status()
        case = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == _TAG_STOP:
            break
        result = run_case(case, sweep_name, _solve, time.perf_counter)
        comm.send((case, result), dest=0, tag=_TAG_WORK)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    sweep_name = _sweep_name(args)
    manifest_path = _manifest_path(sweep_name)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size, rank = comm.Get_size(), comm.Get_rank()
    except Exception:
        comm, size, rank = None, 1, 0

    if size == 1:
        cases = _load_cases(args.cases, manifest_path,
                            resume=not args.no_resume)
        print(f"[serial] {len(cases)} case(s) to run (sweep '{sweep_name}')")
        _run_serial(cases, sweep_name, manifest_path)
    elif rank == 0:
        cases = _load_cases(args.cases, manifest_path,
                            resume=not args.no_resume)
        print(f"[mgr] {len(cases)} case(s) across {size - 1} worker(s)")
        _run_manager(comm, cases, sweep_name, manifest_path)
    else:
        _run_worker(comm, sweep_name)


if __name__ == "__main__":
    main()
