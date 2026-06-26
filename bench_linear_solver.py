"""Benchmark IPOPT linear solvers on the full 23-state MLTP.

Solves the same OCP once per linear solver and reports iteration count, isolated
23-state solve time (excludes the 7-state warm start), the NLP-eval vs
IPOPT-internal time split, and IPOPT-internal time *per iteration*.

`ip_ms/it` is the apples-to-apples metric: total wall-clock is confounded because
this nonconvex OCP has multiple local optima, so solvers take different iteration
counts to converge. Cost *per iteration* isolates the linear solver and is the
honest way to quantify the Coin-HSL speed-up over MUMPS.

Run from the repo root:

    venv\\Scripts\\python.exe bench_linear_solver.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from MLTP import MLTP

# NLP callback timing keys reported by CasADi's ipopt stats.
_CB_KEYS = ("t_wall_nlp_f", "t_wall_nlp_g", "t_wall_nlp_grad_f",
            "t_wall_nlp_jac_g", "t_wall_nlp_hess_l")


def _row(solver, ctx):
    st = getattr(ctx, "solve_stats", {}) or {}
    total = float(st.get("t_wall_total", float("nan")))
    func = sum(float(st.get(k, 0.0)) for k in _CB_KEYS)
    iters = st.get("iter_count", "?")
    ipopt_s = (total - func) if total == total else float("nan")  # NaN-safe
    # IPOPT-internal time per iteration (ms): dominated by the linear-solver
    # factorisation + back-solve, so it isolates the linear solver from the
    # iteration count (which differs across local optima).
    try:
        ip_ms_it = 1000.0 * ipopt_s / iters if iters else float("nan")
    except (TypeError, ZeroDivisionError):
        ip_ms_it = float("nan")
    return {
        "solver": solver,
        "status": st.get("return_status", "?"),
        "iters": iters,
        "solve_s": float(ctx.elapsed.get("solve", float("nan"))),
        "func_s": func,
        "ipopt_s": ipopt_s,
        "ip_ms_it": ip_ms_it,
        "lap_s": float(ctx.data.lap_time),
    }


def _nan_row(solver, status):
    """Placeholder row for a solver whose solve raised."""
    return {"solver": solver, "status": status, "iters": "-",
            "solve_s": float("nan"), "func_s": float("nan"),
            "ipopt_s": float("nan"), "ip_ms_it": float("nan"),
            "lap_s": float("nan")}


def _print_table(results):
    """Print the comparison table plus total and per-iteration speed-ups."""
    print("\n" + "=" * 88)
    print(f"{'solver':8} {'status':18} {'iters':>6} {'solve_s':>9} "
          f"{'func_s':>9} {'ipopt_s':>9} {'ip_ms/it':>9} {'lap_s':>8}")
    print("-" * 88)
    for r in results:
        print(f"{r['solver']:8} {str(r['status'])[:18]:18} {str(r['iters']):>6} "
              f"{r['solve_s']:>9.2f} {r['func_s']:>9.2f} {r['ipopt_s']:>9.2f} "
              f"{r['ip_ms_it']:>9.2f} {r['lap_s']:>8.3f}")
    print("=" * 88)

    base = next((r for r in results if r["solver"] == "mumps"
                 and r["solve_s"] == r["solve_s"]), None)
    if not base:
        return
    for r in results:
        if r["solver"] == "mumps" or r["solve_s"] != r["solve_s"]:
            continue
        spd = base["solve_s"] / r["solve_s"] if r["solve_s"] else float("nan")
        print(f"  {r['solver']} total solve_s vs mumps: {spd:.2f}x")
        # Per-iteration linear-solver speed-up -- the apples-to-apples figure;
        # the total above is confounded by differing iteration counts.
        if (base["ip_ms_it"] == base["ip_ms_it"]
                and r["ip_ms_it"] == r["ip_ms_it"] and r["ip_ms_it"]):
            spd_it = base["ip_ms_it"] / r["ip_ms_it"]
            print(f"  {r['solver']} linear-solver cost/iter vs mumps: "
                  f"{spd_it:.2f}x faster")


def bench(circuit="Sturn", solvers=("mumps", "ma57", "ma97"), **mltp_kwargs):
    results = []
    for ls in solvers:
        print(f"\n===== linear_solver = {ls} =====")
        try:
            ctx = MLTP(circuit=circuit, linear_solver=ls,
                       save=False, plot=False, **mltp_kwargs)
            results.append(_row(ls, ctx))
        except Exception as exc:
            print(f"  {ls}: FAILED -> {exc}")
            results.append(_nan_row(ls, f"EXC:{exc}"))

    _print_table(results)
    return results


if __name__ == "__main__":
    bench(circuit="Sturn", solvers=("mumps", "ma57"))
