"""Benchmark IPOPT linear solvers on the full 23-state MLTP.

Solves the same OCP once per linear solver and reports iteration count, the
isolated 23-state solve time (excludes the 7-state warm start), and the split
between NLP function-evaluation time and IPOPT-internal time (dominated by the
linear-solver factorisations). Confirms the linear solver is the bottleneck and
quantifies the Coin-HSL speed-up over MUMPS.

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
    return {
        "solver": solver,
        "status": st.get("return_status", "?"),
        "iters": st.get("iter_count", "?"),
        "solve_s": float(ctx.elapsed.get("solve", float("nan"))),
        "func_s": func,
        "ipopt_s": (total - func) if total == total else float("nan"),  # NaN-safe
        "lap_s": float(ctx.data.lap_time),
    }


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
            results.append({"solver": ls, "status": f"EXC:{exc}", "iters": "-",
                            "solve_s": float("nan"), "func_s": float("nan"),
                            "ipopt_s": float("nan"), "lap_s": float("nan")})

    print("\n" + "=" * 78)
    print(f"{'solver':8} {'status':18} {'iters':>6} {'solve_s':>9} "
          f"{'func_s':>9} {'ipopt_s':>9} {'lap_s':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['solver']:8} {str(r['status'])[:18]:18} {str(r['iters']):>6} "
              f"{r['solve_s']:>9.2f} {r['func_s']:>9.2f} {r['ipopt_s']:>9.2f} "
              f"{r['lap_s']:>8.3f}")
    print("=" * 78)

    base = next((r for r in results if r["solver"] == "mumps"
                 and r["solve_s"] == r["solve_s"]), None)
    if base:
        for r in results:
            if r["solver"] != "mumps" and r["solve_s"] == r["solve_s"]:
                spd = base["solve_s"] / r["solve_s"] if r["solve_s"] else float("nan")
                print(f"  {r['solver']} speed-up vs mumps: {spd:.2f}x")
    return results


if __name__ == "__main__":
    bench(circuit="Sturn", solvers=("mumps", "ma57"))
