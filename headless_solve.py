"""Headless solve entry. Invoked by the GUI as a subprocess
(`<exe> --headless cfg.json`, or `python headless_solve.py cfg.json` in dev).
Reads cfg.json, runs MLTP, streams IPOPT output to stdout, exits.
"""
import json
import os
import sys

from app.paths import resource_root


def build_solve_kwargs(cfg, resource_root_dir):
    """Pure: cfg dict -> MLTP() kwargs. No I/O, no solving."""
    output_dir = cfg["output_dir"]
    ni = cfg.get("ni", None)
    ni = float("nan") if ni is None else float(ni)
    kwargs = dict(
        circuit=cfg["circuit"],
        vi=float(cfg["vi"]),
        ni=ni,
        AeroConfig=cfg["AeroConfig"],
        ATD=cfg["ATD"],
        Electric_4Motors=cfg["Electric_4Motors"],
        TyreModel=cfg["TyreModel"],
        linear_solver=cfg.get("linear_solver", "ma57"),
        save=bool(cfg.get("save", True)),
        plot=bool(cfg.get("plot", True)),
        results_dir=os.path.join(output_dir, "Results"),
        plots_dir=os.path.join(output_dir, "Plots"),
        circuits_dir=os.path.join(resource_root_dir, "Circuits"),
        data_dir=os.path.join(resource_root_dir, "Data"),
        vp_overrides=cfg.get("vp_overrides") or None,
    )
    # solver / collocation options -> userOpts (via MLTP **useropts_kwargs)
    kwargs["max_iter"] = int(cfg.get("max_iter", 6000))
    kwargs["OPT_ds"] = float(cfg.get("OPT_ds", 30))
    kwargs["OPT_d"] = int(cfg.get("OPT_d", 3))
    kwargs["OPT_e"] = float(cfg.get("OPT_e", 1e-2))
    kwargs["tol"] = float(cfg.get("tol", 1e-4))
    if cfg.get("warm_start"):
        kwargs["warm_start"] = cfg["warm_start"]
    return kwargs


def main(argv):
    if not argv:
        print("usage: headless_solve.py <cfg.json>", file=sys.stderr)
        return 2
    try:
        with open(argv[0]) as fh:
            cfg = json.load(fh)
        os.makedirs(cfg["output_dir"], exist_ok=True)
        kwargs = build_solve_kwargs(cfg, resource_root())
        from MLTP import MLTP        # imported here so the unit test stays casadi-free
        MLTP(**kwargs)
        return 0
    except Exception:
        # Surface the failure in the captured stdout (the GUI log pane) and exit
        # non-zero rather than letting it escape: a frozen windowed (console=
        # False) exe would otherwise pop a modal traceback dialog from this
        # solve subprocess. flush() because stdout is block-buffered to the pipe.
        import traceback
        print("\n[headless_solve] solve failed:", file=sys.stdout)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
