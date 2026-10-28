"""Headless solve entry. Invoked by the GUI as a subprocess
(`<exe> --headless cfg.json`, or `python headless_solve.py cfg.json` in dev).
Reads cfg.json, runs MLTP, streams IPOPT output to stdout, exits.
The cfg key "mode": "qss" runs MLTP_screen (the QSS lap estimate) instead of MLTP.
"""
import json
import os
import sys

from app.paths import resource_root

# cfg key -> default. Each default is the MLTP or userOpts default. A key that is absent, null or at its
# default adds no argument, so an old cfg.json makes the call that it made before. build_solve_kwargs forwards
# any other value.
FORWARD = {
    "ladder": "auto", "homotopy": None, "refine": None, "continue_from": None,        # MLTP
    "ipopt_overrides": None, "physics": "matlab", "start_state": "matlab",              # userOpts
    "n_bounds": None, "Steering": "NA", "CamberGain": "Off",
    "between_knots": "matlab", "collocation": "legendre", "state_scaling": "matlab", "atd_tbr": None,
}
# MLTP_screen has no NLP and no plots. It takes none of these (userOpts raises TypeError on them).
NLP_ONLY = ("warm_start", "plots_dir", "ladder", "homotopy", "refine", "continue_from")


def run_mode(cfg):
    mode = cfg.get("mode", "nlp")
    if mode not in ("nlp", "qss"):
        raise ValueError(f"mode must be 'nlp' or 'qss', got {mode!r}")
    return mode


def build_solve_kwargs(cfg, resource_root_dir):
    """Pure: cfg dict -> the kwargs of MLTP(), or of MLTP_screen() if the cfg mode is 'qss'. No I/O, no solving.
    A bad mode raises ValueError."""
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
    # speed / fidelity options -> userOpts (same defaults as userOpts)
    kwargs["mesh"] = str(cfg.get("mesh", "auto"))
    kwargs["mesh_opts"] = dict(cfg["mesh_opts"]) if cfg.get("mesh_opts") else None
    kwargs["tyre_set"] = str(cfg.get("tyre_set", "MF205"))
    kwargs["screening"] = bool(cfg.get("screening", False))
    if cfg.get("warm_start"):
        kwargs["warm_start"] = cfg["warm_start"]
    for key, default in FORWARD.items():
        if cfg.get(key) is not None and cfg[key] != default:
            kwargs[key] = cfg[key]
    if run_mode(cfg) == "qss":
        for key in NLP_ONLY:
            kwargs.pop(key, None)
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
        if run_mode(cfg) == "qss":
            from MLTP_screen import MLTP_screen as solve     # no NLP and no IPOPT
        else:
            from MLTP import MLTP as solve                   # imported here so the unit test stays casadi-free
        solve(**kwargs)
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
