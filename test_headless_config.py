"""build_solve_kwargs maps a cfg dict -> MLTP kwargs (absolute resource/output
dirs, ni-null -> nan, overrides passed through). No solving."""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
from headless_solve import build_solve_kwargs

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

cfg = {
    "circuit": "Sturn", "AeroConfig": "Static", "ATD": "On",
    "Electric_4Motors": "Off", "TyreModel": "CombinedSlip",
    "vi": 60.0, "ni": None, "linear_solver": "mumps",
    "save": True, "plot": True, "output_dir": "/out",
    "vp_overrides": {"mb": 2000.0},
}
kw = build_solve_kwargs(cfg, "/res")
ok("circuit mapped", kw["circuit"] == "Sturn")
ok("ni null -> nan", isinstance(kw["ni"], float) and math.isnan(kw["ni"]))
ok("results_dir under output", kw["results_dir"] == os.path.join("/out", "Results"))
ok("plots_dir under output", kw["plots_dir"] == os.path.join("/out", "Plots"))
ok("circuits_dir under resource", kw["circuits_dir"] == os.path.join("/res", "Circuits"))
ok("data_dir under resource", kw["data_dir"] == os.path.join("/res", "Data"))
ok("linear_solver passed", kw["linear_solver"] == "mumps")
ok("vp_overrides passed", kw["vp_overrides"] == {"mb": 2000.0})

cfg2 = dict(cfg, ni=0.3)
ok("numeric ni preserved", build_solve_kwargs(cfg2, "/res")["ni"] == 0.3)

# solver/collocation forwarding — defaults when absent
ok("max_iter default", kw["max_iter"] == 6000 and isinstance(kw["max_iter"], int))
ok("OPT_ds default", kw["OPT_ds"] == 30.0 and isinstance(kw["OPT_ds"], float))
ok("OPT_d default", kw["OPT_d"] == 3 and isinstance(kw["OPT_d"], int))
ok("OPT_e default", kw["OPT_e"] == 1e-2)
ok("tol default", kw["tol"] == 1e-4)

cfg3 = dict(cfg, max_iter=3000, OPT_ds=20, OPT_d=4, OPT_e=5e-3, tol=1e-6)
kw3 = build_solve_kwargs(cfg3, "/res")
ok("max_iter forwarded", kw3["max_iter"] == 3000)
ok("OPT_ds forwarded", kw3["OPT_ds"] == 20.0)
ok("OPT_d forwarded", kw3["OPT_d"] == 4)
ok("OPT_e forwarded", kw3["OPT_e"] == 5e-3)
ok("tol forwarded", kw3["tol"] == 1e-6)
print("\nALL headless config TESTS PASSED")
