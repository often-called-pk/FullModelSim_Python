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

# solver/collocation forwarding - defaults when absent
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

# speed / fidelity options (mesh, mesh_opts, tyre_set, screening) - userOpts defaults when absent
ok("mesh default auto", kw["mesh"] == "auto")
ok("mesh_opts default None", kw["mesh_opts"] is None)
ok("tyre_set default MF205", kw["tyre_set"] == "MF205")
ok("screening default False", kw["screening"] is False)

mo = {"a": 2.0, "ds_max": 60.0}
cfg4 = dict(cfg, mesh="curvature", mesh_opts=mo, tyre_set="CopyB", screening=True)
kw4 = build_solve_kwargs(cfg4, "/res")
ok("mesh forwarded", kw4["mesh"] == "curvature")
ok("mesh_opts forwarded (copy)", kw4["mesh_opts"] == mo and kw4["mesh_opts"] is not mo)
ok("tyre_set forwarded", kw4["tyre_set"] == "CopyB")
ok("screening forwarded", kw4["screening"] is True)
ok("empty / null mesh_opts -> None",
   build_solve_kwargs(dict(cfg, mesh_opts={}), "/res")["mesh_opts"] is None
   and build_solve_kwargs(dict(cfg, mesh_opts=None), "/res")["mesh_opts"] is None)

# every forwarded key is a real userOpts / MLTP argument (userOpts only, no casadi / MLTP import)
import inspect
from userOpts import userOpts
_uo_args = set(inspect.signature(userOpts).parameters)
ok("mesh / mesh_opts / tyre_set / screening are userOpts arguments",
   {"mesh", "mesh_opts", "tyre_set", "screening"} <= _uo_args)
print("\nALL headless config TESTS PASSED")
