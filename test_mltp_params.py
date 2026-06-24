"""MLTP exposes plots_dir and forwards useropts_kwargs to the warm-start call.
Signature-level checks only (no solve)."""
import os, sys, inspect
sys.path.insert(0, os.path.dirname(__file__))
import MLTP as mltp_mod

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

sig = inspect.signature(mltp_mod.MLTP)
ok("MLTP has plots_dir param", "plots_dir" in sig.parameters)
ok("plots_dir default 'Plots'", sig.parameters["plots_dir"].default == "Plots")
ok("MLTP accepts **useropts_kwargs",
   any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))

# the warm-start call must forward kwargs (so vp_overrides reach the init model)
src = inspect.getsource(mltp_mod.MLTP)
ok("warm-start call forwards **useropts_kwargs",
   "MLTP_initial(" in src and "**useropts_kwargs" in src.split("MLTP_initial(")[1].split(")")[0])
ok("plotSDI uses plots_dir", "save_dir=plots_dir" in src)
print("\nALL MLTP param TESTS PASSED")
