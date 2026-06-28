"""userOpts exposes solver/collocation kwargs; defaults unchanged."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
from functions.context import Ctx
from userOpts import userOpts

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ctx = userOpts(Ctx(), circuit="Sturn")
ok("default OPT_ds == 30", ctx.OPT_ds == 30)
ok("default OPT_d == 3", ctx.OPT_d == 3)
ok("default OPT_e == 1e-2", ctx.OPT_e == 1e-2)
ok("default max_iter == 6000", ctx.opts["ipopt"]["max_iter"] == 6000)
ok("default tol == 1e-4", ctx.opts["ipopt"]["tol"] == 1e-4)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ctx2 = userOpts(Ctx(), circuit="Sturn",
                    OPT_ds=20, OPT_d=4, OPT_e=5e-3, max_iter=3000, tol=1e-6)
ok("override OPT_ds", ctx2.OPT_ds == 20)
ok("override OPT_d", ctx2.OPT_d == 4)
ok("override OPT_e", ctx2.OPT_e == 5e-3)
ok("override max_iter", ctx2.opts["ipopt"]["max_iter"] == 3000)
ok("override tol", ctx2.opts["ipopt"]["tol"] == 1e-6)

print("\nALL userOpts solver-opt TESTS PASSED")
