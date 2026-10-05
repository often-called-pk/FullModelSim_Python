"""vehParams two-phase refactor: overrides propagate to derived quantities;
empty overrides reproduce defaults; unknown keys raise; mf overrides apply.
Also checks the override reaches the model via userOpts."""
import os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from functions.context import Ctx
from vehParams import vehParams, default_primaries, PRIMARY_KEYS, MF_KEYS
from userOpts import userOpts

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# --- defaults reproduced with no overrides ---
ctx = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx)
vp = ctx.vp
ok("default ms = 1895", vp.ms == 1895.0)
ok("default m = 2085", vp.m == 2085.0)
ok("default Rw_f=Rw_r=0.355", vp.Rw_f == 0.355 and vp.Rw_r == 0.355)
ok("default c_fl formula", abs(vp.c_fl - 0.7 * 2 * np.sqrt((vp.m*(1-vp.wB))/2) * 75000.0) < 1e-6)

# --- primaries metadata ---
ok("PRIMARY_KEYS has mb,kt,brkB", {"mb", "kt", "brkB"} <= PRIMARY_KEYS)
ok("default_primaries mb=1820", default_primaries()["mb"] == 1820.0)

# --- override a primary -> derived recompute ---
ctx2 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx2, vp_overrides={"mb": 2000.0})
vp2 = ctx2.vp
ok("override mb -> ms = 2075", vp2.ms == 2075.0)
ok("override mb -> m = 2265", vp2.m == 2265.0)
ok("override mb -> c_fl uses new m_eff",
   abs(vp2.c_fl - 0.7 * 2 * np.sqrt((2265.0*(1-vp2.wB))/2) * 75000.0) < 1e-6)

# --- override a spring -> damper recompute ---
ctx3 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx3, vp_overrides={"k_fl": 100000.0})
ok("override k_fl -> c_fl uses new k",
   abs(ctx3.vp.c_fl - 0.7 * 2 * np.sqrt((ctx3.vp.m*(1-ctx3.vp.wB))/2) * 100000.0) < 1e-6)

# --- mf override ---
ctx4 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx4, vp_overrides={"pKy1": -19.0})
ok("mf override applied", ctx4.mf.pKy1 == -19.0)
ok("MF_KEYS has pKy1", "pKy1" in MF_KEYS)

# --- unknown key raises ---
ctx5 = Ctx()
raised = False
try:
    vehParams(ctx5, vp_overrides={"not_a_param": 1.0})
except ValueError:
    raised = True
ok("unknown override key raises ValueError", raised)

# --- reaches model via userOpts ---
ctx6 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    userOpts(ctx6, circuit="Sturn", vp_overrides={"mb": 2000.0})
ok("userOpts threads vp_overrides", ctx6.vp.ms == 2075.0)

print("\nALL vp_overrides TESTS PASSED")
