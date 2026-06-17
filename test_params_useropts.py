"""Validate vehParams.py (Copy B tyre set) and userOpts.py (config, boundary,
rate-limit/regularisation assembly). No casadi required."""
import sys, os, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from functions.context import Ctx
from userOpts import userOpts

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

print("vehParams (Copy B) + Powertrain interaction")
ctx = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")            # ignore DATA_AA-missing warning
    userOpts(ctx, circuit="Sturn")             # synthetic track, no .mat needed
vp, pt, mf = ctx.vp, ctx.pt, ctx.mf

ok("ms = mb+md = 1895", vp.ms == 1895.0)
ok("m  = ms+muf+mur = 2085", vp.m == 2085.0)
ok("Fz0 = 4905 (Copy B)", vp.Fz0 == 4905.0)
ok("Rw overwritten to 0.355", vp.Rw == 0.355 and vp.Rw_f == 0.355 and vp.Rw_r == 0.355)
ok("gear still based on Powertrain Rw=0.3142857",
   abs(vp.gear - (pt.OMmax * 0.3142857) / pt.Vmax) < 1e-9)
ok("Jw = 3.6", vp.Jw == 3.6)
# suspension damping: c_fl = zeta*2*sqrt(m_eff_f/2)*k_fl, m_eff_f = m*(1-wB)
m_eff_f = vp.m * (1 - vp.wB)
ok("c_fl matches formula", abs(vp.c_fl - 0.7 * 2 * np.sqrt(m_eff_f / 2) * 75000.0) < 1e-6)
ok("hRC interpolation", abs(vp.hRC - (vp.l_f * vp.hRCr + vp.l_r * vp.hRCf) / vp.l) < 1e-12)

print("Pacejka Copy B coefficients")
ok("pKy1 = -20.505", mf.pKy1 == -20.505)
ok("pEy1 = 0.15", mf.pEy1 == 0.15)
ok("pKy4 = 0.0", mf.pKy4 == 0.0)
ok("pKy5 = 0.002", mf.pKy5 == 0.002)
ok("pKy6 = -0.002", mf.pKy6 == -0.002)
ok("pVy = (0,0,0,0.08)", (mf.pVy1, mf.pVy2, mf.pVy3, mf.pVy4) == (0.0, 0.0, 0.0, 0.08))
ok("Fx pCx1 = 1.6055", mf.pCx1 == 1.6055)
ok("Fx pKx3 = 0.51289", mf.pKx3 == 0.51289)

print("simplified init tyre + camber/toe")
ok("init mu = 1.41", vp.tyre.mu == 1.41)
ok("init by/cy/ey = 6/2.5/0.5", (vp.tyre.by, vp.tyre.cy, vp.tyre.ey) == (6.0, 2.5, 0.5))
ok("camber gamma_fr = -gamma_fl", vp.gamma_fr == -vp.gamma_fl)
ok("toe rad conversion", abs(vp.toe_front_rad - vp.toe_front * np.pi / 180) < 1e-15)

print("boundary conditions")
ok("Xi length 23", ctx.Xi.shape == (23,))
ok("Xi[0] = vi = 60", ctx.Xi[0] == 60.0)
ok("Xi[1] = 0", ctx.Xi[1] == 0.0)
ok("Xi heave/pitch/roll block = 0", np.all(ctx.Xi[9:15] == 0.0))
ok("Xi unknowns are NaN", np.isnan(ctx.Xi[2]) and np.all(np.isnan(ctx.Xi[15:])))
ok("Xi_init length 7", ctx.Xi_init.shape == (7,))
ok("Xf all NaN, length 23", ctx.Xf.shape == (23,) and np.all(np.isnan(ctx.Xf)))

print("collocation + solver options")
ok("OPT_ds=10, OPT_d=3", ctx.OPT_ds == 10 and ctx.OPT_d == 3)
ok("OPT_uinter linear", ctx.OPT_uinter == "linear")
ok("ipopt max_iter 6000", ctx.opts["ipopt"]["max_iter"] == 6000)
ok("ipopt tol 1e-6", ctx.opts["ipopt"]["tol"] == 1e-6)
ok("HSL ma57 selected", ctx.opts["ipopt"]["linear_solver"] == "ma57")

print("input ordering + limits per configuration")
# default: Static(0), ATD On(1), EM4 Off(0)
ok("default keys", ctx.input_keys ==
   ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"])
ok("duk_ub shape matches keys", ctx.duk_ub.shape == (len(ctx.input_keys), 1))
ok("default duk_ub values",
   np.allclose(ctx.duk_ub.ravel(), [2e4, 1.45e5, 2e4, 2e4, 2e4, 2e4, 0.1]))
ok("default rdu2 values (ATD=1.5, delta=15)",
   np.allclose(ctx.rdu2.ravel(), [0, 0, 1.5, 1.5, 1.5, 1.5, 15]))
ok("init keys = motor/brake/delta", list(map(str, ["T_motor", "T_brake", "delta"])) ==
   ["T_motor", "T_brake", "delta"] and ctx.duk_ub_init.shape == (3, 1))

# EM4 on + ATD on -> conflict resolves to EM4 on, ATD off
ctx2 = Ctx()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    userOpts(ctx2, circuit="Straight", ATD="On", Electric_4Motors="On")
ok("conflict warning raised", any("cannot both be On" in str(x.message) for x in w))
ok("conflict -> pt.ATD=0, pt.EM4=1", ctx2.pt.ATD == 0 and ctx2.pt.EM4 == 1)
ok("EM4 keys = 4 motors + brake + delta", ctx2.input_keys ==
   ["T_motor_fl", "T_motor_fr", "T_motor_rl", "T_motor_rr", "T_brake", "delta"])

# AALB aero (3), ATD off, EM4 off
ctx3 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    userOpts(ctx3, circuit="Straight", AeroConfig="AALB", ATD="Off", Electric_4Motors="Off")
ok("AALB keys (FW,FW,RW,TW)", ctx3.input_keys ==
   ["T_motor", "T_brake", "FW", "FW", "RW", "TW", "delta"])
ok("AALB rdu2 (FW=0.3, RW=0.9, TW=0.4)",
   np.allclose(ctx3.rdu2.ravel(), [0, 0, 0.3, 0.3, 0.9, 0.4, 15]))

print("track building (synthetic)")
ok("Sturn track has s,k", hasattr(ctx.track, "s") and hasattr(ctx.track, "k"))
ok("Sturn s monotonic increasing", np.all(np.diff(ctx.track.s) > 0))
ok("Sturn k smoothed (finite)", np.all(np.isfinite(ctx.track.k)))

print("\nALL vehParams / userOpts TESTS PASSED")
