"""Validate vehParams.py (MF205 default, legacy CopyB tyre sets) and
userOpts.py (config, boundary, rate-limit/regularisation assembly). No casadi
required."""
import sys, os, warnings
import numpy as np
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)

from functions.context import Ctx
from functions.transcription import discretise
from userOpts import userOpts, SCREENING_IPOPT, MESH_AUTO_MIN_LENGTH
from vehParams import vehParams, _default_mf, _MF205_OVERRIDES, MF_KEYS, TYRE_SETS

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

print("vehParams (MF205 default) + Powertrain interaction")
ctx = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")            # ignore DATA_AA-missing warning
    userOpts(ctx, circuit="Sturn")             # synthetic track, no .mat needed
vp, pt, mf = ctx.vp, ctx.pt, ctx.mf

ok("ms = mb+md = 1895", vp.ms == 1895.0)
ok("m  = ms+muf+mur = 2085", vp.m == 2085.0)
ok("Fz0 = 4905", vp.Fz0 == 4905.0)
ok("Rw overwritten to 0.355", vp.Rw == 0.355 and vp.Rw_f == 0.355 and vp.Rw_r == 0.355)
ok("gear still based on Powertrain Rw=0.3142857",
   abs(vp.gear - (pt.OMmax * 0.3142857) / pt.Vmax) < 1e-9)
ok("Jw = 3.6", vp.Jw == 3.6)
# suspension damping: c_fl = zeta*2*sqrt(m_eff_f/2)*k_fl, m_eff_f = m*(1-wB)
m_eff_f = vp.m * (1 - vp.wB)
ok("c_fl matches formula", abs(vp.c_fl - 0.7 * 2 * np.sqrt(m_eff_f / 2) * 75000.0) < 1e-6)
ok("hRC interpolation", abs(vp.hRC - (vp.l_f * vp.hRCr + vp.l_r * vp.hRCf) / vp.l) < 1e-12)

print("Pacejka MF205 coefficients (userOpts default tyre_set)")
ok("pKy1 = 20.505", mf.pKy1 == 20.505)
ok("pEy1 = 0.33443", mf.pEy1 == 0.33443)
ok("pKy4 = 2.0", mf.pKy4 == 2.0)
ok("pKy5 = 0.0", mf.pKy5 == 0.0)
ok("pKy6 = 0.0", mf.pKy6 == 0.0)
ok("pVy = (0.026365,-0.0062119,-0.41389,-0.048038)",
   (mf.pVy1, mf.pVy2, mf.pVy3, mf.pVy4) == (0.026365, -0.0062119, -0.41389, -0.048038))
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
ok("OPT_ds=30, OPT_d=3", ctx.OPT_ds == 30 and ctx.OPT_d == 3)
ok("OPT_uinter linear", ctx.OPT_uinter == "linear")
ok("ipopt max_iter 6000", ctx.opts["ipopt"]["max_iter"] == 6000)
ok("ipopt tol 1e-4", ctx.opts["ipopt"]["tol"] == 1e-4)
ok("default linear_solver ma57", ctx.opts["ipopt"]["linear_solver"] == "ma57")

print("input ordering + limits per configuration")
# default: Static(0), ATD On(1), EM4 Off(0)
ok("default keys", ctx.input_keys ==
   ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"])
ok("duk_ub shape matches keys", ctx.duk_ub.shape == (len(ctx.input_keys), 1))
ok("default duk_ub values",
   np.allclose(ctx.duk_ub.ravel(), [2e4, 1.45e5, 2e4, 2e4, 2e4, 2e4, 0.1]))
ok("default rdu2 values (ATD=1.5, delta=15)",
   np.allclose(ctx.rdu2.ravel(), [0, 0, 1.5, 1.5, 1.5, 1.5, 15]))
ok("init keys = motor/brake/delta (duk_ub_init / ru_init / rdu2_init)",
   ctx.duk_ub_init.shape == (3, 1) and ctx.ru_init.shape == (3, 1)
   and np.allclose(ctx.duk_ub_init.ravel(), [2e4, 1.45e5, 0.1])
   and np.allclose(ctx.rdu2_init.ravel(), [0, 0, 15]))

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

print("tyre_set: MF205 (default, MATLAB-selected MF_205_60R15_V91) vs CopyB (legacy, pKy4 = 0)")
def _vp_ctx(**kw):
    c = Ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vehParams(c, **kw)
    return c

def _kya(m, fz, fz0, gam=0.0):
    """Cornering stiffness exactly as vehModel.py builds it (lambdas = 1)."""
    return (m.pKy1 * fz0 * np.sin(m.pKy4 * np.arctan(fz / ((m.pKy2 + m.pKy5 * gam**2) * fz0)))
            / (1 + m.pKy3 * gam**2))

ok("TYRE_SETS = (CopyB, MF205)", TYRE_SETS == ("CopyB", "MF205"))
cD = _vp_ctx()                                  # no tyre_set kwarg -> default
ok("default tyre_set is MF205", cD.tyre_set == "MF205")
ok("default pKy4 == 2.0 (non-zero cornering stiffness)", cD.mf.pKy4 == 2.0)
ok("default pKy1 == 20.505", cD.mf.pKy1 == 20.505)
ok("default pEy1 == 0.33443", cD.mf.pEy1 == 0.33443)
ok("default pKy5 == 0.0", cD.mf.pKy5 == 0.0)
ok("default pVy1 == 0.026365", cD.mf.pVy1 == 0.026365)
ok("default mf == explicit tyre_set='MF205'", vars(cD.mf) == vars(_vp_ctx(tyre_set="MF205").mf))
cB = _vp_ctx(tyre_set="CopyB")
ok("tyre_set='CopyB' stored on ctx", cB.tyre_set == "CopyB")
ok("CopyB pKy4 == 0.0 (zero cornering stiffness, legacy)", cB.mf.pKy4 == 0.0)
ok("CopyB pKy1 == -20.505", cB.mf.pKy1 == -20.505)
ok("CopyB mf == _default_mf()", vars(cB.mf) == vars(_default_mf()))
ok("CopyB Kya(Fz0) == 0 exactly", _kya(cB.mf, cB.vp.Fz0, cB.vp.Fz0) == 0.0)

c205 = _vp_ctx(tyre_set="MF205")
m205 = c205.mf
ok("MF205 stored on ctx", c205.tyre_set == "MF205")
ok("MF205 pKy4 == 2.0", m205.pKy4 == 2.0)
ok("MF205 pKy1 == 20.505", m205.pKy1 == 20.505)
ok("MF205 pEy1 == 0.33443", m205.pEy1 == 0.33443)
ok("MF205 pKy5 == 0.0", m205.pKy5 == 0.0)
ok("MF205 pVy1 == 0.026365", m205.pVy1 == 0.026365)
ok("MF205 pKy6 == 0.0", m205.pKy6 == 0.0)
ok("MF205 pVy2,pVy3,pVy4 = -0.0062119,-0.41389,-0.048038",
   (m205.pVy2, m205.pVy3, m205.pVy4) == (-0.0062119, -0.41389, -0.048038))
ok("MF205 override keys are valid mf keys", set(_MF205_OVERRIDES) <= MF_KEYS)
changed = {k for k, v in vars(cB.mf).items() if vars(m205)[k] != v}
ok("MF205 changes exactly the override coefficients", changed == set(_MF205_OVERRIDES))
ok("MF205 leaves the primaries untouched",
   (c205.vp.ms, c205.vp.m, c205.vp.Fz0, c205.vp.Rw) == (cB.vp.ms, cB.vp.m, cB.vp.Fz0, cB.vp.Rw))
ok("MF205 Kya(Fz0) > 0 (about 8e4 N/rad)",
   7e4 < _kya(m205, c205.vp.Fz0, c205.vp.Fz0) < 9e4)

bad = Ctx()
raised = False
try:
    vehParams(bad, tyre_set="Slick")
except ValueError as e:
    raised = "Slick" in str(e)
ok("unknown tyre_set raises ValueError", raised)
ok("rejected call leaves ctx untouched", not hasattr(bad, "vp") and not hasattr(bad, "mf"))

cov = _vp_ctx(tyre_set="MF205", vp_overrides={"pKy4": 1.5, "pEy1": 0.2, "mb": 2000.0})
ok("vp_overrides win over MF205 (pKy4, pEy1)", cov.mf.pKy4 == 1.5 and cov.mf.pEy1 == 0.2)
ok("non-overridden MF205 values remain (pKy1)", cov.mf.pKy1 == 20.505)
ok("primary override still propagates with MF205 (ms = 2075)", cov.vp.ms == 2075.0)
ok("default userOpts(circuit='Sturn') lists no mf overrides", ctx.mf_overrides == [])
ok("mf_overrides lists exactly the mf keys that differ from MF205 (pEy1, pKy4; not mb)",
   cov.mf_overrides == ["pEy1", "pKy4"])
ok("an mf override equal to the base value (pKy4 = 2.0 under MF205) is not listed",
   _vp_ctx(vp_overrides={"pKy4": 2.0}).mf_overrides == [])
lat9 = {"pEy1": 0.15, "pKy1": -20.505, "pKy4": 0.0, "pKy5": 0.002, "pKy6": -0.002,
        "pVy1": 0.0, "pVy2": 0.0, "pVy3": 0.0, "pVy4": 0.08}
ok("the nine CopyB lateral values under MF205 list those nine keys, sorted",
   _vp_ctx(vp_overrides=lat9).mf_overrides
   == ["pEy1", "pKy1", "pKy4", "pKy5", "pKy6", "pVy1", "pVy2", "pVy3", "pVy4"])
cpo = _vp_ctx(vp_overrides={"mb": 2000.0})
ok("a primary override (mb) is applied but not listed in mf_overrides",
   cpo.vp.ms == 2075.0 and cpo.mf_overrides == [])

print("userOpts kwargs: mesh / mesh_opts, screening, tyre_set")
def _uo(circuit="Sturn", **kw):
    c = Ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        userOpts(c, circuit=circuit, **kw)
    return c

# mesh: default 'auto' resolved by lap length; explicit values kept; mesh_opts copied and
# validated; unknown value rejected
ok("default mesh_requested 'auto'; short Sturn resolves to 'uniform'; mesh_opts None",
   ctx.mesh_requested == "auto" and ctx.mesh == "uniform" and ctx.mesh_opts is None)
cba = _uo(circuit="BCN")
ok("auto on BCN (long real circuit) resolves to 'curvature', request kept",
   cba.mesh == "curvature" and cba.mesh_requested == "auto")
ok("auto threshold: Sturn lap < MESH_AUTO_MIN_LENGTH (2000 m) <= BCN lap",
   MESH_AUTO_MIN_LENGTH == 2000.0 and ctx.track.s[-1] - ctx.track.s[0] < MESH_AUTO_MIN_LENGTH
   <= cba.track.s[-1] - cba.track.s[0])
cbu = _uo(circuit="BCN", mesh="uniform")
ok("explicit mesh='uniform' on BCN stays 'uniform'",
   cbu.mesh == "uniform" and cbu.mesh_requested == "uniform")
cm = _uo(mesh="curvature")
ok("userOpts(mesh='curvature') sets ctx.mesh", cm.mesh == "curvature")
ok("curvature mesh without mesh_opts -> ctx.mesh_opts None", cm.mesh_opts is None)
mo = {"a": 2.0, "ds_max": 60.0}
cmo = _uo(mesh="curvature", mesh_opts=mo)
ok("mesh_opts stored equal", cmo.mesh_opts == mo)
ok("mesh_opts stored as a copy", cmo.mesh_opts is not mo)
raised = False
try:
    _uo(mesh="adaptive")
except ValueError as e:
    raised = "adaptive" in str(e)
ok("unknown mesh raises ValueError", raised)
ok("mesh_opts={'ds_max': 60.0} accepted",
   _uo(mesh="curvature", mesh_opts={"ds_max": 60.0}).mesh_opts == {"ds_max": 60.0})
allowed = ("a", "b", "ds_min", "ds_max", "smooth_window", "N", "pct", "grid_ds")
ok("all eight curvature_mesh kwargs accepted as mesh_opts keys",
   _uo(mesh="curvature", mesh_opts=dict.fromkeys(allowed, 1.0)).mesh_opts == dict.fromkeys(allowed, 1.0))
for bad_key in ("bogus", "OPT_ds"):             # unknown, and a positional arg discretise() supplies
    raised = False
    try:
        _uo(mesh_opts={bad_key: 1})
    except ValueError as e:
        raised = bad_key in str(e) and "ds_max" in str(e)
    ok(f"mesh_opts key {bad_key!r} raises ValueError listing the allowed keys", raised)

cwa = Ctx()
with warnings.catch_warnings(record=True) as wa:
    warnings.simplefilter("always")
    userOpts(cwa, circuit="Sturn", mesh_opts={"ds_max": 60.0})
ok("mesh_opts with auto -> uniform (Sturn) warns 'mesh_opts ignored'",
   any("mesh_opts ignored" in str(x.message) for x in wa))
ok("mesh_opts with auto -> uniform (Sturn) gives ctx.mesh 'uniform', ctx.mesh_opts None",
   cwa.mesh == "uniform" and cwa.mesh_opts is None)
cwu = Ctx()
with warnings.catch_warnings(record=True) as wu:
    warnings.simplefilter("always")
    userOpts(cwu, circuit="BCN", mesh="uniform", mesh_opts={"ds_max": 60.0})
ok("mesh_opts with explicit mesh='uniform' (BCN) warns 'mesh_opts ignored'",
   any("mesh_opts ignored" in str(x.message) for x in wu))
ok("mesh_opts with explicit mesh='uniform' (BCN) gives ctx.mesh_opts None",
   cwu.mesh == "uniform" and cwu.mesh_opts is None)
cwc = Ctx()
with warnings.catch_warnings(record=True) as wc:
    warnings.simplefilter("always")
    userOpts(cwc, circuit="Sturn", mesh="curvature", mesh_opts={"ds_max": 60.0})
ok("mesh='curvature' keeps the mesh_opts dict",
   cwc.mesh == "curvature" and cwc.mesh_opts == {"ds_max": 60.0})
ok("mesh='curvature' emits no 'mesh_opts ignored' warning",
   not any("mesh_opts ignored" in str(x.message) for x in wc))
raised = False
try:
    _uo(mesh="auto", mesh_opts={"bogus": 1})
except ValueError as e:
    raised = "bogus" in str(e) and "ds_max" in str(e)
ok("unknown mesh_opts key still raises with mesh='auto' on Sturn (resolves to uniform)", raised)

# screening: loose IPOPT tolerances, applied before (so under) ipopt_overrides
ok("SCREENING_IPOPT is the five-tolerance preset",
   SCREENING_IPOPT == {"tol": 1e-3, "acceptable_tol": 1e-2, "dual_inf_tol": 1e-2,
                       "constr_viol_tol": 1e-3, "compl_inf_tol": 1e-3})
ok("default screening is False, default tolerances unchanged",
   ctx.screening is False and ctx.opts["ipopt"]["tol"] == 1e-4
   and ctx.opts["ipopt"]["acceptable_tol"] == 1e-3 and ctx.ipopt_overrides == {})
cs = _uo(screening=True)
ipo = cs.opts["ipopt"]
ok("userOpts(screening=True) stores ctx.screening", cs.screening is True)
ok("screening: tol 1e-3, acceptable_tol 1e-2", ipo["tol"] == 1e-3 and ipo["acceptable_tol"] == 1e-2)
ok("screening: dual_inf_tol 1e-2, constr_viol_tol 1e-3, compl_inf_tol 1e-3",
   ipo["dual_inf_tol"] == 1e-2 and ipo["constr_viol_tol"] == 1e-3 and ipo["compl_inf_tol"] == 1e-3)
ok("screening leaves the other IPOPT options alone",
   ipo["max_iter"] == 6000 and ipo["linear_solver"] == "ma57"
   and ipo["mu_strategy"] == "adaptive" and ipo["mu_init"] == 1e-1)
ok("screening is not recorded as a per-call override", cs.ipopt_overrides == {})
cso = _uo(screening=True, ipopt_overrides={"tol": 1e-5})
ok("ipopt_overrides tol=1e-5 wins over the screening tol",
   cso.opts["ipopt"]["tol"] == 1e-5 and cso.ipopt_overrides == {"tol": 1e-5})
ok("the rest of the screening preset still applies next to an override",
   cso.opts["ipopt"]["acceptable_tol"] == 1e-2 and cso.opts["ipopt"]["compl_inf_tol"] == 1e-3)
ok("screening replaces the tol argument (the GUI always passes tol)",
   _uo(screening=True, tol=1e-6).opts["ipopt"]["tol"] == 1e-3)
ok("ipopt_overrides alone (no screening) still wins over the tol argument",
   _uo(tol=1e-6, ipopt_overrides={"tol": 1e-5}).opts["ipopt"]["tol"] == 1e-5)

# tyre_set forwarded userOpts -> vehParams
cu205 = _uo(tyre_set="MF205")
ok("userOpts(tyre_set='MF205') gives pKy4 == 2.0", cu205.mf.pKy4 == 2.0)
ok("userOpts(tyre_set='MF205') stores ctx.tyre_set", cu205.tyre_set == "MF205")
ok("userOpts default tyre_set is MF205 (pKy4 == 2.0)",
   ctx.tyre_set == "MF205" and ctx.mf.pKy4 == 2.0)
cuB = _uo(tyre_set="CopyB")
ok("userOpts(tyre_set='CopyB') gives the legacy pKy4 == 0.0",
   cuB.tyre_set == "CopyB" and cuB.mf.pKy4 == 0.0)
raised = False
try:
    _uo(tyre_set="Slick")
except ValueError as e:
    raised = "Slick" in str(e)
ok("userOpts(tyre_set='Slick') raises ValueError", raised)

print("track building (real circuit NBR, Nurburgring GP)")
cn = _uo(circuit="NBR")
Ln = float(cn.track.s[-1] - cn.track.s[0])
ok("NBR lap length 5139.104 m (within 0.01 m)", abs(Ln - 5139.104) < 0.01)
ok("NBR track has x and y, same length as s",
   hasattr(cn.track, "x") and hasattr(cn.track, "y")
   and cn.track.x.shape == cn.track.y.shape == cn.track.s.shape)
ok("NBR mesh 'auto' resolves to 'curvature' (lap >= MESH_AUTO_MIN_LENGTH), request kept",
   Ln >= MESH_AUTO_MIN_LENGTH and cn.mesh == "curvature" and cn.mesh_requested == "auto")
for ds, n_exp in ((30, 171), (10, 514)):
    c_ds = _uo(circuit="NBR", OPT_ds=ds)
    d_ds = discretise(c_ds.track, c_ds.OPT_ds, c_ds.OPT_d, mesh=c_ds.mesh, mesh_opts=c_ds.mesh_opts)
    ok(f"NBR OPT_ds={ds}: N = round(L/OPT_ds) = {n_exp}", round(Ln / ds) == n_exp and d_ds["N"] == n_exp)

print("\nALL vehParams / userOpts TESTS PASSED")
