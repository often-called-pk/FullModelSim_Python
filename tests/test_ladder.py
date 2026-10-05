"""functions/ladder.py (multi-fidelity ladder, roadmap Item 10) and its wiring into
MLTP_initial(seed=...) and MLTP(ladder=..., homotopy=...) (plain script, no pytest).
Run from the repo root:

    venv\\Scripts\\python.exe tests\\test_ladder.py

Sections:
  1. registry: resolve_ladder / LADDERS / AUTO_LADDER, homotopy_schedule, friction_overrides
  2. longitudinal rules: split_torque, wheel_torque, lon_model (vs the QSS envelope)
  3. qss_profile on Sturn (= MLTP_screen's lap, = march bit for bit)
  4. seed_m7 with the real vehModel_initial scales (Sturn, Straight, BCN curvature mesh, Circle)
  5. seed_m23 for Static ATD On / ATD Off / EM4 / Active_RW / Active / AALB (+ Circle)
  6. MLTP re-exports ladder.quasi_static_states
  7. MLTP_initial: the constant seed is unchanged, seed='qss' = seed_m7, bad seeds raise
  8. MLTP(ladder=...) wiring with stand-ins (no solve)
  9. one capped real solve (qss23, max_iter=5) and a capped 7-state seed='qss' solve
 10. MLTP(homotopy=...) wiring with a stand-in inner MLTP call, then the record

Sections 1-3 need no casadi; 4-10 need casadi and Data/DATA_AA.mat (SKIP + exit 0
otherwise). Only section 9 runs IPOPT (capped). No NLP numbers are pinned.
"""
import contextlib
import copy
import inspect
import io
import os
import sys
import tempfile
import time
import warnings
from types import SimpleNamespace

import numpy as np

import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)

import functions.ladder as LD
from functions.context import Ctx
from functions.ggv import build_envelope, march, peak_mu_x, peak_mu_y
from functions.transcription import discretise
from userOpts import userOpts
from vehParams import _default_mf, _MF205_OVERRIDES

T_START = time.time()


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout and warnings silenced."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def make_ctx(circuit="Sturn", **kw):
    c = Ctx()
    quiet(userOpts, c, circuit=circuit, **kw)
    return c


# =============================================================================
print("1. registry, homotopy_schedule, friction_overrides")
ok("resolve_ladder('auto') and resolve_ladder(None) -> ('legacy', ('const', 'm7', 'm23'))",
   LD.resolve_ladder("auto") == ("legacy", ("const", "m7", "m23"))
   and LD.resolve_ladder(None) == ("legacy", ("const", "m7", "m23"))
   and LD.resolve_ladder() == ("legacy", ("const", "m7", "m23")))
ok("AUTO_LADDER == 'legacy' (the default until the owner flips it)", LD.AUTO_LADDER == "legacy")
ok("qss7 / qss23 chains", LD.resolve_ladder("qss7") == ("qss7", ("qss", "m7", "m23"))
   and LD.resolve_ladder("qss23") == ("qss23", ("qss", "m23")))
ok("every LADDERS chain starts with a seed tier, ends with 'm23', uses registered tiers",
   all(ch[0] in LD.SEEDS and ch[-1] == "m23" and set(ch) <= set(LD.TIERS) for ch in LD.LADDERS.values())
   and set(LD.TIERS) == {"const", "qss", "m7", "m23"} and LD.SEEDS == ("const", "qss"))
ok("an unknown ladder name raises ValueError ('fast', 'Legacy', 7, ('qss', 'm23'))",
   all(raises(lambda b=b: LD.resolve_ladder(b)) for b in ("fast", "Legacy", 7, ("qss", "m23"))))
ok("MA57_PRE_ALLOC = 3.0", LD.MA57_PRE_ALLOC == 3.0)

ok("homotopy_schedule((1.2, 1.1, 1.0)) -> a tuple of floats; arrays / lists accepted; True -> default",
   LD.homotopy_schedule((1.2, 1.1, 1.0)) == (1.2, 1.1, 1.0)
   and LD.homotopy_schedule(np.array([1.2, 1])) == (1.2, 1.0)
   and all(isinstance(x, float) for x in LD.homotopy_schedule([2, 1]))
   and LD.homotopy_schedule(True) == LD.HOMOTOPY_DEFAULT == (1.2, 1.1, 1.0))
bad = [(), [1.0], (1.2, 1.1), (1.2, 0.9), (0.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0),
       (float("inf"), 1.0), ("1.2", 1.0), (None, 1.0), (True, 1.0), "1.0", 1.0, None, False,
       {"a": 1.0}]
ok(f"homotopy_schedule rejects {len(bad)} bad schedules (empty, too short, last != 1, <= 0, "
   "non-finite, non-numbers, bools, a bare number / string / dict)",
   all(raises(lambda b=b: LD.homotopy_schedule(b)) for b in bad))

mf = _default_mf()
for key, val in _MF205_OVERRIDES.items():
    setattr(mf, key, val)
ok("MF205 overrides none of pDx1, pDx2, pDy1, pDy2", not set(LD.FRICTION_COEFFS) & set(_MF205_OVERRIDES))
base = {"mb": 1900.0, "pDy1": 1.1, "pKy1": 21.0}
base_copy = dict(base)
fo = LD.friction_overrides(mf, 1.2, base)
ok("friction_overrides: base first, then exactly pDx1, pDx2, pDy1, pDy2 x scale (base's value if given)",
   set(fo) == set(base) | set(LD.FRICTION_COEFFS) and fo["mb"] == 1900.0 and fo["pKy1"] == 21.0
   and fo["pDy1"] == 1.2 * 1.1 and fo["pDx1"] == 1.2 * mf.pDx1 and fo["pDx2"] == 1.2 * mf.pDx2
   and fo["pDy2"] == 1.2 * mf.pDy2 and base == base_copy)
ok("scale 1.0 returns base unchanged (a copy), None stays None; no base -> just the four",
   LD.friction_overrides(mf, 1.0, base) == base and LD.friction_overrides(mf, 1.0, base) is not base
   and LD.friction_overrides(mf, 1.0) is None
   and set(LD.friction_overrides(mf, 0.9)) == set(LD.FRICTION_COEFFS))
ok("friction_overrides rejects a scale <= 0 or non-finite",
   all(raises(lambda s=s: LD.friction_overrides(mf, s)) for s in (0.0, -1.0, float("nan"), float("inf"))))
mf_s = copy.copy(mf)
for key, val in LD.friction_overrides(mf, 1.3).items():
    setattr(mf_s, key, val)
Fz = np.linspace(1000.0, 9000.0, 9)
ok("exact mu scaling: peak mu_x and mu_y (vehModel's formulas, functions.ggv) scale by exactly 1.3",
   np.allclose(peak_mu_x(Fz, mf_s, 4905.0), 1.3 * peak_mu_x(Fz, mf, 4905.0), rtol=1e-14, atol=0)
   and np.allclose(peak_mu_y(Fz, mf_s, 4905.0, 0.02), 1.3 * peak_mu_y(Fz, mf, 4905.0, 0.02),
                   rtol=1e-14, atol=0))

rec = LD.ladder_record("qss23", ("qss", "m23"), qss_lap_s=18.7, qss_wall_s=0.01)
ok("ladder_record: savemat-safe defaults for rungs that did not run (-1 / 'none' / NaN), chain joined",
   rec["chain"] == "qss->m23" and rec["m7_iters"] == -1 and rec["m7_status"] == "none"
   and np.isnan(rec["m7_lap_s"]) and np.isnan(rec["m7_wall_s"]) and rec["qss_lap_s"] == 18.7)
hr = LD.homotopy_record([dict(scale=1.2, iters=10, status="Solve_Succeeded", lap=17.5, wall=3.0,
                              warm_start="cold", m7_iters=250),
                         dict(scale=1.0, iters=5, status="Solve_Succeeded", lap=18.0, wall=1.0)])
ok("homotopy_record: arrays per step, statuses / modes joined, missing m7_iters -> -1",
   hr["scales"].tolist() == [1.2, 1.0] and hr["iters"].tolist() == [10, 5]
   and hr["status"] == "Solve_Succeeded, Solve_Succeeded" and hr["warm_start"] == "cold, ?"
   and hr["m7_iters"].tolist() == [250, -1] and hr["lap_s"].tolist() == [17.5, 18.0]
   and raises(lambda: LD.homotopy_record([])))
tmp = tempfile.mkdtemp(prefix="test_ladder_")
import scipy.io as sio                                        # noqa: E402
from functions.importfile import load_solution                # noqa: E402
sio.savemat(os.path.join(tmp, "rec.mat"), {"data": {"lap_time": 1.0, "ladder": rec, "homotopy": hr}},
            do_compression=True)
back = load_solution(os.path.join(tmp, "rec.mat"))
ok("both records survive savemat -> load_solution (strings, ints, NaN, arrays)",
   back.ladder.name == "qss23" and back.ladder.m7_status == "none" and int(back.ladder.m7_iters) == -1
   and np.isnan(back.ladder.m7_lap_s) and np.allclose(back.homotopy.scales, [1.2, 1.0])
   and back.homotopy.status == hr["status"])

# =============================================================================
print("2. longitudinal rules")
ctx = make_ctx("Sturn")
vp, pt = ctx.vp, ctx.pt
T = np.linspace(-20000.0, 9000.0, 2901)
v = np.full_like(T, 30.0)
Tm, Tb = LD.split_torque(T, v, vp, pt, n_motors=1)
ok("split_torque: T_motor * T_brake == 0 everywhere", np.all(Tm * Tb == 0.0))
ok("split_torque: 0 <= T_motor <= Tmax, -Tbrake_max <= T_brake <= 0",
   Tm.min() >= 0.0 and Tm.max() <= pt.Tmax and Tb.min() >= -vp.Tbrake_max and Tb.max() <= 0.0)
ok("split_torque: both non-decreasing in T_w", np.all(np.diff(Tm) >= 0) and np.all(np.diff(Tb) >= 0))
p_cap = pt.Pmax * vp.Rw / (vp.gear * v)
un = (T > -2.0 * vp.Tbrake_max) & (T / vp.gear < np.minimum(pt.Tmax, p_cap))
ok(f"exact inverse on the unclipped range ({un.sum()} of {T.size}): gear*T_motor + 2*T_brake == T_w",
   un.sum() > 1000 and np.allclose((vp.gear * Tm + 2.0 * Tb)[un], T[un], rtol=1e-13, atol=1e-9))
ok("clipped: full braking at -Tbrake_max, full drive at Tmax",
   Tb[0] == -vp.Tbrake_max and Tm[-1] == min(pt.Tmax, p_cap[-1]))
Tw_hi = np.full(5, 3.0 * pt.Tmax * vp.gear)
v_hi = np.array([20.0, 40.0, 60.0, 70.0, 80.0])
Tm_hi, _ = LD.split_torque(Tw_hi, v_hi, vp, pt, n_motors=1)
ok("power cap: one motor at high speed delivers Pmax*Rw/(gear*v) < Tmax (Tmax at low speed)",
   np.allclose(Tm_hi[2:], pt.Pmax * vp.Rw / (vp.gear * v_hi[2:]), rtol=1e-14)
   and np.all(Tm_hi[2:] < pt.Tmax) and Tm_hi[0] == pt.Tmax)
Tm4, Tb4 = LD.split_torque(T, v, vp, pt, n_motors=4)
un4 = un & (T / (4 * vp.gear) < pt.Tmax)
ok("EM4: per-motor torque = the single-motor torque / 4 where unclipped, same brake, no power cap",
   np.allclose(4.0 * Tm4[un4], Tm[un4], rtol=1e-13) and np.array_equal(Tb4, Tb)
   and np.allclose(LD.split_torque(Tw_hi, v_hi, vp, pt, n_motors=4)[0],
                   np.minimum(Tw_hi / vp.gear / 4, pt.Tmax), rtol=1e-14))
has_aero = getattr(ctx, "aero", None) is not None
lon7 = LD.lon_model(ctx, "m7")
q = 0.5 * vp.rho * vp.A
ok("lon_model('m7'): M = vp.m, two wheels, c0 = f*(Wfl0+Wfr0+Wrl0+Wrr0), c2 = q*(Cd_max + f*Cl_max)",
   lon7["M"] == vp.m and lon7["n_wheels"] == 2
   and np.isclose(lon7["c0"], vp.f * (vp.Wfl0 + vp.Wfr0 + vp.Wrl0 + vp.Wrr0), rtol=1e-15)
   and np.isclose(lon7["c2"], q * (np.max(vp.Cd) + vp.f * np.max(vp.Cl)), rtol=1e-15)
   and np.isclose(lon7["c_drag"], q * np.max(vp.Cd), rtol=1e-15))
v_s, a_s = np.array([10.0, 35.0, 70.0]), np.array([-12.0, 0.0, 4.0])
Tw, F = LD.wheel_torque(lon7, vp, v_s, a_s)
ok("wheel_torque: F = M ax + c0 + c2 v^2, T_w = Rw F + n_w Jw ax / Rw",
   np.allclose(F, lon7["M"] * a_s + lon7["c0"] + lon7["c2"] * v_s ** 2, rtol=1e-15)
   and np.allclose(Tw, vp.Rw * F + 2 * vp.Jw * a_s / vp.Rw, rtol=1e-15))
ok("lon_model rejects an unknown tier", raises(lambda: LD.lon_model(ctx, "m9")))
if has_aero:
    env = build_envelope(ctx, load_model="vehModel")
    lon23 = LD.lon_model(ctx, "m23")
    ok("lon_model('m23') == the QSS envelope's M / c_res0 / c_res2 (load_model='vehModel'), four wheels",
       lon23["M"] == env.M == vp.ms and lon23["c0"] == env.c_res0 and lon23["c2"] == env.c_res2
       and lon23["n_wheels"] == 4
       and np.isclose(lon23["c0"], vp.f * (vp.ms + 4 * vp.mus) * vp.g, rtol=1e-14))
    ok("lon_model('m7'): c2 < q*Cd_max (the 7-state model sees aero LIFT: DATA_AA Cl < 0)",
       float(np.max(vp.Cl)) < 0 and lon7["c2"] < q * float(np.max(vp.Cd)))
else:                                                           # pragma: no cover
    print("  [SKIP] lon_model('m23') / aero checks: Data/DATA_AA.mat missing")

# =============================================================================
print("3. qss_profile on Sturn")
if has_aero:
    from MLTP_screen import MLTP_screen                         # noqa: E402
    prof = LD.qss_profile(ctx)
    scr = quiet(MLTP_screen, circuit="Sturn", save=False, verbose=False)
    ref = march(build_envelope(ctx, load_model="vehModel"), ctx.track.s, ctx.track.k, ctx.vi, ds_fine=1.0)
    ok(f"lap {prof['lap_time']:.4f} s == MLTP_screen's ({scr.data.lap_time:.4f} s, documented 18.744)",
       prof["lap_time"] == float(scr.data.lap_time) and round(prof["lap_time"], 3) == 18.744)
    ok("== march(build_envelope(ctx), track.s, track.k, vi, ds_fine=1) bit for bit, + 'env'",
       all(np.array_equal(prof[k], ref[k]) for k in ("s", "v", "ax", "ay", "t"))
       and prof["lap_time"] == ref["lap_time"] and hasattr(prof["env"], "ay_max"))
    ok("profile finite, 0 < v <= Vmax, v(0) == vi, 1 m grid",
       all(np.all(np.isfinite(prof[k])) for k in ("v", "ax", "t")) and prof["v"].min() > 0
       and prof["v"].max() <= pt.Vmax + 1e-9 and prof["v"][0] == ctx.vi and abs(prof["ds"] - 1.0) < 0.01)
else:                                                           # pragma: no cover
    print("  [SKIP] Data/DATA_AA.mat missing")

# =============================================================================
try:
    import casadi  # noqa: F401
except ImportError as exc:                                      # pragma: no cover
    print(f"\nSKIP sections 4-10: casadi not importable ({exc})")
    print("\nALL ladder TESTS PASSED (casadi sections skipped)")
    sys.exit(0)
if not has_aero:                                                # pragma: no cover
    print("\nSKIP sections 4-10: Data/DATA_AA.mat missing")
    print("\nALL ladder TESTS PASSED (casadi sections skipped)")
    sys.exit(0)

from functions.mesh import solution_knots                       # noqa: E402
from functions.transcription import build_and_solve_nlp         # noqa: E402
from functions.warmstart import nlp_structure                   # noqa: E402
from vehModel import vehModel                                   # noqa: E402
from vehModel_initial import vehModel_initial                   # noqa: E402
import MLTP as mltp_mod                                          # noqa: E402
import MLTP_initial as mi_mod                                    # noqa: E402


def in_bounds(a, lo, hi, tol=1e-12):
    return bool(np.all(a >= np.asarray(lo)[:, None] - tol) and np.all(a <= np.asarray(hi)[:, None] + tol))


def disc_of(c):
    return discretise(c.track, c.OPT_ds, c.OPT_d, mesh=c.mesh, mesh_opts=c.mesh_opts)


# =============================================================================
print("4. seed_m7 with the real vehModel_initial scales")
vehModel_initial(ctx)
m7 = ctx.m7
disc = disc_of(ctx)
N, d = disc["N"], ctx.OPT_d
g7 = LD.seed_m7(ctx, m7, disc, prof)
ok(f"shapes x0 (7, {N + 1}), u0 (3, {N + 1}), y0 (1, {N + 1}), xc0 (7, {N * d})",
   g7["x0"].shape == (7, N + 1) and g7["u0"].shape == (3, N + 1) and g7["y0"].shape == (1, N + 1)
   and g7["xc0"].shape == (7, N * d))
ok("finite and within the model's scaled bounds",
   all(np.all(np.isfinite(g7[k])) for k in g7)
   and in_bounds(g7["x0"], m7.x_min, m7.x_max) and in_bounds(g7["xc0"], m7.x_min, m7.x_max)
   and in_bounds(g7["u0"], m7.u_min, m7.u_max) and in_bounds(g7["y0"], m7.y_min, m7.y_max))
X = g7["x0"] * m7.x_s[:, None]
vk = np.interp(disc["s_knot"], prof["s"], prof["v"])
ak = np.interp(disc["s_knot"], prof["s"], prof["ax"])
kk = disc["k_knot"]
ok("path speed vx/cos(eps) == QSS v and yaw rate r == v*k at the knots",
   np.allclose(X[0] / np.cos(X[4]), vk, rtol=1e-13) and np.allclose(X[2], vk * kk, rtol=1e-13, atol=1e-15))
ok("no n drift: vx*sin(eps) + vy*cos(eps) == 0, n == 0",
   np.allclose(X[0] * np.sin(X[4]) + X[1] * np.cos(X[4]), 0.0, atol=1e-12) and np.all(X[3] == 0.0))
ok("sideslip only in the turns: vy == eps == 0 where k == 0, vy != 0 in the corners",
   np.all(X[1][kk == 0] == 0.0) and np.all(X[4][kk == 0] == 0.0) and np.any(X[1] != 0.0))
U = g7["u0"] * m7.u_s[:, None]
ok("steering: sign(delta) == sign(k) at every knot, |delta| ~ L|k| (Ackermann + slip difference)",
   np.array_equal(np.sign(U[2]), np.sign(kk))
   and np.allclose(U[2], (vp.l_f + vp.l_r) * kk, rtol=0.2, atol=1e-6))
ok("T_drive * T_brake == 0 at every knot; = split_torque(wheel_torque(lon_model('m7'), v, ax))",
   np.all(g7["u0"][0] * g7["u0"][1] == 0.0)
   and np.allclose(U[:2], np.vstack(LD.split_torque(LD.wheel_torque(lon7, vp, vk, ak)[0], vk, vp, pt)),
                   rtol=1e-13, atol=1e-9))
ok("scaled Om_f == scaled Om_r == scaled vx (rolling wheels, Om = vx / Rw)",
   np.array_equal(g7["x0"][5], g7["x0"][6]) and np.allclose(g7["x0"][5], g7["x0"][0], rtol=1e-14))
Fk = LD.wheel_torque(lon7, vp, vk, ak)[1]
ok("aux ltx == (F + q Cd_max v^2) hcg / l (the ltx_eq row with zero steer force)",
   np.allclose(g7["y0"][0] * m7.y_s[0], (Fk + lon7["c_drag"] * vk ** 2) * vp.hcg / vp.l, rtol=1e-13))
Xc = g7["xc0"] * m7.x_s[:, None]
vc = np.interp(disc["s_col"], prof["s"], prof["v"])
ok("xc0 = the same rule at disc['s_col'] (profile values, not the kron of the knots)",
   np.allclose(Xc[0] / np.cos(Xc[4]), vc, rtol=1e-13)
   and np.allclose(Xc[2], vc * disc["k_col"], rtol=1e-13, atol=1e-15)
   and not np.array_equal(g7["xc0"], np.kron(g7["x0"][:, :-1], np.ones((1, d)))))

cs = make_ctx("Straight")
vehModel_initial(cs)
ds_ = disc_of(cs)
gs = LD.seed_m7(cs, cs.m7, ds_, LD.qss_profile(cs))
ok("Straight: vy = r = eps = delta = 0 (no lateral part), the speed profile still rises",
   np.all(gs["x0"][1:5] == 0.0) and np.all(gs["xc0"][1:5] == 0.0) and np.all(gs["u0"][2] == 0.0)
   and gs["x0"][0, -1] > gs["x0"][0, 0])
cb = make_ctx("BCN")
vehModel_initial(cb)
db = disc_of(cb)
pb = LD.qss_profile(cb)
gb = LD.seed_m7(cb, cb.m7, db, pb)
Xb = gb["xc0"] * cb.m7.x_s[:, None]
ok(f"BCN curvature mesh (N={db['N']}, mesh {cb.mesh}): shapes, bounds, finite, xc0 in s_col order",
   cb.mesh == "curvature" and gb["x0"].shape == (7, db["N"] + 1) and gb["xc0"].shape == (7, db["N"] * d)
   and gb["y0"].shape == (1, db["N"] + 1) and all(np.all(np.isfinite(gb[k])) for k in gb)
   and in_bounds(gb["x0"], cb.m7.x_min, cb.m7.x_max) and in_bounds(gb["u0"], cb.m7.u_min, cb.m7.u_max)
   and np.allclose(Xb[0] / np.cos(Xb[4]), np.interp(db["s_col"], pb["s"], pb["v"]), rtol=1e-12))
cc = make_ctx("Circle")
vehModel_initial(cc)
dcc = disc_of(cc)
pcc = LD.qss_profile(cc)
gc7 = LD.seed_m7(cc, cc.m7, dcc, pcc)
lo, hi = LD.boundary_box(cc.m7.x_min, cc.m7.x_max, cc.Xi_init, cc.m7.x_s, cc.OPT_e)
ok(f"Circle: the march cannot hold vi (v(0) = {pcc['v'][0]:.1f} < {cc.vi:g} m/s); knot 0 is clipped "
   "into the Xi box, wheel speeds re-derived from the clipped vx",
   pcc["v"][0] < cc.vi - 1.0 and np.all(gc7["x0"][:, 0] >= lo - 1e-15) and np.all(gc7["x0"][:, 0] <= hi + 1e-15)
   and np.allclose(gc7["x0"][5:7, 0], gc7["x0"][0, 0], rtol=1e-14)
   and np.allclose(gc7["x0"][0, 1:] * 100 / np.cos(gc7["x0"][4, 1:]),
                   np.interp(dcc["s_knot"][1:], pcc["s"], pcc["v"]), rtol=1e-12))

# =============================================================================
print("5. seed_m23 across configurations")
CONFIGS = [dict(), dict(ATD="Off"), dict(ATD="Off", Electric_4Motors="On"), dict(AeroConfig="Active_RW"),
           dict(AeroConfig="Active"), dict(AeroConfig="AALB")]
m23_static = None
for cfg in CONFIGS:
    c = make_ctx("Sturn", **cfg)
    quiet(vehModel, c)
    m = c.m23
    if not cfg:
        m23_static = m
    dc = disc_of(c)
    pc = LD.qss_profile(c)
    g = LD.seed_m23(c, m, dc, pc, c.input_keys)
    keys = list(c.input_keys)
    tag = "/".join(f"{k}={v}" for k, v in cfg.items()) or "Static/ATD=On"
    n1 = dc["N"] + 1
    U = g["u0"] * m.u_s[:, None]
    X = g["x0"] * m.x_s[:, None]
    vk = np.interp(dc["s_knot"], pc["s"], pc["v"])
    ak = np.interp(dc["s_knot"], pc["s"], pc["ax"])
    Tw = LD.wheel_torque(LD.lon_model(c, "m23"), c.vp, vk, ak)[0]
    mot = [i for i, k in enumerate(keys) if k.startswith("T_motor")]
    T1, Tb1 = LD.split_torque(Tw, vk, c.vp, c.pt, n_motors=1)
    T4, _ = LD.split_torque(Tw, vk, c.vp, c.pt, n_motors=len(mot))
    checks = [
        g["x0"].shape == (23, n1) and g["u0"].shape == (len(keys), n1) and g["xc0"].shape == (23, dc["N"] * d),
        in_bounds(g["u0"], m.u_min, m.u_max) and in_bounds(g["x0"], m.x_min, m.x_max),
        all(np.all(np.isfinite(g[k])) for k in g),
        np.allclose(U[keys.index("T_brake")], Tb1, rtol=1e-14, atol=1e-9)
        and np.all(g["u0"][mot[0]] * g["u0"][keys.index("T_brake")] == 0),
        all(np.array_equal(U[i], U[mot[0]]) for i in mot),
        np.allclose(U[mot[0]], T4, rtol=1e-13, atol=1e-12),
        np.all(U[keys.index("delta")] == 0.0),
        all(np.all(U[i] == 0.0) for i, k in enumerate(keys) if k in ("FW", "RW", "TW")),
        np.allclose(X[5:], LD.quasi_static_states(c.vp, X[0]), rtol=1e-14),
        np.allclose(X[0], vk, rtol=1e-14) and np.all(X[4] == 0.0)
        and np.allclose(X[1:4], c.OPT_e, rtol=1e-14),
        np.allclose(g["xc0"][0] * m.x_s[0], np.interp(dc["s_col"], pc["s"], pc["v"]), rtol=1e-14),
    ]
    if "ATD" in keys:
        checks.append(np.allclose(sum(U[i] for i, k in enumerate(keys) if k == "ATD"), 1.0, rtol=1e-15))
    if len(mot) == 4:                   # each motor = the total motor torque T_w / gear, / 4
        sel = (Tw > 0) & (Tw / c.vp.gear / 4 < c.pt.Tmax)
        checks.append(sel.sum() > 3 and np.allclose(4.0 * U[mot[0]][sel], Tw[sel] / c.vp.gear, rtol=1e-13)
                      and np.all(U[mot[0]][Tw / c.vp.gear / 4 >= c.pt.Tmax] == c.pt.Tmax))
    ok(f"{tag}: u0 rows follow input_keys {keys}; inputs in bounds; T_motor*T_brake = 0; motor rows equal"
       f"{' (each = total T_w/gear / 4)' if len(mot) == 4 else ''}; ATD sum 1; wings 0; delta 0; "
       "rows 5-22 quasi-static; vy = r = n = OPT_e; eps 0; xc0 at s_col", all(checks))
cc23 = make_ctx("Circle")
dcc23 = disc_of(cc23)
g = LD.seed_m23(cc23, m23_static, dcc23, LD.qss_profile(cc23), cc23.input_keys)
lo, hi = LD.boundary_box(m23_static.x_min, m23_static.x_max, cc23.Xi, m23_static.x_s, cc23.OPT_e)
X0 = g["x0"][:, 0] * m23_static.x_s
ok("Circle: knot 0 inside the Xi box (vx clipped up to vi - OPT_e*x_s), its chassis rows re-derived",
   np.all(g["x0"][:, 0] >= lo - 1e-15) and np.all(g["x0"][:, 0] <= hi + 1e-15)
   and X0[0] > pcc["v"][0] + 1.0
   and np.allclose(X0[5:], LD.quasi_static_states(cc23.vp, [X0[0]])[:, 0], rtol=1e-14))
ok("seed_m23 rejects an unknown input channel / a channel count != nu",
   raises(lambda: LD.seed_m23(ctx, m23_static, disc, prof, ["T_motor", "T_brake", "XX", "delta"]))
   and raises(lambda: LD.seed_m23(ctx, m23_static, disc, prof, ["T_motor", "T_brake", "delta"])))

# =============================================================================
print("6. MLTP re-exports functions.ladder.quasi_static_states")
ok("MLTP.quasi_static_states is functions.ladder.quasi_static_states (one object for warmstart_guesses, "
   "warmstart_refined and seed_m23)", mltp_mod.quasi_static_states is LD.quasi_static_states)

# =============================================================================
print("7. MLTP_initial: constant seed unchanged, seed='qss' = seed_m7, bad seeds raise")
_SIG = inspect.signature(build_and_solve_nlp)


class _Captured(Exception):
    pass


def capture(module, fn, **kw):
    """module.fn(**kw) with module.build_and_solve_nlp replaced by a stand-in that records
    its bound arguments and aborts. Returns (arguments, called)."""
    seen = {}

    def stand_in(*args, **kwargs):
        seen.update(_SIG.bind(*args, **kwargs).arguments)
        raise _Captured()

    real = module.build_and_solve_nlp
    module.build_and_solve_nlp = stand_in
    try:
        quiet(getattr(module, fn), **kw)
    except _Captured:
        pass
    finally:
        module.build_and_solve_nlp = real
    return seen


a = capture(mi_mod, "MLTP_initial", circuit="Sturn", save=False)
ma, ga = a["m"], a["guesses"]
Nn = a["disc"]["N"]
vi0, e0 = 60.0, ctx.OPT_e
legacy = {
    "x0": np.vstack([vi0 * np.ones(Nn + 1), e0 * np.ones(Nn + 1), e0 * np.ones(Nn + 1), e0 * np.ones(Nn + 1),
                     np.zeros(Nn + 1), vi0 * np.ones(Nn + 1) / vp.Rw, vi0 * np.ones(Nn + 1) / vp.Rw]) / ma.x_s[:, None],
    "u0": np.vstack([0.85 * pt.Tmax * np.ones(Nn + 1), np.zeros(Nn + 1), np.zeros(Nn + 1)]) / ma.u_s[:, None],
    "y0": np.zeros((1, Nn + 1)) / ma.y_s[:, None],
}
legacy["xc0"] = np.kron(legacy["x0"][:, :-1], np.ones((1, d)))
ok("seed='const' (default): guesses == the legacy constant formula bit for bit (vx = vi; vy = r = n = "
   "OPT_e; eps 0; Om = vx/Rw; T 0.85 Tmax, 0, 0; ltx 0; xc0 = kron)",
   inspect.signature(mi_mod.MLTP_initial).parameters["seed"].default == "const"
   and set(ga) == set(legacy) and all(np.array_equal(ga[k], legacy[k]) for k in legacy))
ok("seed='const': no ma57_pre_alloc in the IPOPT options", "ma57_pre_alloc" not in a["opts"]["ipopt"])
b = capture(mi_mod, "MLTP_initial", circuit="Sturn", save=False, seed="qss")
exp7 = LD.seed_m7(ctx, m7, disc, prof)
ok("seed='qss': guesses == seed_m7(ctx, m7, disc, qss_profile(ctx)) bit for bit; ma57_pre_alloc 3.0",
   all(np.array_equal(b["guesses"][k], exp7[k]) for k in exp7) and set(b["guesses"]) == set(exp7)
   and b["opts"]["ipopt"].get("ma57_pre_alloc") == 3.0)
b2 = capture(mi_mod, "MLTP_initial", circuit="Sturn", save=False, seed="qss",
             ipopt_overrides={"ma57_pre_alloc": 1.5})
ok("seed='qss' with ipopt_overrides ma57_pre_alloc=1.5: the explicit value wins",
   b2["opts"]["ipopt"]["ma57_pre_alloc"] == 1.5)
called = []
real_b = mi_mod.build_and_solve_nlp
mi_mod.build_and_solve_nlp = lambda *x, **y: called.append(1)
try:
    ok("seed='bogus' / 'QSS' / None raise ValueError before the NLP build",
       all(raises(lambda s=s: quiet(mi_mod.MLTP_initial, circuit="Sturn", save=False, seed=s))
           for s in ("bogus", "QSS", None)) and not called)
finally:
    mi_mod.build_and_solve_nlp = real_b

# =============================================================================
print("8. MLTP(ladder=...) wiring (stand-ins: no NLP is solved)")
N7 = 9
s7 = np.linspace(0.0, 1.0, N7 + 1)
X7 = np.vstack([np.linspace(30, 50, N7 + 1), np.zeros(N7 + 1), np.full(N7 + 1, 0.05),
                0.5 * np.sin(6 * s7), 0.02 * np.cos(4 * s7), np.full(N7 + 1, 100.0), np.full(N7 + 1, 100.0)])
U7 = np.vstack([np.full(N7 + 1, 300.0), np.linspace(0, -500, N7 + 1), 0.05 * np.sin(5 * s7)])
FAKE_INIT = SimpleNamespace(x_opt=X7, u_opt=U7, lap_time=19.5)
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=U7)}


def run_mltp(statuses=None, init_calls=None, **kw):
    """MLTP(...) with MLTP_initial replaced by a stand-in (records its kwargs, returns
    FAKE_INIT) and build_and_solve_nlp by one that returns the packed guesses as the
    solution (zero duals) with the next status of ``statuses``. Returns (ctx, calls, out)."""
    calls, statuses = [], list(statuses or [])
    init_calls = [] if init_calls is None else init_calls

    def init_stand_in(**k):
        init_calls.append(k)
        return SimpleNamespace(data=SimpleNamespace(init=FAKE_INIT),
                               elapsed=dict(ipopt_iters=123, return_status="Solve_Succeeded", qss=0.0))

    def nlp_stand_in(*args, **kwargs):
        a_ = _SIG.bind(*args, **kwargs).arguments
        m_, disc_, g_ = a_["m"], a_["disc"], a_["guesses"]
        st = nlp_structure(m_.nx, m_.nu, m_.ny, disc_["N"], a_["OPT_d"], np.size(a_["h_lb"]))
        w = np.concatenate([g_["x0"].reshape(-1, order="F"), g_["u0"].reshape(-1, order="F"),
                            np.asarray(g_["xc0"]).reshape(-1, order="F")])
        status = statuses.pop(0) if statuses else "Solve_Succeeded"
        warm = a_.get("warm")
        calls.append(dict(N=disc_["N"], disc=disc_, guesses=g_, opts=a_["opts"], warm=warm,
                          m=m_, status=status))
        stats = {"iter_count": 7, "return_status": status}
        wi = dict(x0=False, lam_g0=False, lam_x0=False, ipopt=False, duals=False)
        if warm and warm.get("x0") is not None:              # what build_and_solve_nlp would use
            wi["x0"] = True
            if warm.get("lam_g0") is not None:
                wi.update(lam_g0=True, lam_x0=True, ipopt=True, duals=True)
        return dict(sol={"x": w}, solver=SimpleNamespace(stats=lambda: dict(stats)), w_opt=w,
                    lam_g=np.zeros(st["n_g"]), lam_x=np.zeros(st["n_w"]), structure=st,
                    warm_info=wi, sym_type="SX", linear_solver="ma57", N=disc_["N"])

    real_i, real_n = mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp
    mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp = init_stand_in, nlp_stand_in
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = mltp_mod.MLTP(circuit="Sturn", save=False, plot=False, mesh="uniform", **kw)
    finally:
        mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp = real_i, real_n
    return c, calls, buf.getvalue(), init_calls


ok("MLTP(ladder='auto', homotopy=None) are the defaults",
   inspect.signature(mltp_mod.MLTP).parameters["ladder"].default == "auto"
   and inspect.signature(mltp_mod.MLTP).parameters["homotopy"].default is None)
ref = make_ctx("Sturn", mesh="uniform")
for lad in ("auto", "legacy"):
    c, calls, out, ic = run_mltp(ladder=lad)
    gi = calls[0]["guesses"]
    exp = mltp_mod.warmstart_guesses(ref, calls[0]["m"], X7, U7, calls[0]["disc"]["s_knot"],
                                     solution_knots(FAKE_INIT, X7.shape[1]))
    ok(f"ladder={lad!r}: one 7-state init with seed='const' (kwargs forwarded), guesses == "
       "warmstart_guesses(init), no ma57_pre_alloc, data.ladder 'legacy' with the m7 record, no suffix",
       len(ic) == 1 and ic[0]["seed"] == "const" and ic[0]["mesh"] == "uniform" and ic[0]["save"] is False
       and all(np.array_equal(gi[k], exp[k]) for k in ("x0", "u0", "xc0"))
       and "ma57_pre_alloc" not in calls[0]["opts"]["ipopt"]
       and c.data.ladder["name"] == "legacy" and c.data.ladder["chain"] == "const->m7->m23"
       and c.data.ladder["m7_iters"] == 123 and c.data.ladder["m7_status"] == "Solve_Succeeded"
       and c.data.ladder["m7_lap_s"] == 19.5 and np.isnan(c.data.ladder["qss_lap_s"])
       and c.data.nlp["warm_start"] == "cold" and c.elapsed["ladder"] == "legacy" and "ladder=" not in out)
c, calls, out, ic = run_mltp(ladder="qss7")
ok("ladder='qss7': the 7-state init is called with seed='qss'; ma57_pre_alloc 3.0 on the 23-state "
   "solve; summary ends with ladder=qss7",
   len(ic) == 1 and ic[0]["seed"] == "qss" and calls[0]["opts"]["ipopt"].get("ma57_pre_alloc") == 3.0
   and c.data.ladder["name"] == "qss7" and c.data.ladder["chain"] == "qss->m7->m23"
   and "ladder=qss7" in out)
c, calls, out, ic = run_mltp(ladder="qss23")
exp = LD.seed_m23(ref, calls[0]["m"], calls[0]["disc"], LD.qss_profile(ref), ref.input_keys)
ok("ladder='qss23': no 7-state init; guesses == seed_m23(ctx, m23, disc, qss_profile(ctx)) bit for bit; "
   "ma57_pre_alloc 3.0; data.ladder records the QSS lap, m7 'none'; warm start stays 'cold'",
   len(ic) == 0 and all(np.array_equal(calls[0]["guesses"][k], exp[k]) for k in ("x0", "u0", "xc0"))
   and calls[0]["opts"]["ipopt"].get("ma57_pre_alloc") == 3.0
   and c.data.ladder["name"] == "qss23" and c.data.ladder["chain"] == "qss->m23"
   and c.data.ladder["qss_lap_s"] == prof["lap_time"] and c.data.ladder["qss_wall_s"] > 0
   and c.data.ladder["m7_iters"] == -1 and c.data.ladder["m7_status"] == "none"
   and c.data.nlp["warm_start"] == "cold" and "ladder=qss23" in out)
c, calls, out, ic = run_mltp(ladder="qss23", ipopt_overrides={"ma57_pre_alloc": 2.0})
ok("an explicit ipopt_overrides ma57_pre_alloc wins over the ladder's", calls[0]["opts"]["ipopt"]["ma57_pre_alloc"] == 2.0)
n_prof = []
real_q = mltp_mod.qss_profile
mltp_mod.qss_profile = lambda c_, **k: (n_prof.append(1), real_q(c_, **k))[1]
try:
    c, calls, out, ic = run_mltp(["Solve_Succeeded", "Maximum_Iterations_Exceeded", "Solve_Succeeded"],
                                 ladder="qss23", refine=dict(passes=1, tol=1e-9))
finally:
    mltp_mod.qss_profile = real_q
exp = LD.seed_m23(ref, calls[2]["m"], calls[2]["disc"], LD.qss_profile(ref), ref.input_keys)
ok("qss23 + refine: the failed reseeded pass retries from seed_m23 on the NEW mesh, the profile "
   "computed once (cached)", len(calls) == 3 and calls[2]["N"] == 36 and len(n_prof) == 1 and len(ic) == 0
   and all(np.array_equal(calls[2]["guesses"][k], exp[k]) for k in ("x0", "u0", "xc0")))
for bad_kw in (dict(ladder="fast"), dict(ladder="QSS23"), dict(ladder="qss23", warm_start=INIT),
               dict(ladder="qss7", warm_start=INIT), dict(homotopy=(1.2, 1.1)), dict(homotopy=(0.0, 1.0))):
    called_i, called_n = [], []
    real_i, real_n = mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp
    mltp_mod.MLTP_initial = lambda **k: called_i.append(k)
    mltp_mod.build_and_solve_nlp = lambda *x, **y: called_n.append(1)
    try:
        r = raises(lambda: quiet(mltp_mod.MLTP, circuit="Sturn", save=False, plot=False, **bad_kw))
    finally:
        mltp_mod.MLTP_initial, mltp_mod.build_and_solve_nlp = real_i, real_n
    ok(f"MLTP({', '.join(f'{k}={v!r}' if k != 'warm_start' else 'warm_start=<7-state init>' for k, v in bad_kw.items())}) "
       "raises ValueError before any solve", r and not called_i and not called_n)
c, calls, out, ic = run_mltp(ladder="auto", warm_start=INIT)
ok("warm_start=<7-state init> with ladder='auto' (or 'legacy') still runs 'init7', no 7-state solve",
   len(ic) == 0 and c.data.nlp["warm_start"] == "init7" and c.data.ladder["m7_iters"] == -1
   and run_mltp(ladder="legacy", warm_start=INIT)[0].data.nlp["warm_start"] == "init7")
src = inspect.getsource(mltp_mod.MLTP)
seg = src.split("MLTP_initial(")[1].split(")")[0]
ok("test_mltp_params' check still holds: the MLTP_initial call forwards **useropts_kwargs (and seed), "
   "the docstring does not contain the call",
   "**useropts_kwargs" in seg and "seed=chain[0]" in seg and "MLTP_initial(" not in mltp_mod.MLTP.__doc__)

# =============================================================================
print("9. one capped real solve (IPOPT max_iter=5)")
t9 = time.time()
cr = quiet(mltp_mod.MLTP, circuit="Sturn", ladder="qss23", save=False, plot=False, max_iter=5)
ok(f"MLTP(Sturn, ladder='qss23', max_iter=5): 5 iterations, Maximum_Iterations_Exceeded, data.ladder "
   f"'qss23', finite lap ({cr.data.lap_time:.3f} s), ma57_pre_alloc 3.0 ({time.time() - t9:.1f} s)",
   cr.elapsed["ipopt_iters"] == 5 and cr.solve_stats["return_status"] == "Maximum_Iterations_Exceeded"
   and cr.data.ladder["name"] == "qss23" and np.isfinite(cr.data.lap_time)
   and cr.opts["ipopt"].get("ma57_pre_alloc") == 3.0 and cr.data.nlp["warm_start"] == "cold")
t9 = time.time()
ci = quiet(mi_mod.MLTP_initial, circuit="Sturn", seed="qss", save=False, max_iter=5)
ok(f"MLTP_initial(Sturn, seed='qss', max_iter=5): ctx.elapsed / ctx.solve_stats / data.init record seed, "
   f"iterations and status ({time.time() - t9:.1f} s)",
   ci.elapsed["ipopt_iters"] == 5 and ci.elapsed["seed"] == "qss"
   and ci.elapsed["return_status"] == "Maximum_Iterations_Exceeded" and ci.elapsed["qss"] > 0
   and ci.data.init.seed == "qss" and ci.data.init.ipopt_iters == 5
   and ci.data.init.return_status == "Maximum_Iterations_Exceeded"
   and set(ci.elapsed) == {"setup", "qss", "solve", "ipopt_iters", "return_status", "seed"})

# =============================================================================
print("10. MLTP(homotopy=...) wiring")
inner = []


def inner_stand_in(**k):
    i = len(inner)
    inner.append(k)
    return SimpleNamespace(elapsed=dict(ipopt_iters=10 + i, warm_start="cold" if i == 0 else "full+duals"),
                           solve_stats=dict(return_status="Solve_Succeeded"),
                           data=SimpleNamespace(lap_time=18.0 + i, ladder=dict(m7_iters=250 if i == 0 else -1)),
                           tag=i)


real_M = mltp_mod.MLTP
mltp_mod.MLTP = inner_stand_in
try:
    out_c = quiet(real_M, circuit="Sturn", homotopy=(1.2, 1.1, 1.0), vp_overrides={"mb": 1850.0},
                  refine=True, save=True, plot=True, results_dir="nowhere", ladder="qss23", mesh="uniform")
finally:
    mltp_mod.MLTP = real_M
mf0 = make_ctx("Sturn", vp_overrides={"mb": 1850.0}).mf
ok("3 steps; vp_overrides = the caller's + pDx1/pDx2/pDy1/pDy2 x 1.2, x 1.1, then the caller's own",
   len(inner) == 3
   and inner[0]["vp_overrides"] == LD.friction_overrides(mf0, 1.2, {"mb": 1850.0})
   and inner[0]["vp_overrides"]["pDy1"] == 1.2 * mf0.pDy1 and inner[0]["vp_overrides"]["mb"] == 1850.0
   and inner[1]["vp_overrides"]["pDx2"] == 1.1 * mf0.pDx2 and inner[2]["vp_overrides"] == {"mb": 1850.0})
ok("warm-start chain: step 1 from the caller's warm_start (None: cold through the ladder), "
   "each later step from the previous step's ctx; the ctx returned is the last step's",
   inner[0]["warm_start"] is None and inner[1]["warm_start"].tag == 0 and inner[2]["warm_start"].tag == 1
   and out_c.tag == 2)
ok("refine / save / plot only on the last step; homotopy=None, ladder and the other kwargs on every step",
   [s["refine"] for s in inner] == [None, None, True] and [s["save"] for s in inner] == [False, False, True]
   and [s["plot"] for s in inner] == [False, False, True] and all(s["homotopy"] is None for s in inner)
   and all(s["ladder"] == "qss23" and s["mesh"] == "uniform" and s["results_dir"] == "nowhere"
           and s["circuit"] == "Sturn" for s in inner))
hs = inner[2].get("_homotopy_steps")
ok("the last step receives the earlier steps' record (scale, iterations, status, lap, warm start, m7)",
   "_homotopy_steps" not in inner[0] and "_homotopy_steps" not in inner[1] and len(hs) == 2
   and [h_["scale"] for h_ in hs] == [1.2, 1.1] and [h_["iters"] for h_ in hs] == [10, 11]
   and [h_["lap"] for h_ in hs] == [18.0, 19.0] and [h_["m7_iters"] for h_ in hs] == [250, -1]
   and [h_["warm_start"] for h_ in hs] == ["cold", "full+duals"])
c, calls, out, ic = run_mltp(homotopy=(1.2, 1.0), warm_start=INIT)
hm = c.data.homotopy
ok("real recursion (stand-in solves): data['homotopy'] = {scales, iters, status, lap_s, wall_s, ...}; "
   "step 1 from the 7-state init, step 2 full+duals from step 1",
   len(calls) == 2 and hm["scales"].tolist() == [1.2, 1.0] and hm["iters"].tolist() == [7, 7]
   and hm["status"] == "Solve_Succeeded, Solve_Succeeded" and hm["warm_start"] == "init7, full+duals"
   and hm["lap_s"].shape == (2,) and np.all(hm["wall_s"] > 0) and calls[1]["warm"] is not None
   and c.data.nlp["warm_start"] == "full+duals" and c.elapsed["homotopy"] is hm
   and "homotopy step 2/2" in out)
c0, _, _, _ = run_mltp(warm_start=INIT)
ok("homotopy=None (default): no 'homotopy' record", "homotopy" not in vars(c0.data))

print(f"\nALL ladder TESTS PASSED ({time.time() - T_START:.0f} s)")
