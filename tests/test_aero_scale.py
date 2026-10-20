"""E3 Cd_scale and Cl_scale, and the honest Cd / Cl placeholder labels (docs/laptime_sim_improvements.md,
item E3; plan docs/superpowers/plans/2026-10-20-finish-partials.md, Task 2).

Plain script, no pytest. Each check prints PASS or FAIL, and the first FAIL stops the file. Run from the repo root:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_aero_scale.py

  1. defaults (float 1.0), vp_overrides plumbing, registry rows (primary, META, group), placeholder labels
  2. the Cd and Cl placeholders: with DATA_AA.mat an override changes no 23-state graph and no QSS lap;
     without it the 7-state model reads them
  3. full model, both physics: f_drag x Cd_scale, f_lift and the four f_lift_i x Cl_scale, f_side unchanged
  4. QSS: aero_coefficients and the envelope scale like vehModel; the lap moves in the right direction
  5. corrected drag pitch moment (A4): rear-wing drag at hw, body drag at hcg, none in matlab; scales with
     Cd_scale; the QSS axle shift matches the coast equilibrium; the QSS lap moves less than 0.01 %
  6. parity export: Cd_scale and Cl_scale left out at 1.0, both exported when one of them is not 1.0

Sections 2 to 6 need casadi and Data/DATA_AA.mat (SKIP and exit 0 otherwise). Nothing is solved.
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import hashlib
import io
import os
import sys
import tempfile
import warnings

import numpy as np

from functions.context import Ctx
from vehParams import vehParams, PRIMARY_KEYS
from app.vp_params import PARAM_GROUPS, META, meta_for


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *args, **kwargs):
    """fn(*args, **kwargs) with stdout swallowed and warnings ignored."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def rel_err(a, b, ratio=1.0):
    """Max relative error of a against ratio * b."""
    a, b = np.asarray(a, float), ratio * np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


# =============================================================================
print("1. defaults, overrides, registry, labels")
c = Ctx()
quiet(vehParams, c)
ok("vp.Cd_scale and vp.Cl_scale are float 1.0",
   isinstance(c.vp.Cd_scale, float) and isinstance(c.vp.Cl_scale, float) and c.vp.Cd_scale == c.vp.Cl_scale == 1.0)
c = Ctx()
quiet(vehParams, c, vp_overrides={"Cd_scale": 1.2, "Cl_scale": 0.9})
ok("vp_overrides Cd_scale 1.2 and Cl_scale 0.9 land in ctx.vp", c.vp.Cd_scale == 1.2 and c.vp.Cl_scale == 0.9)
ok("registry: Cd_scale and Cl_scale are primaries, have a META row, and are the last two of 'Aero & Environment'",
   {"Cd_scale", "Cl_scale"} <= PRIMARY_KEYS and {"Cd_scale", "Cl_scale"} <= set(META)
   and dict(PARAM_GROUPS)["Aero & Environment"][-2:] == ["Cd_scale", "Cl_scale"])
ok("labels: the Cd and Cl rows say placeholder",
   "placeholder" in meta_for("Cd").label.lower() and "placeholder" in meta_for("Cl").label.lower())

if not os.path.exists(os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")):
    print("SKIP sections 2-6 (test_aero_scale): Data/DATA_AA.mat missing")
    sys.exit(0)

# model imports come after the DATA_AA guard
import casadi as ca  # noqa: E402
from MLTP_screen import MLTP_screen  # noqa: E402
from functions.ggv import aero_coefficients, build_envelope  # noqa: E402
from userOpts import userOpts  # noqa: E402
from vehModel import rolling_start, vehModel  # noqa: E402
from vehModel_initial import vehModel_initial  # noqa: E402
import validation.py_export as py_export  # noqa: E402

KW = dict(circuit="Sturn", vi=60.0, AeroConfig="Static", ATD="On", Electric_4Motors="Off", mesh="uniform")
TILT = {"alpha_TW": 3.0}     # rear-wing tilt in deg: gives a nonzero side force (f_side is exactly 0 at the static set)
NAMES = ("f_drag", "f_lift", "f_lift_fl", "f_lift_fr", "f_lift_rl", "f_lift_rr", "f_side")


def model(physics="matlab", **ov):
    """Full 23-state model on Sturn, vp_overrides = ov."""
    c = Ctx()
    quiet(userOpts, c, physics=physics, vp_overrides=ov, **KW)
    quiet(vehModel, c)
    return c


def digest(fn):
    """sha256 of a casadi Function: equal digests mean equal graphs."""
    return hashlib.sha256(fn.serialize().encode()).hexdigest()


def graph23(c):
    m = c.m23
    return digest(ca.Function("f", [m.x, m.u, m.pv], [m.dx, m.sf]))


def static_state(c, vx):
    """Physical state at vx: rolling wheels, static tyre deflections, all else zero."""
    vp = c.vp
    X = np.zeros(c.m23.nx)
    X[0] = vx
    X[5:9] = vx / vp.Rw
    X[19:23] = [vp.xti_fl, vp.xti_fr, vp.xti_rl, vp.xti_rr]
    return X


def coast(c):
    """Physical inputs: ATD 0.25 at U[2:6] (as in test_mf_scaling), torque and steer 0."""
    U = np.zeros(c.m23.nu)
    U[2:6] = 0.25
    return U


def signals(c, names):
    """(X, U) -> the named model signals in physical units, on a straight road (kappa 0)."""
    m = c.m23
    F = ca.Function("F", [m.x, m.u, m.pv], [getattr(m, n) for n in names])
    xs, us = np.asarray(m.x_s, float), np.asarray(m.u_s, float)
    return lambda X, U: [float(v) for v in F(X / xs, U / us, 0.0)]


def qss_lap(physics="matlab", **ov):
    """QSS lap time on Sturn, vp_overrides = ov."""
    return quiet(MLTP_screen, circuit="Sturn", save=False, verbose=False, physics=physics,
                 vp_overrides=ov).data.lap_time


# =============================================================================
print("2. Cd and Cl placeholders")
c0, c1 = model(), model(Cd=2.0, Cl=3.0)
ok("with DATA_AA.mat the override is overwritten: vp.Cd = Cd0 and vp.Cl = Cl0_front + Cl0_rear",
   c1.vp.Cd == c1.vp.Cd0 and c1.vp.Cl == c1.vp.Cl0_front + c1.vp.Cl0_rear)
ok("with DATA_AA.mat the 23-state graph does not read the Cd / Cl override", graph23(c0) == graph23(c1))
lap0, lap1 = qss_lap(), qss_lap(Cd=2.0, Cl=3.0)
ok(f"with DATA_AA.mat the QSS lap does not read it either ({lap1!r} == {lap0!r})", lap0 == lap1)
with tempfile.TemporaryDirectory() as empty:
    n0, n1 = Ctx(), Ctx()
    quiet(userOpts, n0, data_dir=empty, **KW)
    quiet(userOpts, n1, data_dir=empty, vp_overrides={"Cd": 2.0, "Cl": 3.0}, **KW)

    def graph7(c):
        quiet(vehModel_initial, c)
        m = c.m7
        return digest(ca.Function("f", [m.x, m.u, m.y, m.pv], [m.dx, m.sf]))
    ok("without DATA_AA.mat the 7-state graph reads the placeholders (the override changes it)",
       graph7(n0) != graph7(n1))

# =============================================================================
print("3. full model, both physics: scales on the aero forces, side force untouched")
for physics in ("matlab", "corrected"):
    a, b = model(physics, **TILT), model(physics, Cd_scale=1.2, Cl_scale=0.85, **TILT)
    X, U = static_state(a, 55.0), coast(a)
    s0, s1 = signals(a, NAMES)(X, U), signals(b, NAMES)(X, U)
    ok(f"{physics}: f_drag x 1.2, f_lift and the four f_lift_i x 0.85 (rel 1e-12), f_side unchanged "
       f"(f_side {s0[6]:.2f} N, nonzero)",
       rel_err(s1[0], s0[0], 1.2) < 1e-12 and rel_err(s1[1:6], s0[1:6], 0.85) < 1e-12
       and abs(s0[6]) > 1e-3 * abs(s0[0]) and s1[6] == s0[6])

# =============================================================================
print("4. QSS: aero forces and envelope follow the scales as vehModel does; the lap moves")
c = model(Cd_scale=1.2, Cl_scale=0.85, **TILT)
ac = aero_coefficients(c.vp, c.aero)
q = 0.5 * c.vp.rho * c.vp.A * 50.0 ** 2
qss = [q * ac["Cd"], -q * ac["Cl"], -q * ac["Cl_fl"], -q * ac["Cl_fr"], -q * ac["Cl_rl"], -q * ac["Cl_rr"]]
veh = signals(c, NAMES[:6])(static_state(c, 50.0), coast(c))
ok("QSS aero_coefficients == vehModel at Cd_scale 1.2, Cl_scale 0.85 (f_drag, f_lift, four f_lift_i; rel 1e-9)",
   rel_err(qss, veh) < 1e-9)
env0 = quiet(build_envelope, model(**TILT))
env1 = quiet(build_envelope, model(Cd_scale=1.2, Cl_scale=0.85, **TILT))
ok("envelope: F_drag x 1.2 and F_lift x 0.85 on the speed grid (rel 1e-12)",
   np.allclose(env1.F_drag, 1.2 * env0.F_drag, rtol=1e-12, atol=0.0)
   and np.allclose(env1.F_lift, 0.85 * env0.F_lift, rtol=1e-12, atol=0.0))
lap_d, lap_l = qss_lap(Cd_scale=1.2), qss_lap(Cl_scale=1.2)
ok(f"QSS lap: Cd_scale 1.2 is slower ({lap_d - lap0:+.4f} s), Cl_scale 1.2 is faster ({lap_l - lap0:+.4f} s), "
   f"each by more than 1 ms", lap_d - lap0 > 1e-3 and lap0 - lap_l > 1e-3)


# =============================================================================
print("5. corrected drag pitch moment (A4): rear-wing drag at hw, body drag at hcg")


def dxdt(c, X, U):
    """Physical time derivative dx/dt of the state on a straight road. m.dx is the normalised space
    derivative, so dx/dt = m.dx * x_s / sf."""
    m = c.m23
    F = ca.Function("F", [m.x, m.u, m.pv], [m.dx, m.sf])
    xs, us = np.asarray(m.x_s, float), np.asarray(m.u_s, float)
    dx, sf = F(X / xs, U / us, 0.0)
    return np.array(dx).ravel() * xs / float(sf)


def drag_moment(c, v):
    """Closed form of the drag term in dthetadot: -q RW_Cd (hw - hcg) / I_y, RW_Cd at the static wing angle."""
    vp = c.vp
    q = 0.5 * vp.rho * vp.A * v * v
    cd_rw = sum(float(c.aero.RW.Cd[i]) * vp.alpha_RW ** p for i, p in enumerate((2, 1)))
    return -q * cd_rw * (vp.hw - vp.hcg) / vp.I_y


cc, ch = model("corrected"), model("corrected", hw=0.5)      # hw 1.28 (default) and hw = hcg = 0.5
for v in (20.0, 60.0):
    X, U = static_state(cc, v), coast(cc)
    want = drag_moment(cc, v)
    got = dxdt(cc, X, U)[12] - dxdt(ch, X, U)[12]
    ok(f"corrected: dthetadot at hw 1.28 minus at hw = hcg equals the closed form at {v:.0f} m/s (rel 1e-9)",
       abs(got - want) <= 1e-9 * abs(want))
mm, mh = model(), model(hw=0.5)
X, U = static_state(mm, 60.0), coast(mm)
env_m = quiet(build_envelope, mm, v_grid=np.array([60.0]))
env_mh = quiet(build_envelope, mh, v_grid=np.array([60.0]))
ok("matlab: the state derivative and the QSS axle loads do not depend on hw (exact)",
   np.array_equal(dxdt(mm, X, U), dxdt(mh, X, U))
   and np.array_equal(env_m.Fz_front, env_mh.Fz_front) and np.array_equal(env_m.Fz_rear, env_mh.Fz_rear))
cd12 = model("corrected", Cd_scale=1.2)
X, U = static_state(cc, 60.0), coast(cc)
want = 0.2 * drag_moment(cc, 60.0)
got = dxdt(cd12, X, U)[12] - dxdt(cc, X, U)[12]
ok("corrected: the drag term scales with Cd_scale (Cd_scale 1.2 adds 0.2 times the closed form, rel 1e-9)",
   abs(got - want) <= 1e-9 * abs(want))
fz_r = [quiet(build_envelope, c, v_grid=np.array([60.0])).Fz_rear[0] for c in (ch, cc, cd12)]
ok("corrected: the QSS axle shift scales with Cd_scale too (Cd_scale 1.2 adds 0.2 times the shift, rel 1e-9)",
   abs((fz_r[2] - fz_r[1]) - 0.2 * (fz_r[1] - fz_r[0])) <= 1e-9 * abs(fz_r[1] - fz_r[0]))
for v in (30.0, 50.0, 70.0):
    X_hw, _ = rolling_start(cc, cc.m23, v)
    X_hcg, _ = rolling_start(ch, ch.m23, v)
    fz_hw, fz_hcg = X_hw[19:23] * cc.vp.kt, X_hcg[19:23] * ch.vp.kt
    d_front = (fz_hw[0] + fz_hw[1]) - (fz_hcg[0] + fz_hcg[1])
    d_rear = (fz_hw[2] + fz_hw[3]) - (fz_hcg[2] + fz_hcg[3])
    ok(f"corrected: coast equilibrium at {v:.0f} m/s: front load falls, rear rises, sum 0, theta falls "
       f"({d_front:+.2f} / {d_rear:+.2f} N)",
       d_rear > 0 > d_front and abs(d_front + d_rear) < 0.05 and X_hw[11] < X_hcg[11])
    env_hw = quiet(build_envelope, cc, v_grid=np.array([v]))
    env_hcg = quiet(build_envelope, ch, v_grid=np.array([v]))
    q_front = float(env_hw.Fz_front[0] - env_hcg.Fz_front[0])
    q_rear = float(env_hw.Fz_rear[0] - env_hcg.Fz_rear[0])
    ok(f"corrected: QSS axle shift at {v:.0f} m/s equals the coast equilibrium shift (0.05 N)",
       abs(q_front - d_front) < 0.05 and abs(q_rear - d_rear) < 0.05)
lap_hw, lap_hcg = qss_lap("corrected"), qss_lap("corrected", hw=0.5)
ok(f"corrected: QSS lap on Sturn is lower by less than 0.01 % ({100 * (lap_hw / lap_hcg - 1):+.4f} %)",
   -1e-4 < lap_hw / lap_hcg - 1 < 0)


# =============================================================================
def export(ov):
    """validation.py_export.params at Sturn, ds 30, with vp_overrides = ov. params() reads its userOpts
    keyword arguments from PARITY, so PARITY gets the key for the call."""
    saved = py_export.PARITY
    py_export.PARITY = {**saved, "vp_overrides": ov}
    try:
        return quiet(py_export.params, "Sturn", 30)
    finally:
        py_export.PARITY = saved


print("6. parity export: joint identity filter")
vp0 = export(None)["vp"]
ok("parity export at default: no Cd_scale or Cl_scale in the vp block (sanity: hw is there)",
   "Cd_scale" not in vp0 and "Cl_scale" not in vp0 and "hw" in vp0)
for ov in ({"Cd_scale": 1.1}, {"Cl_scale": 0.9}):     # one key alone must not hide the other
    vp1, want = export(ov)["vp"], {"Cd_scale": 1.0, "Cl_scale": 1.0, **ov}
    got = {k: vp1.get(k) for k in want}
    ok(f"parity export, vp_overrides {ov}: the vp block has {got} == {want}", got == want)

print("ALL aero_scale TESTS PASSED")
