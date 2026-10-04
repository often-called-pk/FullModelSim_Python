"""23-state NLP bounds vs MLTP.m / vehModel.m (plain script, no pytest). Run from the repo root:

    venv\\Scripts\\python.exe test_mltp_constraints.py

Pins the two port fixes of 2026-10-04 (MATLAB is the reference; both were Python gaps):
  1. MLTP.build_path_constraints follows MLTP.m's `switch TyreModel`: CombinedSlip (the
     default, the only case MATLAB's MLTP runs) has powertrain rows only (nh = 3 / 4 / 12
     for ATD Off / ATD On / EM4); PureSlip puts the four friction circles rho_lim_* first
     (nh = 7 / 8 / 16)
  2. input-rate bounds are divided by u_s (vehModel.m L377-379): the NLP bounds the rate
     of the NORMALISED inputs, so vehModel exposes m.duk_* = ctx.duk_* / u_s (motor
     33.2226 and brake 36.25 per second as in MATLAB; steering 0.1 rad/s / delta_max,
     where MATLAB divides by a delta_s = pi/8 leaked from vehModel_initial.m)
  3. MLTP() and MLTP_paramOptim hand exactly these to build_and_solve_nlp (a stand-in
     captures the call: nothing is solved, an in-memory 7-state init skips the init solve)
  4. plotSDI.friction_usage: the rho_lim rows when present, else the same formula on
     data.vehicle (CombinedSlip results have no rho_lim rows)

Needs casadi and Data/DATA_AA.mat (prints SKIP and exits 0 otherwise); section 4 needs plotly.
"""
import contextlib
import inspect
import io
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)                                   # MLTP() reads Circuits/ and Data/ relatively

try:
    import casadi as ca
except ImportError as exc:                                      # pragma: no cover
    print(f"SKIP test_mltp_constraints: casadi not importable ({exc})")
    sys.exit(0)
if not os.path.exists(os.path.join(HERE, "Data", "DATA_AA.mat")):   # pragma: no cover
    print("SKIP test_mltp_constraints: Data/DATA_AA.mat missing")
    sys.exit(0)

from functions.context import Ctx
from functions.transcription import build_and_solve_nlp
from userOpts import userOpts
from vehModel import vehModel
from vehModel_initial import vehModel_initial
import MLTP as mltp_mod
import MLTP_paramOptim as po_mod
from MLTP import build_path_constraints


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def model(TyreModel="CombinedSlip", **kw):
    c = Ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # ATD + EM4 guard warning
        userOpts(c, circuit="Sturn", **kw)
    vehModel(c, TyreModel=TyreModel)
    return c


PT_ROWS = {   # MLTP.m case 'CombinedSlip': (names, lb, ub) per config
    "ATD Off": (["motor_power", "motor_rpm", "BrTh_1"], [0, 0, -1], [1, 1, 1]),
    "ATD On": (["motor_power", "motor_rpm", "BrTh_1", "ATD_eq"], [0, 0, -1, -1e-3], [1, 1, 1, 1e-3]),
    "EM4": ([f"motor_power_{w}" for w in ("fl", "fr", "rl", "rr")]
            + [f"motor_rpm_{w}" for w in ("fl", "fr", "rl", "rr")]
            + [f"BrTh_{w}" for w in ("fl", "fr", "rl", "rr")],
            [0] * 8 + [-1] * 4, [1] * 12),
}
CONFIG_KW = {"ATD Off": dict(ATD="Off"), "ATD On": dict(ATD="On"),
             "EM4": dict(ATD="Off", Electric_4Motors="On")}
RHO = ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr"]

# =============================================================================
print("1. path constraints per TyreModel (MLTP.m L85-141)")
rs = np.random.RandomState(11)
for cfg, (names, lb, ub) in PT_ROWS.items():
    rows = {}
    for tm in ("CombinedSlip", "PureSlip"):
        c = model(tm, **CONFIG_KW[cfg])
        m = c.m23
        hn, h, h_lb, h_ub = build_path_constraints(ca, m, c.pt)
        exp_n = (RHO + names) if tm == "PureSlip" else names
        exp_lb = ([0] * 4 + lb) if tm == "PureSlip" else lb
        exp_ub = ([1] * 4 + ub) if tm == "PureSlip" else ub
        ok(f"{cfg}, {tm}: m.TyreModel recorded, rows {len(exp_n)} = MLTP.m's names / bounds, in order",
           m.TyreModel == tm and hn == exp_n and h.shape == (len(exp_n), 1)
           and np.array_equal(h_lb, np.array(exp_lb, dtype=float))
           and np.array_equal(h_ub, np.array(exp_ub, dtype=float)))
        rows[tm] = (c, hn, ca.Function("h", [m.x, m.u, m.pv], [h]))
    # the same point in both models: the powertrain rows do not involve the tyre forces
    c_ps, hn_ps, f_ps = rows["PureSlip"]
    _, _, f_cs = rows["CombinedSlip"]
    m_ps = c_ps.m23
    x = rs.uniform(0.2, 0.6, 23)
    u = rs.uniform(-0.4, 0.4, m_ps.nu)
    k = 0.01
    v_ps = np.array(f_ps(x, u, k)).ravel()
    v_cs = np.array(f_cs(x, u, k)).ravel()
    sig = ca.Function("sig", [m_ps.x, m_ps.u, m_ps.pv],
                      [ca.vertcat(*[getattr(m_ps, f"{q}_{w}") for w in ("fl", "fr", "rl", "rr")])
                       for q in ("fx", "fy", "fz")]
                      + [ca.vertcat(*[getattr(m_ps, f"mu_{w}_{a}") for w in ("fl", "fr", "rl", "rr")])
                         for a in ("x", "y")])
    fx, fy, fz, mux, muy = (np.array(s).ravel() for s in sig(x, u, k))
    ok(f"{cfg}: PureSlip rho_lim_* = sqrt((fx/(mu_x fz))^2 + (fy/(mu_y fz))^2) per tyre",
       np.allclose(v_ps[:4], np.sqrt((fx / (mux * fz))**2 + (fy / (muy * fz))**2), rtol=1e-12, atol=0))
    ok(f"{cfg}: the remaining PureSlip rows == the CombinedSlip rows (same powertrain expressions)",
       np.array_equal(v_ps[4:], v_cs))

c = model("CombinedSlip")
hn_o = build_path_constraints(ca, c.m23, c.pt, TyreModel="PureSlip")[0]
ok("an explicit TyreModel argument overrides the model's own", hn_o == RHO + PT_ROWS["ATD On"][0])
try:
    build_path_constraints(ca, c.m23, c.pt, TyreModel="Combined")
    raised = False
except ValueError:
    raised = True
ok("an unknown TyreModel raises ValueError", raised)

# =============================================================================
print("2. input-rate bounds divided by u_s (vehModel.m L377-379)")
PHYS = {"T_motor": 2e4, "T_motor_fl": 2e4, "T_motor_fr": 2e4, "T_motor_rl": 2e4, "T_motor_rr": 2e4,
        "T_brake": 1.45e5, "ATD": 2e4, "delta": 0.1, "FW": 20.0, "RW": 60.0, "TW": 24.0}
for label, kw in (("Static, ATD On (default)", {}), ("Static, ATD Off", dict(ATD="Off")),
                  ("EM4", dict(ATD="Off", Electric_4Motors="On")),
                  ("AALB, ATD On", dict(AeroConfig="AALB")),
                  ("Active, EM4", dict(AeroConfig="Active", ATD="Off", Electric_4Motors="On"))):
    c = model(**kw)
    m = c.m23
    phys = np.array([PHYS[k] for k in c.input_keys])
    ok(f"{label}: ctx.duk_* stay physical (userOpts units per second), m.duk_* = ctx.duk_* / u_s",
       np.array_equal(c.duk_ub.ravel(), phys) and np.array_equal(c.duk_lb.ravel(), -phys)
       and np.array_equal(m.duk_ub, phys / m.u_s) and np.array_equal(m.duk_lb, -phys / m.u_s))
c = model()
m = c.m23
delta_max = 35.0 * np.pi / 180.0
ok("u_s = [Tmax 602, Tbrake_max 4000, ATD 1 x4, delta_max] for the default config",
   np.allclose(m.u_s, [602.0, 4000.0, 1, 1, 1, 1, delta_max], rtol=1e-15, atol=0))
ok(f"default bounds: motor {m.duk_ub[0]:.4f} and brake {m.duk_ub[1]:.2f} per second = MATLAB "
   "(vehModel.m run: 33.2226, 36.25), ATD 2e4 (u_s = 1)",
   abs(m.duk_ub[0] - 33.2226) < 5e-5 and m.duk_ub[1] == 36.25 and np.all(m.duk_ub[2:6] == 2e4))
ok(f"steering: {m.duk_ub[-1]:.5f} per second = 0.1 rad/s / delta_max, the documented 0.1 rad/s "
   "(MATLAB as run: 0.1 / (pi/8) = 0.25465)",
   m.duk_ub[-1] == 0.1 / delta_max and abs(m.duk_ub[-1] * m.u_s[-1] - 0.1) < 1e-15)
cA = model(AeroConfig="AALB")
ok("AALB: wing-rate bounds 20/10, 20/10, 60/30, 24/12 = 2 per second (FW, FW, RW, TW)",
   np.allclose(cA.m23.duk_ub[6:10], 2.0, rtol=1e-15, atol=0))
vehModel_initial(c)
ok("the 7-state init model uses the same convention (m7.duk_*_init = ctx.duk_*_init / its u_s)",
   np.array_equal(c.m7.duk_ub_init, c.duk_ub_init.ravel() / c.m7.u_s)
   and np.array_equal(c.m7.duk_lb_init, c.duk_lb_init.ravel() / c.m7.u_s))

# =============================================================================
print("3. what MLTP() and MLTP_paramOptim pass to build_and_solve_nlp (captured, no solve)")
_SIG = inspect.signature(build_and_solve_nlp)


class _Captured(Exception):
    pass


def captured_call(module, fn, **kw):
    """Call module.fn(**kw) with module.build_and_solve_nlp replaced by a stand-in
    that records its bound arguments and aborts; returns them."""
    seen = {}

    def stand_in(*args, **kwargs):
        seen.update(_SIG.bind(*args, **kwargs).arguments)
        raise _Captured()

    real = module.build_and_solve_nlp
    module.build_and_solve_nlp = stand_in
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            getattr(module, fn)(**kw)
    except _Captured:
        pass
    finally:
        module.build_and_solve_nlp = real
    return seen


N0 = 9                                    # a 7-state init solution held in memory
X7 = np.vstack([np.linspace(30, 50, N0 + 1), np.zeros(N0 + 1), np.zeros(N0 + 1), np.zeros(N0 + 1),
                np.zeros(N0 + 1), np.full(N0 + 1, 100.0), np.full(N0 + 1, 100.0)])
U7 = np.vstack([np.full(N0 + 1, 300.0), np.zeros(N0 + 1), np.zeros(N0 + 1)])
INIT = {"init": SimpleNamespace(x_opt=X7, u_opt=U7)}
phys = np.array([2e4, 1.45e5, 2e4, 2e4, 2e4, 2e4, 0.1])     # default keys: T_motor T_brake ATD x4 delta

a = captured_call(mltp_mod, "MLTP", circuit="Sturn", warm_start=INIT, save=False, plot=False,
                  mesh="uniform")
ma = a.get("m")
ok("MLTP() reached build_and_solve_nlp with the 23-state model", ma is not None and ma.nx == 23)
ok("MLTP(): duk_lb / duk_ub = m.duk_* = userOpts limits / u_s (not the physical values)",
   np.array_equal(a["duk_ub"], ma.duk_ub) and np.array_equal(a["duk_lb"], ma.duk_lb)
   and np.array_equal(a["duk_ub"], phys / ma.u_s) and not np.allclose(a["duk_ub"], phys))
ok("MLTP() default (CombinedSlip, ATD On): 4 path rows, h_eq returns 4, bounds as MLTP.m",
   a["h_eq"].size1_out(0) == 4 and np.array_equal(a["h_lb"], [0, 0, -1, -1e-3])
   and np.array_equal(a["h_ub"], [1, 1, 1, 1e-3]))
b = captured_call(mltp_mod, "MLTP", circuit="Sturn", warm_start=INIT, save=False, plot=False,
                  mesh="uniform", TyreModel="PureSlip")
ok("MLTP(TyreModel='PureSlip'): 8 path rows (4 friction circles first), same rate bounds",
   b["m"].TyreModel == "PureSlip" and b["h_eq"].size1_out(0) == 8
   and np.array_equal(b["h_lb"][:4], [0, 0, 0, 0]) and np.array_equal(b["duk_ub"], a["duk_ub"]))
p = captured_call(po_mod, "MLTP_paramOptim", circuit="Sturn", warm_start=INIT, save=False,
                  mesh="uniform")
ok("MLTP_paramOptim: duk_* = m.duk_* = limits / u_s, 4 path rows, 6 design parameters",
   np.array_equal(p["duk_ub"], p["m"].duk_ub) and np.array_equal(p["duk_lb"], p["m"].duk_lb)
   and np.array_equal(p["duk_ub"], phys / p["m"].u_s)
   and p["h_eq"].size1_out(0) == 4 and p["param"]["sym"].numel() == 6)

# =============================================================================
print("4. plotSDI.friction_usage (rho_lim rows, else computed from data.vehicle)")
try:
    import plotSDI
except ImportError as exc:                                      # pragma: no cover
    print(f"  [SKIP] plotly not importable ({exc})")
    plotSDI = None
if plotSDI is not None:
    n = 6
    r = np.random.RandomState(5)
    veh = {}
    for w in ("fl", "fr", "rl", "rr"):
        veh[f"fx_{w}"], veh[f"fy_{w}"] = r.uniform(-3e3, 3e3, n), r.uniform(-4e3, 4e3, n)
        veh[f"fz_{w}"] = r.uniform(3e3, 7e3, n)
        veh[f"mu_{w}_x"], veh[f"mu_{w}_y"] = r.uniform(1.2, 1.5, n), r.uniform(1.1, 1.4, n)
    base = dict(s_full=np.linspace(0.0, 100.0, 4 * (n - 1) + 1), N=n - 1, OPT_d=3, vehicle=veh)
    d_cs = dict(base, constraints={"motor_power": np.ones(n)})
    rho = plotSDI.friction_usage(d_cs)
    ok("no rho_lim rows (CombinedSlip): rho_fl..rr from fx, fy, fz, mu_x, mu_y in data.vehicle",
       list(rho) == ["rho_fl", "rho_fr", "rho_rl", "rho_rr"]
       and all(np.allclose(rho[f"rho_{w}"], np.sqrt((veh[f"fx_{w}"] / (veh[f"mu_{w}_x"] * veh[f"fz_{w}"]))**2
                                                    + (veh[f"fy_{w}"] / (veh[f"mu_{w}_y"] * veh[f"fz_{w}"]))**2))
               for w in ("fl", "fr", "rl", "rr")))
    rows = {k: np.full(n, 0.5) for k in RHO}
    ok("rho_lim rows present (PureSlip): those rows are used as saved",
       list(plotSDI.friction_usage(dict(base, constraints=rows))) == RHO)
    ok("neither available: nothing to plot", plotSDI.friction_usage(dict(N=2, s_full=np.arange(9.0))) == {})
    ok("plot_friction draws one trace per tyre for a CombinedSlip result",
       len(plotSDI.plot_friction(d_cs).data) == 4)

print("\nALL MLTP constraint TESTS PASSED")
