"""vehModel.py vs the MATLAB reference vehModel.m (plain script, no pytest).

Section 1 pins MATLAB reference values: vehModel.m (FullModel_4EM_Suspension_FullTyre)
was evaluated in MATLAB R2025a + CasADi 3.7.2 (2026-10-04) at the normalised point X/U/K
below, with Powertrain.m + vehParams.m defaults (= tyre_set 'MF205') and Static aero.
vehModel.py reproduced every value bit for bit there and at 14 random points (single
motor, ATD, EM4, AALB at the static wing angles; AALB with other wing inputs differs,
see the active-aero note in section 1).
Sections 2-5 pin the four relations audited on 2026-10-04 (the model review pass).
All four are inherited from vehModel.m, not port bugs:
  2. EM4 motor speed Om_motor_fl = Om_fl/gear vs single motor gear*mean(Om):
     the per-motor power/rpm rows are gear^2 (~52.6x) looser and never bind
  3. every unsprung corner carries the TOTAL vp.mus: static tyre-load sum
     (ms + 4*mus)*g = 26.0 kN, not m*g = 20.5 kN (vehParams' Wfl0 basis sums to m*g)
  4. the front/rear Cl split leaves the steady axle loads unchanged (aero load reaches
     the axles in the ratio l_r/l : l_f/l); it only changes the body attitude
  5. with ATD On, brkB and Tdist are inert (ATD_* split drive and 2*T_brake)
A deliberate physics change must update these pins, functions/ggv.py
(load_model='vehModel') and CLAUDE.md together.
Needs casadi and Data/DATA_AA.mat (prints SKIP and exits 0 otherwise). Run from the repo root:

    venv\\Scripts\\python.exe test_vehmodel_matlab.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "Data")

try:
    import casadi as ca
except ImportError as exc:                                      # pragma: no cover
    print(f"SKIP test_vehmodel_matlab: casadi not importable ({exc})")
    sys.exit(0)
if not os.path.exists(os.path.join(DATA, "DATA_AA.mat")):       # pragma: no cover
    print("SKIP test_vehmodel_matlab: Data/DATA_AA.mat missing")
    sys.exit(0)

from functions.context import Ctx
from userOpts import userOpts
from vehModel import vehModel


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)))


def model(ATD="On", EM4="Off", aero="Static", **kw):
    c = Ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        userOpts(c, circuit="Sturn", AeroConfig=aero, ATD=ATD, Electric_4Motors=EM4,
                 data_dir=DATA, **kw)
    return c


def build(c):
    vehModel(c)
    return c


def signals(c):
    """f(x, u, kappa) -> dx, fz, T, motor, f_lift corners (all physical except dx)."""
    m = c.m23
    mot = ([m.Om_motor, m.P_motor] if c.pt.EM4 == 0 else
           [m.Om_motor_fl, m.Om_motor_fr, m.Om_motor_rl, m.Om_motor_rr,
            m.P_motor_fl, m.P_motor_fr, m.P_motor_rl, m.P_motor_rr])
    outs = [m.dx, ca.vertcat(m.fz_fl, m.fz_fr, m.fz_rl, m.fz_rr),
            ca.vertcat(m.T_fl, m.T_fr, m.T_rl, m.T_rr), ca.vertcat(*mot),
            ca.vertcat(m.f_lift_fl, m.f_lift_fr, m.f_lift_rl, m.f_lift_rr)]
    f = ca.Function("sig", [m.x, m.u, m.pv], outs)
    return lambda x, u, k: [np.array(v).ravel() for v in f(x, u, k)]


def vertical_equilibrium(c, v):
    """Straight running at v (zero torques, free rolling): solve heave/pitch/roll +
    4 unsprung for (zs, theta, phi, zt_*). Returns (z, fz[4], f_lift, f_lift_front_pair)."""
    m, vp = c.m23, c.vp
    z = ca.SX.sym("z", 7)
    xp = ca.SX.zeros(23)
    xp[0] = v
    for i in (5, 6, 7, 8):
        xp[i] = v / vp.Rw
    xp[9], xp[11], xp[13] = z[0], z[1], z[2]
    for j, i in enumerate((19, 20, 21, 22)):
        xp[i] = z[3 + j]
    syms = ca.vertcat(m.x, m.u, m.kappa)
    vals = ca.vertcat(xp / ca.DM(m.x_s), ca.DM.zeros(m.nu), 0)
    dx = ca.substitute(m.dx, syms, vals)
    resid = ca.vertcat(*[dx[i] for i in (10, 12, 14, 15, 16, 17, 18)])
    sig = ca.substitute(ca.vertcat(m.fz_fl, m.fz_fr, m.fz_rl, m.fz_rr, m.f_lift,
                                   m.f_lift_fl + m.f_lift_fr), syms, vals)
    rf = ca.rootfinder("vert_eq", "newton", ca.Function("resid", [z], [resid]))
    zz = np.array(rf(np.r_[0.0, 0.0, 0.0, np.full(4, 6500.0 / vp.kt)])).ravel()
    s = np.array(ca.Function("sig", [z], [sig])(zz)).ravel()
    res = np.array(ca.Function("resid", [z], [resid])(zz)).ravel()
    assert np.max(np.abs(res)) < 1e-8, "vertical equilibrium did not converge"
    return zz, s[:4], s[4], s[5]


# =============================================================================
print("1. MATLAB reference values (vehModel.m evaluated in MATLAB, 2026-10-04)")
# normalised point: x (23, MATLAB order), kappa; u per config
X = np.array([0.45, 0.05, 0.2, 0.3, 0.05, 0.46, 0.455, 0.47, 0.465,
              -0.01, 0.05, 0.004, -0.02, -0.006, 0.03,
              0.02, -0.03, 0.01, -0.015, 0.021, 0.018, 0.024, 0.02])
K = 0.02
# dx entries 0-4 and 9-22 do not depend on the powertrain branch at this point
DX_COMMON = {0: 0.0017518611162892067, 1: -0.0074422644757377015, 2: 0.12404996858332513,
             3: 0.011870247063097604, 4: -0.01568109298658056, 9: 0.0010797267533548599,
             10: -5.940777803725136, 11: -0.000431890701341944, 12: 5.756692480348896,
             13: 0.0006478360520129159, 14: -1.4984170087763218, 15: 23.49593416573808,
             16: 22.304560537196142, 17: 10.722186824782433, 18: 2.254739898884998,
             19: -0.000431890701341944, 20: 0.0006478360520129159, 21: -0.000215945350670972,
             22: 0.00032391802600645797}
FZ = [6300.0, 5400.0, 7200.0, 6000.0]
FLIFT = [431.8334739203659, 431.8334739203659, 214.01003339662745, 214.01003339662745]
REF = {   # config: (ATD, EM4, u, dOm_fl..rr, T_fl..rr, motor signals)
    "1 motor, ATD Off": ("Off", "Off", [0.6, -0.1, 0.15],
                         [-0.020928391394166617, -0.005420041458908816, -0.04199418899812003, -0.022780667916155965],
                         [85.46908450541832, 85.46908450541832, 824.237000472214, 824.237000472214],
                         [944.7991270134842, 341261.4446772705]),
    "1 motor, ATD On": ("On", "Off", [0.6, -0.1, 0.3, 0.2, 0.25, 0.25, 0.15],
                        [-0.011125319993173855, 0.0005086625069481589, -0.0498600766815449, -0.030646555599580832],
                        [545.8236509865794, 363.88243399105295, 454.8530424888162, 454.8530424888162],
                        [944.7991270134842, 341261.4446772705]),
    "EM4": ("Off", "On", [0.5, 0.4, 0.7, 0.6, -0.1, 0.15],
            [0.017971206794210985, 0.024182996208383396, 0.0027752575032708817, 0.012692218064149557],
            [1912.203474962721, 1475.6347799701766, 2926.620864947809, 2490.0521699552646],
            [17.867894491186345, 17.67367824671693, 18.256326980125177, 18.06211073565576,
             5378.236241847089, 4255.821721809437, 7693.216189424749, 6524.03439771886]),
}
TOL = 1e-10   # bit-identical on the reference machine; slack for other libm builds
for name, (atd, em4, u, dom, T, mot) in REF.items():
    c = build(model(ATD=atd, EM4=em4))
    dx_ref = np.array([DX_COMMON[i] if i in DX_COMMON else dom[i - 5] for i in range(23)])
    dx, fz, Tq, mo, fl = signals(c)(X, np.array(u), K)
    ok(f"{name}: dx (23 states) == MATLAB (rel {rel(dx, dx_ref):.0e})", rel(dx, dx_ref) < TOL)
    ok(f"{name}: tyre loads, f_lift_* == MATLAB", rel(fz, FZ) < TOL and rel(fl, FLIFT) < TOL)
    ok(f"{name}: wheel torques T_* == MATLAB", rel(Tq, T) < TOL)
    ok(f"{name}: motor speed/power == MATLAB", rel(mo, mot) < TOL)

# AALB with the aero inputs at the static wing angles == the static ATD-On reference
cA = build(model(ATD="On", aero="AALB"))
vpA = cA.vp
u_static = np.array([0.6, -0.1, 0.3, 0.2, 0.25, 0.25, vpA.alpha_FL / 10.0, vpA.alpha_FR / 10.0,
                     vpA.alpha_RW / 30.0, vpA.alpha_TW / 12.0, 0.15])
dx_s, fz_s, T_s, _, fl_s = signals(cA)(X, u_static, K)
_, _, _, dom_on, T_on, _ = REF["1 motor, ATD On"]
dx_on = np.array([DX_COMMON[i] if i in DX_COMMON else dom_on[i - 5] for i in range(23)])
ok("AALB, aero inputs at the static angles: dx == MATLAB", rel(dx_s, dx_on) < TOL and rel(fl_s, FLIFT) < TOL)
# Known deviation (owner decision pending): vehModel.m defines the active-aero inputs
# but its aero coefficients read the fixed vp.alpha_* (vehModel.m L429-449), so in
# MATLAB those inputs are dead; vehModel.py wires them into the wing angles.
u_rw = u_static.copy()
u_rw[-3] = 20.0 / 30.0
dx_rw, _, _, _, fl_rw = signals(cA)(X, u_rw, K)
ok(f"AALB: rear-wing input is live in vehModel.py (f_lift {fl_s.sum():.0f} -> {fl_rw.sum():.0f} N; "
   "dead in vehModel.m: known deviation)", rel(fl_rw.sum(), fl_s.sum()) > 1e-3 and rel(dx_rw[0], dx_s[0]) > 1e-6)

# =============================================================================
print("2. EM4 motor speed (vehModel.m L975-978 / L944, inherited)")
c1 = build(model(ATD="Off", EM4="Off"))
c4 = build(model(ATD="Off", EM4="On"))
vp, pt = c1.vp, c1.pt
gear = vp.gear
v = 50.0
xw = np.zeros(23); xw[0] = v; xw[5:9] = v / vp.Rw; xw[19:23] = 6500.0 / vp.kt
Om = v / vp.Rw
_, _, T1, mot1, _ = signals(c1)(xw / c1.m23.x_s, np.array([0.7, 0.0, 0.0]), 0.0)
_, _, T4, mot4, _ = signals(c4)(xw / c4.m23.x_s, np.array([0.7, 0.4, 0.9, 0.2, 0.0, 0.0]), 0.0)
ok("single motor: Om_motor = gear * mean(Om_wheel)", rel(mot1[0], gear * Om) < 1e-12)
ok("EM4: Om_motor_* = Om_wheel / gear (MATLAB divides; wheel torque still uses *gear)",
   rel(mot4[:4], Om / gear) < 1e-12 and rel(T4, np.array([0.7, 0.4, 0.9, 0.2]) * pt.Tmax * gear) < 1e-12)
ok(f"speed ratio single/EM4 = gear^2 = {gear**2:.2f} (the '52x' observation)",
   rel(mot1[0] / mot4[0], gear**2) < 1e-12 and 52.0 < gear**2 < 53.0)
ok("single motor: P_motor = wheel drive power sum(T_i*Om_i)", rel(mot1[1], np.sum(T1) * Om) < 1e-12)
ok("EM4: sum P_motor_* = wheel drive power / gear^2", rel(np.sum(mot4[4:]) * gear**2, np.sum(T4) * Om) < 1e-12)
Om_top = pt.Vmax / vp.Rw
ok(f"EM4 rows cannot bind: at Vmax and Tmax, P_motor_fl = {pt.Tmax * Om_top / gear / 1e3:.1f} kW "
   f"(cap {pt.Pmax / 1e3:.0f} kW), Om_motor_fl = {Om_top / gear:.1f} rad/s (cap {pt.OMmax:.0f})",
   pt.Tmax * Om_top / gear < 0.05 * pt.Pmax and Om_top / gear < 0.02 * pt.OMmax)
ok("single motor at Vmax and Tmax exceeds Pmax (its power row does bind)",
   pt.Tmax * gear * Om_top > pt.Pmax)

# =============================================================================
print("3. unsprung-mass load accounting (vehModel.m L1034-1037, inherited)")
cd = build(model())
vp = cd.vp
z, fz, f_lift, _ = vertical_equilibrium(cd, 1.0)
cc = np.cos(z[1]) * np.cos(z[2])
ok("equilibrium: sum fz == (ms*cos(phi)*cos(theta) + 4*mus)*g + f_lift (rel 1e-9)",
   rel(fz.sum(), (vp.ms * cc + 4 * vp.mus) * vp.g + f_lift) < 1e-9)
ok(f"near-static tyre-load sum {fz.sum():.1f} N = (ms + 4*mus)*g = {(vp.ms + 4 * vp.mus) * vp.g:.1f} N "
   f"(not m*g = {vp.m * vp.g:.1f} N)",
   abs(fz.sum() - (vp.ms + 4 * vp.mus) * vp.g) < 1.0 and fz.sum() - vp.m * vp.g > 3 * vp.mus * vp.g - 1.0)
ok("vp.mus is the TOTAL unsprung mass (muf + mur), subtracted at every corner", vp.mus == vp.muf + vp.mur)
ok("vehParams' static corner loads Wfl0..Wrr0 use muf/2, mur/2 and sum to m*g (internally inconsistent)",
   rel(2 * (vp.Wfl0 + vp.Wrl0), vp.m * vp.g) < 1e-12)

# =============================================================================
print("4. front/rear Cl split vs axle loads (vehModel.m L508-511, L1028-1030, inherited)")
v = 60.0
z1, fz1, fl1, flf1 = vertical_equilibrium(cd, v)
cs = model()
cs.vp.Cl0_front -= 0.1          # move downforce to the front axle, total unchanged
cs.vp.Cl0_rear += 0.1
build(cs)
z2, fz2, fl2, flf2 = vertical_equilibrium(cs, v)
L = vp.l_f + vp.l_r
front_cf = 2 * vp.mus * vp.g + (vp.ms * vp.g * np.cos(z1[1]) * np.cos(z1[2]) + fl1) * vp.l_r / L
ok(f"split shift moves {flf2 - flf1:.0f} N of lift to the front pair, total unchanged",
   flf2 - flf1 > 400.0 and rel(fl2, fl1) < 1e-12)
ok(f"front axle load unchanged by the split ({fz1[0] + fz1[1]:.2f} -> {fz2[0] + fz2[1]:.2f} N)",
   abs((fz2[0] + fz2[1]) - (fz1[0] + fz1[1])) < 1.0)
ok(f"aero load reaches the axles in l_r/l (front lift share {flf1 / fl1:.2f}, front axle = closed form, rel 1e-4)",
   rel(fz1[0] + fz1[1], front_cf) < 1e-4)
ok(f"the split only changes attitude; more front downforce pitches nose-up (theta {z1[1]:.2e} -> {z2[1]:.2e})",
   z2[1] < z1[1] - 1e-4)

# =============================================================================
print("5. ATD On: brkB and Tdist inert (vehModel.m L958-962, by design)")
uA = np.array([0.6, -0.4, 0.4, 0.1, 0.3, 0.2, 0.1])
xa = X.copy()
Ta = signals(build(model(ATD="On")))(xa, uA, K)[2]
Tb = signals(build(model(ATD="On", vp_overrides=dict(brkB=0.3, Tdist=0.2))))(xa, uA, K)[2]
ok("ATD On: T_* identical for brkB/Tdist = defaults vs 0.3/0.2", np.array_equal(Ta, Tb))
uO = np.array([0.6, -0.4, 0.1])
To = signals(build(model(ATD="Off")))(xa, uO, K)[2]
Tp = signals(build(model(ATD="Off", vp_overrides=dict(brkB=0.3, Tdist=0.2))))(xa, uO, K)[2]
ok("ATD Off: the same change moves T_* (brkB/Tdist live)", rel(To, Tp) > 1e-3)
pt = cd.pt
tot = 0.6 * pt.Tmax * cd.vp.gear + 2 * (-0.4 * cd.vp.Tbrake_max)
ok("ATD On: sum T_* = (T_motor*gear + 2*T_brake) * sum(ATD), the fixed-split total when sum(ATD) = 1",
   rel(np.sum(Ta), tot * uA[2:6].sum()) < 1e-12 and rel(np.sum(To), tot) < 1e-12)

print("ALL vehModel vs MATLAB TESTS PASSED")
