"""QSS g-g-v screen (functions/ggv.py + MLTP_screen.py).

Plain top-level assertions, like the other test_*.py files. Sections:
  1. analytic straights (constant accel, power limit, braking into a hairpin)
  2. constant-radius circles (synthetic and real envelope)
  3. causality / monotonicity of the march on BCN
  4. CasADi anchors: numpy peak-mu, aero and quasi-steady loads vs vehModel
     (skipped with a note if casadi is missing)
  5. MLTP_screen end-to-end + timing + .mat round trip
  6. ggv runs with casadi blocked (numpy-only at runtime)
  7. screen_sweep
"""
import os, sys, math, tempfile, subprocess, warnings
import numpy as np
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
ROOT = _bootstrap.ROOT

from functions.context import Ctx
from functions.ggv import (Envelope, build_envelope, march, apex_speeds, aero_coefficients,
                           peak_mu_x, peak_mu_y, lateral_load_transfer)
from functions.importfile import load_solution
from userOpts import userOpts
from MLTP_screen import MLTP_screen, screen_sweep


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# =============================================================================
print("1. analytic straights")
# (a) constant tyre acceleration up to the speed cap
env = Envelope.from_constants(ay_max=10.0, ax_trac=5.0, ax_brk=8.0, v_cap=50.0, M=1000.0)
p = march(env, [0.0, 1000.0], [0.0, 0.0], vi=10.0)
t_ref = (50 - 10) / 5.0 + (1000 - (50**2 - 10**2) / (2 * 5.0)) / 50.0          # 8 + 15.2
ok(f"const accel + cap: lap {p['lap_time']:.6f} == {t_ref:.6f} s", rel(p["lap_time"], t_ref) < 1e-9)

# (b) tyre-limited -> power-limited (P/v) -> speed cap: t_power = M*(v2^2 - v1^2)/(2P)
M, a_t, P, vcap, vi, L = 1000.0, 5.0, 300e3, 70.0, 10.0, 2000.0
env = Envelope.from_constants(ay_max=10.0, ax_trac=a_t, ax_brk=8.0, v_cap=vcap, M=M, P_mot=P)
v_p = P / (M * a_t)                                         # 60 m/s: tyre limit meets power limit
s1, t1 = (v_p**2 - vi**2) / (2 * a_t), (v_p - vi) / a_t
s2, t2 = M * (vcap**3 - v_p**3) / (3 * P), M * (vcap**2 - v_p**2) / (2 * P)
t_ref = t1 + t2 + (L - s1 - s2) / vcap
p = march(env, [0.0, L], [0.0, 0.0], vi=vi)
ok(f"accel -> power -> cap: lap {p['lap_time']:.5f} vs analytic {t_ref:.5f} s (rel {rel(p['lap_time'], t_ref):.1e})",
   rel(p["lap_time"], t_ref) < 1e-4)

# (c) accelerate, cruise, brake (backward pass) into a constant-radius hairpin
a_t, b, vcap, ay, R, vi = 4.0, 8.0, 60.0, 12.5, 50.0, 20.0
v_c = math.sqrt(ay * R)                                     # 25 m/s
env = Envelope.from_constants(ay_max=ay, ax_trac=a_t, ax_brk=b, v_cap=vcap, M=1000.0)
s_tr = [0.0, 1000.0 - 1e-9, 1000.0, 1300.0]
k_tr = [0.0, 0.0, 1.0 / R, 1.0 / R]
d_acc, d_brk = (vcap**2 - vi**2) / (2 * a_t), (vcap**2 - v_c**2) / (2 * b)
t_ref = (vcap - vi) / a_t + (1000.0 - d_acc - d_brk) / vcap + (vcap - v_c) / b + 300.0 / v_c
# The step into a curvature discontinuity brakes at half rate (e = 0 at the corner
# point), shifting the braking curve by ds/2: an O(ds) error of ~ds/2*(1/v_c - 1/vcap).
p_fine = march(env, s_tr, k_tr, vi=vi, ds_fine=0.1)
ok(f"accel/cruise/brake/hairpin (ds=0.1): lap {p_fine['lap_time']:.5f} vs analytic {t_ref:.5f} s "
   f"(rel {rel(p_fine['lap_time'], t_ref):.1e})", rel(p_fine["lap_time"], t_ref) < 1e-4)
p = march(env, s_tr, k_tr, vi=vi)
ok(f"same at default ds=1 m (rel {rel(p['lap_time'], t_ref):.1e}, O(ds) corner-entry step)",
   rel(p["lap_time"], t_ref) < 1e-3)
ok("hairpin held at sqrt(ay*R) after entry", np.max(np.abs(p["v"][p["s"] >= 1000.0] - v_c)) < 1e-9)
ok("braking zone ends at the hairpin entry", abs(np.interp(1000.0, p["s"], p["v"]) - v_c) < 1e-9)

# =============================================================================
print("2. constant-radius circles")
R = 50.0
env = Envelope.from_constants(ay_max=12.5, ax_trac=4.0, ax_brk=8.0, v_cap=60.0, M=1000.0)
p = march(env, [0.0, 2000.0], [1.0 / R, 1.0 / R], vi=5.0)
ok("synthetic circle: v -> sqrt(ay_max R) = 25 m/s (2nd half)",
   np.max(np.abs(p["v"][p["s"] > 1000.0] - 25.0)) < 1e-9)
ok("synthetic circle: accelerates on the ellipse from vi", p["v"][0] == 5.0 and np.all(np.diff(p["v"]) >= -1e-12))

ctx = Ctx()
userOpts(ctx, circuit="Sturn")
env = build_envelope(ctx)
R = 80.0
p = march(env, [0.0, 3000.0], [1.0 / R, 1.0 / R], vi=10.0)
v_ss = p["v"][-1]
v_ap = apex_speeds(env, np.array([1.0 / R]))[0]
e_ss = math.sqrt(1.0 - (v_ss**2 / R / np.interp(v_ss, env.v_grid, env.ay_max))**2)
balance = np.interp(v_ss, env.v_grid, env.Fx_trac) * e_ss / float(env.F_res(v_ss))
ok(f"real circle R=80: v_ss {v_ss:.3f} <= apex {v_ap:.3f} m/s and within 3%", v_ap * 0.97 <= v_ss <= v_ap + 1e-9)
ok(f"real circle: ellipse traction balances drag+rolling at v_ss (ratio {balance:.5f})", abs(balance - 1.0) < 1e-3)
ok("real circle: v_ap = sqrt(ay_max(v_ap)*R)",
   rel(v_ap, math.sqrt(np.interp(v_ap, env.v_grid, env.ay_max) * R)) < 1e-9)

# =============================================================================
HAVE_BCN = os.path.exists(os.path.join(ROOT, "Circuits", "Barcelona_circuit.mat"))
print(f"3. causality / monotonicity on {'BCN' if HAVE_BCN else 'VirtualTrack (BCN .mat missing)'}")
ctxb = Ctx()
userOpts(ctxb, circuit="BCN" if HAVE_BCN else "VirtualTrack")
envb = build_envelope(ctxb)
p = march(envb, ctxb.track.s, ctxb.track.k, vi=60.0)
v, s, ds = p["v"], p["s"], p["ds"]
ok("v <= apex speed everywhere", np.all(v <= p["v_apex"] + 1e-9))
ok("v <= v_cap (pt.Vmax) and v > 0", np.all(v <= envb.v_cap + 1e-9) and np.all(v > 0))
ok("v = min(forward, backward)", np.array_equal(v, np.minimum(p["v_fwd"], p["v_bwd"])))
ok("|ay| = v^2|k| <= ay_max(v)", np.all(np.abs(p["ay"]) <= np.interp(v, envb.v_grid, envb.ay_max) + 1e-6))
ok("v(0) = vi", p["vi_feasible"] and v[0] == 60.0)
ok("t strictly increasing, lap = t[-1]", np.all(np.diff(p["t"]) > 0) and p["lap_time"] == p["t"][-1])
def ax_limit(env, vv, brake):
    """Pure-longitudinal limit at vv with the analytic powertrain/brake terms
    (the ax_acc table itself is linear across the torque->power kink)."""
    res = env.F_res(vv)
    if brake:
        return np.minimum((np.interp(vv, env.v_grid, env.Fx_brk) + res) / env.M,
                          (env.F_brk_trq + res) / env.M_eff)
    return np.minimum((np.interp(vv, env.v_grid, env.Fx_trac) - res) / env.M,
                      (env.F_mot(vv) - res) / env.M_eff)


a_seg = (v[1:]**2 - v[:-1]**2) / (2 * ds)
acc_lim = np.maximum(ax_limit(envb, v[:-1], False), ax_limit(envb, v[1:], False))
brk_lim = np.maximum(ax_limit(envb, v[:-1], True), ax_limit(envb, v[1:], True))
ok(f"segment accel <= pure-longitudinal limit (max excess {np.max(a_seg - acc_lim):.1e})",
   np.all(a_seg <= acc_lim + 1e-6))
ok(f"segment decel <= pure-longitudinal limit (max excess {np.max(-a_seg - brk_lim):.1e})",
   np.all(-a_seg <= brk_lim + 1e-6))
ok("envelope sane: 0 < ay_max, ax_acc(v_cap) < ax_acc(2), ax_brk > 0",
   np.all(envb.ay_max > 0) and envb.ax_acc[-1] < envb.ax_acc[0] and np.all(envb.ax_brk > 0))

# =============================================================================
print("4. CasADi anchors (numpy envelope vs symbolic vehModel)")
try:
    import casadi as ca
    from vehModel import vehModel
    HAVE_CA = True
except Exception as exc:                                       # pragma: no cover
    HAVE_CA = False
    print(f"  [SKIP] casadi not importable ({exc}); symbolic anchors skipped")

if HAVE_CA:
    def x_state(vp, vx, Fz=(6500.0,) * 4, vy=0.0, Om=None):
        """Physical 23-state vector: planar speed, wheel speeds, tyre deflections."""
        x = np.zeros(23)
        x[0], x[1] = vx, vy
        x[5:9] = vx / vp.Rw if Om is None else Om
        x[19:23] = np.asarray(Fz, dtype=float) / vp.kt
        return x

    def vertical_equilibrium(c, v, vy=0.0, sx=0.0):
        """Quasi-steady vertical equilibrium of the 23-state model for a frozen
        planar state: solve heave/pitch/roll + 4 unsprung for (zs, theta, phi,
        zt_*). Returns [fz_fl, fz_fr, fz_rl, fz_rr, sum fx, Fy_front, Fy_rear]."""
        m, vp = c.m23, c.vp
        z = ca.SX.sym("z", 7)
        xp = ca.SX.zeros(23)
        xp[0], xp[1] = v, vy
        for i in (5, 6, 7, 8):
            xp[i] = (1.0 + sx) * v / vp.Rw
        xp[9], xp[11], xp[13] = z[0], z[1], z[2]
        for j, i in enumerate((19, 20, 21, 22)):
            xp[i] = z[3 + j]
        u = np.zeros(m.nu)
        for i, key in enumerate(c.input_keys):
            if key.strip() == "ATD":
                u[i] = 0.25
        syms = ca.vertcat(m.x, m.u, m.kappa)
        vals = ca.vertcat(xp / ca.DM(m.x_s), ca.DM(u / m.u_s), 0)
        dx = ca.substitute(m.dx, syms, vals)
        resid = ca.vertcat(*[dx[i] for i in (10, 12, 14, 15, 16, 17, 18)])
        sig = ca.substitute(ca.vertcat(m.fz_fl, m.fz_fr, m.fz_rl, m.fz_rr,
                                       m.fx_fl + m.fx_fr + m.fx_rl + m.fx_rr,
                                       m.fy_fl + m.fy_fr, m.fy_rl + m.fy_rr), syms, vals)
        solver = ca.rootfinder("vert_eq", "newton", ca.Function("resid", [z], [resid]))
        zz = solver(np.r_[0.0, 0.0, 0.0, np.full(4, 6500.0 / vp.kt)])
        return np.array(ca.Function("sig", [z], [sig])(zz)).ravel()

    # (a) peak mu(Fz): numpy vs symbolic mu_*_x / mu_*_y (camber + wing angles off-default)
    c4 = Ctx()
    userOpts(c4, circuit="Sturn",
             vp_overrides=dict(gamma_fl=-2.5, gamma_rl=-1.5, alpha_RW=11.0, alpha_TW=2.0))
    vehModel(c4)
    m, vp, mf = c4.m23, c4.vp, c4.mf
    f_mu = ca.Function("f_mu", [m.x], [m.mu_fl_x, m.mu_fl_y, m.mu_fr_y,
                                       m.mu_rl_x, m.mu_rl_y, m.mu_rr_y])
    worst = 0.0
    for Fz in (800.0, 3000.0, 4905.0, 6500.0, 9000.0, 13000.0):
        sym = [float(val) for val in f_mu(x_state(vp, 30.0, (Fz,) * 4) / m.x_s)]
        num = [peak_mu_x(Fz, mf, vp.Fz0), peak_mu_y(Fz, mf, vp.Fz0, vp.gamma_fl_rad),
               peak_mu_y(Fz, mf, vp.Fz0, vp.gamma_fr_rad), peak_mu_x(Fz, mf, vp.Fz0),
               peak_mu_y(Fz, mf, vp.Fz0, vp.gamma_rl_rad), peak_mu_y(Fz, mf, vp.Fz0, vp.gamma_rr_rad)]
        worst = max(worst, max(rel(float(a), b) for a, b in zip(num, sym)))
    ok(f"peak mu_x / mu_y(Fz, camber) == vehModel mu_*_x / mu_*_y (worst rel {worst:.1e})", worst < 1e-6)

    # (b) aero forces at 50 m/s: numpy coefficients vs symbolic f_lift_* / f_drag
    ac = aero_coefficients(vp, c4.aero)
    f_aero = ca.Function("f_aero", [m.x], [m.f_lift_fl, m.f_lift_fr, m.f_lift_rl, m.f_lift_rr,
                                           m.f_lift, m.f_drag])
    sym = [float(val) for val in f_aero(x_state(vp, 50.0) / m.x_s)]
    q = 0.5 * vp.rho * vp.A * 50.0**2
    num = [-q * ac["Cl_fl"], -q * ac["Cl_fr"], -q * ac["Cl_rl"], -q * ac["Cl_rr"], -q * ac["Cl"], q * ac["Cd"]]
    ok("aero f_lift_* / f_lift / f_drag == vehModel (rel 1e-9)",
       all(rel(a, b) < 1e-9 for a, b in zip(num, sym)))

    # (c) peak longitudinal force: max over slip ratio of vehModel fx == D = mu_x*Fz/Fz0_shift
    Fz = 7000.0
    f_fx = ca.Function("f_fx", [m.x, m.u], [m.fx_fl])          # u = 0: zero steer
    sx = np.linspace(-0.6, 0.6, 2401)
    fx = np.array([float(f_fx(x_state(vp, 30.0, (Fz,) * 4, Om=30.0 * (1 + s_) / vp.Rw) / m.x_s,
                              np.zeros(m.nu))) for s_ in sx])
    Dx = peak_mu_x(Fz, mf, vp.Fz0) * Fz / vp.Fz0_shift
    ok(f"max_sx fx = +D ({fx.max():.1f} vs {Dx:.1f} N), min = -D", rel(fx.max(), Dx) < 1e-3 and rel(-fx.min(), Dx) < 1e-3)

    # (d) quasi-steady loads ('vehModel' basis) vs the model's vertical equilibrium
    # Legacy CopyB set on purpose: the vertical-load model is tyre-set agnostic, but
    # MF205's pVy shifts give a lateral force at zero slip, so the 23-state
    # straight-running equilibrium is no longer symmetric (rel 5e-5 > tolerance).
    cb = Ctx()
    userOpts(cb, circuit="Sturn", tyre_set="CopyB")
    vehModel(cb)
    vg = np.array([20.0, 50.0, 80.0])
    e3 = build_envelope(cb, v_grid=vg)
    worst = 0.0
    for i, vv in enumerate(vg):
        fz = vertical_equilibrium(cb, vv)
        worst = max(worst, rel(fz[0] + fz[1], e3.Fz_front[i]), rel(fz[2] + fz[3], e3.Fz_rear[i]))
    # residual = the model's ms*g*cos(phi)*cos(theta) heave term (theta ~ -3 mrad at 80 m/s)
    ok(f"static + aero axle loads == 23-state equilibrium (worst rel {worst:.1e}, O(theta^2))", worst < 1e-5)
    ok("static tyre-load sum = ms*g + 4*mus*g (vehModel basis)",
       rel(e3.info["Fz_static"], cb.vp.ms * cb.vp.g + 4 * cb.vp.mus * cb.vp.g) < 1e-12)
    e40 = build_envelope(cb, v_grid=np.array([40.0]))
    fz = vertical_equilibrium(cb, 40.0, sx=-0.02)
    dF_eq = fz[0] + fz[1] - e40.Fz_front[0]
    dF_qss = -fz[4] * e40.info["h_x"] / (cb.vp.l_f + cb.vp.l_r)
    ok(f"longitudinal transfer, braking Fx={fz[4]:.0f} N: QSS {dF_qss:.0f} vs equilibrium {dF_eq:.0f} N (5%)",
       rel(dF_qss, dF_eq) < 0.05)

    # (e) lateral transfer + peak fy need non-zero cornering stiffness: the legacy
    #     CopyB base has pKy4 = 0 (Kya = 0, so vehModel's fy is identically zero); the
    #     overrides restore it (MF5.2 pKy4 = 2, pKy1 sign-flipped to vehModel's slip
    #     convention) without MF205's pVy shifts, so fy.max() equals Dy.
    fy_ship = vertical_equilibrium(cb, 30.0, vy=-1.0)[5:7]
    if np.all(np.abs(fy_ship) < 1e-6):
        print("  [NOTE] legacy tyre_set='CopyB' (pKy4=0): vehModel lateral tyre force is identically 0")
    cf = Ctx()
    userOpts(cf, circuit="Sturn", tyre_set="CopyB", vp_overrides=dict(pKy4=2.0, pKy1=20.505))
    vehModel(cf)
    fz = vertical_equilibrium(cf, 30.0, vy=-1.0)
    dF_f, dF_r = lateral_load_transfer(cf.vp, fz[5], fz[6])
    ok(f"lateral transfer (Fy_f={fz[5]:.0f}, Fy_r={fz[6]:.0f} N) == 23-state equilibrium",
       rel(dF_f, (fz[1] - fz[0]) / 2) < 1e-3 and rel(dF_r, (fz[3] - fz[2]) / 2) < 1e-3)
    mc = cf.m23
    f_fy = ca.Function("f_fy", [mc.x, mc.u], [mc.fy_rl])
    vys = np.linspace(-20.0, 0.0, 2001)
    fy = np.array([float(f_fy(x_state(cf.vp, 30.0, (Fz,) * 4, vy=vy_) / mc.x_s, np.zeros(mc.nu)))
                   for vy_ in vys])
    Dy = peak_mu_y(Fz, cf.mf, cf.vp.Fz0, cf.vp.gamma_rl_rad) * Fz / cf.vp.Fz0_shift
    ok(f"max_alpha fy = D = mu_y*Fz/Fz0_shift ({fy.max():.1f} vs {Dy:.1f} N, pKy4=2 set)", rel(fy.max(), Dy) < 1e-3)

    # (f) powertrain / brake mapping: wheel-force limits from vehModel's own motor
    #     speed and wheel-torque expressions (single motor and EM4, incl. Om/gear)
    for em4 in ("Off", "On"):
        ce = Ctx()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                     # ATD+EM4 guard warning
            userOpts(ce, circuit="Sturn", Electric_4Motors=em4)
        vehModel(ce)
        me, vpe, pte = ce.m23, ce.vp, ce.pt
        Om_sym = me.Om_motor_fl if pte.EM4 else me.Om_motor
        f_pt = ca.Function("f_pt", [me.x, me.u], [Om_sym, me.T_fl + me.T_fr + me.T_rl + me.T_rr])
        envp = build_envelope(ce, v_grid=np.array([20.0, 40.0, 70.0]))
        good = True
        for vv in envp.v_grid:
            u = np.zeros(me.nu)
            u[list(ce.input_keys).index("T_brake")] = -vpe.Tbrake_max
            if pte.ATD:
                u[[i for i, key in enumerate(ce.input_keys) if key == "ATD"]] = 0.25
            Om_m, T_sum = (float(val) for val in f_pt(x_state(vpe, vv) / me.x_s, u / me.u_s))
            T_mot = min(pte.Tmax, pte.Pmax / Om_m)              # motor_power / torque bounds
            F_sym = (4.0 if pte.EM4 else 1.0) * T_mot * vpe.gear / vpe.Rw
            good &= rel(float(envp.F_mot(vv)), F_sym) < 1e-9
            good &= rel(envp.F_brk_trq, -T_sum / vpe.Rw) < 1e-9
        ok(f"EM4={em4}: drive (torque/power) and brake-torque wheel forces == vehModel", good)
    # (g) fixed Tdist / brkB splits can never beat the free (ATD) distribution
    co = Ctx()
    userOpts(co, circuit="Sturn", ATD="Off")
    env_fix, env_free = build_envelope(co), build_envelope(cb)
    ok("fixed-split traction/braking capacity <= free split (ATD)",
       np.all(env_fix.Fx_trac <= env_free.Fx_trac + 1e-6) and np.all(env_fix.Fx_brk <= env_free.Fx_brk + 1e-6))

# =============================================================================
print("5. MLTP_screen end-to-end")
c5 = MLTP_screen("Sturn", vi=60.0, save=False)
d = c5.data
t_core = c5.elapsed["envelope"] + c5.elapsed["march"]
print(f"     Sturn: envelope {1e3 * c5.elapsed['envelope']:.1f} ms + march {1e3 * c5.elapsed['march']:.1f} ms "
      f"= {1e3 * t_core:.1f} ms (target < 100 ms)")
ok("Sturn envelope + march well under 1 s", t_core < 1.0)
ok("fidelity tag 'qss' + schema (s_full/k_full/v/t_opt/ax/ay/track/envelope)",
   d.fidelity == "qss" and all(hasattr(d, f) for f in
                               ("s_full", "k_full", "v", "t_opt", "ax", "ay", "track", "envelope", "N")))
ok("profile on the NLP collocation grid (N*(d+1)+1 points)",
   d.v.size == d.s_full.size == d.N * (d.OPT_d + 1) + 1 and d.t_opt.size == d.s_full.size)
ok("t_opt[-1] == lap_time and Sturn lap in a sane band (15-25 s)",
   abs(d.t_opt[-1] - d.lap_time) < 1e-9 and 15.0 < d.lap_time < 25.0)
ok("racing line = centreline (n = 0)", np.allclose(d.track["xopt"], d.track["x"]) and
   np.allclose(d.track["yopt"], d.track["y"]))
if HAVE_BCN:
    cb5 = MLTP_screen("BCN", vi=60.0, save=False)
    t_core = cb5.elapsed["envelope"] + cb5.elapsed["march"]
    print(f"     BCN:   envelope {1e3 * cb5.elapsed['envelope']:.1f} ms + march {1e3 * cb5.elapsed['march']:.1f} ms "
          f"= {1e3 * t_core:.1f} ms, lap ~ {cb5.data.lap_time:.2f} s")
    ok("BCN envelope + march well under 1 s", t_core < 1.0)
with tempfile.TemporaryDirectory() as tmp:
    c6 = MLTP_screen("Sturn", vi=60.0, save=True, results_dir=tmp, verbose=False)
    path = os.path.join(tmp, "Sturn_Static_ATDOn_EM4Off_qss.mat")
    back = load_solution(path)
    ok("saved Results/<circuit>_<cfg>_qss.mat round-trips (lap, fidelity, envelope)",
       os.path.exists(path) and abs(float(back.lap_time) - c6.data.lap_time) < 1e-12
       and str(back.fidelity) == "qss" and np.asarray(back.envelope.v_grid).size == 40)
    ok("default QSS file records mesh 'uniform', empty mesh_opts, tyre_set 'MF205'",
       str(back.mesh) == "uniform" and vars(back.mesh_opts) == {} and str(back.tyre_set) == "MF205")
    c7 = MLTP_screen("Sturn", vi=60.0, save=True, results_dir=tmp, verbose=False, mesh="curvature",
                     mesh_opts={"a": 2.0, "ds_max": None}, tyre_set="CopyB")
    back7 = load_solution(c7.out_path)
    ok("QSS file records mesh / mesh_opts (None dropped) / tyre_set",
       str(back7.mesh) == "curvature" and vars(back7.mesh_opts) == {"a": 2.0}
       and str(back7.tyre_set) == "CopyB")
cn = MLTP_screen("Sturn", vi=60.0, save=False, verbose=False, load_model="nominal")
ok("load_model='nominal': textbook m*g basis, slower than the vehModel basis",
   rel(cn.envelope.info["Fz_static"], cn.vp.m * cn.vp.g) < 1e-12 and cn.data.lap_time > d.lap_time)

# =============================================================================
print("6. numpy-only at runtime (casadi blocked)")
code = r"""
import sys
sys.modules["casadi"] = None                       # any 'import casadi' now raises
import numpy as np
from functions.context import Ctx
from Powertrain import Powertrain
from vehParams import vehParams
import functions.ggv as G
ctx = Ctx(); Powertrain(ctx); vehParams(ctx)
ctx.pt.EM4, ctx.pt.ATD, ctx.vp.ActAero = 0, 1, 0
s = np.linspace(0.0, 600.0, 301); k = np.where((s > 200) & (s < 260), 1.0 / 30.0, 0.0)
p = G.march(G.build_envelope(ctx), s, k, 50.0)
assert "casadi" not in sys.modules or sys.modules["casadi"] is None
print("LAP %.12f" % p["lap_time"])
"""
r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
ok("build_envelope + march run with casadi unavailable", r.returncode == 0 and "LAP" in r.stdout)
if r.returncode != 0:
    print(r.stdout, r.stderr)

# =============================================================================
print("7. screen_sweep")
res = screen_sweep("Sturn", [{}, {"Tbrake_max": 6000.0}, {"mb": 1600.0}])
laps = [r_["lap_time"] for r_ in res]
print(f"     laps: {[round(x, 3) for x in laps]}")
ok("one result per override set, overrides echoed", len(res) == 3 and res[1]["overrides"] == {"Tbrake_max": 6000.0})
ok("baseline sweep entry == MLTP_screen lap", abs(laps[0] - d.lap_time) < 1e-9)
ok("more brake torque / less mass -> faster", laps[1] < laps[0] and laps[2] < laps[0])

print("\nALL QSS SCREEN TESTS PASSED")
