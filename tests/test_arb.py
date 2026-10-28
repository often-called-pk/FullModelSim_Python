"""C3 anti-roll bars (docs/laptime_sim_improvements.md, item C3). vp.k_arb_f and vp.k_arb_r are the bar rates
at the wheel, in N/m. The default is 0.0, which means no bar.

A bar of rate k adds k*(xs_left - xs_right) to the left strut force. The right strut gets the opposite force.
Heave and pitch see no bar force. The roll moment resists roll, and the axle roll stiffness gains k*t^2 N m/rad.
The QSS (functions/ggv.lateral_load_transfer) adds 4*k_arb to the axle sum of the spring rates.

Sections:
  1 registry and defaults (no casadi)
  2 graph: a plain 0 adds nothing, a rate or a symbol adds the term
  3 force sign and shape
  4 roll stiffness
  5 steady load transfer against the QSS
  6 QSS lap
  7 sweep classification, RunConfig and the parity export
  8 zero-solve gradient of the stored Sturn optimum

Sections 2 to 8 need casadi and Data/DATA_AA.mat. Without the file, the test prints SKIP and exits 0.
Section 8 also needs Results/Sturn_Static_ATDOn_EM4Off.mat. No section runs IPOPT.
Run from anywhere:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_arb.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import sys
import warnings

import numpy as np

from app.runconfig import RunConfig
from app.vp_params import PARAM_GROUPS, meta_for
from vehParams import PRIMARY_KEYS, all_vp_defaults, default_primaries


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *args, **kwargs):
    """fn(*args, **kwargs) with stdout swallowed and warnings ignored."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# =============================================================================
print("1. registry and defaults")
ok("k_arb_f and k_arb_r are vehParams primaries, default 0.0",
   {"k_arb_f", "k_arb_r"} <= PRIMARY_KEYS and default_primaries()["k_arb_f"] == default_primaries()["k_arb_r"] == 0.0)
ok("k_arb_f and k_arb_r have GUI rows in 'Suspension stiffness', unit N/m",
   {"k_arb_f", "k_arb_r"} <= set(dict(PARAM_GROUPS)["Suspension stiffness"])
   and meta_for("k_arb_f").unit == meta_for("k_arb_r").unit == "N/m")
ok("ksD stays a primary (the MATLAB parity dumps hold it) and its row says unused",
   "ksD" in PRIMARY_KEYS and "unused" in meta_for("ksD").label.lower())

if not os.path.exists(os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")):   # pragma: no cover
    print("SKIP sections 2-8 (test_arb): Data/DATA_AA.mat missing")
    sys.exit(0)

# model imports come after the DATA_AA guard
import casadi as ca  # noqa: E402
import setup_sweep  # noqa: E402
import functions.transcription as T  # noqa: E402
from functions.ggv import lateral_load_transfer  # noqa: E402
from functions.context import Ctx  # noqa: E402
from MLTP_screen import MLTP_screen  # noqa: E402
from userOpts import userOpts  # noqa: E402
import validation.py_export as py_export  # noqa: E402
from vehModel import vehModel  # noqa: E402


def _no_ipopt(*args, **kwargs):
    raise RuntimeError("test_arb never runs IPOPT: a build reached the real _make_solver")


T._make_solver = _no_ipopt  # fake_ipopt() swaps in its own stand-in for the rebuilds of section 8


def build(ov=None, physics="matlab"):
    """Full 23-state model on Sturn with vp_overrides = ov."""
    c = Ctx()
    quiet(userOpts, c, circuit="Sturn", physics=physics, vp_overrides=dict(ov or {}))
    quiet(vehModel, c)
    return c


def physical_state(zs=0.0, theta=0.0, phi=0.0):
    """23-state vector at 40 m/s with the body at zs [m], theta and phi [rad]. Everything else 0."""
    x = np.zeros(23)
    x[0] = 40.0
    x[9], x[11], x[13] = zs, theta, phi
    return x


# =============================================================================
print("2. graph: a plain 0 adds nothing, a rate or a symbol adds the term")


def nodes(c):
    m = c.m23
    return ca.Function("f", [m.x, m.u, m.pv], [m.dx, m.sf]).n_nodes()


c_def = build()
ok("vp holds the defaults after userOpts", c_def.vp.k_arb_f == 0.0 and c_def.vp.k_arb_r == 0.0)
for physics in ("matlab", "corrected"):
    c0 = build(physics=physics)
    n0 = nodes(c0)
    c_zero = build({"k_arb_f": 0.0, "k_arb_r": 0}, physics)
    ok(f"{physics}: explicit 0.0 and 0 give the default graph",
       nodes(c_zero) == n0 and (c_zero.m23.nx, c_zero.m23.nu) == (c0.m23.nx, c0.m23.nu))
    n1 = nodes(build({"k_arb_f": 2e4}, physics))
    c_rear = build({"k_arb_r": 3e4}, physics)
    n2 = nodes(build({"k_arb_f": 2e4, "k_arb_r": 2e4}, physics))
    ok(f"{physics}: one front bar adds nodes, the rear bar adds more ({n0} < {n1} < {n2})", n0 < n1 < n2)
    ok(f"{physics}: a bar adds no state and no input", (c_rear.m23.nx, c_rear.m23.nu) == (c0.m23.nx, c0.m23.nu))

c = Ctx()
quiet(userOpts, c, circuit="Sturn")
k_f, k_r = ca.SX.sym("k_f"), ca.SX.sym("k_r")
c.vp.k_arb_f, c.vp.k_arb_r = k_f, k_r  # optimise_design sets a symbol like this
quiet(vehModel, c)
ok("symbolic k_arb_f and k_arb_r stay in the graph", ca.depends_on(c.m23.dx, k_f) and ca.depends_on(c.m23.dx, k_r))


# =============================================================================
print("3. force sign and shape")


def strut_reader(m):
    """Physical state -> [fzs_fl, fzs_fr, fzs_rl, fzs_rr, xs_fl, xs_fr, xs_rl, xs_rr] of model m (N, m)."""
    f = ca.Function("strut", [m.x, m.u, m.pv], [ca.vertcat(m.fzs_fl, m.fzs_fr, m.fzs_rl, m.fzs_rr,
                                                          m.xs_fl, m.xs_fr, m.xs_rl, m.xs_rr)])
    x_s = np.asarray(m.x_s, float)
    return lambda x: np.array(f(x / x_s, np.zeros(m.nu), 0.0)).ravel()


for physics in ("matlab", "corrected"):
    base = strut_reader(build(physics=physics).m23)
    front = strut_reader(build({"k_arb_f": 2e4}, physics).m23)
    rear = strut_reader(build({"k_arb_r": 3e4}, physics).m23)
    both = strut_reader(build({"k_arb_f": 2e4, "k_arb_r": 3e4}, physics).m23)

    x = physical_state(phi=5e-3)                   # phi > 0: right side down, right struts in bump
    x[19] = 1e-3                                   # zt_fl: corrected front and rear travel differences differ
    xs = base(x)[4:]
    d = front(x) - base(x)
    ok(f"{physics}: front bar: left strut pulled down, right pushed up, equal and opposite, rear untouched",
       d[0] < 0 < d[1] and abs(d[0] + d[1]) < 1e-9 * abs(d[0]) and abs(d[2]) + abs(d[3]) < 1e-6)
    ok(f"{physics}: front bar force is k (xs_left - xs_right) = {d[0]:.1f} N",
       abs(d[0] - 2e4 * (xs[0] - xs[1])) < 1e-9 * abs(d[0]))
    dr = rear(x) - base(x)
    ok(f"{physics}: rear bar force is k (xs_rl - xs_rr) = {dr[2]:.1f} N, front untouched",
       abs(dr[2] - 3e4 * (xs[2] - xs[3])) < 1e-9 * abs(dr[2]) and abs(dr[0]) + abs(dr[1]) < 1e-6)
    for name, xh in (("heave", physical_state(zs=0.01)), ("pitch", physical_state(theta=2e-3))):
        ok(f"{physics}: {name} gives no bar force", np.abs(both(xh) - base(xh))[:4].max() < 1e-6)
    if physics == "matlab":
        # xs_fr = 79 mm is past the MATLAB bump cut (xs >= 75 mm): the cut sets the strut force to 0. The bar acts
        # after the cut, on the capped travel 75 mm.
        xc = physical_state(zs=-0.07, phi=0.01)
        sc, dc = base(xc), front(xc) - base(xc)
        ok(f"matlab: past the bump cut the strut force is 0 and the front bar adds -k (xs_fl - 0.075) = {dc[1]:.1f} N",
           sc[1] == 0.0 and sc[5] == 0.075 and abs(dc[1] + 2e4 * (sc[4] - sc[5])) < 1e-9 * abs(dc[1]))


# =============================================================================
print("4. roll stiffness: d(phi_ddot)/d(phi) falls by k t^2 / I_x")


def roll_jacobian(c):
    """d(phi_ddot)/d(phi) of the model at 40 m/s with the body level. dx/dt = x_s * m.dx / sf."""
    m = c.m23
    xs_ = ca.SX.sym("x", 23)
    dxdt = ca.Function("dxdt", [m.x, m.u, m.pv], [m.dx * ca.DM(m.x_s) / m.sf])
    J = ca.Function("J", [xs_], [ca.jacobian(dxdt(xs_ / ca.DM(m.x_s), np.zeros(m.nu), 0.0)[14], xs_[13])])
    return float(J(physical_state()))


for physics in ("matlab", "corrected"):
    c0 = build(physics=physics)
    base = roll_jacobian(c0)
    dk_f = roll_jacobian(build({"k_arb_f": 2e4}, physics)) - base
    dk_r = roll_jacobian(build({"k_arb_r": 3e4}, physics)) - base
    t2_over_ix = c0.vp.t ** 2 / c0.vp.I_x
    ok(f"{physics}: front bar 2e4 changes d(phi_ddot)/d(phi) by {dk_f:.2f} 1/s^2 = -k t^2 / I_x",
       abs(dk_f + 2e4 * t2_over_ix) < 1e-6 * abs(dk_f))
    ok(f"{physics}: rear bar 3e4 changes d(phi_ddot)/d(phi) by {dk_r:.2f} 1/s^2 = -k t^2 / I_x",
       abs(dk_r + 3e4 * t2_over_ix) < 1e-6 * abs(dk_r))


# =============================================================================
print("5. steady lateral load transfer of the 23-state model against the QSS")


def equilibrium(c, v, vy):
    """Steady vertical equilibrium at speed v and lateral speed vy: frozen planar state, the seven vertical
    residuals solved for (zs, theta, phi, zt_*). Harness of test_screen.py section 4.
    Returns [fz x4, sum fx, Fy_front, Fy_rear] and z."""
    m, vp = c.m23, c.vp
    z = ca.SX.sym("z", 7)
    xp = ca.SX.zeros(23)
    xp[0], xp[1] = v, vy
    for i in (5, 6, 7, 8):
        xp[i] = v / vp.Rw
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
    solve = ca.rootfinder("vert_eq", "newton", ca.Function("resid", [z], [resid]))
    zz = solve(np.r_[0.0, 0.0, 0.0, np.full(4, 6500.0 / vp.kt)])
    return np.array(ca.Function("sig", [z], [sig])(zz)).ravel(), np.array(zz).ravel()


SETUPS = [{}, {"k_arb_f": 2e4}, {"k_arb_r": 2e4}, {"k_arb_f": 5e4, "k_arb_r": 3e4}]
share, roll = [], []
for ov in SETUPS:
    c = build(ov)
    fz, z = equilibrium(c, 30.0, -1.0)
    dFf, dFr = (fz[1] - fz[0]) / 2, (fz[3] - fz[2]) / 2
    qf, qr = lateral_load_transfer(c.vp, fz[5], fz[6])
    label = ", ".join(f"{k} = {v:g}" for k, v in ov.items()) or "no bars"
    ok(f"matlab, {label}: QSS transfer equals the model (front {dFf:.1f} N, rel 1e-6)",
       abs(qf - dFf) < 1e-6 * abs(dFf) and abs(qr - dFr) < 1e-6 * abs(dFr))
    share.append(dFf / (dFf + dFr))
    roll.append(z[2])
ok("a front bar moves the transfer to the front, a rear bar to the rear", share[1] > share[0] > share[2])
ok("every bar lowers the roll angle", roll[1] < roll[0] and roll[2] < roll[0] and roll[3] < min(roll[1], roll[2]))
c = build({"k_arb_f": 2e4}, "corrected")
fz, z = equilibrium(c, 30.0, -1.0)
dFf, dFr = (fz[1] - fz[0]) / 2, (fz[3] - fz[2]) / 2
qf, qr = lateral_load_transfer(c.vp, fz[5], fz[6])
ok(f"corrected, front bar: QSS split within 10 % of the model ({abs(qf - dFf) / abs(dFf):.1%} / "
   f"{abs(qr - dFr) / abs(dFr):.1%})",
   abs(qf - dFf) < 0.10 * abs(dFf) and abs(qr - dFr) < 0.10 * abs(dFr))


# =============================================================================
print("6. QSS lap")


def lap(ov=None, physics="matlab"):
    """QSS lap time on Sturn, vp_overrides = ov."""
    return float(quiet(MLTP_screen, "Sturn", save=False, verbose=False, physics=physics,
                       vp_overrides=ov or {}).data.lap_time)


t0 = lap()
# measured before C3, with the bars at 0: the default QSS lap must not move
ok(f"QSS Sturn matlab: the default lap is unchanged by C3 ({t0:.12f} s)", abs(t0 - 18.743926140324547) < 1e-9)
ok("QSS: explicit 0 gives the default lap, bit for bit", lap({"k_arb_f": 0.0, "k_arb_r": 0.0}) == t0)
tf, tr = lap({"k_arb_f": 2e4}), lap({"k_arb_r": 2e4})
tb = lap({"k_arb_f": 2e4, "k_arb_r": 2e4})
ok(f"QSS Sturn matlab: a front bar of 2e4 changes the lap by {100 * (tf / t0 - 1):+.3f} % (0.05 to 0.25 %)",
   1.0005 < tf / t0 < 1.0025)
ok(f"QSS Sturn matlab: a rear bar of 2e4 changes the lap by {100 * (tr / t0 - 1):+.3f} % (0.3 to 0.7 %)",
   1.003 < tr / t0 < 1.007)
ok(f"QSS Sturn matlab: equal bars change the lap by {100 * (tb / t0 - 1):+.3f} % (below 0.1 %)",
   abs(tb / t0 - 1) < 1e-3)
vp = build({"k_arb_f": 2e4, "k_arb_r": 3e4}).vp
Fy_f, Fy_r = 4000.0, 3500.0
k_front = vp.k_fl + vp.k_fr + 4 * 2e4
k_rear = vp.k_rl + vp.k_rr + 4 * 3e4
phi_f = k_front / (k_front + k_rear)
Mx = Fy_f * (vp.hcg - vp.hRCf) + Fy_r * (vp.hcg - vp.hRCr)
want = ((phi_f * Mx + Fy_f * vp.hRCf) / vp.t, ((1 - phi_f) * Mx + Fy_r * vp.hRCr) / vp.t)
ok("lateral_load_transfer: front share (k_fl + k_fr + 4 k_arb_f) / (sum of k + 4 k_arb_f + 4 k_arb_r)",
   np.allclose(lateral_load_transfer(vp, Fy_f, Fy_r), want, rtol=1e-12, atol=0))


# =============================================================================
print("7. sweep classification, RunConfig and the parity export")
flags = quiet(setup_sweep.classify_fields, [("k_arb_f", 0.0, 5e4), ("k_arb_r", 0.0, 5e4), ("ksD", 0.3, 0.7)],
              circuit="Sturn")
ok("k_arb_f, k_arb_r: primaries, NLP-live, promotable, and the QSS ranks them",
   all(flags[f]["primary"] and flags[f]["nlp_live"] and flags[f]["promotion_safe"] and flags[f]["qss_blind"] is False
       for f in ("k_arb_f", "k_arb_r")))
ok("ksD: NLP-inert, so a sweep drops it", flags["ksD"]["nlp_live"] is False and not flags["ksD"]["promotion_safe"])
old = {k: v for k, v in all_vp_defaults().items() if not k.startswith("k_arb")}
ok("a saved RunConfig from before C3 loads and forwards nothing", RunConfig.from_dict({"vp": old}).vp_overrides() == {})
ok("a RunConfig with a bar forwards that bar",
   RunConfig(vp={**all_vp_defaults(), "k_arb_f": 2e4}).vp_overrides() == {"k_arb_f": 2e4})


def export_vp(ov):
    """vp block of validation.py_export.params at Sturn, ds 30, with vp_overrides = ov (params() reads PARITY)."""
    saved = py_export.PARITY
    py_export.PARITY = {**saved, "vp_overrides": ov}
    try:
        return quiet(py_export.params, "Sturn", 30)["vp"]
    finally:
        py_export.PARITY = saved


vp0 = export_vp(None)
ok("parity export at default: no k_arb_f or k_arb_r in the vp block (sanity: ksD is there)",
   "k_arb_f" not in vp0 and "k_arb_r" not in vp0 and "ksD" in vp0)
for ov in ({"k_arb_f": 2e4}, {"k_arb_r": 2e4}):     # one bar alone must not hide the other key
    vp1, want = export_vp(ov), {"k_arb_f": 0.0, "k_arb_r": 0.0, **ov}
    got = {k: vp1.get(k) for k in want}
    ok(f"parity export, vp_overrides {ov}: the vp block has {got} == {want}", got == want)


# =============================================================================
print("8. zero-solve gradient of the stored Sturn optimum")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
if os.path.exists(STURN):
    from functions.importfile import load_solution  # noqa: E402
    from functions.sensitivity import setup_gradients  # noqa: E402
    import MLTP as mltp_mod  # noqa: E402

    os.environ["MLTP_KEEP_NLP"] = "1"  # MLTP keeps ctx.nlp_fg (never saved)

    @contextlib.contextmanager
    def fake_ipopt():
        """T._make_solver -> a stand-in: x = x0 with zero duals. No IPOPT."""
        def fake(ca_, nlp, opts, info=None):
            if info is not None:
                info["linear_solver"] = "fake"

            def solver(**k):
                return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                        "lam_x": np.zeros(nlp["x"].numel())}

            solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
            return solver

        real, T._make_solver = T._make_solver, fake
        try:
            yield
        finally:
            T._make_solver = real

    d = load_solution(STURN)
    w, lg = d.nlp.w_opt, d.nlp.lam_g

    def lagr(ov):
        """J + lam_g' g of the rebuilt Sturn NLP at the stored point w, vp_overrides = ov."""
        with fake_ipopt():
            cc = quiet(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False, vp_overrides=ov)
        J, g = cc.nlp_fg(w)
        return float(J) + lg @ np.asarray(g).ravel()

    bars = ("k_arb_f", "k_arb_r")
    G = quiet(setup_gradients, STURN, list(bars), circuit="Sturn")
    for p in bars:
        fd = (lagr({p: 500.0}) - lagr({p: -500.0})) / 1000.0
        ok(f"{p}: setup_gradients {G[p]:+.3e} s per N/m equals the central difference (rel 1e-6)",
           abs(G[p] - fd) <= 1e-6 * abs(fd))
        ok(f"{p}: the gradient is not zero", abs(fd) > 1e-9)
    ok("at ATD on the optimum a 2e4 N/m bar moves the lap by less than 0.02 s",
       all(abs(G[p]) * 2e4 < 0.02 for p in bars))
    print("\nALL anti-roll bar TESTS PASSED")
else:
    print("  SKIP 8: Results/Sturn_Static_ATDOn_EM4Off.mat missing")
    print("\nanti-roll bar TESTS PASSED (section 8 skipped)")
