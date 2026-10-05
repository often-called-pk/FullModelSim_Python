"""MLTP.py warm-start helpers: warmstart_guesses, warmstart_guesses_full, warmstart_full
(plain script, no pytest). Run from the repo root:

    venv\\Scripts\\python.exe tests\\test_mltp_warmstart.py

test_warmstart.py covers functions/warmstart.py (the casadi-free engine). This file covers
the three MLTP.py functions on top of it, the ones that hand the starting point to
build_and_solve_nlp:

  1. warmstart_guesses on the real 23-state model (Sturn): shapes, state / input row layout
  2. every row takes its own parameter and scale (distinct Rw_f / Rw_r, W0, kt, x_s, u_s)
  3. input rows by channel name across configurations (4 motors, wings, ATD, ...)
  4. the legacy scalar-N call form
  5. non-uniform grids: a curvature-mesh target, an init solution on its own knots
  6. length mismatches raise
  7. warmstart_guesses_full + warmstart_full: full+duals, full-primal, full-interp, cold

The 23-state vehModel is built (needs casadi, ~10 ms) but nothing is solved and no result
file is read, so the script runs in a few seconds. Without casadi it prints a SKIP line and
exits 0.
"""
import contextlib, copy, io, os, sys
from types import SimpleNamespace
import numpy as np
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)

try:
    import casadi as ca
except ImportError as exc:                                     # pragma: no cover
    print(f"[SKIP] casadi not importable ({exc}); MLTP warm-start helpers not tested")
    sys.exit(0)

from functions.context import Ctx
from functions.mesh import solution_knots
from functions.transcription import discretise, unpack_solution, reconstruct_x_full
from functions.warmstart import guesses_from_full, nlp_record, nlp_structure, warm_start_ipopt_opts
from userOpts import userOpts
from vehModel import vehModel
from MLTP import build_path_constraints, warmstart_full, warmstart_guesses, warmstart_guesses_full


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def same(a, b, rtol=1e-12, atol=1e-14):
    """Equal shapes AND allclose (np.allclose alone would broadcast a shape bug away)."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=rtol, atol=atol))


def identical(g1, g2):
    return set(g1) == set(g2) and all(np.array_equal(g1[k], g2[k]) for k in g1)


def raised(exc, fn, *args, **kw):
    """The message of the ``exc`` that fn(...) raises, or None if it does not raise."""
    try:
        fn(*args, **kw)
    except exc as e:
        return str(e)
    return None


def quiet(fn, *args, **kw):
    """(fn(...), what it printed): warmstart_full reports its mode on stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kw)
    return out, buf.getvalue()


# ---- setup: Sturn, default config (Static aero, ATD On, 1 motor), real model ----------------
print("setup: Sturn, default config, real 23-state model")
ctx = Ctx()
userOpts(ctx, circuit="Sturn", mesh="uniform")
vehModel(ctx)
m, vp, d = ctx.m23, ctx.vp, ctx.OPT_d
disc = discretise(ctx.track, ctx.OPT_ds, d, mesh=ctx.mesh, mesh_opts=ctx.mesh_opts)
N, s_knot = disc["N"], disc["s_knot"]
nu = len(ctx.input_keys)
nh = len(build_path_constraints(ca, m, ctx.pt)[0])
ok(f"model: nx = 23, nu = {nu} == len(input_keys), uniform mesh with N = {N} intervals",
   m.nx == 23 and m.nu == nu and disc["mesh"] == "uniform" and np.allclose(disc["dsk"], disc["dsk"][0]))

# synthetic 7-state init solution (MLTP_initial layout: vx vy r n eps Om_f Om_r / T_drive T_brake
# delta) on a uniform grid with another interval count than the target
N0 = 11
L = float(ctx.track.s[-1] - ctx.track.s[0])
s0 = ctx.track.s[0] + np.linspace(0.0, L, N0 + 1)
rs = np.random.RandomState(7)
X = np.vstack([rs.uniform(20, 60, N0 + 1), rs.uniform(-2, 2, N0 + 1), rs.uniform(-0.5, 0.5, N0 + 1),
               rs.uniform(-3, 3, N0 + 1), rs.uniform(-0.2, 0.2, N0 + 1),
               np.full(N0 + 1, 777.0), np.full(N0 + 1, -555.0)])    # Om_f, Om_r rows: never read
U = np.vstack([rs.uniform(0, 600, N0 + 1), rs.uniform(-4000, 0, N0 + 1), rs.uniform(-0.6, 0.6, N0 + 1)])
ok(f"init solution: 7 x {N0 + 1} states, 3 x {N0 + 1} inputs, N0 = {N0} != N = {N}",
   X.shape == (7, N0 + 1) and U.shape == (3, N0 + 1) and N0 != N)


def at(s_new, row, s_old=s0):
    return np.interp(s_new, s_old, row)


# =============================================================================
print("1. warmstart_guesses on the real model (init interpolated by arc length onto the target knots)")
X_in, U_in = X.copy(), U.copy()
g = warmstart_guesses(ctx, m, X, U, s_knot)
x0, u0, xc0 = g["x0"], g["u0"], g["xc0"]
ok("returns exactly {x0, u0, xc0}", set(g) == {"x0", "u0", "xc0"})
ok(f"shapes: x0 (23, N+1), u0 ({nu}, N+1), xc0 (23, N*OPT_d)",
   x0.shape == (23, N + 1) and u0.shape == (nu, N + 1) and xc0.shape == (23, N * d))
ok("all finite", all(np.all(np.isfinite(v)) for v in g.values()))
ok("the init arrays are not modified", np.array_equal(X, X_in) and np.array_equal(U, U_in))

ok("vx, vy, r, n, eps = np.interp of the init rows at the target knots / x_s",
   all(same(x0[i], at(s_knot, X[i]) / m.x_s[i]) for i in range(5)))
vx0 = at(s_knot, X[0])
ok("Om_fl = Om_fr = vx / Rw_f and Om_rl = Om_rr = vx / Rw_r, scaled; init Om rows ignored",
   same(x0[5], vx0 / vp.Rw_f / m.x_s[5]) and same(x0[6], vx0 / vp.Rw_f / m.x_s[6])
   and same(x0[7], vx0 / vp.Rw_r / m.x_s[7]) and same(x0[8], vx0 / vp.Rw_r / m.x_s[8]))
ok("zs, zsdot, theta, thetadot, phi, phidot, wu_fl..rr (rows 9-18) are exactly 0",
   np.all(x0[9:19] == 0.0))
W0 = np.array([vp.Wfl0, vp.Wfr0, vp.Wrl0, vp.Wrr0])
ok("zt_fl..rr = static tyre deflection W0 / kt (== vp.xti_*), scaled, constant along the lap",
   same(x0[19:23], np.tile((W0 / vp.kt / m.x_s[19:23])[:, None], (1, N + 1)))
   and same(W0 / vp.kt, [vp.xti_fl, vp.xti_fr, vp.xti_rl, vp.xti_rr]))

Ui = dict(T=at(s_knot, U[0]), B=at(s_knot, U[1]), D=at(s_knot, U[2]))
rows = [Ui["T"], Ui["B"]] + [np.full(N + 1, 0.25)] * 4 + [Ui["D"]]
ok("default keys [T_motor, T_brake, ATD x4, delta]: init torque / brake / steering by key, "
   "ATD rows 0.25, all scaled by u_s",
   list(ctx.input_keys) == ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"]
   and same(u0, np.vstack(rows) / m.u_s[:, None]))
ok("xc0: knot k's state repeated OPT_d times for interval k",
   all(np.array_equal(xc0[:, k * d:(k + 1) * d], np.repeat(x0[:, [k]], d, axis=1)) for k in range(N)))
st = nlp_structure(m.nx, m.nu, m.ny, N, d, nh)
w0 = np.concatenate([x0.reshape(-1, order="F"), u0.reshape(-1, order="F"), xc0.reshape(-1, order="F")])
ok(f"packed like build_and_solve_nlp's w0 the guesses fill the decision vector exactly (n_w = {st['n_w']})",
   w0.size == st["n_w"])

# =============================================================================
print("2. every row takes its own parameter and its own scale")
vp2 = SimpleNamespace(**vars(vp))
vp2.Rw_f, vp2.Rw_r, vp2.kt = 0.30, 0.40, 2.5e5
vp2.Wfl0, vp2.Wfr0, vp2.Wrl0, vp2.Wrr0 = 4000.0, 4500.0, 5000.0, 5500.0
ctx2, m2 = copy.copy(ctx), copy.copy(m)
ctx2.vp = vp2
m2.x_s = 1.0 + 0.37 * np.arange(23)                     # a different scale for every state
m2.u_s = 2.0 + 0.9 * np.arange(nu)                      # and every input
g2 = warmstart_guesses(ctx2, m2, X, U, s_knot)
phys = np.zeros((23, N + 1))                            # the documented state order, spelled out
phys[0:5] = [at(s_knot, X[i]) for i in range(5)]
phys[5] = phys[6] = vx0 / 0.30                          # fl, fr
phys[7] = phys[8] = vx0 / 0.40                          # rl, rr
phys[19:23] = np.array([4000.0, 4500.0, 5000.0, 5500.0])[:, None] / 2.5e5   # zt_fl, fr, rl, rr
ok("x0 = [vx vy r n eps | Om fl fr rl rr | 10 zero rows | zt fl fr rl rr] / x_s, row by row",
   same(g2["x0"], phys / m2.x_s[:, None]))
phys_u = np.vstack([Ui["T"], Ui["B"]] + [np.full(N + 1, 0.25)] * 4 + [Ui["D"]])
ok("u0 = [T_motor, T_brake, ATD x4, delta] / u_s, row by row", same(g2["u0"], phys_u / m2.u_s[:, None]))
ok("xc0 follows the same scaled states", same(g2["xc0"], np.kron(g2["x0"][:, :-1], np.ones((1, d)))))

# =============================================================================
print("3. input rows by channel name across configurations (real model per config)")
# (label, userOpts kwargs, input_keys, physical seed per row: T / B / D = the interpolated init
#  T_motor / T_brake / delta rows, a number = constant seed)
CONFIGS = [
    ("Static, ATD Off", dict(ATD="Off"),
     ["T_motor", "T_brake", "delta"], ["T", "B", "D"]),
    ("Active_RW, ATD On", dict(AeroConfig="Active_RW", ATD="On"),
     ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "RW", "delta"],
     ["T", "B", .25, .25, .25, .25, 0., "D"]),
    ("Active (FW + RW), ATD Off", dict(AeroConfig="Active", ATD="Off"),
     ["T_motor", "T_brake", "FW", "RW", "delta"], ["T", "B", 0., 0., "D"]),
    ("AALB + ATD", dict(AeroConfig="AALB", ATD="On"),
     ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "FW", "FW", "RW", "TW", "delta"],
     ["T", "B", .25, .25, .25, .25, 0., 0., 0., 0., "D"]),
    ("4 motors (ATD forced Off)", dict(ATD="Off", Electric_4Motors="On"),
     ["T_motor_fl", "T_motor_fr", "T_motor_rl", "T_motor_rr", "T_brake", "delta"],
     ["T", "T", "T", "T", "B", "D"]),
    ("4 motors + AALB", dict(AeroConfig="AALB", ATD="Off", Electric_4Motors="On"),
     ["T_motor_fl", "T_motor_fr", "T_motor_rl", "T_motor_rr", "T_brake", "FW", "FW", "RW", "TW", "delta"],
     ["T", "T", "T", "T", "B", 0., 0., 0., 0., "D"]),
]
for label, kw, keys, seeds in CONFIGS:
    cc = Ctx()
    userOpts(cc, circuit="Sturn", mesh="uniform", **kw)
    vehModel(cc)
    mc = cc.m23
    gc = warmstart_guesses(cc, mc, X, U, s_knot)
    expect = np.vstack([Ui[t] if isinstance(t, str) else np.full(N + 1, t) for t in seeds])
    ok(f"{label}: nu = {len(keys)}, rows {keys[0]}..{keys[-1]} seeded by key and scaled by u_s",
       list(cc.input_keys) == keys and gc["u0"].shape == (len(keys), N + 1)
       and same(gc["u0"], expect / mc.u_s[:, None]) and gc["xc0"].shape == (23, N * d))
    ok(f"{label}: the state guesses do not depend on the config", np.array_equal(gc["x0"], x0))

# =============================================================================
print("4. legacy scalar-N call: both grids taken as uniform on [0, 1]")
u01 = np.linspace(0.0, 1.0, N + 1)
g_legacy = warmstart_guesses(ctx, m, X, U, N)
ok("scalar N == the array call on the uniform unit-interval grid, bit for bit (x0, u0, xc0)",
   identical(g_legacy, warmstart_guesses(ctx, m, X, U, u01)))
ok("legacy x0 = np.interp of the init rows from linspace(0, 1, N0+1) onto linspace(0, 1, N+1), / x_s",
   same(g_legacy["x0"][:5], np.vstack([np.interp(u01, np.linspace(0.0, 1.0, N0 + 1), X[i])
                                       for i in range(5)]) / m.x_s[:5, None]))
ok("the physical uniform grid gives the same guesses (same grids up to scale; equal to rounding)",
   same(g_legacy["x0"], g["x0"]) and same(g_legacy["u0"], g["u0"]) and same(g_legacy["xc0"], g["xc0"]))
ok("legacy shapes follow N", g_legacy["x0"].shape == (23, N + 1) and g_legacy["u0"].shape == (nu, N + 1)
   and g_legacy["xc0"].shape == (23, N * d))

# =============================================================================
print("5. non-uniform grids: interpolation is by physical arc length")
a_x, b_x = np.array([20.0, 1.5, -0.2, 2.0, 0.05, 9.0, 9.0]), np.array([0.04, -0.003, 0.0004, 0.005, -0.0002, 0, 0])
a_u, b_u = np.array([100.0, -2000.0, 0.1]), np.array([0.5, 2.0, -0.0005])


def line_x(s):
    return a_x[:, None] + b_x[:, None] * np.asarray(s)[None, :]


def line_u(s):
    return a_u[:, None] + b_u[:, None] * np.asarray(s)[None, :]


disc_c = discretise(ctx.track, ctx.OPT_ds, d, mesh="curvature")
sk_c = disc_c["s_knot"]
ok(f"curvature-mesh target: same N = {disc_c['N']}, genuinely non-uniform "
   f"(intervals {disc_c['dsk'].min():.1f}..{disc_c['dsk'].max():.1f} m)",
   disc_c["N"] == N and disc_c["dsk"].max() > 1.5 * disc_c["dsk"].min())
gc_rand = warmstart_guesses(ctx, m, X, U, sk_c)
ok("curvature target, random init rows: x0 / u0 = np.interp at the PHYSICAL knots (states / inputs)",
   all(same(gc_rand["x0"][i], at(sk_c, X[i]) / m.x_s[i]) for i in range(5))
   and all(same(gc_rand["u0"][j], at(sk_c, U[k]) / m.u_s[j]) for j, k in ((0, 0), (1, 1), (6, 2))))
gc_line = warmstart_guesses(ctx, m, line_x(s0), line_u(s0), sk_c)
ok("curvature target, init rows linear in s: the line evaluated at the physical knots (no np.interp)",
   same(gc_line["x0"][:5], line_x(sk_c)[:5] / m.x_s[:5, None])
   and same(gc_line["u0"][[0, 1, 6]], line_u(sk_c) / m.u_s[[0, 1, 6], None]))
gu_line = warmstart_guesses(ctx, m, line_x(s0), line_u(s0), s_knot)
ok("... which differs from the uniform-mesh guesses (the knot positions matter)",
   not same(gc_line["x0"][0], gu_line["x0"][0], rtol=1e-3, atol=0.0)
   and same(gu_line["x0"][:5], line_x(s_knot)[:5] / m.x_s[:5, None]))

disc_i = discretise(ctx.track, 45, d, mesh="curvature")      # an init solution on its own mesh
N_i = disc_i["N"]
s_i = solution_knots(SimpleNamespace(s_full=disc_i["s_full"], OPT_d=d), N_i + 1)   # as MLTP() does
ok(f"init on its own non-uniform knots: N0 = {N_i} != N = {N}, knots = solution_knots(s_full, OPT_d)",
   N_i != N and s_i.size == N_i + 1 and np.array_equal(s_i, disc_i["s_knot"]) and np.ptp(np.diff(s_i)) > 1.0)
gi = warmstart_guesses(ctx, m, line_x(s_i), line_u(s_i), sk_c, s_i)
ok("s_knot_init given: the line again at the target knots (curvature -> curvature, other N)",
   same(gi["x0"][:5], line_x(sk_c)[:5] / m.x_s[:5, None])
   and same(gi["u0"][[0, 1, 6]], line_u(sk_c) / m.u_s[[0, 1, 6], None]))
gi_u = warmstart_guesses(ctx, m, line_x(s_i), line_u(s_i), s_knot, s_i)
ok("s_knot_init given: uniform target from a curvature-mesh init",
   same(gi_u["x0"][:5], line_x(s_knot)[:5] / m.x_s[:5, None]))
ok("... while omitting it assumes a uniform init grid and lands elsewhere",
   not same(warmstart_guesses(ctx, m, line_x(s_i), line_u(s_i), sk_c)["x0"][0], gi["x0"][0], rtol=1e-3, atol=0.0))
s_short = np.linspace(0.0, 0.9 * L, N0 + 1)                  # an init solution that stops short of the lap end
g_short = warmstart_guesses(ctx, m, X, U, s_knot, s_short)
beyond = s_knot > s_short[-1]
ok(f"{int(beyond.sum())} target knots beyond the init span hold the init end values (clamped, no extrapolation)",
   beyond.sum() >= 2 and same(g_short["x0"][0][beyond], np.full(beyond.sum(), X[0][-1] / m.x_s[0]))
   and same(g_short["u0"][6][beyond], np.full(beyond.sum(), U[2][-1] / m.u_s[6]))
   and same(g_short["x0"][3][~beyond], at(s_knot[~beyond], X[3], s_short) / m.x_s[3]))
Xr = np.vstack([rs.uniform(20, 60, N_i + 1) for _ in range(7)])
Ur = np.vstack([rs.uniform(0, 600, N_i + 1), rs.uniform(-4000, 0, N_i + 1), rs.uniform(-0.6, 0.6, N_i + 1)])
gr = warmstart_guesses(ctx, m, Xr, Ur, sk_c, list(s_i))
ok("s_knot_init may be a list; random rows = np.interp(s_knot, s_knot_init, row) / scale",
   all(same(gr["x0"][i], at(sk_c, Xr[i], s_i) / m.x_s[i]) for i in range(5))
   and same(gr["u0"][6], at(sk_c, Ur[2], s_i) / m.u_s[6]))

# =============================================================================
print("6. length mismatch raises")
msg = raised(ValueError, warmstart_guesses, ctx, m, X, U, s_knot, np.linspace(0, L, N0 + 2))
ok(f"s_knot_init with {N0 + 2} points for {N0 + 1} init knots raises ValueError naming s_knot_init",
   msg is not None and "s_knot_init" in msg and str(N0 + 2) in msg and str(N0 + 1) in msg)
ok("one point too few raises too", raised(ValueError, warmstart_guesses, ctx, m, X, U, s_knot, s0[:-1]) is not None)
ok("init_u with another number of knots than init_x raises ValueError",
   raised(ValueError, warmstart_guesses, ctx, m, X, U[:, :-1], s_knot) is not None)

# =============================================================================
print("7. warmstart_guesses_full / warmstart_full on an in-memory full result (nothing solved)")
# a fake data namespace of the Sturn structure, built the way MLTP() builds it from w_opt
rs = np.random.RandomState(0)
w_opt = rs.uniform(0.1, 0.9, st["n_w"])
lam_g, lam_x = rs.randn(st["n_g"]), rs.randn(st["n_w"])
x_opt, u_opt, _, xc_opt = unpack_solution(w_opt, m.nx, m.nu, m.ny, N, d, m.x_s, m.u_s, None)
x_full = reconstruct_x_full(x_opt, xc_opt, m.nx, N, d)
rec = nlp_record(dict(w_opt=w_opt, lam_g=lam_g, lam_x=lam_x, structure=st, sym_type="SX", linear_solver="ma57"),
                 {"iter_count": 257, "return_status": "Solve_Succeeded"}, m.x_s, m.u_s, "cold")
src = SimpleNamespace(x_opt=x_opt, u_opt=u_opt, x_full=x_full, s_full=disc["s_full"], N=N, OPT_d=d,
                      input_keys=list(ctx.input_keys), tyre_set=ctx.tyre_set, nlp=rec)
ok(f"source shapes from the real discretise + model: x_opt (23, {N + 1}), u_opt ({nu}, {N + 1}), "
   f"x_full (23, {disc['s_full'].size}), n_w = {st['n_w']}, n_g = {st['n_g']}",
   x_opt.shape == (23, N + 1) and u_opt.shape == (nu, N + 1) and x_full.shape == (23, disc["s_full"].size)
   and w_opt.size == st["n_w"] and lam_g.size == st["n_g"] and ctx.tyre_set == "MF205")
col = np.array([k * (d + 1) + 1 + j for k in range(N) for j in range(d)])   # collocation columns of s_full
ok("collocation points of s_full are disc['s_col']", np.array_equal(disc["s_full"][col], disc["s_col"]))

g_full = warmstart_guesses_full(ctx, m, src, disc)
ok("warmstart_guesses_full = guesses_from_full(src, s_knot, s_col, input_keys, x_s, u_s)",
   identical(g_full, guesses_from_full(src, disc["s_knot"], disc["s_col"], ctx.input_keys, m.x_s, m.u_s)))
ok("same grid: x0 * x_s == x_opt, u0 * u_s == u_opt, xc0 * x_s == the collocation states",
   same(g_full["x0"] * m.x_s[:, None], x_opt) and same(g_full["u0"] * m.u_s[:, None], u_opt)
   and same(g_full["xc0"] * m.x_s[:, None], x_full[:, col]) and same(g_full["xc0"] * m.x_s[:, None], xc_opt))

# --- identical structure: 'full+duals' ---------------------------------------------------
(gf, warm, mode), out = quiet(warmstart_full, ctx, m, src, disc, nh)
ok("identical structure -> mode 'full+duals', reported on stdout", mode == "full+duals" and mode in out)
ok("... with no variable-rescaling note (m.x_s / m.u_s equal what the source was solved with)",
   "x_s" not in out and "u_s" not in out)
ok("warm = the exact saved w_opt / lam_g / lam_x + the IPOPT dual warm-start recipe",
   set(warm) == {"x0", "lam_g0", "lam_x0", "ipopt"} and np.array_equal(warm["x0"], w_opt)
   and np.array_equal(warm["lam_g0"], lam_g) and np.array_equal(warm["lam_x0"], lam_x)
   and warm["ipopt"] == warm_start_ipopt_opts())
ok("guesses come from the source primal (same arrays as warmstart_guesses_full)", identical(gf, g_full))
wg = np.concatenate([gf["x0"].reshape(-1, order="F"), gf["u0"].reshape(-1, order="F"),
                     gf["xc0"].reshape(-1, order="F")])
ok("re-packed like build_and_solve_nlp's w0, the guesses reproduce the saved primal vector w_opt",
   same(wg, w_opt))
ctx_o = copy.copy(ctx)
ctx_o.ipopt_overrides = {"mu_init": 1e-4, "tol": 1e-6}
(_, warm_o, mode_o), _ = quiet(warmstart_full, ctx_o, m, src, disc, nh)
ok("ctx.ipopt_overrides win over the recipe, the rest of the recipe stays",
   mode_o == "full+duals" and warm_o["ipopt"]["mu_init"] == 1e-4 and warm_o["ipopt"]["tol"] == 1e-6
   and warm_o["ipopt"]["warm_start_init_point"] == "yes")
(gd, _, mode_d), _ = quiet(warmstart_full, ctx, m, dict(vars(src)), disc, nh)
ok("the source may be a dict (data dict) as well as a namespace", mode_d == "full+duals" and identical(gd, gf))
src_rescaled = SimpleNamespace(**dict(vars(src), nlp=dict(rec, x_s=m.x_s * 1.05)))
(_, warm_r, mode_r), out_r = quiet(warmstart_full, ctx, m, src_rescaled, disc, nh)
ok("a source with another variable scaling (setup change, the sweep case) still re-injects, noting x_s",
   mode_r == "full+duals" and warm_r is not None and "x_s" in out_r)

# --- identical structure, use_duals=False: 'full-primal' ---------------------------------
(gp, warm_p, mode_p), out_p = quiet(warmstart_full, ctx, m, src, disc, nh, use_duals=False)
ok("use_duals=False -> mode 'full-primal', reported on stdout", mode_p == "full-primal" and mode_p in out_p)
ok("warm = {x0: the exact saved w_opt} only (no multipliers, no IPOPT recipe)",
   set(warm_p) == {"x0"} and np.array_equal(warm_p["x0"], w_opt))
ok("full-primal guesses are the same arrays as with duals", identical(gp, gf))

# --- structure differs: 'full-interp' ----------------------------------------------------
disc_f = discretise(ctx.track, 15, d)                                  # OPT_ds = 15: N = 36, same degree
Nf = disc_f["N"]
(gi2, warm_i, mode_i), out_i = quiet(warmstart_full, ctx, m, src, disc_f, nh)
ok(f"another N ({N} -> {Nf}) -> mode 'full-interp', no warm vectors, the N change named in the note",
   Nf != N and mode_i == "full-interp" and warm_i is None and mode_i in out_i and f"N {N} -> {Nf}" in out_i)
ok("guess shapes follow the new grid: x0 (23, Nf+1), u0 (nu, Nf+1), xc0 (23, Nf*OPT_d)",
   gi2["x0"].shape == (23, Nf + 1) and gi2["u0"].shape == (nu, Nf + 1) and gi2["xc0"].shape == (23, Nf * d))
ok("knot states / inputs = the source primal interpolated by arc length at the new knots",
   all(same(gi2["x0"][i] * m.x_s[i], at(disc_f["s_knot"], x_opt[i], disc["s_knot"])) for i in range(23))
   and all(same(gi2["u0"][j] * m.u_s[j], at(disc_f["s_knot"], u_opt[j], disc["s_knot"])) for j in range(nu)))
ok("collocation states = the source x_full interpolated at the new collocation points",
   all(same(gi2["xc0"][i] * m.x_s[i], at(disc_f["s_col"], x_full[i], disc["s_full"])) for i in range(23)))

(gm, warm_m, mode_m), out_m = quiet(warmstart_full, ctx, m, src, disc_c, nh)
ok("same N on another mesh (curvature) -> 'full-interp': the collocation grid s_full differs",
   mode_m == "full-interp" and warm_m is None and "s_full" in out_m)
ok("... guesses at the curvature knots, interpolated from the uniform-mesh source",
   all(same(gm["x0"][i] * m.x_s[i], at(sk_c, x_opt[i], disc["s_knot"])) for i in range(23))
   and gm["xc0"].shape == (23, N * d))

ctx_d2 = copy.copy(ctx)
ctx_d2.OPT_d = 2
disc_d2 = discretise(ctx.track, ctx.OPT_ds, 2)
(gd2, warm_d2, mode_d2), out_d2 = quiet(warmstart_full, ctx_d2, m, src, disc_d2, nh)
ok("another collocation degree (OPT_d 3 -> 2, same N and knots) -> 'full-interp', the change named in the note",
   disc_d2["N"] == N and mode_d2 == "full-interp" and warm_d2 is None and "OPT_d 3 -> 2" in out_d2)
ok("... xc0 has OPT_d = 2 columns per interval, taken from the source x_full; the knot guesses are unchanged",
   gd2["xc0"].shape == (23, N * 2)
   and all(same(gd2["xc0"][i] * m.x_s[i], at(disc_d2["s_col"], x_full[i], disc["s_full"])) for i in range(23))
   and same(gd2["x0"] * m.x_s[:, None], x_opt))
g_d2 = warmstart_guesses(ctx_d2, m, X, U, s_knot)
ok("warmstart_guesses follows ctx.OPT_d too: xc0 (23, N*2) with the same x0",
   g_d2["xc0"].shape == (23, N * 2) and np.array_equal(g_d2["x0"], x0))

(_, warm_h, mode_h), out_h = quiet(warmstart_full, ctx, m, src, disc, nh + 1)
ok("another path-constraint count nh changes n_g -> 'full-interp' (nh is part of the structure)",
   mode_h == "full-interp" and warm_h is None and f"n_g {st['n_g']} ->" in out_h)

src_nonlp = SimpleNamespace(**{k: v for k, v in vars(src).items() if k != "nlp"})
(g_n, warm_n, mode_n), _ = quiet(warmstart_full, ctx, m, src_nonlp, disc, nh)
ok("a full result without data.nlp (older .mat) -> 'full-interp', primal guesses still taken from it",
   mode_n == "full-interp" and warm_n is None and same(g_n["x0"] * m.x_s[:, None], x_opt))
(_, warm_nd, mode_nd), _ = quiet(warmstart_full, ctx, m, src, disc_f, nh, use_duals=False)
ok("use_duals=False cannot help a different structure: still 'full-interp'",
   mode_nd == "full-interp" and warm_nd is None)

# a config change: 4-motor target seeded from the 1-motor + ATD source (channels matched by name)
ctx_e = Ctx()
userOpts(ctx_e, circuit="Sturn", mesh="uniform", ATD="Off", Electric_4Motors="On")
vehModel(ctx_e)
m_e = ctx_e.m23
nh_e = len(build_path_constraints(ca, m_e, ctx_e.pt)[0])
(ge, warm_e, mode_e), out_e = quiet(warmstart_full, ctx_e, m_e, src, disc, nh_e)
ok("config change (4-motor target, 1-motor + ATD source) -> 'full-interp', input_keys named in the note",
   mode_e == "full-interp" and warm_e is None and "input_keys" in out_e)
ok("... every corner motor seeded with the source motor torque, brake / steering matched by name",
   ge["u0"].shape == (6, N + 1)
   and same(ge["u0"] * m_e.u_s[:, None], np.vstack([u_opt[0]] * 4 + [u_opt[1], u_opt[6]])))

# --- 'cold': a source from another tyre set is no start at all ---------------------------
src_notyre = SimpleNamespace(**{k: v for k, v in vars(src).items() if k != "tyre_set"})
res_c, out_c = quiet(warmstart_full, ctx, m, src_notyre, disc, nh)
ok("source without tyre_set (a pre-flip CopyB result) vs the MF205 target -> (None, None, 'cold')",
   res_c == (None, None, "cold"))
ok("... and says so on stdout, naming both tyre sets", "ignored" in out_c and "CopyB" in out_c and "MF205" in out_c)
src_copyb = SimpleNamespace(**dict(vars(src), tyre_set="CopyB"))
ok("an explicit CopyB source is cold too, with duals or primal-only",
   quiet(warmstart_full, ctx, m, src_copyb, disc, nh)[0] == (None, None, "cold")
   and quiet(warmstart_full, ctx, m, src_copyb, disc, nh, use_duals=False)[0] == (None, None, "cold"))
ctx_cb = copy.copy(ctx)
ctx_cb.tyre_set = "CopyB"
ok("the target's tyre set is ctx.tyre_set: a CopyB target rejects the MF205 source -> cold",
   quiet(warmstart_full, ctx_cb, m, src, disc, nh)[0] == (None, None, "cold"))
ok("... and accepts a CopyB source ('full+duals')",
   quiet(warmstart_full, ctx_cb, m, src_copyb, disc, nh)[0][2] == "full+duals")

print("\nALL MLTP warm-start TESTS PASSED")
