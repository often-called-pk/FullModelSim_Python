"""B6: the mesh report (interval durations against the vertical modes) and the opt-in speed term
of the curvature mesh (docs/laptime_sim_improvements.md, item B6).

Sections 1 and 2 need no model file. They cover functions.mesh.mesh_report, the speed term of
monitor and curvature_mesh on a synthetic lap, and discretise. Section 3 needs Data/DATA_AA.mat
for the QSS speed (track.v_qss). Sections 4 to 6 also need the stored Sturn result. Section 4
compares vehModel.vertical_eigs with the AD block eigenvalues of tests/test_physics_suspension_aero.py.
Sections 5 and 6 run MLTP with IPOPT faked (no solve). Without the files the run prints SKIP and
exits 0. Run from any directory:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_mesh_report.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np

from functions.context import Ctx
from functions.mesh import curvature_mesh, mesh_opts_record, mesh_report, monitor
from functions.transcription import discretise

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
MODEL = os.path.exists(DATA_AA) and os.path.exists(STURN)


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout and warnings swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def loud(fn, *a, **kw):
    """(result, stdout text) of fn(*a, **kw), warnings swallowed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fn(*a, **kw)
    return out, buf.getvalue()


def error_of(fn, *a, **kw):
    """The exception that fn(*a, **kw) raises, or None when it returns."""
    try:
        quiet(fn, *a, **kw)
    except Exception as exc:
        return exc
    return None


def n_braking(m):
    """Knots in the braking zone, 400 m to 560 m."""
    return int(np.sum((m >= 400) & (m <= 560)))


# =============================================================================================
print("1. mesh_report: the text")
pairs = [-7.4 + 12.0j, -13.0 + 19.1j, -12.1 + 32.7j]
# the last two are slower than the longest interval
reals = [-11.9, -18.4, -35.9, -84.5, -93.8, -145.6, -162.6, -191.1, -0.02, -0.05]
ev = np.array(pairs + [np.conj(p) for p in pairs] + reals)
lines = mesh_report([0.31, 0.75, 1.50], ev)
text = "\n".join(lines)
ok("3 lines, line 1: interval count and durations",
   len(lines) == 3 and lines[0] == "mesh: 3 intervals last 0.31 s to 1.50 s (median 0.75 s).")
ok("line 2: damped frequency |Im|/2pi and its period, then the 8 decaying modes faster than the longest interval",
   lines[1] == "vertical modes: oscillating 1.91, 3.04, 5.20 Hz (periods 0.52 s, 0.33 s, 0.19 s). "
               "8 decaying, time constants 84 ms to 5 ms.")
ok("decaying: a 1.00 s time constant is kept when the longest interval is 1.50 s",
   "1 decaying, time constants 1.00 s to 1.00 s" in "\n".join(mesh_report([0.31, 1.50], np.array([-1.0, -0.5]))))
ok("line 3: no interval is at most half the slowest period (0.26 s): 0 of 3, and the line says so",
   lines[2] == "Intervals that last at most half the slowest period (0.52 s): 0 of 3. "
               "No interval resolves a vertical mode.")
fine = "\n".join(mesh_report([0.1, 0.2, 0.3], ev))
ok("short intervals: 2 of 3 resolve the slowest oscillation, and the sentence is gone",
   "2 of 3" in fine and "No interval" not in fine)
stiff = mesh_report([0.5, 1.0], np.array([-13002.1, -27028.0, -3e-2 + 21.3j, -3e-2 - 21.3j]))
ok("microsecond time constants print in us", "77 us to 37 us" in "\n".join(stiff))
ok("only decaying modes: two lines, no resolution line", len(mesh_report([0.5], np.array([-50.0, -80.0]))) == 2)
osc = mesh_report([0.5], np.array([-1.0 + 20.0j, -1.0 - 20.0j]))
ok("only an oscillation: 3 lines, no decaying part",
   len(osc) == 3 and osc[1] == "vertical modes: oscillating 3.18 Hz (periods 0.31 s).")
ok("no mode faster than the longest interval: the durations line only",
   len(mesh_report([0.5], np.array([-0.01]))) == 1)
ok("ASCII only", all(ord(ch) < 128 for ch in text + fine + "".join(stiff)))

# =============================================================================================
print("2. the speed term of the curvature mesh (synthetic lap: braking 400-560 m, corner 600-700 m)")
s = np.linspace(0.0, 1000.0, 501)
k = np.where((s > 600) & (s < 700), 0.01, 0.0)
v = np.interp(s, [0, 400, 560, 700, 800, 1000], [70, 70, 30, 30, 70, 70])
prof = (s, v)
m0 = curvature_mesh(s, k, 30)
m1 = curvature_mesh(s, k, 30, speed=1.0, v=prof)
ok("speed=0 with a profile given: the same knots bit for bit",
   np.array_equal(curvature_mesh(s, k, 30, speed=0.0, v=prof), m0))
ok("speed > 0 without a profile raises ValueError",
   isinstance(error_of(curvature_mesh, s, k, 30, speed=1.0), ValueError))
d1 = np.diff(m1)
ok(f"speed=1: same N ({m1.size - 1}), strictly increasing, ends fixed, spacing inside [7.5, 75] m",
   m1.size == m0.size and bool(np.all(d1 > 0)) and m1[0] == 0.0 and m1[-1] == 1000.0
   and d1.min() >= 7.5 - 1e-9 and d1.max() <= 75.0 + 1e-9)
ok(f"more knots in the braking zone: {n_braking(m1)} against {n_braking(m0)}", n_braking(m1) > n_braking(m0))
_, M0, i0 = monitor(s, k, 30)
sg, M1, i1 = monitor(s, k, 30, speed=1.0, v=prof)
at50, at480 = np.searchsorted(sg, 50.0), np.searchsorted(sg, 480.0)
ok("the monitor gains the term where the speed changes: equal on the 50 m straight, larger at 480 m",
   abs(M1[at50] - M0[at50]) < 1e-9 and M1[at480] > M0[at480] and i0["dv_ref"] == 0.0 and i1["dv_ref"] > 0.0)
flat = (s, np.full(s.size, 70.0))
ok("a constant speed drops the term: dv_ref 0, and the knots of speed=0",
   monitor(s, k, 30, speed=1.0, v=flat)[2]["dv_ref"] == 0.0
   and np.array_equal(curvature_mesh(s, k, 30, speed=1.0, v=flat), m0))
ok("discretise passes track.v_qss to the speed term (the knots of curvature_mesh)",
   np.array_equal(discretise(SimpleNamespace(s=s, k=k, v_qss=prof), 30, 3, mesh="curvature",
                             mesh_opts={"speed": 1.0})["s_knot"], m1))
ok("discretise without v_qss and with a speed weight raises ValueError",
   isinstance(error_of(discretise, SimpleNamespace(s=s, k=k), 30, 3, mesh="curvature",
                       mesh_opts={"speed": 1.0}), ValueError))

# =============================================================================================
if os.path.exists(DATA_AA):
    from userOpts import _MESH_OPTS_KEYS, userOpts

    print("3. userOpts(mesh_opts={'speed': w}) puts the QSS speed on the track")
    ok("'speed' is a mesh_opts key and the profile 'v' is not",
       "speed" in _MESH_OPTS_KEYS and "v" not in _MESH_OPTS_KEYS)
    err = error_of(userOpts, Ctx(), circuit="Sturn", mesh="curvature", mesh_opts={"v": 1.0})
    ok("mesh_opts={'v': ...} raises ValueError that names the key",
       isinstance(err, ValueError) and "['v']" in str(err))
    c0, c1 = Ctx(), Ctx()
    quiet(userOpts, c0, circuit="Sturn", mesh="curvature")
    quiet(userOpts, c1, circuit="Sturn", mesh="curvature", mesh_opts={"speed": 1.0})
    vq = getattr(c1.track, "v_qss", None)
    ok("default: no v_qss. speed=1 gives (s, v) with the QSS speeds in m/s",
       not hasattr(c0.track, "v_qss") and vq is not None and len(vq[0]) == len(vq[1])
       and 10.0 < float(np.min(vq[1])) < float(np.max(vq[1])) < 120.0)
    g0 = discretise(c0.track, c0.OPT_ds, c0.OPT_d, mesh=c0.mesh, mesh_opts=c0.mesh_opts)
    g1 = discretise(c1.track, c1.OPT_ds, c1.OPT_d, mesh=c1.mesh, mesh_opts=c1.mesh_opts)
    ok(f"Sturn N = {g1['N']} on both meshes, the speed weight moves the knots, the record is {{speed: 1.0}}",
       g0["N"] == g1["N"] and not np.array_equal(g0["s_knot"], g1["s_knot"])
       and mesh_opts_record(c1.mesh_opts) == {"speed": 1.0})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings(record=True) as wa:
        warnings.simplefilter("always")
        c2 = userOpts(Ctx(), circuit="Sturn", mesh_opts={"speed": 1.0})
    ok("mesh='auto' on short Sturn resolves to uniform: the weight is ignored with the old warning, no v_qss",
       c2.mesh == "uniform" and not hasattr(c2.track, "v_qss")
       and any("mesh_opts ignored" in str(x.message) for x in wa))
    import setup_sweep as SS
    with tempfile.TemporaryDirectory() as tmp:
        err = error_of(SS.setup_sweep, [("alpha_RW", 4, 16)], 4, circuit="Sturn", results_root=tmp, confirm=False,
                       plot=False, verbose=False, workers=0, mesh="curvature", mesh_opts={"speed": 1.0})
        ok("setup_sweep refuses the speed term (each setup moves the knots) with SweepError and writes no file",
           isinstance(err, SS.SweepError) and "'speed'" in str(err) and os.listdir(tmp) == [])
else:
    print("SKIP sections 3-6: Data/DATA_AA.mat is missing")

# =============================================================================================
if MODEL:
    import casadi as ca

    import MLTP as mltp_mod
    import functions.transcription as T
    from functions.casadi_opts import fn_opts
    from vehModel import vehModel, vertical_eigs

    def model(physics):
        """The Sturn model at one physics setting, default options."""
        c = Ctx()
        quiet(userOpts, c, circuit="Sturn", physics=physics)
        quiet(vehModel, c)
        return c

    def static_x(c, vx):
        """The static coast point at vx: rolling wheels, suspension at rest, static tyre deflection."""
        vp, X = c.vp, np.zeros(c.m23.nx)
        X[0], X[5:9] = vx, vx / vp.Rw
        X[19:23] = [vp.xti_fl, vp.xti_fr, vp.xti_rl, vp.xti_rr]
        return X

    def ad_block_eigs(c, vx):
        """Eigenvalues of the 14x14 vertical block of the physical-time Jacobian, by AD."""
        m = c.m23
        xs, us = ca.DM(np.asarray(m.x_s, float)), ca.DM(np.asarray(m.u_s, float))
        F = ca.Function("F", [m.x, m.u, m.pv], [m.dx, m.sf])
        Xp, Up, K = ca.SX.sym("Xp", m.nx), ca.SX.sym("Up", m.nu), ca.SX.sym("K")
        o = F(Xp / xs, Up / us, K)
        A = ca.Function("A", [Xp, Up, K], [ca.jacobian(o[0] * xs / o[1], Xp)])
        u0 = np.zeros(m.nu)
        u0[2:6] = 0.25  # ATD 1/4 each: inputs 2 to 5 (one motor, no active aero)
        return np.linalg.eigvals(np.array(A.call([static_x(c, vx), u0, 0.0])[0], float)[9:23, 9:23])

    print("4. vertical_eigs: central differences of f_dyn equal the AD block eigenvalues")
    for physics, hz, fast in (("corrected", (1.90, 3.03, 5.21), 191.1), ("matlab", (3.38, 3.90, 4.55), 27028.0)):
        c = model(physics)
        m = c.m23
        f_dyn = ca.Function("f_dyn", [m.x, m.u, m.pv], [m.dx, m.sf], ["x", "u", "pv"], ["dx", "L"], fn_opts(c))
        ev_fd = vertical_eigs(c, m, f_dyn, 20.0)
        ev_ad = ad_block_eigs(c, 20.0)
        rel = np.max(np.abs(np.sort_complex(ev_fd) - np.sort_complex(ev_ad))) / np.max(np.abs(ev_ad))
        got = np.sort(ev_fd.imag[ev_fd.imag > 1e-9] / (2 * np.pi))
        ok(f"{physics}: max relative difference to AD {rel:.1e} (limit 1e-6), frequencies "
           f"{np.round(got, 2).tolist()} Hz, fastest {np.max(np.abs(ev_fd)):.0f} 1/s",
           rel <= 1e-6 and np.allclose(got, hz, atol=0.01) and abs(np.max(np.abs(ev_fd)) - fast) < 5.0)

    @contextlib.contextmanager
    def fake_ipopt():
        """Replaces the IPOPT solver: x = x0, zero duals, no iterations."""
        def fake(ca_, nlp, opts, info=None):
            if info is not None:
                info["linear_solver"] = "fake"

            def solver(**kw):
                return {"x": kw["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                        "lam_x": np.zeros(nlp["x"].numel())}

            solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
            return solver

        real, T._make_solver = T._make_solver, fake
        try:
            yield
        finally:
            T._make_solver = real

    print("5. MLTP prints the report after the solve, and a failing report keeps the solve")
    real_ve = mltp_mod.vertical_eigs
    seen_vx = []

    def spy(c_, m_, f_, vx, *a, **kw):
        seen_vx.append(vx)
        return real_ve(c_, m_, f_, vx, *a, **kw)

    mltp_mod.vertical_eigs = spy
    try:
        with fake_ipopt():
            ctx, out = loud(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False)
    finally:
        mltp_mod.vertical_eigs = real_ve
    rep = [ln for ln in out.splitlines()
           if ln.startswith(("[MLTP] mesh: ", "[MLTP] vertical modes: ",
                             "[MLTP] Intervals that last at most half the slowest period ("))]
    ok("3 report lines after the summary, with the durations of the stored Sturn mesh",
       len(rep) == 3 and out.index(rep[0]) > out.index("lap time =")
       and rep[0].startswith("[MLTP] mesh: 18 intervals last 0.48 s to 1.67 s"))
    ok("the modes are taken at the start speed vi, in one call", seen_vx == [ctx.vi])
    ok("default Sturn (matlab physics): 3.38, 3.90 and 4.55 Hz, 4 decaying modes of 77 us to 37 us, "
       "no interval resolves a mode",
       rep[1:] == ["[MLTP] vertical modes: oscillating 3.38, 3.90, 4.55 Hz (periods 0.30 s, 0.26 s, 0.22 s). "
                   "4 decaying, time constants 77 us to 37 us.",
                   "[MLTP] Intervals that last at most half the slowest period (0.30 s): 0 of 18. "
                   "No interval resolves a vertical mode."])

    def boom(*a, **kw):
        raise RuntimeError("boom")

    real_ve = mltp_mod.vertical_eigs
    mltp_mod.vertical_eigs = boom
    try:
        with fake_ipopt():
            ctx2, out2 = loud(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False)
    finally:
        mltp_mod.vertical_eigs = real_ve
    ok("a failing report prints one skip line, and the solve result is the same",
       out2.count("[MLTP] mesh report skipped: boom") == 1 and ctx2.data.lap_time == ctx.data.lap_time)

    print("6. MLTP(mesh='curvature', mesh_opts={'speed': 1.0}) on Sturn")
    with fake_ipopt():
        cs, outs = loud(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False,
                        mesh="curvature", mesh_opts={"speed": 1.0})
    with fake_ipopt():
        cb, _ = loud(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False, mesh="curvature")
    ok("same N as the speed=0 curvature mesh, other knots, data.mesh_opts == {speed: 1.0} (empty by default), "
       "report printed",
       cs.data.N == cb.data.N and not np.array_equal(cs.data.s_full, cb.data.s_full)
       and cs.data.mesh_opts == {"speed": 1.0} and cb.data.mesh_opts == {}
       and "[MLTP] mesh: 18 intervals last " in outs)
elif os.path.exists(DATA_AA):
    print("SKIP sections 4-6: the stored Sturn result is missing")

print("\nALL mesh report TESTS PASSED" if MODEL else "\nnumpy mesh report tests PASSED (model sections skipped)")
