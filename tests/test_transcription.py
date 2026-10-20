"""Validate the casadi-free numerics of transcription.py: discretisation grid and
the column-major packing/unpacking/reconstruction (the reshape-order danger zone),
plus the curvature-weighted mesh (functions/mesh.py) and the arc-length
interpolation behind MLTP.warmstart_guesses."""
import sys, os
import numpy as np
from types import SimpleNamespace
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)

from functions.transcription import (discretise, unpack_solution,
                                      reconstruct_x_full, reconstruct_track,
                                      interp_inputs)
from functions.mesh import curvature_mesh, mesh_stats, monitor, solution_knots

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

print("discretise")
# simple track: s from 0..100 (so N = 100/10 = 10), curvature ramp
track = SimpleNamespace(s=np.linspace(0, 100, 201), k=np.linspace(0, 0.02, 201))
OPT_ds, OPT_d = 10, 3
disc = discretise(track, OPT_ds, OPT_d)
N = disc["N"]
ok("N = round(100/10) = 10", N == 10)
ok("s_knot length N+1", disc["s_knot"].size == N + 1)
ok("s_full length N*(d+1)+1", disc["s_full"].size == N * (OPT_d + 1) + 1)
ok("s_full strictly increasing", np.all(np.diff(disc["s_full"]) > 0))
ok("s_full starts/ends at track ends",
   abs(disc["s_full"][0] - 0) < 1e-9 and abs(disc["s_full"][-1] - 100) < 1e-9)
ok("s_full contains all knots",
   np.allclose(disc["s_full"][::OPT_d + 1][:N], disc["s_knot"][:-1], atol=1e-9))
ok("k_full matches interp on s_full",
   np.allclose(disc["k_full"], np.interp(disc["s_full"], track.s, track.k)))
ok("C shape (d+1,d), D (d+1,1), B (d,1)",
   disc["C"].shape == (OPT_d + 1, OPT_d) and disc["D"].shape == (OPT_d + 1, 1)
   and disc["B"].shape == (OPT_d, 1))

print("pack -> unpack round trip (column-major)")
nx, nu, ny = 7, 3, 1
x_s = np.arange(1.0, nx + 1)
u_s = np.array([2.0, 3.0, 4.0])
y_s = np.array([5.0])
rng = np.arange(1.0, nx * (N + 1) + 1).reshape(nx, N + 1)         # deterministic x_opt
x_opt = rng
u_opt = np.arange(1.0, nu * (N + 1) + 1).reshape(nu, N + 1) * 0.1
y_opt = np.arange(1.0, ny * (N + 1) + 1).reshape(ny, N + 1) * 0.01
xc_opt = np.arange(1.0, nx * N * OPT_d + 1).reshape(nx, N * OPT_d) * 0.5

# Build w exactly as MATLAB: w = [Xk(:); Uk(:); Yk(:); Xkj(:)], scaled decision vars
w = np.concatenate([
    (x_opt / x_s[:, None]).flatten(order="F"),
    (u_opt / u_s[:, None]).flatten(order="F"),
    (y_opt / y_s[:, None]).flatten(order="F"),
    (xc_opt / x_s[:, None]).flatten(order="F"),
])
xo, uo, yo, xco = unpack_solution(w, nx, nu, ny, N, OPT_d, x_s, u_s, y_s)
ok("x_opt recovered", np.allclose(xo, x_opt))
ok("u_opt recovered", np.allclose(uo, u_opt))
ok("y_opt recovered", np.allclose(yo, y_opt))
ok("xc_opt recovered", np.allclose(xco, xc_opt))

print("reconstruct_x_full structure")
x_full = reconstruct_x_full(x_opt, xc_opt, nx, N, OPT_d)
ok("x_full length N*(d+1)+1", x_full.shape == (nx, N * (OPT_d + 1) + 1))
ok("knot columns equal x_opt",
   np.allclose(x_full[:, ::OPT_d + 1][:, :N], x_opt[:, :-1]))
ok("last column = final knot", np.allclose(x_full[:, -1], x_opt[:, -1]))
# collocation columns of interval i equal xc_opt[:, i*d : i*d+d]
good = all(np.allclose(x_full[:, i * (OPT_d + 1) + 1: i * (OPT_d + 1) + 1 + OPT_d],
                       xc_opt[:, i * OPT_d: i * OPT_d + OPT_d]) for i in range(N))
ok("collocation columns equal xc_opt", good)

print("interp_inputs")
u_full = interp_inputs(u_opt, disc["s_knot"], disc["s_full"], "linear")
ok("u_full shape", u_full.shape == (nu, disc["s_full"].size))
ok("u_full equals u_opt at knots",
   np.allclose(u_full[:, ::OPT_d + 1][:, :N + 1][:, :N], u_opt[:, :N]))

print("reconstruct_track (straight, lateral offset)")
# straight centreline, offset n=+1.5 -> racing line shifted in +y, width band +/-2
s_full = np.linspace(0, 100, 101)
k_full = np.zeros_like(s_full)
n_full = 1.5 * np.ones_like(s_full)
tr0 = SimpleNamespace(s=s_full)         # no x,y -> curv2cart used
rec = reconstruct_track(tr0, s_full, k_full, n_full, 4.0)
ok("racing line offset ~ +1.5 in y", abs(np.mean(rec["yopt"]) - 1.5) < 1e-6)
ok("left/right limits span ~ 4 m apart",
   abs(np.mean(rec["Xl"][:, 1] - rec["Xr"][:, 1]) - 4.0) < 1e-6)


def legacy_uniform(trk, ds, d, tau):
    """The pre-mesh discretise() grid, verbatim (N from s[-1], start from 0)."""
    s_ = np.asarray(trk.s, dtype=float).reshape(-1)
    k_ = np.asarray(trk.k, dtype=float).reshape(-1)
    n_ = int(round(s_[-1] / ds))
    sk_ = np.linspace(s_.min(), s_.max(), n_ + 1)
    dsk_ = np.diff(sk_)
    start_ = np.concatenate(([0.0], np.cumsum(dsk_[:-1])))
    sc_ = np.kron(dsk_, tau) + np.kron(start_, np.ones(d))
    blk = np.vstack([np.zeros((1, n_)), sc_.reshape(d, n_, order="F")])
    sf_ = np.kron(sk_[:-1], np.concatenate(([1.0], np.zeros(d)))) + blk.reshape(-1, order="F")
    sf_ = np.append(sf_, sk_[-1])
    return n_, sk_, sc_, sf_, np.interp(sc_, s_, k_), np.interp(sf_, s_, k_)


# synthetic hairpin-like track: 1200 m (OPT_ds=30 -> N=40), a 60 m hairpin
# (R = 12.5 m) and a 100 m medium corner (R = 33 m), straights elsewhere
s_h = np.linspace(0.0, 1200.0, 601)
k_h = np.zeros_like(s_h)
k_h[(s_h >= 300) & (s_h <= 360)] = 0.08
k_h[(s_h >= 800) & (s_h <= 900)] = -0.03
trk_h = SimpleNamespace(s=s_h, k=k_h)

print("discretise: uniform grid bit-identical to the legacy formula")
ss_track = SimpleNamespace(s=np.linspace(0, 540, 270),
                           k=np.sin(np.linspace(0, 6, 270)) * 0.05)
for trk_, ds_, d_ in [(track, 10, 3), (trk_h, 30, 3), (ss_track, 30, 3),
                      (ss_track, 45, 3), (ss_track, 15, 2)]:
    du = discretise(trk_, ds_, d_)
    n_, sk_, sc_, sf_, kc_, kf_ = legacy_uniform(trk_, ds_, d_, du["tau"])
    ok(f"uniform N={n_} d={d_}: N, s_knot, s_col, s_full, k_col, k_full identical",
       du["N"] == n_ and du["mesh"] == "uniform"
       and np.array_equal(du["s_knot"], sk_) and np.array_equal(du["s_col"], sc_)
       and np.array_equal(du["s_full"], sf_) and np.array_equal(du["k_col"], kc_)
       and np.array_equal(du["k_full"], kf_))

print("curvature_mesh (synthetic hairpin track)")
OPT_ds_h = 30
sk = curvature_mesh(s_h, k_h, OPT_ds_h)
dsk_h = np.diff(sk)
ok("N = round(L/OPT_ds) (same as the uniform mesh)", sk.size - 1 == 40)
ok("endpoints fixed at the track ends", sk[0] == s_h[0] and sk[-1] == s_h[-1])
ok("knots strictly increasing", np.all(dsk_h > 0))
ok("mean spacing == OPT_ds", abs(dsk_h.mean() - OPT_ds_h) < 1e-9)
mid = 0.5 * (sk[:-1] + sk[1:])
corner = np.abs(np.interp(mid, s_h, k_h)) > 0
ok("denser in the corners (mean corner dsk < 0.5 x mean straight dsk)",
   corner.sum() >= 8 and dsk_h[corner].mean() < 0.5 * dsk_h[~corner].mean())
hair = (mid >= 300) & (mid <= 360)
med = (mid >= 800) & (mid <= 900)
ok("tighter corner gets the finer spacing", dsk_h[hair].mean() < dsk_h[med].mean())
ok("default bounds 0.25*OPT_ds <= dsk <= 2.5*OPT_ds",
   dsk_h.min() >= 0.25 * OPT_ds_h - 1e-9 and dsk_h.max() <= 2.5 * OPT_ds_h + 1e-9)
for kw in [dict(ds_min=5.0, ds_max=33.0), dict(ds_min=20.0, ds_max=40.0),
           dict(a=6.0, b=6.0, ds_min=15.0, ds_max=36.0)]:
    d_b = np.diff(curvature_mesh(s_h, k_h, OPT_ds_h, **kw))
    active = (abs(d_b.min() - kw["ds_min"]) < 1e-6) or (abs(d_b.max() - kw["ds_max"]) < 1e-6)
    ok(f"bounds {kw['ds_min']:g}..{kw['ds_max']:g} m respected (and active)",
       d_b.min() >= kw["ds_min"] - 1e-9 and d_b.max() <= kw["ds_max"] + 1e-9 and active
       and abs(d_b.mean() - OPT_ds_h) < 1e-9)
ok("zero curvature -> uniform mesh",
   np.allclose(curvature_mesh(s_h, np.zeros_like(s_h), OPT_ds_h),
               np.linspace(0, 1200, 41), rtol=0, atol=1e-9))
# float rounding noise must not be rescaled to O(1) by the monitor normalisation
s_c = np.linspace(0.0, 1000.0, 501)
for kc in (0.02, 0.1, 1.0 / 3.0, 0.5):
    ok(f"constant curvature k={kc:.3g} (constant-radius circle) -> uniform mesh",
       np.allclose(curvature_mesh(s_c, kc * np.ones_like(s_c), OPT_ds_h),
                   np.linspace(0, 1000, 34), rtol=0, atol=1e-9))
rng_n = np.random.default_rng(0)
ok("numerically straight track (|k| ~ 1e-14 noise) -> uniform mesh",
   np.allclose(curvature_mesh(s_h, 1e-14 * rng_n.standard_normal(s_h.size), OPT_ds_h),
               np.linspace(0, 1200, 41), rtol=0, atol=1e-9))
k_one = np.zeros_like(s_h)                        # one 60 m hairpin: under 10% of the lap
k_one[(s_h >= 300) & (s_h <= 360)] = 0.08
sk_one = curvature_mesh(s_h, k_one, OPT_ds_h)
ok("exact-zero straights + one hairpin: corner denser, mesh bounded",
   np.diff(sk_one).min() < 0.5 * OPT_ds_h and np.diff(sk_one).max() <= 2.5 * OPT_ds_h + 1e-9)
for amp in (1e-17, 1e-13):
    ok(f"{amp:g} noise on the straights leaves the one-corner mesh unchanged",
       np.allclose(curvature_mesh(s_h, k_one + amp * rng_n.standard_normal(s_h.size), OPT_ds_h),
                   sk_one, rtol=0, atol=1e-6))
# noise ABOVE the absolute floors (1e-6 / L ~ 8e-10 1/m on this lap) but far below the
# corner sets the 90th percentile; the relative guard (_NOISE_REL) then uses the max, as
# for exact zeros, instead of rescaling the noise to O(1) and pinning the corner at ds_min
_, _, inf_one = monitor(s_h, k_one, OPT_ds_h)
for amp in (1e-9, 1e-6):
    k_nz = k_one + amp * rng_n.standard_normal(s_h.size)
    sk_nz = curvature_mesh(s_h, k_nz, OPT_ds_h)
    _, _, inf_nz = monitor(s_h, k_nz, OPT_ds_h)
    ok(f"{amp:g} noise (above the floors) on the straights: one-corner mesh within 1 cm of the "
       "exact-zero one, corner not pinned at ds_min, k_ref / dk_ref set by the corner",
       np.max(np.abs(sk_nz - sk_one)) < 1e-2
       and np.diff(sk_nz).min() > 0.25 * OPT_ds_h + 1.0
       and abs(inf_nz["k_ref"] / inf_one["k_ref"] - 1.0) < 1e-3
       and abs(inf_nz["dk_ref"] / inf_one["dk_ref"] - 1.0) < 1e-3)
_, _, inf_lo = monitor(s_h, k_one + 1e-6, OPT_ds_h)      # background ~1e-5 of the peak
_, _, inf_hi = monitor(s_h, k_one + 2e-3, OPT_ds_h)      # background ~2e-2 of the peak: data
_, _, inf_h = monitor(s_h, k_h, OPT_ds_h)                # corners over > 10% of the lap
ok("percentile guard: a background under 1e-3 of the peak gives k_ref = max, one above it "
   "keeps its 90th percentile, and so do corners covering > 10% of the lap",
   inf_lo["k_ref"] == inf_lo["K"].max()
   and inf_hi["k_ref"] == np.percentile(inf_hi["K"], 90) and abs(inf_hi["k_ref"] - 2e-3) < 1e-12
   and inf_h["k_ref"] == np.percentile(inf_h["K"], 90)
   and inf_h["dk_ref"] == np.percentile(inf_h["DK"], 90))
sk20 = curvature_mesh(s_h, k_h, OPT_ds_h, N=20)
ok("explicit N honoured (mean spacing L/N)",
   sk20.size == 21 and abs(np.diff(sk20).mean() - 60.0) < 1e-9)
try:
    curvature_mesh(s_h, k_h, OPT_ds_h, ds_max=20.0)
    raised = False
except ValueError:
    raised = True
ok("infeasible bounds (ds_max < L/N) raise ValueError", raised)
st = mesh_stats(sk)
ok("mesh_stats fields", st["N"] == 40 and abs(st["ds_mean"] - 30) < 1e-9
   and abs(st["ds_min"] - dsk_h.min()) < 1e-12 and st["max_adjacent_ratio"] >= 1.0)

print("discretise(mesh='curvature')")
dc = discretise(trk_h, OPT_ds_h, 3, mesh="curvature")
Nc = dc["N"]
ok("mesh label + N = len(s_knot)-1", dc["mesh"] == "curvature" and Nc == dc["s_knot"].size - 1 == 40)
ok("s_knot is curvature_mesh's", np.array_equal(dc["s_knot"], sk))
ok("dsk = diff(s_knot)", np.array_equal(dc["dsk"], np.diff(dc["s_knot"])))
ok("s_full length N*(d+1)+1", dc["s_full"].size == Nc * 4 + 1)
ok("s_full strictly increasing, knots at every (d+1)-th entry",
   np.all(np.diff(dc["s_full"]) > 0) and np.array_equal(dc["s_full"][::4], dc["s_knot"]))
sc_mat = dc["s_col"].reshape(3, Nc, order="F")
ok("s_col strictly inside each interval",
   np.all(sc_mat > dc["s_knot"][:-1]) and np.all(sc_mat < dc["s_knot"][1:]))
ok("s_col at the Legendre points of each interval",
   np.allclose(sc_mat, dc["s_knot"][:-1] + np.outer(dc["tau"], dc["dsk"]), rtol=0, atol=1e-9))
ok("k_knot / k_col / k_full = interp of k",
   np.array_equal(dc["k_full"], np.interp(dc["s_full"], s_h, k_h))
   and np.array_equal(dc["k_col"], np.interp(dc["s_col"], s_h, k_h))
   and np.array_equal(dc["k_knot"], np.interp(dc["s_knot"], s_h, k_h)))
ok("pv shapes", dc["pv_knot"].shape == (1, Nc + 1) and dc["pv_col"].shape == (1, Nc * 3)
   and dc["pv_full"].shape == (1, Nc * 4 + 1))
ok("mesh_opts forwarded", discretise(trk_h, 30, 3, mesh="curvature", mesh_opts={"N": 30})["N"] == 30)
custom = np.array([0.0, 100.0, 150.0, 600.0, 1200.0])
dx = discretise(trk_h, 30, 3, s_knot=custom)
ok("explicit s_knot override -> mesh 'custom'",
   dx["mesh"] == "custom" and dx["N"] == 4 and np.array_equal(dx["s_knot"], custom)
   and dx["s_full"].size == 4 * 4 + 1)
trk_off = SimpleNamespace(s=s_h + 500.0, k=k_h)          # track not starting at s = 0
for mesh_ in ("uniform", "curvature"):
    do = discretise(trk_off, 30, 3, mesh=mesh_)
    sco = do["s_col"].reshape(3, do["N"], order="F")
    ok(f"offset track ({mesh_}): N=40, s_full spans 500..1700, s_col inside intervals",
       do["N"] == 40 and do["s_full"][0] == 500.0 and do["s_full"][-1] == 1700.0
       and np.all(sco > do["s_knot"][:-1]) and np.all(sco < do["s_knot"][1:]))
try:
    discretise(trk_h, 30, 3, mesh="adaptive")
    raised = False
except ValueError:
    raised = True
ok("unknown mesh name raises ValueError", raised)

print("arc-length interpolation of init signals: uniform-grid equivalence with the legacy index version")


def legacy_interp(N_new, row):
    """MLTP._interp_to as it was: both grids taken as uniform on [0, 1]."""
    return np.interp(np.linspace(0.0, 1.0, N_new + 1), np.linspace(0.0, 1.0, row.size), row)


rng = np.random.default_rng(0)
for ds_src, ds_dst in [(30, 30), (10, 10), (30, 15), (15, 40)]:
    d_src = discretise(trk_h, ds_src, 3)
    d_dst = discretise(trk_h, ds_dst, 3)
    init = SimpleNamespace(s_full=d_src["s_full"], OPT_d=3)
    src = solution_knots(init, d_src["N"] + 1)
    rows = rng.standard_normal((3, d_src["N"] + 1))
    new = np.vstack([np.interp(d_dst["s_knot"], src, r) for r in rows])
    old = np.vstack([legacy_interp(d_dst["N"], r) for r in rows])
    if ds_src == ds_dst:
        ok(f"uniform N={d_src['N']} -> same grid: bit-identical to the index version",
           np.array_equal(new, old) and np.array_equal(new, rows))
    else:
        ok(f"uniform N={d_src['N']} -> N={d_dst['N']}: equal to the index version (1e-12)",
           np.allclose(new, old, rtol=0, atol=1e-12))
# a curvature-mesh init onto the same curvature mesh is reproduced exactly; the
# index version would misplace it (knot i is not at the same s on both meshes)
init_c = SimpleNamespace(s_full=dc["s_full"], OPT_d=3)
row_c = np.interp(dc["s_knot"], s_h, np.cumsum(np.abs(k_h)))
src_c = solution_knots(init_c, Nc + 1)
ok("curvature init -> same curvature grid: exact",
   np.array_equal(np.interp(dc["s_knot"], src_c, row_c), row_c))
d_u = discretise(trk_h, 30, 3)
lin_c = 2.0 * src_c + 1.0                    # a signal linear in s is reproduced exactly
ok("curvature init -> uniform grid: right values at the physical s (index version is not)",
   np.allclose(np.interp(d_u["s_knot"], src_c, lin_c), 2.0 * d_u["s_knot"] + 1.0,
               rtol=0, atol=1e-9)
   and not np.allclose(legacy_interp(d_u["N"], lin_c), 2.0 * d_u["s_knot"] + 1.0, atol=1.0))
ok("solution_knots: dict input, missing fields / wrong length -> None",
   np.array_equal(solution_knots({"s_full": dc["s_full"], "OPT_d": 3}), dc["s_knot"])
   and solution_knots(SimpleNamespace(OPT_d=3)) is None
   and solution_knots(init_c, Nc) is None)

print("\nALL TRANSCRIPTION NUMERIC TESTS PASSED")
