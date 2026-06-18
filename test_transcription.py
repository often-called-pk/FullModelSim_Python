"""Validate the casadi-free numerics of transcription.py: discretisation grid and
the column-major packing/unpacking/reconstruction (the reshape-order danger zone)."""
import sys, os
import numpy as np
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(__file__))

from functions.transcription import (discretise, unpack_solution,
                                      reconstruct_x_full, reconstruct_track,
                                      interp_inputs)

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

print("\nALL TRANSCRIPTION NUMERIC TESTS PASSED")
