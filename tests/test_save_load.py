"""Validate that a saved optimal-solution .mat round-trips: save -> reload ->
the racing line and channels are all recoverable without re-solving.
This is the user's core requirement. Uses scipy only (no casadi/plotly)."""
import sys, os, tempfile
import numpy as np
import scipy.io as sio
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)

from functions.importfile import load_solution

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# ---- build a fake solution exactly like MLTP.py saves --------------------
N, d = 10, 3
nfull = N * (d + 1) + 1
s_full = np.linspace(0, 100, nfull)
x_full = np.random.RandomState(0).rand(23, nfull) * 50
xopt = np.cos(np.linspace(0, 2 * np.pi, nfull)) * 50
yopt = np.sin(np.linspace(0, 2 * np.pi, nfull)) * 50
track = {"s": s_full, "k": np.zeros(nfull), "x": xopt * 1.01, "y": yopt * 1.01,
         "xopt": xopt, "yopt": yopt,
         "Xl": np.column_stack([xopt, yopt + 2]), "Xr": np.column_stack([xopt, yopt - 2])}
vehicle = {"fx_fl": np.linspace(0, 500, N + 1), "fy_fl": np.linspace(0, 800, N + 1),
           "fz_fl": np.linspace(3000, 5000, N + 1), "P_motor": np.linspace(0, 4e5, N + 1),
           "E_motor": np.linspace(0, 1.2, N + 1), "zs": np.zeros(N + 1)}
constraints = {"rho_lim_fl": np.linspace(0, 0.95, N + 1)}
data = {"s_full": s_full, "x_full": x_full, "x_opt": np.random.rand(23, N + 1),
        "u_opt": np.random.rand(7, N + 1), "t_opt": np.linspace(0, 5.5, N + 1),
        "lap_time": 5.5, "track": track, "vehicle": vehicle, "constraints": constraints,
        "input_keys": ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"],
        "N": N, "OPT_d": d, "circuit": "Sturn",
        "AeroConfig": "Static", "ATD": "On", "EM4": "Off"}

tmp = tempfile.mkdtemp()
full_path = os.path.join(tmp, "Sturn_Static.mat")
sio.savemat(full_path, {"data": data}, do_compression=True)

print("full-solve .mat round-trip")
d2 = load_solution(full_path)
ok("lap_time recovered", abs(float(d2.lap_time) - 5.5) < 1e-9)
ok("racing line xopt recovered", np.allclose(np.asarray(d2.track.xopt).reshape(-1), xopt))
ok("racing line yopt recovered", np.allclose(np.asarray(d2.track.yopt).reshape(-1), yopt))
ok("track boundaries present",
   np.asarray(d2.track.Xl).shape == (nfull, 2) and np.asarray(d2.track.Xr).shape == (nfull, 2))
ok("x_full shape 23 x nfull", np.asarray(d2.x_full).shape == (23, nfull))
ok("velocity (x_full[0]) recoverable for colouring",
   np.allclose(np.asarray(d2.x_full)[0, :], x_full[0, :]))
ok("vehicle channel fx_fl recovered",
   np.allclose(np.asarray(d2.vehicle.fx_fl).reshape(-1), vehicle["fx_fl"]))
ok("energy channel recovered",
   np.allclose(np.asarray(d2.vehicle.E_motor).reshape(-1), vehicle["E_motor"]))
ok("constraint rho_lim_fl recovered",
   np.allclose(np.asarray(d2.constraints.rho_lim_fl).reshape(-1), constraints["rho_lim_fl"]))

print("warm-start (data.init) .mat round-trip")
init_path = os.path.join(tmp, "init_Sturn.mat")
sio.savemat(init_path, {"data": {"init": data}}, do_compression=True)
d3 = load_solution(init_path)
ok("init lap_time recovered", abs(float(d3.lap_time) - 5.5) < 1e-9)
ok("init racing line recovered", np.allclose(np.asarray(d3.track.xopt).reshape(-1), xopt))

print("primal/dual NLP record (data.nlp) round-trip")
rs = np.random.RandomState(1)
n_w, n_g = 23 * (N + 1) + 7 * (N + 1) + 23 * N * d, 2 * 23 + (d + 1) * N * 23 + 8 * (N + 1) + 7 * N
structure = {"nx": 23, "nu": 7, "ny": 0, "N": N, "OPT_d": d, "n_w": n_w, "n_g": n_g, "n_param": 0}
nlp = {"w_opt": rs.rand(n_w), "lam_g": rs.randn(n_g), "lam_x": rs.randn(n_w),
       "structure": structure, "sym_type": "SX", "linear_solver": "ma57",
       "ipopt_iters": 3209, "return_status": "Solve_Succeeded", "warm_start": "cold",
       "x_s": rs.rand(23) + 1.0, "u_s": rs.rand(7) + 1.0}
nlp_path = os.path.join(tmp, "Sturn_Static_nlp.mat")
sio.savemat(nlp_path, {"data": dict(data, nlp=nlp)}, do_compression=True)
d4 = load_solution(nlp_path)
ok("data.nlp present", hasattr(d4, "nlp"))
for key in ("w_opt", "lam_g", "lam_x", "x_s", "u_s"):
    v = getattr(d4.nlp, key)
    ok(f"nlp.{key} is a 1-D float array, values exact",
       isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype == float
       and np.array_equal(v, nlp[key]))
ok("nlp.structure ints recovered",
   all(int(getattr(d4.nlp.structure, k)) == v for k, v in structure.items()))
ok("nlp strings recovered",
   d4.nlp.sym_type == "SX" and d4.nlp.linear_solver == "ma57"
   and d4.nlp.return_status == "Solve_Succeeded" and d4.nlp.warm_start == "cold")
ok("nlp.ipopt_iters recovered", int(d4.nlp.ipopt_iters) == 3209)
ok("the rest of data unchanged by the nlp field", abs(float(d4.lap_time) - 5.5) < 1e-9
   and np.allclose(np.asarray(d4.x_full), x_full))

# loadmat squeezes a length-1 vector to a scalar: load_solution must undo it
edge_path = os.path.join(tmp, "edge_nlp.mat")
sio.savemat(edge_path, {"data": dict(data, nlp=dict(nlp, lam_g=np.array([0.25])))})
d5 = load_solution(edge_path)
ok("length-1 lam_g comes back as a 1-element array",
   isinstance(d5.nlp.lam_g, np.ndarray) and d5.nlp.lam_g.shape == (1,) and d5.nlp.lam_g[0] == 0.25)

print("config fields (mesh, mesh_opts, tyre_set) round-trip")
from functions.mesh import mesh_opts_record
ok("mesh_opts_record: None and {} give {}", mesh_opts_record(None) == {} and mesh_opts_record({}) == {})
rec = mesh_opts_record({"a": np.float64(2.0), "b": 1, "ds_max": None, "N": np.int64(18), "flag": True})
ok("mesh_opts_record drops None values and coerces numpy scalars",
   rec == {"a": 2.0, "b": 1, "N": 18, "flag": 1} and type(rec["a"]) is float and type(rec["N"]) is int)
cfg_path = os.path.join(tmp, "cfg_uniform.mat")
sio.savemat(cfg_path, {"data": dict(data, mesh="uniform", mesh_opts=mesh_opts_record(None),
                                    tyre_set="CopyB")}, do_compression=True)
d6 = load_solution(cfg_path)
ok("mesh / tyre_set strings recovered", d6.mesh == "uniform" and d6.tyre_set == "CopyB")
ok("empty mesh_opts reloads as an empty namespace", vars(d6.mesh_opts) == {})
cfg_path = os.path.join(tmp, "cfg_curvature.mat")
sio.savemat(cfg_path, {"data": dict(data, mesh="curvature", tyre_set="MF205", mesh_opts=mesh_opts_record(
    {"a": 2.0, "ds_max": None, "N": 18}))}, do_compression=True)   # a raw None would make savemat raise
d7 = load_solution(cfg_path)
ok("curvature mesh + MF205 recovered", d7.mesh == "curvature" and d7.tyre_set == "MF205")
ok("mesh_opts values recovered, None entry dropped",
   vars(d7.mesh_opts) == {"a": 2.0, "N": 18})

print("\nSAVE/LOAD ROUND-TRIP TESTS PASSED")
