"""Validate a saved optimal-solution .mat round-trips (save -> reload): the racing
line and channels recover without re-solving. scipy only (no casadi/plotly)."""
import sys, os, tempfile
import numpy as np
import scipy.io as sio
sys.path.insert(0, os.path.dirname(__file__))

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

print("\nSAVE/LOAD ROUND-TRIP TESTS PASSED")
