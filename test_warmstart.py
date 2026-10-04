"""functions/warmstart.py: structure-compatibility check, s-based interpolation,
warm-start source classification and planning (plain script, no pytest, no
casadi). Run from the repo root:

    venv\\Scripts\\python.exe test_warmstart.py
"""
import os, sys, tempfile, warnings
import numpy as np
import scipy.io as sio
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import functions.warmstart as W
from functions.importfile import importfile, load_solution
from functions.transcription import discretise

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

KEYS = ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"]   # Static / ATD On / EM4 Off

# ---- 1. IPOPT warm-start recipe -------------------------------------------------
print("warm_start_ipopt_opts")
o = W.warm_start_ipopt_opts()
ok("warm_start_init_point = yes", o["warm_start_init_point"] == "yes")
ok("all five push/frac options = push (1e-10 default)",
   all(o[k] == 1e-10 for k in ("warm_start_bound_push", "warm_start_bound_frac",
                              "warm_start_slack_bound_push", "warm_start_slack_bound_frac",
                              "warm_start_mult_bound_push")))
ok("mu_init default 1e-6, mu_strategy untouched", o["mu_init"] == 1e-6 and "mu_strategy" not in o)
o2 = W.warm_start_ipopt_opts(mu_init=1e-4, push=1e-6)
ok("custom mu_init / push", o2["mu_init"] == 1e-4 and o2["warm_start_bound_push"] == 1e-6)
ok("fresh dict per call", W.warm_start_ipopt_opts() is not W.warm_start_ipopt_opts())

# ---- 2. NLP sizes (same packing as build_and_solve_nlp) -------------------------
print("nlp_structure")
st = W.nlp_structure(23, 7, 0, 18, 3, 8)
ok("Sturn N=18 d=3 sizes: n_w = 30*19 + 23*18*3 = 1812", st["n_w"] == 1812)
ok("n_g = 2*23 + 4*18*23 + 8*19 + 7*18 = 1980", st["n_g"] == 1980)
ok("n_param default 0, adds to n_w only",
   st["n_param"] == 0 and W.nlp_structure(23, 7, 0, 18, 3, 8, n_param=6)["n_w"] == 1818)
ok("7-state model with an aux variable",
   W.nlp_structure(7, 3, 1, 10, 3, 6)["n_w"] == 11 * 11 + 7 * 30)

# ---- 3. structure_mismatch -------------------------------------------------------
print("structure_mismatch")
s_full = np.linspace(0.0, 540.0, 18 * 4 + 1)
x_s, u_s = np.arange(1.0, 24.0), np.arange(1.0, 8.0)
exp = dict(st, input_keys=KEYS, s_full=s_full, x_s=x_s, u_s=u_s)
ok("identical -> compatible", W.structure_mismatch(dict(exp), exp) == [])
ok("no saved structure -> incompatible", W.structure_mismatch(None, exp) != [])
r = W.structure_mismatch(dict(exp, N=36, n_w=3000), exp)
ok("different N -> reasons name N and n_w", any(x.startswith("N ") for x in r)
   and any(x.startswith("n_w") for x in r))
ok("different n_g -> incompatible", W.structure_mismatch(dict(exp, n_g=1979), exp) != [])
sv = dict(exp); sv.pop("n_param")
ok("missing n_param in the saved record counts as 0", W.structure_mismatch(sv, exp) == [])
padded = np.array(["T_motor", "T_brake", "ATD    ", "ATD    ", "ATD    ", "ATD    ", "delta  "])
ok("loadmat-padded input_keys still match", W.structure_mismatch(dict(exp, input_keys=padded), exp) == [])
r = W.structure_mismatch(dict(exp, input_keys=["T_motor", "T_brake", "RW", "delta"]), exp)
ok("different input_keys -> incompatible", any("input_keys" in x for x in r))
r = W.structure_mismatch(dict(exp, s_full=s_full + 1e-2), exp)
ok("different collocation grid (e.g. another mesh) -> incompatible", any("s_full" in x for x in r))
ok("grid equal within tolerance -> compatible",
   W.structure_mismatch(dict(exp, s_full=s_full + 1e-9), exp) == [])
ok("different variable scaling (setup change, e.g. vp.Rw) does NOT block re-injection",
   W.structure_mismatch(dict(exp, x_s=x_s * 1.1, u_s=u_s * 2), exp) == [])
ok("scaling_changed reports it (informational)",
   W.scaling_changed(dict(exp, x_s=x_s * 1.1), exp) == ["x_s"]
   and W.scaling_changed(dict(exp, u_s=u_s * 2), exp) == ["u_s"] and W.scaling_changed(exp, exp) == [])
sv = dict(exp); sv.pop("x_s"); sv.pop("s_full")
ok("optional entries compared only when present in both",
   W.structure_mismatch(sv, exp) == [] and W.scaling_changed(sv, exp) == [])

# ---- 4. knots + row interpolation ------------------------------------------------
print("knots_from_full / interp_rows")
track = SimpleNamespace(s=np.linspace(0.0, 540.0, 271), k=0.01 * np.sin(np.linspace(0, 6, 271)))
disc = discretise(track, 30, 3)
N, d = disc["N"], 3
ok("knots_from_full(s_full, d) == s_knot", np.array_equal(W.knots_from_full(disc["s_full"], d), disc["s_knot"]))
try:
    W.knots_from_full(disc["s_full"], d, n_knots=N + 5); raised = False
except ValueError:
    raised = True
ok("knot-count check raises", raised)
M = np.vstack([2.0 + 0.5 * disc["s_full"], -disc["s_full"]])
ok("interp_rows is exact at the source nodes", np.allclose(W.interp_rows(disc["s_full"], M, disc["s_full"]), M))
ok("interp_rows is exact for a linear profile anywhere",
   np.allclose(W.interp_rows(disc["s_full"], M, [3.3, 271.7])[0], 2.0 + 0.5 * np.array([3.3, 271.7])))
ok("interp_rows clamps outside the source range",
   np.allclose(W.interp_rows([0.0, 1.0], [[5.0, 7.0]], [-1.0, 2.0]), [[5.0, 7.0]]))
ok("interp_rows accepts a 1-D row", W.interp_rows([0.0, 2.0], [0.0, 4.0], [1.0]).shape == (1, 1))
try:
    W.interp_rows([0.0, 1.0, 1.0], [[1.0, 2.0, 3.0]], [0.5]); raised = False
except ValueError:
    raised = True
ok("non-increasing source grid raises", raised)

# ---- 5. input-row mapping by channel name ----------------------------------------
print("map_input_rows")
U = np.arange(7.0).reshape(-1, 1) * np.ones((1, 4))         # row i == i
ok("same keys -> identity", np.array_equal(W.map_input_rows(KEYS, U, KEYS), U))
ok("loadmat-padded source keys -> identity", np.array_equal(W.map_input_rows(padded, U, KEYS), U))
em4 = ["T_motor_fl", "T_motor_fr", "T_motor_rl", "T_motor_rr", "T_brake", "delta"]
ok("EM4 On: each corner motor takes the source motor row; brake/steer by name",
   np.array_equal(W.map_input_rows(KEYS, U, em4)[:, 0], [0, 0, 0, 0, 1, 6]))
aero = ["T_motor", "T_brake", "FW", "RW", "delta"]
src_no_atd = ["T_motor", "T_brake", "delta"]
ok("missing channels get neutral seeds (wings 0)",
   np.array_equal(W.map_input_rows(src_no_atd, U[[0, 1, 6]], aero)[:, 0], [0, 1, 0, 0, 6]))
ok("missing ATD rows seeded at 0.25",
   np.array_equal(W.map_input_rows(src_no_atd, U[[0, 1, 6]], KEYS)[:, 0], [0, 1, .25, .25, .25, .25, 6]))
UA = U.copy(); UA[2:6] = np.array([[10.0], [11.0], [12.0], [13.0]])
ok("repeated keys map occurrence by occurrence",
   np.array_equal(W.map_input_rows(KEYS, UA, KEYS)[2:6, 0], [10, 11, 12, 13]))

# ---- 6. primal guesses from a full result (s-based interpolation) -----------------
print("guesses_from_full")
sf = disc["s_full"]
xs = np.linspace(1.0, 3.0, 23)                              # scaling vectors
us = np.linspace(1.0, 2.0, 7)
x_full = np.vstack([1.0 + 0.1 * i + 0.01 * i * sf for i in range(23)])     # linear in s
knot_idx = np.arange(0, sf.size, d + 1)
col_idx = np.array([k * (d + 1) + 1 + j for k in range(N) for j in range(d)])
src = SimpleNamespace(x_opt=x_full[:, knot_idx], x_full=x_full, s_full=sf, OPT_d=d,
                      u_opt=np.vstack([np.sin(disc["s_knot"] / 50.0 + i) for i in range(7)]),
                      input_keys=KEYS)
ok("collocation columns of s_full are disc['s_col']", np.allclose(sf[col_idx], disc["s_col"]))
g = W.guesses_from_full(src, disc["s_knot"], disc["s_col"], KEYS, xs, us)
ok("same grid: x0 reproduces x_opt", np.allclose(g["x0"] * xs[:, None], src.x_opt))
ok("same grid: xc0 reproduces the collocation states", np.allclose(g["xc0"] * xs[:, None], x_full[:, col_idx]))
ok("same grid: u0 reproduces u_opt", np.allclose(g["u0"] * us[:, None], src.u_opt))
ok("guess shapes", g["x0"].shape == (23, N + 1) and g["u0"].shape == (7, N + 1)
   and g["xc0"].shape == (23, N * d))
disc2 = discretise(track, 15, 2)                            # finer N, other degree
g2 = W.guesses_from_full(src, disc2["s_knot"], disc2["s_col"], KEYS, xs, us)
ok("new grid: shapes follow the new N / OPT_d",
   g2["x0"].shape == (23, disc2["N"] + 1) and g2["xc0"].shape == (23, disc2["N"] * 2))
ok("new grid: linear-in-s states interpolated exactly at knots",
   np.allclose(g2["x0"][5] * xs[5], 1.5 + 0.05 * disc2["s_knot"]))
ok("new grid: and at collocation points",
   np.allclose(g2["xc0"][5] * xs[5], 1.5 + 0.05 * disc2["s_col"]))
src_nofull = SimpleNamespace(**{k: v for k, v in vars(src).items() if k != "x_full"})
g3 = W.guesses_from_full(src_nofull, disc["s_knot"], disc["s_col"], KEYS, xs, us)
ok("no x_full: collocation states held from the knots",
   np.allclose(g3["xc0"], np.kron(g3["x0"][:, :-1], np.ones((1, d)))))

# ---- 7. resolve_source: init vs full, file vs memory ---------------------------
print("resolve_source")
tmp = tempfile.mkdtemp()
init_path = os.path.join(tmp, "init_Sturn.mat")
sio.savemat(init_path, {"data": {"init": {"x_opt": np.ones((7, N + 1)), "u_opt": np.ones((3, N + 1))}}})
kind, obj = W.resolve_source(init_path)
ok("init file -> ('init', data.init)", kind == "init" and np.shape(obj.x_opt) == (7, N + 1))
legacy = importfile(init_path)["data"].init
ok("init path returns the same struct the legacy importfile branch read",
   np.array_equal(obj.x_opt, legacy.x_opt) and np.array_equal(obj.u_opt, legacy.u_opt))
full_path = os.path.join(tmp, "Sturn_full.mat")
sio.savemat(full_path, {"data": {"x_opt": src.x_opt, "u_opt": src.u_opt, "s_full": sf,
                                 "OPT_d": d, "input_keys": KEYS}})
kind, obj = W.resolve_source(full_path)
ok("full result file -> ('full', data)", kind == "full" and np.shape(obj.x_opt) == (23, N + 1))
ctx_like = SimpleNamespace(data=SimpleNamespace(x_opt=src.x_opt))
ok("MLTP ctx in memory -> full", W.resolve_source(ctx_like)[0] == "full")
ok("data namespace -> full", W.resolve_source(ctx_like.data)[0] == "full")
ok("data dict -> full", W.resolve_source({"x_opt": src.x_opt})[0] == "full")
ctx_init = SimpleNamespace(data=SimpleNamespace(init=SimpleNamespace(x_opt=np.ones((7, 4)))))
ok("MLTP_initial ctx in memory -> init", W.resolve_source(ctx_init)[0] == "init")
try:
    W.resolve_source(SimpleNamespace(data=SimpleNamespace(foo=1))); raised = False
except ValueError:
    raised = True
ok("unrecognised warm_start raises ValueError", raised)

# ---- 8. plan_full_warm_start ------------------------------------------------------
print("plan_full_warm_start")
stN = W.nlp_structure(23, 7, 0, N, d, 8)
rs = np.random.RandomState(0)
res = {"w_opt": rs.rand(stN["n_w"]), "lam_g": rs.randn(stN["n_g"]), "lam_x": rs.randn(stN["n_w"]),
       "structure": stN, "sym_type": "SX", "linear_solver": "ma57"}
rec = W.nlp_record(res, {"iter_count": 3209, "return_status": "Solve_Succeeded"}, xs, us, "cold")
ok("nlp_record fields", set(rec) >= {"w_opt", "lam_g", "lam_x", "structure", "sym_type", "linear_solver",
                                     "ipopt_iters", "return_status", "x_s", "u_s"}
   and rec["ipopt_iters"] == 3209 and rec["structure"] == stN)
expected = dict(stN, input_keys=KEYS, s_full=sf, x_s=xs, u_s=us)
src_mem = SimpleNamespace(**vars(src), nlp=rec)            # like ctx.data (nlp is a dict)
warm, mode, note = W.plan_full_warm_start(src_mem, expected)
ok("identical structure -> full+duals", mode == "full+duals" and warm is not None)
ok("warm carries the exact saved w_opt / lam_g / lam_x",
   np.array_equal(warm["x0"], res["w_opt"]) and np.array_equal(warm["lam_g0"], res["lam_g"])
   and np.array_equal(warm["lam_x0"], res["lam_x"]))
ok("warm carries the IPOPT recipe", warm["ipopt"] == W.warm_start_ipopt_opts())
warm, _, _ = W.plan_full_warm_start(src_mem, expected, ipopt_overrides={"mu_init": 1e-4, "tol": 1e-6})
ok("per-call ipopt_overrides win over the recipe",
   warm["ipopt"]["mu_init"] == 1e-4 and warm["ipopt"]["tol"] == 1e-6
   and warm["ipopt"]["warm_start_init_point"] == "yes")
warm, mode, _ = W.plan_full_warm_start(src_mem, expected, use_duals=False)
ok("use_duals=False -> full-primal: x0 only, no recipe",
   mode == "full-primal" and set(warm) == {"x0"})
warm, mode, note = W.plan_full_warm_start(src_mem, dict(expected, N=N + 1))
ok("different structure -> full-interp, no warm vectors", mode == "full-interp" and warm is None
   and "differs" in note)
# the sweep case: the source was solved with other vp / tyre values (vp_overrides);
# nothing setup-dependent is in the structure, and a moved scaling only adds a note
src_other_setup = SimpleNamespace(**vars(src), nlp=dict(rec, x_s=xs * 1.05),
                                  vp_overrides={"brkB": 0.69, "pKy4": 2.0})
warm, mode, note = W.plan_full_warm_start(src_other_setup, expected)
ok("source with different vp_overrides / scaling -> still full+duals",
   mode == "full+duals" and np.array_equal(warm["lam_g0"], res["lam_g"]) and "x_s" in note)
warm, mode, note = W.plan_full_warm_start(src, expected)
ok("full result without data.nlp -> full-interp", mode == "full-interp" and warm is None)
bad = SimpleNamespace(**vars(src), nlp=dict(rec, return_status="Maximum_Iterations_Exceeded"))
with warnings.catch_warnings(record=True) as wrec:
    warnings.simplefilter("always")
    warm, mode, _ = W.plan_full_warm_start(bad, expected)
ok("failed source solve -> warns, still re-injects",
   mode == "full+duals" and any("Maximum_Iterations" in str(x.message) for x in wrec))

# the tyre-SET guard: a result solved with another tyre set (every pre-flip result has no
# data.tyre_set, i.e. the legacy CopyB set) is no usable start, primal-only included
exp_mf, exp_cb = dict(expected, tyre_set="MF205"), dict(expected, tyre_set="CopyB")
src_mf = SimpleNamespace(**vars(src_mem), tyre_set="MF205")
src_cb = SimpleNamespace(**vars(src_mem), tyre_set="CopyB")
warm, mode, note = W.plan_full_warm_start(src_mem, exp_mf)
ok("source without tyre_set (pre-flip CopyB) vs MF205 target -> cold, no warm vectors, note names both",
   mode == "cold" and warm is None and "'CopyB'" in note and "'MF205'" in note)
ok("tyre-set mismatch -> cold even primal-only (use_duals=False) and with no data.nlp",
   W.plan_full_warm_start(src_mem, exp_mf, use_duals=False)[:2] == (None, "cold")
   and W.plan_full_warm_start(src, exp_mf)[:2] == (None, "cold"))
ok("source MF205 vs CopyB target -> cold", W.plan_full_warm_start(src_mf, exp_cb)[:2] == (None, "cold"))
warm, mode, _ = W.plan_full_warm_start(src_mf, exp_mf)
ok("source MF205 vs MF205 target -> full+duals", mode == "full+duals" and warm is not None)
warm, mode, _ = W.plan_full_warm_start(src_cb, exp_cb)
ok("source CopyB vs CopyB target -> full+duals", mode == "full+duals" and warm is not None)
ok("expected without a tyre_set key -> unchanged, whatever the source carries",
   W.plan_full_warm_start(src_mem, expected)[1] == "full+duals"
   and W.plan_full_warm_start(src_mf, expected)[1] == "full+duals"
   and W.plan_full_warm_start(src, expected)[1] == "full-interp")
ok("dict source with a blank-padded tyre_set still matches",
   W.plan_full_warm_start(dict(vars(src_mem), tyre_set="MF205 "), exp_mf)[1] == "full+duals")

# the file route: savemat -> importfile / load_solution -> same decision
nlp_path = os.path.join(tmp, "Sturn_nlp.mat")
sio.savemat(nlp_path, {"data": {"x_opt": src.x_opt, "u_opt": src.u_opt, "x_full": x_full,
                                 "s_full": sf, "OPT_d": d, "input_keys": KEYS, "nlp": rec}},
            do_compression=True)
for label, loaded in (("load_solution", load_solution(nlp_path)),
                      ("resolve_source", W.resolve_source(nlp_path)[1])):
    warm, mode, _ = W.plan_full_warm_start(loaded, expected)
    ok(f"{label}: saved record re-injects (padded keys, squeezed ints handled)",
       mode == "full+duals" and np.array_equal(warm["x0"], res["w_opt"])
       and np.array_equal(warm["lam_g0"], res["lam_g"]))

print("\nALL warmstart TESTS PASSED")
