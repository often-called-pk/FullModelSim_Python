"""Python port vs the MATLAB original, no solve (plain script, no pytest). Run from anywhere:

    venv\\Scripts\\python.exe tests\\test_matlab_parity.py

Section 1 is validation gate 1 (docs/validation_matlab_vs_python.md section 4): the parameter dump of the
Python port (validation/py_export.params, parity settings: uniform mesh, tol 1e-8, MUMPS, monotone mu) against
the dump of the MATLAB original that tests/data/matlab_params_<track>_ds<ds>.json stores. MATLAB R2025a ran
userOpts.m -> MLTP_initial.m -> vehModel.m -> MLTP.m up to nlpsol at IPOPT max_iter 0 on a patched scratch
copy (validation/matlab_batch.py export; the shipped IPOPT options are the ones dumped). Re-create a dump with

    python validation/matlab_batch.py export --circuit BCN --opt-ds 10 --out tests\\data

Tracks: Sturn at OPT_ds 30, BCN and NBR at OPT_ds 10 (the MATLAB default). Floats rel 1e-12, ints and strings
exact. The only expected hit is the dormant camber-gain pitch table. Needs casadi and Data/DATA_AA.mat
(prints SKIP and exits 0 otherwise).

Section 2 is validation gate 2 (docs section 5): the Python NLP built at IPOPT max_iter 0 (validation/py_export.nlp,
env MLTP_KEEP_NLP=1 keeps f, g and the bounds) against tests/data/matlab_nlp_Sturn_ds30.mat: the bounds bit-exact,
f and g at MATLAB's w0, a seeded random point and its 2026-10-04 solution agree to 1e-10 (scaled), and the start
point w0 (built under validation.py_export.matlab_seed: states 9-22 at OPT_e as MLTP.m:229-251) equals MATLAB's.
BCN and NBR (N 465 / 514, ~85 s and 8 GB each) are not run here, see validation/compare.py nlp --help.
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import copy
import json
import math
import os
import sys

import scipy.io as sio

ROOT = _bootstrap.ROOT

try:
    import casadi  # noqa: F401
except ImportError as exc:                                      # pragma: no cover
    print(f"SKIP test_matlab_parity: casadi not importable ({exc})")
    sys.exit(0)
if not os.path.exists(os.path.join(ROOT, "Data", "DATA_AA.mat")):   # pragma: no cover
    print("SKIP test_matlab_parity: Data/DATA_AA.mat missing")
    sys.exit(0)

from validation.compare import diff_params
from validation.py_export import params

DATA = os.path.join(ROOT, "tests", "data")


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


# =============================================================================
print("1. gate 1: parameter parity, Python vs the stored MATLAB dump")
# dormant: CamberGain is 'Off' in both codes and the Python table is None (MATLAB: 14 values, vehParams.m L445-460)
EXPECTED = {"vp.CG_p_deg_per_deg_table"}
CASES = (("Sturn", 30, 18, None), ("BCN", 10, 465, "a9150a44"), ("NBR", 10, 514, "bb6589ab"))

for circuit, ds, N, track_sha in CASES:
    with open(os.path.join(DATA, f"matlab_params_{circuit}_ds{ds}.json")) as f:
        mat = json.load(f)
    py = params(circuit, ds)
    info = []
    hits = diff_params(mat, py, info)
    print(f"  {circuit} OPT_ds {ds}: N {py['N']}, n_w {py['n_w']}, n_g {py['n_g']}")
    ok(f"{circuit}: the stored dump is the intended case (N {N}, 59 Pacejka values, aero file 098f156f)",
       mat["circuit"] == circuit and mat["N"] == N and len(mat["mf"]) == 59
       and mat["aero_sha256"].startswith("098f156f")
       and (mat["track_sha256"] or "").startswith(track_sha or ""))
    ok(f"{circuit}: shipped IPOPT options (tol 1e-8, acceptable 1e-6, max_iter 6000, MUMPS and monotone unset)",
       mat["ipopt"]["tol"] == 1e-8 and mat["ipopt"]["acceptable_tol"] == 1e-6 and mat["ipopt"]["max_iter"] == 6000
       and "linear_solver" not in mat["ipopt"] and "mu_strategy" not in mat["ipopt"])
    ok(f"{circuit}: steering u_s is pi/8 in both codes", mat["u_s"][-1] == math.pi / 8 == py["u_s"][-1])
    ok(f"{circuit}: no hit but the expected one", {k for k, _ in hits} == EXPECTED)
    ok(f"{circuit}: Python-only IPOPT keys are print_timing_statistics only", [k for k, _ in info] == ["ipopt.print_timing_statistics"])
    if circuit == "Sturn":                  # the diff must see a real difference and ignore rounding noise
        bad = copy.deepcopy(py)
        bad["mf"]["pKy1"] *= 1 + 1e-9
        bad["hnames"] = bad["hnames"][::-1]
        bad["ipopt"]["mu_strategy"] = "adaptive"
        bad["ipopt"]["bound_push"] = 0.1
        bad["Xi"][0] = float("nan")
        near = copy.deepcopy(py)
        near["mf"]["pKy1"] *= 1 + 1e-13
        ok("diff_params reports 1e-9 / reordered names / adaptive mu / bound_push / NaN",
           {k for k, _ in diff_params(mat, bad)} == EXPECTED | {"mf.pKy1", "hnames", "ipopt.mu_strategy", "ipopt.bound_push", "Xi"})
        ok("diff_params ignores a 1e-13 relative change", {k for k, _ in diff_params(mat, near)} == EXPECTED)
        n_w, n_g = mat["n_w"], mat["n_g"]

# the stored Sturn NLP numbers belong to that dump (layout: validation/matlab/vexport.m, used by section 2)
nlp = sio.loadmat(os.path.join(DATA, "matlab_nlp_Sturn_ds30.mat"), squeeze_me=True)
ok("Sturn NLP file: columns w0, w_rand, w_star; sizes n_w 1812, n_g 1904 as in the dump",
   list(nlp["W_names"]) == ["w0", "w_rand", "w_star"] and (n_w, n_g) == (1812, 1904)
   and nlp["W"].shape == (n_w, 3) and nlp["G"].shape == (n_g, 3) and nlp["F"].shape == (3,)
   and nlp["lbw"].size == nlp["ubw"].size == n_w and nlp["lbg"].size == nlp["ubg"].size == n_g)
ok("Sturn NLP file: f(w_star) = 18.0540140701883, the objective of the MATLAB solve of 2026-10-04 (tol 1e-8, 299 iterations)",
   abs(nlp["F"][2] - 18.054014070188256) <= 1e-12 * 18.054014070188256)

print("\nsection 1 (gate 1) passed")

# --- section 2: NLP-function equivalence (builder D) ---

# =============================================================================
print("\n2. gate 2: NLP functions, Python vs the stored MATLAB export (no iteration run)")
# Docs section 5. tests/data/matlab_nlp_Sturn_ds30.mat holds MATLAB's bounds and f, g at w0, a seeded random point and
# w_star (the MATLAB solution of 2026-10-04). nlp_py builds the Python NLP through MLTP at IPOPT max_iter 0
# (MLTP_KEEP_NLP=1 keeps f, g and the bounds; the start point is MATLAB's, via matlab_seed) and evaluates f, g at the
# same three columns.
# Bounds are bit-exact (bound tol 0, the gate's default): vehModel.m:130-138 forms `1/x_s * lim` and vehModel.py:88
# `lim * (1 / x_s)`; the earlier `lim / x_s` was 1 ulp off at the four wheel-speed limits.
# BCN (N 465) and NBR (N 514) take ~85 s and 8 GB each: run them with `python validation/compare.py nlp ...`, not here.
import numpy as np
from types import SimpleNamespace

import MLTP as mltp_mod
from functions.ladder import quasi_static_states
from validation.compare import NLP_TOL, nlp_gate
from validation.py_export import matlab_seed, nlp as nlp_py

nlp_path = os.path.join(DATA, "matlab_nlp_Sturn_ds30.mat")
py = nlp_py("Sturn", 30, W=sio.loadmat(nlp_path, squeeze_me=True)["W"])
gate = nlp_gate(nlp_path, py)
cols = {c["name"]: c for c in gate["columns"]}
print("  bounds, max |a-b|/|a|: " + ", ".join(f"{k} {v:.2e}" for k, v in gate["bounds"].items()) + " (0 = bit-exact)")
print("  f, g: " + ", ".join(f"{n} dF {c['dF']:.1e} dG {c['dG']:.1e}" for n, c in cols.items()))
print(f"  start point, max |w0_py - w0_matlab| = {gate['w0']:.1e}")

ok("same sizes as the MATLAB export: n_w 1812, n_g 1904", gate["n_w"] == (1812, 1812) and gate["n_g"] == (1904, 1904))
ok("evaluated at the three MATLAB columns w0, w_rand, w_star", list(cols) == ["w0", "w_rand", "w_star"])
ok("bounds bit-exact (max |a-b|/|a| = 0)", max(gate["bounds"].values()) == 0)
ok("f and g agree to 1e-10 (scaled by max(1, |a|)) at w0, w_rand and w_star",
   all(max(c["dF"], c["dG"]) <= NLP_TOL for c in cols.values()))
ok("start point w0 equals MATLAB's (matlab_seed: states 9-22 at OPT_e), max abs diff 0", gate["w0"] == 0)
ok("gate 2 verdict", gate["ok"])
ok("f(w_star) = 18.054014070188256 (the MATLAB objective) in the Python NLP, rel 1e-10",
   abs(py["F"][2] - 18.054014070188256) <= 1e-10 * 18.054014070188256)

# the gate must see a real difference and ignore rounding noise
def copy_of(d):
    return {k: v.copy() for k, v in d.items()}


def gate_of(d):
    return nlp_gate(nlp_path, d)


near = copy_of(py)
near["G"] *= 1 + 1e-13
ok("noise (g x (1 + 1e-13)) passes", gate_of(near)["ok"])
bad = copy_of(py)
bad["ubw"][0] = np.nextafter(bad["ubw"][0], 0.0)
ok("a bound 1 ulp off is reported (bound tol 0) and passes only with an explicit bound_tol of 5e-16",
   not gate_of(bad)["ok"] and gate_of(bad)["bounds"]["ubw"] > 0 and nlp_gate(nlp_path, bad, bound_tol=5e-16)["ok"])
bad = copy_of(py)
bad["lbg"][-1] *= 1 + 1e-9
ok("a bound off by 1e-9 rel is reported", gate_of(bad)["bounds"]["lbg"] > 0 and not gate_of(bad)["ok"])
bad = copy_of(py)
bad["F"][1] *= 1 + 1e-9
ok("f off by 1e-9 rel is reported", gate_of(bad)["columns"][1]["dF"] > NLP_TOL)
bad = copy_of(py)
bad["G"][np.argmax(np.abs(py["G"][:, 2])), 2] *= 1 + 1e-9
ok("the largest g off by 1e-9 rel is reported", gate_of(bad)["columns"][2]["dG"] > NLP_TOL)
bad = copy_of(py)
bad["G"][7, 0] = np.nan
ok("a NaN where MATLAB has a number is reported", gate_of(bad)["columns"][0]["dG"] == np.inf)

# matlab_seed: states 5-8 stay the quasi-static seed, states 9-22 are OPT_e, and the seed is restored on exit
# (also when the build raises), so production code keeps its quasi-static seed
vp = SimpleNamespace(Rw_f=0.3, Rw_r=0.4, kt=2.0, Wfl0=1.0, Wfr0=2.0, Wrl0=3.0, Wrr0=4.0)
vx = np.array([10.0, 20.0, 30.0])
ref = quasi_static_states(vp, vx)
try:
    with matlab_seed():
        seeded = mltp_mod.quasi_static_states(vp, vx)
        raise KeyError
except KeyError:
    pass
ok("matlab_seed: states 5-8 are the quasi-static ones, states 9-22 are 1e-2", seeded.shape == ref.shape == (18, 3)
   and np.array_equal(seeded[:4], ref[:4]) and np.all(seeded[4:] == 1e-2) and not np.array_equal(seeded, ref))
ok("matlab_seed: the quasi-static seed is back after the with", mltp_mod.quasi_static_states is quasi_static_states)

# the hook is opt-in: without MLTP_KEEP_NLP nothing is kept on ctx
# (and nothing reaches the saved .mat: MLTP builds `data` key by key)
from MLTP import MLTP
from validation.py_export import PARITY

os.environ.pop("MLTP_KEEP_NLP", None)
ctx = MLTP("Sturn", OPT_ds=30, mesh="uniform", max_iter=0, save=False, plot=False, warm_start=None, **PARITY)
ok("MLTP_KEEP_NLP unset: ctx has no nlp_fg / nlp_bounds", not hasattr(ctx, "nlp_fg") and not hasattr(ctx, "nlp_bounds"))

print("\nsection 2 (gate 2) passed")
