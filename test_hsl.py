"""HSL integration tests (plain script, no pytest).

Sections 1-3 are casadi-free (resolution / path / opts-rewriting via an injected
fake probe). Section 4 is a guarded smoke that only runs if a real CoinHSL DLL
resolves on this machine. Run from the repo root:

    venv\\Scripts\\python.exe test_hsl.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))

import functions.hsl as H

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# ---- 1. resolve_hsl_dir precedence: env > explicit > default > None ----------
print("resolve_hsl_dir")
_saved_env = os.environ.pop("COINHSL_DIR", None)
_saved_default = H._DEFAULT_HSL_DIR
try:
    real_a = tempfile.mkdtemp(prefix="hsl_a_")   # stands in for an explicit dir
    real_b = tempfile.mkdtemp(prefix="hsl_b_")   # stands in for COINHSL_DIR
    H._DEFAULT_HSL_DIR = r"Z:\definitely_not_a_real_hsl_dir"

    ok("env unset, explicit existing -> explicit",
       H.resolve_hsl_dir(real_a) == os.path.abspath(real_a))
    os.environ["COINHSL_DIR"] = real_b
    ok("env set -> env wins over explicit",
       H.resolve_hsl_dir(real_a) == os.path.abspath(real_b))
    del os.environ["COINHSL_DIR"]
    ok("nothing resolves -> None",
       H.resolve_hsl_dir(r"Z:\also_not_real") is None)
finally:
    H._DEFAULT_HSL_DIR = _saved_default
    if _saved_env is not None:
        os.environ["COINHSL_DIR"] = _saved_env

# ---- 2. hsllib_path ----------------------------------------------------------
print("hsllib_path")
_d = tempfile.mkdtemp(prefix="hsl_lib_")
ok("empty dir -> None", H.hsllib_path(_d) is None)
_lib = os.path.join(_d, "libhsl.dll")
open(_lib, "wb").close()
ok("dir with libhsl.dll -> that path", H.hsllib_path(_d) == _lib)
ok("None dir -> None", H.hsllib_path(None) is None)

# ---- 3. apply_linear_solver opts-rewriting (injected fake probe) -------------
print("apply_linear_solver")
base = {"ipopt": {"max_iter": 10, "tol": 1e-4}}

m = H.apply_linear_solver(base, linear_solver="mumps", hsl_dir=_d, probe=lambda *_: True)
ok("mumps passes through, no hsllib",
   m["ipopt"]["linear_solver"] == "mumps" and "hsllib" not in m["ipopt"])

good = H.apply_linear_solver(base, linear_solver="ma57", hsl_dir=_d, probe=lambda *_: True)
ok("ma57 + working probe -> ma57 + hsllib",
   good["ipopt"]["linear_solver"] == "ma57" and good["ipopt"]["hsllib"] == _lib)

bad = H.apply_linear_solver(base, linear_solver="ma57", hsl_dir=_d, probe=lambda *_: False)
ok("ma57 + failing probe -> fallback mumps",
   bad["ipopt"]["linear_solver"] == "mumps" and "hsllib" not in bad["ipopt"])

none = H.apply_linear_solver(base, linear_solver="ma57", hsl_dir=None, probe=lambda *_: True)
ok("ma57 + no dir -> fallback mumps (probe not reached)",
   none["ipopt"]["linear_solver"] == "mumps" and "hsllib" not in none["ipopt"])

ok("input opts not mutated", base["ipopt"].get("linear_solver") is None)

# ---- 4. guarded real-DLL smoke (skips if no CoinHSL on this machine) ---------
print("real probe smoke")
_dir = H.resolve_hsl_dir()
_lib_real = H.hsllib_path(_dir) if _dir else None
if _lib_real:
    ok("real ma57 probes True", H.probe_linear_solver("ma57", _lib_real) is True)
else:
    print("  [SKIP] no CoinHSL DLL resolved; skipping real-probe smoke")

print("ALL HSL TESTS PASSED")
