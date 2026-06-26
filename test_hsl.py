"""HSL integration tests (plain script, no pytest).

Sections 1-3 are casadi-free (resolution / path / opts-rewriting via an injected
fake probe); section 4's real-DLL smoke runs only if a CoinHSL DLL resolves here.
Run from the repo root:

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

def _boom_probe(*_):
    raise AssertionError("probe must not run for an unsupported ma* name")

unsup = H.apply_linear_solver(base, linear_solver="ma86", hsl_dir=_d, probe=_boom_probe)
ok("unsupported ma* name -> fallback mumps, probe never reached",
   unsup["ipopt"]["linear_solver"] == "mumps" and "hsllib" not in unsup["ipopt"])

# ---- 4. guarded real-DLL smoke (skips if no CoinHSL on this machine) ---------
print("real probe smoke")
_dir = H.resolve_hsl_dir()
_lib_real = H.hsllib_path(_dir) if _dir else None
if _lib_real:
    ok("real ma57 probes True", H.probe_linear_solver("ma57", _lib_real) is True)
else:
    print("  [SKIP] no CoinHSL DLL resolved; skipping real-probe smoke")

# ---- 5. _make_solver: pops _hsl_dir, falls back, uses HSL when present -------
print("_make_solver")
import casadi as ca
from functions.transcription import _make_solver

_x = ca.MX.sym("x")
_nlp = {"x": _x, "f": (_x - 1) ** 2, "g": _x}

# (a) bogus _hsl_dir + ma57 -> must NOT crash (key popped) and must solve (mumps)
S = _make_solver(ca, _nlp, {"ipopt": {"linear_solver": "ma57", "print_level": 0,
                                       "max_iter": 50},
                            "_hsl_dir": r"Z:\no_such_dir"})
S(x0=0, lbg=-10, ubg=10)
ok("ma57 + bogus dir -> falls back and solves",
   S.stats().get("return_status") == "Solve_Succeeded")

# (b) explicit mumps -> solves, _hsl_dir absent is fine
S2 = _make_solver(ca, _nlp, {"ipopt": {"linear_solver": "mumps", "print_level": 0,
                                        "max_iter": 50}})
S2(x0=0, lbg=-10, ubg=10)
ok("mumps still works",
   S2.stats().get("return_status") == "Solve_Succeeded")

# (c) if a real DLL resolves, ma57 via the default resolver solves on HSL
if _lib_real:   # set in Section 4
    S3 = _make_solver(ca, _nlp, {"ipopt": {"linear_solver": "ma57",
                                           "print_level": 0, "max_iter": 50}})
    S3(x0=0, lbg=-10, ubg=10)
    ok("ma57 via default resolver solves",
       S3.stats().get("return_status") == "Solve_Succeeded")
else:
    print("  [SKIP] no CoinHSL DLL resolved; skipping _make_solver HSL check")

# ---- 6. userOpts wiring: intent flows into ctx.opts -------------------------
print("userOpts wiring")
from functions.context import Ctx
from userOpts import userOpts

c1 = Ctx(); userOpts(c1, circuit="Sturn")          # synthetic track, no files needed
ok("default linear_solver is ma57",
   c1.opts["ipopt"]["linear_solver"] == "ma57")
ok("_hsl_dir present (default None)",
   "_hsl_dir" in c1.opts and c1.opts["_hsl_dir"] is None)

c2 = Ctx(); userOpts(c2, circuit="Sturn", linear_solver="mumps")
ok("explicit mumps honored", c2.opts["ipopt"]["linear_solver"] == "mumps")

c3 = Ctx(); userOpts(c3, circuit="Sturn", linear_solver="ma97", hsl_dir=r"X:\hsl\bin")
ok("ma97 + hsl_dir threaded",
   c3.opts["ipopt"]["linear_solver"] == "ma97" and c3.opts["_hsl_dir"] == r"X:\hsl\bin")

print("ALL HSL TESTS PASSED")
