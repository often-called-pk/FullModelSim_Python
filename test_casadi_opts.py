"""casadi_opts: CSE and JIT are both opt-in, JIT falls back safely (plain script, no pytest).

Compiler detection and the compile probe are injected, so this runs on a machine
with no C compiler. Section 8 is a guarded smoke that only runs a real JIT build if
a working compiler is on PATH. Run from the repo root:

    venv\\Scripts\\python.exe test_casadi_opts.py
"""
import os, re, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import casadi as ca
import functions.casadi_opts as C
from functions.context import Ctx
from userOpts import userOpts

HERE = os.path.dirname(os.path.abspath(__file__))

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

def run(fn, *a, **k):
    """Call fn, returning (result, [RuntimeWarning messages])."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fn(*a, **k)
    return res, [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]

os.environ.pop("MLTP_JIT", None)          # process-local: keep the env switches out of every section
os.environ.pop("MLTP_CSE", None)
_saved = dict(avail=C.jit_available, probe=C._probe_jit, warned=C._warned)
try:
    # ---- 1. default: nothing requested -> {} (CSE and JIT are both opt-in) ------
    print("default")
    ok("fn_opts() == {}", C.fn_opts() == {})
    ok("fn_opts(ctx) without cse/jit == {}", C.fn_opts(Ctx()) == {})
    ok("fn_opts(jit=False) == {}", C.fn_opts(jit=False) == {})
    d = C.fn_opts(); d["junk"] = 1
    ok("each call returns a fresh dict", C.fn_opts() == {})
    for val in ("0", "false", "OFF", "no", ""):
        os.environ["MLTP_CSE"] = val
        ok(f"MLTP_CSE={val!r} leaves CSE off", C.fn_opts() == {})
    for val in ("1", "true", "yes", "on", "ON", " True "):
        os.environ["MLTP_CSE"] = val
        ok(f"MLTP_CSE={val!r} turns CSE on", C.fn_opts() == {"cse": True})
    d = C.fn_opts(); d["junk"] = 1
    ok("each call returns a fresh dict (CSE on)", C.fn_opts() == {"cse": True})
    del os.environ["MLTP_CSE"]
    ok("MLTP_CSE unset -> {}", C.fn_opts() == {})
    c = Ctx(); c.cse = True
    ok("ctx.cse=True turns CSE on", C.fn_opts(c) == {"cse": True})
    c = Ctx(); c.cse = False
    ok("ctx.cse=False leaves CSE off", C.fn_opts(c) == {})
    ok("ctx.cse does not leak into a fresh ctx", C.fn_opts(Ctx()) == {})

    # ---- 2. JIT requested, no compiler -> warn once, JIT stays off ----------------
    print("JIT requested, no compiler")
    C.jit_available = lambda: False
    C._warned = False
    (o1, w1) = run(C.fn_opts, jit=True)
    (o2, w2) = run(C.fn_opts, jit=True)
    ok("jit=True w/o compiler -> no JIT, default opts", o1 == {} and o2 == {})
    ok("warns exactly once per process", len(w1) == 1 and len(w2) == 0)
    ok("warning names the problem", "JIT" in w1[0] and "compiler" in w1[0])
    ok("warning does not claim CSE is on", "CSE" not in w1[0])
    c = Ctx(); c.jit = True
    ok("ctx.jit w/o compiler -> no JIT, default opts", run(C.fn_opts, c)[0] == {})
    os.environ["MLTP_JIT"] = "1"
    ok("MLTP_JIT=1 w/o compiler -> no JIT, default opts", run(C.fn_opts)[0] == {})
    del os.environ["MLTP_JIT"]
    c = Ctx(); c.jit = True; c.cse = True
    ok("cse + jit w/o compiler -> CSE kept, no JIT", run(C.fn_opts, c)[0] == {"cse": True})

    # ---- 3. JIT requested, compiler found and working ----------------------------
    print("JIT requested, compiler works")
    C.jit_available = lambda: True
    C._probe_jit = lambda jit_options: True
    C._warned = False
    jopts, w = run(C.fn_opts, jit=True)
    ok("jit=True -> jit/compiler/jit_options added, no cse by default",
       "cse" not in jopts and jopts.get("jit") is True
       and jopts.get("compiler") == "shell" and isinstance(jopts.get("jit_options"), dict))
    ok("jit_options carry an optimisation flag, not verbose",
       len(jopts["jit_options"]["flags"]) == 1 and jopts["jit_options"]["verbose"] is False)
    ok("no warning when JIT is granted", w == [])
    c = Ctx(); c.jit = True
    ok("ctx.jit=True requests JIT", C.fn_opts(c).get("jit") is True)
    ok("ctx.jit=False does not", "jit" not in C.fn_opts(Ctx()))
    os.environ["MLTP_JIT"] = "1"
    ok("MLTP_JIT=1 requests JIT", C.fn_opts().get("jit") is True)
    os.environ["MLTP_JIT"] = "0"
    ok("MLTP_JIT=0 does not", "jit" not in C.fn_opts())
    del os.environ["MLTP_JIT"]
    ok("nothing requested -> no JIT even with a compiler", "jit" not in C.fn_opts())
    c = Ctx(); c.jit = True; c.cse = True
    both = C.fn_opts(c)
    ok("cse and jit are independent: both requested -> both present",
       both.get("cse") is True and both.get("jit") is True)
    c = Ctx(); c.cse = True
    ok("cse alone does not request JIT", C.fn_opts(c) == {"cse": True})

    # ---- 4. compiler on PATH but a test build fails -> warn once, JIT stays off ----
    print("JIT requested, compiler broken")
    C._probe_jit = lambda jit_options: False
    C._warned = False
    (ob, wb) = run(C.fn_opts, jit=True)
    ok("failed probe -> no JIT, default opts", ob == {})
    ok("failed probe -> one warning", len(wb) == 1 and "failed" in wb[0])
    c = Ctx(); c.cse = True
    ok("failed probe + cse -> CSE kept, no JIT", run(C.fn_opts, c, jit=True)[0] == {"cse": True})
finally:
    C.jit_available, C._probe_jit, C._warned = _saved["avail"], _saved["probe"], _saved["warned"]
    os.environ.pop("MLTP_JIT", None)
    os.environ.pop("MLTP_CSE", None)

# ---- 5. userOpts(jit=..., cse=...) -> ctx.jit / ctx.cse -> fn_opts(ctx) ---------------
print("userOpts")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ok("userOpts default jit is False", userOpts(Ctx(), circuit="Sturn").jit is False)
    ok("userOpts(jit=True) sets ctx.jit", userOpts(Ctx(), circuit="Sturn", jit=True).jit is True)
    ok("userOpts default cse is False", userOpts(Ctx(), circuit="Sturn").cse is False)
    ok("userOpts(cse=True) sets ctx.cse", userOpts(Ctx(), circuit="Sturn", cse=True).cse is True)
    ok("userOpts default -> fn_opts(ctx) == {}", C.fn_opts(userOpts(Ctx(), circuit="Sturn")) == {})
    ok("userOpts(cse=True) -> fn_opts(ctx) == {'cse': True}",
       C.fn_opts(userOpts(Ctx(), circuit="Sturn", cse=True)) == {"cse": True})

# ---- 6. casadi accepts the options and results are unchanged -------------------------
print("casadi Function")
x = ca.SX.sym("x", 2)
y = ca.vertcat(ca.sin(x[0] * x[1]) * ca.sin(x[0] * x[1]) + ca.cos(x[0] * x[1]),
               ca.sin(x[0] * x[1]) + 1)
cse_ctx = Ctx(); cse_ctx.cse = True
f_plain = ca.Function("f", [x], [y], ["x"], ["y"])
f_def = ca.Function("f", [x], [y], ["x"], ["y"], C.fn_opts())             # default: {}
f_cse = ca.Function("f", [x], [y], ["x"], ["y"], C.fn_opts(cse_ctx))      # CSE requested
g_cse = ca.Function("g", [x], [y], C.fn_opts(cse_ctx))      # signature without names
ok("default fn_opts() accepted with names", f_def.name_in() == ["x"] and f_def.name_out() == ["y"])
ok("fn_opts(cse) accepted with names", f_cse.name_in() == ["x"] and f_cse.name_out() == ["y"])
ok("fn_opts(cse) accepted without names", g_cse.n_in() == 1 and g_cse.n_out() == 1)
ok("default leaves the graph untouched", f_def.n_instructions() == f_plain.n_instructions())
ok("CSE reduces the instruction count", f_cse.n_instructions() < f_plain.n_instructions())
a = ca.DM([0.3, 0.7])
ok("CSE does not change values", float(ca.norm_inf(f_plain(a) - f_cse(a))) == 0.0)

# ---- 7. every ca.Function in the MLTP scripts gets fn_opts(ctx) ----------------------
print("call sites")
def function_calls(src):
    for mt in re.finditer(r"ca\.Function\(", src):
        depth, i = 1, mt.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        yield src[mt.start():i]
for fname, n_expected in (("MLTP.py", 4), ("MLTP_initial.py", 4), ("MLTP_paramOptim.py", 3)):
    calls = list(function_calls(open(os.path.join(HERE, fname)).read()))
    ok(f"{fname}: {n_expected} Functions, all built with fn_opts(ctx)",
       len(calls) == n_expected and all(c.rstrip().endswith("fn_opts(ctx))") for c in calls))

# ---- 8. guarded: a real JIT build, only if a working compiler is on PATH -------------
print("real JIT (guarded)")
if C.jit_available():
    jo = C.fn_opts(jit=True)
    if jo.get("jit"):
        fj = ca.Function("fj", [x], [y], ["x"], ["y"], jo)
        ok("JIT-built Function matches the plain one",
           float(ca.norm_inf(f_plain(a) - fj(a))) < 1e-12)
    else:
        print("  [SKIP] a compiler is on PATH but casadi cannot drive it (fallback worked)")
else:
    print("  [SKIP] no C compiler on PATH")

print("\nALL casadi_opts TESTS PASSED")
