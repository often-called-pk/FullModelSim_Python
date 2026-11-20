"""HSL integration tests (plain script, no pytest).

Sections 1-3 are casadi-free (resolution / path / opts-rewriting via an injected
fake probe). Section 1c-1d check the Linux scoped preload with tiny shared
objects built by gcc (SKIP when gcc is missing, and on Windows). Section 4 is a
guarded smoke that only runs if a real CoinHSL DLL resolves on this machine.
Section 5 solves toy NLPs through IPOPT (no HSL needed). Run from the repo root:

    venv\\Scripts\\python.exe tests\\test_hsl.py
"""
import ctypes
import os, shutil, subprocess, sys, tempfile, warnings
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)

import functions.hsl as H

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# One root for every temp folder of this file. cleanup() at the end removes it.
# ignore_cleanup_errors: Windows cannot delete a library that this process loaded.
_TMP = tempfile.TemporaryDirectory(prefix="hsl_", ignore_cleanup_errors=True)

# ---- 1. resolve_hsl_dir precedence: COINHSL_DIR > explicit > frozen root > None ----
# The seed folder is gone: no module default, so an unset env and no explicit folder resolve to None.
print("resolve_hsl_dir")
_saved_env = os.environ.pop("COINHSL_DIR", None)
try:
    real_a = tempfile.mkdtemp(prefix="hsl_a_", dir=_TMP.name)   # stands in for an explicit dir
    real_b = tempfile.mkdtemp(prefix="hsl_b_", dir=_TMP.name)   # stands in for COINHSL_DIR
    missing = os.path.join(real_a, "no_such_dir")
    ok("no module seed (H._DEFAULT_HSL_DIR is gone)", not hasattr(H, "_DEFAULT_HSL_DIR"))
    ok("env unset, explicit existing -> explicit",
       H.resolve_hsl_dir(real_a) == os.path.abspath(real_a))
    os.environ["COINHSL_DIR"] = real_b
    ok("env set -> env wins over explicit",
       H.resolve_hsl_dir(real_a) == os.path.abspath(real_b))
    del os.environ["COINHSL_DIR"]
    ok("env unset, no explicit folder -> None",
       H.resolve_hsl_dir() is None and H.resolve_hsl_dir(None) is None)
    ok("env unset, explicit folder missing -> None", H.resolve_hsl_dir(missing) is None)
finally:
    if _saved_env is not None:
        os.environ["COINHSL_DIR"] = _saved_env

# ---- 1a. a missing explicit folder: the warning names it -----------------------------
print("missing explicit folder")
_saved_env = os.environ.pop("COINHSL_DIR", None)
try:
    _gone = os.path.join(tempfile.mkdtemp(prefix="hsl_gone_", dir=_TMP.name), "coinhsl")
    H.resolve_hsl_dir(_gone)                      # records the folder that was asked for
    with warnings.catch_warnings(record=True) as _seen:
        warnings.simplefilter("always")
        _out = H.apply_linear_solver({}, linear_solver="ma57", hsl_dir=None,
                                     probe=lambda *_: True)
    _msgs = [str(w.message) for w in _seen]
    ok("missing explicit folder -> MUMPS fallback",
       _out["ipopt"]["linear_solver"] == "mumps" and "hsllib" not in _out["ipopt"])
    ok("the warning names the missing folder",
       any(_gone in m and "does not exist" in m for m in _msgs))
    # COINHSL_DIR wins and holds no library: the warning names it first, then the missing explicit folder
    _envdir = tempfile.mkdtemp(prefix="hsl_env_", dir=_TMP.name)
    os.environ["COINHSL_DIR"] = _envdir
    _used = H.resolve_hsl_dir(_gone)
    with warnings.catch_warnings(record=True) as _seen:
        warnings.simplefilter("always")
        H.apply_linear_solver({}, linear_solver="ma57", hsl_dir=_used, probe=lambda *_: True)
    del os.environ["COINHSL_DIR"]
    _m = str(_seen[-1].message)
    ok("COINHSL_DIR without a library: the warning names it before the missing explicit folder",
       _used == os.path.abspath(_envdir) and "has no library" in _m and "does not exist" in _m
       and -1 < _m.find(_used) < _m.find(_gone))
    H.resolve_hsl_dir(_used)
    with warnings.catch_warnings(record=True) as _seen:
        warnings.simplefilter("always")
        H.apply_linear_solver({}, linear_solver="ma57", hsl_dir=_used, probe=lambda *_: True)
    ok("no missing explicit folder: the warning names only the folder in use",
       _used in str(_seen[-1].message) and "does not exist" not in str(_seen[-1].message))
finally:
    if _saved_env is not None:
        os.environ["COINHSL_DIR"] = _saved_env

# ---- 1b. resolve_hsl_dir: frozen bundle folder (sys._MEIPASS/hsl) -----------
# In a PyInstaller-frozen exe the spec puts the Coin-HSL libraries into the `hsl`
# subfolder of the bundle root (sys._MEIPASS/hsl); resolve_hsl_dir must find that
# folder. A library at the bundle root itself is not a candidate. Only when frozen.
print("resolve_hsl_dir (frozen bundle)")
_saved_env = os.environ.pop("COINHSL_DIR", None)
_saved_frozen = getattr(sys, "frozen", None)
_saved_meipass = getattr(sys, "_MEIPASS", None)
try:
    meipass = tempfile.mkdtemp(prefix="hsl_meipass_", dir=_TMP.name)   # stands in for the bundle root
    bundle_hsl = os.path.join(meipass, "hsl")
    os.mkdir(bundle_hsl)                                 # the spec's hsl subfolder
    open(os.path.join(bundle_hsl, "libhsl.so"), "wb").close()
    open(os.path.join(meipass, "libhsl.so"), "wb").close()   # root copy: not a candidate
    bare = tempfile.mkdtemp(prefix="hsl_bare_", dir=_TMP.name)          # a frozen root without hsl/
    open(os.path.join(bare, "libhsl.so"), "wb").close()
    explicit = tempfile.mkdtemp(prefix="hsl_explicit_", dir=_TMP.name)
    sys._MEIPASS = meipass

    # attr present but not frozen -> _MEIPASS/hsl must be ignored
    if hasattr(sys, "frozen"):
        del sys.frozen
    ok("not frozen -> _MEIPASS/hsl ignored",
       H.resolve_hsl_dir() is None)

    # frozen -> the hsl subfolder resolves when env/explicit absent
    sys.frozen = True
    ok("frozen -> _MEIPASS/hsl resolves",
       H.resolve_hsl_dir() == os.path.abspath(bundle_hsl))
    ok("frozen -> hsllib_path finds the library in hsl/",
       H.hsllib_path(H.resolve_hsl_dir()) == os.path.join(bundle_hsl, "libhsl.so"))

    # frozen root WITHOUT an hsl/ folder -> no candidate, even with a library at the root
    sys._MEIPASS = bare
    ok("frozen root without hsl/ -> no candidate",
       H.resolve_hsl_dir() is None)
    sys._MEIPASS = meipass

    # explicit arg still wins over the bundle hsl/ folder
    ok("explicit wins over _MEIPASS/hsl",
       H.resolve_hsl_dir(explicit) == os.path.abspath(explicit))

    # a missing explicit folder falls through to the bundle hsl/ folder
    ok("missing explicit -> _MEIPASS/hsl",
       H.resolve_hsl_dir(os.path.join(explicit, "no_such_dir")) == os.path.abspath(bundle_hsl))

    # COINHSL_DIR env still wins over everything
    os.environ["COINHSL_DIR"] = explicit
    ok("env wins over _MEIPASS/hsl",
       H.resolve_hsl_dir() == os.path.abspath(explicit))
    ok("env wins over explicit",
       H.resolve_hsl_dir(bundle_hsl) == os.path.abspath(explicit))
    del os.environ["COINHSL_DIR"]
finally:
    if _saved_env is not None:
        os.environ["COINHSL_DIR"] = _saved_env
    if _saved_meipass is None:
        if hasattr(sys, "_MEIPASS"):
            del sys._MEIPASS
    else:
        sys._MEIPASS = _saved_meipass
    if _saved_frozen is None:
        if hasattr(sys, "frozen"):
            del sys.frozen
    else:
        sys.frozen = _saved_frozen

# ---- 1c. Linux scoped preload: the folder's own libraries, RTLD_LOCAL -------------
# Chain: libhsl -> liba -> libb -> libc3. Without the preload libhsl does not load (checked in a child
# process, so this process stays clean). With it libhsl loads, and no symbol of the folder reaches the
# global scope. A folder without libhsl loads nothing. An unreadable folder raises nothing.
print("linux preload (scoped)")
_path0 = os.environ.get("PATH")
if sys.platform.startswith("win"):
    print("  [SKIP] Linux preload checks: Windows (add_dll_directory path)")
else:
    _CC = shutil.which("gcc") or shutil.which("cc")
    _SYMS = ("c3_fn", "b_fn", "a_fn", "hsl_fn")

    def _maps_of(folder):
        """Basenames of the mapped files that live in `folder` (from /proc/self/maps)."""
        real = os.path.realpath(folder) + os.sep
        found = set()
        with open("/proc/self/maps") as fh:
            for ln in fh:
                parts = ln.split()
                if len(parts) >= 6 and parts[5].startswith(real):
                    found.add(os.path.basename(parts[5]))
        return found

    def _build_chain(folder, with_hsl=True):
        """Build libc3 -> libb -> liba (-> libhsl) as shared objects; sonames = file names."""
        src = {"libc3.so": ("c3_fn", "", "int c3_fn(void){return 3;}"),
               "libb.so": ("b_fn", "libc3.so", "int c3_fn(void);int b_fn(void){return c3_fn()+1;}"),
               "liba.so": ("a_fn", "libb.so", "int b_fn(void);int a_fn(void){return b_fn()+1;}"),
               "libhsl.so": ("hsl_fn", "liba.so", "int a_fn(void);int hsl_fn(void){return a_fn()+1;}")}
        names = [n for n in ("libc3.so", "libb.so", "liba.so", "libhsl.so") if with_hsl or n != "libhsl.so"]
        for name in names:
            _sym, dep, code = src[name]
            cfile = os.path.join(folder, name[:-3] + ".c")
            with open(cfile, "w") as fh:
                fh.write(code + "\n")
            cmd = [_CC, "-shared", "-fPIC", f"-Wl,-soname,{name}", "-o", os.path.join(folder, name), cfile]
            if dep:
                cmd += [f"-L{folder}", f"-l:{dep}"]
            subprocess.run(cmd, check=True, capture_output=True)

    if _CC is None:
        print("  [SKIP] chain checks: no gcc on PATH")
    else:
        _chain = tempfile.mkdtemp(prefix="hsl_chain_", dir=_TMP.name)
        _build_chain(_chain)
        # control: in a child process the chain does not load without the preload
        _probe = subprocess.run(
            [sys.executable, "-c", "import ctypes,sys; ctypes.CDLL(sys.argv[1])",
             os.path.join(_chain, "libhsl.so")], capture_output=True)
        ok("control: libhsl alone (child process) fails to load", _probe.returncode != 0)

        H.register_hsl_dll_dir(_chain)
        _mapped = _maps_of(_chain)
        ok("preload maps liba, libb and libc3 of the folder",
           {"liba.so", "libb.so", "libc3.so"} <= _mapped)
        _hsl = ctypes.CDLL(os.path.join(_chain, "libhsl.so"))   # loads after the preload
        _hsl.hsl_fn.restype = ctypes.c_int
        ok("libhsl loads after the preload and its chain resolves (hsl_fn() == 6)", _hsl.hsl_fn() == 6)
        ok("RTLD_LOCAL: no folder symbol in the global scope",
           not any(hasattr(ctypes.CDLL(None), s) for s in _SYMS))
        ok("Linux: PATH is not touched", os.environ.get("PATH") == _path0)

        # a folder without libhsl loads nothing
        _nolib = tempfile.mkdtemp(prefix="hsl_nolib_", dir=_TMP.name)
        _build_chain(_nolib, with_hsl=False)
        H.register_hsl_dll_dir(_nolib)
        ok("a folder without libhsl loads nothing", _maps_of(_nolib) == set())

# an unreadable folder raises nothing: the library is an invalid object, so IPOPT's own load fails,
# and the solve falls back to MUMPS (section 5 runs the solve). Runs on Linux only.
_unread = None
if not sys.platform.startswith("win"):
    _unread = tempfile.mkdtemp(prefix="hsl_unread_", dir=_TMP.name)
    with open(os.path.join(_unread, "libhsl.so"), "wb") as fh:
        fh.write(b"not an ELF object, so no load can succeed")
    os.chmod(_unread, 0o311)                   # search and write, no read: listing the folder fails
    try:
        H.register_hsl_dll_dir(_unread)
        ok("unreadable folder: register raises nothing", True)
    except Exception as exc:                   # the check reports the exception, it does not hide it
        ok(f"unreadable folder: register raises nothing (got {exc!r})", False)

# ---- 2. hsllib_path ----------------------------------------------------------
print("hsllib_path")
_d = tempfile.mkdtemp(prefix="hsl_lib_", dir=_TMP.name)
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

# (a) a missing folder + ma57 -> must NOT crash (key popped), must solve (mumps), must name the folder
_saved_env = os.environ.pop("COINHSL_DIR", None)
try:
    _gone5 = os.path.join(tempfile.mkdtemp(prefix="hsl_gone5_", dir=_TMP.name), "coinhsl")
    with warnings.catch_warnings(record=True) as _seen5:
        warnings.simplefilter("always")
        S = _make_solver(ca, _nlp, {"ipopt": {"linear_solver": "ma57", "print_level": 0,
                                               "max_iter": 50},
                                    "_hsl_dir": _gone5})
    S(x0=0, lbg=-10, ubg=10)
    ok("ma57 + missing dir -> falls back and solves",
       S.stats().get("return_status") == "Solve_Succeeded")
    ok("the warning names the missing folder",
       any(_gone5 in str(w.message) for w in _seen5))
finally:
    if _saved_env is not None:
        os.environ["COINHSL_DIR"] = _saved_env

# (a2) an unreadable folder + ma57 -> the real probe fails, MUMPS solves, no exception
if _unread is not None:
    _saved_env = os.environ.pop("COINHSL_DIR", None)
    try:
        S4 = _make_solver(ca, _nlp, {"ipopt": {"linear_solver": "ma57", "print_level": 0,
                                                "max_iter": 50},
                                     "_hsl_dir": _unread})
        S4(x0=0, lbg=-10, ubg=10)
        ok("unreadable folder + ma57 -> MUMPS fallback solves",
           S4.stats().get("return_status") == "Solve_Succeeded")
    except Exception as exc:
        ok(f"unreadable folder + ma57 -> MUMPS fallback solves (got {exc!r})", False)
    finally:
        os.chmod(_unread, 0o755)
        if _saved_env is not None:
            os.environ["COINHSL_DIR"] = _saved_env

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

c3 = Ctx(); userOpts(c3, circuit="Sturn", linear_solver="ma97", hsl_dir=_d)
ok("ma97 + hsl_dir threaded",
   c3.opts["ipopt"]["linear_solver"] == "ma97" and c3.opts["_hsl_dir"] == _d)

if _unread is not None:
    os.chmod(_unread, 0o700)                   # else cleanup cannot list the folder
_TMP.cleanup()
print("ALL HSL TESTS PASSED")
