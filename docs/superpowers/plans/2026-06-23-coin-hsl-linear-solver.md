# Coin-HSL Linear Solver Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make IPOPT's linear solver configurable (default `ma57`) using the user's prebuilt Coin-HSL DLL, with a one-time pre-flight probe that transparently falls back to MUMPS if HSL is unavailable, plus a benchmark proving the speed-up on the 23-state MLTP.

**Architecture:** All solves funnel through one chokepoint — `userOpts` builds `ctx.opts` → `MLTP*`/wrappers → `build_and_solve_nlp(..., ctx.opts)` → `_make_solver`. A new isolated helper `functions/hsl.py` owns DLL-dir registration, path resolution, the probe, and opts-rewriting. `userOpts` only carries user *intent* (`linear_solver`, optional `hsl_dir`); `_make_solver` resolves, probes, and either wires HSL (`hsllib` + `linear_solver=ma*`) or downgrades to MUMPS.

**Tech Stack:** Python 3.x, CasADi 3.7.2 (bundled IPOPT with HSL runtime loader), the Julia `CoinHSL_jll` MinGW/libgfortran5 DLL, NumPy. No new pip dependencies. No compilation.

## Global Constraints

- **No new pip dependencies and no IPOPT/CasADi/HSL rebuild.** Mechanism proven against CasADi 3.7.2; IPOPT's bundled `IpLibraryLoader` loads an external HSL DLL at solve time.
- **CasADi pin:** the installed wheel is `casadi==3.7.2`; code must work there (and `>=3.6` per `requirements.txt`).
- **Windows DLL loading recipe (proven):** `os.add_dll_directory(<bin>)` **and** prepend `<bin>` to `os.environ["PATH"]`, then set IPOPT option `hsllib` to the full path of `libhsl.dll`. The user's DLL bin is self-contained (its own OpenBLAS/METIS/gfortran runtime).
- **Default HSL location:** seeded fallback `D:\Software Installs\CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin`; env var `COINHSL_DIR` overrides it. Never *require* a machine-specific path in committed code — it is a fallback only.
- **Probe success criterion:** a trivial NLP solve returns `return_status == "Solve_Succeeded"`. A failed HSL load returns `"Invalid_Option"` (verified empirically), which the probe treats as failure → fall back to MUMPS.
- **Tests are plain scripts** (no pytest/unittest): module-top-level asserts via an `ok(name, cond)` helper that prints `[PASS]`/`[FAIL]`. Run each file directly with the in-repo venv **from the repo root** (paths are relative): `venv\Scripts\python.exe test_hsl.py`.
- **Excluded from defaults:** `ma86` segfaulted in a no-DLL loader probe; only `ma57`/`ma97`/`ma27`/`mumps` are offered. Default is `ma57`.
- **Preserve the existing safety guarantee:** a solve must never hard-crash on a missing/incompatible HSL DLL.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `functions/hsl.py` | DLL-dir registration, path resolution, pre-flight probe, opts-rewriting with MUMPS fallback | **Create** |
| `test_hsl.py` | Plain-script tests: casadi-free resolution/rewriting + guarded real-DLL/solver checks | **Create** |
| `functions/transcription.py` | Rewrite `_make_solver` (lines 326–342) to consume `_hsl_dir` and delegate to `functions/hsl.apply_linear_solver` | **Modify** |
| `userOpts.py` | Add `linear_solver` / `hsl_dir` kwargs; set `ctx.opts["ipopt"]["linear_solver"]` + `ctx.opts["_hsl_dir"]`; refresh comment | **Modify** |
| `MLTP.py` | Stash `ctx.solve_stats` + `ctx.elapsed` so the benchmark can read iteration count / timing split | **Modify** |
| `bench_linear_solver.py` | Solve the 23-state MLTP across solvers; tabulate iters / solve-time / linear-vs-eval split | **Create** |
| `requirements.txt`, `CLAUDE.md` | Doc updates: enable-HSL note, `COINHSL_DIR`, `_make_solver` gotcha change | **Modify** |

**Task ordering rationale (do not reorder Tasks 2 and 3):** `_make_solver` must learn to *pop* the new top-level `_hsl_dir` key **before** `userOpts` starts *emitting* it. If `userOpts` emitted `_hsl_dir` while the old `_make_solver` still passed it straight to `ca.nlpsol`, CasADi would reject the unknown option and every solve would break. So: Task 2 rewrites the consumer (safe no-op while `userOpts` still requests `mumps`), Task 3 flips the producer.

---

### Task 1: `functions/hsl.py` helper + casadi-free tests

**Files:**
- Create: `functions/hsl.py`
- Test: `test_hsl.py`

**Interfaces:**
- Consumes: nothing (new leaf module; `casadi` imported lazily, only inside the probe).
- Produces:
  - `resolve_hsl_dir(explicit: str | None = None) -> str | None`
  - `hsllib_path(hsl_dir: str | None) -> str | None`
  - `register_hsl_dll_dir(hsl_dir: str | None) -> None`
  - `probe_linear_solver(solver_name: str, hsllib: str | None) -> bool` (cached per `(solver_name, hsllib)`)
  - `apply_linear_solver(opts: dict, *, linear_solver: str, hsl_dir: str | None, probe=probe_linear_solver) -> dict`
  - module constant `_DEFAULT_HSL_DIR: str` (overridable by tests)

- [ ] **Step 1: Write the failing test file `test_hsl.py`**

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv\Scripts\python.exe test_hsl.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'functions.hsl'` (the module does not exist yet).

- [ ] **Step 3: Implement `functions/hsl.py`**

```python
"""Coin-HSL linear-solver support for IPOPT.

CasADi's bundled IPOPT already contains the HSL runtime loader; it loads an
external HSL shared library at *solve* time when `linear_solver` is an `ma*`
solver and the `hsllib` option points at the library. This module locates that
library, makes it (and its co-located dependency DLLs) findable on Windows,
probes once that IPOPT can actually use it, and rewrites the IPOPT options dict
to either use HSL or fall back to MUMPS.

Everything here is casadi-free except `probe_linear_solver`, which imports
casadi lazily so the rest of the module stays importable without it.
"""
import copy
import os
import sys
import warnings

# Seeded fallback only -- the COINHSL_DIR env var overrides it. This is the
# user's CoinHSL_jll (MinGW / libgfortran5) bin folder; nothing requires it.
_DEFAULT_HSL_DIR = r"D:\Software Installs\CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin"

# Candidate library filenames, most-preferred first. `libhsl.dll` is the
# IPOPT-compatible shim shipped by CoinHSL_jll.
_LIB_NAMES = ("libhsl.dll", "libcoinhsl.dll",
              "libhsl.so", "libcoinhsl.so",
              "libhsl.dylib", "libcoinhsl.dylib")

_registered = set()      # dirs already added to the DLL search path
_probe_cache = {}        # (solver_name, hsllib) -> bool


def resolve_hsl_dir(explicit=None):
    """Resolve the directory that should contain the HSL library.

    Precedence: COINHSL_DIR env var > `explicit` arg > seeded default. Returns
    the first candidate that is an existing directory, as an absolute path, or
    None if none exist.
    """
    for cand in (os.environ.get("COINHSL_DIR"), explicit, _DEFAULT_HSL_DIR):
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


def hsllib_path(hsl_dir):
    """Full path to the HSL library inside `hsl_dir`, or None if not found."""
    if not hsl_dir or not os.path.isdir(hsl_dir):
        return None
    for name in _LIB_NAMES:
        p = os.path.join(hsl_dir, name)
        if os.path.isfile(p):
            return p
    return None


def register_hsl_dll_dir(hsl_dir):
    """Make `hsl_dir` (and its co-located dependency DLLs) findable by the
    Windows loader when IPOPT LoadLibrary's the HSL library. Idempotent; no-op
    if the dir is missing. Uses both os.add_dll_directory and a PATH prepend so
    transitive dependencies resolve regardless of the loader's search mode.
    """
    if not hsl_dir or hsl_dir in _registered or not os.path.isdir(hsl_dir):
        return
    if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(hsl_dir)
        except OSError:
            pass
    os.environ["PATH"] = hsl_dir + os.pathsep + os.environ.get("PATH", "")
    _registered.add(hsl_dir)


def probe_linear_solver(solver_name, hsllib):
    """Return True iff IPOPT can actually load+use `solver_name` (with `hsllib`)
    in THIS process. Builds and solves a trivial NLP. Cached per (solver, lib).

    A failed HSL load makes IPOPT return status 'Invalid_Option'; a working one
    reaches 'Solve_Succeeded'. We treat only the latter as success.
    """
    key = (solver_name, hsllib)
    if key in _probe_cache:
        return _probe_cache[key]
    ok = False
    try:
        import casadi as ca
        x = ca.MX.sym("x")
        nlp = {"x": x, "f": (x - 1) ** 2, "g": x}
        ip = {"linear_solver": solver_name, "print_level": 0, "max_iter": 20}
        if hsllib:
            ip["hsllib"] = hsllib
        S = ca.nlpsol("hsl_probe", "ipopt", nlp,
                      {"ipopt": ip, "print_time": False})
        S(x0=0, lbg=-10, ubg=10)
        ok = (S.stats().get("return_status") == "Solve_Succeeded")
    except Exception:
        ok = False
    _probe_cache[key] = ok
    return ok


def apply_linear_solver(opts, *, linear_solver, hsl_dir, probe=probe_linear_solver):
    """Return a copy of `opts` with the IPOPT linear solver wired.

    For a non-HSL solver (e.g. 'mumps'), pass it through. For an `ma*` solver,
    register the DLL dir, locate the library, and probe it; on success set
    `linear_solver` + `hsllib`, otherwise warn and fall back to 'mumps'.
    """
    opts = copy.deepcopy(opts)
    ip = opts.setdefault("ipopt", {})

    if not str(linear_solver).startswith("ma"):
        ip["linear_solver"] = linear_solver
        ip.pop("hsllib", None)
        return opts

    register_hsl_dll_dir(hsl_dir)
    lib = hsllib_path(hsl_dir)
    if lib and probe(linear_solver, lib):
        ip["linear_solver"] = linear_solver
        ip["hsllib"] = lib
        return opts

    warnings.warn(
        f"HSL linear_solver '{linear_solver}' unavailable "
        f"(hsl_dir={hsl_dir!r}, lib={lib!r}); falling back to MUMPS.",
        RuntimeWarning, stacklevel=2)
    ip["linear_solver"] = "mumps"
    ip.pop("hsllib", None)
    return opts
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv\Scripts\python.exe test_hsl.py`
Expected: every line `[PASS]`, the real-probe smoke either `[PASS] real ma57 probes True` (DLL present on this machine) or `[SKIP]`, and a final `ALL HSL TESTS PASSED`. Exit code 0.

- [ ] **Step 5: Commit**

```bash
git add functions/hsl.py test_hsl.py
git commit -m "feat(hsl): add Coin-HSL resolver, DLL registration, probe, opts-rewriting"
```

---

### Task 2: Rewrite `_make_solver` to consume `_hsl_dir` and delegate to HSL helper

**Files:**
- Modify: `functions/transcription.py:326-342` (the `_make_solver` function)
- Modify: `test_hsl.py` (append Section 5: `_make_solver` behavior)
- Modify: `CLAUDE.md` (the `_make_solver` gotcha bullet)

**Interfaces:**
- Consumes: `functions.hsl.apply_linear_solver`, `functions.hsl.resolve_hsl_dir` (Task 1).
- Produces: unchanged signature `_make_solver(ca, nlp, opts) -> ca.Function`. New contract: pops a top-level `opts["_hsl_dir"]` hint (if present) before constructing the solver, and applies HSL/fallback when `opts["ipopt"]["linear_solver"]` starts with `ma`.

This task is a safe no-op on real solves until Task 3, because `userOpts` still sets `linear_solver="mumps"` and emits no `_hsl_dir` yet.

- [ ] **Step 1: Append the failing Section 5 to `test_hsl.py`** (before the final `print("ALL HSL TESTS PASSED")` line)

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv\Scripts\python.exe test_hsl.py`
Expected: FAIL during Section 5 — the **old** `_make_solver` passes the unknown top-level option `_hsl_dir` straight to `ca.nlpsol`, raising a CasADi error like `Unrecognized option: _hsl_dir` (or the `S(...)` call errors). This proves the new contract is needed.

- [ ] **Step 3: Replace the body of `_make_solver` in `functions/transcription.py`**

Replace lines 326–342 (the current `_make_solver` and its commented-out JIT variant) with:

```python
def _make_solver(ca, nlp, opts):
    """Create the IPOPT solver with a configurable linear solver.

    If an HSL solver (ma*) is requested, register the Coin-HSL DLL directory,
    probe once that IPOPT can load it, and use it via the `hsllib` option; if
    HSL is unavailable or fails to load, transparently fall back to MUMPS
    (bundled in the casadi wheel) so a solve never crashes on a missing or
    incompatible HSL DLL. The HSL directory is taken from a private top-level
    `opts["_hsl_dir"]` hint (set by userOpts) resolved against COINHSL_DIR and
    a seeded default; the hint is always stripped before reaching CasADi.
    """
    import copy
    from functions.hsl import apply_linear_solver, resolve_hsl_dir

    opts = copy.deepcopy(opts)
    hsl_dir = resolve_hsl_dir(explicit=opts.pop("_hsl_dir", None))
    ip = opts.setdefault("ipopt", {})
    linear_solver = ip.get("linear_solver", "mumps")
    if str(linear_solver).startswith("ma"):
        opts = apply_linear_solver(opts, linear_solver=linear_solver,
                                   hsl_dir=hsl_dir)
    return ca.nlpsol("solver", "ipopt", nlp, opts)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv\Scripts\python.exe test_hsl.py`
Expected: all `[PASS]` (Section 5 included), `[SKIP]` only where no DLL resolves, final `ALL HSL TESTS PASSED`, exit 0.

- [ ] **Step 5: Update the `_make_solver` gotcha bullet in `CLAUDE.md`**

Find this bullet under "The transcription engine":

```
`_make_solver()` **forces `linear_solver='mumps'` and strips any HSL config**
(so requesting `ma57` silently downgrades). After the solve: `unpack_solution`,
```

Replace the first sentence so it reads:

```
`_make_solver()` **selects the configured `linear_solver`**: it uses HSL
(`ma*`) when a working Coin-HSL DLL is found (via the `COINHSL_DIR` env var or a
seeded default, registered on the Windows DLL path and probed once), otherwise
it transparently falls back to `mumps` — so a solve never crashes on a missing
HSL DLL. Default is `ma57`. After the solve: `unpack_solution`,
```

- [ ] **Step 6: Commit**

```bash
git add functions/transcription.py test_hsl.py CLAUDE.md
git commit -m "feat(solver): _make_solver uses HSL when available, falls back to MUMPS"
```

---

### Task 3: Make `linear_solver` / `hsl_dir` configurable from `userOpts`

**Files:**
- Modify: `userOpts.py` (signature at lines 181–189; IPOPT options dict at 244–263)
- Modify: `test_hsl.py` (append Section 6: userOpts wiring)

**Interfaces:**
- Consumes: nothing from `functions.hsl` directly — `userOpts` only writes dict keys that Task 2's `_make_solver` consumes.
- Produces: `userOpts(..., linear_solver="ma57", hsl_dir=None)` sets `ctx.opts["ipopt"]["linear_solver"] = linear_solver` and `ctx.opts["_hsl_dir"] = hsl_dir`. Reachable through `MLTP(..., linear_solver=..., hsl_dir=...)` because `MLTP` forwards `**useropts_kwargs` to `userOpts` (`MLTP.py:144-145`).

- [ ] **Step 1: Append the failing Section 6 to `test_hsl.py`** (before the final `print("ALL HSL TESTS PASSED")`)

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv\Scripts\python.exe test_hsl.py`
Expected: FAIL in Section 6 — `userOpts()` does not yet accept `linear_solver`/`hsl_dir` (`TypeError: userOpts() got an unexpected keyword argument 'linear_solver'`), and the default `linear_solver` is still `mumps`.

- [ ] **Step 3a: Add the kwargs to the `userOpts` signature**

In `userOpts.py`, change the signature (lines 181–189) from:

```python
def userOpts(ctx,
             AeroConfig="Static",          # 'Static' | 'Active_RW' | 'Active' | 'AALB'
             ATD="On",                     # 'On' | 'Off'
             Electric_4Motors="Off",       # 'On' | 'Off'
             circuit="BCN",
             vi=60.0,                      # initial velocity [m/s]
             ni=np.nan,                    # initial lateral position [m]
             circuits_dir="Circuits",
             data_dir="Data"):
```

to:

```python
def userOpts(ctx,
             AeroConfig="Static",          # 'Static' | 'Active_RW' | 'Active' | 'AALB'
             ATD="On",                     # 'On' | 'Off'
             Electric_4Motors="Off",       # 'On' | 'Off'
             circuit="BCN",
             vi=60.0,                      # initial velocity [m/s]
             ni=np.nan,                    # initial lateral position [m]
             circuits_dir="Circuits",
             data_dir="Data",
             linear_solver="ma57",         # 'ma57'|'ma97'|'ma27'|'mumps'; ma* uses Coin-HSL
             hsl_dir=None):                # Coin-HSL bin dir; None -> COINHSL_DIR env / default
```

- [ ] **Step 3b: Wire the options dict**

In `userOpts.py`, replace the IPOPT comment block + the `linear_solver` line (lines 244–263). Change the comment (244–247) from:

```python
    # ---- solver options (IPOPT) -------------------------------------------
    # MUMPS is the linear solver: it ships inside the casadi wheel and needs no
    # external library. (HSL/MA57 would be faster but requires a licensed HSL
    # DLL that IPOPT loads at solve time; not used here.)
```

to:

```python
    # ---- solver options (IPOPT) -------------------------------------------
    # `linear_solver` is configurable (default 'ma57'). An ma* solver uses
    # Coin-HSL, which IPOPT loads at solve time from `hsl_dir` (resolved against
    # COINHSL_DIR / a seeded default in functions/hsl.py); _make_solver probes it
    # and transparently falls back to MUMPS if it is unavailable. 'mumps' (the
    # casadi-bundled solver) is always available and needs no external library.
```

Then change the `linear_solver` line (260) from:

```python
        "linear_solver": "mumps",
```

to:

```python
        "linear_solver": linear_solver,
```

And immediately after `ctx.opts = {"ipopt": ipopt}` (line 263), add the hint:

```python
    ctx.opts = {"ipopt": ipopt}
    ctx.opts["_hsl_dir"] = hsl_dir          # consumed + stripped by _make_solver
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv\Scripts\python.exe test_hsl.py`
Expected: all `[PASS]` including Section 6; final `ALL HSL TESTS PASSED`; exit 0.

- [ ] **Step 5: Confirm existing tests still pass**

Run: `venv\Scripts\python.exe test_params_useropts.py`
Expected: the existing PASS output is unchanged (this file calls `userOpts`; the added kwargs are optional and must not alter its assertions).

- [ ] **Step 6: Commit**

```bash
git add userOpts.py test_hsl.py
git commit -m "feat(userOpts): expose linear_solver/hsl_dir, default ma57 via Coin-HSL"
```

---

### Task 4: `bench_linear_solver.py` — profile the 23-state solve across solvers

**Files:**
- Modify: `MLTP.py` (stash `ctx.solve_stats` after the solve; `ctx.elapsed` before return)
- Create: `bench_linear_solver.py` (repo root)

**Interfaces:**
- Consumes: `MLTP.MLTP(...)` returning `ctx` with new `ctx.solve_stats` (CasADi `solver.stats()` dict) and `ctx.elapsed` (`{"init": float, "solve": float}`).
- Produces: a console table comparing solvers, plus a returned list of per-solver result dicts.

**Why the MLTP hook:** `MLTP` currently discards the solver object (`res["solver"]`, `MLTP.py:185-189`). The benchmark needs IPOPT's `iter_count`, `return_status`, and per-callback wall times to compute the linear-solver-vs-function-eval split; these live in `solver.stats()`. Two lines expose them.

- [ ] **Step 1: Stash solve stats in `MLTP.py`**

After `sol = res["sol"]` (`MLTP.py:189`), add:

```python
    sol = res["sol"]
    ctx.solve_stats = res["solver"].stats()
```

Then, just before the final `return ctx` (`MLTP.py:281`), add:

```python
    ctx.elapsed = elapsed
    return ctx
```

- [ ] **Step 2: Write `bench_linear_solver.py`**

```python
"""Benchmark IPOPT linear solvers on the full 23-state MLTP.

Solves the same OCP once per linear solver and reports iteration count, the
isolated 23-state solve time (excludes the 7-state warm start), and the split
between NLP function-evaluation time and IPOPT-internal time (dominated by the
linear-solver factorisations). Confirms the linear solver is the bottleneck and
quantifies the Coin-HSL speed-up over MUMPS.

Run from the repo root:

    venv\\Scripts\\python.exe bench_linear_solver.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from MLTP import MLTP

# NLP callback timing keys reported by CasADi's ipopt stats.
_CB_KEYS = ("t_wall_nlp_f", "t_wall_nlp_g", "t_wall_nlp_grad_f",
            "t_wall_nlp_jac_g", "t_wall_nlp_hess_l")


def _row(solver, ctx):
    st = getattr(ctx, "solve_stats", {}) or {}
    total = float(st.get("t_wall_total", float("nan")))
    func = sum(float(st.get(k, 0.0)) for k in _CB_KEYS)
    return {
        "solver": solver,
        "status": st.get("return_status", "?"),
        "iters": st.get("iter_count", "?"),
        "solve_s": float(ctx.elapsed.get("solve", float("nan"))),
        "func_s": func,
        "ipopt_s": (total - func) if total == total else float("nan"),  # NaN-safe
        "lap_s": float(ctx.data.lap_time),
    }


def bench(circuit="Sturn", solvers=("mumps", "ma57", "ma97"), **mltp_kwargs):
    results = []
    for ls in solvers:
        print(f"\n===== linear_solver = {ls} =====")
        try:
            ctx = MLTP(circuit=circuit, linear_solver=ls,
                       save=False, plot=False, **mltp_kwargs)
            results.append(_row(ls, ctx))
        except Exception as exc:
            print(f"  {ls}: FAILED -> {exc}")
            results.append({"solver": ls, "status": f"EXC:{exc}", "iters": "-",
                            "solve_s": float("nan"), "func_s": float("nan"),
                            "ipopt_s": float("nan"), "lap_s": float("nan")})

    print("\n" + "=" * 78)
    print(f"{'solver':8} {'status':18} {'iters':>6} {'solve_s':>9} "
          f"{'func_s':>9} {'ipopt_s':>9} {'lap_s':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['solver']:8} {str(r['status'])[:18]:18} {str(r['iters']):>6} "
              f"{r['solve_s']:>9.2f} {r['func_s']:>9.2f} {r['ipopt_s']:>9.2f} "
              f"{r['lap_s']:>8.3f}")
    print("=" * 78)

    base = next((r for r in results if r["solver"] == "mumps"
                 and r["solve_s"] == r["solve_s"]), None)
    if base:
        for r in results:
            if r["solver"] != "mumps" and r["solve_s"] == r["solve_s"]:
                spd = base["solve_s"] / r["solve_s"] if r["solve_s"] else float("nan")
                print(f"  {r['solver']} speed-up vs mumps: {spd:.2f}x")
    return results


if __name__ == "__main__":
    bench(circuit="Sturn", solvers=("mumps", "ma57"))
```

- [ ] **Step 3: Run the benchmark (integration proof)**

Run: `venv\Scripts\python.exe bench_linear_solver.py`
Expected (timings are machine-dependent; shape and direction are what matter): both solvers reach `status = Solve_Succeeded` with the **same `lap_s`** (identical optimum), a printed table with columns `solver status iters solve_s func_s ipopt_s lap_s`, and a final `ma57 speed-up vs mumps: <N>x` line with N > 1 (ma57's `ipopt_s` should drop noticeably vs mumps). If `ma57` shows `status` reflecting a MUMPS fallback warning instead, that means no DLL resolved — set `COINHSL_DIR` to the bin folder and re-run.

- [ ] **Step 4: Commit**

```bash
git add MLTP.py bench_linear_solver.py
git commit -m "feat(bench): add 23-state linear-solver benchmark; expose solve stats on ctx"
```

---

### Task 5: Documentation — enable-HSL note and `COINHSL_DIR`

**Files:**
- Modify: `requirements.txt` (Linear solver comment block, lines 12–21)
- Modify: `CLAUDE.md` (add a short "Using Coin-HSL" note near the Environment & commands section)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the `requirements.txt` Linear-solver note**

Replace the block (lines 12–21) with:

```
# ---------------------------------------------------------------------------
# Linear solver
# ---------------------------------------------------------------------------
# IPOPT ships *inside* the casadi wheel. It defaults here to the HSL solver
# MA57 (much faster than MUMPS on the 23-state MLTP). MA57/MA97/MA27 require a
# Coin-HSL shared library (licensed, free for academic use); IPOPT loads it at
# solve time. Point the framework at it by setting the COINHSL_DIR environment
# variable to the folder containing libhsl.dll / libcoinhsl.dll, e.g.:
#     setx COINHSL_DIR "D:\path\to\CoinHSL\bin"     (Windows, new shells)
# If no Coin-HSL library is found, the solve transparently falls back to MUMPS
# (bundled in the casadi wheel, needs no external library). To force MUMPS, call
# with linear_solver="mumps" (e.g. MLTP(..., linear_solver="mumps")).
```

- [ ] **Step 2: Add a "Using Coin-HSL" note to `CLAUDE.md`**

In `CLAUDE.md`, directly after the "Running a solve." paragraph block under "## Environment & commands", insert:

```markdown
**Using Coin-HSL (faster linear solver).** The solve defaults to IPOPT's HSL
`ma57` solver, which is much faster than MUMPS on the 23-state problem. Provide a
Coin-HSL library (the MinGW/`libgfortran5` `CoinHSL_jll` build matches the casadi
wheel's ABI) by setting `COINHSL_DIR` to its `bin/` folder, or rely on the seeded
default in `functions/hsl.py`. The directory is registered on the Windows DLL
path and probed once; if HSL can't load, `_make_solver` falls back to MUMPS with a
warning. Choose the solver per call: `MLTP(circuit='BCN', linear_solver='ma97')`
or `linear_solver='mumps'`. Benchmark all three with `python bench_linear_solver.py`.
```

- [ ] **Step 3: Verify the docs render and nothing else changed**

Run: `git diff --stat`
Expected: only `requirements.txt` and `CLAUDE.md` modified.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt CLAUDE.md
git commit -m "docs: document Coin-HSL setup (COINHSL_DIR) and MUMPS fallback"
```

---

## Self-Review

**Spec coverage** (each spec section → task):
- A1 env-var-first resolver → Task 1 `resolve_hsl_dir` (COINHSL_DIR > explicit > default > None).
- B1 pre-flight probe + MUMPS fallback → Task 1 `probe_linear_solver`/`apply_linear_solver`, exercised in Task 2 `_make_solver`.
- Component 1 `functions/hsl.py` → Task 1.
- Component 2 `userOpts` kwargs + comment + `_hsl_dir` hint → Task 3.
- Component 3 `_make_solver` rewrite (single chokepoint) → Task 2.
- Component 4 `bench_linear_solver.py` (iters / wall / linear-vs-eval split) → Task 4 (+ MLTP stats hook).
- Component 5 tests + docs → `test_hsl.py` across Tasks 1–3; docs in Tasks 2 (CLAUDE.md gotcha), 5 (requirements.txt, CLAUDE.md note).
- Acceptance criteria (ma57 solves 23-state; missing DLL falls back without crashing; bench reports the split; solver selectable; `test_hsl.py` passes; existing tests unaffected) → Tasks 2–4 tests + Task 3 Step 5.
- Risk register items (loader present, ABI match, no committed required path, no mid-solve crash, ma86 excluded, idempotent register/probe) → all encoded in Task 1 helper + Global Constraints.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step states the exact command and expected outcome.

**Type/name consistency:** `resolve_hsl_dir`, `hsllib_path`, `register_hsl_dll_dir`, `probe_linear_solver`, `apply_linear_solver`, `_DEFAULT_HSL_DIR`, the `_hsl_dir` opts key, `ctx.solve_stats`, and `ctx.elapsed` are used identically across Tasks 1–4. `apply_linear_solver`'s injected `probe` parameter matches `probe_linear_solver`'s `(solver_name, hsllib) -> bool` signature.

**Known limitation (acceptable, documented):** `MLTP` does not forward `linear_solver`/`hsl_dir` into its internal `MLTP_initial(...)` warm-start call (`MLTP.py:150-151`); the 7-state init therefore uses the `userOpts` default (`ma57`). Since the default is HSL, the init also benefits; only an explicit non-default `linear_solver` on `MLTP(...)` would not propagate to the init solve. Out of scope for this plan.
