"""Coin-HSL linear-solver support for IPOPT.

CasADi's bundled IPOPT already contains the HSL runtime loader; it loads an
external HSL shared library at *solve* time when `linear_solver` is an `ma*`
solver and the `hsllib` option points at the library. This module locates that
library, makes it (and its co-located dependency DLLs) findable on Windows,
loads its dependencies by hand on Linux when the library does not load alone,
probes once that IPOPT can actually use it, and rewrites the IPOPT options dict
to either use HSL or fall back to MUMPS.

Everything here is casadi-free except `probe_linear_solver`, which imports
casadi lazily so the rest of the module stays importable without it.
"""
import copy
import os
import re
import sys
import warnings

# Candidate library filenames, most-preferred first. `libhsl.dll` is the
# IPOPT-compatible shim shipped by CoinHSL_jll.
_LIB_NAMES = ("libhsl.dll", "libcoinhsl.dll",
              "libhsl.so", "libcoinhsl.so")

# HSL solvers we let reach the probe. `ma86` is deliberately excluded: it
# segfaulted in a loader probe, and a native segfault is NOT a catchable Python
# exception -- it would take down the whole process. An unrecognised `ma*` name
# (typo, or an unsupported solver) is downgraded to MUMPS rather than probed.
_SUPPORTED_HSL = ("ma27", "ma57", "ma97")

_registered = set()      # dirs already added to the DLL search path
_probe_cache = {}        # (solver_name, hsllib) -> bool
_bad_explicit = None     # the last explicit folder that was not a directory


def _frozen_bundle_dir():
    """The PyInstaller bundle root (sys._MEIPASS) when frozen, else None.

    The build spec drops the Coin-HSL DLLs at the bundle root (`"."`), so in a
    frozen exe that root is where the HSL library lives. Guarded on sys.frozen
    so a stray _MEIPASS attribute in a dev process is never picked up.
    """
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", None)
    return None


def resolve_hsl_dir(explicit=None):
    """Resolve the directory that should contain the HSL library.

    Precedence: COINHSL_DIR env var > `explicit` arg (GUI field or kwarg) >
    the frozen bundle root (PyInstaller sys._MEIPASS, where the spec drops the
    Coin-HSL DLLs). Returns the first candidate that is an existing directory,
    as an absolute path, or None if none exist.
    """
    global _bad_explicit
    _bad_explicit = explicit if explicit and not os.path.isdir(explicit) else None
    for cand in (os.environ.get("COINHSL_DIR"), explicit, _frozen_bundle_dir()):
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
    """Make the HSL library of `hsl_dir` loadable with its co-located
    dependencies. Idempotent; no-op if the dir is missing.

    Windows: os.add_dll_directory and a PATH prepend, so transitive
    dependencies resolve regardless of the loader's search mode.
    Linux: the loader does not read PATH, so `_preload_deps` loads the
    dependencies by hand.
    """
    if not hsl_dir or hsl_dir in _registered or not os.path.isdir(hsl_dir):
        return
    if sys.platform.startswith("win"):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(hsl_dir)
            except OSError:
                pass
        os.environ["PATH"] = hsl_dir + os.pathsep + os.environ.get("PATH", "")
    else:
        _preload_deps(hsl_dir)
    _registered.add(hsl_dir)


def _preload_deps(hsl_dir):
    """Linux: load the other shared libraries of the folder (libgfortran,
    metis, blas), so the loader finds them by name when IPOPT opens libhsl.

    It loads nothing when the folder has no HSL library, or when the library
    loads alone. The loads are local (RTLD_LOCAL): no symbol of these
    libraries replaces a symbol of the process. The loop stops when a pass
    loads no new file. A failure is harmless: IPOPT then reports the load
    error and the solve uses MUMPS.
    """
    lib = hsllib_path(hsl_dir)
    if not lib or _loads(lib):
        return
    try:
        names = sorted(os.listdir(hsl_dir))
    except OSError:
        return
    todo = [f for f in names
            if re.search(r"\.so(\.\d+)*$", f) and f != os.path.basename(lib)]
    while todo:
        left = [f for f in todo if not _loads(os.path.join(hsl_dir, f))]
        if len(left) == len(todo):
            break
        todo = left


def _loads(path):
    import ctypes
    try:
        ctypes.CDLL(path)               # default mode: RTLD_LOCAL
        return True
    except OSError:
        return False


def probe_linear_solver(solver_name, hsllib):
    """Return True iff IPOPT can actually load+use `solver_name` (with `hsllib`)
    in THIS process. Builds and solves a trivial NLP. Cached per (solver, lib).

    A failed HSL load makes IPOPT return status 'Invalid_Option'; a working one
    reaches 'Solve_Succeeded'. We treat only the latter as success.

    Also registers the HSL library's directory with the loader (on first call for
    a given library) so the library and its co-located dependency DLLs can load.
    """
    key = (solver_name, hsllib)
    if key in _probe_cache:
        return _probe_cache[key]
    ok = False
    try:
        import casadi as ca
        if hsllib:
            register_hsl_dll_dir(os.path.dirname(os.path.abspath(hsllib)))
        x = ca.MX.sym("x")
        nlp = {"x": x, "f": (x - 1) ** 2, "g": x}
        ip = {"linear_solver": solver_name, "print_level": 0, "max_iter": 20}
        if hsllib:
            ip["hsllib"] = hsllib
        S = ca.nlpsol("hsl_probe", "ipopt", nlp,
                      {"ipopt": ip, "print_time": False})
        S(x0=0, lbg=-10, ubg=10)
        ok = (S.stats().get("return_status") == "Solve_Succeeded")
    except Exception as exc:
        warnings.warn(
            f"HSL probe for linear_solver '{solver_name}' failed: {exc}",
            RuntimeWarning, stacklevel=2)
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

    if linear_solver not in _SUPPORTED_HSL:
        warnings.warn(
            f"Unsupported HSL linear_solver {linear_solver!r} "
            f"(supported: {_SUPPORTED_HSL}); falling back to MUMPS without "
            f"probing it.",
            RuntimeWarning, stacklevel=2)
        ip["linear_solver"] = "mumps"
        ip.pop("hsllib", None)
        return opts

    register_hsl_dll_dir(hsl_dir)
    lib = hsllib_path(hsl_dir)
    if lib and probe(linear_solver, lib):
        ip["linear_solver"] = linear_solver
        ip["hsllib"] = lib
        # MA57 does no internal scaling by default (unlike MUMPS, which scales
        # the augmented system automatically). On a badly-scaled NLP like the
        # 23-state MLTP this stalls IPOPT with oscillating dual infeasibility,
        # so MA57 can hit max_iter where MUMPS converges. Enabling MC64
        # auto-scaling restores convergence parity. Respects an explicit
        # caller-set value.
        if linear_solver == "ma57":
            ip.setdefault("ma57_automatic_scaling", "yes")
        return opts

    why = (f"the Coin-HSL folder {_bad_explicit} does not exist"
           if lib is None and _bad_explicit else f"hsl_dir={hsl_dir!r}, lib={lib!r}")
    warnings.warn(
        f"HSL linear_solver '{linear_solver}' unavailable "
        f"({why}); falling back to MUMPS.",
        RuntimeWarning, stacklevel=2)
    ip["linear_solver"] = "mumps"
    ip.pop("hsllib", None)
    return opts
