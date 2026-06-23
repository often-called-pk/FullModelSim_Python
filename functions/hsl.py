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

    warnings.warn(
        f"HSL linear_solver '{linear_solver}' unavailable "
        f"(hsl_dir={hsl_dir!r}, lib={lib!r}); falling back to MUMPS.",
        RuntimeWarning, stacklevel=2)
    ip["linear_solver"] = "mumps"
    ip.pop("hsllib", None)
    return opts
