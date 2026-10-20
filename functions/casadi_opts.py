"""Shared options for every ``casadi.Function`` the MLTP scripts build.

``fn_opts(ctx)`` is the trailing options dict of each symbolic Function
(``f_dyn``, ``sf``, ``h_eq``, ``f_veh`` in MLTP / MLTP_initial / MLTP_paramOptim):

* ``cse`` is OPT-IN (``MLTP_CSE=1``, or ``userOpts(cse=True)`` -> ``ctx.cse``).
  casadi runs common-subexpression elimination while it builds the Function's
  algorithm, so the vehicle/tyre graph is stored with shared sub-expressions
  merged (~11% fewer instructions for f_dyn / h_eq). Function values, gradients
  and Jacobians are bit-identical; only the Lagrangian Hessian changes, by ~1
  ulp. The 23-state NLP is path-sensitive, so that is enough to change IPOPT's
  iterates (iteration count, local optimum, or even convergence), while
  function evaluation is only ~10% of the per-iteration cost (MA57
  factorisation dominates). Until the NLP is robust to such perturbations CSE
  therefore stays off by default.
* ``jit`` is opt-in (``userOpts(jit=True)`` -> ``ctx.jit``, ``fn_opts(jit=True)``,
  or the environment variable ``MLTP_JIT=1``): casadi generates C for the
  Function and compiles it with a system compiler (gcc, clang, or MSVC ``cl`` on
  PATH). If JIT is requested but no working compiler is found, a single
  RuntimeWarning is issued and the Function is built without JIT -- a solve never
  fails because of a missing or broken compiler (same policy as the HSL solver).

Everything here is casadi-free except the one-off compile probe, which imports
casadi lazily, so the module stays importable without it.
"""
import os
import shutil
import sys
import warnings

_COMPILERS = ("gcc", "clang", "cl")
_warned = False          # the "JIT unavailable" warning is issued once per process
_probe_cache = {}        # repr(jit_options) -> bool (can casadi really JIT with these?)


def jit_available():
    """True if a C compiler (gcc, clang or MSVC ``cl``) is on PATH."""
    return any(shutil.which(c) is not None for c in _COMPILERS)


def _env_jit():
    """True if the MLTP_JIT environment variable asks for JIT (``1``)."""
    return os.environ.get("MLTP_JIT", "").strip().lower() in ("1", "true", "yes", "on")


def _env_cse():
    """CSE is off unless MLTP_CSE is explicitly ``1`` (or true/yes/on)."""
    return os.environ.get("MLTP_CSE", "").strip().lower() in ("1", "true", "yes", "on")


def _jit_options():
    """``jit_options`` for casadi's 'shell' compiler plugin, matched to the compiler on PATH.

    The plugin's built-in command lines are fixed when the casadi wheel is built.
    On Windows they drive MSVC (``cl.exe``), so a machine that only has a
    gcc/clang (e.g. MinGW) needs the GNU-style command lines spelled out;
    elsewhere the defaults are already GNU-style (``-O3``, as wanted).
    """
    if sys.platform.startswith("win"):
        if shutil.which("cl") is not None:
            return {"flags": ["/O2"], "verbose": False}          # cl rejects -O3
        cc = "gcc" if shutil.which("gcc") is not None else "clang"
        return {"flags": ["-O3"], "verbose": False,
                "compiler": cc, "linker": cc,
                "compiler_setup": "-c", "linker_setup": "-shared",
                "compiler_output_flag": "-o ", "linker_output_flag": "-o "}
    return {"flags": ["-O3"], "verbose": False}


def _probe_jit(jit_options):
    """True iff casadi can JIT-compile and run a trivial Function with ``jit_options``
    in THIS process. Cached per option set. A compiler on PATH is not enough -- it
    can be the wrong flavour for the plugin or lack headers/linker -- and a failed
    JIT build raises from ``ca.Function(...)``, which would kill the solve."""
    key = repr(jit_options)
    if key not in _probe_cache:
        try:
            import casadi as ca
            x = ca.SX.sym("x")
            f = ca.Function("jit_probe", [x], [x * x + 1],
                            {"jit": True, "compiler": "shell", "jit_options": dict(jit_options)})
            ok = abs(float(f(3.0)) - 10.0) < 1e-9
        except Exception:
            ok = False
        _probe_cache[key] = ok
    return _probe_cache[key]


def fn_opts(ctx=None, jit=None):
    """Options dict to pass as the trailing argument of ``ca.Function(...)``.

    ``{}`` by default. ``{"cse": True}`` when CSE is requested -- ``MLTP_CSE=1``
    or a truthy ``ctx.cse`` (set by ``userOpts(cse=True)``). JIT is added when
    requested -- ``jit is True``, a truthy ``ctx.jit`` (set by
    ``userOpts(jit=True)``), or ``MLTP_JIT=1`` -- and a working compiler exists;
    otherwise a single RuntimeWarning is issued and JIT stays off.
    """
    global _warned
    opts = {"cse": True} if (getattr(ctx, "cse", False) or _env_cse()) else {}
    if not (jit is True or getattr(ctx, "jit", False) or _env_jit()):
        return opts

    reason = None
    jit_options = None
    if not jit_available():
        reason = f"no C compiler ({', '.join(_COMPILERS)}) found on PATH"
    else:
        jit_options = _jit_options()
        if not _probe_jit(jit_options):
            reason = "a test JIT build with the compiler on PATH failed"
            jit_options = None
    if jit_options is None:
        if not _warned:
            _warned = True
            warnings.warn(f"JIT requested but {reason}; building casadi Functions "
                          "without JIT.", RuntimeWarning, stacklevel=2)
        return opts

    opts.update({"jit": True, "compiler": "shell", "jit_options": jit_options})
    return opts
