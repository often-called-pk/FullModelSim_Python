"""B5: MLTP(continue_from='auto') (plain script, no pytest). Run from anywhere:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_continue_from.py

continue_from='auto' starts one off-default solve from the stored default result of the same
circuit, config, tyre set, physics, mesh request and start state. The solve takes the primal and
the duals of that result (the primal only with warm_start_duals=False). Every other case is refused. There is no fallback to a primal-only seed,
to a cold start or to the default of another start state.

  1. functions.warmstart.default_source: a missing file, knobs in data.provenance.setup, an
     unconverged solve, a result without provenance (saved before E2, trusted by its name), a
     default provenance, an acceptable solve
  2. MLTP argument errors and a missing default (physics corrected too), raised before any NLP is built
  3. MLTP through a fake IPOPT (needs casadi and Data/DATA_AA.mat, SKIP otherwise): LMUY 0.95,
     n_bounds 4.2 and vi 65 with ni 0.5 continue with full+duals. warm_start_duals=False gives
     full-primal. Another start_state needs its own default. OPT_ds 45, PureSlip and a default
     without data.tyre_set are refused with the reason. A file at the default name with knobs is refused. A hop that does not converge saves
     its status and warns. continue_from=None reads nothing.
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import scipy.io as sio

from functions.importfile import load_solution
from functions.warmstart import default_source

ROOT = _bootstrap.ROOT
DEF = "Sturn_Static_ATDOn_EM4Off.mat"       # the default file name of Sturn, ATD On, EM4 Off
STURN = os.path.join(ROOT, "Results", DEF)  # the stored default result
DATA_AA = os.path.join(ROOT, "Data", "DATA_AA.mat")
TMPDIR = tempfile.TemporaryDirectory(prefix="continue_from_test_")   # removed at the end, SKIP included
TMP = TMPDIR.name


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def raised(exc, fn, *a, **kw):
    """str(exc) of the exception that fn(...) raises, or None when it raises none."""
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    return None


def todict(o):
    return {k: todict(v) for k, v in vars(o).items()} if hasattr(o, "__dict__") else o


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def variant(name, mutate):
    """A copy of the stored default result with mutate(dict) applied, in its own folder: returns the path."""
    d = todict(load_solution(STURN))
    mutate(d)
    path = os.path.join(TMP, name, DEF)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sio.savemat(path, {"data": d}, do_compression=True)
    return path


def accepted(path):
    """default_source(path), or None when it raises ValueError (a refusal)."""
    try:
        return default_source(path)
    except ValueError:
        return None


def with_provenance(setup):
    return lambda d: d.update(provenance={"code": "abc1234", "setup": setup, "run": {"source": "none"}})


def with_status(status):
    return lambda d: d["nlp"].update(return_status=status)


if not os.path.exists(STURN):
    print(f"SKIP test_continue_from: Results/{DEF} missing")
    TMPDIR.cleanup()
    sys.exit(0)

# ---- 1. default_source ------------------------------------------------------------------
print("1. default_source")
missing = os.path.join(TMP, "none", DEF)
msg = raised(ValueError, default_source, missing)
ok("missing file: ValueError names the path", msg is not None and missing in msg)
src = accepted(STURN)
ok("the stored default result is accepted", src is not None and str(src.nlp.return_status) == "Solve_Succeeded")
ok("a result without data.provenance (saved before E2) is accepted by its file name",
   accepted(variant("legacy", lambda d: d.pop("provenance", None))) is not None)
src = accepted(variant("default", with_provenance({})))
ok("a default provenance (empty setup) is accepted", src is not None and src.provenance.code == "abc1234")
msg = raised(ValueError, default_source,
             variant("knob", with_provenance({"OPT_ds": 45.0, "vp_overrides": {"LMUY": 0.9}})))
ok("knobs in data.provenance.setup: ValueError lists them", msg is not None and "OPT_ds" in msg
   and "vp_overrides" in msg)
msg = raised(ValueError, default_source, variant("unconverged", with_status("Maximum_Iterations_Exceeded")))
ok("an unconverged default: ValueError names the status", msg is not None and "Maximum_Iterations_Exceeded" in msg)
ok("Solved_To_Acceptable_Level is accepted",
   accepted(variant("acceptable", with_status("Solved_To_Acceptable_Level"))) is not None)

# ---- 2 and 3 need casadi and Data/DATA_AA.mat -------------------------------------------
try:
    import casadi  # noqa: F401
    have_nlp = os.path.exists(DATA_AA)
except ImportError:
    have_nlp = False
if not have_nlp:
    print("  SKIP sections 2 and 3: casadi or Data/DATA_AA.mat missing")
    TMPDIR.cleanup()
    print("\nALL continue_from TESTS PASSED (sections 2 and 3 skipped)")
    sys.exit(0)

import functions.transcription as T  # noqa: E402
import MLTP as mltp_mod  # noqa: E402


@contextlib.contextmanager
def fake_ipopt(status="Solve_Succeeded"):
    """T._make_solver -> a solver that returns x = x0 and zero duals. The real NLP is built, nothing is solved."""
    def fake(ca_, nlp, opts, info=None):
        if info is not None:
            info["linear_solver"] = "fake"

        def solver(**k):
            return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                    "lam_x": np.zeros(nlp["x"].numel())}

        solver.stats = lambda: {"iter_count": 0, "return_status": status}
        return solver

    real, T._make_solver = T._make_solver, fake
    try:
        yield
    finally:
        T._make_solver = real


WARNS = []   # the warning texts of the last quiet() call


def quiet(fn, *a, status="Solve_Succeeded", **kw):
    """(fn(...), its printed text) with a fake IPOPT that ends with ``status``."""
    buf = io.StringIO()
    with fake_ipopt(status), contextlib.redirect_stdout(buf), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    WARNS[:] = [str(x.message) for x in w]
    return out, buf.getvalue()


# ---- 2. argument errors, before any NLP is built ----------------------------------------
print("2. argument errors")
RES = os.path.join(TMP, "Results")
os.makedirs(RES)
shutil.copy(STURN, os.path.join(RES, DEF))
EMPTY = os.path.join(TMP, "empty")
os.makedirs(EMPTY)
base = dict(circuit="Sturn", save=False, plot=False, results_dir=RES)
built = []
real_build = mltp_mod.build_and_solve_nlp


def counted_build(*a, **k):
    built.append(1)
    return real_build(*a, **k)


def arg_error(**kw):
    """(the ValueError text of MLTP(**kw), whether an NLP was built). No solve runs."""
    built.clear()
    return raised(ValueError, quiet, mltp_mod.MLTP, **dict(base, **kw)), bool(built)


mltp_mod.build_and_solve_nlp = counted_build
try:
    msg, was_built = arg_error(continue_from="manual")
    ok("continue_from='manual': ValueError names the allowed values, no NLP built",
       msg is not None and "'auto'" in msg and not was_built)
    msg, was_built = arg_error(continue_from="auto", warm_start=STURN)
    ok("continue_from with warm_start: ValueError names warm_start, no NLP built",
       msg is not None and "warm_start" in msg and not was_built)
    msg, was_built = arg_error(continue_from="auto", homotopy=True)
    ok("continue_from with homotopy: ValueError names homotopy, no NLP built",
       msg is not None and "homotopy" in msg and not was_built)
    msg, was_built = arg_error(continue_from="auto", results_dir=EMPTY)
    ok("a missing default: ValueError names the default file, no NLP built",
       msg is not None and DEF in msg and not was_built)
    msg, was_built = arg_error(continue_from="auto", physics="corrected")   # the planner does not compare physics
    ok("physics corrected: ValueError names the corrected default, the matlab default is not used, no NLP built",
       msg is not None and "Sturn_Static_ATDOn_EM4Off_physCorrected.mat" in msg and not was_built)
finally:
    mltp_mod.build_and_solve_nlp = real_build

# ---- 3. MLTP(continue_from='auto') through a fake IPOPT ---------------------------------
print("3. MLTP(continue_from='auto'), fake IPOPT")
default_path = os.path.join(RES, DEF)
default_bytes = read_bytes(default_path)
auto = dict(circuit="Sturn", save=True, plot=False, results_dir=RES, continue_from="auto")

c, out = quiet(mltp_mod.MLTP, vp_overrides={"LMUY": 0.95}, **auto)
started = [ln for ln in out.splitlines() if ln.startswith("[MLTP] continue_from") and ln.endswith(default_path)]
ok("(a) LMUY 0.95: full+duals, the print names the default file it started from",
   c.elapsed["warm_start"] == "full+duals" and bool(c.elapsed["duals"]) and len(started) == 1)
ok("(a) saved under its own tag, the default file is not changed",
   os.path.basename(c.out_path) == "Sturn_Static_ATDOn_EM4Off_8c76aec0.mat"
   and read_bytes(default_path) == default_bytes)
d = load_solution(c.out_path)
ok("(a) provenance.run.source is the default file name, nlp.warm_start is full+duals",
   str(d.provenance.run.source) == DEF and str(d.nlp.warm_start) == "full+duals")

c, _ = quiet(mltp_mod.MLTP, n_bounds=4.2, **auto)
ok("(b) n_bounds 4.2: full+duals from the default (the band is not in the source name), saved as _hw4p2",
   c.elapsed["warm_start"] == "full+duals" and os.path.basename(c.out_path) == "Sturn_Static_ATDOn_EM4Off_hw4p2.mat")
c, _ = quiet(mltp_mod.MLTP, vi=65.0, ni=0.5, **auto)
ok("(d) vi 65 and ni 0.5: full+duals", c.elapsed["warm_start"] == "full+duals")
c, _ = quiet(mltp_mod.MLTP, warm_start_duals=False, vp_overrides={"LMUY": 0.95}, **auto)
ok("warm_start_duals=False: full-primal, no duals", c.elapsed["warm_start"] == "full-primal" and not c.elapsed["duals"])

msg = raised(ValueError, quiet, mltp_mod.MLTP, start_state="rolling", **auto)
ok("(c) start_state rolling without its own default: ValueError names that file, no matlab default used",
   msg is not None and "Sturn_Static_ATDOn_EM4Off_startRolling.mat" in msg)
quiet(mltp_mod.MLTP, circuit="Sturn", save=True, plot=False, results_dir=RES, warm_start=default_path,
      start_state="rolling")                                   # the rolling default, a primal-only seed
c, _ = quiet(mltp_mod.MLTP, start_state="rolling", vp_overrides={"LMUY": 0.95}, **auto)
ok("(c) with that default: full+duals, saved as _startRolling_8c76aec0",
   c.elapsed["warm_start"] == "full+duals"
   and os.path.basename(c.out_path) == "Sturn_Static_ATDOn_EM4Off_startRolling_8c76aec0.mat")

msg = raised(ValueError, quiet, mltp_mod.MLTP, OPT_ds=45, **auto)
ok("OPT_ds 45 (N 18 -> 12): ValueError with the reason, no primal-only fallback",
   msg is not None and "does not fit this NLP" in msg and "N 18 -> 12" in msg)
msg = raised(ValueError, quiet, mltp_mod.MLTP, TyreModel="PureSlip", **auto)
ok("TyreModel PureSlip (more path rows): ValueError names n_g", msg is not None and "n_g" in msg)
pre_flip = variant("preflip", lambda d: d.pop("tyre_set"))   # no tyre_set field = CopyB: planned mode cold
msg = raised(ValueError, quiet, mltp_mod.MLTP, **dict(auto, results_dir=os.path.dirname(pre_flip)))
ok("a default without data.tyre_set (CopyB): ValueError names the tyre set, no fallback to the 7-state init",
   msg is not None and "does not fit this NLP" in msg and "tyre_set 'CopyB'" in msg)

c, _ = quiet(mltp_mod.MLTP, vp_overrides={"LMUY": 0.95}, status="Maximum_Iterations_Exceeded", **auto)
ok("a hop that ends Maximum_Iterations_Exceeded: warning, no exception, the status is saved",
   any("continue_from" in w and "Maximum_Iterations_Exceeded" in w for w in WARNS)
   and str(load_solution(c.out_path).nlp.return_status) == "Maximum_Iterations_Exceeded")
quiet(mltp_mod.MLTP, vp_overrides={"LMUY": 0.95}, **auto)
ok("a converged hop: no continue_from warning", not any("continue_from" in w for w in WARNS))

knob_dir = os.path.join(TMP, "knobdir")
os.makedirs(knob_dir)
shutil.copy(variant("knobfile", with_provenance({"OPT_ds": 45.0})), os.path.join(knob_dir, DEF))
msg = raised(ValueError, quiet, mltp_mod.MLTP, **dict(auto, results_dir=knob_dir))
ok("a file at the default name with knobs in its provenance: ValueError names them",
   msg is not None and "OPT_ds" in msg)

calls = []
real_ds = mltp_mod.default_source


def counted_ds(*a, **k):
    calls.append(a)
    return real_ds(*a, **k)


mltp_mod.default_source = counted_ds
try:
    c, _ = quiet(mltp_mod.MLTP, circuit="Sturn", save=False, plot=False, results_dir=EMPTY,
                 warm_start=default_path)
finally:
    mltp_mod.default_source = real_ds
ok("continue_from=None: default_source is never called, the explicit warm start works without a default file",
   not calls and c.elapsed["warm_start"] == "full+duals" and c.data.provenance["run"]["source"] == DEF)

TMPDIR.cleanup()
print("\nALL continue_from TESTS PASSED")
