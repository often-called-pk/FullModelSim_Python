"""E2: setup knobs in the result file name and data.provenance (plain script, no pytest). Run from anywhere:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_provenance.py

A setup knob that differs from its default adds 8 hex digits to the result stem, so a knob run never
overwrites the default result. The knob is also listed in data.provenance.setup. The default setup keeps
its stem.

  1. userOpts.setup_knobs / knob_tag: the default setup, values equal to the default, int vs float, key order,
     the tyre set that decides what a default is, A5 keys, n_bounds, the MLTP_screen knobs load_model and
     ds_fine, the arguments that never count, and a rule that classifies every userOpts argument (hashed
     unless listed in _NOT_SETUP)
  2. result_stem(setup_tag=): the tag is the last part, the default stem is unchanged
  3. code_version and provenance_record: git or 'unknown', savemat round trip (ipopt_overrides as JSON text,
     also an option name longer than 31 characters), an empty setup
  4. the GUI finds what MLTP saves: RunConfig.setup_tag, app.results paths, build_solve_kwargs
  5. MLTP, MLTP_initial, MLTP_screen, optimise_design through a fake IPOPT: names, fields, the default file
     is not touched by a knob run (needs casadi and Data/DATA_AA.mat, SKIP otherwise)
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import hashlib
import inspect
import io
import json
import numbers
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import scipy.io as sio

import userOpts as UO
from app import results
from app.runconfig import RunConfig
from functions.importfile import load_solution, result_stem
from headless_solve import build_solve_kwargs
from userOpts import setup_knobs, knob_tag, provenance_record, code_version
from vehParams import PRIMARY_KEYS, MF_KEYS

ROOT = _bootstrap.ROOT
DATA_AA = os.path.join(ROOT, "Data", "DATA_AA.mat")
TMPDIR = tempfile.TemporaryDirectory(prefix="provenance_test_")   # removed at the end, SKIP included
TMP = TMPDIR.name
nan = float("nan")


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def tag_of(**kw):
    return knob_tag(setup_knobs(kw))


# ---- 1. setup_knobs / knob_tag ------------------------------------------------------
print("1. setup_knobs and knob_tag")
SIG = {k: p.default for k, p in inspect.signature(UO.userOpts).parameters.items() if p.default is not p.empty}
ok("no kwargs: no knobs, empty tag", setup_knobs({}) == {} and tag_of() == "")
ok("the userOpts defaults, written out, give no knob and the empty tag",
   setup_knobs(SIG) == {} and knob_tag(setup_knobs(SIG)) == "")
ok("equal to the default: MF205 mb and pKy4, None and {} for the dict arguments",
   tag_of(vp_overrides={"mb": 1820.0, "pKy4": 2.0}) == "" and tag_of(vp_overrides=None, mesh_opts=None) == ""
   and tag_of(vp_overrides={}, mesh_opts={}) == "")
ok("circuit, config, tyre set, physics, start state, mesh and a scalar n_bounds are in the stem: no knob",
   tag_of(circuit="BCN", AeroConfig="AALB", ATD="Off", Electric_4Motors="On", tyre_set="CopyB",
          physics="corrected", start_state="rolling", mesh="curvature", n_bounds=3.5) == "")
ok("linear solver, HSL dir, jit, cse, paths and ipopt_overrides change the route, not the setup",
   tag_of(linear_solver="mumps", hsl_dir="x", jit=True, cse=True, circuits_dir="c", data_dir="d",
          ipopt_overrides={"max_wall_time": 100.0, "tol": 1e-8}) == "")
ok("MLTP arguments that are not userOpts arguments are never read (save, warm_start, ladder ...)",
   tag_of(save=False, plot=False, warm_start="p", ladder="qss23", refine=True, homotopy=True,
          results_dir="r") == "")
t1 = tag_of(vp_overrides={"LMUY": 0.95})
ok("LMUY 0.95: 8 hex digits", len(t1) == 8 and all(c in "0123456789abcdef" for c in t1))
ok("45 and 45.0 give one tag", tag_of(OPT_ds=45) == tag_of(OPT_ds=45.0) != "")
ok("key order does not matter",
   tag_of(vp_overrides={"LMUY": 0.95, "LMUX": 0.9}) == tag_of(vp_overrides={"LMUX": 0.9, "LMUY": 0.95}))
ok("an override equal to the default next to a real one changes nothing",
   tag_of(vp_overrides={"LMUY": 0.95, "mb": 1820.0}) == t1)
ok("another value gives another tag", tag_of(vp_overrides={"LMUY": 0.96}) != t1)
ok("the tyre set decides what a default is (pKy4 2.0: default for MF205, a knob for CopyB)",
   tag_of(vp_overrides={"pKy4": 2.0}) == "" and tag_of(vp_overrides={"pKy4": 2.0}, tyre_set="CopyB") != "")
ok("an A5 key is always a knob", tag_of(vp_overrides={"Preg": 5e4}, physics="corrected") != "")
ok("ni = 0 is a knob (NaN is the default)", tag_of(ni=0.0) != "")
ok("screening is a knob (a loose solve must not overwrite the default)", tag_of(screening=True) != "")


def probe(default):
    """A value that differs from `default`, chosen by type alone. None takes a list of pairs: dict() and
    the n_bounds pair test both accept it."""
    if default is None:
        return [("__probe__", 1.0)]
    if isinstance(default, bool):
        return not default
    if isinstance(default, numbers.Real):
        return 0.0 if np.isnan(default) else default + 1
    if isinstance(default, str):
        return default + "_probe"
    raise AssertionError(f"probe(): no rule for the default {default!r}")


# a new userOpts argument needs no edit here: it is hashed unless it is listed in _NOT_SETUP
knob_args = set(SIG) - UO._NOT_SETUP
hashed = {k for k in knob_args if set(setup_knobs({k: probe(SIG[k])})) == {k}}
ok("every _NOT_SETUP name is a userOpts argument", UO._NOT_SETUP <= set(SIG))
ok(f"every other userOpts argument is hashed (not hashed: {sorted(knob_args - hashed)})", hashed == knob_args)
ok("TyreModel, an MLTP argument, is hashed", set(setup_knobs({"TyreModel": "PureSlip"})) == {"TyreModel"})
ok("load_model and ds_fine, MLTP_screen arguments, are hashed (ds_fine 1 is the default 1.0)",
   set(setup_knobs({"load_model": "nominal"})) == {"load_model"} and set(setup_knobs({"ds_fine": 0.5})) == {"ds_fine"}
   and setup_knobs({"load_model": "vehModel", "ds_fine": 1}) == {})
ok("n_bounds: a custom pair is hashed by value, a number is left to the stem",
   tag_of(n_bounds=(-2.0, 5.0)) != tag_of(n_bounds=(-3.0, 5.0))
   and tag_of(n_bounds=(np.full(4, -2.0), np.full(4, 3.0))) != ""
   and "n_bounds" not in setup_knobs({"n_bounds": 3.5}))
ok("the tag is a function of the knobs only (not of the numpy type)",
   knob_tag({"vp_overrides": {"LMUY": 0.95}}) == t1 == knob_tag({"vp_overrides": {"LMUY": np.float64(0.95)}}))
pin = hashlib.sha256(b'{"vp_overrides":{"LMUY":0.95}}').hexdigest()[:8]
ok("pinned: LMUY 0.95 is tag 8c76aec0, the sha256 of the canonical JSON", t1 == pin == "8c76aec0")

# ---- 2. result_stem -----------------------------------------------------------------
print("2. result_stem(setup_tag=)")
ok("default stem unchanged", result_stem("BCN", "Static_ATDOn_EM4Off") == "BCN_Static_ATDOn_EM4Off"
   and result_stem("BCN", "Static_ATDOn_EM4Off", setup_tag="") == "BCN_Static_ATDOn_EM4Off")
ok("the tag is the last part", result_stem("BCN", "Static_ATDOn_EM4Off", "CopyB", "curvature", "corrected",
                                           "rolling", "hw4p2", "8c76aec0")
   == "BCN_Static_ATDOn_EM4Off_CopyB_physCorrected_startRolling_hw4p2_meshCurvature_8c76aec0")
ok("the tag alone", result_stem("Sturn", "Static_ATDOn_EM4Off", setup_tag=t1) == f"Sturn_Static_ATDOn_EM4Off_{t1}")

# ---- 3. code_version, provenance_record ---------------------------------------------
print("3. code_version and provenance_record")
code_version.cache_clear()
v = code_version()
ok("code_version is a non-empty string, the same on a second call",
   isinstance(v, str) and bool(v) and code_version() == v)
real_run = subprocess.run


def no_git(*a, **k):
    raise FileNotFoundError("git")


subprocess.run = no_git
code_version.cache_clear()
try:
    ok("no git: 'unknown'", code_version() == "unknown")
finally:
    subprocess.run = real_run
    code_version.cache_clear()
OV = {"mu_init": 1e-3, "nlp_scaling_constr_target_gradient": 1.0}    # 34 characters: savemat refuses it as a field
rec = provenance_record(setup_knobs({"vp_overrides": {"LMUY": 0.95}, "mesh_opts": {"a": 2.0, "ds_max": None},
                                     "OPT_ds": 45}),
                        source="Sturn_Static_ATDOn_EM4Off.mat", ipopt_overrides=dict(OV), jit=None)
p = os.path.join(TMP, "prov.mat")
sio.savemat(p, {"data": {"lap_time": 1.0, "provenance": rec}}, do_compression=True)
pv = load_solution(p).provenance
ok("the knobs survive a .mat round trip",
   float(pv.setup.vp_overrides.LMUY) == 0.95 and float(pv.setup.OPT_ds) == 45
   and float(pv.setup.mesh_opts.a) == 2.0 and not hasattr(pv.setup.mesh_opts, "ds_max"))
ok("run: source and ipopt_overrides as JSON text (a long option name too), a None entry is dropped",
   str(pv.run.source) == "Sturn_Static_ATDOn_EM4Off.mat" and json.loads(str(pv.run.ipopt_overrides)) == OV
   and not hasattr(pv.run, "jit"))
sio.savemat(p, {"data": {"lap_time": 1.0, "provenance": provenance_record({})}}, do_compression=True)
ok("the default setup: an empty setup struct", vars(load_solution(p).provenance.setup) == {})

# ---- 4. the GUI finds what MLTP saves ------------------------------------------------
print("4. RunConfig.setup_tag, app.results and build_solve_kwargs")
rc0 = RunConfig()
ok("a default RunConfig has the empty tag", rc0.setup_tag() == "")
ok("... and its result path is the default stem",
   os.path.basename(results.result_mat_path("/o", rc0.circuit, rc0.AeroConfig, rc0.ATD, rc0.Electric_4Motors,
                                            rc0.tyre_set, rc0.mesh, rc0.setup_tag()))
   == "Sturn_Static_ATDOn_EM4Off.mat")
rc1 = RunConfig(vp={**rc0.vp, "LMUY": 0.95})
ok("an edited Setup row gives the tag that a script gets with vp_overrides={'LMUY': 0.95}",
   rc1.setup_tag() == t1)
ok("the result path and the plot folder carry the tag",
   os.path.basename(results.result_mat_path("/o", "Sturn", "Static", "On", "Off", "MF205", "auto", t1))
   == f"Sturn_Static_ATDOn_EM4Off_{t1}.mat"
   and results.plot_dir("/o", "Sturn", "Static", "On", "Off", "MF205", "auto", t1)
   == os.path.join("/o", "Plots", "Sturn", f"Static_ATDOn_EM4Off_{t1}"))
rc2 = RunConfig(OPT_ds=45.0, tol=1e-6, max_iter=2500)
ok("Advanced-tab values: 45.0 from the GUI equals 45 from a script",
   rc2.setup_tag() == tag_of(OPT_ds=45, tol=1e-6, max_iter=2500) != "")
ok("the solve kwargs of the GUI hold the same knobs",
   setup_knobs(build_solve_kwargs(rc2._cfg(), ROOT)) == {"OPT_ds": 45.0, "tol": 1e-6, "max_iter": 2500})
ok("a mesh or tyre set choice is in the stem, not in the tag",
   RunConfig(mesh="curvature", tyre_set="CopyB").setup_tag() == "")

# ---- 5. MLTP and the other savers through a fake IPOPT --------------------------------
print("5. saved names and fields, fake IPOPT")
STURN = os.path.join(ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
try:
    import casadi  # noqa: F401
    have_nlp = os.path.exists(DATA_AA) and os.path.exists(STURN)
except ImportError:
    have_nlp = False
if not have_nlp:
    print("  SKIP section 5: casadi, Data/DATA_AA.mat or Results/Sturn_Static_ATDOn_EM4Off.mat missing")
    TMPDIR.cleanup()
    print("\nALL provenance TESTS PASSED (section 5 skipped)")
    sys.exit(0)

import functions.transcription as T
import MLTP as mltp_mod
import MLTP_initial as mi_mod
import MLTP_paramOptim as po_mod
from MLTP_screen import MLTP_screen
from plotSDI import plotSDI


@contextlib.contextmanager
def fake_ipopt():
    """T._make_solver -> a solver that returns x = x0 and zero duals: the real NLP is built, nothing solved."""
    def fake(ca_, nlp, opts, info=None):
        if info is not None:
            info["linear_solver"] = "fake"

        def solver(**k):
            return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                    "lam_x": np.zeros(nlp["x"].numel())}

        solver.stats = lambda: {"iter_count": 0, "return_status": "Solve_Succeeded"}
        return solver

    real, T._make_solver = T._make_solver, fake
    try:
        yield
    finally:
        T._make_solver = real


def quiet(fn, *a, **kw):
    with fake_ipopt(), contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


RES = os.path.join(TMP, "Results")
base = dict(circuit="Sturn", warm_start=STURN, save=True, plot=False, results_dir=RES)
c0 = quiet(mltp_mod.MLTP, **base)
default_path = os.path.join(RES, "Sturn_Static_ATDOn_EM4Off.mat")
ok("default run: Sturn_Static_ATDOn_EM4Off.mat, ctx.out_path says so",
   os.path.isfile(default_path) and c0.out_path == default_path)
d0 = load_solution(default_path)
ok("default run: provenance.setup is empty, no setup_tag field, the code version and casadi are recorded",
   vars(d0.provenance.setup) == {} and not hasattr(d0, "setup_tag") and str(d0.provenance.code) == code_version()
   and str(d0.provenance.run.casadi) == casadi.__version__)
ok("default run: the source is the file the warm start came from",
   str(d0.provenance.run.source) == "Sturn_Static_ATDOn_EM4Off.mat")
with open(default_path, "rb") as fh:
    default_bytes = fh.read()

c1 = quiet(mltp_mod.MLTP, **dict(base, warm_start=default_path, vp_overrides={"LMUY": 0.95}))
knob_path = os.path.join(RES, f"Sturn_Static_ATDOn_EM4Off_{t1}.mat")
ok("LMUY 0.95: the file name carries the tag", os.path.isfile(knob_path) and c1.out_path == knob_path)
with open(default_path, "rb") as fh:
    ok("LMUY 0.95: the default result is untouched (bytes)", fh.read() == default_bytes)
d1 = load_solution(knob_path)
ok("LMUY 0.95: data.setup_tag, provenance.setup.vp_overrides, mf_overrides names",
   str(d1.setup_tag) == t1 and float(d1.provenance.setup.vp_overrides.LMUY) == 0.95
   and list(np.atleast_1d(d1.mf_overrides)) == ["LMUY"] and set(vars(d1.provenance.setup)) == {"vp_overrides"})
ok("LMUY 0.95: the warm start source is recorded and the mode is full+duals",
   str(d1.provenance.run.source) == "Sturn_Static_ATDOn_EM4Off.mat" and c1.elapsed["warm_start"] == "full+duals")
ok("provenance.params holds the values the model was built with (default and knob run)",
   float(d0.provenance.params.LMUY) == 1.0 and float(d1.provenance.params.LMUY) == 0.95
   and float(d1.provenance.params.mb) == 1820.0 and float(d1.provenance.params.pKy4) == 2.0
   and float(d1.provenance.params.Pmax) == 456000.0 and float(d1.provenance.params.gear) == c1.vp.gear
   and set(vars(d1.provenance.params)) >= PRIMARY_KEYS | MF_KEYS)

OV2 = {"max_wall_time": 99.0, "nlp_scaling_constr_target_gradient": 1.0}
c2 = quiet(mltp_mod.MLTP, **dict(base, warm_start=default_path, vi=65.0, ipopt_overrides=dict(OV2),
                                 linear_solver="mumps"))
ok("vi 65 + route knobs: tag of vi alone, ipopt_overrides (JSON text, in the file too) and linear_solver only recorded",
   os.path.basename(c2.out_path) == f"Sturn_Static_ATDOn_EM4Off_{tag_of(vi=65.0)}.mat"
   and json.loads(c2.data.provenance["run"]["ipopt_overrides"]) == OV2
   and json.loads(str(load_solution(c2.out_path).provenance.run.ipopt_overrides)) == OV2
   and c2.data.provenance["run"]["linear_solver"] == "mumps")
c3 = quiet(mltp_mod.MLTP, **dict(base, warm_start=default_path, n_bounds=4.2, start_state="matlab"))
ok("n_bounds 4.2 alone: the stem has _hw4p2 and no tag",
   os.path.basename(c3.out_path) == "Sturn_Static_ATDOn_EM4Off_hw4p2.mat" and not hasattr(c3.data, "setup_tag"))
c4 = quiet(mltp_mod.MLTP, **dict(base, warm_start=default_path, n_bounds=4.2, vp_overrides={"LMUY": 0.95}))
ok("n_bounds 4.2 + LMUY 0.95: _hw4p2 first, the tag last",
   os.path.basename(c4.out_path) == f"Sturn_Static_ATDOn_EM4Off_hw4p2_{t1}.mat")
c5 = quiet(mltp_mod.MLTP, **dict(base, warm_start=default_path, TyreModel="PureSlip"))
ok("TyreModel PureSlip is a knob, with its own file",
   os.path.basename(c5.out_path) == f"Sturn_Static_ATDOn_EM4Off_{tag_of(TyreModel='PureSlip')}.mat"
   and c5.data.provenance["setup"] == {"TyreModel": "PureSlip"})
ok("ctx.data holds the same provenance as the file (in memory)",
   c1.data.provenance["setup"] == {"vp_overrides": {"LMUY": 0.95}})

# the GUI path: the file a RunConfig run would write is the file the Output tab looks for
rcg = RunConfig(vp={**rc0.vp, "LMUY": 0.95}, output_dir=TMP)
kw = build_solve_kwargs(rcg._cfg(), ROOT)
kw.update(circuit="Sturn", warm_start=default_path, plot=False, save=True, results_dir=RES)
cg = quiet(mltp_mod.MLTP, **kw)
ok("GUI: result_mat_path(..., rc.setup_tag()) names the file MLTP wrote",
   os.path.normpath(cg.out_path) == os.path.normpath(os.path.join(
       RES, os.path.basename(results.result_mat_path(TMP, rcg.circuit, rcg.AeroConfig, rcg.ATD,
                                                     rcg.Electric_4Motors, rcg.tyre_set, rcg.mesh,
                                                     rcg.setup_tag())))))

# MLTP_initial, MLTP_screen, optimise_design
quiet(mi_mod.MLTP_initial, circuit="Sturn", save=True, results_dir=RES)
quiet(mi_mod.MLTP_initial, circuit="Sturn", save=True, results_dir=RES, vp_overrides={"LMUY": 0.95})
ok("MLTP_initial: init_Sturn.mat, and the tag for a knob run",
   os.path.isfile(os.path.join(RES, "init_Sturn.mat")) and os.path.isfile(os.path.join(RES, f"init_Sturn_{t1}.mat")))
ok("MLTP_initial: provenance in data.init",
   float(load_solution(os.path.join(RES, f"init_Sturn_{t1}.mat")).provenance.setup.vp_overrides.LMUY) == 0.95)
cs = quiet(MLTP_screen, "Sturn", save=True, results_dir=RES, verbose=False, vp_overrides={"LMUY": 0.95})
quiet(MLTP_screen, "Sturn", save=True, results_dir=RES, verbose=False)
ds_ = load_solution(cs.out_path)
ok("MLTP_screen: <stem>_<tag>_qss.mat for a knob run, <stem>_qss.mat for the default, setup_tag and provenance",
   os.path.basename(cs.out_path) == f"Sturn_Static_ATDOn_EM4Off_{t1}_qss.mat"
   and os.path.isfile(os.path.join(RES, "Sturn_Static_ATDOn_EM4Off_qss.mat"))
   and str(ds_.setup_tag) == t1 and float(ds_.provenance.setup.vp_overrides.LMUY) == 0.95)
qss_default = os.path.join(RES, "Sturn_Static_ATDOn_EM4Off_qss.mat")
with open(qss_default, "rb") as fh:
    qss_bytes = fh.read()
cn = quiet(MLTP_screen, "Sturn", save=True, results_dir=RES, verbose=False, load_model="nominal")
with open(qss_default, "rb") as fh:
    ok("MLTP_screen load_model='nominal': its own file, the default QSS file is untouched (bytes)",
       os.path.basename(cn.out_path) == f"Sturn_Static_ATDOn_EM4Off_{tag_of(load_model='nominal')}_qss.mat"
       and fh.read() == qss_bytes)
cp = quiet(po_mod.MLTP_paramOptim, warm_start=default_path, save=True, results_dir=RES,
           vp_overrides={"mb": 1900.0})
dp = load_solution(cp.out_path)
tag_mb = tag_of(vp_overrides={"mb": 1900.0})
ok("optimise_design: Sturn_paramOptim_<tag>.mat for a knob run, setup_tag and provenance.setup in the file",
   os.path.basename(cp.out_path) == f"Sturn_paramOptim_{tag_mb}.mat" and str(dp.setup_tag) == tag_mb
   and float(dp.provenance.setup.vp_overrides.mb) == 1900.0 and set(vars(dp.provenance.setup)) == {"vp_overrides"})

# plotSDI names its folder from the record
PLOTS = os.path.join(TMP, "Plots")
quiet(plotSDI, knob_path, save_dir=PLOTS)
ok("plotSDI: Plots/Sturn/Static_ATDOn_EM4Off_<tag>/racing_line.html",
   os.path.isfile(os.path.join(PLOTS, "Sturn", f"Static_ATDOn_EM4Off_{t1}", "racing_line.html")))
quiet(plotSDI, default_path, save_dir=PLOTS)
ok("plotSDI: the default folder has no tag",
   os.path.isfile(os.path.join(PLOTS, "Sturn", "Static_ATDOn_EM4Off", "racing_line.html")))

TMPDIR.cleanup()
print("\nALL provenance TESTS PASSED")
