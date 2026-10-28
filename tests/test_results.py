"""Results parser: path scheme, .mat summary (lap_time + final energy), plot listing."""
import os, sys, atexit, shutil, tempfile
import numpy as np
import scipy.io as sio
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
from app import results

# scratch files: CLAUDE_JOB_DIR_TMP when set, else a fresh temp dir removed at exit (never the repo)
TMP = os.environ.get("CLAUDE_JOB_DIR_TMP")
if not TMP:
    TMP = tempfile.mkdtemp(prefix="results_test_")
    atexit.register(shutil.rmtree, TMP, ignore_errors=True)

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# path scheme matches MLTP/plotSDI naming
mp = results.result_mat_path("/out", "Sturn", "Static", "On", "Off")
ok("mat path scheme", mp == os.path.join("/out", "Results", "Sturn_Static_ATDOn_EM4Off.mat"))
pd = results.plot_dir("/out", "Sturn", "Static", "On", "Off")
ok("plot dir scheme", pd == os.path.join("/out", "Plots", "Sturn", "Static_ATDOn_EM4Off"))
# E2: a setup knob adds its tag as the last part of the stem, in the file and in the plot folder
mp = results.result_mat_path("/out", "Sturn", "Static", "On", "Off", "CopyB", "curvature", "8c76aec0")
ok("mat path with a setup tag",
   mp == os.path.join("/out", "Results", "Sturn_Static_ATDOn_EM4Off_CopyB_meshCurvature_8c76aec0.mat"))
pd = results.plot_dir("/out", "Sturn", "Static", "On", "Off", setup_tag="8c76aec0")
ok("plot dir with a setup tag", pd == os.path.join("/out", "Plots", "Sturn", "Static_ATDOn_EM4Off_8c76aec0"))

# summary from a crafted .mat (E_motor is stored in Wh)
data = {"lap_time": 12.5, "vehicle": {"E_motor": np.array([0.0, 1000.0, 2300.0])}}
mat = os.path.join(TMP, "_res.mat")
sio.savemat(mat, {"data": data}, do_compression=True)
summ = results.parse_summary(mat)
ok("lap_time parsed", abs(summ["lap_time_s"] - 12.5) < 1e-9)
ok("energy parsed (last E_motor, Wh shown as kWh)", abs(summ["energy_kWh"] - 2.3) < 1e-9)

# plot listing
pdir = os.path.join(TMP, "_plots")
os.makedirs(pdir, exist_ok=True)
for nm in ("speed.html", "racing_line.html"):
    open(os.path.join(pdir, nm), "w").write("<html></html>")
plots = results.list_plots(pdir)
ok("two plots listed sorted", [os.path.basename(p) for p in plots]
   == ["racing_line.html", "speed.html"])
ok("missing dir -> empty list", results.list_plots(os.path.join(TMP, "_nope")) == [])
# --- F2: result_paths(rc) names the file the solve writes ---------------------------------------------
import contextlib, inspect, io, re, warnings
from app.runconfig import RunConfig
from functions.context import Ctx
from functions.importfile import result_stem
from userOpts import userOpts, setup_knobs, knob_tag
import headless_solve

ok("result_paths of the default RunConfig is the pre-F2 name",
   results.result_paths(RunConfig(output_dir="/out"))
   == (os.path.join("/out", "Results", "Sturn_Static_ATDOn_EM4Off.mat"),
       os.path.join("/out", "Plots", "Sturn", "Static_ATDOn_EM4Off")))

USEROPTS = set(inspect.signature(userOpts).parameters)


def nlp_stem(kw):
    """The stem MLTP saves under for the solver kwargs kw (headless_solve.build_solve_kwargs output).
    Same calls as MLTP: a userOpts build for the ctx fields, then result_stem with the knob tag. No IPOPT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # the BCN track QA warning
        ctx = userOpts(Ctx(), **{k: v for k, v in kw.items() if k in USEROPTS})
    cfg = f"{kw['AeroConfig']}_ATD{ctx.ATD}_EM4{ctx.Electric_4Motors}"
    return result_stem(kw["circuit"], cfg, ctx.tyre_set, ctx.mesh_requested, ctx.physics, ctx.start_state,
                       n_tag=ctx.n_tag, setup_tag=knob_tag(setup_knobs(kw)))


# (RunConfig fields, regex of the file name). A knob that differs from its default adds 8 hex digits.
GRID = [
    (dict(), r"Sturn_Static_ATDOn_EM4Off\.mat"),
    (dict(physics="corrected"), r"Sturn_Static_ATDOn_EM4Off_physCorrected\.mat"),
    (dict(start_state="rolling", n_bounds=3.5), r"Sturn_Static_ATDOn_EM4Off_startRolling_hw3p5\.mat"),
    (dict(n_bounds=[-2.0, 3.0], Steering="Ackermann"), r"Sturn_Static_ATDOn_EM4Off_nbCustom_[0-9a-f]{8}\.mat"),
    (dict(tyre_set="CopyB", mesh="curvature", max_iter=2500),
     r"Sturn_Static_ATDOn_EM4Off_CopyB_meshCurvature_[0-9a-f]{8}\.mat"),
    (dict(mode="qss"), r"Sturn_Static_ATDOn_EM4Off_qss\.mat"),
    (dict(mode="qss", TyreModel="PureSlip", physics="corrected"),
     r"Sturn_Static_ATDOn_EM4Off_physCorrected_qss\.mat"),
    (dict(circuit="BCN", AeroConfig="AALB", ATD="Off", Electric_4Motors="On", physics="corrected",
          start_state="periodic", n_bounds=4.25, mesh="uniform"),
     r"BCN_AALB_ATDOff_EM4On_physCorrected_startPeriodic_hw4p25_meshUniform\.mat"),
]


def check_grid(entry):
    kw, pattern = entry
    with tempfile.TemporaryDirectory(prefix="results_grid_") as out:
        rc = RunConfig(output_dir=out, **kw)
        cfg_path = os.path.join(out, "cfg.json")
        rc_cfg = rc.write_cfg(cfg_path)
        mat, pdir = results.result_paths(rc)
        if rc.mode == "qss":                             # the real QSS solve, through headless_solve.main
            with contextlib.redirect_stdout(io.StringIO()):
                code = headless_solve.main([cfg_path])
            res_dir = os.path.join(out, "Results")
            names = os.listdir(res_dir) if os.path.isdir(res_dir) else []
            ok(f"{kw}: the qss run writes exactly one file and no plots",
               code == 0 and len(names) == 1 and not os.path.isdir(os.path.join(out, "Plots")))
            solved = os.path.join(res_dir, names[0])
            stem = names[0][:-len("_qss.mat")]
        else:
            solved = os.path.join(out, "Results", nlp_stem(headless_solve.build_solve_kwargs(rc_cfg, _bootstrap.ROOT)) + ".mat")
            stem = os.path.basename(solved)[:-len(".mat")]
        ok(f"{kw}: result_paths names the file the solve writes", mat == solved)
        ok(f"{kw}: the file name matches {pattern}", re.fullmatch(pattern, os.path.basename(mat)) is not None)
        ok(f"{kw}: the plot folder follows the same stem (qss: no folder)",
           pdir == ("" if rc.mode == "qss" else os.path.join(out, "Plots", rc.circuit, stem[len(rc.circuit) + 1:])))


if os.path.isfile(os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")):
    for entry in GRID:
        check_grid(entry)
    ok("TyreModel does not rename a qss file (MLTP_screen ignores it)",
       results.result_paths(RunConfig(mode="qss", TyreModel="PureSlip", physics="corrected", output_dir="/out"))[0]
       == results.result_paths(RunConfig(mode="qss", physics="corrected", output_dir="/out"))[0])
else:
    print("  [SKIP] F2 grid: Data/DATA_AA.mat missing (the qss run needs ctx.aero)")

print("\nALL results TESTS PASSED")
