"""Offscreen GUI logic: window builds, collect_runconfig reflects widgets,
EM4-on disables the ATD control (the config-conflict rule), the Tyre set / Mesh
combos default correctly and reach RunConfig, and the inert Setup rows are
greyed (ATD on: brkB + Tdist; 4 Motors on: Tdist only)."""
import os, sys, warnings
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
from PySide6.QtWidgets import QApplication
from app.mainwindow import MainWindow, TYRE_SETS, MESHES
from app.runconfig import RunConfig
from functions.context import Ctx
from userOpts import userOpts, MESH_AUTO_MIN_LENGTH
from vehParams import TYRE_SETS as VP_TYRE_SETS

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

app = QApplication.instance() or QApplication([])
win = MainWindow()

rc = win.collect_runconfig()
ok("collect returns RunConfig", isinstance(rc, RunConfig))
ok("default circuit collected", rc.circuit == "Sturn")

# conflict rule: turning 4-motors on disables ATD
win.set_em4(True)
ok("ATD control disabled when EM4 on", win.atd_enabled() is False)
win.set_em4(False)
ok("ATD control re-enabled when EM4 off", win.atd_enabled() is True)

# --- Tyre set / Mesh combos (Advanced tab) ---------------------------------
def items(combo):
    return [combo.itemText(i) for i in range(combo.count())]

def userOpts_accepts(**kw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            userOpts(Ctx(), circuit="Sturn", **kw)
        return True
    except ValueError:
        return False

ok("Tyre set combo lists MF205 (default) then CopyB", items(win.tyre_set) == ["MF205", "CopyB"])
ok("Tyre set combo choices are the vehParams tyre sets", set(items(win.tyre_set)) == set(VP_TYRE_SETS))
ok("Tyre set combo default is MF205", win.tyre_set.currentText() == "MF205")
ok("Mesh combo lists auto (default), uniform, curvature", items(win.mesh) == ["auto", "uniform", "curvature"])
ok("Mesh combo default is auto", win.mesh.currentText() == "auto")
ok("module choice lists match the combos", items(win.tyre_set) == TYRE_SETS and items(win.mesh) == MESHES)
ok("collect carries the default tyre_set / mesh", rc.tyre_set == "MF205" and rc.mesh == "auto")
ok("default combos equal the RunConfig field defaults",
   (rc.tyre_set, rc.mesh) == (RunConfig().tyre_set, RunConfig().mesh))
ok("every Mesh item is a mesh userOpts accepts", all(userOpts_accepts(mesh=m) for m in items(win.mesh)))
ok("every Tyre set item is a tyre_set userOpts accepts", all(userOpts_accepts(tyre_set=t) for t in items(win.tyre_set)))
ok("that acceptance check can fail (unknown mesh / tyre_set rejected)",
   not userOpts_accepts(mesh="adaptive") and not userOpts_accepts(tyre_set="bogus"))

tip = win.tyre_set.toolTip()
ok("Tyre set tooltip: MF205 = MATLAB-run set (default)", "MF205 = MATLAB-run set (default)" in tip)
ok("Tyre set tooltip: CopyB = legacy set, pKy4=0 (zero cornering stiffness), kept to reproduce old results",
   all(s in tip for s in ("CopyB = legacy shipped set with pKy4=0", "zero cornering stiffness",
                          "kept to reproduce old results")))
ok("Mesh tooltip states the auto rule",
   "auto = curvature-weighted knots for laps >= 2000 m, uniform otherwise" in win.mesh.toolTip())
ok("Mesh tooltip threshold is userOpts.MESH_AUTO_MIN_LENGTH", MESH_AUTO_MIN_LENGTH == 2000.0)

win.tyre_set.setCurrentText("CopyB")
win.mesh.setCurrentText("curvature")
rc2 = win.collect_runconfig()
ok("collect carries the selected tyre_set / mesh", rc2.tyre_set == "CopyB" and rc2.mesh == "curvature")
win.mesh.setCurrentText("uniform")
ok("collect carries mesh=uniform", win.collect_runconfig().mesh == "uniform")
win.tyre_set.setCurrentText("MF205")
win.mesh.setCurrentText("auto")
rc3 = win.collect_runconfig()
ok("combos back to their defaults", (rc3.tyre_set, rc3.mesh) == ("MF205", "auto"))

# --- inert Setup rows in the 23-state model ----------------------------------
# ATD on: vehModel takes the wheel-torque split from the four ATD inputs, so brkB and
# Tdist are both unused. 4 Motors on: wheel torque is T_motor_* * gear + T_brake * brkB,
# so only Tdist is unused.
INERT = {"brkB", "Tdist"}
EM4_INERT = {"Tdist"}

def row_enabled(key, w=win):
    return w.vp_spins[key].isEnabled() and w.vp_labels[key].isEnabled()

def row_tips(key, w=win):
    return (w.vp_spins[key].toolTip(), w.vp_labels[key].toolTip())

fresh = MainWindow()                 # startup state (ATD defaults to on)
ok("startup: ATD is on", fresh.atd.isChecked())
ok("startup: the inert set is exactly brkB + Tdist", fresh.inert_vp_keys() == INERT)
ok("startup: brkB and Tdist rows are greyed out", not row_enabled("brkB", fresh) and not row_enabled("Tdist", fresh))
ok("startup: the greyed rows say why (ATD, ignored)",
   all("ATD" in t and "ignored" in t for k in INERT for t in row_tips(k, fresh)))
ok("startup: every other Setup row stays editable",
   all(row_enabled(k, fresh) for k in fresh.vp_spins if k not in INERT))

# the EM4 round trip above left ATD unchecked (4 Motors on unchecks it, off does not re-check it)
ok("after the EM4 round trip ATD is off and the rows are editable",
   not win.atd.isChecked() and win.inert_vp_keys() == set() and row_enabled("brkB") and row_enabled("Tdist"))
win.atd.setChecked(True)
ok("ATD on: the inert set is exactly brkB + Tdist", win.inert_vp_keys() == INERT)
ok("ATD on: brkB and Tdist rows are greyed out", not row_enabled("brkB") and not row_enabled("Tdist"))
ok("ATD on: the greyed rows say why (ATD, ignored)",
   all("ATD" in t and "ignored" in t for k in INERT for t in row_tips(k)))
ok("ATD on: every other Setup row stays editable", all(row_enabled(k) for k in win.vp_spins if k not in INERT))

win.atd.setChecked(False)
ok("ATD off: nothing is inert", win.inert_vp_keys() == set())
ok("ATD off: brkB and Tdist rows are editable again", row_enabled("brkB") and row_enabled("Tdist"))
ok("ATD off: tooltips are back to the plain description",
   all("ATD" not in t for k in INERT for t in row_tips(k)) and row_tips("brkB")[0] == "Front brake-torque fraction")
win.atd.setChecked(True)
ok("ATD on again: greyed again", win.inert_vp_keys() == INERT and not row_enabled("brkB") and not row_enabled("Tdist"))

# an inert row still holds (and reports) its value: the 7-state warm-start model reads it
win.vp_spins["brkB"].setValue(0.55)
rc4 = win.collect_runconfig()
ok("a value held by a greyed row is still collected", rc4.vp["brkB"] == 0.55 and rc4.vp_overrides() == {"brkB": 0.55})
win._reset_all_params()
ok("Reset still reaches a greyed row", win.vp_spins["brkB"].value() == 0.6766)

# 4 Motors forces ATD off; Tdist goes inert but brkB stays live (T_brake * brkB is still used)
win.set_em4(True)
ok("4 Motors on: ATD forced off", not win.atd.isChecked())
ok("4 Motors on: the inert set is exactly Tdist", win.inert_vp_keys() == EM4_INERT)
ok("4 Motors on: Tdist row greyed out, brkB row enabled", not row_enabled("Tdist") and row_enabled("brkB"))
ok("4 Motors on: the Tdist row says why (4 Motors, ignored, not ATD)",
   all("4 Motors" in t and "ignored" in t and "ATD" not in t for t in row_tips("Tdist")))
ok("4 Motors on: the brkB row keeps its plain tooltip", row_tips("brkB") == ("Front brake-torque fraction",) * 2)
ok("4 Motors on: every other Setup row stays editable", all(row_enabled(k) for k in win.vp_spins if k not in EM4_INERT))
win.atd.setChecked(True)             # programmatic: the box itself is disabled for the user
ok("4 Motors wins over a stray ATD check (userOpts forces ATD off): only Tdist inert",
   win.inert_vp_keys() == EM4_INERT and not row_enabled("Tdist") and row_enabled("brkB"))
ok("4 Motors wins over a stray ATD check: the Tdist tooltip names 4 Motors, not ATD",
   all("4 Motors" in t and "ATD" not in t for t in row_tips("Tdist")))
win.atd.setChecked(False)
win.set_em4(False)
ok("4 Motors off again: ATD stays off, nothing inert",
   not win.atd.isChecked() and win.inert_vp_keys() == set() and row_enabled("brkB") and row_enabled("Tdist"))
ok("4 Motors off again: tooltips are back to the plain description",
   all("4 Motors" not in t and "ATD" not in t for k in INERT for t in row_tips(k)))
win.set_em4(True)                    # ATD already off, so only the 4 Motors toggle can refresh the rows
ok("4 Motors on with ATD already off: Tdist greyed, brkB enabled",
   win.inert_vp_keys() == EM4_INERT and not row_enabled("Tdist") and row_enabled("brkB"))
win.set_em4(False)
ok("4 Motors off with ATD off: nothing inert", win.inert_vp_keys() == set() and row_enabled("brkB") and row_enabled("Tdist"))
win.atd.setChecked(True)
ok("restored: ATD on, rows greyed", win.inert_vp_keys() == INERT and not row_enabled("brkB") and not row_enabled("Tdist"))

# --- Track QA gate: GUI circuit list (Task 2, F1) ---------------------------------
from app.mainwindow import CIRCUITS
from userOpts import _load_track, _REAL_CIRCUITS, _synthetic_curvature

ok("GUI list: BCNAssetto out, NBR in", "BCNAssetto" not in CIRCUITS and "NBR" in CIRCUITS)
ok("GUI list: each circuit is a real track or a synthetic one",
   all(c in _REAL_CIRCUITS or _synthetic_curvature(c) is not None for c in CIRCUITS))
gui_bad = []
for c in CIRCUITS:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # QA warnings are allowed here
            _load_track(c, "Circuits")
    except Exception as e:                       # any other error fails the check
        gui_bad.append((c, repr(e)))
if gui_bad:
    print("  load errors:", gui_bad)
ok("GUI list: every circuit loads (QA warnings allowed)", not gui_bad)
# --- F2: Advanced-tab options (run mode, physics, start state, homotopy, "More options" JSON box) ----------
import json, contextlib, io, tempfile
from app.mainwindow import MORE_KEYS
from app import results
from functions.ladder import LADDERS, resolve_ladder

print("F2 options")
combos = win.opt_combos
ok("combos: Run, Physics and Start state (the OPTION_COMBOS widgets)", set(combos) == {"mode", "physics", "start_state"})
ok("Run combo lists nlp (default) then qss", items(combos["mode"]) == ["nlp", "qss"])
ok("Physics combo lists matlab (default) then corrected", items(combos["physics"]) == ["matlab", "corrected"])
ok("Start state combo lists matlab (default), rolling, periodic",
   items(combos["start_state"]) == ["matlab", "rolling", "periodic"])


def accepts(circuit="Sturn", **kw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            userOpts(Ctx(), circuit=circuit, **kw)
        return True
    except ValueError:
        return False


ok("every Physics and Start state item is accepted by userOpts (periodic on a closed circuit)",
   all(accepts(physics=v) for v in items(combos["physics"]))
   and all(accepts(circuit="Circle" if v == "periodic" else "Sturn", start_state=v)
           for v in items(combos["start_state"])))
ok("that acceptance check can fail", not accepts(physics="bogus") and not accepts(start_state="periodic"))

F2F = ("mode", "physics", "start_state", "n_bounds", "Steering", "CamberGain", "ladder", "homotopy", "refine",
       "ipopt_overrides", "pt_overrides", "between_knots", "collocation", "state_scaling", "atd_tbr", "continue_from")


def cfg_of(rc):
    with tempfile.TemporaryDirectory(prefix="gui_cfg_") as d:
        return rc.write_cfg(os.path.join(d, "cfg.json"))


fresh_win = MainWindow()                                 # a fresh window: startup state
rc0 = fresh_win.collect_runconfig()
ok("startup: every F2 field equals the RunConfig default", all(getattr(rc0, k) == getattr(RunConfig(), k) for k in F2F))
ok("startup: the cfg has no vp or A5 override", cfg_of(rc0)["vp_overrides"] == {})

for k, v in dict(mode="qss", physics="corrected", start_state="rolling").items():
    combos[k].setCurrentText(v)
win.homotopy.setChecked(True)
rc1 = win.collect_runconfig()
ok("collect carries the chosen combos and homotopy True",
   (rc1.mode, rc1.physics, rc1.start_state, rc1.homotopy) == ("qss", "corrected", "rolling", True))

# the "More options" JSON box: RunConfig fields without a widget of their own
MORE = {"n_bounds": 4.5, "ladder": "qss23", "ipopt_overrides": {"mu_strategy": "monotone", "max_iter": 400},
        "Steering": "Ackermann", "CamberGain": "Linear", "pt_overrides": {"Tmax": 301, "Preg": 5e4},
        "between_knots": "linear", "collocation": "radau", "state_scaling": "typical", "atd_tbr": 2.0,
        "continue_from": "auto"}
ok("More options keys are exactly the eleven of the ruling (no refine, no widget key)",
   set(MORE_KEYS) == {"n_bounds", "ladder", "ipopt_overrides", "Steering", "CamberGain", "pt_overrides",
                      "between_knots", "collocation", "state_scaling", "atd_tbr", "continue_from"})
ok("MORE_KEYS are RunConfig fields", set(MORE_KEYS) <= set(RunConfig.__dataclass_fields__))
ok("the ladder names the box may take resolve (auto and every LADDERS key)",
   all(resolve_ladder(v)[0] for v in ("auto",) + tuple(LADDERS)))
win.more_json.setPlainText(json.dumps(MORE))
rc2 = win.collect_runconfig()
ok("collect carries every JSON key into its RunConfig field", all(getattr(rc2, k) == v for k, v in MORE.items()))
ok("... and leaves the combos and the checkbox alone", (rc2.mode, rc2.physics, rc2.homotopy) == ("qss", "corrected", True))
cfg_w = cfg_of(rc2)
ok("the cfg carries the box, the A5 keys inside vp_overrides only",
   cfg_w["n_bounds"] == 4.5 and cfg_w["ladder"] == "qss23" and cfg_w["vp_overrides"] == {"Tmax": 301, "Preg": 5e4}
   and "pt_overrides" not in cfg_w)

win.more_json.setPlainText("")
rc3 = win.collect_runconfig()
ok("empty box: no JSON field set", rc3.n_bounds is None and rc3.pt_overrides is None and rc3.continue_from is None)
for bad, why in (("{'a': 1}", "not valid JSON"), ("not json", "not valid JSON"), ("[1, 2]", "must be a JSON object"),
                 ('{"refine": true}', "unknown key"), ('{"physics": "corrected"}', "unknown key"),
                 ('{"mode": "qss"}', "unknown key"), ('{"start_state": "rolling"}', "unknown key"),
                 ('{"homotopy": true}', "unknown key"), ('{"nope": 1}', "unknown key")):
    win.more_json.setPlainText(bad)
    try:
        win.collect_runconfig()
        msg = ""
    except ValueError as exc:
        msg = str(exc)
    ok(f"More options {bad!r} raises ValueError ({why})", msg.startswith("More options") and why in msg)

# Run: a bad box logs an [error] and starts nothing; a good run clears the last result on show
real_start, started = win._runner.start, []
win._runner.start = started.append                       # records the cfg path, starts no process
prev_out = win.output_dir.text()
try:
    win.log.clear()
    win.more_json.setPlainText('{"refine": true}')
    win._on_run()
    ok("Run with a bad box logs an [error], starts nothing and stays idle",
       "[error] More options" in win.log.toPlainText() and started == []
       and win.run_btn.isEnabled() and not win.cancel_btn.isEnabled())
    win.more_json.setPlainText("")
    with tempfile.TemporaryDirectory(prefix="gui_run_") as run_out:
        win.output_dir.setText(run_out)
        win.summary.setText("Lap time: 1.000 s")
        win.plot_list.addItem("stale.html")
        win._on_run()
        ok("Run clears the last summary and plot list (a failed run must not leave an old lap on show)",
           win.summary.text() == "No results yet." and win.plot_list.count() == 0 and len(started) == 1
           and not win.run_btn.isEnabled())
finally:
    win._runner.start = real_start
    win.output_dir.setText(prev_out)
    win.run_btn.setEnabled(True)
    win.cancel_btn.setEnabled(False)

# the Output tab reads the file that a qss run writes (numpy only; needs Data/DATA_AA.mat)
if os.path.isfile(os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")):
    from MLTP_screen import MLTP_screen
    with tempfile.TemporaryDirectory(prefix="gui_qss_") as out:
        with contextlib.redirect_stdout(io.StringIO()):
            MLTP_screen(circuit="Sturn", save=True, results_dir=os.path.join(out, "Results"))
        lap = results.parse_summary(os.path.join(out, "Results", "Sturn_Static_ATDOn_EM4Off_qss.mat"))["lap_time_s"]
        nlp_plots = os.path.join(out, "Plots", "Sturn", "Static_ATDOn_EM4Off")     # an NLP run of the same setup
        os.makedirs(nlp_plots)
        open(os.path.join(nlp_plots, "speed.html"), "w").close()
        win._show_results(RunConfig(output_dir=out))
        ok("the nlp run of that setup lists its plot", win.plot_list.count() == 1)
        win._show_results(RunConfig(mode="qss", output_dir=out))
        ok("qss summary: labelled as an estimate, with the QSS lap, and no plots (not the NLP plots of the setup)",
           win.summary.text().startswith(f"QSS estimate (not an optimum): {lap:.3f} s") and win.plot_list.count() == 0)
        win._show_results(RunConfig(mode="qss", physics="corrected", output_dir=out))
        ok("a stem the solve did not write is reported, not crashed on", win.summary.text().startswith("Results unreadable"))
else:
    print("  [SKIP] QSS summary: Data/DATA_AA.mat missing (the qss run needs ctx.aero)")

# back to the startup state, then the 4 Motors tick
for k in ("mode", "physics", "start_state"):
    combos[k].setCurrentText(items(combos[k])[0])
win.more_json.setPlainText("")
win.homotopy.setChecked(False)
ok("back to the defaults: the collected F2 fields equal the RunConfig defaults",
   all(getattr(win.collect_runconfig(), k) == getattr(RunConfig(), k) for k in F2F))
win.set_em4(True)
ok("4 Motors on ticks Homotopy (the GUI default for four motors)",
   win.homotopy.isChecked() and win.collect_runconfig().homotopy is True)
win.homotopy.setChecked(False)
ok("the user can untick Homotopy with 4 Motors on", win.collect_runconfig().homotopy is None)
win.set_em4(False)
ok("4 Motors off unticks Homotopy again (the one-motor default is unchanged)",
   not win.homotopy.isChecked() and win.collect_runconfig().homotopy is None)
ok("the library default stays off for four motors (RunConfig homotopy None)",
   RunConfig(Electric_4Motors="On").homotopy is None)
win.atd.setChecked(True)

print("\nALL GUI logic TESTS PASSED")
