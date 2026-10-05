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
print("\nALL GUI logic TESTS PASSED")
