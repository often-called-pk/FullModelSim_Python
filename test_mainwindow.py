"""MainWindow constructs headlessly (offscreen Qt); Setup tab exposes all 110
vehicle params; collect_runconfig + reset behave correctly; the Tyre set / Mesh
combos reach RunConfig / cfg.json and the Setup table (defaults, reset,
changed-value highlighting) follows the selected tyre set."""
import os, sys, shutil, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))
from PySide6.QtWidgets import QApplication
from app.mainwindow import MainWindow
from app.vp_params import all_vp_defaults
from app import results
import headless_solve
from vehParams import PRIMARY_KEYS, MF_KEYS, _MF205_OVERRIDES

ROOT = os.path.dirname(os.path.abspath(__file__))

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

app = QApplication.instance() or QApplication([])
win = MainWindow()

ok("vp_spins covers all 110 keys", set(win.vp_spins) == (PRIMARY_KEYS | MF_KEYS))

rc = win.collect_runconfig()
ok("collect vp has 110 keys", set(rc.vp) == (PRIMARY_KEYS | MF_KEYS))
ok("collect vp equals defaults (no rounding loss)",
   all(rc.vp[k] == float(all_vp_defaults()[k]) for k in rc.vp))
ok("no spurious overrides at startup", rc.vp_overrides() == {})

# change one param -> exactly that key appears in overrides
win.vp_spins["alpha_RW"].setValue(12.0)
rc2 = win.collect_runconfig()
ov = rc2.vp_overrides()
ok("changed param in overrides", ov.get("alpha_RW") == 12.0)
ok("only the changed param overridden", set(ov) == {"alpha_RW"})

# a tiny Pacejka coeff round-trips through the widget
win.vp_spins["rHy1"].setValue(-1.2345e-10)
ok("sci widget round-trips tiny value", win.vp_spins["rHy1"].value() == -1.2345e-10)

# reset restores defaults -> overrides empty again
win._reset_all_params()
rc3 = win.collect_runconfig()
ok("reset clears overrides", rc3.vp_overrides() == {})

# --- Advanced-tab solver/collocation widgets wired into RunConfig ---
win._reset_all_params()
rc4 = win.collect_runconfig()
ok("max_iter default wired", rc4.max_iter == 6000)
ok("OPT_ds default wired", rc4.OPT_ds == 30.0)
ok("OPT_d default wired", rc4.OPT_d == 3)
ok("OPT_e default wired", rc4.OPT_e == 1e-2)
ok("tol default wired", rc4.tol == 1e-4)

win.max_iter.setValue(2500)
win.opt_ds.setValue(25.0)
win.opt_d.setValue(4)
win.opt_e.setValue(2e-3)
win.tol.setValue(5e-5)
rc5 = win.collect_runconfig()
ok("max_iter override wired", rc5.max_iter == 2500)
ok("OPT_ds override wired", rc5.OPT_ds == 25.0)
ok("OPT_d override wired", rc5.OPT_d == 4)
ok("OPT_e override wired", rc5.OPT_e == 2e-3)
ok("tol override wired", rc5.tol == 5e-5)

# --- Advanced-tab Tyre set / Mesh combos: RunConfig, cfg.json, Setup table ----
LAT = "Pacejka 5.2, lateral"
MF = {k: float(v) for k, v in all_vp_defaults("MF205").items()}
CB = {k: float(v) for k, v in all_vp_defaults("CopyB").items()}
LAT9 = set(_MF205_OVERRIDES)

def type_into(key, text):
    """Simulate a user typing into a Pacejka (ScientificField) cell: text, then editingFinished."""
    field = win.vp_spins[key]
    field._edit.setText(text)
    field._edit.editingFinished.emit()

def flagged(key):
    field = win.vp_spins[key]
    return bool(getattr(field, "_edit", field).styleSheet())

def n_flagged():
    return sum(flagged(k) for k in win.vp_spins)

def badges():
    return [t for t, s in win._sections.items() if "changed" in s._btn.text()]

ok("the two tyre sets differ in exactly the nine lateral coefficients",
   {k for k in MF if MF[k] != CB[k]} == LAT9 and len(LAT9) == 9)

win._reset_all_params()
rc6 = win.collect_runconfig()
ok("MF205 (default): tyre_set carried, vp['pKy4'] == 2.0, no overrides",
   rc6.tyre_set == "MF205" and rc6.vp["pKy4"] == 2.0 and rc6.vp_overrides() == {})

win.tyre_set.setCurrentText("CopyB")
rcb = win.collect_runconfig()
ok("CopyB selected: RunConfig.tyre_set == 'CopyB'", rcb.tyre_set == "CopyB")
ok("CopyB: the pKy4 field is re-seeded to 0.0", win.vp_spins["pKy4"].value() == 0.0)
ok("CopyB: collected vp['pKy4'] == 0.0", rcb.vp["pKy4"] == 0.0)
ok("CopyB: all nine lateral coefficients re-seeded", all(rcb.vp[k] == CB[k] for k in LAT9))
ok("CopyB: the whole collected vp equals the CopyB defaults", rcb.vp == CB)
ok("CopyB: vp_overrides() == {} (a re-seeded table is not an edit)", rcb.vp_overrides() == {})
ok("CopyB: nothing flagged as changed, no section badge", n_flagged() == 0 and badges() == [])

# cfg.json / headless kwargs / results path carry the selection
tmp = tempfile.mkdtemp()                      # scratch dir: no stray cfg file in the repo
cfg = rcb.write_cfg(os.path.join(tmp, "cfg.json"))
ok("cfg.json: tyre_set is a top-level field, not a vp override",
   cfg["tyre_set"] == "CopyB" and cfg["vp_overrides"] == {})
kw = headless_solve.build_solve_kwargs(cfg, ROOT)
ok("headless kwargs: tyre_set reaches MLTP, no vp_overrides",
   kw["tyre_set"] == "CopyB" and kw["vp_overrides"] is None)
def mat_name(rc):
    return os.path.basename(results.result_mat_path(rc.output_dir, rc.circuit, rc.AeroConfig, rc.ATD,
                                                    rc.Electric_4Motors, rc.tyre_set, rc.mesh))
ok("results lookup follows the tyre set (CopyB stem)", mat_name(rcb) == "Sturn_Static_ATDOn_EM4Off_CopyB.mat")

win.mesh.setCurrentText("curvature")
rcm = win.collect_runconfig()
cfg_m = rcm.write_cfg(os.path.join(tmp, "cfg_m.json"))
ok("Mesh combo reaches RunConfig and cfg.json", rcm.mesh == "curvature" and cfg_m["mesh"] == "curvature")
ok("headless kwargs: mesh reaches MLTP", headless_solve.build_solve_kwargs(cfg_m, ROOT)["mesh"] == "curvature")
ok("mesh is no vp override and leaves the vp table alone", rcm.vp_overrides() == {} and rcm.vp == CB)
ok("results lookup follows tyre set + mesh", mat_name(rcm) == "Sturn_Static_ATDOn_EM4Off_CopyB_meshCurvature.mat")
win.mesh.setCurrentText("auto")
shutil.rmtree(tmp, ignore_errors=True)

win.tyre_set.setCurrentText("MF205")
rcf = win.collect_runconfig()
ok("back to MF205: vp == MF205 defaults, no overrides, nothing flagged",
   rcf.vp == MF and rcf.vp_overrides() == {} and n_flagged() == 0)

# a user edit survives a switch and is judged against the new set's defaults
type_into("pKy4", "1.5")
ok("edit under MF205 is flagged and is the only override",
   flagged("pKy4") and win.collect_runconfig().vp_overrides() == {"pKy4": 1.5})
win.tyre_set.setCurrentText("CopyB")
ok("CopyB: the user edit is kept across the switch", win.vp_spins["pKy4"].value() == 1.5)
ok("CopyB: ... and flagged against the CopyB default (1.5 != 0.0)", flagged("pKy4"))
ok("CopyB: the untouched lateral coefficients follow the new set",
   all(win.vp_spins[k].value() == CB[k] for k in LAT9 - {"pKy4"}))
ok("CopyB: only the edit is an override", win.collect_runconfig().vp_overrides() == {"pKy4": 1.5})
ok("CopyB: only the edit is flagged, and the lateral badge counts it",
   n_flagged() == 1 and badges() == [LAT] and "(1 changed)" in win._sections[LAT]._btn.text())
win._sections[LAT].set_changed_count(0)
win._update_section_badge(LAT)               # no explicit defaults: must use the selected set's
ok("badge refresh without explicit defaults follows the selected set",
   "(1 changed)" in win._sections[LAT]._btn.text())

# an edit that equals the new default is kept but stops being a change
win._reset_all_params()
win.tyre_set.setCurrentText("MF205")
type_into("pKy4", "0")
ok("typing the CopyB value under MF205 is a flagged edit",
   flagged("pKy4") and win.collect_runconfig().vp_overrides() == {"pKy4": 0.0})
win.tyre_set.setCurrentText("CopyB")
ok("CopyB: an edit equal to the new default is kept, no longer flagged or overridden",
   win.vp_spins["pKy4"].value() == 0.0 and not flagged("pKy4") and badges() == []
   and win.collect_runconfig().vp_overrides() == {})

# typing is judged against the selected set (same value, opposite verdict per set)
type_into("pKy4", "2")
ok("under CopyB the MF205 value 2.0 is a change", flagged("pKy4"))
win.tyre_set.setCurrentText("MF205")
ok("under MF205 the same 2.0 is the default again", win.vp_spins["pKy4"].value() == 2.0 and not flagged("pKy4"))

# an unrelated edit survives a switch too
win.vp_spins["alpha_RW"].setValue(12.0)
win.tyre_set.setCurrentText("CopyB")
ok("an unrelated edit survives the switch and stays the only override",
   win.vp_spins["alpha_RW"].value() == 12.0 and win.collect_runconfig().vp_overrides() == {"alpha_RW": 12.0})

# Reset follows the selected set
type_into("pKy1", "-10")
type_into("pKy4", "1")
win._reset_section(LAT)
ok("section Reset under CopyB restores the CopyB lateral values, other sections untouched",
   all(win.vp_spins[k].value() == CB[k] for k in LAT9)
   and win.collect_runconfig().vp_overrides() == {"alpha_RW": 12.0})
type_into("pKy4", "1")
win._reset_all_params()
rcr = win.collect_runconfig()
ok("Reset all under CopyB restores the whole CopyB set (pKy4 == 0.0)",
   rcr.tyre_set == "CopyB" and rcr.vp == CB and rcr.vp_overrides() == {} and n_flagged() == 0)

# a loaded preset / expert value is judged against the selected set too
win._apply_param_dict({"pKy4": 2.0, "mb": 1900.0})
ok("a loaded MF205 lateral value under CopyB is flagged and overridden",
   flagged("pKy4") and flagged("mb")
   and win.collect_runconfig().vp_overrides() == {"pKy4": 2.0, "mb": 1900.0})
win._reset_all_params()

# half-typed text is left alone by the re-seed; collect falls back to the selected set's default
win.tyre_set.setCurrentText("MF205")
win.vp_spins["pKy1"]._edit.setText("-")
win.tyre_set.setCurrentText("CopyB")
ok("a half-typed cell is left alone by the re-seed", win.vp_spins["pKy1"]._edit.text() == "-")
rch = win.collect_runconfig()
ok("collect falls back to the CopyB default for unparsable text (pKy1 == -20.505)",
   rch.vp["pKy1"] == CB["pKy1"] and rch.vp_overrides() == {})

win.tyre_set.setCurrentText("MF205")
win._reset_all_params()
rc_end = win.collect_runconfig()
ok("Reset all under MF205 restores the MF205 set (pKy4 == 2.0)",
   rc_end.tyre_set == "MF205" and rc_end.vp == MF and rc_end.vp["pKy4"] == 2.0
   and rc_end.vp_overrides() == {} and n_flagged() == 0 and badges() == [])

print("\nALL MainWindow TESTS PASSED")
