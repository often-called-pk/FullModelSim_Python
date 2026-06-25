"""MainWindow constructs headlessly (offscreen Qt); Setup tab exposes all 110
vehicle params; collect_runconfig + reset behave correctly."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))
from PySide6.QtWidgets import QApplication
from app.mainwindow import MainWindow
from app.vp_params import all_vp_defaults
from vehParams import PRIMARY_KEYS, MF_KEYS

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

print("\nALL MainWindow TESTS PASSED")
