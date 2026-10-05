"""Offscreen GUI logic: window builds, collect_runconfig reflects widgets,
EM4-on disables the ATD control (the config-conflict rule)."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))
from PySide6.QtWidgets import QApplication
from app.mainwindow import MainWindow
from app.runconfig import RunConfig

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
print("\nALL GUI logic TESTS PASSED")
