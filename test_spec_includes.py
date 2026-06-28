"""Static lint of the PyInstaller spec: required data dirs + collected packages
are referenced so a frozen build can find them."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

spec = open(os.path.join(os.path.dirname(__file__), "build", "windows-app.spec")).read()
for token in ("Circuits", "Data", "app/presets", "casadi", "PySide6", "headless_solve"):
    ok(f"spec references {token}", token in spec)
ok("spec sets app name", "name='FullModelSim'" in spec or 'name="FullModelSim"' in spec)
print("\nALL spec-include TESTS PASSED")
