"""Static lint of the PyInstaller spec: required data dirs + collected packages
are referenced so a frozen build can find them."""
import os, sys
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

spec = open(os.path.join(_bootstrap.ROOT, "build", "windows-app.spec")).read()
for token in ("Circuits", "Data", "app/presets", "casadi", "PySide6", "headless_solve"):
    ok(f"spec references {token}", token in spec)
ok("spec sets app name", "name='FullModelSim'" in spec or 'name="FullModelSim"' in spec)
# F2: a frozen exe re-runs itself for every multiprocessing worker (MLTP_screen.screen_batch with
# workers > 1). The first statement of the __main__ block must be freeze_support(), before main() builds a QApplication.
import ast
with open(os.path.join(_bootstrap.ROOT, "app", "main.py")) as fh:
    main_tree = ast.parse(fh.read())
guard = next(n for n in main_tree.body if isinstance(n, ast.If) and "__main__" in ast.dump(n.test))
ok("app/main.py: the first statement of the __main__ block is multiprocessing.freeze_support()",
   ast.unparse(guard.body[0]) == "multiprocessing.freeze_support()")
ok("app/main.py imports multiprocessing at module level",
   any(isinstance(n, ast.Import) and any(a.name == "multiprocessing" for a in n.names) for n in main_tree.body))
with open(os.path.join(_bootstrap.ROOT, "headless_solve.py")) as fh:
    hs_tree = ast.parse(fh.read())
ok("headless_solve imports MLTP_screen statically (PyInstaller finds a static import, not a dynamic one)",
   any(isinstance(n, ast.ImportFrom) and n.module == "MLTP_screen" for n in ast.walk(hs_tree)))

print("\nALL spec-include TESTS PASSED")
