"""solve_command (dev branch) + headless arg dispatch. Offscreen Qt."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
from app.solve_runner import solve_command
from app.main import is_headless
from app.paths import resource_root

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

cmd = solve_command("cfg.json")
ok("dev command runs headless_solve.py",
   cmd[1] == os.path.join(resource_root(), "headless_solve.py") and cmd[-1] == "cfg.json")
ok("dev command uses python", cmd[0] == sys.executable)

ok("is_headless true", is_headless(["app", "--headless", "cfg.json"]))
ok("is_headless false", not is_headless(["app"]))
print("\nALL solve_runner TESTS PASSED")
