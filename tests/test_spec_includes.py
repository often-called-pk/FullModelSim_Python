"""Static lint of the PyInstaller spec: required data dirs + collected packages
are referenced so a frozen build can find them."""
import os, sys
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

spec = open(os.path.join(_bootstrap.ROOT, "build", "app.spec")).read()
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

# Data entries: the spec's datas / binaries are (source, destination) tuples. Parse them as AST so the
# check does not depend on quoting or line breaks. The spec itself is never executed (no PyInstaller run).
from userOpts import _REAL_CIRCUITS
spec_tree = ast.parse(spec)

def _str_consts(node):
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}

def _dest_tuples():
    return [n for n in ast.walk(spec_tree)
            if isinstance(n, ast.Tuple) and len(n.elts) == 2
            and isinstance(n.elts[1], ast.Constant) and isinstance(n.elts[1].value, str)]

def _assigned(name):
    for node in spec_tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise AssertionError(f"spec has no {name} assignment")

circuit_files = ast.literal_eval(_assigned("CIRCUIT_FILES"))
ok("CIRCUIT_FILES has no duplicates", len(circuit_files) == len(set(circuit_files)))
ok("CIRCUIT_FILES == the file names of userOpts._REAL_CIRCUITS",
   set(circuit_files) == set(_REAL_CIRCUITS.values()))
ok("circuit files go to the Circuits destination via CIRCUIT_FILES",
   any(isinstance(c, ast.ListComp) and isinstance(c.elt, ast.Tuple) and "Circuits" in _str_consts(c.elt)
       and any(ast.unparse(g.iter) == "CIRCUIT_FILES" for g in c.generators)
       for c in ast.walk(spec_tree)))
ok("spec bundles Data/DATA_AA.mat into Data",
   any("DATA_AA.mat" in _str_consts(t.elts[0]) and t.elts[1].value == "Data" for t in _dest_tuples()))
ok("spec bundles app/presets into app/presets",
   any("presets" in _str_consts(t.elts[0]) and t.elts[1].value == "app/presets" for t in _dest_tuples()))
hsl_tuples = [t for t in _dest_tuples() if t.elts[1].value == "hsl"]
ok("HSL libraries go to the hsl destination folder",
   any("hsl_dir" in ast.unparse(t.elts[0]) for t in hsl_tuples))
with open(os.path.join(_bootstrap.ROOT, "functions", "hsl.py")) as fh:
    hsl_tree = ast.parse(fh.read())
ok("functions/hsl.py reads the same hsl subfolder name as the spec",
   any(isinstance(n, ast.Constant) and n.value == "hsl" for n in ast.walk(hsl_tree)))

print("\nALL spec-include TESTS PASSED")
