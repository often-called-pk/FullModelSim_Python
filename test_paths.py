"""Validate app/paths.py resource + output resolution (dev / non-frozen)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import paths

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

repo_root = os.path.dirname(os.path.abspath(__file__))
ok("not frozen in dev", paths.is_frozen() is False)
ok("resource_root is repo root", os.path.abspath(paths.resource_root()) == repo_root)
ok("resource_path joins", paths.resource_path("Data", "DATA_AA.mat")
   == os.path.join(paths.resource_root(), "Data", "DATA_AA.mat"))
out = paths.default_output_dir()
ok("output under Documents/FullModelSim",
   out.replace("\\", "/").endswith("Documents/FullModelSim"))
print("\nALL paths TESTS PASSED")
