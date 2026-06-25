"""app/paths.user_presets_dir resolution."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app.paths import user_presets_dir, default_output_dir

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

ok("presets under output root",
   user_presets_dir() == os.path.join(default_output_dir(), "presets"))
ok("presets basename is 'presets'", os.path.basename(user_presets_dir()) == "presets")
print("\nALL paths TESTS PASSED")
