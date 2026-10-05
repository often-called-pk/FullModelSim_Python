"""Shipped presets are valid override files and default.json reproduces the
baseline car."""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(__file__))
from functions.context import Ctx
from vehParams import vehParams, default_primaries, PRIMARY_KEYS, MF_KEYS

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

ppath = os.path.join(os.path.dirname(__file__), "app", "presets", "default.json")
ok("default.json exists", os.path.exists(ppath))
preset = json.load(open(ppath))
ok("all keys are known overrides", set(preset) <= (PRIMARY_KEYS | MF_KEYS))

# default.json must equal the baseline primaries (it is the baseline car)
base = default_primaries()
ok("default.json matches baseline primaries", all(preset[k] == base[k] for k in preset))

# loading it through vehParams reproduces the baseline (ms=1895)
ctx = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx, vp_overrides=preset)
ok("baseline ms reproduced", ctx.vp.ms == 1895.0)
print("\nALL preset TESTS PASSED")
