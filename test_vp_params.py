"""app/vp_params.py registry invariants (pure Python, no Qt)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from vehParams import default_primaries, _default_mf, PRIMARY_KEYS, MF_KEYS
from app.vp_params import PARAM_GROUPS, meta_for, all_vp_defaults, fmt_sci

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

grouped = [k for _, keys in PARAM_GROUPS for k in keys]
ok("no duplicate keys across groups", len(grouped) == len(set(grouped)))
ok("groups cover exactly the override keys", set(grouped) == (PRIMARY_KEYS | MF_KEYS))

exp = {**default_primaries(), **vars(_default_mf())}
ok("all_vp_defaults == primaries+mf merge", all_vp_defaults() == exp)
ok("all_vp_defaults has 110 keys", len(all_vp_defaults()) == len(PRIMARY_KEYS | MF_KEYS))

defs = all_vp_defaults()
ok("lo <= default <= hi for all", all(meta_for(k).lo <= v <= meta_for(k).hi for k, v in defs.items()))
ok("decimals >= 0 for all", all(meta_for(k).decimals >= 0 for k in defs))
ok("kind sci iff mf or eps", all(
    (meta_for(k).kind == "sci") == ((k in MF_KEYS) or k.startswith("eps_")) for k in defs))
ok("spin defaults representable at their decimals", all(
    round(v, meta_for(k).decimals) == v for k, v in defs.items() if meta_for(k).kind == "spin"))
ok("fmt_sci round-trips every default exactly", all(float(fmt_sci(v)) == float(v) for v in defs.values()))

print("\nALL vp_params TESTS PASSED")
