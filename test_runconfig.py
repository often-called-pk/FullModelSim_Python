"""RunConfig: round-trip, Tier-1 -> vp_overrides folding, expert-file merge,
unknown-key rejection."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from app.runconfig import RunConfig, TIER1_FIELDS

TMP = os.environ.get("CLAUDE_JOB_DIR_TMP", os.path.dirname(__file__))

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# round-trip
rc = RunConfig(circuit="BCN", vi=55.0, alpha_RW=12.0)
ok("round-trip from_dict(to_dict)", RunConfig.from_dict(rc.to_dict()) == rc)

# Tier-1 folds into vp_overrides, top-level Tier-1 keys dropped from cfg
cfg = rc.write_cfg(os.path.join(TMP, "_rc_cfg.json")) or json.load(
    open(os.path.join(TMP, "_rc_cfg.json")))
ok("cfg has vp_overrides", "vp_overrides" in cfg)
ok("alpha_RW folded into overrides", cfg["vp_overrides"]["alpha_RW"] == 12.0)
ok("no top-level Tier-1 key in cfg", all(k not in cfg for k in TIER1_FIELDS))
ok("cfg keeps run fields", cfg["circuit"] == "BCN" and cfg["vi"] == 55.0)

# expert file merge: expert provides mb + pKy1; Tier-1 GUI value overrides expert's brkB
expert = {"mb": 2000.0, "pKy1": -19.0, "brkB": 0.5}
epath = os.path.join(TMP, "_expert.json")
json.dump(expert, open(epath, "w"))
rc2 = RunConfig(expert_config=epath, brkB=0.7)
ov = rc2.vp_overrides()
ok("expert mb present", ov["mb"] == 2000.0)
ok("expert mf key present", ov["pKy1"] == -19.0)
ok("Tier-1 overrides expert brkB", ov["brkB"] == 0.7)

# unknown key in expert file raises
bad = os.path.join(TMP, "_bad.json")
json.dump({"not_a_param": 1.0}, open(bad, "w"))
raised = False
try:
    RunConfig(expert_config=bad).vp_overrides()
except ValueError:
    raised = True
ok("unknown expert key raises", raised)

print("\nALL RunConfig TESTS PASSED")
