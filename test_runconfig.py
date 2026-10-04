"""RunConfig: vp-dict round-trip, diff-from-default -> vp_overrides, expert-file
merge, solver fields at top level (not in vp_overrides), unknown-key rejection."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from app.runconfig import RunConfig
from app.vp_params import all_vp_defaults
from vehParams import _MF205_OVERRIDES

TMP = os.environ.get("CLAUDE_JOB_DIR_TMP", os.path.dirname(__file__))

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# round-trip
vp = dict(all_vp_defaults()); vp["alpha_RW"] = 12.0
rc = RunConfig(circuit="BCN", vi=55.0, vp=vp)
ok("round-trip from_dict(to_dict)", RunConfig.from_dict(rc.to_dict()) == rc)

# vp diff -> vp_overrides; cfg drops raw vp; run fields kept
cfg = rc.write_cfg(os.path.join(TMP, "_rc_cfg.json"))
ok("cfg has vp_overrides", "vp_overrides" in cfg)
ok("alpha_RW folded into overrides", cfg["vp_overrides"]["alpha_RW"] == 12.0)
ok("only the diff is overridden", set(cfg["vp_overrides"]) == {"alpha_RW"})
ok("no raw vp dict in cfg", "vp" not in cfg)
ok("cfg keeps run fields", cfg["circuit"] == "BCN" and cfg["vi"] == 55.0)

# solver fields: top-level defaults, never inside vp_overrides
for k, d in [("max_iter", 6000), ("OPT_ds", 30.0), ("OPT_d", 3), ("OPT_e", 1e-2), ("tol", 1e-4)]:
    ok(f"{k} top-level default", cfg[k] == d)
    ok(f"{k} not in vp_overrides", k not in cfg["vp_overrides"])

# solver overrides carried through
rc_s = RunConfig(max_iter=3000, OPT_ds=20.0, OPT_d=4, OPT_e=5e-3, tol=1e-6)
cfg_s = rc_s.write_cfg(os.path.join(TMP, "_rc_cfg_s.json"))
ok("max_iter override", cfg_s["max_iter"] == 3000)
ok("OPT_ds override", cfg_s["OPT_ds"] == 20.0)
ok("OPT_d override", cfg_s["OPT_d"] == 4)
ok("tol override", cfg_s["tol"] == 1e-6)

# speed / fidelity options: top-level cfg fields with userOpts defaults, never in vp_overrides
for k, d in [("mesh", "auto"), ("mesh_opts", None), ("tyre_set", "MF205"), ("screening", False)]:
    ok(f"{k} top-level default", cfg[k] == d)
    ok(f"{k} not in vp_overrides", k not in cfg["vp_overrides"])
rc_f = RunConfig(mesh="curvature", mesh_opts={"a": 2.0}, tyre_set="CopyB", screening=True)
fpath = os.path.join(tempfile.mkdtemp(), "cfg_f.json")       # scratch dir: no stray file in the repo
cfg_f = rc_f.write_cfg(fpath)
ok("mesh override carried", cfg_f["mesh"] == "curvature")
ok("mesh_opts override carried", cfg_f["mesh_opts"] == {"a": 2.0})
ok("tyre_set override carried", cfg_f["tyre_set"] == "CopyB")
ok("screening override carried", cfg_f["screening"] is True)
ok("speed options survive the JSON file", json.load(open(fpath)) == cfg_f)
ok("speed options round-trip from_dict(to_dict)", RunConfig.from_dict(rc_f.to_dict()) == rc_f)
ok("an old saved config without the new fields still loads (defaults)",
   RunConfig.from_dict({"circuit": "BCN"}).mesh == "auto"
   and RunConfig.from_dict({"circuit": "BCN"}).tyre_set == "MF205")

rc_cb = RunConfig(tyre_set="CopyB")
ok("CopyB seeds vp from the CopyB set (pKy4 == 0.0)", rc_cb.vp["pKy4"] == 0.0)
ok("CopyB untouched vp gives no overrides", rc_cb.vp_overrides() == {})
rc_cb.vp["pKy4"] = 2.0
ok("MF205 value entered under CopyB is an explicit override", rc_cb.vp_overrides() == {"pKy4": 2.0})
ok("default RunConfig seeds MF205 (pKy4 == 2.0) with no overrides",
   RunConfig().vp["pKy4"] == 2.0 and RunConfig().vp_overrides() == {})
cb = all_vp_defaults("CopyB")
rc_old = RunConfig.from_dict(json.loads(json.dumps({"circuit": "BCN", "vp": cb})))
ok("old JSON with full CopyB vp and no tyre_set key reproduces the stored vp",
   rc_old.tyre_set == "MF205" and rc_old.vp == cb)
ok("old CopyB-valued vp under default MF205 becomes the nine explicit overrides",
   rc_old.vp_overrides() == {k: cb[k] for k in _MF205_OVERRIDES} and len(rc_old.vp_overrides()) == 9)

# expert-file merge: expert provides mb + pKy1; GUI vp value overrides expert's brkB
expert = {"mb": 2000.0, "pKy1": -19.0, "brkB": 0.5}
epath = os.path.join(TMP, "_expert.json"); json.dump(expert, open(epath, "w"))
vp2 = dict(all_vp_defaults()); vp2["brkB"] = 0.7
rc2 = RunConfig(expert_config=epath, vp=vp2)
ov = rc2.vp_overrides()
ok("expert mb present", ov["mb"] == 2000.0)
ok("expert mf key present", ov["pKy1"] == -19.0)
ok("GUI vp overrides expert brkB", ov["brkB"] == 0.7)

# unknown key in expert file raises
bad = os.path.join(TMP, "_bad.json"); json.dump({"not_a_param": 1.0}, open(bad, "w"))
raised = False
try:
    RunConfig(expert_config=bad).vp_overrides()
except ValueError:
    raised = True
ok("unknown expert key raises", raised)

# unknown key inside the vp dict raises a friendly ValueError (not KeyError)
vp_bad = dict(all_vp_defaults()); vp_bad["bogus_key"] = 1.0
raised_vp = False
try:
    RunConfig(vp=vp_bad).vp_overrides()
except ValueError:
    raised_vp = True
except KeyError:
    raised_vp = False
ok("unknown vp key raises ValueError not KeyError", raised_vp)

print("\nALL RunConfig TESTS PASSED")
