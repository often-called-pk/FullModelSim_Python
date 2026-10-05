"""RunConfig: vp-dict round-trip, diff-from-default -> vp_overrides, expert-file
merge, solver fields at top level (not in vp_overrides), unknown-key rejection."""
import os, sys, json, tempfile, atexit, shutil
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
from app.runconfig import RunConfig
from app.vp_params import all_vp_defaults
from vehParams import _MF205_OVERRIDES

# scratch files: CLAUDE_JOB_DIR_TMP when set, else a fresh temp dir removed at exit (never the repo)
TMP = os.environ.get("CLAUDE_JOB_DIR_TMP")
if not TMP:
    TMP = tempfile.mkdtemp(prefix="runconfig_test_")
    atexit.register(shutil.rmtree, TMP, ignore_errors=True)

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

# expert-file merge: the table (rc.vp) alone decides every key it holds; the expert file
# only contributes keys absent from it
expert = {"mb": 2000.0, "pKy1": -19.0, "brkB": 0.5}
epath = os.path.join(TMP, "_expert.json"); json.dump(expert, open(epath, "w"))
vp2 = dict(all_vp_defaults()); vp2["brkB"] = 0.7
rc2 = RunConfig(expert_config=epath, vp=vp2)
ov = rc2.vp_overrides()
ok("full table: expert mb does not override a table value at its default", "mb" not in ov)
ok("full table: expert mf key does not override a table value at its default", "pKy1" not in ov)
ok("GUI vp overrides expert brkB", ov["brkB"] == 0.7)
ok("full table: the table edit is the only override", ov == {"brkB": 0.7})
ok("an explicit vp is not rewritten by the expert file",
   rc2.vp["mb"] == all_vp_defaults()["mb"] and rc2.vp["pKy1"] == all_vp_defaults()["pKy1"] and rc2.vp["brkB"] == 0.7)
vp_sp = {k: v for k, v in vp2.items() if k not in ("mb", "pKy1")}
ov_sp = RunConfig(expert_config=epath, vp=vp_sp).vp_overrides()
ok("expert-only primary key absent from the table still passes through", ov_sp["mb"] == 2000.0)
ok("expert-only mf key absent from the table still passes through", ov_sp["pKy1"] == -19.0)
ok("sparse table: table brkB still overrides expert brkB", ov_sp["brkB"] == 0.7)
rc_e = RunConfig(expert_config=epath)
ok("default vp + expert file: the expert values are written into rc.vp", rc_e.vp == {**all_vp_defaults(), **expert})
ok("default vp + expert file: vp_overrides() carries exactly the expert values", rc_e.vp_overrides() == expert)
epath_k = os.path.join(tempfile.mkdtemp(), "_expert_pky4.json"); json.dump({"pKy4": 2.0}, open(epath_k, "w"))
rc_kd = RunConfig(tyre_set="CopyB", expert_config=epath_k)
ok("default vp under CopyB: the table shows the expert pKy4 == 2.0, so the override is visible",
   rc_kd.vp["pKy4"] == 2.0 and rc_kd.vp_overrides() == {"pKy4": 2.0})
rc_k = RunConfig(tyre_set="CopyB", expert_config=epath_k, vp=all_vp_defaults("CopyB"))
ok("CopyB table shows pKy4 == 0.0 with an expert pKy4 == 2.0 loaded", rc_k.vp["pKy4"] == 0.0)
ok("expert pKy4 does not override the CopyB table value the user sees", "pKy4" not in rc_k.vp_overrides())

# unknown key in expert file raises
bad = os.path.join(TMP, "_bad.json"); json.dump({"not_a_param": 1.0}, open(bad, "w"))
raised = False
try:
    RunConfig(expert_config=bad).vp_overrides()
except ValueError:
    raised = True
ok("unknown expert key raises", raised)
raised_ev = False
try:
    RunConfig(expert_config=bad, vp=dict(all_vp_defaults())).vp_overrides()
except ValueError:
    raised_ev = True
ok("unknown expert key raises through vp_overrides() with an explicit vp", raised_ev)

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
