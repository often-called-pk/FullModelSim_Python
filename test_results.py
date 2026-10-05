"""Results parser: path scheme, .mat summary (lap_time + final energy), plot listing."""
import os, sys
import numpy as np
import scipy.io as sio
sys.path.insert(0, os.path.dirname(__file__))
from app import results

TMP = os.environ.get("CLAUDE_JOB_DIR_TMP", os.path.dirname(__file__))

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# path scheme matches MLTP/plotSDI naming
mp = results.result_mat_path("/out", "Sturn", "Static", "On", "Off")
ok("mat path scheme", mp == os.path.join("/out", "Results", "Sturn_Static_ATDOn_EM4Off.mat"))
pd = results.plot_dir("/out", "Sturn", "Static", "On", "Off")
ok("plot dir scheme", pd == os.path.join("/out", "Plots", "Sturn", "Static_ATDOn_EM4Off"))

# summary from a crafted .mat
data = {"lap_time": 12.5, "vehicle": {"E_motor": np.array([0.0, 1.0, 2.3])}}
mat = os.path.join(TMP, "_res.mat")
sio.savemat(mat, {"data": data}, do_compression=True)
summ = results.parse_summary(mat)
ok("lap_time parsed", abs(summ["lap_time_s"] - 12.5) < 1e-9)
ok("energy parsed (last E_motor)", abs(summ["energy_kWh"] - 2.3) < 1e-9)

# plot listing
pdir = os.path.join(TMP, "_plots")
os.makedirs(pdir, exist_ok=True)
for nm in ("speed.html", "racing_line.html"):
    open(os.path.join(pdir, nm), "w").write("<html></html>")
plots = results.list_plots(pdir)
ok("two plots listed sorted", [os.path.basename(p) for p in plots]
   == ["racing_line.html", "speed.html"])
ok("missing dir -> empty list", results.list_plots(os.path.join(TMP, "_nope")) == [])
print("\nALL results TESTS PASSED")
