"""Tests for the MPI sweep driver's pure logic (functions/sweep.py) and
run_sweep.py's serial path. No casadi, no mpi4py — a stub solve fn is injected.
Plain-script style (CLAUDE.md): run with `python test_sweep.py`."""
import os, sys, csv, tempfile
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import sweep

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

def _write_csv(rows, header):
    fd, path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header); w.writeheader()
        for r in rows:
            w.writerow(r)
    return path

print("read_cases + case_kwargs")
hdr = ["case_id", "circuit", "vi", "ATD", "Electric_4Motors", "linear_solver"]
path = _write_csv([
    {"case_id": "0", "circuit": "BCN", "vi": "40", "ATD": "On",
     "Electric_4Motors": "Off", "linear_solver": "ma57"},
    {"case_id": "1", "circuit": "Spa", "vi": "60", "ATD": "",
     "Electric_4Motors": "On", "linear_solver": ""},   # blanks -> dropped
], hdr)
cases = sweep.read_cases(path)
ok("two cases parsed", len(cases) == 2)
ok("case_id preserved", cases[0]["case_id"] == "0")
ok("blank ATD dropped", "ATD" not in cases[1])
ok("blank linear_solver dropped", "linear_solver" not in cases[1])

kw = sweep.case_kwargs(cases[0])
ok("vi coerced to float", isinstance(kw["vi"], float) and kw["vi"] == 40.0)
ok("plot forced False", kw["plot"] is False)
ok("case_id not a kwarg", "case_id" not in kw)
ok("circuit passed through", kw["circuit"] == "BCN")

# missing case_id column -> row index used
path2 = _write_csv([{"circuit": "BCN", "vi": "50"}], ["circuit", "vi"])
ok("auto case_id from row index", sweep.read_cases(path2)[0]["case_id"] == "0")

# unknown column rejected
raised = False
try:
    sweep.case_kwargs({"case_id": "9", "bogus": "x"})
except ValueError:
    raised = True
ok("unknown column raises ValueError", raised)
os.remove(path); os.remove(path2)

print("output_dir_for + manifest formatting")
ok("vi-only-differing cases get distinct dirs",
   sweep.output_dir_for("run1", "0") != sweep.output_dir_for("run1", "1"))
ok("output dir shape",
   sweep.output_dir_for("run1", "3").replace("\\", "/")
   == "Results/run1/case_3")

cs = [
    {"case_id": "0", "circuit": "BCN", "vi": "40"},
    {"case_id": "1", "circuit": "Spa", "vi": "60", "ATD": "On"},
]
fields = sweep.manifest_fieldnames(cs)
ok("case_id is first field", fields[0] == "case_id")
ok("input cols before outcome", fields.index("circuit") < fields.index("status"))
ok("ATD column included once", fields.count("ATD") == 1)
ok("outcome fields present at end",
   fields[-len(sweep.OUTCOME_FIELDS):] == list(sweep.OUTCOME_FIELDS))

row = sweep.manifest_row(cs[0], {"status": "ok", "lap_time": 12.3})
ok("manifest_row merges inputs", row["circuit"] == "BCN")
ok("manifest_row merges outcome", row["status"] == "ok" and row["lap_time"] == 12.3)
