"""Run the plain-script tests one file at a time and print a summary.

Usage: python tests/run_tests.py [--quick | --all | test_x.py ...]
--quick (default) skips the files of REAL_IPOPT. They run IPOPT iterations on a vehicle NLP.
The exit code is 1 when a file fails. The last lines of a failed log go to the console.
"""
import subprocess
from collections import deque
import sys
import tempfile
import time
from pathlib import Path

REAL_IPOPT = {"test_setup_sweep.py", "test_paramoptim_warmstart.py", "test_ladder.py",
              "test_refine.py"}
TIMEOUT = 1800  # s per file

tests = Path(__file__).resolve().parent
args = sys.argv[1:] or ["--quick"]
files = sorted(tests.glob("test_*.py"))
if args[0] == "--quick":
    files = [f for f in files if f.name not in REAL_IPOPT]
elif args[0] != "--all":
    files = [tests / a for a in args]

fails = []
for f in files:
    t0 = time.time()
    with tempfile.TemporaryFile("w+") as log:     # the log goes to disk, not to memory
        try:
            rc = subprocess.run([sys.executable, str(f)], cwd=tests.parent, stdout=log,
                                stderr=subprocess.STDOUT, timeout=TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        if rc != 0:
            log.seek(0)
            print("".join(deque(log, maxlen=30)))
    print(f"{'ok' if rc == 0 else f'FAIL({rc})':12s} {time.time() - t0:7.1f} s  {f.name}", flush=True)
    if rc != 0:
        fails.append(f.name)
print(f"{len(files) - len(fails)}/{len(files)} passed" + (f"; failed: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
