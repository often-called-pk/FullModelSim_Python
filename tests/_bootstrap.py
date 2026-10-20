"""Bootstrap shared by the plain-script tests in this folder (not a test itself).

Every test starts with ``import _bootstrap``: running ``python tests/test_x.py`` puts this
folder on sys.path[0], whatever the working directory is. Importing it makes a test
independent of where it was launched from:

* the repo root goes first on sys.path, so ``from MLTP import MLTP``,
  ``from functions.x import y`` and ``from app.x import y`` resolve to the repo modules;
* the working directory becomes the repo root, because the code under test reads
  ``Circuits/``, ``Data/``, ``Results/`` and ``Plots/`` relative to it. A test that spawns
  ``python -c`` children passes ``cwd=ROOT`` so they import the repo the same way.

``ROOT`` is the repo root, for everything a test reads from the repo (Data/, Circuits/,
app/presets/, build/, the *.py sources, .gitignore). Scratch files never go there or into
this folder: a test that writes files uses tempfile, or CLAUDE_JOB_DIR_TMP when that is set.
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)

# a relative scratch dir given through the environment keeps meaning what the caller meant
# once the working directory moves to the repo root below
if os.environ.get("CLAUDE_JOB_DIR_TMP"):
    os.environ["CLAUDE_JOB_DIR_TMP"] = os.path.abspath(os.environ["CLAUDE_JOB_DIR_TMP"])

if ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)                     # first, so no file in this folder can shadow a repo module
os.chdir(ROOT)
