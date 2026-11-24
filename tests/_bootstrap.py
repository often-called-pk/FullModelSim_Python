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
import atexit
import os
import shutil
import sys
import tempfile

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

# the GUI tests keep their settings in a temp file of this process, never in the user's
# real settings (app.paths.settings reads this variable)
if "FULLMODELSIM_SETTINGS" not in os.environ:
    _fd, _ini = tempfile.mkstemp(prefix="fms_settings_", suffix=".ini")
    os.close(_fd)
    os.environ["FULLMODELSIM_SETTINGS"] = _ini
    atexit.register(lambda: os.path.exists(_ini) and os.remove(_ini))

# the GUI tests keep their Documents folder (default output, presets, saved cars) in a temp
# folder of this process, never in the user's real Documents (app.paths.documents_dir reads
# this variable). On Windows, Qt takes Documents from the shell, not from HOME or USERPROFILE.
if "FULLMODELSIM_DOCUMENTS" not in os.environ:
    _docs = tempfile.mkdtemp(prefix="fms_documents_")
    os.environ["FULLMODELSIM_DOCUMENTS"] = _docs
    atexit.register(shutil.rmtree, _docs, True)


GP3_SHA8 = "5da29437"       # the GP3 2013 base car of the vendor Models folder, named by its content


def vendor_car(cs_dir, sha8=GP3_SHA8):
    """Return the path of the car file under the parent of cs_dir whose sha256 starts with sha8, or "".
    The tests select a vendor car by its content, so no vendor file name is in the repo."""
    import hashlib
    from pathlib import Path
    for p in sorted(Path(cs_dir).resolve().parent.rglob("*.car")):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest().startswith(sha8):
            return str(p)
    return ""
