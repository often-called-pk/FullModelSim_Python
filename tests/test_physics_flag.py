"""Physics flag plumbing (plan docs/superpowers/plans/2026-10-18-physics-fixes.md, Task 2).

userOpts(physics=...) sets ctx.physics: 'matlab' by default, 'corrected' opt-in; any other
value raises ValueError. result_stem() appends _physCorrected to corrected result files, so
they never overwrite the default ones. MLTP_screen forwards physics to userOpts and records
it in data.physics (the screen runs with save off, so no file is written).
Needs casadi and Data/DATA_AA.mat (prints SKIP and exits 0 otherwise). Run from the repo root:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_physics_flag.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import os
import sys
import warnings

DATA = os.path.join(_bootstrap.ROOT, "Data")
if not os.path.exists(os.path.join(DATA, "DATA_AA.mat")):       # pragma: no cover
    print("SKIP test_physics_flag: Data/DATA_AA.mat missing")
    sys.exit(0)

from functions.context import Ctx
from functions.importfile import result_stem
from MLTP_screen import MLTP_screen
from userOpts import userOpts
from vehModel import vehModel


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def model(physics="matlab", **kw):          # kw -> userOpts (circuit, ATD, Electric_4Motors, vp_overrides, Steering, CamberGain)
    c = Ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        userOpts(c, **dict(dict(circuit="Sturn", physics=physics), **kw))
    vehModel(c)
    return c


def rejected(physics):
    """True when userOpts raises a ValueError whose message names the physics flag."""
    try:
        model(physics)
    except ValueError as exc:
        return "physics" in str(exc).lower()
    return False


# =============================================================================
print("1. userOpts(physics=...): default, kept, rejected")
c0 = Ctx()                                  # no physics kw: the userOpts default is what is tested
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    userOpts(c0, circuit="Sturn")
ok("default: ctx.physics == 'matlab' when physics is not passed",
   getattr(c0, "physics", None) == "matlab")
ok("physics='corrected' is kept on ctx.physics",
   getattr(model("corrected"), "physics", None) == "corrected")
ok("physics='Corrected' (case-sensitive) raises ValueError", rejected("Corrected"))
ok("physics='foo' raises ValueError", rejected("foo"))

# =============================================================================
print("2. result_stem: physics suffix")
ok("matlab stem unchanged: 'BCN_Static_ATDOn_EM4Off'",
   result_stem("BCN", "Static_ATDOn_EM4Off") == "BCN_Static_ATDOn_EM4Off")
ok("physics='corrected' appends _physCorrected",
   result_stem("BCN", "Static_ATDOn_EM4Off", physics="corrected")
   == "BCN_Static_ATDOn_EM4Off_physCorrected")
ok("tyre set, physics and mesh suffixes in that order",
   result_stem("BCN", "Static_ATDOn_EM4Off", "CopyB", "curvature", "corrected")
   == "BCN_Static_ATDOn_EM4Off_CopyB_physCorrected_meshCurvature")

# =============================================================================
print("3. MLTP_screen: physics forwarded to userOpts and recorded in data.physics (save off)")
cs = MLTP_screen(circuit="Sturn", save=False, verbose=False, physics="corrected")
ok("MLTP_screen(physics='corrected'): data.physics == 'corrected'",
   getattr(cs.data, "physics", None) == "corrected")

print("ALL physics flag TESTS PASSED")
