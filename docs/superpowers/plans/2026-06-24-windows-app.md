# Windows App (GUI + standalone exe) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the MLTP simulator as a user-friendly PySide6 desktop app, frozen to a standalone Windows `.exe` that runs on a clean machine with no Python.

**Architecture:** One frozen exe with two entry modes — launched normally it runs the GUI; launched with `--headless cfg.json` it runs a solve and exits. The GUI writes a JSON config, spawns the solve as a subprocess (re-invoking itself), and streams the IPOPT log. Vehicle parameters become overridable through a two-phase `vehParams` refactor (defaults dict + overrides → derived), enabling presets and an expert config file.

**Tech Stack:** Python, PySide6/Qt, CasADi+IPOPT (MUMPS floor, Coin-HSL bonus), numpy/scipy/plotly, PyInstaller.

## Global Constraints

- **Platform:** Windows. Shell for commands below is the Bash tool (POSIX `sh`).
- **No pytest.** Tests are plain scripts with asserts at module top level, run with `python test_<name>.py`. New tests live at repo root, mirror the existing `ok(name, cond)` helper style, and start with `sys.path.insert(0, os.path.dirname(__file__))`.
- **Keep existing tests green.** `test_params_useropts.py`, `test_foundation.py`, `test_transcription.py`, `test_save_load.py` must continue to pass **unchanged**. The `vehParams` refactor must reproduce every current default byte-for-byte (incl. the `Rw=0.355` / `gear`-from-0.3142857 quirk and the Copy-B tyre set).
- **Commit discipline:** commit a task only after that task's own test passes **and** the full suite still passes. The full-suite command is:
  ```bash
  for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
  ```
- **Push discipline:** push to `origin/windows-app` **only** in the final task, after the entire suite is green. No pushes before then.
- **Branch:** all work on `windows-app` (already checked out off `coin-hsl`).
- **vehParams override key validation:** an override key not in `PRIMARY_KEYS ∪ MF_KEYS` raises `ValueError`; never silently ignored.
- **Real end-to-end solve (minutes, needs IPOPT) is NOT in the automated suite** — it is the manual acceptance floor in Task 11. The automated gate covers the casadi-free logic.

---

### Task 1: Project scaffold + frozen-aware paths

**Files:**
- Create: `app/__init__.py` (empty)
- Create: `app/paths.py`
- Modify: `requirements.txt` (append `PySide6` and `pyinstaller`)
- Test: `test_paths.py`

**Interfaces:**
- Produces: `app.paths.is_frozen() -> bool`, `app.paths.resource_root() -> str`, `app.paths.resource_path(*parts) -> str`, `app.paths.default_output_dir() -> str`.

- [ ] **Step 1: Add deps and install**

Append to `requirements.txt`:
```
PySide6
pyinstaller
```
Run:
```bash
venv/Scripts/python.exe -m pip install PySide6 pyinstaller
```
Expected: both install successfully.

- [ ] **Step 2: Create the empty package marker**

Create `app/__init__.py` with a single line:
```python
"""windows-app GUI package for the MLTP simulator."""
```

- [ ] **Step 3: Write the failing test**

Create `test_paths.py`:
```python
"""Validate app/paths.py resource + output resolution (dev / non-frozen)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import paths

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

repo_root = os.path.dirname(os.path.abspath(__file__))
ok("not frozen in dev", paths.is_frozen() is False)
ok("resource_root is repo root", os.path.abspath(paths.resource_root()) == repo_root)
ok("resource_path joins", paths.resource_path("Data", "DATA_AA.mat")
   == os.path.join(paths.resource_root(), "Data", "DATA_AA.mat"))
out = paths.default_output_dir()
ok("output under Documents/FullModelSim",
   out.replace("\\", "/").endswith("Documents/FullModelSim"))
print("\nALL paths TESTS PASSED")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_paths.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.paths'`.

- [ ] **Step 5: Implement `app/paths.py`**

```python
"""Frozen-aware path resolution for the windows-app GUI.

Bundled (PyInstaller) read-only resources live under sys._MEIPASS; in a dev
checkout they live in the repo root (app/paths.py -> app/ -> repo root).
Solve output (Results/Plots) defaults to a user-writable Documents folder.
"""
import os
import sys


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def resource_root():
    if is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(*parts):
    return os.path.join(resource_root(), *parts)


def default_output_dir():
    return os.path.join(os.path.expanduser("~"), "Documents", "FullModelSim")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `venv/Scripts/python.exe test_paths.py`
Expected: PASS (all lines).

- [ ] **Step 7: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add app/__init__.py app/paths.py requirements.txt test_paths.py
git commit -m "feat(app): add frozen-aware paths module and GUI deps"
```

---

### Task 2: vehParams two-phase refactor + userOpts override threading

**Files:**
- Modify: `vehParams.py` (full rewrite of the function body; same defaults)
- Modify: `userOpts.py:181-195` (add `vp_overrides=None`, pass to `vehParams`)
- Test: `test_vp_overrides.py`

**Interfaces:**
- Produces: `vehParams.default_primaries() -> dict`, `vehParams.PRIMARY_KEYS` (frozenset), `vehParams.MF_KEYS` (frozenset), and `vehParams(ctx, data_dir="Data", vp_overrides=None)`. `userOpts(ctx, ..., vp_overrides=None, ...)`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

Create `test_vp_overrides.py`:
```python
"""vehParams two-phase refactor: overrides propagate to derived quantities;
empty overrides reproduce defaults; unknown keys raise; mf overrides apply.
Also checks the override reaches the model via userOpts."""
import os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from functions.context import Ctx
from vehParams import vehParams, default_primaries, PRIMARY_KEYS, MF_KEYS
from userOpts import userOpts

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# --- defaults reproduced with no overrides ---
ctx = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx)
vp = ctx.vp
ok("default ms = 1895", vp.ms == 1895.0)
ok("default m = 2085", vp.m == 2085.0)
ok("default Rw_f=Rw_r=0.355", vp.Rw_f == 0.355 and vp.Rw_r == 0.355)
ok("default c_fl formula", abs(vp.c_fl - 0.7 * 2 * np.sqrt((vp.m*(1-vp.wB))/2) * 75000.0) < 1e-6)

# --- primaries metadata ---
ok("PRIMARY_KEYS has mb,kt,brkB", {"mb", "kt", "brkB"} <= PRIMARY_KEYS)
ok("default_primaries mb=1820", default_primaries()["mb"] == 1820.0)

# --- override a primary -> derived recompute ---
ctx2 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx2, vp_overrides={"mb": 2000.0})
vp2 = ctx2.vp
ok("override mb -> ms = 2075", vp2.ms == 2075.0)
ok("override mb -> m = 2265", vp2.m == 2265.0)
ok("override mb -> c_fl uses new m_eff",
   abs(vp2.c_fl - 0.7 * 2 * np.sqrt((2265.0*(1-vp2.wB))/2) * 75000.0) < 1e-6)

# --- override a spring -> damper recompute ---
ctx3 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx3, vp_overrides={"k_fl": 100000.0})
ok("override k_fl -> c_fl uses new k",
   abs(ctx3.vp.c_fl - 0.7 * 2 * np.sqrt((ctx3.vp.m*(1-ctx3.vp.wB))/2) * 100000.0) < 1e-6)

# --- mf override ---
ctx4 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx4, vp_overrides={"pKy1": -19.0})
ok("mf override applied", ctx4.mf.pKy1 == -19.0)
ok("MF_KEYS has pKy1", "pKy1" in MF_KEYS)

# --- unknown key raises ---
ctx5 = Ctx()
raised = False
try:
    vehParams(ctx5, vp_overrides={"not_a_param": 1.0})
except ValueError:
    raised = True
ok("unknown override key raises ValueError", raised)

# --- reaches model via userOpts ---
ctx6 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    userOpts(ctx6, circuit="Sturn", vp_overrides={"mb": 2000.0})
ok("userOpts threads vp_overrides", ctx6.vp.ms == 2075.0)

print("\nALL vp_overrides TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_vp_overrides.py`
Expected: FAIL — `ImportError: cannot import name 'default_primaries'`.

- [ ] **Step 3: Rewrite `vehParams.py`**

Replace the entire file with (keeps the module docstring; restructures the function):
```python
"""vehParams.py - two-phase build of ctx.vp (+ ctx.mf / ctx.aero / ctx.cg).

Phase 1 builds a dict of primary (typed-in) inputs and merges any
``vp_overrides`` into it; phase 2 computes every derived quantity from the
merged primaries, so an override of any primary propagates correctly. With no
overrides the result is byte-for-byte identical to the original hardcoded
values (Copy-B tyre set; Rw overwritten to 0.355 with gear left on 0.3142857).
"""
import os
import warnings
import numpy as np
from types import SimpleNamespace

from functions.importfile import mat_to_namespace


def default_primaries():
    """Primary (leaf) vehicle inputs and their default values."""
    return {
        # balance / aero inputs
        "brkB": 0.6766, "Tdist": 0.7281, "ksD": 0.4620,
        "alpha_FL": 10.0, "alpha_FR": 10.0, "alpha_RW": 8.0, "alpha_TW": 0.0,
        # constants
        "g": 9.81, "rho": 1.204,
        # masses
        "mb": 1820.0, "md": 75.0, "muf": 90.0, "mur": 100.0,
        # dimensions
        "A": 1.95, "t": 1.8, "l": 3.0, "wB": 0.5,
        "hcg": 0.5, "huf": 0.2968771, "hur": 0.2968771, "hw": 1.28,
        "hRCf": 0.07, "hRCr": 0.11, "hride": 0.117,
        # inertias
        "I_z": 1960.0, "I_y": 1600.0, "I_x": 1000.0,
        # aero placeholders (overwritten by DATA_AA when present)
        "Cd": 0.75, "Cl": 1.45,
        # tyre
        "Rw": 0.355, "Jw": 3.6, "f": 0.01, "kt": 300000.0,
        "Fz0": 4905.0, "Fz0_shift": 1.0,
        # suspension
        "k_fl": 75000.0, "k_fr": 75000.0, "k_rl": 80000.0, "k_rr": 80000.0,
        "zeta_fl": 0.7, "zeta_fr": 0.7, "zeta_rl": 0.7, "zeta_rr": 0.7,
        # brakes
        "Tbrake_max": 4000.0,
        # camber / toe
        "gamma_fl": 0.0, "gamma_rl": 0.0, "toe_front": 0.0, "toe_rear": 0.0,
        # numerical epsilons
        "eps_x": 1e-6, "eps_y": 1e-6, "eps_K": 1e-6,
    }


def _default_mf():
    """Pacejka 5.2 coefficients (MF_205_60R15_V91, lateral block = Copy B)."""
    return SimpleNamespace(
        pCx1=1.6055, pDx1=1.1703, pDx2=-0.081328, pDx3=0.0,
        pEx1=0.53409, pEx2=-0.019956, pEx3=0.18089, pEx4=0.0,
        pKx1=36.411, pKx2=0.12615, pKx3=0.51289,
        pHx1=0.0, pHx2=0.0, pVx1=0.0, pVx2=0.0,
        rBx1=18.456, rBx2=16.314, rBx3=0.0,
        rCx1=1.091, rEx1=0.0, rEx2=0.0, rHx1=0.0058715,
        pCy1=2.1322, pDy1=1.0283, pDy2=-0.16758, pDy3=-1.5821,
        pEy1=0.15, pEy2=-1.8733, pEy3=0.0, pEy4=0.0, pEy5=0.0,
        pKy1=-20.505, pKy2=2.0284, pKy3=0.89994, pKy4=0.0,
        pKy5=0.002, pKy6=-0.002, pKy7=0.0,
        pHy1=0.0031377, pHy2=0.00051596,
        pVy1=0.0, pVy2=0.0, pVy3=0.0, pVy4=0.08,
        rBy1=22.003, rBy2=-13.623, rBy3=-0.0093616, rBy4=0.0,
        rCy1=0.98294, rEy1=0.0, rEy2=0.0,
        rHy1=-9.1492e-11, rHy2=0.0,
        rVy1=22.965, rVy2=0.37981, rVy3=1.8552,
        rVy4=0.08767, rVy5=-8.8234e-11, rVy6=0.90374,
    )


PRIMARY_KEYS = frozenset(default_primaries())
MF_KEYS = frozenset(vars(_default_mf()))


def vehParams(ctx, data_dir="Data", vp_overrides=None):
    if not hasattr(ctx, "vp") or ctx.vp is None:
        ctx.vp = SimpleNamespace()
    vp = ctx.vp

    overrides = dict(vp_overrides or {})
    unknown = set(overrides) - PRIMARY_KEYS - MF_KEYS
    if unknown:
        raise ValueError(f"Unknown vehParams override keys: {sorted(unknown)}")
    mf_over = {k: overrides.pop(k) for k in list(overrides) if k in MF_KEYS}

    # ---- phase 1: primaries (+ overrides) ---------------------------------
    p = default_primaries()
    p.update(overrides)
    for k, v in p.items():
        setattr(vp, k, v)

    # ---- phase 2: derived quantities --------------------------------------
    vp.ms = vp.mb + vp.md
    vp.mus = vp.muf + vp.mur
    vp.m = vp.ms + vp.muf + vp.mur

    vp.l_f = vp.l * (1 - vp.wB)
    vp.l_r = vp.l * vp.wB

    vp.hRC = (vp.l_f * vp.hRCr + vp.l_r * vp.hRCf) / vp.l
    vp.d = vp.hcg - vp.hRC

    vp.Rw_r = vp.Rw
    vp.Rw_f = vp.Rw

    vp.m_eff_f = vp.m * (1 - vp.wB)
    vp.m_eff_r = vp.m * vp.wB

    vp.c_fl = vp.zeta_fl * 2 * np.sqrt(vp.m_eff_f / 2) * vp.k_fl
    vp.c_fr = vp.zeta_fr * 2 * np.sqrt(vp.m_eff_f / 2) * vp.k_fr
    vp.c_rl = vp.zeta_rl * 2 * np.sqrt(vp.m_eff_r / 2) * vp.k_rl
    vp.c_rr = vp.zeta_rr * 2 * np.sqrt(vp.m_eff_r / 2) * vp.k_rr

    vp.Wfl0 = 0.5 * vp.g * (vp.muf + vp.wB * vp.ms)
    vp.Wfr0 = 0.5 * vp.g * (vp.muf + vp.wB * vp.ms)
    vp.Wrl0 = 0.5 * vp.g * (vp.mur + (1 - vp.wB) * vp.ms)
    vp.Wrr0 = 0.5 * vp.g * (vp.mur + (1 - vp.wB) * vp.ms)

    vp.xti_fl = vp.Wfl0 / vp.kt
    vp.xti_fr = vp.Wfr0 / vp.kt
    vp.xti_rl = vp.Wrl0 / vp.kt
    vp.xti_rr = vp.Wrr0 / vp.kt

    vp.xsi_fl = 0.5 * vp.g * vp.ms * vp.l_f / (vp.l * vp.k_fl)
    vp.xsi_fr = 0.5 * vp.g * vp.ms * vp.l_f / (vp.l * vp.k_fr)
    vp.xsi_rl = 0.5 * vp.g * vp.ms * vp.l_r / (vp.l * vp.k_rl)
    vp.xsi_rr = 0.5 * vp.g * vp.ms * vp.l_r / (vp.l * vp.k_rr)

    vp.lsi_fl = vp.hcg - (vp.Rw - vp.xti_fl)
    vp.lsi_fr = vp.hcg - (vp.Rw - vp.xti_fr)
    vp.lsi_rl = vp.hcg - (vp.Rw - vp.xti_rl)
    vp.lsi_rr = vp.hcg - (vp.Rw - vp.xti_rr)

    # ---- aerodynamics (DATA_AA overwrites Cd/Cl when present) -------------
    aa_path = os.path.join(data_dir, "DATA_AA.mat")
    if os.path.exists(aa_path):
        import scipy.io as sio
        raw = sio.loadmat(aa_path, squeeze_me=True, struct_as_record=False)
        ctx.aero = mat_to_namespace(raw["aero"])
        a = ctx.aero
        vp.Cl0_right = float(a.veh.Cl0_right)
        vp.Cl0_left = float(a.veh.Cl0_left)
        vp.Cl0_front = float(a.veh.Cl0_front)
        vp.Cl0_rear = float(a.veh.Cl0_rear)
        vp.Cd0 = float(a.veh.Cd0)
        vp.Cs0_front = float(a.veh.Cs0_front)
        vp.Cs0_rear = float(a.veh.Cs0_rear)
        vp.Cl = vp.Cl0_front + vp.Cl0_rear
        vp.Cd = vp.Cd0
    else:
        ctx.aero = None
        warnings.warn(
            f"DATA_AA.mat not found at '{aa_path}'. Aerodynamic coefficients are "
            "unset; place DATA_AA.mat in the Data/ folder before a real run.")

    # ---- Pacejka 5.2 coefficients (+ mf overrides) ------------------------
    ctx.mf = _default_mf()
    for k, v in mf_over.items():
        setattr(ctx.mf, k, v)

    # ---- simplified Pacejka used by the 7-state init model ----------------
    vp.tyre = SimpleNamespace(
        mu=1.41, pD2=-0.6,
        bx=6.0, cx=2.3, ex=0.9,
        by=6.0, cy=2.5, ey=0.5,
    )

    deg2rad = np.pi / 180.0

    # ---- camber (mirrored across the axle) --------------------------------
    vp.gamma_fr = -vp.gamma_fl
    vp.gamma_rr = -vp.gamma_rl
    vp.gamma_fl_rad = vp.gamma_fl * deg2rad
    vp.gamma_fr_rad = vp.gamma_fr * deg2rad
    vp.gamma_rl_rad = vp.gamma_rl * deg2rad
    vp.gamma_rr_rad = vp.gamma_rr * deg2rad

    # ---- toe --------------------------------------------------------------
    vp.toe_front_rad = vp.toe_front * deg2rad
    vp.toe_rear_rad = vp.toe_rear * deg2rad

    # ---- camber-gain coefficients (linear option) -------------------------
    ctx.cg = SimpleNamespace(
        CG_h_deg_per_mm_linear=-0.03,
        CG_r_deg_per_deg_linear=0.1,
        CG_p_deg_per_deg_linear=0.05,
    )

    vp.CG_h_deg_per_mm_table = np.array([
        -0.12, 0.038, -0.08, 0.027, -0.04, 0.012, 0.00, 0.000,
        0.02, -0.006, 0.05, -0.015, 0.08, -0.025, 0.10, -0.032])
    vp.CG_r_deg_per_deg_table = np.array([
        -0.105, -0.8, -0.070, -0.75, -0.035, -0.7, 0.000, -0.65,
        0.035, -0.6, 0.070, -0.55, 0.105, -0.5])
    vp.CG_p_deg_per_deg_table = None

    return ctx
```

- [ ] **Step 4: Thread `vp_overrides` through `userOpts`**

In `userOpts.py`, add the parameter to the signature (after `hsl_dir=None`):
```python
             hsl_dir=None,                 # Coin-HSL bin dir; None -> COINHSL_DIR env / default
             vp_overrides=None):           # dict of vehParams primary/mf overrides
```
And change the `vehParams` call (currently `vehParams(ctx, data_dir=data_dir)`):
```python
    vehParams(ctx, data_dir=data_dir, vp_overrides=vp_overrides)
```

- [ ] **Step 5: Run the refactor test and the regression test**

Run:
```bash
venv/Scripts/python.exe test_vp_overrides.py
venv/Scripts/python.exe test_params_useropts.py
```
Expected: both PASS (the second proves byte-for-byte defaults preserved).

- [ ] **Step 6: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add vehParams.py userOpts.py test_vp_overrides.py
git commit -m "refactor(vehParams): two-phase primaries+derived; thread vp_overrides"
```

---

### Task 3: MLTP output-dir + warm-start override plumbing

**Files:**
- Modify: `MLTP.py:136-138` (add `plots_dir="Plots"`), `MLTP.py:150-151` (forward `**useropts_kwargs` to `MLTP_initial`), `MLTP.py:278` (`plotSDI(ctx, save_dir=plots_dir)`)
- Test: `test_mltp_params.py`

**Interfaces:**
- Produces: `MLTP(..., plots_dir="Plots", **useropts_kwargs)` accepts `vp_overrides`, `circuits_dir`, `data_dir`, `linear_solver` via kwargs; writes plots under `plots_dir`. The internal warm-start `MLTP_initial` call receives the same `**useropts_kwargs` (so overrides apply to the warm start too).

- [ ] **Step 1: Write the failing test**

Create `test_mltp_params.py`:
```python
"""MLTP exposes plots_dir and forwards useropts_kwargs to the warm-start call.
Signature-level checks only (no solve)."""
import os, sys, inspect
sys.path.insert(0, os.path.dirname(__file__))
import MLTP as mltp_mod

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

sig = inspect.signature(mltp_mod.MLTP)
ok("MLTP has plots_dir param", "plots_dir" in sig.parameters)
ok("plots_dir default 'Plots'", sig.parameters["plots_dir"].default == "Plots")
ok("MLTP accepts **useropts_kwargs",
   any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))

# the warm-start call must forward kwargs (so vp_overrides reach the init model)
src = inspect.getsource(mltp_mod.MLTP)
ok("warm-start call forwards **useropts_kwargs",
   "MLTP_initial(" in src and "**useropts_kwargs" in src.split("MLTP_initial(")[1].split(")")[0])
ok("plotSDI uses plots_dir", "save_dir=plots_dir" in src)
print("\nALL MLTP param TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_mltp_params.py`
Expected: FAIL — `MLTP has plots_dir param` assertion.

- [ ] **Step 3: Edit `MLTP.py`**

Change the signature (lines ~136-138) to add `plots_dir`:
```python
def MLTP(circuit="Sturn", vi=60.0, ni=np.nan, warm_start=None,
         AeroConfig="Static", ATD="On", Electric_4Motors="Off", TyreModel="CombinedSlip",
         save=True, plot=True, results_dir="Results", plots_dir="Plots", **useropts_kwargs):
```
Change the warm-start branch (lines ~150-151) to forward kwargs:
```python
        ctx_init = MLTP_initial(circuit=circuit, vi=vi, ni=ni, AeroConfig=AeroConfig,
                                ATD=ATD, Electric_4Motors=Electric_4Motors, save=False,
                                **useropts_kwargs)
```
Change the plotting call (line ~278):
```python
            plotSDI(ctx, save_dir=plots_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe test_mltp_params.py`
Expected: PASS.

- [ ] **Step 5: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add MLTP.py test_mltp_params.py
git commit -m "feat(mltp): add plots_dir output + forward overrides to warm start"
```

---

### Task 4: RunConfig (GUI state <-> solve config)

**Files:**
- Create: `app/runconfig.py`
- Test: `test_runconfig.py`

**Interfaces:**
- Consumes: `vehParams.PRIMARY_KEYS`, `vehParams.MF_KEYS`, `app.paths.default_output_dir`.
- Produces: `app.runconfig.RunConfig` dataclass with `to_dict()`, `from_dict(d)`, `vp_overrides()`, `write_cfg(path)`, `from_json(path)`; module constant `app.runconfig.TIER1_FIELDS`.

- [ ] **Step 1: Write the failing test**

Create `test_runconfig.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_runconfig.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.runconfig'`.

- [ ] **Step 3: Implement `app/runconfig.py`**

```python
"""RunConfig: the GUI's editable state, serialisable to the solve cfg.json.

`to_dict`/`from_dict` persist the full GUI state. `write_cfg` emits the JSON
the headless solve consumes: run-config fields plus a single merged
`vp_overrides` dict (Tier-1 tunables + any expert config file).
"""
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

from app.paths import default_output_dir
from vehParams import PRIMARY_KEYS, MF_KEYS

TIER1_FIELDS = ("brkB", "Tdist", "ksD",
                "alpha_FL", "alpha_FR", "alpha_RW", "alpha_TW")


@dataclass
class RunConfig:
    circuit: str = "Sturn"
    AeroConfig: str = "Static"
    ATD: str = "On"
    Electric_4Motors: str = "Off"
    TyreModel: str = "CombinedSlip"
    vi: float = 60.0
    ni: Optional[float] = None
    linear_solver: str = "ma57"
    warm_start: Optional[str] = None
    save: bool = True
    plot: bool = True
    output_dir: str = field(default_factory=default_output_dir)
    # Tier-1 tunables (vehParams primaries)
    brkB: float = 0.6766
    Tdist: float = 0.7281
    ksD: float = 0.4620
    alpha_FL: float = 10.0
    alpha_FR: float = 10.0
    alpha_RW: float = 8.0
    alpha_TW: float = 0.0
    expert_config: Optional[str] = None

    def vp_overrides(self):
        ov = {}
        if self.expert_config:
            with open(self.expert_config) as fh:
                ov.update(json.load(fh))
        for k in TIER1_FIELDS:
            ov[k] = getattr(self, k)
        unknown = set(ov) - PRIMARY_KEYS - MF_KEYS
        if unknown:
            raise ValueError(f"Unknown vehParams override keys: {sorted(unknown)}")
        return ov

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def _cfg(self):
        d = asdict(self)
        d.pop("expert_config", None)
        for k in TIER1_FIELDS:
            d.pop(k, None)
        d["vp_overrides"] = self.vp_overrides()
        return d

    def write_cfg(self, path):
        cfg = self._cfg()
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe test_runconfig.py`
Expected: PASS.

- [ ] **Step 5: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add app/runconfig.py test_runconfig.py
git commit -m "feat(app): add RunConfig state + solve-cfg serialisation"
```

---

### Task 5: Headless solve entry

**Files:**
- Create: `headless_solve.py`
- Test: `test_headless_config.py`

**Interfaces:**
- Consumes: `app.paths.resource_root`, the cfg dict produced by `RunConfig.write_cfg`.
- Produces: `headless_solve.build_solve_kwargs(cfg, resource_root_dir) -> dict`, `headless_solve.main(argv) -> int`.

- [ ] **Step 1: Write the failing test**

Create `test_headless_config.py`:
```python
"""build_solve_kwargs maps a cfg dict -> MLTP kwargs (absolute resource/output
dirs, ni-null -> nan, overrides passed through). No solving."""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
from headless_solve import build_solve_kwargs

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

cfg = {
    "circuit": "Sturn", "AeroConfig": "Static", "ATD": "On",
    "Electric_4Motors": "Off", "TyreModel": "CombinedSlip",
    "vi": 60.0, "ni": None, "linear_solver": "mumps",
    "save": True, "plot": True, "output_dir": "/out",
    "vp_overrides": {"mb": 2000.0},
}
kw = build_solve_kwargs(cfg, "/res")
ok("circuit mapped", kw["circuit"] == "Sturn")
ok("ni null -> nan", isinstance(kw["ni"], float) and math.isnan(kw["ni"]))
ok("results_dir under output", kw["results_dir"] == os.path.join("/out", "Results"))
ok("plots_dir under output", kw["plots_dir"] == os.path.join("/out", "Plots"))
ok("circuits_dir under resource", kw["circuits_dir"] == os.path.join("/res", "Circuits"))
ok("data_dir under resource", kw["data_dir"] == os.path.join("/res", "Data"))
ok("linear_solver passed", kw["linear_solver"] == "mumps")
ok("vp_overrides passed", kw["vp_overrides"] == {"mb": 2000.0})

cfg2 = dict(cfg, ni=0.3)
ok("numeric ni preserved", build_solve_kwargs(cfg2, "/res")["ni"] == 0.3)
print("\nALL headless config TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_headless_config.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'headless_solve'`.

- [ ] **Step 3: Implement `headless_solve.py`**

```python
"""Headless solve entry. Invoked by the GUI as a subprocess
(`<exe> --headless cfg.json`, or `python headless_solve.py cfg.json` in dev).
Reads cfg.json, runs MLTP, streams IPOPT output to stdout, exits.
"""
import json
import math
import os
import sys

from app.paths import resource_root


def build_solve_kwargs(cfg, resource_root_dir):
    """Pure: cfg dict -> MLTP() kwargs. No I/O, no solving."""
    output_dir = cfg["output_dir"]
    ni = cfg.get("ni", None)
    ni = float("nan") if ni is None else float(ni)
    kwargs = dict(
        circuit=cfg["circuit"],
        vi=float(cfg["vi"]),
        ni=ni,
        AeroConfig=cfg["AeroConfig"],
        ATD=cfg["ATD"],
        Electric_4Motors=cfg["Electric_4Motors"],
        TyreModel=cfg["TyreModel"],
        linear_solver=cfg.get("linear_solver", "ma57"),
        save=bool(cfg.get("save", True)),
        plot=bool(cfg.get("plot", True)),
        results_dir=os.path.join(output_dir, "Results"),
        plots_dir=os.path.join(output_dir, "Plots"),
        circuits_dir=os.path.join(resource_root_dir, "Circuits"),
        data_dir=os.path.join(resource_root_dir, "Data"),
        vp_overrides=cfg.get("vp_overrides") or None,
    )
    if cfg.get("warm_start"):
        kwargs["warm_start"] = cfg["warm_start"]
    return kwargs


def main(argv):
    if not argv:
        print("usage: headless_solve.py <cfg.json>", file=sys.stderr)
        return 2
    with open(argv[0]) as fh:
        cfg = json.load(fh)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    kwargs = build_solve_kwargs(cfg, resource_root())
    from MLTP import MLTP            # imported here so the unit test stays casadi-free
    MLTP(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe test_headless_config.py`
Expected: PASS.

- [ ] **Step 5: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add headless_solve.py test_headless_config.py
git commit -m "feat: add headless solve entry (cfg.json -> MLTP)"
```

---

### Task 6: Results parser

**Files:**
- Create: `app/results.py`
- Test: `test_results.py`

**Interfaces:**
- Produces: `app.results.result_mat_path(output_dir, circuit, AeroConfig, ATD, EM4) -> str`, `app.results.plot_dir(output_dir, circuit, AeroConfig, ATD, EM4) -> str`, `app.results.parse_summary(mat_path) -> dict`, `app.results.list_plots(plot_directory) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Create `test_results.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_results.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.results'`.

- [ ] **Step 3: Implement `app/results.py`**

```python
"""Parse a saved solve .mat into summary numbers and list its Plotly plots.
Path scheme mirrors MLTP.save (Results/<circuit>_<cfg>.mat) and plotSDI
(Plots/<circuit>/<cfg>/<name>.html), with cfg = <AeroConfig>_ATD<ATD>_EM4<EM4>.
"""
import glob
import os
import numpy as np
import scipy.io as sio


def _cfg(AeroConfig, ATD, EM4):
    return f"{AeroConfig}_ATD{ATD}_EM4{EM4}"


def result_mat_path(output_dir, circuit, AeroConfig, ATD, EM4):
    return os.path.join(output_dir, "Results",
                        f"{circuit}_{_cfg(AeroConfig, ATD, EM4)}.mat")


def plot_dir(output_dir, circuit, AeroConfig, ATD, EM4):
    return os.path.join(output_dir, "Plots", circuit, _cfg(AeroConfig, ATD, EM4))


def parse_summary(mat_path):
    raw = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = raw["data"]
    lap_time = float(np.asarray(data.lap_time).reshape(-1)[-1])
    energy = None
    veh = getattr(data, "vehicle", None)
    if veh is not None and hasattr(veh, "E_motor"):
        energy = float(np.asarray(veh.E_motor).reshape(-1)[-1])
    return {"lap_time_s": lap_time, "energy_kWh": energy}


def list_plots(plot_directory):
    if not os.path.isdir(plot_directory):
        return []
    return sorted(glob.glob(os.path.join(plot_directory, "*.html")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe test_results.py`
Expected: PASS.

- [ ] **Step 5: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add app/results.py test_results.py
git commit -m "feat(app): add results .mat summary parser + plot listing"
```

---

### Task 7: Solve runner (QProcess) + app entry / headless dispatch

**Files:**
- Create: `app/solve_runner.py`
- Create: `app/main.py`
- Test: `test_solve_runner.py`

**Interfaces:**
- Consumes: `app.paths.is_frozen`, `app.paths.resource_root`.
- Produces: `app.solve_runner.solve_command(cfg_path) -> list[str]`, `app.solve_runner.SolveRunner` (QObject: `start(cfg_path)`, `cancel()`, signals `output(str)`, `finished(int)`), `app.main.is_headless(argv) -> bool`, `app.main.main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `test_solve_runner.py`:
```python
"""solve_command (dev branch) + headless arg dispatch. Offscreen Qt."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))
from app.solve_runner import solve_command
from app.main import is_headless
from app.paths import resource_root

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

cmd = solve_command("cfg.json")
ok("dev command runs headless_solve.py",
   cmd[1] == os.path.join(resource_root(), "headless_solve.py") and cmd[-1] == "cfg.json")
ok("dev command uses python", cmd[0] == sys.executable)

ok("is_headless true", is_headless(["app", "--headless", "cfg.json"]))
ok("is_headless false", not is_headless(["app"]))
print("\nALL solve_runner TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_solve_runner.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.solve_runner'`.

- [ ] **Step 3: Implement `app/solve_runner.py`**

```python
"""Spawn the headless solve as a subprocess and stream its merged stdout.
In a frozen app the exe re-invokes itself (`<exe> --headless cfg`); in dev it
runs `python headless_solve.py cfg`.
"""
import os
import sys

from PySide6.QtCore import QObject, QProcess, Signal

from app.paths import is_frozen, resource_root


def solve_command(cfg_path):
    if is_frozen():
        return [sys.executable, "--headless", cfg_path]
    return [sys.executable, os.path.join(resource_root(), "headless_solve.py"), cfg_path]


class SolveRunner(QObject):
    output = Signal(str)
    finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None

    def start(self, cfg_path):
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(lambda code, _status: self.finished.emit(int(code)))
        argv = solve_command(cfg_path)
        self._proc.start(argv[0], argv[1:])

    def _on_output(self):
        text = bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.output.emit(text)

    def cancel(self):
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()
```

- [ ] **Step 4: Implement `app/main.py`**

```python
"""Application entry. Normal launch -> GUI; `--headless cfg.json` -> solve."""
import sys


def is_headless(argv):
    return "--headless" in argv


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if is_headless(argv):
        i = argv.index("--headless")
        cfg_path = argv[i + 1]
        import headless_solve
        return headless_solve.main([cfg_path])
    from PySide6.QtWidgets import QApplication
    from app.mainwindow import MainWindow
    app = QApplication(argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

Note: `app/main.py` imports `app.mainwindow` only in the GUI branch, so this task's test (which never enters that branch) passes before Task 8 exists.

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/Scripts/python.exe test_solve_runner.py`
Expected: PASS.

- [ ] **Step 6: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add app/solve_runner.py app/main.py test_solve_runner.py
git commit -m "feat(app): add QProcess solve runner + headless dispatch entry"
```

---

### Task 8: Main window (GUI)

**Files:**
- Create: `app/mainwindow.py`
- Test: `test_gui_logic.py`

**Interfaces:**
- Consumes: `app.runconfig.RunConfig`, `app.solve_runner.SolveRunner`, `app.results`, `app.paths`.
- Produces: `app.mainwindow.MainWindow` (QMainWindow) with `collect_runconfig() -> RunConfig` and the EM4/ATD conflict rule wired so checking 4-motors disables the ATD control.

- [ ] **Step 1: Write the failing test**

Create `test_gui_logic.py`:
```python
"""Offscreen GUI logic: window builds, collect_runconfig reflects widgets,
EM4-on disables the ATD control (the config-conflict rule)."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))
from PySide6.QtWidgets import QApplication
from app.mainwindow import MainWindow
from app.runconfig import RunConfig

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

app = QApplication.instance() or QApplication([])
win = MainWindow()

rc = win.collect_runconfig()
ok("collect returns RunConfig", isinstance(rc, RunConfig))
ok("default circuit collected", rc.circuit == "Sturn")

# conflict rule: turning 4-motors on disables ATD
win.set_em4(True)
ok("ATD control disabled when EM4 on", win.atd_enabled() is False)
win.set_em4(False)
ok("ATD control re-enabled when EM4 off", win.atd_enabled() is True)
print("\nALL GUI logic TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_gui_logic.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.mainwindow'`.

- [ ] **Step 3: Implement `app/mainwindow.py`**

```python
"""Main window: Main / Setup / Output / Advanced tabs, Run/Cancel, live log,
results summary. Non-visual logic (collect_runconfig, conflict rule) is exposed
as methods for testing.
"""
import os
import tempfile
import webbrowser

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QFormLayout, QVBoxLayout, QHBoxLayout,
    QComboBox, QDoubleSpinBox, QCheckBox, QPushButton, QPlainTextEdit, QLabel,
    QLineEdit, QFileDialog, QListWidget,
)

from app.runconfig import RunConfig, TIER1_FIELDS
from app.solve_runner import SolveRunner
from app import results, paths

CIRCUITS = ["Sturn", "Straight", "Hairpin", "Circle", "ZigZag", "ZigZagMirror",
            "VirtualTrack", "BCN", "BCN_S1", "BCN_S2", "BCN_S3", "Jarama", "Spa",
            "BCNAssetto"]
AEROS = ["Static", "Active_RW", "Active", "AALB"]
TYRES = ["CombinedSlip", "PureSlip"]
SOLVERS = ["ma57", "ma97", "ma27", "mumps"]


def _spin(value, lo, hi, step=1.0, decimals=4):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setValue(value)
    return s


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FullModelSim — MLTP")
        self._runner = SolveRunner(self)
        self._runner.output.connect(self._append_log)
        self._runner.finished.connect(self._on_finished)
        self._defaults = RunConfig()

        tabs = QTabWidget()
        tabs.addTab(self._build_main_tab(), "Main")
        tabs.addTab(self._build_setup_tab(), "Setup")
        tabs.addTab(self._build_output_tab(), "Output")
        tabs.addTab(self._build_advanced_tab(), "Advanced")

        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self._runner.cancel)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.cancel_btn)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(tabs)
        lay.addLayout(buttons)
        lay.addWidget(QLabel("Solver log:"))
        lay.addWidget(self.log)
        self.setCentralWidget(central)

    # ---- tab builders -----------------------------------------------------
    def _build_main_tab(self):
        w = QWidget(); form = QFormLayout(w)
        self.circuit = QComboBox(); self.circuit.addItems(CIRCUITS)
        self.aero = QComboBox(); self.aero.addItems(AEROS)
        self.atd = QCheckBox("ATD on"); self.atd.setChecked(True)
        self.em4 = QCheckBox("4 Motors on")
        self.em4.toggled.connect(self._on_em4_toggled)
        self.tyre = QComboBox(); self.tyre.addItems(TYRES)
        self.vi = _spin(60.0, 0.0, 150.0, 1.0, 2)
        self.ni_free = QCheckBox("free"); self.ni_free.setChecked(True)
        self.ni = _spin(0.0, -10.0, 10.0, 0.1, 3)
        self.ni.setEnabled(False)
        self.ni_free.toggled.connect(lambda f: self.ni.setEnabled(not f))
        ni_row = QWidget(); nilay = QHBoxLayout(ni_row); nilay.setContentsMargins(0, 0, 0, 0)
        nilay.addWidget(self.ni); nilay.addWidget(self.ni_free)
        form.addRow("Circuit", self.circuit)
        form.addRow("Aero config", self.aero)
        form.addRow("ATD", self.atd)
        form.addRow("4 Motors", self.em4)
        form.addRow("Tyre model", self.tyre)
        form.addRow("Initial speed [m/s]", self.vi)
        form.addRow("Initial lateral n [m]", ni_row)
        return w

    def _build_setup_tab(self):
        w = QWidget(); form = QFormLayout(w)
        d = self._defaults
        self.brkB = _spin(d.brkB, 0.0, 1.0, 0.01)
        self.Tdist = _spin(d.Tdist, 0.0, 1.0, 0.01)
        self.ksD = _spin(d.ksD, 0.0, 1.0, 0.01)
        self.alpha_FL = _spin(d.alpha_FL, 0.0, 10.0, 0.5, 2)
        self.alpha_FR = _spin(d.alpha_FR, 0.0, 10.0, 0.5, 2)
        self.alpha_RW = _spin(d.alpha_RW, 0.0, 30.0, 0.5, 2)
        self.alpha_TW = _spin(d.alpha_TW, -12.0, 12.0, 0.5, 2)
        form.addRow("Brake bias (front)", self.brkB)
        form.addRow("Torque dist (rear)", self.Tdist)
        form.addRow("Roll stiff (rear)", self.ksD)
        form.addRow("Front wing L [deg]", self.alpha_FL)
        form.addRow("Front wing R [deg]", self.alpha_FR)
        form.addRow("Rear wing [deg]", self.alpha_RW)
        form.addRow("Rear wing tilt [deg]", self.alpha_TW)
        return w

    def _build_output_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        form = QFormLayout()
        self.output_dir = QLineEdit(self._defaults.output_dir)
        browse = QPushButton("Browse…"); browse.clicked.connect(self._pick_output)
        odir = QWidget(); ol = QHBoxLayout(odir); ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(self.output_dir); ol.addWidget(browse)
        self.save_cb = QCheckBox("Save .mat"); self.save_cb.setChecked(True)
        self.plot_cb = QCheckBox("Generate plots"); self.plot_cb.setChecked(True)
        form.addRow("Output folder", odir)
        form.addRow("", self.save_cb)
        form.addRow("", self.plot_cb)
        lay.addLayout(form)
        self.summary = QLabel("No results yet.")
        self.plot_list = QListWidget()
        self.plot_list.itemDoubleClicked.connect(
            lambda it: webbrowser.open(it.data(256)))
        lay.addWidget(self.summary)
        lay.addWidget(QLabel("Plots (double-click to open):"))
        lay.addWidget(self.plot_list)
        return w

    def _build_advanced_tab(self):
        w = QWidget(); form = QFormLayout(w)
        self.solver = QComboBox(); self.solver.addItems(SOLVERS)
        self.warm_start = QLineEdit(); self.warm_start.setPlaceholderText("(auto warm start)")
        ws_btn = QPushButton("…"); ws_btn.clicked.connect(self._pick_warm)
        ws = QWidget(); wl = QHBoxLayout(ws); wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(self.warm_start); wl.addWidget(ws_btn)
        self.expert = QLineEdit(); self.expert.setPlaceholderText("(no expert config)")
        ex_btn = QPushButton("…"); ex_btn.clicked.connect(self._pick_expert)
        ex = QWidget(); el = QHBoxLayout(ex); el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(self.expert); el.addWidget(ex_btn)
        form.addRow("Linear solver", self.solver)
        form.addRow("Warm start .mat", ws)
        form.addRow("Expert config .json", ex)
        return w

    # ---- conflict rule ----------------------------------------------------
    def _on_em4_toggled(self, on):
        if on:
            self.atd.setChecked(False)
        self.atd.setEnabled(not on)

    def set_em4(self, on):           # test hook
        self.em4.setChecked(on)

    def atd_enabled(self):           # test hook
        return self.atd.isEnabled()

    # ---- collect / run ----------------------------------------------------
    def collect_runconfig(self):
        return RunConfig(
            circuit=self.circuit.currentText(),
            AeroConfig=self.aero.currentText(),
            ATD="On" if self.atd.isChecked() else "Off",
            Electric_4Motors="On" if self.em4.isChecked() else "Off",
            TyreModel=self.tyre.currentText(),
            vi=self.vi.value(),
            ni=None if self.ni_free.isChecked() else self.ni.value(),
            linear_solver=self.solver.currentText(),
            warm_start=self.warm_start.text() or None,
            save=self.save_cb.isChecked(),
            plot=self.plot_cb.isChecked(),
            output_dir=self.output_dir.text(),
            brkB=self.brkB.value(), Tdist=self.Tdist.value(), ksD=self.ksD.value(),
            alpha_FL=self.alpha_FL.value(), alpha_FR=self.alpha_FR.value(),
            alpha_RW=self.alpha_RW.value(), alpha_TW=self.alpha_TW.value(),
            expert_config=self.expert.text() or None,
        )

    def _on_run(self):
        rc = self.collect_runconfig()
        try:
            os.makedirs(rc.output_dir, exist_ok=True)
            cfg_path = os.path.join(tempfile.gettempdir(), "fms_solve_cfg.json")
            rc.write_cfg(cfg_path)
        except Exception as exc:
            self._append_log(f"[error] {exc}\n")
            return
        self._active_rc = rc
        self.log.clear()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._runner.start(cfg_path)

    def _on_finished(self, code):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if code == 0:
            self._show_results(self._active_rc)
        else:
            self._append_log(f"\n[solve exited with code {code}]\n")

    def _show_results(self, rc):
        mat = results.result_mat_path(rc.output_dir, rc.circuit, rc.AeroConfig,
                                      rc.ATD, rc.Electric_4Motors)
        try:
            summ = results.parse_summary(mat)
            e = summ["energy_kWh"]
            self.summary.setText(
                f"Lap time: {summ['lap_time_s']:.3f} s"
                + (f"   |   Energy: {e:.3f} kWh" if e is not None else ""))
        except Exception as exc:
            self.summary.setText(f"Results unreadable: {exc}")
        pdir = results.plot_dir(rc.output_dir, rc.circuit, rc.AeroConfig,
                                rc.ATD, rc.Electric_4Motors)
        self.plot_list.clear()
        for p in results.list_plots(pdir):
            from PySide6.QtWidgets import QListWidgetItem
            it = QListWidgetItem(os.path.basename(p))
            it.setData(256, p)
            self.plot_list.addItem(it)

    # ---- helpers ----------------------------------------------------------
    def _append_log(self, text):
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(text)

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.output_dir.text())
        if d:
            self.output_dir.setText(d)

    def _pick_warm(self):
        f, _ = QFileDialog.getOpenFileName(self, "Warm start .mat", "", "MAT (*.mat)")
        if f:
            self.warm_start.setText(f)

    def _pick_expert(self):
        f, _ = QFileDialog.getOpenFileName(self, "Expert config", "", "JSON (*.json)")
        if f:
            self.expert.setText(f)
            self._load_expert_tier1(f)

    def _load_expert_tier1(self, path):
        import json
        try:
            data = json.load(open(path))
        except Exception as exc:
            self._append_log(f"[expert config error] {exc}\n")
            return
        for k in TIER1_FIELDS:
            if k in data and hasattr(self, k):
                getattr(self, k).setValue(float(data[k]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe test_gui_logic.py`
Expected: PASS.

- [ ] **Step 5: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add app/mainwindow.py test_gui_logic.py
git commit -m "feat(app): add main window GUI (tabs, run/cancel, results)"
```

---

### Task 9: Default preset + preset validation

**Files:**
- Create: `app/presets/default.json`
- Test: `test_presets.py`

**Interfaces:**
- Consumes: `vehParams.PRIMARY_KEYS`, `vehParams.MF_KEYS`, `vehParams.default_primaries`, `vehParams.vehParams`.
- Produces: `app/presets/default.json` (baseline car = current defaults).

- [ ] **Step 1: Write the failing test**

Create `test_presets.py`:
```python
"""Shipped presets are valid override files and default.json reproduces the
baseline car."""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(__file__))
from functions.context import Ctx
from vehParams import vehParams, default_primaries, PRIMARY_KEYS, MF_KEYS

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

ppath = os.path.join(os.path.dirname(__file__), "app", "presets", "default.json")
ok("default.json exists", os.path.exists(ppath))
preset = json.load(open(ppath))
ok("all keys are known overrides", set(preset) <= (PRIMARY_KEYS | MF_KEYS))

# default.json must equal the baseline primaries (it is the baseline car)
base = default_primaries()
ok("default.json matches baseline primaries", all(preset[k] == base[k] for k in preset))

# loading it through vehParams reproduces the baseline (ms=1895)
ctx = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vehParams(ctx, vp_overrides=preset)
ok("baseline ms reproduced", ctx.vp.ms == 1895.0)
print("\nALL preset TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_presets.py`
Expected: FAIL — `default.json exists` assertion.

- [ ] **Step 3: Generate `app/presets/default.json` from the baseline primaries**

Run this one-off to write the file from the single source of truth:
```bash
venv/Scripts/python.exe -c "import os,json,sys; sys.path.insert(0,'.'); from vehParams import default_primaries; os.makedirs('app/presets',exist_ok=True); json.dump(default_primaries(), open('app/presets/default.json','w'), indent=2); print('wrote app/presets/default.json')"
```
Expected: `wrote app/presets/default.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe test_presets.py`
Expected: PASS.

- [ ] **Step 5: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add app/presets/default.json test_presets.py
git commit -m "feat(app): ship default car preset + preset validation"
```

---

### Task 10: PyInstaller spec + build/run docs

**Files:**
- Create: `build/windows-app.spec`
- Create: `build/README.md`
- Test: `test_spec_includes.py`

**Interfaces:**
- Produces: a PyInstaller spec that bundles `Circuits/`, `Data/`, `app/presets/`, collects CasADi + PySide6, and (best-effort) the Coin-HSL DLLs; documented manual build + acceptance steps.

- [ ] **Step 1: Write the failing test**

Create `test_spec_includes.py`:
```python
"""Static lint of the PyInstaller spec: required data dirs + collected packages
are referenced so a frozen build can find them."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

spec = open(os.path.join(os.path.dirname(__file__), "build", "windows-app.spec")).read()
for token in ("Circuits", "Data", "app/presets", "casadi", "PySide6", "headless_solve"):
    ok(f"spec references {token}", token in spec)
ok("spec sets app name", "name='FullModelSim'" in spec or 'name="FullModelSim"' in spec)
print("\nALL spec-include TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe test_spec_includes.py`
Expected: FAIL — spec file does not exist.

- [ ] **Step 3: Implement `build/windows-app.spec`**

```python
# PyInstaller spec for the FullModelSim windows-app.
# Build from repo root:  venv\Scripts\pyinstaller.exe build\windows-app.spec
# Onedir first (easier DLL debugging); flip EXE(console=...) / onefile later.
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
ROOT = os.path.abspath(os.getcwd())

# Bundle read-only resources next to the frozen root (sys._MEIPASS).
datas = [
    (os.path.join(ROOT, "Circuits"), "Circuits"),
    (os.path.join(ROOT, "Data"), "Data"),
    (os.path.join(ROOT, "app", "presets"), "app/presets"),
]
binaries = []
hiddenimports = collect_submodules("scipy") + ["headless_solve"]

# CasADi ships its own compiled libs (IPOPT/MUMPS) — collect everything.
for pkg in ("casadi", "PySide6", "plotly"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# Best-effort: bundle Coin-HSL DLLs if COINHSL_DIR is set at build time.
hsl_dir = os.environ.get("COINHSL_DIR")
if hsl_dir and os.path.isdir(hsl_dir):
    for fn in os.listdir(hsl_dir):
        if fn.lower().endswith(".dll"):
            binaries.append((os.path.join(hsl_dir, fn), "."))

a = Analysis(
    ["app/main.py"],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="FullModelSim", console=True,        # console=True so the IPOPT log is visible while debugging
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="FullModelSim")
```

- [ ] **Step 4: Write `build/README.md`**

```markdown
# Building the FullModelSim Windows app

## Build (onedir, for debugging)

```powershell
venv\Scripts\Activate.ps1
# optional: bundle Coin-HSL DLLs
$env:COINHSL_DIR = "C:\path\to\coinhsl\bin"
pyinstaller build\windows-app.spec
```

Output: `dist\FullModelSim\FullModelSim.exe`.

## Patch needed for HSL in a frozen app

`functions/hsl.py` must also look for the HSL DLL directory at the frozen
resource root. Add `sys._MEIPASS` to its search order (alongside `COINHSL_DIR`
and the seeded default). If HSL still fails to load, the solve falls back to
MUMPS automatically — the app always solves.

## Acceptance floor (manual, on a clean Windows VM with no Python)

1. Copy `dist\FullModelSim\` to the VM and launch `FullModelSim.exe`.
2. Main tab: circuit `Sturn`, Aero `Static`, ATD on, linear solver `mumps`.
3. Run. The IPOPT iteration log streams into the log pane.
4. On finish: summary shows a lap time; the Output tab lists plots; double-click
   opens one in the browser. Results/Plots are written under
   `%USERPROFILE%\Documents\FullModelSim`.

HSL working in the frozen app is a stretch goal beyond this floor.
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/Scripts/python.exe test_spec_includes.py`
Expected: PASS.

- [ ] **Step 6: Run full suite, then commit**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
git add build/windows-app.spec build/README.md test_spec_includes.py
git commit -m "build: add PyInstaller spec + build/acceptance docs"
```

---

### Task 11: Final suite gate + push

**Files:** none (verification + push only)

- [ ] **Step 1: Run the entire suite**

```bash
for f in test_*.py; do echo "== $f =="; python "$f" || { echo "FAILED: $f"; exit 1; }; done
```
Expected: every file prints its `ALL ... TESTS PASSED` and the loop exits 0. This includes the four pre-existing tests plus `test_paths`, `test_vp_overrides`, `test_mltp_params`, `test_runconfig`, `test_headless_config`, `test_results`, `test_solve_runner`, `test_gui_logic`, `test_presets`, `test_spec_includes`.

- [ ] **Step 2: Push the branch (only after Step 1 is fully green)**

```bash
git push -u origin windows-app
```
Expected: branch published to origin.

- [ ] **Step 3: Report**

State the final lap of work: all tests green, branch pushed. Note the two manual follow-ups that are outside the automated gate: (a) the `functions/hsl.py` frozen-path patch, and (b) the clean-VM acceptance-floor smoke test from `build/README.md`.

---

## Self-Review

**Spec coverage:**
- Standalone exe / clean Windows → Tasks 1 (paths), 10 (spec + acceptance). ✓
- PySide6 GUI, thin/testable logic → Tasks 7, 8. ✓
- Subprocess + live IPOPT log + Cancel → Task 7 (`SolveRunner`), Task 8 (log pane, run/cancel). ✓
- Summary + open-plots-in-browser → Tasks 6, 8. ✓
- Expert params via editable config file + Load picker → Tasks 4 (`expert_config`, merge), 8 (`_pick_expert`/`_load_expert_tier1`). ✓
- Ship `default.json` preset → Task 9. ✓
- vehParams two-phase refactor + propagation + key validation → Task 2. ✓
- Thread `vp_overrides` through userOpts/MLTP/warm start → Tasks 2, 3. ✓
- Documents-folder writable output → Tasks 1 (`default_output_dir`), 5 (results/plots dirs under output), 8. ✓
- ATD+EM4 conflict disabled-in-GUI (no silent flip) → Task 8 (`_on_em4_toggled`). ✓
- Bundling CasADi + Coin-HSL (MUMPS floor) → Task 10. ✓
- Keep existing tests green → Global Constraints + Task 2 regression step. ✓
- Test per feature; commit/push only when green → per-task tests + Global Constraints + Task 11. ✓
- Out-of-scope items (param-optim, QWebEngine, in-app form, batch, live plotting) → not present. ✓

**Placeholder scan:** no TBD/TODO/"handle edge cases"/"similar to"; every code step shows full code. ✓

**Type consistency:** `vp_overrides` is a `dict` everywhere; `RunConfig.write_cfg` emits `vp_overrides` key consumed by `headless_solve.build_solve_kwargs` via `cfg["vp_overrides"]`; `result_mat_path`/`plot_dir` signatures match their callers in `mainwindow._show_results`; `solve_command` shape (`[python, script, cfg]`) matches `test_solve_runner`; `MainWindow.set_em4/atd_enabled/collect_runconfig` match `test_gui_logic`. ✓

**Known manual gaps (intentionally outside the automated gate):** the `functions/hsl.py` frozen-path patch and the clean-VM exe smoke are documented in Task 10 / Task 11 Step 3, not asserted by a test, because they require a built exe and a clean VM.
