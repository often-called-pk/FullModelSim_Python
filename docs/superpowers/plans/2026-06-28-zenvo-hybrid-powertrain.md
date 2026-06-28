# Zenvo Hybrid Powertrain (Drive-Source Abstraction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded `EM4`/`ATD` powertrain branches with a data-driven *drive-source* abstraction, then express the Zenvo Aurora 3-motor + ICE topology as data on top of it — without changing `nx` (stays 23) or breaking the legacy single/four-motor configs.

**Architecture:** A new casadi-free module `functions/drive_sources.py` becomes the single source of truth for (a) the control-vector ordering and (b) the path-constraint name ordering. `userOpts`, `vehModel`, and `MLTP` all loop over the source/split lists this module produces instead of branching on `pt.EM4`/`pt.ATD`. A new `Hybrid='On'` flag selects the Aurora topology, whose per-source ratings/gears + ICE torque curve are in-code constants in `build_topology`'s `hybrid` branch (the Aurora *chassis* numbers are already the baseline car — DP2 was reversed at spec review).

**Tech Stack:** Python 3, NumPy, CasADi (IPOPT), SciPy `.mat` I/O. Tests are plain scripts (assertions at module top level; run a file directly — no pytest).

## Global Constraints

- **`nx` stays 23.** Drive-source count is a control/param concern, never a state concern. Wheel-speed states stay at indices 5–8 (`Om_fl..Om_rr`).
- **Decision-vector packing is column-major (`order='F'`)** in `functions/transcription.py` and is **not** touched; new channels only widen `nu`.
- **Dimension chain must stay equal:** `nu == len(input_keys) == m.u rows == len(u_s) == len(u_min) == len(u_max) == len(ru) == len(rdu) == len(rdu2) == len(duk_lb) == len(duk_ub) == warmstart u0 rows`. Break one → CasADi dimension error or silent mis-scaling.
- **`delta` is always last; `T_brake` always immediately after the source torques;** split fractions occupy the legacy ATD slot (after brake, before aero).
- **`userOpts._col` raises bare `AttributeError` on an unknown channel** (`getattr` with no default). Every channel name emitted by `control_keys` MUST have a matching `_build_c` rate/reg row in the **same** change, or `userOpts` crashes at the `_col` calls.
- **Aero-key naming quirk preserved:** `ActAero==2` → `['FW','RW']`; `ActAero==3` → `['FW','FW','RW','TW']` (two distinct front symbols alias to channel name `'FW'`).
- **Legacy `.mat`/plot contract:** `T_fl..T_rr`, `rho_lim_*`, `E_motor`, `input_keys`, and the `P_motor*`/`Om_motor*` channel prefixes are preserved; new keys are additive; the legacy `cfg` string stays byte-identical (`<AeroConfig>_ATD<ATD>_EM4<EM4>`) for non-hybrid configs.
- **Init 7-state model is unchanged** (lumped single motor; control vector `['T_motor','T_brake','delta']`).
- **Out of scope, do not touch:** the 23-state chassis/suspension/tyre/aero dynamics, the wheel-speed state scales, `gg_plots.py`, and `vehModel_initial.py`.
- **Run a test file with** `venv\Scripts\python.exe <file>.py` from the repo root. A passing file ends with its `ALL ... TESTS PASSED` banner; a failing assert aborts the file at the first `[FAIL]`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `functions/drive_sources.py` | **Create** | Casadi-free single source of truth: `DriveSource`/`SplitPolicy` descriptors, `build_topology`, `control_keys`, `aero_keys`, `node_wheels`, `path_constraint_names`, `source_signal_keys`, `_topology`. |
| `test_drive_sources.py` | **Create** | Casadi-free unit tests for the above (topology, ordering, back-compat, constraint names). |
| `userOpts.py` | Modify | New `Hybrid`/`ice_gear` kwargs; 3-way guard; build + hang `pt.sources/pt.splits/pt.topology` + `ctx.Hybrid`; `input_keys` from `control_keys`; new `_build_c` rows for `T_motor_r`/`T_ice_r`/`split_R`. |
| `vehModel.py` | Modify | Source/split loops for symbol creation, `u_list`, torque split (DP1 gear fix), and signal exposure (`m.src` + legacy aliases). |
| `MLTP.py` | Modify | Per-source `build_path_constraints`; rewritten `warmstart_guesses`; hybrid `cfg`; per-source + `E_fuel` save schema; `data['Hybrid']`. |
| `MLTP_paramOptim.py` | Modify | Topology-aware default param set (drop `Tdist` unless plain `single`); inherits the generalized shared functions. |
| `plotSDI.py` | Modify | Hybrid `cfg` (mirror MLTP); `plot_powertrain` matches `P_ice*`/`Om_ice*` and plots `E_fuel`. |
| `functions/sweep.py` | Modify | Add `'Hybrid'` to `ACCEPTED_COLUMNS`. |
| `test_foundation.py` | Modify | Powertrain asserts → Zenvo baseline numbers. |
| `test_params_useropts.py` | Modify | Mass/Fz0/Rw/gear/Jw/`c_fl` asserts → Zenvo baseline; add a hybrid-config assert. |
| `test_save_load.py` | Modify | Add `T_fl..T_rr`, `E_fuel`, `Hybrid`, per-source channel round-trip. |
| `test_sweep.py` | Modify | Assert `'Hybrid'` accepted; manifest placement. |

---

## Task 1: Fix stale baseline tests (Zenvo numbers)

The branch already carries Zenvo Aurora numbers in `Powertrain.py`/`vehParams.py` (DP2 reversed). `test_foundation.py` and `test_params_useropts.py` still assert the *original* framework numbers and fail at their first powertrain/mass assert. This **must run first**: `test_params_useropts.py` aborts at L21, so its input-key back-compat asserts (L67-95) that later tasks rely on are otherwise unreachable. This is test maintenance (the code is already correct), not TDD.

**Files:**
- Modify: `test_foundation.py:81-86`
- Modify: `test_params_useropts.py:21-30`

**Interfaces:**
- Consumes: current `ctx.pt`/`ctx.vp` values from `Powertrain.py`/`vehParams.py` (already Zenvo).
- Produces: a green casadi-free suite so Tasks 4+ can verify back-compat asserts below L21.

- [ ] **Step 1: Run both files to confirm the current failures**

```
venv\Scripts\python.exe test_foundation.py
venv\Scripts\python.exe test_params_useropts.py
```
Expected: `test_foundation.py` aborts at `[FAIL] Pmax = 456000 W`; `test_params_useropts.py` aborts at `[FAIL] ms = mb+md = 1895`.

- [ ] **Step 2: Update the Powertrain block in `test_foundation.py`**

Replace lines 81-85 (keep line 86 `gear` assert as-is — it is self-consistent):

```python
ok("Pmax = 450000 W", ctx.pt.Pmax == 450000.0)
ok("Tmax = 429 Nm", ctx.pt.Tmax == 429.0)
ok("OMmax = 25000*pi/30", abs(ctx.pt.OMmax - 25000 * np.pi / 30) < 1e-9)
ok("Vmax = 105.56 m/s", abs(ctx.pt.Vmax - 380 / 3.6) < 1e-9)
ok("Rw = 0.35", ctx.vp.Rw == 0.35)
```

- [ ] **Step 3: Update the mass/Fz0/Rw/gear/Jw/c_fl block in `test_params_useropts.py`**

Replace lines 21-30 with (note `Rw_r=0.37`; the Rw/gear quirk is now numerically dissolved — both radii are 0.35):

```python
ok("ms = mb+md = 1592", vp.ms == 1592.0)
ok("m  = ms+muf+mur = 1692", vp.m == 1692.0)
ok("Fz0 = 4300", vp.Fz0 == 4300.0)
ok("Rw=0.35 front, Rw_r=0.37", vp.Rw == 0.35 and vp.Rw_f == 0.35 and vp.Rw_r == 0.37)
ok("gear based on Powertrain Rw=0.35",
   abs(vp.gear - (pt.OMmax * 0.35) / pt.Vmax) < 1e-9)
ok("Jw = 1.6", vp.Jw == 1.6)
# suspension damping: c_fl = zeta*2*sqrt(m_eff_f/2)*k_fl, m_eff_f = m*(1-wB)
m_eff_f = vp.m * (1 - vp.wB)
ok("c_fl matches formula", abs(vp.c_fl - 0.7 * 2 * np.sqrt(m_eff_f / 2) * 40000.0) < 1e-6)
```

- [ ] **Step 4: Run both files to verify they pass end-to-end**

```
venv\Scripts\python.exe test_foundation.py
venv\Scripts\python.exe test_params_useropts.py
```
Expected: each prints its final banner (`ALL FOUNDATION TESTS PASSED` / `ALL vehParams / userOpts TESTS PASSED`).

- [ ] **Step 5: Commit**

```bash
git add test_foundation.py test_params_useropts.py
git commit -m "test: update baseline asserts to Zenvo Aurora numbers (DP2 reversed)"
```

---

## Task 2: `drive_sources.py` — descriptors, topology, control ordering

The casadi-free core. This task delivers `DriveSource`, `SplitPolicy`, `node_wheels`, `aero_keys`, `control_keys`, `_topology`, and `build_topology` (all topologies). Constraint-name + signal-key helpers come in Task 3.

**Files:**
- Create: `functions/drive_sources.py`
- Test: `test_drive_sources.py`

**Interfaces:**
- Produces:
  - `DriveSource(name, type, node, gear, Tmax, Pmax, OMmax, ctrl_key, curve=None, regen=False, eff=0.90)` — dataclass.
  - `SplitPolicy(node, kind, keys)` — dataclass (`kind ∈ {'fixed','atd','tv'}`, `keys` defaults `[]`).
  - `node_wheels(node) -> list[str]`.
  - `aero_keys(ActAero) -> list[str]`.
  - `control_keys(sources, splits, ActAero) -> list[str]`.
  - `_topology(EM4, ATD, Hybrid) -> str` (`'single'|'single_atd'|'four_motor'|'hybrid'`).
  - `build_topology(topology, pt, vp, ice_gear=1.0) -> (sources, splits)`.

- [ ] **Step 1: Write the failing tests** — create `test_drive_sources.py`

```python
"""Casadi-free unit tests for the drive-source abstraction."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from types import SimpleNamespace
import numpy as np

from functions.drive_sources import (
    DriveSource, SplitPolicy, node_wheels, aero_keys, control_keys,
    _topology, build_topology)

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# fake pt/vp with the fields build_topology reads
pt = SimpleNamespace(Pmax=450000.0, Tmax=429.0, OMmax=25000 * np.pi / 30,
                     Vmax=380 / 3.6, eff=0.9)
vp = SimpleNamespace(gear=(25000 * np.pi / 30) * 0.35 / (380 / 3.6),
                     Tdist=0.33, brkB=0.65)

print("node_wheels / aero_keys")
ok("node all -> 4 wheels", node_wheels("all") == ["fl", "fr", "rl", "rr"])
ok("node rear -> rl,rr", node_wheels("rear") == ["rl", "rr"])
ok("node fl -> fl", node_wheels("fl") == ["fl"])
ok("aero 0 -> []", aero_keys(0) == [])
ok("aero 2 -> FW,RW", aero_keys(2) == ["FW", "RW"])
ok("aero 3 -> FW,FW,RW,TW", aero_keys(3) == ["FW", "FW", "RW", "TW"])

print("_topology")
ok("single", _topology(0, 0, 0) == "single")
ok("single+atd", _topology(0, 1, 0) == "single_atd")
ok("four_motor", _topology(1, 0, 0) == "four_motor")
ok("hybrid overrides", _topology(1, 1, 1) == "hybrid")

print("build_topology + control_keys back-compat")
s, sp = build_topology("single", pt, vp)
ok("single one source node all", len(s) == 1 and s[0].node == "all" and s[0].ctrl_key == "T_motor")
ok("single fixed split", len(sp) == 1 and sp[0].kind == "fixed" and sp[0].keys == [])
ok("single Static keys", control_keys(s, sp, 0) == ["T_motor", "T_brake", "delta"])
ok("single AALB keys", control_keys(s, sp, 3) ==
   ["T_motor", "T_brake", "FW", "FW", "RW", "TW", "delta"])

s_atd, sp_atd = build_topology("single_atd", pt, vp)
ok("single+ATD keys", control_keys(s_atd, sp_atd, 0) ==
   ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"])

s4, sp4 = build_topology("four_motor", pt, vp)
ok("four_motor 4 sources", [x.ctrl_key for x in s4] ==
   ["T_motor_fl", "T_motor_fr", "T_motor_rl", "T_motor_rr"])
ok("four_motor no splits", sp4 == [])
ok("four_motor Static keys", control_keys(s4, sp4, 0) ==
   ["T_motor_fl", "T_motor_fr", "T_motor_rl", "T_motor_rr", "T_brake", "delta"])

sh, sph = build_topology("hybrid", pt, vp, ice_gear=1.0)
ok("hybrid source ctrl_keys", [x.ctrl_key for x in sh] ==
   ["T_motor_fl", "T_motor_fr", "T_motor_r", "T_ice_r"])
ok("hybrid has one ice", sum(1 for x in sh if x.type == "ice") == 1)
ok("hybrid tv split on rear", len(sph) == 1 and sph[0].kind == "tv" and sph[0].keys == ["split_R"])
ok("hybrid Static keys", control_keys(sh, sph, 0) ==
   ["T_motor_fl", "T_motor_fr", "T_motor_r", "T_ice_r", "T_brake", "split_R", "delta"])
ok("hybrid front emotor gear 6.0", sh[0].gear == 6.0)
ok("hybrid ice gear = ice_gear*3.5", sh[3].gear == 1.0 * 3.5)
ok("hybrid rear emotor gear = 1.5*ice_gear*3.5", sh[2].gear == 1.5 * 1.0 * 3.5)
ok("hybrid front emotor 150kW/143Nm", sh[0].Pmax == 150e3 and sh[0].Tmax == 143.0)

print("\nALL DRIVE-SOURCE TESTS PASSED")
```

- [ ] **Step 2: Run to verify it fails**

```
venv\Scripts\python.exe test_drive_sources.py
```
Expected: FAIL — `ModuleNotFoundError: No module named 'functions.drive_sources'`.

- [ ] **Step 3: Create `functions/drive_sources.py`**

```python
"""Drive-source abstraction (MPI-free, CasADi-free).

Single source of truth for the powertrain topology, the control-vector
ordering, and the path-constraint name ordering. Consumed by userOpts.py,
vehModel.py and MLTP.py so the control order is defined exactly once.

Topology strings: 'single' | 'single_atd' | 'four_motor' | 'hybrid'.
"""
from dataclasses import dataclass, field


@dataclass
class DriveSource:
    name: str            # 'motor'|'fl'|'fr'|'r'|'ice' (forms channel/constraint names)
    type: str            # 'emotor' | 'ice'
    node: str            # wheels it feeds: 'fl'|'fr'|'rl'|'rr'|'front'|'rear'|'all'
    gear: float          # per-source ratio (motor spins gear x wheel)
    Tmax: float          # torque scale / cap (Nm)
    Pmax: float          # power cap (W)      — emotor flat cap
    OMmax: float         # speed cap (rad/s)  — emotor flat cap / ice redline
    ctrl_key: str        # control-channel name it owns
    curve: tuple = None  # ICE rpm->torque poly coeffs (high->low order); None for emotor
    regen: bool = False  # False -> torque in [0,Tmax]; True -> [-Tmax,Tmax]
    eff: float = 0.90    # post-solve energy efficiency


@dataclass
class SplitPolicy:
    node: str            # multi-wheel node it distributes: 'front'|'rear'|'all'
    kind: str            # 'fixed' | 'atd' | 'tv'
    keys: list = field(default_factory=list)   # fraction-control channel names


_NODE_WHEELS = {
    "all":   ["fl", "fr", "rl", "rr"],
    "front": ["fl", "fr"],
    "rear":  ["rl", "rr"],
    "fl": ["fl"], "fr": ["fr"], "rl": ["rl"], "rr": ["rr"],
}


def node_wheels(node):
    """Wheels driven by a node, in canonical order."""
    return list(_NODE_WHEELS[node])


def aero_keys(ActAero):
    """Active-aero control channels for a given ActAero level (naming quirk preserved)."""
    if ActAero == 1:
        return ["RW"]
    if ActAero == 2:
        return ["FW", "RW"]
    if ActAero == 3:
        return ["FW", "FW", "RW", "TW"]
    return []


def control_keys(sources, splits, ActAero):
    """THE control-vector ordering: sources -> brake -> split fractions -> aero -> delta."""
    keys = [s.ctrl_key for s in sources]
    keys.append("T_brake")
    for sp in splits:
        keys.extend(sp.keys)
    keys.extend(aero_keys(ActAero))
    keys.append("delta")
    return keys


def _topology(EM4, ATD, Hybrid):
    """Map the resolved flags to a topology string. Hybrid wins."""
    if Hybrid == 1:
        return "hybrid"
    if EM4 == 1:
        return "four_motor"
    if ATD == 1:
        return "single_atd"
    return "single"


def build_topology(topology, pt, vp, ice_gear=1.0):
    """Return (sources, splits) for a topology.

    Legacy topologies reuse the shared scalars pt.Pmax/Tmax/OMmax as-is (so
    'single' caps at pt.Pmax as an aggregate and each 'four_motor' source caps
    at pt.Pmax per motor — the existing dual semantics, preserved). Only
    'hybrid' uses explicit per-source ratings/gears (Zenvo Aurora constants).
    """
    if topology in ("single", "single_atd"):
        src = [DriveSource("motor", "emotor", "all", vp.gear,
                           pt.Tmax, pt.Pmax, pt.OMmax, "T_motor", eff=pt.eff)]
        if topology == "single_atd":
            splits = [SplitPolicy("all", "atd", ["ATD", "ATD", "ATD", "ATD"])]
        else:
            splits = [SplitPolicy("all", "fixed", [])]
        return src, splits

    if topology == "four_motor":
        src = [DriveSource(w, "emotor", w, vp.gear, pt.Tmax, pt.Pmax, pt.OMmax,
                           f"T_motor_{w}", eff=pt.eff)
               for w in ("fl", "fr", "rl", "rr")]
        return src, []

    if topology == "hybrid":
        # ----- Zenvo Aurora source list (in-code constants, spec sec.10) -----
        OM_EM = 25000 * 3.141592653589793 / 30.0       # 25000 rpm -> rad/s (2618)
        EM_P, EM_T, EM_EFF = 150e3, 143.0, 0.90        # per front/rear e-motor
        rear_em_gear = 1.5 * ice_gear * 3.5
        ice_drive_gear = ice_gear * 3.5
        # PLACEHOLDER ICE map: flat peak-torque cap until the engine-map image
        # is digitized. Replace `ice_curve` with real coeffs (high->low order)
        # and set ice_peak / ICE_OM_REDLINE / ICE_EFF from the supplied sheet.
        ice_peak = 700.0                               # Nm (hypercar placeholder)
        ice_curve = (ice_peak,)                        # degree-0 poly => flat cap
        ICE_OM_REDLINE = 8000 * 3.141592653589793 / 30.0   # 8000 rpm crank (placeholder)
        ICE_EFF = 0.35                                  # thermal eff (placeholder)
        src = [
            DriveSource("fl", "emotor", "fl", 6.0, EM_T, EM_P, OM_EM, "T_motor_fl", eff=EM_EFF),
            DriveSource("fr", "emotor", "fr", 6.0, EM_T, EM_P, OM_EM, "T_motor_fr", eff=EM_EFF),
            DriveSource("r",  "emotor", "rear", rear_em_gear, EM_T, EM_P, OM_EM, "T_motor_r", eff=EM_EFF),
            DriveSource("ice", "ice", "rear", ice_drive_gear, ice_peak, ice_peak,
                        ICE_OM_REDLINE, "T_ice_r", curve=ice_curve, eff=ICE_EFF),
        ]
        splits = [SplitPolicy("rear", "tv", ["split_R"])]
        return src, splits

    raise ValueError(f"Unknown topology '{topology}'.")
```

- [ ] **Step 4: Run to verify it passes**

```
venv\Scripts\python.exe test_drive_sources.py
```
Expected: `ALL DRIVE-SOURCE TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add functions/drive_sources.py test_drive_sources.py
git commit -m "feat(powertrain): drive-source descriptors, build_topology, control_keys"
```

---

## Task 3: `drive_sources.py` — constraint names + signal keys

Add the two helpers that fix the path-constraint ordering and the save-channel naming, with back-compat to the legacy names.

**Files:**
- Modify: `functions/drive_sources.py`
- Modify: `test_drive_sources.py`

**Interfaces:**
- Produces:
  - `path_constraint_names(sources, splits) -> list[str]` — 4 `rho_lim_*` first, then per-source emotor/ice rows grouped by type, then `ATD_eq` for an `atd` split.
  - `source_signal_keys(source) -> (P_key, Om_key)`.

- [ ] **Step 1: Add failing tests** — insert before the final `print("\nALL DRIVE-SOURCE TESTS PASSED")` line in `test_drive_sources.py`

```python
from functions.drive_sources import path_constraint_names, source_signal_keys

print("path_constraint_names back-compat")
s, sp = build_topology("single", pt, vp)
ok("single ATD0 names", path_constraint_names(s, sp) ==
   ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr",
    "motor_power", "motor_rpm", "BrTh_1"])
s, sp = build_topology("single_atd", pt, vp)
ok("single ATD1 names", path_constraint_names(s, sp) ==
   ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr",
    "motor_power", "motor_rpm", "BrTh_1", "ATD_eq"])
s4, sp4 = build_topology("four_motor", pt, vp)
ok("four_motor names (grouped by type)", path_constraint_names(s4, sp4) ==
   ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr",
    "motor_power_fl", "motor_power_fr", "motor_power_rl", "motor_power_rr",
    "motor_rpm_fl", "motor_rpm_fr", "motor_rpm_rl", "motor_rpm_rr",
    "BrTh_fl", "BrTh_fr", "BrTh_rl", "BrTh_rr"])
sh, sph = build_topology("hybrid", pt, vp)
ok("hybrid names", path_constraint_names(sh, sph) ==
   ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr",
    "motor_power_fl", "motor_power_fr", "motor_power_r",
    "motor_rpm_fl", "motor_rpm_fr", "motor_rpm_r",
    "ice_curve_ice", "ice_rpm_ice",
    "BrTh_fl", "BrTh_fr", "BrTh_r", "BrTh_ice"])

print("source_signal_keys back-compat")
s, _ = build_topology("single", pt, vp)
ok("single signals", source_signal_keys(s[0]) == ("P_motor", "Om_motor"))
s4, _ = build_topology("four_motor", pt, vp)
ok("four_motor fl signals", source_signal_keys(s4[0]) == ("P_motor_fl", "Om_motor_fl"))
sh, _ = build_topology("hybrid", pt, vp)
ok("hybrid rear emotor signals", source_signal_keys(sh[2]) == ("P_motor_r", "Om_motor_r"))
ok("hybrid ice signals", source_signal_keys(sh[3]) == ("P_ice_r", "Om_ice_r"))
```

- [ ] **Step 2: Run to verify it fails**

```
venv\Scripts\python.exe test_drive_sources.py
```
Expected: FAIL — `ImportError: cannot import name 'path_constraint_names'`.

- [ ] **Step 3: Append the two helpers to `functions/drive_sources.py`**

```python
def path_constraint_names(sources, splits):
    """Path-constraint row names: 4 rho_lim first, then powertrain rows, then ATD_eq.

    Grouped by constraint type so the legacy single/four_motor names + order are
    reproduced byte-for-byte. The single aggregate emotor (node 'all') uses the
    unsuffixed legacy names motor_power/motor_rpm/BrTh_1.
    """
    names = ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr"]
    emotors = [s for s in sources if s.type == "emotor"]
    ices = [s for s in sources if s.type == "ice"]
    aggregate = len(emotors) == 1 and emotors[0].node == "all" and not ices
    if aggregate:
        names += ["motor_power", "motor_rpm", "BrTh_1"]
    else:
        names += [f"motor_power_{s.name}" for s in emotors]
        names += [f"motor_rpm_{s.name}" for s in emotors]
        names += [f"ice_curve_{s.name}" for s in ices]
        names += [f"ice_rpm_{s.name}" for s in ices]
        names += [f"BrTh_{s.name}" for s in (emotors + ices)]
    for sp in splits:
        if sp.kind == "atd":
            names.append("ATD_eq")
    return names


def source_signal_keys(source):
    """(power_key, speed_key) for the saved vehicle dict / plot prefixes."""
    if source.type == "ice":
        return "P_ice_r", "Om_ice_r"
    if source.node == "all":
        return "P_motor", "Om_motor"          # legacy single aggregate
    return f"P_motor_{source.name}", f"Om_motor_{source.name}"
```

- [ ] **Step 4: Run to verify it passes**

```
venv\Scripts\python.exe test_drive_sources.py
```
Expected: `ALL DRIVE-SOURCE TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add functions/drive_sources.py test_drive_sources.py
git commit -m "feat(powertrain): path_constraint_names + source_signal_keys with legacy back-compat"
```

---

## Task 4: Rewire `userOpts.py` onto drive_sources

Replace the `EM4`/`ATD` key/guard logic with the abstraction; add the `Hybrid`/`ice_gear` flags; add the three new rate/reg rows in the same change (or `_col` raises).

**Files:**
- Modify: `userOpts.py` (`_build_c` 104-146; remove `_input_keys` 149-167; guard + flags 201-213; key build ≈272)
- Modify: `test_params_useropts.py` (add a hybrid-config assert)

**Interfaces:**
- Consumes: `build_topology`, `control_keys`, `_topology` from Task 2.
- Produces: `ctx.input_keys`; `ctx.Hybrid` (`'On'/'Off'`); `pt.topology` (str), `pt.sources` (list[DriveSource]), `pt.splits` (list[SplitPolicy]), `pt.Hybrid` (0/1); `pt.EM4`/`pt.ATD` retained as derived compat shims.

- [ ] **Step 1: Add a failing hybrid-config test** — insert before the final `print("\nALL vehParams / userOpts TESTS PASSED")` line in `test_params_useropts.py`

```python
# Hybrid overrides EM4/ATD and produces the Aurora control order
ctx4 = Ctx()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    userOpts(ctx4, circuit="Straight", Hybrid="On", ATD="On", Electric_4Motors="On")
ok("Hybrid topology selected", ctx4.pt.topology == "hybrid")
ok("Hybrid forces EM4/ATD off", ctx4.pt.EM4 == 0 and ctx4.pt.ATD == 0)
ok("Hybrid keys", ctx4.input_keys ==
   ["T_motor_fl", "T_motor_fr", "T_motor_r", "T_ice_r", "T_brake", "split_R", "delta"])
ok("Hybrid duk_ub shape matches keys", ctx4.duk_ub.shape == (7, 1))
ok("Hybrid rdu2 split_R=1.5", np.allclose(
   ctx4.rdu2.ravel(), [0, 0, 0, 0, 0, 1.5, 15]))
```

- [ ] **Step 2: Run to verify it fails**

```
venv\Scripts\python.exe test_params_useropts.py
```
Expected: FAIL — `TypeError: userOpts() got an unexpected keyword argument 'Hybrid'`.

- [ ] **Step 3: Add the new rate/reg rows in `_build_c`** (`userOpts.py`)

In the `rate` dict (after the `"T_motor_rr": (2e4, -2e4),` line) add:

```python
        "T_motor_r":  (2e4, -2e4),
        "T_ice_r":    (2e4, -2e4),
        "split_R":    (2e4, -2e4),
```

In the `reg` dict (after the `"T_motor_rr": (0.0, 0.0, 0.0),` line) add:

```python
        "T_motor_r":  (0.0, 0.0, 0.0),
        "T_ice_r":    (0.0, 0.0, 0.0),
        "split_R":    (0.0, 0.0, 1.5),
```

- [ ] **Step 4: Replace `_input_keys` with the drive_sources import**

Delete the entire `_input_keys` function (lines 149-167). Add to the import block near the top (after `from functions.importfile import mat_to_namespace`):

```python
from functions.drive_sources import build_topology, control_keys, _topology
```

- [ ] **Step 5: Add the `Hybrid`/`ice_gear` kwargs and rewrite the guard**

Change the signature (add two kwargs):

```python
def userOpts(ctx,
             AeroConfig="Static",          # 'Static' | 'Active_RW' | 'Active' | 'AALB'
             ATD="On",                     # 'On' | 'Off'
             Electric_4Motors="Off",       # 'On' | 'Off'
             Hybrid="Off",                 # 'On' | 'Off'  (overrides ATD/EM4)
             circuit="BCN",
             vi=60.0,                      # initial velocity [m/s]
             ni=np.nan,                    # initial lateral position [m]
             ice_gear=1.0,                 # fixed ICE gear (default 6th = 1.0)
             circuits_dir="Circuits",
             data_dir="Data",
             linear_solver="ma57",         # 'ma57'|'ma97'|'ma27'|'mumps'; ma* uses Coin-HSL
             hsl_dir=None):                # Coin-HSL bin dir; None -> COINHSL_DIR env / default
```

Replace the guard + flag block (current lines 201-213) with:

```python
    # conflict guard (3-way). Hybrid overrides EM4/ATD; otherwise the legacy
    # "ATD and 4 motors cannot both be On" rule applies.
    if Hybrid == "On":
        if ATD == "On" or Electric_4Motors == "On":
            warnings.warn("Hybrid='On' overrides ATD/Electric_4Motors; "
                          "both forced Off.")
        ATD = "Off"
        Electric_4Motors = "Off"
    elif ATD == "On" and Electric_4Motors == "On":
        warnings.warn("Electric_4Motors and ATD cannot both be On. "
                      "Setting ATD to Off and ElectricMotors to On.")
        ATD = "Off"
        Electric_4Motors = "On"
    pt.ATD = 1 if ATD == "On" else 0
    pt.EM4 = 1 if Electric_4Motors == "On" else 0
    pt.Hybrid = 1 if Hybrid == "On" else 0

    # build the drive-source topology (single source of truth for ordering)
    pt.topology = _topology(pt.EM4, pt.ATD, pt.Hybrid)
    pt.sources, pt.splits = build_topology(pt.topology, pt, vp, ice_gear=ice_gear)

    ctx.AeroConfig = AeroConfig
    ctx.ATD = ATD
    ctx.Electric_4Motors = Electric_4Motors
    ctx.Hybrid = Hybrid
    ctx.circuit = circuit
```

- [ ] **Step 6: Build `input_keys` from `control_keys`**

Replace the line `keys = _input_keys(pt.EM4, pt.ATD, vp.ActAero)` (≈272) with:

```python
    keys = control_keys(pt.sources, pt.splits, vp.ActAero)
```

- [ ] **Step 7: Run the full casadi-free config suite**

```
venv\Scripts\python.exe test_params_useropts.py
venv\Scripts\python.exe test_drive_sources.py
```
Expected: both pass — including the legacy `default keys` (L67-68), `EM4 keys` (L84-85), `AALB keys` (L92-93), positional `duk_ub`/`rdu2` (L70-73, L94-95), and the new Hybrid asserts.

- [ ] **Step 8: Commit**

```bash
git add userOpts.py test_params_useropts.py
git commit -m "feat(powertrain): userOpts on drive_sources (Hybrid flag, 3-way guard, new rate/reg rows)"
```

---

## Task 5: `vehModel.py` — consume the source list (symbols, controls, split, signals)

One atomic refactor: replacing the motor-torque symbols renames the locals every later block uses, so symbol creation, `u_list` assembly, the torque split (with the DP1 gear fix), and signal exposure all change together. Legacy aliases are preserved so `MLTP`'s still-unmodified `build_path_constraints` keeps working after this task.

**Files:**
- Modify: `vehModel.py` (import; symbol block 88-97; ATD block 108-114; `u_list` 139-166; torque split 476-508; signal exposure 574-588)

**Interfaces:**
- Consumes: `pt.sources`, `pt.splits` (Task 4); `node_wheels` (Task 2); existing locals `Om_fl..Om_rr`, `xs_fl..xs_rr`, `T_brake`/`T_brake_n`.
- Produces on `ctx.m23`: `m.src[name] = SimpleNamespace(T, T_n, Om, P, source)` per source; legacy aliases (`m.T_motor/Om_motor/P_motor/T_motor_n`; `m.T_motor_*/Om_motor_*/P_motor_*/T_motor_*_n`; `m.T_ice_r/Om_ice_r/P_ice_r/T_ice_r_n`; `m.ATD_*`); always `m.T_fl..T_rr`, `m.T_brake_n`, `m.Om_fl..Om_rr`.

- [ ] **Step 1: Add the import** (with the other imports at the top of `vehModel.py`)

```python
from functions.drive_sources import node_wheels
```

- [ ] **Step 2: Replace the motor-torque symbol block (lines 88-97)** with a source loop

```python
    # ===================== model (A): inputs (config-dependent) ===========
    # source torques (one normalized symbol per drive source)
    T_by_key = {}        # ctrl_key -> physical torque SX
    T_n_by_key = {}      # ctrl_key -> normalized torque SX
    T_s_by_key = {}      # ctrl_key -> scale
    T_lim_by_key = {}    # ctrl_key -> [lo,hi] normalized limit row
    for s in pt.sources:
        sym = SX.sym(f"{s.ctrl_key}_n")
        T_n_by_key[s.ctrl_key] = sym
        T_s_by_key[s.ctrl_key] = s.Tmax
        T_by_key[s.ctrl_key] = s.Tmax * sym
        lo = -1.0 if s.regen else 0.0
        T_lim_by_key[s.ctrl_key] = np.array([lo, 1.0])
```

- [ ] **Step 3: Replace the ATD symbol block (lines 108-114)** with a positional split-fraction builder

```python
    # split-fraction controls (ATD x4 or split_R), one normalized symbol each,
    # built positionally so the four same-named 'ATD' channels are distinct.
    frac_syms = []       # list of (ctrl_key, SX) in split order
    for sp in pt.splits:
        for k in sp.keys:
            frac_syms.append((k, SX.sym(f"{k}_n_{len(frac_syms)}")))
    if any(sp.kind == "atd" for sp in pt.splits):
        atd = [sym for (k, sym) in frac_syms if k == "ATD"]
        ATD_FL, ATD_FR, ATD_RL, ATD_RR = atd[0], atd[1], atd[2], atd[3]
    split_R = next((sym for (k, sym) in frac_syms if k == "split_R"), None)
    frac_lim = np.array([0.0, 1.0])
```

- [ ] **Step 4: Replace the `u_list` assembly (lines 139-166)** with the `control_keys` order

```python
    # assemble u, u_s, u_lim in control_keys order: sources, brake, fracs, aero, delta
    u_list = []   # (sym_norm, scale, lim_row)
    for s in pt.sources:
        u_list.append((T_n_by_key[s.ctrl_key], T_s_by_key[s.ctrl_key], T_lim_by_key[s.ctrl_key]))
    u_list.append((T_brake_n, T_brake_s, T_brake_lim))
    for (k, sym) in frac_syms:
        u_list.append((sym, 1.0, frac_lim))
    if vp.ActAero == 1:
        u_list.append((activeAeroRW_n, activeAeroRW_s, activeAeroRW_lim))
    elif vp.ActAero == 2:
        u_list += [(activeAeroFW_n, activeAeroFW_s, activeAeroFW_lim),
                   (activeAeroRW_n, activeAeroRW_s, activeAeroRW_lim)]
    elif vp.ActAero == 3:
        u_list += [(activeAeroFL_n, activeAeroFL_s, activeAeroFL_lim),
                   (activeAeroFR_n, activeAeroFR_s, activeAeroFR_lim),
                   (activeAeroRW_n, activeAeroRW_s, activeAeroRW_lim),
                   (activeAeroTW_n, activeAeroTW_s, activeAeroTW_lim)]
    u_list.append((delta_n, delta_max, delta_lim))

    u = ca.vertcat(*[s for (s, _, _) in u_list])
    u_s = np.array([sc for (_, sc, _) in u_list], dtype=float)
    u_lim = np.vstack([lr for (_, _, lr) in u_list])
    u_min, u_max = u_lim[:, 0], u_lim[:, 1]
    nu = u.shape[0]
```

- [ ] **Step 5: Replace the torque-split block (lines 476-508)** with the source loop + DP1 gear fix

```python
    # ===================== powertrain torque split ========================
    Om_w = {"fl": Om_fl, "fr": Om_fr, "rl": Om_rl, "rr": Om_rr}

    def _share(node, wheel):
        """Fraction of a node's torque sent to `wheel` (0 if not driven)."""
        if wheel not in node_wheels(node):
            return 0
        if node in ("fl", "fr", "rl", "rr"):
            return 1
        pol = next((sp for sp in pt.splits if sp.node == node), None)
        if pol is None or pol.kind == "fixed":
            return 0.5 * (1 - vp.Tdist) if wheel in ("fl", "fr") else 0.5 * vp.Tdist
        if pol.kind == "atd":
            return {"fl": ATD_FL, "fr": ATD_FR, "rl": ATD_RL, "rr": ATD_RR}[wheel]
        if pol.kind == "tv":
            return split_R if wheel == "rl" else (1 - split_R)
        raise ValueError(f"Unknown split kind '{pol.kind}'")

    def _brake_term(wheel):
        if any(sp.kind == "atd" for sp in pt.splits):
            return {"fl": ATD_FL, "fr": ATD_FR, "rl": ATD_RL, "rr": ATD_RR}[wheel] * T_brake * 2
        return T_brake * vp.brkB if wheel in ("fl", "fr") else T_brake * (1 - vp.brkB)

    # per-wheel total drive torque (DP1: Om_source = mean(node wheels) x gear)
    T_phys = {"fl": 0, "fr": 0, "rl": 0, "rr": 0}
    for s in pt.sources:
        for w in node_wheels(s.node):
            T_phys[w] = T_phys[w] + T_by_key[s.ctrl_key] * s.gear * _share(s.node, w)
    T_fl = T_phys["fl"] + _brake_term("fl")
    T_fr = T_phys["fr"] + _brake_term("fr")
    T_rl = T_phys["rl"] + _brake_term("rl")
    T_rr = T_phys["rr"] + _brake_term("rr")
    T_fl = ca.if_else(xs_fl >= 0.075, 0, T_fl)
    T_fr = ca.if_else(xs_fr >= 0.075, 0, T_fr)
    T_rl = ca.if_else(xs_rl >= 0.075, 0, T_rl)
    T_rr = ca.if_else(xs_rr >= 0.075, 0, T_rr)

    # per-source speed / power
    m_src = {}
    for s in pt.sources:
        wl = node_wheels(s.node)
        Om_mean = sum(Om_w[w] for w in wl) / len(wl)
        Om_s = Om_mean * s.gear
        T_s = T_by_key[s.ctrl_key]
        m_src[s.name] = SimpleNamespace(T=T_s, T_n=T_n_by_key[s.ctrl_key],
                                        Om=Om_s, P=T_s * Om_s, source=s)
```

> Reproduces legacy single (`node='all'`: `Om_s = mean(4 wheels)*gear` as old line 478; fixed split share `0.5*(1-Tdist)` front; brake `T_brake*brkB`) and four_motor (`Om_s = Om_w*gear` — **was `/gear`**, the DP1 fix; `T_w = T_motor_w*gear + brake`).

- [ ] **Step 6: Replace the powertrain signal-exposure block (lines 574-588)**

```python
    # torques / powertrain
    m.T_fl, m.T_fr, m.T_rl, m.T_rr = T_fl, T_fr, T_rl, T_rr
    m.Lon_acc, m.Lat_acc = Lon_acc, Lat_acc
    m.src = m_src
    m.Om_fl, m.Om_fr, m.Om_rl, m.Om_rr = Om_fl, Om_fr, Om_rl, Om_rr
    for s in pt.sources:
        sd = m_src[s.name]
        if s.type == "ice":
            m.T_ice_r, m.T_ice_r_n, m.Om_ice_r, m.P_ice_r = sd.T, sd.T_n, sd.Om, sd.P
        elif s.node == "all":
            m.T_motor, m.T_motor_n, m.Om_motor, m.P_motor = sd.T, sd.T_n, sd.Om, sd.P
        else:
            setattr(m, f"T_motor_{s.name}", sd.T)
            setattr(m, f"T_motor_{s.name}_n", sd.T_n)
            setattr(m, f"Om_motor_{s.name}", sd.Om)
            setattr(m, f"P_motor_{s.name}", sd.P)
    if any(sp.kind == "atd" for sp in pt.splits):
        m.ATD_FL, m.ATD_FR, m.ATD_RL, m.ATD_RR = ATD_FL, ATD_FR, ATD_RL, ATD_RR
```

> The old lines 574-575 (`m.T_fl..` / `m.Lon_acc..`) and the old `if pt.EM4 == 0:` block (576-588) are entirely replaced by the block above — make sure the assignment happens only once.

- [ ] **Step 7: Smoke-build EM4=0 / EM4=1 / hybrid and check dimensions + signals**

```
venv\Scripts\python.exe -c "from functions.context import Ctx; from userOpts import userOpts; from vehModel import vehModel; \
import warnings; warnings.simplefilter('ignore'); \
[ (lambda c: (userOpts(c, circuit='Straight', **k), vehModel(c), \
   print(k, 'nx', c.m23.nx, 'nu', c.m23.nu, 'keys', len(c.input_keys), 'src', sorted(c.m23.src)), \
   __import__('sys').exit(1) if c.m23.nu != len(c.input_keys) or c.m23.nx != 23 else None))(Ctx()) \
  for k in ({'ATD':'On','Electric_4Motors':'Off'}, {'ATD':'Off','Electric_4Motors':'On'}, {'Hybrid':'On'}) ]"
```
Expected: three lines; single `nu 7`, four `nu 6`, hybrid `nu 7 ... src ['fl', 'fr', 'ice', 'r']`; `nx 23` throughout; no exit-1.

- [ ] **Step 8: Confirm the legacy EM4=0 solve still runs** (old `build_path_constraints` + new aliases)

```
venv\Scripts\python.exe -c "from MLTP import MLTP; import warnings; warnings.simplefilter('ignore'); \
ctx=MLTP(circuit='Sturn', vi=60.0, ATD='On', Electric_4Motors='Off', save=False, plot=False); \
print('EM4=0 still solves, lap', round(ctx.data.lap_time,3))"
```
Expected: a finite lap time (the EM4=0 path is unchanged by the DP1 fix — single uses multiply already).

- [ ] **Step 9: Commit**

```bash
git add vehModel.py
git commit -m "refactor(vehModel): source/split-driven symbols, controls, torque split (DP1 fix), m.src signals"
```

---

## Task 6: `MLTP.build_path_constraints` — per-source generation

Replace the `EM4`/`ATD` constraint branches with a per-source generator that assembles in `path_constraint_names` order (legacy names byte-identical; ICE rows added).

**Files:**
- Modify: `MLTP.py` (import; `build_path_constraints` 41-93)

**Interfaces:**
- Consumes: `m.src` (Task 5), `m.ATD_*`, `m.T_brake_n`; `path_constraint_names` (Task 3); per-source `s.Pmax/s.OMmax/s.Tmax/s.curve` (Task 2).
- Produces: `(hnames, h, h_lb, h_ub)` with `len(hnames)==h.shape[0]`, 4 `rho_lim` first, finite bounds.

- [ ] **Step 1: Add the import** (top of `MLTP.py`)

```python
from functions.drive_sources import path_constraint_names
```

- [ ] **Step 2: Replace `build_path_constraints` (lines 41-93)**

```python
def _poly(coeffs, x):
    """Evaluate a polynomial (coeffs high->low order) at x (CasADi-compatible)."""
    y = 0
    for c in coeffs:
        y = y * x + c
    return y


def build_path_constraints(ca, m, pt):
    """Config-dependent path constraints (friction circles + powertrain limits).
    Returns (hnames, h_expr, h_lb, h_ub). Names/order come from
    drive_sources.path_constraint_names; per-source ratings come from pt.sources."""
    def rho(fx, fy, mux, muy, fz):
        return ca.sqrt((fx / (mux * fz))**2 + (fy / (muy * fz))**2)

    exprs = {}   # name -> (expr, lb, ub)
    exprs["rho_lim_fl"] = (rho(m.fx_fl, m.fy_fl, m.mu_fl_x, m.mu_fl_y, m.fz_fl), 0.0, 1.0)
    exprs["rho_lim_fr"] = (rho(m.fx_fr, m.fy_fr, m.mu_fr_x, m.mu_fr_y, m.fz_fr), 0.0, 1.0)
    exprs["rho_lim_rl"] = (rho(m.fx_rl, m.fy_rl, m.mu_rl_x, m.mu_rl_y, m.fz_rl), 0.0, 1.0)
    exprs["rho_lim_rr"] = (rho(m.fx_rr, m.fy_rr, m.mu_rr_x, m.mu_rr_y, m.fz_rr), 0.0, 1.0)

    for s in pt.sources:
        sd = m.src[s.name]
        BrTh = (sd.T_n * m.T_brake_n) / 1e-3
        if s.type == "emotor":
            mp = (s.Pmax - sd.Om * sd.T) / s.Pmax
            mr = (s.OMmax - sd.Om) / s.OMmax
            if s.node == "all":
                exprs["motor_power"] = (mp, 0.0, 1.0)
                exprs["motor_rpm"] = (mr, 0.0, 1.0)
                exprs["BrTh_1"] = (BrTh, -1.0, 1.0)
            else:
                exprs[f"motor_power_{s.name}"] = (mp, 0.0, 1.0)
                exprs[f"motor_rpm_{s.name}"] = (mr, 0.0, 1.0)
                exprs[f"BrTh_{s.name}"] = (BrTh, -1.0, 1.0)
        else:  # ice: torque under the (placeholder) curve + redline; ub non-binding
            ic = (_poly(s.curve, sd.Om) - sd.T) / s.Tmax
            ir = (s.OMmax - sd.Om) / s.OMmax
            exprs[f"ice_curve_{s.name}"] = (ic, 0.0, 5.0)
            exprs[f"ice_rpm_{s.name}"] = (ir, 0.0, 1.0)
            exprs[f"BrTh_{s.name}"] = (BrTh, -1.0, 1.0)

    for sp in pt.splits:
        if sp.kind == "atd":
            ATD_eq = 1 - (m.ATD_FL + m.ATD_FR + m.ATD_RL + m.ATD_RR)
            exprs["ATD_eq"] = (ATD_eq, -1e-3, 1e-3)

    hnames = path_constraint_names(pt.sources, pt.splits)
    h = ca.vertcat(*[exprs[n][0] for n in hnames])
    h_lb = np.array([exprs[n][1] for n in hnames], dtype=float)
    h_ub = np.array([exprs[n][2] for n in hnames], dtype=float)
    assert len(hnames) == h.shape[0], "Number of path constraints not consistent"
    return hnames, h, h_lb, h_ub
```

- [ ] **Step 3: Verify the constraint set builds for all three topologies**

```
venv\Scripts\python.exe -c "import casadi as ca; from functions.context import Ctx; \
from userOpts import userOpts; from vehModel import vehModel; from MLTP import build_path_constraints; \
import warnings; warnings.simplefilter('ignore'); \
[ (lambda c: (userOpts(c, circuit='Straight', **k), vehModel(c), \
   (lambda r: print(k, len(r[0]), r[0][:5]))(build_path_constraints(ca, c.m23, c.pt))))(Ctx()) \
  for k in ({'ATD':'On','Electric_4Motors':'Off'}, {'ATD':'Off','Electric_4Motors':'On'}, {'Hybrid':'On'}) ]"
```
Expected: single → `8`, four_motor → `16`, hybrid → `16`; each list starts with the four `rho_lim_*` names.

- [ ] **Step 4: Commit**

```bash
git add MLTP.py
git commit -m "refactor(MLTP): per-source build_path_constraints (emotor + ICE)"
```

---

## Task 7: `MLTP.warmstart_guesses` — source-aware seeds

Rewrite the warm-start input loop to iterate `pt.sources` (seed each source torque by `Tmax`-share of the lumped init drive, `split_R=0.5`) so no `u0` row is silently dropped. Then run the EM4=0/EM4=1 full-solve regression.

**Files:**
- Modify: `MLTP.py` (`warmstart_guesses` input loop 116-131)

**Interfaces:**
- Consumes: `ctx.pt.sources`, `ctx.input_keys`, init drive `init_u` (3-row: `T_motor,T_brake,delta`).
- Produces: `guesses['u0']` with exactly `len(ctx.input_keys)` rows.

- [ ] **Step 1: Replace the input-seed loop (lines 116-131)**

```python
    T_brake_0 = _interp_to(grid, init_u[1])
    delta_0 = _interp_to(grid, init_u[2])
    T_motor_0 = _interp_to(grid, init_u[0])
    src_by_key = {s.ctrl_key: s for s in ctx.pt.sources}
    total_Tmax = sum(s.Tmax for s in ctx.pt.sources)
    u0_rows = []
    for key in ctx.input_keys:
        if key in src_by_key:                      # source torque: Tmax-share of init drive
            u0_rows.append(T_motor_0 * (src_by_key[key].Tmax / total_Tmax))
        elif key == "T_brake":
            u0_rows.append(T_brake_0)
        elif key == "split_R":
            u0_rows.append(0.5 * np.ones(N + 1))
        elif key == "ATD":
            u0_rows.append(0.25 * np.ones(N + 1))
        elif key in ("FW", "RW", "TW"):
            u0_rows.append(np.zeros(N + 1))
        elif key == "delta":
            u0_rows.append(delta_0)
        else:
            raise KeyError(f"warmstart_guesses: no seed for input channel '{key}'")
    u0 = np.vstack(u0_rows) / m.u_s[:, None]
```

> `else: raise` replaces the old silent fall-through, so any future channel without a seed fails loudly instead of dropping a row. Four-motor seeds change from `T_motor_0` per motor to `T_motor_0/4` (Tmax-share) — a warm-start guess only; no test pins it.

- [ ] **Step 2: EM4=0 full-solve regression** (short synthetic solve)

```
venv\Scripts\python.exe -c "from MLTP import MLTP; import warnings; warnings.simplefilter('ignore'); \
ctx=MLTP(circuit='Sturn', vi=60.0, ATD='On', Electric_4Motors='Off', save=False, plot=False); \
print('EM4=0 status', ctx.solve_stats['return_status'], 'lap', round(ctx.data.lap_time,3))"
```
Expected: a finite lap time and a normal IPOPT return status (e.g. `Solve_Succeeded`).

- [ ] **Step 3: EM4=1 full-solve regression**

```
venv\Scripts\python.exe -c "from MLTP import MLTP; import warnings; warnings.simplefilter('ignore'); \
ctx=MLTP(circuit='Sturn', vi=60.0, ATD='Off', Electric_4Motors='On', save=False, plot=False); \
print('EM4=1 status', ctx.solve_stats['return_status'], 'lap', round(ctx.data.lap_time,3))"
```
Expected: a finite lap time (numbers differ from pre-DP1-fix runs — this is the intended EM4=1 change).

- [ ] **Step 4: Commit**

```bash
git add MLTP.py
git commit -m "refactor(MLTP): source-aware warmstart seeds (no dropped u0 rows)"
```

---

## Task 8: Save schema, hybrid `cfg`, plots, sweep

Wire the additive outputs: per-source channels + `E_fuel`, the hybrid `cfg` string (MLTP **and** plotSDI), the plot prefix extension, the sweep column, and `.mat` round-trip tests.

**Files:**
- Modify: `MLTP.py` (import; veh_syms 226-232; energy 238-247; data dict + cfg 261-268)
- Modify: `plotSDI.py` (cfg 233; `plot_powertrain` 195-203)
- Modify: `functions/sweep.py` (`ACCEPTED_COLUMNS` 13-16)
- Modify: `test_save_load.py`, `test_sweep.py`

**Interfaces:**
- Consumes: `source_signal_keys` (Task 3), `m.src` + `m.P_ice_r/Om_ice_r` (Task 5), `ctx.Hybrid`.
- Produces: saved `vehicle` dict with per-source `P_*/Om_*`, `E_motor`, `E_fuel` (when an ICE exists), `data['Hybrid']`; hybrid `cfg = "<AeroConfig>_Hybrid"`.

- [ ] **Step 1: Add the import** (top of `MLTP.py`)

```python
from functions.drive_sources import source_signal_keys
```

- [ ] **Step 2: Replace the EM4 veh_syms branch (lines 226-232)** with a source loop

```python
    for s in pt.sources:
        p_key, om_key = source_signal_keys(s)
        sd = m.src[s.name]
        veh_syms += [(p_key, sd.P), (om_key, sd.Om)]
```

- [ ] **Step 3: Replace the energy block (lines 238-247)** with per-source split

```python
    # energy [kWh] — electric (back-compat) + fuel (ICE)
    def _integ(power, eff):
        E = np.zeros(N + 1)
        for i in range(N):
            E[i + 1] = E[i] + 0.5 * (power[i] + power[i + 1]) \
                * (t_opt[i + 1] - t_opt[i]) * 2.7778e-4 / eff
        return E

    emotors = [s for s in pt.sources if s.type == "emotor"]
    ices = [s for s in pt.sources if s.type == "ice"]
    P_em = sum(vehicle[source_signal_keys(s)[0]] for s in emotors)
    vehicle["E_motor"] = _integ(P_em, pt.eff)
    if ices:
        P_ice = sum(vehicle[source_signal_keys(s)[0]] for s in ices)
        vehicle["E_fuel"] = _integ(P_ice, ices[0].eff)
```

- [ ] **Step 4: Add `Hybrid` to the data dict and the hybrid `cfg`** (lines 261, 267)

Change the config line in the `data` dict (line 261) to add `Hybrid`:

```python
        "AeroConfig": AeroConfig, "ATD": ctx.ATD, "EM4": ctx.Electric_4Motors,
        "Hybrid": ctx.Hybrid,
```

Replace the `cfg` line (267) inside the `if save:` block:

```python
        if ctx.Hybrid == "On":
            cfg = f"{AeroConfig}_Hybrid"
        else:
            cfg = f"{AeroConfig}_ATD{ctx.ATD}_EM4{ctx.Electric_4Motors}"
```

- [ ] **Step 5: Mirror the `cfg` in `plotSDI.py` (line 233)**

```python
    if str(_get(data, "Hybrid", "")) == "On":
        cfg = f"{_get(data, 'AeroConfig', 'cfg')}_Hybrid"
    else:
        cfg = f"{_get(data, 'AeroConfig', 'cfg')}_ATD{_get(data, 'ATD', '')}_EM4{_get(data, 'EM4', '')}"
```

- [ ] **Step 6: Extend `plot_powertrain` (lines 195-203)** to include ICE + `E_fuel`

```python
    # power
    for key in [k for k in veh if k.startswith(("P_motor", "P_ice"))]:
        fig.add_trace(go.Scatter(x=s_knot, y=_arr(veh[key]) * 1e-3, name=key), row=1, col=2)
    # speed
    for key in [k for k in veh if k.startswith(("Om_motor", "Om_ice"))]:
        fig.add_trace(go.Scatter(x=s_knot, y=_arr(veh[key]), name=key), row=2, col=1)
    # energy
    for ekey in ("E_motor", "E_fuel"):
        if ekey in veh:
            fig.add_trace(go.Scatter(x=t_opt, y=_arr(veh[ekey]), name=ekey), row=2, col=2)
            fig.update_xaxes(title_text="t [s]", row=2, col=2)
```

- [ ] **Step 7: Add `'Hybrid'` to `ACCEPTED_COLUMNS` (functions/sweep.py 13-16)**

```python
ACCEPTED_COLUMNS = (
    "circuit", "vi", "ni", "warm_start", "AeroConfig", "ATD",
    "Electric_4Motors", "Hybrid", "TyreModel", "linear_solver",
)
```

- [ ] **Step 8: Add hybrid round-trip data to `test_save_load.py`**

In the fixture's `vehicle` dict construction add `T_fl..T_rr` + ICE + fuel channels, and set `data["Hybrid"]`. Place these right after the existing `vehicle = {...}` / `data = {...}` literals (adjust the array length `6` to match the fixture's `N+1`):

```python
vehicle["T_fl"] = np.full(6, 100.0); vehicle["T_fr"] = np.full(6, 100.0)
vehicle["T_rl"] = np.full(6, 80.0); vehicle["T_rr"] = np.full(6, 80.0)
vehicle["P_ice_r"] = np.full(6, 5e4); vehicle["Om_ice_r"] = np.full(6, 300.0)
vehicle["E_fuel"] = np.linspace(0.0, 0.4, 6)
data["Hybrid"] = "On"
```

After the existing reload asserts add:

```python
ok("T_fl recovered", np.allclose(d2.vehicle.T_fl.reshape(-1), vehicle["T_fl"]))
ok("P_ice_r recovered", np.allclose(d2.vehicle.P_ice_r.reshape(-1), vehicle["P_ice_r"]))
ok("E_fuel recovered", np.allclose(d2.vehicle.E_fuel.reshape(-1), vehicle["E_fuel"]))
ok("Hybrid flag recovered", str(d2.Hybrid) == "On")
```

- [ ] **Step 9: Add a `'Hybrid'` column assert to `test_sweep.py`** (place near the other `case_kwargs` asserts)

```python
ok("Hybrid is an accepted column", "Hybrid" in sweep.ACCEPTED_COLUMNS)
ok("Hybrid case kwargs pass", sweep.case_kwargs({"case_id": "h", "Hybrid": "On"}) ==
   {"Hybrid": "On", "plot": False})
```

- [ ] **Step 10: Run the affected casadi-free tests**

```
venv\Scripts\python.exe test_save_load.py
venv\Scripts\python.exe test_sweep.py
```
Expected: both print their final `ALL ... TESTS PASSED` banners.

- [ ] **Step 11: Commit**

```bash
git add MLTP.py plotSDI.py functions/sweep.py test_save_load.py test_sweep.py
git commit -m "feat(powertrain): per-source + E_fuel save, hybrid cfg, plot/sweep wiring"
```

---

## Task 9: Co-optimization wrappers — topology-aware defaults + regression

`MLTP_paramOptim`/`MLTP_TyreOptim` import `build_path_constraints` and `warmstart_guesses` from `MLTP` — both changed — so they inherit the generalization but need a regression. Also make the default param set topology-aware: `Tdist` only feeds the `fixed` split (plain `single`), so promoting it elsewhere creates an unused, unconstrained decision variable.

**Files:**
- Modify: `MLTP_paramOptim.py` (`MLTP_paramOptim` default `params` 123-131)

**Interfaces:**
- Consumes: the refactored `build_path_constraints`/`warmstart_guesses` (via `from MLTP import ...`, unchanged import line); `Hybrid`/`Electric_4Motors`/`ATD` flags.
- Produces: a default param list without `Tdist` unless the topology is plain `single`.

- [ ] **Step 1: Make the default param list topology-aware** — replace the `if params is None:` block (lines 123-131)

```python
    if params is None:
        # Tdist only feeds the 'fixed' split (plain single); a no-op (and an
        # unconstrained decision var) for single_atd / four_motor / hybrid.
        single = (ATD != "On" and Electric_4Motors != "On"
                  and useropts_kwargs.get("Hybrid", "Off") != "On")
        params = [("brkB", 0.0, 1.0)]              # brake balance (front fraction)
        if single:
            params.append(("Tdist", 0.0, 1.0))     # torque distribution (rear fraction)
        params += [
            ("alpha_FL", 0.0, 10.0),   # front-left wing angle  [deg]
            ("alpha_FR", 0.0, 10.0),   # front-right wing angle [deg]
            ("alpha_RW", 0.0, 30.0),   # rear wing angle        [deg]
            ("alpha_TW", -12.0, 12.0), # rear wing tilt         [deg]
        ]
```

- [ ] **Step 2: Regression — a plain-single paramOptim solve still converges**

```
venv\Scripts\python.exe -c "from MLTP_paramOptim import MLTP_paramOptim; import warnings; warnings.simplefilter('ignore'); \
ctx=MLTP_paramOptim(circuit='Sturn', vi=60.0, ATD='Off', Electric_4Motors='Off', save=False); \
print('paramOptim lap', round(ctx.data.lap_time,3), 'params', ctx.data.optimal_params)"
```
Expected: a finite lap time and an `optimal_params` dict containing `brkB` and `Tdist` (plain single → `Tdist` promoted).

- [ ] **Step 3: Regression — paramOptim under four-motor drops `Tdist`**

```
venv\Scripts\python.exe -c "from MLTP_paramOptim import MLTP_paramOptim; import warnings; warnings.simplefilter('ignore'); \
ctx=MLTP_paramOptim(circuit='Sturn', vi=60.0, ATD='Off', Electric_4Motors='On', save=False); \
print('four-motor params', list(ctx.data.optimal_params))"
```
Expected: a finite solve; `Tdist` is **not** in the params list (`['brkB', 'alpha_FL', 'alpha_FR', 'alpha_RW', 'alpha_TW']`).

- [ ] **Step 4: Commit**

```bash
git add MLTP_paramOptim.py
git commit -m "feat(paramOptim): topology-aware default params (drop Tdist unless plain single)"
```

---

## Task 10: First Aurora hybrid solve (end-to-end regression)

Prove the whole stack (Tasks 2-9) runs in `Hybrid='On'` on a synthetic circuit, producing a finite lap time and the new channels.

**Files:**
- None modified (verification only). This task records the commands + expected output.

**Interfaces:**
- Consumes: the full stack from Tasks 2-9.

- [ ] **Step 1: Run a hybrid solve on `Sturn` and check the new outputs**

```
venv\Scripts\python.exe -c "from MLTP import MLTP; import warnings; warnings.simplefilter('ignore'); \
ctx=MLTP(circuit='Sturn', vi=60.0, Hybrid='On', save=False, plot=False); \
v=ctx.data.vehicle; \
assert ctx.data.lap_time > 0; \
assert {'P_motor_fl','P_motor_fr','P_motor_r','P_ice_r','E_motor','E_fuel'} <= set(v); \
assert ctx.m23.nx == 23; \
assert ctx.input_keys[-1] == 'delta' and 'split_R' in ctx.input_keys; \
print('HYBRID OK status', ctx.solve_stats['return_status'], 'lap', round(ctx.data.lap_time,3), \
      'E_fuel[-1]', round(float(v['E_fuel'][-1]),4))"
```
Expected: `HYBRID OK status <...> lap <finite> E_fuel[-1] <finite>` (status `Solve_Succeeded` or `Solved_To_Acceptable_Level`; a finite lap time is the pass criterion).

- [ ] **Step 2: Confirm a saved hybrid `.mat` uses the `_Hybrid` config name**

```
venv\Scripts\python.exe -c "from MLTP import MLTP; import warnings, os; warnings.simplefilter('ignore'); \
MLTP(circuit='Sturn', vi=60.0, Hybrid='On', save=True, plot=False, results_dir='Results'); \
print('saved', os.path.exists('Results/Sturn_Static_Hybrid.mat'))"
```
Expected: `saved True`.

- [ ] **Step 3: Re-run the full casadi-free suite to confirm no regressions**

```
venv\Scripts\python.exe test_foundation.py
venv\Scripts\python.exe test_params_useropts.py
venv\Scripts\python.exe test_drive_sources.py
venv\Scripts\python.exe test_transcription.py
venv\Scripts\python.exe test_save_load.py
venv\Scripts\python.exe test_sweep.py
```
Expected: all six print their `ALL ... TESTS PASSED` banners.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test(powertrain): end-to-end Aurora hybrid solve regression on synthetic circuit"
```

---

## Notes for the implementer

- **DP1 is a deliberate numeric change to EM4=1.** No test pins EM4=1 *numbers* (only channel names), so the regression in Task 7 checks convergence + finiteness, not specific values.
- **ICE map is a flagged placeholder** (`curve=(700.0,)` flat cap, redline/eff guessed) in `build_topology`'s `hybrid` branch. When the engine-map image arrives, replace `ice_curve` with real coeffs (high→low order), and set `ice_peak`/`ICE_OM_REDLINE`/`ICE_EFF`. Nothing else changes.
- **`transcription.py` is intentionally untouched** — adding controls only widens `nu`; the column-major packing already follows it.
- **If `_col` raises `AttributeError`**, a channel name was emitted without a `_build_c` row — they must change together (Task 4 already pairs them).
- **`gg_plots.py` and the chassis/suspension/tyre/aero dynamics are out of scope** — do not touch their channels.
- **Known data gaps (carried forward):** ICE torque map (placeholder), split tyre stiffness `kt_f≠kt_r` and split `Jw` (single-field model), and active-aero/AALB for Aurora (needs the `DATA_AA` polynomial map). Static aero only for hybrid.
