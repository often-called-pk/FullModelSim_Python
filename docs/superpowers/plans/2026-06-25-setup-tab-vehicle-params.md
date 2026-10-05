# Setup-tab Vehicle-Parameter Editor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose all ~110 adjustable vehicle parameters on the GUI Setup tab (collapsible sections, reset, save/load presets, changed-from-default highlight, tooltips/units) and five solver/collocation options on the Advanced tab.

**Architecture:** A pure-Python presentation registry (`app/vp_params.py`) drives Qt widgets built in `app/mainwindow.py`. `RunConfig` moves from 7 named tunables to a single `vp` dict (diff-from-default → `vp_overrides`) plus five top-level solver fields. The five solver knobs are plumbed by adding keyword args to `userOpts` (identical defaults) and forwarding them through `headless_solve`; `MLTP`/`MLTP_initial` already pass `**useropts_kwargs` through, so no solve-internal change.

**Tech Stack:** Python 3, PySide6 (Qt), CasADi (not touched here), plain-script tests (no pytest).

**Spec:** `docs/superpowers/specs/2026-06-25-setup-tab-vehicle-params-design.md`

## Global Constraints

- **Run Python via the in-repo venv from the repo root:** `venv\Scripts\python.exe <script>`. All paths are relative to the repo root.
- **No test runner.** Tests are plain scripts whose asserts run at module top level; run each file directly (`venv\Scripts\python.exe test_x.py`). The helper is `def ok(name, cond): print(...); assert cond, name`.
- **Qt tests run headless** with the offscreen platform: set `QT_QPA_PLATFORM=offscreen` before importing PySide6.
- **Defaults must not change behaviour.** An unmodified `userOpts(...)` call and a no-override `vehParams(...)` call must remain byte-for-byte identical to today. `RunConfig` solver-field defaults equal the `userOpts` hardcoded values (`max_iter=6000, OPT_ds=30, OPT_d=3, OPT_e=1e-2, tol=1e-4`).
- **Override validation stays:** `vp_overrides()` rejects any key not in `PRIMARY_KEYS ∪ MF_KEYS` with `ValueError`.
- **Full adjustable set = 110 keys** = `PRIMARY_KEYS ∪ MF_KEYS` = 51 primaries (`vehParams.default_primaries()`) + 59 Pacejka coefficients (`vehParams._default_mf()`).
- **Commits:** follow the repo's existing message conventions (trailers included on real commits).

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `app/vp_params.py` | Create | Pure registry: `PARAM_GROUPS`, `ParamMeta`, `META`, `meta_for`, `all_vp_defaults`, `fmt_sci`. No Qt. |
| `app/widgets.py` | Create | Reusable Qt widgets: `ScientificField`, `CollapsibleSection`. |
| `app/paths.py` | Modify | Add `user_presets_dir()`. |
| `app/runconfig.py` | Modify | Replace 7 named fields + `TIER1_FIELDS` with `vp` dict + 5 solver fields. |
| `userOpts.py` | Modify | Accept `OPT_ds/OPT_d/OPT_e/max_iter/tol` kwargs (identical defaults). |
| `headless_solve.py` | Modify | Forward the 5 solver cfg keys into MLTP kwargs. |
| `app/mainwindow.py` | Modify | Setup tab rebuild + Advanced-tab "Solver & Collocation" group + `collect_runconfig` wiring. |
| `test_vp_params.py` | Create | Registry invariants. |
| `test_paths.py` | Create | `user_presets_dir()` resolution. |
| `test_runconfig.py` | Rewrite | `vp` dict + solver-field cfg behaviour. |
| `test_useropts_solveropts.py` | Create | `userOpts` kwarg defaults + overrides. |
| `test_headless_config.py` | Modify | Forwarding of the 5 solver keys. |
| `test_mainwindow.py` | Create | Headless MainWindow: 110-param coverage, collect/override round-trip, reset. |

**Task order & dependencies:** 1 (vp_params) → 2 (widgets) → 3 (paths) → 4 (runconfig) → 5 (userOpts) → 6 (headless) → 7 (Setup tab) → 8 (Advanced tab). Tasks 5 and 6 are independent of the GUI. Between Task 4 and Task 7 `app/mainwindow.py` references removed names; no test imports it until Task 7, and the app is not shipped mid-plan.

---

### Task 1: `app/vp_params.py` — presentation registry

**Files:**
- Create: `app/vp_params.py`
- Test: `test_vp_params.py`

**Interfaces:**
- Consumes: `vehParams.default_primaries()`, `vehParams._default_mf()`, `vehParams.PRIMARY_KEYS`, `vehParams.MF_KEYS`.
- Produces:
  - `ParamMeta(label, unit="", tooltip="", lo=-1e6, hi=1e6, step=0.1, decimals=6, kind="spin")` (dataclass).
  - `PARAM_GROUPS: list[tuple[str, list[str]]]`.
  - `META: dict[str, ParamMeta]`.
  - `meta_for(key) -> ParamMeta`.
  - `all_vp_defaults() -> dict` (= `{**default_primaries(), **vars(_default_mf())}`).
  - `fmt_sci(x) -> str` (= `"%.12g" % float(x)`).

- [ ] **Step 1: Write the failing test** — create `test_vp_params.py`:

```python
"""app/vp_params.py registry invariants (pure Python, no Qt)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from vehParams import default_primaries, _default_mf, PRIMARY_KEYS, MF_KEYS
from app.vp_params import PARAM_GROUPS, meta_for, all_vp_defaults, fmt_sci

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

grouped = [k for _, keys in PARAM_GROUPS for k in keys]
ok("no duplicate keys across groups", len(grouped) == len(set(grouped)))
ok("groups cover exactly the override keys", set(grouped) == (PRIMARY_KEYS | MF_KEYS))

exp = {**default_primaries(), **vars(_default_mf())}
ok("all_vp_defaults == primaries+mf merge", all_vp_defaults() == exp)
ok("all_vp_defaults has 110 keys", len(all_vp_defaults()) == len(PRIMARY_KEYS | MF_KEYS))

defs = all_vp_defaults()
ok("lo <= default <= hi for all", all(meta_for(k).lo <= v <= meta_for(k).hi for k, v in defs.items()))
ok("decimals >= 0 for all", all(meta_for(k).decimals >= 0 for k in defs))
ok("kind sci iff mf or eps", all(
    (meta_for(k).kind == "sci") == ((k in MF_KEYS) or k.startswith("eps_")) for k in defs))
ok("spin defaults representable at their decimals", all(
    round(v, meta_for(k).decimals) == v for k, v in defs.items() if meta_for(k).kind == "spin"))
ok("fmt_sci round-trips every default exactly", all(float(fmt_sci(v)) == float(v) for v in defs.values()))

print("\nALL vp_params TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe test_vp_params.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.vp_params'`.

- [ ] **Step 3: Write minimal implementation** — create `app/vp_params.py`:

```python
"""app/vp_params.py — presentation registry for the Setup-tab vehicle-parameter
editor. Pure Python (no Qt). Parameter *values* come from vehParams; this module
only describes how each parameter is labelled, ranged, and edited.
"""
from dataclasses import dataclass

from vehParams import default_primaries, _default_mf, PRIMARY_KEYS, MF_KEYS


@dataclass
class ParamMeta:
    label: str
    unit: str = ""
    tooltip: str = ""
    lo: float = -1.0e6
    hi: float = 1.0e6
    step: float = 0.1
    decimals: int = 6
    kind: str = "spin"          # "spin" -> QDoubleSpinBox, "sci" -> ScientificField


def all_vp_defaults():
    """Full adjustable parameter set (primaries + Pacejka mf) as a flat dict."""
    return {**default_primaries(), **vars(_default_mf())}


def fmt_sci(x):
    """Format a float for a ScientificField; round-trips exactly via float()."""
    return "%.12g" % float(x)


_PACEJKA_LONG = [
    "pCx1", "pDx1", "pDx2", "pDx3", "pEx1", "pEx2", "pEx3", "pEx4",
    "pKx1", "pKx2", "pKx3", "pHx1", "pHx2", "pVx1", "pVx2",
    "rBx1", "rBx2", "rBx3", "rCx1", "rEx1", "rEx2", "rHx1",
]
_PACEJKA_LAT = [
    "pCy1", "pDy1", "pDy2", "pDy3", "pEy1", "pEy2", "pEy3", "pEy4", "pEy5",
    "pKy1", "pKy2", "pKy3", "pKy4", "pKy5", "pKy6", "pKy7", "pHy1", "pHy2",
    "pVy1", "pVy2", "pVy3", "pVy4", "rBy1", "rBy2", "rBy3", "rBy4",
    "rCy1", "rEy1", "rEy2", "rHy1", "rHy2",
    "rVy1", "rVy2", "rVy3", "rVy4", "rVy5", "rVy6",
]

PARAM_GROUPS = [
    ("Balance & Aero", ["brkB", "Tdist", "ksD",
                        "alpha_FL", "alpha_FR", "alpha_RW", "alpha_TW"]),
    ("Masses", ["mb", "md", "muf", "mur"]),
    ("Dimensions", ["A", "t", "l", "wB", "hcg", "huf", "hur", "hw",
                    "hRCf", "hRCr", "hride"]),
    ("Inertias", ["I_z", "I_y", "I_x"]),
    ("Tyre & Wheel", ["Rw", "Jw", "f", "kt", "Fz0", "Fz0_shift"]),
    ("Suspension stiffness", ["k_fl", "k_fr", "k_rl", "k_rr"]),
    ("Suspension damping", ["zeta_fl", "zeta_fr", "zeta_rl", "zeta_rr"]),
    ("Brakes", ["Tbrake_max"]),
    ("Camber & Toe", ["gamma_fl", "gamma_rl", "toe_front", "toe_rear"]),
    ("Aero & Environment", ["rho", "g", "Cd", "Cl"]),
    ("Numerical", ["eps_x", "eps_y", "eps_K"]),
    ("Pacejka 5.2 — longitudinal", _PACEJKA_LONG),
    ("Pacejka 5.2 — lateral", _PACEJKA_LAT),
]

META = {
    # Balance & Aero
    "brkB": ParamMeta("Brake bias (front)", "", "Front brake-torque fraction", 0.0, 1.0, 0.01, 4),
    "Tdist": ParamMeta("Torque dist (rear)", "", "Rear drive-torque fraction", 0.0, 1.0, 0.01, 4),
    "ksD": ParamMeta("Roll stiffness (rear)", "", "Rear roll-stiffness distribution", 0.0, 1.0, 0.01, 4),
    "alpha_FL": ParamMeta("Front wing L", "deg", "Front wing angle of attack, left", 0.0, 10.0, 0.5, 2),
    "alpha_FR": ParamMeta("Front wing R", "deg", "Front wing angle of attack, right", 0.0, 10.0, 0.5, 2),
    "alpha_RW": ParamMeta("Rear wing", "deg", "Rear wing angle of attack", 0.0, 30.0, 0.5, 2),
    "alpha_TW": ParamMeta("Rear wing tilt", "deg", "Rear wing tilt trim", -12.0, 12.0, 0.5, 2),
    # Masses
    "mb": ParamMeta("Body mass", "kg", "Sprung body mass", 0.0, 5000.0, 10.0, 1),
    "md": ParamMeta("Driver mass", "kg", "Driver mass", 0.0, 200.0, 5.0, 1),
    "muf": ParamMeta("Unsprung mass front", "kg", "Front unsprung mass", 0.0, 300.0, 5.0, 1),
    "mur": ParamMeta("Unsprung mass rear", "kg", "Rear unsprung mass", 0.0, 300.0, 5.0, 1),
    # Dimensions
    "A": ParamMeta("Frontal area", "m^2", "Reference frontal area", 0.5, 5.0, 0.05, 3),
    "t": ParamMeta("Track width", "m", "Axle track width", 0.5, 3.0, 0.05, 3),
    "l": ParamMeta("Wheelbase", "m", "Wheelbase", 1.0, 5.0, 0.05, 3),
    "wB": ParamMeta("Weight balance (rear)", "", "Rear weight fraction", 0.0, 1.0, 0.01, 4),
    "hcg": ParamMeta("CoG height", "m", "Centre-of-gravity height", 0.0, 1.5, 0.01, 4),
    "huf": ParamMeta("Unsprung CoG h front", "m", "Front unsprung CoG height", 0.0, 1.0, 0.001, 7),
    "hur": ParamMeta("Unsprung CoG h rear", "m", "Rear unsprung CoG height", 0.0, 1.0, 0.001, 7),
    "hw": ParamMeta("Wing height", "m", "Aero application height", 0.0, 2.0, 0.01, 4),
    "hRCf": ParamMeta("Roll centre h front", "m", "Front roll-centre height", -0.5, 1.0, 0.01, 4),
    "hRCr": ParamMeta("Roll centre h rear", "m", "Rear roll-centre height", -0.5, 1.0, 0.01, 4),
    "hride": ParamMeta("Ride height", "m", "Static ride height", 0.0, 0.5, 0.001, 4),
    # Inertias
    "I_z": ParamMeta("Yaw inertia", "kg*m^2", "Yaw moment of inertia", 0.0, 10000.0, 50.0, 1),
    "I_y": ParamMeta("Pitch inertia", "kg*m^2", "Pitch moment of inertia", 0.0, 10000.0, 50.0, 1),
    "I_x": ParamMeta("Roll inertia", "kg*m^2", "Roll moment of inertia", 0.0, 10000.0, 50.0, 1),
    # Tyre & Wheel
    "Rw": ParamMeta("Wheel radius", "m", "Loaded wheel radius", 0.1, 0.6, 0.005, 4),
    "Jw": ParamMeta("Wheel inertia", "kg*m^2", "Rotational inertia per wheel", 0.0, 20.0, 0.1, 3),
    "f": ParamMeta("Rolling resistance", "", "Rolling-resistance coefficient", 0.0, 0.1, 0.001, 4),
    "kt": ParamMeta("Tyre vert. stiffness", "N/m", "Tyre vertical stiffness", 0.0, 1.0e6, 1000.0, 1),
    "Fz0": ParamMeta("Nominal tyre load", "N", "Pacejka nominal vertical load", 0.0, 20000.0, 50.0, 1),
    "Fz0_shift": ParamMeta("Nominal load shift", "", "Scale on nominal load Fz0", 0.0, 5.0, 0.05, 4),
    # Suspension stiffness
    "k_fl": ParamMeta("Spring rate FL", "N/m", "Front-left spring rate", 0.0, 300000.0, 1000.0, 1),
    "k_fr": ParamMeta("Spring rate FR", "N/m", "Front-right spring rate", 0.0, 300000.0, 1000.0, 1),
    "k_rl": ParamMeta("Spring rate RL", "N/m", "Rear-left spring rate", 0.0, 300000.0, 1000.0, 1),
    "k_rr": ParamMeta("Spring rate RR", "N/m", "Rear-right spring rate", 0.0, 300000.0, 1000.0, 1),
    # Suspension damping
    "zeta_fl": ParamMeta("Damping ratio FL", "", "Front-left damping ratio", 0.0, 2.0, 0.05, 3),
    "zeta_fr": ParamMeta("Damping ratio FR", "", "Front-right damping ratio", 0.0, 2.0, 0.05, 3),
    "zeta_rl": ParamMeta("Damping ratio RL", "", "Rear-left damping ratio", 0.0, 2.0, 0.05, 3),
    "zeta_rr": ParamMeta("Damping ratio RR", "", "Rear-right damping ratio", 0.0, 2.0, 0.05, 3),
    # Brakes
    "Tbrake_max": ParamMeta("Max brake torque", "N*m", "Maximum total brake torque", 0.0, 20000.0, 100.0, 1),
    # Camber & Toe
    "gamma_fl": ParamMeta("Camber front (L)", "deg", "Front camber, left (mirrored to right)", -10.0, 10.0, 0.1, 3),
    "gamma_rl": ParamMeta("Camber rear (L)", "deg", "Rear camber, left (mirrored to right)", -10.0, 10.0, 0.1, 3),
    "toe_front": ParamMeta("Toe front", "deg", "Front toe angle", -10.0, 10.0, 0.1, 3),
    "toe_rear": ParamMeta("Toe rear", "deg", "Rear toe angle", -10.0, 10.0, 0.1, 3),
    # Aero & Environment
    "rho": ParamMeta("Air density", "kg/m^3", "Ambient air density", 0.5, 2.0, 0.001, 4),
    "g": ParamMeta("Gravity", "m/s^2", "Gravitational acceleration", 0.0, 20.0, 0.01, 4),
    "Cd": ParamMeta("Drag coefficient", "", "OVERWRITTEN by Data/DATA_AA.mat when present", 0.0, 5.0, 0.01, 4),
    "Cl": ParamMeta("Lift coefficient", "", "OVERWRITTEN by Data/DATA_AA.mat when present", 0.0, 5.0, 0.01, 4),
    # Numerical (scientific notation; tiny values)
    "eps_x": ParamMeta("eps_x", "", "Longitudinal-slip smoothing epsilon", 1.0e-12, 1.0, 0.0, 12, kind="sci"),
    "eps_y": ParamMeta("eps_y", "", "Lateral-slip smoothing epsilon", 1.0e-12, 1.0, 0.0, 12, kind="sci"),
    "eps_K": ParamMeta("eps_K", "", "Curvature smoothing epsilon", 1.0e-12, 1.0, 0.0, 12, kind="sci"),
}


def meta_for(key):
    """Annotated metadata, or a generic scientific-notation fallback (used by
    every Pacejka mf coefficient — wide range, no rounding)."""
    if key in META:
        return META[key]
    return ParamMeta(label=key, unit="", tooltip=key,
                     lo=-1.0e6, hi=1.0e6, step=0.0, decimals=12, kind="sci")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe test_vp_params.py`
Expected: PASS — final line `ALL vp_params TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add app/vp_params.py test_vp_params.py
git commit -m "feat(app): vehicle-parameter presentation registry (vp_params)"
```

---

### Task 2: `app/widgets.py` — `ScientificField` + `CollapsibleSection`

**Files:**
- Create: `app/widgets.py`
- Test: covered indirectly by `fmt_sci` (Task 1) and the offscreen MainWindow test (Task 7). No standalone test file — these are thin Qt wrappers; their pure logic lives in `fmt_sci`.

**Interfaces:**
- Consumes: `app.vp_params.fmt_sci`.
- Produces:
  - `ScientificField(QWidget)` with `value() -> float`, `setValue(float)`, `set_changed(bool)`, `setToolTip(str)`, and signal `valueChanged(float)`.
  - `CollapsibleSection(QWidget)` with `addRow(label, field)`, `set_expanded(bool)`, `set_changed_count(int)`, and signal `reset_requested()`.

- [ ] **Step 1: Write the implementation** — create `app/widgets.py`:

```python
"""app/widgets.py — small reusable Qt widgets for the parameter editor."""
from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QWidget, QToolButton, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
)

from app.vp_params import fmt_sci


class ScientificField(QWidget):
    """A line-edit numeric field that preserves full float precision (including
    tiny/exponent values such as -9.1492e-11), exposing the QDoubleSpinBox-like
    API the Setup tab relies on: value()/setValue()/valueChanged."""
    valueChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._edit = QLineEdit(self)
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.ScientificNotation)
        self._edit.setValidator(validator)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._edit)
        self._edit.editingFinished.connect(self._emit)

    def _emit(self):
        try:
            self.valueChanged.emit(self.value())
        except ValueError:
            pass

    def value(self):
        return float(self._edit.text())

    def setValue(self, x):
        self._edit.setText(fmt_sci(x))

    def set_changed(self, changed):
        self._edit.setStyleSheet("font-weight: bold; color: #b30000;" if changed else "")

    def setToolTip(self, text):
        super().setToolTip(text)
        self._edit.setToolTip(text)


class CollapsibleSection(QWidget):
    """A titled section whose body collapses. Header shows an arrow, the title,
    and a '(N changed)' badge; an inline 'Reset' button emits reset_requested."""
    reset_requested = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title = title
        self._changed = 0

        self._btn = QToolButton(self)
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.setStyleSheet("QToolButton { border: none; font-weight: bold; }")

        self._reset = QToolButton(self)
        self._reset.setText("Reset")
        self._reset.setStyleSheet("QToolButton { border: none; color: #555; }")
        self._reset.clicked.connect(self.reset_requested)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._btn)
        header.addStretch(1)
        header.addWidget(self._reset)

        self._body = QWidget(self)
        self._form = QFormLayout(self._body)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(header)
        lay.addWidget(self._body)

        self._btn.toggled.connect(self._on_toggled)
        self._refresh()

    def _on_toggled(self, on):
        self._body.setVisible(on)
        self._refresh()

    def _refresh(self):
        arrow = "▾" if self._btn.isChecked() else "▸"
        badge = f"   ({self._changed} changed)" if self._changed else ""
        self._btn.setText(f"{arrow}  {self._title}{badge}")

    def addRow(self, label, field):
        self._form.addRow(label, field)

    def set_expanded(self, on):
        self._btn.setChecked(on)

    def set_changed_count(self, n):
        self._changed = n
        self._refresh()
```

- [ ] **Step 2: Verify it imports** (offscreen, since PySide6 needs a platform plugin)

Run: `$env:QT_QPA_PLATFORM='offscreen'; venv\Scripts\python.exe -c "from app.widgets import ScientificField, CollapsibleSection; from PySide6.QtWidgets import QApplication; a=QApplication([]); f=ScientificField(); f.setValue(-9.1492e-11); print(f.value()); assert f.value()==-9.1492e-11; print('OK')"`
Expected: prints `-9.1492e-11` then `OK` (proves the tiny value round-trips, the core reason this widget exists).

- [ ] **Step 3: Commit**

```bash
git add app/widgets.py
git commit -m "feat(app): ScientificField + CollapsibleSection widgets"
```

---

### Task 3: `app/paths.py` — `user_presets_dir()`

**Files:**
- Modify: `app/paths.py` (add one function after `default_output_dir`, around line 26)
- Test: `test_paths.py`

**Interfaces:**
- Consumes: existing `default_output_dir()`.
- Produces: `user_presets_dir() -> str` = `os.path.join(default_output_dir(), "presets")`.

- [ ] **Step 1: Write the failing test** — create `test_paths.py`:

```python
"""app/paths.user_presets_dir resolution."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from app.paths import user_presets_dir, default_output_dir

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

ok("presets under output root",
   user_presets_dir() == os.path.join(default_output_dir(), "presets"))
ok("presets basename is 'presets'", os.path.basename(user_presets_dir()) == "presets")
print("\nALL paths TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe test_paths.py`
Expected: FAIL — `ImportError: cannot import name 'user_presets_dir'`.

- [ ] **Step 3: Implement** — append to `app/paths.py`:

```python
def user_presets_dir():
    """Writable directory for user-saved setup presets (created on demand by
    the caller). The bundled app/presets/ is read-only when frozen."""
    return os.path.join(default_output_dir(), "presets")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe test_paths.py`
Expected: PASS — `ALL paths TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add app/paths.py test_paths.py
git commit -m "feat(app): user_presets_dir() for writable setup presets"
```

---

### Task 4: `app/runconfig.py` — `vp` dict + solver fields

**Files:**
- Modify: `app/runconfig.py` (full rewrite of the dataclass; remove `TIER1_FIELDS`)
- Test: `test_runconfig.py` (rewrite)

**Interfaces:**
- Consumes: `app.vp_params.all_vp_defaults`, `vehParams.PRIMARY_KEYS`, `vehParams.MF_KEYS`.
- Produces: `RunConfig` with fields `circuit, AeroConfig, ATD, Electric_4Motors, TyreModel, vi, ni, linear_solver, warm_start, save, plot, output_dir, expert_config, max_iter:int=6000, OPT_ds:float=30.0, OPT_d:int=3, OPT_e:float=1e-2, tol:float=1e-4, vp:dict`. Methods: `vp_overrides()`, `to_dict()`, `from_dict()`, `from_json()`, `write_cfg()`. **`TIER1_FIELDS` no longer exists.**

- [ ] **Step 1: Rewrite the test** — replace the entire contents of `test_runconfig.py`:

```python
"""RunConfig: vp-dict round-trip, diff-from-default -> vp_overrides, expert-file
merge, solver fields at top level (not in vp_overrides), unknown-key rejection."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from app.runconfig import RunConfig
from app.vp_params import all_vp_defaults

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

print("\nALL RunConfig TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe test_runconfig.py`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'vp'` (old RunConfig has no `vp`).

- [ ] **Step 3: Implement** — replace the entire contents of `app/runconfig.py`:

```python
"""RunConfig: the GUI's editable state, serialisable to the solve cfg.json.

`to_dict`/`from_dict` persist the full GUI state. `write_cfg` emits the JSON the
headless solve consumes: run-config fields, the five top-level solver/collocation
options, and a single merged `vp_overrides` dict (vp diff-from-default + any
expert config file).
"""
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

from app.paths import default_output_dir
from app.vp_params import all_vp_defaults
from vehParams import PRIMARY_KEYS, MF_KEYS


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
    expert_config: Optional[str] = None
    # solver / collocation options (top-level cfg fields, NOT vp_overrides)
    max_iter: int = 6000
    OPT_ds: float = 30.0
    OPT_d: int = 3
    OPT_e: float = 1e-2
    tol: float = 1e-4
    # full vehicle-parameter set (primaries + Pacejka mf), seeded from defaults
    vp: dict = field(default_factory=all_vp_defaults)

    def vp_overrides(self):
        ov = {}
        if self.expert_config:
            with open(self.expert_config) as fh:
                ov.update(json.load(fh))            # expert is the base layer
        defaults = all_vp_defaults()
        for k, v in self.vp.items():                 # GUI diff overrides expert
            if v != defaults[k]:
                ov[k] = v
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
        d.pop("vp", None)
        d["vp_overrides"] = self.vp_overrides()
        return d

    def write_cfg(self, path):
        cfg = self._cfg()
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe test_runconfig.py`
Expected: PASS — `ALL RunConfig TESTS PASSED`.

> **Note:** `app/mainwindow.py` still imports `TIER1_FIELDS` and references `self.brkB` etc.; it is repaired in Task 7. No test imports `app.mainwindow` until then.

- [ ] **Step 5: Run the presets test (must still pass — `default.json` untouched)**

Run: `venv\Scripts\python.exe test_presets.py`
Expected: PASS — `ALL preset TESTS PASSED`.

- [ ] **Step 6: Commit**

```bash
git add app/runconfig.py test_runconfig.py
git commit -m "refactor(app): RunConfig vp-dict + top-level solver fields"
```

---

### Task 5: `userOpts.py` — accept solver/collocation kwargs

**Files:**
- Modify: `userOpts.py` (signature at line 181; collocation block at lines 242–245; ipopt dict at lines 253–267)
- Test: `test_useropts_solveropts.py`

**Interfaces:**
- Produces: `userOpts(...)` additionally accepts `OPT_ds=30, OPT_d=3, OPT_e=1e-2, max_iter=6000, tol=1e-4`, writing `ctx.OPT_ds/OPT_d/OPT_e` and `ctx.opts["ipopt"]["max_iter"]/["tol"]`. All defaults equal the previous hardcoded values.

- [ ] **Step 1: Write the failing test** — create `test_useropts_solveropts.py`:

```python
"""userOpts exposes solver/collocation kwargs; defaults unchanged."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
from functions.context import Ctx
from userOpts import userOpts

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ctx = userOpts(Ctx(), circuit="Sturn")
ok("default OPT_ds == 30", ctx.OPT_ds == 30)
ok("default OPT_d == 3", ctx.OPT_d == 3)
ok("default OPT_e == 1e-2", ctx.OPT_e == 1e-2)
ok("default max_iter == 6000", ctx.opts["ipopt"]["max_iter"] == 6000)
ok("default tol == 1e-4", ctx.opts["ipopt"]["tol"] == 1e-4)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ctx2 = userOpts(Ctx(), circuit="Sturn",
                    OPT_ds=20, OPT_d=4, OPT_e=5e-3, max_iter=3000, tol=1e-6)
ok("override OPT_ds", ctx2.OPT_ds == 20)
ok("override OPT_d", ctx2.OPT_d == 4)
ok("override OPT_e", ctx2.OPT_e == 5e-3)
ok("override max_iter", ctx2.opts["ipopt"]["max_iter"] == 3000)
ok("override tol", ctx2.opts["ipopt"]["tol"] == 1e-6)

print("\nALL userOpts solver-opt TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe test_useropts_solveropts.py`
Expected: FAIL — `TypeError: userOpts() got an unexpected keyword argument 'OPT_ds'`.

- [ ] **Step 3a: Implement — extend the signature.** In `userOpts.py`, replace the signature block (lines 181–192):

```python
def userOpts(ctx,
             AeroConfig="Static",          # 'Static' | 'Active_RW' | 'Active' | 'AALB'
             ATD="On",                     # 'On' | 'Off'
             Electric_4Motors="Off",       # 'On' | 'Off'
             circuit="BCN",
             vi=60.0,                      # initial velocity [m/s]
             ni=np.nan,                    # initial lateral position [m]
             circuits_dir="Circuits",
             data_dir="Data",
             linear_solver="ma57",         # 'ma57'|'ma97'|'ma27'|'mumps'; ma* uses Coin-HSL
             hsl_dir=None,                 # Coin-HSL bin dir; None -> COINHSL_DIR env / default
             vp_overrides=None,            # dict of vehParams primary/mf overrides
             OPT_ds=30,                    # collocation step (m)
             OPT_d=3,                      # degree of interpolating polynomials
             OPT_e=1e-2,                   # slack for path constraints / guesses
             max_iter=6000,                # IPOPT max iterations
             tol=1e-4):                    # IPOPT convergence tolerance
```

- [ ] **Step 3b: Implement — use the collocation kwargs.** Replace the collocation block (lines 242–245):

```python
    # ---- collocation options ----------------------------------------------
    ctx.OPT_ds = OPT_ds         # collocation step (m)
    ctx.OPT_d = OPT_d           # degree of interpolating polynomials
    ctx.OPT_uinter = "linear"   # 'linear' or 'constant' inputs (not exposed)
    ctx.OPT_e = OPT_e           # slack for path constraints / initial guesses
```

- [ ] **Step 3c: Implement — use the IPOPT kwargs.** In the `ipopt = {...}` dict (lines 253–267), change the `max_iter` and `tol` entries:

```python
        "max_iter": max_iter,
```
```python
        "tol": tol,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe test_useropts_solveropts.py`
Expected: PASS — `ALL userOpts solver-opt TESTS PASSED`.

- [ ] **Step 5: Run the existing config test (defaults must be unchanged)**

Run: `venv\Scripts\python.exe test_params_useropts.py`
Expected: PASS (no regression in the existing userOpts/vehParams behaviour).

- [ ] **Step 6: Commit**

```bash
git add userOpts.py test_useropts_solveropts.py
git commit -m "feat: expose OPT_ds/OPT_d/OPT_e/max_iter/tol as userOpts kwargs"
```

---

### Task 6: `headless_solve.py` — forward the five solver keys

**Files:**
- Modify: `headless_solve.py` (`build_solve_kwargs`, before the `if cfg.get("warm_start")` block, ~line 33)
- Test: `test_headless_config.py` (extend)

**Interfaces:**
- Consumes: cfg keys `max_iter, OPT_ds, OPT_d, OPT_e, tol` (optional; defaults applied).
- Produces: those five added to the MLTP kwargs dict with correct types; they land in MLTP's `**useropts_kwargs`.

- [ ] **Step 1: Extend the test** — append before the final `print` of `test_headless_config.py` (after line 29):

```python
# solver/collocation forwarding — defaults when absent
ok("max_iter default", kw["max_iter"] == 6000 and isinstance(kw["max_iter"], int))
ok("OPT_ds default", kw["OPT_ds"] == 30.0 and isinstance(kw["OPT_ds"], float))
ok("OPT_d default", kw["OPT_d"] == 3 and isinstance(kw["OPT_d"], int))
ok("OPT_e default", kw["OPT_e"] == 1e-2)
ok("tol default", kw["tol"] == 1e-4)

cfg3 = dict(cfg, max_iter=3000, OPT_ds=20, OPT_d=4, OPT_e=5e-3, tol=1e-6)
kw3 = build_solve_kwargs(cfg3, "/res")
ok("max_iter forwarded", kw3["max_iter"] == 3000)
ok("OPT_ds forwarded", kw3["OPT_ds"] == 20.0)
ok("OPT_d forwarded", kw3["OPT_d"] == 4)
ok("OPT_e forwarded", kw3["OPT_e"] == 5e-3)
ok("tol forwarded", kw3["tol"] == 1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe test_headless_config.py`
Expected: FAIL — `KeyError: 'max_iter'` (kwargs lacks the new keys).

- [ ] **Step 3: Implement** — in `headless_solve.py`, inside `build_solve_kwargs`, immediately after the `kwargs = dict(...)` assignment and before `if cfg.get("warm_start"):`, insert:

```python
    # solver / collocation options -> userOpts (via MLTP **useropts_kwargs)
    kwargs["max_iter"] = int(cfg.get("max_iter", 6000))
    kwargs["OPT_ds"] = float(cfg.get("OPT_ds", 30))
    kwargs["OPT_d"] = int(cfg.get("OPT_d", 3))
    kwargs["OPT_e"] = float(cfg.get("OPT_e", 1e-2))
    kwargs["tol"] = float(cfg.get("tol", 1e-4))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe test_headless_config.py`
Expected: PASS — `ALL headless config TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add headless_solve.py test_headless_config.py
git commit -m "feat: forward solver/collocation options through headless_solve"
```

---

### Task 7: `app/mainwindow.py` — Setup tab rebuild

**Files:**
- Modify: `app/mainwindow.py` (imports; `_build_setup_tab`; new param helpers; `collect_runconfig` vp wiring; rename `_load_expert_tier1` → `_load_expert_into_widgets`)
- Test: `test_mainwindow.py` (create)

**Interfaces:**
- Consumes: `app.vp_params.{PARAM_GROUPS, meta_for, all_vp_defaults}`, `app.widgets.{CollapsibleSection, ScientificField}`, `app.paths.user_presets_dir`, `RunConfig` (Task 4).
- Produces: `self.vp_spins: dict[str, QDoubleSpinBox|ScientificField]` (110 entries); methods `_field_for`, `_on_param_changed`, `_refresh_all_changed`, `_set_field_changed_style`, `_update_section_badge`, `_reset_all_params`, `_reset_section`, `_save_preset`, `_load_preset`, `_apply_param_dict`, `_load_expert_into_widgets`. `collect_runconfig()` returns a `RunConfig` whose `vp` holds all 110 widget values (solver fields still default until Task 8).

- [ ] **Step 1: Write the failing test** — create `test_mainwindow.py`:

```python
"""MainWindow constructs headlessly (offscreen Qt); Setup tab exposes all 110
vehicle params; collect_runconfig + reset behave correctly."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))
from PySide6.QtWidgets import QApplication
from app.mainwindow import MainWindow
from app.vp_params import all_vp_defaults
from vehParams import PRIMARY_KEYS, MF_KEYS

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

app = QApplication.instance() or QApplication([])
win = MainWindow()

ok("vp_spins covers all 110 keys", set(win.vp_spins) == (PRIMARY_KEYS | MF_KEYS))

rc = win.collect_runconfig()
ok("collect vp has 110 keys", set(rc.vp) == (PRIMARY_KEYS | MF_KEYS))
ok("collect vp equals defaults (no rounding loss)",
   all(rc.vp[k] == float(all_vp_defaults()[k]) for k in rc.vp))
ok("no spurious overrides at startup", rc.vp_overrides() == {})

# change one param -> exactly that key appears in overrides
win.vp_spins["alpha_RW"].setValue(12.0)
rc2 = win.collect_runconfig()
ov = rc2.vp_overrides()
ok("changed param in overrides", ov.get("alpha_RW") == 12.0)
ok("only the changed param overridden", set(ov) == {"alpha_RW"})

# a tiny Pacejka coeff round-trips through the widget
win.vp_spins["rHy1"].setValue(-1.2345e-10)
ok("sci widget round-trips tiny value", win.vp_spins["rHy1"].value() == -1.2345e-10)

# reset restores defaults -> overrides empty again
win._reset_all_params()
rc3 = win.collect_runconfig()
ok("reset clears overrides", rc3.vp_overrides() == {})

print("\nALL MainWindow TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:QT_QPA_PLATFORM='offscreen'; venv\Scripts\python.exe test_mainwindow.py`
Expected: FAIL — currently an `ImportError` (mainwindow imports `TIER1_FIELDS`, removed in Task 4).

- [ ] **Step 3a: Fix imports.** In `app/mainwindow.py`, replace the import section (lines 5–18) with:

```python
import os
import json
import tempfile
import webbrowser

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QFormLayout, QVBoxLayout, QHBoxLayout,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QPlainTextEdit,
    QLabel, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QScrollArea,
    QGroupBox,
)

from app.runconfig import RunConfig
from app.solve_runner import SolveRunner
from app import results, paths
from app.vp_params import PARAM_GROUPS, meta_for, all_vp_defaults
from app.widgets import CollapsibleSection, ScientificField
from app.paths import user_presets_dir
```

- [ ] **Step 3b: Replace `_build_setup_tab`.** Replace the whole `_build_setup_tab` method (lines 99–116) with the following (keep the module-level `_spin` helper — the Main tab still uses it):

```python
    def _build_setup_tab(self):
        self.vp_spins = {}            # key -> QDoubleSpinBox | ScientificField
        self._sections = {}           # title -> CollapsibleSection
        self._section_keys = {}       # title -> [keys]
        defaults = all_vp_defaults()

        container = QWidget()
        vlay = QVBoxLayout(container)

        bar = QHBoxLayout()
        reset_all = QPushButton("Reset all to defaults")
        reset_all.clicked.connect(self._reset_all_params)
        save_btn = QPushButton("Save preset…"); save_btn.clicked.connect(self._save_preset)
        load_btn = QPushButton("Load preset…"); load_btn.clicked.connect(self._load_preset)
        bar.addWidget(reset_all); bar.addWidget(save_btn); bar.addWidget(load_btn)
        bar.addStretch(1)
        vlay.addLayout(bar)

        for title, keys in PARAM_GROUPS:
            sec = CollapsibleSection(title)
            self._sections[title] = sec
            self._section_keys[title] = list(keys)
            for key in keys:
                m = meta_for(key)
                field = self._field_for(key)
                field.setValue(float(defaults[key]))
                field.valueChanged.connect(lambda _v=None, k=key: self._on_param_changed(k))
                self.vp_spins[key] = field
                lbl = QLabel(m.label); lbl.setToolTip(m.tooltip or m.label)
                sec.addRow(lbl, field)
            sec.set_expanded(title == "Balance & Aero")
            sec.reset_requested.connect(lambda t=title: self._reset_section(t))
            vlay.addWidget(sec)

        vlay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        return scroll

    def _field_for(self, key):
        m = meta_for(key)
        if m.kind == "sci":
            field = ScientificField()
        else:
            field = QDoubleSpinBox()
            field.setRange(m.lo, m.hi)
            field.setSingleStep(m.step)
            field.setDecimals(m.decimals)
            if m.unit:
                field.setSuffix(f" {m.unit}")
        field.setToolTip(m.tooltip or m.label)
        return field

    # ---- changed-from-default highlighting --------------------------------
    def _on_param_changed(self, key):
        defaults = all_vp_defaults()
        self._set_field_changed_style(self.vp_spins[key], self._is_changed(key, defaults))
        for title, keys in self._section_keys.items():
            if key in keys:
                self._update_section_badge(title, defaults)
                break

    def _is_changed(self, key, defaults):
        try:
            return float(self.vp_spins[key].value()) != float(defaults[key])
        except ValueError:
            return True

    def _set_field_changed_style(self, field, changed):
        if isinstance(field, ScientificField):
            field.set_changed(changed)
        else:
            field.setStyleSheet("font-weight: bold; color: #b30000;" if changed else "")

    def _update_section_badge(self, title, defaults=None):
        if defaults is None:
            defaults = all_vp_defaults()
        n = sum(1 for k in self._section_keys[title] if self._is_changed(k, defaults))
        self._sections[title].set_changed_count(n)

    def _refresh_all_changed(self):
        defaults = all_vp_defaults()
        for k in self.vp_spins:
            self._set_field_changed_style(self.vp_spins[k], self._is_changed(k, defaults))
        for title in self._section_keys:
            self._update_section_badge(title, defaults)

    # ---- reset / preset ---------------------------------------------------
    def _reset_all_params(self):
        defaults = all_vp_defaults()
        for k, field in self.vp_spins.items():
            field.setValue(float(defaults[k]))
        self._refresh_all_changed()

    def _reset_section(self, title):
        defaults = all_vp_defaults()
        for k in self._section_keys[title]:
            self.vp_spins[k].setValue(float(defaults[k]))
        self._refresh_all_changed()

    def _save_preset(self):
        os.makedirs(user_presets_dir(), exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save preset", os.path.join(user_presets_dir(), "setup.json"),
            "JSON (*.json)")
        if not path:
            return
        data = {k: float(self.vp_spins[k].value()) for k in self.vp_spins}
        try:
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
            self._append_log(f"[preset saved] {path}\n")
        except Exception as exc:
            self._append_log(f"[preset save error] {exc}\n")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load preset", user_presets_dir(), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:
            self._append_log(f"[preset load error] {exc}\n")
            return
        self._apply_param_dict(data)

    def _apply_param_dict(self, data):
        applied, ignored = 0, 0
        for k, v in data.items():
            if k in self.vp_spins:
                try:
                    self.vp_spins[k].setValue(float(v))
                    applied += 1
                except (TypeError, ValueError):
                    ignored += 1
            else:
                ignored += 1
        self._refresh_all_changed()
        msg = f"[preset] applied {applied} values"
        if ignored:
            msg += f", ignored {ignored} unknown/invalid keys"
        self._append_log(msg + "\n")
```

- [ ] **Step 3c: Wire `collect_runconfig` to the vp dict.** Replace the whole `collect_runconfig` method (lines 169–187) with:

```python
    def collect_runconfig(self):
        defaults = all_vp_defaults()
        vp = {}
        for k, field in self.vp_spins.items():
            try:
                vp[k] = float(field.value())
            except (TypeError, ValueError):
                vp[k] = float(defaults[k])
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
            expert_config=self.expert.text() or None,
            vp=vp,
        )
```

- [ ] **Step 3d: Rename the expert loader.** Replace `_load_expert_tier1` (lines 254–264) with:

```python
    def _load_expert_into_widgets(self, path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:
            self._append_log(f"[expert config error] {exc}\n")
            return
        self._apply_param_dict(data)
```

And in `_pick_expert` (line ~252) change the call `self._load_expert_tier1(f)` to `self._load_expert_into_widgets(f)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:QT_QPA_PLATFORM='offscreen'; venv\Scripts\python.exe test_mainwindow.py`
Expected: PASS — `ALL MainWindow TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add app/mainwindow.py test_mainwindow.py
git commit -m "feat(app): full vehicle-parameter editor on the Setup tab"
```

---

### Task 8: `app/mainwindow.py` — Advanced tab "Solver & Collocation" group

**Files:**
- Modify: `app/mainwindow.py` (`_build_advanced_tab`; `collect_runconfig` solver wiring)
- Test: `test_mainwindow.py` (extend)

**Interfaces:**
- Produces: `self.max_iter (QSpinBox)`, `self.opt_ds (QDoubleSpinBox)`, `self.opt_d (QSpinBox)`, `self.opt_e (ScientificField)`, `self.tol (ScientificField)`; `collect_runconfig` now passes `max_iter, OPT_ds, OPT_d, OPT_e, tol` into `RunConfig`.

- [ ] **Step 1: Extend the test** — append before the final `print` of `test_mainwindow.py`:

```python
# --- Advanced-tab solver/collocation widgets wired into RunConfig ---
win._reset_all_params()
rc4 = win.collect_runconfig()
ok("max_iter default wired", rc4.max_iter == 6000)
ok("OPT_ds default wired", rc4.OPT_ds == 30.0)
ok("OPT_d default wired", rc4.OPT_d == 3)
ok("OPT_e default wired", rc4.OPT_e == 1e-2)
ok("tol default wired", rc4.tol == 1e-4)

win.max_iter.setValue(2500)
win.opt_ds.setValue(25.0)
win.opt_d.setValue(4)
win.opt_e.setValue(2e-3)
win.tol.setValue(5e-5)
rc5 = win.collect_runconfig()
ok("max_iter override wired", rc5.max_iter == 2500)
ok("OPT_ds override wired", rc5.OPT_ds == 25.0)
ok("OPT_d override wired", rc5.OPT_d == 4)
ok("OPT_e override wired", rc5.OPT_e == 2e-3)
ok("tol override wired", rc5.tol == 5e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:QT_QPA_PLATFORM='offscreen'; venv\Scripts\python.exe test_mainwindow.py`
Expected: FAIL — `AttributeError: 'MainWindow' object has no attribute 'max_iter'`.

- [ ] **Step 3a: Replace `_build_advanced_tab`.** Replace the whole `_build_advanced_tab` method (lines 140–154) with:

```python
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

        grp = QGroupBox("Solver & Collocation")
        gform = QFormLayout(grp)
        self.max_iter = QSpinBox(); self.max_iter.setRange(1, 100000); self.max_iter.setValue(6000)
        self.opt_ds = QDoubleSpinBox(); self.opt_ds.setRange(1.0, 500.0)
        self.opt_ds.setSingleStep(1.0); self.opt_ds.setDecimals(2)
        self.opt_ds.setValue(30.0); self.opt_ds.setSuffix(" m")
        self.opt_d = QSpinBox(); self.opt_d.setRange(1, 6); self.opt_d.setValue(3)
        self.opt_e = ScientificField(); self.opt_e.setValue(1e-2)
        self.tol = ScientificField(); self.tol.setValue(1e-4)
        gform.addRow("Max iterations", self.max_iter)
        gform.addRow("Collocation step", self.opt_ds)
        gform.addRow("Polynomial degree", self.opt_d)
        gform.addRow("Path-constraint slack", self.opt_e)
        gform.addRow("IPOPT tolerance", self.tol)
        form.addRow(grp)
        return w
```

- [ ] **Step 3b: Add solver fields to `collect_runconfig`.** In the `return RunConfig(...)` call (edited in Task 7 Step 3c), add these keyword arguments just before `vp=vp,`:

```python
            max_iter=self.max_iter.value(),
            OPT_ds=self.opt_ds.value(),
            OPT_d=self.opt_d.value(),
            OPT_e=float(self.opt_e.value()),
            tol=float(self.tol.value()),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:QT_QPA_PLATFORM='offscreen'; venv\Scripts\python.exe test_mainwindow.py`
Expected: PASS — `ALL MainWindow TESTS PASSED`.

- [ ] **Step 5: Run the full casadi-free test suite (no regressions)**

Run each in turn:
```
venv\Scripts\python.exe test_vp_params.py
venv\Scripts\python.exe test_paths.py
venv\Scripts\python.exe test_runconfig.py
venv\Scripts\python.exe test_presets.py
venv\Scripts\python.exe test_useropts_solveropts.py
venv\Scripts\python.exe test_headless_config.py
$env:QT_QPA_PLATFORM='offscreen'; venv\Scripts\python.exe test_mainwindow.py
venv\Scripts\python.exe test_params_useropts.py
venv\Scripts\python.exe test_vp_overrides.py
```
Expected: every file prints its `ALL … TESTS PASSED` line.

- [ ] **Step 6: Smoke-test the symbolic model still builds (defaults unchanged)**

Run: `venv\Scripts\python.exe -c "from functions.context import Ctx; from Powertrain import Powertrain; from vehParams import vehParams; from userOpts import userOpts; from vehModel import vehModel; ctx=Ctx(); Powertrain(ctx); vehParams(ctx); userOpts(ctx); vehModel(ctx); print(ctx.m23.nx, ctx.m23.nu)"`
Expected: prints `23` and the nu value with no exception.

- [ ] **Step 7: Commit**

```bash
git add app/mainwindow.py test_mainwindow.py
git commit -m "feat(app): Solver & Collocation options on the Advanced tab"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** Setup tab full editor (Tasks 1,2,7); reset + save/load + highlight + tooltips/units (Tasks 1,2,7); `vp` dict model + diff-from-default + expert precedence (Task 4); `ScientificField` for tiny Pacejka/eps values (Tasks 1,2,7); `user_presets_dir` writable presets (Task 3); five solver options on Advanced tab + `userOpts`/`headless_solve` plumbing + top-level cfg fields (Tasks 5,6,8); all tests from the spec's Testing section (Tasks 1,3,4,5,6,7,8). No gaps.
- **Precision guard:** `huf`/`hur` (0.2968771) given `decimals=7`; `test_vp_params.py` asserts every spin default is exactly representable, preventing a spurious startup "changed"/override.
- **`_spin` retained:** the Main tab uses it for `vi`/`ni`; only `_build_setup_tab`'s body is rebuilt.
- **Type consistency:** `vp_spins` values expose `value()/setValue()/valueChanged` for both widget kinds; `meta_for` always returns a `ParamMeta`; `collect_runconfig` returns a `RunConfig` whose `vp_overrides()` is validated against `PRIMARY_KEYS ∪ MF_KEYS`.
