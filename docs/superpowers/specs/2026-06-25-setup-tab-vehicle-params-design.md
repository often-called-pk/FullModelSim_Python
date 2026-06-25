# Setup-tab full vehicle-parameter editor — design

**Date:** 2026-06-25
**Status:** Approved (design); implementation pending plan.
**Component:** `app/` Windows GUI (PySide6).

## Goal

Expand the GUI **Setup** tab from the current 7 hardcoded tunables to **every
adjustable vehicle parameter** `vehParams` accepts as an override
(`PRIMARY_KEYS ∪ MF_KEYS` — ~49 primaries + ~60 Pacejka 5.2 coefficients, ~110
total). Organize them into collapsible sections with reset, save/load presets,
changed-from-default highlighting, and tooltips/units.

This is a **front-end + config-storage change only**. The solve pipeline
downstream of `cfg.json` (`headless_solve → MLTP → vehParams`) is unchanged.

## Background — the existing pipeline

The plumbing to override any vehicle parameter already exists end to end:

```
Setup widgets → MainWindow.collect_runconfig() → RunConfig
            → RunConfig.vp_overrides() → RunConfig.write_cfg() → cfg.json
            → headless_solve.build_solve_kwargs() → MLTP(vp_overrides=…)
            → vehParams(ctx, vp_overrides=…)
```

`vehParams` validates every override key against `PRIMARY_KEYS ∪ MF_KEYS` and
recomputes all derived quantities from the merged primaries. Today only 7 keys
(`TIER1_FIELDS = brkB, Tdist, ksD, alpha_FL, alpha_FR, alpha_RW, alpha_TW`) are
exposed as GUI widgets; the other ~103 are reachable only via the Advanced
tab's "Expert config .json" file picker. So the entire task is **GUI widgets +
`RunConfig` storage** — no new solve capability is required.

## Full adjustable parameter set (grouping for the UI)

All keys below are accepted overrides. Defaults are **not** re-declared in the
GUI — they come from `vehParams.default_primaries()` and
`vehParams._default_mf()`.

| Section | Keys |
|---|---|
| Balance & Aero (existing 7) | `brkB, Tdist, ksD, alpha_FL, alpha_FR, alpha_RW, alpha_TW` |
| Masses | `mb, md, muf, mur` |
| Dimensions | `A, t, l, wB, hcg, huf, hur, hw, hRCf, hRCr, hride` |
| Inertias | `I_z, I_y, I_x` |
| Tyre & Wheel | `Rw, Jw, f, kt, Fz0, Fz0_shift` |
| Suspension stiffness | `k_fl, k_fr, k_rl, k_rr` |
| Suspension damping | `zeta_fl, zeta_fr, zeta_rl, zeta_rr` |
| Brakes | `Tbrake_max` |
| Camber & Toe | `gamma_fl, gamma_rl, toe_front, toe_rear` |
| Aero & Environment | `rho, g, Cd, Cl` |
| Numerical | `eps_x, eps_y, eps_K` |
| Pacejka — longitudinal | `pCx1, pDx1, pDx2, pDx3, pEx1…pEx4, pKx1…pKx3, pHx1, pHx2, pVx1, pVx2, rBx1…rBx3, rCx1, rEx1, rEx2, rHx1` |
| Pacejka — lateral | `pCy1, pDy1…pDy3, pEy1…pEy5, pKy1…pKy7, pHy1, pHy2, pVy1…pVy4, rBy1…rBy4, rCy1, rEy1, rEy2, rHy1, rHy2, rVy1…rVy6` |

Notes carried into tooltips:
- `Cd` / `Cl` are **overwritten by `Data/DATA_AA.mat`** at solve time when that
  file is present, so editing them usually has no effect — the tooltip says so.
- `eps_x/eps_y/eps_K` are solver smoothing epsilons, not physical car
  parameters; exposed (scope = "everything") but grouped under **Numerical**.

## Approved decisions

1. **Migrate the 7 named `RunConfig` fields into a single `vp` dict** (one
   source of truth). `TIER1_FIELDS` and the named fields are removed;
   `test_runconfig.py` is rewritten against the `vp` dict. No persistence today
   stores top-level `brkB`-style keys, so this breaks nothing beyond that test.
2. **User-saved presets are written to a writable `user_presets_dir()`** under
   `~/Documents/FullModelSim/presets/`, not the read-only bundled
   `app/presets/` (which is inside `sys._MEIPASS` when frozen). Loading reads
   from both the bundled and user directories.

## Architecture

### New module — `app/vp_params.py` (pure Python, no Qt; unit-testable)

Single source of truth for **presentation** of each parameter (values still
come from `vehParams`).

- `PARAM_GROUPS: list[tuple[str, list[str]]]` — ordered `(section_title,
  [keys])` as in the table above. Covers `PRIMARY_KEYS ∪ MF_KEYS` exactly once.
- `@dataclass ParamMeta`: `label, unit, tooltip, lo, hi, step, decimals`.
- `META: dict[str, ParamMeta]` — hand-authored entries for the ~49 physical
  primaries (sensible labels, SI units `kg/m/N·m/N/m/deg/…`, tight ranges).
- `meta_for(key) -> ParamMeta` — returns the annotated entry, else a **generic
  fallback** (wide symmetric range, 6 decimals, no unit, label = key). This is
  what the ~60 Pacejka coefficients use, so they need no hand entries and are
  never silently clamped.
- `all_vp_defaults() -> dict` — `{**default_primaries(), **vars(_default_mf())}`.
  This is (a) the seed for `RunConfig.vp`, (b) the reset baseline, and (c) the
  reference for changed-from-default highlighting.

**Invariant (enforced by test):** the union of `PARAM_GROUPS` keys equals
`PRIMARY_KEYS ∪ MF_KEYS`. If a parameter is later added to `vehParams` without
being placed in a group, the test fails — preventing silent omission.

### New module — `app/widgets.py`

`CollapsibleSection(QWidget)` — a reusable collapsible group (Qt ships none):
a `QToolButton` header (▶/▼) that toggles a content widget's visibility, plus a
live "(N changed)" badge in the header. `addRow(label_widget, field_widget)`
delegates to an inner `QFormLayout`. `set_expanded(bool)`.

### `app/paths.py`

Add `user_presets_dir()` → `os.path.join(default_output_dir_root, "presets")`
(writable, created on demand). `app/presets/` (bundled) remains read-only and is
resolved via `resource_path("app", "presets")`.

### `app/runconfig.py` — `vp` dict model

```python
from app.vp_params import all_vp_defaults

@dataclass
class RunConfig:
    # …run fields unchanged: circuit, AeroConfig, ATD, Electric_4Motors,
    #   TyreModel, vi, ni, linear_solver, warm_start, save, plot, output_dir,
    #   expert_config…
    vp: dict = field(default_factory=all_vp_defaults)

    def vp_overrides(self):
        ov = {}
        if self.expert_config:
            with open(self.expert_config) as fh:
                ov.update(json.load(fh))          # expert is the base layer
        defaults = all_vp_defaults()
        for k, v in self.vp.items():               # GUI diff overrides expert
            if v != defaults[k]:
                ov[k] = v
        unknown = set(ov) - PRIMARY_KEYS - MF_KEYS
        if unknown:
            raise ValueError(f"Unknown vehParams override keys: {sorted(unknown)}")
        return ov
```

- Precedence preserved: **expert file < GUI**. With the GUI at defaults the diff
  is empty, so expert values pass through unchanged (keeps `test_runconfig`'s
  expert-merge semantics: GUI `brkB≠default` overrides expert `brkB`).
- `_cfg()` drops `vp` and `expert_config`, emits `vp_overrides` only — so
  **`headless_solve` is unchanged** (it already reads only `vp_overrides`).
- `to_dict`/`from_dict` round-trip the `vp` dict (it is a dataclass field).

### `app/mainwindow.py` — Setup tab rebuild

- `_build_setup_tab()` returns a `QScrollArea` containing one
  `CollapsibleSection` per `PARAM_GROUPS` entry, built by iteration. **Balance &
  Aero** starts expanded; the rest collapsed.
- Each row: a tooltip'd `QLabel` + a `QDoubleSpinBox` configured from
  `meta_for(key)` (range/step/decimals, unit via `setSuffix`). A generalized
  `_spin_for(key)` replaces the ad-hoc `_spin`. All spins kept in
  `self.vp_spins: dict[str, QDoubleSpinBox]`.
- **Reset:** a global "Reset all to defaults" button and a per-section reset,
  both writing `all_vp_defaults()` back into the relevant spins.
- **Changed highlight:** each spin's `valueChanged` compares to its default and
  applies a bold/colored style when different; the owning section's "(N
  changed)" badge updates. Mirrors exactly what `vp_overrides()` will send.
- **Save/Load preset:** Save writes the full param set (the existing
  `default.json` flat-key format) via `QFileDialog`, defaulting to
  `user_presets_dir()`. Load reads a JSON and sets matching spins; unknown keys
  are ignored with a log line. Shipped presets load from the bundled
  `app/presets/`.
- `collect_runconfig()` builds `vp = {k: s.value() for k, s in
  self.vp_spins.items()}` and passes `vp=vp`; the 7 named kwargs are removed.
- The Advanced-tab "Expert config" picker stays; `_load_expert_tier1` is renamed
  `_load_expert_into_widgets` and generalized to set any matching spin in
  `self.vp_spins` (intersection of file keys with known spins).

## Data flow (unchanged below `collect_runconfig`)

```
vp_spins (dict of QDoubleSpinBox)
  → collect_runconfig(): vp = {key: spin.value()}
  → RunConfig(vp=vp, …run fields, expert_config)
  → write_cfg(): cfg["vp_overrides"] = vp_overrides()   # expert ∪ (vp diff)
  → cfg.json
  → headless_solve.build_solve_kwargs(): vp_overrides passed through
  → MLTP(vp_overrides=…) → vehParams(ctx, vp_overrides=…)
```

## Error handling

- Loading a preset / expert file with **unknown keys**: ignored at widget-load
  time (logged), and still hard-rejected by `vp_overrides()`'s
  `ValueError` if they reach an override dict.
- Out-of-range loaded values are clamped by the spinbox; physical ranges are set
  generously and Pacejka coeffs use the wide fallback range to avoid surprising
  clamps.
- Save to a read-only location is avoided by targeting `user_presets_dir()`
  (created on demand); failures are caught and surfaced in the log pane.

## Components & isolation

| Unit | Purpose | Depends on | Qt? |
|---|---|---|---|
| `app/vp_params.py` | param presentation registry + defaults helper | `vehParams` | no |
| `app/widgets.py` | `CollapsibleSection` reusable widget | PySide6 | yes |
| `app/runconfig.py` | `vp` dict config model + override emission | `vp_params`, `vehParams` | no |
| `app/paths.py` | `user_presets_dir()` | stdlib | no |
| `app/mainwindow.py` | builds Setup widgets from the registry | all of the above | yes |

The two testable-without-Qt units (`vp_params`, `runconfig`) hold all the
non-trivial logic; the Qt units are thin construction code.

## Testing

- **Rewrite `test_runconfig.py`** for the `vp` dict: round-trip
  `from_dict(to_dict)`; `vp` diff folds into `vp_overrides`; expert-file merge
  with GUI-overrides-expert precedence; unknown expert key raises; run fields
  preserved in `_cfg()`.
- **New `test_vp_params.py`:** `PARAM_GROUPS` key-union == `PRIMARY_KEYS ∪
  MF_KEYS` (no gaps/extras, no duplicates); `all_vp_defaults()` equals the merge
  of `default_primaries()` + `_default_mf()`; every `meta_for(key)` has
  `lo ≤ default ≤ hi` and `decimals ≥ 0`.
- **`test_presets.py`:** unchanged (`default.json` untouched, still a subset of
  `PRIMARY_KEYS ∪ MF_KEYS`).
- GUI construction (`MainWindow`) remains not unit-tested (no `QApplication`
  harness exists today); all new logic lives in the pure modules above.

## Out of scope (YAGNI)

- No changes to the solve, `vehModel`, `userOpts`, or `headless_solve`.
- No per-parameter physical validation beyond spinbox ranges + the existing
  unknown-key check.
- No special-casing of `Cd`/`Cl` overwrite — surfaced via tooltip only.
- No search/filter box over the param list (revisit only if the section UI
  proves hard to navigate).
