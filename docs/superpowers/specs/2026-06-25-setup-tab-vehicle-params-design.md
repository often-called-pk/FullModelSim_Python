# Setup-tab full vehicle-parameter editor — design

**Date:** 2026-06-25
**Status:** Approved (design); implementation pending plan.
**Component:** `app/` Windows GUI (PySide6).

## Goal

Two related additions to the GUI:

1. **Setup tab — full vehicle-parameter editor.** Expand from the current 7
   hardcoded tunables to **every adjustable vehicle parameter** `vehParams`
   accepts as an override (`PRIMARY_KEYS ∪ MF_KEYS` — **51 primaries + 59
   Pacejka 5.2 coefficients = 110**), in collapsible sections with reset,
   save/load presets, changed-from-default highlighting, and tooltips/units.
2. **Advanced tab — solver & collocation options.** Expose five knobs currently
   hardcoded inside `userOpts` — **max iterations, collocation step (`OPT_ds`),
   interpolating-polynomial degree (`OPT_d`), path-constraint slack (`OPT_e`),
   and IPOPT tolerance (`tol`)** — next to the existing linear-solver control.

The vehicle params flow through the existing `vp_overrides` path (no solve
change). The five solver options need a small plumbing addition: `userOpts`
gains them as keyword arguments (identical defaults) and `headless_solve`
forwards them from `cfg.json`. The transcription/model internals
(`MLTP`, `vehModel`, `functions/transcription.py`) are unchanged — `MLTP`
already forwards `**useropts_kwargs` to `userOpts` and to `MLTP_initial`.

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
tab's "Expert config .json" file picker. So the **vehicle-parameter** half of
this task is purely **GUI widgets + `RunConfig` storage** — no new solve
capability is required.

The **solver/collocation** half is different: `OPT_ds, OPT_d, OPT_e, max_iter,
tol` are hardcoded inside `userOpts` (lines ~242–267) and are *not* currently
parameterised. `MLTP`/`MLTP_initial` already forward `**useropts_kwargs` to
`userOpts`, so exposing them needs only: make `userOpts` accept them as kwargs
(identical defaults) and make `headless_solve` forward them from `cfg.json`.

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
3. **The five solver/collocation options live on the Advanced tab** (a new
   "Solver & Collocation" group beside the linear-solver control), are carried
   as **top-level `RunConfig`/`cfg.json` fields** (not `vp_overrides`), and are
   plumbed by adding keyword arguments to `userOpts` with defaults identical to
   today's hardcoded values. They apply to **both** the 7-state warm-start solve
   and the 23-state solve (as the hardcoded values do today).

## Architecture

### New module — `app/vp_params.py` (pure Python, no Qt; unit-testable)

Single source of truth for **presentation** of each parameter (values still
come from `vehParams`).

- `PARAM_GROUPS: list[tuple[str, list[str]]]` — ordered `(section_title,
  [keys])` as in the table above. Covers `PRIMARY_KEYS ∪ MF_KEYS` exactly once.
- `@dataclass ParamMeta`: `label, unit, tooltip, lo, hi, step, decimals, kind`,
  where `kind ∈ {"spin", "sci"}` selects the widget (see **Widget
  representation** below).
- `META: dict[str, ParamMeta]` — hand-authored entries for the 51 physical
  primaries (sensible labels, SI units `kg/m/N·m/N/m/deg/…`, tight ranges).
- `meta_for(key) -> ParamMeta` — returns the annotated entry, else a **generic
  fallback** (wide symmetric range, `kind="sci"`, label = key). This is what the
  59 Pacejka coefficients use, so they need no hand entries and are never
  silently clamped or rounded to zero.
- `all_vp_defaults() -> dict` — `{**default_primaries(), **vars(_default_mf())}`.
  This is (a) the seed for `RunConfig.vp`, (b) the reset baseline, and (c) the
  reference for changed-from-default highlighting.

**Invariant (enforced by test):** the union of `PARAM_GROUPS` keys equals
`PRIMARY_KEYS ∪ MF_KEYS`. If a parameter is later added to `vehParams` without
being placed in a group, the test fails — preventing silent omission.

### Widget representation — why two field kinds

A plain `QDoubleSpinBox` stores its value rounded to its `decimals` setting, so
it **cannot represent** the small-magnitude tyre coefficients in this set —
e.g. `rHy1 = -9.1492e-11`, `rVy5 = -8.8234e-11`, `pHy2 = 0.00051596`, and the
`1e-6` numerical epsilons. Loading those into a spinbox would silently collapse
them to `0`, corrupting the tyre model and producing a bogus
"changed-from-default" diff. Because the chosen scope is "everything (~110)",
this must be handled, not deferred.

Resolution: `ParamMeta.kind` picks the editor per parameter.
- `kind="spin"` → `QDoubleSpinBox` (range/step/decimals/unit) for the
  well-scaled physical primaries — keeps nice step arrows for tuning.
- `kind="sci"` → a `ScientificField` (see `app/widgets.py`) backed by a
  `QLineEdit` + `QDoubleValidator(ScientificNotation)`. It displays via
  `"%.12g"` and reports `value() = float(text)`, so it round-trips
  `-9.1492e-11` and `1e-6` exactly with no rounding.

Assignment rule (in the registry, covered by a test): **all `MF_KEYS` and the
three `eps_*` keys use `kind="sci"`; every other key uses `kind="spin"`.** Both
kinds expose the same `value()` / `setValue()` / `valueChanged` API so the
Setup tab treats them uniformly.

### New module — `app/widgets.py`

- `CollapsibleSection(QWidget)` — a reusable collapsible group (Qt ships none):
  a `QToolButton` header (▶/▼) that toggles a content widget's visibility, plus
  a live "(N changed)" badge in the header. `addRow(label_widget, field_widget)`
  delegates to an inner `QFormLayout`. `set_expanded(bool)`.
- `ScientificField(QWidget)` — a `QLineEdit` + `QDoubleValidator` (with
  `setNotation(QDoubleValidator.ScientificNotation)`) exposing the same minimal
  API as `QDoubleSpinBox`: `value() -> float` (= `float(text)`),
  `setValue(float)` (text set via `"%.12g"`), and a `valueChanged` signal
  emitted on `editingFinished`/text change. Used for `kind="sci"` parameters so
  tiny tyre coefficients and epsilons round-trip without rounding.

### `app/paths.py`

Add `user_presets_dir()` → `os.path.join(default_output_dir(), "presets")`
(i.e. `~/Documents/FullModelSim/presets/`; writable, created on demand —
`default_output_dir()` already returns the `FullModelSim` folder). The bundled
`app/presets/` remains read-only and is resolved via
`resource_path("app", "presets")`.

### `app/runconfig.py` — `vp` dict model

```python
from app.vp_params import all_vp_defaults

@dataclass
class RunConfig:
    # …run fields unchanged: circuit, AeroConfig, ATD, Electric_4Motors,
    #   TyreModel, vi, ni, linear_solver, warm_start, save, plot, output_dir,
    #   expert_config…
    # NEW top-level solver/collocation fields (defaults == userOpts defaults):
    max_iter: int = 6000
    OPT_ds: float = 30.0
    OPT_d: int = 3
    OPT_e: float = 1e-2
    tol: float = 1e-4
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
- `_cfg()` drops `vp` and `expert_config` and emits `vp_overrides`; the five
  solver fields (`max_iter, OPT_ds, OPT_d, OPT_e, tol`) **stay as top-level cfg
  keys**, alongside `linear_solver`.
- `to_dict`/`from_dict` round-trip the `vp` dict and the solver fields (all
  dataclass fields).

### `app/mainwindow.py` — Setup tab rebuild

- `_build_setup_tab()` returns a `QScrollArea` containing one
  `CollapsibleSection` per `PARAM_GROUPS` entry, built by iteration. **Balance &
  Aero** starts expanded; the rest collapsed.
- Each row: a tooltip'd `QLabel` + a field built from `meta_for(key)`. A
  `_field_for(key)` factory returns a `QDoubleSpinBox` (`kind="spin"`,
  range/step/decimals, unit via `setSuffix`) or a `ScientificField`
  (`kind="sci"`); the ad-hoc `_spin` helper is removed. All fields kept in
  `self.vp_spins: dict[str, QDoubleSpinBox | ScientificField]` (both share the
  `value()`/`setValue()`/`valueChanged` API).
- **Reset:** a global "Reset all to defaults" button and a per-section reset,
  both writing `all_vp_defaults()` back into the relevant fields.
- **Changed highlight:** driven purely by **current value vs. default**, so it
  stays correct after any reset, preset load, or expert-file load. Each field's
  `valueChanged` re-evaluates and applies a bold/colored style when different;
  the owning section's "(N changed)" badge updates. Mirrors exactly what
  `vp_overrides()` will send.
- **Save/Load preset:** presets use the same **flat `{key: value}` JSON shape**
  as `app/presets/default.json`. Save writes the full current param set (all 110
  keys, including the Pacejka coefficients that `default.json` omits) via
  `QFileDialog`, defaulting to `user_presets_dir()`. Load reads a JSON and sets
  matching fields; a partial file (e.g. `default.json`, which has only the 51
  primaries) is fine — keys it omits keep their current value. Unknown keys are
  ignored with a log line. Shipped presets load from the bundled `app/presets/`.
- `collect_runconfig()` builds `vp = {k: f.value() for k, f in
  self.vp_spins.items()}` and passes `vp=vp`. **Only the 7 vehicle-param kwargs
  are replaced by `vp=`; all run-config kwargs (`circuit, AeroConfig, ATD,
  Electric_4Motors, TyreModel, vi, ni, linear_solver, warm_start, save, plot,
  output_dir, expert_config`) are unchanged.**
- The Advanced-tab "Expert config" picker stays; `_load_expert_tier1` is renamed
  `_load_expert_into_widgets` and generalized to set any matching spin in
  `self.vp_spins` (intersection of file keys with known spins).

### `app/mainwindow.py` — Advanced tab "Solver & Collocation" group

A new `QGroupBox` ("Solver & Collocation") added to `_build_advanced_tab`,
beside the existing Linear solver / Warm start / Expert config rows. Five
widgets:

| Label | field | widget | default | range | note |
|---|---|---|---|---|---|
| Max iterations | `max_iter` | `QSpinBox` (int) | 6000 | 1 … 100000 | IPOPT `max_iter` |
| Collocation step [m] | `OPT_ds` | `QDoubleSpinBox` | 30.0 | 1 … 500 | grid spacing; smaller ⇒ more knots |
| Polynomial degree | `OPT_d` | `QSpinBox` (int) | 3 | 1 … 6 | degree of interpolating polynomials |
| Path-constraint slack | `OPT_e` | `ScientificField` | 1e-2 | 1e-6 … 1 | `OPT_e` slack/seed |
| IPOPT tolerance | `tol` | `ScientificField` | 1e-4 | 1e-10 … 1e-1 | IPOPT `tol` |

`collect_runconfig()` reads these into the new `RunConfig` fields. They are
**not** part of `self.vp_spins` and never enter `vp_overrides`. (`OPT_e` and
`tol` reuse the `ScientificField` from `app/widgets.py`.)

### `userOpts.py` — accept the five options as keyword arguments

Add parameters with defaults equal to today's hardcoded values, so an
unmodified call is byte-for-byte identical:

```python
def userOpts(ctx, …existing args…,
             OPT_ds=30, OPT_d=3, OPT_e=1e-2,   # collocation
             max_iter=6000, tol=1e-4):          # IPOPT
    …
    ctx.OPT_ds = OPT_ds
    ctx.OPT_d  = OPT_d
    ctx.OPT_e  = OPT_e
    ctx.OPT_uinter = "linear"                   # unchanged (not exposed)
    …
    ipopt = { …, "max_iter": max_iter, "tol": tol, … }
```

`MLTP` and `MLTP_initial` already pass `**useropts_kwargs` straight through, so
no change is needed in either; the same values reach both the 7-state and
23-state solves.

### `headless_solve.py` — forward the five cfg keys

`build_solve_kwargs(cfg, …)` adds the five to the MLTP kwargs, reading with the
userOpts defaults so an older `cfg.json` without them still works:

```python
kwargs["max_iter"] = int(cfg.get("max_iter", 6000))
kwargs["OPT_ds"]   = float(cfg.get("OPT_ds", 30))
kwargs["OPT_d"]    = int(cfg.get("OPT_d", 3))
kwargs["OPT_e"]    = float(cfg.get("OPT_e", 1e-2))
kwargs["tol"]      = float(cfg.get("tol", 1e-4))
```

These land in MLTP's `**useropts_kwargs` and are forwarded to `userOpts`.

## Data flow (vehicle params unchanged below `collect_runconfig`)

```
vp_spins (dict of QDoubleSpinBox | ScientificField)  + 5 solver widgets
  → collect_runconfig(): vp = {key: field.value()}; max_iter/OPT_ds/OPT_d/OPT_e/tol
  → RunConfig(vp=vp, max_iter=…, OPT_ds=…, …run fields, expert_config)
  → write_cfg():
        cfg["vp_overrides"] = vp_overrides()              # expert ∪ (vp diff)
        cfg["max_iter"], cfg["OPT_ds"], cfg["OPT_d"],     # top-level keys
        cfg["OPT_e"], cfg["tol"]
  → cfg.json
  → headless_solve.build_solve_kwargs():
        vp_overrides passed through; 5 solver keys → MLTP **useropts_kwargs
  → MLTP(vp_overrides=…, OPT_ds=…, …)
        → vehParams(ctx, vp_overrides=…)
        → userOpts(ctx, OPT_ds=…, OPT_d=…, OPT_e=…, max_iter=…, tol=…)
```

## Error handling

- Loading a preset / expert file with **unknown keys**: ignored at widget-load
  time (logged), and still hard-rejected by `vp_overrides()`'s
  `ValueError` if they reach an override dict.
- Values typed into a spin field are clamped to its range by Qt; `ScientificField`
  validates typed input but accepts programmatically-loaded values as-is.
  Physical ranges are set generously and Pacejka coeffs use the wide `sci`
  fallback, so legitimate values are never surprised by a clamp.
- Save to a read-only location is avoided by targeting `user_presets_dir()`
  (created on demand); failures are caught and surfaced in the log pane.

## Components & isolation

| Unit | Purpose | Depends on | Qt? |
|---|---|---|---|
| `app/vp_params.py` | param presentation registry + defaults helper | `vehParams` | no |
| `app/widgets.py` | `CollapsibleSection` + `ScientificField` reusable widgets | PySide6 | yes |
| `app/runconfig.py` | `vp` dict + 5 solver fields; override/cfg emission | `vp_params`, `vehParams` | no |
| `app/paths.py` | `user_presets_dir()` | stdlib | no |
| `app/mainwindow.py` | builds Setup + Advanced-tab widgets | all of the above | yes |
| `userOpts.py` | accept `OPT_ds/OPT_d/OPT_e/max_iter/tol` kwargs | — | no |
| `headless_solve.py` | forward the 5 solver cfg keys to MLTP | stdlib | no |

The two testable-without-Qt units (`vp_params`, `runconfig`) hold all the
non-trivial logic; the Qt units are thin construction code.

## Testing

- **Rewrite `test_runconfig.py`** for the `vp` dict: round-trip
  `from_dict(to_dict)`; `vp` diff folds into `vp_overrides`; expert-file merge
  with GUI-overrides-expert precedence; unknown expert key raises; run fields
  preserved in `_cfg()`; **the five solver fields appear as top-level cfg keys
  (not inside `vp_overrides`) with both defaults and overridden values.**
- **Extend `test_headless_config.py`:** `build_solve_kwargs` forwards
  `max_iter/OPT_ds/OPT_d/OPT_e/tol` (correct types) into the MLTP kwargs, and a
  cfg missing them falls back to the userOpts defaults.
- **Extend `test_vp_overrides.py` (or add a small userOpts test):** calling
  `userOpts()` with no new kwargs still gives `ctx.OPT_ds==30, OPT_d==3,
  OPT_e==1e-2`, `ipopt["max_iter"]==6000`, `ipopt["tol"]==1e-4`; passing each
  kwarg overrides the corresponding `ctx`/`ipopt` value.
- **New `test_vp_params.py`:** `PARAM_GROUPS` key-union == `PRIMARY_KEYS ∪
  MF_KEYS` (no gaps/extras, no duplicates); `all_vp_defaults()` equals the merge
  of `default_primaries()` + `_default_mf()`; every `meta_for(key)` has
  `lo ≤ default ≤ hi` and `decimals ≥ 0`; **`kind="sci"` for every `MF_KEYS`
  member and every `eps_*`, `kind="spin"` otherwise**; and the `"%.12g"`
  format/parse helper round-trips **every** default value exactly (catches the
  tiny-coefficient rounding bug — `float("%.12g" % v) == v` for all 110).
- **`test_presets.py`:** unchanged (`default.json` untouched, still a subset of
  `PRIMARY_KEYS ∪ MF_KEYS`).
- GUI construction (`MainWindow`) remains not unit-tested (no `QApplication`
  harness exists today); all new logic lives in the pure modules above.

## Out of scope (YAGNI)

- No changes to the transcription/model internals (`MLTP`, `MLTP_initial`,
  `vehModel`, `functions/transcription.py`). `userOpts` and `headless_solve`
  change only to accept/forward the five solver kwargs.
- Of the solver/collocation knobs, only the five requested are exposed. The
  input-interpolation mode (`OPT_uinter`) and the other IPOPT options
  (`acceptable_tol`, `mu_*`, bound pushes, etc.) stay hardcoded in `userOpts`.
- No per-parameter physical validation beyond spinbox ranges + the existing
  unknown-key check.
- No special-casing of `Cd`/`Cl` overwrite — surfaced via tooltip only.
- No search/filter box over the param list (revisit only if the section UI
  proves hard to navigate).
