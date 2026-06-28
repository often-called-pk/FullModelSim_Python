# Drive-Source Abstraction + Zenvo Aurora Hybrid Powertrain — Design Spec

**Date:** 2026-06-28
**Branch context:** `coin-hsl`
**Approach:** B — refactor the `EM4`/`ATD` powertrain branching into a generic, data-driven
*drive-source* abstraction, then express the Zenvo Aurora 3-motor + ICE topology as data on top
of it. Add the full Aurora vehicle as an **opt-in parameter profile** (baseline car unchanged).

---

## 1. Goal & motivation

Model the **Zenvo Aurora** powertrain in the MLTP framework: **two independent front e-motors**
(per-corner torque vectoring), a **rear axle driven by one e-motor *and* an ICE**, with a
**controllable rear torque-vectoring split**. The current framework only supports two hard-coded
topologies (`pt.EM4=0` single aggregated motor; `pt.EM4=1` four independent corner motors), and
Aurora is neither.

Rather than add a third `if`-branch everywhere (Approach A), we refactor the powertrain into a
**drive-source list** that makes all three topologies *data, not branches*. This removes the
duplicated control-vector ordering (today split across `vehModel.u_list` and
`userOpts._input_keys`, which can silently desync) by giving both a single shared source of truth.

### Locked decisions (from brainstorming)

| # | Decision |
|---|---|
| Energy model | **Pure torque sources** — no SoC/fuel/thermal *states*. `nx` stays **23**. (A post-solve energy *integral* is still reported, §9.) |
| Rear sources | Rear e-motor and ICE are **two independent torque controls**, each with its own gear & cap, summed at the rear axle. |
| Front | Two **independent** front e-motors (per-corner). |
| Rear diff | **Controllable** torque-vectoring split `split_R ∈ [0,1]` (one new control). |
| ICE envelope | **Digitized rpm→torque curve** `T_ice ≤ poly(Om_ice)`; numeric curve is a flagged **placeholder** until the engine-map image is supplied. |
| ICE gear | **Fixed representative gear** per solve (config param `ice_gear`, default 6th = 1.0); no shifting. |
| Selector | New **`Hybrid='On'/'Off'`** flag; when `On` it overrides `EM4`/`ATD`. |
| Aurora params | **Opt-in profile** `apply_aurora_params(ctx)`; baseline default car numbers unchanged (DP2). |
| Gear convention | **Fix the latent bug** (DP1): unify to `Om_source = mean(target wheels) × gear`. |
| Energy report | **Split** `E_motor` (electric, back-compat) + `E_fuel` (ICE), per-source efficiency (DP3). |

---

## 2. Background: how the powertrain works today (verified by codebase map)

- **Control vector** (`vehModel.u_list`, lines 139–166) is assembled in a fixed MATLAB order:
  `[motor torque(s)] → T_brake → [4 ATD if ATD==1] → [aero per ActAero] → delta (LAST)`. It is
  packed into the NLP decision vector `w` **column-major (`order='F'`)** in `transcription.py`.
- **The control order is defined twice** — `vehModel.u_list` *and* `userOpts._input_keys`
  (149–167) — and must stay row-for-row aligned. This duplication is the chief fragility the
  refactor removes.
- **Torque split** (`vehModel.py` 476–508): converts motor torque(s) → per-corner wheel torques
  `T_fl/T_fr/T_rl/T_rr` using the single shared scalar `vp.gear`, `vp.Tdist` (EM4=0 only),
  `vp.brkB`. The split torques feed **only** the wheel-rotation dynamics `dOm_*` (522–525); the
  chassis force balance uses tyre forces directly, so it is **topology-agnostic**.
- **Ratings** `pt.Pmax/Tmax/OMmax` do **not** appear in `vehModel.py`; power/rpm limits are
  enforced in `MLTP.build_path_constraints` (41–93), which branches on `pt.EM4`/`pt.ATD`.
- **`pt.EM4`/`pt.ATD` are set in `userOpts.py`** (≈201–213), not `Powertrain.py`. Guard: if `ATD`
  and `Electric_4Motors` are both `On`, ATD is forced Off with a warning.
- **The init 7-state model** (`vehModel_initial.py`) is always a single aggregated motor with a
  fixed 3-control vector `[T_motor, T_brake, delta]`, independent of `input_keys`. It seeds the
  full solve.

### The latent bug (DP1)

`vehModel.py:478` (EM4=0) computes motor speed as `Om_motor = mean_wheel × gear` (**multiply**),
but `vehModel.py:496–497` (EM4=1) computes `Om_motor_* = Om_wheel / gear` (**divide**). With
`gear = OMmax·Rw/Vmax ≈ 8.7`, the motor spins *faster* than the wheel, so **multiply is correct**;
the EM4=1 divide makes its `motor_power`/`motor_rpm` constraints operate on a speed that is wrong
by a factor of `gear` (power off by `gear²`). The abstraction adopts one convention everywhere:

```
Om_source = mean(speeds of wheels the source drives) × source.gear
T_wheel   = Σ_sources(T_source · source.gear · share_to_wheel) + brake_term
P_source  = T_source · Om_source            # = T_wheel · Om_wheel (power-consistent)
```

This **changes EM4=1 numeric results** (deliberate). No test pins EM4=1 *numbers* (only its
control-channel *names*), so nothing breaks mechanically; documented as a fix.

---

## 3. Core data model — `functions/drive_sources.py` (new, casadi-free)

A new MPI-free, CasADi-free module — testable like `functions/sweep.py`. It is the **single source
of truth** for the powertrain topology and the control-vector ordering.

```python
# Plain SimpleNamespace/dataclass-style descriptors (no CasADi).

class DriveSource:
    name:    str            # 'fl','fr','r','ice', ...  (also forms channel/signal names)
    type:    str            # 'emotor' | 'ice'
    node:    str            # which wheels it feeds: 'fl'|'fr'|'rl'|'rr'|'front'|'rear'|'all'
    gear:    float          # per-source gear ratio (motor spins gear× wheel)
    Tmax:    float          # torque scale / cap (Nm)
    Pmax:    float          # power cap (W)        — emotor flat cap
    OMmax:   float          # rpm cap (rad/s)      — emotor flat cap
    curve:   tuple | None   # ICE rpm→torque polynomial coeffs (None for emotor)
    regen:   bool           # False → torque ∈ [0, Tmax]; True → [-Tmax, Tmax]
    ctrl_key: str           # control-channel name it owns, e.g. 'T_motor_fl', 'T_ice_r'

class SplitPolicy:
    node:    str            # multi-wheel node the policy distributes: 'front'|'rear'|'all'
    kind:    str            # 'fixed' | 'atd' | 'tv'
    keys:    list[str]      # fraction-control channel names ([] for 'fixed')
    # 'fixed' carries vp.Tdist (front/rear) + 50/50 side split, applied in vehModel
```

### `build_topology(topology, pt, vp) -> (sources, splits)`

Maps each topology string to data. `topology ∈ {'single','four_motor','hybrid'}` (derived from the
`EM4`/`ATD`/`Hybrid` flags in `userOpts`).

**Where ratings come from.** For the legacy topologies, every source is assigned the existing shared
scalars `pt.Pmax/Tmax/OMmax` as-is — so `single` caps at `pt.Pmax` *as an aggregate* and each of the
four `four_motor` sources caps at `pt.Pmax` *per motor*, **reproducing today's dual semantics
unchanged** (the map flagged that `pt.Pmax/Tmax` mean "3-motor total" in EM4=0 but "per-motor" in
EM4=1; we preserve that rather than reinterpret it). Only the `hybrid` source list uses explicit
per-source ratings/gears (from `apply_aurora_params`, §10). `pt.Vmax`/`pt.eff` stay vehicle-level.

| topology | sources | splits |
|---|---|---|
| `single` (EM4=0, ATD=0) | 1× emotor `node='all'`, `ctrl_key='T_motor'` | `fixed` on `all` (uses `vp.Tdist`, `vp.brkB`) |
| `single`+ATD (EM4=0, ATD=1) | 1× emotor `node='all'`, `ctrl_key='T_motor'` | `atd` on `all`, `keys=['ATD','ATD','ATD','ATD']` |
| `four_motor` (EM4=1) | 4× emotor `node='fl'/'fr'/'rl'/'rr'`, `ctrl_key='T_motor_fl'…'T_motor_rr'` | none |
| **`hybrid`** | emotor@`fl` (`T_motor_fl`), emotor@`fr` (`T_motor_fr`), emotor@`rear` (`T_motor_r`), **ice@`rear`** (`T_ice_r`) | `tv` on `rear`, `keys=['split_R']` |

### `control_keys(sources, splits, ActAero) -> list[str]`

The **one** ordering definition. Returns, in order:

```
[s.ctrl_key for s in sources]      # source torques, in source-list order
+ ['T_brake']
+ [k for sp in splits for k in sp.keys]   # split fractions (ATD ×4, or split_R)
+ aero_keys(ActAero)               # [] | ['RW'] | ['FW','RW'] | ['FW','FW','RW','TW']
+ ['delta']                        # ALWAYS last
```

**Back-compat (byte-for-byte) check** — this reproduces every legacy `input_keys` list the tests
pin:

| config | `control_keys` output |
|---|---|
| EM4=0, ATD=1, Static | `['T_motor','T_brake','ATD','ATD','ATD','ATD','delta']` ✓ |
| EM4=1, Static | `['T_motor_fl','T_motor_fr','T_motor_rl','T_motor_rr','T_brake','delta']` ✓ |
| EM4=0, ATD=0, AALB | `['T_motor','T_brake','FW','FW','RW','TW','delta']` ✓ |
| **Hybrid, Static** | `['T_motor_fl','T_motor_fr','T_motor_r','T_ice_r','T_brake','split_R','delta']` (new) |

> Aero key naming preserves the current quirk: ActAero=2 emits `['FW','RW']`, ActAero=3 emits
> `['FW','FW','RW','TW']` (matching `_input_keys` today and `test_params_useropts`).

---

## 4. `vehModel.py` — consume the source list

Replace the three `pt.EM4`/`pt.ATD` branches with loops over `pt.sources` / `pt.splits`.

1. **Symbol creation** (replaces 88–97): for each `source`, create one normalized torque symbol
   `<ctrl_key>_n`, scale `= source.Tmax`, limit row `[0,1]` (or `[-1,1]` if `source.regen`). For
   each `split.keys`, create a fraction symbol (scale 1, limit `[0,1]`), exactly as ATD is today.
2. **`u_list` assembly** (replaces 139–160): build by calling the **same** `control_keys` ordering
   — sources → brake → split fractions → aero → delta. `u/u_s/u_lim/u_min/u_max/nu` unchanged
   downstream.
3. **Torque split** (replaces 476–508): for each wheel `w ∈ {fl,fr,rl,rr}`:
   ```
   T_w = Σ_{s feeds w} ( T_s · s.gear · share(s.node → w) ) + brake_term(w)
   T_w = if_else(xs_w ≥ 0.075, 0, T_w)          # per-wheel lift cutoff (preserved)
   ```
   - `share` from the node's split policy: `fixed` → `(1-Tdist)/2` front, `Tdist/2` rear;
     `atd` → the 4 ATD fractions; `tv` → `split_R` / `(1-split_R)` for rear L/R; corner nodes → 1.
   - `brake_term(w)` = the existing `T_brake·brkB` (front) / `T_brake·(1-brkB)` (rear); the legacy
     ATD branch's `ATD_*·T_brake·2` brake overlay is preserved for the `atd` policy.
   - Per source: `Om_s = mean(speeds of s.node's wheels) · s.gear` (DP1 convention);
     `P_s = T_s · Om_s`.
4. **Signal exposure** (replaces 576–588): expose a uniform per-source collection on `ctx.m23`
   (e.g. `m.src['fl'].T/T_n/Om/P`, plus the ICE `m.src['ice'].*`). **Keep legacy aliases**
   (`m.T_motor`, `m.Om_motor`, `m.P_motor` for `single`; `m.T_motor_*`, `m.Om_motor_*`,
   `m.P_motor_*`, `m.ATD_*` for the others) so existing consumers keep working. Always keep
   `m.T_fl/T_fr/T_rl/T_rr` (topology-agnostic) and `m.T_brake_n`.

**Untouched:** the entire 23-state chassis/suspension/tyre/aero dynamics, the wheel-speed state
scales (`Om_*_s` via `Rw_f/Rw_r`), `dOm_*` (522–525) consume only the four `T_*` totals.

---

## 5. `MLTP.build_path_constraints` — per-source generation

Replace the `pt.EM4`/`pt.ATD` branch (41–93) with:

1. **Friction circles** — always the four `rho_lim_fl/fr/rl/rr` first (`h_lb=0,h_ub=1`).
   Topology-independent; unchanged.
2. **Per source** (loop `pt.sources`):
   - `emotor`: `motor_power_<name> = (Pmax_s − Om_s·T_s)/Pmax_s ≥ 0`,
     `motor_rpm_<name> = (OMmax_s − Om_s)/OMmax_s ≥ 0`,
     `BrTh_<name> = (T_s_n · T_brake_n)/1e-3 ∈ [−1,1]` (drive/brake non-simultaneity).
   - `ice`: `ice_curve_<name> = (poly(Om_s) − T_s)/Tmax_s ≥ 0` (torque under the curve),
     `ice_rpm_<name>` idle/redline bound, `BrTh_<name>` as above.
3. **Controllable splits**: `atd` → existing `ATD_eq = 1 − Σ ATD_* = 0`; `tv` → `split_R ∈ [0,1]`
   (a simple bound; no equality needed for a 2-way fractional split).

Constraint **count/names are now data-driven**; the `assert len(hnames)==h.shape[0]` invariant and
the `order='F'`-compatible row ordering (4 `rho_lim` first) are preserved. `h_lb/h_ub` stay
finite/NaN-free.

**Back-compat:** for `single`/`four_motor` the generated names must equal today's
(`motor_power`/`motor_rpm`/`BrTh_1`/`ATD_eq`; `motor_power_fl…`/`motor_rpm_fl…`/`BrTh_fl…`). A unit
test pins this.

---

## 6. `userOpts.py` — flag, ordering, rate/reg tables, guard

- **New `Hybrid='On'/'Off'`** kwarg, threaded exactly like `ATD`/`Electric_4Motors`
  (`MLTP(...)` → `userOpts(**useropts_kwargs)`). Stored as `ctx.Hybrid`.
- **Topology + sources built here** (after flags resolve): call
  `build_topology(topology, pt, vp)` and hang `pt.sources` / `pt.splits` / `pt.topology` on `ctx`.
  Keep `pt.EM4`/`pt.ATD` set as **derived compat shims** (so `test_params_useropts` L83 and any
  unrefactored reader still work); add `pt.Hybrid`.
- **`_input_keys` delegates** to `control_keys(pt.sources, pt.splits, ActAero)`.
- **`_build_c` rate/reg table** (104–146): add rows for new channels:
  `T_motor_r` (≈ e-motor limits), `T_ice_r` (ICE-appropriate torque-rate), `split_R` (modeled like
  `ATD`). Legacy entries keep exact values (tests pin them positionally).
- **Guard redesign** (3-way): `Hybrid='On'` selects `hybrid` and **overrides** EM4/ATD (warn if
  the user also set them). The legacy "ATD and EM4 both On → force ATD off" warning is retained for
  the legacy clash. The guard must **not** reject Hybrid's legitimately dual-source rear axle.

---

## 7. Warm-start — `MLTP.warmstart_guesses`

The current dispatch (116–131) prefix-matches a **closed** key set; an unknown drive-source key
silently drops a `u0` row → `vstack` shape mismatch. Rewrite to iterate `ctx.input_keys` against
`pt.sources`:

- Source torque keys: seed from the single 7-state init torque `T_motor_0`, **split by Tmax-share**
  across sources feeding the driven wheels (so the hybrid sources sum to roughly the lumped init
  drive). ICE key seeded from its share likewise.
- `split_R`: seed `0.5`. ATD fractions: `0.25` (unchanged). aero: `0`. `T_brake`/`delta` from init.

Init (7-state) model stays a lumped single motor (`init_keys=['T_motor','T_brake','delta']`,
unchanged). `Om_fl/fr` seeds use `Rw_f`, `Om_rl/rr` use `Rw_r` (unchanged).

---

## 8. Save schema, plots, sweep, paramOptim

### Save (`MLTP.py`)
- **Keep** `T_fl/T_fr/T_rl/T_rr` and `E_motor` (electric aggregate) for back-compat; rear wheel
  torques sum emotor + ICE.
- **Add** per-source channels `P_<name>`/`Om_<name>` and the ICE `P_ice_r`/`Om_ice_r`, plus
  `E_fuel` (DP3), and `split_R` trace data.
- **Config string** (`cfg`, MLTP.py:267 *and* plotSDI.py:233 — must stay identical):
  ```
  cfg = f"{AeroConfig}_Hybrid"               if Hybrid on
        f"{AeroConfig}_ATD{ATD}_EM4{EM4}"    otherwise   # legacy byte-identical
  ```
  Add `data['Hybrid'] = ctx.Hybrid`. Legacy `data['ATD']/['EM4']` retained.

### Plots (`functions/plotSDI.py`)
- `plot_powertrain` already discovers `P_motor*/Om_motor*` by prefix (e-motors auto-plot). **Extend**
  it to also match `P_ice*/Om_ice*` and to plot `E_fuel` and the `split_R` control. `gg_plots.py`
  is **not** a powertrain consumer — leave untouched.

### Sweep (`functions/sweep.py`, `run_sweep.py`)
- Add **`Hybrid`** to `ACCEPTED_COLUMNS` (additive — safe for new sweeps; do not change the column
  set of a manifest you intend to resume). It threads to `MLTP` via `**useropts_kwargs`. Keep
  `ATD`/`Electric_4Motors` columns for back-compat.

### Co-optimization (`MLTP_paramOptim.py`, `MLTP_TyreOptim.py`)
- Both reuse `build_path_constraints` verbatim → inherit the generalization for free.
- Make the **default param set topology-aware**: `Tdist` is meaningful only for the aggregated
  `single` source (a no-op otherwise); `split_R` is a *control*, not a static param, so it is **not**
  promoted. `brkB` stays valid across topologies. (Promoting per-source ratings/gears is possible
  later but out of scope here, since they live on `pt.sources`, not `vp`.)

---

## 9. Post-solve energy integral (DP3)

No new *states* (pure torque sources). The reported energy becomes per-source:

```
E_motor = ∫ ( Σ_{emotor sources} P_s ) / eff_em  dt   [kWh]   # back-compat name
E_fuel  = ∫ ( P_ice ) / eff_ice              dt   [kWh]
```

Per-source `eff` lives on the source/`pt` (e-motor 0.90 default; ICE eff a flagged placeholder).
Legacy configs have no ICE → `E_fuel` absent/0.

---

## 10. Aurora parameter profile — `functions/aurora_params.py` (new, opt-in)

`apply_aurora_params(ctx)` overrides `vp`/`pt` with Zenvo Aurora values from the spec sheet
(`Zenvo_PublicAcademicProjects_VehicleModelParameters.xlsx`). **Baseline defaults unchanged** (DP2);
this is opt-in and is the recommended companion to `Hybrid='On'`.

**Chassis (direct drop-in / derived):** `m=1692`, sprung `mb=1517`, `muf=45`, `mur=55`, `md=75`;
`l=2.8`, track `t=1.74` (front; rear 1.67 — single-track model), `wB=0.57` (rear fraction = 1−0.43),
`hcg=0.45`; `A=2.0`; `Rw_f=0.35`, `Rw_r=0.37`; wheel rates `k_f=40000`, `k_r=50000`;
`I_x=500`, `I_y=2500`, `I_z=2900`; `brkB=0.65`; `Jw≈1.6` (front)/`1.8` (rear, single-`Jw` model
caveat); aero `Cd=0.32`, **`Cl=+0.15`** (sign-flipped: sheet −0.15 = downforce → MLTP positive).

**Caveats carried as comments (from the mapping doc):** single-`kt` model can't take the split
280k/330k tyre stiffness; no full Pacejka set (Static aero only, no AALB/active-aero map); roll-centre
heights, camber/toe, wing geometry keep framework defaults; `huf/hur/hw/d/hride/ksD` are
informational (no model consumer).

**Powertrain source ratings (the `hybrid` source list):**
- Front e-motors (`fl`,`fr`): `Pmax=150 kW`, `Tmax=143 Nm`, `OMmax=25000 rpm` (2618 rad/s),
  `gear=6.0`, `eff=0.90`.
- Rear e-motor (`r`): same ratings; `gear = 1.5 × ice_gear × 3.5` (sheet formula).
- ICE (`ice`): `gear = ice_gear × 3.5`; **torque curve = flagged placeholder** (`curve=…`,
  peak Nm/kW assumed for a hypercar) until the engine-map image is digitized; `eff_ice` placeholder.
- `ice_gear` config param (default 6th = 1.0). `Vmax` stays vehicle-level for state bounds.

> **Data gaps to be filled by the user:** ICE torque-map image (→ `poly` coeffs + peak Nm/kW),
> ICE efficiency, and (optionally) a chosen `ice_gear` per circuit. These are placeholders flagged
> in-code.

---

## 11. Invariants the refactor MUST preserve (from the codebase map)

1. **`nu == len(input_keys) == m.u rows == len(u_s) == len(u_min) == len(u_max) == len(ru) ==
   len(rdu) == len(rdu2) == len(duk_lb) == len(duk_ub) == warmstart u0 rows.`** Break one →
   CasADi dimension error or silent mis-scaling.
2. **Single ordering source of truth:** both `vehModel.u_list` and `userOpts._input_keys` derive
   from `control_keys(...)`. `delta` is always last; `T_brake` always immediately after source
   torques; split fractions occupy the legacy ATD slot (after brake).
3. **Column-major (`order='F'`) packing** in `transcription.py` is unchanged — new channels just
   extend `nu`.
4. **`nx` stays 23** (4 wheel-speed states at indices 5–8). Drive-source count is a control/param
   concern, never a state concern.
5. **Every `input_keys` name has a `_build_c` entry** (else `_col` raises) **and** a
   `warmstart_guesses` branch (else `u0` row dropped).
6. **`build_path_constraints` returns `(hnames,h,h_lb,h_ub)`** with `len(hnames)==h.shape[0]`, 4
   `rho_lim` first, bounds finite.
7. **Legacy `.mat`/plot contract:** `T_fl..T_rr`, `rho_lim_*`, `E_motor`, `input_keys`,
   `P_motor*/Om_motor*` prefixes preserved; new keys are additive; legacy `cfg` string byte-identical.
8. **Init 7-state model unchanged** (lumped single motor; 3-control vector).
9. `gg_plots.py` and the chassis/suspension/tyre/aero dynamics are **out of scope** — do not touch
   their channels.

---

## 12. Testing strategy

**New casadi-free tests** (`test_drive_sources.py` + extend `test_params_useropts.py`):
1. `build_topology` returns correct sources/splits per topology (`single`, `single`+ATD,
   `four_motor`, `hybrid`).
2. `control_keys` reproduces the three legacy `input_keys` lists **byte-for-byte**, and produces the
   new Hybrid list in the right order.
3. `_build_c`/`_col` resolve all Hybrid channels (`T_motor_r`, `T_ice_r`, `split_R`) — no
   `AttributeError`; legacy `duk_ub`/`rdu2` vectors unchanged positionally.
4. A casadi-free constraint **name generator** (extracted from `build_path_constraints`) yields the
   legacy names for `single`/`four_motor` and the new per-source names for `hybrid`.
5. Hybrid config-string + `.mat` round-trip (`test_save_load.py`): `data['Hybrid']`, `E_fuel`,
   per-source channels persist and reload.
6. `ACCEPTED_COLUMNS` gains `Hybrid`; `manifest_fieldnames` test updated (`test_sweep.py`).

**Smoke / regression (CasADi + IPOPT):**
7. EM4=0 and EM4=1 still **build** (`ctx.m23.nx==23`, `nu` matches `len(input_keys)`).
8. A short synthetic-circuit solve in `hybrid` mode converges (e.g. `Sturn`), producing finite
   `lap_time` and the new channels.

**Open issue (see §13):** `test_foundation.py` (81–86) and `test_params_useropts.py` (21–27)
already fail on this branch against the partially-applied Aurora baseline numbers.

---

## 13. Open issues / out of scope

- **Stale baseline tests (DP2 consequence).** Earlier commits half-applied Aurora numbers to the
  *baseline* `Powertrain.py`/`vehParams.py`, so `test_foundation`/`test_params_useropts` already
  fail. With Aurora moved to an opt-in profile, the baseline *should* be reverted to the original
  framework numbers so those tests pass, **or** the tests updated to the current baseline. **Recommended
  follow-up:** revert baseline `pt`/`vp` to original values and put all Zenvo numbers solely in
  `apply_aurora_params`. Flagged for decision at spec review; not blocking the powertrain refactor.
- **ICE torque-map image** not yet provided → placeholder curve. **Split tyre stiffness**
  (`kt_f`≠`kt_r`) and **split `Jw`** unsupported by the single-field model (mapping-doc caveats).
- **Active aero / AALB** for Aurora needs the DATA_AA polynomial map (not in the Zenvo data) →
  Static aero only.
- **Hybrid energy management** (battery SoC, fuel mass, thermal) explicitly out of scope (pure
  torque sources).

---

## 14. Build order (phased; each independently verifiable)

1. `functions/drive_sources.py` (`DriveSource`, `SplitPolicy`, `build_topology`, `control_keys`) +
   casadi-free tests. **No solver needed.**
2. Rewire `userOpts._input_keys` + `_build_c` + 3-way guard onto `drive_sources`; legacy configs
   reproduce identical keys (tests green).
3. Refactor `vehModel.py` symbol/split/signal loops (DP1 gear fix); smoke-test EM4=0/1 build.
4. Generalize `MLTP.build_path_constraints` + `warmstart_guesses`; EM4=0/1 full-solve regression.
5. Add `hybrid` topology end-to-end + `apply_aurora_params`; first Aurora solve on a synthetic
   circuit.
6. Save/plot/sweep/paramOptim wiring + `E_fuel` + back-compat checks.
7. (Follow-up, see §13) resolve baseline vs profile test status.
