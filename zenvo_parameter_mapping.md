# Zenvo Aurora → MLTP Framework Parameter Mapping

## Vehicle Data Source
Zenvo Aurora Tur | Michelin Pilot Sport Cup 2/2R tyres
275/30/R20 front | 335/30/R21 rear
ISO 8855 coordinate system

---

## CATEGORY 1: DIRECT DROP-IN (change the number, nothing else)

### Masses — vehParams.py

| Zenvo Parameter | Value | MLTP Variable | Current Value | Notes |
|---|---|---|---|---|
| Total vehicle mass (with driver, fuel, coolant, lubricant, luggage) | 1692 kg | `vp.m` via `vp.ms + vp.muf + vp.mur` | 2085 kg | Must be decomposed into sprung + unsprung |
| Driver mass | 75 kg | `vp.md` | 75 kg | Already matches |
| Unsprung mass front (both wheels combined) | 45 kg | `vp.muf` | 90 kg | Zenvo value is per-axle total |
| Unsprung mass rear (both wheels combined) | 55 kg | `vp.mur` | 100 kg | Zenvo value is per-axle total |
| Sprung mass (derived) | 1692 - 45 - 55 - 75 = 1517 kg | `vp.mb` | 1820 kg | mb = total - muf - mur - md |

### Geometry — vehParams.py

| Zenvo Parameter | Value | MLTP Variable | Current Value | Notes |
|---|---|---|---|---|
| Wheelbase | 2.8 m | `vp.l` | 3.0 m | Direct |
| Front track width | 1.74 m | `vp.t` | 1.8 m | Model uses single track width; rear is 1.67 m — use average ~1.705 m or front |
| Mass distribution front | 0.43 | `vp.wB` | 0.5 | NOTE: wB is rear fraction; wB = 1 - 0.43 = 0.57 |
| Semi-wheelbase front (l_f) | 1.596 m | `vp.l_f = vp.l*(1-vp.wB)` | 1.5 m | Auto-derived from wB and l |
| Semi-wheelbase rear (l_r) | 1.204 m | `vp.l_r = vp.l*vp.wB` | 1.5 m | Auto-derived from wB and l |
| CoM height (Road mode) | 0.45 m | `vp.hcg` | 0.5 m | Use road mode; track mode = 0.44 m |
| Frontal area | 2.0 m² | `vp.A` | 1.95 m² | Direct |

### Wheel & Tyre — vehParams.py

| Zenvo Parameter | Value | MLTP Variable | Current Value | Notes |
|---|---|---|---|---|
| Loaded radius front (at Fz=4300 N) | 0.35 m | `vp.Rw_f` | 0.355 m | Direct |
| Loaded radius rear (at Fz=5300 N) | 0.37 m | `vp.Rw_r` | 0.355 m | Direct; also update `vp.Rw` to whichever is used as nominal |
| Tyre vertical stiffness front | 280,000 N/m | `vp.kt` | 300,000 N/m | Model uses single kt; must choose or average |
| Tyre vertical stiffness rear | 330,000 N/m | (no separate `vp.kt_r`) | 300,000 N/m | Framework has one kt — split-axle not currently supported |
| Wheel rotational inertia front | 1.6 kg·m² | `vp.Jw` | 3.6 kg·m² | Model uses single Jw; current = 0.9*2*2 (2 motors * 2 axles). Zenvo: per corner ~1.6 kg·m² front, 1.8 rear |

### Suspension — vehParams.py

| Zenvo Parameter | Value | MLTP Variable | Current Value | Notes |
|---|---|---|---|---|
| Wheel rate front | 40,000 N/m | `vp.k_fl`, `vp.k_fr` | 75,000 N/m | Direct (wheel rate, not spring rate) |
| Wheel rate rear | 50,000 N/m | `vp.k_rl`, `vp.k_rr` | 80,000 N/m | Direct (wheel rate, not spring rate) |

### Inertias — vehParams.py

| Zenvo Parameter | Value | MLTP Variable | Current Value | Notes |
|---|---|---|---|---|
| Ixx (Roll) | 500 kg·m² | `vp.I_x` | 1000 kg·m² | Direct |
| Iyy (Pitch) | 2500 kg·m² | `vp.I_y` | 1600 kg·m² | Direct |
| Izz (Yaw) | 2900 kg·m² | `vp.I_z` | 1960 kg·m² | Direct |

### Aerodynamics — vehParams.py (via DATA_AA.mat nominally, or direct override)

| Zenvo Parameter | Value | MLTP Variable | Current Value | Notes |
|---|---|---|---|---|
| Drag coefficient Cd | 0.32 | `vp.Cd0` (from DATA_AA aero.veh.Cd0) | 0.75 (placeholder) | Zenvo Cd is non-dimensional; MLTP uses Cd * ρ * A * v² / 2 |
| Lift coefficient CL | -0.15 | `vp.Cl` = `Cl0_front + Cl0_rear` | 1.45 (placeholder) | Zenvo sign: -ve = downforce; MLTP convention: positive Cl produces downforce (f_lift negative). Negate: Cl = +0.15 or split front/rear. CRITICAL sign check required |

### Brakes — vehParams.py

| Zenvo Parameter | Value | MLTP Variable | Current Value | Notes |
|---|---|---|---|---|
| Mechanical pressure bias front | 0.65 | `vp.brkB` | 0.6766 | Direct; brkB = fraction to front |

---

## CATEGORY 2: REQUIRES CALCULATION/DERIVATION BEFORE USE

### Powertrain — Powertrain.py

| Zenvo Parameter | Raw Data | Derivation | MLTP Variable |
|---|---|---|---|
| e-Motor power (each of 3) | 150 kW peak (from power map: [0 10000 25000] rpm → [0 150 150] kW) | Pt.Pmax = 3 × 150,000 = 450,000 W total (or per-motor if EM4=1: 150,000 W each front, 150,000 W rear) | `pt.Pmax` (currently 456,000 W for 2 × 228 kW) |
| e-Motor max speed | 25,000 rpm (from power map x-axis end) | pt.OMmax = 25,000 × π/30 = 2618.0 rad/s | `pt.OMmax` (currently 17,750 rpm → 1858.8 rad/s) |
| e-Motor peak torque (each) | Derived: T = P/ω at low speed. At 10,000 rpm (1047 rad/s): T = 150,000/1047 = 143 Nm per motor | pt.Tmax = 143 Nm per motor (or 143×3=429 Nm total if single-motor model) | `pt.Tmax` (currently 2×301=602 Nm) |
| e-Motor efficiency | 84–96% | Use midpoint or worst-case: pt.eff ≈ 0.90 | `pt.eff` (currently 0.90 — already matches) |
| Top speed (to derive gear) | Not given directly in sheet | Use track-mode top speed. Must be assumed or sourced externally. Current: 290 km/h | `pt.Vmax` |

### Transmission & Gear Ratio — Powertrain.py / vehParams.py

| Zenvo Parameter | Raw Data | Derivation | MLTP Variable |
|---|---|---|---|
| Front e-motor transmission ratio | 6:1 | For EM4=1: gear_f = 6. Motor wheel speed = Om_wheel × 6 | `vp.gear` (single gear in model) |
| Rear e-motor transmission ratio | 1.5 × gear_ratio × final_drive. At e.g. gear 4 (1.4) × 3.5 final = 7.35; × 1.5 = 11.025 | Complex: rear motor sees additional ICE gearbox and final drive. The 1.5 factor likely reflects a fixed planetary. In pure EV mode on rear axle: vp.gear_r = 1.5 × g_ICE × 3.5 | Framework has single `vp.gear`; would need split-axle extension for Zenvo's hybrid topology |
| ICE gearbox ratios (8 gears) | 3.4, 2.35, 1.75, 1.4, 1.2, 1.0, 0.9, 0.8 | For OCP: pick a representative gear (e.g. gear 6 = 1.0, or gear 5 = 1.2 for mixed-speed circuits). The single-gear MLTP does not shift. Effective total ratio = g_ICE × 3.5 | Not directly in framework; factored into `vp.gear` |
| Final drive ratio | 3.5 | Combined with gearbox: overall ratio = g_ICE × 3.5. In gear 6: 1.0 × 3.5 = 3.5 | Folded into `vp.gear` |

### Torque Distribution — vehParams.py

| Zenvo Parameter | Raw Data | Derivation | MLTP Variable |
|---|---|---|---|
| Torque distribution front/rear | 3 motors: 2 front (150 kW each) + 1 rear (150 kW) → front: 2/3, rear: 1/3 by motor count; but rear also has ICE | If electric-only: Tdist (rear fraction) ≈ 0.33. With ICE adding rear torque, effective Tdist shifts rear-heavy. Must decide on ICE contribution | `vp.Tdist` (currently 0.7281 rear) |

### Brake Torque Capacity — vehParams.py

| Zenvo Parameter | Raw Data | Derivation | MLTP Variable |
|---|---|---|---|
| Front disc OD/ID | 410 / 230 mm | Effective radius r_eff ≈ (0.205 + 0.115)/2 = 0.160 m | — |
| Front caliper pistons | 6 pistons: 4 × Ø36 mm + 2 × Ø40 mm | Piston area = 4×π(18)² + 2×π(20)² = 4072 + 2513 = 6585 mm² | — |
| Front pad friction | μ_pad = 0.3 | With booster ratio 5.1, pedal ratio 3:1, master cyl Ø25 mm: hydraulic pressure P = F_pedal × 3 × 5.1 / (π × 12.5²). At realistic pedal force ~150 N: P ≈ 150×15.3/490 = 4.7 MPa. T_brake_f = P × A_piston × μ_pad × r_eff × n_calipers | Detailed brake model. For vp.Tbrake_max: sum front+rear at peak hydraulic pressure. Rough estimate: 4.7 MPa × 6585 mm² × 0.3 × 0.160 m × 2 ≈ 2800 Nm front. Rear similar. Total: ~5000–6000 Nm. `vp.Tbrake_max` currently 4000 Nm |
| Rear disc OD/ID | 400 / 270 mm | r_eff ≈ (0.200 + 0.135)/2 = 0.1675 m | — |
| Rear caliper pistons | 6: 4 × Ø28 mm + 2 × Ø35 mm | A_piston = 4×π(14)² + 2×π(17.5)² = 2463 + 1924 = 4387 mm² | — |
| Brake balance (mechanical) | 65% front | `vp.brkB = 0.65` | `vp.brkB` (currently 0.6766) |

### Suspension Damping — vehParams.py

| Zenvo Parameter | Raw Data | Derivation | MLTP Variable |
|---|---|---|---|
| Suspension travel (bump/rebound) | Front: 70/60 mm road, 55/75 mm track. Rear: 80/70 mm road, 65/85 mm track | Not used directly — travel limits state bounds only. The framework uses zeta (damping ratio) instead | `vp.zeta_fl/fr/rl/rr` |
| Motion ratio | Front: 1.5, Rear: 1.5 | Spring rate = Wheel_rate × motion_ratio² = 40,000 × 1.5² = 90,000 N/m front; rear = 112,500 N/m. This is informational (spring rate behind the rocker); the model uses wheel rate directly | Wheel rate already correct; motion ratio not used directly in model |
| Damping ratio zeta | Not given | Must assume. Typical sports car: 0.6–0.8. Framework default: 0.7 for all corners — reasonable assumption | `vp.zeta_fl/fr/rl/rr` = 0.7 (keep current) |

### Tyre Nominal Load — vehParams.py

| Zenvo Parameter | Raw Data | Derivation | MLTP Variable |
|---|---|---|---|
| Static front axle load | m × g × wB_front = 1692 × 9.81 × 0.43 = 7129 N total → 3565 N per front corner | Matches given Fz=4300 N (loaded radius reference). Difference: 4300 N includes some downforce at speed | `vp.Fz0` (currently 4905 N, based on current heavier vehicle) |
| Static rear axle load | 1692 × 9.81 × 0.57 = 9454 N total → 4727 N per rear corner | Zenvo rear tyre reference: Fz=5300 N (includes downforce) | Framework uses single `vp.Fz0`; Zenvo has different front/rear |

---

## CATEGORY 3: CONTEXT-DEPENDENT / FRAMEWORK EXTENSION NEEDED

### Powertrain Configuration — userOpts.py

| Zenvo Configuration | Framework Mapping | Required Action |
|---|---|---|
| 3 electric motors (2 front + 1 rear) + ICE rear | `Electric_4Motors="Off"` gives 1 rear motor + brake; `"On"` gives 4 corner motors. Zenvo's 2F+1R is neither | Closest approximation: `Electric_4Motors="On"` with EM4=1 and zero-torque on T_motor_rr (right rear, which has no dedicated e-motor). Or: use single-motor model with effective combined output |
| ICE (engine) with 8-speed gearbox driving rear | Framework has no ICE model. Engine torque map referenced but not provided numerically | Use the e-motor model only and disable ICE contribution, OR add ICE as additional torque source on rear axle |
| ATD (Active Torque Distribution) | `ATD="On"` enables per-corner fractions | Zenvo's 2 front independent + 1 rear effectively has front-corner torque vectoring already from independent front motors |

### Aerodynamics — DATA_AA.mat structure

| Zenvo Data | Framework Need | Gap |
|---|---|---|
| Cd = 0.32, CL = -0.15 (total, undivided) | Framework needs front/rear Cl split: `Cl0_front`, `Cl0_rear`, `Cl0_left`, `Cl0_right` | Zenvo gives only total CL. Assume 40/60 front-rear split as approximation: Cl0_front ≈ 0.06, Cl0_rear ≈ 0.09 (note: inverted — downforce positive in MLTP). Side force coefficients `Cs0_front/rear` not provided — set to 0. Wing polynomial gains (FW_L, FW_R, RW, TW) not provided — no AALB possible |
| Aerodynamic map "not yet available" | Full DATA_AA.mat polynomial coefficients for wing surface dependencies | Cannot run AALB or Active aero modes without this. Static aero mode only possible with current Zenvo data |
| Plan view area 9 m² | Not used in current MLTP formulation | Informational |

### Tyre Model — vehParams.py (ctx.mf)

| Zenvo Data | Framework Need | Gap |
|---|---|---|
| Michelin PSC2/2R identified | Full Pacejka 5.2 MF coefficient set (~60 parameters per tyre) | Zenvo provides only: cornering stiffness (150,000 N/rad front, 180,000 N/rad rear), linear range slip angle (3.5°), peak slip angle (6.0°). These are insufficient to populate the MF 5.2 model. Pacejka coefficients must be sourced from Michelin, measured on a flat-trac, or estimated via published PSC2 literature |
| Cornering stiffness front | 150,000 N/rad | Can be used to constrain/validate pKy1-pKy2 combination: Ky = pKy1 × Fz0 × sin(2 arctan(Fz0/(pKy2 × Fz0))) | Cross-check to current mf coefficients |
| Peak slip angle | ~6° (0.1047 rad) | Constrains location of Fy peak; informs pKy3/pEy parameters | Validation only |
| Simplified tyre model (7-state warm start) | mu, bx, cx, ex, by, cy, ey (8 params) | Cornering stiffness and peak force allow rough estimation of by, cy, mu: by ≈ Cs/(mu × Fz0) | `vp.tyre` can be approximated |

### Roll Centre Heights — vehParams.py

| Zenvo Data | Framework Parameter | Gap |
|---|---|---|
| Not provided | `vp.hRCf` = 0.07 m, `vp.hRCr` = 0.11 m | Must retain framework defaults or estimate from suspension geometry (double-wishbone typical: RC 30–100 mm above ground). Zenvo data does not include kinematic data |

### Unsprung Mass CoG Heights — vehParams.py

| Zenvo Data | Framework Parameter | Gap |
|---|---|---|
| Not provided | `vp.huf` = 0.2968771 m, `vp.hur` = 0.2968771 m | Geometric estimate: hub centre ≈ loaded radius ≈ 0.35/0.37 m from ground. Use huf ≈ 0.35, hur ≈ 0.37 |

### Rear Wing & Active Aero Geometry — vehParams.py

| Zenvo Data | Framework Parameter | Gap |
|---|---|---|
| Not provided | `vp.hw` = 1.28 m (rear wing height) | Must be estimated from photographs/CAD. Aurora is a hypercar — wing height ~1.0–1.3 m |

---

## CATEGORY 4: PARAMETERS WITH NO ZENVO EQUIVALENT (keep framework defaults)

| MLTP Parameter | Value | Reason no Zenvo equivalent |
|---|---|---|
| `vp.ksD` = 0.4620 | Roll stiffness distribution (rear fraction) | Not provided by Zenvo; lumped roll stiffness block inactive in 23-state model |
| `vp.f` = 0.01 | Rolling resistance coefficient | Not provided; 0.01 is standard for high-performance tyres |
| `vp.hride` = 0.117 m | Ride height | Not provided; must estimate |
| `vp.gamma_*` = 0.0° | Camber angles | Not provided; set up camber separately if known |
| `vp.toe_*` = 0.0° | Toe angles | Not provided |
| `ctx.cg.*` | Camber-gain coefficients | Requires full suspension kinematic model |
| Pacejka MF 5.2 coefficients (all ~60) | Full tyre model | Not provided by Zenvo |
| `vp.alpha_FL/FR/RW/TW` | Wing AoA settings | Not provided; no aero map available |
| `vp.Cs0_front/rear` | Side force coefficients | Not provided |

---

## SUMMARY: MAPPING STATUS

| Category | Count | Status |
|---|---|---|
| Direct drop-in replacements | 17 | Update numbers in vehParams.py |
| Requires calculation | 11 | Derive from Zenvo raw data |
| Requires assumptions/extensions | 8 | Use approximations or flag as unknown |
| No Zenvo data (keep defaults) | 10 | Retain framework values |

### Critical Notes
1. **Sign convention on CL**: Zenvo uses +ve = lift, -ve = downforce. MLTP `f_lift = -½ρCl_fl·A·vx²` — negative force acts downward (downforce). Zenvo CL = -0.15 → MLTP Cl = +0.15.
2. **wB convention**: Zenvo gives front mass fraction (0.43). MLTP `wB` is the rear fraction: `vp.wB = 1 - 0.43 = 0.57`.
3. **Wheel vs spring rate**: Zenvo gives wheel rates (40/50 kN/m) directly. Do NOT divide by motion ratio² again — the model uses wheel rates.
4. **Front vs rear tyre radius**: MLTP currently uses single `vp.Rw`. For Aurora (Rw_f=0.35, Rw_r=0.37), both `vp.Rw_f` and `vp.Rw_r` must be set independently.
5. **No full aero map**: AALB and active aero modes require DATA_AA polynomial coefficients not present in Zenvo data.
6. **Hybrid topology**: Aurora's 2F+1R e-motor + ICE rear is not directly representable in the current EM4/single-motor switch. Closest: EM4=0 (single aggregated motor, rear-biased) or custom torque-split.
