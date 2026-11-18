# ChassisSim phase 2 plan: GP3 car physics and QSS

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** phase 2 of `docs/superpowers/specs/2026-11-18-chassissim-integration-design.md`. `MLTP(car='chassissim:<path>', circuit='chassissim:Eastern Creek', physics='corrected')` solves the GP3 2013 car from its own `.car` file. Exit: D19 gate 2 (fixed-line lap within 2% of ChassisSim). Backlog items 4 (aeromap), 6 (DRS), 7 (differentials), 9 (dampers), 10 (roll-centre migration) and the item 1 axle balance channel ship here.
**Architecture:** one new model module `vehModel_cs.py` (m23-compatible, nx 23 in this phase) built from `ctx.cs`. `vehModel(ctx)` dispatches to it as its first statement. Shared fields (mass, geometry, inertia, wheel radius) go to `vp` through `vp_overrides`; car-only data (tyre, aeromap, suspension curves, engine, diff) lives in `ctx.cs`. The corrected chassis equations move into one helper that both models call (snapshot gate none). `functions/chassissim.load_car` turns decoded records into `ctx.cs`.
**Tech stack:** Python, CasADi SX, numpy, scipy.
**Spec inputs (local, git-ignored, binding unless a later note corrects them):** `.superpowers/sdd/chassissim/map/p2_fms_seams.md` (seams, file:line, hardcoded 23/4EM list), `p2_equations.md` (equations, smooth replacements), `gp3_sheet.md` (decoded GP3 values, cid meanings, gaps), `gp3_sheet.py` (decoder scratch). Codec: `functions/carfile.py`. Teammate answers to the 15 cid questions arrive in the ledger.

## Order
T1 car record, T2 chassis helper, T3 `vehModel_cs` core, T4 car source wiring, T5 QSS envelope, V1 first real GP3 solve, T6 kinematics curves, T7 differentials, T8 aero corrections and DRS, T9 axle balance channel, V2 gate runs, T10 docs.

## Global constraints
- Phase 1 constraints hold (worktree, branch `chassissim-cars`, venv python, commit identity, no push, ponytail, STE, cavecrew receipts, no vendor data or vendor file names in git, synthetic tests with a SKIP vendor section, at most one solve at a time with a memory check, no tracked text that names files on the dev machine).
- Snapshot gate after every task: `.superpowers/sdd/chassissim/map/nlp_snapshot.py` prints `SNAPSHOT DIFF: none` (14 configs incl. corrected and corrected EM4). The 4EM car stays byte-identical.
- A ChassisSim car runs only with `physics='corrected'` (ValueError otherwise). It refuses options it cannot honour (ValueError, STE): `start_state='rolling'`, `state_scaling='typical'`, `atd_tbr`, `Electric_4Motors='On'`, `AeroConfig` other than 'Static', `between_knots` other than 'matlab', `collocation='radau'` (until tested), a 7-state init `warm_start`. ATD is forced Off with one warning. `ladder` is forced to 'qss23'.
- Smooth NLP: no `ca.if_else`, no `fabs`/`fmax`/`fmin` in SX; use the tools of `p2_equations.md` (smax, sabs, softplus, tanh, splines). Tables become splines once at car load (numpy/scipy), evaluated in SX by `ca.interpolant('bspline')` or polynomial.
- One real solve per new knob or model before a task closes (memory rule): Sturn or Eastern Creek, MUMPS, cold, one at a time, `free -g` available >= 4 GB.

## Rulings
- Ruling: GP3 Fx data is absent (tyre tables T7-T9 are zero). Fx uses the same peak force F'(Fz) as Fy and the lateral normalised curve shape, scaled by the peak slip ratio (tyre scalar 9), combined by the normalised slip vector (`p2_equations.md` 1.5). Cost if wrong: longitudinal grip off, small effect on a power-limited lap. Revisit when the teammate answers Q2.
- Ruling: brakes use cids 107 (force) and 108 (front bias) as decoded, until the teammate confirms (Q8). Cost if wrong: braking zones off by the bias error.
- Ruling: the GP3 tyre has no temperature, pressure or speed effect (tables flat). Phase 2 evaluates the load curve at the background temperature. Thermal states stay in phase 4.
- Ruling: car text readers are not needed for GP3 (aeromaps, hardpoints and tyre tables are inline). Skipped: add a reader when a car references a text file that the `.car` does not inline.
- Ruling: shared fields reach `vp` through `vp_overrides` (one source for code that reads `vp`). `ctx.cs` holds the rest. Cost if wrong: one more seam.
- Ruling: the chassis EoM is extracted once into a helper (`vehModel._chassis_corrected` or a small module) and both models call it; no copied block. Gate: snapshot none on all 14 configs.
- Ruling: `ctx.grip` (circuit base grip) multiplies the ChassisSim tyre peak force F' too.

---

### Task 1: Car record (`chassissim.load_car`)

**Files:** modify `functions/chassissim.py`. Tests: `tests/test_cs_car.py` (Haiku).
- [ ] `load_car(path) -> SimpleNamespace` from `carfile.decode(path)`: a module table `CAR_FIELDS = {cid: (name, unit factor to SI)}` for the consumed fields only (`gp3_sheet.md` section 2: mass block, ride heights 117-120, tyres 80-83, springs 1-6, bump rubbers 13-18, motion ratios 23-30, ARBs 19/21, dampers 31-35, aeromap records and axes, CLA/CDA/AB/offset, DRS records, engine curve 101/0x65, gears 0x6e, driveline efficiency 0x64, split 0x6f, diff types 0x70/0x71 and the custom map, brakes 107/108, hardpoints 49-52). Values in SI (mm -> m, deg -> rad, lbf/in -> N/m only where the record stores imperial, rpm -> rad/s for the engine axis).
- [ ] `car_vp(car) -> dict` of `vehParams` primaries for the shared fields (mass split, wheelbase, CG height and position, track, inertias, wheel radius, wheel inertia). Totals must reproduce the car mass and weight distribution (assert).
- [ ] Field names follow `gp3_sheet.md`. A cid the car lacks gives None, never a guess. Docstring STE.
- [ ] Tests: a synthetic car built with `carfile.encode` holding every consumed cid: SI conversions, `car_vp` totals. Vendor section: GP3 decodes, `car_vp` reproduces 650 kg and 0.41 front weight, spring rates equal 1200 / 800 lbf/in in N/m.

### Task 2: Chassis helper

**Files:** modify `vehModel.py`. Tests: extend `tests/test_physics_mass_frame.py` only if a check is needed.
- [ ] Move the corrected-physics chassis block (sf, dvx, dvy, dr, dn, deps, wheel spin dOm_*, heave/pitch/roll, unsprung, tyre deflection rows; `p2_fms_seams.md` section 2, vehModel.py ~625-690) into one function that takes the forces and moments and returns the state derivatives. The matlab branch stays where it is.
- [ ] Gate: snapshot none on all 14 configs (corrected and corrected EM4 included). No numeric change.

### Task 3: `vehModel_cs` core and the build path

**Files:** create `vehModel_cs.py`; modify `vehModel.py` (first statement dispatch only), `userOpts.py` (minimal car source), `MLTP.py` (path-row dispatch, channel guard). Tests: `tests/test_vehmodel_cs.py` (Haiku). Read first: `p2_fms_seams.md` (all), `p2_equations.md` 1, 3, 4, 5, `gp3_sheet.md` 2, `teammate_p2_answers.md`, `vehModel.py` (states 93-143, inputs 145-234, suspension 292-342, slip geometry 344-390, tyre outputs, `_chassis_corrected`).
- [ ] Build path (minimal, T4 adds the rest): `userOpts(car=None)`. A `'chassissim:<path>'` value sets `ctx.cs = chassissim.load_car(path)` and merges `chassissim.car_vp(ctx.cs)` under the user `vp_overrides`. It raises ValueError unless `physics='corrected'`. `ctx.cs = None` otherwise. `_input_keys` / `_build_c` give the car inputs `T_drive`, `T_brake`, `delta` with rate bounds and regulariser rows (zero regulariser like the motor; T_drive rate from the car engine torque scale, T_brake and delta as the 4EM rows). `vehModel(ctx)` first statement: `if getattr(ctx, "cs", None) is not None: return vehModel_cs(ctx)` (lazy import). `MLTP`: `cs_path_constraints` for a car at the single call site, no 4EM motor channels or energy block for a car (guard), and the ladder forced to 'qss23' (until T5 the build test uses the fake solver only).
- [ ] States: the 23 rows, order, names and scales of the 4EM model. Bounds from the car where they differ (wheel speeds from the car speed range and radius).
- [ ] Inputs: `T_drive` = total rear drive force [N], u_s = the largest gear-envelope force, bounds [0, 1]. `T_brake` = total brake force [N] with the 4EM sign (bounds [-1, 0]), u_s = car brake force (cid 107), split front/rear by the bias (cid 108), half per wheel, wheel torque = force x radius. `delta` as the 4EM car (u_s pi/8, delta = 35 deg x delta_n).
- [ ] Path rows in `cs_path_constraints(ca, m, ctx)`: `drive_env` = (F_env(v_r) - T_drive)/u_s in [0, inf], v_r = mean rear wheel speed x rear radius. `BrTh_1` as the 4EM row (no drive and brake together).
- [ ] Gear envelope (D7): F_env(v) = smooth max over the gears (log-sum-exp, error bound in `p2_equations.md` 5) of eff x T_e(w_e) x G_g / R with w_e = v x G_g / R. T_e = spline of the engine curve over [idle, limit]. Below idle the clutch holds w_e at idle. Above the limit a smooth window fades the torque to 0 (width about 2% of the limit). Ratios already hold the final drive. Open rear differential: T_rl = T_rr = T_drive x R_r / 2.
- [ ] Tyre (D4, Ruling Fx, teammate Q2): Fz_eff = softplus floor of Fz (20 N scale). F' = `ctx.grip` x spline(load -> peak force) at Fz_eff (the T1 record holds it at the background temperature). Camber factor = spline of the camber curve at the static camber (curves in T6). Normalised slip: sa_n = alpha / peak slip angle (per axle, rad), ss_n = slip ratio / peak slip ratio (tyre scalar 9), s = sqrt(sa_n^2 + ss_n^2 + 1e-6). h(s) = fn(s)/s as its own spline (fn = odd spline of the normalised slip table, duplicate abscissae removed). Fy = F' x camber factor x h(s) x sa_n, Fx = F' x h(s) x ss_n. mu_*_x = mu_*_y = F' / Fz_eff. Rolling drag = tyre scalar 7 x Fz. Slip angles, wheel speeds and slip ratios: the same geometry as `vehModel` (extract a helper only if the snapshot stays none, else copy with a one-line comment). Static toe from tyre scalar 0 (positive = toe-out) mapped to the 4EM toe convention.
- [ ] Aero (items 4 base): ride heights rh_f, rh_r = static ride height of the axle minus the mean suspension travel of its corners minus the change of tyre deflection (signs checked by the static test). CLA, CDA and AB maps: `ca.interpolant('bspline')` on the grid (X = rear ride height, Z = front, data index ix*nz + iz, axes in m) with a smooth clamp of both inputs at the grid edges (teammate Q4: ChassisSim clamps; the GP3 rear static height sits above the grid). q = 0.5 rho v^2. Lift L = q CLA, front share AB + offset, half per corner. Drag D = q CDA at the CG (no pitch moment: the car holds no drag height). No corrections, no DRS (T8).
- [ ] Suspension (item 9): wheel rates = spring rate x MR^2 (constant MR or the polynomial), preload so that the static spring force equals the static sprung corner load at the given ride heights (closed static state). Bars per C3 (F = k_w (xs_l - xs_r)). Bump rubbers: softplus beyond the static ground gap. Dampers: C2 spline through the origin built from (v, bump) for v > 0 and (-v, -rebound) for v < 0, damper velocity = MR x wheel travel rate, wheel force = MR x damper force. Static roll-centre heights from the hardpoints (front-view instant centres at static, numpy, once; T6 makes them curves). Jacking forces from those heights as the 4EM corrected model does.
- [ ] Chassis: `_chassis_corrected` (T2). Jw: vehParams default (the car holds none). f_side = 0, RW_Cd = 0.
- [ ] Outputs: the 4EM channel names (fx/fy/fz, mu_*, sa_*, sx_*, xs_*, T_*, Om_*, delta, zs, theta, phi, f_drag, f_lift, f_lift_*, dynamic_camber_*, ax, ay, Lon_acc, Lat_acc, sf) plus `m.veh_extra` = [(name, expr)] for engine speed and F_env. `m.TyreModel = 'CombinedSlip'`.
- [ ] Tests (Haiku): a synthetic car file (arbitrary values) through `MLTP(car=..., circuit=<synthetic>, physics='corrected', save=False, plot=False)` with the fake solver: sizes, names, input keys, path row names; f_dyn finite at the static point; left/right symmetry on a straight (vy = r = delta = 0 gives zero lateral forces and roll rate); static residual of the vertical rows small at the static ride heights; F_env positive and below the single-gear maximum; the default 4EM snapshot none. Vendor: GP3 builds the same way.

### Task 4: Car source wiring (rest)

**Files:** `userOpts.py`, `MLTP.py`, `functions/warmstart.py`, `functions/ladder.py`, `functions/importfile.py`, `functions/sensitivity.py` (guards). Tests: `tests/test_cs_wiring.py` (Haiku).
- [ ] Refusals of the global constraints (ValueError, STE), ATD forced Off with one warning, ladder forced to 'qss23' with one note. Knob `car_file` = absolute path plus `car.sha8`. Stem part `_car_<base name>` (slug) before the physics part, in `result_stem` and every caller.
- [ ] `("car", "4em")` in `WARM_KNOBS`; a source of another nx gives 'cold'. `ladder.seed_m23` / `quasi_static_states`: a `T_drive` seed and the car's static rows. `sensitivity.limit_prices` returns NaN for the motor prices of a car.
- [ ] Tests: refusals, stems and knobs, warm-start refusal across car sources, seeds finite. Snapshot none.

### Task 5: QSS envelope for a car

**Files:** modify `functions/ggv.py` (one dispatch at the top of `build_envelope`, one optional drive-force table in `Envelope` and `_make_accel`). Tests: `tests/test_cs_qss.py` (Haiku).
- [ ] `build_envelope_cs(ctx)`: tyre peak force vs load from the car (background temperature), aero at quasi-static ride heights (fixed point), load transfer from the car geometry and rates, drive force table = the gear envelope, brake force and bias. `march` and `qss_profile` unchanged. The default `_make_accel` closure stays verbatim (pinned QSS laps).
- [ ] Tests: synthetic car envelope pieces, QSS lap finite on a synthetic circuit, the 4EM pinned QSS laps unchanged. Vendor: GP3 QSS lap at Eastern Creek finite and within 15% of the `.ldx` lap.

### Validation V1 (Opus): first real GP3 solve
- [ ] `MLTP(car=<GP3 base>, circuit='chassissim:Eastern Creek', physics='corrected', linear_solver='mumps', save=True, results_dir=<scratch>)`, cold. Record status, iterations, lap, wall, RSS. If it fails, diagnose (systematic-debugging) before T6.

### Task 6: Kinematics curves (item 10)

**Files:** modify `functions/chassissim.py` (numpy solve), `vehModel_cs.py`. Tests: `tests/test_cs_kinematics.py`.
- [ ] Solve the double-wishbone geometry from the hardpoints (cids 49-52) once per car: camber, toe and roll-centre height vs wheel travel (and roll for the RC), fitted to low-order polynomials over the travel range. vehModel_cs uses them in place of the static values.
- [ ] Tests: a synthetic parallel-equal-arm geometry gives zero camber gain, a known swing-arm geometry gives its analytic camber gain and RC height. Vendor: GP3 curves finite and smooth over the travel range.

### Task 7: Differentials (item 7, D8)

**Files:** `vehModel_cs.py`, `functions/ggv.py` (QSS split). Tests: `tests/test_cs_diff.py`.
- [ ] The five types (`p2_equations.md` 6): open, super diff, spool, custom 2-D map (GP3 rear: one-way power-on LSD from its map), Detroit locker. Smooth laws only. Spool: equal wheel speeds by a stiff smooth coupling, not a state removal.
- [ ] Tests per type on a synthetic car: torque split at a speed difference, symmetry, the one-way behaviour of the GP3-like map.

### Task 8: Aero corrections and DRS (items 4, 6)

**Files:** `vehModel_cs.py`, `functions/ggv.py`, `userOpts.py` (DRS zones as a per-point track row only if a car enables DRS). Tests: `tests/test_cs_aero.py`.
- [ ] Sideslip, speed, steer and roll corrections (multiplicative on CLA/CDA, additive on AB), DRS deltas inside zones through a smooth window of s (`p2_equations.md` 3.3-3.4). All skipped when the car has them at zero (GP3), so the GP3 graph does not grow.
- [ ] Tests on a synthetic car with non-zero corrections and one DRS zone.

### Task 9: Axle balance channel (item 1, D25)

**Files:** `MLTP.py` (`veh_syms`), `plotSDI.py` (one trace). Tests: `tests/test_axle_balance.py`.
- [ ] `balance` = rear axle grip share minus front axle grip share (oversteer positive), from the friction usage of each corner, for any car. Diagnostic only, no NLP row.

### Validation V2 (Opus): gate runs
- [ ] Fixed line: GP3 base at Eastern Creek with `n_bounds=(-0.05, 0.05)` (ChassisSim drives the trace, corr -0.999). Lap within 2% of the `.ldx` lap of `gp3_baseline_easterncreek` (76.742 s). Free line (default band): lap <= the fixed-line lap. Overlay against `read_ld`: channel RMS reported (speed, ax, ay, steer).

### Task 10: Docs
- [ ] CLAUDE.md bullets (car source, vehModel_cs, QSS branch, kinematics, diffs, aero corrections and DRS, axle balance), tests list, CONTEXT.md terms if resolved. STE lint.
