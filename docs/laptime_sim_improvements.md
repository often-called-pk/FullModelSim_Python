# Lap-time simulation: improvement list

Compiled 17 October 2026 from `docs/lap_time_simulation_literature.md` and a review of the code.

## Scope and method

- **Sources.** The literature review is `docs/lap_time_simulation_literature.md`. "Section N" below refers to its sections.
- **Code.** The review read branch `worktree-speed-phase1`, checked out at `.claude/worktrees/speed-phase1`. All line numbers refer to that branch. The MATLAB original is `IRP/FullModel_4EM_Suspension_FullTyre`.
- **Method.** Six readers extracted 347 facts from the review and the code. Five proposers wrote 56 items, which merged into 38. Each item then had three independent checks: support in the literature, accuracy against the code, and value for this kind of solver.
- **Evidence.** No NLP solve ran during the review. The numbers come from stored results, stored multipliers and the QSS screener. Treat them as estimates until an NLP run checks them.
- **Outcome.** 21 items passed all three checks, 11 passed two, and 6 failed two. This list keeps 30 items, some of them merged or narrowed. The section "Skipped for now" holds the rest, each with a trigger.

## Status on 19 October 2026

Branch `sim-phase2` holds the second batch of items. The batch contains D1 (stage S), F1, E1 (stage S), B1, B2, C1, C2 (stage S) and the port gap of C6. Each item is opt-in or changes only outputs. The default NLP stays byte-identical in seven test builds, and the parity gates stay green. All 40 test files pass on Linux.

The runs below used MUMPS on Linux. The column "Seed" names the warm start. A lap change compares with the stored baseline: Sturn 18.01633 s, BCN 116.44407 s.

| Run | Case | Seed | Result |
|---|---|---|---|
| V0 | Sturn, defaults | Cold | 18.01857 s, 152 iterations, the same as the first batch |
| V1 | Sturn, `start_state='rolling'` | Stored Sturn result | 18.08490 s, 95 iterations, +0.069 s (first-order estimate +0.045 s) |
| V1c | Sturn, `start_state='rolling'` | Cold | Before the fix: 18.61105 s, 530 iterations, a worse local optimum. After the fix: 18.06169 s, 149 iterations, +0.045 s (first-order estimate +0.045 s). |
| V2 | BCN, `start_state='rolling'` | Stored BCN result | 116.72927 s, 118 iterations, +0.285 s (first-order estimate +0.197 s) |
| V3a | Sturn, `physics='corrected'`, vi = 70 m/s | Cold | 20.18902 s, 404 iterations, body height at s = 0 on the -10 mm box edge |
| V3b | V3a with `start_state='rolling'` | Cold | No convergence in 6000 iterations |
| V3c | V3a with `start_state='rolling'` | The V3a result | No convergence in 2000 iterations |
| V4 | Circle, `start_state='periodic'` | Cold | 21.71527 s, 65 iterations, end state equal to the start state |
| V5 | BCN, `start_state='periodic'` | Stored BCN result | 116.91133 s, 74 iterations, +0.467 s (forecast about +0.3 s) |
| V6 | Sturn, `n_bounds=4.2` | Stored Sturn result with duals | 17.97682 s, 6 iterations. The objective changes by -0.0401 s. The price estimate is -0.0405 s. |
| V7 | BCN, `n_bounds=4.1` | Stored BCN result with duals | 116.26122 s, 17 iterations. The objective changes by -0.1841 s. The price estimate is -0.1861 s. |
| V8 | Sturn TyreOptim with the Fz0 coupling | Stored Sturn result with duals | 17.49913 s, 51 iterations, Fz0_shift 0.787 |

Findings:

- **Rolling start (B1), cold seed.** A seeded rolling start works with the default physics (V1, V2). B1 uses the equilibrium of the model itself, not `ladder.quasi_static_states`. Before the fix, the cold seed set rows 9-22 (body, unsprung and tyre states) to static values without aero, not to the rolling-start equilibrium. So the cold V1c took 530 iterations to a worse local optimum. The fix sets rows 9-22 of the cold seed to the start equilibrium at all knots and collocation points. Now V1c takes 149 iterations, and its lap is 0.023 s shorter than the lap of the seeded V1.
- **Rolling start (B1), start speed.** With `physics='corrected'` at 60 m/s, a cold rolling start takes 220 iterations to 20.03744 s (before the fix: 270 iterations, 20.02972 s). At 70 m/s, the rolling start is infeasible as posed (V3b, V3c), and the solver and the transcription have no bug. Two solves, from two seeds, found the highest start speed with every other start state in the rolling box: 68.50 m/s and 68.53 m/s. The vx start box at 70 m/s has its lower edge at 69 m/s. When only this box widens to 59-71 m/s, the cold V3b solve converges in 348 iterations, with a start speed of 65.475 m/s.
- **Braking-limit warning.** `MLTP` now warns when the vx box of a rolling start has its lower edge above the QSS braking limit at s = 0. On the Sturn centreline, this limit is 66.87 m/s with `physics='corrected'` and 70.79 m/s with the default physics. The QSS limit is about 1.6 m/s below the highest start speed that the NLP found. So the warning can also fire for a feasible start. The solve still runs after the warning.
- **Start limit.** With `physics='corrected'`, a start speed above 62 m/s does not make the start infeasible (V3a). The body height stays at the edge of the start box instead. At 70 m/s, the solve with the default start box converges only from a pre-braked start away from the equilibrium. At s = 0, the rear wheels lock, and the unsprung states sit on their bounds.
- **Periodic lap (B2).** The periodic rows hold exactly (V4, V5). The BCN curvature does not close the lap (-1.023 laps), so treat the +0.467 s of V5 as provisional until a closed BCN track exists.
- **Width prices (C1, C2).** The first-order prices predict the change of the objective within 1.1% (V6, V7). One metre of half-width is worth about 1.86 s on BCN.
- **TyreOptim (C6).** The fix moves the Sturn optimum from the 0.5 bound (15.750 s) to an interior point (17.499 s at Fz0_shift 0.787). A second solve that started at Fz0_shift 0.53 ended on a worse optimum (21.832 s).

Open items:

- B1: the refine reseed (`warmstart_refined`) still sets rows 9-22 to static values. Change it when a rolling start runs with refine. No solve checks a rolling start with the qss23 ladder. Only Sturn solves check the braking-limit warning. The fix does not clip the ends of the cold seed to the start box. A clip gives no measured gain, and the proposed form changes the default seed.
- C6 part 2, the Magic Formula scaling factors, waits for the owner's sign-off, together with E3.
- Stage M of D1, E1 and C2 is not built.
- BCN keeps its original file for the parity gates, so it loads with one QA warning.

## Status on 18 October 2026

Items A1 to A6 are built on branch `physics-fixes`, behind the opt-in flag `physics='corrected'`. The default `physics='matlab'` keeps the MATLAB behaviour bit for bit, and the parity gates stay green. All 33 test files pass on Linux.

| Circuit | `physics='matlab'` | `physics='corrected'` | Change | QSS forecast |
|---|---|---|---|---|
| Sturn | 18.019 s, 152 iterations | 20.024 s, 218 iterations | +11.1% | +10.0% |
| BCN | 116.444 s, 207 iterations | 130.035 s, 179 iterations | +11.7% | +11.1% |

The runs used MUMPS on Linux, from a cold start. The default BCN lap matches the Windows ma57 baseline within 0.2 ms.

Two limits remain:

- With `physics='corrected'` the body sits 9.4 mm low at 60 m/s, and the start box allows 10 mm. This limit is softer than this list first said. At 70 m/s the solve converges, but the body height at s = 0 stays at the edge of the box. So the car starts away from its equilibrium (run V3a in the status of 19 October). The rolling start of item B1 is infeasible in this case (runs V3b and V3c, see the status of 19 October).
- A5 changes only the motor speed. The per-motor ratings, the gear as a setup field and regenerative braking are still open. A4 has no drag pitch moment yet.

## Read this first: parity is not physics

The MATLAB validation of 7 and 8 October 2026 proved port fidelity. The parameters, the NLP functions and the laps agree within 0.01%. That validation did not check the physics of the original.

This review found several inherited MATLAB behaviours that move the lap more than any setup change. The largest is a mass bookkeeping error. The QSS estimate puts it at about 10% of lap time. Section A lists these behaviours.

Any fix in Section A moves the lap away from the MATLAB reference. Put these fixes behind one opt-in flag, for example `physics='matlab' | 'corrected'`, with `'matlab'` as the default. The parity gates then stay valid for the default. Build new reference results with the flag on, one solve at a time.

## Order of work

1. **D1, track QA gate.** Jarama runs on an unusable curvature file today, and the check costs hours.
2. **E1, output channels.** The stored lateral acceleration and the energy unit are wrong. The fix does not touch the NLP.
3. **A1 to A3, physics fixes behind the flag.** These are the largest lap-accuracy items. They need an owner decision, because they change inherited MATLAB behaviour.
4. **B1 and B2, start and boundary states.** They remove a start bias of about 0.2 s on BCN and the dependence on an arbitrary start speed.
5. **C1 and C2, track width and limit prices.** Every track is 8 m wide today, and the width is worth about 1.6% of the BCN lap.
6. **C6, TyreOptim port gap.** The Python tyre co-optimisation does not do what MATLAB does.

Effort: S = hours, M = days, L = weeks. Gains are QSS or first-order estimates unless the text says otherwise.

## A. Model physics (inherited from MATLAB, fix behind the flag)

### A1. Mass and load bookkeeping (S code, M validation)

- **Problem.** Each unsprung corner carries the total unsprung mass of 190 kg (`vehModel.py:552-555`). The planar equations divide by the sprung mass of 1895 kg, not by the total of 2085 kg (`vehModel.py:531-534`). The static tyre loads sum to 26.05 kN against m g = 20.45 kN. MATLAB does the same (`vehModel.m:1034-1037`).
- **Change.** Use `muf/2` and `mur/2` per corner, and divide the planar equations by `m`. Apply the same load basis in the QSS screener (`functions/ggv.py`).
- **Gain.** The load-sensitive tyre turns the extra 27% of load into grip. The QSS estimate makes the lap about 10% slower: Sturn +10.0%, BCN +11.0%, NBR +10.3%.
- **Why.** Section 5 states that load sensitivity sets the cost of load transfer. Section 10, check 1, asks for this check.
- **Risk.** The fix breaks the parity pins unless the flag gates it. All results need new baselines under the flag.

### A2. Corner suspension: damper formula, spring path, smooth stops (M)

- **Problem.** The damper rate is `c = zeta * 2 * sqrt(m_eff/2) * k` (`vehParams.py:150-153`, from `vehParams.m:84`). The stiffness sits outside the square root, so `c` is about 2.4e6 N s/m. For a damping ratio of 0.7 the value must be about 8.3e3 to 8.8e3 N s/m. The effective damping ratio is about 200, so the dampers lock the suspension.
- **Problem.** The spring acts from the body to the ground (`vehModel.py:245-248`), while the damper acts from the body to the wheel. The travel limits are hard switches (`vehModel.py:250-261`).
- **Change.** Use `c = 2 * zeta * sqrt(k * m_corner)` with the sprung corner mass. Connect the spring to the wheel. Replace the hard limits with smooth bump stops. Do A1 first, because the wheel-hop mode needs the true unsprung mass.
- **Gain.** The ride and load-transfer dynamics become physical. With A1 in place, the stiffest mode falls from 1.77e4 to about 150 1/s, which helps the collocation.
- **Why.** Section 2.1 gives Nowlan's damper guide: omega_0 = sqrt(K/m) and zeta = C / (2 m omega_0), in strict SI units.
- **Risk.** Convergence can change. A softer car moves more, so the smooth stops are necessary.

### A3. Frame-consistent chassis equations (S)

- **Problem.** On a flat road the planar equations add gravity from body pitch and roll, and the heave equation adds body-frame rate terms (`vehModel.py:531-534`, `545-546`). The tyre forces stay in the road plane, so the equations mix two frames. MATLAB does the same (`vehModel.m:1014-1015`, `1028`).
- **Change.** Remove these terms behind the flag. Keep a place for grade and bank terms for a later 3D road.
- **Gain.** The QSS estimate makes the lap about 0.8% to 1.0% faster.
- **Why.** Section 3.2: roll and pitch act through the tyre loads, not as planar forces.
- **Risk.** No run measured the effect of the rate terms yet. Judge the change against a fine reference.

### A4. Aero load path through the springs (S for stage 1)

- **Problem.** Downforce enters each corner suspension force, and the heave equation cancels it (`vehModel.py:254-257`, `545`). The springs never carry downforce, and front downforce pitches the nose up. The axles take the aero load by `l_r/l`, whatever the front and rear lift split is.
- **Change, stage 1.** Send the downforce through the springs. Add the aero pitch moment and the drag pitch moment about the CG.
- **Gain.** The aero balance becomes live, and about 17% of the downforce moves to the front axle. The QSS lap moves +0.03% on Sturn and +0.13% on BCN. The wing-angle co-optimisation and the active aero become physical.
- **Status.** The checks rejected the first version of this item. The ride-height aeromap stage is now in "Skipped for now".

### A5. Four-motor powertrain (M)

- **Problem.** With four motors, the model divides the wheel speed by the gear ratio to get the motor speed (`vehModel.py:514-515`). The single-motor model multiplies by it (`vehModel.py:496`). So the per-motor power and speed rows are 52.6 times too loose. Each motor also gets the full 602 Nm and 456 kW.
- **Change.** Add a per-motor model behind the flag: speed times gear, per-motor ratings, `gear` as a setup field and an optional pack power row. Regenerative braking also needs a new upper bound on the power rows (`MLTP.py:93-96`).
- **Gain.** The four-motor car becomes power limited. The QSS BCN lap moves +0.13% to +7.6%, depending on the ratings that the owner picks.
- **Risk.** The four-motor NLP is already fragile and needs the friction homotopy.

### A6. Wheel kinematics (S)

- **Problem.** The rear-right slip angle uses the front toe (`vehModel.py:281`). The Ackermann front wheels use the rear toe (`vehModel.py:276-277`). MATLAB has the same indices. `MLTP` cannot reach the CamberGain or Steering options, and the camber `'Table'` branch has no table.
- **Change.** Fix the indices, forward the two options, and port or remove the table branch.
- **Gain.** Every setup with different front and rear toe gets the right slip angles. Nothing changes at the default toe of 0.

## B. Boundary conditions and transcription

### B1. Rolling start state (S)

- **Problem.** The wheel speeds, unsprung velocities and tyre deflections are free at s = 0 (`userOpts.py:277-279`). The optimiser starts every lap with all four wheels at the speed bound, at a slip ratio of 0.32.
- **Change.** Add `start_state='rolling'`, which fills these states from `ladder.quasi_static_states`. Also reduce the start boxes on pitch and roll to physical sizes.
- **Gain.** It removes about 0.06 s of start bonus on BCN. The total start bias on BCN is about 0.2 s, and the pitch and roll boxes hold most of the rest.

### B2. Periodic flying-lap conditions (S)

- **Problem.** The start speed sits on the edge of the box 60 +- 1 m/s on every circuit (`userOpts.py:277`, `functions/transcription.py:301-310`). The edge is worth 0.070 s per m/s on BCN, so the lap depends on an arbitrary start speed.
- **Change.** Add opt-in periodic rows X(0) = X(N) for closed circuits.
- **Gain.** The BCN lap moves by about 0.1 s to 0.4 s, and it no longer depends on the start speed.
- **Dissent.** The literature review has no source on boundary conditions. The code and value checks passed.

### B3. Inputs between knots and the power row (S code, M validation)

- **Problem.** Inside each interval the inputs hardly move (`functions/transcription.py:321-325`, as in `MLTP.m`). The Gauss-point input uses a per-metre slope times a fraction from 0 to 1. The motor power row applies only at the knots. At the Gauss points the dynamics reach 1.38 times the power limit.
- **Change.** First export the inputs that the dynamics saw. Then add opt-in true linear inputs and the power row at the Gauss points.
- **Gain.** The default BCN lap is probably 0.1% to 0.2% too fast today. That is the same size as the production mesh error.

### B4. Knot ringing (S, then M)

- **Problem.** Gauss-Legendre collocation is not L-stable. Unresolved fast modes swing from knot to knot. The stored BCN rear-left tyre load reads 13.3, 2.1, 1.2 and 5.8 kN at four adjacent knots.
- **Change, S.** Store a second channel set at the collocation points, where the solution is accurate.
- **Change, M.** Add an opt-in Radau IIA scheme. CasADi already supplies the points.
- **Risk.** Radau puts duplicate points into `s_full`, which can break code that expects a strictly increasing grid.

### B5. Continuation for single off-default solves (S-M)

- **Problem.** A cold solve at a non-default setup can land on a slow branch. On Sturn such solves were 0.47 s to 1.93 s slower than a solve continued from the default.
- **Change.** Add `MLTP(continue_from='auto')`, which warm-starts with duals from the stored default result. It needs E2, so that the source result is unambiguous.
- **Gain.** No wrong-branch laps, and an estimated four times faster than a cold solve.

### B6. Braking-aware mesh (S-M, narrowed)

- **Problem.** Intervals on BCN last 0.31 s to 1.50 s, so the mesh resolves no suspension or tyre-lag mode. The curvature mesh puts its longest intervals in braking zones.
- **Change.** Print the interval durations against the mode periods after each solve. Add an opt-in mesh term on the speed change from the QSS profile.
- **Gain.** At most 0.2% of lap time, mostly in the braking profile. A fine-mesh reference already exists: the production mesh is +0.20% on BCN and -0.04% on NBR.

### B7. State scaling (M, measure first)

- **Problem.** The 14 chassis states have a scale of 1, but their values are near 1e-3 to 1e-2 (`vehModel.py:62-75`).
- **Change.** Scale them by typical values, and test whether IPOPT still needs the MC64 scaling.
- **Dissent.** No run measured the gain. The Sturn tests suggest that IPOPT is not sensitive to scaling here. Do this only after A2.

## C. Setup sensitivity and analysis

### C1. Track width and per-point lateral bounds (S-M)

- **Problem.** The model limits the lateral position to +-4 m on every circuit (`vehModel.py:79`), so every track is 8 m wide.
- **Change.** Add `userOpts(n_bounds=...)`: a half-width, a pair of bounds, or a reference line with a band, evaluated per point. This also gives a fixed-line mode for correlation.
- **Gain.** The BCN lap changes by about 1.86 s per metre of half-width, about 1.6%. That is larger than most setup effects.
- **Risk.** The repo has no real track widths yet. Tight bands can need a homotopy.

### C2. Limit prices from the stored multipliers (S, then M)

- **Problem.** Every result stores the NLP multipliers (`functions/warmstart.py:334-353`), but only the warm starts read them.
- **Change.** Add `functions/sensitivity.py` (numpy only) for limit prices and costate maps. Later, add setup gradients at fixed parameters.
- **Gain.** Sensitivities to the width, the torque, brake and power limits and the start speed, at zero solve cost.
- **Risk.** The prices are first order only. They are wrong across a change of active constraints.

### C3. Anti-roll bars (S)

- **Problem.** The model has no anti-roll bar. The rear roll-stiffness field `ksD` is a GUI field, but no model reads it. MATLAB does not read it either.
- **Change.** Add `k_arb_f` and `k_arb_r` in N/m at the wheel, default 0. Remove `ksD` or label it as unused.
- **Gain.** The QSS BCN lap moves +0.21% to +1.20% across front roll-stiffness shares from 0.3 to 0.7.
- **Risk.** The QSS model puts the optimum at the current split by construction. With torque vectoring on, the NLP can trim the balance by torque, so it can show less sensitivity. Check with ATD off as well.

### C4. Differential realism (M)

- **Problem.** The default (ATD on) is ideal lossless torque vectoring for drive and brake (`vehModel.py:503-506`). The alternative is an open split with equal left and right torque. So the default lap is an active torque-vectoring lap.
- **Change.** First compare ATD on and off on Sturn, which takes two solves. Then add opt-in rows for bounded authority or a torque-bias ratio. Add a limited-slip model only if the first steps show a large effect.
- **Dissent.** The literature review has only blog titles on differentials.

### C5. Tyre data path (M)

- **Problem.** The tyre set is a hard-coded road-tyre fit (205/60 R15) on a 2.1 t race car. The model has no pressure terms.
- **Change.** Add a `.tir` reader, `tyre_set='file:<path>'` and MF6.1 pressure terms with an inflation-pressure field. Apply the same terms in the QSS screener.
- **Gain.** It enables car-specific tyres. The repo has no race-tyre data yet.
- **Risk.** Never commit licensed tyre data to the public repo.

### C6. TyreOptim port gap and tyre scaling factors (S)

- **Problem.** MATLAB TyreOptim sets `vp.Fz0 = 4905*vp.Fz0_shift` before it builds the model (`MLTP_TyreOptim.m:128`, `134`). The Python `optimise_design` replaces only the named field (`MLTP_paramOptim.py:81-89`). So in Python, `Fz0_shift` acts as a plain grip multiplier.
- **Effect.** This is why the Python TyreOptim ran to the 0.5 bound (Sturn 15.750 s against 18.016 s). The parity gates do not cover the co-optimisation scripts. With the gap closed, the Sturn optimum is interior: 17.499 s at Fz0_shift 0.787 (run V8). On these two results, the gap explains about 77% of the old gain of 2.266 s.
- **Change.** Substitute the nominal load in the Python TyreOptim path as MATLAB does, and add a test. Add Magic Formula scaling factors for load, grip and stiffness, default 1, shared with E3.
- **Status.** The checks rejected the first version of this item, because the literature has no source on scaling factors. The code check found the port gap itself.

## D. Track data

### D1. Track QA gate, then a circuit builder (S, then M)

- **Problem.** The loader reads s and k without checks (`userOpts.py:77-106`). The Jarama curvature integrates to +78.7 laps of turning, against +1.0 from its own x and y. Every Jarama result runs on a fictitious track.
- **Problem.** BCN turns -1.023 laps against -1.000, with about 17 degrees of false heading near s = 2000 m. The BCNAssetto file does not load.
- **Change, S.** Add `functions/track_qa.py`: turning, closure, sign agreement with x and y, spacing and monotonic s. Warn at load time. Rebuild Jarama from its x and y.
- **Change, M.** Add a heading-conserving builder from GPS or x and y. `IRP/MLTP_Python/track_processor.py` already does part of this.
- **Risk.** The smoothing level is itself a fit parameter. Smoothing moved the QSS lap on the racing line by 1.7%.

## E. Outputs and correlation with data

### E1. Logger-equivalent output channels (S, then M)

- **Problem.** The stored lateral acceleration is the derivative of the lateral speed (`vehModel.py:558-559`), not the value that an accelerometer measures. On BCN the difference has an RMS of 10.6 m/s^2.
- **Problem.** The motor energy is in Wh but has the label kWh (`MLTP.py:599-609` and the GUI). MATLAB stores the motor power in kW (`MLTP.m:592`), so its energy value is in kWh, and the label matches the value.
- **Change, S.** Add accelerometer-equivalent ax and ay, the steer angle and the wheel speeds. Fix the energy unit, and update `tests/test_results.py`.
- **Change, M.** Store channels at the collocation points, and export at 1 m or 50 Hz in the ChassisSim "monster file" format.

### E2. Provenance and a defaults table (S, file-name change M)

- **Problem.** A result does not store the parameter values, the override values, the start speed, the width, the start state or the code version (`MLTP.py:613-634`). A non-default solve saves under the default file name.
- **Change, S.** Add `data['provenance']`. Add a README table of the defaults that move the lap, with a price where known. Add a "not modelled" list with a trigger for each feature.
- **Change, M.** Give each setup a unique file name. This touches the result lookup in the app.
- **Why.** Section 8: lap time alone proves little, and the GT3 study uses grouped validation and a placebo test.

### E3. Correlation knobs (S)

- **Problem.** The model accepts the `Cd` and `Cl` fields, but the aero data file overwrites them and the model does not read them (`vehParams.py:175-190`).
- **Change.** Add `Cd_scale`, `Cl_scale`, `mu_x_scale` and `mu_y_scale`, default 1, in the model and the QSS screener. Use one implementation with the C6 scaling factors. A "driver margin" is the same knob.
- **Gain.** The staged matching of Section 8 becomes possible with the existing sweep tools. In QSS, 5% more grip changes the lap by -1.3% to -1.4%.
- **Risk.** The knobs absorb model error. Do A1 to A3 first.

### E4. Logging specification (S)

- **Change.** Write one page: channels, rates, units, signs, and the stage that uses each channel. Note that the model takes left-turn curvature as positive, while ChassisSim takes right-turn curvature as positive.
- **Note.** Do not add a channel-map file until E5 can read it.

### E5. Telemetry reader and overlay report (M)

- **Problem.** Nothing reads logged data. The only validation is MATLAB parity.
- **Change.** Add `functions/telemetry.py` with readers for the monster file and for mapped CSV. Add a lap split and a projection to s and n. Add an overlay of speed, time delta, ax, ay and steer. Stream long files.
- **Prior art.** `IRP/MLTP_Python` holds three AiM CSV laps of another car, a track processor and an overlay script. Use them for tests.

### E6. Staged correlation runner (M, then L)

- **Change, M.** Build a grip stage, an identifiability check and a stage ledger on top of `setup_sweep` and `optimise_design`.
- **Change, L, later.** Add a tracking objective inside the NLP.
- **Dissent.** The first version claimed more than its own identifiability evidence supports. It needs C1 and the track import first.

### E7. Tyre grip against load from data (L)

- **Change.** First fit `mu_x_scale` and `mu_y_scale` to the measured peak acceleration against speed. Then fit the peak friction against load per axle from mid-corner and braking samples.
- **Gain.** Grip is the largest lap lever.
- **Risk.** It needs near-limit logs, and the loads depend on the model. Do A1 and A2 first.

### E8. Aero coefficients from straights (M, low priority)

- **Change.** Get the drag area from a power balance, and the downforce from ride height against speed squared.
- **Dissent.** A 20% drag error moves the QSS lap by only about 0.1%. It needs inverter torque, airspeed and ride-height data.

## F. Workflow and GUI

### F1. GUI cleanup (S)

- **Change.** Remove BCNAssetto and Jarama from the circuit list until D1 rebuilds them (`app/mainwindow.py:25-27`). Add NBR. Mark the CopyB tyre set as legacy.
- **Note.** MATLAB runs MF205, so CopyB is necessary only to reproduce old Python results.

### F2. Headless and GUI options (S-M)

- **Change.** Add `freeze_support` to `app/main.py`. Forward `ladder`, `homotopy`, `refine` and `ipopt_overrides` from the run configuration. Make the homotopy the default for four motors after one BCN check. Add a QSS screen mode.
- **Deferred.** A GUI for sweeps and sensitivity (L).

## Skipped for now

- **3D road (grade, bank, vertical curvature).** The QSS lap moves at most 0.33% on closed laps, but local speeds move by 1.6 to 2.3 m/s. Add it before drag or braking correlation on NBR or Spa, when usable altitude data exists.
- **Ride-height aeromap.** No aeromap data exists, and the CasADi interpolant needs the slower MX route. Add it after A4, when ride-height or damper data exists.
- **Full-model g-g-v envelope by trim.** QSS on the NLP's own BCN line is within 0.14% of the NLP lap, so the QSS gap comes from the fixed line. Add it together with a free-line QSS.
- **Mesh-consistent regularisation.** The lap effect is at most 0.12%, with an unknown sign. Add it when refinement studies compare objectives across meshes.
- **Seven-post rig and bump-grip derating.** The lap effect is about 0.05% with the vendor demo profile, and the result has no meaning before A2. Add it after A2, when road-profile data exists.
- **Heading-consistent curvature sampling.** The lap effect is at most 0.005%. Include it in the D1 builder.
- **Driver model, thermal tyre, wear, relaxation length.** Section 10 of the review skips these. Add a thermal state when data shows a grip drift that load and slip cannot explain.

## Verification record

| Item | Checks passed | Note |
|---|---|---|
| A1, A2, A3, A5, A6 | 3 of 3 | |
| A4 | rewritten | stage 1 kept, aeromap stage skipped |
| B1, B3, B4, B5 | 3 of 3 | |
| B2 | 2 of 3 | no literature source |
| B6 | narrowed | a fine-mesh reference already exists |
| B7 | 2 of 3 | gain not measured |
| C1, C2, C3, C5 | 3 of 3 | |
| C4 | 2 of 3 | literature has titles only |
| C6 | rewritten | the code check found the port gap |
| D1 | 3 of 3 | curvature-sampling item merged in |
| E1, E4, E5, E7 | 3 of 3 | |
| E2 | 3 of 3 | defaults-table and scope items merged in |
| E3 | 3 of 3 | driver-margin item merged in |
| E6 | 2 of 3 | split into two steps |
| E8 | 2 of 3 | small lap effect |
| F1 | 2 of 3 | inert-field claim is an inference |
| F2 | rewritten | split, sweep GUI deferred |
| Skipped items | 1 or 2 of 3 | see "Skipped for now" |
