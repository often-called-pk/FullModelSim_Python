# Lap-time simulation: improvement list

Compiled 17 October 2026 from `docs/lap_time_simulation_literature.md` and a review of the code.

## Scope and method

- **Sources.** The literature review is `docs/lap_time_simulation_literature.md`. "Section N" below refers to its sections.
- **Code.** The review read branch `worktree-speed-phase1`, checked out at `.claude/worktrees/speed-phase1`. All line numbers refer to that branch. The MATLAB original is `IRP/FullModel_4EM_Suspension_FullTyre`.
- **Method.** Six readers extracted 347 facts from the review and the code. Five proposers wrote 56 items, which merged into 38. Each item then had three independent checks: support in the literature, accuracy against the code, and value for this kind of solver.
- **Evidence.** No NLP solve ran during the review. The numbers come from stored results, stored multipliers and the QSS screener. Treat them as estimates until an NLP run checks them.
- **Outcome.** 21 items passed all three checks, 11 passed two, and 6 failed two. This list keeps 30 items, some of them merged or narrowed. The section "Skipped for now" holds the rest, each with a trigger.

## Status on 21 October 2026

Branch `finish-partials` closes the open part of seven items: A4, A5, C2, C6, D1, E1 and E3. Each new knob has a default that keeps the old behaviour. With the default `physics='matlab'`, the NLP stays byte-identical in seven test builds, and the parity gates stay green. With `physics='corrected'`, only A4 changes the NLP, by design. All 44 test files pass on Linux.

Done:

- **A4, drag pitch moment.** With `physics='corrected'`, the rear-wing drag acts at the wing height `hw` (1.28 m), and the body drag acts at the CG height (0.5 m). The MATLAB co-optimisation scripts use the same split. A rearward force above the CG lifts the nose, and the QSS moves the matching load to the rear axle. At 60 m/s, the coast pitch changes from 2.2569 mrad to 2.0886 mrad. The QSS lap changes by -0.003% on Sturn and by -0.012% on BCN.
- **A5, the rest of the powertrain.** With `physics='corrected'`, `vp_overrides` takes five keys: `gear`, `Tmax`, `Pmax`, `Preg` and `Ppack`. `Tmax` and `Pmax` set the ratings of one motor. `Preg` sets the regenerative braking power of each motor, and `Ppack` limits the sum of the motor powers. With the default physics, each key raises `ValueError`.
- **C2 stage M, setup gradients.** `functions/sensitivity.setup_gradients` gives the first-order change of the objective per unit of a setup parameter. It rebuilds the NLP at the stored optimum with the stored multipliers, and it does not solve. A parameter can be a vehicle primary or a Magic Formula factor. It takes 0.11 s on Sturn and 1.01 s on BCN. For 27 parameters, the gradients agree with central differences of the rebuilt NLP to 1.8e-6 or better. `optimise_design` can now also promote a Magic Formula factor.
- **C6, the other Magic Formula factors.** The branch adds LFZO (nominal load), LKX (longitudinal slip stiffness) and LKY (cornering stiffness), default 1, as in the PAC2002 manual. LFZO = 0.9 gives the same full-model tyre forces and the same QSS lap as the override Fz0 = 0.9 x 4905 N. That QSS lap on Sturn is 18.89453 s, 0.80% longer than the default. LKX and LKY change the slip stiffness, and the peak forces do not change. The QSS uses only the peak friction, so LKX and LKY do not change the QSS lap. The 7-state model ignores the three factors.
- **D1 stage M, track builder.** `functions/track_qa.write_track` builds a track file from a polyline, runs the track QA and writes the file. `xy_from_gps` converts GPS fixes to east and north metres, and `smooth_xy` smooths GPS noise. `write_track` writes no file in these cases:
  - The start and the end have a gap between them.
  - A closed line does not turn +1 or -1 laps.
  - The track QA finds an error.
- **D1, test on BCN.** BCN, built again from its own x and y, turns -1.000000000000 laps and has no QA issue. Its QSS lap is 134.768 s, 4.1% longer than with the parity file. So BCN keeps its old file, and the branch registers no new circuit.
- **E1 stage M, channels at the collocation points.** `MLTP` now evaluates the vehicle channels at every knot and collocation point. It uses the inputs that the NLP dynamics use. With the linear inputs (`u_full`), ax and ay at the collocation points are wrong by 0.70 m/s^2 and 1.15 m/s^2 RMS. The result now holds `vehicle_full` and `t_full`, and the 66 old result arrays do not change. `functions/telemetry.py` writes a ChassisSim monster file at 50 Hz and a CSV file at 1 m steps.
- **E3, aero scales.** `Cd_scale` scales the whole drag coefficient, and `Cl_scale` scales all four lift coefficients. The default is 1. The full model and the QSS apply the scales with both physics settings. On Sturn, `Cd_scale` = 1.2 adds 0.0084 s to the QSS lap, and `Cl_scale` = 1.2 removes 0.0041 s. The `Cd` and `Cl` fields stay MATLAB placeholders, and the Setup tab labels them so.

The runs below used MUMPS on Linux. The column "Seed" names the warm start. A lap change compares with the stored Sturn baseline of 18.01633 s. For V13, it compares with V10.

| Run | Case | Seed | Result |
|---|---|---|---|
| V10 | Sturn, `physics='corrected'` | Cold | 20.02081 s, 392 iterations. On 18 October, before A4: 20.02413 s, 218 iterations. |
| V11 | Sturn, LKY = 0.9 | Stored Sturn result with duals | 18.09017 s, 6 iterations, +0.07384 s. The objective changes by +0.07146 s. The first-order estimate is +0.06404 s. |
| V12 | Sturn, LKY = 0.95 | Stored Sturn result with duals | 18.05111 s, 3 iterations, +0.03478 s. The objective changes by +0.03379 s. The first-order estimate is +0.03202 s. |
| V13 | Sturn, `physics='corrected'`, `Preg` = 50 kW | The V10 result with duals | 19.97510 s, 49 iterations, -0.04571 s (-0.23%). The QSS forecast is -0.42%. |

Findings:

- **A4 in the NLP.** With `physics='corrected'`, the cold Sturn lap is 0.0033 s (0.017%) shorter than the lap of 18 October (V10). This difference is smaller than the cold-solve scatter of about 0.01 s on Sturn (ladder study, default physics). So the solve shows only that the effect of A4 is small, as the QSS forecast of -0.003% says. The solve took 392 iterations, against 218 on 18 October. On this path-sensitive NLP, the iteration count does not measure the change.
- **Setup gradient (C2).** For LKY = 0.9, the first-order estimate is 10.4% below the change of the objective (V11). For LKY = 0.95, the gap is 5.2% (V12). The gap halves with the step. This is the sign of a second-order term, not of a gradient error. The objective includes the input regularisation, so the lap changes are about 3% larger than the objective changes.
- **Regenerative braking (A5).** With `Preg` = 50 kW, the single motor brakes at 7 of the 19 knots (V13). Its power reaches -50.000 kW at the knots and -50.069 kW between them, because the power row applies only at the knots (B3). The motor energy falls from 1016.4 Wh to 982.1 Wh. The QSS forecast of -0.42% is 1.8 times the NLP change of -0.23%.
- **A5 in the QSS.** With `physics='corrected'` and four motors, half ratings (301 Nm, 228 kW) make the QSS lap 1.10% longer on Sturn and 0.89% longer on BCN. A 456 kW pack limit adds 4.88% and 4.07%. A regen power of 456 kW for each motor removes 0.58% and 0.44%.

Open items, each with a trigger:

- No cold BCN solve with `physics='corrected'` checks A4 yet. A BCN solve needs 6 GB of free memory, and less than 5 GB was free during this work. Run it when 6 GB is free. The QSS forecast for BCN is -0.012%.
- The owner still has to pick the per-motor ratings, the pack limit and the regen power. The defaults keep the current car: 602 Nm and 456 kW for each motor, no pack row and no regen.
- No solve tests the four-motor car with the A5 keys. Run one before you use such a result.
- The vendor document does not state the ChassisSim signs of ay and steer. The export assumes that ChassisSim takes a right turn as positive. Check this with the first ChassisSim import.
- A result file name does not show the knob values (E2). So a knob run with `save=True` overwrites the default result. Use `save=False` or a separate `results_dir` until E2 gives unique names.
- `limit_prices` gives no price for the regen bounds or the pack row. Add these prices when a regen study needs them.
- The research report says that lap times, tyre temperature and tyre pressure cannot identify LFZO, LKX, LKY and the aero scales. Keep these knobs at 1 until the data of Table 5 in `docs/tyre_scaling_track_method.md` exist.
- The branch skips the items below. Add each one when its trigger occurs.
  - The ride-height aeromap stage of A4. Add it when ride-height or damper data exist.
  - The side-force roll moment. The side force is zero at the default twist-wing angle of 0 degrees. Add it when a run tilts the twist wing.
  - GUI rows for the A5 keys. Add them with F2, when the GUI offers `physics='corrected'`.
  - A tool to fit the new factors and scales. Add it when the data of Table 5 exist.
  - A command line for the track builder. Add it when a new circuit arrives.
  - The optional strain columns of the monster file. Add them when a rig needs them.

## Status on 20 October 2026

Branch `sim-phase2` now holds part 2 of C6 and the grip part of E3. The owner decisions of 19 October 2026 and Table 5 of `docs/tyre_scaling_track_method.md` set the scope. That research report gives a candidate method to estimate the factors from lap times, tyre temperature and tyre pressure. No human reviewed the report.

Built:

- **LMUX and LMUY.** These are the standard Magic Formula 5.2 peak-grip factors (lambda_mux and lambda_muy), default 1. As in MF 5.2, they scale the peak friction and the vertical shift on each axis. The slip stiffness does not change.
- **Scope.** The full model and the QSS screener apply the factors. The 7-state model ignores them, and `Fz0_shift` does not change.
- **Default.** At exactly 1, the model skips the multiplication. The default NLP stays byte-identical in seven test builds, and the parity gates stay green. The parity export leaves out the two factors while both are 1, because MATLAB has no counterpart.
- **Tests.** The new file `tests/test_mf_scaling.py` passes. The verification ran 13 of the 41 test files, and all 13 pass.

Not built, with the reason from the research report:

- **LKX and LKY (slip stiffness).** The QSS uses only the peak friction, so these factors cannot change the QSS lap time. This is a structural limit: no lap-time data can identify them through the QSS. A better estimate needs steer, wheel or motor speeds and IMU data.
- **LFZO (nominal load).** In the QSS, LFZO shares the peak-friction channel with LMUX, LMUY and the Cl scale. On one circuit, it is nearly collinear with the peak grip. The report says: never fit it from lap times.
- **Cd and Cl scales (E3).** A 20% drag error moves the QSS lap by only about 0.1%. Power, wind and rolling resistance confound this small effect. On one circuit, the Cl scale is collinear with LMUY. The report names coast-down runs for Cd, and ride-height or GPS data for Cl.
- **A QSS tool for the grip scale.** The tool would turn stint median lap times into a grip scale. Add it when the first session data exist.
- **LMUX and LMUY as design variables.** `optimise_design` does not promote them. Add this when a co-optimisation needs it.

The run below used MUMPS on Linux. The lap change compares with the stored Sturn baseline of 18.01633 s.

| Run | Case | Seed | Result |
|---|---|---|---|
| V9 | Sturn, LMUX = LMUY = 0.95 | Stored Sturn result with duals | 18.24010 s, 20 iterations, +0.22377 s (+1.24%), 3.5 s wall |

Findings:

- **Lap response.** With 5% less peak grip on both axes, the Sturn lap is 1.24% longer. The QSS forecast is +1.26%: the QSS lap changes from 18.74393 s at 1 to 18.98036 s at 0.95. At 1, the QSS lap is equal to the default QSS lap.
- **Use of the factors.** The report links LMUX and LMUY to one value. It sets that value from data only in the QSS, after four gates. It keeps the full model at 1 until a simulation study (B2) gives the full-model elasticity. The code has two keys, so set both keys to the same value for this use.

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
- **Rolling start (B1), start speed.** With `physics='corrected'` at 60 m/s, a cold rolling start takes 220 iterations to 20.03744 s (before the fix: 270 iterations, 20.02972 s). With the fix, this lap is 0.008 s longer. The solve ends at another local optimum. At 70 m/s, the rolling start is infeasible as posed (V3b, V3c). The failure is not a solver or transcription bug. Two local solves, from two seeds, maximise the start speed vx(0) with every other start state in the rolling box at 70 m/s. Both put the highest start speed at about 68.5 m/s (68.50 m/s and 68.53 m/s). The solves are local, so they give no global proof. The vx start box at 70 m/s has its lower edge at 69 m/s. When only this box widens to 59-71 m/s, the cold V3b solve converges in 348 iterations, with a start speed of 65.475 m/s.
- **Braking-limit warning.** `MLTP` now warns when the vx box of a rolling start has its lower edge above the QSS braking limit at s = 0. On the Sturn centreline, this limit is 66.87 m/s with `physics='corrected'` and 70.79 m/s with the default physics. The QSS limit is about 1.6 m/s below the highest start speed that the NLP found. So the warning can also fire for a feasible start. With `physics='corrected'` on Sturn, the warning fires for vi from 67.9 m/s to 69.5 m/s. In this band, the vx box still holds start speeds at or below 68.5 m/s, so a feasible point is likely. No solve tests this band. The solve still runs after the warning.
- **Start limit.** With `physics='corrected'`, a start speed above 62 m/s does not make the start infeasible (V3a). The body height stays at the edge of the start box instead. At 70 m/s, the solve with the default start box converges only from a pre-braked start away from the equilibrium. At s = 0, the rear wheels lock, and the unsprung states sit on their bounds.
- **Periodic lap (B2).** The periodic rows hold exactly (V4, V5). The BCN curvature does not close the lap (-1.023 laps), so treat the +0.467 s of V5 as provisional until a closed BCN track exists.
- **Width prices (C1, C2).** The first-order prices predict the change of the objective within 1.1% (V6, V7). One metre of half-width is worth about 1.86 s on BCN.
- **TyreOptim (C6).** The fix moves the Sturn optimum from the 0.5 bound (15.750 s) to an interior point (17.499 s at Fz0_shift 0.787). A second solve that started at Fz0_shift 0.53 ended on a worse optimum (21.832 s).

Open items:

- B1: the refine reseed (`warmstart_refined`) still sets rows 9-22 to static values. Change it when a rolling start runs with refine. No solve checks a rolling start with the qss23 ladder. Only Sturn solves check the braking-limit warning. The fix does not clip the ends of the cold seed to the start box. A clip gives no measured gain, and the proposed form changes the default seed. At the default vi = 60 m/s, the braking-limit warning also fires on the circuits below. No solve checks these cases. The margin is the lower edge of the vx box (59 m/s) minus the QSS braking limit. On Sturn, the QSS limit is about 1.6 m/s below the highest start speed that the NLP found. No other circuit has a measurement of this difference.
  - Jarama: +0.6 m/s with the default physics, +8.5 m/s with `physics='corrected'`.
  - BCN_S2: +4.9 m/s with the default physics, +9.0 m/s with `physics='corrected'`.
  - Circle: +11.1 m/s with the default physics, +17.5 m/s with `physics='corrected'`. Circle has a curved start, so the straight-start warning also fires.
  - Hairpin: +1.7 m/s, only with `physics='corrected'`.
  - ZigZag and ZigZagMirror: +1.8 m/s each, only with `physics='corrected'`.
  - Sturn, Spa, NBR and BCN: no warning with either physics.
- C6 part 2, the Magic Formula scaling factors, waits for the owner's sign-off, together with E3. The status of 20 October gives the result.
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
- **Status on 21 October 2026.** Stage 1 is complete. The downforce goes through the springs, with the aero pitch and roll moments (status of 18 October). The rear-wing drag now also gives a pitch moment. With `physics='corrected'`, the QSS lap changes by -0.003% on Sturn and by -0.012% on BCN. The cold Sturn solve gives 20.02081 s (run V10).

### A5. Four-motor powertrain (M)

- **Problem.** With four motors, the model divides the wheel speed by the gear ratio to get the motor speed (`vehModel.py:514-515`). The single-motor model multiplies by it (`vehModel.py:496`). So the per-motor power and speed rows are 52.6 times too loose. Each motor also gets the full 602 Nm and 456 kW.
- **Change.** Add a per-motor model behind the flag: speed times gear, per-motor ratings, `gear` as a setup field and an optional pack power row. Regenerative braking also needs a new upper bound on the power rows (`MLTP.py:93-96`).
- **Gain.** The four-motor car becomes power limited. The QSS BCN lap moves +0.13% to +7.6%, depending on the ratings that the owner picks.
- **Risk.** The four-motor NLP is already fragile and needs the friction homotopy.
- **Status on 21 October 2026.** Done, with `physics='corrected'`. Each motor turns at the wheel speed times the gear ratio (status of 18 October). The per-motor ratings, the gear, regenerative braking and the pack power row are now `vp_overrides` keys. Run V13 checks regen on Sturn: 50 kW makes the lap 0.23% shorter. The owner picks the ratings, and no solve checks the four-motor car with the keys yet.

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
- **Status on 21 October 2026.** Done. Stage S, the limit prices, is in the status of 19 October. Stage M adds `setup_gradients`, which gives the setup gradients at zero solve cost. Runs V11 and V12 check one gradient against two solves.

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
- **Status on 20 October 2026.** Part 2 adds only the grip factors LMUX and LMUY, default 1, in the full model and the QSS screener. It does not add the load factor LFZO or the stiffness factors LKX and LKY. The status of 20 October gives the reasons.
- **Status on 21 October 2026.** Done. The branch adds LFZO, LKX and LKY, default 1, so all five standard factors exist. LFZO = 0.9 gives the same QSS lap as Fz0 = 0.9 x 4905 N: 18.89453 s on Sturn. At LKY = 0.9, the first-order gradient predicts the change of the objective within 10.4% (run V11). The data limits of the status of 20 October still apply: keep the new factors at 1 without data.

## D. Track data

### D1. Track QA gate, then a circuit builder (S, then M)

- **Problem.** The loader reads s and k without checks (`userOpts.py:77-106`). The Jarama curvature integrates to +78.7 laps of turning, against +1.0 from its own x and y. Every Jarama result runs on a fictitious track.
- **Problem.** BCN turns -1.023 laps against -1.000, with about 17 degrees of false heading near s = 2000 m. The BCNAssetto file does not load.
- **Change, S.** Add `functions/track_qa.py`: turning, closure, sign agreement with x and y, spacing and monotonic s. Warn at load time. Rebuild Jarama from its x and y.
- **Change, M.** Add a heading-conserving builder from GPS or x and y. `IRP/MLTP_Python/track_processor.py` already does part of this.
- **Risk.** The smoothing level is itself a fit parameter. Smoothing moved the QSS lap on the racing line by 1.7%.
- **Status on 21 October 2026.** Done. Stage S, the QA gate, is in the status of 19 October. Stage M adds `write_track`, `xy_from_gps` and `smooth_xy` to `functions/track_qa.py`. BCN, built again from its own x and y, turns -1.000000000000 laps. Its QSS lap is 4.1% longer than with the parity file, so the smoothing risk is real, and BCN keeps its file.

## E. Outputs and correlation with data

### E1. Logger-equivalent output channels (S, then M)

- **Problem.** The stored lateral acceleration is the derivative of the lateral speed (`vehModel.py:558-559`), not the value that an accelerometer measures. On BCN the difference has an RMS of 10.6 m/s^2.
- **Problem.** The motor energy is in Wh but has the label kWh (`MLTP.py:599-609` and the GUI). MATLAB stores the motor power in kW (`MLTP.m:592`), so its energy value is in kWh, and the label matches the value.
- **Change, S.** Add accelerometer-equivalent ax and ay, the steer angle and the wheel speeds. Fix the energy unit, and update `tests/test_results.py`.
- **Change, M.** Store channels at the collocation points, and export at 1 m or 50 Hz in the ChassisSim "monster file" format.
- **Status on 21 October 2026.** Done. Stage S is in the status of 19 October. Stage M stores `vehicle_full` and `t_full` and adds `functions/telemetry.py`, which writes the monster file and a 1 m CSV. At the collocation points, ax and ay agree with the kinematics of the state polynomials to 5.2e-5 m/s^2 and 7.4e-7 m/s^2. The signs of ay and steer in the monster file are not verified. The E5 readers can go in the same module.

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
- **Status on 20 October 2026.** The C6 factors LMUX and LMUY are the grip knobs `mu_x_scale` and `mu_y_scale`, in one implementation. The branch does not add `Cd_scale` and `Cl_scale`, and the problem with the `Cd` and `Cl` fields stays open. The status of 20 October gives the reasons.
- **Status on 21 October 2026.** Done. `Cd_scale` and `Cl_scale` exist, default 1, in the full model and the QSS screener. The `Cd` and `Cl` fields stay MATLAB placeholders, and the Setup tab labels them "(placeholder)". On Sturn, `Cd_scale` = 1.2 adds 0.0084 s to the QSS lap. The research report still advises against a fit of the scales from lap times.
- **Driver margin.** The research report advises a different use. It keeps the effective grip level, which includes the driver margin, out of the vehicle model, as a lap-time ratio.

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
