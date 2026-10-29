# Lap-time simulation: improvement list

Compiled 17 October 2026 from `docs/lap_time_simulation_literature.md` and a review of the code.

## Scope and method

- **Sources.** The literature review is `docs/lap_time_simulation_literature.md`. "Section N" below refers to its sections.
- **Code.** The review read branch `worktree-speed-phase1`, checked out at `.claude/worktrees/speed-phase1`. All line numbers refer to that branch. The MATLAB original is `IRP/FullModel_4EM_Suspension_FullTyre`.
- **Method.** Six readers extracted 347 facts from the review and the code. Five proposers wrote 56 items, which merged into 38. Each item then had three independent checks: support in the literature, accuracy against the code, and value for this kind of solver.
- **Evidence.** No NLP solve ran during the review. The numbers come from stored results, stored multipliers and the QSS screener. Treat them as estimates until an NLP run checks them.
- **Outcome.** 21 items passed all three checks, 11 passed two, and 6 failed two. This list keeps 30 items, some of them merged or narrowed. The section "Skipped for now" holds the rest, each with a trigger.

## Status on 28 October 2026

Branch `phase3` closes the last 15 open items: B3, B4, B5, B6, B7, C3, C4, C5, E2, E4, E5, E6, E7, E8 and F2. Each new knob is opt-in, and its default keeps the old behaviour. The NLP snapshots show no change: seven builds with `physics='matlab'` and two with `physics='corrected'`. All 55 test files pass on Linux. Two validation runs found a bug each, and the branch removes both (B4 and B7 below).

Done:

- **E2, one file name per setup.** A setup knob that differs from its default adds 8 hex digits at the end of the result stem. The digits are the first 8 hex digits of a sha256 hash of the knob values, for example `_8c76aec0` for `vp_overrides={'LMUY': 0.95}`. Default stems do not change. Every `userOpts` argument is a knob, except the stem, solver and path keys in `userOpts._NOT_SETUP`. `data['provenance']` holds the code version, the knobs, the run route and every parameter value. The GUI computes the same digits. So a knob run no longer overwrites the default result.
- **B5, continuation from the stored default.** `MLTP(continue_from='auto')` starts a setup change from the stored default result, with its multipliers. It raises `ValueError` before the solve if that result is missing, holds knobs, did not converge or does not fit the new NLP. On Sturn, a 3% mass step takes 15 iterations and 3.9 s. A cold solve of the same setup takes 500 iterations and 79.6 s, and its lap is 1.007 s slower.
- **B3, inputs between knots.** `between_knots='linear'` feeds the dynamics a linear input between two knots. `'linear+power'` also applies the motor power rows at the collocation points. With `'linear+power'`, the cold lap is 0.20% longer on Sturn and 0.029% longer on BCN. So the near hold of MATLAB makes the default lap 0.03% to 0.2% too fast.
- **B4, knot ringing.** Stage S was in place before this branch: `vehicle_full` (E1 stage M) holds every channel at the collocation points. On the stored BCN result, the loads and the longitudinal channels ring less there than at the knots, and the lateral channels do not. Stage M adds `collocation='radau'`, the Radau IIA points, for the 23-state NLP. The first Radau solve on Sturn took 5325 iterations to a lap 5.3% too slow. The last point of each interval is a copy of the next knot, and the copy had the bounds of the knot. When the knot sat on a bound, that gave three active constraints on two variables. Commit fc611ac frees the bounds of the copy. Now Sturn takes 278 iterations to 17.99226 s (-0.146%). The racing line and the warm-start seed take the knot, not the copy. Radau needs `OPT_d` of 2 or more: with one point, the CasADi quadrature weight is 0.5, so `userOpts` refuses `OPT_d=1`.
- **B6, mesh against the vertical modes.** After each solve, `MLTP` prints three lines that compare the interval durations with the vertical modes of the model. On BCN, the 155 intervals last 0.31 s to 1.50 s, and none resolves a mode. The opt-in term `mesh_opts={'speed': w}` puts knots where the QSS speed changes fast. On BCN at w = 1, the cold lap is 116.35533 s in 218 iterations, 0.076% shorter than on the default mesh. This meets the keep rule (lap at most 116.385 s, at most 248 iterations), so the term stays.
- **B7, state scaling.** `state_scaling='typical'` scales the 14 chassis states by typical sizes. Only 7 of the 14 states are small, so the premise of the item was half wrong. With MUMPS it shows no gain. The median of five Sturn starts goes from 178 to 207 iterations, with a tighter physical stop rule (see Findings). The validation found a bug. With `'typical'`, the start-box code of `userOpts` reused the name `tol`, so IPOPT got an array for `tol`. Commit 6451820 keeps `tol` a number. The knob stays opt-in for a Windows MA57 check by the owner.
- **C4, differential realism.** With ATD Off, the lap is 0.371 s (2.06%) longer on Sturn and 3.434 s (2.95%) longer on BCN. The QSS forecast for Sturn was only +0.022 s, because the QSS does not model the yaw moment of a left-right torque split. `atd_tbr` bounds the torque bias ratio of both axle differentials: |T_l - T_r| <= k |T_l + T_r|, with k = (TBR - 1)/(TBR + 1). On Sturn, TBR 3, 2 and 1 cost 0.008 s, 0.027 s and 0.315 s. On BCN, TBR 2 costs 0.198 s (0.170%).
- **C3, anti-roll bars.** `k_arb_f` and `k_arb_r` (N/m at the wheel, default 0) add the force k (xs_l - xs_r) to the left strut. The right strut gets the opposite force. The QSS adds 4 k to the axle sum of the spring rates. The GUI labels `ksD` as unused. The NLP is about 100 times less sensitive to a bar than the QSS, with ATD On and with ATD Off.
- **C5, tyre files and pressure.** `tyre_set='file:<path>'` reads a .tir file as written, in the sign convention of the model. Eleven new Magic Formula keys hold the MF 6.1 pressure terms and the two pressures. At the defaults, every pressure factor is 1. With made-up pressure terms, a 10% pressure step changes the Sturn objective by +0.03343 s in 3 iterations. The first-order forecast is +0.03319 s.
- **F2, headless and GUI options.** `cfg.json` reaches every new option, and the GUI reaches them through combos and a JSON box. The cfg key `"mode": "qss"` runs the QSS screen in about 1 s, without IPOPT. The Advanced tab has a Run combo, a Physics combo, a Start state combo, a Homotopy check box and a "More options" JSON box. When you choose 4 Motors, the GUI ticks Homotopy. The BCN four-motor check with the homotopy converged at every step, to 111.17993 s.
- **E4, logging specification.** `docs/logging_spec.md` lists the channels, units, rates and signs, and the stages that use each channel. The model takes a left turn as positive. Five ChassisSim manuals say that an import flag of 1 takes a right turn as positive.
- **E5, telemetry readers.** `functions/telemetry.py` reads a monster file and a mapped CSV file, splits a log into laps, projects it on the centreline and writes an overlay report. The CSV reader streams the file. At the default chunk, the read of 300000 rows and 40 columns (a 115 MB file) peaks at 21.1 MB of memory. On synthetic BCN laps at n = +-1.5 m, the projection is within 0.165 m in s and 0.028 m in n.
- **E6, E7 and E8, correlation runner.** `functions/correlate.py` fits QSS knobs to a log in three stages. The grip stage fits LMUX and LMUY from ax and ay. The drag stage fits `Cd_scale` on the straights, and the downforce stage fits `Cl_scale` from the damper travel. A gate refuses a fit when the collinearity index gamma is above 15. It also refuses a knob that a lap cannot identify (LFZO, Fz0_shift, LKX, LKY). A JSON ledger records each stage. The fits apply to the QSS only.

The runs below used MUMPS on Linux, one at a time, with `save=False` or a scratch `results_dir`. The column "Seed" names the warm start. A lap change of a cold run compares with the cold default solve of the same day. That is Sturn 18.01857 s (152 iterations) and BCN 116.44387 s (207 iterations). With `physics='corrected'`, it compares with V10 (20.02081 s, 392 iterations). A warm run compares with its seed: the stored Sturn result (18.01633 s) or the stored BCN result (116.44407 s).

| Run | Case | Seed | Result |
|---|---|---|---|
| VB3-1 | Sturn, `between_knots='linear'` | Cold | 18.05249 s, 143 iterations, +0.03392 s (+0.188%) |
| VB3-2 | Sturn, `between_knots='linear+power'` | Cold | 18.05521 s, 125 iterations, +0.03664 s (+0.203%). n_g grows from 1904 to 1958. |
| VB3-3 | BCN, `between_knots='linear+power'` | Cold | 116.47793 s, 117 iterations, +0.03406 s (+0.029%). n_g grows from 16015 to 16480. |
| VB4-1 | Sturn, `collocation='radau'` | Cold | Before commit fc611ac: 18.97522 s, 5325 iterations (+5.3%). After it: 17.99226 s, 278 iterations (-0.146%). |
| VB4-2 | Sturn, `physics='corrected'`, Radau | Cold | Before commit fc611ac: 20.29610 s, 426 iterations (+1.4%). After it: 20.05447 s, 249 iterations (+0.168%). |
| VB4-1q | Sturn, Radau, `ladder='qss23'` | Cold | 17.99499 s, 312 iterations (-0.131%) |
| VB5-1 | Sturn, LMUX = LMUY = 0.95 | `continue_from='auto'` | 18.24010 s, 20 iterations, 4.1 s wall, the same as V9 |
| VB5-2 | Sturn, `mb` +3% (1874.6 kg) | `continue_from='auto'` | 18.16919 s, 15 iterations, 3.9 s wall |
| VB5-3 | The setup of VB5-2 | Cold | 19.17591 s, 500 iterations, 79.6 s wall, 1.00672 s slower than VB5-2 |
| VB6-1 | BCN, `mesh_opts={'speed': 1.0}` | Cold | 116.35533 s, 218 iterations (-0.076%) |
| VB6-1v | The setup of VB6-1 at vi = 60.001 m/s | Cold | 116.39807 s, 156 iterations. The default mesh at this start: 116.44196 s, 277 iterations. |
| VB7-B | Sturn, `state_scaling='typical'`, five starts | Cold | 491, 190, 389, 207 and 199 iterations, median 207 (default 178). Two laps are slow: 19.11574 s (vi = 60.002 m/s) and 18.08485 s (vi = 59.999 m/s). |
| VB7-C | Sturn, `'typical'`, MUMPS scaling off | Cold | 18.02153 s, 170 iterations. The default scaling with MUMPS scaling off: 18.01857 s, 157 iterations. |
| VB7-D | Sturn, `physics='corrected'`, `'typical'` | Cold | 20.05873 s, 860 iterations (V10: 392) |
| VC4-2 | Sturn, `atd_tbr` 3, 2 and 1 | Cold | 18.02621 s (196 iterations), 18.04571 s (281), 18.33339 s (119). ATD Off: 18.38951 s (151). |
| VC4-3 | BCN, `atd_tbr=2` | Cold | 116.64141 s, 186 iterations (+0.170%). ATD Off: 119.87823 s (156). |
| VC3-1 | Sturn, `k_arb_f` 2e4 N/m | Stored Sturn result with duals | 3 iterations. The objective changes by -1.028 ms. The first-order estimate is -1.193 ms. |
| VC3-2 | Sturn, `k_arb_r` 2e4 N/m | Stored Sturn result with duals | 22 iterations. The objective changes by -1.213 ms. The first-order estimate is +0.225 ms. |
| VC3-3 | Sturn, ATD Off, `k_arb_f` 2e4 N/m | Cold, then the ATD Off result with duals | Cold: 18.42626 s, 105 iterations. Warm: 7 iterations, objective -0.306 ms against an estimate of +1.270 ms. |
| VC5-1 | Sturn, INFLPRES +10%, made-up pressure terms | Stored Sturn result with duals | 3 iterations. The objective changes by +0.03343 s. The first-order estimate is +0.03319 s. |
| VE7-1 | Sturn, LMUX = LMUY = 0.90 | Stored Sturn result with duals | 18.48051 s, 22 iterations. `grip_fit` moves LMUY by -0.1013. `grip_from_lap` gives 0.90199. |
| VE7-2 | Sturn, LMUY = 0.90 | Stored Sturn result with duals | 22 iterations. `grip_fit` moves LMUY by -0.0941 and LMUX by -0.0200. |
| VE7-3 | Sturn, LMUX = 0.90 | Stored Sturn result with duals | 23 iterations. `grip_fit` moves LMUX by -0.0270 and LMUY by -0.0011. |
| VE8-1 | BCN, defaults | Stored BCN result with duals | 116.44407 s, 0 iterations: the stored point, now with `vehicle_full` |
| VE8-2 | BCN, `Cd_scale` 1.3 | The VE8-1 result with duals | 116.76333 s, 71 iterations, +0.31926 s. `drag_fit` on the main straight fails. |
| F2-V1 | BCN, four motors, ATD Off, `homotopy=True` | Cold | 111.17993 s. The 7-state init takes 938 iterations, the three steps 234, 98 and 59. 496 s wall. |

Findings:

- **Stop rule (B7).** IPOPT applies its tolerances to scaled errors. With `'typical'`, the largest physical defect of the chassis states is 8 to 18 times smaller than with the default scaling (VB7-B, VB7-C, VB7-D). So the two scalings stop at different accuracy, and the iteration counts are exploration, not a speed result. Without the MUMPS scaling, both scalings still converge (VB7-C).
- **Local optima (B7).** In five Sturn starts 1 mm/s apart, one start of the default scaling and two starts of `'typical'` end on a slower local optimum. With the default scaling it is vi = 59.998 m/s (18.70993 s). With `'typical'` it is vi = 60.002 m/s (19.11574 s) and vi = 59.999 m/s (18.08485 s). That second lap is 0.07 s above the other three typical starts (18.015 s to 18.019 s). With `'typical'`, vi = 59.998 m/s reaches 18.01736 s. So the scaling moves the risk to other starts and does not remove it. For this reason and the stop rule, the BCN run of B7 (VB7-E) did not take place.
- **Radau (B4).** With the default physics, the Radau solve takes 278 iterations against 152 and 19.5 s against 12.5 s, at 0.85 times the time per iteration. With `physics='corrected'`, it takes 249 iterations against 392 and 12.0 s against 75.3 s. Against the Legendre knots, the Radau knots ring less in fz_rl: 0.18 of the roughness with the default physics, 0.44 with the corrected physics. They also ring less in zs (0.12, default physics) and fx_rl (0.32, corrected physics). The gate of 0.6 fails for fx_rl with the default physics (0.73), and for ax (0.67) and zs (1.86) with the corrected physics. Radau also damps the three undamped body modes of the default physics. So MATLAB parity runs stay on Legendre points.
- **Radau seed (B4).** Before commit fc611ac, a seed that puts the next knot into the end-point copy made the solve worse (1697 iterations, +10.1%). After it, Radau from the qss23 ladder takes 312 iterations, close to the 278 of the default ladder. So the seed did not change.
- **Inputs between knots (B3).** On BCN, the measured lap cost of the linear input (+0.029%) is about 14 times smaller than the first-order forecast for `'linear+power'` (+0.4% to +0.45%). It is about the size of the scatter between cold BCN solves (13 ms to 33 ms). With `'linear+power'`, the solves take fewer iterations (Sturn 125 against 152, BCN 117 against 207). On BCN, each iteration takes 1.19 times as long.
- **Speed term (B6).** The term gains 0.089 s at vi = 60.000 m/s and 0.044 s at vi = 60.001 m/s. Between these two starts, the lap on the speed mesh moves by 0.043 s, and on the default mesh by 0.002 s. Against the fine MATLAB reference (116.211 s), the error falls from 0.200% to 0.124% and 0.161%. At vi = 60.000 m/s, the 7-state init on the speed mesh ended with `Error_In_Step_Computation`, and the 23-state solve still converged. A primal-only start from the stored default (full-interp) did not finish in 20 minutes.
- **Anti-roll bars (C3).** The first-order bar gradients hold for small steps. A 2e3 N/m step, solved to a tolerance of 1e-7, agrees within 5% (rear bar with ATD On +4.0%, front bar with ATD Off -4.3%). Two of three 2e4 N/m hops went to another local optimum. A hop from the rear-bar result back to the default setup ends 1.444 ms of objective below the stored Sturn default. So the stored default is not the best local optimum, and laps a few ms apart do not rank bar setups. With ATD Off, the NLP front-bar slope is +6.35e-8 s per N/m, about 90 times smaller than the QSS slope and of the opposite sign.
- **Torque bias (C4).** TBR 2 keeps 93% of the Sturn gain of ideal torque vectoring and 94% of the BCN gain. TBR 1 keeps 15% on Sturn. On BCN at TBR 2, both axles reach the bias bound (k = 0.333) somewhere on the lap.
- **Grip fit on NLP data (E7).** On the stored Sturn optimum, `grip_fit` reads LMUY 0.9726 and LMUX 0.8693, not 1. So the QSS reads the grip of the NLP 2.7% low and its trail braking 13% low. The fit finds a change of LMUY within 0.01 (-0.1013 and -0.0941 for -0.10). It finds a change of LMUX of -0.10 as -0.027 only, because only trail braking shows LMUX. So keep LMUX and LMUY linked (`link=True`, the default). The first-order NLP forecast of the step to 0.90 is 6.4% low (+0.43542 s against +0.46538 s).
- **Drag fit on NLP data (E8).** On the BCN main straight, ax at the collocation points rings (standard deviation 3.60 m/s^2 around a mean of 2.14 m/s^2). So `check_ax` refuses the log (correlation 0.00 with d vx / dt). Without ax, the straight has no window, because |ay| rings up to 3.64 m/s^2. A whole-lap fit without ax and with 0.5 s windows reads 1.0297 and 1.3445. The difference +0.3148 is inside the gate of +0.30 +- 0.05.
- **Pressure (C5).** The warm 10% pressure step changes the objective by 1.007 times the first-order forecast. The pressure terms are not data of a real tyre, so this shows only that the code is consistent.

Open items, each with a trigger:

- E2: the promoted parameters of `optimise_design` (`param_specs`, `Fz0_shift_bounds`) and the `refine` options are not in the file name. So `MLTP_paramOptim(params=..., save=True)` overwrites `Sturn_paramOptim.mat`, and two refine runs share one `_meshAdaptive` file. Add them to the knobs when two such runs must exist side by side.
- E2: `ipopt_overrides` is not in the file name. A looser tolerance through it can overwrite the default result. `data.provenance.run` records the overrides. Add them to the knobs if a run with other IPOPT options must keep its own file.
- E2: the tracked `Results/*.mat` files are older than E2 and have no provenance. `continue_from='auto'` trusts them by their names. Solve them again when a check needs their provenance.
- B5: `continue_from='auto'` does not compare the linear solver, the IPOPT options or the code version of the source. Only small steps have a measurement (grip up to 10%, mass 3%). For a larger step, use the bridges of `setup_sweep`.
- B3: `limit_prices` gives NaN for Pmax when the result has power rows at the collocation points. Sum the multipliers of those rows when a price study needs it.
- B3: no motor-speed row applies between the knots. The Om box keeps the motor speed below 0.885 OMmax at the default gear and wheel radius. Add the rows when a gear above 8.19 or a wheel radius below 0.3143 m lifts that cap.
- B3: `optimise_design`, `setup_gradients` and `setup_sweep` refuse a value other than `'matlab'`. Add support when a co-optimisation or a sweep needs it.
- B4: each interval keeps the end-point copy (Choice A). Choice B removes it and makes the KKT system 22% smaller. Add it when a Radau study with the default physics needs the wall time.
- B4: `setup_gradients` of a Radau result needs `collocation='radau'` by hand. A Radau result or a Radau QSS screen needs casadi, because the numpy fallback has no Radau points.
- B4: Radau leaves part of the ringing (see Findings). Use a finer mesh when a study must read such a channel at the knots.
- B6: the speed term moves the knots with the setup. So a homotopy step or a chain of setups with the term seeds the primal only, and `setup_sweep` refuses the term. Use it for single solves.
- B7: the Windows MA57 check (`ipopt_overrides={'ma57_automatic_scaling': 'no'}` with both scalings) is for the owner. Recommend `'typical'` only if it rescues a solve there.
- B7: a result of one scaling seeds the other scaling with the primal only. Map the multipliers when a sweep must seed `'typical'` solves from default results.
- C4: the rows bound the drive torque and the brake torque alike. So above TBR 1, the brake torque can also differ between left and right. Add a separate brake split when the owner asks for equal brake torque on the left and right.
- C4: the passive limited-slip model is not built. Add it when the owner says the car has one and TBR 2 costs 1% or more against ideal torque vectoring. The measured cost is 0.15% on Sturn and 0.17% on BCN.
- C4: `limit_prices` gives no price for the differential rows. `setup_sweep` compares only the row count of its base. So a PureSlip base for a TBR sweep fails at the hub check, after one solve.
- C3: the QSS cannot rank bars for the NLP. Use small NLP hops or `setup_gradients`.
- C3: the damper rate follows the spring rate only. So a bar lowers the roll damping ratio, from 0.56 to 0.48 for a 2e4 N/m front bar with `physics='corrected'`. Add a damper rate that includes the bar when a bar above 3e4 N/m runs with `physics='corrected'`.
- C3: with `physics='corrected'`, the QSS keeps the roll split of the springs (error up to 9% in the load transfer). Add the series law when `physics='corrected'` becomes the default or a study ranks bars with it.
- C5: no solve used a real tyre file. A file in ISO signs (pKy1 < 0) needs a conversion by hand, and the sign check covers only the cornering stiffness at Fz0. Add an ISO switch when a real ISO file exists.
- C5: the 7-state init and the friction homotopy ignore the tyre file and the pressure. All four wheels use the same coefficients (no TYRESIDE mirror). The equations are MF 5.2 with the MF 6.1 pressure terms. The file sets no wheel radius, tyre stiffness, inertia, rolling resistance or gear. Add these when a real file needs them.
- C5: the GUI has no file picker. A cfg.json can give `"tyre_set": "file:<path>"`. Add a picker when GUI users need tyre files.
- F2: `refine` is a cfg.json key only, because the Output tab does not look for `_meshAdaptive` files. Add a widget and the file lookup when GUI users need refinement.
- F2: the A5 keys reach the GUI as `pt_overrides` in the "More options" box. The GUI has no rows for them. Add rows when the owner picks the ratings.
- F2: the frozen exe has not run the QSS mode on Windows. `build/README.md` lists the check.
- E5: no real log went through the readers. The signs of the sample AiM channels are not verified, so the repo ships no AiM channel map. `overlay_report` warns when ay or the steer of a log runs against the simulation. Check the signs with the first real log.
- E5: `project` estimates no shift and no rotation. A GPS log fits a track built from the GPS points of the same log (`write_track(xy_from_gps(...))`). The stored circuits are not in east and north axes. Add an alignment when a GPS log meets one of them.
- E6: the tracking objective inside the NLP (step L) waits for a real log and its E5 projection.
- E7: the grip against load per axle (E7b) waits for wheel-load channels. On lap data the gate refuses it (gamma 50 to 940).
- E7: `grip_fit` takes its values from raw samples. One ax spike of -13 m/s^2 moved LMUX from 0.927 to 1.005. Use median or smoothed values if real logs show spikes.
- E8: without a brake channel, `drag_fit` counts a deceleration below -3 m/s^2 as braking. A coast at 90 m/s with `Cd_scale` 1.5 reaches -2.81 m/s^2. So light braking can count as coast. Use a log with a brake channel or a coast-down run.
- E4: the owner inputs of the spec are open, and the owner gives them. They are the logger, its channels and its rates. They also include the steering ratio, the gear ratio, the damper motion ratios and the signs of ay and steer.
- E4: the export writes ay and steer right positive (`sign=(-1, -1)`), so set both ChassisSim import flags to 1. No file went through a ChassisSim import yet, so this stays unverified. Check it with the first ChassisSim import.
- The other open items of 21 October stay open. This branch closes three of them: the file names (E2), the fit tool (E6 to E8) and the A5 keys in the GUI (F2).

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
| V14 | BCN, `physics='corrected'` | Cold | 130.04167 s, 212 iterations, 224 s wall. On 18 October, before A4: 130.03517 s, 179 iterations. |
| V11 | Sturn, LKY = 0.9 | Stored Sturn result with duals | 18.09017 s, 6 iterations, +0.07384 s. The objective changes by +0.07146 s. The first-order estimate is +0.06404 s. |
| V12 | Sturn, LKY = 0.95 | Stored Sturn result with duals | 18.05111 s, 3 iterations, +0.03478 s. The objective changes by +0.03379 s. The first-order estimate is +0.03202 s. |
| V13 | Sturn, `physics='corrected'`, `Preg` = 50 kW | The V10 result with duals | 19.97510 s, 49 iterations, -0.04571 s (-0.23%). The QSS forecast is -0.42%. |

Findings:

- **A4 in the NLP.** With `physics='corrected'`, the cold Sturn lap is 0.0033 s (0.017%) shorter than the lap of 18 October (V10). This difference is smaller than the cold-solve scatter of about 0.01 s on Sturn (ladder study, default physics). So the solve shows only that the effect of A4 is small, as the QSS forecast of -0.003% says. The solve took 392 iterations, against 218 on 18 October. On this path-sensitive NLP, the iteration count does not measure the change. On BCN, the cold lap is 0.0065 s (0.005%) longer than on 18 October (V14). The QSS forecast is 0.012% shorter, so the two signs differ. So the BCN solve also shows only that the effect of A4 is small.
- **Setup gradient (C2).** For LKY = 0.9, the first-order estimate is 10.4% below the change of the objective (V11). For LKY = 0.95, the gap is 5.2% (V12). The gap halves with the step. This is the sign of a second-order term, not of a gradient error. The objective includes the input regularisation, so the lap changes are about 3% larger than the objective changes.
- **Regenerative braking (A5).** With `Preg` = 50 kW, the single motor brakes at 7 of the 19 knots (V13). Its power reaches -50.000 kW at the knots and -50.069 kW between them, because the power row applies only at the knots (B3). The motor energy falls from 1016.4 Wh to 982.1 Wh. The QSS forecast of -0.42% is 1.8 times the NLP change of -0.23%.
- **A5 in the QSS.** With `physics='corrected'` and four motors, half ratings (301 Nm, 228 kW) make the QSS lap 1.10% longer on Sturn and 0.89% longer on BCN. A 456 kW pack limit adds 4.88% and 4.07%. A regen power of 456 kW for each motor removes 0.58% and 0.44%.

Open items, each with a trigger:

- The owner still has to pick the per-motor ratings, the pack limit and the regen power. The defaults keep the current car: 602 Nm and 456 kW for each motor, no pack row and no regen.
- No solve tests the four-motor car with the A5 keys. Run one before you use such a result.
- The vendor document does not state the ChassisSim signs of ay and steer. The export assumes that ChassisSim takes a right turn as positive. Check this with the first ChassisSim import.
- `limit_prices` gives no price for the regen bounds or the pack row. For a result with regen, it gives NaN for Tmax and Pmax, because their formulas assume no regen. Add these prices when a regen study needs them.
- The research report says that lap times, tyre temperature and tyre pressure cannot identify LFZO, LKX, LKY and the aero scales. Keep these knobs at 1 until the data of Table 5 in `docs/tyre_scaling_track_method.md` exist.
- The branch skips the items below. Add each one when its trigger occurs.
  - The ride-height aeromap stage of A4. Add it when ride-height or damper data exist.
  - The side-force roll moment. The side force is zero at the default twist-wing angle of 0 degrees. Add it when a run tilts the twist wing.
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
- **Status on 28 October 2026.** Done. `userOpts(between_knots='linear')` feeds the dynamics the linear input between two knots. `'linear+power'` also applies the motor power rows at the collocation points. The default stays `'matlab'`. With `'linear+power'`, the cold lap is 0.20% longer on Sturn and 0.029% longer on BCN (runs VB3-2 and VB3-3). The first-order forecast of 0.4% to 0.45% for BCN with `'linear+power'` was too large.

### B4. Knot ringing (S, then M)

- **Problem.** Gauss-Legendre collocation is not L-stable. Unresolved fast modes swing from knot to knot. The stored BCN rear-left tyre load reads 13.3, 2.1, 1.2 and 5.8 kN at four adjacent knots.
- **Change, S.** Store a second channel set at the collocation points.
- **Change, M.** Add an opt-in Radau IIA scheme. CasADi already supplies the points.
- **Risk.** Radau puts duplicate points into `s_full`, which can break code that expects a strictly increasing grid.
- **Status on 28 October 2026.** Done. Stage S: `vehicle_full` (E1 stage M) holds the channels at the collocation points. On the stored BCN result, the loads and the longitudinal channels ring less there than at the knots. The rear-left load has 0.59 and ax 0.35 of the knot roughness. The lateral channels ring as much (ay 1.02). Stage M: `userOpts(collocation='radau')` for the 23-state NLP. After commit fc611ac, Radau converges on Sturn with both physics settings (runs VB4-1 and VB4-2). It removes part of the ringing, not all of it. Keep the MATLAB parity runs on Legendre points, because Radau damps the undamped body modes of the default physics.

### B5. Continuation for single off-default solves (S-M)

- **Problem.** A cold solve at a non-default setup can land on a slow branch. On Sturn such solves were 0.47 s to 1.93 s slower than a solve continued from the default.
- **Change.** Add `MLTP(continue_from='auto')`, which warm-starts with duals from the stored default result. It needs E2, so that the source result is unambiguous.
- **Gain.** No wrong-branch laps, and an estimated four times faster than a cold solve.
- **Status on 28 October 2026.** Done. `MLTP(continue_from='auto')` starts from the stored default result with its multipliers, and it refuses every case that does not fit. On Sturn, a 3% mass step takes 15 iterations and 3.9 s. A cold solve of the same setup takes 500 iterations and 79.6 s, and its lap is 1.007 s slower (runs VB5-2 and VB5-3).

### B6. Braking-aware mesh (S-M, narrowed)

- **Problem.** Intervals on BCN last 0.31 s to 1.50 s, so the mesh resolves no suspension or tyre-lag mode. The curvature mesh puts its longest intervals in braking zones.
- **Change.** Print the interval durations against the mode periods after each solve. Add an opt-in mesh term on the speed change from the QSS profile.
- **Gain.** At most 0.2% of lap time, mostly in the braking profile. A fine-mesh reference already exists: the production mesh is +0.20% on BCN and -0.04% on NBR.
- **Status on 28 October 2026.** Done. `MLTP` prints the interval durations against the vertical modes after each solve. On Sturn and BCN, no interval resolves a mode. The opt-in speed term (`mesh_opts={'speed': w}`) passed its keep rule on BCN: 116.35533 s in 218 iterations, 0.076% below the default mesh (run VB6-1). The gain depends on the start: at vi = 60.001 m/s it is 0.044 s.

### B7. State scaling (M, measure first)

- **Problem.** The 14 chassis states have a scale of 1 (`vehModel.py:62-75`). Only 7 of them are small: zs, theta, phi and the four tyre deflections, with RMS values of 4e-3 to 3e-2 in the stored optima. The three body rates are near 0.05 to 0.14, and the four unsprung velocities near 0.15 to 0.37.
- **Change.** Scale them by typical values, and test whether IPOPT still needs the MC64 scaling.
- **Dissent.** No run measured the gain. The Sturn tests suggest that IPOPT is not sensitive to scaling here. Do this only after A2.
- **Status on 28 October 2026.** Done, as an opt-in knob. `userOpts(state_scaling='typical')` scales the chassis states by typical sizes. With MUMPS it shows no gain (run VB7-B). IPOPT applies its tolerances to scaled errors, so the two scalings stop at different physical accuracy. Treat the iteration counts as exploration. The Windows MA57 check by the owner stays open.

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
- **Gain.** The QSS BCN lap moves -0.21% to +1.20% across front roll-stiffness shares from 0.3 to 0.7. The QSS optimum share is 0.52, against 0.48 today.
- **Risk.** With torque vectoring on, the NLP can trim the balance by torque, so it can show less sensitivity. Check with ATD off as well.
- **Status on 28 October 2026.** Done. `k_arb_f` and `k_arb_r` (N/m at the wheel, default 0) act with both physics settings and in the QSS. The NLP slope is about 100 times smaller than the QSS slope, with ATD On and with ATD Off. So torque vectoring does not explain the gap, and the QSS cannot rank bars for the NLP. `ksD` stays as a field, labelled unused.

### C4. Differential realism (M)

- **Problem.** The default (ATD on) is ideal lossless torque vectoring for drive and brake (`vehModel.py:503-506`). The alternative is an open split with equal left and right torque. So the default lap is an active torque-vectoring lap.
- **Change.** First compare ATD on and off on Sturn, which takes two solves. Then add opt-in rows for bounded authority or a torque-bias ratio. Add a limited-slip model only if the first steps show a large effect.
- **Dissent.** The literature review has only blog titles on differentials.
- **Status on 28 October 2026.** Done. ATD Off makes the lap 2.06% longer on Sturn and 2.95% longer on BCN. The knob `atd_tbr` adds rows for the torque bias ratio. TBR 2 costs 0.15% on Sturn and 0.17% on BCN against ideal torque vectoring (runs VC4-2 and VC4-3). The limited-slip model is not built, see the open items of 28 October.

### C5. Tyre data path (M)

- **Problem.** The tyre set is a hard-coded road-tyre fit (205/60 R15) on a 2.1 t race car. The model has no pressure terms.
- **Change.** Add a `.tir` reader, `tyre_set='file:<path>'` and MF6.1 pressure terms with an inflation-pressure field. Apply the same terms in the QSS screener.
- **Gain.** It enables car-specific tyres. The repo has no race-tyre data yet.
- **Risk.** Never commit licensed tyre data to the public repo.
- **Status on 28 October 2026.** Done. `functions/importfile.read_tir` reads a .tir file, and `tyre_set='file:<path>'` uses it. In the full model, the MF 6.1 pressure terms act on the peak friction and the slip stiffness. In the QSS, they act on the peak friction only. `.gitignore` keeps `*.tir` files and `Data/tyres/` out of the repo. The repo still has no race-tyre data.

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
- **Status on 28 October 2026.** Done. Each setup of the knobs gets its own file name, and `data['provenance']` records the code, the knobs, the run and the parameter values. The open items of 28 October list the inputs that are not knobs. The README has the table of the defaults that move the lap and the list of what the model leaves out.

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
- **Status on 28 October 2026.** Done: `docs/logging_spec.md`. E5 reads a JSON channel map, and the spec gives its form. The repo ships no map file, because the signs of the sample logger are not verified.

### E5. Telemetry reader and overlay report (M)

- **Problem.** Nothing reads logged data. The only validation is MATLAB parity.
- **Change.** Add `functions/telemetry.py` with readers for the monster file and for mapped CSV. Add a lap split and a projection to s and n. Add an overlay of speed, time delta, ax, ay and steer. Stream long files.
- **Prior art.** `IRP/MLTP_Python` holds three AiM CSV laps of another car, a track processor and an overlay script. Use them for tests.
- **Status on 28 October 2026.** Done. `functions/telemetry.py` has `read_monster`, `read_csv` (it streams the file), `split_laps`, `project`, `time_delta` and `overlay_report`. Only synthetic logs went through them.

### E6. Staged correlation runner (M, then L)

- **Change, M.** Build a grip stage, an identifiability check and a stage ledger on top of `setup_sweep` and `optimise_design`.
- **Change, L, later.** Add a tracking objective inside the NLP.
- **Dissent.** The first version claimed more than its own identifiability evidence supports. It needs C1 and the track import first.
- **Status on 28 October 2026.** The M step is in place: `functions/correlate.run_stage` with the gate and the ledger. Step L (the tracking objective) is not built, because it needs a real log.

### E7. Tyre grip against load from data (L)

- **Change.** First fit `mu_x_scale` and `mu_y_scale` to the measured peak acceleration against speed. Then fit the peak friction against load per axle from mid-corner and braking samples.
- **Gain.** Grip is the largest lap lever.
- **Risk.** It needs near-limit logs, and the loads depend on the model. Do A1 and A2 first.
- **Status on 28 October 2026.** The first step is in place (`grip_fit`, `grip_from_lap`). On NLP data it finds a change of LMUY within 0.01. It finds a change of LMUX of -0.10 as -0.027 only (runs VE7-1 to VE7-3). The second step (per axle) is not built: on lap data the gate refuses it.

### E8. Aero coefficients from straights (M, low priority)

- **Change.** Get the drag area from a power balance, and the downforce from ride height against speed squared.
- **Dissent.** A 20% drag error moves the QSS lap by only about 0.1%. It needs inverter torque, airspeed and ride-height data.
- **Status on 28 October 2026.** Done as `drag_fit` and `lift_fit`, for the QSS only. On BCN, ax of the NLP rings at the collocation points, so the fit on the main straight fails (run VE8-2). A whole-lap fit without ax finds the change of `Cd_scale` within the gate.

## F. Workflow and GUI

### F1. GUI cleanup (S)

- **Change.** Remove BCNAssetto and Jarama from the circuit list until D1 rebuilds them (`app/mainwindow.py:25-27`). Add NBR. Mark the CopyB tyre set as legacy.
- **Note.** MATLAB runs MF205, so CopyB is necessary only to reproduce old Python results.

### F2. Headless and GUI options (S-M)

- **Change.** Add `freeze_support` to `app/main.py`. Forward `ladder`, `homotopy`, `refine` and `ipopt_overrides` from the run configuration. Make the homotopy the default for four motors after one BCN check. Add a QSS screen mode.
- **Deferred.** A GUI for sweeps and sensitivity (L).
- **Status on 28 October 2026.** Done. `app/main.py` calls `freeze_support`. cfg.json forwards `ladder`, `homotopy`, `refine`, `ipopt_overrides` and the other new options. The GUI reaches all of them except `refine`. The cfg key `"mode": "qss"` runs the QSS screen. When you choose 4 Motors, the GUI ticks Homotopy. A BCN check supports this default (run F2-V1).

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
