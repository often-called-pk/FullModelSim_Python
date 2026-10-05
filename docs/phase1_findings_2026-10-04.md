# Phase 1 findings (2026-10-04)

Branch `worktree-speed-phase1`. All numbers were measured today on one machine: IPOPT 3.14.11, CasADi 3.7.2, HSL ma57 with MC64 scaling unless stated. Benchmark case: Sturn (`OPT_ds` 30, N=18, `OPT_d` 3, uniform mesh), Static aero, ATD On, EM4 Off, `vi` 60. Roadmap: [speed_capability_roadmap.md](speed_capability_roadmap.md) (section "Phase 1 status"). Wall times move ~30% with machine load (the identical 3209-iteration solve took 243 to 360 s of solve time across six logs), so iteration counts are the primary metric. Raw data and logs: `C:\Users\ASUS\.claude\jobs\9b76333d\tmp\study\` (scratch; `results/<tag>.json`).

"Corrected tyre" means the lateral values MATLAB actually runs (MF_205_60R15_V91). The options-study, mesh and warm-start runs used the five-coefficient override `{pEy1, pKy1, pKy4, pKy5, pVy1}`; `vehParams(tyre_set='MF205')` applies all nine lateral coefficients. Iteration counts and laps quoted below for the "corrected" tyre (257 iterations, 18.016 s) are therefore for that five-coefficient proxy; the full `tyre_set='MF205'` Sturn solve with default options takes 489 iterations for 18.009 s (pre-fix NLP, see the status note below). "Shipped tyre" and "Python shipped default" below mean Copy-B, the Python default until the 2026-10-04 flip (see Decisions).

> **Status note (2026-10-05).** Every iteration count, lap time and wall time in this document was measured before the 23-state NLP constraint fix of 2026-10-04 (commit 37e1e4e: input-rate bounds scaled by `u_s`, friction-circle rows only for `TyreModel='PureSlip'`). They are kept unchanged as the historical record, and the headline default-call figures are labelled pre-fix. After the fix the default Sturn solve takes 179 iterations for 18.0086 s (not 489 / 18.009 s) and BCN 247 iterations for 116.441 s (not 614 / 116.523 s). Before/after table, model-audit verdicts and the other follow-ups: [Update 2026-10-05: follow-ups](#update-2026-10-05-follow-ups).

## Summary

* **The Copy-B tyre set, the Python default until 2026-10-04, has no lateral force.** Its lateral block (`pKy4 = 0`, `pKy1 = -20.505`) is the MATLAB 'Test' case, not the set MATLAB runs. Cornering stiffness is then zero, the car corners by drifting, and the Sturn solve takes 3209 IPOPT iterations (202 in restoration) for a non-physical 25.57 s. With the MATLAB-run set (`tyre_set='MF205'`) it takes 489 iterations for 18.009 s (pre-fix NLP), within 0.05% of MATLAB (18.008 to 18.022 s); the five-coefficient proxy used for the benchmark runs below takes 257 iterations for 18.016 s (pre-fix NLP). The default is now MF205 (see Decisions); `tyre_set='CopyB'` keeps the legacy set.
* **The cost is iterations times factorisation.** MA57 factorisation is 81 to 87% of IPOPT time and NLP function evaluation 9 to 13%; the NLP build is about 3.5 to 4 s on Sturn. Per-evaluation work (CSE, JIT, `f_dyn.map`, tyre tabulation) cannot gain more than a few percent, and a C++ rewrite would not help.
* **What does help.** Warm start with duals: a +3% setup change re-solves in 11 iterations / 5.8 s instead of 364 / 41.6 s. Curvature mesh: on BCN the same lap (117.40 vs 117.42 s) in 392 s instead of 968 s, but that run also used fewer intervals (curvature at `OPT_ds` 45, N=103, vs uniform at `OPT_ds` 30, N=155); at equal N it is more accurate (Sturn N=18: +0.16% vs +0.72% lap error). QSS screen: ~7 ms (Sturn) and ~30 ms (BCN) per setup, +3.9% and +11.1% against the corrected-tyre NLP.
* **CSE and the MX route are opt-in.** Both are mathematically harmless but move IPOPT to a different optimum on this path-sensitive NLP, and CSE fails on VirtualTrack.
* **IPOPT options.** Keep the current defaults; a loose-tolerance `screening=True` preset exists (its iteration saving is not robust); never disable MC64.
* **Model issues found.** EM4 per-motor power cap ~52x looser than the single-motor cap; `brkB`/`Tdist` inert with ATD On; unsprung-mass load accounting; no steady-state effect of the front/rear Cl split; Hairpin track defect. Verdicts (see the Update below): the first four are inherited from `vehModel.m` and kept for parity, not port bugs; the Hairpin defect is still open.
* **Decisions taken 2026-10-04:** tyre default flipped to MF205, mesh default `auto` (curvature knot placement on tracks of 2000 m or more, N unchanged), CSE stays opt-in (below).

## Tyre-set defect and MATLAB reference

Python's Copy-B lateral Pacejka block (the default until the flip) equals the `'Test'` case of `vehParams.m` (lines 279-346); MATLAB selects `'MF_205_60R15_V91'` (line 141). Nine lateral coefficients differ, and only those (50 of 59 Pacejka coefficients match):

| | pEy1 | pKy1 | pKy4 | pKy5 | pKy6 | pVy1 | pVy2 | pVy3 | pVy4 |
|---|---|---|---|---|---|---|---|---|---|
| Python Copy-B (default until the flip) | 0.15 | -20.505 | 0 | 0.002 | -0.002 | 0 | 0 | 0 | 0.08 |
| MATLAB-run MF_205_60R15_V91 | 0.33443 | 20.505 | 2.0 | 0 | 0 | 0.026365 | -0.0062119 | -0.41389 | -0.048038 |

Mechanism (`vehModel.py:390`): `Kya = pKy1*Fz0*sin(pKy4*atan(...))`, so `pKy4 = 0` makes the cornering stiffness identically zero, hence `By = 0` and `Fy = 0` at zero camber (`pKy5`, `pKy6`, `pVy3`, `pVy4` act only through camber). Nominal `Kya(Fz0)` is 0 (Copy-B) vs 7.978e4 N/rad per tyre (MF205); at slip angle -2.86 deg and Fz 5000 N, Fy is 0.0 N vs -3180 N per tyre. Setting `pKy4 = 2` alone gives a force with the wrong sign (`pKy1` negative).

Consequences, from the two shipped-tyre optima:

| 23-state, N=18 | 25.57 s (CSE off) | 32.62 s (CSE on) |
|---|---|---|
| max abs(fy), any tyre | 2e-5 N | 2e-5 N |
| min / mean vx at corner knots (200 to 340 m) | 1.61 / 7.2 m/s | 1.38 / 5.0 m/s |
| time over 200 to 340 m | 12.6 s | 18.7 s |
| IPOPT iterations / restoration iterations | 3209 / 202 | 1286 / 0 |
| max abs slip angle | 81 deg | 88 deg |

Both are non-physical local optima of a degenerate problem (6.1 s of the 7.0 s gap is the corner segment); a 1-ulp Hessian change (CSE) moved IPOPT from one to the other.

MATLAB reference (R2025a, original scripts, Sturn, `OPT_ds` 30, N=18, MUMPS):

| run | lap [s] | iterations |
|---|---|---|
| MATLAB, MF_205_60R15_V91, tol 1e-8 | 18.008 | 299 |
| MATLAB, MF_205_60R15_V91, tol 1e-4 | 18.022 | 317 |
| Python, `tyre_set='MF205'` (nine lateral coefficients, default options), pre-fix NLP | 18.009 | 489 |
| Python, `tyre_set='MF205'`, default options, after the 2026-10-04 NLP fix (see Update) | 18.0086 | 179 |
| Python, corrected-tyre study default (`A_fixed`, 5-coefficient override), pre-fix NLP | 18.016 | 257 |
| MATLAB forced onto the Copy-B coefficients, tol 1e-4 | 26.03 (vx min 1.65 m/s) | 327 |
| Python Copy-B (shipped default until the flip), pre-fix NLP | 25.57 (vx min 1.61 m/s) | 3209 |

Audit (MATLAB `userOpts.m`/`vehParams.m` executed and dumped, compared field by field): Pacejka 50/59 match; vehicle 112/113 (only `vp.CG_p_deg_per_deg_table` differs, dormant: `None` in Python, read only with `CamberGain='Table'`); powertrain 7/7; rate-limit tables 55/55 and all 16 aero x ATD x EM4 input-limit combinations; the 7 real circuit files are md5-identical. Other mismatch: the synthetic Hairpin track has 73 trailing zeros vs 146 in MATLAB (port defect, not fixed). Deliberate Python speed defaults differ from MATLAB: `tol` 1e-4 (1e-8), `acceptable_tol` 1e-3 (1e-6), `mu_strategy` adaptive (unset = monotone), `OPT_ds` 30 (10), ma57 (MUMPS).

Provenance: `vehModel.py` lines 8-9 say the set is "Copy B, per the user's confirmation", so it may have been chosen deliberately; see Decisions (taken 2026-10-04).

## Where the time goes

| quantity (23-state IPOPT run) | Sturn, shipped tyre (`A_base`) | Sturn, corrected (`A_fixed`) | BCN N=155 uniform, corrected |
|---|---|---|---|
| iterations | 3209 | 257 | 852 |
| IPOPT overall | 309.0 s | 22.3 s | 853.1 s |
| MA57 factorisation | 269.2 s (87%) | 18.1 s (81%) | 733.1 s (86%) |
| NLP function evaluation | 27.4 s (9%) | 3.0 s (13%) | 82.0 s (10%) |
| linear solve per iteration | 86 ms | 73 ms | 892 ms |
| total wall | 319 s | 33 s | 968 s |

The NLP build (CasADi SX) takes about 3.5 to 4 s on Sturn and 34.0 s on BCN N=155 (3.5% of its wall); the Python assembly loop 0.02 to 0.17 s; the 7-state stage 4.9 s of 33 s (Sturn) and 78.4 s of 968 s (BCN). Removing function evaluation entirely would save 9 to 13% of IPOPT time, so per-evaluation work is bounded; the shipped-tyre solve costs 12.5x the iterations of the five-coefficient corrected one (3209 vs 257) and 6.6x those of `tyre_set='MF205'` (3209 vs 489); all of these counts are pre-fix.

## Per-item results

### Items 1 and 2: CSE, JIT, `f_dyn.map` (`functions/casadi_opts.py`, `functions/transcription.py`, `test_casadi_opts.py`)

| change | result |
|---|---|
| CSE (`MLTP_CSE=1`, `userOpts(cse=True)`): f, g, Jacobian bit-identical, Hessian 1 ulp, ~11% fewer instructions in `f_dyn`/`h_eq` | shipped tyre: lap 32.62 vs 25.57 s (1286 vs 3209 iterations, a different optimum); corrected tyre: 8 runs, 238 to 370 iterations, laps 18.0106 to 18.0249 s; VirtualTrack: `Restoration_Failed` after 1767 iterations (default solved in 812). Opt-in |
| JIT (`userOpts(jit=True)`, `MLTP_JIT=1`; gcc through casadi's `shell` plugin, falls back with one warning) | no compiler on the machine PATH (measured with a process-local gcc): 1.1 to 1.2x on direct evaluation of the Functions, no effect on the inlined SX NLP. Opt-in |
| `f_dyn.map`, SX route (default) | bit-identical to the old loop: 3209 iterations, same lap, oracle functions identical |
| MX route (`MLTP_SYM_TYPE=MX`) | BCN NLP build 27 s to 1.9 s, but derivatives ~8x slower and a different optimum (Sturn shipped tyre: 31.862 s, 453 iterations). Opt-in |

The Python loop was never the cost (0.02 to 0.17 s); build time is CasADi SX derivative generation. With function evaluation at 9 to 13% of IPOPT time, both items are low value on this problem.

### Item 3: curvature mesh (`functions/mesh.py`, `discretise(mesh='curvature')`, `userOpts(mesh=, mesh_opts=)`)

Reference: fine uniform `OPT_ds` 15 (N=36), corrected tyre, Sturn: 17.887 s, 3391 iterations, 638 s wall.

| N | uniform: lap (error), iterations, wall | curvature a = b = 1: lap (error), iterations, wall |
|---|---|---|
| 18 | 18.016 s (+0.72%), 257, 32 s | 17.915 s (+0.16%), 240, 29 s |
| 12 | 18.409 s (+2.9%), 637, 43 s | 18.084 s (+1.1%), 182, 18 s |
| 9 | 21.612 s (+20.8%), 558, 33 s | 19.389 s (+8.4%), 291, 21 s |

BCN (corrected tyre): uniform N=155 (`OPT_ds` 30): 117.423 s, 852 iterations, 968 s wall (its 7-state init ended `Error_In_Step_Computation`); curvature N=103 (`OPT_ds` 45): 117.403 s, 497 iterations, 392 s wall. Factorisation per iteration also falls (860 vs 500 ms), so IPOPT time drops from 853 to 300 s. Default is now `mesh='auto'` (curvature on tracks of 2000 m or more, i.e. BCN-size; uniform below; see Decisions), which only redistributes the knots: N stays round(L / `OPT_ds`), so BCN at the default `OPT_ds` 30 gets a curvature mesh with N=155, the same interval count as before. A reviewer run of the default call (ma57, full `tyre_set='MF205'`) measured auto on BCN at `OPT_ds` 30: 614 iterations, a 116.523 s lap and 566 s solve (pre-fix NLP; after the fix 247 iterations, 116.441 s and ~250 s) against 852 iterations, 117.42 s and 968 s for the uniform N=155 run above (five-coefficient corrected tyre, pre-fix), i.e. 1.7x wall at equal N, and its 7-state init now converges (1072 iterations, Optimal; the uniform-mesh init ended `Error_In_Step_Computation`). ZigZag (auto resolves to curvature, N=79): `Solve_Succeeded`, 481 iterations, 56.216 s; Jarama and Spa also resolve to curvature under auto and are unmeasured. The BCN speedup above (2.5x in wall time) is curvature at `OPT_ds` 45 (N=103) against uniform at `OPT_ds` 30 (N=155); to get the interval reduction pass `OPT_ds=45` or larger with the curvature mesh, e.g. `MLTP(circuit='BCN', OPT_ds=45)`. At equal N the gain is also accuracy (Sturn N=18, table above: lap error +0.72% to +0.16%, 257 to 240 iterations). A result saved on another mesh seeds a new solve by interpolation only (no dual re-injection), so warm-start chains must keep the mesh fixed.

### Items 4 and 7: warm start with duals (`functions/warmstart.py`, `test_warmstart.py`)

`data["nlp"]` is saved with every result; `MLTP(warm_start=<full result .mat or the ctx of an earlier call>, warm_start_duals=True)` re-injects primal and dual solution when the NLP structure matches; `userOpts(ipopt_overrides=)` merges any IPOPT option last. IPOPT recipe (`warm_start_ipopt_opts`): `warm_start_init_point=yes`, all push/frac options 1e-10, `mu_init` 1e-6 (push 1e-8 re-centres for 18 iterations on an identical re-solve).

| case (Sturn, corrected tyre, pre-fix NLP) | cold: iterations / wall / lap | warm, primal + dual: iterations / wall / lap |
|---|---|---|
| identical re-solve | 257 / 32.5 s / 18.0161 | 0 / 4.7 s / 18.0161 (bit-identical) |
| `alpha_RW` +3% | 364 / 41.6 s / 18.0094 | 11 / 5.8 s / 18.0183 |
| `mb` (body mass) +3% | 327 / 36.3 s / 18.1627 | 32 / 8.3 s / 18.1680 |
| chain: `mb` +3%, then `hcg` +3% (in memory) | 385 / 39.6 s / 18.1764 | 9 / 5.3 s / 18.1858 |

Wall speedup 4.4x to 7.5x. Primal-only (no duals): identical re-solve 30 iterations / 6.0 s; `mb` +3% 76 / 9.7 s. Re-check with the full `tyre_set='MF205'` (nine lateral coefficients), +3% mass: 10 iterations warm with duals, 109 primal-only, 505 cold. Caveats: (1) `brkB` and `Tdist` are inert with ATD On (`vehModel.py:492-496`); a +3% `brkB` re-solved in 0 iterations. (2) Cold solves scatter ~0.01 s between local optima, more than many setup effects: `alpha_RW` +3% reads -6.7 ms cold-vs-cold but +2.2 ms warm-vs-base, and the three warm laps sit 5.3 to 9.4 ms above their cold counterparts. Sweeps should chain warm starts from one reference.

### Item 5: QSS screen (`functions/ggv.py`, `MLTP_screen.py`, `test_screen.py`)

Envelope + march ~7 ms (Sturn) and ~30 ms (BCN) per setup. It copies `vehModel`'s aero, loads (26.0 kN total with unsprung mass subtracted per corner; aero load split l_r/L), peak mu from the Pacejka coefficients (exact match to the symbolic mu) and the powertrain/brake limits. Fixed centreline, point mass, no transients, static aero angles, no combined ax+ay load transfer.

| circuit | QSS | corrected-tyre NLP | error |
|---|---|---|---|
| Sturn | 18.744 s | 18.034 s | +3.9% |
| Sturn, QSS marched on the NLP's own line | 17.903 s | 18.034 s | -0.7% |
| BCN | 129.467 s | 116.544 s | +11.1% (~6% line, ~5% corner speed) |

On the NLP's own line the screen is 0.7% fast, so the Sturn error is all fixed centreline. These NLP references are from the screener's comparison runs, not the study runs; 18.034 s sits inside the study's 18.007 to 18.050 s scatter.

## IPOPT options study and recommendation

Sturn N=18, corrected tyre, 23-state stage; 32 runs (full tables, logs and solution comparison in the study `REPORT.md`; tags are `results/<tag>.json`). dLap is relative to the default run; wall times are load-sensitive.

| Tag | Change vs default | Its | Resto | Lap [s] | dLap [ms] | Wall [s] | IPOPT [s] | Lin. solve [ms/it] |
|---|---|---|---|---|---|---|---|---|
| A_fixed | default | 257 | 0 | 18.0161 | +0.0 | 33 | 22.3 | 73 |
| F2_screen_loose | tol 1e-3, acceptable_tol 1e-2, dual_inf_tol 1e-2, constr_viol_tol 1e-3, compl_inf_tol 1e-3 (F2) | 196 | 0 | 18.0223 | +6.3 | 28 | 16.8 | 72 |
| F_fixed | tol 1e-3, acceptable_tol 1e-2 only (identical path to default) | 257 | 0 | 18.0161 | +0.0 | 32 | 20.8 | 68 |
| J_fixed | tol 1e-6 | 355 | 2 | 18.0155 | -0.6 | 41 | 30.2 | 72 |
| I_fixed | mu_init 1e-2 (adaptive ignores it) | 257 | 0 | 18.0161 | +0.0 | 31 | 20.2 | 66 |
| C_fixed | mu_strategy monotone, both stages | 371 | 0 | 18.0273 | +11.2 | 44 | 33.7 | 78 |
| C23_fixed | mu_strategy monotone, 23-state only | 501 | 31 | 18.0207 | +4.6 | 63 | 50.5 | 87 |
| I2_mono_mu1e-2 | monotone + mu_init 1e-2, 23-state only | 370 | 0 | 18.0167 | +0.6 | 44 | 33.2 | 77 |
| B_fixed | nlp_scaling_method none, both stages | 373 | 0 | 18.0105 | -5.6 | 49 | 35.5 | 82 |
| B23_fixed | nlp_scaling_method none, 23-state only | 231 | 0 | 18.0143 | -1.8 | 31 | 20.2 | 74 |
| D_fixed | L-BFGS (history 20), both stages | 765 | 0 | 18.0122 | -3.8 | 64 | 23.5 | 23 |
| D23_fixed | L-BFGS, 23-state only | 1063 | 12 | 18.0086 | -7.5 | 41 | 32.7 | 23 |
| E_fixed | linear_solver mumps | 318 | 0 | 18.0290 | +12.9 | 42 | 33.8 | 93 |
| M_ma27 | linear_solver ma27 | 213 | 0 | 18.0315 | +15.5 | 28 | 18.1 | 72 |
| N_ma97 | linear_solver ma97 | 285 | 0 | 18.0073 | -8.8 | 43 | 33.3 | 104 |
| G_fixed | bound_relax_factor 0, honor_original_bounds yes | 294 | 0 | 18.0126 | -3.5 | 39 | 25.3 | 73 |
| L_noauto | ma57_automatic_scaling no (MC64 off) | 262 | 0 | 18.0231 | +7.0 | 27 | 16.0 | 48 |
| Q_combo | MC64 off + unscaled 23-state | 261 | 0 | 18.0267 | +10.7 | 25 | 15.3 | 46 |
| S_screen | F2 + MC64 off | 243 | 2 | 18.0237 | +7.6 | 27 | 15.5 | 50 |
| H_* (8 runs) | the options above with CSE | 238 to 370 | 0 to 2 | 18.0106 to 18.0249 | -5.4 to +8.8 | 24 to 97 | | |
| P_* (4 runs) | default 23-state stage, changed 7-state stage options | 210 to 342 | 0 to 9 | 18.0153 to 18.0497 | -0.7 to +33.6 | 32 to 40 | | |

VirtualTrack (N=52), same options applied as stated:

| run | change | lap [s] | status | iterations | resto | wall [s] |
|---|---|---|---|---|---|---|
| VT_A | default | 58.8547 | ok | 812 | 0 | 297 |
| VT_A_cse | CSE | 54.4320 (last iterate) | Restoration_Failed | 1767 | 136 | 781 |
| VT_C23 | monotone, 23-state only | 58.9446 | ok | 1180 | 273 | 390 |
| VT_L | MC64 off | killed | stalled at iteration 267, inf_du 1.6e15 (its 7-state stage ended in restoration failure) | 267 | 29 | 296 |

Shipped tyre (the degenerate problem), Sturn N=18:

| run | change | lap [s] | status | iterations | resto | wall [s] | lin. solve [ms/it] |
|---|---|---|---|---|---|---|---|
| A_base | default | 25.5744 | ok | 3209 | 202 | 319 | 86 |
| B_noscale | nlp_scaling_method none | 81.4867 | max_iter | 6000 | 3790 | 477 | 67 |
| C_monotone | mu_strategy monotone | 26.1853 | ok | 424 | 12 | 52 | 93 |
| D_lbfgs | L-BFGS | 27.2655 | max_iter | 6000 | 0 | 232 | 24 |
| E_mumps | linear_solver mumps | 32.7601 | ok | 1562 | 139 | 516 | 316 |

Reading the tables:

* **The optimum is flat.** The 32 Sturn runs span 18.0073 to 18.0497 s (42 ms, 0.23%) while their profiles differ by up to 0.95 m/s in vx and 3.5 m in n; the tol-1e-6 run is not the lowest-objective run. Lap differences below ~0.04 s between option sets are noise.
* **Iteration counts are path-dependent.** Eight pairs that differ only by a 1-ulp Hessian change (CSE on or off) differ by -36% to +40% in iterations (257/238, 231/244, 501/321, 294/370, 262/300, 196/275, 261/242, 243/301). `A_fixed`, `I_fixed`, `K_blas1` and `F_fixed` follow one identical path, so the solver is deterministic and scatter appears only when the problem or options change. Trust consistent directions and per-iteration costs, not single-run differences.
* **Consistent results.** L-BFGS needs 3 to 4x the iterations and its 3x cheaper linear solve does not repay it (total wall 64 s vs 33 s; its 7-state stage takes 3813 iterations). Monotone mu is higher than adaptive in every comparison (371, 501, 370 vs 257; 1180 vs 812 on VirtualTrack). MC64 off saves 35% per iteration (48 vs 73 ms) but stalls on VirtualTrack. `tol` 1e-6 costs +38% iterations for 0.6 ms of lap time.
* **Linear solver**, per-iteration cost (path-independent): ma27 72 ms, ma57 73, MUMPS 93 (+26%), ma97 104 (+42%). ma27 is equivalent to ma57 here; ma97 and MUMPS are slower in wall (43 and 42 s vs 33 s). Tested on Sturn only.
* **Other.** `mu_init` is ignored by the adaptive strategy; `OPENBLAS_NUM_THREADS=1` changes nothing; `bound_relax_factor` 0 gives +14% iterations; the 7-state stage matters (`P_*`: changing only its options moves the 23-state result by -18% to +33% iterations and up to +34 ms of lap).
* **Screening preset (F2).** 196 vs 257 iterations (-24%) at +6.3 ms in the head-to-head run, but its path diverges from the default from iteration 0, and the four F2 pairs (with and without CSE, with and without MC64) give -24%, +16%, -7% and +0.3% in iterations, with lap changes of +6.3, -4.4, +0.5 and +11.0 ms. A fifth pair, measured later on the full `tyre_set='MF205'` (default options vs `screening=True`, both cold; pre-fix NLP), gives 489 vs 290 iterations (-41%) and 18.0093 vs 18.0295 s (+20.2 ms; wall 40 s vs 29 s). Across the five pairs the preset moved iterations by -41% to +16% and the lap by -4 to +20 ms. Sturn only.

**Recommendation.**

1. Keep the current defaults (adaptive mu, exact Hessian, gradient-based scaling, ma57 with MC64, `tol` 1e-4): no alternative was consistently better, and the default 23-state solve converged on every circuit tried (Sturn, VirtualTrack, BCN uniform and curvature).
2. Keep `userOpts(screening=True)` (F2 tolerances) as an opt-in looser stopping rule for sweeps and ranking, and do not promise a saving (best observed: -41% on the `tyre_set='MF205'` pair, -24% on the best 5-coefficient pair; worst: +16%). The shipped `SCREENING_IPOPT` sets the five tolerances only; the study run also set `acceptable_iter` 5 and `max_iter` 1500, but it ended on the optimal criteria, so neither acted.
3. Never disable MC64 (`ma57_automatic_scaling=no`).
4. CSE stays opt-in.
5. Linear solver choice is second order: ma57 default (ma27 equivalent), MUMPS remains the automatic fallback.
6. Do not adopt L-BFGS, monotone mu, `nlp_scaling_method=none` (both stages: +45% iterations; shipped tyre: `max_iter`), `bound_relax_factor=0` or `tol` 1e-6.

## Model observations

* **EM4 power cap.** In the four-motor branch `vehModel.py:504-505` computes motor speed as wheel speed / `gear`, while the single-motor branch (`vehModel.py:486`) uses wheel speed x `gear`; the per-motor power cap is therefore ~52x looser (gear^2, gear 7.252). `functions/ggv.py` mirrors it (`Pmax*gear^2/v`).
* **`brkB` and `Tdist` are inert with ATD On** (`vehModel.py:492-496`: the ATD branch uses only `ATD_*`), yet both are default parameters of `MLTP_paramOptim.py`.
* **Unsprung-mass load accounting.** `vehModel`'s unsprung equation subtracts `vp.mus*g` (the total unsprung mass) at every corner, so the static tyre-load sum is ~26.0 kN, not `m*g` = 20.5 kN (`functions/ggv.py` header).
* **Aero balance.** The front/rear Cl split has no steady-state effect on axle loads in the 23-state model (the pitch balance acts on `fzs`, which already contains the lift terms; the split only changes body attitude, and aero load reaches the axles in the CG ratio). Only total downforce matters in steady state.
* **Hairpin track** (`userOpts.py`): 73 vs 146 trailing zeros in MATLAB; port defect, not fixed. `vp.CG_p_deg_per_deg_table` is `None` (dormant; `CamberGain='Table'` would raise).
* **Benchmark note.** The corrected-tyre runs used five coefficients, i.e. without `pVy2`, `pKy6`, `pVy3`, `pVy4` (camber-only or small); `tyre_set='MF205'` applies all nine.
* **Tyre set in result files and warm starts.** Result file names now carry `_CopyB` and `_mesh<Name>` suffixes for a non-default `tyre_set` or an explicit `mesh` (`functions/importfile.result_stem`). A warm start from a result solved with a different tyre set is refused and falls back to the cold 7-state init, because a CopyB optimum used as an MF205 start sat in IPOPT restoration for 4000+ iterations.

## Decisions (taken 2026-10-04)

1. **Tyre default flipped to `tyre_set='MF205'` (was Copy-B).** Rationale: the MATLAB reference gives 18.008 to 18.022 s and Python MF205 18.009 s (489 iterations pre-fix, 257 with the five-coefficient proxy, instead of 3209). `tyre_set='CopyB'` stays selectable as the legacy shipped set (`pKy4 = 0`) for reproducing old results; all pre-flip Python results came from a zero-cornering-stiffness car. The flip changes every baseline (Sturn 25.57 s to 18.009 s).
2. **Mesh default `mesh='auto'`.** Resolved in `userOpts` after the track is loaded: `'curvature'` when the track length (`track.s[-1] - track.s[0]`) is at least 2000 m (BCN-size circuits), `'uniform'` otherwise (Sturn and the other short synthetic tracks). An explicit `mesh='uniform'` or `mesh='curvature'` still overrides; `ctx.mesh` holds the resolved value and `ctx.mesh_requested` the requested one. `'auto'` only redistributes the knots: N stays round(L / `OPT_ds`), so BCN at the default `OPT_ds` 30 gets a curvature mesh with N=155, the same interval count as before. A reviewer run of the default call (ma57, full `tyre_set='MF205'`) measured auto on BCN at `OPT_ds` 30: 614 iterations, 116.523 s, 566 s solve (pre-fix NLP) against uniform N=155's 852, 117.42 s, 968 s (five-coefficient corrected tyre), i.e. 1.7x wall at equal N, and its 7-state init now converges (1072 iterations, Optimal; the uniform-mesh init ended `Error_In_Step_Computation`). ZigZag (auto resolves to curvature, N=79): `Solve_Succeeded`, 481 iterations, 56.216 s; Jarama and Spa also resolve to curvature under auto and are unmeasured. Rationale: better accuracy at equal N (Sturn N=18: lap error +0.16% curvature vs +0.72% uniform, 240 vs 257 iterations). The measured BCN 2.5x (117.40 vs 117.42 s, 392 s vs 968 s wall, 497 vs 852 iterations) was curvature at `OPT_ds` 45 (N=103) against uniform at `OPT_ds` 30 (N=155); to get the interval reduction pass `OPT_ds=45` or larger with the curvature mesh, e.g. `MLTP(circuit='BCN', OPT_ds=45)`. The evidence covers two circuits only, and warm-start chains must keep the resolved mesh fixed.
3. **CSE stays opt-in.** A flip would first need the NLP to be robust to 1-ulp perturbations: CSE ended `Restoration_Failed` on VirtualTrack (after 1767 iterations; the default solved in 812) even with the corrected tyre. Re-test after the model review.

## Next steps

_As written on 2026-10-04. Since then the model review pass and items 8 and 9 are done; see [Update 2026-10-05: follow-ups](#update-2026-10-05-follow-ups) and the roadmap's Phase 2 status._

* Carry the decisions above through code, tests and docs (default `tyre_set='MF205'`, `mesh='auto'`, CSE opt-in); no owner decision is pending.
* Model review pass on the observations above (EM4 cap, inert `brkB`/`Tdist`, unsprung-mass loads, aero balance).
* Item 8, sweep orchestrator: `MLTP_screen.screen_sweep` to screen, then confirm the top-k with warm-started `MLTP` runs chained on a fixed mesh.
* Item 10, ladder (QSS rung 0, 7-state, 23-state), consuming items 5 and 7.
* Re-measure the screening preset on VirtualTrack and BCN, and the other study defaults with `tyre_set='MF205'` (nine coefficients; only the Sturn default and screening pair, 489 vs 290 iterations (pre-fix), is measured so far), before quoting any iteration saving.
* Item 11 (free-line QSS) if the +3.9% to +11.1% screen error, mostly the fixed line, is too coarse for ranking setups; items 9 and 12 to 14 unchanged; item 6 dropped.

## Update 2026-10-05: follow-ups

Same machine and tools as above (IPOPT 3.14.11, CasADi 3.7.2, HSL ma57 with MC64, `tyre_set='MF205'`, `mesh='auto'`), measured on 2026-10-04 and 2026-10-05 after the model review pass. The sections above keep their original numbers, all measured before the NLP fix described here; where they disagree, this section is current.

### Model audit verdicts

`vehModel.py` was compared with `vehModel.m` (MATLAB R2025a + CasADi 3.7.2, the same normalised point, `Powertrain.m` + `vehParams.m` defaults = `tyre_set='MF205'`, Static aero). It reproduces every reference value bit for bit at the pinned point and at 14 random points (single motor, ATD, EM4, AALB at the static wing angles). `test_vehmodel_matlab.py` pins the values and the four relations below. All four "Model observations" above are inherited from `vehModel.m`: they are kept for parity and are not port bugs.

| observation (2026-10-04) | verdict |
|---|---|
| EM4 per-motor power cap ~52x looser | inherited: EM4 motor speed is `Om_wheel/gear` (single motor: `gear*mean(Om)`), so the per-motor power and rpm rows are gear^2 ~ 52.6x looser and never bind |
| unsprung-mass load accounting | inherited: every unsprung corner subtracts the total `vp.mus`, so the static tyre loads sum to `(ms + 4*mus)*g` = 26.0 kN, not `m*g` = 20.5 kN (planar dynamics use `ms`) |
| front/rear Cl split without steady-state effect | inherited: aero load reaches the axles in the ratio `l_r/l : l_f/l`; the split only changes the body attitude |
| `brkB` / `Tdist` inert with ATD On | inherited: the `ATD_*` inputs split the drive torque and `2*T_brake` |

Still open: the active-aero inputs are live in `vehModel.py` and dead in `vehModel.m` (AALB with wing inputs other than the static angles differs from MATLAB), and the synthetic Hairpin track still has 73 trailing zeros instead of 146. A deliberate physics change must update the test pins, `functions/ggv.py` (`load_model='vehModel'`) and CLAUDE.md together.

### Two NLP port bugs (fixed in commit 37e1e4e)

The audit found no model bug, but the 23-state NLP assembly differed from `MLTP.m` / `vehModel.m` in two places. `test_mltp_constraints.py` pins both fixes.

1. **Input-rate bounds were not divided by `u_s`.** The NLP bounds the rate of the normalised inputs, and `vehModel.m` (L377-379) divides the physical Nm/s, rad/s and deg/s limits by the input scales; the port applied the physical values to the normalised rates. Steering was held to 0.061 rad/s (documented: 0.1 rad/s) while the motor, brake and wing rates were 602x, 4000x and 10-30x looser than specified. Now `m.duk_lb/ub = ctx.duk_* / u_s`. Steering is divided by the model's own scale `delta_max` (0.1 rad/s); `vehModel.m` divides by a `delta_s = pi/8` leaked from `vehModel_initial.m` (0.156 rad/s as MATLAB runs), so matching that instead would be a one-line owner decision in `vehModel.py`.
2. **Friction-circle path rows for every `TyreModel`.** `MLTP.m` adds the four friction-circle rows (`rho_lim_*` <= 1) only under `TyreModel='PureSlip'`. The default `'CombinedSlip'` has powertrain rows only (nh = 3 / 4 / 12 for ATD Off / ATD On / EM4; PureSlip 7 / 8 / 16). The default Sturn NLP now has MATLAB's size (n_w = 1812, n_g = 1904).

Default call (ma57, MC64, `tyre_set='MF205'`, `mesh='auto'`), before and after the fix:

| case | before the fix | after the fix |
|---|---|---|
| Sturn, N=18 uniform | 489 iterations, 18.0093 s | 179 iterations, 18.0086 s, ~16 s of solve |
| BCN, N=155 (auto = curvature) | 614 iterations, 116.523 s, 566 s of solve | 247 iterations, 116.441 s, ~250 s of solve |
| BCN, AALB | not measured | 241 iterations, 116.065 s |

MATLAB gives 18.008 s on Sturn (tol 1e-8). Iterations fall 2.7x (Sturn) and 2.5x (BCN); the laps move by -0.7 ms and -82 ms (the 0.1 rad/s steering bound is active on BCN). The QSS screen does not use the NLP and is unchanged (Sturn 18.744 s, BCN 129.467 s), so its error against the NLP laps is now +4.1% and +11.2% (+3.9% and +11.1% against the pre-fix reference laps in the QSS section). The `Results/` baselines were regenerated under the fixed NLP; the `*_CopyB.mat` files remain as legacy pre-fix results.

### Design co-optimisation finding

Sturn, default parameters (`brkB, Tdist, alpha_FL/FR/RW/TW`), after the fix:

| start | iterations | lap |
|---|---|---|
| 7-state init (the old path) | 220 | 18.244 s |
| the fixed-parameter full solution, with duals (`optimise_design(warm_start=<full result>)`, commit f729753) | 5 | 18.008 s |

The fixed-parameter solution (18.0086 s) is a feasible point of the joint problem, so the 18.244 s run from the 7-state init is a worse local optimum, 0.235 s above it. Started from the converged full solution the joint solve takes 5 iterations (26 primal-only, same point) to 18.00805 s, 0.5 ms below the fixed-parameter lap, and that result is the committed `Results/Sturn_paramOptim.mat` baseline (commit 281c9c3). On this case the warm start is therefore the more robust way to run the co-optimisation as well as the faster one. `MLTP_TyreOptim` (`Fz0_shift`, 121 iterations from the 7-state init) gives 15.768 s with `Fz0_shift` at its lower bound 0.5, so that result is bound-limited, not an interior optimum.

### Tests

There are now 27 `test_*.py` files (21 at the Phase 1 docs commit, 24 at the NLP fix; `test_paramoptim_warmstart.py`, `test_setup_sweep.py` and `test_refine.py` were added since). All 27 pass with casadi (2026-10-05, about 2 minutes in total, 72 s of it `test_setup_sweep.py`), and 23 of the 27 exit 0 with casadi blocked (four of those only print SKIP; `test_casadi_opts.py`, `test_hsl.py`, `test_mltp_params.py` and `test_setup_sweep.py` need it). Only `test_setup_sweep.py` (section 5) and `test_paramoptim_warmstart.py` (section 4) run the real 23-state NLP through IPOPT.

### Landed since, and still open

Items 8 (setup sweep orchestrator) and 9 (adaptive mesh refinement) landed after this review; their measured outcomes are in the roadmap's [Phase 2 status (2026-10-05)](speed_capability_roadmap.md#phase-2-status-2026-10-05). Still open from the lists above: the CSE re-test on VirtualTrack (decision 3 waited for the model review, now done) and the screening-preset re-measure on the fixed NLP.
