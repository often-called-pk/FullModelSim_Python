# MATLAB vs Python MLTP validation (BCN + NBR)

Status: plan, 2026-10-07. Results appended per phase in section 14.

## 1. Purpose

- Prove Python port solves same NLP as MATLAB original and lands on same answer.
- Measure speed and accuracy of both, two real tracks: Barcelona (BCN), Nurburgring GP (NBR).
- Truth = reference bundle per track (section 6), not either code alone.
- Skipped physical validation against real laps, add when measured telemetry exists.

## 2. Facts (2026-10-07)

- Tracks, byte-identical in Python `circuits/` and MATLAB `Circuits/` (sha256):
  - BCN `Barcelona_circuit.mat` a9150a44.., L 4653.6 m.
  - NBR `Nurburgring_circuit.mat` bb6589ab.., GP-Strecke, L 5139.1 m (official 5148 m), 1029 points
    at ~5 m, clockwise, R_min 13.8 m, closure gap 5 m. Provenance unknown, oldest copy 2024-07-18.
    Registered in neither loader before Phase 1.
- Loaders equivalent: s, k used (x, y plots only), no smoothing for real tracks, linear interp,
  uniform knots `linspace(s0, sN, N+1)`, lateral bound n +-4 m.
- N = round(L/OPT_ds) both (MLTP.m:163, transcription.py:68): BCN 465 at 10 m, 155 at 30 m;
  NBR 514 at 10 m, 171 at 30 m.
- Parameters: pt bit-equal; vp 101/102 equal (CG_p table differs, dormant, CamberGain Off both);
  mf 59/59 bit-equal (Python MF205 = MATLAB MF_205_60R15_V91); `DATA_AA.mat` same sha 098f156f..;
  Xi/Xf, rate and regularisation tables equal; Rw/gear quirk same both sides.
- Breaks found: steering scale u_s (MATLAB vehModel.m:318 uses `delta_s = pi/8` leaked from
  vehModel_initial.m, rate bound 0.1556 rad/s; Python used delta_max, 0.1 rad/s; owner call
  2026-10-07: Python adopts pi/8); active aero dead in MATLAB, live in Python (moot: Static).
- NLP layout identical: w = [Xk; Uk; Xkj] (MLTP.m:371, transcription.py:376-390), g order
  (MLTP.m:377, transcription.py:403), J with per-interval regularisation, no dsk (MLTP.m:355,
  transcription.py:350). Same w valid in both at equal N.
- casadi 3.7.2, IPOPT 3.14.11, MUMPS 5.4.1 both sides. MATLAB R2025a, run headless (`matlab -batch`).
- MATLAB MLTP.m: runs MLTP_initial inline (:27); saves no .mat (global `data`, Plots txt, SDI);
  no IPOPT stats; tic starts after 7-state init (:33).
- Python MLTP.py: saves .mat with `data.nlp`; timing only in memory (`ctx.elapsed`,
  `ctx.solve_stats`); solver object dropped after solve.
- Recorded numbers:
  - Sturn N=18 (same laptop, 2026-10-04): MATLAB tol 1e-8 18.008274 s, 299 it, 23.2 s solve;
    Python default 18.00859 s, 179 it, 14.9 s.
  - BCN MATLAB as shipped (N=465, 2025-08-04, machine unknown, init excluded): 116.2025 s,
    solve 244.8 s, NLP build 37.7 s.
  - BCN Python default (N=155 curvature, 0.1 rad/s steering): 116.44076 s, 247 it, IPOPT 232.9 s.
  - No Python solve at N 233-514; no NBR solve anywhere.
- Laptop: Ryzen 9 5980HS 8C/16T, 31.4 GB. Repo public; all 12 `circuits/` files (NBR included)
  already tracked and pushed.

## 3. Settings: MATLAB as shipped vs Python default

| Setting | MATLAB (userOpts.m) | Python default | Parity tier (Python) |
|---|---|---|---|
| OPT_ds | 10 (:189) | 30 | 10 |
| mesh | uniform (MLTP.m:166) | auto: curvature if L >= 2000 m | uniform |
| tol | 1e-8 (:200) | 1e-4 | 1e-8 |
| acceptable_tol | 1e-6 (:201) | 1e-3 | 1e-6 |
| mu_strategy | unset = monotone | adaptive | monotone |
| linear_solver | unset = MUMPS | ma57 + MC64 scaling | mumps |
| steering u_s | pi/8 (0.1556 rad/s) | pi/8 from Phase 1 (was delta_max) | same |
| active aero inputs | dead | live | moot (Static) |
| CG_p table | set | None | dormant, listed |
| 7-state init | inline every run | ladder legacy, same path | same |
| 23-state seed, states 9-22 | OPT_e (MLTP.m:229-251) | quasi-static (suspension 0, zt static) | OPT_e, validation runner only |
| result, timing | workspace + txt, no stats | .mat, no timing | runners capture |

Equal: max_iter 6000, fixed_variable_treatment make_constraint, mu_init 0.1, bound_push and
bound_frac 1e-2, constr_viol/dual_inf/compl_inf tol 1e-4, OPT_d 3 Legendre, linear inputs,
OPT_e 1e-2, vi 60, ni free, Xi/Xf, rate and regularisation tables, Static / ATD On / EM4 Off /
CombinedSlip, MF205.

Parity call: `MLTP(circuit, OPT_ds=10, mesh='uniform', tol=1e-8, linear_solver='mumps',
ipopt_overrides={'acceptable_tol': 1e-6, 'mu_strategy': 'monotone'})`.

## 4. Gate 1: parameter parity (before any solve)

- Dump both sides, validation config (BCN and NBR; Static, ATD On, EM4 Off, CombinedSlip, vi 60):
  vp, pt, mf (59), aero file sha + scalars, track sha, Xi/Xf, OPT_*, N, x_s/u_s, x/u limits,
  scaled rate bounds duk, path bounds h_lb/h_ub, full IPOPT options.
- Diff exact; floats 1e-12 rel. Every hit fixed or listed in section 14 with reason.
- Expected listed hits: IPOPT defaults (differ by design, parity tier overrides), CG_p (dormant).
- Tools: `validation/matlab_batch.py` (MATLAB export), `validation/py_export.py`,
  `validation/compare.py`; regression `tests/test_matlab_parity.py` section 1 (stored MATLAB
  JSON in `tests/data/`).
- Reused base: scratch `dump_params.m`, `audit_compare.py`, `matlab_ref` harness (2026-10-04).

## 5. Gate 2: NLP-function equivalence (no solve)

- Both codes build same NLP at parity settings, IPOPT max_iter 0.
- Bounds lbw, ubw, lbg, ubg: exact.
- f, g at 3 shared w: MATLAB start point w0, MATLAB solution w*, seeded random point near w0
  (clipped to bounds): |a - b| <= 1e-10 * max(1, |a|) per element.
- BCN/NBR w* exists only after Phase 2: Phase 1 runs w0 + random, w* added in Phase 2.
  Sturn N=18 (MATLAB w* from 2026-10-04 run) = permanent regression, test section 2.
- Python hook: env `MLTP_KEEP_NLP=1` keeps NLP function and bounds on ctx (never saved to .mat).
- Info check: start point w0 equal with parity seed (states 9-22 at OPT_e).
- Skipped gradient/Hessian comparison, add when f, g pass but solves disagree.

## 6. Reference bundle (single source of truth)

Per track, `Results/validation/reference/<track>/`:

1. Inputs: track sha, DATA_AA sha, both parameter dumps, gate 1 diff (empty or listed).
2. Gate 2 numbers.
3. Reference solution: lowest-lap converged run among MATLAB as shipped and Python parity that is
   a valid optimum in both codes (max constraint violation <= 1e-6 in other code's NLP,
   objective within 1e-9 rel).
4. `summary.json`: lap, objective, N, settings, source run, hashes.

Every result, Python production included, scored against item 3.

- Skipped Richardson mesh-converged truth: regularisation summed per interval without dsk, so
  objective depends on N (0.26-0.34% of lap); add when regularisation gets dsk scaling.
- Skipped forward re-integration of controls, add when discretisation accuracy itself questioned.

## 7. Run matrix (per track)

| Tag | Code | Settings | Reps |
|---|---|---|---|
| mat_ship | MATLAB | as shipped | 3 |
| py_par | Python | parity call (section 3) | 3 |
| py_prod | Python | defaults: OPT_ds 30, mesh auto, tol 1e-4, ma57, adaptive, pi/8 | 3 |
| py_prod8 | Python | py_prod with MATLAB's stopping rule: tol 1e-8, acceptable_tol 1e-6 | 3 |
| xcheck | both | each w* evaluated in other code's NLP (no solve) | 1 |
| xseed | both | only if lap gap > 0.01%: seed each code from other's w* | 1 each |

- Order: mat_ship r1, py_par r1, py_prod r1, then r2, then r3.
- Files: `Results/validation/<track>_<code>_<tier>_r<k>.mat` + `.json` (gitignored).
- Python production BCN run doubles as new BCN baseline (Phase 5).
- Skipped other configs (ATD Off, EM4, AALB, PureSlip), add when default config passes and they matter.
- Skipped MATLAB with ma57, add when verdict needs solver-matched speed.
- Skipped sector times, add when lap gap needs localising (BCN_S1-S3 splits exist).

## 8. Acceptance

Parity (mat_ship vs py_par, per rep pair, and reference):

- Lap |dT| <= 0.01% of lap (BCN ~12 ms) AND cross-feasible (max violation <= 1e-6, objective
  within 1e-9 rel).
- Gap > 0.01%: xseed. Both stay put within 5 iterations: distinct local optima, documented.
  Else bug hunt.
- Reported, flagged not failed: RMS on 1 m grid of vx, n, steering (normalised), torques;
  flag RMS vx > 0.1 m/s or RMS n > 0.05 m.

Production (py_prod vs reference): signed lap error, sanity band 0.5%.

Basis: Sturn 2026-10-04 gap 0.3 ms; BCN cold-start scatter ~10 ms; distinct optima 13-33 ms apart.

## 9. Speed protocol

- Per run: model build, 7-state init (wall, iterations), NLP build, 23-state IPOPT (wall,
  iterations, s per iteration, function-eval share from `solver.stats()`), peak working set,
  end-to-end wall (MATLAB: runner times `matlab -batch`, startup reported apart).
- Idle laptop on AC, other tool closed, codes alternated, 3 reps: median (min, max).
- MATLAB plots and SDI cut from runner copy, not timed.
- Headline speed = py_prod8 vs mat_ship: same stopping rule (tol 1e-8, acceptable_tol 1e-6), only
  mesh and linear solver differ; reported with its lap cost vs reference.
- py_prod (tol 1e-4) = loose stopping rule, exploration setting: second row only, never a headline.
- Parity tier (py_par vs mat_ship) = only solver-vs-solver speed comparison.

## 10. Budget and failure policy

- ~10 solves per track, ONE at a time, foreground. Before each: no stray python or MATLAB process.
- Cap 30 min per solve: IPOPT `max_wall_time` 1800 s in both runners, hard kill at 45 min.
- On cap or failure: retry case once in BOTH codes at OPT_ds 20 (BCN N=233, NBR N=257); then seed
  from other code's w*; then report.

## 11. Deliverables

- This doc: plan now, results in section 14.
- `validation/`: `matlab_batch.py` (pristine copy, patch, `matlab -batch`), `matlab/*.m` drivers,
  `py_export.py`, `run_py.py`, `compare.py`.
- `tests/test_matlab_parity.py`: gates 1 + 2, no solve, MATLAB data in `tests/data/`.
- `Results/validation/` gitignored. MATLAB repo untouched; scratch copies under
  `C:\Users\ASUS\.claude\jobs\9b76333d\tmp\validation\`.

## 12. Phases

| Phase | Work | Builds | Checks |
|---|---|---|---|
| 0 | this plan | Opus | - |
| 1 | no solves: steering pi/8 default + pins; NBR registration; MATLAB batch infra; gate 1; NLP hook; gate 2 (Sturn full, BCN/NBR w0 + random); runners + compare, smoke at max_iter 0 | Sonnet, Opus orchestrates | Fable audits gates |
| 2 | BCN run matrix, xcheck, reference bundle | Sonnet runs, one solve at a time | Fable audits numbers |
| 3 | NBR, same | same | same |
| 4 | results section, verdict | Opus | Fable |
| 5 | regenerate `Results/` baselines and documented laps (pi/8 default) | Sonnet | Fable |

## 13. Risks

1. MATLAB BCN as shipped (245 s at N=465, unknown machine) matches Python default (233 s IPOPT
   at N=155): speed verdict may not favour port. Re-measured here, same laptop (resolved: section 14,
   equal speed at equal settings; 2.1-4.8x at equal tolerance from mesh and linear solver).
2. Python never solved N 465/514 with MUMPS: memory unknown (3-5 GB at N=155 ma57). Watch run 1.
3. NBR never solved by either code; uniform 7-state init may fail (Python uniform BCN init once
   ended `Error_In_Step_Computation`).
4. Path sensitivity: same binaries, different SX op order, iterates may diverge. Judge by
   cross-evaluation, not identical iterates.
5. No mesh-converged truth (regularisation depends on N): reference pinned to fixed NLP.
6. MATLAB timer excludes init, `elapsedTime(4)` includes NLP build: runner times externally and
   reads `solver.stats()`.
7. pi/8 default inherits MATLAB reporting quirk: `u_opt` steering row and plotSDI trace =
   delta_n * pi/8 = 9/14 of physical steer delta_n * 35 deg; `data.vehicle` has no steering
   channel. Compare steering as normalised delta_n. Skipped physical steering channel, add when a
   plot needs true steer. Documented laps and `Results/` stale until Phase 5.
8. NBR file provenance unknown, already public in `circuits/`.
9. Parity proves fidelity, not physics: inherited quirks (sa_rr uses toe_front vehModel.m:544,
   EM4 motor speed via gear, unsprung mass sum) stay in both.
10. Laptop thermal drift: alternate order, report spread.

## 14. Results

### Phase 1 (2026-10-07, no solves)

Code changes: steering u_s = pi/8 default (owner call, vehModel.py:169); state bounds formed as
x_lim * (1/x_s) like vehModel.m:130-138 (was 1 ulp off on wheel speeds, vehModel.py:88); 'NBR'
registered; parity seed `matlab_seed()` in validation tooling only.

Gate 1, parameter parity (MATLAB export vs Python, 1e-12 rel):

| Track | Hits | Info |
|---|---|---|
| Sturn ds30 | vp.CG_p_deg_per_deg_table (MATLAB 14 values, Python None; dormant, CamberGain Off) | print_timing_statistics Python only |
| BCN ds10 | same single hit | same |
| NBR ds10 | same single hit | same |

Everything else equal: vp (102), pt, mf (59), aero, Xi/Xf, OPT_*, N, x_s/u_s (steering pi/8 both),
x/u limits, scaled duk, h bounds and names, ru/rdu/rdu2, track sha, DATA_AA sha.

Gate 2, NLP-function equivalence (strict, IPOPT max_iter 0 both):

| Track | n_w / n_g | Bounds | w0 (parity seed) | max scaled diff f / g | Python build, peak commit |
|---|---|---|---|---|---|
| Sturn ds30 | 1812 / 1904 | bit-exact | bit-exact | 0 / 4.1e-15 (F(w*) = 18.054014070188256 both) | 3.4 s, 3.3 GB |
| BCN ds10 | 46065 / 47945 | bit-exact | bit-exact | 1.0e-15 / 2.7e-13 | 81 s, 7.7 GB |
| NBR ds10 | 50916 / 52992 | bit-exact | bit-exact | 2.2e-15 / 2.9e-13 | 82 s, 8.3 GB |

BCN/NBR w* column follows in Phase 2-3. MATLAB export (startup + build): BCN 96 s, NBR 97 s.
Peak commit 7.7-8.3 GB for a build alone: full N=465-514 solves likely 8-12 GB, one at a time.

### Phase 2: BCN (2026-10-07, one solve at a time)

Runs in order. Its = 7-state + 23-state IPOPT iterations. Build = 23-state model + NLP build.
End-to-end = external wall of the whole run (MATLAB incl. startup).

| Run | Status | Lap [s] | Its | 23-state IPOPT [s] | s/it | Build [s] | 7-state wall [s] | End-to-end [s] | Peak |
|---|---|---|---|---|---|---|---|---|---|
| mat_ship r1 | Solve_Succeeded | 116.21127 | 2185 + 284 | 654.3 | 2.30 | 66.0 | 384.2 | 1120 | WS 5.2 GB |
| py_par r1 | Solve_Succeeded | 116.22177 | 1788 + 302 | 819.8 | 2.71 | 60.0 | 301.8 | 1186 | WS 3.5 GB, commit 8.0 GB |
| py_prod r1 | Solve_Succeeded | 116.44407 | 1072 + 165 | 110.2 | 0.67 | 20.1 | 99.9 | 233 | WS 1.3 GB, commit 6.4 GB |
| mat_ship r2 | Solve_Succeeded | 116.21127 | 2185 + 284 | 764.0 | 2.69 | 67.1 | 372.3 | 1219 | WS 5.3 GB |
| py_par r2 | Solve_Succeeded | 116.22177 | 1788 + 302 | 856.5 | 2.84 | 61.3 | 306.0 | 1229 | WS 3.5 GB, commit 8.0 GB |
| py_prod r2 | Solve_Succeeded | 116.44407 | 1072 + 165 | 113.6 | 0.69 | 20.9 | 109.0 | 247 | WS 1.3 GB, commit 6.4 GB |
| mat_ship r3 | Solve_Succeeded | 116.21127 | 2185 + 284 | 826.2 | 2.91 | 67.3 | 375.4 | 1286 | WS 5.3 GB |
| py_par r3 | Solve_Succeeded | 116.22177 | 1788 + 302 | 769.7 | 2.55 | 61.2 | 301.3 | 1137 | WS 3.5 GB, commit 8.0 GB |
| py_prod r3 | Solve_Succeeded | 116.44407 | 1072 + 165 | 113.5 | 0.69 | 21.1 | 103.7 | 241 | WS 1.3 GB, commit 6.4 GB |
| py_prod8 r1 | Solve_Succeeded | 116.44626 | 967 + 172 | 138.4 | 0.80 | 22.7 | 91.3 | 256 | WS 1.2 GiB, commit 6.0 GiB |
| py_prod8 r2 | Solve_Succeeded | 116.44626 | 967 + 172 | 134.0 | 0.78 | 22.2 | 89.0 | 249 | WS 1.2 GiB, commit 6.0 GiB |
| py_prod8 r3 | Solve_Succeeded | 116.44626 | 967 + 172 | 138.4 | 0.80 | 23.9 | 91.2 | 257 | WS 1.2 GiB, commit 6.0 GiB |

Reps deterministic per code (same iterations, same lap to all printed digits); only wall times vary.
MATLAB IPOPT wall drifted 654 -> 764 -> 826 s over the session (thermal), so medians are used.

**BCN verdict: PASS.**

- Parity, all 3 rep pairs: 116.21127 (MATLAB) vs 116.22177 s (Python), gap 10.5 ms = 0.0090%, rule 0.01%.
- Cross-check ship r1 / par r1, PASS both ways: Python NLP at MATLAB w*: bound viol 8.8e-9, constraint
  viol 9.9e-9, f 116.329963605 equal (rel 0); MATLAB NLP at Python w*: 8.8e-9 / 9.9e-9, f 116.338613427
  equal (rel 1.2e-16). Same numbers as each code's self-check, so gate 2 holds at w* too.
- Nearby distinct optima: MATLAB objective lower by 0.0087. The 7-state inits take different paths
  (2185 vs 1788 iterations), so the 23-state starts differ. Profile RMS vs reference: vx 0.085 m/s,
  n 0.129 m (flagged, > 0.05 m), delta_n 0.0048, T_motor 26 Nm, T_brake 109 Nm. Gap < 0.01%: no xseed.
- Reference bundle `Results/validation/reference/BCN/`: BCN_mat_ship_r1, lap 116.211267 s,
  f 116.329964, N 465.
- Production (N=155 curvature, tol 1e-4, ma57): 116.44407 s, +0.2328 s = +0.200% vs reference (band
  0.5% ok); RMS vx 0.54 m/s, n 0.69 m (coarse mesh). Its result in standard form for Phase 5:
  `Results/validation/baseline_BCN/BCN_Static_ATDOn_EM4Off.mat`.

Speed, median (min, max) of 3 reps:

| Metric | MATLAB as shipped | Python parity | Python prod8 (tol 1e-8) | Python prod (tol 1e-4) |
|---|---|---|---|---|
| End-to-end [s] | 1219 (1120, 1286) | 1186 (1137, 1229) | 256 (249, 257) | 241.5 (233.5, 246.7) |
| Lap vs reference | reference | +0.0090% | +0.202% | +0.200% |
| Startup / import [s] | 7.8 | 0.9 | 0.9 | 0.9 |
| 7-state init: its, wall [s] | 2185, 375 | 1788, 302 | 967, 91 | 1072, 104 |
| Model + NLP build [s] | 67.1 | 61.2 | 22.7 | 20.9 |
| 23-state: its, IPOPT wall [s] | 284, 764 (654, 826) | 302, 820 (770, 857) | 172, 138 (134, 138) | 165, 113.5 (110, 114) |
| IPOPT s per iteration | 2.69 | 2.71 | 0.80 | 0.69 |
| Function-eval share | 7.3% | 7.3% | 8.6% | 9.5% |
| Peak working set [GiB] | 4.89 | 3.30 (commit 7.45) | 1.18 (commit 6.01) | 1.18 (commit 5.98) |

BCN speed verdict:

- Headline, equal stopping rule (tol 1e-8, acceptable_tol 1e-6; only mesh and linear solver differ):
  Python prod8 256 s vs MATLAB as shipped 1219 s end to end = 4.8x faster, lap cost +0.202% (0.235 s)
  vs reference, mostly mesh (N=155 curvature vs N=465 uniform).
- Loose stopping rule, exploration setting (tol 1e-4): 241.5 s, +0.200%. Not a speed claim.
- Solver vs solver (parity tier, identical settings, the only like-for-like comparison): equal speed
  (2.71 vs 2.69 s per iteration, 1186 vs 1219 s end to end, within path noise).
### Phase 3: NBR (2026-10-07, one solve at a time)

Ledger (resume point): order mat_ship r1, py_par r1, py_prod r1, then r2, r3; then xcheck (ship r1 /
par r1), reference, runs; then py_prod8 r1-r3 for NBR and BCN (`run_py.py --tier prod8`), commit per
track. Commands: section 7 tags via `validation/matlab_batch.py solve` and
`validation/run_py.py` (see BCN). mat_ship r1 attempt 1 killed by a forced agent handback, rerun.
Status: Phases 2-3 complete incl. py_prod8 on both tracks. Next: Phase 4 (verdict), Phase 5 (baselines).

| Run | Status | Lap [s] | Its | 23-state IPOPT [s] | s/it | Build [s] | 7-state wall [s] | End-to-end [s] | Peak |
|---|---|---|---|---|---|---|---|---|---|
| mat_ship r1 | Solve_Succeeded | 125.15790 | 923 + 274 | 701.3 | 2.56 | 67.1 | 175.7 | 965 | WS 5.4 GiB |
| py_par r1 | Solve_Succeeded | 125.15202 | 1555 + 329 | 858.0 | 2.61 | 62.8 | 295.7 | 1222 | WS 3.7 GiB, commit 7.8 GiB |
| py_prod r1 | Solve_Succeeded | 125.10282 | 773 + 369 | 274.1 | 0.74 | 20.4 | 80.7 | 379 | WS 1.3 GiB, commit 6.1 GiB |
| mat_ship r2 | Solve_Succeeded | 125.15790 | 923 + 274 | 721.9 | 2.63 | 66.4 | 172.3 | 977 | WS 5.3 GiB |
| py_par r2 | Solve_Succeeded | 125.15202 | 1555 + 329 | 896.4 | 2.72 | 61.7 | 292.4 | 1256 | WS 3.7 GiB, commit 7.8 GiB |
| py_prod r2 | Solve_Succeeded | 125.10282 | 773 + 369 | 322.4 | 0.87 | 24.1 | 88.0 | 438 | WS 1.3 GiB, commit 6.1 GiB |
| mat_ship r3 | Solve_Succeeded | 125.15790 | 923 + 274 | 847.6 | 3.09 | 81.6 | 214.8 | 1166 | WS 5.3 GiB |
| py_par r3 | Solve_Succeeded | 125.15202 | 1555 + 329 | 1015.8 | 3.09 | 68.7 | 360.6 | 1452 | WS 3.7 GiB, commit 7.8 GiB |
| py_prod r3 | Solve_Succeeded | 125.10282 | 773 + 369 | 311.2 | 0.84 | 24.3 | 89.3 | 428 | WS 1.3 GiB, commit 6.1 GiB |
| py_prod8 r1 | Solve_Succeeded | 125.12154 | 793 + 436 | 364.2 | 0.84 | 23.7 | 88.0 | 480 | WS 1.3 GiB, commit 6.1 GiB |
| py_prod8 r2 | Solve_Succeeded | 125.12154 | 793 + 436 | 342.1 | 0.78 | 22.4 | 87.2 | 456 | WS 1.3 GiB, commit 6.1 GiB |
| py_prod8 r3 | Solve_Succeeded | 125.12154 | 793 + 436 | 342.9 | 0.79 | 24.7 | 93.8 | 465 | WS 1.3 GiB, commit 6.1 GiB |

Reps deterministic per code again; wall times drifted up over the session (thermal).

**NBR verdict: PASS.**

- Parity, all 3 rep pairs: 125.15790 (MATLAB) vs 125.15202 s (Python), gap 5.9 ms = 0.0047%, rule 0.01%.
  This time Python holds the lower optimum.
- Cross-check ship r1 / par r1, PASS both ways: Python NLP at MATLAB w*: bound viol 7.2e-9, constraint
  viol 9.5e-9, f 125.249955688 equal (rel 2.3e-16); MATLAB NLP at Python w*: 7.2e-9 / 9.5e-9,
  f 125.248054447 equal (rel 1.1e-16). The Python JSON's inf_pr 2.8e-3 is the last IPOPT trace entry,
  not the final point: direct evaluation gives 9.5e-9.
- Profile RMS of MATLAB vs reference: vx 0.085 m/s, n 0.124 m (flagged), delta_n 0.0035, T_motor 23 Nm,
  T_brake 26 Nm. 7-state inits again take different paths (923 vs 1555 iterations).
- Reference bundle `Results/validation/reference/NBR/`: NBR_py_par_r1, lap 125.152023 s, f 125.248054,
  N 514.
- Production (N=171 curvature, tol 1e-4, ma57): 125.10282 s, -0.0492 s = -0.039% vs reference (band ok;
  faster lap on the coarser mesh); RMS vx 0.48 m/s, n 0.71 m.

Speed, median (min, max) of 3 reps:

| Metric | MATLAB as shipped | Python parity | Python prod8 (tol 1e-8) | Python prod (tol 1e-4) |
|---|---|---|---|---|
| End-to-end [s] | 977 (965, 1166) | 1256 (1222, 1452) | 465 (456, 480) | 428.5 (378.6, 438.2) |
| Lap vs reference | +0.0047% | reference | -0.024% | -0.039% |
| 7-state init: its, wall [s] | 923, 176 | 1555, 296 | 793, 88 | 773, 88 |
| Model + NLP build [s] | 67.1 | 62.8 | 23.7 | 24.1 |
| 23-state: its, IPOPT wall [s] | 274, 722 (701, 848) | 329, 896 (858, 1016) | 436, 343 (342, 364) | 369, 311 (274, 322) |
| IPOPT s per iteration | 2.64 | 2.73 | 0.79 | 0.84 |
| Peak working set [GiB] | 5.32 | 3.67 (commit 7.83) | 1.30 (commit 6.07) | 1.30 (commit 6.07) |

NBR speed verdict:

- Headline, equal stopping rule (tol 1e-8, acceptable_tol 1e-6; only mesh and linear solver differ):
  Python prod8 465 s vs MATLAB as shipped 977 s end to end = 2.1x faster, lap cost -0.024% (0.030 s)
  vs reference.
- Loose stopping rule, exploration setting (tol 1e-4): 428 s, -0.039%. Not a speed claim.
- Solver vs solver (parity tier, identical settings, the only like-for-like comparison): equal speed
  per iteration (2.73 vs 2.64 s); Python's end-to-end 1256 s is longer only through path noise
  (7-state init 1555 vs 923 iterations, 23-state 329 vs 274).
