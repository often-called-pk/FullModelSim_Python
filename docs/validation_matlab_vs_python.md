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
- Python hook: NLP function and bounds kept on ctx after build (never saved to .mat).
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
   at N=155): speed verdict may not favour port. Re-measured here, same laptop.
2. Python never solved N 465/514 with MUMPS: memory unknown (3-5 GB at N=155 ma57). Watch run 1.
3. NBR never solved by either code; uniform 7-state init may fail (Python uniform BCN init once
   ended `Error_In_Step_Computation`).
4. Path sensitivity: same binaries, different SX op order, iterates may diverge. Judge by
   cross-evaluation, not identical iterates.
5. No mesh-converged truth (regularisation depends on N): reference pinned to fixed NLP.
6. MATLAB timer excludes init, `elapsedTime(4)` includes NLP build: runner times externally and
   reads `solver.stats()`.
7. pi/8 default inherits MATLAB reporting quirk: `u_opt` steering row = delta_n * pi/8, physical
   steer = delta_n * 35 deg (use `data.vehicle`). Documented laps and `Results/` stale until Phase 5.
8. NBR file provenance unknown, already public in `circuits/`.
9. Parity proves fidelity, not physics: inherited quirks (sa_rr uses toe_front vehModel.m:544,
   EM4 motor speed via gear, unsprung mass sum) stay in both.
10. Laptop thermal drift: alternate order, report spread.

## 14. Results

Pending. Phase 1 gate numbers, then Phase 2-3 tables (lap, gaps, cross-feasibility, RMS, timers).
