# Rolling-start fix plan (B1 follow-up)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** rolling start warns before a likely infeasible solve; cold rolling seed puts chassis rows at the pinned equilibrium.
**Architecture:** two local edits in `MLTP.py` (rolling block, `_init_guesses`), fake-solver tests, then Opus validation solves + docs.
**Tech stack:** Python, CasADi, IPOPT (MUMPS on this box), numpy.
**Spec:** no separate spec. Binding evidence = section "Evidence" below (debug run wf_4aa3e6d4-f63: investigator + skeptic). Raw notes: `/tmp/phase2_scratch/rolling_dbg/challenge.md`.

## Global constraints
- Python `/home/pk/.venvs/fullmodelsim/bin/python`, cwd = worktree root `.claude/worktrees/physics-fixes`, branch `sim-phase2`. Commit per task: `git -c user.name="Prashant Kumar" -c user.email="prashant.jps@gmail.com" commit`. No attribution lines. No push.
- Defaults byte-identical: `start_state='matlab'` and `'periodic'` builds unchanged. NLP snapshot `/tmp/phase2_scratch/nlp_snapshot.py <worktree> <out.json> /tmp/phase2_scratch/nlp_base.json` prints "SNAPSHOT DIFF: none". Parity guards `tests/test_vehmodel_matlab.py`, `tests/test_matlab_parity.py` green, asserts untouched.
- `MLTP.py` keeps 4 `ca.Function` calls (`tests/test_casadi_opts.py`). No new module (`setup_sweep._CODE_FILES` unchanged).
- No IPOPT in build/test steps: fake solver only (`tests/test_start_wiring.py` helpers `fake_ipopt`, `quiet`, `run_stub`). Solves only in Task 2, Opus, one at a time, foreground or one background job, `free -g` available >= 6 GB, MUMPS, `save=False`.
- Real-IPOPT test files (`test_setup_sweep`, `test_ladder`, `test_paramoptim_warmstart`, `test_refine`, `test_hsl`): Opus only, one file at a time, never next to a solve.
- No em dash anywhere. ASCII. Warning text, comments, docs: STE (short sentences, active voice, no semicolons). Ponytail + karpathy surgical diffs.
- Tests: plain scripts, `import _bootstrap` first, top-level `ok()`. Extend `tests/test_start_wiring.py` (new section 12), no new test file. Haiku edits only that file. Sonnet never edits tests.

## Evidence (binding)
- Corrected Sturn vi 70 rolling (V3b cold, V3c seeded) = infeasible as posed. Frontier solve (max vx(0) under the rolling rows 1-22): 68.499 / 68.528 m/s from 2 seeds < vx box edge 69. Only the vx(0) box widened to [59, 71]: Solve_Succeeded 348 its, vx(0) 65.475 interior. No solver or transcription bug.
- QSS braking limit `qss_profile(ctx)["v_bwd"][0]`: corrected 66.87 m/s, matlab 70.79 m/s (Sturn centreline). Conservative vs the NLP frontier by ~1.6 m/s. Flags only corrected vi 70 among V1, V1c, V2, S3, S4.
- Matlab start at vi 70 (V3a) converges only via a pre-braked off-equilibrium start (rear wheels locked, wu on bounds).
- Cold rolling fragility (V1c: 530 its, 18.611 s vs seeded V1 95 its, 18.085 s) = chassis seed. `warmstart_guesses` seeds rows 9-22 quasi-statically (zs = theta = 0, zt = W0/kt, no aero); rolling pins them at the equilibrium; matlab chassis modes are slow (tau 8-35 s). Rows 9-22 seeded at `Xi` everywhere: 149 its, 18.06169 s (no clip); 132 its, 18.06556 s (with knot-0 clip).
- Knot-0 / knot-N clip: no effect on the outcome, moves default seed bytes (synthetic INIT vx(0) 0.29 outside the matlab box; stored inits 6.66e-9 outside). Rejected.

## Decisions (rulings)
- Ruling: approved item "cold-seed fix" = rows 9-22 equilibrium seed, not the knot-0 clip. Why: clip has no effect (149 vs 132 its, same optimum) and breaks default bytes. Cost if wrong: rolling seeds keep knot-0 offsets <= 0.02 scaled, no measured effect.
- Ruling: precheck = warning, not error. Why: QSS limit conservative ~1.6 m/s, feasible starts exist up to 68.5 m/s on Sturn. Cost if wrong: an infeasible solve still runs to max_iter after the warning.
- Precheck compares the vx box lower edge `vi - Xi_tol[0]*x_s[0]` with the centreline `v_bwd[0]`. Local variable only: no hand-off to the ladder `prof` (`MLTP.py:420-431` resets it and records `qss_lap_s`).
- Skipped: `warmstart_refined` rows 9-22 override (refine reseed), add when rolling + refine is used. qss23 + rolling solve check, add when used. Braking-trim rolling start, add when starts inside braking zones are needed.

## Review focus
1. Default bytes: the seed override sits behind `ctx.start_state == "rolling"`. Snapshot "none".
2. Both `_init_guesses` returns (qss branch `return g`, legacy `return warmstart_guesses(...)`) get the override. init7 warm starts route through it too.
3. Warning: fires for corrected vi 70 rolling; silent for matlab vi 70 rolling, corrected vi 60 rolling, corrected vi 70 matlab start.
4. Override shapes: `x0` (nx, N+1) and `xc0` (nx, N*d), scaled = `Xi[9:23] / x_s[9:23]`.
5. Docs keep the label "the old 62 m/s infeasibility claim is wrong" and never write "limited to about 67 m/s".

---

### Task 1: rolling start: braking-limit warning + chassis seed at the equilibrium (Sonnet build, Haiku test, Opus verify)

**Files:**
- Modify: `MLTP.py` rolling block (after `ctx.Xi[idx] = X0[idx]`, ~:405) and `_init_guesses` (~:422-463, both returns).
- Test: `tests/test_start_wiring.py`, new section 12 before the final done print.

**Interfaces:**
- Consumes: `ladder.qss_profile(ctx)` (imported in MLTP.py) -> dict with `"v_bwd"` (m/s, fine grid, `[0]` = s 0). `m.x_s` (23,). `ctx.Xi` (23,) physical, rows 9-22 filled by `rolling_start`. `ctx.Xi_tol` (23,) scaled half-widths or None. Guess dict keys `"x0"` (nx, N+1), `"xc0"` (nx, N*d), scaled.
- Produces: warning text containing "braking limit". Rolling cold guesses: rows 9-22 == `ctx.Xi[9:23] / m.x_s[9:23]` at every knot and collocation point.

- [ ] **Step 1 (Haiku): tests, section 12**, fake solver, DATA_AA guard as the file does:
  1. `MLTP(circuit="Sturn", physics="corrected", vi=70.0, start_state="rolling", mesh="uniform", save=False, plot=False)` under `fake_ipopt` + `warnings.catch_warnings(record=True)`: one warning contains "braking limit". Same call with (physics "matlab", vi 70, rolling), (physics "corrected", vi 60, rolling), (physics "corrected", vi 70, start_state "matlab"): no warning contains "braking limit".
  2. Rolling build `MLTP(circuit="Sturn", start_state="rolling", mesh="uniform", save=False, plot=False)` captured (`rec["x0"]` is the 23-state w0 or use the file's capture helper): rows 9-22 of the knot and collocation state blocks == `Xi[9:23]/x_s[9:23]` (max abs diff <= 1e-15).
  3. Matlab-start build, same call without `start_state`: quasi-static seed kept (`ladder.quasi_static_states`, ladder.py:110-132, is constant along the lap in rows 9-22): rows 9-18 == 0 at every knot and collocation point, rows 19-22 == `quasi_static_states(ctx.vp, [1.0], ctx.physics)[14:18, 0] / m.x_s[19:23]` (abs diff <= 1e-15). Unpack w0 with `functions/sensitivity.nlp_index` or the file's capture helper.
  4. Rolling with an init7 warm start (`warm_start={"init": <init from a fake MLTP_initial>}` or a saved init file in a temp dir): rows 9-22 == `Xi/x_s` as in 2.
- [ ] **Step 2 (Sonnet): code**
  - a. In `if ctx.start_state == "rolling":` after `ctx.Xi[idx] = X0[idx]`:
    `v_brk = float(qss_profile(ctx)["v_bwd"][0])`, `lo = ctx.vi - ctx.Xi_tol[0] * m.x_s[0]`; if `lo > v_brk`: `warnings.warn(...)` with text verbatim:
    `f"start_state='rolling': the vx start box begins at {lo:.1f} m/s. This is above the QSS braking limit {v_brk:.1f} m/s at s = 0. The NLP can be infeasible. If IPOPT does not converge, use a lower vi or start_state='matlab'."`
  - b. In `_init_guesses`, one small local helper applied at both returns: if `ctx.start_state == "rolling"`: `g["x0"][9:23] = (ctx.Xi[9:23] / m.x_s[9:23])[:, None]` and the same for `g["xc0"]`. One STE comment line: the quasi-static seed rows miss the pinned start, and the matlab chassis modes are slow.
- [ ] **Step 3 (Opus verify):** review diff vs this task; run section 12 + whole `tests/test_start_wiring.py`, `tests/test_start_state.py`, `tests/test_mltp_warmstart.py`, `tests/test_casadi_opts.py`, `tests/test_output_channels.py`, parity guards, then alone: `tests/test_ladder.py`, `tests/test_refine.py`; snapshot "none". Fix, rerun, commit `fix(start): warn when the rolling start box is above the braking limit; seed the chassis rows at the rolling equilibrium`.

### Task 2: validation solves + docs (Opus)

- [ ] **Solves** (one at a time, MUMPS, `save=False`, record status / iterations / lap / wall):
  - V0 `MLTP(circuit="Sturn", save=False, plot=False)` cold: 152 its, 18.018569 s exactly (default guard).
  - V1c `MLTP(circuit="Sturn", start_state="rolling", save=False, plot=False)` cold: expect ~149 its, ~18.062 s (S8c). Pass: GOOD_STATUS, lap <= 18.10 s, start rows in their boxes.
  - S3 `MLTP(circuit="Sturn", physics="corrected", vi=60.0, start_state="rolling", save=False, plot=False)` cold: report vs 270 its, 20.02972 s.
  - V1 `MLTP(circuit="Sturn", start_state="rolling", warm_start=STURN, ...)`: full-interp path, report (expect unchanged 95 its, 18.08490 s).
  - V3b: no solve. Zero-solve: the warning fires (section 12 test 1).
- [ ] **Docs** (`asd-ste100` loaded; lint each changed doc file, fix hard hits in new text):
  - `CLAUDE.md` start-state bullet: replace "Open defect (B1) ... Cause unknown ... Until fixed" with the measured cause (frontier 68.5 m/s < box edge 69; QSS 66.9 conservative; matlab start pre-braked), the warning, the chassis seed and the new V1c numbers. Physics bullet: keep "wrong" label, add "a coast rolling start on Sturn reaches at most 68.5 m/s".
  - `docs/laptime_sim_improvements.md` status of 19 October: B1 finding + open-items lines updated, V1c after the fix.
  - Commit `docs: rolling start cause and fix (braking limit, chassis seed)`.

### Final review (Opus)
Whole-branch review of the Task 1-2 commits: spec + quality, the ledger rulings, residual risks.
