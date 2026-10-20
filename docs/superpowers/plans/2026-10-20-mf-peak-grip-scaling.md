# MF peak-grip scaling factors plan (C6 part 2, E3 part)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** add the standard Magic Formula scaling factors LMUX and LMUY (default 1) to the full model and the QSS screener. Defaults byte-identical.
**Architecture:** two MF keys in `vehParams` (both tyre sets), applied where MF 5.2 puts the peak-friction scales (D and SV terms) in `vehModel.py` and in `functions/ggv.py` peak friction; Python-level skip at exactly 1.0 so the SX graph does not change; registry rows; parity export drops the factors while they are 1.
**Tech stack:** Python, CasADi, numpy.
**Spec:** owner decisions of 2026-10-19 (add the factors; standard MF definitions; `Fz0_shift` unchanged; full model and QSS, not the 7-state model) + research report `docs/tyre_scaling_track_method.md` Table 5 (add LMUX and LMUY only, default 1; LKX, LKY, LFZO and the Cd / Cl scales are not identifiable from temperature, pressure and lap times, so they stay out).

## Global constraints
- Python `/home/pk/.venvs/fullmodelsim/bin/python`, cwd = worktree root `.claude/worktrees/physics-fixes`, branch `sim-phase2`. Commit per task: `git -c user.name="Prashant Kumar" -c user.email="prashant.jps@gmail.com" commit`. No attribution lines. No push.
- Defaults byte-identical: NLP snapshot `/tmp/phase2_scratch/nlp_snapshot.py <worktree> <out.json> /tmp/phase2_scratch/nlp_base.json` prints "SNAPSHOT DIFF: none". Parity guards `tests/test_vehmodel_matlab.py`, `tests/test_matlab_parity.py` green, asserts untouched (`EXPECTED = set()`).
- `MLTP.py` / `MLTP_initial.py` / `MLTP_paramOptim.py` keep 4 / 4 / 3 `ca.Function` calls. No new module (`setup_sweep._CODE_FILES` unchanged).
- No IPOPT in build/test steps. Real-IPOPT test files (`test_setup_sweep`, `test_ladder`, `test_paramoptim_warmstart`, `test_refine`, `test_hsl`): Opus only, one at a time, `free -g` available >= 4 GB, no other solve.
- No em dash anywhere. ASCII. Comments, docs: STE. Ponytail + karpathy surgical diffs.
- Tests: plain scripts, `import _bootstrap` first, top-level `ok()`, casadi sections SKIP when `Data/DATA_AA.mat` is missing (`tests/test_physics_flag.py:16-19`). New file `tests/test_mf_scaling.py` (Haiku). Sonnet never edits tests.

## Evidence (binding)
- MF 5.2 places the peak-friction scales in the peak and the vertical shift: `mu_x = (pDx1 + pDx2*dfz) * LMUX`, `SVx = Fz*(pVx1 + pVx2*dfz) * LVX * LMUX`; `mu_y = (pDy1 + pDy2*dfz)*(camber term) * LMUY`, `SVy = Fz*((pVy1 + pVy2*dfz)*LVY + (pVy3 + pVy4*dfz)*gamma) * LMUY`. No degressive factor (that is MF 6.1). LVX, LVY are not added (= 1). Source: PAC2002 (= MF 5.2) manual, eqs 20-23, 28, 34-35, 41 and 74-75 (copy at pdfcoffee.com/pac2002-magic-formula-tire-model-pdf-free.html; its text drops the Greek letters, and the leftover tokens read `Fz (pVx1 + pVx2 dfz) lambda_Vx lambda_mux zeta1` and `Fz {(pVy1 + pVy2 dfz) lambda_Vy + (pVy3 + pVy4 dfz) gamma_y} lambda_muy zeta4`). The combined-slip `DVyk = mu_y Fz (...)` carries LMUY through `mu_y`.
- The code builds `Bx = Kxk/(Cx*Dx)` and `By = Kya/(Cy*Dy)` (`vehModel.py:400-401, 452-453`), so a peak scale leaves the slip stiffness unchanged, as MF intends.
- Code sites: `vehModel.py:382-385` (mu_x), `:403` (`_svx`), `:430-433` (mu_y), `:443-446` (Svy with `Svy_gamma_*`); `functions/ggv.py:220-231` (`peak_mu_x`, `peak_mu_y`). `vehModel_initial.py` uses the simplified `vp.tyre`: untouched.
- `tests/test_vp_params.py:13` requires registry groups to cover exactly `PRIMARY_KEYS | MF_KEYS`; `validation/compare._walk` flags every Python-only key, so the parity export must drop the factors while they are 1.

## Decisions (rulings)
- Ruling: LMUX and LMUY are MF keys (`ctx.mf`), not vp primaries. Why: the MF standard keeps scaling factors with the coefficient set (`[SCALING_COEFFICIENTS]` of a TIR file). Cost if wrong: one rename.
- Ruling: skip each factor when it is exactly the float 1.0 (Python branch), so default SX graphs stay byte-identical. A symbolic value (promotion) always multiplies.
- Ruling: `validation/py_export.params` leaves out LMUX and LMUY while both are 1.0 (identity, no MATLAB counterpart), one comment line. Any other value is exported, so the gate flags it.
- Ruling: registry rows in the tyre group with lo 0.5, hi 1.5, step 0.01, so the GUI parameter table lists them like the other MF coefficients. Cost if wrong: two extra rows in that table.
- Skipped: LKX, LKY, LFZO, Cd / Cl scales (report Table 5: not identifiable from the core data). Add each when the data of its Table 5 row ("Better verdict needs") exist. Never fit LFZO from lap times. QSS inversion tool (stint median lap time -> grip scale), add when the first session data exist. Promotion of LMUX / LMUY in `optimise_design`, add when a co-optimisation needs it.

## Review focus
1. Default bytes: snapshot "none"; both tyre sets default to 1.0.
2. Placement: D and SV terms only, both axes, camber part of SVy included; Kxk / Kya untouched.
3. QSS: `peak_mu_x` / `peak_mu_y` scale; envelope and lap move with the factor; at 1.0 identical to default.
4. Parity: gate 1 green with unchanged asserts; the export drops the factors only at 1.0.
5. 7-state model and `Fz0_shift` untouched.

---

### Task 1: LMUX and LMUY in the full model and the QSS (Sonnet build, Haiku test, Opus verify)

**Files:** `vehParams.py` (MF defaults, both sets), `vehModel.py` (sites above), `functions/ggv.py` (`peak_mu_x`, `peak_mu_y`), `app/vp_params.py` (registry rows + meta), `validation/py_export.py` (identity filter). Test: `tests/test_mf_scaling.py` (new).

**Interfaces:**
- Produces: `ctx.mf.LMUX`, `ctx.mf.LMUY` (float, default 1.0); `vp_overrides={"LMUX": x, "LMUY": y}` accepted; model signals `m.mu_*_x`, `m.mu_*_y` include the factors.

- [ ] **Step 1 (Haiku): tests** (`tests/test_mf_scaling.py`, casadi + DATA_AA SKIP guard for model sections):
  1. Defaults: MF205 and CopyB sets give `mf.LMUX == mf.LMUY == 1.0`; `vp_overrides={"LMUX": 0.9, "LMUY": 1.1}` lands in `ctx.mf`; an unknown key still raises.
  2. Full model at LMUX = 0.9, LMUY = 0.8 vs default, at 3 states (wheel slip -6..+8%, slip angles up to 0.15 rad, camber nonzero, both physics): `mu_*_x` ratio 0.9 and `mu_*_y` ratio 0.8 (rel err <= 1e-12); the vertical shifts scale the same way; `Kxk`, `Kya` equal (rel <= 1e-12); the slopes dfx/dkappa and dfy/dalpha at zero slip (central difference, step 1e-6) equal within 1e-6 relative. The force at zero slip itself moves by design, because the vertical shift scales with the factor.
  3. QSS: `peak_mu_x` / `peak_mu_y` scale by the factor; `MLTP_screen(circuit="Sturn", save=False, vp_overrides={"LMUX": 0.95, "LMUY": 0.95})` lap > default lap; with both at 1.0 the lap equals the default exactly.
  4. 7-state model: `vehModel_initial` builds with LMUX = 0.9 and matches the default build (factors unused there).
  5. Parity export: `validation.py_export.params("Sturn", 30)` has no LMUX / LMUY at default; with `vp_overrides={"LMUY": 0.9}` it contains `mf.LMUY` (call shape: read the module).
- [ ] **Step 2 (Sonnet): code** per Evidence and Decisions. One STE comment per site that names the MF 5.2 term. Do not touch `Fz0_shift` logic or the 7-state model.
- [ ] **Step 3 (Opus verify):** review vs this task; run `tests/test_mf_scaling.py`, `tests/test_vp_params.py`, `tests/test_vp_overrides.py`, `tests/test_params_useropts.py`, `tests/test_screen.py`, `tests/test_tyreoptim_fz0.py`, `tests/test_physics_flag.py`, `tests/test_output_channels.py`, parity guards; alone: `tests/test_setup_sweep.py`; snapshot "none". Fix, rerun, commit `feat(tyre): standard MF peak-grip scaling factors LMUX and LMUY (default 1) in the full model and the QSS screener`.

### Task 2: docs (Opus)
- [ ] `CLAUDE.md`: tyre bullet (LMUX / LMUY: MF 5.2 placement, default 1, skipped at 1, full model + QSS, not the 7-state model, parity export drops them at 1, Fz0_shift unchanged); test list 41 + `test_mf_scaling.py`; pointer to `docs/tyre_scaling_track_method.md`.
- [ ] `docs/laptime_sim_improvements.md`: C6 part 2 and E3 status per the research (built: LMUX / LMUY; not built: LKX, LKY, LFZO, Cd / Cl scales, with the reason).
- [ ] STE lint each changed doc file. Commit `docs: MF peak-grip scaling factors and the tyre data method`.

### Final review (Opus)
Whole-branch review of the Task 1-2 commits, rulings, residual risks.
