# Finish the partly done items plan (A4, A5, C2, C6, D1, E1, E3)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** close the open part of every partly done item in `docs/laptime_sim_improvements.md` with the least code and zero cost at the defaults.
**Architecture:** reuse existing helpers (`vehModel._mf_scale`, `track_qa.k_from_xy`/`check_track`, `refine.eval_states`/`nlp_inputs`, `optimise_design`, the parity identity filter); every new knob defaults to today's behaviour; one new module only (`functions/telemetry.py`).
**Tech stack:** Python, CasADi, numpy, scipy.
**Spec:** owner request of 2026-10-20 ("finish off all the partly done tasks ... minimal code with the fastest performance", ponytail) + item texts in `docs/laptime_sim_improvements.md` + research Table 5 (`docs/tyre_scaling_track_method.md`). Design notes, reference diffs and draft tests from the read-only mapping run: `/tmp/phase2_scratch/partials/` (`aero.md`, `power.md` + `power.patch` + `power_test_a5.py`, `tyre_grad/` with `c6.diff`, `c2.diff`, `c6_checks.py`, `c2_check.py`, `track_chan.md` with `track_build.py`, `e1_proto.py`, `test_track_build_proto.py`, `test_channels_proto.py`). The diffs were made on 185331b: rebase by hand, never apply blind.

## Global constraints
- Python `/home/pk/.venvs/fullmodelsim/bin/python`, cwd = worktree root `.claude/worktrees/physics-fixes`, branch `finish-partials`. One commit per task: `git -c user.name="Prashant Kumar" -c user.email="prashant.jps@gmail.com" commit`, STE body, no attribution lines, no push.
- Ponytail: shortest correct diff, reuse before writing, no new dependency, one new module (`functions/telemetry.py`), no speculative knobs beyond the item text. Zero cost at defaults: no extra NLP rows, variables or SX nodes unless a knob is set.
- matlab defaults byte-identical: `/tmp/phase2_scratch/nlp_snapshot.py <wt> <out> /tmp/phase2_scratch/nlp_base.json` prints "SNAPSHOT DIFF: none" after every task. Parity guards `tests/test_vehmodel_matlab.py`, `tests/test_matlab_parity.py` green, asserts untouched.
- corrected defaults: `/tmp/phase2_scratch/partials/power_nlp_snapshot_corrected.py <wt> <out> <base>` prints "none" after every task except Task 3 (A4 changes corrected by design). Base: `/tmp/phase2_scratch/partials/power_nlp_corr_base.json` for Tasks 1-2; Opus writes a new base after Task 3 verify (`/tmp/phase2_scratch/partials/corr_base_after_a4.json`) for Tasks 4-7.
- Parity export (`validation/py_export.params`): a new key is dropped only while it and its group are at identity; groups are joint (MF scales: LMUX LMUY LFZO LKX LKY; aero scales: Cd_scale Cl_scale).
- No IPOPT in build and test steps (fake solver, numpy). Real-IPOPT test files (`test_setup_sweep`, `test_ladder`, `test_paramoptim_warmstart`, `test_refine`, `test_hsl`) and every solve: Opus only, one at a time, `free -g` available >= 4 GB (Sturn) or >= 6 GB (BCN), no other fullmodelsim python.
- Tests: plain scripts, `import _bootstrap` first, top-level `ok()` asserts, DATA_AA SKIP guard for model sections. Haiku edits only the task's test files. Sonnet never edits tests. Lift the draft tests; keep only checks that kill a mutant (ponytail: one runnable check per logic branch).
- No em dash. ASCII. Never copy `/home/pk/Downloads/pk-backup-d-drive/IRP/MLTP_Python/*.csv` (another team's telemetry).

## Evidence (binding, from the mapping run)
- C6 (PAC2002 eq 16, 25, 37; MFeval 4.E1, 4.E2a, 4.E15, 4.E25; OpenTire Ky0/Kx): `Fz0' = LFZO*Fz0` feeds dfz and Kya; `Kxk *= LKX`; `Kya *= LKY`; B = K/(C D) follows. LFZO = a equals `vp_overrides Fz0 = a*4905` in the full model and the QSS exactly. LKX, LKY have no QSS site (peak-only QSS). Fz0_shift / TyreOptim coupling untouched.
- E3: scale TOTAL coefficients: `Cd = (Cd0 + RW_Cd)*Cd_scale`; all four of Cl_front, Cl_rear, Cl_left, Cl_right `*= Cl_scale` (scaling only front/rear cancels). Both physics. 7-state ignores. The `Cd`, `Cl` vp fields are MATLAB placeholders read only by the 7-state model and only without DATA_AA.mat: label them, do not make them live.
- A4: rear-wing drag acts at `vp.hw` (1.28 m, existing field), body drag at hcg (MATLAB split, `vehModel.m:480-481`, `MLTP_paramOptim.m:209`). Corrected only: add `- q*_mf_scale(RW_Cd, vp.Cd_scale)*(vp.hw - vp.hcg)` to the corrected dthetadot bracket, and the matching axle shift `dFz = q*Cd_rw*(hw - hcg)/L` in `ggv.build_envelope` (corrected only). Corrected QSS lap: Sturn -0.003 %, BCN -0.012 %; corrected coast pitch at 60 m/s 2.2569 -> 2.0886 mrad.
- A5: keys `gear, Tmax, Pmax, Preg, Ppack` via `vp_overrides`, routed in vehParams (`PT_KEYS`), corrected only (ValueError otherwise), set only when given. Regen: torque lower bound -Tmax, power rows ub 1 + Preg/Pmax, BrTh ub inf, energy: regen credit eff*|P|. Pack: one row `(Ppack - sum P)/Ppack in [0, inf]` only when set; QSS `P_mot = min(P_mot, Ppack)`; QSS regen adds `min(F_mot_trq, P_regen/v)` to braking below v_rpm.
- C2: `dJ*/dp = gradient(f + lam_g.g, w)[P rows]` at the stored optimum; f does not depend on P; stored lam_g reusable (KKT residual 5.8e-7 Sturn); AD vs central difference rel <= 7.7e-8; derived vp fields give silent zeros, so a guard refuses them.
- E1: channels at collocation points need `refine.nlp_inputs` (the NLP near-hold inputs), never `u_full` (linear: ax off rms 0.70 m/s^2). `t_full[::d+1] == t_opt` exactly with a Lagrange-quadrature `time_at_points`. Monster file (vendor PDF `CSim_monster_file_creation.pdf`): 11 columns, tab, no header, 50 Hz: distance m, rpm, ay g, ax g, damper FL FR RL RR mm, steer deg at tyre, throttle %, speed km/h. Sign of ay / steer not stated by the vendor.
- D1: `k_from_xy` already conserves heading on closed tracks. Missing: GPS projection (WGS84 tangent plane), duplicate-fix drop, overshoot trim, gap and winding guards, optional smoothing, writer. BCN rebuilt from its own x,y: turning -1.000000000, QSS +4.1 % vs the parity file (not registered).

## Decisions (rulings)
- Ruling: A4 uses the MATLAB split (wing drag at hw, body at hcg), no new field. Cost if wrong: add `h_drag` later.
- Ruling: A4 changes corrected results by design; Task 3 updates the corrected pitch pin `tests/test_start_wiring.py` (2.2-2.3 -> 2.0-2.1 mrad) and re-bases the corrected snapshot. Cost if wrong: none, the value is the model equilibrium.
- Ruling: E3 Cd / Cl fields get honest labels, not live behaviour. Cost if wrong: a script can still pass them silently.
- Ruling: A5 knobs only through `vp_overrides`, corrected only, no GUI rows. Cost if wrong: GUI users wait for F2.
- Ruling: C6 builds LFZO too (MF-standard name, redundant with the Fz0 override). Cost if wrong: two routes to one effect.
- Ruling: C2 uses a `build_only` hook (+9 lines, opt-in) and makes MF keys promotable (2 lines). Cost if wrong: a reviewer must read the hot-builder branch.
- Ruling: D1 registers no circuit; `smooth_xy` is built because GPS curvature is unusable without it. Cost if wrong: 8 lines unused until GPS data arrive.
- Ruling: E1 monster sign default `(-1, -1)` (right positive, from the E4 note that ChassisSim takes right-turn curvature as positive), unverified, a parameter. The 1 m export is an own documented CSV. Cost if wrong: the user flips one argument.
- Skipped: aeromap stage (Skipped list), side-force roll moment (zero for Static), GUI rows for A5 (F2), a fit tool for the new factors, CLI for the track builder, strain columns. Add when their triggers in the notes fire.

## Review focus
1. Defaults: matlab snapshot none after every task; corrected snapshot none except Task 3.
2. Placement: C6 per PAC2002; E3 all four Cl corners; A4 sign (rearward force above CG = nose up); A5 bounds only, no new rows unless Ppack.
3. E1 uses `nlp_inputs`, and `vehicle[k] == vehicle_full[k][::d+1]` bitwise.
4. C2 guard refuses derived fields and mismatched results; gradients match central differences.
5. D1 refuses gaps, figure-eights and QA errors before writing.

---

### Task 1: C6 LFZO, LKX, LKY (Sonnet build, Haiku test, Opus verify)
**Files:** `vehParams.py` (MF defaults), `vehModel.py` (dfz via Fz0p, `_kxk`, `_kya`), `functions/ggv.py` (`Fz0*LFZO`), `app/vp_params.py` (rows), `validation/py_export.py` (joint MF_SCALES filter); tests `tests/test_mf_scaling.py` (extend), `tests/test_vp_params.py` / `tests/test_mainwindow.py` labels. Reference: `tyre_grad/c6.diff`, `c6_checks.py`.
- [ ] Step 1 (Haiku): extend `tests/test_mf_scaling.py` per `c6_checks.py`: defaults 1.0 both sets; LFZO = 0.9 equals Fz0 override 0.9*4905 (full model and QSS lap exactly); LKX / LKY slope x1.2 at zero slip (pHy1 = pHy2 = 0), peaks unchanged; QSS blind to LKX / LKY; export joint filter both directions.
- [ ] Step 2 (Sonnet): code per Evidence, reuse `_mf_scale`.
- [ ] Step 3 (Opus): both snapshots none; run test_mf_scaling, test_vp_params, test_mainwindow, test_vp_overrides, test_screen, test_tyreoptim_fz0, parity guards. Commit `feat(tyre): MF scaling factors LFZO, LKX, LKY (default 1) per PAC2002`.

### Task 2: E3 Cd_scale, Cl_scale + honest Cd / Cl labels
**Files:** `vehParams.py` (primaries, comments), `vehModel.py:247-251`, `functions/ggv.py` aero_coefficients, `app/vp_params.py` (rows + placeholder labels), `validation/py_export.py`; test `tests/test_aero_scale.py` (new, from the draft, sections 1-4 and 6). Reference: `aero.md`, `aero.diff`.
- [ ] Step 1 (Haiku): defaults / override / registry; drag x1.2, every corner lift x0.85 (rel 1e-12), side force unchanged; QSS anchored to vehModel at scales; Cd / Cl placeholders inert with DATA_AA; export joint filter.
- [ ] Step 2 (Sonnet): code per Evidence.
- [ ] Step 3 (Opus): both snapshots none; tests as Task 1 plus test_aero_scale, test_presets, test_setup_sweep (alone). Commit `feat(aero): Cd_scale and Cl_scale (default 1); label the Cd and Cl placeholders`.

### Task 3: A4 drag pitch moment (corrected)
**Files:** `vehModel.py` corrected dthetadot, `functions/ggv.py` build_envelope (corrected), `app/vp_params.py` hw tooltip; tests `tests/test_aero_scale.py` section 5, `tests/test_start_wiring.py` corrected pitch bound.
- [ ] Step 1 (Haiku): dthetadot diff vs hw = hcg equals `-q*RW_Cd*(hw-hcg)/I_y` (rel 1e-9 at vx 20, 60); matlab independent of hw; scales with Cd_scale; QSS axle shift equals the rolling_start shift (0.05 N at 30, 50, 70 m/s), sum 0, nose up; corrected QSS lap lower by < 0.01 %. Move the pin to 2.0-2.1 mrad.
- [ ] Step 2 (Sonnet): code per Evidence.
- [ ] Step 3 (Opus): matlab snapshot none; corrected snapshot changes only in the corrected builds: write the new corrected base; run test_aero_scale, test_start_wiring, test_physics_suspension_aero, test_physics_mass_frame, test_screen, parity guards. Commit `feat(physics): rear-wing drag pitch moment under physics='corrected' (A4)`.

### Task 4: A5 per-motor ratings, gear, regen, pack row (corrected)
**Files:** `vehParams.py` (PT_KEYS routing), `vehModel.py:134-143` (torque bounds), `MLTP.py` (regen bounds, pack row, energy), `functions/ggv.py` (P_regen, Ppack, F_brk), docstrings; test `tests/test_power_a5.py` (new, from `power_test_a5.py`). Reference: `power.patch` (rebase onto Tasks 1-3).
- [ ] Step 1 (Haiku): defaults pinned; keys refused under matlab; ratings, gear, regen bounds and energy credit, pack row last with [0, inf] and n_g + (N+1); QSS regen anchored to vehModel wheel torques; QSS pack min.
- [ ] Step 2 (Sonnet): code per Evidence; default paths execute no new branch.
- [ ] Step 3 (Opus): both snapshots none (corrected base after Task 3); run test_power_a5, test_mltp_constraints, test_screen, test_physics_em4_kinematics, test_vp_params, test_runconfig, parity guards. Commit `feat(powertrain): per-motor ratings, gear, regenerative braking and pack power limit under physics='corrected' (A5)`.

### Task 5: C2 setup gradients (zero solves)
**Files:** `functions/transcription.py` (`build_only`), `MLTP_paramOptim.py` (`build_only`, MF keys promotable), `functions/sensitivity.py` (`setup_gradients(data, params, tol=1e-3, **kw) -> dict`); test `tests/test_sensitivity.py` section 8. Reference: `tyre_grad/c2.diff`, `c2_check.py`, `oracle_fd.py`.
- [ ] Step 1 (Haiku): Sturn brkB = Tdist = 0.0 exact (ATD on), wing signs; central-difference oracle on the rebuilt Lagrangian rel 1e-6 for alpha_RW, LMUY, LKY, Fz0_shift, rho, f; ValueError for derived fields, mismatched options and missing data.nlp.
- [ ] Step 2 (Sonnet): code per Evidence; default `build_only=False` leaves the hot path byte-identical.
- [ ] Step 3 (Opus): both snapshots none; run test_sensitivity, test_paramoptim_warmstart (alone), test_transcription, test_warmstart, test_tyreoptim_fz0. Commit `feat(analysis): setup gradients from the stored optimum, zero solves (C2 stage M)`.

### Task 6: E1 channels at collocation points + exports
**Files:** `MLTP.py` (veh_syms om_gear, throttle, xs_*, sf; one f_veh call on all nodes with `nlp_inputs`; `vehicle_full`, `t_full`), `functions/transcription.py` (`time_at_points`), `functions/telemetry.py` (new: `export_monster(data, path, step=0.02, sign=(-1, -1))`, `export_csv(data, path, step=1.0, by="distance")`); test `tests/test_telemetry.py` (new, from `test_channels_proto.py`).
- [ ] Step 1 (Haiku): time_at_points exact; resample keeps input jumps, ax / ay inside node range; monster 11 tab columns, no header, distance = integral of speed; CSV header; wiring on the stored Sturn point with fake IPOPT: `vehicle[k] == vehicle_full[k][::4]` bitwise, `t_full[::4] == t_opt`, ax / ay at nodes equal state-polynomial kinematics (pins `nlp_inputs`).
- [ ] Step 2 (Sonnet): code per Evidence.
- [ ] Step 3 (Opus): both snapshots none; run test_telemetry, test_output_channels, test_results, test_save_load, test_transcription, test_refine (alone). Commit `feat(outputs): channels at the collocation points and ChassisSim / CSV export (E1 stage M)`.

### Task 7: D1 track builder
**Files:** `functions/track_qa.py` (`xy_from_gps`, `smooth_xy`, `write_track`); test `tests/test_track_qa.py` sections 12-15 (from `test_track_build_proto.py`).
- [ ] Step 1 (Haiku): projection lengths and axes; GPS circle CW / CCW turning +-1; duplicate fixes and overlap give +1; 30 % gap and figure-eight raise; BCN rebuilt turning -1, issues [], QSS 134.768 s within 0.1 %; noisy circle with smoothing within 3 %.
- [ ] Step 2 (Sonnet): code per Evidence; no registry change.
- [ ] Step 3 (Opus): both snapshots none; run test_track_qa, test_setup_sweep (alone). Commit `feat(tracks): track builder from x,y or GPS with closure guards (D1 stage M)`.

### Task 8: docs + validation solves (Opus)
- [ ] Solves, one at a time, MUMPS, save=False: corrected Sturn and BCN cold (new corrected numbers after A4); Sturn `LKY = 0.9` warm with duals vs `setup_gradients` first-order prediction; Sturn corrected single motor `Preg = 50e3` warm (regen smoke: lap <= base, min P_motor < 0).
- [ ] `CLAUDE.md`: knobs (LFZO LKX LKY, Cd_scale Cl_scale, A5 keys, drag moment), new helpers (`setup_gradients`, telemetry exports, track builder), test list 44. `docs/laptime_sim_improvements.md`: "Status on 21 October 2026", each item closed with its numbers; move items to done. STE lint. Commit `docs: partly done items closed (A4, A5, C2, C6, D1, E1, E3)`.

### Final review (Opus)
Whole-branch review of Tasks 1-8, rulings, residual risks; one fix wave if needed.
