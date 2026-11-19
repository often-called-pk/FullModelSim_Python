# ChassisSim phase 2b plan: third element, steer camber, bump steer, rubber hysteresis, stability index

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** owner ask 2026-11-18 after phase 2: build the five items phase 2 left out. Spec: `docs/superpowers/specs/2026-11-18-chassissim-integration-design.md`. Same constraints as phase 2 (global.md of the phase 2 SDD folder). Every law comes from a teammate decode; check each against a ChassisSim log where one exists; a law no log can check gets synthetic tests only and says so.
**Order:** T12 third element, T13 steer camber, T16 stability index, T14 bump steer, T15 rubber hysteresis (decode order may reorder), V4 validation solves, T17 docs + final review.

## Global constraints
- Phase 2 global.md holds (worktree, branch `chassissim-cars`, venv, commit identity, no push, ponytail, STE, no vendor data or vendor file names in tracked files: tests find cars by folder glob, one solve at a time with a memory check, snapshot none after every task, fake IPOPT in tests).
- Zero cost when unused: a car without the feature builds the same graph (skip rule). The GP3 gate stays (fixed line within 2 % of 76.742 s).
- Smooth NLP only. A ChassisSim state machine becomes a smooth law or a new state, never `if_else`.

## Rulings
- Ruling (D17 by owner ask, assumption): the stability index ships as an output channel for every car (diagnostic, checked against the GP3 log channel `StabIND`, corner median -9 to -11 %) and as the apex-speed map correction in the QSS when a car holds the map record. The NLP is not derated by it (an optimal-control lap has no apex-speed rule). Cost if wrong: owner wants an NLP speed cap; add it as an opt-in later.
- Ruling: rubber hysteresis in the NLP is a smooth approximation of ChassisSim's two-threshold switch (exact form after the decode: a blend by the damper velocity sign, or one smooth state per corner). The QSS uses the loading curve (quasi-static: no unloading). Say so in the docstring.
- Ruling: the F2 at Adelaide stays a non-gate check until phase 3 adds the bump profile (ChassisSim runs it far under its grip there).

---

### Task 12: Third element (heave spring, damper, rubber)
**Files:** `functions/chassissim.py` (record, refusals), `vehModel_cs.py`, `functions/ggv.py`. Tests: `tests/test_cs_third.py`.
- [ ] Law from the decode: heave travel of an axle (from the two wheel or damper travels), third MR (cids 29, 30), spring (5, 6), damper (35), rubber gap (11, 12) with its flags (208, 209) and table (17, 18); force back to the two wheels. Remove the third-spring and third-rubber refusals.
- [ ] Checks: F3 builds; F2 builds; F2 log (Adelaide): rear third rubber engaged in about 47 % of the lap above 28 m/s, max about 569 kgf, rear ride height at 70+ m/s about 19.3 mm (model without it 13.3 mm); QSS uses the same law.

### Task 13: Camber from steer (castor, kingpin)
**Files:** `vehModel_cs.py`, `functions/ggv.py`. Tests: `tests/test_cs_kinematics.py` (add).
- [ ] Law from the decode (castor = tyre scalar 2; GP3 front 9 deg). Dynamic camber of a steered wheel adds the steer term, sign per side.
- [ ] Check: GP3 log front camber vs steer slope +-0.1506 deg per deg (sin 9 deg = 0.1564): within 5 %. QSS: the camber at the apex steer (state the steer the QSS uses).

### Task 16: Stability index
**Files:** `vehModel_cs.py` (channel), `MLTP.py` (veh_syms), `functions/ggv.py` (map correction), `functions/chassissim.py` (map record). Tests: `tests/test_cs_stbi.py`.
- [ ] Per-step stbi from the decode (axle cornering stiffness basis, % normalisation, sign); output channel `stbi` for a car (and the 4EM car if the inputs exist, else ChassisSim cars only).
- [ ] QSS: apex speed x map(average stbi of the corner) when the car holds the map record (none shipped: synthetic test).
- [ ] Check: GP3 fixed-line result vs log `StabIND` along s: median and corner medians within 2 points.

### Task 14: Bump steer
**Files:** `functions/chassissim.py` (drop the refusal), `vehModel_cs.py`. Tests: add to `tests/test_cs_kinematics.py`.
- [ ] Toe change vs wheel travel (cids 74-77, units and sign from the decode) added to the static toe of that corner, in the slip angle. Skip when the table is zero (all shipped cars).
- [ ] Synthetic tests only (no shipped car holds a non-zero table): sign, units, the skip rule.

### Task 15: Rubber hysteresis
**Files:** `functions/chassissim.py` (cids 657-664 per corner), `vehModel_cs.py`, `functions/ggv.py`. Tests: `tests/test_cs_rubber.py` (add).
- [ ] Decode split (enable, loading curve, unloading curve, thresholds) and switching law; smooth form per the ruling; skip when disabled (all shipped cars).
- [ ] Synthetic tests: loading vs unloading force at the same compression, thresholds, skip rule.

### Validation V4 (Opus)
- [ ] GP3 fixed line again (gate holds), F3 at its own circuit if it has one (no log: build and converge), F2 at Adelaide fixed line (report only), a synthetic-feature car (bump steer + hysteresis on) on Sturn converges.

### Task 17: Docs and final review
- [ ] CLAUDE.md car paragraph (laws, refusals removed), tests list, CONTEXT.md terms if new. STE lint. Final whole-branch review (Opus) and fix wave (Fable).
