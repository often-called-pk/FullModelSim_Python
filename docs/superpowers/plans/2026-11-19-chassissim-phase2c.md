# ChassisSim phase 2c plan: powertrain gaps, hybrid layer, GUI on Windows and Linux

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** owner decisions D26-D40 (spec `docs/superpowers/specs/2026-11-18-chassissim-integration-design.md`, grilling 2026-11-19). Decode inputs: `.superpowers/sdd/chassissim/map/teammate_p2c_powertrain.md` (sections 1-7; section 7 corrects the cid labels of section 6).
**Lanes:** G (GUI, packaging, CI) and P (powertrain) touch different files and can run side by side. Order inside each lane as listed. P4 needs P2 (front deploy needs front drive).
**Routing:** Sonnet builds, Opus audits (read-only), Fable fixes audit findings, Haiku writes tests, Opus commits.

## Global constraints
- Phase 2 / 2b rules hold (worktree physics-fixes, branch `chassissim-cars`, venv, commit identity, ponytail, STE, no vendor data or vendor file names in tracked files, no machine paths in tracked files, one IPOPT solve at a time with a memory check, snapshot none after every task, fake IPOPT in tests).
- Zero cost when unused: a car without the feature builds the same graph (hash). The 4EM car and every shipped car stay byte-identical unless the task says so.
- Smooth NLP only. A ChassisSim state machine becomes a smooth law or a new control, never `if_else`.
- No push: the owner pushes. CI runs after the owner's push.

---

### G1: uv packaging, tracked test runner, setup scripts (D29, D30)
**Files:** `pyproject.toml` (new), `uv.lock` (new), `requirements.txt` (exported), `start.py`, `start.bat`, `start.sh` (new), `tests/run_tests.py` (new: the quick list as a tracked runner), README section.
- [ ] `pyproject.toml`: name, python range, dependencies from `requirements.txt` (casadi, numpy, scipy, plotly, kaleido, PySide6), dev group (pyinstaller), script `fullmodelsim = "app.main:main"` (add `main()` if missing). `uv lock`; `uv export` -> `requirements.txt` (header comment says it is generated).
- [ ] `tests/run_tests.py`: runs the plain-script tests, `--quick` (no real IPOPT solve, no vendor data), exit code = failures. The setup script and CI call it.
- [ ] `start.py` steps: [1] uv (missing: ask "Install it now? [Y/n]", run Astral's official installer for the OS; `--yes` skips), [2] `uv sync`, [3] Linux: Qt xcb libs present (else print the apt / dnf line), [4] `Data/DATA_AA.mat`, [5] extras report (Coin-HSL loads or MUMPS, `CHASSISSIM_DIR`, C compiler for jit), [6] verify: model build (nx 23) + a QSS lap on Sturn, [7] quick tests (`--no-tests`), [8] GUI (`--no-gui`). error / fix hints as ball-drop-sim. `start.bat` / `start.sh` thin wrappers (find uv or python, pass the args).
- [ ] Checks: on this Linux box from a clean clone dir: `./start.sh --yes --no-gui` passes; `uv run fullmodelsim` starts (offscreen smoke); snapshot none.

### G2: app fixes (D31, D32)
**Files:** `app/paths.py`, `app/mainwindow.py`, `app/runconfig.py`, `headless_solve.py`, `functions/hsl.py`, tests.
- [ ] Output folder default: QStandardPaths DocumentsLocation + `FullModelSim` (home fallback); QSettings remembers the last chosen folder.
- [ ] Coin-HSL folder field + Browse (Advanced tab), QSettings; reaches the solve (cfg key, `FORWARD`); `hsl.resolve_hsl_dir` order: `COINHSL_DIR` > explicit arg (GUI field) > frozen > seeded default. Linux: a folder with `libhsl.so` loads (preload its dependent `.so` files when the loader needs it).
- [ ] Unique temp cfg file per run (no fixed `fms_solve_cfg.json`).
- [ ] Checks: GUI tests offscreen pass; a fake-HSL folder path reaches the solve kwargs; MUMPS fallback message.

### G3: PyInstaller spec for both OSes, CI workflows (D27, D28)
**Files:** `build/app.spec` (rename / generalise `windows-app.spec`), `build/README.md`, `.github/workflows/tests.yml`, `.github/workflows/release.yml`.
- [ ] One spec: Windows keeps today's behaviour (version info, `.ico`, casadi DLL hook); Linux: no version info, HSL bundling of `.so` only for local builds with `COINHSL_DIR`.
- [ ] `tests.yml`: push + PR, matrix windows-latest / ubuntu-22.04, `uv sync --locked`, `QT_QPA_PLATFORM=offscreen`, `python tests/run_tests.py --quick`.
- [ ] `release.yml`: tag `v*`: build on both, smoke = frozen app `--headless` with a QSS cfg (both OSes), package `FullModelSim-<tag>-windows.zip` / `-linux.tar.gz`, Linux clean-container check (`ubuntu:22.04`: run the headless QSS; `ldd` of the Qt xcb plugin shows no missing lib), create the GitHub Release with both assets. MUMPS only, no vendor data.
- [ ] Checks: local Linux build with the spec (`uv run pyinstaller build/app.spec`), the frozen binary runs a headless QSS cfg; workflow YAML lint (actionlint if available, else a careful review). Windows and the container step verify in CI after the owner's push.

### P1: refusals, atmospheric correction, new cids (D33, D40)
**Files:** `functions/chassissim.py` (CAR_FIELDS for cids 104-106, 160, 177-183, 464-468, 535, 571, 573, 590, 598, 649, 836, 870, 871, 161, 520, 419, 420, 423-429, 897, 906 as needed), `userOpts.py` (refusals with STE messages), `vehModel_cs.py` + `functions/ggv.py` (atmos factor).
- [ ] Raise ValueError: brake ramp on (571), split recompute (598), battery plugin present (897 / 906), push-to-pass (649), torque multiplier maps (520), throttle trims (419, 420, 423-429 on), non-zero engine-brake table (161).
- [ ] Atmospheric correction: read `Atmos_Cond.ini` from the ChassisSim folder when present; Eng Corr = 1 -> T x sqrt(Teng_K / Tcur_K) x ((Patm_cur - Pw_cur) / (Patm_eng - Pw_eng)) on the engine torque (NLP and QSS); else factor 1.0 (no node).
- [ ] Checks: all 31 shipped files load as before; GP3 / GT3 / F2 graph hashes equal; synthetic files for each refusal and the atmos factor.

### P2: front-axle drive (D34)
**Files:** `vehModel_cs.py` (one axle-generic differential function used for both axles, front share of T_drive), `functions/ggv.py` (QSS traction limit with both axles), `userOpts.py` (drop the torque-split refusal).
- [ ] Front force = split x drive force, rear = (1 - split) x drive force; each through its axle's five-type law; front wheels get their drive torque. Split 0: graph unchanged (hash).
- [ ] Checks: WRX builds (nx 23) and converges on Sturn (one real solve, memory check); synthetic: split 0.5 open / spool front; QSS lap of the WRX; GP3 / GT3 / F2 hashes equal.

### P3: shift cut (D35)
**Files:** `vehModel_cs.py` (envelope), `functions/ggv.py` (same envelope in the QSS).
- [ ] Dip per upshift speed v_g = rpm_limit R / ratio_g (g < top): factor 1 - (1 - minF) w(v), w a smooth bump over [v_g, v_g + dv], dv = speed gained in `delay` at the dipped full drive (impulse matched); ramp time adds half its length. delay 0: no node.
- [ ] Checks: GT3 (LP560): QSS and NLP envelope dips at its upshift speeds; the GT3 Adelaide log torque drop 0.575 at 12 upshifts for 0.1 s vs the model's impulse; the GT3 fixed-line solve again (lap change reported); other cars' hashes equal.

### P4: hybrid layer, NLP (D36-D38)
**Files:** `vehModel_cs.py` (new control F_dep >= 0, and F_pad >= 0 when cid 573 is on; harvest power from the brake blend; deploy through the differentials with the 464 / 465 front share), `functions/transcription.py` or `MLTP.py` (two lap-integral rows: deploy <= harvest + E_free, deploy <= E_cap + E_free; the quadrature of sf as for the lap time), result channels (P_dep, P_harv, E_dep, E_harv).
- [ ] Master cid 177 = 0: no new input, no row, graph unchanged.
- [ ] Deploy power bound: cid 180 x map(rpm, full throttle) x cid 183 (map blend 464 / 465; maps fractions of the scale). Brake-blend harvest = smooth min(cid 179, regen-axle brake force x v) x cid 182 (axle 870 / 871). Paddle: extra decel on the driven axles (engine split), power <= cid 179 shared with brake regen, harvest counted.
- [ ] Warm starts: a hybrid NLP has other inputs and rows (structure mismatch -> full-interp); the planner says so.
- [ ] Checks: a synthetic hybrid car (fake IPOPT): rows, bounds, channels, zero cost when off; GP3 copy with the layer on (scratch .car through carfile.write) at Eastern Creek fixed line (one real solve): deploy <= power bound, budget rows active, harvest <= clamp, lap gain reported.

### P5: hybrid layer, QSS (D39)
**Files:** `functions/ggv.py` (march), `MLTP_screen.py` (record fields).
- [ ] Budget = min(brake-regen harvest of the braking zones (backward pass), cid 181) + cid 836; deploy mode 178: 0 = whole budget on the last acceleration segment before S/F, 1 = split over the acceleration segments by length; deploy power = the P4 bound. Store-mode power uses cid 180 (store-max power undecoded: ponytail).
- [ ] Checks: hand budget on a synthetic lap; both modes; QSS lap of the hybrid GP3 copy vs its NLP lap; shipped cars bit-equal.

### V: validation solves (Opus)
- [ ] WRX on Sturn (front drive), GT3 Adelaide fixed line (shift cut, vs log), hybrid GP3 copy at Eastern Creek fixed line, local frozen Linux build headless QSS. One solve at a time.

### D: docs and final review
- [ ] CLAUDE.md (setup routes, uv commands, car paragraph: front drive, shift cut, hybrid layer, refusals, atmos), README (download and developer routes), build/README.md (both OSes, CI), tests list, CONTEXT.md terms (done in the grilling). STE lint on new text. Final whole-branch Opus review, Fable fix wave.
