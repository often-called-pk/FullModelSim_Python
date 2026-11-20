# ChassisSim integration: program spec

**Goal:** all 10 items of the ChassisSim backlog (`IRP/ChassisSim_analysis/ChassisSim_mechanism_and_FullModelSim_backlog.md`, PART 2). Data FullModelSim lacks comes from ChassisSim input files directly. A ChassisSim car loads whole. The 4EM car stays the default, unchanged.
**Source:** owner grilling session 2026-11-18 (25 decisions below), owner reply "yes we have reached an understanding". Terms: `CONTEXT.md`. Codec decision: `docs/adr/0001-chassissim-car-decoder-in-public-repo.md`.
**Branch:** `chassissim-cars`. One phase plan per phase in `docs/superpowers/plans/`.

## Owner decisions (binding)
- D1 Car source: one solve = one car source, 4EM car (default) or whole ChassisSim car. No splicing of car data.
- D2 Car data: decode the binary `.car` (ChassisSim v3.41). Layout map from the teammate session `chassissim-vs-fullsim-comparison` (`IRP/ChassisSim_analysis/car_format_v3_41.{json,md}`, `_evidence.md`).
- D3 First target: GP3 2013 at Eastern Creek (`Models/GP3 2013/Eastern Creek, AU`: 7 `.car` variants, 8 `.ld` laps, `rc_data.dat`).
- D4 Tyre: native ChassisSim tyre (load fn x temperature fn, slip table, camber and ellipse terms). Load floor for wheel lift.
- D5 Tyre temperature: 4 dynamic states. D6 start at the background temperature, periodic start stays an option.
- D7 Powertrain: gear envelope (smooth max over gears of engine torque x ratio). Shift time ignored at first.
- D8 Differential: all five ChassisSim types (open, super diff, spool, custom 2-D map, Detroit locker).
- D9 Codec in the public repo, decode AND encode. Encoder writes new files only, never overwrites a source. Gate: encode(decode(f)) == f for every shipped `.car` before any edit is saved. No vendor file or vendor data in git. Tests use synthetic files.
- D10 Racing line: free line by default, +-4 m band. Gate runs use a fixed-line band.
- D11 Kinematics: numpy solve once per car from the hardpoints, smooth fitted curves in the NLP (camber, toe, roll-centre height vs travel and roll).
- D12 Road input: full road profile wanted. D13 on this machine (15 GB): long waves only (wavelength >= 2x mesh spacing) on the normal mesh. Full fine mesh: later, bigger machine.
- D14 Relaxation length (item 8): build it, MF 6.1 load-dependent form, literature PTY1/PTY2 defaults, on for ChassisSim cars. 4EM car only through a knob (parity kept).
- D15 Circuits: any car on any circuit. ChassisSim circuits carry curvature, altitude/camber, grip factors, road profile.
- D16 QSS: full QSS path for ChassisSim cars (tyre peak at background temperature, gear envelope, aeromap at quasi-static ride heights, DRS in zones, circuit grip/grade/banking).
- D17 STBI correction map: teammate decodes the per-step stability formula first, then decide. Until then the map is read, not applied.
- D18 Comparison: MoTeC `.ld` reader feeding the E5 log tools (overlay channel by channel).
- D19 Acceptance: gate 1 decoded data exact (byte round trip of every shipped car, GP3 variant record sets pinned as measured, values match the `Car_*.ini` lists where they overlap). Gate 2 fixed-line GP3 lap within 2% of ChassisSim. Channel RMS reported, no hard gate. Free-line lap <= fixed-line lap.
- D20 EV energy model (pack SoC, sag, regen budget, pack thermal, motor envelope) for the 4EM car: separate later plan, after the owner supplies pack data. ChassisSim cars get ChassisSim's hybrid layer instead (D36).
- D21 API: `car='chassissim:<path to .car>'`, `circuit='chassissim:<name>'`, install folder from kwarg `chassissim_dir` or env `CHASSISSIM_DIR`.
- D22 GUI: car and circuit combos, full setup editing. D23 edits saved as a new `.car` (D9 gate first).
- D24 Order: codec first. Phases below.
- D25 Item 1: own axle balance channel (rear grip share minus front, oversteer positive) for any car, diagnostic. ChassisSim stability index waits for D17.
- New physics lives on the corrected path. A ChassisSim car never runs `physics='matlab'`. matlab defaults byte-identical forever.

## Phase 2c decisions (owner grilling 2026-11-19, binding)
- D26 GUI routes: both. Ready-to-run downloads per OS for people without Python, and a one-command developer install.
- D27 Builds: GitHub Actions. Windows and Linux runners build the frozen app on each tag `v*` and attach the downloads to a GitHub Release. The tests run on both OSes on every push. The local build scripts are the CI steps.
- D28 Linux download: `.tar.gz` of the PyInstaller onedir folder, built on Ubuntu 22.04, Qt system libs (libxcb-cursor0 etc.) installed on the runner so the bundle carries them, checked in a clean Ubuntu container. AppImage only on request.
- D29 Developer route: `uv` + `pyproject.toml` + `uv.lock`; `uv run fullmodelsim` starts the GUI; `requirements.txt` exported from the lock for pip users; CI builds from the lock.
- D30 Setup scripts: `start.bat` / `start.sh` -> `start.py` (ball-drop-sim pattern: numbered steps, error / fix hints). Steps: uv (missing: ask, then Astral's official installer; `--yes` skips the prompt), `uv sync`, Linux Qt libs (print the apt / dnf line), `Data/DATA_AA.mat`, optional extras report (Coin-HSL, `CHASSISSIM_DIR`, C compiler), verify (model build + 1 s QSS lap), quick tests, GUI. Flags `--yes`, `--no-gui`, `--no-tests`.
- D31 Output folder: QStandardPaths DocumentsLocation + `FullModelSim`; the GUI remembers the last folder per user (QSettings).
- D32 Coin-HSL: folder field + Browse in the Advanced tab, remembered per user; `COINHSL_DIR` wins; MUMPS fallback. Downloads ship MUMPS only (HSL licence) and no vendor data. The ChassisSim folder field comes with the phase 5 pickers.
- D33 Brake delay: cid 109 is dead data in ChassisSim: not built. A car with the brake ramp on (cid 571 = 1) raises ValueError. The ramp as a brake-rate limit is a later upgrade.
- D34 Front-axle drive: cid 111 = constant front share of the drive force (no centre diff); the front diff follows the same five-type law (cids 112, 114); NLP and QSS; split 0 keeps the rear-drive graph; cid 598 set raises.
- D35 Shift cut: a smooth dip of the drive envelope at each upshift speed (gear g at the rev limit) that costs the time of the cut. Per upshift tau = (1 - minF) F t / (F - R): F = envelope and R = resistance at the upshift speed, t = delay (cid 104) + ramp (cid 106) / 2, minF = cid 105. Band width max((1 - minF) F t / m_eff, 3 m/s) (mesh-resolvable), depth solved so that crossing the band costs tau. No band where F <= R; zero cost when the delay is 0. (Corrected 2026-11-19 after the P3 audit: a dip to minF x F ignores R and stalls the GT3 at 59.4 m/s.) ponytail: a car holding a band speed feels the dip; upgrade = gear state.
- D36 Hybrid layer (ChassisSim cars, master cid 177): the optimizer chooses the deploy force (control), power <= cid 180 x map (front 467 / rear 468 at full throttle, blend 464 / 465) x cid 183, through the differentials. Per lap: deploy energy <= harvest + cid 836, and <= cid 181 + cid 836. Deploy and harvest traces in the results. The battery-pack plugin stays out.
- D37 Brake regen: automatic blend: regen takes over the regen axle's friction share (cids 870 / 871, rear by default) up to cid 179 x cid 182; axle forces unchanged; no new control.
- D38 Paddle regen: a second optimizer control when cid 573 is on: extra deceleration on the driven axles (engine split), harvest counted, shares the cid 179 clamp.
- D39 QSS of a hybrid car: ChassisSim's rule: budget = min(brake-regen harvest of the braking zones, cid 181) + cid 836, spent by the deploy mode (cid 178) at the map power.
- D40 Still not modelled, raise ValueError when on: battery plugin (cids 897 / 906), push-to-pass (649), torque multiplier maps (520), throttle trims (419, 420, 423-429), a non-zero engine-brake table (161). Atmospheric correction: the exact formula when an `Atmos_Cond.ini` in the ChassisSim folder has Eng Corr = 1, else 1.0. The gear choice stays (the log shows upshifts at the rev limit).

## Phases and exit gates
1. Data access. Codec (decode, encode, round trip on all shipped `.car`, GP3 variant diff), circuit text readers, ChassisSim circuit source, `.ld` reader. Exit: gates green, 4EM car solves on `chassissim:Eastern Creek`, snapshot none.
2. GP3 physics + QSS. Car source `car=`, car text readers the `.car` does not inline, tyre (D4), gear envelope, diffs, aeromap + corrections, DRS, kinematics curves, springs, bars, dampers (item 9), roll-centre migration (item 10). Exit: D19 gate 2.
3. Circuit items: grip factor along s (item 2), grade and banking (item 3), road profile long waves (D13).
4. Tyre temperature states (item 5), relaxation length (item 8).
5. GUI: combos, full setup editing, save as new `.car`.
6. Stability index (item 1 map part) after the D17 decode (built in phase 2b).
2c. Powertrain gaps, hybrid layer and the GUI on Windows and Linux (D26-D40).

## Data facts (measured 2026-11-18, binding)
- Curvature files: 2 columns `k [1/m] s [m]`, no header, CRLF, tab or space, irregular spacing 0.15-7.6 m. Right turn positive (docs + Brands Hatch clockwise +0.985 laps, Phillip Island anticlockwise -1.025 laps). FullModelSim negates k. Traces do not close (Eastern Creek -0.646 laps, end gap 971 m): load as open `s, k`, no closure correction, no periodic start.
- `Circ_prop.ini`: `[CircuitN]` blocks, keys `Circuit name`, `Circuit file name`, `Base Grip factor`, `Circuit bp name/flag`, optional gsf/bsf/alt keys. Eastern Creek = base grip 1.0, `tr_eastcreek.txt`, `bump_profile_ec.dat`. Four curvature files have no block (orphans: Brands Hatch, Magny Cours, Portimao, FSAE Sample circuit).
- alt/camber: line 1 base altitude m, rows `s, pitch rad, camber rad`, one row fewer than the curvature file. Only Brands Hatch and Portimao, camber all 0. Bump: `FL FR RL RR [mm] s [m]`.
- `.ld`: 124-byte channel records, int32 dec 2 (value = raw/100), 50 Hz, no time channel. Sim files: 203 channels, 3 segments (lead-in, lap, 4 s tail) split at the Lap Distance wrap. Signs measured on GP3 Eastern Creek against the curvature: `G Force - Lateral` left positive (corr -0.999 with v^2 k right-positive), `Steered Angle` and `Yaw rate` right positive. Gate lap time = the variant's `.ldx` fastest time (= `.ld` lap 1, up to 0.03 s above the true lap). `rc_data.dat` (44-column CSV, flying lap, true lap time) holds only the LAST run of its folder.

## Risks
- Gate 2 can fail on method alone (transient sim on a fixed line vs optimal control).
- ChassisSim car NLP grows to about 31 states (thermal + relaxation): slower solves.
- No ChassisSim on Linux: a written `.car` is checked by the byte gates only; the owner opens one in ChassisSim on Windows as the last step.
