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
- D20 EV energy model (pack SoC, sag, regen budget, pack thermal, motor envelope): separate later plan, after the owner supplies pack data.
- D21 API: `car='chassissim:<path to .car>'`, `circuit='chassissim:<name>'`, install folder from kwarg `chassissim_dir` or env `CHASSISSIM_DIR`.
- D22 GUI: car and circuit combos, full setup editing. D23 edits saved as a new `.car` (D9 gate first).
- D24 Order: codec first. Phases below.
- D25 Item 1: own axle balance channel (rear grip share minus front, oversteer positive) for any car, diagnostic. ChassisSim stability index waits for D17.
- New physics lives on the corrected path. A ChassisSim car never runs `physics='matlab'`. matlab defaults byte-identical forever.

## Phases and exit gates
1. Data access. Codec (decode, encode, round trip on all shipped `.car`, GP3 variant diff), circuit text readers, ChassisSim circuit source, `.ld` reader. Exit: gates green, 4EM car solves on `chassissim:Eastern Creek`, snapshot none.
2. GP3 physics + QSS. Car source `car=`, car text readers the `.car` does not inline, tyre (D4), gear envelope, diffs, aeromap + corrections, DRS, kinematics curves, springs, bars, dampers (item 9), roll-centre migration (item 10). Exit: D19 gate 2.
3. Circuit items: grip factor along s (item 2), grade and banking (item 3), road profile long waves (D13).
4. Tyre temperature states (item 5), relaxation length (item 8).
5. GUI: combos, full setup editing, save as new `.car`.
6. Stability index (item 1 map part) after the D17 decode.

## Data facts (measured 2026-11-18, binding)
- Curvature files: 2 columns `k [1/m] s [m]`, no header, CRLF, tab or space, irregular spacing 0.15-7.6 m. Right turn positive (docs + Brands Hatch clockwise +0.985 laps, Phillip Island anticlockwise -1.025 laps). FullModelSim negates k. Traces do not close (Eastern Creek -0.646 laps, end gap 971 m): load as open `s, k`, no closure correction, no periodic start.
- `Circ_prop.ini`: `[CircuitN]` blocks, keys `Circuit name`, `Circuit file name`, `Base Grip factor`, `Circuit bp name/flag`, optional gsf/bsf/alt keys. Eastern Creek = base grip 1.0, `tr_eastcreek.txt`, `bump_profile_ec.dat`. Four curvature files have no block (orphans: Brands Hatch, Magny Cours, Portimao, FSAE Sample circuit).
- alt/camber: line 1 base altitude m, rows `s, pitch rad, camber rad`, one row fewer than the curvature file. Only Brands Hatch and Portimao, camber all 0. Bump: `FL FR RL RR [mm] s [m]`.
- `.ld`: 124-byte channel records, int32 dec 2 (value = raw/100), 50 Hz, no time channel. Sim files: 203 channels, 3 segments (lead-in, lap, 4 s tail) split at the Lap Distance wrap. Signs measured on GP3 Eastern Creek against the curvature: `G Force - Lateral` left positive (corr -0.999 with v^2 k right-positive), `Steered Angle` and `Yaw rate` right positive. Gate lap time = the variant's `.ldx` fastest time (= `.ld` lap 1, up to 0.03 s above the true lap). `rc_data.dat` (44-column CSV, flying lap, true lap time) holds only the LAST run of its folder.

## Risks
- Gate 2 can fail on method alone (transient sim on a fixed line vs optimal control).
- ChassisSim car NLP grows to about 31 states (thermal + relaxation): slower solves.
- No ChassisSim on Linux: a written `.car` is checked by the byte gates only; the owner opens one in ChassisSim on Windows as the last step.
