# ChassisSim phase 5: GUI car and circuit combos, car setup editing, save new .car, powertrain group

Date 2026-11-20. Owner asks 2026-11-20 13:04: select `.car` files in the GUI (preset Load), save the current settings as a `.car` (preset Save), powertrain options in the GUI. Spec decisions D22 (car and circuit combos, full setup editing), D23 (edits saved as a new `.car`), D9 (encoder never overwrites, `encode(decode(f)) == f` gate before a save). Code map: `.superpowers/sdd/2026-11-20-gui-phase5/map.md` (local). Flow per lane: Sonnet build -> Opus audit -> Fable fix -> Haiku tests -> commit.

Assumptions (owner may override): the 4EM car saves JSON only (no `.car` source, a 4EM -> `.car` export would lose the tyre and powertrain data). Scalars are editable, tables / curves / maps are read-only. Edited cars go to `<presets>/cars`, never into the ChassisSim folder.

## Lane P5-A: ChassisSim folder, circuits, car selector, run with a car

Files: app/mainwindow.py, app/runconfig.py, app/results.py, headless_solve.py, functions/chassissim.py.

- [ ] `chassissim.list_circuits(cs_dir)`: the ini names plus the orphan `tr_*.txt` circuits, factored out of `load_circuit` (one source). `list_cars(cs_dir)`: sorted absolute paths of `<cs_dir>/*.car`, `<parent>/Models/**/*.car` and `<presets>/cars/*.car`.
- [ ] RunConfig and FORWARD: `chassissim_dir` (None) and `car` (None, `"chassissim:<abs path>"`). Kept in qss mode (not NLP_ONLY).
- [ ] Advanced tab: ChassisSim folder field (QLineEdit + Browse + QSettings `chassissim_dir`, saved on Run as `hsl_dir`). Tooltip: the field wins, env `CHASSISSIM_DIR` is the fallback.
- [ ] Circuit combo: `CIRCUITS` plus `chassissim:<name>` entries from `list_circuits` (field, else env). Rebuilt on a folder change; the selection stays. The `CIRCUITS` constant does not change (test pin).
- [ ] Main tab Car combo: `4EM (FullModelSim)` first, then `list_cars`, plus Browse (any `.car`, added to the combo). QSettings `car`.
- [ ] Car coupling: physics corrected, 4 Motors off, Aero Static, ATD off, start state not rolling, Homotopy off. Each control is set and disabled, with a tooltip that says why. The 4EM car restores them.
- [ ] Preset Load: filter `Presets (*.json *.car)`. A `.car` selects that car.
- [ ] `result_paths(rc)` passes `car` and `chassissim_dir`. `_show_results` catches errors and logs `[error]`.
- [ ] Checks: offscreen GUI cfg capture with a synthetic car (runner trick). headless_solve qss run with a synthetic car cfg. A vendor car qss run (SKIP without `CHASSISSIM_DIR`). The result path equals the MLTP stem.

## Lane P5-B: car setup editing, save a new .car

Files: app/mainwindow.py, app/car_edit.py (new), functions/carfile.py (helpers only if needed).

- [ ] `car_edit`: from `carfile.decode` records to editable fields. These are the tag 1 scalars and tag 6 ints of the CAR_FIELDS groups mass, geom, susp, engine, brake, aero, hybrid and env. Also the scalars inside the tag 11 springs (preload, rate), the tag 12 dampers and the tag 13 tyres. Labels and units come from CAR_FIELDS (SI display = file value x factor).
- [ ] Setup tab car mode: the 4EM groups and the Powertrain group are hidden. The car groups show the file values as defaults, with the changed highlight and Reset against the file.
- [ ] Edit by record: only edited records change (value / factor back to file units), and unread records stay byte-equal. A left record copied from the right is written out when the user edits it.
- [ ] D9 gate on the source bytes before any write: `encode(decode_bytes(b)) == b`, else `[error]` and no editing.
- [ ] Validate the new bytes: `load_car`, `check_car`, `power_refusals`, `aero_refusals`. Errors go to the log as `[error]` lines.
- [ ] Run with edits: write `<presets>/cars/<base>_<sha8 of the new bytes>.car` (reuse a file with equal bytes, `write` never overwrites), then run with `car=<new path>`.
- [ ] Save preset in car mode: dialog `ChassisSim car (*.car)`, DontConfirmOverwrite, an existing path gives `[error]` (choose a new name), `carfile.write`. 4EM mode saves JSON as today.
- [ ] Checks: a synthetic car with an edited FR spring rate and brake bias decodes to the edited values, and every other record is bitwise equal. Also the gate, the validation errors and the run cfg on the new file. A vendor GP3 copy edit moves the QSS lap.

## Lane P5-C: powertrain group (4EM car)

Files: app/mainwindow.py, app/runconfig.py.

- [ ] Setup group "Powertrain (physics corrected)": gear, Tmax [N m per motor], Pmax [kW per motor], Preg [kW per motor, check box, off], Ppack [kW, check box, off]. Defaults from Powertrain.py and vp.gear. A separate dict (not PARAM_GROUPS: 130-row pin, vp_overrides raises on unknown keys).
- [ ] Enabled only with Physics corrected, hidden in car mode. It sends `pt_overrides` only for changed or switched-on values (A5 keys always count as knobs). The More box `pt_overrides` and the group both set gives `[error]`.
- [ ] Checks: cfg vp_overrides gets the keys; matlab physics greys the group; the tag changes only when a value changes.

## D and final

- [ ] README GUI section (car, circuits, editing, save `.car`, powertrain group); CLAUDE.md desktop app and test list; STE lint.
- [ ] Opus whole-phase review, Fable fix wave. The owner opens one saved `.car` in ChassisSim on Windows (spec risk: no ChassisSim on Linux).

Skipped: table / curve / map editing (add when a curve edit is needed); 4EM -> `.car` export (add on owner ask); the 4EM battery model (D20, waits for pack data).
