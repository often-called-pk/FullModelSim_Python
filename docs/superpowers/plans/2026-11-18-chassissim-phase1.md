# ChassisSim phase 1 plan: data access (circuit source, .ld reader, .car codec)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** phase 1 of `docs/superpowers/specs/2026-11-18-chassissim-integration-design.md`. Any car runs on a ChassisSim circuit. ChassisSim `.ld` laps read into the E5 log tools. The `.car` codec decodes and re-encodes every shipped car byte for byte.
**Architecture:** one new module `functions/chassissim.py` (circuit readers + circuit source), `read_ld` appended to `functions/telemetry.py`, one new module `functions/carfile.py` (codec, T3). Wiring only in `userOpts._load_track`, `importfile.result_stem` and two plot-folder slices. Defaults untouched.
**Tech stack:** Python, numpy, CasADi (no new dependency).
**Spec:** the program spec above (owner decisions D1-D25). Mapping notes (local, git-ignored): `.superpowers/sdd/chassissim/map/{cs_text.md, cs_text_proto.py, fms_track.md, ld_reader.md, ld_proto.py, sign_check.py}`. Prototypes parse every install file: lift, do not apply blind.

## Order
T1 circuit source, T2 `.ld` reader (uses T1 in its vendor check), T3 codec (brief completed when the teammate layout v1 lands; T1 and T2 do not wait), T4 docs, V validation.

## Global constraints
- Worktree `.claude/worktrees/physics-fixes`, branch `chassissim-cars`, cwd = worktree root, never cd outside. Python `/home/pk/.venvs/fullmodelsim/bin/python`. One commit per task by Opus: `git -c user.name="Prashant Kumar" -c user.email="prashant.jps@gmail.com" commit`, STE body, no attribution lines, no push.
- Ponytail: shortest correct diff, reuse before writing, no new dependency, no knob beyond the task text. Zero cost at defaults.
- Snapshot gate after every task: `python .superpowers/sdd/chassissim/map/nlp_snapshot.py . <out.json> .superpowers/sdd/chassissim/map/nlp_base.json` prints `SNAPSHOT DIFF: none`. Default result stems unchanged.
- Parity gates `tests/test_vehmodel_matlab.py`, `tests/test_matlab_parity.py` green, never edited.
- Vendor data: ChassisSim install `CHASSISSIM_DIR` = `/home/pk/Downloads/pk-backup-d-drive/Software Installs/ChassisSimTechnologies/ChassisSim_v3_41`, models `<CHASSISSIM_DIR>/../Models`. Never copy a vendor file, value table or log into the repo. Tests write synthetic files into `tempfile.TemporaryDirectory`. Vendor checks sit in a final test section that prints SKIP and exits 0 when `CHASSISSIM_DIR` is unset or missing. `.gitignore` gets `*.car`, `*.ld`, `*.ldx`.
- Never read `/home/pk/Downloads/pk-backup-d-drive/IRP/MLTP_Python/*.csv`.
- No IPOPT in build and test steps (fake solver via `functions.transcription._make_solver`, as `nlp_snapshot.py`). Real solves: Opus only, one at a time, `free -g` available >= 6 GB.
- Tests: plain scripts, `import _bootstrap` first, top-level asserts with `[PASS]` lines, exit non-zero on failure. Builders (Sonnet) never edit tests. Test writers (Haiku) edit only the task's test file and must run it.
- Style: briefs and agent replies caveman. Docstrings, comments, commit bodies, new error and warning messages STE (short sentences, active voice, no semicolons). Code ponytail. No em dash, ASCII only.
- cavecrew receipts: builder and test writer end with `<path:line-range> - <change, max 10 words>.` lines then `verified: <re-read OK | mismatch @ path:line>.`, or one terminal token `too-big.` / `needs-confirm.` / `ambiguous.` / `regressed.`. Reviewer (`caveman:cavecrew-reviewer`) returns `path:line: <emoji> <severity>: <problem>. <fix>.` then totals, or `No issues.`.

## Evidence (binding, measured 2026-11-18)
- Curvature files right-turn positive (docs + Brands Hatch clockwise +0.985 laps, Phillip Island anticlockwise -1.025 laps). Traces do not close (Eastern Creek -0.646 laps, gap 971 m). `check_track` with s, k only runs finite / monotone / spacing / |k| checks, no closure (track_qa.py:48).
- Eastern Creek: `[Circuit2]` of `Circ_prop.ini`, `tr_eastcreek.txt` 971 rows, s 0 to 3824.181 m, base grip 1.0, bump `bump_profile_ec.dat` flag 1, no alt file.
- `.ld` on GP3 Eastern Creek against the right-positive curvature: `G Force - Lateral` corr -0.999 with v^2 k (left positive), `Steered Angle` +0.981, `Yaw rate` +0.945 (right positive). ax forward positive (mapper C). `Speed FL` = ground speed (distance rate / Speed FL = 1.000 to 1.002).
- pv path, stems, GUI list rules: `fms_track.md`. `tests/test_gui_logic.py:155-170` requires every GUI circuit in `_REAL_CIRCUITS` or synthetic: no GUI change in phase 1.

## Rulings
- Ruling: ChassisSim circuits load as open `s, k` traces, k negated, no x, y on the track (plots use `curv2cart` through `reconstruct_track`), no closure correction, periodic start refused (not in `CLOSED_CIRCUITS`). Cost if wrong: one closure option later.
- Ruling: name resolution = `Circuit name` of `Circ_prop.ini`, case-insensitive. Else the orphan pattern `Circuits/tr_<name>.txt`, `bp_<name>.dat`, `alt_rcamb_<name>.txt` (base grip 1.0). Cost if wrong: one wrong pairing of an orphan file.
- Ruling: the circuit base grip factor applies now for the MF tyre: `ctx.mf.LMUX *= g`, `ctx.mf.LMUY *= g`, skipped at g == 1.0. The varying grip factor (GSF) waits for phase 3 (no sample file exists). Cost if wrong: one rescale rule.
- Ruling: phase 1 track = s, k. The alt/camber and bump readers exist and are tested, `load_circuit` returns their file paths, the model reads them in phase 3. Car-side text readers (aeromap, hardpoints, tyre curves, Car ini) move to phase 2, when the codec shows what the `.car` holds inline. The `car=` kwarg comes in phase 2 with its physics. Cost if wrong: none.
- Ruling: stem slug `cs_<name>` (`\W+` -> `_`) for `chassissim:` names only, in `result_stem` and in both plot-folder slices. `chassissim_dir` is a route key (`_NOT_SETUP`). No content-hash knob for circuit files (same as the `.mat` circuits). Cost if wrong: an edited circuit file overwrites its old result.
- Ruling: `read_ld` returns the `read_monster` keys (`t`, `s` = Lap Distance, `vx`, `ax`, `ay`, `delta`, `xs_*`, `throttle`, `om_gear`) plus `r` and `lap`, so `overlay_report` takes it with no `project` call. `xs_*` = damper travel (motion ratio not applied). Cost if wrong: one map entry.

---

### Task 1: ChassisSim circuit source

**Files:** create `functions/chassissim.py`; modify `userOpts.py` (`_load_track` ~82-119, signature ~423-431, `_NOT_SETUP` ~254, after `ctx.track = ...` ~535), `functions/importfile.py` (`result_stem` ~105-130), `plotSDI.py` (~275), `app/results.py` (`plot_dir` ~31), `.gitignore`. Tests: `tests/test_chassissim_circuit.py` (Haiku).

- [ ] `functions/chassissim.py` (module docstring STE: ChassisSim v3.41 input files, readers and circuit source). Lift from `map/cs_text_proto.py`:
  - `PREFIX = "chassissim:"`.
  - `install_dir(chassissim_dir=None)`: kwarg, else env `CHASSISSIM_DIR`. ValueError "Set CHASSISSIM_DIR to the ChassisSim install folder, or pass chassissim_dir." when neither, and "The folder '<p>' does not exist." for a missing folder.
  - `read_curvature(path)` -> `(s, k)` as stored (right positive), float arrays. Line by line, `split()`, skip blank lines. ValueError when s is not strictly increasing or fewer than 3 rows.
  - `read_circ_prop(path)` -> list of dicts, one per `[CircuitN]` block (keys as written, values str). `configparser.ConfigParser(strict=False, interpolation=None)`, `optionxform = str`, latin-1.
  - `read_alt_camber(path)` -> `(base_alt, s, pitch, camber)`: line 1 one number (trailing tabs allowed), then rows `s pitch camber` in m, rad, rad.
  - `read_bump(path)` -> `(s, h)`: h shape (n, 4), columns FL FR RL RR, converted mm -> m.
  - `load_circuit(name, cs_dir)` -> `SimpleNamespace(s, k, grip, bump, alt)`: `k = -k_file` (model kappa > 0 = left), `grip` = float `Base Grip factor` (1.0 for orphans), `bump` / `alt` = absolute paths or None (flag key == "1" and the file exists; `default.dat` = None). Case-insensitive file lookup inside `<cs_dir>/Circuits`. ValueError listing the known names for an unknown name. FileNotFoundError for a listed curvature file that is absent.
- [ ] `userOpts._load_track(circuit, circuits_dir, chassissim_dir=None)`: first branch `if circuit.startswith(chassissim.PREFIX)`: `track = chassissim.load_circuit(circuit[len(PREFIX):], chassissim.install_dir(chassissim_dir))`, then the same QA as the `.mat` branch (move the 6 QA lines into one helper used by both branches, same messages). Pass `chassissim_dir` from `userOpts` (new kwarg after `circuits_dir`, default None, added to `_NOT_SETUP`).
- [ ] After `ctx.track = _load_track(...)`: `g = getattr(ctx.track, "grip", 1.0)`; when `g != 1.0`: `ctx.mf.LMUX *= g` and `ctx.mf.LMUY *= g` (comment: circuit base grip, ChassisSim `Base Grip factor`).
- [ ] `importfile.circuit_slug(circuit)`: `"cs_" + re.sub(r"\W+", "_", circuit[len("chassissim:"):]).strip("_")` for the prefix, else `circuit` unchanged. `result_stem` starts with `f"{circuit_slug(circuit)}_{cfg}"`. `plotSDI.py` ~275 and `app/results.plot_dir`: folder `circuit_slug(circuit)`, rest `stem[len(slug) + 1:]`. Import without a new dependency (casadi-free).
- [ ] `.gitignore`: add `*.car`, `*.ld`, `*.ldx`.
- [ ] Test file draft (Haiku writes it, keeps only checks that kill a mutant):
  1. Readers on synthetic files (CRLF, mixed tab/space, a space in the file name, blank trailing line): `read_curvature` values; non-increasing s raises; `read_circ_prop` with a 5-key and a 9-key block; `read_alt_camber` with trailing tabs on line 1; `read_bump` mm -> m and column order.
  2. `load_circuit` on a synthetic install (`Circ_prop.ini` + `Circuits/`): k sign flipped, s kept, grip from the ini, bump path with flag 1, None with flag 0 or `default.dat`; orphan `tr_Orphan.txt` + `alt_rcamb_Orphan.txt` found with grip 1.0; name lookup case-insensitive; unknown name raises ValueError naming a known circuit; listed but absent curvature file raises FileNotFoundError.
  3. `install_dir`: kwarg beats env, env used without kwarg, ValueError without both, ValueError for a missing folder.
  4. Stems: `result_stem("chassissim:Eastern Creek, AU", "Static_ATDOn_EM4Off")` starts `cs_Eastern_Creek_AU_Static`; `result_stem("Sturn", ...)` unchanged; `app.results.plot_dir` folder is the slug; `userOpts.setup_knobs({"chassissim_dir": "x"})` is empty.
  5. Fake-solver `MLTP(circuit="chassissim:<synthetic name>", chassissim_dir=<tmp>, save=False, plot=False, linear_solver="mumps")` builds (stand-in `_make_solver`), track s / k match the file (k negated). Grip 1.1 in the ini gives `ctx.mf.LMUX == ctx.mf.LMUY == 1.1`, grip 1.0 leaves both a plain float 1.0.
  6. Vendor section (SKIP without `CHASSISSIM_DIR`): every `Circ_prop.ini` block whose curvature file exists loads; Eastern Creek s[-1] == 3824.181, 971 points, sum(k ds) > 0 after the flip (net left), bump path ends `bump_profile_ec.dat`, alt None; `Brands Hatch` (orphan) finds its alt file; `MLTP_screen(circuit="chassissim:Eastern Creek", save=False)` gives a finite lap.
- [ ] Opus: run the test file, `tests/test_track_qa.py`, `tests/test_provenance.py`, `tests/test_results.py`, `tests/test_gui_logic.py`, snapshot none, review, commit `feat(circuit): add the ChassisSim circuit source (phase 1 T1)`.

### Task 2: MoTeC .ld reader

**Files:** modify `functions/telemetry.py` (append), `docs/logging_spec.md` (Table C). Tests: `tests/test_read_ld.py` (Haiku).

- [ ] Append to `functions/telemetry.py`, lifted from `map/ld_proto.py` (layout in `map/ld_reader.md` sections 1-3): `LD_MAP` (default map) and `read_ld(path, channel_map=None)`.
  - Header: magic u32 0x40 at 0 (else ValueError), channel list pointer u32 at 0x08, channel count u16 at 0x56. Channel record 124 bytes, linked list: prev, next, data pointer, count, id, class, size, rate, shift, mul, scale, dec, name[32], short[8], unit[12] (offsets per the note).
  - Values: `raw / scale * 10**-dec * mul + shift` reduced to the verified case: ValueError unless shift/mul/scale = 0/1/1; dtype from (class, size): int16, int32 (ValueError for others). One `np.fromfile(path, dtype, count, offset)` per mapped channel (streaming: only mapped channels in memory).
  - Map forms as `read_csv`: `{key: name}` or `{key: (name_or_tuple_of_names, unit, factor)}`; a tuple takes the first name present; unit checked against the record unit (ValueError on mismatch); a user-map channel that is absent raises ValueError; `channel_map=None` uses `LD_MAP` and skips absent channels.
  - `LD_MAP`: `s` Lap Distance m 1; `vx` (Speed FL, Wheel Speed FL) km/h 1/3.6; `ax` G Force - Longitudinal G +G; `ay` G Force - Lateral G +G; `delta` Steered Angle deg -pi/180; `r` Yaw rate deg/s -pi/180; `xs_fl..xs_rr` Damper Position - Front L / Front R / Rear L / Rear R mm 1e-3; `throttle` Throttle Pos % 0.01; `om_gear` Engine RPM rpm pi/30. Use the module constant `G` (exact unit strings from the note, section 5).
  - `t = arange(n) / rate` of the first mapped channel; a channel with another rate or count is `np.interp`-ed onto it. `lap = cumsum(r_[0, diff(s) < -0.5 * max(s)])` when `s` is mapped (lap 0 lead-in, 1 the lap, 2 the tail).
  - Docstring STE: sources, signs (measured: ay left positive, steer and yaw right positive in ChassisSim `.ld` files), `xs_*` = damper travel.
- [ ] `docs/logging_spec.md` Table C: add the measured ChassisSim `.ld` output signs (ay left positive, steer and yaw rate right positive, measured 2026-11-18 on GP3 Eastern Creek against the curvature), next to the import flags of the manuals. STE.
- [ ] Test file draft (Haiku): a test-local writer packs a synthetic `.ld` (header, 3 linked records: int32 dec 2 "Lap Distance" m with a wrap, int32 dec 2 "G Force - Lateral" G, int16 dec 0 "Engine RPM" rpm at another count). Checks: values (raw/100), units, `t`, `lap` 0/1/2, default map skips absent channels, signs of `delta` / `r` / `ay` factors, tuple fallback, user-map missing channel raises, unit mismatch raises, bad magic raises, shift != 0 raises, unknown class raises, resampling of the short channel. Vendor section (SKIP without `CHASSISSIM_DIR`): every `.ld` under `<CHASSISSIM_DIR>/../Models` reads with the default map; GP3 baseline lap 1 lasts within 0.05 s of its `.ldx` Fastest Time; corr(ay, vx^2 k) > 0.99 with k from `load_circuit("Eastern Creek")` at s (model signs both).
- [ ] Opus: run the test file, `tests/test_telemetry_logs.py`, `tests/test_telemetry.py`, snapshot none, STE lint on `docs/logging_spec.md`, review, commit `feat(telemetry): read ChassisSim MoTeC .ld laps (phase 1 T2)`.

### Task 3: .car codec

**Files:** create `functions/carfile.py`, `tests/test_carfile.py`. Layout: teammate files `IRP/ChassisSim_analysis/car_format_v3_41.{json,md}` and `_evidence.md`.

- [ ] Opus completes this brief when layout v1 lands: record codec (header, records, strings, arrays), the schema (named fields, units), checksum / version / padding / derived-field rules. API: `decode(path) -> dict`, `encode(car) -> bytes`, `write(car, path)` raises FileExistsError for an existing path (never overwrite).
- [ ] Gates: (1) `encode(decode(f)) == bytes(f)` for every shipped `.car` (31 files: `<CHASSISSIM_DIR>/*.car`, `<CHASSISSIM_DIR>/../Models/**/*.car`, `<CHASSISSIM_DIR>/../*.car`); fields the layout marks non-deterministic are listed, not masked silently. (2) The 7 GP3 2013 Eastern Creek variants differ only in the fields their names state (ride height 17 / 20 mm, toe 0 / 0.33 / 0.66, front damper 20000). (3) Synthetic round trip in the test without vendor files: a synthetic decoded car -> encode -> decode equal. Fields that fail go back to the teammate.
- [ ] Opus: tests, snapshot none, review, commit `feat(carfile): decode and encode ChassisSim car files (phase 1 T3)`.

### Task 4: Docs

- [ ] CLAUDE.md: one bullet each for the ChassisSim circuit source (`circuit='chassissim:<name>'`, `chassissim_dir` / `CHASSISSIM_DIR`, sign flip, open trace, base grip, slug stem), `read_ld` (map, signs, no positions, `s` = Lap Distance), the codec (gates, never overwrite), and the spec pointer; the tests list gets the new files. `CONTEXT.md`: new terms only if a task resolved one. STE, run `python3 -I ~/.claude/skills/asd-ste100/scripts/ste-lint.py <file>` on changed docs, fix hard violations in new text.
- [ ] Commit `docs: ChassisSim phase 1 (circuit source, .ld reader, codec)`.

### Validation (Opus, one at a time)
- [ ] V1 snapshot none and the quick suite (`.superpowers/sdd/chassissim/run_tests.py . <out> --quick`), then the six real-IPOPT files one at a time.
- [ ] V2 one real cold solve: `MLTP(circuit="chassissim:Eastern Creek", linear_solver="mumps", save=False, plot=False)` (4EM car, default physics). Record status, iterations, lap, wall. `MLTP_screen` lap on the same circuit. Then `telemetry.overlay_report` of that result against `read_ld(GP3 baseline)` runs (plumbing check: different cars).
- [ ] V3 final review (`caveman:cavecrew-reviewer`, Opus) of the whole branch diff, one fix wave, re-review.
