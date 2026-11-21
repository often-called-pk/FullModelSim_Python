# ChassisSim cs2c: brake ramp, strut and live axle, roll-centre modes, anti-dive

Date 2026-11-21. Owner ask 01:45: work through the open cs2c board cards in order (brake delay closed: cid 109 is dead; brake ramp; strut kinematics). Grilling answers: Q1 build the live axle now (V8 rear). Q2 validation = stored roll-centre values + real solves + owner ChassisSim runs of the WRX and the V8 at Eastern Creek (board card CS2c-CSRUN, the build does not wait). Q3 fix the roll-centre reading in this card. Q4 build the anti-dive / anti-squat term in this card. Decode notes (local): `.superpowers/sdd/chassissim/map/teammate_strut.md`, `teammate_rc_modes.md`, `teammate_p2c_powertrain.md` section 3; ledger `.superpowers/sdd/2026-11-21-cs2c-ramp-strut/progress.md`. Flow per lane: Sonnet build -> Opus audit -> Fable fix -> Haiku tests -> commit. One lane at a time (shared files).

Decode facts this plan rests on (O = read in the binary, I = inferred):
- Hardpoint record (tag 10) type byte [O]: 0 double wishbone (6 points), 1 MacPherson strut (4 points: lower arm inner front, inner aft, lower ball joint, top mount), 2 live axle (6 link points, not read by the solve). load_car drops the byte today. Shipped cars: 27 all type 0, WRX all type 1, V8 rear type 2.
- Geometry mode per axle, cid 53 front / 61 rear [O]: 0 live solve; 1 manual roll centre = cid 54 / 62 + (ride height - datum ride height, cid 142 / 144), static camber, no steer camber (cid 53 = 1); 2 live-axle roll centre 0; 3 solve with static camber. Shipped: mode 0 everywhere except the V8 rear and two demo cars (mode 1). Modes 2 and 3: no shipped car.
- Mode 0 [O]: contact patch at z = 0 of the hardpoint frame, at the datum ride height (cids 142 / 144); the solve pose takes ride height - datum ride height as heave. Front-view block of the loader: each inner pair cut at the axle plane (x = 0 front, x = wheelbase cid 90 rear); outer y raw, outer z = height of the arm plane (3 points) at (axle plane, outer y). Our check: the block reproduces the stored front-view height cid 54 / 62 to 0.01 mm on 7 cars; older files carry stale values (GP3 front, V8 front), and mode 0 never reads them.
- Cids 55 / 56 front, 63 / 64 rear [O]: side-view anti-dive / anti-squat heights, not roll-centre heights. The phase 2b ground back-out from cid 55 is wrong for every mode-0 car. GP3 log, rear roll centre: corrected median error +2.5 mm (projection rule) against -3.0 mm today; front -1.9 mm against +5.0 mm.
- Strut [O]: instant centre = lower-arm line x the line through the top mount perpendicular to the strut axis; camber = change of the strut-axis angle; motion ratios = the stored polynomials. WRX stored heights reproduced to 0.03 / 0.17 mm.
- Live axle (V8 rear: type 2, mode 1, cid 69 = 1) [O]: roll centre by mode 1; camber = static +- d, d = (hub height rear right - rear left) / rear track in the road frame (the live-axle law overrides mode 1); no body roll added; two independent wheel states (no vertical coupling) [I strong].
- Steer camber [O]: one linear K1 x steer term after the solve, same for struts and wishbones. Our K1 formula takes the strut axis (lower ball joint -> top mount) as the steering axis.
- Brake ramp [O]: once per braking zone, each wheel force = (t / T) x its capped onset force for t < T, then free.

## Lane R: brake ramp (done)

- [x] cid 571 on: brake input rate bound -1 / T (cid 572) in the application direction, release unchanged, tighter bound kept. 572 checks. 571 decode `brake.ramp_on`. QSS ignores the ramp.
- [x] One real solve with a binding T (GP3 on Sturn, T = 2 s, cold, MUMPS): 16.313 s against 16.150 s, fastest brake application -0.495 1/s at the -0.5 bound (the base reaches -1.22 1/s). Committed 8cadb01.

## Lane K1: type byte, geometry modes, gates

Files: functions/chassissim.py.

- [ ] load_car keeps the tag-10 type byte: `hardpoints.<corner>_type`. Decode cid 53 `geom.mode_f`, 61 `geom.mode_r` (int), 142 / 144 `geom.datum_rh_f` / `_r` (m), 69 `susp.live_axle`.
- [ ] check_car gate per axle, (type, points, mode): front (0, 6, 0 or 1), (1, 4, 0 or 1); rear the same plus (2, 6, 1) with cid 69 = 1. Anything else raises one ValueError naming the cid, the type and the mode (modes 2 and 3, type 2 in mode 0 or at the front, cid 69 without a type-2 rear, types 3-9). Replaces the 6-point refusal.
- [ ] Missing fields: a mode-1 axle needs cid 54 / 62 and the datum ride height; a mode-0 axle needs the hardpoints and the datum ride height.
- [ ] Checks: every shipped car decodes; WRX and V8 pass the gate; the 27 type-0 cars unchanged in load_car output apart from the new fields.

## Lane K2: kinematics by the ChassisSim rules

Files: functions/chassissim.py (axle_kinematics, suspension_curves), vehModel_cs.py (camber per corner), functions/ggv.py (QSS camber and roll centre).

- [ ] Front-view block by the loader rule (inner pairs cut at the axle plane, outer z on the arm plane). ponytail: float64; ChassisSim rounds to float32 (below 0.01 mm).
- [ ] Mode 0: ground at z = 0 of the hardpoint frame. The curves stay indexed by our wheel travel w (static = 0, bump positive); the geometry is solved at w + (datum ride height - static ride height of the axle). Remove the cid 55 back-out, the plausibility check and the cid 54 fallback. Keep the no-solution error.
- [ ] Strut (type 1): the lower ball joint swings on the lower-arm circle, the top mount is fixed, the upright follows the strut axis, the contact patch is rigid to the upright. Instant centre and camber as decoded. K1 from the strut axis.
- [ ] Mode 1: rc_h(w) = cid 54 / 62 + (static - datum ride height) - w, camber curve 0, K1 0 on a mode-1 front.
- [ ] Live axle: rear camber = static +- d from the road-frame hub heights of the model (wheel states), no body roll, no kinematic curve; QSS the same from the steady tyre deflections.
- [ ] Checks: the stored front-view heights where they hold (7 cars exact, WRX 0.03 / 0.17 mm), GP3 log medians, the strut and live-axle laws on hand cases, mode-1 slope -1, snapshot of the 4EM NLP unchanged, every ChassisSim QSS pin that moves listed with old and new values.

## Lane K3: anti-dive / anti-squat (pitch-moment arm)

Files: vehModel_cs.py. Decode [O] (FUN_14012a600, equations of motion of the time march): Fx = signed tyre longitudinal force, drive positive, brake negative. The pitch moment of the axle tyre forces uses the arm (h_cg - h_axle) instead of h_cg: pitch accel = (Fx_front_sum (h_cg - h_f) + Fx_rear_sum (h_cg - h_r)) / I_y. h_f = cid 56 when the front force sum > 0 (drive), else cid 55 (brake, coast); h_r = cid 64 / 63 likewise. Positive h = anti (less pitch). The vertical jacking force (cids 800-807) is off in every shipped car: not built. The QSS and the apex search do not use the heights: QSS unchanged.

- [ ] Pitch arm per axle reduced by the side-view height, NLP only. The switch on the force sign is a smooth blend (SX smooth), scale named in a ponytail note.
- [ ] ponytail: mode 0 adds a small ride-height shift to h (divisor not decoded); skipped, add when decoded. Left copies (59, 60, 67, 68) only behind gates no shipped car sets: not read.
- [ ] Checks: hand values of the moment on a synthetic car (drive, brake, coast); all four heights 0 builds the graph of today; snapshot of the 4EM NLP unchanged; the GP3 pitch under braking smaller than today.

## Lane V: validation (Opus, one solve at a time, memory check before each)

- [ ] QSS laps of every car that builds (GP3, GT3, F2, WRX, V8, the two mode-1 demo cars, others), old vs new.
- [ ] Real solves: GP3 Eastern Creek fixed line (re-pin, overlay corr ay, RMS speed, stbi against the log), WRX and V8 on Sturn and at Eastern Creek. Overlays of the WRX and V8 wait for the owner runs (CS2c-CSRUN).

## Lane D: docs (Opus)

- [ ] CLAUDE.md ChassisSim paragraphs (roll-centre reading, modes, strut, live axle, anti-dive, ramp), the test list, the moved numbers. README only where a user step changes.
