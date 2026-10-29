# Logging specification (E4)

This page lists the channels that a log of the car must hold. It gives their units, rates and signs, and the stages that use each channel. The readers of `functions/telemetry.py` (E5) and the fits of `functions/correlate.py` (E6 to E8) take these channels. A log keeps the units of the logger. The reader converts each channel to SI units and to the signs of the model.

Stages: D1 is the track builder and E1 the model export. E5 is the reader and the overlay report, and E6 to E8 are the fits. T is the tyre method of `docs/tyre_scaling_track_method.md`. Need: C (core, E5 cannot run without it), S (one stage needs it) or O (optional). A rate or a sign marked "proposal" has no source. It is a judgement.

## Table A. Continuous channels

| Channel | Unit in the log | Rate | Sign and zero | Stages | Need |
|---|---|---|---|---|---|
| time | s | one clock for all rows, at the rate of the fastest channel | 0 at the session start, the same clock as the timing | all | C |
| lap trigger | beacon times (s) or a lap number | each crossing | lap i runs from beacon i to beacon i+1. Mark an out-lap, in-lap, yellow-flag or traffic lap as not valid, with the reason. | E5, E6, T | C |
| lap distance | m, at least 2 decimals | row rate | 0 at the lap start. The integral of the speed. GPS and wheel-speed distances differ by about 0.25%. | E1, E5 | C |
| GPS latitude and longitude | deg (WGS84) | 10 Hz or more (proposal) | north and east positive. `xy_from_gps` gives east as +x and north as +y. | D1, E5 | C |
| GPS quality: satellites, accuracy | unit of the logger | with the GPS | Drop rows with few satellites (proposal). | D1, E5 | O |
| speed | km/h | 50 Hz in the export | ground speed, 0 or more, from GPS or wheel speed. The model vx is the speed in the body frame (0.1% to 0.3% apart). | E1, E5, E6, E7, E8 | C |
| lateral acceleration ay | g | 50 Hz in the export | positive in a left turn in the model. Record the sign of the log (Table C). | E1, E5, E7, T | C |
| longitudinal acceleration ax | g | 50 Hz in the export | positive when the car accelerates | E1, E5, E7, E8 | C |
| steering angle at the tyre | deg | 50 Hz in the export | positive to the left in the model (delta). A sensor at the wheel needs the steering ratio. | E1, E5, E7, T | C |
| yaw rate | deg/s | 20 Hz or more (proposal) | positive to the left in the model (r) | E5, T | O |
| damper travel FL, FR, RL, RR | mm | 50 Hz in the export | positive in bump, 0 on the ground at ride height. The model xs is the wheel travel (motion ratio 1). | E1, E8 | S |
| motor speed | rpm | 50 Hz in the export | The model `om_gear` is the mean wheel speed times `vp.gear` (7.252, a model constant). Log the true gear ratio. | E1, E8, T | S |
| throttle | % | 50 Hz in the export | The model value is the motor torque in % of Tmax (0 under regen), not the pedal. State which one the log holds. | E1, E8 | S |
| motor torque, each motor | Nm | 50 Hz or more (proposal) | positive in drive, negative in regen | E7, E8 | S |
| DC bus voltage and current | V, A | 10 Hz or more (proposal) | positive in discharge | E8, T | O |
| brake pressure, front and rear | bar | 50 Hz or more (proposal) | positive when the car brakes | E7, E8 | S |
| wheel speeds, 4 | rad/s or km/h | 50 Hz or more (proposal) | positive forward | E7, T | O |
| IMU acceleration and rate, 3 axes | g, deg/s | 50 Hz or more (proposal) | at the CG, zeroed on flat ground. Check the sign of each axis. | T | O |
| side-slip angle | deg | 20 Hz or more (proposal) | positive on the side of the yaw (proposal) | T | O |
| airspeed (Pitot) | km/h | 10 Hz or more (proposal) | 0 or more | E8 | O |
| wheel load or strain, 4 | kgf | 50 Hz | positive in compression (proposal) | E7 | O |
| ride height, front and rear | mm | 20 Hz or more (proposal) | positive up (proposal) | E8, T | O |
| altitude and slope | m, deg | with the GPS | | E8 | O |
| ambient air: temperature, pressure, humidity, wind | deg C, hPa, %, m/s | each stint (Table B) | | E6, E8, T | S |
| tyre surface temperature, each corner | deg C | each stop, or IR in the session | | T | O |

## Table B. Manual log sheet

The manual log sheet is Table 3 of `docs/tyre_scaling_track_method.md`. Use it as it is. Its blocks are:

| Block | When | Serves |
|---|---|---|
| Session | start | session and driver terms, QSS mass |
| Stint: track and air conditions | stint start | b1, air density, Equation 3 |
| Stint: pressure level, tyre set, warmers | before the stint | design, set term, tyre age |
| Each corner: cold pressure | before the stint | dp, Equation 3 |
| Stint: battery state | start and end | QSS power limit |
| Stint: laps before the pit entry | pit entry | stint validity |
| Each corner: hot pressure, tread temperatures | pit entry | safety, second outcome, Equation 3 check |
| Lap: number, time, validity | every lap | stint median, time term |

## Table C. Signs

| Quantity | Model (this repo) | ChassisSim | AiM sample (another car) |
|---|---|---|---|
| curvature | positive in a left turn | right turn positive (a box in the curvature dialog) | no channel |
| yaw rate | r, positive to the left | not in the monster file | GPS Gyro, positive to the right |
| lateral acceleration | ay, positive to the left | import flag 1: right positive, flag -1: left positive | GPS LatAcc and Lateral Acc, positive to the right |
| steer | delta, positive to the left, rad at the tyre | import flag as for ay, deg at the tyre | Steering Pos, positive to the right, deg at the steering wheel |
| longitudinal acceleration | ax, positive forward | positive forward | GPS LonAcc, positive forward. Inline Acc runs against the speed change (inverted or badly mounted). |
| damper | xs, positive in bump, m | positive in bump, mm, 0 at ride height | no channel |
| roll and pitch | phi positive with the right side down, theta positive with the nose down | | |
| lateral offset | n, positive to the left of the centreline | | |

Status of the signs:

- Model: numerical checks on the model and on the stored results agree with every sign of the model column.
- ChassisSim: five ChassisSim manuals state the import flags. Flag 1 (right turn positive) is their base case, and -1 marks a left-positive log. The monster-file manual states no sign. No file went through a ChassisSim import yet, so this column is not verified in use.
- AiM: the signs come from one log of another car. They are not verified for the logger of this car.
- Export: `export_monster` has the default `sign=(-1, -1)` for (ay, steer). So the file is right positive, and both ChassisSim flags are 1. With `sign=(1, 1)` the file keeps the signs of the model, and both flags are -1. `read_monster` takes the same pair, so a file reads back with the signs of the model.
- Check: `overlay_report` gives the correlation of ay and steer of the log with the simulation. It warns when one of them is below 0. Then change the sign of that channel in the channel map.

## Channel map of `read_csv`

`read_csv(path, channel_map)` takes `{channel: column}` or `{channel: [column, unit, factor]}`, or the path of a JSON file with the same content. The channel is the factor times the column value. If the file has a units row, it must show the unit of the map. The factor holds the unit change and the sign. For example, a lateral acceleration in g that is right positive has the factor -9.80665. The lap end times of the beacon row take the factor of the channel t. So a time column in ms (factor 1e-3) also gives the laps. The repo ships no map file, because the signs of the sample logger are not verified for this car.

## Rates

- The vendor manuals ask for every monster column at 50 Hz (a sample time of 0.02 s).
- The sample logger exports at 20 Hz.
- The model holds no detail above about 1 Hz to 3 Hz: the BCN intervals last 0.31 s to 1.50 s. So the 50 Hz export is an interpolation. Compare a log with the model after a low-pass filter of about 0.5 s (at 60 m/s).
- Proposal: log each channel at its source rate, and state the rate. Do not up-sample. Use at least 20 Hz for the channels of E5 to E7 and 10 Hz for GPS and slow channels.

## Minimum channels per stage

The fits read a dict of SI arrays on one time base, with the signs of the model.

| Stage | Minimum | Adds |
|---|---|---|
| E5 overlay | t, lap trigger, latitude and longitude or lap distance, speed, ax, ay, steer | yaw rate, dampers |
| E6 grip (`grip_fit`) | t, vx, ax, ay | brake pressure for trail braking |
| E6 drag (`drag_fit`) | t, vx, ay, and Tm (total motor torque, Nm) or throttle | ax, a brake channel (0 or 1), air temperature and pressure, airspeed or wind, slope |
| E6 downforce (`lift_fit`, `physics='corrected'` only) | t, vx, ax, ay, xs_fl, xs_fr, xs_rl, xs_rr | spring rates within 3%, damper motion ratios |
| D1 track builder | latitude and longitude of one clean lap | smoothing |
| T, core data | lap times and the Table B sheet | sector times, then GPS speed |

## Risks

- A steering angle at the wheel, not at the tyre, makes the comparison wrong. Divide by the steering ratio.
- An IMU can give spikes or an axis with the wrong sign. The sample log shows spikes, and one of its axes runs against the speed change. Use the GPS acceleration when the IMU has no calibration.
- The telemetry module uses g = 9.80665 m/s^2, and the model uses `vp.g` = 9.81 m/s^2. Use 9.80665 to convert g in both directions.
- The integrated lap distance drifts. Use the projection on the centreline (`project`) for the time delta.
- The start of the track file (s = 0) is not the timing line. Log the position of the timing line or of the beacon (proposal).

## Owner inputs

1. The logger, its channels, rates and units, and the inverter channels on CAN.
2. The location and ratio of the steering sensor, the gear ratio and the damper motion ratios.
3. The signs of ay and steer in that logger. They decide the import flags and the channel map.
4. Sector timing, ride-height sensors and a Pitot tube, if the car has them.
