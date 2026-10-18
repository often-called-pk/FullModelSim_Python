# Lap-time simulation: sources and the physics behind accuracy

Compiled 17 October 2026 for the FullModelSim_Python project.

## Scope and method

This document collects the public sources on lap-time simulation that explain why a tool such as
ChassisSim correlates well with telemetry. It covers six themes: transient versus quasi-steady-state
simulation, driver models, tyre models, damper and ride models, aeromaps, and telemetry correlation.
Each source has a short summary. The closing sections explain the physics and the numerical methods,
and relate them to this project.

Sources came from web search, from the ChassisSim blog and its PDFs, and from the documents that ship
with ChassisSim v3.41. The Racecar Engineering website blocks automated readers, so Nowlan's magazine
articles are collected through their republication on the ChassisSim site and through his book.

Each citation carries a tag:

- **[V]** verified online during this review, with a link.
- **[K]** cited from prior knowledge. The work exists, but check the volume and page details before
  you cite it in a thesis.

## 1. The method landscape

Massaro and Limebeer (2021) [V] give the standard taxonomy. A lap-time method makes three choices.

1. **Vehicle model.** Quasi-steady-state (QSS) or transient. A QSS model assumes force equilibrium at
   each point on the track, so it needs no time integration. A transient model integrates the
   equations of motion, so yaw inertia, roll, pitch, dampers and tyre lag all act.
2. **Trajectory.** Fixed or free. A fixed method drives a given line, often the measured one. A free
   method optimises the line and the speed together.
3. **Solution.** Closed-loop simulation or open-loop optimal control. A simulation drives the model
   with a driver model, in forward time. Optimal control finds the controls that minimise time, with
   no driver model, by direct transcription to a nonlinear program or by an indirect method.

ChassisSim is a transient, closed-loop simulation. It integrates a multibody model in time and drives
it with a control-theory driver model. The user supplies the line as a curvature profile. The
FullModelSim_Python project is a transient, free-trajectory, open-loop method. It transcribes the
problem with direct Legendre collocation in the arc-length domain and solves it with IPOPT. The two
tools answer different questions. ChassisSim predicts what a real car does on a known line with a
given setup. An optimal-control solver predicts the best possible lap for a given model.

## 2. Danny Nowlan and ChassisSim sources

Nowlan is the founder of ChassisSim Technologies and a long-time contributor to Racecar Engineering.
His book grew out of those articles.

### 2.1 Primary technical documents

**Nowlan, D. (2011). "The ChassisSim approach to lap time simulation: what makes us unique."** [V]
PDF: https://www.chassissim.com/wp-content/uploads/CSim_laptime_sim_approach.pdf

- Quasi-static simulation rests on the d'Alembert equilibrium assumption. At each point the inertial
  force at the centre of gravity, m V^2 / R, balances the tyre forces. The lap is many such snapshots
  stitched together.
- Nowlan credits the method: it runs fast, and it judges wings and gear ratios well.
- It misses the transients. His V8 Supercar data shows measured roll 11 to 13 percent above the
  quasi-static value. His worked example puts the damper force near 20 percent of the spring force.
- Quasi-static simulation cannot model bumps. Tyre load variation through a corner matters almost as
  much as the mean load, because grip follows load. Both points favour a time-domain model.
- ChassisSim uses a fully transient multibody engine for every function. He states that one lap
  simulation equals a run on a seven-post rig, and that the method depends less on grip factors.
- The paper does not describe the driver model or the integration scheme.

**Nowlan, D. "Race car damping: some food for thought" (Damper guide part 1).** [V]
Originally in Racecar Engineering. PDF: https://www.chassissim.com/wp-content/uploads/DN_Damp_guide_part_1.pdf

- Starts from the quarter-car model under a sinusoidal road input.
- Asks what damping holds the body still. The answer, a damping rate proportional to tan(2 pi f t),
  goes to infinity when the damper velocity passes through zero. A damper cannot hold position.
- A bilinear, or dual-rate, damper approximates the ideal curve. The high initial slope acts like
  spring preload, which can hurt the car under braking and cornering loads.
- An inerter of inertance b = K / (2 pi f)^2 holds the body still at one frequency only. At low
  frequency it needs an impractical value.
- Conclusion: no passive element replaces active suspension, so damper tuning is track specific.

**Nowlan, D. "Race car damping 2" (Damper guide part 2).** [V]
Originally in Racecar Engineering. PDF: https://www.chassissim.com/wp-content/uploads/DN_Damp_guide_part_2.pdf

- Assumes the tyre is much stiffer than the spring and the unsprung mass is small. The sprung mass
  then behaves as a second-order system.
- Natural frequency: omega_0 = sqrt(K / m). Damping ratio: zeta = C / (2 m omega_0). Use wheel
  rates and strict SI units. The damping rate C is the local slope of the damper force curve.
- Worked example: 175,000 N/m, 157 kg, 21,333 N s/m gives omega_0 = 33.4 rad/s and zeta = 2.03.
- Guide: zeta of 0.3 to 0.4 filters bumps. Zeta of 0.5 to 1.0 controls the body. Zeta above 1.0
  gives extreme body control and puts temperature into the tyre.
- A damper's ratio table reads like a design intent. High ratios at low speed show body control,
  low ratios at high speed show bump absorption.
- Key sentence for this review: Nowlan rejected the quasi-static lap-time algorithm because it could
  not evaluate dampers. That choice created ChassisSim.

**ChassisSim v3.41 shipped documents.** [V] These are summarised in the companion file
`chassissim-matlab-interop/MODEL_NOTES.md`. In brief:

- `CSim_v3_tyre_model_documentation.pdf`: a semi-empirical tyre model. A 4th-order load polynomial
  times a temperature peak function sets the traction-circle radius. A ten-point normalised slip table
  in the Radt form shapes the force. Camber and the traction ellipse share one set of terms.
- `CSim_aero_model_creation.pdf`: a ride-height aeromap built from telemetry "slivers", fitted by a
  second-order surface or by per-slice peak-minus-curvature optimisation.
- `CSim_model_refinement.pdf`: the correlation workflow. The aero toolbox reverse-engineers the
  aeromap first. The tyre force toolbox then fits the tyre model. Each tyre fit runs for hours.
- `CSim_process_made_simple.pdf`: the model-building order. Start from a template, build the
  "monster file" (a 50 Hz tab-delimited flying lap with optional loads), run one-touch modelling,
  then create the circuit.

### 2.2 Book

**Nowlan, D. (2010, hardcover 2019). The Dynamics of the Race Car. ChassisSim Technologies.
ISBN 978-0-9870593-0-7, 241 pages.** [V]
Review: https://speedreaders.info/23030-the-dynamics-of-the-race-car/

The review states that the book has two origins, and one is Nowlan's writing for Racecar
Engineering. It describes nearly every page as dense with mathematics. This book is the collected
form of the magazine series.

### 2.3 ChassisSim technical articles

The blog has 16 archive pages. The table lists the technical posts found in this review, by theme.
Many posts introduce a video, so the text is short and the method is in the video.

| Date | Title | Theme |
|---|---|---|
| 2011-02-23 | [What makes ChassisSim unique for Lap Time Simulation](https://www.chassissim.com/what-makes-chassissim-unique-for-laptime-simulation/) | Transient vs quasi-static |
| 2017-08-09 | [What Makes ChassisSim Unique?](https://www.chassissim.com/what-makes-chassissim-unique/) | Transient vs quasi-static |
| 2011-09-22 | [How to use lap time simulation properly](https://www.chassissim.com/how-to-use-lap-time-simulation-properly/) | Method, expectations |
| 2017-04-27 | [Why Simulation needs to be one of the first things you do](https://www.chassissim.com/why-simulation-needs-to-be-one-of-the-first-things-you-do/) | Method |
| 2018-08-09 | [The Racecar Engineering/Simulation Process](https://www.chassissim.com/the-racecar-engineering-simulation-process/) | Correlation process |
| 2026-09-24 | [Simulation: why you need to do it yourself](https://www.chassissim.com/simulation-why-you-need-to-do-it-yourself-if-you-can/) | Correlation |
| 2017-03-16 | [Incorporating Race car handling into Lap Time Simulation](https://www.chassissim.com/incorporating-race-car-handling-into-lap-time-simulation/) | Driver and handling |
| 2020-05-15 | [Incorporating Handling into Lap Time Simulation Revisited](https://www.chassissim.com/incorporating-handling-into-lap-time-simulation-revisited/) | Driver and handling |
| 2026-05-21 | [Neutral steer and car stability](https://www.chassissim.com/neutral-steer-and-sar-stability-connecting-the-dots/) | Handling |
| 2020-10-06 | [Lap-time simulation and driver in the loop combined](https://www.chassissim.com/combining-lap-time-simulation-driver-in-the-loop-simulation/) | Driver in the loop, one engine |
| 2023-07-07 | [ChassisSim Driver in the Loop: Getting Started](https://www.chassissim.com/chassissim-driver-in-the-loop-getting-started/) | Driver in the loop |
| 2015-09-24 | [Creating tire models with no test rig data](https://www.chassissim.com/creating-tire-models-with-no-test-rig-data-using-chassissim/) | Tyre from data |
| 2016-03-31 | [Creating tyre models from a blank sheet of paper](https://www.chassissim.com/creating-tyre-models-from-nothing-2/) | Tyre from data |
| 2026-07-09 | [Dialling in the Tyre's Traction Circle radius](https://www.chassissim.com/dialling-in-the-tyres-traction-circle-radius-with-chassissim/) | Tyre |
| 2023-09-07 | [Modelling Tyre Pressures and their effects](https://www.chassissim.com/modelling-tyre-pressures-in-chassissim-and-their-effects/) | Tyre |
| 2023-09-21 | [Modelling Tyre Surface Temperature](https://www.chassissim.com/modelling-tyre-surface-temperature-in-chassissim-and-their-effects-on-vehicle-performance/) | Thermal tyre |
| 2013-07-04 | [The Damper Guide Part 1](https://www.chassissim.com/the-damper-guide-part-1-using-damping-ratios/) | Dampers, Racecar Engineering series |
| 2017-08-18 | [Improve dampers using ChassisSim](https://www.chassissim.com/evaluating-dampers-using-chassissim/) | Dampers |
| 2016-05-05 | [Using ChassisSim to evaluate suspension geometry](https://www.chassissim.com/using-chassissim-to-evaluate-suspension-geometry/) | Suspension kinematics |
| 2020-06-11 | [The ChassisSim Suspension Geometry Viewer](https://www.chassissim.com/the-chassissim-suspension-geometry-viewer/) | Suspension kinematics |
| 2020-08-13 | [Lateral load transfer at the front](https://www.chassissim.com/lateral-load-transfer-at-the-front-magic-number-what-really-drives-it/) | Load transfer |
| 2024-10-31 | [Working out Aeromaps using Pitch Data](https://www.chassissim.com/working-out-aeromaps-using-pitch-data/) | Aero from data |
| 2026-09-17 | [Aero Surface Optimisation](https://www.chassissim.com/aero-surface-optimisation-a-really-effective-way-of-dialling-in-your-aeromaps/) | Aero from data |
| 2018-01-18 | [3 minute Winning Tips: Ride Heights](https://www.chassissim.com/chassissim-3-minute-winning-tips-ride-heights/) | Ride height sensitivity |
| 2024-05-09 | [Circuit model creation revisited](https://www.chassissim.com/circuit-model-creation-connecting-the-dots/) | Track from data |
| 2025-02-13 | [Generating altitude and road camber from data](https://www.chassissim.com/generating-altitude-and-road-camber-to-your-circuit-models-from-data/) | Track from data |
| 2024-03-08 | [Running ChassisSim on Ovals](https://www.chassissim.com/running-chassissim-on-ovals/) | Track |
| 2019-09-05 | [Using Advanced Simulation Channels](https://www.chassissim.com/using-advanced-simulation-channels/) | Output channels |
| 2016-09-13 | [Front Wheel Drive dynamics and tuning](https://www.chassissim.com/front-wheel-drive-its-dynamics-and-tuning-using-chassissim/) | Drivetrain |
| 2025-07-24 | [Understanding the Limited Slip Differential](https://www.chassissim.com/understanding-the-limited-slip-differential/) | Drivetrain |
| 2026-06-12 | [Quantifying the effects of the Differential](https://www.chassissim.com/quantifying-the-effects-on-the-differential-on-racecar-handling/) | Drivetrain |
| 2026-10-15 | [Differential Optimisation](https://www.chassissim.com/differential-optimisation-getting-the-most-of-the-racing-differential/) | Drivetrain |
| 2018-02-09 | [v3.31 Release and Setup Sweeping](https://www.chassissim.com/the-chassissim-v3-31-release-and-setup-sweeping/) | Setup optimisation |
| 2026-08-20 | [Setup Sweeping Redux](https://www.chassissim.com/setup-sweeping-redux/) | Setup optimisation |
| 2014-06-12 | [ChassisSim Altair webinar](https://www.chassissim.com/chassissim-altair-vehicle-simulation-webinar/) | Tyre toolbox, FTire |

Related: a 2019 Altair webinar by Nowlan on tyre modelling and dampers
(https://web.altair.com/2019-chassissim-apa-webinar) [V]. A 2025 SAE paper uses ChassisSim beside
the point-mass tool OpenLAP for Formula SAE design (SAE 2025-01-0380) [V].

### 2.4 What is not public

The control law of the ChassisSim driver model is not documented in any public source found here. The
vendor states that it is a control-theory model. The same numerical engine serves both lap-time
and driver-in-the-loop modes, and handling behaviour is included. The internal formulation stays
proprietary.

## 3. Transient versus quasi-steady-state

### 3.1 Sources

**Siegler, B., Deakin, A., Crolla, D. (2000). "Lap time simulation: comparison of steady state,
quasi-static and transient racing car cornering strategies." SAE 2000-01-3563.** [V]
https://saemobilus.sae.org/content/2000-01-3563
Compares the three strategies against measured behaviour. It notes that quasi-static models ignore
dynamic effects such as yaw damping.

**Kelly, D. P. (2008). Lap time simulation with transient vehicle and tyre dynamics. PhD thesis,
Cranfield University.** [V] https://dspace.lib.cranfield.ac.uk/handle/1826/4791
Finds the optimal line and the steer, throttle and brake inputs together. The method integrates the
vehicle model forward in time only, so it captures the full transient response. It includes a
thermodynamic tyre model and compares results with a quasi-steady-state method.

**Brayshaw, D. L., Harrison, M. F. (2005). "A quasi steady state approach to race car lap simulation
in order to understand the effects of racing line and centre of gravity location." Proc. IMechE Part D,
219(6).** [K, journal issue checked online]
A seven-degree-of-freedom model and optimal control, with a g-g-speed diagram as an intermediate
result. Low cost compared with transient optimal control. A 6 percent centre-of-gravity shift changed
the lap time in both the QSS and the transient method, but moved the optimal line very little.

**Casanova, D., Sharp, R. S., Symonds, P. (2000). "Minimum time manoeuvring: the significance of yaw
inertia." Vehicle System Dynamics 34(2).** [K]
Shows that yaw inertia changes the minimum-time solution. A QSS model has no yaw inertia, so it cannot
see this effect.

**Lot, R., Dal Bianco, N. (2015). "The significance of high-order dynamics in lap time simulations."
IAVSD 2015.** [K]
Compares models of rising order on the same track and quantifies which dynamics change the lap time.

**Veneri, M., Massaro, M. (2019, online). "A free-trajectory quasi-steady-state optimal-control method
for minimum lap-time of race vehicles." Vehicle System Dynamics.** [V, volume unconfirmed]
https://www.research.unipd.it/handle/11577/3297111
Builds g-g maps from a full vehicle model, then solves a point-mass QSS problem with a free trajectory.
It is a middle path between fixed-line QSS and full transient optimal control.

**Massaro, M., Limebeer, D. J. N. (2021). "Minimum-lap-time optimisation and simulation." Vehicle
System Dynamics 59(7), 1069 to 1113. DOI 10.1080/00423114.2021.1910718.** [V]
The survey. It covers QSS and transient models, fixed and free trajectories, nonlinear optimal control
fundamentals, three-dimensional road models and vehicle positioning.

**Nowlan (2011), above.** The practitioner's case: measured roll above the quasi-static value, damper
force near 20 percent of spring force, no bumps in a quasi-static model.

### 3.2 The physics

A quasi-steady-state model solves force and moment balance at each track point. The solution is
exact only when the car does not accelerate in roll, pitch or yaw, and when no bump acts. Four
effects break that assumption in a real corner.

1. **Yaw inertia.** The car must build and shed yaw rate at turn-in and exit. The yaw moment needed
   to do that comes from a front and rear tyre force imbalance. The imbalance is not available for
   lateral grip. Casanova, Sharp and Symonds quantified the lap-time cost.
2. **Load transfer dynamics.** Roll and pitch do not reach their steady values at once. Springs,
   dampers and bars set the time constants. Transient load transfer changes tyre loads during the
   phases that decide the corner, which are braking, turn-in and exit.
3. **Tyre load variation.** Tyre force rises with load at a falling rate. The force versus load curve
   is concave. For a given mean load, a fluctuating load gives less mean grip than a constant load.
   This is why bumps and poor damping cost grip, and why Nowlan writes that load variation counts
   almost as much as mean load. A QSS model holds load constant along each segment, so it cannot show
   the loss.
4. **Tyre lag and temperature.** Tyre force builds over a relaxation length, and grip follows carcass
   temperature with a lag of seconds. Both are dynamic states.

A transient model carries all four as states. That is why it predicts damper changes, bump response
and balance through a corner, and why a QSS model cannot.

### 3.3 The numerical methods

**Quasi-steady-state.** Build the performance envelope, the g-g or g-g-v diagram, from the vehicle
model at each speed. Then find the speed profile along a fixed line. The common scheme runs a forward
pass limited by traction and a backward pass limited by braking, and takes the lower speed at each
point. Cost is low, and the result is a good first estimate.

**Transient closed-loop simulation (ChassisSim).** Integrate the multibody equations of motion in
time. A driver model generates steering, throttle and brake to follow the line at the limit. The speed
profile emerges from the driver's search for the limit, not from an envelope. Each run needs a driver
model that behaves like a real driver, which is the hardest part to get right.

**Transient open-loop optimal control (this project).** State the lap as an optimal control problem.
Use the arc length along the track as the independent variable. Lap time then becomes an integral of
the inverse speed, and the track geometry enters as curvature. Place the car in curvilinear
coordinates relative to the centreline. Discretise with direct collocation, and solve the nonlinear
program with IPOPT. Perantoni and Limebeer (2014) [V] used this form for a Formula One car, with
analytic derivatives and non-dimensional scaling to cut solve times. Dal Bianco, Bertolazzi, Biral and
Massaro (2019) [V] compare this direct route with the indirect route, which solves the Pontryagin
boundary-value problem.

The three methods trade cost for fidelity. QSS runs in seconds. A closed-loop transient lap runs in
minutes and needs a tuned driver. A transient optimal control lap can take minutes to an hour, but it
needs no driver model and gives the true minimum for the model.

## 4. Driver models

A closed-loop simulation needs a controller that plays the driver. The literature has three families.

**MacAdam, C. C. (1981). "Application of an optimal preview control for simulation of closed-loop
automobile driving." IEEE Trans. Systems, Man and Cybernetics SMC-11(6).** [V]
https://trid.trb.org/View/174360
A single-point optimal preview controller on a linear single-track model. It minimises the path error
at one preview time, with a perception and neuromuscular lag. Still the reference lateral driver in
commercial tools, including the MathWorks Lateral Driver block.

**Sharp, R. S., Casanova, D., Symonds, P. (2000). "A mathematical model for driver steering control,
with design, tuning and performance results." Vehicle System Dynamics 33(5).** [K, existence checked]
A multi-point preview, optimal linear quadratic steering controller. The driver sees the road ahead at
several points and weights them.

**Thommyppillai, M., Evangelou, S., Sharp, R. S. (2009). "Optimal path-tracking of virtual race-cars
using gain-scheduled preview control." Vehicle System Dynamics 47(12).** [V]
https://spiral.imperial.ac.uk/entities/publication/4efcfdd4-a829-4611-81ba-fa055f1c470e
Extends linear preview control to large lateral accelerations by scheduling the gains with the
operating point. Companion work adds low-pass filters for driver bandwidth. The aim is a fast virtual
racing driver that can hold the car at the limit of steady cornering.

**Anderson, J., Ayalew, B. (2016, 2017). "Modeling professional drivers with model predictive
control." American Control Conference 2016, and Vehicle System Dynamics 2017.** [V]
Models the driver as a receding-horizon MPC controller. An outer loop tunes the cost weights to cut
the global lap time, because a finite horizon is sub-optimal on its own.

**Nowlan.** ChassisSim drives the car with a control-theory driver model and includes handling
behaviour in the lap-time result. The law is not public.

**Why it matters for accuracy.** A driver model adds behaviour. A good one reproduces how a real
driver finds the limit, including the sub-optimal parts, so the simulated lap looks like a real lap.
Optimal control removes the driver and returns the best possible lap. The first correlates to a real
driver. The second bounds what any driver can do. Both are valid, for different questions.

## 5. Tyre models

**Pacejka, H. B. Tire and Vehicle Dynamics (3rd ed., 2012).** [K] The Magic Formula family. An
empirical fit of force against slip, with load, camber and pressure terms. The industry default for
handling simulation. A Brookes study [V] notes that it describes motorsport extremes poorly, where
temperature and sliding speed change a lot.

**Radt, H. S., Milliken, W. F. (1960). "Motions of skidding automobiles." SAE.** [V, via Milliken
Research] The origin of the nondimensional tyre treatment.

**Radt, H. S., Glemming, D. A. (1993). "Normalization of tire force and moment data." Tire Science and
Technology 21(2).** [V] https://tst.kglmeridian.com/view/journals/tist/21/2/article-p91.xml
Normalises force, moment, slip angle and slip ratio with semi-empirical theory, so data at all loads
collapse onto one curve.

**Radt, H. S. (1994). "An efficient method for treating race tire force-moment data." SAE 942536.** [V]
**Radt, H. S. (1995). "Processing of tire force/moment data." SAE 951048.** [V] Applies the
normalisation to race tyres and to combined slip through a normalised resultant force.

**Kasprzak, E. M., Lewis, K. E., Milliken, D. L. (2006). "Tire asymmetries and pressure variations in
the Radt/Milliken nondimensional tire model." SAE 2006-01-1968.** [V] Adds pressure and asymmetry.

**Milliken, W. F., Milliken, D. L. (1995). Race Car Vehicle Dynamics. SAE.** [K] The textbook that
presents the nondimensional model, the friction circle and the g-g diagram.

**Kelly (2008), above.** A thermodynamic tyre model inside a time-optimal lap solver.

**Tremlett, A. J., Limebeer, D. J. N. (2016). "Optimal tyre usage for a Formula One car." Vehicle
System Dynamics 54(10).** [V] https://pure.uj.ac.za/en/publications/optimal-tyre-usage-for-a-formula-one-car/
A minimum-lap-time optimal control problem with a thermodynamic tyre wear model. Small control changes
cut wear with little lap-time cost.

**West, W. J., Limebeer, D. J. N. (2022). "Optimal tyre management for a Formula One car." Vehicle
System Dynamics.** [V] Adds a carcass temperature state. Wear depends on surface temperature and
contact-patch power. Multi-lap optimisation schedules tyre changes.

**Nowlan, ChassisSim v3 tyre documentation and 2023 thermal post.** [V] Uses the Radt normalised
form with a lookup table instead of the Magic Formula, because the author distrusts the Magic
Formula after the peak. Grip peaks at an optimum carcass temperature, with a parabolic or
exponential fall each side.

### The physics

Three tyre properties drive lap-time accuracy.

1. **Load sensitivity.** The friction coefficient falls as load rises. This sets the cost of load
   transfer and of load variation, and it decides the balance of the car.
2. **Combined slip.** Lateral and longitudinal force share one friction limit. The friction circle,
   or ellipse, is the first-order model. The normalised resultant-force method of Radt gives a
   data-driven form.
3. **Temperature and lag.** Grip follows carcass temperature with a peak and a lag. Force builds over
   a relaxation length. Both need states in the model, and only a transient model carries them.

## 6. Dampers and ride

**Nowlan, damper guides 1 and 2, above.** [V] Quarter-car theory, damping ratios, the inerter result,
and the ratio guide. The direct link between damper evaluation and the choice of a transient engine.

**Kowalczyk, H. (2002). "Damper tuning with the use of a seven post shaker rig." SAE 2002-01-0804.** [V]
https://saemobilus.sae.org/content/2002-01-0804
Pairs quarter-car and seven-degree-of-freedom models with a real Champ Car rig test. The rig minimises
tyre load variation while it keeps body control.

**AVL Racing damper model (MathWorks Student Lounge, 2021).** [V]
https://blogs.mathworks.com/student-lounge/2021/10/11/avl-racing-models-shock-absorbers-for-race-cars/
Conventional damper models omit hysteresis from oil compressibility. AVL built a restoring-force,
force-state-map model from dynamometer runs for transient lap simulation.

**Milliken and Milliken (1995), chapter on dampers.** [K]

### The physics

A damper does no work at steady state. Its force exists only while the suspension moves, so it has no
place in a QSS model. In a transient model the damper sets how fast load transfer builds and how much
tyre load varies over bumps. It also sets how the ride height moves, and with it the aeromap. Nowlan's ratio
guide gives the design intent in one number per velocity range. The seven-post rig measures the same
quantity, tyre load variation, that a transient simulation predicts.

## 7. Aeromaps

**Nowlan, CSim_aero_model_creation.pdf and the 2024 and 2026 aero posts.** [V] A map of CLA, CDA and
aero balance against front and rear ride height, built from telemetry. Each run gives one path, a
"sliver", through the ride-height space. A test matrix of ride-height steps and wing sweeps supplies
many slivers. A second-order surface fit, or a per-slice peak-minus-curvature fit, produces the map.

**Carreira, M. (Técnico Lisboa thesis). Formula Student aerodynamic map from a surrogate model.** [V]
https://fenix.tecnico.ulisboa.pt/downloadFile/845043405586591/MiguelCarreira_Thesis.pdf
Over 100 CFD points feed a surrogate map. Ride height changed CdA by about 10 percent and ClA by about
20 percent between peaks. Yaw was the largest sensitivity. Airspeed estimation was the main
uncertainty in track correlation.

### The physics

Downforce depends on ride height, pitch, roll and yaw, and ground-effect sensitivity rises as the car
gets lower. Ride height moves with speed, braking, cornering and bumps. So the aero load changes
through the corner, which changes the tyre loads, which changes the ride height again. Only a
transient model that carries ride height as a state closes this loop. A fixed CLA and CDA cannot.

## 8. Telemetry correlation

**Nowlan, CSim_model_refinement.pdf and CSim_process_made_simple.pdf.** [V] The monster file links
measured data to the model. The aero toolbox then back-calculates the aeromap sliver from the run.
The tyre force toolbox then fits the tyre model by optimisation against the same run. This is
parameter identification with the simulator in the loop.

**Canopy Simulations. "Getting started 3: matching telemetry."** [V]
https://simulation.michelin.com/canopy/technical-articles/getting-started-3-matching-telemetry
A staged order: match grip first on low-speed apex speeds, then drag on end-of-straight speed, then
downforce on the high-speed sector. Each stage sweeps one parameter and overlays the speed trace.

**Dal Bianco, N., Lot, R., Gadola, M. (2017). "Minimum time optimal control simulation of a GP2 race
car." Proc. IMechE Part D.** [V] https://eprints.soton.ac.uk/416216
A 14-degree-of-freedom multibody model with an indirect optimal control method, checked against a
2012 GP2 qualifying lap at Barcelona. Shows that a full transient optimal-control model can match real
telemetry channel by channel.

**Dal Bianco, N., Bertolazzi, E., Biral, F., Massaro, M. (2019). "Comparison of direct and indirect
methods for minimum lap time optimal control problems." Vehicle System Dynamics.** [V]
https://research.unipd.it/handle/11577/3271001

**MODSIM 2026 abstract 0091.** [V] https://modsimworld.org/papers/2026/MODSIM_2026_abstract_0091.pdf
Uses the speed delta between simulation and data to locate errors in the aero, powertrain and
suspension sub-models.

**Setup-conditioned telemetry modelling of a GT3 car (Vehicles, 2025, DOI
10.3390/vehicles8070146).** [V] A machine-learning study, but its validation hygiene is the lesson:
grouped cross-validation by telemetry file, and a shuffled-setup placebo test.

### The method

Lap time alone proves little, because errors cancel. Match channels in a fixed order, from the least
coupled to the most coupled. Low-speed apex speed isolates grip. End-of-straight speed isolates drag.
High-speed corner speed then isolates downforce. Then compare roll, pitch, damper position and tyre
temperature traces, which only a transient model produces. Judge each stage on its own channel before
you move to the next.

## 9. Why a simulator is accurate: synthesis

The sources agree on six requirements.

1. **Integrate in time.** Yaw inertia, roll and pitch dynamics, damper forces, tyre lag and bumps are
   all transient. A model that holds equilibrium at each point cannot show them, and they decide
   setup. Nowlan's measured roll exceeded the quasi-static value by 11 to 13 percent.
2. **Carry tyre load variation.** The tyre force curve is concave in load, so load variation costs
   grip. Dampers, springs and bumps act on the lap only through this path.
3. **Model the tyre as the data shows it.** Normalise by load in the Radt form, keep a lookup or a fit
   that behaves after the peak, and add temperature and relaxation states.
4. **Couple aero to ride height.** Downforce follows ride height, ride height follows load, and load
   follows downforce. A fixed coefficient breaks the loop.
5. **Identify parameters from telemetry.** Reverse-engineer the aeromap and the tyre model from
   measured runs with the simulator in the loop. Then correlate channel by channel in a staged order.
6. **Choose the solver for the question.** A closed-loop driver model predicts a real driver on a known
   line. Open-loop optimal control predicts the best possible lap. Direct collocation in the arc-length
   domain with curvilinear coordinates is the standard numerical route for the second.

## 10. Implications for FullModelSim_Python

This project already sits in the transient, free-trajectory, direct-collocation family of Perantoni
and Limebeer, Dal Bianco and Massaro. The literature suggests four checks, in order of value.

1. **Tyre load sensitivity and combined slip.** Check that the tyre model's force falls off with load
   and that combined slip shares one limit. Without this the optimiser over-rewards load transfer.
2. **Ride-height aero.** If CLA and CDA are constants, a ride-height map in the Nowlan form is a cheap
   upgrade. Use 11 by 9 points over front and rear ride height. The suspension states already exist.
3. **Damper and tyre lag states.** Check that the collocation grid resolves the damper and relaxation
   time constants. A coarse grid in arc length under-resolves them at low speed.
4. **Correlation protocol.** Adopt the staged channel order from Section 8 for any comparison with
   data. Document which parameters each stage fixed.

Skipped for now: a driver model and a thermal tyre state. Add a thermal state when the comparison with
data shows a grip drift that load and slip cannot explain.

## References by theme

Transient versus QSS: Siegler et al. 2000 [V]. Kelly 2008 [V]. Brayshaw and Harrison 2005 [K].
Casanova, Sharp and Symonds 2000 [K]. Lot and Dal Bianco 2015 [K]. Veneri and Massaro 2019 [V].
Massaro and Limebeer 2021 [V]. Nowlan 2011 [V].

Optimal control: Casanova 2000, PhD thesis, Cranfield [V],
https://dspace.lib.cranfield.ac.uk/handle/1826/1091. Perantoni and Limebeer 2014, Vehicle System
Dynamics 52, 653 to 678 [V]. Dal Bianco, Lot and Gadola 2017 [V]. Dal Bianco, Bertolazzi, Biral and
Massaro 2019 [V]. Limebeer and Rao 2015, "Faster, higher, and greener," IEEE Control Systems Magazine
35(2) [K].

Driver models: MacAdam 1981 [V]. Sharp, Casanova and Symonds 2000 [K]. Thommyppillai, Evangelou and
Sharp 2009 [V]. Anderson and Ayalew 2016 and 2017 [V].

Tyres: Pacejka 2012 [K]. Radt and Milliken 1960 [V]. Radt and Glemming 1993 [V]. Radt 1994 [V].
Radt 1995 [V]. Kasprzak, Lewis and Milliken 2006 [V]. Milliken and Milliken 1995 [K]. Tremlett and
Limebeer 2016 [V]. West and Limebeer 2022 [V]. Nowlan tyre documentation [V].

Dampers: Nowlan damper guides 1 and 2 [V]. Kowalczyk 2002 [V]. AVL and MathWorks 2021 [V].

Aero: Nowlan aero documentation and posts [V]. Carreira thesis [V].

Correlation: Nowlan model refinement and process documents [V]. Canopy [V]. Dal Bianco, Lot and Gadola
2017 [V]. MODSIM 2026 [V]. Vehicles 2025 GT3 study [V].

Nowlan and ChassisSim: all items in Section 2 [V].
