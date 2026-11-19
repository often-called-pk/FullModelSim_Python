# FullModelSim

FullModelSim finds the minimum lap time of a race car on a circuit. It poses the racing line as an optimal-control problem and solves it as one nonlinear program.

## Language

### Cars

**Car source**:
The origin of the car data that one solve uses. A solve has one car source, the 4EM car or a ChassisSim car.
_Avoid_: car model, vehicle file

**4EM car**:
The four-motor electric race car of the MATLAB original (FullModel_4EM). It is the default car source.
_Avoid_: default car, MATLAB car, our car

**ChassisSim car**:
One complete car that ChassisSim input files define: a car file plus its text inputs. FullModelSim takes all of its data from those files, never part of it from another car.
_Avoid_: ChassisSim model, sample car, demo car

### ChassisSim inputs

**Car file**:
The binary `.car` file in which ChassisSim stores one car. It holds the mass, geometry, inertia, tyre model, motion ratios, damper curves, differential map, engine, DRS zones and the axis ranges of the aeromaps.
_Avoid_: car model, setup file

**Text inputs**:
The plain-text files that ChassisSim reads beside a car file. They hold aeromaps, circuit curvature, altitude and road camber, grip factors, suspension hardpoints and tyre load curves. They also hold the lists of springs, bars and wing levels.
_Avoid_: ini files, data files

### Tyres

**MF tyre**:
The Pacejka Magic Formula 5.2 tyre model of the 4EM car, with its coefficient sets and scaling factors.
_Avoid_: Pacejka model, full tyre

**ChassisSim tyre**:
The tyre model of a ChassisSim car. Its maximum force is a load function times a temperature function. A slip table and camber and ellipse terms then shape the force.
_Avoid_: traction-circle model, lookup tyre

**Tyre temperature**:
The mean temperature of one tyre, which changes along the lap with heat input and cooling. Only a car with thermal data has it.
_Avoid_: carcass temperature, surface temperature, internal temperature

**Background temperature**:
The tyre temperature that the car file gives for a tyre that is up to temperature. A lap starts from it.
_Avoid_: ambient temperature, cold temperature

### Handling

**Axle balance**:
The rear axle's share of its grip limit minus the front axle's share. A positive value means the rear is nearer its limit, which is oversteer.
_Avoid_: STBI, stability index, handling balance

**Stability index**:
ChassisSim's own measure of under- and oversteer, in percent and negative for understeer. It comes from the cornering stiffness of each axle. ChassisSim can average it per corner and use the average to correct the corner speed. It is not the axle balance.
_Avoid_: STBI channel, balance

### Circuits

**Circuit source**:
The origin of the circuit data that one solve uses: a FullModelSim circuit or a ChassisSim circuit. Any car can run on a circuit from either source.
_Avoid_: track file, track source

**ChassisSim circuit**:
A circuit that ChassisSim text inputs define. It has the curvature of the driven line. It can also have an altitude and road-camber file, grip factors and a road profile. It has no track width.
_Avoid_: ChassisSim track, curvature file

**Grip factor**:
A multiplier on tyre grip that belongs to the circuit. It is one base value for the whole circuit, times an optional value that changes along the track.
_Avoid_: friction factor, track grip, mu scale

**Road profile**:
The road height under each tyre along the circuit. A solve uses only the waves that its mesh can resolve.
_Avoid_: bumps, bump file, road roughness

### Powertrain

**Gear envelope**:
The largest wheel force that the engine can give at each speed, over all gears. It replaces a gear choice, which the nonlinear program cannot make.
_Avoid_: CVT, ideal gearbox, gear map

**Differential type**:
The rule that splits the drive force between the two driven wheels: open, super, spool, custom map or Detroit. The car file gives it per axle. The faster wheel always loses force.
_Avoid_: diff mode, LSD setting

### Suspension

**Roll-centre height**:
The height of the roll centre above the contact patches of an axle. It changes with the wheel travel.
_Avoid_: stored roll-centre value, roll axis height

**Dynamic camber**:
The camber of a wheel against the road. It is the static camber, plus the camber change from the wheel travel, plus the body roll angle. A steered wheel also adds the camber change from the steer, which the castor and the kingpin inclination set.
_Avoid_: camber gain, kinematic camber

**Bump steer**:
The toe change of a wheel with its travel. It adds to the static toe of that wheel.
_Avoid_: roll steer, toe curve

**Third element**:
A spring, a bump rubber and a damper that act on the heave of an axle, the mean travel of its two wheels. It does not resist roll.
_Avoid_: heave spring, monoshock

**Rubber hysteresis**:
A bump rubber that pushes back less while it extends than while it compresses, at the same compression.
_Avoid_: rubber damping
