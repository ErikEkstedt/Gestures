# Defining Mujoco xml
[Mujoco modeling](http://www.mujoco.org/book/modeling.html)

## joint
* armature - rotor inertia,
* damping - Damping applied to all degrees of freedom created by this joint.
* limited - If this attribute is "true", the length limits defined by the range attribute below are imposed by the constraint solver.
	* (not in default) range : float(2), "0 0",  Range of allowed tendon lengths. To enable length limits, set the limited attribute to "true" in addition to defining the present value.
* stiffness - Joint stiffness. If this value is positive, a spring will be created with equilibrium position given by springref below. The spring force is computed along with the other passive forces.

* pos - Position of the joint, specified in local or global coordinates as determined by the coordinate attribute of compiler. For free joints this attribute is ignored.


## geom
* Collisions:
	* contype - int, type of contact of one body in collision 
	* conaffinity - int, type of contact of jhe other body in collision.  
	* condim - int, dimension of contact "dynamics(?)"	
* friction - (float, float, float) = (slide , torsional , rolling) friction.
* margin - The distance threshold below which limits become active.
* rgba - color

## motor
* ctrllimited - boolean, if true clamps the actuators at runtime
* ctrlrange - the range of the clamping
* gear - This attribute scales the length (and consequently moment arms, velocity and force) of the actuator, for all transmission types. It is different from the gain in the force generation mechanism, because the gain only scales the force output and does not affect the length, moment arms and velocity.

## option
* integrator - This attribute selects the numerical integrator to be used. Currently the available integrators are the semi-implicit Euler method and the fixed-step 4-th order Runge Kutta method.
* iterations - int, maximum number of iterations of the integrator.
* sover - solver algorithm, PGS or Newton
* timestep - Simulation time step in seconds. This is the single most important parameter affecting the speed-accuracy trade-off which is inherent in every physics simulation. Smaller values result in better accuracy and stability. To achieve real-time performance, the time step must be larger than the CPU time per step (or 4 times larger when using the RK4 integrator).


## Worldbody

Defining the worldbody. The regular 'roboschool' humanoids does not have any 'geom's in the world body but starts a 'body' inside it.
The reacher environment which is stuck contains two 'geom's which are a plane and a cylinder (where the arm is connected), and also four "walls". 

The Roboschool humanoids are fre to move around in the world but the reacher is stuck.

In this "social" implementation the pelvis is fixed in the worldbody and then connects to other 'bodies'.

# [Social Torso](Social_torso.xml)

The soical torso implemented in this project is a simple upper torso model.

The torso's lowest part, the pelvis, is static in the world. This part is then connected through three joints to the lower waist.

Joints:
* Abdomen
	* x, y, z
* Upper abs
* Neck
	* x, y, z
* Arms
	* Right/left
		* shoulder 1
		* shoulder 2
		* elbow 

```xml
	<actuator>
		<motor gear="100" joint="abdomen_y" name="abdomen_y"/>
		<motor gear="100" joint="abdomen_z" name="abdomen_z"/>
		<motor gear="100" joint="abdomen_x" name="abdomen_x"/>
		<motor gear="100" joint="upper_abs" name="upper_abs"/>
		<motor gear="25" joint="neck_y" name="neck_y"/>
		<motor gear="25" joint="neck_z" name="neck_z"/>
		<motor gear="25" joint="neck_x" name="neck_x"/>
		<motor gear="25" joint="right_shoulder1" name="right_shoulder1"/>
		<motor gear="25" joint="right_shoulder2" name="right_shoulder2"/>
		<motor gear="25" joint="right_elbow" name="right_elbow"/>
		<motor gear="25" joint="left_shoulder1" name="left_shoulder1"/>
		<motor gear="25" joint="left_shoulder2" name="left_shoulder2"/>
		<motor gear="25" joint="left_elbow" name="left_elbow"/>
	</actuator>
```


