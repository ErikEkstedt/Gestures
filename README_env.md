# Environment: Social Torso


## Action Space

Thirteen controllable joints

```python
["abdomen_z",
"abdomen_y", 
"abdomen_x", 
"upper_abs",
"neck_y",
"neck_z",
"neck_x",
"right_shoulder1",
"right_shoulder2",
"right_elbow",
"left_shoulder1",
"left_shoulder2",
"left_elbow"]
```

## Defining Mujoco xml
### default
*joint*
* armature - rotor inertia,
* damping - Damping applied to all degrees of freedom created by this joint.
* limited - If this attribute is "true", the length limits defined by the range attribute below are imposed by the constraint solver.
	* (not in default) range : float(2), "0 0",  Range of allowed tendon lengths. To enable length limits, set the limited attribute to "true" in addition to defining the present value.

*geom*
* Collisions:
	* contype - int, type of contact of one body in collision 
	* conaffinity - int, type of contact of the other body in collision.  
	* condim - int, dimension of contact "dynamics(?)"	
* friction - (float, float, float) = (slide , torsional , rolling) friction.
* margin - The distance threshold below which limits become active.
* rgba - color

*motor*
* ctrllimited - boolean, if true clamps the actuators at runtime
* ctrlrange - the range of the clamping

*option*
* integrator - This attribute selects the numerical integrator to be used. Currently the available integrators are the semi-implicit Euler method and the fixed-step 4-th order Runge Kutta method.
* iterations - int, maximum number of iterations of the integrator.
* sover - solver algorithm, PGS or Newton
* timestep - Simulation time step in seconds. This is the single most important parameter affecting the speed-accuracy trade-off which is inherent in every physics simulation. Smaller values result in better accuracy and stability. To achieve real-time performance, the time step must be larger than the CPU time per step (or 4 times larger when using the RK4 integrator).


