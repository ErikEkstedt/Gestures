Roboschool
==========

# Custom Environment

Wants:
- [ ] **Customising initial pose**
	- [ ] fix joints, fix such that robot always standing.
- [ ] **Customizing rewards**
	* [x] How to get positions and information from joints?
	* [x] coordinates
	* [x] distances

- [ ] **Multiple Processor training:**
	* rgb
	* joint states 
	* Simple to get pixels from single proc
		* obs = env.render('rgb_array')
		* cam = env.unwrapped.scene.cpp_world.new_camera_free_float(self.height, self.width, name)

- [x] **Control the camera**

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





# Code 
## Roboschool Inspired Environment
These scripts mimicks the structure of roboschool.

Here I try to make an environment that fixates the hips and legs of a humonoid in order to train the upper body. This task does not want to solve balance or walking, just gestures.

### [gym_social](gym_social.py)
Base class which defines my custom environment.

Contains:
* step
* reset
* render

### [gym_mujoco_social](gym_mujoco_social.py)
Extensions of gym_social and is used in roboschool for selecting different robots.
For now this only contains the Humanoid robot.

### [gym_mujoco_xml_env](gym_mujoco_xml_env.py)
The same class as found in `roboschool` except that this uses an absolute path to the xml files
describing the different mujoco robots.

## Other
### [camera](camera.py)
Contains the camera class. Used to get rgb observation (optonal: depth, label).


