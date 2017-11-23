Roboschool
==========

## Custom Environment
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


## PPO on random roboschool task

1. Make Roboschool Agent
2. Main training loop
3. Start training on Desktop


