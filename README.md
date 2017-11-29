Roboschool
==========

# Todo

- [x] **Get back relative coordinates**
- [ ] **Rewards**
	- Start: distance between left hand and a target.
		- train and see if it works.
- [ ] **Targets**

1. Fix test method so debugging can be made.
	 * A test method for visualization - one processor
	 * A test method for mult. procs for progress measuring.
	
2. Define "done" in the environment in a good way that gives monotonical increasing returns for better policies.


# Custom Environment
Wants:
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

Here I try to make an environment that fixates the hips and legs of a humonoid in order to train the upper body. This task does not want to solve balance or walking, only upper body gestures.

## [Environment](environment.py)
The script for my custom gym/roboschool environment.

### Social_Torso 
Base class which defines my custom environment.

Contains:
* init
* robot_specific_reset
* set_initial_orientation
* apply_action

Inherits from....

### GYM_XML_MEM
Class that inherits from both Shared_Mem and GYM_XML

Inherits from....

### GYM_XML

Wrapper for OpenAI's gym (inherits gym.Env).
Contains:
* init
* \_seed
* \_reset
* \_render
* calc_potential (return 0) ?
* HUD

### Shared_Mem
Inherits from roboschool.multiplayer.SharedMemoryClientEnv.
This is used in order to enable multiplayer setup. Several robots in one world.
Also controls the singleplayer setup.

Contains:
* init
* create_single_player_scene 
* robot_specific_reset
* move_robot - used in multiplayer to not have every robot on top of another.
* apply_action
* calc_state
* \_step
* episode_over


### [camera](camera.py)
Contains the camera class. Used to get rgb observation (optonal: depth, label).


