Roboschool
==========

## Custom Environment
Wants:
- [ ] Customizing rewards
	* How to get positions and information from joints?
	* coordinates
	* distances
- [ ] Customising initial pose
	- [ ] fix joints, fix such that robot always standing.
- [ ] Multiple Processor training:
	* rgb
	* joint states 
- [x] Control the camera

# Code
## Roboschool Inspired Environment
These scripts mimicks the structure of roboschool.

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


