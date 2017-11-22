Roboschool
==========

## Custom Environment

Wants:
- [ ] Customizing rewards
- [ ] Customising initial pose
- [ ] fix joints, fix such that robot always standing.
- [ ] Multiple Processor training:
	* rgb
	* joint states 
- [x] Control the camera


### First Task

Follow the [lead](RoboEnvironment.py)!
* `class RoboschoolMujocoXmlEnv(gym.Env)`
* `class RoboschoolForwardWalker(SharedMemoryClientEnv)`
* `class RoboschoolForwardWalkerMujocoXML(RoboschoolForwardWalker, RoboschoolMujocoXmlEnv)`
* `class RoboschoolHumanoid(RoboschoolForwardWalkerMujocoXML)`

## PPO on random roboschool task

1. Make Roboschool Agent
2. Main training loop
3. Start training on Desktop

# Code

### [camera](camera.py)
Contains the camera class. Used to get rgb observation (optonal: depth, label).


