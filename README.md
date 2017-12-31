# Todo

# Day 1 
1. [x] Create 2 target 3 DoF env.
	* get angles between elbow and upper arm.
	* collect rgb from training
3. [x] Start training with 2 targets.

# Day 2
5. [x] Torso
	* angles of shoulders?
	* can it reach? what points?
	* [x] train and record best
6. [x] Grab random target positions.
	* Now just a script that makes random moves and chooses a datapoint by some probability. Stops when correct amount of samples has been gathered.
	* show image, coords

# Day 3
StackedState and RolloutStorage

1. [ ] Make RGB training results. Same rewards but just pixel info.
	* [ ] Make 2dof plane viewd from above take actions from pixels alone
	* [ ] Combin state and pixels
	* [ ] Memory, include pixel observations.
		* [ ] StackedState
		* [ ] RolloutStorage

## ML:
* [ ] Annealing learning rate
* [ ] make CNN autoencoder module
* [ ] Train jointly with rgb info and state
	* [ ] Compare convergence rate 
	* [ ] Compare training time
* [ ] predictive dynamics model which tries to predict a robots movement and instantly choose that predictive state as target.


## Program:
* [x] Make target rgb-visible during rendering.
* [ ] Think about mimic in practice
	* [ ] Have rgb at certain intervals be the target.
	* [ ] Update after one target is reached.
* [ ] Continue training. Save checkpoint
* [ ] After som easy convergence put two robots in an environment and have one mimic the other.


## Environment:
* [ ] Create 2 target 3 DoF env.
* [x] Humanoid with appropriate limits.


## Dynamics Model
* Go through tha main again.
* envs - learn from mujoco gym?
* train on humanoid-reacher-custom-evns



----------------------------------------------------------------------------

## Why does training improve but when loading state dict the result sucks?
Don't know why... might come back...
But for now simply just save models with good score and then restart scirpt until correct behaviour. [This pains me]


### Possible Sources:
* [x] Check that StackedState gives same output for all numbers of processes. YES! (memory.py - test())
*	[x] torch.load? - loads corrupt file?
	* torch.load and state dict works well.
	* same input gives same output everytime
* [ ] Environment
	* [x] moters, motor_names - different order? actions goes to wrong joints? NO
	* Beahviour:
		* same between resets.
		* different between init's

------------------------------------------------------------------------------

Project
==========

		Roboschool is broken on my setup.
		After calling env.reset(), if there has been some type of rendering, the robot and target disappears and all numbers are *nan*.

		However, my own CustomReacher environment is now working correctly and using a single process convergence is easily reached.
		There still seems to be something wrong with the multiple process setup.

# [Agent](Agent/)
Pytorch training.

* [main.py](Agent/main.py)
* [model.py](Agent/model.py)
* [memory.py](Agent/memory.py)
* [test.py](Agent/test.py)
* [train.py](Agent/train.py)
* Help Scripts:
	* [vislogger.py](Agent/vislogger.py)
	* [utils.py](Agent/utils.py)
	* [arguments.py](Agent/arguments.py)
	* [loadtest.py](Agent/loadtest.py)

# [Environments](environments/)

##  [gym_env.py](environments/custom_envs.py)

Extends OpenAI's gym.Env class.
* _seed
* _render
* _reset
* _step
* HUD

## environment code

All robot functions in `envs` containing functions:
* robot_specific_reset()
	* robot_reset()
	* target_reset()
* calc_state()
	* calc_to_target_vec()
* calc_reward(action)
	* calc_potential()
* get_rgb()

##   [reacher_envs.py ](environments/reacher_envs.py) (DoF = Degrees of Freedom)
All custom environments.

* **Reacher_plane**
* **Reacher2Dof**
* **Reacher3Dof**
* **Reacher6Dof**

##   [humanoid_envs.py ](environments/humanoid_envs.py)
All custom environments.

* **Humanoid**
* **Humanoid_right3DoF**
* **Humanoid6DoF**



## [xml_files](environments/xml_files)
Directory for the xml files.

### reacher
*  Reacher2DoF.xml
*  Reacher3DoF.xml
*  Reacher3DoF_2Targets.xml
*  Reacher6DoF.xml
*  Reacher_plane
*  ReacherHumanoid.xml

### humanoid

* humanoid.xml
	* Roboschool's symmetric humanoid
* humanoid6DoF.xml
* humanoid_right3DoF.xml
* upper_torso.xml


# [Baselines](Baselines/)
OpenAI baselines training
Mostly used for debugging training, is env wrong or algorithm...


