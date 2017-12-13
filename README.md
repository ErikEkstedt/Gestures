More important fixes
* [ ] Continue training. Save checkpoint
	* [checkpoint](https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3)

* [ ] Annealing learning rate


## Why does training improve but when loading state dict the result sucks?


Don't know why... might come back.
For now if the tested saved model was good just try til you get rewards like it suppose to have.


Possible Sources:
* [x] Check that StackedState gives same output for all numbers of processes. YES! (memory.py - test())
*	[x] torch.load? - loads corrupt file?
	* torch.load and state dict works well.
	* same input gives same output everytime
* [ ] Environment
	* [x] moters, motor_names - different order? actions goes to wrong joints? NO
	* Behaviour the same between resets.
	* Behaviour different between creation.


* good score with Dict_Dof3_bklabla on DoF2 env..jk.
Plan
* Train and get stable models reliably
	* [ ] Dof2
	* [ ] Dof3

* Get rgb arrays from training
	* [ ] Dof2
	* [ ] Dof3


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

##  [ custom_envs.py ](environments/custom_envs.py)
All custom environments.

* **Reacher_plane**
* **CustomReacher2Dof**
* **CustomReacher3Dof**
* **CustomReacher6Dof**

All containing functions:
* robot_specific_reset()
	* robot_reset()
	* target_reset()
* calc_state()
	* calc_to_target_vec()
* calc_reward(action)
	* calc_potential()
* get_rgb()

##  [gym_env.py](environments/custom_envs.py)

Extends OpenAI's gym.Env class.
* _seed
* _render
* _reset
* _step
* HUD

## [xml_files](environments/xml_files)
Directory for the xml files.

* custom_reacher
* half_humanoid
* humanoid
* fixed torso

# [Baselines](Baselines/)
OpenAI baselines training
Mostly used for debugging training, is env wrong or algorithm...


