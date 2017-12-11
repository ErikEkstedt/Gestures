Not so important fixes
* [ ] Make Result class do plotting for less clutter.
	* Combine vislogger and results. LOGGER


More important fixes
* [ ] Continue training. Save checkpoint
	* [checkpoint](https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3)

TODO
1. Must have rigid baseline
	* [ ] Clean code.
	* [x] Make multiprocess work
		* [x] Result class
		* [x] Stacked_State
		* [x] Rollouts
		* [x] Subproc vec
		* [ ] MP like a3c  - [x] not worth diving in here
	* [x] Saved state_dict should always show the expected result.
		* Test score in name

2. Make environment more complex
	* [ ] Make CustomReacher more complex
		* [ ] Reach several targets
	* [ ] HalfHumanoid

3. Train with RGB
	* [ ] Back to social movements.


Project
==========

Roboschool is broken on my setup.
After calling env.reset(), if there has been some type of rendering, the robot and target disappears and all numbers are *nan*.

However, my own CustomReacher environment is now working correctly and using a single process convergence is easily reached.
There still seems to be something wrong with the multiple process setup.


# [Agent](Agent/)
Pytorch training

## Results
A training on my CustomReacher environment.

![Training rewards](Agent/Result/Dec9/training_score.png)
![Value loss](Agent/Result/Dec9/value_loss.png)
![Action std](Agent/Result/Dec9/action_std.png)


# [Baselines](Baselines/)
OpenAI baselines training

# [Environments](environments/)

##  custom.py

All custom environments. Wrapper for gym_env code.

##  gym_env.py

Base script that communicates with cpp_houshold, scene and so forth.

## [xml_files](environments/xml_files)
Directory for the xml files.

* custom_reacher
* half_humanoid
* humanoid
* fixed torso
