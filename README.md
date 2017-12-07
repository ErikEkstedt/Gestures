Project
==========
# [Agent](Agent/)

# [Baselines](Baselines/)

Testing out ppo from OpenAI's Baselines.

# [Environments](environments/)

Directory for the environments.
## [xml_files](environments/xml_files)

* custom_reacher
* half_humanoid
* humanoid
* fixed torso


### [camera](camera.py)
Contains the camera class. Used to get rgb observation (optonal: depth, label).


todo 
* [x] Train a roboschoolreacher with baselines and watch play.
* [x ] Write train/enjoy for custom env.
* [ ] train different versions of the half_humanoid.
	* [x] Reacher2d two arms
	* [x] Fixed so gravity is passed as argument
	* [ ] only move limbs in 2D plane
	* [x] two targets.
* [ ] Plot baseline results.
* [ ] Look into PPO2 for gpu
* [ ] Add video observation and think how that might help.
* [ ] Try with different baseline algorithms.
* [ ] fix Pytorch - full insight
