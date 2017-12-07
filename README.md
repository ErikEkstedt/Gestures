Project
==========

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


# [Agent](Agent/)
Pytorch training

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



# Old
all old code implementations not used...
