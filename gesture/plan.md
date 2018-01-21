# Plan

Environment with: 
* [ ] Clean Reacher with visible target dots. Small state space.
* [x] Clean SocialReacher state space only implementation.
* [ ] SocialHumanoidHlaf??

Reward
* [ ] Try several runs with reward function
* [ ] Show/record best test score and relate to training data.

Train
* [x] Clean training loop.
* [x] Change environmnet easy (now my environments should not change anymore)

Test
* [ ] Set up a test script that utilizes state_target approximation

Values
* [ ] The value/policy loss always depends on size of steps.
	* [ ] Normalize this data.
* [ ] A specific reward function will yield the same reward for the same DoF
	* Baseline (play with random agent average)
	* [ ] SocialReacher
	* [ ] SocialHumanoid
	* [ ] SocialHumanoidHalf?

# Todo

* Model
	* [ ] Results from trained model + cnn_translation
	* [ ] Using stateless targets. See results. training now
	* [x] Translation in combine training

* [ ] Clean social.py
	* [x] render functions
	* [x] target datatype - consistency. NUMPY IN ENV

* [x] Training loop for social.Social
	* [x] record directly from test


# Trainings

Reward
1. [ ] Absolute
2. [ ] Diff
3. [ ] Diff + cost
4. [ ] inverse square distance 
5. [ ] Bonus points for being on top 

SocialReacher
1. [ ] Train Understanding module.
2. [ ] MLP, trained with state target + Understand
3. [ ] SemiCombine: CNN+MLP, trained with state target + Understand
4. [ ] Combine: CNN+MLP, trained without state target. [intrinsic understanding]



