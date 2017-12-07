# Agent

Pytorch implementation


TODO:

* [x] Check rollouts/currentstate
	* [x] single proc
	* [x] mult proc
* [x] Make subproc env
* [x] Obs_stats work correctly
	* Obs normalizes away information about target! 
	* If target data is explicetly in the state this destroys a lot of information.

* [x] Calculate and print running reward mean.
* [ ] Fix plot 
* [ ] fix annealing STD on actions (baselines and paper)
* [x] Train on RoboschoolReacher-v1
* [x] Create Load_model and run test.

Questions:
* PPO update old_policy every iter of ppo_epoch??

