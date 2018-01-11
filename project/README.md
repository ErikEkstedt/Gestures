# Todo

# Coding 



* [ ] make environment stpe method to take target as input.
	* env.step(a, target)
* [x] 'test_combine.py',  see performance of loaded state_dict.
* [ ] Gather trajectory.
	* [ ] Collect complete episode trajectory
* [ ] Script
	* test script but deterministic targets
	* loads a state dict 
	* loads a target trajectory
	* renders both




# improvements

* [ ] ReacherPlaneCombi environments take args and target as input. Uses targets to set video dims.



## Data Science: 
* Data collection/result sampling

1. Training: 
	* [x] Make RGB training results. Same rewards but just pixel info.
	* [x] CNN model
	* [x] Make 2dof plane viewd from above take actions from pixels alone
	* [x] Combine state and pixels
	* [ ] Train models
		* [x] Vanilla CNN
		* [x] Vanilla MLP
		* [x] Combine
		* [ ] CLSTM (Don't start before Vanilla done)
	* [ ] Continue training from checkpoint with smaller learning rate

2. [x] UNDERSTANDING MODULE
	* [x] Create Data - Supervised Learning
		* [x] Dataset
		* [x] only positions, no speed
			* Only joint_speed codes dynamic behavior and they are last.
			* ReacherPlane: s[:-2]
			* Reacher3D:		s[:-3]
		* [x] Velocities, if velocities are to be model by a cnn we need stacked frames.
			* [ ] Stack frames
			* [ ]	smaller network 

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

-------------------------------------------
# plan
The experiments that I run I make a script that does just that experiment.
Easy to rerun experiments and all hyperparameters will be the same easily.
## Reward
1. [x] Only absolute distance
2. [x] Only difference distance 
3. [x] both above with:
	* [x] Electricity cost, EC
	* [x] Torque cost, TC
	* [x] stuck cost, SC
4. [x] Hierarchical scaling
5. [ ] RGB Reward function.
	* [ ] MSE
	* [ ] MSE + diff_penalty = movement regularizer

These are hyperparameters defined in Agent.arguments. 
## architecture
1. [ ] FC
	* Layers/Hidden
2. [ ] LSTM

## State
Stacking of frames
* [ ] no-stack (=1)
* [ ] 2-stack
* [ ] 4-stack

## Observation
1. Define simple CNN
	* rgb -> state
	* rgb -> action
	* clstm -> action
2. Combination of understanding and coordination
	
## Experiment to run
1. [x] 2DoF
	* target: plane
	* reward
		* abs. dist
		* diff. dist 
		* regular costs

2. [x] 3DoF
	* reward
		* diff. dist 
		* regular costs
