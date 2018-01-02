# Todo
1. [x] Memory, include pixel observations.
	* [x] StackedState
	* [x] RolloutStorage

2. [ ] Make RGB training results. Same rewards but just pixel info.
	* [ ] CNN model
	* [ ] Make 2dof plane viewd from above take actions from pixels alone
	* [ ] Combine state and pixels

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

--------------------------------------------
-------------------------------------------
# plan
The experiments that I run I make a script that does just that experiment.
Easy to rerun experiments and all hyperparameters will be the same easily.

## Environment
1. [x] 2 Dof
	* [x] Plane
	* [x] 3D 
2. [x] 3 Dof
	* [x] Plane
	* [x] 3D 
	* [ ] two networks for each arm.
3. [ ] 2-3-DoF = 6 DoF
	* [ ] Plane
	* [ ] 3D 
4. [x] 6 Dof Humanoid and rgb
	* [ ] 3D 

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

3. 6Dof
	* same target:
		* plane
		* one octagon
		* quart
		* all reachable 3D
	* two targets:
		* plane
		* one octagon
		* quart
		* all reachable 3D
	* reward
		* abs. dist
		* diff. dist 
		* regular costs
