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
