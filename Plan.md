# plan
## Environment
1. [ ] 2 Dof
	* [ ] Plane
	* [ ] 3D 
2. [ ] 3 Dof
	* [ ] Plane
	* [ ] 3D 
	* [ ] two networks for each arm.
3. [ ] 2-3-DoF = 6 DoF
	* [ ] Plane
	* [ ] 3D 
4. [ ] 2 Dof
	* [ ] Plane
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

