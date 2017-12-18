## Why does training improve but when loading state dict the result sucks?
Don't know why... might come back...
But for now simply just save models with good score and then restart scirpt until correct behaviour. [This pains me]

### Possible Sources:
* [x] Check that StackedState gives same output for all numbers of processes. YES! (memory.py - test())
*	[x] torch.load? - loads corrupt file?
	* torch.load and state dict works well.
	* same input gives same output everytime
* [ ] Environment
	* [x] moters, motor_names - different order? actions goes to wrong joints? NO
	* Behaviour:
		* same between resets.
		* different between init's
	
* Could it be because of the seed?
	* Changed the seed function in the environment such that one needs to explicitly call that function from `the outside`.

* It "feels" like either the torques are way over exaggerated or not strong enough. Why ??






