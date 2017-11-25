General Agent for Roboschool-"esque" Environments
=================================================


General Memory
--------------

Keep memory in a circular "rollerdeck" fashion. Push/Pop dynamics.

Functions:
* Off-Policy, DQN Memory Style:
  * Give back data in batch of M size: `experience = (s, a, r, s_next)`.
* On-Policy,  Policy Optimization Style:
  * Calculate Returns for latest N data points.
* Store Data for convenincy and effiency    

* Store data to disk and retrieve by using pytorch dataset/dataloader. (off-policy)


### Starting point

- [x] Data recieved from roboschool (humanoid):
	* state:	np.array, dtype: float32, shape: (num_proc, 44)
	* reward: np.array, dtype: float64, shape: (num_proc,)
	* done:		np.array, dtype: bool, shape: (num_proc,)

Agent
------

Seems to work now! 
1. Test tomorrow and get some results.
2. Clean code.
3. Back to environment

Value estimation based on rgb-obswervation as well?

### PPO

Old policy/two policies needed ?

Instead of having an old policy evaluating log probs for everything in training we calculate log probs
directly when sampling in exploration and saves those values.

then during training epochs we have those static values and jsut uses our current policy to evaluate and update its values. Effectively the same thing as having two networks and doing two batch_size passes through.




