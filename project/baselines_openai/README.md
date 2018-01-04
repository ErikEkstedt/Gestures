# Baselines

## [main.py](main.py)

Reads input args and executes training.
calls on pposgd_simple for all the work.

## [main.py](pposgd_simple.py)

* learn(env, policy_func, **kwargs)
	* The complete learning algorithm with exploration and
	* learning/optimization
* add_vtarg_and_adv(seg, gamma, lam):
	* calculates returns and advantages.
* traj_segment_generator(pi, env, horizon, stochastic)
	* exploration

TODO:
Connect to vislogger - or tensorboard to visualize results.

# RLAlgorithm
## Model

**Policy core**:
* Merge: Value estimation/Policy optimization
* One common base for actor/critic
* Separate Nets
 
	`MLP`

**Policy type**:
* Categorical discrete output, `n` separate classes.
* Continous Diagonal Gaussian, `n` separate distributions.
		
	`Continous Diagonal Gaussian`
		
## Optimization
How does the algorithm learn.

In what kind of way is the logprob estimation done.
How to set the direction of update. 

> `Strange loopiness`
>
> AI research and `Programming RL` is where an agent tries to optimize how to learn how agents inside a simulation learns.


**Policy-algorithm**: Poly gradient estimation
* automatic backprop:
	* Loss: negative logprob of pi(a|s) * Advantage
	* Advantage: 
		* gae ? 
		* Mean value
* How large should the update be?
	* Trust Region: move only inside a `trusted` region. ratio between old policy and new.
	* PPO: Pessimisstic lower bound. cap the trusted region. 








