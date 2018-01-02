Project
==========

This project aims to engineer a simulation environment where an agent might learn to
move and understand how that movement is preceived in the world.  The idea is
to have an agent train to move limbs to random targets in joint state space and
at the same time learn a model which translates rgb-values onto the state
space.  After convergence the agent will be able to then see an image of a pose
and then execute that same pose, possibly also a trajectory of images to mimic.

(having to robot opposite of eachother mimicing eachother... could be fun)

For this setup a custom version of OpenAI's Roboschool is the environment used.
The custom environments consists of a robot (reacher/humanoid) which has a
"torso" connected to the environment and some limbs with different degrees of
freedom, DoF. A limb then has a number of "important points" which purpose is
to be as close to some target points as possible, these points are the
definition of a pose.


In reinforcement learning the reward function is what is going to define the
behaviour of the trained agents. One part of this project is to find some
suitable reward function which makes convergence as fast and the resulting
behaviours as good as possible. The reward function can be dependent on the
state space as well as the rgb space, as well as a combination [R(s), R(obs), R(s, obs)]. 

The starting point is the p2-norm of the difference vector between the robot
points, the `potential` and the target points in state space and MSE as the
analogue in pixel space. Then different reward function will be tested in order
to find one that produces a desirable behaviour. In Roboschool/Mujoco the
reward function for both reacher and the running tasks are dependent on the
difference in potential between concurrent timesteps and some
regularizing/penalty term for electricity used/torque-force. 

The agent will train by using PPO (Proximal Policy Optimization) implemented in
pytorch.  The policy will be an MLP, operating in joint state space, a CNN
pixels -> actions and a combination of them both.  An convolutional/CLSTM -
"translation/understanding" module will be used to translate between pixel
space and joint state space.


This project aims to engineer a suitable simulation environment, a reward
function, and a network architecture that will make learning to move and mimic
movements, in the way explained above, possible.

# [project.agent](project.agent/)
Pytorch Agent

* [main.py](project.agent/main.py)
* [model.py](project.agent/model.py)
* [memory.py](project.agent/memory.py)
* [test.py](project.agent/test.py)
* [train.py](project.agent/train.py)
* Help Scripts:
	* [vislogger.py](project.agent/vislogger.py)
	* [utils.py](project.agent/utils.py)
	* [arguments.py](project.agent/arguments.py)
	* [loadtest.py](project.agent/loadtest.py)


-----------------------------------------------
# [project.environments](project.environments/)
The simulation environments are a big part of this project. 
##   [reacher.py ](project.environments/reacher.py) (DoF = Degrees of Freedom)
All custom project.environments.
* **Reacher_Plane**
* **Reacher3D**

##   [humanoid.py ](project.environments/humanoid.py)
All custom project.environments.
* **Humanoid**
* TargetHumanoid

##  [my_gym_env.py](project.environments/my_gym_env.py)
Extends OpenAI's gym.Env class.
* _seed
* _render
* _reset
* _step
* HUD

## [xml_files](project.environments/xml_files)
Directory for the xml files.
* xml_files/
	* bullet_mjcf/
	* gym_assets/
	* humanoid/
	* reacher/

# [baselines](baselines/)
OpenAI baselines training
Mostly used for debugging training, is env wrong or algorithm...
