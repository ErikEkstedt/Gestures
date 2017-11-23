import os
import gym
import torch

from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import roboschool

from memory import RolloutStorage
from memory import StackedState

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, "{}.monitor.json".format(rank)))
        return env
    return _thunk

def print_info(x):
    print(type(x))
    print(len(x))
    print(x.shape)
    print(x.dtype)
    input()


# env =============
monitor_log_dir = "/tmp/"
env_id = "RoboschoolHumanoid-v1"
seed = 10
num_processes = 2
num_stack = 4
num_steps = 10

envs = SubprocVecEnv([make_env(env_id, seed, i, monitor_log_dir) for i in range(num_processes)])
state_shape = envs.observation_space.shape
stacked_state_shape = (state_shape[0] * num_stack,)
action_shape = envs.action_space.shape

# memory
memory = RolloutStorage(num_steps, num_processes, stacked_state_shape, action_shape)
CurrentState = StackedState(num_processes, num_stack, state_shape)

# init
s = envs.reset()
CurrentState.update(s)
memory.states[0] = CurrentState()

steps = 10
from itertools import count
for step in count(1):
    a = envs.action_space.sample()
    a = [a]*num_processes

    s, r, done, i = envs.step(a)

    if done[0] or done[1]:
        print('done')
        print(done)
        input()

    # CurrentState.update(s)

    # v = torch.rand(4)
    # mask = torch.Tensor([int(d) for d in done])
    # a = torch.Tensor(a)
    # r = torch.Tensor(r)
    # memory.insert(step, CurrentState(), a, v, r, mask)

v = torch.rand(4)
memory.compute_returns(v, use_gae=True, gamma=0.99, tau=0.95)

print(memory.rewards.size())
print(memory.actions.size())
print(memory.states.size())
print(memory.masks.size())
print(memory.returns.size())
print(memory.returns[:,0,:])
