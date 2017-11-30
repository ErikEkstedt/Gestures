import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from itertools import count
import os
import gym
import numpy as np
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import roboschool

from memory import RolloutStorage, StackedState
from arguments import FakeArgs, get_args
from AgentRobo import AgentRoboSchool
from environment import Social_Torso

def make_social_torso(seed, rank):
    def _thunk():
        env = Social_Torso()
        env.seed(seed + rank)
        return env
    return _thunk

def random_action(idx, size):
    z = np.zeros(size)
    if type(idx) is int:
        z[idx] = np.random.rand()*2 - 1
    else:
        z[idx] = np.random.rand(len(idx))*2 - 1
    return z


def main():
    # env = Social_Torso()
    # s = env.reset()
    # t = env.get_target()

    seed = 10
    num_processes = 2
    env = SubprocVecEnv([
        make_social_torso(seed, i)
        for i in range(num_processes)])

    asize = env.action_space.shape[0]
    alls = list(np.arange(asize))

    s = env.reset()
    t1,t2 = env.get_target()

    target1 = t1['target']
    target2 = t2['target']

    random_r = 0
    for i in range(4000):
        a = random_action(alls, asize)
        s, r, d, _ = env.step(num_processes*[a])
        t1,t2 = env.get_target()

        if not (t1['target'] == target1).all():
            print('target 1 changed at ', i)
            target1 = t1['target']
        if not (t2['target'] == target2).all():
            print('target 2 changed at ', i)
            target2 = t2['target']

        random_r += r
        if sum(d) > 0:
            print(i)
            print(d)
            print(random_r)
            input()








if __name__ == '__main__':
    main()
