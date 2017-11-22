import os
import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import roboschool

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env,
                            os.path.join(log_dir,
                                         "{}.monitor.json".format(rank)))
        return env
    return _thunk

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)

log_dir = "/tmp/"

env_id = "RoboschoolHumanoid-v1"
seed = 10
num_processes = 4
envs = SubprocVecEnv([
    make_env(env_id, seed, i, log_dir)
    for i in range(num_processes)
])

steps = 100
s = envs.reset()
for i in range(steps):
    a = envs.action_space.sample()
    s, r, d, i = envs.step([a]*num_processes)
    print(r)
    print(s[0].shape)


