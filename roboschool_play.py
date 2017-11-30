import roboschool
import gym

import numpy as np

env = gym.make('RoboschoolHumanoid-v1')

s = env.reset()

for i in range(100):
    env.render()
    a = np.random.rand(env.action_space.shape[0])
    s, r, d, _ = env.step(a)

print('reset')
s = env.reset()

for i in range(100):
    env.render()
    a = np.random.rand(env.action_space.shape[0])
    s, r, d, _ = env.step(a)

print('done')
