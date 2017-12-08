# from OpenGL import GLX, GLE, GL  # fix for opengl issues on desktop  / nvidia
from OpenGL import GLE
import gym, roboschool
import numpy as np
import time

env = gym.make('RoboschoolHumanoid-v1')
a = env.action_space.shape[0]

print('\n------------')
env.reset()
while True:
    obs = env.render("human")
    s, r, d, _ = env.step(np.random.rand(a)*2-1)
    if d:
        break

env.close()
del env
