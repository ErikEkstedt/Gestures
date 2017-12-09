# from OpenGL import GLX, GLE, GL  # fix for opengl issues on desktop  / nvidia
from OpenGL import GL
import gym, roboschool
import numpy as np
import time

env = gym.make('RoboschoolHumanoid-v1')

while True:
    print('\n------------')
    env.reset()
    while True:
        s, r, d, _ = env.step(env.action_space.sample())
        # obs = env.render("rgb_array")
        obs = env.render("human")
        if d:
            break

env.close()
del env
