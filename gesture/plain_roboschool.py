from OpenGL import GL
import gym, roboschool
import numpy as np
import time

'''
For me every state value becomes `nan` after reset() is called after render()
has been called. It does not matter if render is only done once, after reset
everything is `nan`. Can't get it to work. I get around this on my own custom
environments.
'''

env = gym.make('RoboschoolHumanoid-v1')
while True:
    print('\n------------')
    env.reset()
    env.render()  # Does not seem to matter where this is.
    while True:
        s, r, d, _ = env.step(env.action_space.sample())
        # obs = env.render("rgb_array")
        if d:
            break
