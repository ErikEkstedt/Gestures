from OpenGL import GLU # fix for opengl issues on desktop  / nvidia
import roboschool
import gym
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Plain Roboschool')
parser.add_argument('--env_id', type=str, default='RoboschoolReacher-v1',
                    help='Roboschool Environment')



args = parser.parse_args()

env = gym.make(args.env_id)
a = env.action_space.shape[0]

s = env.reset()
while True:
    env.render()
    s, r, d, _ = env.step(np.random.rand(a)*2-1)
    print(r)


