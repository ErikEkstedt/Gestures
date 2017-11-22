import gym, roboschool
from gym_mujoco_social import RoboschoolSocialHumanoid

env = RoboschoolSocialHumanoid()
env.reset()
steps = 1000

for step in range(steps):
    env.render()
    a = env.action_space.sample()
    s, r, d, i = env.step(a)
