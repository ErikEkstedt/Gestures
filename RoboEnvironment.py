import numpy as np

def test():
    import gym
    import roboschool
    import cv2
    # env.set_initial_orientation(task=1, yaw_center=0, yaw_random_spread=1)

    steps = 100
    turn = 3.14/500

    env = gym.make('RoboschoolHumanoid-v1')
    s = env.reset()


    for i in range(steps):
        env.render()
        a = env.action_space.sample()
        s, r, d, i = env.step(a)

if __name__=="__main__":
    test()
