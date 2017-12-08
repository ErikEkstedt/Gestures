# from OpenGL import *
import OpenGL
import os.path, gym
import numpy as np
import roboschool
from OpenGL import GLE


def demo_run():
    env = gym.make("RoboschoolAnt-v1")

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        #create another camera, the behavior is different from the built-in camera of the robot
        #I dont understand why because both of them seem to be created from new_camera_free_float
        mycam = env.unwrapped.scene.cpp_world.new_camera_free_float(320, 200, "my_camera")
        mycam.move_and_look_at(1,1,1, 0,0,0)

        while 1:
            a = np.zeros(env.action_space.shape)
            obs, r, done, _ = env.step(a)

            # world test_window, cannot work with camera, but can go with mycam if mycam.render is not called, while it outputs garbage pixels as the render is not called
            still_open = env.render("human")

            # rgb_array camera, cannot work
            img = env.render('rgb_array')
            env.unwrapped.camera.test_window()

            rgb, _, _, _, _ = mycam.render(False, False, False)
            mycam.test_window()


if __name__=="__main__":
    demo_run()
