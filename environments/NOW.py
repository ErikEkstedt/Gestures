from humanoid_envs import  Humanoid6DoF_2target
from reacher_envs import  Reacher3DoF_2Target
import cv2
import numpy as np
import torch


def joint_info(j):
    print('-'*52,
          '\nName: {}, (type: {}, power: {})'.format(j.name, j.type, j.power_coef),
          '\n\nPosition: \t{}, \nRelative Position: {}'.format(j.current_position(), j.current_relative_position()),
          '\n\nPossible actions:\n\tset_motor_torque()\n\tset_servo_target()\n\tset_target_speed()\n')
    print('-'*52)

def parts_info(Part):
    p = Part.pose()
    print('-'*52,
          '\nName:', Part.name,
          '\nSpeed:',  Part.speed(),
          '\nContact list:', Part.contact_list(),
          '\n\nPose:',
          '\nxyz: {}, \nrpy: {}\nquatertion: {}'.format(p.xyz(), p.rpy(), p.quatertion()),
          '\n\nPossible actions:\n\tset_xyz()\n\tset_rpy()\n\tset_quatertion()',
          '\n\tmove_xyz()\n\tdot()\n\trotate_z()')
    print('-'*52)


# humanoid/humanoid6DoF.xml
# env = Humanoid6DoF_2target()
env = Reacher3DoF_2Target()

s = env.reset()
jdict = env.jdict
parts = env.parts

# el_r = jdict['robot_right_elbow']
# el_l = jdict['robot_left_elbow']
# hand_r = parts['robot_right_hand']
# hand_l = parts['robot_left_hand']
#
# joint_info(el_r)
# joint_info(el_l)
# parts_info(hand_r)
# parts_info(hand_l)
#

# env.custom_camera_adjust()
x, z = np.random.rand()*1, np.random.rand()*0.2+0.4
while True:
    env.reset()
    for i in range(100):
        s, r, d, _ = env.step(env.action_space.sample())
        rgb = env.render('rgb_array')

        cv2.imshow('frame', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # joint_info(el_r)
    # joint_info(el_l)
    # parts_info(hand_r)
    # parts_info(hand_l)
