from roboschool.scene_abstract import Scene, SingleRobotEmptyScene
import os
import numpy as np
import gym
from itertools import count

# from OpenGL import GL # fix for opengl issues on desktop  / nvidia
from OpenGL import GLE # fix for opengl issues on desktop  / nvidia

try:
    from environments.gym_env import MyGymEnv
except:
    from gym_env import MyGymEnv


def Target(r0, r1, x0=0, y0=0, z0=0.41):
    """TODO: Docstring for Target.

    :arg1: TODO
    :returns: TODO
    """

    # circle in xy-plane
    theta = 2 * np.pi * np.random.rand()
    x = x0 + r0*np.cos(theta)
    y = y0 + r0*np.sin(theta)
    z = z0

    # sphere, r=0.2, origo in x,y,z
    theta = np.pi * np.random.rand()
    phi = 2 * np.pi * np.random.rand()
    x1 = x + r*np.sin(theta)*np.cos(phi)
    y1 = y + r*np.sin(theta)*np.sin(phi)
    z1 = z + r*np.cos(theta)

    verbose = False
    for name, j in self.target_joints.items():
        if "0" in name:
            if "z" in name:
                j.reset_current_position(z, 0)
                if verbose:
                    print('z0')
                    print(name)
            elif "x" in name:
                if verbose:
                    print('x0')
                    print(name)
                j.reset_current_position(x,0)
            else:
                if verbose:
                    print('y0')
                    print(name)
                j.reset_current_position(y, 0)
        else:
            if "z" in name:
                if verbose:
                    print('z1')
                    print(name)
                j.reset_current_position(z1, 0)
            elif "x" in name:
                if verbose:
                    print('x1')
                    print(name)
                j.reset_current_position(x1,0)
            else:
                if verbose:
                    print('y1')
                    print(name)
                j.reset_current_position(y1, 0)

class Reacher(Base):
    '''
    2DoF Reacher in a plane
    No joint limits
    1 DoF each joint (only z-axis)
    target random in 3D space, not every point is
    '''
    def __init__(self, args=None):
        Base.__init__(self, XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot_arm',
                        target_name='target',
                        model_xml='reacher/Reacher_plane.xml',
                        ac=2, obs=13, args=args)
        print('I am', self.model_xml)
