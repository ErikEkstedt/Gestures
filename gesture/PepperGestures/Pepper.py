from __future__ import print_function
import gym
from gym import spaces
import numpy as np
import time
import cv2

import qi
import motion

from screen import ObsRGB

#============== Help Functions ======================

def getLimits(motion_service, name="Body"):
    limits = motion_service.getLimits(name)
    jointNames = motion_service.getBodyNames(name)
    for i in range(0,len(limits)):
        print(jointNames[i] + ":")
        print("minAngle", limits[i][0],\
            "maxAngle", limits[i][1],\
            "maxVelocity", limits[i][2],\
            "maxTorque", limits[i][3])

class Pepper_v0_with_goal(gym.Env):
    '''
    metadata = { 'render.modes': ['human', 'rgb_array'],
                 'video.frames_per_second' : 50 }

    # Directions:
    #           Environment - Choregraphe
    #           Open Choregraphe and connect to a virtual session.
    #           Copy the PORT the simulation runs on.
    #           Detach the "Robot view" window and have it
    #           visable on desktop (we get pixel values from the screen).
    #           Create a qi session with the PORT number.

    '''
    def __init__(self, session, rgb_shape=(64,64), head=False, goal=None, precision=1e-2):
        '''
        Arguments:
            session - a qi.session
            head - boolean, if true actions include head actions.
            precision - precision used for end run condition
        '''
        self.session = session
        self.motion_service = session.service("ALMotion")
        self.motion_service.setStiffnesses = 1  # movement possible
        self.posture_service = session.service("ALRobotPosture")
        self.posture_service.goToPosture("StandInit", 0.5)

        o_low, o_high, a, b = self.get_limits(head)  # different sizes w/ or w/out head
        self.state_space = spaces.Box(o_low, o_high)
        self.action_space = spaces.Box(0.3*o_low, 0.2*o_high)

        self.RGB_observer = ObsRGB()
        self.rgb_shape = rgb_shape
        self.observation_shape = (3, rgb_shape[0], rgb_shape[1])

        self.useSensors = False
        self.step_time = 0.1
        self.head = head
        if head:
            self.names = ["LShoulderRoll", "LShoulderPitch",
                    "LElbowYaw", "LElbowRoll",
                    "LWristYaw", "LHand",
                    "RShoulderRoll", "RShoulderPitch",
                    "RElbowYaw", "RElbowRoll",
                    "RWristYaw", "RHand",
                    "HeadYaw", "HeadPitch"]
        else:
            self.names= ["LShoulderRoll", "LShoulderPitch",
                    "LElbowYaw", "LElbowRoll",
                    "LWristYaw", "LHand",
                    "RShoulderRoll", "RShoulderPitch",
                    "RElbowYaw", "RElbowRoll",
                    "RWristYaw", "RHand"]
        self.fractionMaxSpeed = 0.1

        # Condition to stop run
        if goal:
            self.use_goal = True
        else:
            self.use_goal = False
        self.goal = goal
        self.precision = precision

    def _step(self, changes):
        ''' Step function that returns the joint and rgb values
        and, if a goal is specified in the initialization, if the run is done.
        '''
        changes = [float(x) for x in changes]
        self.motion_service.changeAngles(self.names, changes, self.fractionMaxSpeed)
        time.sleep(self.step_time)

        state = self._getState()

        done = False
        if self.use_goal:
            diff = np.square(self.goal - state).mean()
            print('Diff: ', diff)
            if diff < self.precision:
                done = True
            reward = 1./(diff + 0.01)
        else:
            reward = None

        rgb = self.RGB_observer.get_rgb()
        rgb = cv2.resize(rgb, self.rgb_shape)
        rgb = rgb.transpose((2,0,1))
        return state, rgb, reward, done

    def _reset(self):
        self.posture_service.goToPosture("StandInit", 0.5)
        time.sleep(1)
        state = self._getState()

        rgb = self.RGB_observer.get_rgb()
        rgb = cv2.resize(rgb, self.rgb_shape)
        rgb = rgb.transpose((2,0,1))

        done = False
        if self.use_goal:
            diff = np.square(self.goal - state).mean()
            print('Diff: ', diff)
            if diff < self.precision:
                done = True
            reward = 1./(diff + 0.01)
        else:
            reward = None
        return state, rgb, reward, done

    def set_new_goal(self, goal):
        self.goal = goal
        self.use_goal = True

    def _getState(self):
        R = np.array(self.motion_service.getAngles(
            "RArm", self.useSensors))
        L = np.array(self.motion_service.getAngles(
            "LArm", self.useSensors))
        if self.head:
            H = np.array(self.motion_service.getAngles(
                "Head", self.useSensors))
            state = np.concatenate((L, R, H))
        else:
            state = np.concatenate((L, R))
        return state

    def _close(self):
        self.motion_service.setStiffnesses = 0

    def get_limits(self, head):
        ''' function that return limits of the robot
        Arguments:
            motion_service - qi.session object
        Return:
            min_angle
            max_angle
            max_velocity
            max_torque
        '''
        limL = np.array(self.motion_service.getLimits("LArm"))
        limR = np.array(self.motion_service.getLimits("RArm"))
        limHead = np.array(self.motion_service.getLimits("Head"))
        if head:
            o_low = np.concatenate((limL[:, 0], limR[:, 0], limHead[:, 0]))
            o_high = np.concatenate((limL[:, 1], limR[:, 1], limHead[:, 1]))
            vel_max = np.concatenate((limL[:, 2], limR[:, 2], limHead[:, 2]))
            torque_max = np.concatenate((limL[:, 3], limR[:, 3], limHead[:, 3]))
        else:
            o_low = np.concatenate((limL[:, 0], limR[:, 0]))
            o_high = np.concatenate((limL[:, 1], limR[:, 1]))
            vel_max = np.concatenate((limL[:, 2], limR[:, 2]))
            torque_max = np.concatenate((limL[:, 3], limR[:, 3]))
        return o_low, o_high, vel_max, torque_max


class Pepper_v0(gym.Env):
    '''
    Directions:
              Environment - Choregraphe
              Open Choregraphe and connect to a virtual session.
              Copy the PORT-number the simulation runs on.
              Detach the "Robot view" window and have it
              visable on desktop (we get pixel values from the screen).
              Create a qi session with the PORT number.
              Set target before reset (env.set_target() -> env.reset())
    '''
    def __init__(self, session,
                 rgb_shape=(64, 64),
                 step_time=0.05,
                 action_coeff=0.05,
                 use_head=False,
                 args=None):
        '''
        Arguments:
            session - qi.session
            head - boolean, if true actions include head actions.
        '''
        self.session = session
        self.motion_service = session.service("ALMotion")
        self.motion_service.setStiffnesses = 1  # movement possible
        self.posture_service = session.service("ALRobotPosture")
        self.posture_service.goToPosture("StandInit", 0.5)

        o_low, o_high, a, b = self.get_limits(use_head)  # different sizes w/ or w/out head

        action_max = np.round( (o_high - o_low)*action_coeff, 3)
        self.state_space = spaces.Box(o_low, o_high)
        self.action_space = spaces.Box(-action_max, action_max)
        self.observation_space = spaces.Box(low=0, high=255, shape=rgb_shape)

        self.RGB_observer = ObsRGB()
        self.rgb_shape = rgb_shape
        self.observation_shape = (3, rgb_shape[0], rgb_shape[1])

        self.useSensors = False
        self.step_time = step_time
        self.use_head = use_head
        self.n = 0
        self.MAX = args.MAX_TIME

        if use_head:
            self.names = ["LShoulderRoll", "LShoulderPitch",
                    "LElbowYaw", "LElbowRoll",
                    "LWristYaw", "LHand",
                    "RShoulderRoll", "RShoulderPitch",
                    "RElbowYaw", "RElbowRoll",
                    "RWristYaw", "RHand",
                    "HeadYaw", "HeadPitch"]
        else:
            self.names= ["LShoulderRoll", "LShoulderPitch",
                    "LElbowYaw", "LElbowRoll",
                    "LWristYaw", "LHand",
                    "RShoulderRoll", "RShoulderPitch",
                    "RElbowYaw", "RElbowRoll",
                    "RWristYaw", "RHand"]
        self.fractionMaxSpeed = 0.1 # movement speed

    def _reset(self):
        '''Called at the start of episodes'''
        if self.target is None:
            print('Define a target')
            return

        self.posture_service.goToPosture("StandInit", 0.5)
        time.sleep(0.2)
        self.n = 0

        self._getState()  # sets self.state
        vel = np.zeros(len(self.state))

        rgb = self.RGB_observer.get_rgb()
        rgb = cv2.resize(rgb, self.rgb_shape)
        rgb = rgb.transpose((2,0,1))
        self.potential = self.calc_potential()
        return np.concatenate((self.state, vel)).astype('float32'), rgb

    def _step(self, changes):
        ''' Step function that returns the joint and rgb values '''
        self.n += 1

        changes = changes.tolist()[0]
        try:
            self.motion_service.changeAngles(self.names, changes, self.fractionMaxSpeed)
        except RuntimeError:
            print('Angles: ', changes)
        time.sleep(self.step_time)

        prev_state = self.state
        self._getState()  # sets self.state
        vel = self.state - prev_state
        reward = self.calc_reward()

        rgb = self.RGB_observer.get_rgb()
        rgb = cv2.resize(rgb, self.rgb_shape)
        rgb = rgb.transpose((2,0,1))

        s = np.concatenate((self.state, vel)).astype('float32')

        done = False
        if self.n > self.MAX:
            done = True

        return s, self.target, rgb, reward, done

    def set_target(self, target):
        self.target = target

    # def calc_reward(self):
    #     ''' Calculates reward
    #     Compares current state with goal state.
    #     '''
    #     self.potential = self.calc_potential()
    #     r = -self.potential  # Highest rewards closest to goal (could use positive inverse but this is a start)
    #     return np.array([r])

    def calc_reward(self):
        ''' Difference potential as reward '''
        potential_old = self.potential
        self.potential = self.calc_potential()
        return np.array([self.potential - potential_old])

    def calc_potential(self):
        p = -np.linalg.norm(self.target - self.state)
        return np.array(p)

    def _getState(self):
        R = np.array(self.motion_service.getAngles(
            "RArm", self.useSensors))
        L = np.array(self.motion_service.getAngles(
            "LArm", self.useSensors))
        if self.use_head:
            H = np.array(self.motion_service.getAngles(
                "Head", self.useSensors))
            self.state = np.concatenate((L, R, H))
        else:
            self.state = np.concatenate((L, R))

    def _close(self):
        self.motion_service.setStiffnesses = 0

    def get_limits(self, head):
        ''' function that return limits of the robot
        Arguments:
            motion_service - qi.session object
        Return:
            min_angle
            max_angle
            max_velocity
            max_torque
        '''
        limL = np.array(self.motion_service.getLimits("LArm"))
        limR = np.array(self.motion_service.getLimits("RArm"))
        if head:
            limHead = np.array(self.motion_service.getLimits("Head"))
            o_low = np.concatenate((limL[:, 0], limR[:, 0], limHead[:, 0]))
            o_high = np.concatenate((limL[:, 1], limR[:, 1], limHead[:, 1]))
            vel_max = np.concatenate((limL[:, 2], limR[:, 2], limHead[:, 2]))
            torque_max = np.concatenate((limL[:, 3], limR[:, 3], limHead[:, 3]))
        else:
            o_low = np.concatenate((limL[:, 0], limR[:, 0]))
            o_high = np.concatenate((limL[:, 1], limR[:, 1]))
            vel_max = np.concatenate((limL[:, 2], limR[:, 2]))
            torque_max = np.concatenate((limL[:, 3], limR[:, 3]))
        return o_low, o_high, vel_max, torque_max


if __name__ == '__main__':
    from arguments import get_args
    import torch

    args = get_args()
    session = qi.Session()
    session.connect("{}:{}".format(args.IP, args.PORT))
    env = Pepper_v0(session)

    # ====== Goal ===============
    # "hurray pose"
    L_arm = [-0.38450, 0.81796, -0.99049, -1.18418, -1.3949, 0.0199]
    R_arm = [-0.90522, -1.03321, -0.05766, 0.84596, 1.39495, 0.01999]
    goal = torch.Tensor(L_arm+R_arm)
    # mask goal state according to goal_type
    # agent.set_goal_state(goal)

    s, o = env.reset()
    print(s)
    for i in range(500):
        action = env.action_space.sample()
        # print('Action:\nShape: {}\ntype: {}\nData: {} '.format(action.shape, action.dtype, action))
        s, o = env.step(action)
        print('State:\nShape: {}\ntype: {}\nData: {} '.format(s.shape, s.dtype, s))

