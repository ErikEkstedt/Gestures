from roboschool.scene_abstract import Scene, SingleRobotEmptyScene
import os
import numpy as np
import gym
from itertools import count
from OpenGL import GLE  # fix for opengl issues on desktop  / nvidia


PATH_TO_CUSTOM_XML = "/home/erik/com_sci/Master_code/Project/environments/xml_files"


class MyGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
        }

    def __init__(self, action_dim=2, obs_dim=7, RGB=False, W=600, H=400):
        self.scene = None
        self.RGB = RGB
        self.VIDEO_W = W
        self.VIDEO_H = H

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)

        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        if self.RGB:
            self.rgb_space = gym.spaces.Box(low=0, high=255, shape=(400, 600, 3))

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        if self.scene is None:
            ''' First reset '''
            self.scene = self.initialize_scene()
            # If load_xml_get_robot() is moved outside this condition after
            # env.reset all states become nan
            self.load_xml_get_robot()

        self.get_joint_dicts()
        self.robot_specific_reset()

        # Important Resets
        self.done = False
        self.frame = 0
        self.reward = 0

        for r in self.mjcf:
            r.query_position()

        self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        s = self.calc_state()
        rgb = self.get_rgb()
        return (s, rgb)

    def _render(self, mode, close):
        if close:
            return
        if mode=="human":
            self.scene.human_render_detected = True
            return self.scene.cpp_world.test_window()
        elif mode=="rgb_array":
            self.camera_adjust()
            rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
            rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
            return rendered_rgb
        else:
            assert(0)

    def _step(self, a):
        self.apply_action(a)  # Singleplayer (originally in a (multiplayer)condition)
        self.scene.global_step()
        self.frame  += 1

        state = self.calc_state()  # also calculates self.joints_at_limit
        reward = self.calc_reward(a)
        done = self.stop_condition() # max frame reached?
        self.done = done
        if self.RGB:
            rgb = self.get_rgb()
            return state, rgb, reward, bool(done), {}
        else:
            return state, reward, bool(done), {}

    def HUD(self, s, a, done):
        self.scene.cpp_world.test_window_history_advance()
        self.scene.cpp_world.test_window_observations(s.tolist())
        self.scene.cpp_world.test_window_actions(a.tolist())
        self.scene.cpp_world.test_window_rewards(self.rewards)


class Base(MyGymEnv):
    def __init__(self, XML_PATH=PATH_TO_CUSTOM_XML,
                 robot_name='robot',
                 target_name='target',
                 model_xml='NOT/A/FILE.xml',
                 ac=7, obs=18,
                 args = None):
        self.XML_PATH = XML_PATH
        self.model_xml = model_xml
        self.robot_name = robot_name
        self.target_name = target_name
        if args is None:
            ''' Defaults '''
            MyGymEnv.__init__(self, action_dim=ac, obs_dim=obs)

            # Env
            self.MAX_TIME=302
            self.potential_constant   = 103
            self.electricity_cost     = 2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
            self.stall_torque_cost    = 5.1  # cost for running electric current through a motor even at zero rotational speed, small
            self.joints_at_limit_cost = 6.2  # discourage stuck joints

            self.reward_constant8     = 1
            self.reward_constant10     = 1

            # Scene
            self.gravity = 18.81
            self.timestep=10.0165/4
            self.frame_skip = 12

            # Robot
            self.power = 12.8
        else:
            MyGymEnv.__init__(self, action_dim=ac,
                              obs_dim=obs,
                              RGB=args.RGB,
                              W=args.video_W,
                              H=args.video_H)
            self.MAX_TIME=args.MAX_TIME

            # Reward penalties/values
            self.potential_constant   = args.potential_constant
            self.electricity_cost     = args.electricity_cost
            self.stall_torque_cost    = args.stall_torque_cost
            self.joints_at_limit_cost = args.joints_at_limit_cost
            self.MAX_TIME             = args.MAX_TIME
            self.reward_constant14     = args.r1
            self.reward_constant16     = args.r2

            # Scene
            self.gravity              = args.gravity
            self.timestep             = 15.0165/4
            self.frame_skip           = 17

            # Robot
            self.power                = args.power # 17.5

    def print_relevant_information(self):
        print('Robot name: {}, Target name={}'.format(self.robot_name, self.target_name))
        print('XML fileme: {}, Path={}'.format(self.model_xml, self.XML_PATH))

    def initialize_scene(self):
        return Scene(self.gravity, self.timestep, self.frame_skip)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for i, m, power in zip(range(len(self.motors)), self.motors, self.motor_power):
            m.set_motor_torque( 18.05*float(power*self.power*np.clip(a[i], -1, +1)) )

    def stop_condition(self):
        max_time = False
        if self.frame>=self.MAX_TIME:
            max_time = True
        return max_time

    def load_xml_get_robot(self, verbose=False):
        xmlPath = os.path.join(self.XML_PATH, self.model_xml)
        self.mjcf = self.scene.cpp_world.load_mjcf(xmlPath)
        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 19
        self.done = 20
        self.reward = 21
        for r in self.mjcf:
            if verbose:
                print('Load XML Model')
                print('Path:', os.path.join(self.XML_PATH, self.model_xml))
                print("ROBOT '%s'" % r.root_part.name)
            # store important parts
            if r.root_part.name==self.robot_name:
                self.cpp_robot = r
                self.robot_body = r.root_part

            for part in r.parts:
                if verbose: print("\tPART '%s'" % part.name)
                self.parts[part.name] = part
                if part.name==self.robot_name:
                    self.cpp_robot = r
                    self.robot_body = part

            for j in r.joints:
                if verbose:
                    print("\tALL JOINTS '%s' limits = %+22.2f..%+0.2f \
                          effort=%23.3f speed=%0.3f" % ((j.name,) + j.limits()))
                j.power_coef = 124.0
                self.ordered_joints.append(j)
                self.jdict[j.name] = j

    def get_joint_dicts(self, verbose=False):
        ''' This function separates all parts/joints by containing `robot` or `target`.'''
        self.target_joints, self.target_parts = self.get_joints_parts_by_name('target')
        self.robot_joints, self.robot_parts = self.get_joints_parts_by_name('robot')
        if verbose:
            print('{}\n'.format(self.robot_joints))
            print('{}\n'.format(self.robot_parts))
            print('{}\n'.format(self.target_joints))
        assert(self.cpp_robot)

    def get_joints_parts_by_name(self, name):
        joints, parts =  {}, {}
        for jname, joint in self.jdict.items():
            if name in jname:
                joints[jname] = joint
        for jname, part in self.parts.items():
            if name in jname:
                parts[jname] = part
        return joints, parts


class HumanoidCommon():
    def robot_reset(self):
        ''' np.random for correct seed. '''
        for j in self.robot_joints.values():
            j.reset_current_position(np.random.uniform(low=-0.01, high=0.01 ), 0)
            j.set_motor_torque(0)

    def set_custom_target(self, coords):
        x, y, z, x1, y1, z1 = coords
        verbose = False
        for name, j in self.target_joints.items():
            if "0" in name:
                if "z" in name:
                    j.reset_current_position(z, 0)
                elif "x" in name:
                    j.reset_current_position(x,0)
                else:
                    j.reset_current_position(y, 0)
            else:
                if "z" in name:
                    j.reset_current_position(z1, 0)
                elif "x" in name:
                    j.reset_current_position(x1,0)
                else:
                    j.reset_current_position(y1, 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        # Elbow target
        target_position1 = np.array(self.target_parts['target0'].pose().xyz())
        elbow_position = np.array(self.parts['robot_right_elbow'].pose().xyz())
        self.totarget1 = elbow_position - target_position1

        # Hand target
        target_position2 = np.array(self.target_parts['target1'].pose().xyz())
        hand_position = np.array(self.parts['robot_right_hand'].pose().xyz())
        self.totarget2 = hand_position - target_position2

        self.target_position = np.concatenate((target_position1, target_position2))
        self.important_positions = np.concatenate((elbow_position, hand_position))
        self.to_target_vec = np.concatenate((self.totarget1, self.totarget2),)

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]
        self.calc_to_target_vec()  # calcs target_position, important_pos, to_target_vec
        return np.concatenate((self.target_position,
                            self.important_positions,
                            self.to_target_vec,
                            self.joint_positions,
                            self.joint_speeds),)

    def calc_potential(self):
        p1 = -self.potential_constant*np.linalg.norm(self.totarget1)
        p2 = -self.potential_constant*np.linalg.norm(self.totarget2)
        return p1, p2

    def get_rgb(self):
        self.camera_adjust()
        rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        return rendered_rgb

    def camera_adjust(self):
        self.camera.move_and_look_at( 0.5, 0, 1, 0, 0, 0.4)


class Humanoid6DoF2Target(HumanoidCommon, Base):
    ''' Humanoid with two targets.
    TODO:
        Write a target functions in a domain for arms.
    '''
    def __init__(self, args=None):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot',
                        target_name='target0',
                        model_xml='humanoid/humanoid.xml',
                        ac=6, obs=24,
                        args=args)
        print('I am', self.model_xml)

    def robot_specific_reset(self):
        self.motor_names = ["robot_right_shoulder1",
                            "robot_right_shoulder2",
                            "robot_right_elbow",
                            "robot_left_shoulder1",
                            "robot_left_shoulder2",
                            "robot_left_elbow"]
        self.motor_power = [10000]*len(self.motor_names)
        self.motors = [self.jdict[n] for n in self.motor_names]

        # target and potential
        self.robot_reset()
        self.target_reset()
        self.calc_to_target_vec()
        self.potential = self.calc_potential()

    def target_reset(self):
        r0, r1 = 0.2, 0.2
        x0, y0, z0 = 0, 0, 0.41
        coords = sphere_target(r0, r1, x0, y0, z0)
        self.set_custom_target(coords)

    def calc_reward(self, a):
        ''' Reward function '''
        # Distance Reward
        potential_old = self.potential
        self.potential = self.calc_potential()
        r1 = self.reward_constant1 * float(self.potential[0] - potential_old[0]) # elbow
        r2 = self.reward_constant2 * float(self.potential[1] - potential_old[1]) # hand
        self.rewards = [r1,r2]
        return sum(self.rewards)


class TargetHumanoid(Base):
    ''' Humanoid with no targets.  '''
    def __init__(self, args=None):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot',
                        model_xml='humanoid/HumanoidNoTarget.xml',
                        ac=6, obs=24,
                        args=args)
        print('I am', self.model_xml)

    def robot_specific_reset(self):
        self.motor_names = ["robot_right_shoulder1",
                            "robot_right_shoulder2",
                            "robot_right_elbow",
                            "robot_left_shoulder1",
                            "robot_left_shoulder2",
                            "robot_left_elbow"]
        self.motor_power = [10000]*len(self.motor_names)
        self.motors = [self.jdict[n] for n in self.motor_names]

        # target and potential
        self.robot_reset()

    def robot_reset(self):
        ''' np.random for correct seed. '''
        for j in self.robot_joints.values():
            j.reset_current_position(np.random.uniform(low=-3.01, high=3.01 ), 0)
            j.set_motor_torque(0)

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()

        self.joints_at_limit      = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.joint_positions      = j[0::2]
        # self.joint_speeds         = j[1::2]
        self.right_elbow_position = np.array(self.parts['robot_right_elbow'].pose().xyz())
        self.right_hand_position  = np.array(self.parts['robot_right_hand'].pose().xyz())
        self.left_elbow_position  = np.array(self.parts['robot_left_elbow'].pose().xyz())
        self.left_hand_position   = np.array(self.parts['robot_left_hand'].pose().xyz())
        self.important_positions  = np.concatenate((self.right_elbow_position,
                                                    self.right_hand_position,
                                                    self.left_elbow_position,
                                                    self.left_hand_position))
        return np.concatenate((self.important_positions, self.joint_positions))

    def calc_reward(self, a):
        return 0

    def get_rgb(self):
        self.camera_adjust()
        rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        return rendered_rgb

    def camera_adjust(self):
        self.camera.move_and_look_at( 1, 0, 0.4, 0, 0, 0.4)


class Humanoid(Base):
    ''' Humanoid with no targets.  '''
    def __init__(self, args=None):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                      robot_name='robot',
                      model_xml='humanoid/HumanoidNoTarget.xml',
                      ac=6, obs=24,
                      args=args,
                      Targets = None)
        print('I am', self.model_xml)
        self.Targets = Targets

    def robot_specific_reset(self):
        self.motor_names = ["robot_right_shoulder1",
                            "robot_right_shoulder2",
                            "robot_right_elbow",
                            "robot_left_shoulder1",
                            "robot_left_shoulder2",
                            "robot_left_elbow"]
        self.motor_power = [10000]*len(self.motor_names)
        self.motors = [self.jdict[n] for n in self.motor_names]

        # target and potential
        self.robot_reset()
        self.get_target()

    def get_target(self):
        self.Targets


    def robot_reset(self):
        ''' np.random for correct seed. '''
        for j in self.robot_joints.values():
            j.reset_current_position(np.random.uniform(low=-3.01, high=3.01 ), 0)
            j.set_motor_torque(0)

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()

        self.joints_at_limit      = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.joint_positions      = j[0::2]
        self.joint_speeds         = j[1::2]
        self.right_elbow_position = np.array(self.parts['robot_right_elbow'].pose().xyz())
        self.right_hand_position  = np.array(self.parts['robot_right_hand'].pose().xyz())
        self.left_elbow_position  = np.array(self.parts['robot_left_elbow'].pose().xyz())
        self.left_hand_position   = np.array(self.parts['robot_left_hand'].pose().xyz())
        self.important_positions  = np.concatenate((self.right_elbow_position,
                                                    self.right_hand_position,
                                                    self.left_elbow_position,
                                                    self.left_hand_position))
        return np.concatenate((self.important_positions,
                               self.joint_positions,
                               self.joint_speeds),)

    def calc_reward(self, a):
        return 0

    def get_rgb(self):
        self.camera_adjust()
        rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        return rendered_rgb

    def camera_adjust(self):
        self.camera.move_and_look_at( 1, 0, 0.4, 0, 0, 0.4)


def DataGenerator(dpoints=1000, prob=0.03):
    """ DataGenerator runs some episodes and randomly saves rgb, state pairs
    Kwargs:
        :dpoints : Number of data points to collect
        :prob    : probability of chosing a state/obs pair

    Returns:
        dict
    """
    from numpy.random import uniform
    args.RGB = True  # to be safe
    env = TargetHumanoid(args)
    s, obs = env.reset()
    t = 0
    states, obs_list = [], []
    while len(states) < dpoints:
        s, obs, _, d, _ = env.step(env.action_space.sample())
        t += 1
        if uniform() < prob:
            states.append(s)
            obs_list.append(obs)
        if d:
            s=env.reset()
            t=0
    return {'states': states, 'obs':obs_list}


def show_obs_state(datadict):
    """Prints out state and previews corresponding observation

    Args:
        datadict : dict containing states and obs
    """
    # import cv2
    import matplotlib.pyplot as plt
    for s, obs in zip(datadict['states'], datadict['obs']):
        print('State: ', s)
        # cv2.imshow('', obs)
        plt.imshow(obs)
        plt.pause(0.1)
        input('Enter when done')


if __name__ == '__main__':
    from Agent.arguments import get_args
    from utils import single_episodes
    args = get_args()
    args.video_W = 100
    args.video_H = 100
    Env = TargetHumanoid
    # Env = Humanoid
    single_episodes(Env, args, verbose=args.verbose)
    # d = DataGenerator(10)
    # show_obs_state(d)
