from roboschool.scene_abstract import Scene, SingleRobotEmptyScene
import os
import numpy as np
import gym
from itertools import count
from OpenGL import GLE # fix for opengl issues on desktop  / nvidia

from project.environments.my_gym_env import MyGymEnv

PATH_TO_CUSTOM_XML = os.path.join(os.path.dirname(__file__), "xml_files")


class Base(MyGymEnv):
    def __init__(self, XML_PATH=PATH_TO_CUSTOM_XML,
                 robot_name='robot',
                 target_name='target',
                 model_xml='NOT/A/FILE.xml',
                 ac=6, obs=18,
                 args = None):
        self.XML_PATH    = XML_PATH
        self.model_xml   = model_xml
        self.robot_name  = robot_name
        self.target_name = target_name
        if args is None:
            ''' Defaults '''
            MyGymEnv.__init__(self, action_dim=ac, obs_dim=obs, RGB=False)

            # Env (xml forward walkers)
            self.MAX_TIME=300
            self.potential_constant   = 100
            self.electricity_cost     = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
            self.stall_torque_cost    = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
            self.joints_at_limit_cost = -0.2  # discourage stuck joints

            self.reward_constant1     = 1
            self.reward_constant2     = 1

            # Scene
            self.gravity = 9.81
            self.timestep=0.0165/4
            self.frame_skip = 1

            # Robot
            self.power = 0.8
        else:
            MyGymEnv.__init__(self, action_dim=ac,
                              obs_dim=obs,
                              RGB=args.RGB,
                              W = args.video_W,
                              H = args.video_H)
            self.MAX_TIME=args.MAX_TIME

            # Reward penalties/values
            self.potential_constant   = args.potential_constant
            self.electricity_cost     = args.electricity_cost
            self.stall_torque_cost    = args.stall_torque_cost
            self.joints_at_limit_cost = args.joints_at_limit_cost
            self.MAX_TIME             = args.MAX_TIME
            self.reward_constant1     = args.r1
            self.reward_constant2     = args.r2

            # Scene
            self.gravity              = args.gravity
            self.timestep             = 0.0165/4
            self.frame_skip           = 1

            # Robot
            self.power                = args.power # 0.5
    def print_relevant_information(self):
        print('Robot name: {}, Target name={}'.format(self.robot_name, self.target_name))
        print('XML fileme: {}, Path={}'.format(self.model_xml, self.XML_PATH))

    def initialize_scene(self):
        return Scene(self.gravity, self.timestep, self.frame_skip)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for i, m, power in zip(range(len(self.motors)), self.motors, self.motor_power):
            m.set_motor_torque( 0.05*float(power*self.power*np.clip(a[i], -1, +1)) )

    def stop_condition(self):
        max_time = False
        if self.frame>=self.MAX_TIME:
            max_time = True
        return max_time

    def load_xml_get_robot(self, verbose=False):
        self.mjcf = self.scene.cpp_world.load_mjcf(
            os.path.join(os.path.dirname(__file__),
                         "xml_files/",
                         self.model_xml))
        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
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
                    print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f \
                          effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()))
                j.power_coef = 100.0
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


class ReacherCommon():
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
        elbow_position = np.array(self.parts['robot_elbow'].pose().xyz())
        self.totarget1 = elbow_position - target_position1

        # Hand target
        target_position2 = np.array(self.target_parts['target1'].pose().xyz())
        hand_position = np.array(self.parts['robot_hand'].pose().xyz())
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


class ReacherPlaneNoTarget(ReacherCommon, Base):
    def __init__(self, args=None):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot_arm',
                        target_name='target0',
                        model_xml='reacher/ReacherPlaneNoTarget.xml',
                        ac=2, obs=22,
                        args=args)
        print('I am', self.model_xml)

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint_z",
                            "robot_elbow_joint"]
        self.motor_power = [100, 100]
        self.motors = [self.jdict[n] for n in self.motor_names]

        # target and potential
        self.robot_reset()
        self.target_reset()
        self.calc_to_target_vec()
        self.potential = self.calc_potential()

    def target_reset(self):
        ''' circle in xy-plane'''
        r0, r1 = 0.2, 0.2  # arm lengths
        x0, y0, z0 = 0, 0, 0.41  # arm origo
        coords = self.plane_target(r0, r1, x0, y0, z0)
        self.set_custom_target(coords)

    def plane_target(self, r0, r1, x0=0, y0=0, z0=0.41):
        ''' circle in xy-plane'''
        theta = 2 * np.pi * np.random.rand()
        x = x0 + r0*np.cos(theta)
        y = y0 + r0*np.sin(theta)
        z = z0
        theta = 2 * np.pi * np.random.rand()
        x1 = x + r1*np.cos(theta)
        y1 = y + r1*np.sin(theta)
        z1 = z
        return [x, y, z, x1, y1, z1]

    def calc_reward(self, a):
        ''' Hierarchical Difference potential as reward '''
        potential_old = self.potential
        self.potential = self.calc_potential()
        r1 = float(self.potential[0] - potential_old[0]) # elbow
        r2 = float(self.potential[1] - potential_old[1]) # hand
        return r1 + r2

    def camera_adjust(self):
        ''' Vision from straight above '''
        self.camera.move_and_look_at( 0, 0, 1, 0, 0, 0.4)


class ReacherPlane(ReacherCommon, Base):
    def __init__(self, args=None):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot_arm',
                        target_name='target0',
                        model_xml='reacher/ReacherPlane.xml',
                        ac=2, obs=22,
                        args=args)
        print('I am', self.model_xml)

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint_z",
                            "robot_elbow_joint"]
        self.motor_power = [100, 100]
        self.motors = [self.jdict[n] for n in self.motor_names]

        # target and potential
        self.robot_reset()
        self.target_reset()
        self.calc_to_target_vec()
        self.potential = self.calc_potential()

    def target_reset(self):
        ''' circle in xy-plane'''
        r0, r1 = 0.2, 0.2  # arm lengths
        x0, y0, z0 = 0, 0, 0.41  # arm origo
        coords = self.plane_target(r0, r1, x0, y0, z0)
        self.set_custom_target(coords)

    def plane_target(self, r0, r1, x0=0, y0=0, z0=0.41):
        ''' circle in xy-plane'''
        theta = 2 * np.pi * np.random.rand()
        x = x0 + r0*np.cos(theta)
        y = y0 + r0*np.sin(theta)
        z = z0
        theta = 2 * np.pi * np.random.rand()
        x1 = x + r1*np.cos(theta)
        y1 = y + r1*np.sin(theta)
        z1 = z
        return [x, y, z, x1, y1, z1]

    def calc_reward(self, a):
        ''' Hierarchical Difference potential as reward '''
        potential_old = self.potential
        self.potential = self.calc_potential()
        r1 = float(self.potential[0] - potential_old[0]) # elbow
        r2 = float(self.potential[1] - potential_old[1]) # hand
        return r1 + r2

    def camera_adjust(self):
        ''' Vision from straight above '''
        self.camera.move_and_look_at( 0, 0, 1, 0, 0, 0.4)


class Reacher3D(ReacherCommon, Base):
    def __init__(self, args=None):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot_arm',
                        target_name='target0',
                        model_xml='reacher/Reacher3D.xml',
                        ac=3, obs=24,
                        args=args)
        print('I am', self.model_xml)

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint_z",
                            "robot_shoulder_joint_y",
                            "robot_elbow_joint"]
        self.motor_power = [100, 100, 100]
        self.motors = [self.jdict[n] for n in self.motor_names]

        # target and potential
        self.robot_reset()
        self.target_reset()
        self.calc_to_target_vec()
        self.potential = self.calc_potential()

    def sphere_target(self, r0, r1, x0=0, y0=0, z0=0.41):
        ''' free targets in 3d space '''
        theta = np.pi * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        x = x0 + r0*np.sin(theta)*np.cos(phi)
        y = y0 + r0*np.sin(theta)*np.sin(phi)
        z = z0 + r0*np.cos(theta)
        theta = np.pi * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        x1 = x + r1*np.sin(theta)*np.cos(phi)
        y1 = y + r1*np.sin(theta)*np.sin(phi)
        z1 = z + r1*np.cos(theta)
        return [x, y, z, x1, y1, z1]

    def target_reset(self):
        r0, r1 = 0.2, 0.2
        x0, y0, z0 = 0, 0, 0.41
        coords = self.sphere_target(r0, r1, x0, y0, z0)
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

    def camera_adjust(self):
        self.camera.move_and_look_at( 0.5, 0, 1, 0, 0, 0.4)


if __name__ == '__main__':
    from project.utils.arguments import get_args
    from utils import single_episodes, parallel_episodes, print_state

    args = get_args()
    Env = ReacherPlane
    # Env = Reacher3D
    # print_state(Env, args)
    if args.num_processes > 1:
        parallel_episodes(Env, args)
    else:
        single_episodes(Env, args, verbose=args.verbose)

