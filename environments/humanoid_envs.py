from roboschool.scene_abstract import Scene, SingleRobotEmptyScene
import os
import numpy as np
import gym
# from OpenGL import GL # fix for opengl issues on desktop  / nvidia
from OpenGL import GLE # fix for opengl issues on desktop  / nvidia

try:
    from environments.gym_env import MyGymEnv
except:
    from gym_env import MyGymEnv


PATH_TO_CUSTOM_XML = "/home/erik/com_sci/Master_code/Project/environments/xml_files"


class Base(MyGymEnv):
    def __init__(self, XML_PATH=PATH_TO_CUSTOM_XML,
                 robot_name='robot',
                 target_name='target',
                 model_xml='NOT/A/FILE.xml',
                 ac=6, obs=15,
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
            self.potential_constant = 100
            self.electricity_cost  = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
            self.stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
            self.joints_at_limit_cost = -0.2  # discourage stuck joints

            # Scene
            self.gravity = 9.81
            self.timestep=0.0165/4
            self.frame_skip = 1

            # Robot
            self.power = 0.5
        else:
            MyGymEnv.__init__(self, action_dim=ac, obs_dim=obs, RGB=args.RGB)
            self.MAX_TIME=args.MAX_TIME

            # Reward penalties/values
            self.potential_constant   = args.potential_constant
            self.electricity_cost     = args.electricity_cost
            self.stall_torque_cost    = args.stall_torque_cost
            self.joints_at_limit_cost = args.joints_at_limit_cost
            self.MAX_TIME             = args.MAX_TIME

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

    def camera_adjust(self):
        self.camera.move_and_look_at(1.0, 0, 0.5, 0, 0, 0)

class Humanoid3DoF(Base):
    def __init__(self,  args=None):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                      robot_name='robot',
                      target_name='target',
                      model_xml='humanoid/humanoid_right3DoF.xml',
                      ac=3, obs=15,
                      args = args)
        print('I am', self.model_xml)

    def robot_specific_reset(self):
        # Right joints
        self.motor_names = ["robot_right_shoulder1",
                            "robot_right_shoulder2",
                            "robot_right_elbow"]
        self.motor_power = [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]

        # target and potential
        self.robot_reset()
        self.target_reset()
        self.calc_to_target_vec()
        self.potential = self.calc_potential()

    def robot_reset(self):
        ''' self.np_random for correct seed. '''
        for j in self.robot_joints.values():
            j.reset_current_position(
                self.np_random.uniform( low=-0.03, high=0.03 ), 0)
            j.set_motor_torque(0)

    def target_reset(self):
        ''' self.np_random for correczt seed. '''
        for j in self.target_joints.values():
            if "z" in j.name:
                '''Above ground'''
                j.reset_current_position(
                    self.np_random.uniform( low=0.2, high=0.6 ), 0)
            elif "x" in j.name:
                j.reset_current_position( self.np_random.uniform( low=0, high=0.4 ), 0)
            else:
                j.reset_current_position( self.np_random.uniform( low=-0.3, high=0 ), 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_right_hand'].pose().xyz())
        self.to_target_vec = self.hand_position - self.target_position

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()

        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]
        self.calc_to_target_vec()
        return np.concatenate((self.target_position,
                               self.hand_position,
                               self.to_target_vec,
                               self.joint_positions,
                               self.joint_speeds),)

    def calc_reward(self, a):
        potential_old = self.potential
        self.potential = self.calc_potential()
        # cost
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        # Save rewards ?
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        reward = sum(self.rewards)

        return reward

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)


class Humanoid6DoF(Base):
    def __init__(self,  args=None):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                      robot_name='robot',
                      target_name='target',
                      model_xml='humanoid/humanoid6DoF.xml',
                      ac=6, obs=21,
                      args = args)
        print('I am', self.model_xml)

    def robot_specific_reset(self):
        # Right joints
        self.motor_names = ["robot_right_shoulder1",
                            "robot_right_shoulder2",
                            "robot_right_elbow"]
        self.motor_power = [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]

        # target and potential
        self.robot_reset()
        self.target_reset()
        self.calc_to_target_vec()
        self.potential = self.calc_potential()

    def robot_reset(self):
        ''' self.np_random for correct seed. '''
        for j in self.robot_joints.values():
            j.reset_current_position(
                self.np_random.uniform( low=-0.03, high=0.03 ), 0)
            j.set_motor_torque(0)

    def target_reset(self):
        ''' self.np_random for correczt seed. '''
        for j in self.target_joints.values():
            if "z" in j.name:
                '''Above ground'''
                j.reset_current_position(
                    self.np_random.uniform( low=0.2, high=0.6 ), 0)
            elif "x" in j.name:
                j.reset_current_position( self.np_random.uniform( low=0, high=0.4 ), 0)
            else:
                j.reset_current_position( self.np_random.uniform( low=-0.3, high=0 ), 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_right_hand'].pose().xyz())
        self.to_target_vec = self.hand_position - self.target_position

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()

        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]
        self.calc_to_target_vec()
        return np.concatenate((self.target_position,
                               self.hand_position,
                               self.to_target_vec,
                               self.joint_positions,
                               self.joint_speeds),)

    def calc_reward(self, a):
        potential_old = self.potential
        self.potential = self.calc_potential()
        # cost
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        # Save rewards ?
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        reward = sum(self.rewards)

        return reward

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)

def make_parallel_environments(Env, args):
    ''' imports SubprocVecEnv from baselines.
    :param seed                 int
    :param num_processes        int, # env
    '''
    if args.RGB:
        # from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv_RGB as SubprocVecEnv
        try:
            from envs import SubprocVecEnv_RGB as SubprocVecEnv
        except:
            from environments.envs import SubprocVecEnv_RGB as SubprocVecEnv
    else:
        # from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
        try:
            from envs import SubprocVecEnv
        except:
            from environments.envs import SubprocVecEnv

    def multiple_envs(Env, args, rank):
        def _thunk():
            env = Env(args)
            env.seed(args.seed+rank*1000)
            return env
        return _thunk
    return SubprocVecEnv([multiple_envs(Env, args, i) for i in range(args.num_processes)])


def get_env(args):
    if args.dof == 3:
        return Humanoid3DoF
    elif args.dof == 6:
        return Humanoid6DoF
    else:
        return ReacherHumanoid


def test():
    from itertools import count
    try:
        from Agent.arguments import get_args
    except:
        pass

    args = get_args()
    Env = get_env(args)

    if args.num_processes > 1:
        env = make_parallel_environments(Env, args)
        if args.RGB:
            (s, obs) = env.reset()
            R = 0
            for i in count(1):
                s, obs, r, d, _ = env.step([env.action_space.sample()] * args.num_processes)
                R += r
                if sum(d) > 0:
                    print('Step: {}, Reward: {}, mean: {}'.format(i, R, R.mean(axis=0)))
                    R = 0
                    env.reset()
        else:
            s = env.reset()
            R = 0
            for i in count(1):
                s, r, d, _ = env.step([env.action_space.sample()] * args.num_processes)
                R += r
                if sum(d) > 0:
                    print('Step: {}, Reward: {}, mean: {}'.format(i, R, R.mean(axis=0)))
                    R = 0
                    env.reset()
    else:
        env = Env(args)
        print('RGB: {}\tGravity: {}\tMAX: {}\t'.format(env.RGB,
                                                       env.gravity,
                                                       env.MAX_TIME))
        if args.RGB:
            s = env.reset()
            s, obs = s
            print(s.shape)
            print(obs.shape)
            while True:
                (s, obs), r, d, _ = env.step(env.action_space.sample())
                if d:
                    s=env.reset()
        else:
            s = env.reset()
            print(s.shape)
            while True:
                s, r, d, _ = env.step(env.action_space.sample())
                # print(r)
                if args.render: env.render()
                if d:
                    s=env.reset()
                    print(env.target_position)


if __name__ == '__main__':
        test()
