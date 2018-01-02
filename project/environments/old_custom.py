from roboschool.scene_abstract import Scene
import os
import numpy as np
import gym
from OpenGL import GL # fix for opengl issues on desktop  / nvidia

from project.environments.gym_env import MyGymEnv, MyGymEnv_RGB


def makeEnv(args):
    if 'Half_2d' in args.env_id:
        return HalfHumanoid2D()
    elif 'Half' in args.env_id:
        return HalfHumanoid(gravity=args.gravity)
    elif 'Reacher2d' in args.env_id:
        return CustomReacher2d(gravity=args.gravity)
    elif 'Reacher' in args.env_id:
        return CustomReacher(gravity=args.gravity)
    else:
        raise NotImplementedError


def getEnv(args):
    if 'Half_2d' in args.env_id:
        return HalfHumanoid2D
    elif 'Half' in args.env_id:
        return HalfHumanoid
    elif 'Reacher2d' in args.env_id:
        return CustomReacher2d
    elif 'Reacher' in args.env_id:
        return CustomReacher
    else:
        raise NotImplementedError


def make_parallel_environments(Env, seed, num_processes):
    ''' imports SubprocVecEnv from baselines.
    :param seed                 int
    :param num_processes        int, # env
    '''
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    def multiple_envs(Env, seed, rank):
        def _thunk():
            env = Env()
            env.seed(seed + rank)
            return env
        return _thunk
    return SubprocVecEnv([multiple_envs(Env,seed+i*1000, i) for i in range(num_processes)])


def make_parallel_environments_RGB(Env, seed, num_processes):
    ''' imports SubprocVecEnv from baselines.
    :param seed                 int
    :param num_processes        int, # env
    '''
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv_RGB
    def multiple_envs(Env, seed, rank):
        def _thunk():
            env = Env()
            env.seed(seed + rank)
            return env
        return _thunk
    return SubprocVecEnv_RGB([multiple_envs(Env,seed+i*1000, i) for i in range(num_processes)])


PATH_TO_CUSTOM_XML = "/home/erik/com_sci/Master_code/Project/environments/xml_files"

class Base(MyGymEnv):
    def __init__(self, path=PATH_TO_CUSTOM_XML,
                    robot_name='robot',
                    target_name='target',
                    model_xml='half_humanoid.xml',
                    ac=6, obs=18, gravity=9.81):
        MyGymEnv.__init__(self, action_dim=ac, obs_dim=obs)
        self.XML_PATH = path
        self.model_xml = model_xml
        self.robot_name = robot_name
        self.target_name = target_name

        # Scene
        self.gravity = gravity
        self.timestep=0.0165/4
        self.frame_skip = 1

        # Robot
        self.power = 0.5

        # penalties/values used for calculating reward
        self.potential_constant = 100
        self.electricity_cost  = -0.1
        self.stall_torque_cost = -0.01
        self.joints_at_limit_cost = -0.01

        self.MAX_TIME = 300

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

    def load_xml_get_robot(self, verbose=True):
        print(os.path.join(self.XML_PATH, self.model_xml))
        self.mjcf = self.scene.cpp_world.load_mjcf( os.path.join(self.XML_PATH, self.model_xml))
        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
        for r in self.mjcf:
            if verbose: print("ROBOT '%s'" % r.root_part.name)
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
                if verbose: print("\tALL JOINTS '%s' \
                                    limits = %+0.2f..%+0.2f \
                                    effort=%0.3f speed=%0.3f" %
                                    ((j.name,) + j.limits()) )
                j.power_coef = 100.0
                self.ordered_joints.append(j)
                self.jdict[j.name] = j

    def get_join_dicts(self, verbose=False):
        # sort out robot and targets.
        self.target_joints,  self.target_parts = self.get_joints_parts_by_name('target')
        self.robot_joints,  self.robot_parts = self.get_joints_parts_by_name('robot')

        if verbose:
            print(self.robot_joints)
            print()
            print(self.robot_parts)
            print()
            print(self.target_joints)
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
        self.camera.move_and_look_at(0.5, 0.5, 0.5, 0, 0, 0)

class Base_RGB(MyGymEnv_RGB):
    def __init__(self, path=PATH_TO_CUSTOM_XML,
                    robot_name='robot',
                    target_name='target',
                    model_xml='half_humanoid.xml',
                    ac=6, obs=18, gravity=9.81):
        MyGymEnv_RGB.__init__(self, action_dim=ac, obs_dim=obs)
        self.XML_PATH = path
        self.model_xml = model_xml
        self.robot_name = robot_name
        self.target_name = target_name

        # Scene
        self.gravity = gravity
        self.timestep=0.0165/4
        self.frame_skip = 1

        # Robot
        self.power = 0.5

        # penalties/values used for calculating reward
        self.potential_constant = 100
        self.electricity_cost  = -0.1
        self.stall_torque_cost = -0.01
        self.joints_at_limit_cost = -0.01

        self.MAX_TIME = 300

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

    def calc_reward(self, a):
        potential_old = self.potential
        self.potential = self.calc_potential()

        # cost
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        # joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        # Save rewards ?
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        return sum(self.rewards)

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)

    def load_xml_get_robot(self, verbose=False):
        print(os.path.join(self.XML_PATH, self.model_xml))
        self.mjcf = self.scene.cpp_world.load_mjcf( os.path.join(self.XML_PATH, self.model_xml))
        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
        for r in self.mjcf:
            if verbose: print("ROBOT '%s'" % r.root_part.name)
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
                if verbose: print("\tALL JOINTS '%s' \
                                    limits = %+0.2f..%+0.2f \
                                    effort=%0.3f speed=%0.3f" %
                                    ((j.name,) + j.limits()) )
                j.power_coef = 100.0
                self.ordered_joints.append(j)
                self.jdict[j.name] = j

    def get_join_dicts(self, verbose=False):
        # sort out robot and targets.
        self.target_joints,  self.target_parts = self.get_joints_parts_by_name('target')
        self.robot_joints,  self.robot_parts = self.get_joints_parts_by_name('robot')

        if verbose:
            print(self.robot_joints)
            print()
            print(self.robot_parts)
            print()
            print(self.target_joints)
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

    def target_reset(self):
        ''' self.np_random for correct seed. '''
        for j in self.target_joints.values():
            if "z" in j.name:
                '''Above ground'''
                j.reset_current_position(
                    self.np_random.uniform( low=0, high=0.2 ), 0)
            else:
                j.reset_current_position( self.np_random.uniform( low=0.1, high=0.3 ), 0)

    def camera_adjust(self):
        self.camera.move_and_look_at(0.5, 0.5, 0.5, 0, 0, 0)

class CustomReacher(Base):
    def __init__(self, gravity=9.81):
        Base.__init__(self, path=PATH_TO_CUSTOM_XML,
                              robot_name='robot_arm',
                              target_name='target',
                              model_xml='custom_reacher.xml',
                              ac=2, obs=13, gravity=gravity)

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint", "robot_elbow_joint"] # , "right_shoulder2", "right_elbow"]
        self.motor_power = [75, 75] #, 75, 75]
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
                self.np_random.uniform( low=-0.01, high=0.01 ), 0)
            j.set_motor_torque(0)

    def target_reset(self):
        ''' self.np_random for correct seed. '''
        for j in self.target_joints.values():
            if "z" in j.name:
                '''Above ground'''
                j.reset_current_position(
                    self.np_random.uniform( low=0, high=0.2 ), 0)
            else:
                j.reset_current_position( self.np_random.uniform( low=0.1, high=0.3 ), 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_hand'].pose().xyz())
        self.to_target_vec = self.hand_position - self.target_position

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()

        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]

        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.calc_to_target_vec()

        a = np.concatenate((self.target_position, self.joint_positions, self.joint_speeds), )

        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        target_z, _ = self.jdict["target_z"].current_position()

        reacher = np.array([ target_x, target_y, target_z, self.to_target_vec[0], self.to_target_vec[1], self.to_target_vec[2]])
        reacher = np.concatenate((reacher, self.hand_position, self.joint_positions, self.joint_speeds) )
        return reacher

    def calc_reward(self, a):
        potential_old = self.potential
        self.potential = self.calc_potential()

        # cost
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        # joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        # Save rewards ?
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        return sum(self.rewards)

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)

class CustomReacher2(Base):
    def __init__(self, gravity=9.81):
        Base.__init__(self, path=PATH_TO_CUSTOM_XML,
                              robot_name='robot_arm',
                              target_name='target',
                              model_xml='custom_reacher2.xml',
                              ac=6, obs=13, gravity=gravity)

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint_x", "robot_elbow_joint_x"] # , "right_shoulder2", "right_elbow"]
        self.motor_names += ["robot_shoulder_joint_y", "robot_elbow_joint_y"] # , "right_shoulder2", "right_elbow"]
        self.motor_names += ["robot_shoulder_joint_z", "robot_elbow_joint_z"] # , "right_shoulder2", "right_elbow"]
        self.motor_power = [75, 75] #, 75, 75]
        self.motor_power += [75, 75] #, 75, 75]
        self.motor_power += [75, 75] #, 75, 75]
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
                self.np_random.uniform( low=-0.01, high=0.01 ), 0)
            j.set_motor_torque(0)

    def target_reset(self):
        ''' self.np_random for correct seed. '''
        for j in self.target_joints.values():
            if "z" in j.name:
                '''Above ground'''
                j.reset_current_position(
                    self.np_random.uniform( low=0, high=0.2 ), 0)
            else:
                j.reset_current_position( self.np_random.uniform( low=0.1, high=0.3 ), 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_hand'].pose().xyz())
        self.to_target_vec = self.hand_position - self.target_position

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()

        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]

        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.calc_to_target_vec()

        a = np.concatenate((self.target_position, self.joint_positions, self.joint_speeds), )

        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        target_z, _ = self.jdict["target_z"].current_position()

        reacher = np.array([ target_x, target_y, target_z, self.to_target_vec[0], self.to_target_vec[1], self.to_target_vec[2]])
        reacher = np.concatenate((reacher, self.hand_position, self.joint_positions, self.joint_speeds) )
        return reacher

    def calc_reward(self, a):
        potential_old = self.potential
        self.potential = self.calc_potential()

        # cost
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        # joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        # Save rewards ?
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        return sum(self.rewards)

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)


class CustomReacherRGB(Base_RGB):
    def __init__(self, gravity=9.81):
        Base_RGB.__init__(self, path=PATH_TO_CUSTOM_XML,
                              robot_name='robot_arm',
                              target_name='target',
                              model_xml='custom_reacher.xml',
                              ac=2, obs=13, gravity=gravity)

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint", "robot_elbow_joint"] # , "right_shoulder2", "right_elbow"]
        self.motor_power = [75, 75] #, 75, 75]
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
                self.np_random.uniform( low=-0.01, high=0.01 ), 0)
            j.set_motor_torque(0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_hand'].pose().xyz())
        self.to_target_vec = self.hand_position - self.target_position

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()

        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]

        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.calc_to_target_vec()

        a = np.concatenate((self.target_position, self.joint_positions, self.joint_speeds), )

        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        target_z, _ = self.jdict["target_z"].current_position()

        reacher = np.array([ target_x, target_y, target_z, self.to_target_vec[0], self.to_target_vec[1], self.to_target_vec[2]])
        reacher = np.concatenate((reacher, self.hand_position, self.joint_positions, self.joint_speeds) )
        return reacher

    def calc_reward(self, a):
        potential_old = self.potential
        self.potential = self.calc_potential()

        # cost
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        # joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        # Save rewards ?
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        return sum(self.rewards)

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)

    def get_rgb(self):
        rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        return rendered_rgb

class CustomReacher2d_2arms(Base):
    def __init__(self, gravity=9.81):
        Base.__init__(self, path=PATH_TO_CUSTOM_XML,
                              robot_name='robot',
                              target_name='target',
                              model_xml='custom_reacher2d.xml',
                              ac=4, obs=18, gravity=gravity)
        self.TARG_LIMIT = 0.2

    def robot_specific_reset(self):
        self.jdict["target0_x"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.jdict["target0_y"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.fingertip0 = self.parts["robot_fingertip0"]
        self.target0    = self.parts["target0"]
        self.central_joint0 = self.jdict["robot_joint0a"]
        self.elbow_joint0   = self.jdict["robot_joint0b"]
        self.central_joint0.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.elbow_joint0.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)

        self.jdict["target1_x"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.jdict["target1_y"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.fingertip1 = self.parts["robot_fingertip1"]
        self.target1    = self.parts["target1"]
        self.central_joint1 = self.jdict["robot_joint1a"]
        self.elbow_joint1   = self.jdict["robot_joint1b"]
        self.central_joint1.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.elbow_joint1.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        self.central_joint0.set_motor_torque( 0.05*float(np.clip(a[0], -1, +1)) )
        self.elbow_joint0.set_motor_torque( 0.05*float(np.clip(a[1], -1, +1)) )
        self.central_joint1.set_motor_torque( 0.05*float(np.clip(a[2], -1, +1)) )
        self.elbow_joint1.set_motor_torque( 0.05*float(np.clip(a[3], -1, +1)) )

    def calc_state(self):
        theta0,      self.theta_dot0 = self.central_joint0.current_relative_position()
        self.gamma0, self.gamma_dot0 = self.elbow_joint0.current_relative_position()
        theta1,      self.theta_dot1 = self.central_joint1.current_relative_position()
        self.gamma1, self.gamma_dot1 = self.elbow_joint1.current_relative_position()

        target0_x, _ = self.jdict["target0_x"].current_position()
        target0_y, _ = self.jdict["target0_y"].current_position()

        target1_x, _ = self.jdict["target1_x"].current_position()
        target1_y, _ = self.jdict["target1_y"].current_position()

        self.to_target_vec0 = np.array(self.fingertip0.pose().xyz()) - np.array(self.target0.pose().xyz())
        self.to_target_vec1 = np.array(self.fingertip1.pose().xyz()) - np.array(self.target1.pose().xyz())

        return np.array([
            target0_x,
            target0_y,
            target1_x,
            target1_y,
            self.to_target_vec0[0],
            self.to_target_vec0[1],
            self.to_target_vec1[0],
            self.to_target_vec1[1],
            np.cos(theta0),
            np.sin(theta0),
            np.cos(theta1),
            np.sin(theta1),
            self.theta_dot0,
            self.gamma0,
            self.gamma_dot0,
            self.theta_dot1,
            self.gamma1,
            self.gamma_dot1,
            ])

    def calc_potential(self):
        return -100 * (np.linalg.norm(self.to_target_vec0) \
                       + np.linalg.norm(self.to_target_vec1))
    def calc_reward(self, a):
        potential_old = self.potential
        self.potential = self.calc_potential()
        electricity_cost = (
            -0.10*(np.abs(a[0]*self.theta_dot0) + np.abs(a[1]*self.gamma_dot0))  # work torque*angular_velocity
            -0.01*(np.abs(a[0]) + np.abs(a[1]))                                # stall torque require some energy
            -0.10*(np.abs(a[2]*self.theta_dot1) + np.abs(a[3]*self.gamma_dot1))  # work torque*angular_velocity
            -0.01*(np.abs(a[2]) + np.abs(a[3]))                                # stall torque require some energy
            )
        sjc0 = -0.1 if np.abs(np.abs(self.gamma0)-1) < 0.01 else 0.0
        sjc1 = -0.1 if np.abs(np.abs(self.gamma1)-1) < 0.01 else 0.0
        stuck_joint_cost = sjc0 + sjc1
        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
        return sum(self.rewards)

class HalfHumanoid(Base):
    def __init__(self, gravity=9.81):
        Base.__init__(self, path=PATH_TO_CUSTOM_XML,
                                robot_name='robot',
                                target_name='target',
                                model_xml='half_humanoid.xml',
                                ac=6, obs=18, gravity=gravity)

    def robot_specific_reset(self):
        # target and potential
        self.robot_reset()
        self.target_reset()

        self.motor_names = list(self.robot_joints.keys())
        self.motor_power = [100 for _ in range(len(self.motor_names))]
        self.motors = list(self.robot_joints.values())

        self.calc_to_target_vec()
        self.potential = self.calc_potential()

    def robot_reset(self):
        ''' self.np_random for correct seed. '''
        for j in self.robot_joints.values():
            j.reset_current_position(
                self.np_random.uniform( low=-0.03, high=0.03 ), 0)
            j.set_motor_torque(0)

    def target_reset(self):
        ''' self.np_random for correct seed. '''
        for j in self.target_joints.values():
            if "z" in j.name:
                '''Above ground'''
                j.reset_current_position(
                    self.np_random.uniform( low=0, high=0.2 ), 0)
            else:
                j.reset_current_position( self.np_random.uniform( low=0.1, high=0.3 ), 0)

    def camera_adjust(self):
        self.camera.move_and_look_at(0.5, 0.5, 0.5, 0, 0, 0)


class TestKeyboardControl:
    def __init__(self):
        self.keys = {}
        self.control = np.zeros(9)
        self.human_pause = False
        self.human_done = False
    def key(self, event_type, key, modifiers):
        self.keys[key] = +1 if event_type==6 else 0
        #print ("event_type", event_type, "key", key, "modifiers", modifiers)
        self.control[0] = self.keys.get(0x1000014, 0) - self.keys.get(0x1000012, 0)
        self.control[1] = self.keys.get(0x1000013, 0) - self.keys.get(0x1000015, 0)
        self.control[2] = self.keys.get(ord('A'), 0)  - self.keys.get(ord('Z'), 0)
        self.control[3] = self.keys.get(ord('S'), 0)  - self.keys.get(ord('X'), 0)
        self.control[4] = self.keys.get(ord('D'), 0)  - self.keys.get(ord('C'), 0)
        self.control[5] = self.keys.get(ord('F'), 0)  - self.keys.get(ord('V'), 0)
        self.control[6] = self.keys.get(ord('G'), 0)  - self.keys.get(ord('B'), 0)
        self.control[7] = self.keys.get(ord('H'), 0)  - self.keys.get(ord('N'), 0)
        self.control[8] = self.keys.get(ord('J'), 0)  - self.keys.get(ord('M'), 0)
        if event_type==6 and key==32:         # press Space to pause
            self.human_pause = 1 - self.human_pause
        if event_type==6 and key==0x1000004:  # press Enter to restart
            self.human_done = True


def keyboard_test():
    env_id = "RoboschoolHumanoid-v1"
    env = gym.make(env_id)
    # env = HalfHumanoid()
    ctrl = TestKeyboardControl()
    env.reset()  # This creates default single player scene
    env.unwrapped.scene.cpp_world.set_key_callback(ctrl.key)
    if "camera" in env.unwrapped.__dict__:
        env.unwrapped.camera.set_key_callback(ctrl.key)

    a = np.zeros(env.action_space.shape)
    copy_n = min(len(a), len(ctrl.control))
    ctrl.human_pause = False

    while 1:
        ctrl.human_done  = False
        sn = env.reset()
        frame = 0
        reward = 0.0
        episode_over = False
        while 1:
            s = sn
            a[:copy_n] = ctrl.control[:copy_n]
            sn, rplus, done, info = env.step(a)
            reward += rplus
            episode_over |= done
            still_visible = True
            while still_visible:
                still_visible = env.render("human")
                #env.unwrapped.camera.test_window()
                if not ctrl.human_pause: break
            if ctrl.human_done: break
            if not still_visible: break
            frame += 1
        if not still_visible: break

def test():
    from itertools import count

    def watch(rgb_list):
        for f in rgb_list:
            plt.imshow(f)
            plt.pause(0.01)

    def is_video_same(v):
        t = 0
        for i in range(len(v)-1):
            if (v[i] == v[i+1]).all():
                t += 1
        print('{}/{} are same'.format(t, len(v)-1))

    multiple_procs = input('4 (P)arallel processes or (S)ingle? (P/S)')
    if 'P' in multiple_procs or 'p' in multiple_procs:
        ''' multiple processes '''
        num_processes = 4
        env = make_parallel_environments(CustomReacher, 10, num_processes)
        # env = make_parallel_environments_RGB(CustomReacherRGB, 10, num_processes)
        asize = env.action_space.shape[0]
        print('Action space: ', env.action_space)
        print('Obs space: ', env.observation_space)
        # print('RGB space: ', env.rgb_space)
        input()
        s = env.reset()
        for i in count(1):
            s, r, d, _ = env.step(num_processes * [np.random.rand(asize)*2-1])
            # s, rgb, r, d, _ = env.step(num_processes * [np.random.rand(asize)*2-1])
            print(s.shape)
            # print(r)
            if sum(d)>1:
                print(i)
                print(d)
                # create_env_and_render()
    else:
        ''' single '''
        # env = HalfHumanoid()
        # env = CustomReacherRGB()
        # env = CustomReacher()
        env = CustomReacher2()
        s = env.reset()
        print(s.shape)
        print(s[0].shape)
        rgb_list = []
        for i in count(1):
            # s, rgb, r, d, _ = env.step(env.action_space.sample())
            s, r, d, _ = env.step(env.action_space.sample())
            env.render()
            # rgb_list.append(rgb)
            if d:
                # watch(rgb_list)
                # is_video_same(rgb_list)
                # rgb_list=[]
                input('Done')
                s=env.reset()


if __name__ == '__main__':
    test()    # keyboard_test()
