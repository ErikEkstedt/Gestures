from OpenGL import GLU # fix for opengl issues on desktop  / nvidia
from roboschool.scene_abstract import Scene
import os
import numpy as np

try:
    from gym_env import MyGymEnv, make_parallel_environments
except:
    from environments.gym_env import MyGymEnv, make_parallel_environments

PATH_TO_CUSTOM_XML = "/home/erik/com_sci/Master_code/Project/environments/xml_files"

class BasePyBullet(MyGymEnv):
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
            m.set_motor_torque( float(power*self.power*np.clip(a[i], -1, +1)) )

    def stop_condition(self):
        max_time = False
        if self.frame>=self.MAX_TIME:
            max_time = True
        return max_time

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()

        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]

        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.calc_to_target_vec()

        a = np.concatenate((self.target_position,
                            self.to_target_vec,
                            self.joint_positions,
                            self.joint_speeds),)
        # print(self.joint_positions.shape)
        # print(self.joint_speeds.shape)
        return a

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.robot_parts['robot_left_hand'].pose().xyz())
        self.to_target_vec = self.hand_position - self.target_position

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
                if verbose: print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
                j.power_coef = 100.0
                self.ordered_joints.append(j)
                self.jdict[j.name] = j

        # sort out robot and targets.
        self.target_joints,  self.target_parts = self.get_joints_parts_by_name('target')
        self.robot_joints,  self.robot_parts = self.get_joints_parts_by_name('robot')

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

    def robot_specific_reset(self):
        # self.motor_names = ["robot_abd_x"]
        # self.motor_power = [100] #, 75, 75]
        # self.motors = [self.jdict[n] for n in self.motor_names]

        self.motor_names = list(self.robot_joints.keys())
        self.motor_power = [100 for _ in range(len(self.motor_names))]
        self.motors = list(self.robot_joints.values())

        # target and potential
        self.target_reset()
        self.calc_to_target_vec()
        self.potential = self.calc_potential()

    def target_reset(self):
        ''' self.np_random for correct seed. '''
        for j in self.target_joints.values():
            if "z" in j.name:
                '''Above ground'''
                j.reset_current_position(
                    self.np_random.uniform( low=0, high=0.2 ), 0)
            else:
                j.reset_current_position( self.np_random.uniform( low=0.1, high=0.3 ), 0)


class CustomReacher(BasePyBullet):
    def __init__(self, gravity=9.81):
        BasePyBullet.__init__(self, path=PATH_TO_CUSTOM_XML,
                              robot_name='robot_arm',
                              target_name='target',
                              model_xml='custom_reacher.xml',
                              ac=2, obs=13, gravity=gravity)
    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint", "robot_elbow_joint"] # , "right_shoulder2", "right_elbow"]
        self.motor_power = [75, 75] #, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]

        # target and potential
        self.target_reset()
        self.calc_to_target_vec()
        self.potential = self.calc_potential()

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

        # print('Target: {}, {}, {}'.format(target_x, target_y, target_z))
        # print('Hand: {}'.format(self.hand_position))
        # print('Vector: ', self.to_target_vec)
        # print('Potential: ', self.calc_potential())
        # print('state: ', reacher)

        # return np.concatenate((self.target_position,
        #                       self.joint_positions,
        #                       self.joint_speeds), )
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


class CustomReacher2d(BasePyBullet):
    def __init__(self, gravity=9.81):
        BasePyBullet.__init__(self, path=PATH_TO_CUSTOM_XML,
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


class HalfHumanoid(BasePyBullet):
    def __init__(self, gravity=9.81):
        BasePyBullet.__init__(self, path=PATH_TO_CUSTOM_XML,
                              robot_name='robot',
                              target_name='target',
                              model_xml='half_humanoid.xml',
                              ac=6, obs=18, gravity=gravity)


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


def test():
    from itertools import count
    import time

    multiple_procs = input('4 (P)arallel processes or (S)ingle? (P/S)')

    if 'P' in multiple_procs:
        ''' multiple processes '''
        num_processes = 4
        env = make_parallel_environments(HalfHumanoid, 10, num_processes)
        asize = env.action_space.shape[0]
        s = env.reset()
        for i in range(300):
            s, r, d, _ = env.step(num_processes * [np.random.rand(asize)*2-1])
            print(s.shape)
    else:
        ''' single '''
        env = HalfHumanoid()
        # env = CustomReacher()
        asize = env.action_space.shape[0]
        s = env.reset()
        print(s.shape)
        for j in count(1):
            for i in count(1):
                env.render()
                s, r, d, _ = env.step(np.random.rand(asize)*2-1 )
                print(r)
                if d:
                    print('done')
                    time.sleep(2)
                    s=env.reset()


if __name__ == '__main__':
    test()
