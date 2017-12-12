from roboschool.scene_abstract import Scene
import os
import numpy as np
import gym
from OpenGL import GL # fix for opengl issues on desktop  / nvidia

try:
    from environments.gym_env import MyGymEnv
except:
    from gym_env import MyGymEnv

PATH_TO_CUSTOM_XML = "/home/erik/com_sci/Master_code/Project/environments/xml_files"


class Base(MyGymEnv):
    def __init__(self, XML_PATH =PATH_TO_CUSTOM_XML,
                 robot_name='robot',
                 target_name='target',
                 model_xml='NOT/A/FILE.xml',
                 ac=6, obs=18, gravity=9.81, RGB=False, episode_time=300):
        MyGymEnv.__init__(self, action_dim=ac, obs_dim=obs, RGB=RGB)
        self.XML_PATH = XML_PATH
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

        self.MAX_TIME = episode_time

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
        self.mjcf = self.scene.cpp_world.load_mjcf( os.path.join(self.XML_PATH, self.model_xml))
        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
        for r in self.mjcf:
            if verbose:
                print('Load XML Model')
                print('Path:',os.path.join(self.XML_PATH, self.model_xml))
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
        self.camera.move_and_look_at(0.5, 0.5, 0.5, 0, 0, 0)


class Reacher_plane(Base):
    ''' 2DoF Reacher in a plane
    No joint limits
    1 DoF each joint (only z-axis)
    target random in 3D space, not every point is
    '''
    def __init__(self, potential_constant=100,
                 electricity_cost=-0.1,
                 stall_torque_cost=-0.01,
                 joints_at_limit_cost=-0.01,
                 gravity=9.81, RGB=False):
        Base.__init__(self, XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot_arm',
                        target_name='target',
                        model_xml='reacher_plane.xml',
                        ac=2, obs=13, gravity=gravity, RGB=RGB)


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
                j.reset_current_position( self.np_random.uniform( low=-0.2, high=0.2 ), 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_hand'].pose().xyz())
        self.to_target_vec = self.hand_position - self.target_position

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()], dtype=np.float32).flatten()
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

        # Save rewards/they do in roboschool, don't know why.
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        return sum(self.rewards)

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)

    def get_rgb(self):
        rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        return rendered_rgb


class CustomReacher2DoF(Base):
    ''' 2DoF Reacher
    No joint limits
    1 DoF each joint
    target random in 3D space, not every point is
    '''
    def __init__(self, potential_constant=100,
                 electricity_cost=-0.1,
                 stall_torque_cost=-0.01,
                 joints_at_limit_cost=-0.01,
                 gravity=9.81, RGB=False):
        Base.__init__(self, XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot_arm',
                        target_name='target',
                        model_xml='custom_reacher2DoF.xml',
                        ac=2, obs=13, gravity=gravity, RGB=RGB)


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
                    self.np_random.uniform( low=0.2, high=0.6 ), 0)
            else:
                j.reset_current_position( self.np_random.uniform( low=-0.3, high=0.3 ), 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_hand'].pose().xyz())
        self.to_target_vec = self.hand_position - self.target_position

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()], dtype=np.float32).flatten()
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

        # Save rewards/they do in roboschool, don't know why.
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        return sum(self.rewards)

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)

    def get_rgb(self):
        rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        return rendered_rgb


class CustomReacher3DoF(Base):
    ''' 2DoF Reacher
    No joint limits
    1 DoF each joint
    target random in 3D space, not every point is
    '''
    def __init__(self, potential_constant=100,
                 electricity_cost=-0.1,
                 stall_torque_cost=-0.01,
                 joints_at_limit_cost=-0.01,
                 gravity=9.81, RGB=False):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot_arm',
                        target_name='target',
                        model_xml='custom_reacher3DoF.xml',
                        ac=3, obs=15, gravity=gravity, RGB=RGB)

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint", "robot_elbow_joint_x", "robot_elbow_joint_y", ]
        self.motor_power = [75, 75, 75] #, 75, 75]
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
                    self.np_random.uniform( low=0.2, high=0.6 ), 0)
            else:
                j.reset_current_position( self.np_random.uniform( low=-0.3, high=0.3 ), 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_hand'].pose().xyz())
        self.to_target_vec = self.hand_position - self.target_position

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()], dtype=np.float32).flatten()
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
        ''' Calcutates reward
        :param a      np.ndarray action
        '''
        potential_old = self.potential
        self.potential = self.calc_potential()

        # cost
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        # Save rewards ?
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        return sum(self.rewards)

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)

    def get_rgb(self):
        rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        return rendered_rgb


class CustomReacher2DoF_2Target(Base):
    ''' 2DoF Reacher
    No joint limits
    1 DoF each joint
    Two targets
    '''
    def __init__(self, potential_constant=100,
                 electricity_cost=-0.1,
                 stall_torque_cost=-0.01,
                 joints_at_limit_cost=-0.01,
                 gravity=9.81, RGB=False):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                        robot_name='robot_arm',
                        target_name='target_elbow',
                        model_xml='custom_modeling.xml',
                        ac=3, obs=13, gravity=gravity, RGB=RGB)

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint", "robot_elbow_joint_x","robot_elbow_joint_y"] # , "right_shoulder2", "right_elbow"]
        self.motor_power = [75, 75, 75] #, 75, 75]
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
        r=0.2
        theta= 3.14 * self.np_random.rand()
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = 0.41

        r=0.1
        theta= 3.14 * self.np_random.rand()
        x1 = x + r*np.cos(theta)
        y1 = y + r*np.sin(theta)
        z1 = z + r*self.np_random.rand()*2-1
        for name, j in self.target_joints.items():
            if "0" in name:
                if "z" in name:
                    j.reset_current_position(0.41, 0)
                    print('z0')
                    print(name)
                elif "x" in name:
                    print('x0')
                    print(name)
                    j.reset_current_position(x,0)
                else:
                    print('y0')
                    print(name)
                    j.reset_current_position(y, 0)
            else:
                if "z" in name:
                    print('z1')
                    print(name)
                    j.reset_current_position(0.21, 0)
                elif "x" in name:
                    print('x1')
                    print(name)
                    j.reset_current_position(x1,0)
                else:
                    print('y1')
                    print(name)
                    j.reset_current_position(y1, 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        # Elbow target
        target_position1 = np.array(self.target_parts['target_elbow'].pose().xyz())
        elbow_position = np.array(self.parts['robot_arm1'].pose().xyz())
        totarget1 = elbow_position - target_position1

        # Hand target
        target_position2 = np.array(self.target_parts['target_hand'].pose().xyz())
        hand_position = np.array(self.parts['robot_hand'].pose().xyz())
        totarget2 = hand_position - target_position2

        self.target_position = np.concatenate((target_position1, target_position2))
        self.important_positions = np.concatenate((elbow_position, hand_position))
        self.to_target_vec = np.concatenate((totarget1, totarget2),)

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]
        self.calc_to_target_vec()
        return np.concatenate((self.target_position,
                               self.important_positions,
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
        return sum(self.rewards)

    def calc_potential(self):
        return -self.potential_constant*np.linalg.norm(self.to_target_vec)

    def get_rgb(self):
        rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        return rendered_rgb


class CustomReacher6DoF(Base):
    ''' 6DoF Reacher
    No joint limits
    3 DoF each joint
    target random in reachable 3D space
    '''
    def __init__(self,
                 potential_constant=100,
                 electricity_cost=-0.1,
                 stall_torque_cost=-0.01,
                 joints_at_limit_cost=-0.01,
                 episode_time=300,
                 gravity=9.81):
        Base.__init__(self,XML_PATH=PATH_TO_CUSTOM_XML,
                              robot_name='robot_arm',
                              target_name='target_arm',
                              model_xml='custom_reacher6DoF.xml',
                              ac=6, obs=21, gravity=gravity)

        self.potential_constant = potential_constant
        self.electricity_cost = electricity_cost
        self.stall_torque_cost = stall_torque_cost
        self.joints_at_limit_cost = joints_at_limit_cost

        self.MAX_TIME = episode_time

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint_x", "robot_elbow_joint_x"] # , "right_shoulder2", "right_elbow"]
        self.motor_names += ["robot_shoulder_joint_y", "robot_elbow_joint_y"] # , "right_shoulder2", "right_elbow"]
        self.motor_names += ["robot_shoulder_joint_z", "robot_elbow_joint_z"] # , "right_shoulder2", "right_elbow"]
        self.motor_power = [25, 50] #, 75, 75]
        self.motor_power += [25, 50] #, 75, 75]
        self.motor_power += [25, 50] #, 75, 75]
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
            else:
                j.reset_current_position( self.np_random.uniform( low=-0.3, high=0.3 ), 0)

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_hand'].pose().xyz())
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


def make_parallel_environments(Env, seed, num_processes,
                               potential_constant=100,
                               electricity_cost=-0.1,
                               stall_torque_cost=-0.01,
                               joints_at_limit_cost=-0.01,
                               episode_time=300):
    ''' imports SubprocVecEnv from baselines.
    :param seed                 int
    :param num_processes        int, # env
    '''
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    def multiple_envs(Env, seed, rank):
        def _thunk():
            env = Env(potential_constant,
                      electricity_cost,
                      stall_torque_cost,
                      joints_at_limit_cost,
                      episode_time)
            env.seed(seed+rank*1000)
            return env
        return _thunk
    return SubprocVecEnv([multiple_envs(Env,seed, i) for i in range(num_processes)])


def test():
    import sys

    dof = int(sys.argv[1])
    if dof == 2:
        env = CustomReacher2DoF()
        print('CustomReacher2DoF2')
    elif dof == 3:
        env = CustomReacher3DoF()
        print('CustomReacher3DoF')
    elif dof == 6:
        env = CustomReacher6DoF()
        print('CustomReacher6DoF')
    elif dof == 88:
        env = Reacher_plane()
        print('Reacher_plane')
    elif dof == 99:
        nproc=4
        env = make_parallel_environments(CustomReacher2DoF, 10, nproc, 200, -10, -1, -1, 200)
        print('Parallel CustomReacher')

    if dof == 99:
        s = env.reset()
        while True:
            s, r, d, _ = env.step([env.action_space.sample()] * nproc)
            if sum(d) > 0:
                print(env.reset())
    else:
        s = env.reset()
        print(s.shape)
        while True:
            s, r, d, _ = env.step(env.action_space.sample())
            # print(r)
            env.render()
            if d:
                s=env.reset()
                print(env.target_position)

if __name__ == '__main__':
    test()
