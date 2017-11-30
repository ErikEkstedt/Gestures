''' Start with only singleplayer '''

from itertools import count
import time

from roboschool.scene_abstract import cpp_household
from roboschool.scene_abstract import Scene
import gym
import os
import numpy as np

PATH_TO_CUSTOM_XML = "/home/erik/com_sci/Master_code/Roboschool/xml_files"

class MyGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
        }

    VIDEO_W = 600  # for video showing the robot, not for camera ON the robot
    VIDEO_H = 400

    def __init__(self, action_dim=13, obs_dim=29,):
        self.scene = None
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)

        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        if self.scene is None:
            ''' First reset '''
            self.scene = self.initialize_scene()
            self.load_xml_get_robot()

        self.robot_specific_reset()
        # for r in self.mjcf:
        #     r.query_position()

        s = self.calc_state()
        self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        return s

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
        self.apply_action(a)  # Singleplayer (originally in a condition)
        self.scene.global_step()
        self.frame  += 1

        state = self.calc_state()  # also calculates self.joints_at_limit
        reward = self.calc_reward(a)
        done = self.stop_condition()

        self.done = done
        return state, reward, bool(done), {}

class CustomReacher(MyGymEnv):
    def __init__(self, path=PATH_TO_CUSTOM_XML,
                 robot_name='robot_arm',
                 target_name='target',
                 model_xml='custom_reacher.xml'):
        MyGymEnv.__init__(self, action_dim=1, obs_dim=29)
        self.XML_PATH = path
        self.model_xml = model_xml
        self.robot_name = robot_name
        self.target_name = target_name

        # Scene
        self.gravity = 9.81
        self.timestep=0.0165/4
        self.frame_skip = 1

        self.power = 0.5

        # # penalties/values used for calculating reward
        self.potential_constant = 10
        self.electricity_cost  = -0.1
        self.stall_torque_cost = -0.01
        self.joints_at_limit_cost = -0.01

        self.MAX_TIME = 1000

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for i, m, power in zip(range(len(self.motors)), self.motors, self.motor_power):
            m.set_motor_torque( float(power*self.power*np.clip(a[i], -1, +1)) )

    def stop_condition(self):
        done = False
        if self.frame>=self.MAX_TIME:
            done = True
        return done

    def calc_state(self):
        j = np.array([j.current_relative_position()
                    for j in self.robot_joints.values()],
                    dtype=np.float32).flatten()

        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]

        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.calc_to_target_vec()

        return np.concatenate((self.target_position,
                              self.joint_positions,
                              self.joint_speeds))

    def calc_to_target_vec(self):
        ''' gets hand position, target position and the vector in bewteen'''
        self.target_position = np.array(self.target_parts['target'].pose().xyz())
        self.hand_position = np.array(self.parts['robot_hand'].pose().xyz())
        self.to_target_vec = self.hand_position + self.target_position

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
        return self.potential_constant*np.linalg.norm(self.to_target_vec)

    def initialize_scene(self):
        return Scene(self.gravity, self.timestep, self.frame_skip)

    def load_xml_get_robot(self):
        self.mjcf = self.scene.cpp_world.load_mjcf( os.path.join(self.XML_PATH, self.model_xml))
        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 1
        for r in self.mjcf:
            if dump: print("ROBOT '%s'" % r.root_part.name)
            # store important parts
            if r.root_part.name==self.robot_name:
                self.cpp_robot = r
                self.robot_body = r.root_part

            for part in r.parts:
                if dump: print("\tPART '%s'" % part.name)
                self.parts[part.name] = part
                if part.name==self.robot_name:
                    self.cpp_robot = r
                    self.robot_body = part

            for j in r.joints:
                if dump: print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
                j.power_coef = 100.0
                self.ordered_joints.append(j)
                self.jdict[j.name] = j

        # sort out robot and targets.
        self.target_joints,  self.target_parts = self.get_joints_parts_by_name('target')
        self.robot_joints,  self.robot_parts = self.get_joints_parts_by_name('robot')
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
        self.motor_names = ["robot_shoulder_joint", "robot_elbow_joint"] # , "right_shoulder2", "right_elbow"]
        self.motor_power = [100, 100] #, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]

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
                    self.np_random.uniform( low=0, high=0.3 ), 0)
            else:
                j.reset_current_position(
                    self.np_random.uniform( low=-0.3, high=0.3 ), 0)


def make_parallel_customReacher(seed, num_processes):
    ''' imports SubprocVecEnv from baselines.
    :param seed                 int
    :param num_processes        int, # env
    '''
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    def multiple_envs(Env, seed, rank):
        def _thunk():
            env = CustomReacher()
            env.seed(seed + rank)
            return env
        return _thunk
    return SubprocVecEnv([multiple_envs(CustomReacher,seed, i) for i in range(num_processes)])

def test():
    # num_processes = 4
    # env = make_parallel_customReacher(10, num_processes)
    env = CustomReacher()
    print(env.reset())
    for i in range(300):
        env.render()
        s, r, d, _ = env.step(np.random.rand(2)*2-1)
        print(r)

if __name__ == '__main__':
    test()
