''' Start with only singleplayer '''

from itertools import count
import time

from roboschool.scene_abstract import Scene
import gym
import os
import numpy as np


PATH_TO_CUSTOM_XML = "/home/erik/com_sci/Master_code/Roboschool"

class MyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
        }

    VIDEO_W = 600  # for video showing the robot, not for camera ON the robot
    VIDEO_H = 400
    def __init__(self,
                    XML_PATH=PATH_TO_CUSTOM_XML,
                    model_xml='Social_torso.xml',
                    robot_name='lwaist',
                    action_dim=13,
                    obs_dim=29,
                    electricity_cost = - 2.0,
                    stall_torque_cost = - 0.1,
                    joints_at_limit_cost = -0.2,
                    MAX_TIME = 1000,
                    ):

        self.scene = None
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)

        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self._seed()

        self.XML_PATH = XML_PATH
        self.model_xml = model_xml
        self.robot_name = robot_name

        # Scene
        self.gravity = 9.81
        self.timestep=0.0165/4
        self.frame_skip = 1

        self.power = 0.5

        # penalties/values used for calculating reward
        self.electricity_cost  = electricity_cost
        self.stall_torque_cost = stall_torque_cost
        self.joints_at_limit_cost = joints_at_limit_cost

        self.MAX_TIME = MAX_TIME

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        if self.scene is None:
            ''' First reset '''
            self.scene = Scene(self.gravity, self.timestep, self.frame_skip)

           # load xml files
        self.mjcf = self.scene.cpp_world.load_mjcf(
                            os.path.join(self.XML_PATH,
                                        self.model_xml))

        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        for r in self.mjcf:
            if dump: print("ROBOT '%s'" % r.root_part.name)
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
                if j.name[:6]=="ignore":
                    j.set_motor_torque(0)
                    continue
                j.power_coef = 100.0
                if "target" not in j.name:
                    self.ordered_joints.append(j)
                self.jdict[j.name] = j
        assert(self.cpp_robot)

        self.robot_specific_reset()
        for r in self.mjcf:
            r.query_position()

        s = self.calc_state()    # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        return s

    def robot_specific_reset(self):
        # target
        self.jdict["target0_x"].reset_current_position(self.np_random.uniform( low=-0.3, high=0.3 ), 0)
        self.jdict["target0_y"].reset_current_position(self.np_random.uniform( low=-0.3, high=0.3), 0)
        self.jdict["target0_z"].reset_current_position(self.np_random.uniform( low=0, high=0.3), 0)
        self.target = self.parts["target0"]

        self.key_parts  = [self.parts["left_hand"],
                            self.parts["right_hand"],
                            self.parts["left_elbow"],
                            self.parts["right_elbow"],
                            self.parts["left_shoulder"],
                            self.parts["right_shoulder"]]

        self.motor_names  = ["abdomen_z", "abdomen_y", "abdomen_x", "upper_abs"]
        self.motor_power  = [100, 100, 100, 100]
        self.motor_names += [ "neck_y", "neck_z", "neck_x"]
        self.motor_power += [25, 25, 25]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]

    def set_initial_orientation(self, yaw_center, yaw_random_spread):
        cpose = cpp_household.Pose()
        yaw = yaw_center + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)
        pitch = 0
        roll = 0
        #cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 1.4)
        cpose.set_xyz(0, 0, 0 )
        cpose.set_rpy(roll, pitch, yaw)
        self.cpp_robot.set_pose_and_speed(cpose, 0,0,0)
        self.initial_z = 0.0


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

        state = self.calc_state()  # also calculates self.joints_at_limit
        reward = 5

        self.frame  += 1
        # stop condition
        done = False
        if self.frame>=self.MAX_TIME:
            done = True

        self.done = done
        self.reward = reward
        return state, reward, bool(done), {}

    def calc_state(self):
        j = np.array([j.current_relative_position()
                      for j in self.ordered_joints],
                     dtype=np.float32).flatten()
        return j

    # --------------------
    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for i, m, power in zip(range(len(self.motors)), self.motors, self.motor_power):
            m.set_motor_torque( float(power*self.power*np.clip(a[i], -1, +1)) )

    def calc_potential(self):
        '''norm between hand and target. No other input than vector -> frobious norm
        2 p-norm, euclidean distance'''
        return -100*np.linalg.norm(self.to_target_vec)


def test():
        # Scene
        gravity = 9.81
        timestep=0.0165/4
        frame_skip = 1
        scene = Scene(gravity, timestep, frame_skip)

           # load xml files
        model_xml = 'Social_torso.xml'
        mjcf = scene.cpp_world.load_mjcf(os.path.join(PATH_TO_CUSTOM_XML, model_xml))

        ordered_joints = []
        jdict = {}
        parts = {}

        for r in mjcf:
            print("r.root_part: %s'" % r.root_part.name)

            # if r.root_part.name==robot_name:
            #     cpp_robot = r
            #     robot_body = r.root_part

        for r in mjcf:
            for part in r.parts:
                print("\t%s: '%s'" % (r.root_part.name, part.name))
                parts[part.name] = part

                # if part.name==robot_name:
                #     cpp_robot = r
                #     robot_body = part

        for r in mjcf:
            for j in r.joints:
                if dump: print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
                if j.name[:6]=="ignore":
                    j.set_motor_torque(0)
                    continue
                j.power_coef = 100.0
                if "target" not in j.name:
                    ordered_joints.append(j)
                jdict[j.name] = j


if __name__ == '__main__':
    test()
