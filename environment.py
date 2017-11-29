from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.scene_abstract import SingleRobotEmptyScene

from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv

from roboschool.multiplayer import SharedMemoryClientEnv

import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys


PATH_TO_XML = "/home/erik/anaconda3/envs/robo/lib/python3.5/site-packages/roboschool/roboschool/mujoco_assets"
PATH_TO_CUSTOM_XML = "/home/erik/com_sci/Master_code/Roboschool"

class Shared_Mem(SharedMemoryClientEnv):
    def __init__(self, power):
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.camera_x = 0
        self.camera_y = 4.3
        self.camera_z = 45.0
        self.camera_follow = 0

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)
        # return SingleRobotEmptyScene(gravity=0, timestep=0.0165/4, frame_skip=4)

    def robot_specific_reset(self):
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform( low=-0.1, high=0.1 ), 0)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)
        self.start_pos_x, self.start_pos_y, self.start_pos_z = init_x, init_y, init_z

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for n,j in enumerate(self.ordered_joints):
            j.set_motor_torque( self.power*j.power_coef*float(np.clip(a[n], -1, +1)) )

    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_positions = j[0::2]
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        # Target
        target_x, _ = self.jdict["target0_x"].current_position()
        target_y, _ = self.jdict["target0_y"].current_position()
        target_z, _ = self.jdict["target0_z"].current_position()

        # print('target_x' , target_x)
        # print('target_y' , target_y)
        # print('target_z' , target_z)

        hand_coords = np.array(self.key_parts[0].pose().xyz())
        target_coords = np.array(self.target.pose().xyz())
        self.to_target_vec = np.array(hand_coords - target_coords)

        return np.clip( np.concatenate([self.joint_positions]+\
                                       [self.joint_speeds]+\
                                       [self.to_target_vec]+\
                                       np.array((target_x, target_y, target_z)),
                                       ), -5, +5)

    # def calc_state(self):
    #     j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
    #     # even elements [0::2] position, scaled to -1..+1 between limits
    #     # odd elements  [1::2] angular speed, scaled to show -1..+1
    #     self.joint_speeds = j[1::2]
    #     self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
    #
    #     body_pose = self.robot_body.pose()
    #     parts_xyz = np.array( [p.pose().xyz() for p in self.parts.values()] ).flatten()
    #     self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
    #     self.body_rpy = body_pose.rpy()
    #     z = self.body_xyz[2]
    #     r, p, yaw = self.body_rpy
    #     if self.initial_z==None:
    #         self.initial_z = z
    #
    #     self.rot_minus_yaw = np.array(
    #         [[np.cos(-yaw), -np.sin(-yaw), 0],
    #         [np.sin(-yaw),  np.cos(-yaw), 0],
    #         [           0,             0, 1]]
    #         )
    #     vx, vy, vz = np.dot(self.rot_minus_yaw, self.robot_body.speed())  # rotate speed back to body point of view
    #     j = [j]
    #     j.append(vx)
    #     j.append(vy)
    #     j.append(vz)
    #
    #     # return np.clip( np.concatenate([j]), -5, +5)
    #     return j

    def calc_potential(self):
        '''norm between hand and target. No other input than vector -> frobious norm
        2 p-norm, euclidean distance'''
        return -100*np.linalg.norm(self.to_target_vec)

    electricity_cost     = -2.0    # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost    = -0.1    # cost for running electric current through a motor even at zero rotational speed, small
    joints_at_limit_cost = -0.2    # discourage stuck joints

    def _step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        potential_old = self.potential
        self.potential = self.calc_potential()
        progress = float(self.potential - potential_old)

        # electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        # electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        # joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)
        # self.rewards = [ electricity_cost, joints_at_limit_cost]

        self.rewards = [float(self.potential - potential_old)]
        self.frame  += 1
        done = False
        if abs(self.potential) < self.precision:
            done = True
            print('#'*50)
            print('Precision under 1 !!!!')
        if self.frame>=self.MAX_TIME:
            done = True
        if (done and not self.done) or self.frame>=self.MAX_TIME:
            self.episode_over(self.frame)
        self.done   += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        return state, sum(self.rewards), bool(done), {}


    def episode_over(self, frames):
        pass

    def camera_adjust(self):
        #self.camera_dramatic()
        self.camera_simple_follow()

    def camera_simple_follow(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)

    def camera_dramatic(self):
        pose = self.robot_body.pose()
        speed = self.robot_body.speed()
        x, y, z = pose.xyz()
        if 1:
            camx, camy, camz = speed[0], speed[1], 2.2
        else:
            camx, camy, camz = self.walk_target_x - x, self.walk_target_y - y, 2.2

        n = np.linalg.norm([camx, camy])
        if n > 2.0 and self.frame > 50:
            self.camera_follow = 1
        if n < 0.5:
            self.camera_follow = 0
        if self.camera_follow:
            camx /= 0.1 + n
            camx *= 2.2
            camy /= 0.1 + n
            camy *= 2.8
            if self.frame < 1000:
                camx *= -1
                camy *= -1
            camx += x
            camy += y
            camz  = 1.8
        else:
            camx = x
            camy = y + 4.3
            camz = 2.2
        #print("%05i" % self.frame, self.camera_follow, camy)
        smoothness = 0.97
        self.camera_x = smoothness*self.camera_x + (1-smoothness)*camx
        self.camera_y = smoothness*self.camera_y + (1-smoothness)*camy
        self.camera_z = smoothness*self.camera_z + (1-smoothness)*camz
        self.camera.move_and_look_at(self.camera_x, self.camera_y, self.camera_z, x, y, 0.6)


class GYM_XML(gym.Env):
    """  RoboschoolMujocoXmlEnv
    Base class for MuJoCo .xml actors in a Scene.
    These environments create single-player scenes and behave like normal Gym environments, if
    you don't use multiplayer.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
        }

    VIDEO_W = 600  # for video showing the robot, not for camera ON the robot
    VIDEO_H = 400

    def __init__(self, XML_PATH, model_xml, robot_name, action_dim, obs_dim):
        self.scene = None

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self._seed()

        self.XML_PATH = XML_PATH
        self.model_xml = model_xml
        self.robot_name = robot_name

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        if self.scene is None:
            self.scene = self.create_single_player_scene()
        if not self.scene.multiplayer:
            self.scene.episode_restart()
        self.mjcf = self.scene.cpp_world.load_mjcf(os.path.join(self.XML_PATH, self.model_xml))
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
        self.potential = self.calc_potential()
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

    def calc_potential(self):
        return 0

    def HUD(self, s, a, done):
        ''' Visualization on top of rendering, might be usefull for something
        in the future but not important now'''
        active = self.scene.actor_is_active(self)
        if active and self.done<=2:
            self.scene.cpp_world.test_window_history_advance()
            self.scene.cpp_world.test_window_observations(s.tolist())
            self.scene.cpp_world.test_window_actions(a.tolist())
            self.scene.cpp_world.test_window_rewards(self.rewards)
        if self.done<=1: # Only post score on first time done flag is seen, keep this score if user continues to use env
            s = "%04i %07.1f %s" % (self.frame, self.reward, "DONE" if self.done else "")
            if active:
                self.scene.cpp_world.test_window_score(s)
            #self.camera.test_window_score(s)  # will appear on video ("rgb_array"), but not on cameras istalled on the robot (because that would be different camera)

class GYM_XML_MEM(Shared_Mem, GYM_XML):
    def __init__(self, path, model_xml, robot_name, action_dim, obs_dim, power):
        GYM_XML.__init__(self, path, model_xml, robot_name, action_dim, obs_dim)
        Shared_Mem.__init__(self, power)


class Social_Torso(GYM_XML_MEM):
    def __init__(self, path=PATH_TO_CUSTOM_XML, robot_name='lwaist', model_xml='Social_torso.xml', precision=5):
        GYM_XML_MEM.__init__(self, path=path, model_xml=model_xml, robot_name=robot_name, action_dim=13, obs_dim=29, power=0.41)
        self.precision = precision
        # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25
        self.electricity_cost  = 4.25*GYM_XML_MEM.electricity_cost
        self.stall_torque_cost = 4.25*GYM_XML_MEM.stall_torque_cost
        # self.initial_z = 0.8
        self.MAX_TIME = 10000

    def robot_specific_reset(self):
        GYM_XML_MEM.robot_specific_reset(self)
        self.target = self.parts["target0"]
        self.switch_target()

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
        self.humanoid_task()

    def humanoid_task(self):
        self.set_initial_orientation(yaw_center=0, yaw_random_spread=np.pi/16)

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

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for i, m, power in zip(range(len(self.motors)), self.motors, self.motor_power):
            m.set_motor_torque( float(power*self.power*np.clip(a[i], -1, +1)) )

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

    def get_mjcf(self):
        return self.mjcf

    def get_j(self):
        ''' dict containing the {motor_names : motors}'''
        return self.jdict

    def get_scene(self):
        return self.scene

    def get_world(self):
        return self.scene.cpp_world

    def get_motor_names(self):
        return self.motor_names

    def get_key_positions(self):
        ''' returns the x, y, z positions of key body parts'''
        coords = []
        for part in self.key_parts:
            coords.append(part.pose().xyz())
        return coords

    def switch_target(self):
        '''
        self.jdict["target1_x"].reset_current_position(self.np_random.uniform( low=-1, high=1 ), 0)
        self.jdict["target1_y"].reset_current_position(self.np_random.uniform( low=-1, high=1), 0)
        self.jdict["target1_z"].reset_current_position(self.np_random.uniform( low=0, high=1), 0)
        self.jdict["target2_x"].reset_current_position(self.np_random.uniform( low=-1, high=1 ), 0)
        self.jdict["target2_y"].reset_current_position(self.np_random.uniform( low=-1, high=1), 0)
        self.jdict["target2_z"].reset_current_position(self.np_random.uniform( low=0, high=1), 0)
        x, y, z = 0.2, 0.2, 0.2
        for name, part in self.parts.items():
            if "target" in name:
                print(name)
                part.pose().move_xyz(x,y,z)
        '''
        self.jdict["target0_x"].reset_current_position(self.np_random.uniform( low=-0.3, high=0.3 ), 0)
        self.jdict["target0_y"].reset_current_position(self.np_random.uniform( low=-0.3, high=0.3), 0)
        self.jdict["target0_z"].reset_current_position(self.np_random.uniform( low=0, high=0.3), 0)

    def set_precision(self, p):
        self.set_precision = p

def test():
    from environment import Social_Torso
    import numpy as np

    env = Social_Torso()
    asize = env.action_space.shape[0]
    s = env.reset()
    print(len(s))
    print(s)
    input()

    def random_action(idx, size):
        z = np.zeros(size)
        if type(idx) is int:
            z[idx] = np.random.rand()*2 - 1
        else:
            z[idx] = np.random.rand(len(idx))*2 - 1
        return z

    alls = list(np.arange(asize))
    print(alls)
    for i in range(8000):
        env.render()
        a = random_action(alls, asize)
        s, r, d, _ = env.step(a)

        if i % 400 == 0 and i is not 0:
            env.switch_target()

if __name__ == '__main__':
    test()
