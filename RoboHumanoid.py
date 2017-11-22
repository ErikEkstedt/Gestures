import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os

from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv

class RoboschoolMujocoXmlEnv(gym.Env):
    """
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

    def __init__(self, model_xml, robot_name, action_dim, obs_dim):
        self.scene = None

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self._seed()

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
        self.mjcf = self.scene.cpp_world.load_mjcf(os.path.join(os.path.dirname(__file__), "mujoco_assets", self.model_xml))
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


class RoboschoolForwardWalker(SharedMemoryClientEnv):
    def __init__(self, power):
        self.power = power
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.camera_x = 0
        self.camera_y = 4.3
        self.camera_z = 45.0
        self.camera_follow = 0

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)

    def robot_specific_reset(self):
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform( low=-0.1, high=0.1 ), 0)
        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
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
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array( [p.pose().xyz() for p in self.parts.values()] ).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        r, p, yaw = self.body_rpy
        if self.initial_z==None:
            self.initial_z = z
        self.walk_target_theta = np.arctan2( self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0] )
        self.walk_target_dist  = np.linalg.norm( [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]] )
        self.angle_to_target = self.walk_target_theta - yaw

        self.rot_minus_yaw = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw),  np.cos(-yaw), 0],
             [           0,             0, 1]]
            )
        vx, vy, vz = np.dot(self.rot_minus_yaw, self.robot_body.speed())  # rotate speed back to body point of view

        more = np.array([
            z-self.initial_z,
            np.sin(self.angle_to_target), np.cos(self.angle_to_target),
            0.3*vx, 0.3*vy, 0.3*vz,    # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r, p], dtype=np.float32)
        return np.clip( np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        return - self.walk_target_dist / self.scene.dt

    electricity_cost     = -2.0    # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost    = -0.1    # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost  = -1.0    # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.2    # discourage stuck joints

    def _step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.alive_bonus(state[0]+self.initial_z, self.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            #print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0
            if contact_names - self.foot_ground_object_names:
                feet_collision_cost += self.foot_collision_cost

        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        self.rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
            ]

        self.frame  += 1
        if (done and not self.done) or self.frame==self.spec.timestep_limit:
            self.episode_over(self.frame)
        self.done   += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)
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


class RoboschoolForwardWalkerMujocoXML(RoboschoolForwardWalker, RoboschoolMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        RoboschoolMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        RoboschoolForwardWalker.__init__(self, power)


class RoboschoolHumanoid(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["right_foot", "left_foot"]
    TASK_WALK, TASK_STAND_UP, TASK_ROLL_OVER, TASKS = range(4)

    def __init__(self, model_xml='humanoid_symmetric.xml'):
        RoboschoolForwardWalkerMujocoXML.__init__(self, model_xml, 'torso', action_dim=17, obs_dim=44, power=0.41)
        # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25
        self.electricity_cost  = 4.25*RoboschoolForwardWalkerMujocoXML.electricity_cost
        self.stall_torque_cost = 4.25*RoboschoolForwardWalkerMujocoXML.stall_torque_cost
        self.initial_z = 0.8

    def robot_specific_reset(self):
        RoboschoolForwardWalkerMujocoXML.robot_specific_reset(self)
        self.motor_names  = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power  = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]
        self.humanoid_task()

    def humanoid_task(self):
        self.set_initial_orientation(self.TASK_WALK, yaw_center=0, yaw_random_spread=np.pi/16)

    def set_initial_orientation(self, task, yaw_center, yaw_random_spread):
        self.task = task
        cpose = cpp_household.Pose()
        yaw = yaw_center + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)
        if task==self.TASK_WALK:
            pitch = 0
            roll = 0
            cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 1.4)
        elif task==self.TASK_STAND_UP:
            pitch = np.pi/2
            roll = 0
            cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 0.45)
        elif task==self.TASK_ROLL_OVER:
            pitch = np.pi*3/2 - 0.15
            roll = 0
            cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 0.22)
        else:
            assert False
        cpose.set_rpy(roll, pitch, yaw)
        self.cpp_robot.set_pose_and_speed(cpose, 0,0,0)
        self.initial_z = 0.8

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for i, m, power in zip(range(len(self.motors)), self.motors, self.motor_power):
            m.set_motor_torque( float(power*self.power*np.clip(a[i], -1, +1)) )

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


def test():
    import roboschool

    env = gym.make('RoboschoolHumanoid-v1')
    env.set_initial_orientation(task=1, yaw_center=0, yaw_random_spread=1)

    steps = 100

    s = envs.reset()
    for i in range(steps):
        env.render()
        a = envs.action_space.sample()
        s, r, d, i = envs.step([a]*num_processes)
        print(r)
        print(s[0].shape)


if __name__ == '__main__':
    test()
