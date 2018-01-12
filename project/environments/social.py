''' social movement environment (basically just a robot physics simulator)

Write documentation.

This will be the final environment script used.

Write how everything her works in connection to other important parts.

Reference original/inspirations
'''

from roboschool.scene_abstract import Scene, SingleRobotEmptyScene
import os
import numpy as np
import gym
from itertools import count
from OpenGL import GLE # fix for opengl issues on desktop  / nvidia
import cv2


PATH_TO_CUSTOM_XML = os.path.join(os.path.dirname(__file__), "xml_files")


class MyGymEnv(gym.Env):
    ''' OpenAI zGym wrapper

    functions:

        self._reset   : resets the environment (robot)
        self._step  : steps, returns s, st, o, ot, reward, done, info
        self._seed  : sets seed. self.np_random
        self._render  : r
    '''

    metadata = {
        'render.modes': ['human', 'machine', 'target', 'all', 'all_rgb_array'],
        'video.frames_per_second': 60
        }
    def __init__(self, action_dim=2, state_dim=7, obs_dim=(600, 400, 3)):
        self.scene = None
        self.VIDEO_W = obs_dim[0]
        self.VIDEO_H = obs_dim[1]

        self.Human_VIDEO_W = 600   # for human render
        self.Human_VIDEO_H = 400

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)

        high = np.inf*np.ones([state_dim])
        self.state_space = gym.spaces.Box(-high, high)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_dim)

        self.state_target = None
        self.obs_target = None
        if self.scene is None:
            ''' First reset '''
            self.scene = self.initialize_scene()
            # If load_xml_get_robot() is moved outside this condition after
            # env.reset all states become nan
            self.load_xml_get_robot()

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.get_joint_dicts()
        self.robot_specific_reset()
        for r in self.mjcf:
            r.query_position()

        # Important Resets
        self.done = False
        self.frame = 0
        self.reward = 0
        self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W,
                                                                 self.VIDEO_H,
                                                                 "video_camera")
        self.human_camera = self.scene.cpp_world.new_camera_free_float(self.Human_VIDEO_W,
                                                                       self.Human_VIDEO_H,
                                                                       "human_video_camera")

        if self.state_target is None:
            print('Random Targets. Use "env.set_target(state, obs)"')
            self.state_target = np.random.randint(4)
            self.obs_target = np.random.randint(0, 255, (100,100,3)).astype('uint8')

        state_robot = self.calc_state()  # pos and speed
        self.potential = self.calc_potential()  # potential to target
        obs = self.get_rgb()  #observation

        return (state_robot, self.state_target, obs, self.obs_target)

    def _step(self, a):
        self.apply_action(a)  # Singleplayer (originally in a condition)
        self.scene.global_step()
        self.frame  += 1

        state = self.calc_state()  # also calculates self.joints_at_limit
        reward = self.calc_reward(a)
        done = self.stop_condition() # max frame reached?
        self.done = done
        self.reward = reward

        obs = self.get_rgb()
        return (state, self.state_target, obs, self.obs_target, reward, bool(done), {})

    def _render(self, mode, close):
        def cv2_render(rgb, title='frame'):
            cv2.imshow(title, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stop')
                return
        if close:
            return
        if mode=='human':
            self.human_camera_adjust()
            rgb, _, _, _, _ = self.human_camera.render(False, False, False) # render_depth, render_labeling, print_timing)
            rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.Human_VIDEO_H, self.Human_VIDEO_W,3) )
            cv2_render(rendered_rgb, 'human')
        elif mode=="machine":
            self.camera_adjust()
            rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
            rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
            cv2_render(rendered_rgb, 'machine')
        elif mode=="target":
            cv2_render(self.obs_target, 'target')
        elif mode=='all':
            self._render('human', False)
            self._render('machine', False)
            self._render('target', False)
        elif mode=="all_rgb_array":
            self.camera_adjust()
            rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
            machine = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )

            self.human_camera_adjust()
            rgb, _, _, _, _ = self.human_camera.render(False, False, False) # render_depth, render_labeling, print_timing)
            human = np.fromstring(rgb, dtype=np.uint8).reshape( (self.Human_VIDEO_H, self.Human_VIDEO_W,3) )
            return human, machine, self.obs_target
        else:
            assert(0)


class Base(MyGymEnv):
    def __init__(self, XML_PATH=PATH_TO_CUSTOM_XML,
                 robot_name='robot',
                 model_xml='NOT/A/FILE.xml',
                 ac=2, st=6,
                 args=None):
        self.XML_PATH    = XML_PATH
        self.model_xml   = model_xml
        self.robot_name  = robot_name
        if args is None:
            ''' Defaults '''
            self.MAX_TIME             = 300
            self.potential_constant   = 100
            self.electricity_cost     = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
            self.stall_torque_cost    = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
            self.joints_at_limit_cost = -0.2  # discourage stuck joints

            self.reward_constant1     = 1
            self.reward_constant2     = 1

            # Scene
            self.gravity              = 9.81
            self.timestep             = 0.0165/4
            self.frame_skip           = 1

            # Robot
            self.power                = 0.5
            MyGymEnv.__init__(self, action_dim=ac, state_dim=st)
        else:
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
            MyGymEnv.__init__(self,
                              action_dim=ac,
                              state_dim=st,
                              obs_dim=(args.video_w, args.video_h, args.video_c))

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


class Social(Base):
    def __init__(self, args=None):
        Base.__init__(self, XML_PATH=PATH_TO_CUSTOM_XML,
                      robot_name='robot_arm',
                      model_xml='reacher/ReacherPlaneNoTarget.xml',
                      ac=2, st=6, args=args)
        print('I am', self.model_xml)

    def set_target(self, state_target=None, obs_target=None):
        ''' targets should be numpy.ndarray. obs.shape (W,H,C)'''
        assert type(state_target) is np.ndarray, 'target must be numpy'
        self.state_target = state_target
        self.obs_target = obs_target

    def robot_specific_reset(self):
        self.motor_names = ["robot_shoulder_joint_z", "robot_elbow_joint"]
        self.motor_power = [100, 100]
        self.motors = [self.jdict[n] for n in self.motor_names]

        self.robot_reset()
        self.calc_robot_keypoints()

    def robot_reset(self):
        ''' self.np_random for correct seed. '''
        for j in self.robot_joints.values():
            j.reset_current_position(self.np_random.uniform(low=-0.01, high=0.01 ), 0)
            j.set_motor_torque(0)

    def calc_robot_keypoints(self):
        ''' gets hand position, target position and the vector in bewteen'''
        elbow_position = np.array(self.parts['robot_elbow'].pose().xyz())[:2]
        hand_position = np.array(self.parts['robot_hand'].pose().xyz())[:2]
        self.robot_key_points = np.concatenate((elbow_position, hand_position))

    def calc_reward(self, a):
        ''' Difference potential as reward '''
        potential_old = self.potential
        self.potential = self.calc_potential()
        r = self.reward_constant1 * float(self.potential - potential_old) # elbow
        return r

    def calc_potential(self):
        self.diff_key_points = self.state_target - self.robot_key_points
        p = -self.potential_constant*np.linalg.norm(self.diff_key_points)
        return np.array(p)

    def calc_state(self):
        j = np.array([j.current_relative_position()
                      for j in self.robot_joints.values()],
                     dtype=np.float32).flatten()
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.joint_speeds = j[1::2]
        self.calc_robot_keypoints()  # calcs target_position, important_pos, to_target_vec
        return np.concatenate((self.robot_key_points, self.joint_speeds))

    def get_rgb(self):
        self.camera_adjust()
        rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        return rendered_rgb

    def camera_adjust(self):
        ''' Vision from straight above '''
        self.camera.move_and_look_at( 0, 0, 1, 0, 0, 0.4)

    def human_camera_adjust(self):
        ''' Vision from straight above '''
        self.human_camera.move_and_look_at( 0, 0, 1, 0, 0, 0.4)


# Test functions
def random_run(env, render=True):
    ''' Executes random actions and renders '''
    s, s_, o, o_ = env.reset()
    while True:
        action = env.action_space.sample()
        s, s_, o, o_, r, d, _ = env.step(action)
        if render:
            env.render('all')
            # env.render()  # same as env.render('human')
            # env.render('machine')
            # env.render('target')
            # h, m, t = env.render('all_rgb_array')  # returns rgb arrays
            # print(h.shape)
            # print(m.shape)
            # print(t.shape)
        if d:
            s, s_, o, o_ = env.reset()

def random_run_with_changing_targets(env, dset, args, verbose=False):
    ''' Executes random actions and also changes the target.
    targets are set in order from a project.data.dataset.
    renders options:
        'render.modes': ['human', 'machine', 'target', 'all', 'all_rgb_array']
    '''
    t = 0
    while True:
        ob_target, st_target = dset[t]; t += 1
        env.set_target(st_target.numpy(), ob_target.numpy().transpose((1,2,0)))

        state, s_target, obs, o_target = env.reset()
        episode_reward = 0
        for j in count(1):
            if args.render:
                env.render('all')

            # update the target
            if j % args.update_target == 0:
                ob_target, st_target = dset[t]
                env.set_target(st_target.numpy(), ob_target.numpy().transpose((1,2,0)))
                t += 1

            # Observe reward and next state
            actions = env.action_space.sample()
            state, s_target, obs, o_target, reward, done, info = env.step(actions)

            # If done then update final rewards and reset episode reward
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                break

def project_test():
    from project.utils.arguments import get_args
    from project.data.dataset import load_reacherplane_data
    args = get_args()

    # === Targets ===
    print('\nLoading targets from:')
    print('path:\t', args.target_path)
    dset, dloader = load_reacherplane_data(args.target_path, shuffle=False)

    env = Social(args)
    env.seed(args.seed)

    # random_run(env, render=args.render)
    random_run_with_changing_targets(env, dset, args)

def test_env():
    '''Works'''
    env = Social()
    env.seed(4)
    random_run(env, render=True)

if __name__ == '__main__':
    # test_env()
    project_test()
