import gym
import numpy as np

PATH_TO_CUSTOM_XML = "/home/erik/com_sci/Master_code/Roboschool/xml_files"

class MyGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
        }

    VIDEO_W = 600  # for video showing the robot, not for camera ON the robot
    VIDEO_H = 400

    def __init__(self, action_dim=2, obs_dim=7,):
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

        for r in self.mjcf:
            r.query_position()

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


