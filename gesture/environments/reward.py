'''
Reward Funcs
'''
from gesture.environments.social import SocialReacher
from gesture.environments.utils import random_run
from gesture.utils.arguments import get_args
import numpy as np


# Nothing done on this... needed for rendering reward functions again.
class SocialReacherTargets(SocialReacher):
    def __init__(self, target_name, args=None):
        Base.__init__(self, XML_PATH=PATH_TO_CUSTOM_XML,
                      robot_name='robot_arm',
                      model_xml='SocialPlane.xml',
                      ac=2, st=6, args=args)
        print('I am', self.model_xml)
        self.target_name = target_name

    def set_reward_function(self, func):
        self.calc_reward = func

    def set_target(self, targets):
        ''' targets should be a
        list [numpy.ndarray, numpy.ndarray]

        state.shape (N,)
        obs.shape (W,H,C)
        '''
        self.state_target = targets[0]
        self.obs_target = targets[1]


        for j in self.target_joints.values():
            j.reset_current_position(self.np_random.uniform(low=-0.01, high=0.01 ), 0)
            j.set_motor_torque(0)

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
        r = self.reward_constant1 * float(self.potential - potential_old)
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


# Rewards
class ReacherAbs(SocialReacher):
    def __init__(self, args=None):
        """TODO: to be defined1. """
        SocialReacher.__init__(self, args)

    def calc_reward(self, a):
        ''' Abs potential as reward '''
        potential_old = self.potential
        return self.calc_potential()

class ReacherDiff(SocialReacher):
    def __init__(self, args=None):
        """TODO: to be defined1. """
        SocialReacher.__init__(self, args)

    def calc_reward(self, a):
        ''' Difference potential as reward '''
        potential_old = self.potential
        self.potential = self.calc_potential()
        r = float(self.potential - potential_old)
        return r

class ReacherDiffCost(SocialReacher):
    def __init__(self, args=None):
        """TODO: to be defined1. """
        SocialReacher.__init__(self, args)

    def calc_reward(self, a):
        potential_old = self.potential
        self.potential = self.calc_potential()

        # cost/penalties
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        return sum(self.rewards)


if __name__ == '__main__':
    args = get_args()

    # === Targets ===
    print('\nLoading targets from:')
    print('path:\t', args.test_target_path)
    datadict = load_dict(args.test_target_path)
    targets = Targets(args.num_proc, datadict)
    targets.remove_speed(args.njoints)

    if False:
        env = ReacherAbs(args)
        env.seed(args.seed)
        random_run(env, render=args.render, verbose=args.verbose)
    elif False:
        env = ReacherDiff(args)
        env.seed(args.seed)
        random_run(env, render=args.render, verbose=args.verbose)
    elif True:
        env = ReacherDiffCost(args)
        env.seed(args.seed)
        random_run(env, render=args.render, verbose=args.verbose)
    elif True:
        env = ReacherDiffCost(args)
        env.seed(args.seed)
        random_run(env, render=args.render, verbose=args.verbose)
