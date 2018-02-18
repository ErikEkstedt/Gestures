''' Pepper '''
import __future__
import os
import numpy as np
import datetime
import qi
import motion
import matplotlib.pyplot as plt

from arguments import get_args
from Pepper import Pepper_v0
from train import explorationPepper as exploration
from train import trainPepper as train
from memory import Results, Current
from storage import RolloutStoragePepper
from model import MLPPolicy, VanillaCNN

import torch
import torch.optim as optim
import torch.nn as nn

class PoseDefiner(object):
    def __init__(self, thresh=0.1, done_duration=50, max_time=300, target=None):
        self.thresh = thresh
        self.done_duration = done_duration
        self.max_time = max_time
        self.target = target

        self.counter = 0
        self.time = 0
        self.poses_achieved = 0
        self.total_poses = 1

    def reset(self, target):
        self.counter = 0
        self.time = 0
        self.target = target
        self.total_poses += 1

    def update(self, state):
        self.time += 1
        change_target = False
        dist = np.linalg.norm(state[:len(self.target)] - self.target)
        if dist < self.thresh:
            # Pose reached!
            self.counter += 1
            if self.counter >= self.done_duration:
                # Pose achieved!
                self.poses_achieved += 1
                change_target = True
        else:
            self.counter = 0
            if self.time > self.max_time:
                change_target = True
        return dist, change_target

    def distance(self, state):
        return np.linalg.norm(state[:len(self.target)] - self.target)

    def print_result(self):
        print('\nPoses reached/possible: {}/{}'.format(self.poses_achieved, self.total_poses))


def evaluate(env, pi, args, plot=False):
    if args.cuda:
        current.cuda()
        pi.cuda()

    name="Pepper"
    if args.record:
        import skvideo.io
        vidname = name+'.mp4'
        writer = skvideo.io.FFmpegWriter(os.path.join(args.log_dir,vidname))

    # Initialize targets and reset env
    state, real_state_target, obs = env.reset()

    posedefiner = PoseDefiner(target=real_state_target, max_time=args.update_target)
    d = posedefiner.distance(state)
    X = [0]; Y = [d]

    ep_rew, total_reward = 0, 0
    for j in range(args.MAX_TIME):

        current.update(state, real_state_target, obs, obs)
        s ,st, o, _ = current()
        value, action = pi.act(s, st, o, o)

        if args.render:
            env.render('human')
            env.render('target')

        if args.record:
            record(env, writer)

        # Observe reward and next state

        cpu_actions = action.data.cpu().numpy()
        state, real_state_target, obs, reward, done = env.step(cpu_actions)
        total_reward += reward
        ep_rew += reward

        d, pose_done = posedefiner.update(state)
        Y.append(d)
        X.append(j)
        if plot:
            plt.plot(X,Y,'-b', X, [0.1]*len(X), '-r')
            plt.pause(1e-4)

        if pose_done:
            print('episode reward:', ep_rew)
            print('state: ', env.state)
            print('target: ', env.target)
            ep_rew = 0
            if args.random:
                env.set_random_target()
            state, real_state_target, obs = env.reset()
            posedefiner.reset(real_state_target)

    print('Total Reward: ', total_reward)
    if args.record:
        writer.close()

    posedefiner.print_result()
    plt.plot(X,Y,'-b', X, [0.1]*len(X), '-r')
    # plt.show()

    name += str(posedefiner.poses_achieved) +'of'+ str(posedefiner.total_poses)+'.png'
    name = os.path.join(args.log_dir,name)
    print('plotname',name)
    plt.savefig(name, bbox_inches='tight')

def radians_to_degrees(a):
    import math
    return a*180/math.pi

if __name__ == '__main__':
    args = get_args()
    args.num_proc=1
    session = qi.Session()
    session.connect("{}:{}".format(args.IP, args.PORT))
    env = Pepper_v0(session, args=args)

    # Create dirs
    path='/home/erik/DATA/Pepper'
    run = 0
    while os.path.exists("{}/run-{}".format(path, run)):
        run += 1
    path = "{}/run-{}".format(path, run)
    os.mkdir(path)
    args.log_dir = path
    args.checkpoint_dir = path

    # ====== Goal ===============
    # "hurray pose"
    print('Using static Target')
    L_arm = [-0.38450, 0.81796, -0.99049, -1.18418, -1.3949, 0.0199]
    R_arm = [-0.90522, -1.03321, -0.05766, 0.84596, 1.39495, 0.01999]
    st = np.array(L_arm+R_arm).astype('float32')
    d = np.degrees(st)
    env.set_target(st)

    # env.set_angles(st)
    # degre = radians_to_degrees(st)
    # print(env.names)
    # print(st)
    # print(d)
    # raw_input()
    # raw_input()
    # s_shape = env.state_space.shape[0]    # Joints state

    s_shape = 24
    st_shape = 12
    o_shape = env.observation_space.shape  # RGB (W,H,C)
    o_shape = (3,64,64)
    ac_shape = env.action_space.shape[0]   # Actions

    # === Memory ===
    result = Results(200, 10)
    current = Current(args.num_proc, args.num_stack, s_shape, st.shape[0], o_shape, o_shape, ac_shape)

    # === Model ===
    in_shape = current.st_shape + current.s_shape
    pi = MLPPolicy(input_size=in_shape, a_shape=current.ac_shape, args=args)

    print('\n=== Continue Training ===\n')
    print('Loading:', args.state_dict_path)
    sd = torch.load(args.state_dict_path)
    pi.load_state_dict(sd)

    evaluate(env, pi, args, plot=False)
