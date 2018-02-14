'''
Loads a state dict, a target dataset and runs simulations.
example:
    python enjoy.py --render --MAX_TIME=3000

    python enjoy.py --record --MAX_TIME=3000

    python enjoy.py --render --MAX_TIME=3000 \
        --target-path=/PATH/to/target_data_set/ \
        --state-dict-path=/PATH/to/state_dict
'''
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import count
from tqdm import tqdm
import torch
from torch.autograd import Variable

from gesture.utils.arguments import get_args
from gesture.utils.utils import record, load_dict, get_targets
from gesture.agent.memory import Current, Targets
from gesture.models.modular import VanillaCNN
from gesture.models.combine import SemiCombinePolicy
from gesture.environments.social import SocialReacher


class PoseDefiner(object):
    def __init__(self, thresh=0.1, done_duration=30, max_time=300, target=None):
        self.thresh = thresh
        self.done_duration = done_duration
        self.max_time = max_time
        self.target = target

        self.counter = 0
        self.time = 0
        self.poses_achieved = 0

    def reset(self, target):
        self.counter = 0
        self.target = target

    def update(self, state):
        change_target = False
        self.time += 1
        d = np.linalg.norm(state[:len(target)] - self.target)
        if d < self.thresh:
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
        return d, change_target

    def distance(self, state):
        return np.linalg.norm(state[:len(target)] - self.target)


def evaluate(env, targets, pi, understand, args, plot=False, USE_UNDERSTAND=True):
    if args.cuda:
        current.cuda()
        pi.cuda()
        understand.cuda()

    if args.record:
        import skvideo.io
        name = "mimic_{}_update{}.mp4".format(args.env_id, args.update_target)
        writer = skvideo.io.FFmpegWriter(name)

    if args.continuous_targets:
        target = targets[0]
        t = 1
    else:
        target = targets()

    env.set_target(target)
    state, real_state_target, obs, o_target = env.reset()

    posedefiner = PoseDefiner(target=real_state_target)
    d, _ = posedefiner.distance(state)
    X = [0]; Y = [d]

    tt = time.time()
    total_reward = 0
    for j in tqdm(range(args.MAX_TIME)):

        if USE_UNDERSTAND:
            o_tmp = o_target.transpose(2, 0, 1).astype('float')
            o_tmp /= 255
            o_tmp = torch.from_numpy(o_tmp).float().unsqueeze(0)
            s_target = understand(Variable(o_target)).data.numpy()
        else:
            s_target = real_state_target

        current.update(state, s_target, obs, o_target)
        s ,st, o, ot = current()
        value, action = pi.act(s, st, o, ot)

        if args.render:
            env.render('human')
            env.render('target')

        if args.record:
            record(env, writer)

        # if j % args.update_target == 0:
        #     if args.continuous_targets:
        #         target = targets[t]
        #         t += 1
        #         if t > len(targets)-1:
        #             break
        #     else:
        #         target = targets()
        #     env.set_target(target)

        # Observe reward and next state
        cpu_actions = action.data.cpu().numpy()[0]
        state, real_state_target, obs, o_target, reward, done, info = env.step(cpu_actions)
        total_reward += reward

        d, pose_done = posedefiner.update(state)
        Y.append(d)
        X.append(j)
        if plot:
            plt.plot(X,Y,'-b', X, [0.1]*len(X), '-r')
            plt.pause(1e-4)

        if pose_done:
            if args.continuous_targets:
                target = targets[t]
                t += 1
                if t > len(targets)-1:
                    break
            else:
                target = targets()
            env.set_target(target)
            state, real_state_target, obs, o_target = env.reset()

    print('Duration of script: ', time.time()-tt)
    print('Total Reward: ', total_reward)
    if args.record:
        writer.close()
    plt.plot(X,Y,'-b', X, [0.1]*len(X), '-r')
    plt.show()

if __name__ == '__main__':
    print('Evaluation of Modular approach!')
    args = get_args()
    args.num_proc = 1

    # === Environment and targets ===
    env = SocialReacher(args)
    env.seed(np.random.randint(0,20000))  # random seed

    print('\nLoading targets from:')
    print('path:\t', args.test_target_path)
    datadict = load_dict(args.test_target_path)
    targets = Targets(1, datadict)
    targets.remove_speed(args.njoints)

    s_target, o_target = targets()  # random
    st_shape = s_target.shape[0]  # targets
    ot_shape = o_target.shape

    s_shape = env.state_space.shape[0]    # Joints state
    o_shape = env.observation_space.shape  # RGB (W,H,C)
    ac_shape = env.action_space.shape[0]   # Actions
    current = Current(1, args.num_stack, s_shape, st_shape, o_shape, o_shape, ac_shape)

    pi = SemiCombinePolicy(s_shape=current.s_shape,
                           st_shape=current.st_shape,
                           o_shape=current.o_shape,
                           ot_shape=current.ot_shape,
                           a_shape=current.ac_shape,
                           feature_maps=args.feature_maps,
                           kernel_sizes=args.kernel_sizes,
                           strides=args.strides,
                           args=args)

    print('Loading coordination state dict from:')
    print('path:\t', args.state_dict_path)
    coordination_state_dict = torch.load(args.state_dict_path)
    pi.load_state_dict(coordination_state_dict)

    args.hidden=128
    understand = VanillaCNN(input_shape=current.ot_shape,
                            s_shape=current.st_shape,
                            feature_maps=args.feature_maps,
                            kernel_sizes=args.kernel_sizes,
                            strides=args.strides,
                            args=args)

    print('Loading understanding state dict from:')
    print('path:\t', args.state_dict_path2)
    understand_state_dict = torch.load(args.state_dict_path2)
    understand.load_state_dict(understand_state_dict)

    pi.eval()
    understand.eval()
    evaluate(env, targets, pi, understand, args, plot=True, USE_UNDERSTAND=False)
