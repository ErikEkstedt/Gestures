'''
Loads a state dict, a target dataset and runs simulations.

if args.record then a mp4 file will be saved (kind of bad graphics and this
could be done better)

play saved file on vlc:
    vlc --avcodec-hw none MYSAVEDFILE.mp4

An episode normally lasts for 300 frames and is set through
args.MAX_TIME. The script ends when the environment resets so for
longer play args.MAX_TIME should be set to a larger value.

example:

    python enjoy.py --render --MAX_TIME=3000
    python enjoy.py --record --MAX_TIME=3000

    python enjoy.py --render --MAX_TIME=3000 \
        --target-path=/PATH/to/target_data_set/ \
        --state-dict-path=/PATH/to/state_dict

'''
import numpy as np
import time
import torch
from itertools import count
from torchvision.utils import make_grid
import os
import cv2
from tqdm import tqdm

from project.utils.arguments import get_args
from project.agent.memory import Current
from project.models.combine import CombinePolicy
from project.environments.social import Social
from project.data.dataset import load_reacherplane_data
from project.environments.utils import render_and_scale

def enjoy(env, dset, pi, args):
    if args.record:
        import skvideo.io
        writer = skvideo.io.FFmpegWriter("enjoy.mp4")

    # Test environments
    t = 0
    total_reward, episode_reward, best_episode_reward = 0, 0, -9999

    env.set_target(dset[t]); t += 1
    state, s_target, obs, o_target = env.reset()

    tt = time.time()
    # for j in count(1):
    for j in tqdm(range(args.MAX_TIME)):
        current.update(state, s_target, obs, o_target)
        s ,st, o, ot = current()
        value, action = pi.act(s, st, o, ot)

        if args.render:
            env.render('human')
            env.render('target')

        if args.record:
            human, _, target = env.render('all_rgb_array')  # (W, H, C)
            height, width = obs.shape[:2]
            # target: (40,40,3) -> (3, 600,400)
            # human: (600,400, 3) -> (3, 600,400)
            target = cv2.resize(target,(15*width, 10*height), interpolation = cv2.INTER_CUBIC)
            target = target.transpose((2,0,1))
            human = human.transpose((2,0,1))
            imglist = [torch.from_numpy(human), torch.from_numpy(target)]
            img = make_grid(imglist, padding=5).numpy()
            img = img.transpose((1,2,0))
            img *= 255

            writer.writeFrame(img)

        if j % args.update_target == 0:
            env.set_target(dset[t]); t += 1
            t += 1
            if t > len(dset)-1:
                break

        # Observe reward and next state
        cpu_actions = action.data.cpu().numpy()[0]
        state, s_target, obs, o_target, reward, done, info = env.step(cpu_actions)

        if args.verbose:
            print(reward)

        # If done then update final rewards and reset episode reward
        total_reward += reward
        episode_reward += reward
        if done:
            if args.verbose: print(episode_reward)
            episode_reward = 0
            done = False
            break

    print('Time for enjoyment: ', time.time()-tt)
    if args.record:
        writer.close()

if __name__ == '__main__':
    args = get_args()

    print('Social Mimic')
    print('Loading state dict from:')
    print('path:\t', args.state_dict_path)
    state_dict = torch.load(args.state_dict_path)

    print('\nLoading targets from:')
    print('path:\t', args.target_path)
    dset = torch.load(args.target_path)
    s_target, o_target = dset[4]  # choose random data point
    st_shape = s_target.shape[0]  # targets
    ot_shape = o_target.shape

        # === Environment ===
    env = Social(args)
    env.seed(np.random.randint(0,20000))  # random seed

    s_shape = env.state_space.shape[0]    # Joints state
    o_shape = env.observation_space.shape  # RGB (W,H,C)
    ac_shape = env.action_space.shape[0]   # Actions

    current = Current(1, args.num_stack, s_shape, st_shape, o_shape, o_shape)

    pi = CombinePolicy(o_shape=current.o_shape,
                    o_target_shape=current.ot_shape,
                    s_shape=current.s_shape,
                    st_shape=current.st_shape,
                    a_shape=ac_shape,
                    feature_maps=args.feature_maps,
                    kernel_sizes=args.kernel_sizes,
                    strides=args.strides,
                    args=args)
    pi.load_state_dict(state_dict)

    if args.cuda:
        current.cuda()
        pi.cuda()

    enjoy(env, dset, pi, args)
