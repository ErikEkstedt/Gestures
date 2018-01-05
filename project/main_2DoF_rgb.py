''' Reacher2DoF RGB
Only pixelspace as input
Reward function based on state space
'''
import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from utils.utils import make_log_dirs
from utils.arguments import get_args
from utils.vislogger import VisLogger

from models.coordination import MLPPolicy, CNNPolicy

from agent.test import Test_and_Save_Video_RGB as Test_and_Save_Video
from agent.train import explorationRGB as exploration
from agent.train import trainRGB as train
from agent.memory import RolloutStorageObs, Results
from agent.memory import StackedObs, StackedState

from environments.utils import make_parallel_environments
from environments.reacher import ReacherPlane


def main():
    args = get_args()
    make_log_dirs(args)
    args.num_updates   = int(args.num_frames) // args.num_steps // args.num_processes
    if not args.no_vis:
        vis = VisLogger(args)

    # === Environment ===
    args.RGB = True 	# be safe
    Env = ReacherPlane  # Using Env, only change this line between experiments
    env = make_parallel_environments(Env,args)

    tmp_rgb = args.RGB # reset rgb flag
    test_env = Env(args)
    args.RGB = tmp_rgb # reset rgb flag

    ob_shape = env.observation_space.shape # RGB
    st_shape = env.state_space.shape[0]    # Joints state
    ac_shape = env.action_space.shape[0]   # Actions

    # === Memory ===
    CurrentState = StackedState(args.num_processes, args.num_stack, st_shape)
    result     = Results(max_n=200, max_u=10)
    CurrentObs = StackedObs(args.num_processes, args.num_stack, ob_shape)
    rollouts   = RolloutStorageObs(args.num_steps,
                                   args.num_processes,
                                   CurrentState.size()[1],
                                   CurrentObs.size()[1:],
                                   ac_shape)

    # === Model ===
    # CurrentObs.obs_shape - (C, W, H)
    pi = CNNPolicy(input_shape=CurrentObs.obs_shape,
                   action_shape=ac_shape,
                   in_channels=CurrentObs.obs_shape[0],
                   feature_maps=[64, 64, 64],
                   kernel_sizes=[5, 5, 5],
                   strides=[2, 2, 2],
                   args=args)
    pi.train()
    optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)
    print('\nPOLICY:\n', pi)


    # ==== Training ====
    print('Learning {}(ac: {}, ob: {})'.format( args.env_id, ac_shape, ob_shape))
    print('\nTraining for %d Updates' % args.num_updates)
    s, obs = env.reset()
    CurrentState.update(s)
    CurrentObs.update(obs)
    if False:
        print('After env.reset | s.shape', s.shape)
        print('After env.reset | obs.shape', obs.shape)
        print('After env.reset | obs.mean', obs.mean())
        print('CurrentObs().size()', CurrentObs().size())
        print('CurrentObs().mean()', CurrentObs().mean())

    rollouts.states[0].copy_(CurrentState())
    rollouts.observations[0].copy_(CurrentObs())

    if args.cuda:
        CurrentState.cuda()
        CurrentObs.cuda()
        rollouts.cuda()
        pi.cuda()

    MAX_REWARD = -999999
    for j in range(args.num_updates):
        exploration(pi, CurrentState, CurrentObs, rollouts, args, result, env)
        vloss, ploss, ent = train(pi, args, rollouts, optimizer_pi)

        rollouts.last_to_first()
        result.update_loss(vloss.data, ploss.data, ent.data)
        frame = pi.n * args.num_processes

        #  ==== SHELL LOG ======
        if j % args.log_interval == 0 and j > 0:
            result.plot_console(frame)

        #  ==== VISDOM PLOT ======
        if j % args.vis_interval == 0 and j > 0 and not args.no_vis:
            result.vis_plot(vis, frame, pi.get_std())

        #  ==== TEST ======
        nt = 5
        if not args.no_test and j % args.test_interval < nt and j>nt:
            if j % args.test_interval == 0:
                print('-'*45)
                print('Testing {} episodes'.format(args.num_test))

            pi.cpu()
            sd = pi.cpu().state_dict()
            test_reward, BestVideo = Test_and_Save_Video(test_env, CNNPolicy, sd, args)

            # Plot result
            print('Average Test Reward: {}\n '.format(round(test_reward)))
            if args.vis:
                vis.line_update(Xdata=frame,
                                Ydata=test_reward, name='Test Score')

            #  ==== Save best model ======
            if test_reward > MAX_REWARD:
                print('--'*45)
                print('New High Score!\n')
                print('Avg. Reward: ', test_reward)
                name = os.path.join(args.result_dir,
                    'BESTVIDEO{}_{}.pt'.format(round(test_reward, 1), frame))
                print('Saving Best Video')
                torch.save(BestVideo, name)
                name = os.path.join(
                    args.checkpoint_dir,
                    'BESTDICT{}_{}.pt'.format(frame, round(test_reward, 3)))
                torch.save(sd, name)
                MAX_REWARD = test_reward
            else:
                name = os.path.join(
                    args.checkpoint_dir,
                    'dict_{}_TEST_{}.pt'.format(frame, round(test_reward, 3)))
                torch.save(sd, name)

            if args.cuda:
                pi.cuda()


if __name__ == '__main__':
    main()
