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
from models.coordination import MLPPolicy
from agent.memory import RolloutStorage, StackedState, Results
from agent.train import train, exploration
from agent.test import Test_and_Save_Video
from environments.reacher import ReacherPlane
from environments.utils import make_parallel_environments
from environments.data import make_video


def main():
    args = get_args()
    make_log_dirs(args)
    args.num_updates   = int(args.num_frames) // args.num_steps // args.num_processes
    if not args.no_vis:
        vis = VisLogger(args)

    # === Environment ===
    Env = ReacherPlane  # using Env as variable so I only need to change this line between experiments
    env = make_parallel_environments(Env,args)

    tmp_rgb = args.RGB # reset rgb flag
    args.RGB = True
    test_env = Env(args)
    args.RGB = tmp_rgb # reset rgb flag

    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]

    # === Memory ===
    result = Results(max_n=200, max_u=10)
    CurrentState = StackedState(args.num_processes, args.num_stack, ob_shape)
    rollouts = RolloutStorage(args.num_steps,
                              args.num_processes,
                              CurrentState.size()[1],
                              ac_shape)

    # === Model ===
    pi = MLPPolicy(CurrentState.state_shape, ac_shape, args)

    pi.train()
    optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)

    # ==== Training ====
    print('Learning {}(ac: {}, ob: {})'.format( args.env_id, ac_shape, ob_shape))
    print('\nTraining for %d Updates' % args.num_updates)
    s = env.reset()
    CurrentState.update(s)
    rollouts.states[0].copy_(CurrentState())

    if args.cuda:
        CurrentState.cuda()
        rollouts.cuda()
        pi.cuda()

    MAX_REWARD = -999999
    for j in range(args.num_updates):
        exploration(pi, CurrentState, rollouts, args, result, env)
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
            # sd = deepcopy(pi.cpu().state_dict())
            sd = pi.cpu().state_dict()
            # test_reward = test(test_env, MLPPolicy, sd, args)
            # test_reward = test_existing_env(test_env, MLPPolicy, sd, args)
            test_reward, BestVideo = Test_and_Save_Video(test_env, MLPPolicy, sd, args)
            # Plot result
            print('Average Test Reward: {}\n '.format(round(test_reward)))
            if args.vis:
                vis.line_update(Xdata=frame,
                                Ydata=test_reward, name='Test Score')

            #  ==== Save best model ======
            if test_reward > MAX_REWARD:
                print('--'*45)
                print('New High Score!\n')
                print('error: ', test_reward)
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

    =
    make_video()

if __name__ == '__main__':
    main()
