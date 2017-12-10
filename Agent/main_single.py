import numpy as np
import gym
from copy import deepcopy
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from utils import args_to_list, print_args, log_print
from arguments import FakeArgs, get_args
from model import MLPPolicy
from memory import RolloutStorage, StackedState, Results_single
from train import Training
from train import Exploration_single as Exploration
from test import test
from environments.custom_reacher import CustomReacher


def main():
    args = get_args()  # Real argparser
    args.num_processes = 1  # only here temporary
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    args.updates = num_updates
    ds = print_args(args)

    if args.vis:
        from vislogger import VisLogger
        vis = VisLogger(description_list=ds, log_dir=args.log_dir)
        args.log_dir, args.video_dir, args.checkpoint_dir = vis.get_logdir()

    torch.manual_seed(args.seed)
    env = CustomReacher()
    env.seed(args.seed)

    result = Results_single(max_n=200, max_u=10)
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]
    CurrentState = StackedState(args.num_processes,
                                args.num_stack,
                                ob_shape)

    rollouts = RolloutStorage(args.num_steps,
                              args.num_processes,
                              CurrentState.size()[1],
                              ac_shape)

    pi = MLPPolicy(CurrentState.state_shape,
                   ac_shape,
                   hidden=args.hidden,
                   total_frames=args.num_frames)

    pi.train()
    optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)

    # ==== Training ====

    s = env.reset()
    CurrentState.update(s)
    rollouts.states[0].copy_(CurrentState())

    if args.cuda:
        CurrentState.cuda()
        rollouts.cuda()
        pi.cuda()

    MAX_REWARD = -999999
    for j in range(num_updates):
        Exploration(pi, CurrentState, rollouts, args, result, env)
        vloss, ploss, ent = Training(pi, args, rollouts, optimizer_pi)

        rollouts.last_to_first()
        result.update_loss(vloss.data, ploss.data, ent.data)
        frame = pi.n

        #  ==== SHELL LOG ======
        if j % args.log_interval == 0 and j > 0:
            result.plot_console(frame)

        #  ==== VISDOM PLOT ======
        if j % args.vis_interval == 0 and j > 0 and not args.no_vis:
            result.vis_plot(vis, frame, pi.get_std())

        #  ==== TEST ======
        if not args.no_test and j % args.test_interval < 5 and j > 0:
            ''' `j % args.test_interval < 5` is there because:
            If tests are not performed during some interval bad luck might make
            it that although the model becomes better the test occured
            during a bad policy update. The policy adjust for this in the next
            update but we might miss good policies if we test too seldom.
            Thus we test in an interval of 5 every args.test_interval.
            (default: args.num_test = 50) -> test updates [50,54], [100,104], ...
            '''
            print('-'*45)
            print('Testing {} episodes'.format(args.num_test))
            sd = pi.cpu().state_dict()
            test_reward = test(CustomReacher, MLPPolicy, sd, args)
            pi.cuda()

            # Plot result
            vis.line_update(Xdata=frame, Ydata=test_reward, name='Test Score')
            print('Test Average: {}\n'.format(test_reward))

            #  ==== Save best model ======
            print('--'*45)
            name = os.path.join(args.checkpoint_dir,
                                'dict_test_'+str(frame)+'_'+str(test_reward))
            print('Saving after test ({})\nAt location: {}'.format(test_reward, name))
            torch.save(sd, name + '.pt')

            #  ==== Save best model ======
            if test_reward > MAX_REWARD:
                print('--'*45)
                print('New High Score!')
                name = os.path.join(args.checkpoint_dir,
                                    'dict_test_'+str(test_reward))
                print('Saving Max Score test ({})\nAt location: {}'.format(test_reward, name))
                torch.save(sd, name + '.pt')
                MAX_REWARD = test_reward
                print('New Max: {}\n'.format(MAX_REWARD))


if __name__ == '__main__':
    main()
