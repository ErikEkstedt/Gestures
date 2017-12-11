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
from memory import RolloutStorage, StackedState, Results
from train import Training, Exploration
from test import test

from environments.custom_reacher import make_parallel_environments
# from environments.custom_reacher import CustomReacher
from environments.custom_reacher import CustomReacher2 as CustomReacher
# from environments.custom_reacher import CustomReacher3 as CustomReacher

def main():
    args = get_args()  # Real argparser
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    args.num_updates = num_updates
    ds = print_args(args)

    if args.vis:
        from vislogger import VisLogger
        vis = VisLogger(description_list=ds, log_dir=args.log_dir)
        args.log_dir, args.video_dir, args.checkpoint_dir = vis.get_logdir()

    env = make_parallel_environments(CustomReacher,
                                     args.seed,
                                     args.num_processes,
                                     args.potential_constant,
                                     args.electricity_cost,
                                     args.stall_torque_cost,
                                     args.joints_at_limit_cost,
                                     args.episode_time)

    result = Results(max_n=200, max_u=10)

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
    print('Updates: ', num_updates)

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
        frame = pi.n * args.num_processes

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
            if j % args.test_interval == 0:
                print('-'*45)
                print('Testing {} episodes'.format(args.num_test))

            sd = pi.cpu().state_dict()
            test_reward = test(CustomReacher, MLPPolicy, sd, args)
            if args.cuda:
                pi.cuda()

            # Plot result
            vis.line_update(Xdata=frame, Ydata=test_reward, name='Test Score')
            name = os.path.join(args.checkpoint_dir, 'dict_{}_TEST_{}.pt'.format(frame, round(test_reward,3)))
            #  ==== Save best model ======
            torch.save(sd, name)

            #  ==== Save best model ======
            if test_reward > MAX_REWARD:
                print('--'*45)
                print('New High Score!')
                name = os.path.join(args.checkpoint_dir, 'dict_{}_BEST_{}.pt'.format(frame, round(test_reward,3)))
                torch.save(sd, name)
                MAX_REWARD = test_reward


if __name__ == '__main__':
    main()
