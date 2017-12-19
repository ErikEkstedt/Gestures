import numpy as np
import gym
from copy import deepcopy
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from utils import log_print, make_log_dirs
from arguments import FakeArgs, get_args
from model import MLPPolicy
from memory import RolloutStorage, StackedState, Results

from train import train, exploration_rgb
from test import test, test_existing_env, Test_and_Save_Video

from utils import get_env
from environments.reacher_envs import make_parallel_environments

def main():
    args = get_args()
    args.RGB = True

    # === Environment ===
    Env = get_env(args)

    # Logger
    make_log_dirs(args)
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    args.num_updates = num_updates

    if args.vis:
        from vislogger import VisLogger
        vis = VisLogger(args)

    if args.num_processes > 1:
        from train import exploration
        env = make_parallel_environments(Env,args)
    else:
        from train import Exploration_single as exploration
        env = Env(args)
        env.seed(args.seed)

    test_env = Env(args)

    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]
        # === Memory ===
    result = Results(max_n=200, max_u=10)
    CurrentState = StackedState(args.num_processes,
                                args.num_stack,
                                ob_shape)

    rollouts = RolloutStorage(args.num_steps,
                              args.num_processes,
                              CurrentState.size()[1],
                              ac_shape)


    # === Model ===
    pi = MLPPolicy(CurrentState.state_shape,
                   ac_shape,
                   hidden=args.hidden,
                   total_frames=args.num_frames)

    pi.train()
    optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)

    # ==== Training ====
    print('Learning {}(ac: {}, ob: {})'.format( args.env_id, ac_shape, ob_shape))
    print('\nTraining for %d Updates' % num_updates)
    (s, obs) = env.reset()
    print('joint state shape:', s.shape)
    print('observation shaep:',obs.shape)

    def show_state(obs):
        ob = torch.Tensor(obs)
        ob = ob.permute(0,3,1,2)
        img = make_grid(ob, nrow=2)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.show()

    # show_state(obs)

    CurrentState.update(s)
    rollouts.states[0].copy_(CurrentState())

    if args.cuda:
        CurrentState.cuda()
        rollouts.cuda()
        pi.cuda()

    MAX_REWARD = -999999
    for j in range(num_updates):
        exploration_rgb(pi, CurrentState, rollouts, args, result, env)
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
        # if not args.no_test and j % args.test_interval < nt and j > nt:
        if not args.no_test and j % args.test_interval < nt:
            ''' `j % args.test_interval < 5` is there because:
            If tests are not performed during some interval bad luck might make
            it that although the model becomes better the test occured
            during a bad policy update. The policy adjust for this in the next
            update but we might miss good policies if we test too seldom.
            Thus we test in an interval of 5 every args.test_interval.
            (default: args.num_test = 50)
                -> test updates [50,54], [100,104], ...
            '''
            if j % args.test_interval == 0:
                print('-'*45)
                print('Testing {} episodes'.format(args.num_test))

            pi.cpu()
            sd = deepcopy(pi.cpu().state_dict())
            # test_reward = test(test_env, MLPPolicy, sd, args)
            # test_reward = test_existing_env(test_env, MLPPolicy, sd, args)
            test_reward, videolist = Test_and_Save_Video(test_env, MLPPolicy, sd, args)

            # Plot result
            print('Average Test Reward: {}\n '.format(round(test_reward)))
            if args.vis:
                vis.line_update(Xdata=frame,
                                Ydata=test_reward, name='Test Score')

            #  ==== Save best model ======
            name = os.path.join(
                args.checkpoint_dir,
                'dict_{}_TEST_{}.pt'.format(frame, round(test_reward, 3)))
            torch.save(sd, name)

            name = os.path.join(
                args.result_dir,
                'video{}_TEST_{}.pt'.format(frame, round(test_reward, 3)))
            torch.save(videolist, name)

            # name = os.path.join(
            #     args.checkpoint_dir,
            #     'model_{}_TEST_{}.pt'.format(frame, round(test_reward, 3)))
            # torch.save(pi, name )
            #
            #  ==== Save best model ======
            if test_reward > MAX_REWARD:
                print('--'*45)
                print('New High Score!\n')
                name = os.path.join(
                    args.checkpoint_dir,
                    'BESTDICT{}_{}.pt'.format(frame, round(test_reward, 3)))
                torch.save(sd, name)
                name = os.path.join(
                    args.checkpoint_dir,
                    'BESTMODEL{}_{}.pt'.format(frame, round(test_reward, 3)))
                torch.save(pi, name)
                name = os.path.join(
                    args.result_dir,
                    'BESTVIDEO{}_{}.pt'.format(frame, round(test_reward, 3)))
                torch.save(videolist, name)
                MAX_REWARD = test_reward

            if args.cuda:
                pi.cuda()


if __name__ == '__main__':
    main()
