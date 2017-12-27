import argparse
import numpy as np
# from copy import deepcopy
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from Agent.utils import log_print, make_log_dirs
from Agent.arguments import FakeArgs, get_args
from Agent.model import MLPPolicy
from Agent.memory import RolloutStorage, StackedState, Results

from Agent.train import train, exploration
from Agent.test import Test_and_Save_Video

from environments.Reacher import ReacherPlane
from environments.reacher_envs import make_parallel_environments

from Agent.vislogger import VisLogger

def get_args():
    ''' Arguments for 2DoF experiment '''
    parser = argparse.ArgumentParser(description='2DoF')
    parser.add_argument('--num-processes', type=int, default=4)

    # === Environment ===
    parser.add_argument('--env-id', default='ReacherPlane')
    parser.add_argument('--dof', type=int, default=2)

    parser.add_argument('--MAX_TIME', type=int, default=300)
    parser.add_argument('--gravity', type=float, default=9.81)
    parser.add_argument('--power', type=float, default=0.5)
    parser.add_argument('--RGB', action='store_true', default=False)
    parser.add_argument('--video', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)

    # Reward
    parser.add_argument('--potential-constant',   type=float, default=100)
    parser.add_argument('--electricity-cost',     type=float, default=-0.1)
    parser.add_argument('--stall-torque-cost',    type=float, default=-0.01)
    parser.add_argument('--joints-at-limit-cost', type=float, default=-0.01)
    parser.add_argument('--r1', type=float, default=1.0)
    parser.add_argument('--r2', type=float, default=1.0)

    # === PPO Loss ===
    parser.add_argument('--pi-lr', type=float, default=3e-4, help='policy learning rate (default: 3e-4)')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon value(default: 1e-8)')
    parser.add_argument('--no-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.0, help='entropy term coefficient (default: 0.0)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--max-grad-norm', type=float, default=5, help='ppo clip parameter (default: 5)')

    # PPO Training
    parser.add_argument('--num-frames', type=int, default=int(10e6), help='number of frames to train (default: 10e6)')
    parser.add_argument('--num-steps', type=int, default=2048, help='number of exploration steps in ppo (default: ?)')
    parser.add_argument('--batch-size', type=int, default=256, help='ppo batch size (default: 256)')
    parser.add_argument('--max-episode-length', type=int, default=100000, help='maximum steps in one episode (default: 10000)')
    parser.add_argument('--ppo-epoch', type=int, default=8, help='number of ppo epochs, K in paper (default: 8)')
    parser.add_argument('--num-stack', type=int, default=1, help='number of frames to stack (default: 1)')
    parser.add_argument('--hidden', type=int, default=256, help='Number of hidden neurons in policy (default: 128)')
    parser.add_argument('--std-start', type=float, default=-0.6, help='std-start (Hyperparams for Roboschool in paper)')
    parser.add_argument('--std-stop', type=float, default=-1.7, help='std stop (Hyperparams for Roboschool in paper)')
    parser.add_argument('--seed', type=int, default=10, help='random seed (default: 10)')

    # Test
    parser.add_argument('--no-test', action='store_true', default=False, help='disables test during training')
    parser.add_argument('--test-interval', type=int,  default=50, help='how many updates/test (default: 50)')
    parser.add_argument('--num-test', type=int, default=20, help='Number of test episodes during test (default: 20)')
    parser.add_argument('--load-file', default='/tmp/', help='state_dict to load')

    # Log
    parser.add_argument('--vis-interval', type=int, default=1, help='vis interval, one log per n updates (default: 1)')
    parser.add_argument('--log-dir', default='/tmp/', help='directory to save agent logs (default: /tmp/)')
    parser.add_argument('--log-interval', type=int, default=1, help='log interval in console, one log per n updates (default: 1)')

    # Boolean
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-vis', action='store_true', default=False, help='disables visdom visualization')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis
    return args


'''
Trains on ReacherPlane where the reacher has 2 DoF and two targets.

The reward is the sum of the two absolute potentials
(the p2-norm of the difference vector multiplied by a reward_constant)
constant = 100
r1, r2 = 1, 1

The absolute potential is wuite large and results in huge values for the value
loss. (starting at 38M). The best result got -9797. as test average.

'''

def main():
    # === Settings ===
    args = get_args()
    make_log_dirs(args)
    args.num_updates   = int(args.num_frames) // args.num_steps // args.num_processes
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


if __name__ == '__main__':
    main()
