''' Reacher2DoF Combine
State and Obs as input
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

from models.combine import CombinePolicy

from agent.test import Test_and_Save_Video_RGB as Test_and_Save_Video
from agent.train import explorationRGB as exploration
from agent.train import trainRGB as train
from agent.memory import RolloutStorageObs, Results
from agent.memory import StackedObs, StackedState

from data.dataset import load_reacherplane_data

from environments.utils import make_parallel_environments_combine
from environments.reacher import ReacherPlaneCombi

'''
TODO:

Need to fix check of dimensions in test/train targets

exploration
train
'''

def main():
    args = get_args()
    make_log_dirs(args)
    args.num_updates   = int(args.num_frames) // args.num_steps // args.num_processes
    if not args.no_vis:
        vis = VisLogger(args)

    # === Targets ===
    train_path = '/home/erik/DATA/project/ReacherPlaneNoTarget/obsdata_rgb40-40-3_n100000_0.pt'
    Targets, _ = load_reacherplane_data(train_path)

    test_path = '/home/erik/DATA/project/ReacherPlaneNoTarget/obsdata_rgb40-40-3_n100000_1.pt'
    TargetsTest, _ = load_reacherplane_data(test_path)
    # === Environment ===
    args.RGB = True
    Env = ReacherPlaneCombi  # using Env as variable so I only need to change this line between experiments
    env = make_parallel_environments_combine(Env,args, Targets)

    tmp_rgb = args.RGB # reset rgb flag
    test_env = Env(args, TargetsTest)
    args.RGB = tmp_rgb # reset rgb flag

    ob_shape = env.observation_space.shape # RGB
    st_shape = env.state_space.shape[0]    # Joints state
    ac_shape = env.action_space.shape[0]   # Actions

    # === Memory ===
    result             = Results(200, 10)
    CurrentState       = StackedState(args.num_processes, args.num_stack, st_shape)
    CurrentStateTarget = StackedState(args.num_processes, args.num_stack, st_shape)

    CurrentObs         = StackedObs(args.num_processes, args.num_stack, ob_shape)
    CurrentObsTarget   = StackedObs(args.num_processes, args.num_stack, ob_shape)

    rollouts           = RolloutStorageObs(args.num_steps,
                                   args.num_processes,
                                   CurrentState.size()[1],
                                   CurrentObs.size()[1:],
                                   ac_shape)

    # === Model ===
    # CurrentObs.obs_shape - (C, W, H)

    pi = CombinePolicy(o_shape=CurrentObs.obs_shape,
                       o_target_shape=CurrentObs.obs_shape,
                       s_shape=st_shape,
                       s_target_shape=st_shape,
                       a_shape=ac_shape,
                       feature_maps=[64, 64, 8],
                       kernel_sizes=[5, 5, 5],
                       strides=[2, 2, 2],
                       args=args)
    pi.train()
    optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)
    input(pi.total_parameters())
    print('\nPOLICY:\n', pi)

    # ==== Training ====
    print('Learning {}(ac: {}, ob: {})'.format( args.env_id, ac_shape, ob_shape))
    print('\nTraining for %d Updates' % args.num_updates)
    s, s_target, obs, obs_target = env.reset()

    CurrentState.update(s)
    CurrentStateTarget.update(s_target)

    CurrentObs.update(obs)
    CurrentObsTarget.update(obs_target)
    if True:
        print('env.reset | s.shape', s.shape)
        print('env.reset | s_target.shape', s.shape)

        print('env.reset | type(obs)', type(obs))
        print('env.reset | obs.shape', obs.shape)
        print('env.reset | obs.mean', obs.mean())

        print('env.reset | type(obs_target)', type(obs_target))
        print('env.reset | obs_target.shape', obs_target.shape)
        print('env.reset | obs_target.mean', obs_target.mean())

        print('CurrentObs().size()', CurrentObs().size())
        print('CurrentObs().mean()', CurrentObs().mean())

        print('CurrentObsTarget().size()', CurrentObsTarget().size())
        print('CurrentObsTarget().mean()', CurrentObsTarget().mean())

        print('CurrentState().size()', CurrentState().size())
        print('CurrentState().mean()', CurrentState().mean())
        print('CurrentStateTarget().size()', CurrentStateTarget().size())
        print('CurrentStateTarget().mean()', CurrentStateTarget().mean())

    rollouts.states[0].copy_(CurrentState())
    rollouts.observations[0].copy_(CurrentObs())

    if args.cuda:
        CurrentState.cuda()
        CurrentStateTarget.cuda()
        CurrentObs.cuda()
        CurrentObsTarget.cuda()
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
