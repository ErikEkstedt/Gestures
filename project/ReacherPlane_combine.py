''' Reacher2DoF Combine
State and Obs as input
Reward function based on state space
'''
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

from agent.test import Test_and_Save_Video_Combi as Test_and_Save_Video
from agent.train import explorationCombine as exploration
from agent.train import trainCombine as train
from agent.memory import RolloutStorageCombi as RolloutStorage
from agent.memory import Results
from agent.memory import StackedObs, StackedState
from data.dataset import load_reacherplane_data
from environments.utils import make_parallel_environments_combine
from environments.reacher import ReacherPlaneCombi


def print_shapes(s, obs, CurrentState, CurrentStateTarget, CurrentObs, CurrentObsTarget):
    print('-'*80)
    print('s.shape', s.shape)
    print('s_target.shape', s.shape)
    print()
    print('type(obs)', type(obs))
    print('obs.shape', obs.shape)
    print('obs.mean', obs.mean())
    print()
    print('type(obs_target)', type(obs_target))
    print('obs_target.shape', obs_target.shape)
    print('obs_target.mean', obs_target.mean())
    print()
    print('CurrentObs().size()', CurrentObs().size())
    print('CurrentObs().mean()', CurrentObs().mean())
    print()
    print('CurrentObsTarget().size()', CurrentObsTarget().size())
    print('CurrentObsTarget().mean()', CurrentObsTarget().mean())
    print()
    print('CurrentState().size()', CurrentState().size())
    print('CurrentState().mean()', CurrentState().mean())
    print('CurrentStateTarget().size()', CurrentStateTarget().size())
    print('CurrentStateTarget().mean()', CurrentStateTarget().mean())


def get_targets(train_path='', test_path='', verbose=False):
    ''' Return dataset (not loader)'''
    train_path = '/home/erik/DATA/project/ReacherPlaneNoTarget/obsdata_rgb40-40-3_n100000_0.pt'
    trainset, trainloader = load_reacherplane_data(train_path)

    test_path = '/home/erik/DATA/project/ReacherPlaneNoTarget/obsdata_rgb40-40-3_n100000_1.pt'
    testset, testloader = load_reacherplane_data(test_path)

    # check dims
    ob_tr, s_tr = trainset[4]  # choose random data point
    ob_te, s_te = testset[4]  # choose random data point
    if verbose:
        print('Train:\t state shapes: {}\t obs_shapes:{}'.format( s_tr.shape, ob_tr.shape))
        print('Test:\t state shapes: {}\t obs_shapes:{}'.format( s_te.shape, ob_te.shape))

    assert s_tr.shape == s_te.shape, 'training and test shapes do not match'
    assert ob_tr.shape == ob_te.shape, 'training and test shapes do not match'
    return trainset, testset


def main():
    args = get_args()
    print('ReacherPlaneCombi')

    # === Targets ===
    print('Loading target labels...')
    traintargets, testtargets = get_targets()  # asserts same dims on train/test
    ob_sample, st_sample, = traintargets[4]
    ob_target_shape = ob_sample.shape
    st_target_shape = st_sample.shape[0]


    # Force Settings
    args.RGB = True
    args.COMBI = True
    args.video_w = ob_sample.shape[1]
    args.video_h = ob_sample.shape[2]

    if args.verbose:
        print('ob shape: {}, st_shape: {}, COMBI: {}'.format(ob_sample.shape, st_sample.shape, args.COMBI))
        print('args- Video_W: {}, Video_H: {}'.format(args.video_w, args.video_h))
        input('Press Enter to continue')

    # frames -> updates
    args.num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    args.test_thresh = int(args.test_thresh) // args.num_steps // args.num_processes

    make_log_dirs(args)
    if not args.no_vis:
        vis = VisLogger(args)

    # === Environment ===
    Env = ReacherPlaneCombi  # using Env as variable so I only need to change this line between experiments

    # train
    env = make_parallel_environments_combine(Env, args, traintargets)

    st_shape = env.state_space.shape[0]    # Joints state
    ob_shape = env.observation_space.shape # RGB
    ac_shape = env.action_space.shape[0]   # Actions

    # test environment
    test_env = Env(args, testtargets)
    test_env.seed(np.random.randint(0,2000))

    # === Memory ===
    result             = Results(200, 10)
    CurrentState       = StackedState(args.num_processes, args.num_stack, st_shape)
    CurrentStateTarget = StackedState(args.num_processes, args.num_stack, st_target_shape)
    CurrentObs         = StackedObs(args.num_processes, args.num_stack, ob_shape)
    CurrentObsTarget   = StackedObs(args.num_processes, args.num_stack, ob_shape)

    rollouts           = RolloutStorage(args.num_steps,
                                        args.num_processes,
                                        CurrentState.size()[1],
                                        CurrentStateTarget.size()[1],
                                        CurrentObs.size()[1:],
                                        ac_shape)

    # === Model ===
    # CurrentObs.obs_shape - (C, W, H)
    pi = CombinePolicy(o_shape=CurrentObs.obs_shape,
                       o_target_shape=CurrentObs.obs_shape,
                       s_shape=st_shape,
                       s_target_shape=st_target_shape,
                       a_shape=ac_shape,
                       feature_maps=[64, 64, 8],
                       kernel_sizes=[5, 5, 5],
                       strides=[2, 2, 2],
                       args=args)

    pi.train()
    optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)
    print('\nPOLICY:\n', pi)
    print('Total network parameters to train: ', pi.total_parameters())


    # ==== Training ====
    print('Learning {}(ac: {}, st: {}, ob: {})'.format( args.env_id, ac_shape, st_shape, ob_shape))
    print('\nTraining for %d Updates' % args.num_updates)

    s, s_target, obs, obs_target = env.reset()

    CurrentState.update(s)  # keep track of current state (num_proc, num_stack, state_shape)
    CurrentStateTarget.update(s_target)

    CurrentObs.update(obs)  # keep track of current obs (num_proc, num_stack, obd_shape)
    CurrentObsTarget.update(obs_target)

    # print_shapes(s, obs, CurrentState, CurrentState, CurrentObs, CurrentObsTarget)

    rollouts.states[0].copy_(CurrentState())
    rollouts.target_states[0].copy_(CurrentStateTarget())

    rollouts.observations[0].copy_(CurrentObs())
    rollouts.target_observations[0].copy_(CurrentObsTarget())

    if args.cuda:
        CurrentState.cuda()
        CurrentStateTarget.cuda()
        CurrentObs.cuda()
        CurrentObsTarget.cuda()
        rollouts.cuda()
        pi.cuda()


    MAX_REWARD = -99999
    for j in range(args.num_updates):
        exploration(pi, CurrentState, CurrentStateTarget, CurrentObs, CurrentObsTarget, rollouts, args, result, env)
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
        if not args.no_test and j % args.test_interval < nt and j > args.test_thresh:
        # if not args.no_test and j % args.test_interval < nt:
            if j % args.test_interval == 0:
                print('-'*45)
                print('Testing {} episodes'.format(args.num_test))

            pi.cpu()
            sd = pi.cpu().state_dict()
            test_reward_list, BestVideo = Test_and_Save_Video(test_env, testtargets,  CombinePolicy, sd, args)

            test_reward_list = np.array(test_reward_list)
            test_reward = test_reward_list.mean()

            result.update_test(test_reward_list)

            # Plot result
            print('Average Test Reward: {}\n '.format(round(test_reward)))
            if args.vis:
                vis.line_update(Xdata=frame,
                                Ydata=test_reward, name='Test Score')
                # vis.scatter_update(Xdata=frame,
                #                 Ydata=test_reward, name='Test Score Scatter')
            #  ==== Save best model ======
            if test_reward > MAX_REWARD:
                print('--'*45)
                print('New High Score!\n')
                print('Avg. Reward: ', test_reward)
                name = os.path.join(args.result_dir,
                    'BestVideo_targets{}_{}.pt'.format(round(test_reward, 1), frame))
                print('Saving Best Video')
                torch.save(BestVideo, name)
                name = os.path.join(
                    args.checkpoint_dir,
                    'BestDictCombi{}_{}.pt'.format(frame, round(test_reward, 3)))
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
