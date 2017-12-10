import numpy as np
import gym
from copy import deepcopy
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from utils import args_to_list, print_args, log_print, make_parallel_environments
from arguments import FakeArgs, get_args
from model import MLPPolicy
from memory import RolloutStorage, StackedState, Results
from train import Training
from train import Exploration_single as Exploration

from environments.custom_reacher import CustomReacher

def multiprocess():
    p = mp.Process(target=test, args=(params.num_processes, params, shared_model, shared_obs_stats, test_n))
    p.start()
    processes.append(p)
    p = mp.Process(target=chief, args=(params.num_processes, params, traffic_light, counter, shared_model, shared_grad_buffers, optimizer))
    p.start()
    processes.append(p)
    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, traffic_light, counter, shared_model, shared_grad_buffers, shared_obs_stats, test_n))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def main():
    args = get_args()  # Real argparser
    ds = print_args(args)

    if args.vis:
        from vislogger import VisLogger
        vis = VisLogger(description_list=ds, log_dir=args.log_dir)
        args.log_dir, args.video_dir, args.checkpoint_dir = vis.get_logdir()

    if args.num_processes == 1:
        print('Not made for single processes')
        args.num_processes = 4

    env = make_parallel_environments(CustomReacher, args.seed, args.num_processes)
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

    s = env.reset()
    CurrentState.update(s)
    rollouts.states[0].copy_(CurrentState())

    print('CurrentState(): ', CurrentState())
    print('CurrentState().size(): ', CurrentState().size())
    print('CurrentState.size() ', CurrentState.size())
    print()
    print(pi)
    # print('pi.sample(CurrentState())', pi.sample(CurrentState()))
    input()



    if args.cuda:
        CurrentState.cuda()
        rollouts.cuda()
        pi.cuda()

    # ==== Training ====
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    print('Updates: ', num_updates)

    MAX_REWARD = -999999
    rgb_list = []
    for j in range(num_updates):
        Exploration(pi, CurrentState, rollouts, args, result, env)
        vloss, ploss, ent = Training(pi, args, rollouts, optimizer_pi)

        rollouts.last_to_first()
        result.update_loss(vloss.data, ploss.data, ent.data)
        frame = (j + 1) * args.num_steps * args.num_processes

        #  ==== SHELL LOG ======
        if j % args.log_interval == 0 and j > 0:
            v, p, e = result.get_loss_mean()
            print('Steps: ', frame)
            print('Rewards: ', result.get_reward_mean())
            print('Value loss: ', v)
            print('Policy loss: ', p)
            print('Entropy: ', e)
            print()

        #  ==== TEST ======
        if not args.no_test and j % args.test_interval == 0 and j > 0:
            print('Testing {} episodes'.format(args.num_test))
            pi.eval()
            R = Test(pi, args, ob_shape, verbose=True)
            # R = Test_and_See_gym(test_env, pi, args, ob_shape, verbose=True)
            # R = Test_and_Save_Video(test_env, pi, args, ob_shape, verbose=False)
            pi.train()
            vis.line_update(Xdata=frame, Ydata=R, name='Test Score')
            print('Test Average:', R)

            #  ==== Save best model ======
            # if test_reward > MAX_REWARD:
            if True:
                print('--'*45)
                print('Saving after test')
                print('Reward: ', R)
                name = os.path.join(checkpoint_dir, 'model_best'+str(R))
                print(name)
                torch.save(pi, name + '.pt')
                torch.save(pi.state_dict(), name+'dict_cuda.pt')
                pi.cpu()
                torch.save(pi, name+'cpu.pt')
                torch.save(pi.state_dict(), name+'dict_cuda.pt')
                pi.cuda()
                MAX_REWARD = R
                print('MAX:', MAX_REWARD)
                print()
                pi.train()
                # input('Enter to continue')

        #  ==== VISDOM PLOT ======
        if j % args.vis_interval == 0 and j > 0 and not args.no_vis:
            R = result.get_reward_mean()
            vis.line_update(Xdata=frame,
                            Ydata=R,
                            name='Training Score')

            # Draw plots
            v, p, e = result.get_loss_mean()
            vis.line_update(Xdata=frame, Ydata=v,   name ='Value Loss')
            vis.line_update(Xdata=frame, Ydata=p,   name ='Policy Loss')
            vis.line_update(Xdata=frame, Ydata=pi.get_std(), name ='Action std')
            vis.line_update(Xdata=frame, Ydata=-e,  name ='Entropy')

        #  ==== Save model ======
        if j % args.save_interval == 0 and j > 0:
            R = result.get_last_reward()
            print('Interval Saving (last score: ', R)
            fname = 'state_dict%d_%.2f.pt'%(j+1, R)
            name = os.path.join(args.checkpoint_dir, fname)
            print(name)
            sd = pi.cpu().state_dict()
            torch.save(sd, name)
            pi.cuda()

if __name__ == '__main__':
    main()
