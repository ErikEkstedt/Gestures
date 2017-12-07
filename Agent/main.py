import numpy as np
import gym
import roboschool

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from utils import args_to_list, print_args, log_print
from arguments import FakeArgs, get_args
from model import MLPPolicy
from memory import RolloutStorage, StackedState, Results
from training import Training, Exploration
from testing import Test


def make_gym(env_id, seed, num_processes):
    ''' imports SubprocVecEnv from baselines.
    :param seed                 int
    :param num_processes        int, # env
    '''
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    def multiple_envs(env_id, seed, rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        return _thunk
    return SubprocVecEnv([multiple_envs(env_id, seed, i) for i in range(num_processes)])


def main():
    args = get_args()  # Real argparser
    ds = print_args(args)

    if args.vis:
        from vislogger import VisLogger
        vis = VisLogger(description_list=ds, log_dir=args.log_dir)

    # args.env_id = 'RoboschoolReacher-v1'
    env = make_gym(args.env_id, args.seed, args.num_processes)

    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]

    CurrentState = StackedState(args.num_processes,
                                args.num_stack,
                                ob_shape)

    rollouts = RolloutStorage(args.num_steps,
                              args.num_processes,
                              CurrentState.size()[1],
                              ac_shape)

    result = Results(max_n=200, max_u=10)

    pi = MLPPolicy(CurrentState.state_shape, ac_shape, hidden=64)
    optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)

    s = env.reset()
    CurrentState.update(s)
    rollouts.states[0].copy_(CurrentState())

    if args.cuda:
        CurrentState.cuda()
        rollouts.cuda()
        pi.cuda()

    # ==== Training ====
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    print('Updates: ', num_updates)

    MAX_REWARD = -999999
    for j in range(num_updates):
        Exploration(pi, CurrentState, rollouts, args, result, env)
        vloss, ploss, ent = Training(pi, args, rollouts, optimizer_pi)
        rollouts.last_to_first() #updates rollout memory and puts the last state first.

        result.update_loss(vloss.data, ploss.data, ent.data)
        frame = (j +1) * args.num_steps * args.num_processes

        #  ==== SHELL LOG ======
        if j % args.log_interval == 0 and j > 0 :
            v, p, e = result.get_loss_mean()
            print('Steps: ', frame)
            print('Rewards: ', result.get_reward_mean())
            print('Value loss: ', v)
            print('Policy loss: ', p)
            print('Entropy: ', e)
            print()

        #  ==== TEST ======
        if not args.no_test and j % args.test_interval == 0 and j>0:
            print('Testing {} episodes'.format(args.num_test))
            test_reward = Test(pi, args, ob_shape, verbose=False)
            vis.line_update(Xdata=frame, Ydata=test_reward, name='Test Score')
            print('Done testing\n')

            #  ==== Save best model ======
            if test_reward > MAX_REWARD:
                print('Saving best latest episode score')
                print('Reward: ', test_reward)
                name = 'model_tmp_best%.2f'%test_reward+'.pt'
                print(name)
                torch.save(pi.cpu().state_dict(), 'trained_models/tmp_best/'+name)
                pi.cuda()
                MAX_REWARD = test_reward
                print('MAX:', MAX_REWARD)
                print()

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
            # vis.line_update(Xdata=frame, Ydata=std, name ='Action std')
            vis.line_update(Xdata=frame, Ydata=-e,  name ='Entropy')

        #  ==== Save model ======
        if j % args.save_interval == 0 and j > 0:
            R = result.get_reward_mean()
            print('Interval Saving')
            name = 'model%d_%.2f.pt'%(j+1, R)
            print(name)
            torch.save(pi.cpu().state_dict(), 'trained_models/'+name)
            pi.cuda()


if __name__ == '__main__':
    main()
