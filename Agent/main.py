import numpy as np
import gym
# import roboschool
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from utils import args_to_list, print_args, log_print
from arguments import FakeArgs, get_args
from model import MLPPolicy
from memory import RolloutStorage, StackedState, Results
from training import Training, Exploration
from testing import Test, Test_and_Save_Video,Test_and_See_gym

# from environments.custom import HalfHumanoid, make_parallel_environments
# from environments.custom import CustomReacher2d_2arms


def make_env(env_id, seed, num_processes):
    ''' imports SubprocVecEnv from baselines.
    :param seed                 int
    :param num_processes        int, # env
    '''
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    import baselines.bench as bench
    def make_envs(env_id, seed, rank, log_dir='/tmp/'):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            # env = bench.Monitor(env, os.path.join(log_dir, "{}.monitor.json".format(rank)))
            return env
        return _thunk
    return SubprocVecEnv([make_envs(env_id, seed, i)
                          for i in range(num_processes)])


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
    return SubprocVecEnv([multiple_envs(env_id, seed, i)
                          for i in range(num_processes)])

class Optimizer(object):
    def __init__(self, lr, opt, total_len):
        self.optim = opt
        self.lr = lr
        self.final_lr = lr/100

    def adjust_learning_rate(update):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def main():
    args = get_args()  # Real argparser
    ds = print_args(args)

    if args.vis:
        from vislogger import VisLogger
        vis = VisLogger(description_list=ds, log_dir=args.log_dir)
        _, checkpoint_dir = vis.get_logdir()

    # env = make_gym(args.env_id, args.seed, args.num_processes)
    env = make_env(args.env_id, args.seed, args.num_processes)
    test_env = gym.make(args.env_id)

    ob_shape = test_env.observation_space.shape[0]
    ac_shape = test_env.action_space.shape[0]

    # Env = HalfHumanoid
    # Env = CustomReacher2d_2arms
    # env = make_parallel_environments(Env, args.seed, args.num_processes)
    # test_env = Env()

    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]
    print(ob_shape)
    print(ac_shape)

    CurrentState = StackedState(args.num_processes,
                                args.num_stack,
                                ob_shape)

    rollouts = RolloutStorage(args.num_steps,
                              args.num_processes,
                              CurrentState.size()[1],
                              ac_shape)

    result = Results(max_n=200, max_u=10)

    pi = MLPPolicy(CurrentState.state_shape,
                   ac_shape,
                   hidden=args.hidden,
                   total_frames=args.num_frames)

    pi.train()

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
            R = result.get_reward_mean()
            print('Interval Saving')
            fname = 'model%d_%.2f.pt'%(j+1, R)
            name = os.path.join(checkpoint_dir, fname)
            print(name)
            torch.save(pi.cpu().state_dict(), name)
            name = os.path.join(args.save_dir, fname)
            torch.save(pi, name)
            print(name)
            pi.cuda()
            # input()

if __name__ == '__main__':
    main()
