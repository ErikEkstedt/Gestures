import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from itertools import count
import os
import gym
import numpy as np
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import roboschool


from memory import RolloutStorage, StackedState
from arguments import FakeArgs, get_args
from AgentRobo import AgentRoboSchool
from environment import Social_Torso

from training import Exploration, Training
from test import test, test_and_render

# ---------------------
def log_print(agent, dist_entropy, value_loss, floss, action_loss, j):
    print("\nUpdate: {}, frames:    {} \
          \nAverage final reward:   {}, \
          \nentropy:                {:.4f}, \
          \ncurrent value loss:     {:.4f}, \
          \ncurrent policy loss:    {:.4f}".format(j,
                (j + 1) * agent.args.num_steps * agent.args.num_processes,
                agent.final_rewards.mean(),
                -dist_entropy.data[0],
                value_loss.data[0],
                action_loss.data[0],))

def description_string(args):
    '''Some useful descriptions for the logger/visualizer'''
    slist = []
    slist.append('AgentPepper')
    slist.append('\nSteps: ' + str(args.num_steps))
    slist.append('\nEpoch: ' + str(args.ppo_epoch))
    slist.append('\nlr: ' + str(args.pi_lr))
    slist.append('\nFixed std: ' + str(args.fixed_std))
    slist.append('\nStd(if fixed): ' + str(args.std))
    slist.append('\nTotal frames: ' + str(args.num_frames))
    slist.append('\nTest Render: ' + str(args.test_render))
    slist.append('\nTest iters: ' + str(args.num_test))
    slist.append('\nmax test length: ' + str(args.max_test_length))
    slist.append('\nNo-Test: ' + str(args.no_test))
    slist.append('\nVis-interval: ' + str(args.vis_interval))
    slist.append('\nTest-interval: ' + str(args.test_interval))
    slist.append('\n\n\n\n')
    return slist

def print_ds(l):
    for i in l:
        print(i)

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, "{}.monitor.json".format(rank)))
        return env
    return _thunk

def make_social_torso(seed, rank, log_dir):
    def _thunk():
        env = gym.make('RoboschoolReacher-v1')
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, "{}.monitor.json".format(rank)))
        return env
    return _thunk


def main():
    args = get_args()  # Real argparser
    ds = description_string(args)
    print_ds(ds)


    if args.vis:
        from vislogger import VisLogger
        ds = description_string(args)
        # Text is not pretty
        vis = VisLogger(description_list=ds, log_dir=args.log_dir)

    # == Environment ========
    monitor_log_dir = "/tmp/"
    env_id = 'RoboschoolReacher-v1'

    env = SubprocVecEnv([
        make_env(env_id, args.seed, i, monitor_log_dir)
        for i in range(args.num_processes)])

    # env = SubprocVecEnv([
    #     make_social_torso(args.seed, i, monitor_log_dir)
    #     for i in range(args.num_processes)])


    state_shape = env.observation_space.shape
    stacked_state_shape = (state_shape[0] * args.num_stack,)
    action_shape = env.action_space.shape

    # memory
    memory = RolloutStorage(args.num_steps,
                            args.num_processes,
                            stacked_state_shape,
                            action_shape)


    CurrentState = StackedState(args.num_processes,
                                args.num_stack,
                                state_shape,
                                args.cuda)

    # ====== Agent ==============
    torch.manual_seed(args.seed)
    agent = AgentRoboSchool(args,
                    stacked_state_shape=stacked_state_shape,
                    action_shape=action_shape,
                    hidden=args.hidden,
                    fixed_std=False,
                    std=0.5)

    agent.state_shape = state_shape     # Save non-stack state-shape for testing
    VLoss = nn.MSELoss()                     # Value loss function

    agent.final_rewards = torch.zeros([args.num_processes, 1])   # total episode reward
    agent.episode_rewards = torch.zeros([args.num_processes, 1]) # tmp episode reward
    agent.num_done = torch.zeros([args.num_processes, 1])        # how many finished episode, resets on plot
    agent.std = []                                               # list to hold all [action_mean, action_std]

    #  ==== RESET ====
    s = env.reset()
    CurrentState.update(s)
    memory.states[0] = CurrentState()

    if args.cuda:
        agent.cuda()
        CurrentState.cuda()
        memory.cuda()

    agent.CurrentState = CurrentState
    agent.memory = memory

    # ==== Training ====
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    print('-'*55)
    print()
    print('Starting training {} frames in {} updates.\n\n\
            Batch size {}\tStep size {}\tStack {}'.format(
            args.num_frames, num_updates, args.batch_size, args.num_steps, args.num_stack))

    floss_total = 0
    vloss_total = 0
    ploss_total = 0
    ent_total= 0

    for j in range(num_updates):
        Exploration(agent, env)  # Explore the environment for args.num_steps
        value_loss, action_loss, dist_entropy = Training(agent, VLoss, verbose=False)  # Train models for args.ppo_epoch

        vloss_total += value_loss
        ploss_total += action_loss
        ent_total += dist_entropy

        agent.memory.last_to_first() #updates rollout memory and puts the last state first.

        #  ==== LOG ======
        if j % args.log_interval == 0: log_print(agent, dist_entropy, value_loss, 1, action_loss, j)

        if j % args.vis_interval == 0 and j is not 0 and not args.no_vis:
            frame = (j + 1) * args.num_steps * args.num_processes

            if not args.no_test and j % args.test_interval == 0:
                ''' TODO
                Fix so that resetting the environment does not
                effect the data. Equivialent to `done` ?
                should be the same.'''
                print('Testing')
                test_reward = test(agent, runs=10)
                vis.line_update(Xdata=frame, Ydata=test_reward, name='Test Score')
                print('Done testing')
                if args.test_render:
                    print('RENDER')
                    test_and_render(agent)

                #  ==== RESET ====

            vloss_total /= args.vis_interval
            ploss_total /= args.vis_interval
            ent_total   /= args.vis_interval

            # Take mean b/c several processes
            # Training score now resets to zero alot.
            R = agent.final_rewards
            R = R.mean()
            if abs(R) > 0 :
                # when reward mean is exactly zero it does not count.
                vis.line_update(Xdata=frame, Ydata=R, name='Training Score')

            std = torch.Tensor(agent.std).mean()

            # Draw plots
            vis.line_update(Xdata=frame, Ydata=vloss_total, name='Value Loss')
            vis.line_update(Xdata=frame, Ydata=ploss_total, name='Policy Loss')
            vis.line_update(Xdata=frame, Ydata=std, name='Action std')
            vis.line_update(Xdata=frame, Ydata=-ent_total, name='Entropy')

            # reset
            floss_total = 0
            vloss_total = 0
            ploss_total = 0
            ent_total= 0
            del agent.std[:]
            agent.num_done = 0
            agent.final_rewards = 0


    print('saving')
    agent.cpu()
    agent.policy.cpu()
    torch.save(agent.policy.state_dict(), 'model.pt')


if __name__ == '__main__':
    main()
