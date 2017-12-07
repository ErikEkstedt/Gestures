from OpenGL import GLU # fix for opengl issues on desktop  / nvidia
import argparse
import gym
import roboschool
import math

import torch
from itertools import count
from memory import StackedState

from model import MLPPolicy


def get_args():
    parser = argparse.ArgumentParser(description='Test PPOAgent')
    parser.add_argument('--env-id', default='RoboschoolReacher-v1',
                        help='Environment used (default: RoboschoolReacher-v1)')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed (default: 10)')
    parser.add_argument('--load-file',
                        default='trained_models/tmp_best/model_tmp_best37.56.pt',
                        help='Policy data')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden neurons in policy (default: 128)')
    parser.add_argument('--num-test', type=int, default=100,
                        help='Number of test episodes (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-deterministic', action='store_false', default=True,
                        help='Do not test deterministically')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def Test():
    args = get_args()
    env = gym.make(args.env_id)
    env.seed(args.seed)
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]

    print(args.load_file)
    stored_state_dict = torch.load(args.load_file)

    pi = MLPPolicy(CurrentState.state_shape, ac_shape, hidden=args.hidden)
    pi.load_state_dict(stored_state_dict)

    model = torch.load(args.load_file)

    CurrentState = StackedState(1, args.num_stack, ob_shape)

    if args.cuda:
        CurrentState.cuda()
        pi.cuda()

    # Test environments
    env = gym.make(args.env_id)
    total_reward, episode_reward = 0, 0
    for i in range(args.num_test):
        CurrentState.reset()
        state = test_env.reset()
        for j in count(1):
            # Update current state
            CurrentState.update(state)

            # Sample actions
            value, action, _, _ = pi.sample(CurrentState(), deterministic=True)
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            state, reward, done, info = test_env.step(cpu_actions)

            # If done then update final rewards and reset episode reward
            total_reward += reward
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                episode_reward = 0
                done = False
                break
    return total_reward/args.num_test

def Load_and_Test():
    args = get_args()
    env = gym.make(args.env_id)
    env.seed(args.seed)
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]


    CurrentState = StackedState(1, args.num_stack, ob_shape)
    print(args.load_file)
    stored_state_dict = torch.load(args.load_file)

    pi = MLPPolicy(CurrentState.state_shape, ac_shape, hidden=args.hidden)
    pi.load_state_dict(stored_state_dict)

    if args.cuda:
        CurrentState.cuda()
        pi.cuda()

    input('Start')
    total_reward = 0
    for i in range(args.num_test):
        CurrentState.reset()
        s = env.reset()
        episode_reward = 0
        while True:
            CurrentState.update(s)
            value, action, _, _ = pi.sample(CurrentState(),
                                            deterministic=args.no_deterministic)
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            state, reward, done, info = env.step(cpu_actions)

            # If done then update final rewards and reset episode reward
            episode_reward += reward
            if done:
                total_reward += episode_reward
                print('Episode Reward:', episode_reward)
                episode_reward = 0
                done = False
                break

    print('Total reward: ',total_reward/args.num_test)


def Load_and_Test2():
    args = get_args()
    env = gym.make(args.env_id)
    env.seed(args.seed)
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]


    CurrentState = StackedState(1, args.num_stack, ob_shape)
    print(args.load_file)
    stored_state_dict = torch.load(args.load_file)

    # pi = MLPPolicy(CurrentState.state_shape, ac_shape, hidden=args.hidden)
    pi = torch.load(args.load_file)
    # pi.load_state_dict(stored_state_dict)

    if args.cuda:
        CurrentState.cuda()
        pi.cuda()

    input('Start')
    total_reward = 0
    for i in range(args.num_test):
        CurrentState.reset()
        s = env.reset()
        episode_reward = 0
        while True:
            CurrentState.update(s)
            value, action, _, _ = pi.sample(CurrentState(),
                                            deterministic=args.no_deterministic)
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            state, reward, done, info = env.step(cpu_actions)

            # If done then update final rewards and reset episode reward
            episode_reward += reward
            if done:
                total_reward += episode_reward
                print('Episode Reward:', episode_reward)
                episode_reward = 0
                done = False
                break

    print('Total reward: ',total_reward/args.num_test)


if __name__ == '__main__':
    Load_and_Test()
    # Load_and_Test2()


