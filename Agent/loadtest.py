import argparse
import gym
import math

import torch
from itertools import count
from memory import StackedState

from model import MLPPolicy

from environments.custom_reacher import CustomReacher

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

def Load_and_Test():
    args = get_args()

    torch.manual_seed(args.seed)
    env = CustomReacher()
    env.seed(args.seed)
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]

    print('num_stack:', args.num_stack)
    CurrentState = StackedState(1, args.num_stack, ob_shape)
    print(args.load_file)
    saved_state_dict = torch.load(args.load_file)

    pi = MLPPolicy(CurrentState.state_shape, ac_shape, hidden=args.hidden)
    pi.load_state_dict(saved_state_dict)

    total_reward = 0
    for i in range(args.num_test):
        CurrentState.reset()
        input('Reset')
        state = env.reset()
        episode_reward = 0
        while True:
            CurrentState.update(state)
            value, action, _, _ = pi.sample(CurrentState(), deterministic=True)
            cpu_actions = action.data.cpu().numpy()[0]
            print(cpu_actions)

            # Observe reward and next state
            state, reward, done, info = env.step(cpu_actions)
            env.render()

            # If done then update final rewards and reset episode reward
            episode_reward += reward
            if done:
                print('Episode Reward:', episode_reward)
                total_reward += episode_reward
                done = False
                break

    print('Total reward: ', total_reward/args.num_test)


if __name__ == '__main__':
    Load_and_Test()
