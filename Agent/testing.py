import torch
import gym
from itertools import count
from memory import StackedState

def Test(pi, args, ob_shape, verbose=False):
    '''Test
    :param pi - The policy playing
    :param args - arguments

    :output      - Average complete episodic reward
    '''
    # Use only 1 processor for render
    TestState = StackedState(1,
                             args.num_stack,
                             ob_shape)

    if args.cuda:
        TestState.cuda()
    # Test environments
    test_env = gym.make(args.env_id)
    total_reward, episode_reward = 0, 0
    for i in range(args.num_test):
        TestState.reset()
        state = test_env.reset()
        for j in count(1):
            # Update current state
            TestState.update(state)

            # Sample actions
            value, action, _, _ = pi.sample(TestState(), deterministic=True)
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
