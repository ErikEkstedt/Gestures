from itertools import count
import numpy as np
import torch
from tqdm import tqdm, trange

from memory import StackedState, StackedObs

def Test_and_Save_Video(test_env, Model, state_dict, args, verbose=False):
    '''
    Test with video
    :param test_env   - Reacher/Humanoid environment
    :param Model      - The policy network
    :param state_dict - nn.Module.state_dict
    :param verbose    - Boolean, be verbose

    :output           - Float, average complete episodic reward
    :output           - List, containing all videoframes
    '''
    # == Model ==
    ob_shape = test_env.observation_space.shape[0]
    ac_shape = test_env.action_space.shape[0]
    CurrentState = StackedState(1, args.num_stack, ob_shape)

    pi = Model(CurrentState.state_shape, ac_shape, args)
    pi.load_state_dict(state_dict)

    # == Test ==
    total_reward, episode_reward, best_episode_reward = 0, 0, -999
    Video = []
    for i in trange(args.num_test):
        state, obs = test_env.reset()
        for j in count(1):
            CurrentState.update(state)
            value, action = pi.act(CurrentState())
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            state, rgb, reward, done, info = test_env.step(cpu_actions)
            if j % 2 == 0:
                Video.append(rgb)

            # If done then update final rewards and reset episode reward
            total_reward += reward
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                episode_reward = 0
                done = False
                break
    return total_reward/args.num_test, Video

def Test_and_Save_Video_RGB(test_env, Model, state_dict, args, verbose=False):
    '''
    WARNING: Init values for CNNPolicy are hardcoded here....

    Test with video
    :param test_env   - Reacher/HUmanoid environment
    :param Model      - The policy network
    :param state_dict - nn.Module.state_dict
    :param verbose    - Boolean, be verbose

    :output           - Float, average complete episodic reward
    :output           - List, containing all videoframes
    '''
    # == Model
    st_shape = test_env.observation_space.shape
    ob_shape = test_env.rgb_space.shape
    ac_shape = test_env.action_space.shape[0]

    CurrentObs = StackedObs(1, args.num_stack, ob_shape)
    pi = Model(input_shape=CurrentObs.obs_shape,
               action_shape=ac_shape,
               in_channels=CurrentObs.obs_shape[0],
               feature_maps=[64, 64, 64],
               kernel_sizes=[5, 5, 5],
               strides=[2, 2, 2],
               args=args)

    pi.load_state_dict(state_dict)
    # Test environments
    total_reward, episode_reward, best_episode_reward = 0, 0, -999
    Video = []
    for i in trange(args.num_test):
        state, obs = test_env.reset()
        for j in count(1):
            CurrentObs.update(obs)

            value, action = pi.act(CurrentObs())
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            state, obs, reward, done, info = test_env.step(cpu_actions)
            if j % 2 == 0:
                Video.append(obs)

            # If done then update final rewards and reset episode reward
            total_reward += reward
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                episode_reward = 0
                done = False
                break
    return total_reward/args.num_test, Video
