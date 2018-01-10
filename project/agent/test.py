from itertools import count
import numpy as np
import torch
from tqdm import tqdm, trange
from project.agent.memory import StackedState, StackedObs

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
    ob_shape = test_env.observation_space.shape
    st_shape = test_env.state_space.shape
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


def Test_and_Save_Video_Combi(test_env, testset, Model, state_dict, args, verbose=False):
    '''
    WARNING: Init values for CombinePolicy are hardcoded here....

    Test with video
    :param test_env   - Reacher/HUmanoid environment
    :param testset    - Dataset of targets
    :param Model      - The policy network
    :param state_dict - nn.Module.state_dict
    :param verbose    - Boolean, be verbose

    :output           - Float, average complete episodic reward
    :output           - List, containing all videoframes
    '''

    test_env.seed(np.random.randint(0,2000))

    # === Target dims ===
    ob_sample, st_sample, = testset[4]  #random index
    ob_target_shape = ob_sample.shape
    st_target_shape = st_sample.shape[0]

    # == Model
    st_shape = test_env.state_space.shape[0]    # Joints state
    ob_shape = test_env.observation_space.shape # RGB
    ac_shape = test_env.action_space.shape[0]   # Actions

    CurrentState       = StackedState(1, args.num_stack, st_shape)
    CurrentStateTarget = StackedState(1, args.num_stack, st_target_shape)
    CurrentObs         = StackedObs(1, args.num_stack, ob_shape)
    CurrentObsTarget   = StackedObs(1, args.num_stack, ob_target_shape)

    pi = Model(o_shape=CurrentObs.obs_shape,
               o_target_shape=CurrentObs.obs_shape,
               s_shape=st_shape,
               s_target_shape=st_target_shape,
               a_shape=ac_shape,
               feature_maps=[64, 64, 8],
               kernel_sizes=[5, 5, 5],
               strides=[2, 2, 2],
               args=args)

    pi.load_state_dict(state_dict)
    # Test environments
    total_reward, episode_reward, best_episode_reward = 0, 0, -9999
    Video, Targets = [], []
    for i in trange(args.num_test):
        state, s_target, obs, o_target = test_env.reset()
        Targets.append((o_target, s_target))
        CurrentObsTarget.update(o_target)
        CurrentStateTarget.update(s_target)
        for j in count(1):
            CurrentObs.update(obs)
            CurrentState.update(state)

            value, action = pi.act(CurrentObs(),
                                   CurrentObsTarget(),
                                   CurrentState(),
                                   CurrentStateTarget())
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            state, _, obs, _, reward, done, info = test_env.step(cpu_actions)
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
    test_env.close()
    del test_env
    return total_reward/args.num_test, [Video, Targets]