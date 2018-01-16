from itertools import count
import os
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from project.agent.memory import StackedState, StackedObs, Current


def record(env, writer, scale=(15,10)):
    human, _, target = env.render('all_rgb_array')  # (W, H, C)
    height, width = target.shape[:2]
    target = cv2.resize(target,(scale[0]*width, scale[1]*height),
                        interpolation = cv2.INTER_CUBIC)
    # target: (40,40,3) -> (3, 600,400)
    # human: (600,400, 3) -> (3, 600,400)
    target = target.transpose((2,0,1))
    human = human.transpose((2,0,1))
    imglist = [torch.from_numpy(human), torch.from_numpy(target)]
    img = make_grid(imglist, padding=5).numpy()
    img = img.transpose((1,2,0))
    writer.writeFrame(img)


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
    ot_shape = ob_sample.shape
    st_shape = st_sample.shape[0]

    # == Model
    st_shape = test_env.state_space.shape[0]    # Joints state
    ob_shape = test_env.observation_space.shape # RGB
    ac_shape = test_env.action_space.shape[0]   # Actions

    CurrentState       = StackedState(1, args.num_stack, st_shape)
    CurrentStateTarget = StackedState(1, args.num_stack, st_shape)
    CurrentObs         = StackedObs(1, args.num_stack, ob_shape)
    CurrentObsTarget   = StackedObs(1, args.num_stack, ot_shape)

    pi = Model(o_shape=CurrentObs.obs_shape,
               ot_shape=CurrentObs.obs_shape,
               s_shape=st_shape,
               st_shape=st_target_shape,
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


def Test_and_Save_Video_Social(env, testset, Model, state_dict, args, frames):
    '''
    Test with video
    :param env   - Reacher/HUmanoid environment
    :param testset    - Dataset of targets
    :param Model      - The policy network
    :param state_dict - nn.Module.state_dict
    :param verbose    - Boolean, be verbose
    '''
    if args.record:
        import skvideo.io
        name = "{}-test_frame{}.mp4".format(args.env_id, frames)
        name = os.path.join(args.result_dir, name)
        writer = skvideo.io.FFmpegWriter(name)

    # === Target dims ===
    st_sample, ob_sample = testset[4]  #random index
    ot_shape = ob_sample.shape
    st_shape = st_sample.shape[0]

    # == Model
    s_shape = env.state_space.shape[0]    # Joints state
    o_shape = env.observation_space.shape # RGB
    ac_shape = env.action_space.shape[0]   # Actions

    current = Current(num_processes=1,
                      num_stack=args.num_stack,
                      state_dims=s_shape,
                      starget_dims=st_shape,
                      obs_dims=o_shape,
                      otarget_dims=ot_shape)

    pi = Model(o_shape=current.obs.obs_shape,
               ot_shape=current.obs.obs_shape,
               s_shape=s_shape,
               st_shape=st_shape,
               a_shape=ac_shape,
               feature_maps=args.feature_maps,
               kernel_sizes=args.kernel_sizes,
               strides=args.strides,
               args=args)

    pi.load_state_dict(state_dict)

    total_reward = 0
    for i in trange(args.num_test):
        idx = np.random.randint(0,len(testset)) # random target
        env.set_target(testset[idx])
        state, s_target, obs, o_target = env.reset()
        for j in count(1):
            current.update(state, s_target, obs, o_target)

            s, st, o, ot = current()
            value, action = pi.act(s, st, o, ot)
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            state, s_target, obs, o_target, reward, done, info = env.step(cpu_actions)

            if args.record and j % 2 == 0: # every other frame
                record(env, writer)

            # If done then update final rewards and reset episode reward
            total_reward += reward
            if done:
                break

    if args.record:
        writer.close()
    return total_reward/args.num_test
