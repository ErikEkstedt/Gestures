from itertools import count
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
# from OpenGL import GLU # fix for opengl issues on desktop  / nvidia


try:
    from memory import StackedState
except:
    from Agent.memory import StackedState


def test_existing_env(env, Model, state_dict, args, verbose=False):
    ''' Uses existing environment '''
    # == Environment
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]
    CurrentState = StackedState(1, args.num_stack, ob_shape)

    # == Model
    pi = Model(ob_shape, ac_shape)
    pi = Model(CurrentState.state_shape,
               ac_shape,
               hidden=args.hidden)
    pi.load_state_dict(state_dict)

    # Testing
    total_reward = []
    episode_reward = 0
    for i in range(args.num_test):
        state = env.reset()
        while True:
            CurrentState.update(state)
            value, action = pi.act(CurrentState())
            cpu_actions = action.data.cpu().numpy()[0]
            state, reward, done, info = env.step(cpu_actions)
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                total_reward.append(episode_reward)
                episode_reward = 0
                break

    return np.array(total_reward).mean()

def test(Env, Model, state_dict, args, verbose=False):
    '''Creates new env each time '''

    # == Environment
    env = Env()
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]
    CurrentState = StackedState(1, args.num_stack, ob_shape)

    # == Model
    pi = Model(ob_shape, ac_shape)
    pi = Model(CurrentState.state_shape,
               ac_shape,
               hidden=args.hidden)

    pi.load_state_dict(state_dict)

    # Testing
    total_reward, episode_reward = 0, 0
    for i in range(args.num_test):
        state = env.reset()
        for j in count(1):
            CurrentState.update(state)

            value, action, _, _ = pi.act(CurrentState())
            cpu_actions = action.data.cpu().numpy()[0]
            state, reward, done, info = env.step(cpu_actions)
            total_reward += reward
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                episode_reward = 0
                done = False
                break
    return total_reward/args.num_test


def test_and_render(env, Model, state_dict, args, verbose=False):
    ''' Uses existing environment '''
    # == Environment
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]
    CurrentState = StackedState(1, args.num_stack, ob_shape)

    # == Model
    pi = Model(ob_shape, ac_shape)
    pi = Model(CurrentState.state_shape,
               ac_shape,
               hidden=args.hidden)
    pi.load_state_dict(state_dict)

    # Testing
    total_reward = []
    episode_reward = 0
    state = env.reset()
    while True:
        CurrentState.update(state)
        value, action = pi.act(CurrentState())
        cpu_actions = action.data.cpu().numpy()[0]
        state, reward, done, info = env.step(cpu_actions)
        env.render()
        episode_reward += reward
        if done:
            if verbose: print(episode_reward)
            total_reward.append(episode_reward)
            episode_reward = 0
            break


# Video
def make_video(vid, filenname):
    print('-'*50)
    print('Making Video')
    fig = plt.figure()
    ims = []
    for frame in tqdm(vid):
        im = plt.imshow(frame, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)
    ani.save(filenname)


def Test_and_Save_Video(test_env, Model, state_dict, args, verbose=False):
    '''
    Test with video
    :param pi - The policy playing
    :param args - arguments

    :output      - Average complete episodic reward
    '''
    # == Model
    ob_shape = test_env.observation_space.shape[0]
    ac_shape = test_env.action_space.shape[0]
    CurrentState = StackedState(1, args.num_stack, ob_shape)

    pi = Model(ob_shape, ac_shape)
    pi = Model(CurrentState.state_shape,
               ac_shape,
               hidden=args.hidden)
    pi.load_state_dict(state_dict)
    # Test environments
    total_reward, episode_reward, best_episode_reward = 0, 0, -999
    Video = []
    for i in range(args.num_test):
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
                # if episode_reward > best_episode_reward:
                #     BestVideo = Video
                #     best_episode_reward = episode_reward
                # Video = []
                episode_reward = 0
                done = False
                break
    # return total_reward/args.num_test, BestVideo
    return total_reward/args.num_test, Video

def main():
    import torch
    from arguments import get_args
    from model import MLPPolicy
    from environments.custom_reacher import CustomReacher2DoF

    args = get_args()
    args.hidden=128
    sd = torch.load(args.load_file)
    test(CustomReacher2DoF, MLPPolicy, sd, args, True)


if __name__ == '__main__':
    main()
