from OpenGL import GLU # fix for opengl issues on desktop  / nvidia
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
    TestState = StackedState(1, args.num_stack, ob_shape)

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

def Test_and_See(pi, args, ob_shape, verbose=False):
    '''Test
    :param pi - The policy playing
    :param args - arguments

    :output      - Average complete episodic reward
    '''
    # Use only 1 processor for render
    TestState = StackedState(1, args.num_stack, ob_shape)
    if args.cuda:
        TestState.cuda()
    # Test environments
    test_env = gym.make(args.env_id)

    total_reward, episode_reward, best_episode_reward = 0, 0, -999
    for i in range(args.num_test):
        TestState.reset()
        state = test_env.reset()
        Video = []
        for j in count(1):
            # Update current state
            TestState.update(state)
            print(TestState())
            # Video.append(test_env.render('rgb_array'))

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
                if episode_reward > best_episode_reward:
                    BestVideo = Video
                    best_episode_reward = episode_reward
                Video = []
                episode_reward = 0
                done = False
                break

    torch.save(BestVideo, '.Videos/vid'+str(best_episode_reward)+'.pt')
    print('Saved Video')
    return total_reward/args.num_test


class VideoPlayer(object):
    def __init__(self, Video, fps):
        import cv2
        self.video = Video
        self.fps = fps

    def __call__(self):
        print('Playing video')
        for frame in self.video:
            pass

