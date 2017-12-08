import torch
import gym
from itertools import count
from memory import StackedState

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from OpenGL import GLU # fix for opengl issues on desktop  / nvidia

def Test(pi, args, ob_shape, verbose=False):
    '''Test :param pi - The policy playing
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

def Test_and_See_gym(test_env, pi, args, ob_shape, verbose=False):
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

    total_reward, episode_reward, best_episode_reward = 0, 0, -999
    for i in range(args.num_test):
        TestState.reset()
        state = test_env.reset()
        Video = []
        for j in count(1):
            # Update current state
            TestState.update(state)
            # print(TestState())
            Video.append(test_env.render('rgb_array'))

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


def Test_and_Save_Video(test_env, pi, args, ob_shape, verbose=False):
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

    total_reward, episode_reward, best_episode_reward = 0, 0, -999
    for i in range(args.num_test):
        TestState.reset()
        state = test_env.reset()
        Video = []
        for j in count(1):
            # Update current state
            TestState.update(state)
            Video.append(test_env.render('rgb_array'))

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

    vid_name ='Videos/vid_frame'+str(pi.n)+'_rew'+str(round(best_episode_reward))+'.pt'
    make_video(BestVideo, vid_name)
    print('Saved Video: ', vid_name)
    return total_reward/args.num_test


class VideoPlayer(object):
    def __init__(self, Video, fps):
        import cv2
        self.video = Video
        self.fps = fps
        self.wait = 1/fps

    def __call__(self, cv2=False):
        print('Playing video')
        if cv2:
            for i, frame in enumerate(self.video):
                # Display the resulting frame
                plt.imshow(str(i), frame)
                # Press Q on keyboard to  exit
                if plt.waitKey(self.wait) & 0xFF == ord('q'):
                    break
        else:
            for i, frame in enumerate(self.video):
                # Display the resulting frame
                # print(frame.shape)
                # print(frame.dtype)
                plt.imshow(frame)
                # plt.pause(self.wait)
                plt.pause(0.01)

def make_video(vid, name):
    fig = plt.figure()
    ims = []
    for frame in vid:
        im = plt.imshow(frame, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
                                    repeat_delay=1000)
    ani.save(name+'.mp4')

def watch_video():
    import sys
    print(sys.argv[0])
    vid = torch.load(sys.argv[1])
    vp = VideoPlayer(vid, 25)
    vp()


if __name__ == '__main__':
    # watch_video()
    make_video()
