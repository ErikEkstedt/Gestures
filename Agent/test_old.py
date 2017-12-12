import torch
from itertools import count
from memory import StackedState

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from OpenGL import GLU # fix for opengl issues on desktop  / nvidia

def make_video(vid, filenname):
    fig = plt.figure()
    ims = []
    for frame in tqdm(vid):
        im = plt.imshow(frame, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)
    ani.save(filenname)


def Test_and_Save_Video(test_env, pi, args, verbose=False):
    '''
    Test with video
    :param pi - The policy playing
    :param args - arguments

    :output      - Average complete episodic reward
    '''
    # Use only 1 processor for test
    TestState = StackedState(1, args.num_stack, ob_shape)
    if args.cuda:
        TestState.cuda()

    # Test environments
    total_reward, episode_reward, best_episode_reward = 0, 0, -999
    for i in range(args.num_test):
        (s, obs) = test_env.reset()
        Video = []
        for j in count(1):
            # Update current state
            TestState.update(state)
            Video.append(test_env.render('rgb_array'))

            # Sample actions
            value, action, _, _ = pi.sample(TestState(), deterministic=True)
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            (state, rgb), reward, done, info = test_env.step(cpu_actions)
            Video.append(rgb)

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

    videoname='{}/frame_{}_score_{}.mp4'.format(args.video_dir,
                                            pi.n*args.num_processsor,
                                            round(best_episode_reward,2))
    make_video(BestVideo, videoname)
    print('Saved Video: ', vid_name)
    return total_reward/args.num_test


if __name__ == '__main__':
    # watch_video()
    make_video()
