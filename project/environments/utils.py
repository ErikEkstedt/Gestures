import cv2
import torch
import numpy as np
import time
from itertools import count
from torchvision.utils import make_grid


def env_from_args(args):
    if 'umanoid' in args.env_id:
        from project.environments.social import SocialHumanoid
        Env = SocialHumanoid
        args.env_id = 'SocialHumanoid'
        args.njoints = 6
    elif 'eacher' in args.env_id:
        from project.environments.social import SocialReacher
        Env = SocialReacher
        args.env_id = 'SocialReacher'
        args.njoints = 2
    else:
        import sys
        print('Environment not defined! (SocialHumanoid / SocialReacher)')
        sys.exit(0)
    return Env


# ======================== #
# Render                   #
# ======================== #
def rgb_render(obs, title='obs'):
    ''' cv2 as argument such that import is not done redundantly'''
    cv2.imshow(title, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print('Stop')
        return

def rgb_tensor_render(obs, scale=(1,1), title='tensor_obs'):
    assert type(obs) is torch.Tensor
    assert len(obs.shape) == 3
    obs = obs.permute(1,2,0)
    im = obs.numpy().astype('uint8')
    # rgb_render(im, title)
    render_and_scale(im, scale=scale, title=title)

def render_and_scale(obs, scale=(1, 1), title='obs'):
    ''' cv2 as argument such that import is not done redundantly'''
    height, width = obs.shape[:2]
    obs = cv2.resize(obs,(scale[0]*width, scale[0]*height), interpolation = cv2.INTER_CUBIC)
    cv2.imshow(title, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print('Stop')
        return


# ======================== #
# Run envs                 #
# ======================== #

# Single Test functions
def random_run(env, render=False, verbose=False):
    ''' Executes random actions and renders '''
    s, s_, o, o_ = env.reset()
    for i in count(1):
        action = env.action_space.sample()
        s, s_, o, o_, r, d, _ = env.step(action)
        if verbose:
            print('\nframe: {}\ts: {}\to: {}'.format(i, s.shape, o.shape ))
            print('action:', action)
            input('Press Enter to continue')
            # print('absolute mean:', np.abs(np.array(action)).mean())

        if render:
            env.render('all')
            # env.render()  # same as env.render('human')
            # env.render('machine')
            # env.render('target')
            # h, m, t = env.render('all_rgb_array')  # returns rgb arrays
            # print(h.shape)
            # print(m.shape)
            # print(t.shape)
        if d:
            s, s_, o, o_ = env.reset()

def random_run_with_changing_targets(env, dset, args):
    ''' Executes random actions and also changes the target.
    targets are set in order from a project.data.dataset.
    renders options:
        'render.modes': ['human', 'machine', 'target', 'all', 'all_rgb_array']
    '''
    t = 0
    while True:
        env.set_target(dset[t]); t += 1
        state, s_target, obs, o_target = env.reset()
        episode_reward = 0
        for j in count(1):
            if args.render:
                env.render('all')

            if args.verbose: print('frame:', j)
            # update the target
            if j % args.update_target == 0:
                env.set_target(dset[t]); t += 1

            # Observe reward and next state
            actions = env.action_space.sample()
            state, s_target, obs, o_target, reward, done, info = env.step(actions)

            # If done then update final rewards and reset episode reward
            episode_reward += reward
            if done:
                if args.verbose: print(episode_reward)
                break

# Parallel
def random_run_parallel(env, args):
    ''' Executes random actions and renders '''
    from itertools import count
    s, s_, o, o_ = env.reset()
    for i in count(1):
        action = np.random.rand(*(args.num_proc, *env.action_space.shape))
        s, s_, o, o_, r, d, _ = env.step(action)
        if args.verbose:
            print('frame:', i)
        if args.render:
            env.render(['human']*args.num_proc)  # same as env.render('human')
            # env.render('all')  # everything
            # env.render('machine')
            # env.render('target')
            # h, m, t = env.render('all_rgb_array')  # returns rgb arrays
            # mode = ['all_rgb_array'] * args.num_proc
            # H, M, T = env.render(mode)
            # print('H: {}\tM: {}\tT: {}'.format(H.shape, M.shape, T.shape))

        if sum(d) > 0:
            print('one env is finished')

def random_run_with_changing_targets_parallel(env, dset, args):
    ''' Executes random actions and also changes the target.
    targets are set in order from a project.data.dataset.
    renders options:

        'render.modes': ['human', 'machine', 'target', 'all', 'all_rgb_array']

        example of parallel render:

            modes = ['all'] * args.num_proc
    '''
    t = 0
    targets = [dset[t]] * args.num_proc

    env.set_target(targets)
    t += 1
    state, s_target, obs, o_target = env.reset()
    episode_reward = 0
    for j in count(1):
        if args.render:
            modes = ['all'] * args.num_proc
            env.render(modes)

        # update the target
        if j % args.update_target == 0:
            targets = [dset[t]] * args.num_proc
            env.set_target(targets)
            t += 1

        # Observe reward and next state
        actions = -1 + 2*np.random.rand(*(args.num_proc, *env.action_space.shape))
        state, s_target, obs, o_target, reward, done, info = env.step(actions)

        # If done then update final rewards and reset episode reward
        episode_reward += reward
        if sum(done) > 0:
            if args.verbose:
                print(episode_reward)




