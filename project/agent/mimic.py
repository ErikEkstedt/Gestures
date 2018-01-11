import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from itertools import count
from torchvision.utils import make_grid

from project.utils.arguments import get_args
from project.agent.memory import StackedState, StackedObs
from project.models.combine import CombinePolicy
from project.environments.social import Social
from project.data.dataset import load_reacherplane_data
from project.environments.utils import render_and_scale


def mimic(env, dset, Model, state_dict, args, verbose=False):
    ob_sample, st_sample, = dset[4]  #random index
    ob_target_shape       = ob_sample.shape
    st_target_shape       = st_sample.shape[0]

    CurrentStateTarget    = StackedState(1, args.num_stack, st_target_shape)
    CurrentObsTarget      = StackedObs(1, args.num_stack, ob_target_shape)

    st_shape              = env.state_space.shape[0]    # Joints state
    ob_shape              = env.observation_space.shape # RGB
    ac_shape              = env.action_space.shape[0]   # Actions

    CurrentState          = StackedState(1, args.num_stack, st_shape)
    CurrentObs            = StackedObs(1, args.num_stack, ob_shape)

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

    if args.record:
        fig = plt.figure()
        video = []

    # Test environments
    t = 0
    total_reward, episode_reward, best_episode_reward = 0, 0, -9999

    while t < len(dset):
        ob_target, st_target = dset[t]
        env.set_target(st_target, ob_target)
        t += 1

        state, s_target, obs, o_target = env.reset()

        for j in count(1):
            CurrentObs.update(obs)
            CurrentState.update(state)

            CurrentObsTarget.update(o_target)
            CurrentStateTarget.update(s_target)

            value, action = pi.act(CurrentObs(),
                                   CurrentObsTarget(),
                                   CurrentState(),
                                   CurrentStateTarget())

            if args.render:
                target_im = CurrentObsTarget()[0]
                frame = CurrentObs()[0]
                imglist = [frame, target_im]
                img = make_grid(imglist, padding=5).numpy()
                img = img.transpose((1,2,0))
                render_and_scale(img, scale=(9, 9))

                if args.record:
                    img *= 255
                    img = img.astype('uint8')
                    im = plt.imshow(img, animated=True)
                    video.append([im])

            if j % args.update_target == 0:
                if t > len(dset):
                    break
                ob_target, st_target = dset[t]
                env.set_target(st_target, ob_target)
                t += 1

            # Observe reward and next state
            cpu_actions = action.data.cpu().numpy()[0]
            state, s_target, obs, o_target, reward, done, info = env.step(cpu_actions)

            # If done then update final rewards and reset episode reward
            total_reward += reward
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                episode_reward = 0
                done = False
                break

    if args.record:
        import os
        print('Making Video...')
        ani = animation.ArtistAnimation(fig, video, interval=30, blit=True, repeat_delay=1000)
        savepath = os.path.join(os.getcwd(), 'mimic.mp4')
        ani.save(savepath)
        print('Saved video to:', )

if __name__ == '__main__':
    print('Social Mimic')

    args = get_args()

    print('Loading state dict from:')
    print('path:\t', args.state_dict_path)
    state_dict = torch.load(args.state_dict_path)

    # === Targets ===
    print('\nLoading targets from:')
    print('path:\t', args.target_path)

    dset, _ = load_reacherplane_data(args.target_path, shuffle=False)

    # === Environment ===
    env = Social(args)
    env.seed(np.random.randint(0,20000))  # random seed
    env.reset()  # init

    mimic(env, dset, CombinePolicy, state_dict, args, verbose=False)



