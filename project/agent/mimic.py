import numpy as np
import time
import torch
from itertools import count
from torchvision.utils import make_grid
import os

from project.utils.arguments import get_args
from project.agent.memory import StackedState, StackedObs
from project.models.combine import CombinePolicy
from project.environments.social import Social
from project.data.dataset import load_reacherplane_data
from project.environments.utils import render_and_scale


def mimic(env, dset, Model, state_dict, args, verbose=False):
    ob_sample, st_sample = dset[4]  #random index
    ot_shape      = ob_sample.shape
    st_shape      = st_sample.shape[0]

    s_shape             = env.state_space.shape[0]    # Joints state
    o_shape             = env.observation_space.shape # RGB
    ac_shape             = env.action_space.shape[0]   # Actions

    current = Current(1, args.num_stack, s_shape, st_shape, o_shape, o_shape)

    pi = CombinePolicy(o_shape=current.obs.obs_shape,
                       o_target_shape=current.obs.obs_shape,
                       s_shape=s_shape,
                       s_target_shape=s_target_shape,
                       a_shape=ac_shape,
                       feature_maps=[64, 64, 8],
                       kernel_sizes=[5, 5, 5],
                       strides=[2, 2, 2],
                       args=args)

    pi.load_state_dict(state_dict)

    if args.cuda:
        CurrentStateTarget.cuda()
        CurrentObsTarget.cuda()
        CurrentState.cuda()
        CurrentObs.cuda()
        pi.cuda()


    if args.record:
        video = []

    # Test environments
    t = 0
    total_reward, episode_reward, best_episode_reward = 0, 0, -9999

    ob_target, st_target = dset[t]; t += 1
    env.set_target(st_target, ob_target)
    state, s_target, obs, o_target = env.reset()
    tt = time.time()
    for j in count(1):
        CurrentState.update(state); CurrentStateTarget.update(s_target)
        CurrentObs.update(obs); CurrentObsTarget.update(o_target)

        value, action = pi.act(CurrentObs(), CurrentObsTarget(), CurrentState(), CurrentStateTarget())
        if args.render:
            env.render()
            env.render('target')


        if args.record:
            target_im = CurrentObsTarget()[0].cpu()
            frame = CurrentObs()[0].cpu()
            imglist = [frame, target_im]
            img = make_grid(imglist, padding=5).numpy()
            img = img.transpose((1,2,0))
            img *= 255
            video.append(img.astype('uint8'))

        if j % args.update_target == 0:
            ob_target, st_target = dset[t]
            env.set_target(st_target, ob_target)
        t += 1
        if t > len(dset)-1:
            break
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

    print('Time for mimic: ', time.time()-tt)
    if args.record:
        import skvideo.io
        output = os.path.join(os.getcwd(), 'mimic_skip{}frames.mp4'.format(args.update_target))
        writer = skvideo.io.vwrite(output, np.stack(video, axis=0))
        print('Videofile: ', output)

    # if args.record:
    #     print('Making Video...')
    #     ani = animation.ArtistAnimation(fig, video, interval=20, blit=True, repeat_delay=1000)
    #     savepath = os.path.join(os.getcwd(), 'mimic.mp4')
    #     ani.save(savepath)
    #     print('Saved video to:', )



if __name__ == '__main__':
    args = get_args()

    print('Social Mimic')
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
