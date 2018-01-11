''' Collects randomly generated data from enivronments.  '''
import torch
import numpy as np
import os
from tqdm import tqdm
import pathlib


def DataGenerator(env, args):
    """ DataGenerator runs some episodes and randomly saves rgb, state pairs
    :dpoints            : Number of data points to collect
    :Returns            : dict
    """
    s, obs = env.reset()
    states, obs_list = [], []
    steps = args.dpoints // args.num_processes
    print('start collecting')
    for i in tqdm(range(steps)):
        action = np.random.rand(*(args.num_processes, *env.action_space.shape))
        s, obs, _, d, _ = env.step(action)
        for j in range(args.num_processes):
            states.append(s[j])
            obs_list.append(obs[j])
    return {'obs':obs_list, 'states': states}


def get_filename(Env, args):
    ''' uses the environment name to create dir
    args.data_dir / ENV_NAME / obsdata_{C}-{W}-{H}_n{DATAPOINTS}_{RUN}.pt
    '''
    env_string = str(Env).split(".")[-1][:-2]
    dir_ = os.path.join(args.data_dir, env_string)
    if not os.path.exists(dir_):
        print('Creating directory {}...'.format(dir_))
        pathlib.Path(dir_).mkdir(parents=True, exist_ok=True)
    name = 'obsdata_{}-{}-{}_n{}_'.format(args.video_w, args.video_h, args.video_c, args.dpoints)
    filename = os.path.join(dir_, name)
    run = 0
    while os.path.exists("{}_{}.pt".format(filename, run)):
        run += 1
    return os.path.join("{}_{}.pt".format(filename, run))

def get_social_trajectory(Social, args):
    ''' continuous state trajectory '''
    env = Social(args)
    env.seed(np.random.randint(0,2000))  # random seed

    env.reset ()  # init
    env.set_target()
    trajectory = {'states': [], 'obs': []}
    for i in range(args.episodes):
        s, _, o, _ = env.reset()
        while True:
            trajectory['states'].append(s)
            trajectory['obs'].append(o)
            action = env.action_space.sample()
            s, _, o, _, r, d, _ = env.step(action)

            if d:
                trajectory['states'].append(s)
                trajectory['obs'].append(o)
                break
    return trajectory

def old(args):
    from project.environments.reacher import ReacherPlaneNoTarget, ReacherPlane, Reacher3D
    from project.environments.utils import make_parallel_environments
    args = get_args()
    args.RGB = True  # to be safe
    args.video_W = 40
    args.video_H = 40
    Env = ReacherPlaneNoTarget
    env = make_parallel_environments(Env, args)

    print('Generate Data...')
    data = DataGenerator(env, args)
    print('Done')

    # add in correct dir
    filename = get_filename(Env, args)
    print('Saving into: ', filename)
    input('Press Enter to continue')
    torch.save(data, filename)


if __name__ == '__main__':
    from project.utils.arguments import get_args
    from project.environments.social import Social
    args = get_args()

    data = get_social_trajectory(Social, args)
    filename = get_filename(Social, args)

    print('Saving into: ', filename)
    input('Press Enter to continue')
    torch.save(data, filename)






