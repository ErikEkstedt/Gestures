''' Collects randomly generated data from enivronments.  '''
import torch
import numpy as np
import os
from tqdm import tqdm
import pathlib


def collect_data(env, args):
    """ DataGenerator runs some episodes and randomly saves rgb, state pairs
    :dpoints            : Number of data points to collect
    :Returns            : dict
    """
    s, st, o, ot = env.reset()
    env.set_target([np.array(s.shape), np.array(s.shape)])

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


def generate_continous_data(env, args):
    ''' continuous state trajectory collection
    collects args.dpoints frames continuously and saves a dataset to disk.
    '''
    from project.data.dataset import Social_Dataset_numpy
    trajectory = {'states': [], 'obs': []}  # dataset wants dict
    s, _, o, _ = env.reset()
    for i in range(args.dpoints):
        trajectory['states'].append(s)
        trajectory['obs'].append(o)
        action = env.action_space.sample()
        s, _, o, _, r, d, _ = env.step(action)
        if d:
            # if args.MAX_TIME == args.dpoints this should not happen
            print('DONE')
            trajectory['states'].append(s)
            trajectory['obs'].append(o)
            break
    return trajectory

def collect_social_targets():
    data = generate_continous_data()


if __name__ == '__main__':
    from project.utils.arguments import get_args
    from project.environments.social import Social
    args = get_args()

    args.MAX_TIME = args.dpoints  # no need to gather abrupt resets
    env = Social(args)
    env.seed(np.random.randint(0,20000))  # random seed

    # data = get_trajectory(Social, args)

    env = Social(args)
    env.seed(args.seed)
    data = collect_data(env, args)

    filename = get_filename(Social, args)
    print('Saving into: ', filename)
    input('Press Enter to continue')
    torch.save(data, filename)
