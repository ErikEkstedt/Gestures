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
    env_string = str(Env).split(".")[-1][:-2]
    dir_ = os.path.join(args.data_dir, env_string)
    if not os.path.exists(dir_):
        print('Creating directory {}...'.format(dir_))
        pathlib.Path(dir_).mkdir(parents=True, exist_ok=True)
    name = 'obsdata_rgb{}-{}-3_n{}_'.format(args.video_W, args.video_H, args.dpoints)
    filename = os.path.join(dir_, name)
    run = 0
    while os.path.exists("{}{}.pt".format(filename, run)):
        run += 1

    return os.path.join("{}{}.pt".format(filename, run))


def main():
    from project.utils.arguments import get_args
    from project.environments.reacher import ReacherPlaneNoTarget, ReacherPlane, Reacher3D
    from project.environments.utils import make_parallel_environments
    # from project.data.dataset import ProjectDataSet

    args = get_args()
    args.RGB = True  # to be safe
    args.video_W = 40
    args.video_H = 40
    Env = ReacherPlaneNoTarget
    # Env = Reacher3D
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
    main()
