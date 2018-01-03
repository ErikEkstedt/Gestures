''' Collects randomly generated data from enivronments.  '''
import torch
import numpy as np
from tqdm import tqdm

def DataGenerator(env, args, dpoints=1000):
    """ DataGenerator runs some episodes and randomly saves rgb, state pairs

    :dpoints            : Number of data points to collect
    :Returns            : dict
    """
    s, obs = env.reset()
    states, obs_list = [], []
    steps = dpoints // args.num_processes
    print('start collecting')
    for i in tqdm(range(steps)):
        action = np.random.rand(*(args.num_processes, *env.action_space.shape))
        s, obs, _, d, _ = env.step(action)
        for j in range(args.num_processes):
            states.append(s[j])
            obs_list.append(obs[j])
    return {'obs':obs_list, 'states': states}


def main():
    from project.utils.arguments import get_args
    from project.environments.reacher import ReacherPlane, Reacher3D
    from project.environments.utils import make_parallel_environments
    # from project.data.dataset import ProjectDataSet
    import os
    import pathlib

    args = get_args()
    args.RGB = True  # to be safe
    args.video_W = 40
    args.video_H = 40
    Env = ReacherPlane
    # Env = Reacher3D
    env = make_parallel_environments(Env, args)

    print('Generate Data...')
    dpoints = int(1e5)
    data = DataGenerator(env, args, dpoints=dpoints)
    print('Done')

    # add in correct dir
    env_string = str(Env).split(".")[-1][:-2]
    dir_ = os.path.join(args.data_dir, env_string)
    if not os.path.exists(dir_):
        print('Creating directory {}...'.format(dir_))
        pathlib.Path(dir_).mkdir(parents=True, exist_ok=True)

    name = 'obsdata_rgb{}-{}-3_n{}.pt'.format(args.video_W, args.video_H, dpoints)
    filename = os.path.join(dir_, name)
    torch.save(data, filename)



if __name__ == '__main__':
    main()
