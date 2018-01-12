from project.utils.arguments import get_args
from project.environments.social import Social
from project.data.dataset import Social_Dataset_numpy

from tqdm import tqdm
import numpy as np
import os
import pathlib
from torch import save


def generate_continous_data(env, args):
    ''' continuous state trajectory collection

    Collects args.dpoints frames continuously and saves a dataset to disk.
    '''
    trajectory = {'states': [], 'obs': []}  # dataset wants dict
    s, _, o, _ = env.reset()
    s = s[:-2]  # remove speed
    for i in tqdm(range(args.dpoints)):
        trajectory['states'].append(s)
        trajectory['obs'].append(o)
        action = env.action_space.sample()
        s, _, o, _, r, d, _ = env.step(action)
        s = s[:-2]  # remove speed
        if d:
            # if args.MAX_TIME == args.dpoints this should not happen
            print('DONE')
            trajectory['states'].append(s)
            trajectory['obs'].append(o)
            break
    return trajectory


def get_filename(path='/tmp', s_shape=6, o_shape=(40,40,3), n=10000):
    ''' returns string:

        path/socialtargets_s{S}_o{C}-{W}-{H}_n{DATAPOINTS}_{RUN}.pt
    '''
    if not os.path.exists(path):
        print('Creating directory {}...'.format(args.filepath))
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    name = 'socialtargets_s{}_o{}-{}-{}_n{}'.format(s_shape, o_shape[0],o_shape[1],o_shape[2], n)
    filename = os.path.join(path, name)
    run = 0
    while os.path.exists("{}_{}.pt".format(filename, run)):
        run += 1
    return os.path.join("{}_{}.pt".format(filename, run))


if __name__ == '__main__':
    args = get_args()
    args.MAX_TIME = args.dpoints  # no need to gather abrupt resets

    env = Social(args)
    env.seed(np.random.randint(0,20000))  # random seed

    data = generate_continous_data(env, args)
    dset = Social_Dataset_numpy(data)

    s, o = dset[0]
    filename = get_filename(args.filepath,
                            s_shape=4,
                            o_shape=o.shape,
                            n=args.dpoints)
    print('Saving into:\n\t', filename)
    save(dset, filename)
