'''
Collect random targets used in training.

ex:

Bash:

    python collect_targets --dpoints=100000 --num_proc=4 --filepath=/FILE/PATH/

Saves 100000 state and observations using 4 processors to:

    /FILE/PATH/{}_S{s_shape}_O{Color}-{Width}-{Height}_n{DATAPOINTS}_{RUN}.pt

'''
from gesture.utils.arguments import get_args
from gesture.utils.utils import save_dict, load_dict
from gesture.environments.utils import env_from_args
from gesture.environments.social import Social_multiple

from tqdm import tqdm
import numpy as np
import os
import pathlib
from torch import save

def get_filename(path='/tmp', s_shape=6, o_shape=(40,40,3), n=10000, args=None):
    ''' returns string:
        /FILE/PATH/{}_S{s_shape}_O{Color}-{Width}-{Height}_n{DATAPOINTS}_{RUN}.h5
    '''
    print('Directory:', args.filepath)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    name = '{}_S{}_O{}-{}-{}_n{}'.format(args.env_id, s_shape[0], o_shape[0],o_shape[1],o_shape[2], n)
    filename = os.path.join(path, name)
    run = 0
    while os.path.exists("{}_{}.h5".format(filename, run)):
        run += 1
    return os.path.join("{}_{}.h5".format(filename, run))

def collect_random_targets(Env, args):
    """ Runs episodes and saves rgb, state pairs
    :dpoints            : Number of data points to collect
    :Returns            : dict
    """
    states, obs_list = [], []
    if args.num_proc > 1:
        print('Collecting {} datapoints using {} processes'.format(args.dpoints, args.num_proc))
        env = Social_multiple(Env, args)
        s_shape = env.state_space.shape
        o_shape = env.observation_space.shape
        dummy_target = [[np.array(s_shape), np.array(s_shape)]] *args.num_proc # Dummy target
        env.set_target(dummy_target)

        steps = args.dpoints // args.num_proc
        print('Collecting {} data points'.format(args.dpoints))
        s, st, o, ot = env.reset()
        for i in tqdm(range(steps)):
            action = np.random.rand(*(args.num_proc, *env.action_space.shape)) * 2 -1 # [0 1] -> [-1 1]
            s, _, o, _, r, d, _ = env.step(action)
            for j in range(args.num_proc):
                states.append(s[j])
                obs_list.append(o[j])
    else:
        print('Collecting', args.dpoints,'datapoints using a single processes')
        env = Env(args)
        env.seed(np.random.randint(0, 10000))
        s_shape = env.state_space.shape[0]
        o_shape = env.observation_space.shape
        env.set_target([np.array(s_shape), np.array(s_shape)]) # Dummy target

        steps = args.dpoints
        s, st, o, ot = env.reset()
        for i in tqdm(range(steps)):
            states.append(s)
            obs_list.append(o)

            action = env.action_space.sample()
            s, _, o, _, r, d, _ = env.step(action)
            if d:
                s, _, o, _ = env.reset()
    return {'obs':obs_list, 'states': states}, s_shape, o_shape

def collect_continous(Env, args):
    '''
    Continuous state trajectory collection
    Collects args.dpoints frames continuously and saves a dataset to disk.
    '''
    args.MAX_TIME = args.dpoints  # never reset
    trajectory = {'states': [], 'obs': []}  # dataset wants dict

    env.set_target([np.array(s.shape), np.array(s.shape)])  #set random targets
    s, _, o, _ = env.reset()
    for i in tqdm(range(args.dpoints)):
        trajectory['states'].append(s)
        trajectory['obs'].append(o)
        action = env.action_space.sample()
        s, _, o, _, r, d, _ = env.step(action)
        if d:
            env.seed(np.random.randint(0,20000))  # random seed
            print('DONE')
            trajectory['states'].append(s)
            trajectory['obs'].append(o)
            s, _, o, _ = env.reset()
    return trajectory


if __name__ == '__main__':
    args = get_args()
    Env = env_from_args(args)

    if args.continuous_targets:
        datadict, s_shape, o_shape = collect_continous(Env, args)
    else:
        datadict, s_shape, o_shape = collect_random_targets(Env, args)

    filename = get_filename(args.filepath, s_shape, o_shape, args.dpoints, args)
    print('Saving dict to:\n\t', filename)
    save_dict(datadict, filename)
