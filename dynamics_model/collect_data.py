import argparse
from tqdm import trange
import pickle
import torch
import os

# ------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data-size', type=int, default=10000)
parser.add_argument('--dirpath', type=str, default="/home/erik/DATA/Project/data/")
parser.add_argument('-v', '--verbose', type=int, default=1)
parser.add_argument('--mujoco', type=bool, default=False)
parser.add_argument('--env_name', type=str, default="Pendulum-v0")
parser.add_argument('--num_datafiles', type=int, default=1)

# ---------------------------
# Helper funcs
def downsample(I):
    """ prepro 500,500,3 uint8 frame into
    (40,40,3) uint8 vector """
    I = I[::3,::3,:]
    I = I[::2,::2,:] # downsample by factor of 2
    I = I[::2,::2,:] # downsample by factor of 2.
    I = I[1:-1,1:-1,:]
    return I

def collect_random_data(env, data_size = 100, render=False):
    rgb_list, state_list, action_list = [], [], []
    x = env.reset()
    for i in trange(data_size):
        if render: env.render()
        rgb = env.render('rgb_array')
        rgb = downsample(rgb)
        a = env.action_space.sample()
        x_, r, done, _ = env.step(a)
        rgb_list.append(rgb)
        state_list.append(x)
        action_list.append(a)
        x = x_
    return rgb_list, state_list, action_list

def save_data(data_size, dirpath='/home/erik/DATA/Project/', verbose=True):
    ''' Collects random data and saves it to disk through pickle.
    # Arguments:

    :param data_size - Integer of data size
    :param dirpath   - Directory savepath
    '''
    run = 0
    while os.path.exists("%s/data%d.p" % (dirpath, run)):
        run += 1
    fname=dirpath+'data'+str(run)+'.p'
    if args.mujoco:
        from env import ReacherEnv_hacked
        env = ReacherEnv_hacked()
    else:
        import gym
        env = gym.make(args.env_name)
    if verbose: print('Collecting data\n')
    #obs, states, actions = collect_random_data(env, data_size)
    obs, states, actions = collect_random_data(env, data_size)
    env.close()
    DATA = [obs, states, actions]
    pickle.dump(DATA, open(fname, "wb"))
    if verbose: print('Saving data in: ',fname)

#-----------------------------------------------
if __name__ == '__main__':
    args = parser.parse_args()
    for _ in range(args.num_datafiles):
        save_data(args.data_size,  args.dirpath, args.verbose)


