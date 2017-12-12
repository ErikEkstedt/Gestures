import pathlib
import datetime
import os

def make_log_dirs(args):
    ''' Creates dirs:
        ../root/day/DoF/run/
        ../root/day/DoF/run/checkpoints
        ../root/day/DoF/run/results
    '''
    def get_today():
        t = datetime.date.today().ctime().split()[1:3]
        s = "".join(t)
        return s

    rootpath = args.log_dir
    day = get_today()
    dof = 'DoF' + str(args.dof)
    if args.RGB:
        rootpath = os.path.join(rootpath, day, dof, 'RGB')
    else:
        rootpath = os.path.join(rootpath, day, dof)

    run = 0
    while os.path.exists("{}/run-{}".format(rootpath, run)):
        run += 1

    # Create Dirs
    pathlib.Path(rootpath).mkdir(parents=True, exist_ok=True)
    rootpath = "{}/run-{}".format(rootpath, run)
    result_dir = "{}/results".format(rootpath)
    checkpoint_dir = "{}/checkpoints".format(rootpath)
    os.mkdir(rootpath)
    os.mkdir(checkpoint_dir)
    os.mkdir(result_dir)

    # append to args
    args.log_dir = rootpath
    args.result_dir = result_dir
    args.checkpoint_dir = checkpoint_dir

def log_print(agent, dist_entropy, value_loss, floss, action_loss, j):
    print("\nUpdate: {}, frames:    {} \
          \nAverage final reward:   {}, \
          \nentropy:                {:.4f}, \
          \ncurrent value loss:     {:.4f}, \
          \ncurrent policy loss:    {:.4f}".format(j,
                (j + 1) * agent.args.num_steps * agent.args.num_processes,
                agent.final_rewards.mean(),
                -dist_entropy.data[0],
                value_loss.data[0],
                action_loss.data[0],))

def make_gym_env(env_id, seed, rank, log_dir):
    import gym
    from baselines.common import bench
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, "{}.monitor.json".format(rank)))
        return env
    return _thunk

