def args_to_list(args):
    l = []
    for arg, value in args._get_kwargs():
        s = "{}: {}".format(arg, value)
        l.append(s)
    return l

def print_args(args):
    l = args_to_list(args)
    for s in l:
        print(s)
    return l

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

