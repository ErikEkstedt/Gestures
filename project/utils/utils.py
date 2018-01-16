import pathlib
import datetime
import os

def get_env(args):
    if 'eacher' in args.env_id:
        if args.dof == 2:
            from environments.reacher_envs import Reacher2DoF
            args.env_id='Reacher2DoF'
            return Reacher2DoF
        elif args.dof == 3:
            # from environments.reacher_envs import Reacher3DoF
            # args.env_id='Reacher3DoF'
            # return Reacher3DoF
            from environments.Reacher import Reacher
            args.env_id='Reacher3DoF'
            return Reacher
        elif args.dof == 32:
            from environments.reacher_envs import Reacher3DoF_2Target
            args.env_id='Reacher3DoF_2Target'
            return Reacher3DoF_2Target
        elif args.dof == 6:
            from environments.reacher_envs import Reacher6DoF
            args.env_id='Reacher6DoF'
            return Reacher6DoF
        else:
            from environments.reacher_envs import Reacher_plane
            args.env_id='Reacher_plane'
            return Reacher_plane
    elif 'umanoid' in args.env_id:
        if args.dof == 3:
            from environments.humanoid_envs import Humanoid3DoF
            args.env_id='Humanoid3DoF'
            return Humanoid3DoF
        elif args.dof == 6:
            from environments.reacher_envs import Humanoid6DoF
            args.env_id='Humanoid6DoF'
            return Humanoid6DoF
    else:
        print('Unknown environmnet')
        return

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
    if args.RGB:
        rootpath = os.path.join(rootpath, day, args.env_id, 'RGB')
    else:
        rootpath = os.path.join(rootpath, day, args.env_id)

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
    ''' Make parallel gym environments '''
    import gym
    from baselines.common import bench
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, "{}.monitor.json".format(rank)))
        return env
    return _thunk

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
torch.save(state, filename)
