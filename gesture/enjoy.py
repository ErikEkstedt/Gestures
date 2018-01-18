'''
Loads a state dict, a target dataset and runs simulations.

if args.record then a mp4 file will be saved (kind of bad graphics and this
could be done better)

play saved file on vlc:
    vlc --avcodec-hw none MYSAVEDFILE.mp4

An episode normally lasts for 300 frames and is set through
args.MAX_TIME. The script ends when the environment resets so for
longer play args.MAX_TIME should be set to a larger value.

example:

    python enjoy.py --render --MAX_TIME=3000
    python enjoy.py --record --MAX_TIME=3000

    python enjoy.py --render --MAX_TIME=3000 \
        --target-path=/PATH/to/target_data_set/ \
        --state-dict-path=/PATH/to/state_dict

'''
import numpy as np
import time
import torch
from itertools import count
from tqdm import tqdm

from gesture.utils.arguments import get_args
from gesture.utils.utils import record, load_dict, get_model, get_targets
from gesture.agent.memory import Current, Targets
from gesture.models.combine import CombinePolicy, SemiCombinePolicy
from gesture.environments.utils import env_from_args


def enjoy(env, targets, pi, args):
    if args.cuda:
        current.cuda()
        pi.cuda()

    if args.record:
        import skvideo.io
        name = "mimic_{}_update{}.mp4".format(args.env_id, args.update_target)
        writer = skvideo.io.FFmpegWriter(name)

    if args.continuous_targets:
        target = targets[0]
        t = 1
    else:
        target = targets()

    env.set_target(target)
    state, s_target, obs, o_target = env.reset()

    tt = time.time()
    total_reward = 0
    for j in tqdm(range(args.MAX_TIME)):
        current.update(state, s_target, obs, o_target)
        s ,st, o, ot = current()
        value, action = pi.act(s, st, o, ot)
        if args.render:
            env.render('human')
            env.render('target')
        if args.record:
            record(env, writer)
        if j % args.update_target == 0:
            if args.continuous_targets:
                target = targets[t]
                t += 1
                if t > len(dset)-1:
                    break
            else:
                target = targets()
            env.set_target(target)

        # Observe reward and next state
        cpu_actions = action.data.cpu().numpy()[0]
        state, s_target, obs, o_target, reward, done, info = env.step(cpu_actions)
        total_reward += reward

    print('Time for enjoyment: ', time.time()-tt)
    if args.record:
        writer.close()

if __name__ == '__main__':
    print('Time to enjoy!')
    args = get_args()
    args.num_proc = 1
    Env = env_from_args(args)

    print('Loading state dict from:')
    print('path:\t', args.state_dict_path)
    state_dict = torch.load(args.state_dict_path)

    print('\nLoading targets from:')
    print('path:\t', args.test_target_path)
    datadict = load_dict(args.test_target_path)
    targets = Targets(1, datadict)
    targets.remove_speed(args.njoints)

    s_target, o_target = targets()  # random
    st_shape = s_target.shape[0]  # targets
    ot_shape = o_target.shape

    print(st_shape)
    # === Environment ===
    env = Env(args)
    env.seed(np.random.randint(0,20000))  # random seed

    # Model
    s_shape = env.state_space.shape[0]    # Joints state
    o_shape = env.observation_space.shape  # RGB (W,H,C)
    ac_shape = env.action_space.shape[0]   # Actions
    current = Current(1, args.num_stack, s_shape, st_shape, o_shape, o_shape, ac_shape)

    pi, Model = get_model(current, args)
    pi.load_state_dict(state_dict)
    enjoy(env, targets, pi, args)
