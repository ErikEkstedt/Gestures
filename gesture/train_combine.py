''' Coordination: Combine '''

import os
import numpy as np
import torch
import torch.optim as optim

from utils.arguments import get_args
from utils.utils import make_log_dirs, adjust_learning_rate
from utils.utils import get_targets
from environments.social import SocialReacher
from environments.social import Social_multiple
from agent.test import Test_and_Save_Video_MLP
from agent.train import exploration, train
from agent.memory import RolloutStorage, Results, Current, Targets
from models.combine import CombinePolicy as Model

args = get_args()
Env = SocialReacher

# frames -> updates
args.num_updates = int(args.num_frames) // args.num_steps // args.num_proc
args.test_thresh = int(args.test_thresh) // args.num_steps // args.num_proc
args.test_interval = int(args.test_interval) // args.num_steps // args.num_proc

print('\n=== Loading Targets ===')
targets, test_targets = get_targets(args)

st, ot = targets.random_target()
args.video_w = ot.shape[0]  # Match env dims with targets
args.video_h = ot.shape[1]

make_log_dirs(args)  # create dirs for training logging
if not args.no_vis:
    from utils.vislogger import VisLogger
    vis = VisLogger(args)
    if args.verbose:
        vis.print_console()

print('\n=== Create Environment ===\n')
env = Social_multiple(Env, args)

s_shape = env.state_space.shape[0]    # Joints state
o_shape = env.observation_space.shape  # RGB (W,H,C)
ac_shape = env.action_space.shape[0]   # Actions

test_env = Env(args)
test_env.seed(np.random.randint(0, 20000))

# === Memory ===
result = Results(200, 10)
current = Current(args.num_proc, args.num_stack, s_shape, st.shape[0], o_shape, ot.shape, ac_shape)
rollouts = RolloutStorage(args.num_steps,
                          args.num_proc,
                          current.state.size()[1],
                          current.target_state.size()[1],
                          current.obs.size()[1:],
                          ac_shape)

# === Model ===
pi = Model(s_shape=current.s_shape,
            st_shape=current.st_shape,
            o_shape=current.o_shape,
            ot_shape=current.ot_shape,
            a_shape=current.ac_shape,
            feature_maps=args.feature_maps,
            kernel_sizes=args.kernel_sizes,
            strides=args.strides,
            args=args)

if args.continue_training:
    print('\n=== Continue Training ===\n')
    print('Loading:', args.state_dict_path)
    sd = torch.load(args.state_dict_path)
    pi.load_state_dict(sd)

optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)

print('\n=== Training ===')
print('\nEnvironment:', args.env_id)
print('Model:', args.model)
print('Actions:', ac_shape)
print('State:', s_shape)
print('State target:', st.shape[0])
print('Obs:', o_shape)
print('Obs target:', ot.shape)
print('\nPOLICY:\n', pi)
print('Total network parameters to train: ', pi.total_parameters())
print('\nTraining for %d Updates' % args.num_updates)

# Initialize targets and reset env
env.set_target(targets())  # set initial targets
s, s_target, obs, obs_target = env.reset()
current.update(s, s_target, obs, obs_target)
s, st, o ,ot = current()
rollouts.first_insert(s, st, o, ot)
if args.cuda:
    current.cuda()
    rollouts.cuda()
    pi.cuda()

pi.train()
MAX_REWARD = -99999
for j in range(args.num_updates):
    exploration(pi, current, targets, rollouts, args, result, env)
    vloss, ploss, ent = train(pi, args, rollouts, optimizer_pi)
    rollouts.last_to_first()  # reset data, start from last data point

    result.update_loss(vloss.data, ploss.data, ent.data)
    frame = pi.n * args.num_proc

    #  ==== Adjust LR ======
    if args.adjust_lr and j % args.adjust_lr_interval == 0:
        print('Learning rate decay')
        adjust_learning_rate(optimizer_pi, decay=args.lr_decay)

    #  ==== SHELL LOG ======
    if j % args.log_interval == 0:
        result.plot_console(frame)

    #  ==== VISDOM PLOT ======
    if j % args.vis_interval == 0 and j > 0 and not args.no_vis:
        result.vis_plot(vis, frame, pi.get_std())

    #  ==== TEST ======
    nt = 3
    if not args.no_test and j % args.test_interval < nt and j > args.test_thresh:
        print('Testing...')
        pi.cpu()
        sd = pi.cpu().state_dict()
        test_reward = Test_and_Save_Video_MLP(test_env, test_targets, sd, args, frame, Model)
        result.update_test(test_reward)

        print('Average Test Reward: {}\n '.format(round(test_reward)))
        if args.vis:
            vis.scatter_update(Xdata=frame, Ydata=test_reward, name='Test Score Scatter')

        if test_reward > MAX_REWARD:
            print('--' * 45)
            print('New High Score!\nAvg. Reward:', test_reward)
            print('--' * 45)
            name = os.path.join(args.checkpoint_dir,
                                'BestDictCombi{}_{}.pt'.format(frame, round(test_reward, 3)))
            torch.save(sd, name)
            MAX_REWARD = test_reward
        else:
            name = os.path.join(
                args.checkpoint_dir,
                'dict_{}_TEST_{}.pt'.format(frame, round(test_reward, 3)))
            torch.save(sd, name)
        if args.cuda:
            pi.cuda()
