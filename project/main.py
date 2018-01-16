'''
Main training loop for PPO in the Social Environment.
'''
import os
import numpy as np
import torch
import torch.optim as optim

from utils.utils import make_log_dirs, adjust_learning_rate
from utils.arguments import get_args
from utils.vislogger import VisLogger
from models.combine import CombinePolicy, Combine_NoTargetState

from agent.test import Test_and_Save_Video_Social as Test_and_Save_Video
from agent.train import explorationSocial as exploration
from agent.train import trainSocial as train
from agent.memory import RolloutStorageCombi as RolloutStorage
from agent.memory import Results, Current, Targets
from environments.social import Social, Social_multiple


args = get_args()

print('\n=== Loading Targets ===')
train_dset = torch.load(args.target_path)
print('\nTraining:', args.target_path)
test_dset = torch.load(args.target_path2)
print('\nTesting:', args.target_path2)

s_target, o_target = train_dset[4]  # choose random data point
s_te, o_te = test_dset[5]  # check to have same dims as training set
assert s_target.shape == s_te.shape, 'training and test shapes do not match'
assert o_target.shape == o_te.shape, 'training and test shapes do not match'

targets = Targets(n=args.num_processes, dset=train_dset)

args.video_w = o_target.shape[0]  # Environment will use these values
args.video_h = o_target.shape[1]

# frames -> updates
args.num_updates = int(args.num_frames) // args.num_steps // args.num_processes
args.test_thresh = int(args.test_thresh) // args.num_steps // args.num_processes

make_log_dirs(args)  # create dirs for training logging
if not args.no_vis:
    vis = VisLogger(args)

print('\n=== Create Environment ===\n')
Env = Social  # Env as variabe then change this line between experiments
env = Social_multiple(args)

st_shape = s_target.shape[0]  # targets
ot_shape = o_target.shape

s_shape = env.state_space.shape[0]    # Joints state
o_shape = env.observation_space.shape  # RGB (W,H,C)
ac_shape = env.action_space.shape[0]   # Actions

test_env = Env(args)
test_env.seed(np.random.randint(0, 20000))

# === Memory ===
result = Results(200, 10)
current = Current(args.num_processes, args.num_stack, s_shape, st_shape, o_shape, o_shape)
rollouts = RolloutStorage(args.num_steps,
                          args.num_processes,
                          current.state.size()[1],
                          current.target_state.size()[1],
                          current.obs.size()[1:],
                          ac_shape)

# === Model ===
# model assumes o_shape: (C, W, H)
if args.use_target_state:
    print('All inputs to policy')
    Model = CombinePolicy
    pi = CombinePolicy(s_shape=current.s_shape,
                       st_shape=current.st_shape,
                       o_shape=current.o_shape,
                       ot_shape=current.ot_shape,
                       a_shape=ac_shape,
                       feature_maps=args.feature_maps,
                       kernel_sizes=args.kernel_sizes,
                       strides=args.strides,
                       args=args)
else:
    print('No state_target as input to policy')
    Model = Combine_NoTargetState
    pi = Combine_NoTargetState(s_shape=current.s_shape,
                               st_shape=current.st_shape,
                               o_shape=current.o_shape,
                               ot_shape=current.ot_shape,
                               a_shape=ac_shape,
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
print('\nEnvironment', args.env_id)
print('Actions:', ac_shape)
print('State:', s_shape)
print('State target:', st_shape)
print('Obs:', o_shape)
print('Obs target:', ot_shape)
print('\nPOLICY:\n', pi)
print('Total network parameters to train: ', pi.total_parameters())
print('\nTraining for %d Updates' % args.num_updates)

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

    rollouts.last_to_first()
    result.update_loss(vloss.data, ploss.data, ent.data)
    frame = pi.n * args.num_processes

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
    nt = 5
    if not args.no_test and j % args.test_interval < nt and j > args.test_thresh:
        print('Testing...')
        pi.cpu()
        sd = pi.cpu().state_dict()
        test_reward_list = Test_and_Save_Video(test_env, test_dset, Model, sd, args, frame)
        test_reward_np = np.array(test_reward_list)
        result.update_test(test_reward_list)
        test_reward = test_reward_list.mean()

        print('Average Test Reward: {}\n '.format(round(test_reward)))
        if args.vis:
            vis.scatter_update(Xdata=frame, Ydata=test_reward, name='Test Score Scatter')

        if test_reward > MAX_REWARD:
            print('--' * 45)
            print('New High Score!\nAvg. Reward:', test_reward)
            print('--' * 45)
            name = os.path.join(
                args.checkpoint_dir,
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
