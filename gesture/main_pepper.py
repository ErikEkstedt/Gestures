'''
Main training loop for PPO in the Social Environment.
'''
import os
import numpy as np
import torch
import torch.optim as optim
import __future__

import qi
import motion
import datetime
import pathlib

from utils.arguments import get_args
from environments.pepper.pepper import Pepper_v0

from agent.test import Test_and_Save_Video
from agent.train import explorationPepper as exploration
from agent.train import trainPepper as train
from agent.memory import RolloutStorage, Results, Current, Targets
from models.peppermodel import MLPPolicy

path='/tmp/Pepper/'
run = 0
while os.path.exists("{}/run-{}".format(path, run)):
    run += 1
path = "{}/run-{}".format(path, run)
os.mkdir(path)

args = get_args()
args.log_dir = path
session = qi.Session()
session.connect("{}:{}".format(args.IP, args.PORT))
env = Pepper_v0(session)

# ====== Goal ===============
# "hurray pose"
L_arm = [-0.38450, 0.81796, -0.99049, -1.18418, -1.3949, 0.0199]
R_arm = [-0.90522, -1.03321, -0.05766, 0.84596, 1.39495, 0.01999]
st = np.array(L_arm+R_arm).astype('float32')

# frames -> updates
args.num_updates = int(args.num_frames) // args.num_steps // args.num_proc
args.test_thresh = int(args.test_thresh) // args.num_steps // args.num_proc
args.test_interval = int(args.test_interval) // args.num_steps // args.num_proc

# print('\n=== Loading Targets ===')
# targets, test_targets = get_targets(args)
# st, ot = targets.random_target()
# args.video_w = ot.shape[0]  # Match env dims with targets
# args.video_h = ot.shape[1]

if not args.no_vis:
    from utils.vislogger import VisLogger
    vis = VisLogger(args)
    if args.verbose:
        vis.print_console()

print('\n=== Create Environment ===\n')

# s_shape = env.state_space.shape[0]    # Joints state
s_shape = 24
o_shape = env.observation_space.shape  # RGB (W,H,C)
o_shape = (1,64,64)
ac_shape = env.action_space.shape[0]   # Actions

print(s_shape)
print(st.shape)
# === Memory ===
result = Results(200, 10)
current = Current(args.num_proc, args.num_stack, s_shape, st.shape[0], o_shape, o_shape, ac_shape)
rollouts = RolloutStorage(args.num_steps,
                          args.num_proc,
                          current.state.size()[1],
                          current.target_state.size()[1],
                          current.obs.size()[1:],
                          ac_shape)

# === Model ===
pi = MLPPolicy(s_shape, ac_shape, args)

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
print('\nPOLICY:\n', pi)
print('Total network parameters to train: ', pi.total_parameters())
print('\nTraining for %d Updates' % args.num_updates)

# Initialize targets and reset env
s, obs = env.reset()
current.update(state=s, s_target=st)
s, st, _ , _ = current()
rollouts.first_insert(state=s, s_target=st)
if args.cuda:
    current.cuda()
    rollouts.cuda()
    pi.cuda()

pi.train()
MAX_REWARD = -99999
for j in range(args.num_updates):
    exploration(pi, current, _, rollouts, args, result, env)
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

    #  ==== Saving ======
    if j % args.save_interval == 0 and j > 0:
        pi.cpu()
        sd = pi.cpu().state_dict()
        name = os.path.join(
            args.checkpoint_dir,
            'dict_{}_TEST_{}.pt'.format(frame, round(test_reward, 3)))
        torch.save(sd, name)
        if args.cuda:
            pi.cuda()
