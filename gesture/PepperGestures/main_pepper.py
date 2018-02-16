''' Pepper '''
import __future__
import os
import numpy as np
import datetime
import qi
import motion

from arguments import get_args
from Pepper import Pepper_v0
from train import explorationPepper as exploration
from train import trainPepper as train
from memory import Results, Current
from storage import RolloutStoragePepper
from model import MLPPolicy, VanillaCNN

import torch
import torch.optim as optim
import torch.nn as nn

print('\n=== Create Environment ===\n')
args = get_args()
args.num_proc=1
session = qi.Session()
session.connect("{}:{}".format(args.IP, args.PORT))
env = Pepper_v0(session, args=args)

# Create dirs
path='/home/erik/DATA/Pepper'
run = 0
while os.path.exists("{}/run-{}".format(path, run)):
    run += 1
path = "{}/run-{}".format(path, run)
os.mkdir(path)
args.log_dir = path
args.checkpoint_dir = path

# ====== Goal ===============
# "hurray pose"
L_arm = [-0.38450, 0.81796, -0.99049, -1.18418, -1.3949, 0.0199]
R_arm = [-0.90522, -1.03321, -0.05766, 0.84596, 1.39495, 0.01999]
st = np.array(L_arm+R_arm).astype('float32')
env.set_target(st)

# frames -> updates
args.num_updates = int(args.num_frames) // args.num_steps // args.num_proc
args.test_thresh = int(args.test_thresh) // args.num_steps // args.num_proc
args.test_interval = int(args.test_interval) // args.num_steps // args.num_proc

if not args.no_vis:
    from vislogger import VisLogger
    vis = VisLogger(args)
    if args.verbose:
        vis.print_console()

# s_shape = env.state_space.shape[0]    # Joints state
s_shape = 24
st_shape = 12
o_shape = env.observation_space.shape  # RGB (W,H,C)
o_shape = (3,64,64)
ac_shape = env.action_space.shape[0]   # Actions


# === Memory ===
result = Results(200, 10)
current = Current(args.num_proc, args.num_stack, s_shape, st.shape[0], o_shape, o_shape, ac_shape)
rollouts = RolloutStoragePepper(args.num_steps)

# === Model ===
in_shape = current.st_shape + current.s_shape
pi = MLPPolicy(input_size=in_shape, a_shape=current.ac_shape, args=args)
understand = VanillaCNN(input_shape=o_shape,
                        s_shape=12, # Target
                        feature_maps=[64, 64, 64],
                        kernel_sizes=[5, 5, 5],
                        strides=[2, 2, 2],
                        args=args)
Uloss = nn.MSELoss()

if args.continue_training:
    print('\n=== Continue Training ===\n')
    print('Loading:', args.state_dict_path)
    sd = torch.load(args.state_dict_path)
    pi.load_state_dict(sd)

optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)
optimizer_u = optim.Adam(understand.parameters(), lr=args.pi_lr)

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
s, st, obs  = env.reset()
print(obs.shape)
current.update(state=s, s_target=st)
s, st, _ , _ = current()
rollouts.first_insert(state=s, s_target=st)
if args.cuda:
    current.cuda()
    rollouts.cuda()
    pi.cuda()
    understand.cuda()

pi.train()
understand.train()
MAX_REWARD = -99999
for j in range(args.num_updates):
    exploration(pi, current, _, rollouts, args, result, env)
    uloss, vloss, ploss, ent = train(pi, understand, Uloss, args, rollouts, optimizer_pi, optimizer_u)
    rollouts.last_to_first()  # reset data, start from last data point

    result.update_loss(uloss.data, vloss.data, ploss.data, ent.data)
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
            'dict_{}.pt'.format(frame))
        torch.save(sd, name)
        if args.cuda:
            pi.cuda()
