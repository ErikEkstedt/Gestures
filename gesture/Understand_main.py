import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils.arguments import get_args
from utils.utils import make_log_dirs
from utils.utils import load_dict
from data.dataset import UnderstandDataset
from data.dataset import UnderstandDatasetCuda
from models.understanding import VanillaCNN
from torch.utils.data import DataLoader


def obstotensor(obs):
    if len(obs.shape) > 3:
        obs = obs.transpose((0, 3,1,2)) # keep batch dim
    else:
        obs = obs.transpose((2, 0, 1))
    return torch.from_numpy(obs)


def train_understand(model, Loss, opt, tloader, vloader, args, vis=None):
    min_loss = 9999
    for ep in range(args.epochs):
        train_loss = 0
        n = 0
        model.train()
        for states, obs in tloader:
            opt.zero_grad()
            obs, states = Variable(obs), Variable(states, requires_grad=False)
            # if args.cuda:
            #     obs, states = obs.cuda(), states.cuda()
            predicted_states = model(obs)
            loss = Loss(predicted_states, states)
            train_loss += loss.data[0]
            n += 1
            loss.backward()
            opt.step()

        train_loss /= n
        if vis:
            vis.line_update(Xdata=ep, Ydata=train_loss, name='Training Loss')
            vis.line_update(Xdata=ep, Ydata=math.log10(train_loss), name='Training LogLoss')

        # Validation
        val_loss = 0
        n = 0
        model.eval()
        for states, obs in vloader:
            obs, states = Variable(obs), Variable(states, requires_grad=False)
            # if args.cuda:
            #     obs, states = obs.cuda(), states.cuda()
            predicted_states = model(obs)
            vloss = Loss(predicted_states, states)
            val_loss += vloss.data[0]
            n += 1
        val_loss /= n

        if val_loss < min_loss:
            min_loss = val_loss
            name = os.path.join(args.checkpoint_dir,
                                'BestUnderDict{}_{}.pt'.format(ep, val_loss))
            sd = model.cpu().state_dict()
            torch.save(sd, name)
            if args.cuda:
                model.cuda()

        if vis:
            vis.line_update(Xdata=ep, Ydata=val_loss, name='Validation Loss')
            vis.line_update(Xdata=ep, Ydata=math.log10(val_loss), name='Validation LogLoss')

        if ep % args.save_interval == 0:
            name = os.path.join(args.checkpoint_dir,
                                'UnderDict{}_{}.pt'.format(ep, val_loss))
            sd = model.cpu().state_dict()
            torch.save(sd, name)
            if args.cuda:
                model.cuda()
        print('Epoch: {}/{}\t loss: {}\t Vloss:{}'.format(ep,
                                                          args.epochs,
                                                          train_loss,
                                                          val_loss ))


def test_understand(args):
    print('Loading training data')
    train_data = load_dict(args.train_target_path)
    train_dset = UnderstandDatasetCuda(train_data)
    train_dset.transform_to_cuda(vel=2)

    st, ob = train_dset[0]
    tloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    model = VanillaCNN(input_shape=ob.shape,
                       s_shape=st.shape[0],
                       feature_maps=args.feature_maps,
                       kernel_sizes=args.kernel_sizes,
                       strides=args.strides,
                       args=args)

    print('Loading:', args.state_dict_path)
    sd = torch.load(args.state_dict_path)
    model.load_state_dict(sd)

    if args.cuda:
        model.cuda()

    model.eval()
    for states, obs in tloader:
        obs, states = Variable(obs), Variable(states, requires_grad=False)
        predicted_states = model(obs)
        for i in range(5):
            print()
            print('-'*80)
            print('State:', states[i])
            print('Pred :', predicted_states[i])
            print('Diff :', (predicted_states[i]-states[i]).abs())
            print('-'*80)
        input('Press Enter to continue')


def main(args):
    make_log_dirs(args)
    args.num_updates   = int(args.num_frames) // args.num_steps // args.num_proc

    # # === Load Data ===
    # Not Enough room on GPU so regular data handling
    # print('Loading training data')
    # train_data = load_dict(args.train_target_path)
    # train_dset = UnderstandDataset(train_data)
    # print('Loading validation data')
    # val_data = load_dict(args.val_target_path)
    # val_dset = UnderstandDataset(val_data)
    # print('Done')

    # Enough room on GPU so everything is moved there at once
    print('Loading training data')
    train_data = load_dict(args.train_target_path)
    train_dset = UnderstandDatasetCuda(train_data)
    train_dset.transform_to_cuda(vel=2)
    print('Loading validation data')
    val_data = load_dict(args.val_target_path)
    val_dset = UnderstandDatasetCuda(val_data)
    val_dset.transform_to_cuda(vel=2)
    print('Done')

    # trainloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    # valloader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    trainloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    valloader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=True,  num_workers=0)

    # Check dims
    st, ob = train_dset[0]
    vst, vob = val_dset[0]
    assert vob.shape == ob.shape
    assert vst.shape == st.shape

    if args.verbose:
        print('train: ob.shape', ob.shape)
        print('train: st.shape', st.shape)
        print('val: ob.shape', vob.shape)
        print('val: st.shape', vst.shape)
        print('test: ob.shape', tob.shape)
        print('test: st.shape', tst.shape)
        print('State: \nmean: {}\ntype: {}'.format(st.mean(), type(st)))
        print('Obs: \nmean: {}\ntype: {}'.format(ob.mean(), type(ob)))
        input('Press Enter to continue')

    # === Model ===
    model = VanillaCNN(input_shape=ob.shape,
                       s_shape=st.shape[0],
                       feature_maps=args.feature_maps,
                       kernel_sizes=args.kernel_sizes,
                       strides=args.strides,
                       args=args)

    if args.continue_training:
        print('\n=== Continue Training ===\n')
        print('Loading:', args.state_dict_path)
        sd = torch.load(args.state_dict_path)
        model.load_state_dict(sd)

    Loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.cnn_lr)
    print('=== Model ===')
    print(model)

    # === Training ===
    if args.cuda:
        model.cuda()

    if not args.no_vis:
        from utils.vislogger import VisLogger
        vis = VisLogger(args)
        start = time.time()
        train_understand(model, Loss, opt, trainloader, valloader,  args, vis)
        duration = time.time() - start
    else:
        start = time.time()
        train_understand(model, Loss, opt, trainloader, valloader,  args)
        duration = time.time() - start

    print('Duration: ', duration)

if __name__ == '__main__':
    args = get_args()
    test_understand(args)
    # main()
