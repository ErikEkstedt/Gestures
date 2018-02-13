import os
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
        vis.line_update(Xdata=ep, Ydata=train_loss, name='Training Loss')

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

        vis.line_update(Xdata=ep, Ydata=val_loss, name='Validation Loss')
        if ep % args.save_interval == 0:
            name = os.path.join(args.checkpoint_dir,
                                'UnderDict{}_{}.pt'.format(ep, round(vloss.data[0], 3)))
            sd = model.cpu().state_dict()
            torch.save(sd, name)
            if args.cuda:
                model.cuda()
        print('Epoch: {}/{}\t loss: {}\t Vloss:{}'.format(ep,
                                                          args.epochs,
                                                          train_loss,
                                                          val_loss ))

def test_understand(model, tloader, args):
    model.eval()
    for obs, states in tloader:
        obs, states = Variable(obs), Variable(states, volatile=True)
        if args.cuda:
            obs, states = obs.cuda(), states.cuda()
        predicted_states = model(obs)
        for i in range(5):
            print('State:', states[i])
            print('Pred :', predicted_states[i])
            print('Diff :', (predicted_states[i]-states[i]).abs())
        input('Press Enter to continue')

def main():
    args = get_args()
    make_log_dirs(args)
    args.num_updates   = int(args.num_frames) // args.num_steps // args.num_proc

    # # === Load Data ===
    # print('Loading training data')
    # train_data = load_dict(args.train_target_path)
    # train_dset = UnderstandDataset(train_data)
    # print('Loading validation data')
    # val_data = load_dict(args.val_target_path)
    # val_dset = UnderstandDataset(val_data)
    # print('Done')

    print('Loading training data')
    train_data = load_dict(args.train_target_path)
    train_dset = UnderstandDatasetCuda(train_data)
    print('Loading validation data')
    val_data = load_dict(args.val_target_path)
    val_dset = UnderstandDatasetCuda(val_data)
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
    model = VanillaCNN(input_shape=tuple(ob.shape),
                       s_shape=st.shape[0],
                       feature_maps=args.feature_maps,
                       kernel_sizes=args.kernel_sizes,
                       strides=args.strides,
                       args=args)

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
    # test_dset = torch.load(args.test_target_path)
    # teloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True)
    # test_understand(model, teloader, args)

if __name__ == '__main__':
    main()
