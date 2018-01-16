import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils.arguments import get_args
from utils.utils import make_log_dirs
from models.understanding import VanillaCNN
from torch.utils.data import DataLoader
from data.dataset import load_reacherplane_data

def obstotensor(obs):
    if len(obs.shape) > 3:
        obs = obs.transpose((0, 3,1,2)) # keep batch dim
    else:
        obs = obs.transpose((2, 0, 1))
    return torch.from_numpy(obs)

def train_understand(model, Loss, opt, tloader, vloader, args, vis=None):
    model.train()
    for ep in range(args.epochs):
        for states, obs in tloader:
            opt.zero_grad()

            obs = obs.permute(0,3,1,2).float()
            obs = obs / 255
            obs, states = Variable(obs.float()), Variable(states.float(), requires_grad=False)
            if args.cuda:
                obs, states = obs.cuda(), states.cuda()

            predicted_states = model(obs)
            loss = Loss(predicted_states, states)
            loss.backward()
            opt.step()

        # Validation
        for states, obs in vloader:
            obs = obs.permute(0,3,1,2).float()
            obs = obs / 255
            obs, states = Variable(obs.float()), Variable(states.float(), requires_grad=False)
            if args.cuda:
                obs, states = obs.cuda(), states.cuda()
            predicted_states = model(obs)
            vloss = Loss(predicted_states, states)

        if vis and ep > 10:
            # Draw plots
            vis.line_update(Xdata=ep, Ydata=loss.data[0], name='Training Loss')
            vis.line_update(Xdata=ep, Ydata=vloss.data[0], name='Validation Loss')

        if ep % args.save_interval == 0:
            name = os.path.join(args.checkpoint_dir,
                                'UnderDict{}_{}.pt'.format(ep, round(vloss.data[0], 3)))
            sd = model.cpu().state_dict()
            torch.save(sd, name)
            if args.cuda:
                model.cuda()
        print('Epoch: {}/{}\t loss: {}\t Vloss:{}'.format(ep,
                                                          args.epochs,
                                                          loss.data[0],
                                                          vloss.data[0]))

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
    args.num_updates   = int(args.num_frames) // args.num_steps // args.num_processes

    # # === Load Data ===
    train_dset = torch.load(args.target_path)
    val_dset = torch.load(args.target_path2)

    # busted dataset
    for i, (s, o) in enumerate(train_dset):
        if s.shape[0] is not 4:
            train_dset.state.pop(i)
            train_dset.obs.pop(i)
            print('removed index {} from train_dset: ', i)

    for i, (s, o) in enumerate(train_dset):
        if s.shape[0] is not 4:
            val_dset.state.pop(i)
            val_dset.obs.pop(i)
            print('removed index {} from val_dset: ', i)

    trainloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=True)

    # Check dims
    st, ob = train_dset[0]
    vst, vob = val_dset[0]
    assert vob.shape == ob.shape
    assert vst.shape == st.shape
    if args.verbose:
        print('ob.shape', ob.shape)
        print('st.shape', st.shape)
        print('ob.shape', vob.shape)
        print('st.shape', vst.shape)
        input('Press Enter to continue')

    ob = obstotensor(ob)
    # === Model ===
    model = VanillaCNN(input_shape=ob.shape,
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
        train_understand(model, Loss, opt, trainloader, valloader, args, vis)
    else:
        train_understand(model, Loss, opt, trainloader, valloader, args)

    test_dset = torch.load(args.target_path3)
    teloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True)
    test_understand(model, teloader, args)

if __name__ == '__main__':
    main()
