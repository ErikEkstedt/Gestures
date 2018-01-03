import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils.arguments import get_args
from utils.utils import make_log_dirs
from data.dataset import load_dataset

from models.understanding import VanillaCNN


def train_understand(model, Loss, opt, dloader, vloader, args, vis=None):
    model.train()
    for ep in range(args.understand_epochs):
        # Training
        for obs, states in dloader:
            obs, states = Variable(obs), Variable(states, volatile=True)
            if args.cuda:
                obs, states = obs.cuda(), states.cuda()

            predicted_states = model(obs)
            loss = Loss(predicted_states, states)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation
        for obs, states in dloader:
            obs, states = Variable(obs), Variable(states, volatile=True)
            if args.cuda:
                obs, states = obs.cuda(), states.cuda()
            predicted_states = model(obs)
            vloss = Loss(predicted_states, states)

        # Log, Plot and Save
        if vis:
            # Draw plots
            vis.line_update(Xdata=ep, Ydata=loss.data[0], name='Training Loss')
            vis.line_update(Xdata=ep, Ydata=vloss.data[0], name='Validation Loss')

        if ep % args.save_interval == 0:
            # ==== Save model ======
            name = os.path.join(args.checkpoint_dir,
                                'UnderDict{}_{}.pt'.format(ep, round(vloss.data[0], 3)))
            sd = model.cpu().state_dict()
            torch.save(sd, name)
            if args.cuda:
                model.cuda()

        print('Epoch: {}/{}\t loss: {}\t Vloss:{}'.format(ep,
                                                          args.understand_epochs,
                                                          loss.data[0],
                                                          vloss.data[0]))


def main():
    args = get_args()
    make_log_dirs(args)
    args.num_updates   = int(args.num_frames) // args.num_steps // args.num_processes

    if not args.no_vis:
        from utils.vislogger import VisLogger
        vis = VisLogger(args)

    # === Load Data ===
    path = '/home/erik/DATA/project/ReacherPlane/obsdata_rgb40-40-3_n500000_0.pt'
    # path = '/home/erik/DATA/project/ReacherPlane/obsdata_rgb40-40-3.pt'
    dset, dloader = load_dataset(path)

    val_path = '/home/erik/DATA/project/ReacherPlane/obsdata_rgb40-40-3.pt'
    val_dset, vloader = load_dataset(val_path)

    # Check dims
    ob, st = dset[0]
    vob, vst = val_dset[0]
    ob_shape, vob_shape =  ob.shape, vob.shape
    st_shape, vst_shape =  st.shape, vst.shape
    assert vob_shape == ob_shape
    assert vst_shape == st_shape
    if args.verbose:
        print('ob_shape', ob_shape)
        print('st_shape', st_shape)
        print('ob_shape', vob_shape)
        print('st_shape', vst_shape)
        input('Press Enter to continue')

    # === Model ===
    model = VanillaCNN(input_shape=ob_shape,
                       state_shape=st_shape[0],
                       feature_maps=[64, 64, 64],
                       kernel_sizes=[5, 5, 5],
                       strides=[2, 2, 2],
                       args=args)
    Loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.understand_lr)

    if args.cuda:
        model.cuda()

    # === Training ===
    if not args.no_vis:
        train_understand(model, Loss, opt, dloader, vloader, args, vis)
    else:
        train_understand(model, Loss, opt, dloader, vloader, args)











if __name__ == '__main__':
    main()
