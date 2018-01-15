import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader

from project.dynamics_model.robotdataset import RobotDataset
from project.dynamics_model.model import CLSTM
from project.dynamics_model.logger import Logger
import datetime
import pickle
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr'                , type=float , default=1e-4)
    parser.add_argument('--batch_size'        , type=int   , default=64)
    parser.add_argument('--epochs'            , type=int   , default=50)
    parser.add_argument('--seq-len'           , type=int   , default=3)
    parser.add_argument('--num-num_processes' , type=int   , default=4)

    parser.add_argument('--log-dir', default="./"))
    return parser.parse_args()

def make_log_dirs(args):
    ''' Creates dirs:
        ../root/day/clstm/run/
        ../root/day/clstm/run/checkpoints
        ../root/day/clstm/run/results
    '''
    def get_today():
        t = datetime.date.today().ctime().split()[1:3]
        s = "".join(t)
        return s

    rootpath = args.log_dir
    day = get_today()
    type_ = 'clstm' + str(args.dof)
    if args.RGB:
        rootpath = os.path.join(rootpath, type_, dof, 'RGB')
    else:
        rootpath = os.path.join(rootpath, type_, dof)

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

def train(args, dloader, ):
    step = 0
    for epoch in range(args.epochs):
        Loss = []
        for args.batch, data in enumerate(tqdm(dloader)):
            model.reset_hidden()
            opt.zero_grad()
            x = Variable(data['rgb'])
            target = Variable(data['rgb_target'], requires_grad=False)

            if Cuda:
                x = x.cuda()
                target = target.cuda()

            out, state_list = model(x)
            r_loss = criterion(out[:, 0], target[:, 0])
            g_loss = criterion(out[:, 1], target[:, 1])
            b_loss = criterion(out[:, 2], target[:, 2])
            loss = r_loss + g_loss + b_loss
            loss.backward()

            opt.step()

            tmp_loss = loss.data[0]
            Loss.append(tmp_loss)

        logger.add_loss(loss, step)
        logger.add_parameter_data(model, step)
        step += 1

        avg_loss = np.array(Loss).mean()
        print('-'*45)
        print('Epoch: {}/{} \tLoss: {}'.format(epoch, args.pochs, avg_loss))
        img_in = x[0].data.cpu()
        img_out = out[0].data.cpu()
        img_target = target[0].data.cpu()
        epoch_imgs.append({'in': img_in, 'out': img_out, 'target': img_target})

def play():
    testing
    dataiter = iter(dloader)
    data = next(dataiter)

    x = Variable(data['rgb'])
    target = Variable(data['rgb_target'], requires_grad=False)

    out, _ = model(x)
    step=0
    logger.add_images(x[:10], out[:10], step)

def main():
    args = get_args()
    make_log_dirs(args)  # creates: args.log_dir, result_dir, checkpoint_dir

    # Load data
    # should use torch here... consistency.. DONT KNOW IF THIS WORKS:
    data = torch.load(open(args.load_dir))
    dset = RobotDataset(data, args.seq_len)
    dloader = DataLoader(dset, batch_size=args.batch, num_workers=args.num_processes)


    Cuda = torch.cuda.is_available()
    input_size = (3, 40, 40)
    model = CLSTM(input_size, out_channels=3, CUDA=Cuda)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    logger = Logger(logpath)

    print('Cuda set to: ',Cuda)
    if Cuda:
        model.CUDA = True
        model = model.cuda()

    print('Starting training')
    epoch_imgs = []

    train(args, dloader)
    logger._flush()
    model.cpu()
    model.save_state_dict(logpath+'/model_params.pt')


if __name__ == '__main__':
    main()
