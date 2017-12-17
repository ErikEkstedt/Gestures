import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .robotdataset import RobotDataset
from .model import CLSTM
from .logger import Logger
import datetime
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--sequence-length', type=int, default=3)
parser.add_argument('--output_folder', type=str, default="/home/erik/DATA/Project/logs")
parser.add_argument('--output_folder', type=str,
                    default="/home/erik/DATA/Project/logs")


def main():
    args = parser.parse_args()
    today = datetime.date.today()
    t = today.ctime().split()
    logpath = args.output_folder + '_' + t[1] + '_' + t[2]
    os.mkdir(logpath)
    run = 0
    while os.path.exists("%s/run-%d" % (logpath, run)):
            run += 1

    logpath = "%s/run-%d" % (logpath, run)
    os.mkdir(logpath)

    # Desktop path
    seq_len = args.sequence_length
    epochs = args.epochs
    batch = args.batch_size
    nworkers = 4

    f = "/home/erik/DATA/Project/reacher/data4.p"
    data = pickle.load(open(f, "rb"))
    dset = RobotDataset(data, seq_len)
    dloader = DataLoader(dset, batch_size=batch, num_workers=nworkers)

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

    # testing
    # dataiter = iter(dloader)
    # data = next(dataiter)
    #
    # x = Variable(data['rgb'])
    # target = Variable(data['rgb_target'], requires_grad=False)
    #
    # out, _ = model(x)
    # step=0
    # logger.add_images(x[:10], out[:10], step)

    step = 0
    for epoch in range(epochs):
        Loss = []
        for batch, data in enumerate(tqdm(dloader)):
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
        print('Epoch: {}/{} \tLoss: {}'.format(epoch, epochs, avg_loss))
        img_in = x[0].data.cpu()
        img_out = out[0].data.cpu()
        img_target = target[0].data.cpu()
        epoch_imgs.append({'in': img_in, 'out': img_out, 'target': img_target})
    logger._flush()
    model.save_state_dict(logpath+'/model_params.pt')


if __name__ == '__main__':
    main()
