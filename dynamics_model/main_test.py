import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from robotdataset import RobotDataset
# from model import CLSTM
from new_model import Model_older
from logger import Logger


def LossFunction_sep(rgb_out, state_out, rgb_target, state_target, criterion=nn.MSELoss()):
    r_loss = criterion(rgb_out[:, 0], rgb_target[:, 0])
    g_loss = criterion(rgb_out[:, 1], rgb_target[:, 1])
    b_loss = criterion(rgb_out[:, 2], rgb_target[:, 2])
    rgb_loss = r_loss + g_loss + b_loss
    state_loss = criterion(state_out, state_target)
    return rgb_loss, state_loss


def LossFunction_unit(rgb_out, state_out, rgb_target, state_target, criterion=nn.MSELoss()):
    r_loss = criterion(rgb_out[:, 0], rgb_target[:, 0])
    g_loss = criterion(rgb_out[:, 1], rgb_target[:, 1])
    b_loss = criterion(rgb_out[:, 2], rgb_target[:, 2])
    state_loss = criterion(state_out, state_target)
    loss = r_loss + g_loss + b_loss + state_loss
    return loss


def validation(model, criterion, vloader, nImgs=5):
    model.eval()
    Loss = []
    for batch, data in enumerate(tqdm(vloader)):
        model.reset_hidden()

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

        tmp_loss = loss.data[0]
        Loss.append(tmp_loss)
    return np.array(Loss).mean()


def train_unit_loss(model, opt, criterion, epoch, tloader, Cuda=False, nImgs=5):
    model.train()
    Loss = []
    print('Epoch: ', epoch)
    for batch, data in enumerate(tqdm(tloader)):
        # Reset
        model.reset_hidden()
        opt.zero_grad()

        # Data
        rgb = Variable(data['rgb'])
        s = Variable(data['state']).float()
        a = Variable(data['action']).float()
        rgb_target = Variable(data['rgb_target'], requires_grad=False)
        state_target = Variable(data['state_target'], requires_grad=False).float()

        if Cuda:
            rgb = rgb.cuda()
            s = s.cuda()
            a = a.cuda()
            rgb_target = rgb_target.cuda()
            state_target = state_target.cuda()

        rgb_out, state_out, state_list = model(rgb, state=s, action=a)

        s_out = state_out[:, -1]
        tot_loss = LossFunction_unit(rgb_out, s_out, rgb_target, state_target, criterion)
        tot_loss.backward()
        opt.step()

        Loss.append(tot_loss.data[0])

    avg_loss = np.array(Loss).mean()
    return avg_loss, rgb_target[:nImgs], rgb_out[:nImgs]


# Hyperparameters
seq_len = 3
epochs = 200
batch = 64
nworkers = 4

# Save path and logger
resultpath = "/home/erik/DATA/Project"
logger = Logger(resultpath)

# Desktop path
trainfile = "/home/erik/DATA/Project/data/data0.p"
data = pickle.load(open(trainfile, "rb"))
dset = RobotDataset(data, seq_len)
tloader = DataLoader(dset, batch_size=batch, num_workers=nworkers, shuffle=True)

# valfile = "/home/erik/DATA/Project/data/data1.p"
# data = pickle.load(open(valfile, "rb"))
# dset = RobotDataset(data, seq_len)
# vloader = DataLoader(dset, batch_size=batch, num_workers=nworkers, shuffle=True)

# Model
Cuda = torch.cuda.is_available()
print('Cuda set to: ', Cuda)

input_size = (3, 40, 40)
# model = CLSTM(input_size, out_channels=3, CUDA=Cuda)
model = Model_older(input_size, CUDA=Cuda)
if Cuda:
    model.CUDA = True
    model = model.cuda()

opt = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()


# training
print('Starting training')
step = 0
for epoch in range(1, epochs + 1):
    loss, img_target, out = train_unit_loss(model, opt, criterion, epoch, tloader, Cuda=Cuda)
    logger.add_loss(loss, step, name='Combined loss')
    logger.add_images(img_target, out, step)
    logger.add_parameter_data(model, step)
    logger.save_checkpoint(model, opt, epoch, score=loss)
    step += 1
    print('-' * 65)
    print('Epoch: {}/{} \tLoss: {}'.format(epoch, epochs, loss))
