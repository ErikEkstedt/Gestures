''' Understanding Module.

This code is for the understanding/translation model. The function of this
module is to translate from RGB observations to state space (some space where
the coordination module operates)
'''

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from project.dynamics_model.utils import Conv2d_out_shape, ConvTranspose2d_out_shape
from project.dynamics_model.CLSTMCell import CLSTMCell

class CLSTM(nn.Module):
    '''A hardcoded small prediction model to test if the CLSTMCell works.'''
    def __init__(self, input_shape, out_channels, feature_list=[32, 64, 64], batch_first=True, CUDA=False):
        super(CLSTM, self).__init__()
        self.CUDA = CUDA
        self.batch_first = batch_first
        self.clstm_list = []
        self.hidden_states = None  # initialized as None, becomes a list after forward call

        self.feature_list = feature_list
        self.input_shape = input_shape
        self.in_channels = input_shape[0]

        self.out_channels = out_channels

        nfeats1 = feature_list[0]
        self.conv1 = nn.Conv2d(self.in_channels, nfeats1, kernel_size=5, stride=2)
        self.out_shape1 = Conv2d_out_shape(self.conv1, input_shape)  # output shape of layer

        self.clstm1 = CLSTMCell(self.out_shape1, nfeats1, kernel_size=5)
        self.clstm_list.append(self.clstm1)
        self.norm1 = nn.InstanceNorm2d(nfeats1)

        nfeats2 = feature_list[1]
        self.conv2 = nn.Conv2d(nfeats1, nfeats2, kernel_size=5, stride=2)
        self.out_shape2 = Conv2d_out_shape(self.conv2, self.out_shape1)  # output shape of layer
        self.clstm2 = CLSTMCell(self.out_shape2, nfeats2, kernel_size=5)
        self.clstm_list.append(self.clstm2)
        self.norm2 = nn.InstanceNorm2d(nfeats2)

        nfeats3 = feature_list[2]
        self.conv3_trans = nn.ConvTranspose2d(nfeats2, nfeats3, kernel_size=5, stride=2, output_padding=1)
        self.out_shape3 = ConvTranspose2d_out_shape(self.conv3_trans, self.out_shape2)  # output shape of layer
        self.clstm3 = CLSTMCell(self.out_shape3, nfeats3, kernel_size=5)
        self.clstm_list.append(self.clstm3)
        self.norm3 = nn.InstanceNorm2d(nfeats3)

        nfeats4 = self.out_channels
        self.conv4_trans = nn.ConvTranspose2d(nfeats3, nfeats4, kernel_size=5, stride=2, output_padding=1)
        self.out_shape4 = ConvTranspose2d_out_shape(self.conv4_trans, self.out_shape3)  # output shape of layer
        self.clstm4 = CLSTMCell(self.out_shape4, nfeats4, kernel_size=5)
        self.clstm_list.append(self.clstm4)
        self.norm4 = nn.InstanceNorm2d(nfeats4)

    def forward(self, input, hidden=None):
        if self.hidden_states is None:
            self.hidden_states = self.init_states(input.size(0))

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input.permute(1, 0, 2, 3, 4)

        seq_len = input.size(1)  # length of sequence
        if len(input) == 4:
            # assume (b,c,h,w)
            seq_len = 1

        # Sequence
        for t in range(seq_len):
            x = self.conv1(input[:, t])  # the t:th sequence datum
            x, (h1, c1) = self.clstm1(x, self.hidden_states[0])
            x = self.norm1(x)
            self.hidden_states[0] = (h1, c1)  # update hidden state

            x = self.conv2(x)
            x, (h2, c2) = self.clstm2(x, self.hidden_states[1])
            x = self.norm2(x)
            self.hidden_states[1] = (h2, c2)  # update hidden state

            x = self.conv3_trans(x)
            x, (h3, c3) = self.clstm3(x, self.hidden_states[2])
            x = self.norm3(x)
            self.hidden_states[2] = (h3, c3)  # update hidden state

            x = self.conv4_trans(x)
            x, (h4, c4) = self.clstm4(x, self.hidden_states[3])
            x = self.norm4(x)
            self.hidden_states[3] = (h4, c4)  # update hidden state

        return x, self.hidden_states  # might return self.hidden

    def init_states(self, batch_size):
        states = []
        for i in range(len(self.clstm_list)):
            states.append(self.clstm_list[i].init_state(batch_size, CUDA=self.CUDA))
        return states

    def reset_hidden(self):
        self.hidden_states = None

class CNNAutoencoder(nn.Module):
    def __init__(self, input_size, action_shape,
                    hidden=128,
                    std_start=-0.6,
                    std_stop=-1.7,
                    total_frames=1e6):
        super(CNNAutoencoder, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(input, hidden)
        self.conv2 = nn.Conv2d()

    def forward(self, input):
        pass

class Translation(nn.Module):
    def __init__(self, input_shape=(3,100,100),
                    action_shape=2,
                    in_channels=3,
                    feature_maps=[64, 64, 64],
                    kernel_sizes=[5, 5, 5],
                    strides=[2, 2, 2],
                    args=None):

            super(CNNPolicy, self).__init__()
            self.input_shape  = input_shape
            self.action_shape = action_shape
            self.feature_maps = feature_maps
            self.kernel_sizes = kernel_sizes
            self.strides      = strides

            self.conv1          = nn.Conv2d(input_shape[0], feature_maps[0], kernel_size  = kernel_sizes[0], stride = strides[0])
            self.out_shape1     = Conv2d_out_shape(self.conv1, input_shape)
            self.conv2          = nn.Conv2d(feature_maps[0], feature_maps[1], kernel_size = kernel_sizes[1], stride = strides[1])
            self.out_shape2     = Conv2d_out_shape(self.conv2, self.out_shape1)
            self.conv3          = nn.Conv2d(feature_maps[1], feature_maps[2], kernel_size = kernel_sizes[2], stride = strides[2])
            self.out_shape3     = Conv2d_out_shape(self.conv3, self.out_shape2)
            self.total_conv_out = total_params(self.out_shape3)

            self.fc1       = nn.Linear(self.total_conv_out, args.hidden)
            self.value     = nn.Linear(args.hidden, 1)
            self.action    = nn.Linear(args.hidden, action_shape)
            self.train()

            self.n         = 0
            self.total_n   = args.num_frames
            self.std_start = args.std_start
            self.std_stop  = args.std_stop

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = F.tanh(self.fc1(x))
            v = self.value(x)
            ac_mean = self.action(x)
            ac_std = self.std(ac_mean)  #std annealing
            return v, ac_mean, ac_std


