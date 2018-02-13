''' Understanding Module.

This code is for the understanding/translation model. The function of this
module is to translate from RGB observations to state space (some space where
the coordination module operates)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from gesture.utils.utils import Conv2d_out_shape, ConvTranspose2d_out_shape

def total_params(p):
    n = 1
    for i in p:
        n *= int(i)
    return n


class VanillaCNN(nn.Module):
    ''' Simple CNN model RGB -> state
    - 3 Conv w/ stride 2
    - MLP Hidden layer
    - MLP output
    '''
    def __init__(self,
                 input_shape=(3,100,100),
                 s_shape=22,
                 feature_maps=[16, 32, 64],
                 kernel_sizes=[5, 5, 5],
                 strides=[2, 2, 2],
                 args=None):
        super(VanillaCNN, self).__init__()
        self.input_shape    = input_shape
        self.s_shape        = s_shape
        self.feature_maps   = feature_maps
        self.kernel_sizes   = kernel_sizes
        self.strides        = strides

        self.conv1        = nn.Conv2d(input_shape[0], feature_maps[0], kernel_size  = kernel_sizes[0], stride = strides[0])
        self.out_shape1   = Conv2d_out_shape(self.conv1, input_shape)
        self.conv2        = nn.Conv2d(feature_maps[0], feature_maps[1], kernel_size = kernel_sizes[1], stride = strides[1])
        self.out_shape2   = Conv2d_out_shape(self.conv2, self.out_shape1)
        self.conv3        = nn.Conv2d(feature_maps[1], feature_maps[2], kernel_size = kernel_sizes[2], stride = strides[2])
        self.out_shape3   = Conv2d_out_shape(self.conv3, self.out_shape2)
        self.n_out        = total_params(self.out_shape3)
        self.head         = nn.Linear(self.n_out, args.hidden)
        self.out          = nn.Linear(args.hidden, s_shape)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.head(x))
        return self.out(x)


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


def test_VanillaCNN(args):
    from gesture.agent.memory import StackedObs
    from gesture.environments.social import Social
    import numpy as np

    env = Social(args)
    env.seed(args.seed)

    CurrentObs = StackedObs(args.num_proc, args.num_stack, ob_shape)
    obs = np.random.rand(*(args.num_proc,*ob_shape))*255  # env returns numpy
    CurrentObs.update(obs)

    model = VanillaCNN(input_shape=CurrentObs.obs_shape,
                       s_shape=s_shape,
                       feature_maps=args.feature_maps,
                       kernel_sizes=args.kernel_sizes,
                       strides=args.strides,
                       args=args)
    if True:
        CurrentObs.cuda()
        model.cuda()

    in_ = Variable(CurrentObs())
    out = model(in_)

    print('OUT:\n')
    print('Out size:\n', out.data.size())
    print('\n\nDATA:')
    print('Out:\n', out.data)


if __name__ == '__main__':
    from gesture.utils.arguments import get_args
    args = get_args()
    test_VanillaCNN(args)


