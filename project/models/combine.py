import copy
import math
from functools import reduce
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from project.dynamics_model.utils import Conv2d_out_shape, ConvTranspose2d_out_shape
from project.dynamics_model.CLSTMCell import CLSTMCell


def total_params(p):
    n = 1
    for i in p:
        n *= int(i)
    return n


class Policy(object):
    """ Super Class for Policies
    Functions:

    : evaluate_actions : In(s_t, actions), Out(value, a_logprobs, dist_entropy)
    : sample           : In(s_t), Out(value, a, a_logprobs, a_logstd)
    : act              : In(s_t), Out(value, action)
    : std              : In(x), Out(logstd)
    : get_std          : In( ), Out(std)

    Superclass for the different policies (CNN/MLP) containing common funcs.
    """
    def evaluate_actions(self, s, s_target, o , o_target, actions):
        ''' requires_Grad=True for all in training'''
        actions = Variable(actions)
        o, o_target = Variable(o), Variable(o_target)
        s, s_target = Variable(s), Variable(s_target)
        v, action_mean, action_logstd = self(s, s_target, o, o_target)
        action_std = action_logstd.exp()

        # calculate `old_log_probs` directly in exploration.
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2)\
            - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)

        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, action_log_probs, dist_entropy

    def sample(self, s, s_target, o, o_target):
        ''' volatile input here during exploration. We want gradients at training'''
        s, s_target = Variable(s, volatile=True), Variable(s_target, volatile=True)
        o, o_target = Variable(o, volatile=True), Variable(o_target, volatile=True)

        v, action_mean, action_logstd = self(s, s_target, o, o_target)
        action_std = action_logstd.exp()

        noise = Variable(torch.randn(action_std.size()))
        if action_mean.is_cuda:
            noise = noise.cuda()
        action = action_mean +  action_std * noise

        # calculate `old_log_probs` directly in exploration.
        action_log_probs = -0.5 * ((action - action_mean) / action_std).pow(2)\
            - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)

        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, action, action_log_probs, action_std

    def act(self, s, s_target, o , o_target):
        o, o_target = Variable(o, volatile=True), Variable(o_target, volatile=True)
        s, s_target = Variable(s, volatile=True), Variable(s_target, volatile=True)
        v, action, _ = self(s, s_target, o, o_target)
        return v, action


class MLP(nn.Module):
    def __init__(self, input_size, action_shape, args):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, args.hidden)
        self.fc2 = nn.Linear(args.hidden, args.hidden)

        self.value = nn.Linear(args.hidden, 1)
        self.action = nn.Linear(args.hidden, action_shape)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        v = self.value(x)
        ac_mean = self.action(x)
        return v, ac_mean


class PixelEmbedding(nn.Module):
    ''' Simple CNN model RGB -> emb
    - 3 Conv w/ stride 2
    '''
    def __init__(self,
                 input_shape=(3,100,100),
                 state_shape=22,
                 feature_maps=[16, 32, 64],
                 kernel_sizes=[5, 5, 5],
                 strides=[2, 2, 2],
                 args=None):

        super(PixelEmbedding, self).__init__()
        self.input_shape  = input_shape
        self.state_shape  = state_shape
        self.feature_maps = feature_maps
        self.kernel_sizes = kernel_sizes
        self.strides      = strides

        self.conv1        = nn.Conv2d(input_shape[0], feature_maps[0], kernel_size  = kernel_sizes[0], stride = strides[0])
        self.out_shape1   = Conv2d_out_shape(self.conv1, input_shape)
        self.conv2        = nn.Conv2d(feature_maps[0], feature_maps[1], kernel_size = kernel_sizes[1], stride = strides[1])
        self.out_shape2   = Conv2d_out_shape(self.conv2, self.out_shape1)
        self.conv3        = nn.Conv2d(feature_maps[1], feature_maps[2], kernel_size = kernel_sizes[2], stride = strides[2])
        self.out_shape3   = Conv2d_out_shape(self.conv3, self.out_shape2)
        self.n_out        = total_params(self.out_shape3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)


class CombinePolicy(nn.Module, Policy):
        ''' Policy that uses both state and obs
        self(o, o_, s, s_)  : o,s = current state/obs, o_,s_ = target state/obs
        '''
        def __init__(self,
                    s_shape,
                    st_shape,
                    o_shape,
                    ot_shape,
                    a_shape,
                    feature_maps=[64, 32, 16],
                    kernel_sizes=[5, 5, 5],
                    strides=[2, 2, 2],
                    args=None):

            super(CombinePolicy, self).__init__()
            self.o_shape = o_shape
            self.s_shape = s_shape
            self.a_shape = a_shape

            self.ot_shape = ot_shape
            self.st_shape = st_shape

            self.in_channels_cat = o_shape[0]+ot_shape[0]
            self.obs_shape = (self.in_channels_cat, *o_shape[1:])

            self.cnn = PixelEmbedding(self.obs_shape,
                                    feature_maps=feature_maps,
                                    kernel_sizes=kernel_sizes,
                                    strides=strides,
                                    args=None)

            self.nparams_emb = self.cnn.n_out + s_shape + st_shape
            self.mlp = MLP(self.nparams_emb, a_shape, args)
            self.train()

            # self.n         = self.mlp.n
            self.n         = 0
            self.total_n   = args.num_frames
            self.std_start = args.std_start
            self.std_stop  = args.std_stop

        def forward(self, s, st, o, ot):
            o_cat = torch.cat((o, ot), dim=1)
            s_cat = torch.cat((s, st), dim=1)
            x = self.cnn(o_cat)

            x = torch.cat((x, s_cat), dim=1)
            v, ac_mean = self.mlp(x)
            ac_std = self.std(ac_mean)
            return v, ac_mean, ac_std

        def std(self, x):
            ''' linearly decreasing standard deviation '''
            ratio = self.n/self.total_n
            self.log_std_value = self.std_start - (self.std_start - self.std_stop)*ratio
            std = torch.FloatTensor([self.log_std_value])
            ones = torch.ones(x.data.size())
            if x.is_cuda:
                std = std.cuda()
                ones=ones.cuda()
            std = std*ones
            std = Variable(std)
            return std

        def get_std(self):
            return math.exp(self.log_std_value)

        def total_parameters(self):
            p = 0
            for parameter in self.parameters():
                tmp_params = reduce(operator.mul, parameter.shape)
                p += tmp_params
            return p

class Combine_NoTargetState(nn.Module, Policy):
    def __init__(self,
                 s_shape,
                 st_shape,
                 o_shape,
                 ot_shape,
                 a_shape,
                 feature_maps=[64, 32, 16],
                 kernel_sizes=[5, 5, 5],
                 strides=[2, 2, 2],
                 args=None):

        super(Combine_NoTargetState, self).__init__()
        self.o_shape = o_shape
        self.s_shape = s_shape
        self.a_shape = a_shape

        self.ot_shape = ot_shape
        self.st_shape = st_shape

        self.in_channels_cat = o_shape[0]+ot_shape[0]
        self.obs_shape = (self.in_channels_cat, *o_shape[1:])

        self.cnn = PixelEmbedding(self.obs_shape,
                                  feature_maps=feature_maps,
                                  kernel_sizes=kernel_sizes,
                                  strides=strides,
                                  args=None)

        self.nparams_emb = self.cnn.n_out + s_shape
        self.mlp = MLP(self.nparams_emb, a_shape, args)
        self.train()

        # self.n         = self.mlp.n
        self.n         = 0
        self.total_n   = args.num_frames
        self.std_start = args.std_start
        self.std_stop  = args.std_stop

    def forward(self, s, st, o, ot):
        o_cat = torch.cat((o, ot), dim=1)
        x = self.cnn(o_cat)
        x = torch.cat((x, s), dim=1)
        v, ac_mean = self.mlp(x)
        ac_std = self.std(ac_mean)
        return v, ac_mean, ac_std

    def std(self, x):
        ''' linearly decreasing standard deviation '''
        ratio = self.n/self.total_n
        self.log_std_value = self.std_start - (self.std_start - self.std_stop)*ratio
        std = torch.FloatTensor([self.log_std_value])
        ones = torch.ones(x.data.size())
        if x.is_cuda:
            std = std.cuda()
            ones=ones.cuda()
        std = std*ones
        std = Variable(std)
        return std

    def get_std(self):
        return math.exp(self.log_std_value)

    def total_parameters(self):
        p = 0
        for parameter in self.parameters():
            tmp_params = reduce(operator.mul, parameter.shape)
            p += tmp_params
        return p


def test_combinepolicy(args):
    ''' Test for CombinePolicy '''
    s_shape        = 22
    o_shape        = (3, 40, 40)
    ot_shape = o_shape
    st_shape = 4
    a_shape        = 2

    # Tensors
    s  = torch.rand(s_shape).unsqueeze(0)
    s_ = torch.rand(st_shape).unsqueeze(0)
    o  = torch.rand(o_shape).unsqueeze(0)
    o_ = torch.rand(ot_shape).unsqueeze(0)

    pi = CombinePolicy(o_shape, ot_shape, s_shape, st_shape, a_shape, args)

    print('pi.sample()')
    v, a, a_logprobs, a_std = pi.sample(s, s_target, o, o_target)
    print('Value:', v.shape)
    print('a_mean:', a.shape)
    print('a_logprobs:', a_logprobs.shape)
    print('a_std:', a_std.shape)

    print('pi.act()')
    v, a = pi.act(s, s_, o, o_)
    print('Value:', v.shape)
    print('a:', a.shape)


if __name__ == '__main__':
    from project.utils.arguments import get_args
    args = get_args()
    test_combinepolicy(args)
