'''
Here are the models for the "Coordination" module.

The Policy class contains relevant functions for all policies in this
PPO implementation (act, sample, evaluate_actions).

The MLPPolicy is an MLP/fully connected network (S,St) -> V, A, A_std
The CNNPolicy is an ConvNet. (O,Ot) -> V, A, A_std
'''
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from functools import reduce
import operator

from gesture.utils.utils import Conv2d_out_shape, ConvTranspose2d_out_shape

def total_params(p):
    n = 1
    for i in p:
        n *= int(i)
    return n

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(
            m.weight.data.pow(2).sum(1, keepdim=True))

        if m.bias is not None:
            m.bias.data.fill_(0)


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

    def sample(self, s, s_target, o=None, o_target=None):
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


class MLPPolicy(nn.Module, Policy):
    def __init__(self, input_size, a_shape, args):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, args.hidden)
        self.fc2 = nn.Linear(args.hidden, args.hidden)

        self.value = nn.Linear(args.hidden, 1)
        self.action = nn.Linear(args.hidden, a_shape)
        self.train()

        self.n         = 0
        self.total_n   = args.num_frames
        self.std_start = args.std_start
        self.std_stop  = args.std_stop

    def forward(self, s, st, o=None, ot=None):
        print(s.shape)
        print(st.shape)
        s_cat = torch.cat((s, st), dim=1)
        x = F.tanh(self.fc1(s_cat))
        x = F.tanh(self.fc2(x))
        v = self.value(x)
        ac_mean = self.action(x)
        ac_std = self.std(ac_mean)  #std annealing
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
