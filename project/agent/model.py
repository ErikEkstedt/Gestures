import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from project.dynamics_model.utils import Conv2d_out_shape, ConvTranspose2d_out_shape


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(
            m.weight.data.pow(2).sum(1, keepdim=True))

        if m.bias is not None:
            m.bias.data.fill_(0)

def total_params(p):
    n = 1
    for i in p:
        n *= int(i)
    return n


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


class Policy(object):
    """ evaluate_actions, sample, act, std, get_std are
    used in both MLPPolicy and CNNPolicy. should combine
    TODO:
        Get ready... just had som e flow..not tested at all.
    """
    def __init__(self,
                 state_shape,
                 obs_shape,
                 action_shape,
                 args):
        if CNN:
            self.pi = CNNPolicy(input_shape=obs_shape,
                                action_shape=action_shape,
                                in_channels=3,
                                feature_maps=[64, 64, 64],
                                kernel_sizes=[5, 5, 5],
                                strides=[2, 2, 2],
                                hidden=128,
                                args=args)

        else:
            self.pi = MLPPolicy(state_shape, action_shape, args)

    def __call__(self, x):
        return self.pi(x)

    def evaluate_actions(self, x, actions):
        v, action_mean, action_logstd = self(x)
        action_std = action_logstd.exp()

        # calculate `old_log_probs` directly in exploration.
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2)\
            - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)

        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, action_log_probs, dist_entropy

    def sample(self, s_t):
        input = Variable(s_t, volatile=True)
        v, action_mean, action_logstd = self(input)
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

    def act(self, s_t):
        input = Variable(s_t, volatile=True)
        v, action, _ = self(input)
        return v, action

    def std(self, x):
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


class MLPPolicy(nn.Module):
    def __init__(self, input_size, action_shape, args):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, args.hidden)
        self.fc2 = nn.Linear(args.hidden, args.hidden)

        self.value = nn.Linear(args.hidden, 1)
        self.action = nn.Linear(args.hidden, action_shape)
        self.train()

        self.n         = 0
        self.total_n   = args.num_frames
        self.std_start = args.std_start
        self.std_stop  = args.std_stop

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        v = self.value(x)
        ac_mean = self.action(x)
        ac_std = self.std(ac_mean)  #std annealing
        return v, ac_mean, ac_std

    def evaluate_actions(self, x, actions):
        v, action_mean, action_logstd = self(x)
        action_std = action_logstd.exp()

        # calculate `old_log_probs` directly in exploration.
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2)\
            - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)

        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, action_log_probs, dist_entropy

    def sample(self, s_t):
        input = Variable(s_t, volatile=True)
        v, action_mean, action_logstd = self(input)
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

    def act(self, s_t):
        input = Variable(s_t, volatile=True)
        v, action, _ = self(input)
        return v, action

    def std(self, x):
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


class CNNPolicy(nn.Module):
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

    def evaluate_actions(self, x, actions):
        v, action_mean, action_logstd = self(x)
        action_std = action_logstd.exp()

        # calculate `old_log_probs` directly in exploration.
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2)\
            - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)

        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, action_log_probs, dist_entropy

    def sample(self, s_t):
        input = Variable(s_t, volatile=True)
        v, action_mean, action_logstd = self(input)
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

    def act(self, s_t):
        input = Variable(s_t, volatile=True)
        v, action, _ = self(input)
        return v, action

    def get_std(self):
        return math.exp(self.log_std_value)

    def std(self, x):
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


class Obs_stats(object):
    ''' Not very good to do on tasks requiring data about target
    in the state data '''
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs)
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)

    def observes(self, obs):
        # observation mean var updates
        x = obs
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = self.var.sqrt()
        inputs = inputs
        return torch.clamp((inputs-self.mean)/obs_std, -5., 5.)


# ========= Tests ================
def test_roboschool(args):
    """test using roboschool """
    import roboschool
    import gym
    import numpy as np

    env_id = 'RoboschoolReacher-v1'
    env = gym.make(env_id)

    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]

    # obs_stats = Shared_obs_stats(ob_shape)
    obs_stats = Obs_stats(ob_shape)
    pi = MLPPolicy(ob_shape, ac_shape, args)
    print(pi)

    s = env.reset()
    s = torch.from_numpy(s).float()
    for i in range(100):
        print('s: ', s)
        # obs_stats.observes(s)
        # s = obs_stats.normalize(s)
        # print('s_norm: ',s)
        input('Press Enter to continue')
        v, ac, ac_log_probs, ac_std = pi.sample(s)
        print(ac)
        s, r, done, _ = env.step(ac[0].data.numpy())
        s = torch.from_numpy(s).float()
        print(r)

def test_CNNPolicy(args):
    ''' WORKS '''
    from project.environments.utils import rgb_tensor_render
    from project.agent.memory import StackedObs
    import numpy as np

    ob_shape = (64,64,3)
    ac_shape = 2
    CurrentObs = StackedObs(args.num_processes, args.num_stack, ob_shape)

    pi = CNNPolicy(input_shape=CurrentObs.obs_shape,
                   action_shape=ac_shape,
                   in_channels=3,
                   feature_maps=[64, 64, 64],
                   kernel_sizes=[5, 5, 5],
                   strides=[2, 2, 2],
                   args=args)


    obs = np.random.rand(*(args.num_processes,*ob_shape))*255  # env returns numpy
    CurrentObs.update(obs)

    print('-'*55)
    print('\nIN:\n', CurrentObs().size())

    if True:
        CurrentObs.cuda()
        pi.cuda()

    v, action_mean, action_logstd, action_std = pi.sample(CurrentObs())
    print('OUT:\n')
    print('Value:\n'         , v.size())
    print('Action_mean:\n'   , action_mean.size())
    print('Action_logstd:\n' , action_logstd.size())
    print('Action_std:\n'    , action_std.size())
    if False:
        print('\n\nDATA:')
        print('Value:\n'         , v)
        print('Action_mean:\n'   , action_mean)
        print('Action_logstd:\n' , action_logstd)
        print('Action_std:\n'    , action_std)


if __name__ == '__main__':
    from project.agent.arguments import get_args
    args = get_args()
    test_CNNPolicy(args)
    # test_roboschool(args)
