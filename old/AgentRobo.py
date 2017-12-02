import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import copy
import math

# from utils import Conv2d_out_shape, ConvTranspose2d_out_shape
Conv2d_out_shape, ConvTranspose2d_out_shapea = None, None
try:
    from memory import RolloutStorage, StackedState
except:
    from Agent.memory import RolloutStorage, StackedState
# from running_stat import ObsNorm

# This script is heavily inspired by
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

def total_params(p):
    n = 1
    for i in p:
        n *= int(i)
    return n

def fixed_std_function(std=0.5):
    '''Function to use fixed std in DiagonalGaussian'''
    def forward_pass(x, std=std):
        a = torch.ones(x.size(0),1)
        return Variable(torch.log(torch.Tensor([std])) * a)
    return forward_pass


class AddBias(nn.Module):
        ''' Custom "layer" that adds a bias. Trainable nn.Parameter '''
        def __init__(self, size):
            super(AddBias, self).__init__()
            self.size = size
            self.std = nn.Parameter(torch.zeros(size).unsqueeze(1))

        def forward(self, x):
            return x + self.std.t().view(1, -1)

        def __repr__(self):
            return self.__class__.__name__ + '(' + str(self.size) + ')'


class AddBias_fixed(nn.Module):
        ''' Custom "layer" that adds a bias. Trainable nn.Parameter '''
        def __init__(self, std, action_max):
            super(AddBias_fixed, self).__init__()
            self.std = torch.Tensor([std]) * action_max

        def forward(self, x):
            a = Variable(torch.ones(x.size(0),1) * self.std)
            if x.is_cuda:
                a = a.cuda()
            return a

        def __repr__(self):
            return self.__class__.__name__ + '(' + str(self.std[0]) + ')'


class DiagonalGaussian(nn.Module):
    ''' Diagonal Gaussian used as the head of the policy networks'''
    def __init__(self, num_inputs, num_outputs, fixed_std=False, std=None):
        super(DiagonalGaussian, self).__init__()
        self.mean = nn.Linear(num_inputs, num_outputs)
        if fixed_std:
            self.logstd = AddBias_fixed(std)
        else:
            self.logstd = AddBias(num_outputs)
        weights_init_mlp(self)
        self.train()

    def forward(self, x):
        action_mean = F.tanh(self.mean(x))  # tanh to constrain co-domain, image of function.
        zeros = Variable(torch.zeros(action_mean.size()), volatile=x.volatile)
        if x.is_cuda:
            zeros = zeros.cuda()
            action_mean = action_mean.cuda()

        action_logstd = self.logstd(zeros)
        return action_mean, action_logstd

    def cuda(self, *args):
        super(DiagonalGaussian).cuda()


class PolicyBody(nn.Module):
    ''' Todo: should be dynamic in amounts of layers'''
    def __init__(self, num_inputs, hidden=64, input_filter=False):
        super(PolicyBody, self).__init__()
        self.num_inputs = num_inputs
        self.hidden = hidden
        self.input_filter = input_filter

        if input_filter:
            self.obs_filter = ObsNorm((1,num_inputs),clip=5)

        self.body1 = nn.Linear(num_inputs, hidden)
        self.body2 = nn.Linear(hidden, hidden)
        self.head_value = nn.Linear(hidden, 1)
        weights_init_mlp(self)
        self.train()

    def forward(self, input):
        # input.data = self.obs_filter(input.data)  # Not part of Computation Graph
        x = F.relu(self.body1(input))
        x = F.relu(self.body2(x))
        return self.head_value(x), x

    def cuda(self, **args):
        if self.input_filter:
            self.obs_filter.cuda()
        super(PolicyBody, self).cuda(**args)
        print('obs cuda')

    def cpu(self, **args):
        super(PolicyBody, self).cpu(**args)


class Policy(nn.Module):
    def __init__(self, input_size, action_shape, hidden=64, fixed_std=False, std=None):
        super(Policy, self).__init__()
        self.body = PolicyBody(input_size, hidden=hidden)
        self.head = DiagonalGaussian(hidden, action_shape, fixed_std=fixed_std, std=std)

    def forward(self, input):
        assert type(input) is Variable, 'input to' + self.__name__ + 'not a variable'
        v, x = self.body(input)
        action_mean, action_logstd = self.head(x)
        return v, action_mean, action_logstd

    def reset_parameters(self):
        self.apply(weights_init_mlp)
        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """
        if self.head.__class__.__name__ == "DiagonalGaussian":
            self.head.mean.weight.data.mul_(0.01)


class AgentRoboSchool(object):
    ''' Agent with MLP policy PI( a_t | s_t)

    This agent is only for standard Roboschool Tasks.

    :param   stacked_state_shape            (176,)
    :param   action_shape                   (17,)

    :param   optimizer_pi                   torch.optim
    :param   hidden                         int, number of hidden neurons
    :param   fixed_std                      boolean,fix the std of actions
    :param   std                            float, value of std if fixed
    '''

    def __init__(self, args, stacked_state_shape=(176,), action_shape=17, hidden=64, fixed_std=False, std=0.5):
        # Data
        if len(stacked_state_shape)>1:
            self.stacked_state = stacked_state_shape
        else:
            self.stacked_state = stacked_state_shape[0]

        self.action_shape = action_shape[0]  # action state size
        self.args = args                 # Argparse arguments

        self.tmp_steps = 0
        self.use_cuda = args.cuda

        # ======= Policy ========
        self.hidden = hidden
        self.policy = Policy(self.stacked_state,
                             self.action_shape,
                             hidden=hidden,
                             fixed_std=args.fixed_std,
                             std=std)
        self.old_policy = copy.deepcopy(self.policy)
        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=args.pi_lr)

    def get_std(self):
        return self.policy.head.logstd.std

    def sample(self, s_t, deterministic=False):
        '''Samples an action based on input.
        Computes an action (deterministic or stochastic / with or without noise)
        also computes `old_action_log_probs` used in training.

        :param s_t                      torch.Tensor

        :return v                       Variable, value_prediction
        :return action                  Variable, action to take
        :return action_log_probs        Variable, `old_action_log_probs`
        :return action_std              Variable, std
        '''
        input = Variable(s_t, volatile=True)
        v, action_mean, action_logstd = self.policy(input)
        action_std = action_logstd.exp()

        if deterministic:
            action = action_mean
        else:
            # only care about noise if stochastic
            noise = Variable(torch.randn(action_std.size()))
            if action_mean.is_cuda:
                noise = noise.cuda()
            action = action_mean + action_std * noise

        # calculate `old_log_probs` directly in exploration.
        action_log_probs = -0.5 * ((action - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()

        return v, action, action_log_probs, action_std

    def evaluate_actions(self, s_t, actions):
        v, action_mean, action_logstd = self.policy(s_t)
        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, action_log_probs, dist_entropy

    def cuda(self, **args):
        self.policy.cuda()
        self.policy.body.cuda()
        self.old_policy.cuda()
        self.old_policy.body.cuda()
        self.use_cuda = True

    def cpu(self, **args):
        self.policy.cpu()
        self.policy.body.cpu()
        self.old_policy.cpu()
        self.old_policy.body.cpu()
        self.use_cuda = False

class Agent(object):
    ''' Agent with MLP policy PI( a_t | s_t)

    This agent is only for standard Roboschool Tasks.

    :param   stacked_state_shape            (176,)
    :param   action_shape                   (17,)

    :param   optimizer_pi                   torch.optim
    :param   hidden                         int, number of hidden neurons
    :param   fixed_std                      boolean,fix the std of actions
    :param   std                            float, value of std if fixed
    '''
    def __init__(self, args, env, hidden=64):
        self.args = args                 # Argparse arguments

        self.ac_shape = env.action_space.shape[0]  # `always` 1D-vector
        ob_shape = env.observation_space.shape
        # ======= Current State Stacked ========
        self.CurrentState = StackedState(args.num_processes,
                                         args.num_stack,
                                         ob_shape,
                                         args.cuda)


        if len(ob_shape) == 1:
            ob_shape = ob_shape[0]
        self.ob_shape = ob_shape
        if args.num_stack>1:
            self.ob_shape = ob_shape * args.num_stack

        # ======= Memory ========
        self.memory = RolloutStorage(args.num_steps,
                                     args.num_processes,
                                     (self.ob_shape,),
                                     (self.ac_shape,))


        self.tmp_steps = 0
        self.use_cuda = args.cuda
        self.noise_scaler = 1

        self.episode_rewards = 0
        self.final_rewards = 0

        # ======= Policy ========
        self.hidden = hidden
        self.policy = Policy(self.ob_shape,
                             self.ac_shape,
                             hidden=hidden,
                             fixed_std=False,
                             std=0.5)
        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=args.pi_lr)

    def get_std(self):
        return self.policy.head.logstd.std

    def sample(self, s_t, deterministic=False):
        '''Samples an action based on input.
        Computes an action (deterministic or stochastic / with or without noise)
        also computes `old_action_log_probs` used in training.

        :param s_t                      torch.Tensor

        :return v                       Variable, value_prediction
        :return action                  Variable, action to take
        :return action_log_probs        Variable, `old_action_log_probs`
        :return action_std              Variable, std
        '''
        input = Variable(s_t, volatile=True)
        v, action_mean, action_logstd = self.policy(input)
        action_std = action_logstd.exp()

        if deterministic:
            action = action_mean
        else:
            # only care about noise if stochastic
            noise = Variable(torch.rand(action_std.size()))
            if action_mean.is_cuda:
                noise = noise.cuda()
            action = action_mean + action_std * noise * self.noise_scaler

        # calculate `old_log_probs` directly in exploration.
        action_log_probs = -0.5 * ((action - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, action, action_log_probs, action_std

    def evaluate_actions(self, s_t, actions):
        v, action_mean, action_logstd = self.policy(s_t)
        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, action_log_probs, dist_entropy

    def cuda(self, **args):
        self.policy.cuda()
        self.policy.body.cuda()
        self.use_cuda = True

    def cpu(self, **args):
        self.policy.cpu()
        self.policy.body.cpu()
        self.use_cuda = False


if __name__ == '__main__':
    import gym
    from arguments  import FakeArgs
    from training import Exploration, Training
    from test import test, test_and_render

    args = FakeArgs()

    env = gym.make('Pendulum-v0')
    agent = Agent(args, env=env, hidden=64)

    print('env.action_space.high', env.action_space.high)
    print('env.action_space.low', env.action_space.low)

    s = env.reset()
    print('state', s)
    agent.CurrentState.update(s)
    agent.memory.states[0].copy_(agent.CurrentState())

    # ==== Training ====
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    floss_total = 0
    vloss_total = 0
    ploss_total = 0
    ent_total= 0

    for j in range(num_updates):
        Exploration(agent, env)  # Explore the environment for args.num_steps
        value_loss, action_loss, dist_entropy = Training(agent, VLoss, verbose=False)  # Train models for args.ppo_epoch

        vloss_total += value_loss
        ploss_total += action_loss
        ent_total += dist_entropy

        agent.memory.last_to_first() #updates rollout memory and puts the last state first.
        print(value_loss)
        print(action_loss)







