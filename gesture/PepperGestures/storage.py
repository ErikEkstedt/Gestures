import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class StackedState(object):
    ''' stacked state for Pepper '''
    def __init__(self, state_shape=12, num_stack=4, goal_state=None,use_cuda=False):
        self.state = torch.zeros(num_stack, state_shape)
        self.num_stack = num_stack  #
        self.state_shape = state_shape
        self.use_cuda = use_cuda
        self.goal_state = goal_state

    def update(self, s):
        s = torch.from_numpy(s).float()
        if self.use_cuda:
            s = s.cuda()
        if self.num_stack > 1:
            self.state[:-1,:] = self.state[1:,:]
            self.state[-1,:] = s
        else:
            self.state = s

    def __call__(self):
        '''Return:
        self.num_stack consecutive states concatenated with the goal state.
        (self.num_stack + 1) * state.numel() elements.
        '''
        s = torch.cat((self.state, self.goal_state.unsqueeze(0)), dim=0)
        return s.view(1,-1)

    def set_goal(self, new_goal):
        if self.use_cuda:
            self.goal_state = new_goal.cuda()
        else:
            self.goal_state = new_goal

    def cuda(self):
        self.state = self.state.cuda()
        self.goal_state = self.goal_state.cuda()
        self.use_cuda = True

    def cpu(self):
        self.state = self.state.cpu()
        self.goal_state = self.goal_state.cpu()
        self.use_cuda = False

    def __len__(self):
        return self.num_stack

## This script is taken from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
# Redone for single processor and shaved of states
class RolloutStorage(object):
    def __init__(self, num_steps, state_shape, starget_shape, obs_shape, action_shape, args=None):
        self.states = torch.zeros(num_steps+1, state_shape)
        self.target_states = torch.zeros(num_steps+1, starget_shape)
        self.obs = torch.zeros(num_steps+1, *obs_shape)
        self.value_preds = torch.zeros(num_steps+1, 1)
        self.returns = torch.zeros(num_steps+1, 1)

        self.rewards = torch.zeros(num_steps, 1)
        self.masks = torch.ones(num_steps + 1, 1)
        self.actions = torch.zeros(num_steps, action_shape)

        self.num_steps = args.num_steps

    def cuda(self):
        self.states = self.states.cuda()
        self.obs = self.obs.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, state, s_target, obs, action, value_pred, reward, mask):
        self.states[step + 1].copy_(torch.from_numpy(state))
        self.target_states[step + 1].copy_(torch.from_numpy(s_target))
        self.obs[step + 1].copy_(torch.from_numpy(obs))
        self.actions[step].copy_(torch.from_numpy(action))

        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step] + self.rewards[step]

    def last_to_first(self):
        self.obs[0].copy_(self.obs[-1])
        self.target_states[0].copy_(self.target_states[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def first_insert(self, state=None, s_target=None, o=None, ot=None):
        if state is not None:
            self.states[0].copy_(state)
        if s_target is not None:
            self.target_states[0].copy_(s_target)
        if o is not None:
            self.observations[0].copy_(o)
        if ot is not None:
            self.target_observations[0].copy_(ot)

    def get_last(self):
        return self.states[:-1], self.target_states[-1], self.obs[-1]


class RolloutStoragePepper(object):
    ''' Hardcoded values for pepper '''
    def __init__(self, num_steps):
        self.observations        = torch.zeros(num_steps+1, 3, 64, 64)
        self.states              = torch.zeros(num_steps+1, 24)
        self.target_states       = torch.zeros(num_steps+1, 12)
        self.value_preds         = torch.zeros(num_steps+1, 1)
        self.returns             = torch.zeros(num_steps+1, 1)
        self.masks               = torch.ones(num_steps+1, 1)
        self.actions             = torch.zeros(num_steps, 12)
        self.action_log_probs    = torch.zeros(num_steps, 1)
        self.rewards             = torch.zeros(num_steps, 1)
        self.num_steps           = num_steps

    def cuda(self):
        self.observations        = self.observations.cuda()
        self.states              = self.states.cuda()
        self.target_states       = self.target_states.cuda()
        self.rewards             = self.rewards.cuda()
        self.value_preds         = self.value_preds.cuda()
        self.returns             = self.returns.cuda()
        self.actions             = self.actions.cuda()
        self.masks               = self.masks.cuda()
        self.action_log_probs    = self.action_log_probs.cuda()

    def insert(self, step, state, target_state, obs, action, action_log_prob, value_pred, reward, mask):
        self.target_states[step + 1].copy_(target_state)
        self.observations[step + 1].copy_(obs)
        self.states[step + 1].copy_(state)
        self.masks[step + 1].copy_(mask)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)

    def last_to_first(self):
        self.observations[0].copy_(self.observations[-1])
        self.target_states[0].copy_(self.target_states[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def first_insert(self, state=None, s_target=None, o=None, ot=None):
        if state is not None:
            self.states[0].copy_(state)
        if s_target is not None:
            self.target_states[0].copy_(s_target)
        if o is not None:
            self.observations[0].copy_(o)
        if ot is not None:
            self.target_observations[0].copy_(ot)

    def get_last(self):
        return self.states[-1].view(1,-1), self.target_states[-1].view(1,-1), self.observations[-1]

    def compute_returns(self, next_value, no_gae, gamma, tau):
        if not no_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

    def Batch(self, advantages, mini_batch):
        '''
        Batch the data.
        Grab `indices` datapoints from states, rewards, etc...
        Reshape into correct shape such that everything migth be passed through a network
        in one forward pass.

        :param advantages       torch.Tensor
        :param mini_batch       int, size of batch (64, 128 etc)
        '''
        data_size = self.num_steps  # total data size is steps*processsors

        # Choose `mini_batch` indices from total `data_size`.
        # Choose `64` indices from total `2048`.
        sampler = BatchSampler(SubsetRandomSampler(range(data_size)), mini_batch, drop_last=False)

        for indices in sampler:
            indices = torch.LongTensor(indices)

            if advantages.is_cuda:
                indices = indices.cuda()

            # all but last entry
            obs_batch    = self.observations[:-1].view(-1, 3, 64, 64)[indices]
            states_batch = self.states[:-1].view(-1, self.states.size(-1))[indices]
            target_states_batch = self.target_states[:-1].view(-1, self.target_states.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch  = self.masks[:-1].view(-1, 1)[indices]

            # all entries
            actions_batch              = self.actions.view(-1, self.actions.size(-1))[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            yield states_batch, target_states_batch, obs_batch, \
                actions_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ
