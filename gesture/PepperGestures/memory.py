import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time


class Results(object):
    ''' Results
    Class for storing the results during training.
    Could/should be combine with vislogger/logger.
    '''
    def __init__(self, max_n=200, max_u=200):
        '''
        :param max_n     :int, number of final episode rewards for averaging rewards
        :param max_u     :int, number of updates for averaging training losses
        '''
        self.episode_rewards = 0
        self.tmp_final_rewards = 0
        self.final_reward_list = []
        self.n = 0
        self.max_n = max_n

        self.vloss = []
        self.uloss = []
        self.ploss = []
        self.ent = []
        self.updates = 0
        self.max_u = max_u
        self.start_time = time.time()

        # Test
        self.test_episode_list = []
        self.t = 0
        self.max_t = max_n  # same as training for comparison

    def time(self):
        return time.time() - self.start_time

    def update_list(self):
        self.final_reward_list.insert(0, self.tmp_final_rewards.mean())
        self.n += 1
        if self.n > self.max_n:
            self.final_reward_list.pop()

    def update_test(self, test_reward):
        self.test_episode_list.insert(0, test_reward)
        self.n += 1
        if self.n > self.max_n:
            self.final_reward_list.pop()

    def update_loss(self, u, v, p, e):
        self.uloss.insert(0, u)
        self.vloss.insert(0, v)
        self.ploss.insert(0, p)
        self.ent.insert(0, e)
        self.uloss.insert(0, u)

        self.updates += 1
        if self.updates > self.max_u:
            self.vloss.pop()
            self.uloss.pop()
            self.ploss.pop()
            self.ent.pop()
            self.uloss.pop()

    def get_reward_mean(self):
        if len(self.final_reward_list) > 0:
            return torch.Tensor(self.final_reward_list).mean()
        else:
            return 0

    def get_reward_std(self):
        return torch.Tensor(self.final_reward_list).std()

    def get_last_reward(self):
        return self.final_reward_list[0]

    def get_loss_mean(self):
        v = torch.stack(self.vloss).mean()
        p = torch.stack(self.ploss).mean()
        e = torch.stack(self.ent).mean()
        u = torch.stack(self.uloss).mean()
        return v, p, e, u

    def plot_console(self, frame):
        v, p, e, u = self.get_loss_mean()
        v, p, e, u = round(v, 2), round(p,2), round(e,2), round(u, 2)
        r       = round(self.get_reward_mean(), 2)
        print('Time: {}, Steps: {}, Avg.Rew: {}, VLoss: {}, PLoss: {},  Ent: {}, Uloss: {}'.format(
            int(self.time()), frame, r, v, p, e, u))

    def vis_plot(self, vis, frame, std):
        tr_rew_mean = self.get_reward_mean()
        tr_rew_std = self.get_reward_std()
        v, p, e, u = self.get_loss_mean()

        # Draw plots
        vis.line_update(Xdata=frame, Ydata=tr_rew_mean, name='Training Score Mean')
        vis.line_update(Xdata=frame, Ydata=tr_rew_std, name='Training Score Std')
        vis.line_update(Xdata=frame, Ydata=v, name='Value Loss')
        vis.line_update(Xdata=frame, Ydata=u, name='Understand Loss')
        vis.line_update(Xdata=frame, Ydata=p, name='Policy Loss')
        vis.line_update(Xdata=frame, Ydata=std, name='Action std')
        vis.line_update(Xdata=frame, Ydata=-e, name='Entropy')


class StackedObs(object):
    ''' stacked obs for Roboschool

    state: np.array, shape: (num_proc, W, H, 3) (roboschoolhumanoid)
    state: np.array, shape: (num_proc, 3, W, H) (roboschoolhumanoid)

    Thus with defaults:
    current_state.size: (num_proc, 1, 3, 100, 100)

    update: push out the oldest 1*3*100*100 numbers for all processors.
    call:   return current_state.view(4, 1, 3, 100, 100), concat stacked states for each proc.

    :param state_shape      int/tuple shape
    :param num_stack        int
    :param num_proc         int
    :param use_cuda         bool
    '''
    def __init__(self, num_processes=4, num_stack=1, obs_shape=(100, 100, 3), use_cuda=False):
        if obs_shape[0] > obs_shape[2]:
            ''' Change Dims (H,W,C) ->  (C,H,W) '''
            obs_shape = (obs_shape[2], obs_shape[0],obs_shape[1])
        if num_stack > 1:
            self.obs_shape = (obs_shape[0]*num_stack, obs_shape[1], obs_shape[2])
        else:
            self.obs_shape = obs_shape
        self.num_stack = num_stack
        self.current_state = torch.zeros(*self.obs_shape)
        self.num_processes = num_processes
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def update(self, s):
        if type(s) is np.ndarray:
            s /= 255
            s = torch.from_numpy(s).float()
        if self.use_cuda:
            s = s.cuda()
        if self.num_stack > 1:
            self.current_state[:,:-1,:] = self.current_state[:,1:,:] # push out oldest
            self.current_state[:,-1,:] = s  # add in newest
        else:
            self.current_state = s

    def check_and_reset(self, mask):
        '''
        :param mask     torch.Tensor, size: (num_proc, 1), and contains 1 or 0.
        If an element is zero it means that the env for that processor is `done`
        and thus we need to reset the state.
        '''
        tmp = self.current_state.view(self.num_processes, -1)
        tmp *= mask
        self.current_state = tmp.view(self.num_processes, self.num_stack, *self.obs_shape)

    def check_and_reset_target(self, mask, new_target):
        '''
        :param mask     torch.Tensor, size: (num_proc, 1), and contains 1 or 0.
        If an element is zero it means that the env for that processor is `done`
        and thus we need to reset the state.
        '''
        # reset all done envs to zeros. This is if num_stack > 1.
        tmp = self.current_state.view(self.num_processes, -1)
        tmp *= mask
        self.current_state = tmp.view(self.num_processes, self.num_stack, *self.obs_shape)
        for i in range(self.current_state.size(0)):
            if not int(mask[i]):
                self.current_state[i] = torch.from_numpy(new_target[i])

    def reset(self):
        self.current_state = torch.zeros(self.current_state.size())
        if self.use_cuda:
            self.cuda()

    def reset_to(self):
        self.current_state.copy_(state)

    def __call__(self):
        ''' Returns the flatten state (num_processes, -1)'''
        return self.current_state

    def size(self):
        ''' Returns torch.Size '''
        return self.current_state.view(self.num_processes, *self.obs_shape).size()

    def cuda(self):
        self.current_state = self.current_state.cuda()
        self.use_cuda = True

    def cpu(self):
        self.state = self.state.cpu()
        self.use_cuda = False


class StackedState(object):
    ''' stacked state for Roboschool
    state: np.array, shape: (num_proc, 44) (roboschoolhumanoid)

    Thus with defaults:
    current_state.size: (num_proc, 4, 44)

    update: push out the oldest 44 numbers for all procs.
    call:   return current_state.view(4,-1), concat stacked states for each proc.

    :param state_shape      int/tuple shape
    :param num_stack        int
    :param num_proc         int
    :param use_cuda         bool
    '''
    def __init__(self, num_processes=4, num_stack=4, state_shape=44, use_cuda=False):
        if type(state_shape) is tuple:
            self.current_state = torch.zeros(num_processes, num_stack, *state_shape)
        else:
            self.current_state = torch.zeros(num_processes, num_stack, state_shape)

        self.num_stack = num_stack
        self.state_shape = state_shape * num_stack
        self.num_processes = num_processes
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def update(self, s):
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
        assert type(s) is torch.Tensor
        if self.use_cuda:
            s = s.cuda()
        if self.num_stack > 1:
            self.current_state[:,:-1,:] = self.current_state[:,1:,:] # push out oldest
            self.current_state[:,-1,:] = s  # add in newest
        else:
            self.current_state = s

    def check_and_reset(self, mask):
        '''
        :param mask     torch.Tensor, size: (num_proc, 1), and contains 1 or 0.
        If an element is zero it means that the env for that processor is `done`
        and thus we need to reset the state.
        '''
        tmp = self.current_state.view(self.num_processes, -1)
        tmp *= mask
        self.current_state = tmp.view(self.num_processes, self.num_stack, -1)

    def check_and_reset_target(self, mask, new_target):
        '''
        :param mask     torch.Tensor, size: (num_proc, 1), and contains 1 or 0.
        If an element is zero it means that the env for that processor is `done`
        and thus we need to reset the state.
        '''
        # reset all done envs to zeros. This is if num_stack > 1.
        tmp = self.current_state.view(self.num_processes, -1)
        tmp *= mask
        self.current_state = tmp.view(self.num_processes, self.num_stack, -1)
        for i in range(self.current_state.size(0)):
            if not int(mask[i]):
                self.current_state[i] = torch.from_numpy(new_target[i])

    def reset(self):
        self.current_state = torch.zeros(self.current_state.size())
        if self.use_cuda:
            self.cuda()

    def reset_to(self):
        self.current_state.copy_(state)

    def __call__(self):
        ''' Returns the flatten state (num_processes, -1)'''
        return self.current_state.view(self.num_processes, -1)

    def size(self):
        ''' Returns torch.Size '''
        return self.current_state.view(self.num_processes, -1).size()

    def cuda(self):
        self.current_state = self.current_state.cuda()
        self.use_cuda = True

    def cpu(self):
        self.state = self.state.cpu()
        self.use_cuda = False


class Current(object):
        """Current holds all relevant current information"""
        def __init__(self,
                    num_processes,
                    num_stack,
                    state_dims,
                    starget_dims,
                    obs_dims,
                    otarget_dims,
                    ac_shape):
            self.state = StackedState(num_processes, num_stack, state_dims)
            self.obs = StackedObs(num_processes, num_stack, obs_dims)
            self.target_state = StackedState(num_processes, num_stack, starget_dims)
            self.target_obs = StackedObs(num_processes, num_stack, otarget_dims)

            self.o_shape = self.obs.obs_shape
            self.ot_shape = self.target_obs.obs_shape
            self.s_shape = self.state.state_shape
            self.st_shape = self.target_state.state_shape
            self.ac_shape = ac_shape

            self.targets = []
            self.use_cuda = False

        def update(self, state=None, s_target=None, obs=None, o_target=None):
            if state is not None:
                self.state.update(state)
            if s_target is not None:
                self.target_state.update(s_target)
            if obs is not None:
                self.obs.update(obs)
            if o_target is not None:
                self.target_obs.update(o_target)

        def check_and_reset(self, mask):
            self.state.check_and_reset(mask)
            self.obs.check_and_reset(mask)
            self.target_state.check_and_reset(mask)
            self.target_obs.check_and_reset(mask)

        def __call__(self):
            return self.state(), self.target_state(), self.obs(), self.target_obs()

        def add_target_dataset(self, dset):
            self.targets.append(dset)

        def size(self):
            ''' Returns torch.Size '''
            return self.state.size(), self.target_state.size(), \
                self.obs.size(), self.target_obs.size()

        def cuda(self):
            self.state.cuda()
            self.target_state.cuda()
            self.obs.cuda()
            self.target_obs.cuda()
            self.use_cuda = True

        def cpu(self):
            self.state.cpu()
            self.target_state.cpu()
            self.obs.cpu()
            self.target_obs.cpu()
            self.use_cuda = False


class RolloutStorage(object):
    ''' Usage Description
    First manually make the first state be the reset state
    from env (state[0] = env.reset).

    Then gather samples and use self.insert(step, s, a, v, r, mask).

    Then after `num_steps` samples have been gathered manually add the value
    calculated from the last state.

    example:
        RolloutStorage.states[0].copy_(s)
        for step in num_steps:
            self.insert(step, s, a, v, r, mask).
        RolloutStorage.compute_returns(next_value, *args)

    then samples batches from self.state, self.rewards,
    self.value_pred, self.returns, self.masks

    states and values has one extra data point for `next value` when computing
    returns.
    '''
    def __init__(self, num_steps,
                 num_processes,
                 stacked_s_shape,
                 stacked_st_shape,
                 stacked_o_shape,
                 action_shape):
        self.observations        = torch.zeros(num_steps+1, num_processes, *stacked_o_shape)
        self.target_observations = torch.zeros(num_steps+1, num_processes, *stacked_o_shape)
        self.states              = torch.zeros(num_steps+1, num_processes, stacked_s_shape)
        self.target_states       = torch.zeros(num_steps+1, num_processes, stacked_st_shape)
        self.value_preds         = torch.zeros(num_steps+1, num_processes, 1)
        self.returns             = torch.zeros(num_steps+1, num_processes, 1)
        self.masks               = torch.ones(num_steps+1, num_processes, 1)
        self.actions             = torch.zeros(num_steps, num_processes, action_shape)
        self.action_log_probs    = torch.zeros(num_steps, num_processes, 1)
        self.rewards             = torch.zeros(num_steps, num_processes, 1)
        self.num_processes       = num_processes
        self.num_steps           = num_steps
        self.obs_size            = stacked_o_shape

    def cuda(self):
        self.observations        = self.observations.cuda()
        self.target_observations = self.target_observations.cuda()
        self.states              = self.states.cuda()
        self.target_states       = self.target_states.cuda()
        self.rewards             = self.rewards.cuda()
        self.value_preds         = self.value_preds.cuda()
        self.returns             = self.returns.cuda()
        self.actions             = self.actions.cuda()
        self.masks               = self.masks.cuda()
        self.action_log_probs    = self.action_log_probs.cuda()

    def insert(self, step, state, target_state, obs, target_obs, action, action_log_prob, value_pred, reward, mask):
        self.target_observations[step + 1].copy_(target_obs)
        self.target_states[step + 1].copy_(target_state)
        self.observations[step + 1].copy_(obs)
        self.states[step + 1].copy_(state)
        self.masks[step + 1].copy_(mask)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)

    def last_to_first(self):
        self.target_observations[0].copy_(self.target_observations[-1])
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
        o, o_target = self.get_last_obs()
        s, s_target = self.get_last_state()
        return s, s_target, o, o_target,

    def get_last_state(self):
        '''
        Mostly used for calculating `next value_prediction` before training.
        use `view(num_proc, -1)` to get correct dims for policy.
        '''
        s = self.states[-1].view(self.num_processes, -1)
        target_s = self.target_states[-1].view(self.num_processes, -1)
        return s, target_s

    def get_last_obs(self):
        '''
        Mostly used for calculating `next value_prediction` before training.
        use `view(num_proc, -1)` to get correct dims for policy.
        '''
        o = self.observations[-1].view(-1, *self.obs_size)
        target_o = self.target_observations[-1].view(-1, *self.obs_size)
        return o, target_o

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
        data_size = self.num_processes * self.num_steps  # total data size is steps*processsors

        # Choose `mini_batch` indices from total `data_size`.
        # Choose `64` indices from total `2048`.
        sampler = BatchSampler(SubsetRandomSampler(range(data_size)),
                               mini_batch, drop_last=False)

        for indices in sampler:
            indices = torch.LongTensor(indices)

            if advantages.is_cuda:
                indices = indices.cuda()

            # all but last entry
            obs_batch    = self.observations[:-1].view(-1, *self.obs_size)[indices]
            target_obs_batch    = self.target_observations[:-1].view(-1, *self.obs_size)[indices]
            states_batch = self.states[:-1].view(-1, self.states.size(-1))[indices]
            target_states_batch = self.target_states[:-1].view(-1, self.target_states.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch  = self.masks[:-1].view(-1, 1)[indices]

            # all entries
            actions_batch              = self.actions.view(-1, self.actions.size(-1))[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            yield states_batch, target_states_batch, obs_batch, \
                target_obs_batch, actions_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ

