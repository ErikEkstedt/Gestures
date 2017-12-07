import numpy as np
import gym
import roboschool

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from arguments import FakeArgs, get_args
from PPOAgent import MLPPolicy
from memory import RolloutStorage, StackedState
from training import Training, Exploration
from test import test, test_and_render
from utils import args_to_list, print_args, log_print


def make_gym(env_id, seed, num_processes):
    ''' imports SubprocVecEnv from baselines.
    :param seed                 int
    :param num_processes        int, # env
    '''
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    def multiple_envs(env_id, seed, rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        return _thunk
    return SubprocVecEnv([multiple_envs(env_id, seed, i) for i in range(num_processes)])


def Exploration(pi, CurrentState, rollouts, args, result,  env):
    ''' Exploration part of PPO training:
    1. Sample actions and gather rewards trajectory for num_steps.
    2. Reset states and rewards if some environments are done.
    3. Keep track of means and std fo
    visualizing progress.
    '''
    stds = []
    for step in range(args.num_steps):
        # Sample actions
        value, action, action_log_prob, a_std = pi.sample(CurrentState())
        stds.append(a_std.data.mean())  # Averaging the std for all actions (really blunt info)
        cpu_actions = action.data.cpu().numpy()  # gym takes np.ndarrays

        # Observe reward and next state
        state, reward, done, info = env.step(list(cpu_actions))
        reward = torch.from_numpy(reward).view(args.num_processes, -1).float()
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        # If done then update final rewards and reset episode reward
        result.episode_rewards += reward
        if sum(done) > 0:
            idx = (1-masks)
            result.update_list(idx)

        result.episode_rewards *= masks                                # reset episode reward
        if args.cuda:
            masks = masks.cuda()

        # reset current states for envs done
        CurrentState.check_and_reset(masks)

        # Update current state and add data to rollouts
        CurrentState.update(state)
        rollouts.insert(step,
                        CurrentState(),
                        action.data,
                        action_log_prob.data,
                        value.data,
                        reward,
                        masks)


def Training(pi, args, rollouts, optimizer_pi):
    value, _, _, _ = pi.sample(rollouts.get_last_state())
    rollouts.compute_returns(value.data, args.no_gae, args.gamma, args.tau)

    # Calculate Advantage (normalize)
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    vloss, ploss, ent = 0, 0, 0
    for e in range(args.ppo_epoch):
        data_generator = rollouts.Batch(advantages, args.batch_size)
        for sample in data_generator:
            states_batch, actions_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample

            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy = pi.evaluate_actions(
                Variable(states_batch), Variable(actions_batch))

            adv_targ = Variable(adv_targ)
            ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()  # PPO's pessimistic surrogate (L^CLIP)

            value_loss = (Variable(return_batch) - values).pow(2).mean()

            # update
            optimizer_pi.zero_grad()
            (value_loss+action_loss-dist_entropy*args.entropy_coef).backward()
            nn.utils.clip_grad_norm(pi.parameters(), args.max_grad_norm)
            optimizer_pi.step()

            vloss += value_loss
            ploss += action_loss.abs()
            ent += dist_entropy

    vloss /= args.ppo_epoch
    ploss /= args.ppo_epoch
    ent /= args.ppo_epoch
    # return value_loss, action_loss, dist_entropy
    return vloss, ploss, ent


class Results(object):
    def __init__(self, max_n=200, max_u=200):
        self.episode_rewards = 0
        self.final_reward_list = []
        self.n = 0
        self.max_n = max_n

        self.vloss = []
        self.ploss = []
        self.ent = []
        self.updates = 0
        self.max_u = max_u

    def update_list(self, idx):
        # The worlds ugliest score tracker. not all process might be done
        for i in range(len(idx)):
            if idx[i][0]:
                self.final_reward_list.insert(0, self.episode_rewards[i])
                self.n += 1
                if self.n > self.max_n:
                    self.final_reward_list.pop()

    def update_loss(self, v, p, e):
        self.vloss.insert(0, v)
        self.ploss.insert(0, p)
        self.ent.insert(0, e)
        self.updates += 1
        if self.updates > self.max_u:
            self.vloss.pop()
            self.ploss.pop()
            self.ent.pop()

    def get_reward_mean(self):
        return torch.stack(self.final_reward_list).mean()

    def get_loss_mean(self):
        v = torch.stack(self.vloss).mean()
        p = torch.stack(self.ploss).mean()
        e = torch.stack(self.ent).mean()
        return v, p, e

def main():
    args = get_args()  # Real argparser
    ds = print_args(args)

    if args.vis:
        from vislogger import VisLogger
        # Text is not pretty
        vis = VisLogger(description_list=ds, log_dir=args.log_dir)

    args.env_id = 'RoboschoolReacher-v1'
    env = make_gym(args.env_id, args.seed, args.num_processes)

    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]

    CurrentState = StackedState(args.num_processes,
                                args.num_stack,
                                ob_shape)

    rollouts = RolloutStorage(args.num_steps,
                              args.num_processes,
                              CurrentState.size()[1],
                              ac_shape)

    result = Results(max_n=200, max_u=10)

    pi = MLPPolicy(CurrentState.state_shape, ac_shape, hidden=64)
    optimizer_pi = optim.Adam(pi.parameters(), lr=args.pi_lr)

    s = env.reset()
    CurrentState.update(s)
    rollouts.states[0].copy_(CurrentState())

    if args.cuda:
        CurrentState.cuda()
        rollouts.cuda()
        pi.cuda()

    # ==== Training ====
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    print('Updates: ', num_updates)

    MAX_REWARD = -999999
    for j in range(num_updates):
        Exploration(pi, CurrentState, rollouts, args, result, env)
        vloss, ploss, ent = Training(pi, args, rollouts, optimizer_pi)
        rollouts.last_to_first() #updates rollout memory and puts the last state first.

        result.update_loss(vloss.data, ploss.data, ent.data)

        #  ==== Save best model ======
        r = result.final_reward_list[-1][0]
        if r > MAX_REWARD:
            print('Saving best latest episode score')
            print('Reward: ', r)
            name = 'model_tmp_best%.2f'%r+'.pt'
            print(name)
            torch.save(pi.cpu().state_dict(), 'trained_models/tmp_best/'+name)
            pi.cuda()
            MAX_REWARD = r
            print('MAX:', MAX_REWARD)
            print()


        #  ==== LOG ======
        if j % args.log_interval == 0 and j > 0 :
            v, p, e = result.get_loss_mean()
            print('Steps: ', (j +1) * args.num_steps * args.num_processes)
            print('Rewards: ', result.get_reward_mean())
            print('Value loss: ', v)
            print('Policy loss: ', p)
            print('Entropy: ', e)

        if j % args.vis_interval == 0 and j > 0 and not args.no_vis:
            frame = (j + 1) * args.num_steps * args.num_processes

            if not args.no_test and j % args.test_interval == 0:
                ''' TODO
                Fix so that resetting the environment does not
                effect the data. Equivialent to `done` ?
                should be the same.'''
                print('Testing')
                test_reward = test(pi, runs=10)
                vis.line_update(Xdata=frame, Ydata=test_reward, name='Test Score')
                print('Done testing')
                if args.test_render:
                    print('RENDER')
                    test_and_render(agent)

            R = result.get_reward_mean()
            vis.line_update(Xdata=frame,
                            Ydata=R,
                            name='Training Score')

                        # std = torch.Tensor(std).mean()

            # Draw plots
            v, p, e = result.get_loss_mean()
            vis.line_update(Xdata=frame, Ydata=v,   name ='Value Loss')
            vis.line_update(Xdata=frame, Ydata=p,   name ='Policy Loss')
            # vis.line_update(Xdata=frame, Ydata=std, name ='Action std')
            vis.line_update(Xdata=frame, Ydata=-e,  name ='Entropy')


        #  ==== Save best model ======
        if j % args.save_interval == 0 and j > 0:
            R = result.get_reward_mean()
            print('Interval Saving')
            name = 'model%d_%.2f.pt'%(j+1, R)
            print(name)
            torch.save(pi.cpu().state_dict(), 'trained_models/'+name)
            pi.cuda()


if __name__ == '__main__':
    main()
