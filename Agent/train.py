import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_state(ob):
    img = make_grid(ob, nrow=2)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def exploration(pi, CurrentState, rollouts, args, result,  env):
    ''' Exploration part of PPO training:
    1. Sample actions and gather rewards trajectory for num_steps.
    2. Reset states and rewards if some environments are done.
    3. Keep track of means and std fo
    visualizing progress.
    '''
    stds = []
    for step in range(args.num_steps):
        # add step count
        pi.n += 1

        # Sample actions
        value, action, action_log_prob, a_std = pi.sample(CurrentState())
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        # Observe reward and next state
        state, reward, done, info = env.step(cpu_actions)
        reward = torch.from_numpy(reward).view(args.num_processes, -1).float()
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        result.episode_rewards += reward

        if sum(done) > 0:
            # If done then clean episode reward and update final rewards
            result.tmp_final_rewards *= masks
            result.tmp_final_rewards += (1 - masks) * result.episode_rewards
            result.episode_rewards *= masks
            result.update_list()

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


def train(pi, args, rollouts, optimizer_pi):
    value, _, _, _ = pi.sample(rollouts.get_last_state()) # one extra
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

            # PPO loss
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


def exploration_single(pi, CurrentState, rollouts, args, result,  env):
    ''' Exploration part of PPO training:
    1. Sample actions and gather rewards trajectory for num_steps.
    2. Reset states and rewards if some environments are done.
    3. Keep track of means and std fo
    visualizing progress.
    '''
    stds = []
    for step in range(args.num_steps):
        # add step count
        pi.n += 1

        # Sample actions
        value, action, action_log_prob, a_std = pi.sample(CurrentState())
        cpu_actions = action.data.cpu().numpy()[0]

        # Observe reward and next state
        state, reward, done, info = env.step(cpu_actions)
        result.episode_rewards += reward
        reward = torch.Tensor([reward])
        masks = torch.Tensor([not done])

        # If done then update final rewards and reset episode reward
        if done:
            result.tmp_final_rewards *= masks
            result.tmp_final_rewards += (1 - masks) * result.episode_rewards
            result.episode_rewards *= masks
            result.update_list()
            state = env.reset()

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


def exploration_rgb(pi, CurrentState, rollouts, args, result,  env):
    ''' Exploration part of PPO training:
    1. Sample actions and gather rewards trajectory for num_steps.
    2. Reset states and rewards if some environments are done.
    3. Keep track of means and std fo
    visualizing progress.
    '''
    stds = []
    for step in range(args.num_steps):
        # add step count
        pi.n += 1

        # Sample actions
        value, action, action_log_prob, a_std = pi.sample(CurrentState())
        cpu_actions = action.data.cpu().numpy()  # gym takes np.ndarrays

        # Observe reward and next state
        state, obs, reward, done, info = env.step(list(cpu_actions))

        # ob = torch.Tensor(obs)
        # ob = ob.permute(0,3,1,2)
        # if pi.n % 50 ==0:
        #     print('obs.shape', ob.shape)
        #     show_state(ob)

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


# ------------
def Exploration_single_RGB(pi, CurrentState, rollouts, args, result,  env, rgb_list, MAX_REWARD):
    ''' Exploration part of PPO training:
    1. Sample actions and gather rewards trajectory for num_steps.
    2. Reset states and rewards if some environments are done.
    3. Keep track of means and std fo
    visualizing progress.
    '''
    stds = []
    for step in range(args.num_steps):
        # add step count
        pi.n += 1

        # Sample actions
        value, action, action_log_prob, a_std = pi.sample(CurrentState())
        cpu_actions = action.data.cpu().numpy()[0]

        # Observe reward and next state
        state, rgb, reward, done, info = env.step(cpu_actions)
        rgb_list.append(rgb)

        result.episode_rewards += reward
        reward = torch.Tensor([reward])
        masks = torch.Tensor([not done])

        # If done then update final rewards and reset episode reward
        if done:
            print(result.episode_rewards[0])
            if result.episode_rewards[0] > MAX_REWARD:
                MAX_REWARD = result.episode_rewards[0]
                print('Saving Video!')
                name = args.log_dir + '/best_'+pi.n+'_'+str(MAX_REWARD)+'.pt'
                torch.save(rgb_list, name)

            rgb_list = []
            idx = (1-masks)
            result.update_list(idx)
            state,rgb = env.reset()

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
    return rgb_list, MAX_REWARD


def OBSLoss(agent, states, observations, FLoss, goal_state_size=12, verbose=False):
    ''' Loss for the "understanding" module
    :param agent        AgentPepper
    :param states       Batch of states, torch.autograd.Variable
    :param observations Batch of states, torch.autograd.Variable
    :param FLoss        torch.optim loss
    :param verbose      boolean, use print statements for debugging
    '''
    agent.optimizer_f.zero_grad()
    s_hat = agent.understand(Variable(observations))
    s_target = Variable(states[:,-goal_state_size:], requires_grad=False)  # only last joint state (target is not stacked)
    loss = FLoss(s_hat, s_target)
    loss.backward()
    agent.optimizer_f.step()
    return loss



