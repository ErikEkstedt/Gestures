import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import cv2

from gesture.environments.utils import rgb_render, rgb_tensor_render


# === Social ===
def exploration(pi, current, targets, rollouts, args, result,  env):
    ''' Exploration part of PPO training:
    1. Sample actions and gather rewards trajectory for num_steps.
    2. Reset states and rewards if some environments are done.
    3. Keep track of means and std fo
    visualizing progress.
    '''
    stds = []
    s, st, o, ot = current()
    for step in range(args.num_steps):
        # add step count
        pi.n += 1

        # Sample actions
        value, action, action_log_prob, a_std = pi.sample(s, st, o, ot)
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        # Observe reward and next state
        state, s_target, obs, o_target, reward, done, info = env.step(cpu_actions)
        reward = torch.from_numpy(reward).view(args.num_proc, -1).float()
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        result.episode_rewards += reward

        if args.render:
            env.render()

        if sum(done) > 0:
            # Clear episode reward and update final rewards
            result.tmp_final_rewards *= masks
            result.tmp_final_rewards += (1 - masks) * result.episode_rewards
            result.episode_rewards *= masks
            result.update_list()

            env.set_target(targets())

        if args.cuda:
            masks = masks.cuda()

        # Reset current states for envs done and
        # update current state and add data to rollouts
        current.check_and_reset(masks)
        current.update(state, s_target, obs, o_target)
        s, st, o, ot = current()
        rollouts.insert(step, s, st, o, ot,
                        action.data,
                        action_log_prob.data,
                        value.data,
                        reward,
                        masks)

def train(pi, args, rollouts, optimizer_pi):
    last_value, _, _, _ = pi.sample(*rollouts.get_last())
    rollouts.compute_returns(last_value.data, args.no_gae, args.gamma, args.tau)

    # Calculate Advantage (normalize)
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    vloss, ploss, ent = 0, 0, 0
    for e in range(args.ppo_epoch):
        data_generator = rollouts.Batch(advantages, args.batch_size)
        for sample in data_generator:

            states_batch, state_target_batch, obs_batch, \
            obs_target_batch, actions_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample

            # Reshape to do in a single forward pass for all steps
            v, a_logprobs, entro = pi.evaluate_actions(states_batch,
                                                       state_target_batch,
                                                       obs_batch,
                                                       obs_target_batch,
                                                       actions_batch)
            # PPO loss
            adv_targ = Variable(adv_targ)
            ratio = torch.exp(a_logprobs - Variable(old_action_log_probs_batch))
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()  # PPO's pessimistic surrogate (L^CLIP)
            value_loss = (Variable(return_batch) - v).pow(2).mean()

            # update
            optimizer_pi.zero_grad()
            (value_loss+action_loss- entro *args.entropy_coef).backward()
            nn.utils.clip_grad_norm(pi.parameters(), args.max_grad_norm)
            optimizer_pi.step()

            vloss += value_loss
            ploss += action_loss.abs()
            ent += entro

    vloss /= args.ppo_epoch
    ploss /= args.ppo_epoch
    ent /= args.ppo_epoch
    # return value_loss, action_loss, dist_entropy
    return vloss, ploss, ent

