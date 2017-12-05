import numpy as np
import gym
import roboschool

import torch

from arguments import FakeArgs

from PPOAgent import MLPPolicy
from memory import RolloutStorage, StackedState

from training import Training, Exploration
from test import test, test_and_render

def Training(agent, VLoss, verbose=False):
    args = agent.args

    # Calculate `next_value`
    value, _, _, _ = agent.sample(agent.memory.get_last_state())
    agent.memory.compute_returns(value.data, args.use_gae, args.gamma, args.tau)

    if hasattr(agent.policy, 'obs_filter'):
        agent.policy.obs_filter.update(agent.memory.states[:-1])

    # Calculate Advantage
    advantages = agent.memory.returns[:-1] - agent.memory.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    vloss, ploss, ent = 0, 0, 0
    for e in range(args.ppo_epoch):
        data_generator = agent.memory.Batch(advantages, args.batch_size)
        for sample in data_generator:
            states_batch, actions_batch, return_batch, \
            masks_batch, old_action_log_probs_batch, adv_targ = sample

            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy = agent.evaluate_actions(Variable(states_batch),
                                                                            Variable(actions_batch))

            adv_targ = Variable(adv_targ)
            ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

            value_loss = (Variable(return_batch) - values).pow(2).mean()

            # update
            agent.optimizer_pi.zero_grad()
            (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
            nn.utils.clip_grad_norm(agent.policy.parameters(), args.max_grad_norm)
            agent.optimizer_pi.step()

            vloss += value_loss
            ploss += action_loss
            ent += dist_entropy

    vloss /= args.ppo_epoch
    ploss /= args.ppo_epoch
    ent /= args.ppo_epoch
    #return value_loss, action_loss, dist_entropy
    return vloss,  ploss, ent

def Exploration(agent, env):
    ''' Exploration part of PPO training:

    1. Sample actions and gather rewards trajectory for num_steps.
    2. Reset states and rewards if some environments are done.
    3. Keep track of means and std fo
    visualizing progress.
    '''
    stds = []
    for step in range(agent.args.num_steps):
        agent.tmp_steps  += 1

        # Sample actions
        value, action, action_log_prob, a_std = agent.sample(agent.CurrentState())
        stds.append(a_std.data.mean())  # Averaging the std for all actions (really blunt info)

        cpu_actions = action.data.squeeze(1).cpu().numpy()  # gym takes np.ndarrays

        # Observe reward and next state
        state, reward, done, info = env.step(cpu_actions)
        if agent.args.num_processes > 1:
            reward = torch.from_numpy(reward).view(agent.args.num_processes, -1).float()
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            # If done then update final rewards and reset episode reward
            agent.episode_rewards += reward
            agent.final_rewards *= masks  # set final_reward[i] to zero if masks[i] = 0 -> env[i] is done
            agent.final_rewards += (1 - masks) * agent.episode_rewards # update final_reward to cummulative episodic reward
            agent.episode_rewards *= masks # reset episode reward
        else:
            reward = np.array([reward])
            # If done then update final rewards and reset episode reward
            masks = torch.FloatTensor(1-done)
            agent.episode_rewards += reward
            agent.final_rewards *= masks # set final_reward[i] to zero if masks[i] = 0 -> env[i] is done
            agent.final_rewards += (1-masks) * agent.episode_rewards # update final_reward to cummulative episodic reward
            agent.episode_rewards *= masks # reset episode reward

        if agent.args.cuda:
            masks = masks.cuda()

        # if sum(done) > 0:
        #     print('Finished episode')
        #     print('done', done)
        #     print(agent.final_rewards)
        #     print(agent.episode_rewards)

        # reset current states for envs done
        agent.CurrentState.check_and_reset(masks)

        # Update current state and add data to memory
        agent.CurrentState.update(state)
        agent.memory.insert(step,
                            agent.CurrentState(),
                            action.data,
                            action_log_prob.data,
                            value.data,
                            reward,
                            masks)

    agent.std.append(torch.Tensor(stds).mean())

def main():
    args = FakeArgs()
    args.num_processes = 1
    args.num_stack = 1
    args.num_steps = 100

    env_id = 'RoboschoolReacher-v1'
    env = gym.make(env_id)

    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]

    CurrentState = StackedState(args.num_processes,
                                args.num_stack,
                                ob_shape)


    rollouts = RolloutStorage(args.num_steps,
                               args.num_processes,
                               CurrentState.size(),
                               ac_shape)

    pi = MLPPolicy(CurrentState.state_shape, ac_shape, hidden=64)

    s = env.reset()
    CurrentState.update(s)

    episode_reward = []
    total_reward = []

    for i in range(args.num_steps):
        v, a, a_logprob, a_std = pi.sample(CurrentState())
        cpu_action = a.data[0].numpy()
        s, reward, done, _= env.step(cpu_action)

        reward = np.array([reward])
        print(done)
        input()

        CurrentState.update(s)

        # If done then update final rewards and reset episode reward
        masks = torch.FloatTensor(1-done)
        episode_rewards += reward
        final_rewards *= masks # set final_reward[i] to zero if masks[i] = 0 -> env[i] is done
        final_rewards += (1-masks) * episode_rewards # update final_reward to cummulative episodic reward
        episode_rewards *= masks # reset episode reward

        rollouts.insert(i, CurrentState(), a, a_logprob, v, r,mask)

    print(rollouts)

if __name__ == '__main__':
    main()
