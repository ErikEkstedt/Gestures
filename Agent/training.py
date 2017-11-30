import torch
from torch.autograd import Variable
import torch.nn as nn

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
        reward = torch.from_numpy(reward).view(agent.args.num_processes, -1).float()
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        # If done then update final rewards and reset episode reward
        agent.episode_rewards += reward
        agent.final_rewards *= masks  # set final_reward[i] to zero if masks[i] = 0 -> env[i] is done
        agent.final_rewards += (1 - masks) * agent.episode_rewards # update final_reward to cummulative episodic reward
        agent.episode_rewards *= masks # reset episode reward

        if agent.args.cuda:
            masks = masks.cuda()

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

def PPOLoss(agent, states, actions, returns, adv_target, VLoss, verbose=False):
    ''' PPOLOSS when using two policies. This is the older way.'''
    values, action_log_probs, dist_entropy = agent.evaluate_actions(
                                                    Variable(states, volatile=False),
                                                    Variable(actions, volatile=False),
                                                    use_old_model=False)

    _, old_action_log_probs, _ = agent.evaluate_actions(
                                        Variable(states, volatile=True),
                                        Variable(actions, volatile=True),
                                        use_old_model=True)

    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
    surr1 = ratio * adv_target
    surr2 = torch.clamp(ratio, 1.0 - agent.args.clip_param, 1.0 + agent.args.clip_param) * adv_target
    action_loss = torch.min(surr1, surr2).mean()  # PPO's pessimistic surrogate (L^CLIP)
    #value_loss = (Variable(return_batch) - values).pow(2).mean()
    # The advantages are normalized and thus small.
    # the returns are huge by comparison. Should these be normalized?
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    value_loss = VLoss(values, Variable(returns))

    agent.optimizer_pi.zero_grad()
    (value_loss - action_loss - dist_entropy * agent.args.entropy_coef).backward()  # combined loss - same network for value/action
    agent.optimizer_pi.step()

    if verbose:
        print('-'*40)
        print()
        print('ratio:', ratio.size())
        print('ratio[0]:', ratio[0].data[0])
        print()
        print('adv_targ: ', adv_targ.size())
        print('adv[0]:', adv_targ[0].data[0])
        print()
        print('surr1:', surr1.size())
        print('surr1[0]:', surr1[0].data[0])
        print()
        print('surr2:', surr2.size())
        print('surr2[0]:', surr2[0].data[0])
        print()
        print('action loss: ', action_loss)
        print()
        print('value loss: ', value_loss)
        print('values size: ', values.size())
        print('values[0]: ', values[0].data[0])
        print()
        print('return size: ', return_batch.size())
        print('return[0]: ', return_batch[0][0])
        input()
    return value_loss, action_loss, dist_entropy


