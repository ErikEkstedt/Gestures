import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from itertools import count
import os
import gym
import numpy as np
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import roboschool

from memory import RolloutStorage, StackedState
from arguments import FakeArgs, get_args
from AgentRobo import AgentRoboSchool
from environment import Social_Torso

# ---------------------
def log_print(agent, dist_entropy, value_loss, floss, action_loss, j):
    print("\nUpdate: {}, frames:    {} \
          \nAverage final reward:   {}, \
          \nentropy:                {:.4f}, \
          \ncurrent value loss:     {:.4f}, \
          \ncurrent policy loss:    {:.4f}".format(j,
                (j + 1) * agent.args.num_steps,
                agent.final_rewards.mean(),
                -dist_entropy.data[0],
                value_loss.data[0],
                action_loss.data[0],))

def exploration(agent, env):
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

def training(agent, VLoss, verbose=False):
    args = agent.args

    # Calculate `next_value`
    value, _, _, _ = agent.sample(agent.memory.get_last_state())
    agent.memory.compute_returns(value.data, args.use_gae, args.gamma, args.tau)

    if hasattr(agent.policy, 'obs_filter'):
        agent.policy.obs_filter.update(agent.memory.states[:-1])

    # Calculate Advantage
    advantages = agent.memory.returns[:-1] - agent.memory.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    agent.update_old_policy() # update old policy before training loop

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

def test_and_render(agent):
    '''Test
    :param agent - The agent playing
    :param runs - int, number oftest runs

    :output      - Average complete episodic reward
    '''
    # Use only 1 processor for render
    TestState = StackedState(1,
                             agent.args.num_stack,
                             agent.state_shape,
                             agent.use_cuda)

    # Test environments
    test_env = Social_Torso()
    total_reward = 0
    state = test_env.reset()
    for j in count(1):
        test_env.render()

        # Update current state and add data to memory
        TestState.update(state)

        # Sample actions
        value, action, _, _ = agent.sample(TestState(), deterministic=True)
        cpu_actions = action.data.squeeze(1).cpu().numpy()[0]  # gym takes np.ndarrays

        # Observe reward and next state
        state, reward, done, info = test_env.step(cpu_actions)
        total_reward += reward
        if j % 100 == 0:
            print('frame: {}, total_reward: {}'.format(j, total_reward))
        if done:
            break
            print('Total reward: ', total_reward)

    test_env.close()
    del test_env


def test(agent, runs=10, verbose=False):
    '''Test with multiple processes.
    :param agent - The agent playing
    :param runs - int, number oftest runs

    :output      - Average complete episodic reward
    '''
    # Use same number of testing envs as in training.
    TestState = StackedState(agent.args.num_processes,
                             agent.args.num_stack,
                             agent.state_shape,
                             agent.use_cuda)

    # Test environments
    test_env = SubprocVecEnv([
        make_social_torso(agent.args.seed, i, '/tmp')
        for i in range(agent.args.num_processes)])

    total_reward = 0
    done = False
    episode_rewards = 0
    final_rewards = 0

    state = test_env.reset()
    TestState.update(state)

    total_done = 0
    while total_done <= runs:
        # Sample actions
        value, action, _, _ = agent.sample(TestState(), deterministic=True)
        cpu_actions = action.data.squeeze(1).cpu().numpy()  # gym takes np.ndarrays

        # Observe reward and next state
        state, reward, done, info = test_env.step(cpu_actions)
        total_done += sum(done)  # keep track of number of completed runs

        reward = torch.from_numpy(reward).view(agent.args.num_processes, -1).float()
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        # If done then update final rewards and reset episode reward
        episode_rewards += reward
        final_rewards *= masks  # set final_reward[i] to zero if masks[i] = 0 -> test_env[i] is done
        final_rewards += (1 - masks) * episode_rewards # update final_reward to cummulative episodic reward
        episode_rewards *= masks # reset episode reward

        if sum(done)>0:
            final_rewards *= (1-masks)  # keep the actual completed score.
            total_reward += final_rewards.sum() # add it to total

        if agent.args.cuda:
            masks = masks.cuda()

        # reset current states for test_envs done
        TestState.check_and_reset(masks)

        # Update current state and add data to memory
        TestState.update(state)

    # cleaning (unneccessary ? )
    test_env.close()
    del test_env
    del TestState
    return total_reward/total_done

def description_string(args):
    '''Some useful descriptions for the logger/visualizer'''
    slist = []
    slist.append('AgentPepper')
    slist.append('\nSteps: ' + str(args.num_steps))
    slist.append('\nEpoch: ' + str(args.ppo_epoch))
    slist.append('\nlr: ' + str(args.pi_lr))
    slist.append('\nFixed std: ' + str(args.fixed_std))
    slist.append('\nStd(if fixed): ' + str(args.std))
    slist.append('\nTotal frames: ' + str(args.num_frames))
    slist.append('\nTest Render: ' + str(args.test_render))
    slist.append('\nTest iters: ' + str(args.num_test))
    slist.append('\nmax test length: ' + str(args.max_test_length))
    slist.append('\nNo-Test: ' + str(args.no_test))
    slist.append('\nVis-interval: ' + str(args.vis_interval))
    slist.append('\nTest-interval: ' + str(args.test_interval))
    slist.append('\n\n\n\n')
    return slist

def print_ds(l):
    for i in l:
        print(i)

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, "{}.monitor.json".format(rank)))
        return env
    return _thunk

def make_social_torso(seed, rank, log_dir):
    def _thunk():
        env = Social_Torso()
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, "{}.monitor.json".format(rank)))
        return env
    return _thunk


def main():
    args = get_args()  # Real argparser
    ds = description_string(args)
    print_ds(ds)


    if args.vis:
        from vislogger import VisLogger
        ds = description_string(args)
        # Text is not pretty
        vis = VisLogger(description_list=ds, log_dir=args.log_dir)

    # == Environment ========
    monitor_log_dir = "/tmp/"
    env_id = "RoboschoolHumanoid-v1"
    num_stack = 4
    num_steps = 10
    use_cuda = False


    # env = SubprocVecEnv([
    #     make_env(env_id, args.seed, i, monitor_log_dir)
    #     for i in range(args.num_processes)])

    env = SubprocVecEnv([
        make_social_torso(args.seed, i, monitor_log_dir)
        for i in range(args.num_processes)])


    state_shape = env.observation_space.shape
    stacked_state_shape = (state_shape[0] * num_stack,)
    action_shape = env.action_space.shape

    # memory
    memory = RolloutStorage(args.num_steps,
                            args.num_processes,
                            stacked_state_shape,
                            action_shape)


    CurrentState = StackedState(args.num_processes,
                                args.num_stack,
                                state_shape,
                                use_cuda)

    # ====== Agent ==============
    torch.manual_seed(10)
    agent = AgentRoboSchool(args,
                    stacked_state_shape=stacked_state_shape,
                    action_shape=action_shape,
                    hidden=64,
                    fixed_std=False,
                    std=0.5)

    agent.state_shape = state_shape     # Save non-stack state-shape for testing
    VLoss = nn.MSELoss()                     # Value loss function

    agent.final_rewards = torch.zeros([args.num_processes, 1])   # total episode reward
    agent.episode_rewards = torch.zeros([args.num_processes, 1]) # tmp episode reward
    agent.num_done = torch.zeros([args.num_processes, 1])        # how many finished episode, resets on plot
    agent.std = []                                               # list to hold all [action_mean, action_std]

    #  ==== RESET ====
    s = env.reset()
    CurrentState.update(s)
    memory.states[0] = CurrentState()

    if args.cuda:
        agent.cuda()
        CurrentState.cuda()
        memory.cuda()

    agent.CurrentState = CurrentState
    agent.memory = memory

    # ==== Training ====
    num_updates = int(args.num_frames) // args.num_steps

    print('-'*55)
    print()
    print('Starting training {} frames in {} updates.\n\n\
            Batch size {}\tStep size {}\tStack {}'.format(
            args.num_frames, num_updates, args.batch_size, args.num_steps, args.num_stack))

    floss_total = 0
    vloss_total = 0
    ploss_total = 0
    ent_total= 0

    for j in range(num_updates):
        exploration(agent, env)  # Explore the environment for args.num_steps
        value_loss, action_loss, dist_entropy = training(agent, VLoss, verbose=False)  # Train models for args.ppo_epoch

        vloss_total += value_loss
        ploss_total += action_loss
        ent_total += dist_entropy

        agent.memory.last_to_first() #updates rollout memory and puts the last state first.

        #  ==== LOG ======
        if j % args.log_interval == 0: log_print(agent, dist_entropy, value_loss, 1, action_loss, j)

        if j % args.vis_interval == 0 and j is not 0 and not args.no_vis:
            frame = (j + 1) * args.num_steps * args.num_processes

            if not args.no_test and j % args.test_interval == 0:
                ''' TODO
                Fix so that resetting the environment does not
                effect the data. Equivialent to `done` ?
                should be the same.'''
                print('Testing')
                test_reward = test(agent, runs=10)
                vis.line_update(Xdata=frame, Ydata=test_reward, name='Test Score')
                print('Done testing')
                print('RENDER')
                test_and_render(agent)

                #  ==== RESET ====

            vloss_total /= args.vis_interval
            ploss_total /= args.vis_interval
            ent_total   /= args.vis_interval

            # Take mean b/c several processes
            # Training score now resets to zero alot.
            R = agent.final_rewards
            R = R.mean()
            if abs(R) > 0 :
                # when reward mean is exactly zero it does not count.
                vis.line_update(Xdata=frame, Ydata=R, name='Training Score')

            std = torch.Tensor(agent.std).mean()

            # Draw plots
            vis.line_update(Xdata=frame, Ydata=vloss_total, name='Value Loss')
            vis.line_update(Xdata=frame, Ydata=ploss_total, name='Policy Loss')
            vis.line_update(Xdata=frame, Ydata=std, name='Action std')
            vis.line_update(Xdata=frame, Ydata=-ent_total, name='Entropy')

            # reset
            floss_total = 0
            vloss_total = 0
            ploss_total = 0
            ent_total= 0
            del agent.std[:]
            agent.num_done = 0
            agent.final_rewards = 0


    print('saving')
    torch.save(agent.policy.state_dict(), 'model.pt')


if __name__ == '__main__':
    main()
