import torch
from itertools import count

try:
    from Agent.memory import StackedState
except:
    from memory import StackedState

def test_and_render(agent, Env):
    '''Test
    :param agent - The agent playing
    :param Env  - Environment function/constructor

    :output      - Average complete episodic reward
    '''
    # Use only 1 processor for render
    TestState = StackedState(1,
                                agent.args.num_stack,
                                agent.state_shape,
                                agent.use_cuda)

    # Test environments
    test_env = Env()
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
        if done:
            break
            print('Total reward: ', total_reward)

def test(agent, Env, runs=10, verbose=False):
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
    test_env = Env()
    total_reward = 0
    for i in range(runs):
        TestState.reset()
        state = test_env.reset()
        for j in count(1):
            # Update current state
            TestState.update(state)

            # Sample actions
            value, action, _, _ = agent.sample(TestState(), deterministic=True)
            cpu_actions = action.data.squeeze(1).cpu().numpy()[0]  # gym takes np.ndarrays

            # Observe reward and next state
            state, reward, done, info = test_env.step(cpu_actions)
            total_reward += reward
            if done:
                break
    return total_reward/runs


# def test(agent, Env, runs=10, verbose=False):
#     '''Test with multiple processes.
#     :param agent - The agent playing
#     :param runs - int, number oftest runs
#
#     :output      - Average complete episodic reward
#     '''
#     # Use same number of testing envs as in training.
#     TestState = StackedState(agent.args.num_processes,
#                              agent.args.num_stack,
#                              agent.state_shape,
#                              agent.use_cuda)
#
#     # Test environments
#     test_env = Env(agent.args.seed, agent.args.num_processes)
#
#     total_reward = 0
#     done = False
#     episode_rewards = 0
#     final_rewards = 0
#
#     state = test_env.reset()
#     TestState.update(state)
#
#     total_done = 0
#     while total_done <= runs:
#         # Sample actions
#         value, action, _, _ = agent.sample(TestState(), deterministic=True)
#         cpu_actions = action.data.squeeze(1).cpu().numpy()  # gym takes np.ndarrays
#
#         # Observe reward and next state
#         state, reward, done, info = test_env.step(cpu_actions)
#         total_done += sum(done)  # keep track of number of completed runs
#
#         reward = torch.from_numpy(reward).view(agent.args.num_processes, -1).float()
#         masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
#
#         # If done then update final rewards and reset episode reward
#         episode_rewards += reward
#         final_rewards *= masks  # set final_reward[i] to zero if masks[i] = 0 -> test_env[i] is done
#         final_rewards += (1 - masks) * episode_rewards # update final_reward to cummulative episodic reward
#         episode_rewards *= masks # reset episode reward
#
#         if sum(done)>0:
#             final_rewards *= (1-masks)  # keep the actual completed score.
#             total_reward += final_rewards.sum() # add it to total
#
#         if agent.args.cuda:
#             masks = masks.cuda()
#
#         # reset current states for test_envs done
#         TestState.check_and_reset(masks)
#
#         # Update current state and add data to memory
#         TestState.update(state)
#
#     # cleaning (unneccessary ? )
#     test_env.close()
#     del test_env
#     del TestState
#     return total_reward/total_done
#

