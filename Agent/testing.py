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
