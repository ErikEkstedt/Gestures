import torch
from environment import Social_Torso
from AgentRobo import TestAgentRoboSchool
from memory import StackedState

from itertools import count

num_stack = 4
env = Social_Torso()
state_shape = env.observation_space.shape
stacked_state_shape = (state_shape[0] * num_stack,)
action_shape = env.action_space.shape

TestState = StackedState(1, 4, state_shape, use_cuda=False)

# ====== Agent ==============
torch.manual_seed(10)
agent = TestAgentRoboSchool(stacked_state_shape=stacked_state_shape,
                            action_shape=action_shape,
                            hidden=64,
                            use_cuda=False)

agent.state_shape = state_shape     # Save non-stack state-shape for testing

sdict = torch.load('model.pt')
agent.policy.load_state_dict(sdict)

total_reward = 0
state = env.reset()
for j in count(1):
    env.render()
    # Update current state and add data to memory
    TestState.update(state)
    # Sample actions
    value, action, _, _ = agent.sample(TestState(), deterministic=True)
    cpu_actions = action.data.squeeze(1).cpu().numpy()[0]  # gym takes np.ndarrays
    print(cpu_actions)
    # Observe reward and next state
    state, reward, done, info = env.step(cpu_actions)
    total_reward += reward
    if j % 100 == 0:
        print('frame: {}, total_reward: {}'.format(j, total_reward))
    if done:
        break
        print('Total reward: ', total_reward)
