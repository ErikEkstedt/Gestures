from itertools import count
from memory import StackedState
import numpy as np


def test_existing_env(env, Model, state_dict, args, verbose=False):
    ''' Uses existing environment '''
    # == Environment
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]
    CurrentState = StackedState(1, args.num_stack, ob_shape)

    # == Model
    pi = Model(ob_shape, ac_shape)
    pi = Model(CurrentState.state_shape,
               ac_shape,
               hidden=args.hidden)
    pi.load_state_dict(state_dict)

    # Testing
    total_reward = []
    episode_reward = 0
    for i in range(args.num_test):
        state = env.reset()
        while True:
            CurrentState.update(state)
            value, action = pi.act(CurrentState())
            cpu_actions = action.data.cpu().numpy()[0]
            state, reward, done, info = env.step(cpu_actions)
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                total_reward.append(episode_reward)
                episode_reward = 0
                break

    return np.array(total_reward).mean()

def test(Env, Model, state_dict, args, verbose=False):
    '''Creates new env each time '''

    # == Environment
    env = Env()
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]
    CurrentState = StackedState(1, args.num_stack, ob_shape)

    # == Model
    pi = Model(ob_shape, ac_shape)
    pi = Model(CurrentState.state_shape,
               ac_shape,
               hidden=args.hidden)

    pi.load_state_dict(state_dict)

    # Testing
    total_reward, episode_reward = 0, 0
    for i in range(args.num_test):
        state = env.reset()
        for j in count(1):
            CurrentState.update(state)

            value, action, _, _ = pi.sample(CurrentState(), deterministic=True)
            cpu_actions = action.data.cpu().numpy()[0]
            state, reward, done, info = env.step(cpu_actions)
            total_reward += reward
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                episode_reward = 0
                done = False
                break
    return total_reward/args.num_test


def main():
    import torch
    from arguments import get_args
    from model import MLPPolicy
    from environments.custom_reacher import CustomReacher2DoF

    args = get_args()
    args.hidden=128
    sd = torch.load('/home/erik/com_sci/Master_code/Project/Agent/Result/Dec9-SimpleCustomReacher/checkpoints/Good_state_dict251_40.30.pt')
    test(CustomReacher2DoF, MLPPolicy, sd, args, True)


if __name__ == '__main__':
    main()
