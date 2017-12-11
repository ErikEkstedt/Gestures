import torch
from itertools import count
from memory import StackedState


def test(Env, Model, state_dict, args, verbose=False):
    torch.manual_seed(args.seed)

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
            # cpu_actions = action.data.squeeze(1).cpu().numpy()
            state, reward, done, info = env.step(cpu_actions)
            total_reward += reward
            episode_reward += reward
            if done:
                if verbose: print(episode_reward)
                episode_reward = 0
                done = False
                break
    return total_reward/args.num_test
