import math
import torch
from itertools import count

from arguments import get_args
from memory import StackedState
from model import MLPPolicy


def main():
    import time
    args = get_args()
    if args.dof == 2:
        from environments.custom_reacher import CustomReacher2DoF
        env = CustomReacher2DoF()
        print('CustomReacher2DoF2')
    elif args.dof == 3:
        from environments.custom_reacher import CustomReacher3DoF
        env = CustomReacher3DoF()
        print('CustomReacher3DoF')
    elif args.dof == 6:
        from environments.custom_reacher import CustomReacher6DoF
        env = CustomReacher6DoF()
        print('CustomReacher6DoF')
    elif args.dof == 88:
        from environments.custom_reacher import Reacher_plane
        env = Reacher_plane()
        print('Reacher_plane')

    torch.manual_seed(args.seed)
    env.seed(args.seed)
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]

    print('num_stack:', args.num_stack)
    CurrentState = StackedState(1, args.num_stack, ob_shape)
    print(args.load_file)
    saved_state_dict = torch.load(args.load_file)

    pi = MLPPolicy(CurrentState.state_shape,
                   ac_shape,
                   hidden=args.hidden,
                   total_frames=1e6)
    pi.load_state_dict(saved_state_dict)
    pi.train()



    total_reward = 0
    for i in range(args.num_test):
        CurrentState.reset()
        state = env.reset()
        episode_reward = 0
        while True:
            CurrentState.update(state)
            value, action = pi.act(CurrentState())
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            state, reward, done, info = env.step(cpu_actions)
            env.render()
            print(env.potential)

            # If done then update final rewards and reset episode reward
            episode_reward += reward
            if done:
                print(state)
                print('\nEpisode Reward:', episode_reward)
                input()
                total_reward += episode_reward
                done = False
                break

    print('Total reward: ', total_reward/args.num_test)


if __name__ == '__main__':
    main()
