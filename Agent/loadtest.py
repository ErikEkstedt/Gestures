import math
import torch
from itertools import count

from arguments import get_args
from memory import StackedState
from model import MLPPolicy
from utils import get_env


def run_episodes(env, pi, CurrentState, args, verbose=True):
    total_reward = 0
    for j in range(args.num_test):
        state = env.reset()
        episode_reward = 0
        while True:
            CurrentState.update(state)
            value, action = pi.act(CurrentState())
            cpu_actions = action.data.cpu().numpy()[0]
            # Observe reward and next state
            state, reward, done, info = env.step(cpu_actions)
            # check_types(state)
            if args.render: env.render()
            if verbose:
                print('\nPotential:\t{}\nReward:\t{}' \
                      '\nTarget:\t{}\nRobot:\t{}'.format(env.potential,
                                                         reward,
                                                         env.target_position,
                                                         env.hand_position))
            # If done then update final rewards and reset episode reward
            episode_reward += reward
            if done:
                total_reward += episode_reward
                done = False
                if verbose:
                    print('\nEpisode Reward:', episode_reward)
                    print('Avg Reward: ', total_reward/(j+1))
                break
    return total_reward/args.num_test

def test_without_args():
    from environments.custom_reacher import CustomReacher3DoF

    env = CustomReacher3DoF()
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]

    CurrentState = StackedState(1, 1, ob_shape)
    PATH='/home/erik/com_sci/Master_code/Project/Agent/Result/Dec12/DoF3/run-19/checkpoints/BESTDICT9047080_50.482.pt'
    saved_state_dict = torch.load(PATH)

    pi = MLPPolicy(CurrentState.state_shape, ac_shape, hidden=256)
    pi.load_state_dict(saved_state_dict)

    print('Run...')
    run_episodes(env, pi, CurrentState)

# --- Debugging ----

def check_types(state):
    for s in state:
        if not s.dtype == 'float32':
            print('Not `float32` ')
            print(s)

def compare_dicts():
    def unravel_state_dict(sd):
        for n, v in sd.items():
            print(n)
            print(v.size())
            print(v.mean())
            print()
    print('Mean of Trained dict:')
    unravel_state_dict(saved_state_dict)
    print('Mean of Loaded dict:')
    unravel_state_dict(pi.state_dict())
    input('Start test?')

def compare_input_output_pairs(pi, CurrentState):
    print('ANSWER:\nIn:\n', '0 '*15, \
    '\nValue: 3.7403\nAction: 0.1334, 0.3960, -0.3616\n')

    In = CurrentState()
    v, a = pi.act(In)
    print('Pi this time\nIn:\t{}\nValue:\t{}\nAction:\t{}'.format('0 '*ob_shape,
                                                                v.data[0, 0],
                                                                a.data[0].view(1,-1)))

def print_motors(env):
    state = env.reset()
    for m in env.motors:
        print(m.name)

    print('\nMotor_names:')
    for m in env.motor_names:
        print(m)

# -----------------

def main():
    args = get_args()
    Env = get_env(args)
    env = Env()

    torch.manual_seed(args.seed)
    env.seed(args.seed)
    ob_shape = env.observation_space.shape[0]
    ac_shape = env.action_space.shape[0]


    print('num_stack:', args.num_stack)
    CurrentState = StackedState(1, args.num_stack, ob_shape)
    pi = MLPPolicy(CurrentState.state_shape, ac_shape, hidden=args.hidden)

    # Load state dict
    print(args.load_file)
    saved_state_dict = torch.load(args.load_file)

    pi.load_state_dict(saved_state_dict)

    print('\nRunning trained model...')
    avg_rew = run_episodes(env, pi, CurrentState, args, verbose=False)
    print('Avg. Rewards:', avg_rew)
    render = input('Wish to render?')
    if render is 'y':
        args.render = True
        run_episodes(env, pi, CurrentState, args, verbose=False)


if __name__ == '__main__':
    main()
