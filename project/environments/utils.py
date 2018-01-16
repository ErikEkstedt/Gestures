import cv2
import torch
import numpy as np
import time
from itertools import count
from torchvision.utils import make_grid

# ======================== #
# Render                   #
# ======================== #

def rgb_render(obs, title='obs'):
    ''' cv2 as argument such that import is not done redundantly'''
    cv2.imshow(title, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print('Stop')
        return

def rgb_tensor_render(obs, scale=(1,1), title='tensor_obs'):
    assert type(obs) is torch.Tensor
    assert len(obs.shape) == 3
    obs = obs.permute(1,2,0)
    im = obs.numpy().astype('uint8')
    # rgb_render(im, title)
    render_and_scale(im, scale=scale, title=title)

def render_and_scale(obs, scale=(1, 1), title='obs'):
    ''' cv2 as argument such that import is not done redundantly'''
    height, width = obs.shape[:2]
    obs = cv2.resize(obs,(scale[0]*width, scale[0]*height), interpolation = cv2.INTER_CUBIC)
    cv2.imshow(title, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print('Stop')
        return

# ======================== #
# Run Episodes: Reacher    #
# ======================== #

def print_state(Env, args):
    ''' Runs episodes and prints (labeled) states every step '''
    env = Env(args)
    s = env.reset()
    while True:
        s, r, d, _ = env.step(env.action_space.sample())
        print('\nState size', s.shape)
        print('\ntarget position:')
        print(env.target_position)
        print('\nrobot_key_points:')
        print(env.robot_key_points)
        print('\nto_target_vec:')
        print(env.to_target_vec)
        print('\njoint_positions:')
        print(env.joint_positions)
        print('\njoint_speeds:')
        print(env.joint_speeds)
        input('Press Enter to continue')
        if d:
            s = env.reset()

def print_state_noTarget(Env, args):
    ''' Runs episodes and prints (labeled) states every step '''
    env = Env(args)
    s, s_target, o, o_target = env.reset()
    while True:
        s, r, d, _ = env.step(env.action_space.sample())
        print('\nState size', s.shape)
        print('\nrobot_key_points:')
        print(env.robot_key_points)
        print('\njoint_speeds:')
        print(env.joint_speeds)
        input('Press Enter to continue')
        if d:
            s = env.reset()

def print_state_Combi(Env, args, Targets):
    ''' Runs episodes and prints (labeled) states every step '''
    env = Env(args, Targets)
    s, s_target, o, o_target = env.reset()
    while True:
        s, r, d, _ = env.step(env.action_space.sample())
        print('\nState size', s.shape)
        print('\nrobot_key_points:')
        print(env.robot_key_points)
        print('\njoint_speeds:')
        print(env.joint_speeds)
        ans = input("Done? Press 'y' >")
        if ans == "y":
            return
        if d:
            s = env.reset()

def single_episodes(Env, args, Targets=None, verbose=True):
    ''' Runs episode in one single process
    important args:
        args.RGB = True/False    - extracts rgb from episodes
        args.render = True/False - human friendly rendering
    :Env                         - Environment to run
    :args                        - argparse object
    :verbose                     - print out information (rewards, shapes)
    '''
    if not args.COMBI and args.RGB:
        env = Env(args)
        s, obs = env.reset()
        if verbose:
            print(s.shape)
            print(obs.shape)
            print(obs.dtype)
            input('Press Enter to start')
        while True:
            s, obs, r, d, _ = env.step(env.action_space.sample())
            if args.render:
                rgb_render(obs)
                time.sleep(0.10)
            if verbose: print('Reward: ', r)
            if d:
                s=env.reset()
    elif args.COMBI:
        args.RGB = True
        env = Env(args, Targets)
        s, s_target, o, o_target = env.reset()
        if verbose:
            print('state shape:', s.shape)
            print('state target shape:', s_target.shape)
            print('obs shape:', o.shape)
            print('obs target shape:', o_target.shape)
            input('Enter to start')
        while True:
            s, obs, r, d, _ = env.step(env.action_space.sample())
            if args.render:
                rgb_render(obs)
                time.sleep(0.10)
            if verbose: print('Reward: ', r)
            if d:
                s=env.reset()
    else:
        env = Env(args)
        s = env.reset()
        if verbose:
            print("jdict", env.jdict)
            print("robot_joints", env.robot_joints)
            print("motor_names" , env.motor_names)
            print("motor_power" , env.motor_power)
            print(s.shape)
            input()
        while True:
            a = env.action_space.sample()
            s, r, d, _ = env.step(a)
            if verbose: print('Reward: ', r)
            if args.render: env.render()
            if d:
                s=env.reset()
                if verbose: print('Target pos: ',env.target_position)

def parallel_episodes(Env, args, Targets=None, verbose=False):
    from itertools import count
    R = 0
    if args.RGB:
        env = make_parallel_environments(Env, args)
        s, obs = env.reset()
        if verbose:
            print('state shape:', s.shape)
            print('obs shape:', obs.shape)
            input('Enter to start')
        for i in count(1):
            action = np.random.rand(*(args.num_processes, *env.action_space.shape))
            s, obs, r, d, _ = env.step(action)
            if args.render:
                for i in range(args.num_processes):
                    rgb_render(obs[i], str(i))
            R += r
            if sum(d) > 0:
                print('Step: {}, Reward: {}, mean: {}'.format(i, R, R.mean(axis=0)))
                R = 0
    elif args.COMBI:
        env = make_parallel_environments_combine(Env, args, Targets)
        s, s_target, o, o_target = env.reset()
        if verbose:
            print('state shape:', s.shape)
            print('state target shape:', s_target.shape)
            print('obs shape:', o.shape)
            print('obs target shape:', o_target.shape)
            input('Enter to start')
        for i in count(1):
            action = np.random.rand(*(args.num_processes, *env.action_space.shape))
            s, obs, r, d, _ = env.step(action)
            if args.render:
                for i in range(args.num_processes):
                    rgb_render(obs[i], str(i))
            R += r
            if sum(d) > 0:
                print('Step: {}, Reward: {}, mean: {}'.format(i, R, R.mean(axis=0)))
                R = 0
    else:
        env = make_parallel_environments(Env, args)
        s = env.reset()
        if verbose:
            print('state shape:', s.shape)
            input('Enter to start')
        for i in count(1):
            s, r, d, _ = env.step([env.action_space.sample()] * args.num_processes)
            R += r
            if sum(d) > 0:
                print('Step: {}, Reward: {}, mean: {}'.format(i, R, R.mean(axis=0)))
                R = 0

def make_parallel_environments(Env, args):
    ''' imports SubprocVecEnv from baselines.
    :param Env         environment
    :param args        arguments (argparse object)
    '''
    def multiple_envs(Env, args, rank):
        def _thunk():
            env = Env(args)
            env.seed(args.seed+rank*1000)
            return env
        return _thunk

    if args.RGB:
        from project.environments.SubProcEnv import SubprocVecEnv_RGB as SubprocVecEnv
    else:
        from project.environments.SubProcEnv import SubprocVecEnv
    return SubprocVecEnv([multiple_envs(Env, args, i) for i in range(args.num_processes)])

def make_parallel_environments_combine(Env, args, Targets):
    from project.environments.SubProcEnv import SubprocVecEnv_Combine as SubprocVecEnv
    def multiple_envs(Env, args, Targets, rank):
        def _thunk():
            env = Env(args, Targets)
            env.seed(args.seed+rank*100)
            return env
        return _thunk
    return SubprocVecEnv([multiple_envs(Env, args, Targets, i) for i in range(args.num_processes)])

# ======================== #
# Run Episodes: Social     #
# ======================== #

def random_run_parallel(env, args):
    ''' Executes random actions and renders '''
    from itertools import count
    s, s_, o, o_ = env.reset()
    for i in count(1):
        action = np.random.rand(*(args.num_processes, *env.action_space.shape))
        s, s_, o, o_, r, d, _ = env.step(action)
        if args.verbose:
            print('frame:', i)
        if args.render:
            # env.render()  # same as env.render('human')
            # env.render('machine')
            # env.render('target')
            # h, m, t = env.render('all_rgb_array')  # returns rgb arrays
            mode = ['all_rgb_array'] * args.num_processes
            H, M, T = env.render(mode)
            print(H.shape)
            print(M.shape)
            print(T.shape)

        if sum(d) > 0:
            print('one env is finished')

def random_run_with_changing_targets_parallel(env, dset, args):
    ''' Executes random actions and also changes the target.
    targets are set in order from a project.data.dataset.
    renders options:

        'render.modes': ['human', 'machine', 'target', 'all', 'all_rgb_array']

        example of parallel render:

            modes = ['all'] * args.num_processes
    '''
    t = 0
    targets = [dset[t]] * args.num_processes

    env.set_target(targets)
    t += 1
    state, s_target, obs, o_target = env.reset()
    episode_reward = 0
    for j in count(1):
        if args.render:
            modes = ['all'] * args.num_processes
            env.render(modes)

        # update the target
        if j % args.update_target == 0:
            targets = [dset[t]] * args.num_processes
            env.set_target(targets)
            t += 1

        # Observe reward and next state
        actions = -1 + 2*np.random.rand(*(args.num_processes, *env.action_space.shape))
        state, s_target, obs, o_target, reward, done, info = env.step(actions)

        # If done then update final rewards and reset episode reward
        episode_reward += reward
        if sum(done) > 0:
            if args.verbose:
                print(episode_reward)


# Single Test functions
def random_run(env, render=False, verbose=False):
    ''' Executes random actions and renders '''
    s, s_, o, o_ = env.reset()
    for i in count(1):
        action = env.action_space.sample()
        s, s_, o, o_, r, d, _ = env.step(action)
        if verbose:
            print('\nframe: {}\ts: {}\to: {}'.format(i, s.shape, o.shape ))
            print('action:', action)
            # print('absolute mean:', np.abs(np.array(action)).mean())

        if render:
            env.render('all')
            # env.render()  # same as env.render('human')
            # env.render('machine')
            # env.render('target')
            # h, m, t = env.render('all_rgb_array')  # returns rgb arrays
            # print(h.shape)
            # print(m.shape)
            # print(t.shape)
        if d:
            s, s_, o, o_ = env.reset()

def random_run_with_changing_targets(env, dset, args):
    ''' Executes random actions and also changes the target.
    targets are set in order from a project.data.dataset.
    renders options:
        'render.modes': ['human', 'machine', 'target', 'all', 'all_rgb_array']
    '''
    t = 0
    while True:
        env.set_target(dset[t]); t += 1
        state, s_target, obs, o_target = env.reset()
        episode_reward = 0
        for j in count(1):
            if args.render:
                env.render('all')

            if args.verbose: print('frame:', j)
            # update the target
            if j % args.update_target == 0:
                env.set_target(dset[t]); t += 1

            # Observe reward and next state
            actions = env.action_space.sample()
            state, s_target, obs, o_target, reward, done, info = env.step(actions)

            # If done then update final rewards and reset episode reward
            episode_reward += reward
            if done:
                if args.verbose: print(episode_reward)
                break


# ======================== #
# Example Reward functions #
# Only States
# ======================== #
def calc_reward(self, a):
    ''' Reward function '''
    # Distance Reward
    potential_old = self.potential
    self.potential = self.calc_potential()
    r1 = self.reward_constant1 * float(self.potential[0] - potential_old[0]) # elbow
    r2 = self.reward_constant2 * float(self.potential[1] - potential_old[1]) # hand

    # Cost/Penalties. negative sign
    electricity_cost  = -self.electricity_cost * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += -self.stall_torque_cost * float(np.square(a).mean())
    joints_at_limit_cost = float(-self.joints_at_limit_cost * self.joints_at_limit)

    # Save rewards ?
    self.rewards = [r1, r2, electricity_cost, joints_at_limit_cost]
    return sum(self.rewards)

def calc_reward(self, a):
    ''' Absolute potential as reward '''
    self.potential = self.calc_potential()
    r1 = self.reward_constant1 * float(self.potential[0])
    r2 = self.reward_constant2 * float(self.potential[1])
    return r1 + r2

def calc_reward(self, a):
    ''' Difference potential as reward '''
    potential_old = self.potential
    self.potential = self.calc_potential()
    r1 = self.reward_constant1 * float(self.potential[0] - potential_old[0]) # elbow
    r2 = self.reward_constant2 * float(self.potential[1] - potential_old[1]) # hand
    return r1 + r2

def calc_reward(self, a):
    ''' Hierarchical Difference potential as reward '''
    potential_old = self.potential
    self.potential = self.calc_potential()
    r1 = 10 * float(self.potential[0] - potential_old[0]) # elbow
    r2 = 1 * float(self.potential[1] - potential_old[1]) # hand
    return r1 + r2

def calc_reward(self, a):
    ''' Hierarchical Difference potential as reward '''
    potential_old = self.potential
    self.potential = self.calc_potential()
    r1 = 1 * float(self.potential[0] - potential_old[0]) # elbow
    r2 = 10 * float(self.potential[1] - potential_old[1]) # hand
    return r1 + r2


# ======================== #
# Example Reward functions #
# Only States
# ======================== #
