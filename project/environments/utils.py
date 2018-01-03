import cv2
import torch
import numpy as np
import time

def rgb_render(obs, title='obs'):
    ''' cv2 as argument such that import is not done redundantly'''
    cv2.imshow(title, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print('Stop')
        return

def rgb_tensor_render(obs, title='tensor_obs'):
    assert type(obs) is torch.Tensor
    assert len(obs.shape) == 3
    obs = obs.permute(1,2,0)
    im = obs.numpy().astype('uint8')
    rgb_render(im, title)

def single_episodes(Env, args, verbose=True):
    ''' Runs episode in one single process
    important args:
        args.RGB = True/False    - extracts rgb from episodes
        args.render = True/False - human friendly rendering

    :Env                         - Environment to run
    :args                        - argparse object
    :verbose                     - print out information (rewards, shapes)
    '''
    env = Env(args)
    if verbose: print('RGB: {}\tGravity: {}\tMAX: {}\t'.format(env.RGB, env.gravity, env.MAX_TIME))
    if args.RGB:
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
    else:
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

def parallel_episodes(Env, args, verbose=False):
    from itertools import count
    env = make_parallel_environments(Env, args)
    R = 0
    if args.RGB:
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
    else:
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
    if args.RGB:
        from project.environments.SubProcEnv import SubprocVecEnv_RGB as SubprocVecEnv
    else:
        from project.environments.SubProcEnv import SubprocVecEnv
    def multiple_envs(Env, args, rank):
        def _thunk():
            env = Env(args)
            env.seed(args.seed+rank*1000)
            return env
        return _thunk
    return SubprocVecEnv([multiple_envs(Env, args, i) for i in range(args.num_processes)])


# Example Reward functions
def calc_reward(self, a):
    ''' Reward function '''
    # Distance Reward
    potential_old = self.potential
    self.potential = self.calc_potential()
    r1 = self.reward_constant1 * float(self.potential[0] - potential_old[0]) # elbow
    r2 = self.reward_constant2 * float(self.potential[1] - potential_old[1]) # hand

    # Cost
    electricity_cost  = self.electricity_cost * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
    joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

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

def calc_reward(self, a):
    ''' IN PROGRESS Difference potential as reward '''
    potential_old = self.potential
    self.potential = self.calc_potential()
    r1 = float(self.potential[0] - potential_old[0]) # elbow
    r2 = float(self.potential[1] - potential_old[1]) # hand
    return r1 + r2
