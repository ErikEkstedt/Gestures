def rgb_render(obs, cv2):
    ''' cv2 as argument such that import is not done redundantly'''
    cv2.imshow('frame', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print('Stop')
        return

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
        import cv2
        s, obs = env.reset()
        if verbose:
            print(s.shape)
            print(obs.shape)
            print(obs.dtype)
            input('Press Enter to start')
        while True:
            s, obs, r, d, _ = env.step(env.action_space.sample())
            if args.render: rgb_render(obs, cv2)
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
    if args.RGB:
        s, obs = env.reset()
        if verbose:
            print('state shape:', s.shape)
            print('obs shape:', obs.shape)
            input('Enter to start')
    else:
        s = env.reset()
        if verbose:
            print('state shape:', s.shape)
            input('Enter to start')

    R = 0
    for i in count(1):
        if args.RGB:
            s, obs, r, d, _ = env.step([env.action_space.sample()] * args.num_processes)
        else:
            s, r, d, _ = env.step([env.action_space.sample()] * args.num_processes)
        R += r
        if sum(d) > 0:
            print('Step: {}, Reward: {}, mean: {}'.format(i, R, R.mean(axis=0)))
            R = 0

def make_parallel_environments(Env, args):
    ''' imports SubprocVecEnv from baselines.
    :param seed                 int
    :param num_processes        int, # env
    '''
    if args.RGB:
        try:
            from envs import SubprocVecEnv_RGB as SubprocVecEnv
        except:
            from environments.envs import SubprocVecEnv_RGB as SubprocVecEnv
    else:
        try:
            from envs import SubprocVecEnv
        except:
            from environments.envs import SubprocVecEnv
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
