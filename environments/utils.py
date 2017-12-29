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
            R += r
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

