import gym, roboschool

env = gym.make('RoboschoolHumanoid-v1')
env.reset()
steps = 1000

for step in range(steps):
    env.render()
    a = env.action_space.sample()
    s, r, d, i = env.step(a)
    print(r)
