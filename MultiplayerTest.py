'''
Check out roboschool/agent_zoo/demo_race2.py
'''

import os, gym, roboschool
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from environment import Social_Torso
# ======= Own Test =========
class RandomAction(object):
    def __init__(self, env):
        self.env = env

    def act(self,s, cx):
        return self.env.action_space.sample()

stadium = roboschool.scene_stadium.MultiplayerStadiumScene(gravity=0.8, timestep=0.0165/4, frame_skip=4)

participants = []
# Agent 0
# env_id = "RoboschoolHumanoid-v1"
# env = gym.make(env_id)
env = Social_Torso()
env.unwrapped.scene = stadium   # if you set scene before first reset(), it will be used.
env.unwrapped.player_n = 0   # mutliplayer scenes will also use player_n
pi = RandomAction(env)
participants.append((env,pi))

# Agent 1
# env_id = "RoboschoolHumanoid-v1"
# env = gym.make(env_id)

env = Social_Torso()
env.unwrapped.scene = stadium   # if you set scene before first reset(), it will be used.
env.unwrapped.player_n = 1   # mutliplayer scenes will also use player_n
pi = RandomAction(env)
participants.append((env,pi))

episode_n = 0
while 1:
    stadium.episode_restart()
    episode_n += 1

    multi_state = [env.reset() for env, _ in participants]
    frame = 0
    restart_delay = 0
    while 1:
        still_open = stadium.test_window()
        multi_action = [pi.act(s, None) for s, (env, pi) in zip(multi_state, participants)]

        for a, (env, pi) in zip(multi_action, participants):
            env.unwrapped.apply_action(a)  # action sent in apply_action() must be the same that sent into step(),
        # some wrappers will not work

        stadium.global_step()

        state_reward_done_info = [env.step(a) for a, (env, pi) in zip(multi_action, participants)]
        multi_state = [x[0] for x in state_reward_done_info]
        multi_done  = [x[2] for x in state_reward_done_info]

        if sum(multi_done)==len(multi_done):
            break

        frame += 1
        stadium.cpp_world.test_window_score("%04i" % frame)
        if not still_open: break
        if frame==1000: break
    if not still_open: break

