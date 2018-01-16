from project.environments.humanoid import Humanoid
import torch
import cv2
import time

def target_show(obs):
    cv2.imshow('target', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    cv2.waitKey(100)

def rgb_render(obs, cv2):
    ''' cv2 as argument such that import is not done redundantly'''
    cv2.imshow('frame', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print('Stop')
        return

def episodes_and_target(env, args):
    s, obs = env.reset()
    target_state = env.target
    target_obs = env.target_obs
    target_show(target_obs)
    while True:
        s, obs, r, d, _ = env.step(env.action_space.sample())
        if args.render:
            rgb_render(obs, cv2)
        print('Reward: ', r)
        time.sleep(0.5)
        if d:
            s, obs = env.reset()
            target_state = env.target
            target_obs = env.target_obs
            target_show(target_obs)


if __name__ == '__main__':
    from project.utils.arguments import get_args
    args = get_args()
    args.video_W = 100
    args.video_H = 100

    # load training data
    path='/home/erik/DATA/humanoid/test.pt'
    data = torch.load(path)
    env = Humanoid(args, data)
    episodes_and_target(env, args)
