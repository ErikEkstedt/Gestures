from OpenGL import GLU # fix for opengl issues on desktop  / nvidia
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf
from argparse import ArgumentParser

from environments.custom import getEnv, makeEnv


def enjoy(Env, args):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    # import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    # logger.session().__enter__()
    set_global_seeds(args.seed)
    env = makeEnv(args)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    # env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    obs = env.reset()
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    pi = policy_fn('pi', env.observation_space, env.action_space)
    tf.train.Saver().restore(sess, args.path)
    done = False
    ep_rew = 0
    while not done:
        action = pi.act(True, obs)[0]
        obs, reward, done, info = env.step(action)
        ep_rew += reward
        env.render()
    print('Total Episode Reward: ', ep_rew)


def train(Env, args):
    from baselines.ppo1 import mlp_policy
    import pposgd
    # import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=args.num_processes)
    sess.__enter__()
    set_global_seeds(args.seed)
    env = Env()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    pposgd.learn(env, policy_fn,
            max_timesteps=args.num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95,
        )
    saver = tf.train.Saver()
    saver.save(sess, args.path)


def run_random(args):
    from itertools import count
    import numpy as np
    import time

    env = makeEnv(args)
    asize = env.action_space.shape[0]
    s = env.reset()
    print(s.shape)
    for j in count(1):
        for i in count(1):
            env.render()
            s, r, d, _ = env.step(np.random.rand(asize)*2-1 )
            print(r)
            if d:
                print('done')
                time.sleep(2)
                s=env.reset()


def main():
    parser = ArgumentParser()
    # env
    parser.add_argument('--env-id', type=str, default='HalfHumanoid')
    parser.add_argument('--gravity', type=float, default=9.81)

    # training
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--path', type=str, default='/tmp/model/model')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))

    parser.add_argument('--enjoy', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    args = parser.parse_args()

    if args.random:
        run_random(args)

    if args.enjoy:
        Env = getEnv(args)
        enjoy(Env, args)
    else:
        Env = getEnv(args)
        train(Env, args)



if __name__ == '__main__':
    main()
