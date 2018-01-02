from OpenGL import GLU # fix for opengl issues on desktop  / nvidia
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf
import roboschool
from argparse import ArgumentParser

def enjoy(args):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    # import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    # logger.session().__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
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
    while not done:
        action = pi.act(True, obs)[0]
        obs, reward, done, info = env.step(action)
        env.render()


def train(args):
    from baselines.ppo1 import mlp_policy
    from pposgd import pposgd_simple
    # import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=args.num_processes)
    sess.__enter__()
    # logger.session().__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    # env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=args.num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95,
        )
    env.close()
    saver = tf.train.Saver()
    saver.save(sess, args.path)

def main():
    parser = ArgumentParser()
    parser.add_argument('--env_id', type=str, default='RoboschoolReacher-v1')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--enjoy', action='store_true', default=False)
    parser.add_argument('--path', type=str, default='/tmp/model/model')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()

    if args.enjoy:
        while True:
            enjoy(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
