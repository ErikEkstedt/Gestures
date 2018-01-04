from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf

from environments.fixed_torso import FixedTorso

def enjoy(Env, args, seed):
    from baselines.ppo1 import mlp_policy
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    # logger.session().__enter__()
    set_global_seeds(seed)
    env = Env()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    # env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    # gym.logger.setLevel(logging.WARN)
    pi = policy_fn('pi', env.observation_space, env.action_space)
    tf.train.Saver().restore(sess, args.modelpath)
    with tf.Session() as ses:
        done = False
        for i in range(10):
            obs = env.reset()
            done = False
            ep_rew = 0
            while True:
                action = pi.act(True, obs)[0]
                obs, reward, done, info = env.step(action)
                ep_rew += reward
                env.render()
                if done:
                    print('Episode reward: ', ep_rew)
                    ep_rew = 0
                    break


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', help='environment ID', default='RoboschoolReacher-v1')
    parser.add_argument('--modelpath', type=str, default='/tmp/')
    args = parser.parse_args()

    if 'Roboschool' in args.env:
        import roboschool

        def Env():
            return gym.make(args.env)
        env = Env
    else:
        from environments.fixed_torso import FixedTorso
        env = FixedTorso

    enjoy(env, args, seed=0)


if __name__ == '__main__':
    main()
