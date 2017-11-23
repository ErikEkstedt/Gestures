''' Not important.
Might work on this on and off.
Learn how to build gym wrappers as I please, gym wrapper that makes
everything Pytorch'''

import gym

# From ppo implement ikostrikov

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env,
                            os.path.join(log_dir,
                                         "{}.monitor.json".format(rank)))
        # Ugly hack to detect atari.
        if hasattr(env.env, 'env') and hasattr(env.env.env, 'ale'):
            env = wrap_deepmind(env)
            env = WrapPyTorch(env)
        return env

    return _thunk

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)

# ----------------------
class WrapPyTorch(gym.Wrapper):
    def __init__(self):

    def _reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self._observation(observation)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = self.observation(observation)
        reward = self.reward(reward)
        return observation, reward, done, info

    def _step(self, action):
        action = self.action(action)
        return self.env.step(action)

    def observation(self, observation):
        return self._observation(observation)

    def _observation(self, observation):
        obs = torch.from_numpy(observation)
        print(obs.size())
        return obs

    def reward(self, reward):
        return self._reward(reward)

    def _reward(self, reward):
        raise NotImplementedError

    def action(self, action):
        return self._action(action)

    def _action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        return self._reverse_action(action)

    def _reverse_action(self, action):
        raise NotImplementedError


