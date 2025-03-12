""" This module provides a wrapper class around gymansium-style environments. """
import gymnasium as gym


class Simulator:
    """ Wrapper class around gymansium-style environments. """

    def __init__(self, env_name: str, seed: int = 0):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.seed = seed

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __del__(self):
        self.close()