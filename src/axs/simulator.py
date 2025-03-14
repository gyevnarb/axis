""" This module provides a wrapper class around gymansium-style environments. """
import gymnasium as gym

from axs.config import EnvConfig


class Simulator:
    """ Wrapper class around gymansium-style environments. """

    def __init__(self, config: EnvConfig, env: gym.Env = None):
        """ Initialize the simulator with the environment config.

        Args:
            config (Dict[str, Any]): The configuration for the environment.
            env (gym.Env): Optional environment to be used for simulation.
                           If not given, a new internal environment will be created.
        """
        if env is not None:
            self.env = env
        else:
            self.env = gym.make(
                config.name,
                config=config.params,
                render_mode=None)

    def run(self, query):
        return None, None, None

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