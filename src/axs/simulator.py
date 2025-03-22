"""Create a wrapper class around gymansium-style environments."""

import logging
from typing import Any

import gymnasium as gym
import pettingzoo
from pettingzoo.utils.conversions import parallel_to_aec

from axs.config import EnvConfig, SupportedEnv
from axs.query import Query
from axs.wrapper import QueryableAECWrapper, QueryableWrapper

logger = logging.getLogger(__name__)


class SimulationError(Exception):
    """Exception raised for errors in the simulation."""


class Simulator:
    """Wrapper class around gymansium-style environments."""

    def __init__(self, config: EnvConfig, env: SupportedEnv = None) -> "Simulator":
        """Initialize the simulator with the environment config.

        Args:
            config (Dict[str, Any]): The configuration for the environment.
            env (SupportedEnv): Optional environment to be used for simulation.
                           If not given, a new internal environment will be created.

        """
        self.config = config

        if env is not None:
            if not isinstance(env, SupportedEnv):
                error_msg = "Environment must be a supported environment."
                raise TypeError(error_msg)
            self.env = env
        elif config.name in gym.registry:
            self.env = gym.make(config.name, render_mode=None, **config.params)
            self.env = QueryableWrapper.get(config.wrapper_type)(self.env)
        elif config.env_type:
            if config.env_type == "aec":
                self.env = pettingzoo.AECEnv(**config.params)
            elif config.env_type == "parallel":
                self.env = pettingzoo.ParallelEnv(**config.params)
                self.env = parallel_to_aec(self.env)
            else:
                error_msg = (
                    f"Environment simulator {config.name} cannot be initialized."
                )
                raise ValueError(error_msg)
            self.env = QueryableAECWrapper.get(config.wrapper_type)(self.env)
        else:
            error_msg = f"Environment simulator {config.name} cannot be initialized."
            raise ValueError(error_msg)

    def run(
        self,
        query: Query,
        observations: list[Any],
        actions: list[Any],
        infos: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the simulation with the given query.

        Args:
            query (Query): The query to run the simulation with.
            observations (list[Any]): Observations used to set initial state.
            actions (list[Any]): Actions used to set initial state.
            infos (list[dict[str, Any]]): Info dicts used to set initial state.

        """
        self.env.set_state(query.params.get("time", None), observations, actions, infos)

        sim_observations = []
        sim_infos = []
        sim_actions = []

        observation, info = self.env.reset(seed=self.config.seed)

        # Add the queried changes to the environment
        getattr(self.env, query.query_name)(**query.params)

        for _ in range(self.config.max_iter):
            action = None

            # Perform environment step
            observation, rewards, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                logger.debug("Simulator terminated.")
                break

        return {
            "observations": sim_observations,
            "macro_actions": sim_actions,
            "infos": sim_infos,
            "rewards": rewards,
        }
