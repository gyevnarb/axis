"""Create a wrapper class around gymansium-style environments."""

import logging
from collections import defaultdict
from typing import Any

import gymnasium as gym
import pettingzoo
from pettingzoo.utils.conversions import parallel_to_aec

from axs.config import EnvConfig, SupportedEnv
from axs.policy import Policy
from axs.query import Query
from axs.wrapper import QueryableAECWrapper, QueryableWrapper

logger = logging.getLogger(__name__)


class SimulationError(Exception):
    """Exception raised for errors in the simulation."""


class Simulator:
    """Wrapper class around gymansium-style environments."""

    def __init__(
        self,
        config: EnvConfig,
        agent_policies: dict[int, Policy],
        env: SupportedEnv = None,
    ) -> "Simulator":
        """Initialize the simulator with the environment config.

        Args:
            config (Dict[str, Any]): The configuration for the environment.
            agent_policies (Dict[int, Policy]): Agent policies used in the simulator.
            env (SupportedEnv): Optional environment to be used for simulation.
                           If not given, a new internal environment will be created.

        """
        self.config = config
        self.agent_policies = agent_policies

        if len(agent_policies) > 1 and env is not None and not isinstance(env, gym.Env):
            logger.warning("Running multi-agent simulation using one agent policy!")

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
        logger.info("Running simulation with query: %s", str(query))

        logger.debug("Resetting internal simulator state.")
        self.env.reset(seed=self.config.seed)

        init_state = self.env.execute_query(
            self.agent_policies,
            query,
            observations,
            actions,
            infos,
            **self.config.params,
        )

        if isinstance(self.env, QueryableWrapper):
            results = self.run_single_agent(*init_state)
        else:
            results = self.run_multi_agent()  # AEC env uses env.last() for init state

        return results

    def run_single_agent(
        self, observation: Any, info: dict[str, Any],
    ) -> dict[str, dict[int, list[Any]]]:
        """Run the simulation when using a single agent gym environment.

        Returns:
            dict[str, dict[int, list[Any]]]: The simulation results mapping agent IDs to
                corresponding observations, actions, infos, and rewards.

        """
        agent_id = next(self.agent_policies.keys())
        sim_observations = []
        sim_infos = []
        sim_actions = []

        for _ in range(self.config.max_iter):
            action = self.agent_policies[agent_id].next_action(observation, info)

            # Perform environment step
            observation, rewards, terminated, truncated, info = self.env.step(action)

            sim_observations.append(observation)
            sim_infos.append(info)
            sim_actions.append(action)

            if terminated or truncated:
                logger.debug("Simulator terminated.")
                break

        return {
            "observations": {agent_id: sim_observations},
            "macro_actions": {agent_id: sim_actions},
            "infos": {agent_id: sim_infos},
            "rewards": {agent_id: rewards},
        }

    def run_multi_agent(self) -> dict[str, dict[int, list[Any]]]:
        """Run the simulation when using a multi-agent pettingzoo environment.

        Returns:
            dict[str, dict[int, list[Any]]]: The simulation results mapping agent IDs to
                corresponding observations, actions, infos, and rewards.

        """
        sim_observations = defaultdict(list)
        sim_infos = defaultdict(list)
        sim_actions = defaultdict(list)
        sim_rewards = defaultdict(list)

        for agent in self.env.agent_iter():
            observation, rewards, termination, truncation, info = self.env.last()

            if termination or truncation:
                action = None
                logger.debug("Agent %d terminated or truncated.", agent)
            else:
                action = self.agent_policies[agent].next_action(
                    observation,
                    info,
                )

            sim_observations[agent].append(observation)
            sim_infos[agent].append(info)
            sim_actions[agent].append(action)
            sim_rewards[agent].append(rewards)

            self.env.step(action)

        return {
            "observations": sim_observations,
            "macro_actions": sim_actions,
            "infos": sim_infos,
            "rewards": rewards,
        }
