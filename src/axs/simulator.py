"""Create a wrapper class around gymansium-style environments."""

import logging
from collections import defaultdict
from typing import Any

import gymnasium as gym
import pettingzoo
from pettingzoo.utils.conversions import parallel_to_aec

from axs.config import EnvConfig, SupportedEnv
from axs.macroaction import MacroAction
from axs.policy import Policy
from axs.query import Query
from axs.util import load_env
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
        env: SupportedEnv | None = None,
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
        else:
            self.env = load_env(config)
            if isinstance(self.env, gym.Env):
                self.env = QueryableWrapper.get(config.wrapper_type)(self.env)
            else:
                self.env = QueryableAECWrapper.get(config.wrapper_type)(self.env)

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
        logger.info("Running internal simulation with: %s", query)

        logger.debug("Resetting internal simulator state.")
        self.env.reset(seed=self.config.seed)

        time = query.get_time(len(observations))
        observation, info = self.env.set_state(
            query,
            observations,
            infos,
            **self.config.params,
        )

        logger.debug("Applying %s(...) to simulation.", query.query_name)
        observation, info, macro_actions, simulation_needed = self.env.apply_query(
            query,
            observation,
            info,
            **self.config.params,
        )

        logger.debug("Resetting agent policies for simulation...")
        for policy in self.agent_policies.values():
            obs, infs = observations[:time], infos[:time]
            if not obs and not infs:
                obs, infs = [observation], [info]
            policy.reset(obs, infs)

        if simulation_needed:
            if isinstance(self.env, QueryableWrapper):
                results = self.run_single_agent(macro_actions, observation, info)
            else:
                results = self.run_multi_agent(macro_actions)
        else:
            results = {
                "observations": {aid: observations for aid in self.agent_policies},
                "infos": {aid: infos for aid in self.agent_policies},
                "actions": {aid: actions for aid in self.agent_policies},
                "rewards": None,
            }

        return self.env.process_results(query, **results)

    def run_single_agent(
        self,
        macro_actions: list[MacroAction],
        observation: Any,
        info: dict[str, Any],
    ) -> dict[str, dict[int, list[Any]]]:
        """Run the simulation when using a single agent gym environment.

        Returns:
            dict[str, dict[int, list[Any]]]: The simulation results mapping agent IDs to
                corresponding observations, actions, infos, and rewards.

        """
        agent_id = next(iter(self.agent_policies.keys()))
        agent_policy = self.agent_policies[agent_id]
        sim_observations = []
        sim_infos = []
        sim_actions = []
        sim_rewards = []

        current_macro = None
        if macro_actions and agent_id in macro_actions and macro_actions[agent_id]:
            macro_actions = macro_actions[agent_id]
            current_macro = macro_actions.pop(0)

        for t in range(self.config.max_iter):
            # Override the agent's policy with the macro action if applicable
            if current_macro and current_macro.applicable(observation, info):
                action = current_macro.next_action(observation, info)
            else:
                action = agent_policy.next_action(observation, info)

            observation, rewards, terminated, truncated, info = self.env.step(action)

            sim_observations.append(observation)
            sim_infos.append(info)
            sim_actions.append(action)
            sim_rewards.append(rewards)

            # If macro is done then get next macro action or update policy to
            # account for any mismatch between the policy and environment state
            if current_macro and current_macro.done(observation, info):
                current_macro = macro_actions.pop(0) if macro_actions else None
                agent_policy.update(sim_observations, sim_infos)

            if terminated or truncated:
                logger.debug("Simulator terminated in %d steps.", t)
                break

        return {
            "observations": {agent_id: sim_observations},
            "actions": {agent_id: sim_actions},
            "infos": {agent_id: sim_infos},
            "rewards": {agent_id: sim_rewards},
        }

    def run_multi_agent(
        self,
        macro_actions: dict[int, list[MacroAction]],
    ) -> dict[str, dict[int, list[Any]]]:
        """Run the simulation when using a multi-agent pettingzoo environment.

        Returns:
            dict[str, dict[int, list[Any]]]: The simulation results mapping agent IDs to
                corresponding observations, actions, infos, and rewards.

        """
        sim_observations = defaultdict(list)
        sim_infos = defaultdict(list)
        sim_actions = defaultdict(list)
        sim_rewards = defaultdict(list)

        current_macros = {
            aid: macro.pop(0) if macro else None for aid, macro in macro_actions.items()
        }

        for agent in self.env.agent_iter():
            observation, rewards, termination, truncation, info = self.env.last()

            if termination or truncation:
                action = None
                logger.debug("Agent %d terminated or truncated.", agent)
            elif current_macros[agent] and current_macros[agent].applicable(
                observation,
                info,
            ):
                action = current_macros[agent].next_action(observation, info)
            else:
                action = self.agent_policies[agent].next_action(
                    observation,
                    info,
                )

            sim_observations[agent].append(observation)
            sim_infos[agent].append(info)
            sim_actions[agent].append(action)
            sim_rewards[agent].append(rewards)

            if current_macros[agent] and current_macros[agent].done(observation, info):
                current_macros[agent] = (
                    macro_actions[agent].pop(0) if macro_actions[agent] else None
                )
                self.agent_policies[agent].update(
                    sim_observations[agent],
                    sim_infos[agent],
                )

            self.env.step(action)

        logger.debug("Simulator terminated.")

        return {
            "observations": sim_observations,
            "actions": sim_actions,
            "infos": sim_infos,
            "rewards": sim_rewards,
        }
