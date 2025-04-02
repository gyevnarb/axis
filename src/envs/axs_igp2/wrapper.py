"""Simulation query wrapper for IGP2."""

import logging
from copy import copy
from typing import Any

import igp2 as ip
import numpy as np

import axs
from envs.axs_igp2 import IGP2MacroAction, IGP2Query, util

logger = logging.getLogger(__name__)


class IGP2QueryableWrapper(axs.QueryableWrapper):
    """Wrapper class to support simulation querying for IGP2 environments."""

    def __init__(self, env: ip.simplesim.SimulationEnv) -> "IGP2QueryableWrapper":
        """Initialize the IGP2 queryable wrapper with the environment."""
        if not isinstance(env.unwrapped, ip.simplesim.SimulationEnv):
            error_msg = "Environment must be an IGP2 simulation environment."
            raise TypeError(error_msg)
        super().__init__(env)

    def set_state(
        self,
        query: axs.Query,
        observations: list[np.ndarray],
        infos: list[dict[str, ip.AgentState]],
        **kwargs: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, ip.AgentState]]:
        """Execute the query on the simulation.

        Args:
            agent_policies (dict[int, Policy]): Agent policies used in the simulation.
            query (Query): The query to execute.
            observations (list[np.ndarray]): The observations from the environment.
            actions (list[np.ndarray]): The actions from the environment.
            infos (list[dict[str, ip.AgentState]]): The infos from the environment.
            kwargs: Additional optional keyword arguments.

        Returns:
            result (tuple[Any, dict[str, Any]]): The observation and info dict of the
                new state which is the result of applying the query.

        """
        env: ip.simplesim.SimulationEnv = self.env.unwrapped

        time = query.get_time(current_time=len(observations))
        time = max(0, min(len(observations), time) - 1)
        logger.debug("Setting simulation state to timestep %d", time + 1)

        info = {}
        if time == 0:
            info = infos[0]
            if env.t != 0:
                _, info = env.reset(seed=env.np_random_seed)
        else:
            trajectories = util.infos2traj(infos, time, env.fps)
            for agent_id, agent in env.simulation.agents.items():
                agent.reset()
                new_agent_state = infos[time][agent_id]
                agent._vehicle = type(agent._vehicle)(
                    new_agent_state,
                    agent.metadata,
                    agent.fps,
                )
                env.simulation.state[agent_id] = new_agent_state
                agent._trajectory_cl.extend(trajectories[agent_id])
                if hasattr(agent, "observations"):
                    for aid, trajectory in trajectories.items():
                        agent.observations[aid] = (copy(trajectory), copy(infos[0]))

                info[agent_id] = new_agent_state

        return env._get_obs(), info

    def apply_query(
        self,
        query: IGP2Query,
        observation: np.ndarray,
        info: dict[int, ip.AgentState],
        **kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], dict[int, list[IGP2MacroAction]], bool]:
        """Apply the query to the simulation.

        Args:
            query (Query): The query to apply.
            observation (Any): The observation to apply the query to.
            info (dict[str, Any]): The info dict to apply the query to.
            kwargs: Additional optional keyword arguments from config.

        Returns:
            A 4-tuple containing observations, info dict, and macro actions, and
                whether running a simulation is needed.

        """
        info, macros = getattr(self, "_" + query.query_name)(query, info)
        simulation_needed = not (
            query.query_name == "what"
            and query.params["vehicle"] in info
            and info[query.params["vehicle"]].time <= query.params["time"]
        )
        return self.env.unwrapped._get_obs(), info, macros, simulation_needed

    def process_results(
        self,
        query: IGP2Query,
        observations: dict[int, list[np.ndarray]],
        actions: dict[int, list[np.ndarray]],
        infos: dict[int, list[dict[str, ip.AgentState]]],
        rewards: dict[int, list[float]],
    ) -> dict[int, Any]:
        """Process the simulation results according to the query.

        This function is called after the simulation terminates.

        Args:
            query (IGP2Query): The query used to run the simulation.
            observations (dict[int, list[np.ndarray]]): The observations from
                the simulation for each agent ID.
            actions (dict[int, list[np.ndarray]]): The actions from the simulation
                for each agent ID.
            infos (dict[int, list[dict[str, ip.AgentState]]]): The infos from
                the simulation for each agent ID.
            rewards (dict[int, list[float]]): The rewards from the simulation
                for each agent ID.

        Returns:
            result (dict[str, Any]): A dictionary of agent IDs to
                corresponding results.

        """
        agent_id = next(iter(observations.keys()))
        observations = observations[agent_id]
        actions = actions[agent_id]
        infos = infos[agent_id]

        ego_reward = None
        if "reward" in infos[-1]:
            ego_reward = infos[-1].pop("reward")
        time = query.get_time(len(observations))
        vid = query.params.get("vehicle", agent_id)

        ma_config = axs.MacroActionConfig({"params": {"eps": 0.1}})
        macros = IGP2MacroAction.wrap(
            ma_config,
            actions,
            observations,
            infos,
            self.env.unwrapped,
        )
        if query.query_name == "what":
            current_macros = {
                aid: [macro]
                for aid, macros in macros.items()
                for macro in macros
                if macro.start_t <= time < macro.end_t
            }
            start_t = max(0, time - 1)
            end_t = min(len(observations), time + 1)
            if start_t == 0:
                end_t = start_t + 3
            if end_t == len(observations):
                start_t = end_t - 3
            ret = {
                "observations": observations[start_t:end_t],
                "macro_actions": current_macros,
                "infos": infos[start_t:end_t],
            }
            if ego_reward is not None:
                ret["rewards"] = {vid: ego_reward if vid == agent_id else None}
            return ret

        ret = {
            "observations": observations,
            "macro_actions": macros,
            "infos": infos,
        }
        if ego_reward is not None:
            ret["rewards"] = {agent_id: ego_reward}
        return ret

    def _add(
        self,
        query: IGP2Query,
        info: dict[int, ip.AgentState],
    ) -> tuple[np.ndarray, dict]:
        """Add a new TrafficAgent to the IGP2 simulation.

        Args:
            query (IGP2Query): The 'add' query to execute.
            info (dict[int, ip.AgentState]): The current agent states in the simulation.

        """
        env: ip.simplesim.SimulationEnv = self.env.unwrapped
        next_agent_id = max(env.simulation.agents) + 1

        config = {
            "agents": [
                {
                    "id": next_agent_id,
                    "spawn": {
                        "velocity": [
                            ip.Maneuver.MAX_SPEED - 0.01,
                            ip.Maneuver.MAX_SPEED,
                        ],
                        "box": {
                            "center": query.params["location"],
                            "length": 1,
                            "width": 1,
                            "heading": 0.0,
                        },
                    },
                },
            ],
        }
        new_initial_state = env._generate_random_frame(env.scenario_map, config)[
            next_agent_id
        ]
        goal = ip.PointGoal(query.params["goal"], 1.5)
        new_agent = ip.TrafficAgent(next_agent_id, new_initial_state, goal, fps=env.fps)
        env.simulation.add_agent(new_agent)

        env.reset_observation_space()
        info[next_agent_id] = new_initial_state

        return info, {}

    def _remove(
        self,
        query: IGP2Query,
        info: dict[int, ip.AgentState],
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and remove an agent from the IGP2 simulation.

        Args:
            query (IGP2Query): The 'remove' query to execute.
            info (dict[int, ip.AgentState]): The current agent states in the simulation.
                Updated in-place.

        """
        agent_id = query.params["vehicle"]
        self.env.unwrapped.simulation.remove_agent(agent_id)
        info = {aid: state for aid, state in info.items() if aid != agent_id}
        return info, {}

    def _whatif(self, query: axs.Query, info: dict[int, ip.AgentState]) -> None:
        """Set the macro action of the selected agent."""
        env: ip.simplesim.SimulationEnv = self.env.unwrapped
        observation = env._get_obs()
        agent_id = query.params["vehicle"]

        macro_actions = []
        for macro_action in query.params["actions"]:
            macro_action.agent_id = agent_id
            macro_action.scenario_map = env.scenario_map
            ip_macro = macro_action.from_observation(observation, info, fps=env.fps)
            info = ip_macro.action_segments[-1].final_frame
            macro_actions.append(ip_macro)
        env.simulation.agents[agent_id].set_macro_actions(
            [macro.action_segments[0] for macro in macro_actions],
        )

        return info, {agent_id: macro_actions}

    def _what(self, query: axs.Query, info: dict[int, ip.AgentState]) -> None:
        """Get the macro action of the selected agent."""
        return info, {}
