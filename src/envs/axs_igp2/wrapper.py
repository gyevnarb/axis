"""Simulation query wrapper for IGP2."""

import logging
from copy import copy
from typing import Any

import gymnasium as gym
import igp2 as ip
import numpy as np

import axs
from envs.axs_igp2 import IGP2MacroAction, IGP2Policy, IGP2Query, util

logger = logging.getLogger(__name__)


class IGP2QueryableWrapper(axs.QueryableWrapper):
    """Wrapper class to support simulation querying for IGP2 environments."""

    def __init__(self, env: ip.simplesim.SimulationEnv) -> "IGP2QueryableWrapper":
        """Initialize the IGP2 queryable wrapper with the environment."""
        if not isinstance(env.unwrapped, ip.simplesim.SimulationEnv):
            error_msg = "Environment must be an IGP2 simulation environment."
            raise TypeError(error_msg)
        super().__init__(env)

    def execute_query(
        self,
        agent_policies: dict[int, axs.Policy],
        query: axs.Query,
        observations: list[np.ndarray],
        actions: list[np.ndarray],
        infos: list[dict[str, ip.AgentState]],
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the query on the simulation.

        Args:
            agent_policies (dict[int, Policy]): Agent policies used in the simulation.
            query (Query): The query to execute.
            observations (list[np.ndarray]): The observations from the environment.
            actions (list[np.ndarray]): The actions from the environment.
            infos (list[dict[str, Any]]): The infos from the environment.
            kwargs: Additional optional keyword arguments.

        Returns:
            result (dict[str, Any]): The result of the query.

        """
        env: ip.simplesim.SimulationEnv = self.env.unwrapped

        time = query.get_time(current_time=len(observations))
        if time > len(observations):
            error_msg = f"Cannot set simulation to timestep {time} in the future."
            raise axs.SimulationError(error_msg)

        trajectories = util.infos2traj(infos, time, env.fps)
        info = {}

        if time == 0:
            info = infos[0]
            if env.t != 0:
                _, info = env.reset(seed=env.np_random_seed)
        else:
            env.simulation.reset()
            for agent_id, policy in agent_policies.items():
                agent = policy.agent
                new_agent_state = infos[time][agent_id]
                agent._initial_state = new_agent_state
                info[agent_id] = new_agent_state
                agent.reset()

                if hasattr(agent, "observations"):
                    for aid, trajectory in trajectories.items():
                        agent.observations[aid] = (trajectory, copy(infos[0]))

                env.simulation.add_agent(policy.agent)

        # Call appropriate method based on query type
        info = getattr(self, "_" + query.query_name)(query, time, info)

        return env._get_obs(), info

    def _add(
        self, query: IGP2Query, time: int, info: dict[int, ip.AgentState],
    ) -> tuple[np.ndarray, dict]:
        """Add a new TrafficAgent to the IGP2 simulation.

        Args:
            query (IGP2Query): The 'add' query to execute.
            time (int): The current timestep in the simulation.
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
        new_initial_state.time = time
        goal = ip.PointGoal(query.params["goal"], 1.5)
        new_agent = ip.TrafficAgent(next_agent_id, new_initial_state, goal, fps=env.fps)
        env.simulation.add_agent(new_agent)
        env.reset_observation_space()
        info[next_agent_id] = new_initial_state
        return info

    def _remove(
        self,
        query: IGP2Query,
        time: int,
        info: dict[int, ip.AgentState],
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and remove an agent from the IGP2 simulation.

        Args:
            query (IGP2Query): The 'remove' query to execute.
            time (int): The current timestep in the simulation.
            info (dict[int, ip.AgentState]): The current agent states in the simulation.
                Updated in-place.

        """
        agent_id = query.params["vehicle"]
        self.env.unwrapped.simulation.remove_agent(agent_id)
        info.pop(agent_id)
        return info

    def _whatif(self, macro_action: IGP2MacroAction) -> None:
        """Set the macro action of the selected agent."""

    def _what(self) -> None:
        """Get the macro action of the selected agent."""
