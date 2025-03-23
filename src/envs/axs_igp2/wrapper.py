"""Simulation query wrapper for IGP2."""

from copy import copy

import gymnasium as gym
import igp2 as ip
import numpy as np

import axs
from envs.axs_igp2 import IGP2MacroAction, IGP2Policy, IGP2Query, util


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
        agent_policies: dict[int, IGP2Policy],
        query: IGP2Query,
        observations: list[np.ndarray],
        actions: list[np.ndarray],
        infos: list[dict[str, ip.AgentState]],
    ) -> None:
        """Set the state of the simulation.

        Args:
            agent_policies (dict[int, Policy]): The agent policies to set.
            query (Query): The query used to set the state.
            observations (list[Any]): The observations to set.
            actions (list[Any]): The actions to set.
            infos (list[dict[str, Any]]): The infos to set.

        """
        env: ip.simplesim.SimulationEnv = self.env.unwrapped

        time = query.get_time(current_time=len(observations))
        if time > len(observations):
            error_msg = f"Cannot set simulation to timestep {time} in the future."
            raise axs.SimulationError(error_msg)

        trajectories = util.traj2dict(infos, time, env.fps)

        env.simulation.reset()
        for agent_id, policy in agent_policies.items():
            agent = policy.agent
            agent._initial_state = infos[time][agent_id]
            agent.reset()
            if hasattr(agent, "observations"):
                for aid, trajectory in trajectories.items():
                    agent.observations[aid] = (trajectory, copy(infos[0]))
            env.simulation.add_agent(policy.agent)

    def execute_query(
        self, agent_policies, query, observations, actions, infos, **kwargs
    ):
        return super().execute_query(
            agent_policies, query, observations, actions, infos
        )

    # def add(self) -> None:
    #     """Add a new TrafficAgent to the IGP2 simulation.

    #     Args:

    #     """
    #     env = self.env.unwrapped
    #     env.reset()

    # def remove(
    #     self, agent_id: int, **kwargs: dict[str, Any],
    # ) -> tuple[np.ndarray, dict]:
    #     """Reset the environment and remove an agent from the IGP2 simulation.

    #     Args:
    #         agent_id (int): The agent to remove from the simulation.
    #         kwargs: Additional options for the removal.

    #     """
    #     observation, info = self.env.unwrapped.reset()

    #     self.env.unwrapped.simulation.remove_agent(agent_id)
    #     info.pop(agent_id)
    #     keepmap = [i != agent_id for i in range(len(observation))]
    #     return observation[keepmap, :], info

    # def whatif(self, macro_action: IGP2MacroAction) -> None:
    #     """Set the macro action of the selected agent."""

    # def what(self) -> None:
    #     """Get the macro action of the selected agent."""
