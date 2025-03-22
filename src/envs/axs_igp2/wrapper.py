"""Simulation query wrapper for IGP2."""

from typing import Any

import gymnasium as gym
import igp2 as ip
import numpy as np

import axs
from envs.axs_igp2 import IGP2MacroAction


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
        t: int,
        observations: list[np.ndarray],
        actions: list[np.ndarray],
        infos: list[dict[str, ip.AgentState]],
    ) -> None:
        """Set the state of the simulation.

        Args:
            t (int): The time step to set the simulation to.
            observations (list[Any]): The observations to set.
            actions (list[Any]): The actions to set.
            infos (list[dict[str, Any]]): The infos to set.

        """
        env = self.env.unwrapped

    def add(self) -> None:
        """Add a new TrafficAgent to the IGP2 simulation.

        Args:

        """
        env = self.env.unwrapped
        env.reset()

    def remove(
        self, agent_id: int, **kwargs: dict[str, Any],
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and remove an agent from the IGP2 simulation.

        Args:
            agent_id (int): The agent to remove from the simulation.
            kwargs: Additional options for the removal.

        """
        observation, info = self.env.unwrapped.reset()

        self.env.unwrapped.simulation.remove_agent(agent_id)
        info.pop(agent_id)
        keepmap = [i != agent_id for i in range(len(observation))]
        return observation[keepmap, :], info

    def whatif(self, macro_action: IGP2MacroAction) -> None:
        """Set the macro action of the selected agent."""