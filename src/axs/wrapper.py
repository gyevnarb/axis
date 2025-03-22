"""Gym and pettingzoo wrappers for easy simulation querying.

Wrappers need not implement all querying methods.
"""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
from pettingzoo.utils.wrappers import BaseWrapper

from axs.config import Registerable


class QueryableWrapperBase(ABC):
    """Abstract class for simulation querying."""

    @abstractmethod
    def set_state(
        self,
        t: int,
        observations: list[Any],
        actions: list[Any],
        infos: list[dict[str, Any]],
    ) -> None:
        """Set the state of the simulation.

        This function is called ahead of each simulation call to
        set the starting state of the simulation.

        Args:
            t (int): The time step to set the simulation to.
            observations (list[Any]): The observations to set.
            actions (list[Any]): The actions to set.
            infos (list[dict[str, Any]]): The infos to set.

        """
        raise NotImplementedError

    @abstractmethod
    def add(self, agent_id: int):
        """Add a new agent to the start of the simulation with the given state.

        Args:
            agent_id (int): The ID of the agent to add.
        """
        raise NotImplementedError

    @abstractmethod
    def remove(
        self,
        agent_id: int,
        **kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Remove an agent from the simulation.

        Args:
            agent_id (int): The ID of the agent to remove.
            kwargs: Additional options for the removal.

        Returns:
            tuple: The observation and info dict after removing the agent.

        """
        raise NotImplementedError

    @abstractmethod
    def whatif(self, state):
        """Run a simulation with the given state."""
        raise NotImplementedError

    @abstractmethod
    def what(self):
        """Return the current state of the simulation."""
        raise NotImplementedError


class QueryableWrapper(
    gym.Wrapper,
    QueryableWrapperBase,
    Registerable,
    class_type=None,
):
    """Wrapper class to support simulation querying."""


class QueryableAECWrapper(
    BaseWrapper,
    QueryableWrapperBase,
    Registerable,
    class_type=None,
):
    """Wrapper class to support simulation querying for AEC environments."""
