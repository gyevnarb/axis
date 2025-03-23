"""Gym and pettingzoo wrappers for easy simulation querying.

Wrappers need not implement all querying methods.
"""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
from pettingzoo.utils.wrappers import BaseWrapper

from axs.config import Registerable
from axs.policy import Policy
from axs.query import Query


class QueryableWrapperBase(ABC):
    """Abstract class for simulation querying."""

    @abstractmethod
    def set_state(
        self,
        agent_policies: dict[int, Policy],
        query: Query,
        observations: list[Any],
        actions: list[Any],
        infos: list[dict[str, Any]],
    ) -> None:
        """Set the state of the simulation and the agent policies.

        This function is called ahead of each simulation call to
        set the starting state of the simulation.

        Args:
            agent_policies (dict[int, Policy]): The agent policies to set.
            query (Query): The query used to set the state.
            observations (list[Any]): The observations to set.
            actions (list[Any]): The actions to set.
            infos (list[dict[str, Any]]): The infos to set.

        """
        raise NotImplementedError

    @abstractmethod
    def execute_query(
        self,
        agent_policies: dict[int, Policy],
        query: Query,
        observations: list[Any],
        actions: list[Any],
        infos: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute the query on the simulation.

        Args:
            query (Query): The query to execute.

        Returns:
            result (dict[str, Any]): The result of the query.

        """
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
