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
    def execute_query(
        self,
        agent_policies: dict[int, Policy],
        query: Query,
        observations: list[Any],
        actions: list[Any],
        infos: list[dict[str, Any]],
        **kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Execute the query on the simulation.

        Args:
            agent_policies (dict[int, Policy]): Agent policies used in the simulation.
            query (Query): The query to execute.
            observations (list[np.ndarray]): The observations from the environment.
            actions (list[np.ndarray]): The actions from the environment.
            infos (list[dict[str, Any]]): The infos from the environment.
            kwargs: Additional optional keyword arguments.

        Returns:
            result (tuple[Any, dict[str, Any]]): The new observation and info dict
                after executing the query.

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
