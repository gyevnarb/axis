"""Gym and pettingzoo wrappers for easy simulation querying.

Wrappers need not implement all querying methods.
"""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
from pettingzoo.utils.wrappers import BaseWrapper

from axs.config import Registerable
from axs.macroaction import MacroAction
from axs.query import Query


class QueryableWrapperBase(ABC):
    """Abstract class for simulation querying."""

    @abstractmethod
    def set_state(
        self,
        query: Query,
        observations: list[Any],
        infos: list[dict[str, Any]],
        **kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Set the state of the simulation based on the given query.

        Args:
            query (Query): The query to execute.
            observations (list[np.ndarray]): The observations from the environment.
            infos (list[dict[str, Any]]): The infos from the environment.
            kwargs: Additional optional keyword arguments.

        Returns:
            result (tuple[Any, dict[str, Any]]): The observation and info dict of the
                new state which is the result of applying the query.

        """
        raise NotImplementedError

    @abstractmethod
    def apply_query(
        self,
        query: Query,
        observation: Any,
        info: dict[str, Any],
        **kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], dict[int, list[MacroAction]]]:
        """Apply the query to the simulation.

        Args:
            query (Query): The query to apply.
            observation (Any): The observation to apply the query to.
            info (dict[str, Any]): The info dict to apply the query to.
            kwargs: Additional optional keyword arguments from config file.

        Returns:
            A 3-tuple containing observations, info dict, and macro actions.

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
