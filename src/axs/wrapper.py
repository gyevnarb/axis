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
        """Set the starting state of the simulation based on the given query.

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
    ) -> tuple[Any, dict[str, Any], dict[int, list[MacroAction]], bool]:
        """Apply changes to the simulation based on the query.

        Args:
            query (Query): The query to apply.
            observation (Any): The observation after set_state() has been called.
            info (dict[str, Any]): The info dict after set_state() has been called.
            kwargs: Additional optional keyword arguments from config file.

        Returns:
            A 4-tuple containing observations, info dict, macro actions, and whether
                running a simulation is needed.

        """
        raise NotImplementedError

    @abstractmethod
    def process_results(
        self,
        query: Query,
        observations: dict[int, list[Any]],
        actions: dict[int, list[Any]],
        infos: dict[int, list[dict[str, Any]]],
        rewards: dict[int, list[Any]],
    ) -> dict[int, Any]:
        """Process the simulation results according to the query.

        This function is called after the simulation terminates.

        Args:
            query (Query): The query used to run the simulation.
            observations (dict[int, list[Any]]): The observations from the simulation
                for each agent.
            actions (dict[int, list[Any]]): The actions from the simulation
                for each agent.
            infos (dict[int, list[dict[str, Any]]]): The infos from the simulation
                for each agent.
            rewards (dict[int, list[Any]]): The rewards from the simulation
                for each agent.

        Returns:
            result (dict[str, Any]): A dictionary of agent IDs to
                corresponding results.

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
