"""Base class for verbalizing environment/simulation data."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from axs.config import Registerable, SupportedEnv
from axs.macroaction import MacroAction
from axs.query import Query

logger = logging.getLogger(__name__)


class Verbalizer(ABC, Registerable, class_type=None):
    """Abstract base class for verbalizing environment/simulation data."""

    @staticmethod
    @abstractmethod
    def reset() -> None:
        """Reset the verbalizer state.

        This method may be used to reset the verbalizer after each
        call to the AXSAgent.explanation function.
        """

    @staticmethod
    @abstractmethod
    def convert(  # noqa: PLR0913
        observations: list[Any],
        macro_actions: dict[int, list[MacroAction]],
        infos: list[dict[str, Any]] | None = None,
        rewards: dict[int, float] | None = None,
        query: Query | None = None,
        env: SupportedEnv | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[str, str]:
        """Convert all environment data.

        This method is used in AXSAgent to verbalize the context information.
        The string returned will be used as the context in queries to the LLM
        as is, so it should be formatted accordingly.

        Args:
            query (Query): The query to verbalize.
            observations (list[Any]): The observations to verbalize.
            macro_actions (dict[int, list[IGP2MacroAction]]): dictionary of agent IDs to
                    corresponding macro actions.
            infos (list[dict[str, Any]] | None): Information dictionaries to verbalize.
            rewards (dict[str, float] | None): Rewards to verbalize for each agent ID.
            env (SupportedEnv | None): The environment to verbalize.
            kwargs: Additional options for the verbalizer.

        Returns:
            context (dict[str, str]): Dictionary of verbalized data with keys mapping to
                argument names in a axs.Query objetc

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_environment(env: SupportedEnv, **kwargs: dict[str, Any]) -> str:
        """Verbalize the static elements of the environment.

        Keyword arguments are used for additional options.

        Args:
            env (SupportedEnv): The environment to verbalize.
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_observations(observations: list[Any], **kwargs: dict[str, Any]) -> str:
        """Verbalize a sequence of observations of the environment.

        Args:
            observations (list[Any]): The observations to verbalize.
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_macro_actions(
        macro_actions: dict[int, list[Any]],
        **kwargs: dict[str, Any],
    ) -> str:
        """Verbalize the macro actions of all agents taken in the environment.

        Args:
            macro_actions (dict[int, list[Any]]): Dictionary of agent IDs to
                    corresponding macro actions.
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_infos(infos: list[dict[str, Any]], **kwargs: dict[str, Any]) -> str:
        """Verbalize a sequence of information dictionaries of the environment.

        Args:
            infos (list[str, dict[Any]]): List of information dictionaries to verbalize.
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_query(query: Query) -> tuple[str, str]:
        """Convert the query to query and type descriptions.

        Args:
            query (axs.Query): The query to convert.

        Returns:
            tuple: The query and its type descriptions.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_rewards(rewards: dict[str, float], **kwargs: dict[str, Any]) -> str:
        """Verbalize the rewards of a simulation.

        Args:
            rewards (dict[str, float]): The rewards to verbalize.
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError
