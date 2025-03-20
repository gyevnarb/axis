"""Base class for verbalizing environment/simulation data."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from axs.config import Registerable, SupportedEnv

logger = logging.getLogger(__name__)
_verbalizer_library: dict[str, "type[Verbalizer]"] = {}


class Verbalizer(ABC, Registerable, class_type=None):
    """Abstract base class for verbalizing environment/simulation data."""

    @staticmethod
    @abstractmethod
    def convert(
        env: SupportedEnv,
        observations: list[Any],
        macro_actions: list[dict[int, Any]],
        infos: list[dict[str, Any]],
        **kwargs: dict[str, Any],
    ) -> str:
        """Convert all environment data.

        This method is used in AXSAgent to verbalize the context information.
        The string returned will be used as the context in queries to the LLM
        as is, so it should be formatted accordingly.

        Args:
            env (SupportedEnv): The environment to verbalize.
            observations (list[Any]): The observations to verbalize.
            macro_actions (list[dict[int, Any]]): dictionary of agent IDs to
                    corresponding macro actions.
            infos (list[dict[str, Any]]): The information dictionaries to verbalize.
            kwargs: Additional options for the verbalizer.

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
