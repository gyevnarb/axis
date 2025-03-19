"""Base class for verbalizing environment/simulation data."""

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from axs import SupportedEnv

logger = logging.getLogger(__name__)


class Verbalizer(ABC):
    """Abstract base class for verbalizing environment/simulation data."""

    _verbalizer_library: ClassVar[dict[str, "type[Verbalizer]"]] = {}

    @classmethod
    def register(cls, name: str, verbalizer_type: type["Verbalizer"]) -> None:
        """Register a verbalizer type with the factory.

        Args:
            name (str): The name of the verbalizer.
            verbalizer_type (type[Verbalizer]): The type of the verbalizer.

        """
        if not issubclass(verbalizer_type, cls):
            error_msg = f"Verbalizer {verbalizer_type} is not a subclass of Verbalizer."
            raise TypeError(error_msg)

        if name in cls._verbalizer_library:
            error_msg = (
                f"Verbalizer {name} already registered in the factory "
                f"with {cls._verbalizer_library.keys()} keys."
            )
            raise ValueError(error_msg)
        cls._verbalizer_library[name] = verbalizer_type

    @classmethod
    def get(cls, name: str) -> "type[Verbalizer]":
        """Get the verbalizer type from the factory.

        Args:
            name (str): The name of the verbalizer.

        """
        if name not in cls._verbalizer_library:
            error_msg = (
                f"Verbalizer {name} not found in the factory "
                f"with {cls._verbalizer_library.keys()}."
            )
            raise ValueError(error_msg)
        return cls._verbalizer_library[name]

    @abstractmethod
    @staticmethod
    def convert(
        env: SupportedEnv,
        observations: list[Any],
        macro_actions: list[dict[int, Any]],
        infos: list[dict[str, Any]],
        **kwargs,
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

    @abstractmethod
    @staticmethod
    def convert_environment(env: SupportedEnv, **kwargs) -> str:
        """Verbalize the static elements of the environment.

        Keyword arguments are used for additional options.

        Args:
            env (SupportedEnv): The environment to verbalize.
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def convert_observations(observations: list[Any], **kwargs) -> str:
        """Verbalize a sequence of observations of the environment.

        Args:
            observations (list[Any]): The observations to verbalize.
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def convert_macro_actions(macro_actions: dict[int, list[Any]], **kwargs) -> str:
        """Verbalize the macro actions of all agents taken in the environment.

        Args:
            macro_actions (dict[int, list[Any]]): Dictionary of agent IDs to
                    corresponding macro actions.
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def convert_infos(infos: list[dict[str, Any]], **kwargs) -> str:
        """Verbalize a sequence of information dictionaries of the environment.

        Args:
            infos (list[str, dict[Any]]): List of information dictionaries to verbalize.
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError
