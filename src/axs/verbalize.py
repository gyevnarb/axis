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
            error_msg = (f"Verbalizer {name} already registered in the factory "
                         f"with {cls._verbalizer_library.keys()} keys.")
            raise ValueError(error_msg)
        cls._verbalizer_library[name] = verbalizer_type

    @classmethod
    def get(cls, name: str) -> "type[Verbalizer]":
        """Get the verbalizer type from the factory.

        Args:
            name (str): The name of the verbalizer.

        """
        if name not in cls._verbalizer_library:
            error_msg = (f"Verbalizer {name} not found in the factory "
                         f"with {cls._verbalizer_library.keys()}.")
            raise ValueError(error_msg)
        return cls._verbalizer_library[name]

    @abstractmethod
    def convert(
        self,
        env: SupportedEnv,
        observations: list[Any],
        actions: list[dict[int, Any]],
        infos: list[dict[str, Any]],
    ) -> str:
        """Convert all environment data.

        Args:
            env (SupportedEnv): The environment to verbalize.
            observations (List[Any]): The observations to verbalize.
            actions (List[Dict[int, Any]]): Dictionary of agent IDs to their actions.
            infos (List[Dict[str, Any]]): The information dictionaries to verbalize.

        """
        raise NotImplementedError

    @abstractmethod
    def convert_environment(self, env: SupportedEnv, **kwargs) -> str:  # noqa: ANN003
        """Verbalize the static elements of the environment.

        Keyword arguments are used for additional options.

        Args:
            env (SupportedEnv): The environment to verbalize.

        Keyword Args:
            kwargs: Additional options for the verbalizer.

        """
        raise NotImplementedError

    @abstractmethod
    def convert_observations(self, observations: list[Any]) -> str:
        """Verbalize a sequence of observations of the environment.

        Args:
            observations (List[Any]): The observations to verbalize.

        """
        raise NotImplementedError

    @abstractmethod
    def convert_actions(self, actions: list[dict[int, Any]]) -> str:
        """Verbalize the actions of all agents taken in the environment.

        Args:
            actions (List[Dict[int, Any]]): Dictionary of agent IDs to their actions.

        """
        raise NotImplementedError

    @abstractmethod
    def convert_infos(self, infos: list[dict[str, Any]]) -> str:
        """Verbalize a sequence of information dictionaries of the environment.

        Args:
            infos (List[Dict[str, Any]]): The information dictionaries to verbalize.

        """
        raise NotImplementedError
