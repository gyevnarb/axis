"""Contains wrapper class for easy call to next action in the simulation."""

from abc import ABC, abstractmethod
from typing import Any

from axs.config import Registerable, SupportedEnv


class Policy(ABC, Registerable, class_type=None):
    """Interface class for agent policies.

    You can either subclass this interface or wrap an existing policy in it.
    The purpose of this class is to provide a single function, next_action(),
    which the AXS simulator can use to get actions from fixed policies.
    """

    @abstractmethod
    def reset(self, observations: list[Any], infos: list[dict[Any]]) -> None:
        """Reset the internal state of the policy.

        This function is called once before the AXS internal simulator starts.

        Args:
            observations (list[Any]): The observations from the simulator.
            infos (list[dict[str, Any]]): The infos from the simulator.

        """
        raise NotImplementedError

    @abstractmethod
    def update(self, observations: list[Any], infos: list[dict[str, Any]]) -> None:
        """Update the internal policy state.

        This function is called after a macro action is completed in the simulator,
        providing a hook for the policy to update its internal state from the simulator.

        Args:
            observations (list[Any]): The observations from the simulator.
            infos (list[dict[str, Any]]): The infos from the simulator.

        """
        raise NotImplementedError

    @abstractmethod
    def next_action(
        self,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> Any:
        """Get the next action from the policy.

        Args:
            observation (Any): The observation from the environment.
            info (dict[str, Any] | None): The info dict from the environment.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create(cls, env: SupportedEnv | None = None) -> dict[int, "Policy"]:
        """Create a policy for each agent in the environment.

        Args:
            env (SupportedEnv): The environment to create policies for.

        """
        raise NotImplementedError
