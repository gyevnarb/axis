"""Contains wrapper class for easy call to next action in the simulation."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class Policy(ABC):
    """Interface class for agent policies.

    You can either subclass this interface or wrap an existing policy in it.
    The purpose of this class is to provide a single function, next_action(),
    which the AXS simulator can use to get actions from fixed policies.
    """

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
    def from_function(
        cls, func: Callable[[Any, dict[str, Any] | None], Any],
    ) -> "Policy":
        """Create a Policy from a function.

        Creates an instance of the Policy class and sets the next_action
        function to the given function.

        Args:
            func (callable): The function to create the Policy from.
                It should take an observation and an info dict as arguments
                and return an action.

        """
        new_policy = cls()
        new_policy.next_action = func
        return new_policy
