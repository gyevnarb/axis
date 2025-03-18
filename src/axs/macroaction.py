"""This module contains the MacroAction class used for wrapping
low-level actions to higher level abstractions."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from axs.config import MacroActionConfig


@dataclass
class ActionSegment:
    """A segment of the same repeated actions with corresponding times.

    Attributes:
        times (List[int]): The timesteps of the actions.
        actions (List[Any]): The actions taken during the timesteps.
    """

    times: List[int]
    actions: List[Any]
    name: Tuple[str, ...]

    def __post_init__(self):
        """Check if the lengths of times and actions are equal."""
        if len(self.times) != len(self.actions):
            raise ValueError(
                f"Length of times {self.times} and actions {self.actions} must be equal."
            )


class MacroAction(ABC):
    """Abstract base class for macro actions."""

    _macro_library: Dict[str, "type[MacroAction]"] = {}
    macro_names: List[str] = []

    def __init__(self, name: str, action_segments: List[ActionSegment]):
        """Initialize the macro action with an empty list of actions.
        This method also checks whether the macro library has been defined.

        Args:
            name (str): The name of the macro action.
            action_segments (List[ActionSegment]): The action segments of the macro action.
        """
        if not self.macro_names:
            raise ValueError(f"Macro library of {type(self)} is empty.")
        if name not in self.macro_names:
            raise ValueError(
                f"Macro action {name} not found in the factory with {self.macro_names}."
            )
        self.macro_name = name

        if any(not isinstance(seg, ActionSegment) for seg in action_segments):
            raise ValueError(
                f"Action segments {action_segments} must be of type ActionSegment."
            )
        self.action_segments = action_segments
        self.__repr__()  # Call repr to check if it is implemented

    def __repr__(self):
        """String representation of the macro action. Used in verbalization."""
        raise RuntimeError("MacroAction __repr__ must be overriden.")

    @classmethod
    def register(cls, name: str, macro_type: type["MacroAction"]):
        """Register a macro action type with the factory.

        Args:
            name (str): The name of the macro action.
            macro_type (type[MacroAction]): The type of the macro action.
        """
        if not issubclass(macro_type, cls):
            raise ValueError(
                f"Macro action {macro_type} is not a subclass of MacroAction."
            )
        if name in cls._macro_library:
            raise ValueError(
                f"Macro action {name} already registered in the factory "
                f"with {cls._macro_library.keys()}."
            )
        cls._macro_library[name] = macro_type

    @classmethod
    def get(cls, name: str) -> "type[MacroAction]":
        """Get the macro action type from the factory.

        Args:
            name (str): The name of the macro action.

        Returns:
            type[MacroAction]: The type of the macro action.
        """
        if name not in cls._macro_library:
            raise ValueError(
                f"Macro action {name} not found in the factory with {cls._macro_library.keys()}."
            )
        return cls._macro_library[name]

    @classmethod
    @abstractmethod
    def wrap(
        cls, config: MacroActionConfig, actions, observations, infos=None
    ) -> Dict[int, List["MacroAction"]]:
        """Wrap the low-level actions, observations, or other
        information into macro actions for each agent present in the data.
        The actions within a macro action should be grouped into ActionSegments.

        Args:
            config (MacroActionConfig): The configuration for the macro action.
            actions: The low-level trajectory of actions of the agent to wrap.
            observation: The environment observation sequence.
            infos: Optional list of info dictionaries from the environment.

        Returns:
            Dict[int,List[MacroAction]]: A dictionary of agent ids to macro actions.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def unwrap(cls, macro_actions: List["MacroAction"]) -> List[Any]:
        """Unwrap the macro actions into low-level actions."""
        raise NotImplementedError

    @property
    def start_t(self):
        """The start time of the macro action."""
        return self.action_segments[0].times[0]

    @property
    def end_t(self):
        """The end time of the macro action."""
        return self.action_segments[-1].times[-1]
