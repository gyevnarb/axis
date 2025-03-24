"""Contains functions for wrapping low-level actions to higher level abstractions."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, ClassVar

from axs.config import MacroActionConfig, Registerable, SupportedEnv


@dataclass
class ActionSegment:
    """A segment of the same repeated actions with corresponding times.

    Attributes:
        times (list[int]): The timesteps of the actions.
        actions (list[Any]): The actions taken during the timesteps.
        name (Tuple[str]): The name of the action segment.

    """

    times: list[int]
    actions: list[Any]
    name: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check if the lengths of times and actions are equal."""
        if len(self.times) != len(self.actions):
            error_msg = (
                f"Length of times {self.times} and "
                f"actions {self.actions} must be equal."
            )
            raise ValueError(error_msg)


class MacroAction(ABC, Registerable, class_type=None):
    """Abstract base class for macro actions.

    Attributes:
        macro_names (list[str]): The names of the macro actions.

    """

    macro_names: ClassVar[list[str]] = []

    def __init__(
        self,
        macro_name: str,
        agent_id: int | None = None,
        config: MacroActionConfig | None = None,
        action_segments: list[ActionSegment] | None = None,
    ) -> "MacroAction":
        """Initialize the macro action with an empty list of actions.

        This method also checks whether the macro library has been defined.

        Args:
            macro_name (str): The name of the macro action.
            agent_id (int | None): The agent id to whome the macro action belongs.
            config (MacroActionConfig | None): Configuration for the macro action.
            action_segments (list[ActionSegment] | None): Action segments of the
                macroaction. Optional, and may be set later with from_observations().

        """
        if not self.macro_names:
            error_msg = f"Macro library of {type(self)} is empty."
            raise ValueError(error_msg)
        if macro_name not in self.macro_names:
            error_msg = f"Macro {macro_name} not found in library {self.macro_names}."
            raise ValueError(error_msg)
        self.macro_name = macro_name

        if config is not None and not isinstance(config, MacroActionConfig):
            error_msg = f"Configuration {config} must be of type MacroActionConfig."
            raise TypeError(error_msg)
        self.config = config

        if agent_id is not None and not isinstance(agent_id, int):
            error_msg = f"Agent id {agent_id} must be of type int."
            raise TypeError(error_msg)
        self.agent_id = agent_id

        self.action_segments = action_segments
        if self.action_segments is not None and any(
            not isinstance(seg, ActionSegment) for seg in action_segments
        ):
            error_msg = (
                f"Action segments {action_segments} must be of type ActionSegment."
            )
            raise TypeError(error_msg)

    def __iter__(self) -> Generator["MacroAction", None, None]:
        """Return the macro action object as an iterator."""
        if self.action_segments is None:
            error_msg = "Action segments not initialized."
            raise ValueError(error_msg)
        for segment in self.action_segments:
            yield from segment.actions

    @abstractmethod
    def __repr__(self) -> str:
        """Return representation of the macro action. Used in verbalization."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def wrap(
        cls,
        config: MacroActionConfig,
        actions: list[Any],
        observations: list[Any] | None = None,
        infos: list[dict[str, Any]] | None = None,
        env: SupportedEnv | None = None,
    ) -> dict[int, list["MacroAction"]]:
        """Wrap low-level actions, observations, or other infos into macro actions.

        Wrapping is done for each agent present in the data.
        For simple action spaces, this function may just return the actions as they are.
        The actions within a macro action are grouped into ActionSegments.

        Args:
            config (MacroActionConfig): Configuration for the macro action.
            actions (list[Any]): Low-level trajectory of actions of the agent to wrap.
            observations (list[Any] | None): Environment observation sequence.
            infos (list[dict[str, Any]] | None): list of info dictionaries from the env.
            env (SupportedEnv | None): Environment of the agent.

        Returns:
            Dict[int, list[MacroAction]]: A dictionary of agent ids to macro actions.

        """
        raise NotImplementedError

    def unwrap(self) -> list[Any]:
        """Unwrap the macro action into low-level actions."""
        return list(self)

    @abstractmethod
    def applicable(
        self,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> bool:
        """Check if the macro action is applicable in the given observation.

        Args:
            observation (Any): The observation to check applicability.
            info (Any | None): Optional environment info dict.

        """
        raise NotImplementedError

    @abstractmethod
    def done(
        self,
        observation: Any,
        info: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> bool:
        """Check if the macro action is done in the given observation.

        Args:
            observation (Any): The observation to check if the macro action is done.
            info (dict[str, Any] | None): Optional environment info dict.
            kwargs (dict[str, Any]): Additional optional keyword arguments.

        """
        raise NotImplementedError

    @abstractmethod
    def from_observation(
        self,
        observation: Any,
        info: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        """Initialize action segments of the macro action staring from an observation.

        This function will calculate the action segments for the macro action
        which the agent will take when executing the macro action.

        Args:
            observation (list[Any]): The observations to create the macro action.
            info (dict[str, Any]): The info dictionary from the environment.
            kwargs (dict[str, Any]): Additional optional keyword arguments.

        """
        raise NotImplementedError

    def next_action(
        self,
        observation: Any | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        """Return the next action of the macro action.

        By default, this process invokes next() on a generator object built
        from the action segments. It may be overriden to calculate the next action based
        on the current state of the environment.

        Args:
            observation (Any | None): The current observation.
            info (dict[str, Any] | None): The info dictionary from the environment.
            kwargs (dict[str, Any]): Additional optional keyword arguments.

        """
        return next(self)

    @property
    def start_t(self) -> int:
        """The start time of the macro action."""
        if not self.action_segments:
            error_msg = "Action segments are not initialized."
            raise ValueError(error_msg)
        return self.action_segments[0].times[0]

    @property
    def end_t(self) -> int:
        """The end time of the macro action."""
        if not self.action_segments:
            error_msg = "Action segments are not initialized."
            raise ValueError(error_msg)
        return self.action_segments[-1].times[-1]
