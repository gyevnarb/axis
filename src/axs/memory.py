"""Memory module for storing facts about the environment and iterative experience."""

import abc
import logging
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Memory(abc.ABC):
    """Base class for memory components."""

    def __init__(
        self, memory: Iterable | None = None, save_file: Path | None = None,
    ) -> "Memory":
        """Initialize memory with optional memory object.

        Default memory is an empty dictionary. The memory object can be
        overriden to use a different data structure.

        To turn on automatic memory saving, set save_file to a Path object and
        set self.saving to True. The memory will be saved to the file when
        the learn method is called.

        Args:
            memory (Iterable | None): Optional initial memory object.
            save_file (Path | None): Optional file path to save memory.

        """
        self.save_file = save_file
        self.saving = False
        if memory is not None:
            self._mem = memory
        else:
            self._mem = {}

    def __getitem__(self, key: int | str) -> Any:
        """Get item from memory using the key."""
        return self.retrieve(key)

    @abc.abstractmethod
    def retrieve(self, key: int | str) -> Any:
        """Retrieve experience from memory with lookup key."""
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        """Commit experience to memory passed as (keyword) args."""
        raise NotImplementedError

    def reset(self) -> None:
        """Remove all data from memory."""
        self._mem = {}

    def save_memory(self) -> Path | None:
        """Save the memory to file if config.save_results is True."""
        if self.save_file is not None and self.saving:
            with self.save_file.open("wb") as f:
                pickle.dump(self._mem, f)
                logger.debug("Memory saved to %s", self.save_file)

    def load_memory(self, path: str | Path) -> None:
        """Load memory from file."""
        with Path(path).open("rb") as f:
            self._mem = pickle.load(f)
            logger.debug("Memory loaded from %s", path)

    @property
    def memory(self) -> Iterable:
        """Return the memory object."""
        return self._mem


class SemanticMemory(Memory):
    """Semantic memory stores facts about the environment."""

    def retrieve(self, key: str) -> Any:
        """Retrieve experience from memory with lookup key.

        Args:
            key (str): The key to lookup in memory.

        """
        if key not in self._mem:
            error_msg = f"Key: {key} not found in semantic memory."
            raise KeyError(error_msg)
        return self._mem[key]

    def learn(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        """Commit experiences to memory with keys given as keyword arguments.

        Positional arguments are saved with the 'default'.

        Args:
            args: (list[Any]): Experiences to store in memory.
            kwargs (dict[str, Any]): Named experiences to store in memory.

        """
        for value in args:
            if "default" not in self._mem:
                self._mem["default"] = [value]
            else:
                self._mem["default"].append(value)

        for key, value in kwargs.items():
            if key in self._mem:
                if isinstance(self._mem[key], list):
                    self._mem[key].append(value)
                else:
                    self._mem[key] = [self._mem[key], value]
            elif isinstance(value, list):
                self._mem[key] = [value]
            else:
                self._mem[key] = value
        self.save_memory()

    @property
    def keys(self) -> list[str]:
        """Return the keys in semantic memory."""
        return list(self._mem.keys())


class EpisodicMemory(Memory):
    """Episodic memory stores the iterative experiences of the agent."""

    def __init__(
        self, memory: Iterable | None = None, save_file: Path | None = None,
    ) -> "EpisodicMemory":
        """Initialize episodic memory as a list."""
        super().__init__(memory, save_file)
        if memory is None:
            self._mem = []

    def reset(self) -> None:
        """Set internal memory to an empty list."""
        self._mem = []

    def retrieve(self, key: int) -> Any:
        """Retrieve experience from memory with lookup index.

        Raises IndexError if key is out of bounds.

        Args:
            key (int): The index to lookup in memory.

        """
        return self._mem[key]

    def learn(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        """Append the experience to memory. Kwargs are ignored.

        Args:
            args: (list[Any]): Experiences to store in memory.
            kwargs (dict[str, Any]): Named experiences to store in memory.

        """
        for value in args:
            self._mem.append(value)
        self.save_memory()
