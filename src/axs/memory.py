""" Memory module for storing facts about the environment and iterative experience. """
import abc
from typing import Any, Union, List, Iterable


class Memory(abc.ABC):
    """ Base class for memory components. """
    def __init__(self, memory: Iterable = None):
        if memory is not None:
            self._mem = memory
        else:
            self._mem = {}

    @abc.abstractmethod
    def retrieve(self, key: Union[int, str]) -> Any:
        """ Retrieve experience from memory with lookup key. """
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, *args, **kwargs: Any) -> None:
        """ Commit experience to memory passed as (keyword) args. """
        raise NotImplementedError

    def reset(self) -> None:
        """ Remove all data from memory. """
        self._mem = {}

    @property
    def memory(self) -> Iterable:
        """ Return the memory object. """
        return self._mem


class SemanticMemory(Memory):
    """ Semantic memory stores facts about the environment. """

    def retrieve(self, key: str) -> Any:
        """ Retrieve experience from memory with lookup key.

        Args:
            key (str): The key to lookup in memory.
        """
        if key not in self._mem:
            raise KeyError(f"Key: {key} not found in semantic memory.")
        return self._mem[key]

    def learn(self, *args, **kwargs: Any) -> None:
        """ Commit experiences to memory with keys given as keyword arguments,
        or the default key for positional arguments.

        Keywords Args:
            experience (Any): The experiences to store in memory.
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
            else:
                self._mem[key] = value

    @property
    def keys(self) -> List[str]:
        """ Return the keys in semantic memory. """
        return list(self._mem.keys())


class EpisodicMemory(Memory):
    """ Episodic memory stores the iterative experiences of the agent. """

    def __init__(self):
        super().__init__()
        self._mem = []

    def retrieve(self, idx: int) -> Any:
        """ Retrieve experience from memory with lookup index.
        Raises IndexError if key is out of bounds.

        Args:
            key (int): The index to lookup in memory.
        """
        return self._mem[idx]

    def learn(self, *args, **kwargs: Any) -> None:
        """ Append the experience to memory. Kwargs are ignored.

        Keywords Args:
            experience (Any): The experience to store in memory.
        """
        for value in args:
            self._mem.append(value)
