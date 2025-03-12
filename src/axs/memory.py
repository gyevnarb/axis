""" Memory module for storing facts about the environment and iterative experience. """
import abc
from typing import Any, Union


class Memory(abc.ABC):
    """ Base class for memory components. """
    def __init__(self):
        self._mem = {}

    @abc.abstractmethod
    def retrieve(self, key: Union[int, str]) -> Any:
        """ Retrieve experience from memory with lookup key. """
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, experience: Any, key: Union[int, str] = None) -> None:
        """ Commit experience to memory with lookup key. """
        raise NotImplementedError

    def reset(self) -> None:
        """ Remove all data from memory. """
        self._mem = {}


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

    def learn(self, experience: Any, key: str = None) -> None:
        """ Commit experience to memory with lookup key.
        If key already exists, then append to existing memory.

        Args:
            key (str): The key to associate with the experience.
            experience (Any): The experience to store in memory.
        """
        if key is None:
            key = "key" + str(len(self._mem))

        if key in self._mem:
            if isinstance(self._mem[key], list):
                self._mem[key].append(experience)
            else:
                self._mem[key] = [self._mem[key], experience]
        else:
            self._mem[key] = experience


class EpisodicMemory(Memory):
    """ Episodic memory stores the iterative experiences of the agent. """

    def __init__(self):
        super().__init__()
        self._mem = []

    def retrieve(self, key: int) -> Any:
        """ Retrieve experience from memory with lookup index.
        Raises IndexError if key is out of bounds.

        Args:
            key (int): The index to lookup in memory.
        """
        return self._mem[key]

    def learn(self, experience: Any, key: int = None) -> None:
        """ Append or insert at index the experience to memory.

        Args:
            experience (Any): The experience to store in memory.
            key (int): The index to insert the experience at.
        """
        if key is None:
            self._mem.append(experience)
        else:
            if key < 0 or key > len(self._mem):
                raise IndexError(f"Index: {key} out of bounds for episodic memory.")
            self._mem.insert(key, experience)
