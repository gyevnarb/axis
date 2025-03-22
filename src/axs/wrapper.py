"""Gym and pettingzoo wrappers for easy simulation querying.

Wrappers need not implement all querying methods.
"""

from abc import ABC, abstractmethod

import gymnasium as gym
from pettingzoo.utils.wrappers import BaseWrapper


class QueryableWrapperBase(ABC):
    """Abstract class for simulation querying."""

    @abstractmethod
    def add(self, state):
        """Add a state to the simulation."""
        raise NotImplementedError

    @abstractmethod
    def remove(self, agent):
        """Remove an agent from the simulation."""
        raise NotImplementedError

    @abstractmethod
    def whatif(self, state):
        """Run a simulation with the given state."""
        raise NotImplementedError

    @abstractmethod
    def what(self):
        """Return the current state of the simulation."""
        raise NotImplementedError


class QueryableWrapper(gym.Wrapper, QueryableWrapperBase):
    """Wrapper class to support simulation querying."""

class QueryableAECWrapper(BaseWrapper, QueryableWrapperBase):
    """Wrapper class to support simulation querying for AEC environments."""
