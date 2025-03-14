""" Configuration class for accessing JSON-based configuration files. """
import json
import abc
import os
from functools import lru_cache
from typing import Union, Dict, Iterator, Any

from axs.prompt import Prompt


class ConfigBase(abc.ABC):
    """ Abstract base class for configuration classes. """

    def __init__(self, config: Dict[str, Any]):
        """ Initialize the configuration class with the configuration dictionary. """
        self._config = config

    @property
    def config_dict(self) -> Dict[str, Any]:
        """ Return the configuration dictionary. """
        return self._config


class EnvConfig(ConfigBase):
    """ Configuration class for the environment. """

    @property
    def name(self) -> str:
        """ Environment name. """
        return self._config["name"]

    @property
    def max_iter(self) -> int:
        """ Maximum number of iterations for environment executinon.
        Default: 1000. """
        return self._config.get("max_iter", 1000)

    @property
    def n_episodes(self) -> int:
        """ Number of episodes for the environment. Default: 1. """
        return self._config.get("n_episodes", 1)

    @property
    def seed(self) -> int:
        """ Environment seed. Default: 28"""
        return self._config.get("seed", 28)

    @property
    def render_mode(self) -> str:
        """ Environment render mode. Default: 'human' """
        return self._config.get("render_mode", "human")

    @property
    def params(self) -> Dict[str, Any]:
        """ Return additinal environment configuration parameters. """
        return self._config.get("params", {})


class LLMConfig(ConfigBase):
    """ Configuration class for LLM model parameters. """

    @property
    def model(self) -> str:
        """ LLM model name. """
        return self._config["model"]

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        """ vLLM LLM constructor keyword arguments. """
        return self._config.get("model_kwargs", {})

    @property
    def sampling_params(self) -> Dict[str, Any]:
        """ vLLM SamplingParams. """
        return self._config.get("sampling_params", {})


class AXSConfig(ConfigBase):
    """ Configuration class for the AXS agent parameters. """

    @property
    def n_max(self) -> int:
        """ The maximum number of iterations for explanation generation.
        Default: 5. """
        return self._config.get("n_max", 5)

    @property
    def delta(self) -> float:
        """ The minimum distance between explanations for convergence.
        Default: 0.01. """
        return self._config.get("delta", 0.01)

    @property
    def system_prompt(self) -> str:
        """ The system prompt for the AXSAgent to use.
        Either a string or a file path. """
        value = self._config.get("system_prompt", "")
        if os.path.exists(value):
            with open(value, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return value

    @property
    def user_prompts(self) -> Iterator[Prompt]:
        """ The prompts the users asks the agent. """
        return map(lambda x: Prompt(**x),
                   self._config.get("user_prompts", []))

    @property
    @lru_cache(maxsize=64)
    def query_template(self) -> str:
        """ The query template for the AXSAgent to use.
        The template should be a valid string with placeholders for the
        str.format() method and may be specified in the config file or
        as a separate file with a valid path to it. """
        value = self._config.get("query_template", "")
        if os.path.exists(value):
            with open(value, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return value

    @property
    @lru_cache(maxsize=64)
    def explanation_template(self) -> str:
        """ The explanation template for the AXSAgent to use.
        The template should be a valid string with placeholders for the
        str.format() method and may be specified in the config file or
        as a separate file with a valid path to it. """
        value = self._config.get("explanation_template", "")
        if os.path.exists(value):
            with open(value, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return value


class Config(ConfigBase):
    """ Convenience class for accessing configuration values. """

    def __init__(self, config: Union[str, Dict[str, Any]]):
        super().__init__(config)
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            self._config = config

    @property
    def env(self) -> EnvConfig:
        """ Return the gymnasium environment configuration. """
        return EnvConfig(self._config["env"])

    @property
    def llm(self) -> LLMConfig:
        """ Return the LLM model configuration. """
        return LLMConfig(self._config["llm"])

    @property
    def axs(self) -> AXSConfig:
        """ Return the AXS agent configuration. """
        return AXSConfig(self._config["axs"])
