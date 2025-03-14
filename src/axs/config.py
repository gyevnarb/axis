""" Configuration class for accessing JSON-based configuration files. """
import json
import abc
from typing import Union, Dict, Iterator, Any

from vllm import SamplingParams

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
        """ Maximum number of iterations for environment executinon. Default: 1000. """
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
    def sampling_params(self) -> SamplingParams:
        """ vLLM SamplingParams. """
        return SamplingParams(self._config.get("sampling_params", {}))


class AXSConfig(ConfigBase):
    """ Configuration class for the AXS agent parameters. """

    @property
    def n_max(self) -> int:
        """ The maximum number of iterations for explanation generation. Default: 5. """
        return self._config.get("n_max", 5)

    @property
    def user_prompts(self) -> Iterator[Prompt]:
        """ The prompts the users asks the agent. """
        return map(lambda x: Prompt(**x),
                   self._config.get("user_prompts", []))


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
