""" Configuration class for accessing JSON-based configuration files. """
import json
import abc
import os
from functools import lru_cache
from typing import Union, Dict, List, Any, Optional


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
    def env_type(self) -> Optional[str]:
        """ Optional environment type used for pettingzoo environments.
        Either 'parallel' or 'aec'. """
        return self._config.get("env_type", None)

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
    def inference_mode(self) -> str:
        """ Inference mode for the LLM model. Default: 'localhost' """
        value = self._config.get("inference_mode", "online")
        if value not in ["online", "offline", "localhost"]:
            raise ValueError(f"Invalid LLM inference mode: {value}; "
                             f"must be 'online', 'offline', or 'localhost'.")
        return value

    @property
    def base_url(self) -> Optional[str]:
        """ Base URL for online/localhost LLM model. """
        return self._config.get("base_url", None)

    @property
    def model(self) -> str:
        """ LLM model name. Default: meta-llama/Llama-3.2-3B-Instruct"""
        return self._config.get("model", "meta-llama/Llama-3.2-3B-Instruct")

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        """ vLLM LLM constructor keyword arguments. """
        return self._config.get("model_kwargs", {})

    @property
    def sampling_params(self) -> Dict[str, Any]:
        """ vLLM SamplingParams. """
        return self._config.get("sampling_params", {})


class MacroActionConfig(ConfigBase):
    """ Configuration class for MacroAction creation. """

    @property
    def name(self) -> str:
        """ The name of the macro action. """
        return self._config["name"]

    @property
    def params(self) -> Dict[str, Any]:
        """ Additional parameters for the macro action. """
        return self._config.get("params", {})


class VerbalizerConfig(ConfigBase):
    """ Configuration class for the verbalizer. """

    @property
    def name(self) -> str:
        """ The name of the verbalizer. """
        return self._config["name"]

    @property
    def params(self) -> Dict[str, Any]:
        """ Additional parameters for the verbalizer. """
        return self._config.get("params", {})


class AXSConfig(ConfigBase):
    """ Configuration class for the AXS agent parameters. """

    @property
    def n_max(self) -> int:
        """ The maximum number of iterations for explanation generation.
        Default: 5. """
        value = self._config.get("n_max", 5)
        if value < 1:
            raise ValueError(f"Invalid value for n_max: {value}; must be >= 1.")
        return value

    @property
    def delta(self) -> float:
        """ The minimum distance between explanations for convergence.
        Default: 0.01. """
        value = self._config.get("delta", 0.01)
        if value <= 0:
            raise ValueError(f"Invalid value for delta: {value}; must be > 0.")
        return value

    @property
    def macro_action(self) -> MacroActionConfig:
        """ The macro action configuration for the AXSAgent. """
        return MacroActionConfig(self._config["macro_action"])

    @property
    def verbalizer(self) -> VerbalizerConfig:
        """ The verbalizer configuration for the AXSAgent. """
        return VerbalizerConfig(self._config["verbalizer"])

    @property
    def user_prompts(self) -> List[Dict[str, Any]]:
        """ The prompts the users asks the agent. """
        return self._config.get("user_prompts", [])

    @property
    def system_template(self) -> str:
        """ The system prompt template for the AXSAgent to use.
        The template should be a valid string with placeholders for the
        str.format() method and may be specified in the config file or
        as a separate file with a valid path to it. """
        default_value = ("You write helpful explanations based on a multi-round "
                         "sequence of dialog over {n_max} rounds.")
        value = self._config.get("system_prompt", default_value)
        if os.path.exists(value):
            with open(value, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return value

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
