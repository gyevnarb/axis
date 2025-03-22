"""Configuration class for accessing JSON-based configuration files."""

import json
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import pettingzoo

SupportedEnv = Union[gym.Env, pettingzoo.ParallelEnv, pettingzoo.AECEnv]  # noqa: UP007
SupportedEnv.__doc__ = "The supported environment types for the AXS agent."

_registry: dict[str, "type[Registerable]"] = {}


class Registerable:
    """Abstract base class for registerable classes."""

    def __init_subclass__(cls, class_type: type | None = object) -> None:
        """Register a type in the registry.

        Args:
            class_type: The abstract type of the class to register.
                Used for checking the class inheritance.

        """
        if class_type is None:
            return

        name = cls.__name__
        if not issubclass(cls, class_type):
            error_msg = f"Type {cls} is not a subclass of {class_type}."
            raise TypeError(error_msg)
        if name in _registry:
            error_msg = f"Type {name} already registered in the registry."
            raise ValueError(error_msg)
        _registry[name] = cls

    @classmethod
    def get(cls, name: str) -> type:
        """Get the type from the library by name.

        Args:
            name (str): The name of the type.

        """
        if name not in _registry:
            error_msg = (
                f"Type {name} not found in the registry "
                f"with {_registry.keys()}."
            )
            raise ValueError(error_msg)
        return _registry[name]


class ConfigBase:
    """Abstract base class for configuration classes."""

    def __init__(self, config: dict[str, Any]) -> "ConfigBase":
        """Initialize the configuration class with the configuration dictionary."""
        self._config = config

    @property
    def config_dict(self) -> dict[str, Any]:
        """Return the configuration dictionary."""
        return self._config


class EnvConfig(ConfigBase):
    """Configuration class for the environment."""

    @property
    def name(self) -> str:
        """Environment name."""
        return self._config["name"]

    @property
    def wrapper_type(self) -> str:
        """QueryableWrapper type for the simulator to use."""
        return self._config["wrapper_type"]


    @property
    def env_type(self) -> str | None:
        """Optional environment type used for pettingzoo environments.

        Either 'parallel' or 'aec'.
        """
        return self._config.get("env_type", None)

    @property
    def max_iter(self) -> int:
        """Maximum number of iterations for environment executinon.

        Default: 1000.
        """
        return self._config.get("max_iter", 1000)

    @property
    def n_episodes(self) -> int:
        """Number of episodes for the environment.

        Default: 1.
        """
        return self._config.get("n_episodes", 1)

    @property
    def seed(self) -> int:
        """Environment seed.

        Default: 28.
        """
        return self._config.get("seed", 28)

    @property
    def render_mode(self) -> str:
        """Environment render mode.

        Default: 'human'
        """
        return self._config.get("render_mode", "human")

    @property
    def params(self) -> dict[str, Any]:
        """Return additinal environment configuration parameters."""
        return self._config.get("params", {})


class LLMConfig(ConfigBase):
    """Configuration class for LLM model parameters."""

    @property
    def inference_mode(self) -> str:
        """Inference mode for the LLM model.

        Default: 'localhost'
        """
        value = self._config.get("inference_mode", "online")
        if value not in ["online", "offline", "localhost"]:
            error_msg = (f"Invalid LLM inference mode: {value}; "
                         f"must be 'online', 'offline', or 'localhost'.")
            raise ValueError(error_msg)
        return value

    @property
    def base_url(self) -> str | None:
        """Base URL for online/localhost LLM model."""
        return self._config.get("base_url", None)

    @property
    def model(self) -> str:
        """Get LLM model name.

        Default: meta-llama/Llama-3.2-3B-Instruct
        """
        return self._config.get("model", "meta-llama/Llama-3.2-3B-Instruct")

    @property
    def model_kwargs(self) -> dict[str, Any]:
        """Get vLLM LLM constructor keyword arguments."""
        return self._config.get("model_kwargs", {})

    @property
    def sampling_params(self) -> dict[str, Any]:
        """Get vLLM SamplingParams."""
        return self._config.get("sampling_params", {})


class MacroActionConfig(ConfigBase):
    """Configuration class for MacroAction creation."""

    @property
    def type_name(self) -> str:
        """The type name of the macro action."""
        return self._config["type_name"]

    @property
    def params(self) -> dict[str, Any]:
        """Additional parameters for the macro action."""
        return self._config.get("params", {})


class VerbalizerConfig(ConfigBase):
    """Configuration class for the verbalizer."""

    @property
    def type_name(self) -> str:
        """The name of the verbalizer."""
        return self._config["type_name"]

    @property
    def params(self) -> dict[str, Any]:
        """Additional parameters for the verbalizer."""
        return self._config.get("params", {})


class QueryConfig(ConfigBase):
    """Configuration class for queries."""

    @property
    def type_name(self) -> str:
        """The name of the query."""
        return self._config["type_name"]

    @property
    def params(self) -> dict[str, Any]:
        """Additional parameters for the query."""
        return self._config.get("params", {})

class AXSConfig(ConfigBase):
    """Configuration class for the AXS agent parameters."""

    @property
    def n_max(self) -> int:
        """The maximum number of iterations for explanation generation.

        Default: 5.
        """
        value = self._config.get("n_max", 5)
        if value < 1:
            error_msg = f"Invalid value for n_max: {value}; must be >= 1."
            raise ValueError(error_msg)
        return value

    @property
    def delta(self) -> float:
        """Minimum distance between explanations for convergence.

        Default: 0.01.
        """
        value = self._config.get("delta", 0.01)
        if value <= 0:
            error_msg = f"Invalid value for delta: {value}; must be > 0."
            raise ValueError(error_msg)
        return value

    @property
    def macro_action(self) -> MacroActionConfig:
        """The macro action configuration for the AXSAgent."""
        return MacroActionConfig(self._config["macro_action"])

    @property
    def verbalizer(self) -> VerbalizerConfig:
        """The verbalizer configuration for the AXSAgent."""
        return VerbalizerConfig(self._config["verbalizer"])

    @property
    def query(self) -> QueryConfig:
        """The query configuration for the AXSAgent."""
        return QueryConfig(self._config["query"])

    @property
    def user_prompts(self) -> list[dict[str, Any]]:
        """The prompts the users asks the agent."""
        return self._config.get("user_prompts", [])

    @property
    def system_template(self) -> str:
        """The system prompt template for the AXSAgent to use.

        The template should be a valid string with placeholders for the
        str.format() method and may be specified in the config file or
        as a separate file with a valid path to it.
        """
        default_value = (
            "You write helpful explanations based on a multi-round "
            "sequence of dialog over {n_max} rounds."
        )
        value = self._config.get("system_prompt", default_value)
        if Path(value).exists():
            with Path(value).open("r", encoding="utf-8") as f:
                return f.read()
        else:
            return value

    @property
    def query_template(self) -> str:
        """The query template for the AXSAgent to use.

        The template should be a valid string with placeholders for the
        str.format() method and may be specified in the config file or
        as a separate file with a valid path to it.
        """
        value = self._config.get("query_template", "")
        if Path(value).exists():
            with Path(value).open("r", encoding="utf-8") as f:
                return f.read()
        else:
            return value

    @property
    def explanation_template(self) -> str:
        """The explanation template for the AXSAgent to use.

        The template should be a valid string with placeholders for the
        str.format() method and may be specified in the config file or
        as a separate file with a valid path to it.
        """
        value = self._config.get("explanation_template", "")
        if Path(value).exists():
            with Path(value).open("r", encoding="utf-8") as f:
                return f.read()
        else:
            return value


class Config(ConfigBase):
    """Convenience class for accessing configuration values."""

    def __init__(self, config: dict[str, Any] | str) -> "Config":
        """Initialize the configuration class with the configuration dictionary."""
        super().__init__(config)
        if isinstance(config, str):
            with Path(config).open("r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            self._config = config

    @property
    def env(self) -> EnvConfig:
        """Return the gymnasium environment configuration."""
        return EnvConfig(self._config["env"])

    @property
    def llm(self) -> LLMConfig:
        """Return the LLM model configuration."""
        return LLMConfig(self._config["llm"])

    @property
    def axs(self) -> AXSConfig:
        """Return the AXS agent configuration."""
        return AXSConfig(self._config["axs"])
