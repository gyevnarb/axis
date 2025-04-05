"""Configuration class for accessing JSON-based configuration files."""

import json
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import pettingzoo

SupportedEnv = Union[gym.Env, pettingzoo.ParallelEnv, pettingzoo.AECEnv]  # noqa: UP007
SupportedEnv.__doc__ = "The supported environment types for the AXS agent."

registry: dict[str, "type[Registerable]"] = {}

POSSIBLE_PROMPTS = [
    "system",
    "context",
    "no_context",
    "interrogation",
    "explanation",
    "final",
]


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
        if name in registry:
            error_msg = f"Type {name} already registered in the registry."
            raise ValueError(error_msg)
        registry[name] = cls

    @classmethod
    def get(cls, name: str) -> type:
        """Get the type from the library by name.

        Args:
            name (str): The name of the type.

        """
        if name not in registry:
            error_msg = (f"Type {name} not found in registry with {registry.keys()}. "
                         f"Maybe you have forgotten an import.")
            raise ValueError(error_msg)
        return registry[name]


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
        """Name of the QueryableWrapper type for the simulator to use."""
        return self._config["wrapper_type"]

    @property
    def policy_type(self) -> str:
        """Name of the policy type for the simulator to use."""
        return self._config["policy_type"]

    @property
    def pettingzoo_import(self) -> str:
        """Import path of the environment type when using pettingzoo.

        This should be specified as an entry point to a class
        as a string following the setuptools syntax:
        >>> <name> = <package_or_module>:<object>
        """
        return self._config["pettingzoo_import"]

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
    def host_location(self) -> str:
        """Get where the LLM is hosted.

        Default: 'localhost'
        """
        value = self._config.get("inference_mode", "online")
        if value not in ["online", "offline", "localhost"]:
            error_msg = (
                f"Invalid LLM inference mode: {value}; "
                f"must be 'online', 'offline', or 'localhost'."
            )
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
    def use_context(self) -> bool:
        """Whether to add initial context to the LLM.

        Default: True.
        """
        return self._config.get("use_context", True)

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
    def n_tries(self) -> int:
        """The number of tries for the AXSAgent to generate an explanation.

        Default: 5.
        """
        return self._config.get("n_tries", 5)

    @property
    def prompts_dir(self) -> str:
        """The directory for the AXSAgent prompts."""
        return self._config["prompts_dir"]

    @property
    def prompts(self) -> dict[str, str]:
        """The name of prompt template types to template text.

        Prompts should be stored as a *.txt file in the prompts directory.
        The AXS agent relies on five prompt templates:
        'system', 'context', 'interrogation', 'explanation', and 'final'.
        """
        value = {
            path.stem: path.read_text() for path in Path(self.prompts_dir).glob("*.txt")
        }
        if not all(key in POSSIBLE_PROMPTS for key in value):
            error_msg = "Missing prompt templates in the prompts directory."
            raise ValueError(error_msg)
        return value

    @property
    def user_prompts(self) -> list[dict[str, Any]]:
        """The prompts the users asks the agent."""
        return self._config.get("user_prompts", [])


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

    @property
    def save_results(self) -> bool:
        """Whether to save all run information to disk.

        Default: True.
        """
        return self._config.get("save_results", True)

    @property
    def debug(self) -> bool:
        """Whether to use DEBUG level for logging.

        Default: False.
        """
        return self._config.get("debug", False)

    @property
    def dryrun(self) -> bool:
        """Whether to run environment without explanation.

        Default: False.
        """
        return self._config.get("dryrun", False)

    @property
    def output_dir(self) -> Path:
        """The directory for the AXSAgent output."""
        return Path(self._config["output_dir"])
