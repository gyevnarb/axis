"""AXS: Agetnic eXplanations with Simulations for RL Environments.

The axs package allows users to asks questions and generate explanations for RL envs.
AXS currently supports gymnasium and pettingzoo environments using an agentic workflow.

Most of the functionality is encapsulated in the AXSAgent class, which
is the main agent class for the AXS framework. Other classes may be overriden
to support different environments or use cases.

MacroAction:
Verbalize:
Query:
Policy:
Wrapper:
"""

import logging
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import typer

from . import util
from .agent import AXSAgent
from .config import (
    AXSConfig,
    Config,
    EnvConfig,
    LLMConfig,
    MacroActionConfig,
    SupportedEnv,
    VerbalizerConfig,
    registry,
)
from .macroaction import ActionSegment, MacroAction
from .memory import EpisodicMemory, SemanticMemory
from .policy import Policy
from .prompt import Prompt
from .query import Query, QueryError, QueryTypeMap
from .simulator import SimulationError, Simulator
from .verbalize import Verbalizer
from .wrapper import QueryableAECWrapper, QueryableWrapper

__all__ = [
    "AXSAgent",
    "AXSConfig",
    "ActionSegment",
    "Config",
    "EnvConfig",
    "EpisodicMemory",
    "LLMConfig",
    "MacroAction",
    "MacroActionConfig",
    "Policy",
    "Prompt",
    "Query",
    "QueryError",
    "QueryTypeMap",
    "QueryableAECWrapper",
    "QueryableWrapper",
    "SemanticMemory",
    "SimulationError",
    "Simulator",
    "SupportedEnv",
    "Verbalizer",
    "VerbalizerConfig",
    "axs_registry",
    "init_logging",
    "util",
]

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    config_file: Annotated[
        str,
        typer.Option(help="The path to the configuration file."),
    ] = "data/config.json",
    output_dir: Annotated[
        str,
        typer.Option(help="The path to the output directory."),
    ] = "output/",
    debug: Annotated[
        bool,
        typer.Option(help="Enable debug mode.", is_eager=True),
    ] = False,
) -> None:
    """Run an AXS agent according to a configuration file."""
    util.init_logging(
        level="DEBUG" if debug else "INFO",
        warning_only=[
            "igp2.core.velocitysmoother",
            "matplotlib",
            "httpcore",
            "openai",
            "httpx",
        ],
    )

    if not Path(config_file).exists():
        error_msg = f"Configuration file not found: {config_file}"
        raise FileNotFoundError(error_msg)

    config = Config(config_file)

    if config.axs.output_dir is None:
        config.axs.output_dir = output_dir
    output_dir = config.axs.output_dir
    if not output_dir.exists():
        logger.info("Creating output directory %s", output_dir)
        output_dir.mkdir(parents=True)

    env = util.load_env(config.env)
    initial_state = env.reset(seed=config.env.seed)

    agent_policies = registry.get(config.env.policy_type).create(env)
    axs_agent = AXSAgent(config, agent_policies)

    if config.env.name in gym.registry:
        util.run_gym_env(env, axs_agent, config, *initial_state)
    else:
        util.run_aec_env(env, axs_agent, config)


def run() -> None:
    """Entry-point for CLI.

    This function may be called from any package that implements the AXS framework.
    It is the main entry point for the AXS package and should be called
    when the package is run as a script.
    """
    app()
