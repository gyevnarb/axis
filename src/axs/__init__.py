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
from .llm import LLMWrapper
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
    "LLMWrapper",
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
    "init_logging",
    "registry",
    "util",
]

logger = logging.getLogger(__name__)
app = typer.Typer()


def _init_axs(
    config_file: str,
    debug: bool,
    dryrun: bool,
    save_results: bool,
    output_dir: str,
) -> Config:
    """Initialize the AXS package."""
    if not Path(config_file).exists():
        error_msg = f"Configuration file not found: {config_file}"
        raise FileNotFoundError(error_msg)

    config = Config(config_file)
    if debug is not None:
        config.config_dict["debug"] = debug
    if dryrun is not None:
        config.config_dict["dryrun"] = dryrun
    if save_results is not None:
        config.config_dict["save_results"] = save_results
    if output_dir is not None:
        config.config_dict["output_dir"] = output_dir

    output_dir = config.output_dir
    if output_dir is not None:
        logger.info("Creating output directory structure: %s", output_dir)
        if not output_dir.exists():
            logger.info("  Creating output directory %s", output_dir)
            output_dir.mkdir(parents=True)
        agents_dir = Path(output_dir, "agents")
        if not agents_dir.exists():
            logger.info("  Creating agents directory %s", agents_dir)
            agents_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(output_dir, "checkpoints")
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir = Path(output_dir, "results")
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)

    return config


@app.command()
def run(ctx: typer.Context) -> None:
    """Run an AXS agent according to a configuration file."""
    config = ctx.obj["config"]

    env = util.load_env(config.env, config.env.render_mode)
    initial_state = env.reset(seed=config.env.seed)
    logger.info("Created environment %s", config.env.name)

    agent_policies = registry.get(config.env.policy_type).create(env)
    axs_agent = AXSAgent(config, agent_policies)

    import gymnasium as gym
    if config.env.name in gym.registry:
        logger.info("Running gym environment %s", config.env.name)
        util.run_gym_env(env, axs_agent, config, *initial_state)
    else:
        logger.info("Running pettingzoo environment %s", config.env.name)
        util.run_aec_env(env, axs_agent, config)


@app.command()
def evaluate(ctx: typer.Context) -> None:
    """Evaluate the AXS agent on all explanations in a configuration file.

    The scenario must have been run (either completely or using the --dryrun flag),
    with the observation and info data saved to disk (with --save_results flag).
    """
    config = ctx.obj["config"]

    # Find all save files with the prefix "agent_ep"
    # and the suffix ".pkl" in the output directory
    save_files = list(Path(config.output_dir, "agents").glob("agent_ep*.pkl"))
    if not save_files:
        error_msg = f"No save files found in {config.output_dir}"
        raise FileNotFoundError(error_msg)

    env = util.load_env(config.env, config.env.render_mode)
    env.reset(seed=config.env.seed)
    logger.info("Created environment %s", config.env.name)

    agent_policies = registry.get(config.env.policy_type).create(env)
    axs_agent = AXSAgent(config, agent_policies)

    # Iterate over all save files
    import datetime
    import pickle
    start_dt = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    results = {}
    for ep_ix, save_file in enumerate(save_files):
        ep_results = {}

        # Run all explanations.
        for p_ix, prompt_dict in enumerate(config.axs.user_prompts):
            prompt = Prompt(**prompt_dict)

            # Load the state of the agent from the file
            axs_agent.load_state(save_file)
            logger.info("Loaded state from %s", save_file)

            # Truncate the semantic memory until the current time
            if prompt.time is not None:
                semantic_memory = axs_agent.semantic_memory.memory
                for key in axs_agent.semantic_memory.memory:
                    semantic_memory[key] = semantic_memory[key][:prompt.time]

            user_query = prompt.fill()
            _, p_results = axs_agent.explain(user_query)

            if config.save_results:
                save_name = f"checkpoint_{start_dt}_p{p_ix}.pkl"
                save_path = Path(config.output_dir, "checkpoints", save_name)
                with save_path.open("wb") as f:
                    pickle.dump(p_results, f)
                    logger.info("Episode %d checkpoint saved to %s", ep_ix, save_path)
            ep_results[f"p{p_ix}"] = p_results

        if config.save_results:
            save_name = f"results_{start_dt}_ep{ep_ix}.pkl"
            save_path = Path(config.output_dir, "results", save_name)
            with save_path.open("wb") as f:
                pickle.dump(ep_results, f)
                logger.info("Episode %d results saved to %s", ep_ix, save_path)
        results[f"ep{ep_ix}"] = ep_results

    # Save the results to disk
    if config.save_results:
        save_path = Path(config.output_dir, f"final_{start_dt}.pkl")
        with Path(save_path).open("wb") as f:
            pickle.dump(results, f)
            logger.info("Final results saved to %s", save_path)


@app.callback()
def main(  # noqa: PLR0913
    ctx: typer.Context,
    config_file: Annotated[
        str,
        typer.Option("-c", "--config-file", help="The path to the configuration file."),
    ],
    output_dir: Annotated[
        str | None,
        typer.Option("-o", "--output-dir", help="The path to the output directory."),
    ] = None,
    save_results: Annotated[
        bool | None,
        typer.Option(
            "-s",
            "--save-results",
            help="Whether to save all run information to disk.",
        ),
    ] = None,
    debug: Annotated[
        bool | None,
        typer.Option(help="Enable debug mode for logging.", is_eager=True),
    ] = None,
    dryrun: Annotated[
        bool | None,
        typer.Option(help="Run the environment without executing any explanations."),
    ] = None,
    context: Annotated[
        bool | None,
        typer.Option(
            help="Whether to add initial context to the LLM. Requires specifying a "
            "'no_context.txt' file in the prompts directory.",
        ),
    ] = True,
    llm_kwargs: Annotated[
        str | None,
        typer.Option(
            help="Keyword arguments for the LLM formatted as a valid JSON string.",
        ),
    ] = None,
) -> None:
    """Run an AXS agent."""
    config = _init_axs(config_file, debug, dryrun, save_results, output_dir)
    if llm_kwargs is not None:
        import json
        config.config_dict["llm"].update(json.loads(llm_kwargs))
    if context is not None:
        config.config_dict["axs"]["use_context"] = context

    ctx.obj = {"config": config}


def cli() -> None:
    """Entry-point for CLI.

    This function may be called from any package that implements the AXS framework.
    It is the main entry point for the AXS package and should be called
    when the package is run as a script.
    """
    app()
