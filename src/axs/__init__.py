"""The axs package allows users to asks questions and generate explanations for RL envs.

AXS currently supports gymnasium and pettingzoo environments using an agentic workflow.

Most of the functionality is encapsulated in the AXSAgent class, which
is the main agent class for the AXS framework.
"""

import datetime
import logging
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import pettingzoo
import typer
from rich.logging import RichHandler

from .agent import AXSAgent
from .config import Config
from .macroaction import ActionSegment, MacroAction
from .prompt import Prompt
from .verbalize import Verbalizer

__all__ = [
    "AXSAgent",
    "ActionSegment",
    "Config",
    "MacroAction",
    "Prompt",
    "SupportedEnv",
    "Verbalizer",
    "init_logging",
]

SupportedEnv = gym.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv
SupportedEnv.__doc__ = "The supported environment types for the AXS agent."

logger = logging.getLogger(__name__)
app = typer.Typer()


def init_logging(
    muted: list[str] | None = None,
    log_dir: str | None = None,
    log_name: str | None = None,
) -> None:
    """Initialize logging.

    Args:
        muted (List[str]): List of loggers to mute.
        log_dir (str): Directory to save logs.
        log_name (str): Name base of the log file.

    """
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    for mute in muted:
        logging.getLogger(mute).setLevel(logging.WARNING)

    # Add saving to file if arguments are provided
    if log_dir and log_name:
        if not Path.exists(log_dir):
            Path.mkdir(log_dir, parents=True)
        date_time = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        log_path = Path(f"{log_dir}/{log_name}_{date_time}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(threadName)-10.10s:%(name)-20.20s] [%(levelname)-6.6s] %(message)s",
            ),
        )
        logging.getLogger().addHandler(file_handler)


@app.command()
def main(
    config_file: Annotated[
        str, typer.Option(help="The path to the configuration file.")
    ] = "data/config.json",
    output: Annotated[
        str,
        typer.Option(help="The path to the output directory."),
    ] = "output/",
) -> None:
    """Execute the AXS agent according to the configuration file."""
    if not Path.exists(config_file):
        error_msg = f"Configuration file not found: {config_file}"
        raise FileNotFoundError(error_msg)
    if not Path.exists(output):
        Path.mkdir(output, parents=True)

    config_file = "data/igp2/configs/scenario1.json"
    config = Config(config_file)
    env = gym.make(
        config.env.name, config=config.env.params, render_mode=config.env.render_mode
    )
    observation, info = env.reset(seed=config.env.seed)

    axs_agent = AXSAgent(config)
    ego_agent = None  # TODO: Add loading of environment agent(s) here

    for n in range(config.env.n_episodes):
        logger.info("Running episode %d...", n)

        axs_agent.reset()

        # Execute external environment as normal
        for t in range(config.env.max_iter):
            action = None  # ego_agent.next_action(observation)

            # Learning and explanation phase
            axs_agent.semantic_memory.learn(
                observations=observation, actions=action, infos=info
            )
            for prompt in config.axs.user_prompts:
                if t > 0 and prompt.time == t - 1:
                    user_query = prompt.generate()
                    axs_agent.explain(user_query)

            # Perform environment step
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                logger.info("Episode terminated.")
                observation, info = env.reset(seed=config.env.seed)
                break

    env.close()


def run():
    """Entry-point for CLI."""
    init_logging()
    app()


if __name__ == "__main__":
    run()
