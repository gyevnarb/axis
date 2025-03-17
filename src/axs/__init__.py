""" The axs package allows users to asks questions and generate
explanations for any gymnasium environment using an agentic workflow.

Most of the functionality is encapsulated in the AXSAgent class, which
is the main agent class for the AXS framework.
"""
import os
import logging
from typing import Annotated

import typer
import gymnasium as gym
from rich.logging import RichHandler

from axs.config import Config
from axs.agent import AXSAgent
from axs.prompt import Prompt
from axs.macroaction import MacroActionFactory, MacroAction, IGP2MacroAction


logger = logging.getLogger(__name__)
app = typer.Typer()

def init_logging(log_dir: str = None, log_name: str = None):
    """ Initialize logging. """
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )


@app.command()
def main(
    config_file : Annotated[
        str,
        typer.Option(help="The path to the configuration file.")
    ] = "data/config.json",
    output: Annotated[
        str,
        typer.Option(help="The path to the output directory."),
    ] = "output/") -> None:
    """ Execute the AXS agent according to the configuration file. """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    if not os.path.exists(output):
        os.makedirs(output)

    config_file = "data/igp2/configs/scenario1.json"
    config = Config(config_file)
    env = gym.make(config.env.name,
                   config=config.env.params,
                   render_mode=config.env.render_mode)
    observation, info = env.reset(seed=config.env.seed)

    axs_agent = AXSAgent(config)
    ego_agent = None  # TODO: Add environment agent(s) here

    for n in range(config.env.n_episodes):
        logger.info(f"Running episode {n}...")

        axs_agent.reset()

        # Execute external environment as normal
        for t in range(config.env.max_iter):
            action = None  # ego_agent.next_action(observation)

            # Learning and explanation phase
            axs_agent.semantic_memory.learn(
                observations=observation, actions=action, infos=info)
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
    """ Entry-point for CLI. """
    init_logging()
    app()


if __name__ == "__main__":
    run()
