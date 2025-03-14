""" The axs package is package that allows the user to generate explanation for any gymnasium environment with an agentic workflow. """
import logging
from typing import Annotated

import typer
import gymnasium as gym
import igp2 as ip
from rich.logging import RichHandler

from axs.config import Config
from axs.agent import AXSAgent


logger = logging.getLogger(__name__)
app = typer.Typer()

def init_logging():
    """ Initialize logging. """
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )


@app.command()
def main(
    user_query : Annotated[
        str,
        typer.Argument(help="The query to be explained.")
    ],
    config_file : Annotated[
        str,
        typer.Option(help="The path to the configuration file.")
    ] = "data/config.json") -> None:
    """ Execute the AXS agent according to the configuration file. """

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
