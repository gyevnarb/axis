"""Run gymnasium or PettingZoo environments with AXS agent."""

import datetime
import importlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import pettingzoo
from rich.logging import RichHandler

from axs.config import Config, EnvConfig, SupportedEnv
from axs.prompt import Prompt

logger = logging.getLogger(__name__)


def init_logging(
    level: str = "NOTSET",
    warning_only: list[str] | None = None,
    log_dir: str | None = None,
    log_name: str | None = None,
) -> None:
    """Initialize logging.

    Args:
        level (str): Logging level. Default is "NOTSET".
        warning_only (List[str]): List of loggers whose level is set to WARNING.
        log_dir (str): Directory to save logs.
        log_name (str): Name base of the log file.

    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    for mute in warning_only:
        if not mute:
            continue
        logging.getLogger(mute).setLevel(logging.WARNING)

    # Add saving to file if arguments are provided
    if log_dir and log_name:
        if not Path(log_dir).exists():
            Path(log_dir).mkdir(parents=True)
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


def load_env(config: EnvConfig, render_mode: str | None = None) -> SupportedEnv:
    """Load an environment based on the configuration.

    This function determines from the config whether to load a gymnasium or
    PettingZoo environment. It also handles the conversion of parallel
    environments to AEC environments if necessary. Raising an error if the
    environment type is not supported.

    Args:
        config (Config): The configuration for the environment.
        render_mode (str | None): The render mode for the environment.
            If None, the default 'human' render mode is used.

    Returns:
        SupportedEnv: The loaded environment.

    """
    env = None
    if config.name in gym.registry:
        env = gym.make(
            config.name,
            render_mode=render_mode,
            max_iters=config.max_iter,
            **config.params,
        )
    else:
        env_mod, env_class = config.pettingzoo_import.split(":")
        env_mod = importlib.import_module(env_mod)
        env_class = getattr(env_mod, env_class)
        if issubclass(env_class, pettingzoo.AECEnv):
            env = env_class(**config.params)
        elif issubclass(env_class, pettingzoo.ParallelEnv):
            env = pettingzoo.ParallelEnv(**config.params)
            env = pettingzoo.utils.parallel_to_aec(env)
        else:
            error_msg = f"Environment {config.name} cannot be initialized."
            raise ValueError(error_msg)

    if env is None:
        error_msg = f"Environment {config.name} cannot be initialized."
        raise ValueError(error_msg)
    return env


def run_gym_env(
    env: gym.Env,
    axs_agent: "AXSAgent",  # noqa: F821
    config: Config,
    observation: Any,
    info: dict[str, Any],
) -> list[str]:
    """Run a gymnasium environment with the AXS agent.

    Args:
        env (gym.Env): The gymnasium environment to run.
        axs_agent (AXSAgent): The AXS agent to run.
        config (Config): The configuration for the environment.
        observation (Any): The initial observation from the environment.
        info (dict[str, Any]): The initial info dict from the environment.

    Returns:
        list[str]: The resulting explanations of the run.

    """
    ego_agent = next(iter(axs_agent.agent_policies.values()))

    for n in range(config.env.n_episodes):
        logger.info("Running episode %d/%d", n + 1, config.env.n_episodes)

        axs_agent.reset()

        # Execute external environment as normal
        for t in range(config.env.max_iter):
            action = ego_agent.next_action(observation, info)

            # Learn observations and action
            axs_agent.semantic_memory.learn(
                observations=observation,
                actions=action,
                infos=info,
            )

            # Check whether there is anything to explain
            if not config.dryrun:
                for prompt_dict in config.axs.user_prompts:
                    user_prompt = Prompt(**prompt_dict)
                    if t == 0 or user_prompt.time != t:
                        continue
                    axs_agent.explain(user_prompt)
                    if config.save_results:
                        save_file = Path(
                            config.output_dir, "agents", f"agent_ep{n}_t{t}.pkl",
                        )
                        axs_agent.save_state(save_file)

            # Perform environment step
            observation, reward, terminated, truncated, info = env.step(action)

            # Learn step outcomes
            axs_agent.semantic_memory.learn(
                rewards=reward,
                terminated=terminated,
                truncated=truncated,
            )

            if terminated or truncated:
                logger.info("Episode %d terminated.", n + 1)
                axs_agent.semantic_memory.learn(
                    observations=observation,
                    actions=action,
                    infos=info,
                )
                observation, info = env.reset(seed=config.env.seed)
                break

        if config.save_results:
            save_file = Path(config.output_dir, "agents", f"agent_ep{n}.pkl")
            axs_agent.save_state(save_file)

    env.close()
    return axs_agent.semantic_memory["explanations"]


def run_aec_env(
    env: pettingzoo.AECEnv,
    axs_agent: "AXSAgent",
    config: Config,
) -> list[str]:
    """Run a PettingZoo AEC environment with the AXS agent.

    Args:
        env: The PettingZoo AEC environment to run.
        axs_agent: The AXS agent to run.
        config: The configuration for the environment.

    """
    agent_policies = axs_agent.agent_policies

    for n in range(config.env.n_episodes):
        logger.info("Running episode %d/%d", n + 1, config.env.n_episodes)

        axs_agent.reset()
        agent_times = defaultdict(int)

        for agent in env.agent_iter():
            t = agent_times[agent]

            observation, rewards, termination, truncation, info = env.last()

            axs_agent.semantic_memory.learn(
                agent=agent,
                observations=observation,
                infos=info,
                rewards=rewards,
                terminated=termination,
                truncated=truncation,
            )

            if termination or truncation:
                action = None
                logger.debug("Agent %d terminated or truncated.", agent)
            else:
                action = agent_policies[agent].next_action(observation, info)

            axs_agent.semantic_memory.learn(actions=action)

            # Check whether there is anything to explain
            if not config.dryrun:
                for prompt_dict in config.axs.user_prompts:
                    user_prompt = Prompt(**prompt_dict)
                    if t == 0 or user_prompt.time != t:
                        continue
                    axs_agent.explain(user_prompt)
                    if config.save_results:
                        save_file = Path(
                            config.output_dir,
                            "agents",
                            f"agent_ep{n}_ag{agent}_t{t}.pkl",
                        )
                        axs_agent.save_state(save_file)

            env.step(action)
            agent_times[agent] += 1

        if config.save_results:
            save_file = Path(config.output_dir, "agents", f"agent_ep{n}.pkl")
            axs_agent.save_state(save_file)
        logger.debug("Episode %d terminated.", n + 1)

    env.close()
    return axs_agent.semantic_memory["explanations"]
