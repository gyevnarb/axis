"""Run various baselines for AXS agent evaluation."""

import enum
import logging
import pickle
from pathlib import Path
from typing import Annotated, List

import typer
from util import get_params

import axs
from envs import axs_igp2

app = typer.Typer()

logger = logging.getLogger(__name__)


class LLMModels(enum.Enum):
    """Enum for LLM models."""

    llama_70b = "llama-70b"
    qwen_72b = "qwen-72b"
    gpt_4o = "gpt-4o"
    gpt_o1 = "gpt-o1"
    deepseek_v3 = "deepseek-v3"
    deepseek_r1 = "deepseek-r1"


@app.command()
def main(
    scenario: Annotated[
        int,
        typer.Argument(help="The scenario to evaluate.", min=0, max=9),
    ] = 1,
    model: Annotated[
        LLMModels,
        typer.Argument(help="The LLM model to use."),
    ] = "llama-70b",
    complexity: Annotated[
        int | None,
        typer.Option(help="Complexity levels to use in evaluation."),
    ] = None,
    interrogation: Annotated[
        bool,
        typer.Option(help="Whether to use interrogation."),
    ] = True,
    context: Annotated[
        bool, typer.Option(help="Whether to add context to prompts."),
    ] = True,
) -> None:
    """Run AXS agent evaluation with various configurations."""
    complexity = [1, 2] if complexity is None else [complexity]

    save_name = "test"
    if interrogation:
        save_name += "_interrogation"
    if context:
        save_name += "_context"

    axs.util.init_logging(
        level="INFO",
        warning_only=[
            "igp2.core.velocitysmoother",
            "matplotlib",
            "httpcore",
            "openai",
            "httpx",
        ],
        log_dir=f"output/igp2/scenario{scenario}/logs",
        log_name=save_name,
    )

    params = get_params(
        scenarios=[scenario],
        complexity=complexity,
        models=[model.value],
        use_interrogation=interrogation,
        use_context=context,
        n_max=10 if interrogation else 0,
    )

    scenario_config = axs.Config(f"data/igp2/configs/scenario{scenario}.json")
    env = axs.util.load_env(scenario_config.env, scenario_config.env.render_mode)
    env.reset(seed=scenario_config.env.seed)
    agent_policies = axs.registry.get(scenario_config.env.policy_type).create(env)
    logger.info("Created environment %s", scenario_config.env.name)

    agent_file = Path(scenario_config.output_dir, "agents", "agent_ep0.pkl")
    save_path = Path(scenario_config.output_dir, "results", f"{save_name}.pkl")
    if Path(save_path).exists():
        with save_path.open("rb") as f:
            results = pickle.load(f)
    else:
        results = []

    for param in params:
        config = param.pop("config")

        axs_agent = axs.AXSAgent(config, agent_policies)
        prompt = axs.Prompt(**config.axs.user_prompts[1])

        truncations = [True]
        if not interrogation:
            truncations.append(False)

        for truncate in truncations:
            param["truncate"] = truncate
            logger.info(param)

            if any(param == result["param"] for result in results):
                logger.info("Already evaluated %s", param)
                continue

            # Load the state of the agent from the file
            axs_agent.load_state(agent_file)

            # Truncate the semantic memory until the current time
            if truncate and prompt.time is not None:
                semantic_memory = axs_agent.semantic_memory.memory
                for key in axs_agent.semantic_memory.memory:
                    semantic_memory[key] = semantic_memory[key][: prompt.time]

            # Generate explanation to prompt
            user_query = prompt.fill()
            explanation, exp_results = axs_agent.explain(user_query)

            end_msg = f"{exp_results['success']} - {param}"
            logger.info(end_msg)

            # Save results
            results.append(
                {
                    "param": param,
                    "config": config,
                    "truncate": truncate,
                    "results": exp_results,
                },
            )

            with save_path.open("wb") as f:
                pickle.dump(results, f)
                logger.info("Results saved to %s", save_path)


if __name__ == "__main__":
    app()
