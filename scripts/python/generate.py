"""Run various baselines for AXS agent evaluation."""

import json
import logging
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Annotated

import typer
from util import LLMModels, get_params

import axs
from envs import axs_igp2

app = typer.Typer()

logger = logging.getLogger(__name__)


@app.command()
def run(ctx: typer.Context) -> None:
    """Run AXS agent evaluation with various configurations."""
    scenario = ctx.obj["scenario"]
    model = ctx.obj["model"]
    complexity = ctx.obj["complexity"]
    interrogation = ctx.obj["interrogation"]
    complexity = [1, 2] if complexity is None else [complexity]
    save_name = ctx.obj["save_name"]
    features = ctx.obj["features"]
    override = ctx.obj["override"]
    prompt_idx = ctx.obj["prompt"]

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

    logger.info("Running features with model %s; scenario %d", model.value, scenario)

    params = get_params(
        scenarios=[scenario] if scenario != -1 else range(10),
        complexity=complexity,
        models=[model.value] if model != "all" else [m.value for m in LLMModels],
        features=features,
        use_interrogation=interrogation,
        use_context=ctx.obj["context"],
        n_max=ctx.obj["n_max"] if interrogation else 0,
    )

    logger.info("Found %d parameters to evaluate", len(params))

    scenario_config = axs.Config(f"data/igp2/configs/scenario{scenario}.json")
    env = axs.util.load_env(scenario_config.env, scenario_config.env.render_mode)
    env.reset(seed=scenario_config.env.seed)
    agent_policies = axs.registry.get(scenario_config.env.policy_type).create(env)
    logger.info("Created environment %s", scenario_config.env.name)

    agent_file = Path(scenario_config.output_dir, "agents", "agent_ep0.pkl")
    save_path = Path(scenario_config.output_dir, "results", f"{save_name}.pkl")
    logger.info("Using save path %s", save_path)
    if not override and Path(save_path).exists():
        with save_path.open("rb") as f:
            try:
                results = pickle.load(f)
                logger.info("Loaded %d results", len(results))
            except EOFError:
                logger.exception("File is empty, starting fresh.")
                results = []
    else:
        results = []

    # Backward compatibility: drop truncate from result params
    for result in results:
        if "truncate" in result["param"]:
            del result["param"]["truncate"]
        if "truncate" in result:
            del result["truncate"]

    for param in params:
        config = param.pop("config")

        # We are only using prompt 1 for the feature selection evaluation
        if prompt_idx is not None:
            prompts = [axs.Prompt(**config.axs.user_prompts[prompt_idx])]
        else:
            prompts = [
                axs.Prompt(**config.axs.user_prompts[i])
                for i in range(len(config.axs.user_prompts))
            ]
        if not prompts:
            logger.error("No prompts found in config.")
            raise typer.Exit(code=1)
        logger.info("Using %d prompts", len(prompts))

        for prompt in prompts:
            logger.info(param)

            if any(
                param == result["param"] and prompt == result["prompt"]
                for result in results
            ):
                logger.info("Already evaluated %s", param)
                continue

            # Load the state of the agent from the file
            axs_agent = axs.AXSAgent(config, agent_policies)
            axs_agent.load_state(agent_file)

            # Truncate the semantic memory until the current time
            if prompt.time is not None:
                semantic_memory = axs_agent.semantic_memory.memory
                for key in axs_agent.semantic_memory.memory:
                    semantic_memory[key] = semantic_memory[key][: prompt.time]

            # Generate explanation to prompt
            user_query = prompt.fill()
            try:
                _, exp_results = axs_agent.explain(user_query)
            except Exception as e:
                logger.exception("ERROR - %s - %s", prompt, param)
                raise typer.Exit(code=1) from e

            end_msg = f"{exp_results['success']} - {param}"
            logger.info(end_msg)

            exp_results["param"] = deepcopy(param)
            exp_results["prompt"] = prompt
            exp_results["config"] = config

            # Save results
            results.append(exp_results)

            with save_path.open("wb") as f:
                pickle.dump(results, f)
                logger.info("Results saved to %s", save_path)


@app.callback()
def main(  # noqa: PLR0913
    ctx: typer.Context,
    scenario: Annotated[
        int,
        typer.Option(
            "-s",
            "--scenario",
            help="The scenario to evaluate. -1 for all scenarios.",
            min=-1,
            max=9,
        ),
    ] = -1,
    model: Annotated[
        LLMModels | None,
        typer.Option(
            "-m",
            "--model",
            help="The LLM model to use to generate explanations.",
        ),
    ] = "all",
    complexity: Annotated[
        int | None,
        typer.Option(
            "-c", "--complexity", help="Complexity levels to use in evaluation.",
        ),
    ] = None,
    interrogation: Annotated[
        bool,
        typer.Option(help="Whether to use interrogation."),
    ] = True,
    context: Annotated[
        bool,
        typer.Option(help="Whether to add context to prompts."),
    ] = True,
    n_max: Annotated[
        int,
        typer.Option(help="Maximum number of samples to use."),
    ] = 6,
    features: Annotated[
        str | None,
        typer.Option(help="List of features formatted as valid JSON string."),
    ] = None,
    override: Annotated[
        bool,
        typer.Option(
            "-o",
            "--override",
            help="Whether to override the existing results.",
        ),
    ] = False,
    prompt: Annotated[
        int | None,
        typer.Option(
            "-p",
            "--prompt",
            help="The prompt index to use. If not given, all prompts are used.",
        ),
    ] = None,
) -> None:
    """Set feature selection parameters."""
    save_name = f"{model.value}"

    # If no explicit features are given, iterate over feature combinations
    if not features:
        save_name += "_features"
    save_name += "_interrogation" if interrogation else ""
    save_name += "_context" if context else ""

    if features:
        features = json.loads(features)

    ctx.obj = {
        "scenario": scenario,
        "model": model,
        "complexity": complexity,
        "interrogation": interrogation,
        "context": context,
        "n_max": n_max,
        "save_name": save_name,
        "features": features,
        "override": override,
        "prompt": prompt,
    }


if __name__ == "__main__":
    app()
