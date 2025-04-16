"""Run various baselines for AXS agent evaluation."""

import logging
import pickle
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from util import (
    LLMModels,
    get_actionable_accuracy,
    get_actionable_values,
    get_combined_score,
    get_params,
    get_shapley_values,
    plot_actionable_barplot,
    plot_shapley_waterfall,
)

import axs
from envs import axs_igp2

app = typer.Typer()

logger = logging.getLogger(__name__)


@app.callback()
def main(  # noqa: PLR0913
    ctx: typer.Context,
    scenario: Annotated[
        int,
        typer.Option(
            "-s",
            "--scenario",
            help="The scenario to evaluate.",
            min=0,
            max=9,
        ),
    ] = 1,
    model: Annotated[
        LLMModels,
        typer.Option("-m", "--model", help="The LLM model to use."),
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
        bool,
        typer.Option(help="Whether to add context to prompts."),
    ] = True,
    n_max: Annotated[
        int,
        typer.Option(help="Maximum number of samples to use."),
    ] = 6,
) -> None:
    """Set feature selection parameters."""
    save_name = f"{model.value}_features"
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

    ctx.obj = {
        "scenario": scenario,
        "model": model,
        "complexity": complexity,
        "interrogation": interrogation,
        "context": context,
        "n_max": n_max,
        "save_name": save_name,
    }


@app.command()
def shapley(ctx: typer.Context) -> None:
    """Calculate Shapley values for features and rank over all scenarios."""
    model = ctx.obj["model"]
    interrogation = ctx.obj["interrogation"]
    context = ctx.obj["context"]

    save_name = "evaluate_*_features"
    if interrogation:
        save_name += "_interrogation"
    if context:
        save_name += "_context"

    shapley_results = {}
    save_paths = list(Path("output", "igp2").glob(f"scenario*/results/{save_name}.pkl"))
    if not save_paths:
        logger.error("No results found for the given parameters.")
        return

    for save_path in save_paths:
        scenario = re.search(r"scenario(\d+)", str(save_path)).group(1)
        eval_model = re.search(
            rf"evaluate_([A-Za-z0-9-\.]+)_{model.value}_", str(save_path),
        ).group(1)
        if scenario != "1": continue
        with save_path.open("rb") as f:
            eval_results = pickle.load(f)

        scores = {}
        for eval_dict in eval_results:
            param = eval_dict.pop("param")
            features = set(param["verbalizer_features"])
            if param["truncate"]:
                features.add("truncate")
            if param["complexity"] == 2:
                features.add("complexity")
            scores[frozenset(features)] = get_combined_score(eval_dict)

        shapley_results[scenario + eval_model] = get_shapley_values(scores)

    # Combine Shapley values across scenarios, calculate mean and std
    combined_shapley = {}
    for shapley_values in shapley_results.values():
        for feature, value in shapley_values.items():
            if feature not in combined_shapley:
                combined_shapley[feature] = []
            combined_shapley[feature].append(value)
    combined_shapley = dict(sorted(combined_shapley.items(), key=lambda x: x[0]))
    # Drop zero mean, zero std features
    combined_shapley = {
        feature: scores
        for feature, scores in combined_shapley.items()
        if np.mean(scores) != 0 or np.std(scores) != 0
    }
    for feature, scores in combined_shapley.items():
        scores = np.array(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        combined_shapley[feature] = {
            "mean": mean_score,
            "std": std_score,
        }
        logger.info(
            "Feature: %s, Mean: %f, Std: %f, Scores: %s",
            feature,
            mean_score,
            std_score,
            axs_igp2.verbalize.util.ndarray2str(scores, 4),
        )

    plot_shapley_waterfall(combined_shapley)


@app.command()
def actionable(ctx: typer.Context) -> None:
    """Calculate actionability scores for each feature and rank over all scenarios."""
    model = ctx.obj["model"]
    interrogation = ctx.obj["interrogation"]
    context = ctx.obj["context"]

    save_name = "evaluate_*_features"
    if interrogation:
        save_name += "_interrogation"
    if context:
        save_name += "_context"

    actionable_results = {}
    save_paths = list(Path("output", "igp2").glob(f"scenario*/results/{save_name}.pkl"))
    if not save_paths:
        logger.error("No results found for the given parameters.")
        return

    for save_path in reversed(save_paths):
        scenario = re.search(r"scenario(\d+)", str(save_path)).group(1)
        eval_model = re.search(
            rf"evaluate_([A-Za-z0-9-\.]+)_{model.value}_", str(save_path),
        ).group(1)
        with save_path.open("rb") as f:
            eval_results = pickle.load(f)

        scores = {
            "actionable_exp": {},
            "actionable_no_exp": {},
        }
        for eval_dict in eval_results:
            param = eval_dict.pop("param")
            features = set(param["verbalizer_features"])
            if param["truncate"]:
                features.add("truncate")
            if param["complexity"] == 2:
                features.add("complexity")
            for explanation_given in ["actionable_exp", "actionable_no_exp"]:
                scores[explanation_given][frozenset(features)] = (
                    get_actionable_accuracy(eval_dict[explanation_given])
                )

        actionable_results[scenario + eval_model] = get_actionable_values(scores)

    # Combine actionable values across scenarios, calculate mean and std
    combined_actionable = {
        "actionable_exp": {},
        "actionable_no_exp": {},
    }
    for explanation_given in ["actionable_exp", "actionable_no_exp"]:
        val = {}
        for actionable_values in actionable_results.values():
            for feature, value in actionable_values[explanation_given].items():
                if feature not in val:
                    val[feature] = {"goal": [], "maneuver": []}
                val[feature]["goal"].append(value["goal"])
                val[feature]["maneuver"].append(value["maneuver"])
        combined_actionable[explanation_given] = val

    # Drop zero mean, zero std features
    for explanation_given in ["actionable_exp", "actionable_no_exp"]:
        for feature, scores in combined_actionable[explanation_given].items():
            for key in ["goal", "maneuver"]:
                mean_score = np.mean(scores[key])
                std_score = np.std(scores[key])
                scores[key] = {
                    "mean": mean_score,
                    "std": std_score,
                }
                logger.info(
                    "%s -- Feature: %s, Mean %s: %f, Std %s: %f",
                    "No explanation"
                    if explanation_given == "actionable_no_exp"
                    else "With explanation",
                    feature,
                    key,
                    mean_score,
                    key,
                    std_score,
                )

    plot_actionable_barplot(combined_actionable)


@app.command()
def run(ctx: typer.Context) -> None:
    """Run AXS agent evaluation with various configurations."""
    scenario = ctx.obj["scenario"]
    model = ctx.obj["model"]
    complexity = ctx.obj["complexity"]
    interrogation = ctx.obj["interrogation"]
    context = ctx.obj["context"]
    n_max = ctx.obj["n_max"]

    complexity = [1, 2] if complexity is None else [complexity]

    params = get_params(
        scenarios=[scenario],
        complexity=complexity,
        models=[model.value],
        use_interrogation=interrogation,
        use_context=context,
        n_max=n_max if interrogation else 0,
    )

    scenario_config = axs.Config(f"data/igp2/configs/scenario{scenario}.json")
    env = axs.util.load_env(scenario_config.env, scenario_config.env.render_mode)
    env.reset(seed=scenario_config.env.seed)
    agent_policies = axs.registry.get(scenario_config.env.policy_type).create(env)
    logger.info("Created environment %s", scenario_config.env.name)

    agent_file = Path(scenario_config.output_dir, "agents", "agent_ep0.pkl")
    save_name = ctx.obj["save_name"]
    save_path = Path(scenario_config.output_dir, "results", f"{save_name}.pkl")
    if Path(save_path).exists():
        with save_path.open("rb") as f:
            results = pickle.load(f)
    else:
        results = []

    for param in params:
        config = param.pop("config")

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
            axs_agent = axs.AXSAgent(config, agent_policies)
            axs_agent.load_state(agent_file)

            # Truncate the semantic memory until the current time
            if truncate and prompt.time is not None:
                semantic_memory = axs_agent.semantic_memory.memory
                for key in axs_agent.semantic_memory.memory:
                    semantic_memory[key] = semantic_memory[key][: prompt.time]

            # Generate explanation to prompt
            user_query = prompt.fill()
            _, exp_results = axs_agent.explain(user_query)

            end_msg = f"{exp_results['success']} - {param}"
            logger.info(end_msg)

            exp_results["param"] = param
            exp_results["truncate"] = truncate
            exp_results["config"] = config

            # Save results
            results.append(exp_results)

            with save_path.open("wb") as f:
                pickle.dump(results, f)
                logger.info("Results saved to %s", save_path)


if __name__ == "__main__":
    app()
