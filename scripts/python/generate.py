"""Run various baselines for AXS agent evaluation."""

import json
import logging
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from util import (
    LLMModels,
    get_actionable_accuracy,
    get_actionable_values,
    get_combined_score,
    get_params,
    get_save_paths,
    get_shapley_values,
    load_eval_results,
    plot_actionable_barplot,
    plot_shapley_waterfall,
)

import axs
from envs import axs_igp2

app = typer.Typer()

logger = logging.getLogger(__name__)


@app.command()
def shapley(
    ctx: typer.Context,
    eval_model: Annotated[
        LLMModels | None,
        typer.Option("-e", "--eval-model", help="The model used for evaluation."),
    ] = "all",
) -> None:
    """Calculate Shapley values for features and rank over all scenarios."""
    ctx.obj["eval_model"] = eval_model

    save_paths = get_save_paths(ctx)
    if not save_paths:
        return

    shapley_results = {}
    for save_path in save_paths:
        eval_results, (sid, eval_m_str, gen_m_str) = load_eval_results(ctx, save_path)
        if eval_results is None:
            continue

        scores = {}
        for eval_dict in eval_results:
            param = eval_dict.pop("param")
            features = set(param["verbalizer_features"])
            if param["truncate"]:
                features.add("truncate")
            if param["complexity"] == 2:
                features.add("complexity")
            scores[frozenset(features)] = get_combined_score(eval_dict)

        shapley_results[f"{sid}_{eval_m_str}_{gen_m_str}"] = get_shapley_values(
            scores,
        )

    if len(shapley_results) == 0:
        logger.error("No Shapley results found.")
        return
    logger.info(
        "Processed Shapley results for %d scenarios and models",
        len(shapley_results),
    )

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

    plot_shapley_waterfall(combined_shapley, ctx)


@app.command()
def actionable(
    ctx: typer.Context,
    eval_model: Annotated[
        LLMModels | None,
        typer.Option("-e", "--eval-model", help="The model used for evaluation."),
    ] = "all",
) -> None:
    """Calculate actionability scores for features and rank over all scenarios."""
    ctx.obj["eval_model"] = eval_model

    save_paths = get_save_paths(ctx)
    if not save_paths:
        return

    actionable_results = {}
    for save_path in reversed(save_paths):
        eval_results, (sid, eval_m_str, gen_m_str) = load_eval_results(ctx, save_path)
        if eval_results is None:
            continue

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

        actionable_results[f"{sid}_{eval_m_str}_{gen_m_str}"] = (
            get_actionable_values(
                scores,
            )
        )

    logger.info(
        "Processed actionable results for %d scenarios and models",
        len(actionable_results),
    )
    if len(actionable_results) == 0:
        logger.error("No actionable results found.")
        return

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

    # Remove memory_trunaction from the results
    for explanation_given in ["actionable_exp", "actionable_no_exp"]:
        del combined_actionable[explanation_given]["truncate"]
    plot_actionable_barplot(combined_actionable, ctx)


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
    save_name = ctx.obj["save_name"]
    features = ctx.obj["features"]

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
        use_context=context,
        n_max=n_max if interrogation else 0,
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
    if Path(save_path).exists():
        with save_path.open("rb") as f:
            try:
                results = pickle.load(f)
            except EOFError:
                logger.exception("File is empty, starting fresh.")
                results = []
    else:
        results = []

    for param in params:
        config = param.pop("config")

        # We are only using prompt 1 for the feature selection evaluation
        prompt = axs.Prompt(**config.axs.user_prompts[1])

        truncations = [True]
        # if not interrogation:
        #     truncations.append(False)

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

            exp_results["param"] = deepcopy(param)
            exp_results["truncate"] = truncate
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
            "-c", "--complexity",
            help="Complexity levels to use in evaluation."
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
    ] = False,
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

    print(save_name)

    ctx.obj = {
        "scenario": scenario,
        "model": model,
        "complexity": complexity,
        "interrogation": interrogation,
        "context": context,
        "n_max": n_max,
        "save_name": save_name,
        "features": features,
    }


if __name__ == "__main__":
    app()
