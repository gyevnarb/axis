"""Obtain scores from results for a given scenario and model."""

import logging
import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from generate import app
from plot import plot_actionable_barplot, plot_combined_scores, plot_shapley_waterfall
from util import (
    LLMModels,
    get_actionable_accuracy,
    get_actionable_values,
    get_combined_score,
    get_save_paths,
    get_shapley_values,
    load_eval_results,
)

from envs import axs_igp2

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
            high_complexity = 2
            if param["complexity"] == high_complexity:
                features.add("complexity")
            scores[frozenset(features)] = get_combined_score(eval_dict, kind="correct")

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
        np_scores = np.array(scores)
        mean_score = np.mean(np_scores)
        std_score = np.std(np_scores)
        combined_shapley[feature] = {
            "mean": mean_score,
            "std": std_score / np.sqrt(len(save_paths)),
        }
        logger.info(
            "Feature: %s, Mean: %f, Std: %f, Scores: %s",
            feature,
            mean_score,
            std_score,
            axs_igp2.verbalize.util.ndarray2str(np_scores, 4),
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
            high_complexity = 2
            if param["complexity"] == high_complexity:
                features.add("complexity")
            for explanation_given in ["actionable_exp", "actionable_no_exp"]:
                scores[explanation_given][frozenset(features)] = (
                    get_actionable_accuracy(eval_dict[explanation_given])
                )

        actionable_results[f"{sid}_{eval_m_str}_{gen_m_str}"] = get_actionable_values(
            scores,
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

    plot_actionable_barplot(combined_actionable, ctx)


@app.command()
def evolution(
    ctx: typer.Context,
    eval_model: Annotated[
        LLMModels | None,
        typer.Option("-e", "--eval-model", help="The model used for evaluation."),
    ] = "all",
    features: Annotated[bool, typer.Option("--features", is_eager=True)] = False,
) -> None:
    """Plot the evolution of the combined score for a given scenario and model."""
    ctx.obj["eval_model"] = eval_model
    save_paths = get_save_paths(ctx, features=features)
    if not save_paths:
        return

    scores = []
    for save_path in reversed(save_paths):
        eval_results, (sid, eval_m_str, gen_m_str) = load_eval_results(ctx, save_path)
        if eval_results is None:
            continue
        for result in eval_results:
            new_scores = {}
            combined_score = get_combined_score(result, kind="combined")
            new_scores["score"] = combined_score
            new_scores["scenario"] = sid
            new_scores["model"] = gen_m_str
            scores.append(new_scores)
    plot_combined_scores(scores, ctx)


if __name__ == "__main__":
    app()
