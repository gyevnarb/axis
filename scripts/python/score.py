"""Obtain scores from results for a given scenario and model."""

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from generate import app  # Fixed import path
from plot import (
    plot_actionable_barplot,
    plot_evolution_from_csv,
    plot_shapley_waterfall,
)
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
    csv_path: Annotated[
        str,
        typer.Option(
            "--csv",
            "-c",
            help="Path to the CSV file containing evaluation results",
        ),
    ],
    scenario_id: Annotated[
        int,
        typer.Option(
            "--scenario",
            "-s",
            help="Filter by scenario ID (-1 for all scenarios)",
        ),
    ] = -1,
    gen_model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Filter by generation model ('all' for all models)",
        ),
    ] = "all",
    eval_model: Annotated[
        str,
        typer.Option(
            "--eval-model",
            "-e",
            help="Filter by evaluation model ('all' for all models)",
        ),
    ] = "all",
    aggregate_all: Annotated[
        bool,
        typer.Option(
            "--aggregate-all",
            "-a",
            help="Aggregate all scores across explanation indices",
        ),
    ] = False,
    show_all_scores: Annotated[
        bool,
        typer.Option(
            "--all-scores",
            "-as",
            help="Show score types as separate lines",
        ),
    ] = False,
) -> None:
    """Plot the evolution of combined scores across explanation indices from a CSV file.

    This command visualizes how scores change between different explanation indices,
    allowing you to track performance improvements or variations.

    You can filter by scenario ID, generation model, and evaluation model.

    Use --all-scores to visualize all score types (correctness, fluency, actionability)
    as separate lines instead of just the combined score.
    """
    logger.info(
        "Plotting score evolution from %s (scenario=%s, gen_model=%s, eval_model=%s, all_scores=%s)",  # noqa: E501
        csv_path,
        scenario_id,
        gen_model,
        eval_model,
        show_all_scores,
    )
    plot_evolution_from_csv(
        csv_path, scenario_id, gen_model, eval_model, aggregate_all, show_all_scores,
    )


@app.command()
def latex_table(
    csv_path: Annotated[
        str,
        typer.Option(
            "--csv",
            "-c",
            help="Path to the CSV file containing evaluation results",
        ),
    ],
    output_path: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Path to save the LaTeX table",
        ),
    ] = "combined_scores_table.tex",
    eval_model: Annotated[
        str,
        typer.Option(
            "--eval-model",
            "-e",
            help="Filter by evaluation model ('all' for all models)",
        ),
    ] = "all",
    explanation_idx: Annotated[
        int,
        typer.Option(
            "--explanation-idx",
            "-i",
            help="Filter by explanation index (default: -1 for all)",
        ),
    ] = -1,
) -> None:
    """Create a LaTeX table with combined evaluation scores for each model and scenario.

    This function generates a publication-ready LaTeX table showing the combined scores
    for each model across all scenarios. The table is formatted with proper statistical
    notation including standard errors.

    Args:
        csv_path: Path to the CSV file containing evaluation results
        output_path: Path to save the generated LaTeX table
        eval_model: Evaluation model to filter by ('all' for all models)
        explanation_idx: Explanation index to filter by (-1 for all)

    """
    logger.info(
        "Generating LaTeX table from %s (eval_model=%s, explanation_idx=%s)",
        csv_path,
        eval_model,
        explanation_idx,
    )

    # Load the CSV data
    df_results = pd.read_csv(csv_path)

    # Apply filters
    if eval_model != "all":
        df_results = df_results[df_results["eval_llm"] == eval_model]

    if explanation_idx >= 0:
        df_results = df_results[df_results["explanation_idx"] == explanation_idx]

    # Filter to only include combined scores
    df_combined = df_results[df_results["score_type"] == "combined"]

    if df_combined.empty:
        logger.error("No combined score data matches the specified filters")
        return

    # Calculate mean scores and standard errors for each model-scenario combination
    pivot_table = pd.pivot_table(
        df_combined,
        values="combined_score",
        index="gen_llm",
        columns="scenario_id",
        aggfunc=["mean", "std", "count"],
    )

    # Get list of unique scenarios and models
    scenarios = sorted(df_combined["scenario_id"].unique())
    models = sorted(df_combined["gen_llm"].unique())
    
    # Calculate model averages across all scenarios
    model_averages = {}
    model_data = {}  # Store scores for each model/scenario
    
    for model in models:
        model_scores = []
        model_data[model] = {}
        
        for scenario in scenarios:
            try:
                # Access pivot table correctly with 2-level MultiIndex
                mean = pivot_table["mean"][scenario].loc[model]
                std = pivot_table["std"][scenario].loc[model]
                count = pivot_table["count"][scenario].loc[model]

                # Calculate standard error and store the formatted string
                se = std / np.sqrt(count)
                model_scores.append(mean)
                model_data[model][scenario] = f"{mean:.2f} $\\pm$ {se:.2f}"
            except (KeyError, ValueError):
                model_data[model][scenario] = "--"
                
        # Calculate average across scenarios for this model
        if model_scores:
            avg = np.mean(model_scores)
            model_averages[model] = avg
        else:
            model_averages[model] = float('nan')
    
    # Split scenarios into two equal (or nearly equal) groups
    mid_idx = len(scenarios) // 2
    scenarios_first = scenarios[:mid_idx]
    scenarios_second = scenarios[mid_idx:]
    
    # Create LaTeX content
    latex_content = []
    
    # First table with first half of scenarios
    latex_content.append(r"\begin{table}[ht]")
    latex_content.append(r"\centering")
    latex_content.append(r"\caption{Combined Evaluation Scores by Model and Scenario (Part 1)}")
    latex_content.append(r"\label{tab:combined_scores_part1}")
    
    # Create table format
    col_fmt = "l" + "c" * len(scenarios_first)  # Left-align model names, center scores
    latex_content.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    
    # Table header with scenario numbers
    header_row = ["Model"] + [f"S{s}" for s in scenarios_first]
    latex_content.append(" & ".join(header_row) + r" \\")
    latex_content.append(r"\midrule")
    
    # Add data rows
    for model in models:
        row = [f"{model}"]
        for scenario in scenarios_first:
            row.append(model_data[model][scenario])
        latex_content.append(" & ".join(row) + r" \\")
    
    # End first table
    latex_content.append(r"\bottomrule")
    latex_content.append(r"\end{tabular}")
    latex_content.append(r"\end{table}")
    
    # Second table with second half of scenarios and averages
    latex_content.append(r"\begin{table}[ht]")
    latex_content.append(r"\centering")
    latex_content.append(r"\caption{Combined Evaluation Scores by Model and Scenario (Part 2)}")
    latex_content.append(r"\label{tab:combined_scores_part2}")
    
    # Create table format
    col_fmt = "l" + "c" * len(scenarios_second) + "c"  # Include average column
    latex_content.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    
    # Table header with scenario numbers and average
    header_row = ["Model"] + [f"S{s}" for s in scenarios_second] + ["Avg"]
    latex_content.append(" & ".join(header_row) + r" \\")
    latex_content.append(r"\midrule")
    
    # Add data rows
    for model in models:
        row = [f"{model}"]
        for scenario in scenarios_second:
            row.append(model_data[model][scenario])
            
        # Add average column
        if not np.isnan(model_averages[model]):
            row.append(f"{model_averages[model]:.2f}")
        else:
            row.append("--")
            
        latex_content.append(" & ".join(row) + r" \\")
    
    # End second table
    latex_content.append(r"\bottomrule")
    latex_content.append(r"\end{tabular}")
    latex_content.append(r"\end{table}")

    # Join all lines and save to file
    latex_table = "\n".join(latex_content)

    with Path(output_path).open("w") as f:
        f.write(latex_table)

    logger.info("LaTeX table saved to %s", output_path)

    # Print best performing model
    if model_averages:
        best_model = max(model_averages.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('-inf'))
        if not np.isnan(best_model[1]):
            logger.info(
                "Best performing model: %s with average score %.3f",
                best_model[0],
                best_model[1],
            )


if __name__ == "__main__":
    app()
