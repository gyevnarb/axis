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
from scipy.stats import mannwhitneyu
from util import (
    MODEL_NAME_MAP,
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
    ] = "results.csv",
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
    baseline_path: Annotated[
        str,
        typer.Option(
            "--baseline",
            "-b",
            help="Path to a CSV file containing baseline scores to draw as horizontal lines",
        ),
    ] = "baseline.csv",
) -> None:
    """Plot the evolution of combined scores across explanation indices from a CSV file.

    This command visualizes how scores change between different explanation indices,
    allowing you to track performance improvements or variations.

    You can filter by scenario ID, generation model, and evaluation model.

    Use --all-scores to visualize all score types (correctness, fluency, actionability)
    as separate lines instead of just the combined score.

    Use --baseline to provide a CSV file with baseline scores that will be drawn as
    horizontal reference lines on the plot.
    """
    logger.info(
        "Plotting score evolution from %s (scenario=%s, gen_model=%s, eval_model=%s, all_scores=%s, baseline=%s)",  # noqa: E501
        csv_path,
        scenario_id,
        gen_model,
        eval_model,
        show_all_scores,
        baseline_path,
    )
    plot_evolution_from_csv(
        csv_path,
        scenario_id,
        gen_model,
        eval_model,
        aggregate_all,
        show_all_scores,
        baseline_path,
    )


@app.command()
def llm_table(
    csv_path: Annotated[
        str,
        typer.Option(
            "--csv",
            "-c",
            help="Path to the CSV file containing evaluation results",
        ),
    ] = "results.csv",
    baseline_path: Annotated[
        str,
        typer.Option(
            "--baseline",
            "-b",
            help="Path to a CSV file containing baseline scores",
        ),
    ] = "baseline.csv",
    output_path: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Path to save the generated LaTeX table",
        ),
    ] = "neurips_table.tex",
    caption: Annotated[
        str,
        typer.Option(
            "--caption",
            help="Caption for the LaTeX table",
        ),
    ] = "Evaluation results comparing different models and interaction methods.",
    label: Annotated[
        str,
        typer.Option(
            "--label",
            help="Label for the LaTeX table",
        ),
    ] = "tab:eval_results",
) -> None:
    """Generate a LaTeX table from evaluation results in NeurIPS single-column format.

    This command processes the results and baseline CSV files to create
    a LaTeX table that aggregates results over scenarios, comparing
    different models and interaction conditions.

    The table includes metrics for preference score, correctness,
    goal accuracy, and action accuracy.
    """
    logger.info(
        "Generating NeurIPS-style LaTeX table from %s with baseline from %s",
        csv_path,
        baseline_path,
    )

    # Load CSV files
    try:
        results_df = pd.read_csv(csv_path)
        baseline_df = pd.read_csv(baseline_path)
        logger.info("Successfully loaded results and baseline data")
    except Exception as e:
        logger.exception("Failed to load CSV data: %s", e)
        return

    # Initialize table rows
    table_rows = []

    # Calculate mean and SEM for goal accuracy
    noexp_goal_values = results_df[results_df["score_type"] == "actionable_no_exp"][
        "actionable_no_exp_goal"
    ]
    noexp_goal_mean = noexp_goal_values.mean()
    noexp_goal_sem = (
        noexp_goal_values.std() / np.sqrt(len(noexp_goal_values))
        if len(noexp_goal_values) > 0
        else 0
    )

    # Calculate mean and SEM for action accuracy
    noexp_action_values = results_df[results_df["score_type"] == "actionable_no_exp"][
        "actionable_no_exp_maneuver"
    ]
    noexp_action_mean = noexp_action_values.mean()
    noexp_action_sem = (
        noexp_action_values.std() / np.sqrt(len(noexp_action_values))
        if len(noexp_action_values) > 0
        else 0
    )

    table_rows.append(
        f"\\textit{{NoExp}} & --- & "
        f"--- & "
        f"{noexp_goal_mean:.2f}$\\pm${noexp_goal_sem:.2f} & "
        f"{noexp_action_mean:.2f}$\\pm${noexp_action_sem:.2f} \\\\"
    )

    # Process model-specific rows
    models_to_process = ["gpt41", "deepseekv3", "llama70b", "o4mini", "deepseekr1"]

    for model_key in models_to_process:
        model_name = MODEL_NAME_MAP.get(model_key, model_key)

        # Base model results (from baseline - no interrogation)
        base_data = baseline_df[baseline_df["gen_llm"] == model_key]

        # Calculate mean and SEM for combined scores
        base_combined_values = base_data[base_data["score_type"] == "combined"][
            "combined_score"
        ]
        base_combined_mean = base_combined_values.mean()
        base_combined_sem = (
            base_combined_values.std() / np.sqrt(len(base_combined_values))
            if len(base_combined_values) > 0
            else 0
        )

        # Calculate mean and SEM for fluent scores
        base_fluent_values = base_data[base_data["score_type"] == "fluent"][
            "fluent_score"
        ]
        base_fluent_mean = base_fluent_values.mean()
        base_fluent_sem = (
            base_fluent_values.std() / np.sqrt(len(base_fluent_values))
            if len(base_fluent_values) > 0
            else 0
        )

        # Calculate mean and SEM for correct scores
        base_correct_values = base_data[base_data["score_type"] == "correct"][
            "correct_score"
        ]
        base_correct_mean = base_correct_values.mean()
        base_correct_sem = (
            base_correct_values.std() / np.sqrt(len(base_correct_values))
            if len(base_correct_values) > 0
            else 0
        )

        # Calculate mean and SEM for goal accuracy
        base_goal_values = base_data[base_data["score_type"] == "actionable_exp"][
            "actionable_exp_goal"
        ]
        base_goal_mean = base_goal_values.mean()
        base_goal_sem = (
            base_goal_values.std() / np.sqrt(len(base_goal_values))
            if len(base_goal_values) > 0
            else 0
        )

        # Calculate mean and SEM for action accuracy
        base_action_values = base_data[base_data["score_type"] == "actionable_exp"][
            "actionable_exp_maneuver"
        ]
        base_action_mean = base_action_values.mean()
        base_action_sem = (
            base_action_values.std() / np.sqrt(len(base_action_values))
            if len(base_action_values) > 0
            else 0
        )

        table_rows.append(
            f"\\textit{{{model_name}}} & "
            # f"{base_combined_mean:.2f}$\\pm${base_combined_sem:.2f} & "
            f"{base_fluent_mean:.2f}$\\pm${base_fluent_sem:.2f} & "
            f"{base_correct_mean:.2f}$\\pm${base_correct_sem:.2f} & "
            f"{base_goal_mean:.2f}$\\pm${base_goal_sem:.2f} & "
            f"{base_action_mean:.2f}$\\pm${base_action_sem:.2f} \\\\"
        )

        # AXIS results (with interrogation)
        axs_data = results_df[results_df["gen_llm"] == model_key]

        # Aggregate over the best explanation index (usually the last one)
        axs_best_index = (
            axs_data.sort_values("combined_score", ascending=False)
            .groupby(["scenario_id", "result_id"])
            .max()
        )["explanation_idx"]

        axs_data_best = axs_data.merge(
            axs_best_index.reset_index(),
            on=["scenario_id", "result_id", "explanation_idx"],
            how="inner",
        )

        # Calculate mean and SEM for combined scores
        axs_combined_values = axs_data_best[axs_data_best["score_type"] == "combined"][
            "combined_score"
        ]
        axs_combined_mean = axs_combined_values.mean()
        axs_combined_sem = (
            axs_combined_values.std() / np.sqrt(len(axs_combined_values))
            if len(axs_combined_values) > 0
            else 0
        )

        # Calculate mean and SEM for fluent scores
        axs_fluent_values = axs_data_best[axs_data_best["score_type"] == "fluent"][
            "fluent_score"
        ]
        axs_fluent_mean = axs_fluent_values.mean()
        axs_fluent_sem = (
            axs_fluent_values.std() / np.sqrt(len(axs_fluent_values))
            if len(axs_fluent_values) > 0
            else 0
        )

        # Calculate mean and SEM for correct scores
        axs_correct_values = axs_data_best[axs_data_best["score_type"] == "correct"][
            "correct_score"
        ]
        axs_correct_mean = axs_correct_values.mean()
        axs_correct_sem = (
            axs_correct_values.std() / np.sqrt(len(axs_correct_values))
            if len(axs_correct_values) > 0
            else 0
        )

        # Calculate mean and SEM for goal accuracy
        axs_goal_values = axs_data_best[
            axs_data_best["score_type"] == "actionable_exp"
        ]["actionable_exp_goal"]
        axs_goal_mean = axs_goal_values.mean()
        axs_goal_sem = (
            axs_goal_values.std() / np.sqrt(len(axs_goal_values))
            if len(axs_goal_values) > 0
            else 0
        )

        # Calculate mean and SEM for action accuracy
        axs_action_values = axs_data_best[
            axs_data_best["score_type"] == "actionable_exp"
        ]["actionable_exp_maneuver"]
        axs_action_mean = axs_action_values.mean()
        axs_action_sem = (
            axs_action_values.std() / np.sqrt(len(axs_action_values))
            if len(axs_action_values) > 0
            else 0
        )

        def significance_stars(p_value: float) -> str:
            """Convert p-value to significance stars."""
            if p_value > 0.05:
                return ""
            if p_value > 0.01:
                return "*"
            if p_value > 0.001:
                return "**"
            return "***"

        # Calculate the deltas (improvements) - compact format for NeurIPS
        if "base_combined_mean" in locals():
            delta_combined = axs_combined_mean - base_combined_mean
            delta_symbol_combined = "+" if delta_combined > 0 else ""
            delta_combined_str = f"{delta_symbol_combined}{delta_combined:.2f}"
        else:
            delta_combined_str = "-"

        if "base_fluent_mean" in locals():
            delta_fluent = axs_fluent_mean - base_fluent_mean
            delta_symbol_fluent = "+" if delta_fluent > 0 else ""
            delta_fluent_str = f"{delta_symbol_fluent}{delta_fluent:.2f}"
            delta_fluent_utest = mannwhitneyu(
                axs_fluent_values,
                base_fluent_values,
                alternative="greater",
            )
            delta_fluent_stars = significance_stars(
                delta_fluent_utest.pvalue,
            )
        else:
            delta_fluent_str = "-"

        if "base_correct_mean" in locals():
            delta_correct = axs_correct_mean - base_correct_mean
            delta_symbol_correct = "+" if delta_correct > 0 else ""
            delta_correct_str = f"{delta_symbol_correct}{delta_correct:.2f}"
            delta_correct_utest = mannwhitneyu(
                axs_correct_values,
                base_correct_values,
                alternative="greater",
            )
            delta_correct_stars = significance_stars(
                delta_correct_utest.pvalue,
            )
        else:
            delta_correct_str = "-"

        if "base_goal_mean" in locals():
            delta_goal = axs_goal_mean - base_goal_mean
            delta_symbol_goal = "+" if delta_goal > 0 else ""
            delta_goal_str = f"{delta_symbol_goal}{delta_goal:.2f}"
            delta_goal_utest = mannwhitneyu(
                axs_goal_values,
                base_goal_values,
                alternative="greater",
            )
            delta_goal_stars = significance_stars(
                delta_goal_utest.pvalue,
            )
        else:
            delta_goal_str = "-"

        if "base_action_mean" in locals():
            delta_action = axs_action_mean - base_action_mean
            delta_symbol_action = "+" if delta_action > 0 else ""
            delta_action_str = f"{delta_symbol_action}{delta_action:.2f}"
            delta_action_utest = mannwhitneyu(
                axs_action_values,
                base_action_values,
                alternative="greater",
            )
            delta_action_stars = significance_stars(
                delta_action_utest.pvalue,
            )
        else:
            delta_action_str = "-"

        table_rows.append(
            f"$+$AXIS & "
            # f"{axs_combined_mean:.2f}$^{{{delta_combined_str}}}\\pm${axs_combined_sem:.2f} & "
            f"{axs_fluent_mean:.2f}$\\pm${axs_fluent_sem:.2f}$^{{{delta_fluent_stars}}}_{{{delta_fluent_str}}}$ & "
            f"{axs_correct_mean:.2f}$\\pm${axs_correct_sem:.2f}$^{{{delta_correct_stars}}}_{{{delta_correct_str}}}$ & "
            f"{axs_goal_mean:.2f}$\\pm${axs_goal_sem:.2f}$^{{{delta_goal_stars}}}_{{{delta_goal_str}}}$ & "
            f"{axs_action_mean:.2f}$\\pm${axs_action_sem:.2f}$^{{{delta_action_stars}}}_{{{delta_action_str}}}$ \\\\"
        )

    # Create the LaTeX table with NeurIPS single-column formatting
    caption += " Values after $\\pm$ denotes standard error of the mean. Superscript values show change relative to base model."
    latex_table = f"""\\begin{{table}}[t]
\\caption{{{caption}}}
\\label{{{label}}}
\\centering
\\setlength{{\\tabcolsep}}{{4pt}}  % Reduce horizontal spacing
\\begin{{tabular}}{{lllll}}
\\toprule
\\textbf{{Model}} & \\textbf{{Preference}} & \\textbf{{Correctness}} & \\textbf{{Goal Acc.}} & \\textbf{{Action Acc.}} \\\\
\\midrule
{table_rows[0]}
\\midrule
{table_rows[1]}
{table_rows[2]}
\\addlinespace[0.5ex]
{table_rows[3]}
{table_rows[4]}
\\addlinespace[0.5ex]
{table_rows[5]}
{table_rows[6]}
\\midrule
{table_rows[7]}
{table_rows[8]}
\\addlinespace[0.5ex]
{table_rows[9]}
{table_rows[10]}
\\bottomrule
\\end{{tabular}}
\\vspace{{-1ex}}
\\end{{table}}
"""  # noqa: E501

    with Path(output_path).open("w") as f:
        f.write(latex_table)
    logger.info("NeurIPS-style LaTeX table saved to %s", output_path)

    # Also print the table to the console
    logger.info("\nGenerated NeurIPS-style LaTeX table:")
    logger.info(latex_table)


@app.command()
def scenario_table(
    csv_path: Annotated[
        str,
        typer.Option(
            "--csv",
            "-c",
            help="Path to the CSV file containing evaluation results",
        ),
    ] = "results.csv",
    output_path: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Path to save the generated LaTeX table",
        ),
    ] = "scenario_table.tex",
    caption: Annotated[
        str,
        typer.Option(
            "--caption",
            help="Caption for the LaTeX table",
        ),
    ] = "Evaluation results by scenario across all models and explanation indexes.",
    label: Annotated[
        str,
        typer.Option(
            "--label",
            help="Label for the LaTeX table",
        ),
    ] = "tab:scenario_results",
) -> None:
    """Generate a LaTeX table showing evaluation scores aggregated by scenario.

    This command processes the results CSV file to create a LaTeX table that
    aggregates results by scenario across all models and explanation indexes.

    The table includes metrics for preference score, correctness,
    goal accuracy, and action accuracy for each scenario.
    """
    logger.info(
        "Generating scenario-aggregated LaTeX table from %s",
        csv_path,
    )

    # Load CSV file
    try:
        results_df = pd.read_csv(csv_path)
        logger.info("Successfully loaded results data")
    except Exception:
        logger.exception("Failed to load CSV data")
        return

    # Initialize table rows
    table_rows = []

    # Get all unique scenarios
    scenarios = sorted(results_df["scenario_id"].unique())

    # Get best explanation for each scenario/result_id pair
    best_indices = (
        results_df.sort_values("combined_score", ascending=False)
        .groupby(["scenario_id", "result_id"])
        .head(1)
    )

    # Aggregate over the best explanation index (usually the last one)
    # results_best_index = (
    #     results_df.sort_values("combined_score", ascending=False)
    #     .groupby(["scenario_id", "result_id"])
    #     .max()
    # )["explanation_idx"]

    # results_best_data = results_df.merge(
    #     results_best_index.reset_index(),
    #     on=["scenario_id", "result_id", "explanation_idx"],
    #     how="inner",
    # )

    # Process each scenario
    for scenario_id in scenarios:
        scenario_data = results_df[
            results_df["scenario_id"] == scenario_id
        ]

        # Calculate mean and SEM for fluent scores
        fluent_values = scenario_data[scenario_data["score_type"] == "fluent"][
            "fluent_score"
        ]
        fluent_mean = fluent_values.mean()
        fluent_sem = (
            fluent_values.std() / np.sqrt(len(fluent_values))
            if len(fluent_values) > 0
            else 0
        )

        # Calculate mean and SEM for correct scores
        correct_values = scenario_data[scenario_data["score_type"] == "correct"][
            "correct_score"
        ]
        correct_mean = correct_values.mean()
        correct_sem = (
            correct_values.std() / np.sqrt(len(correct_values))
            if len(correct_values) > 0
            else 0
        )

        # Calculate mean and SEM for goal accuracy
        goal_values = scenario_data[scenario_data["score_type"] == "actionable_exp"][
            "actionable_exp_goal"
        ]
        goal_mean = goal_values.mean()
        goal_sem = (
            goal_values.std() / np.sqrt(len(goal_values)) if len(goal_values) > 0 else 0
        )

        # Calculate mean and SEM for action accuracy
        action_values = scenario_data[scenario_data["score_type"] == "actionable_exp"][
            "actionable_exp_maneuver"
        ]
        action_mean = action_values.mean()
        action_sem = (
            action_values.std() / np.sqrt(len(action_values))
            if len(action_values) > 0
            else 0
        )

        # Add row to table
        table_rows.append(
            f"{scenario_id} & "
            f"{fluent_mean:.2f}$\\pm${fluent_sem:.2f} & "
            f"{correct_mean:.2f}$\\pm${correct_sem:.2f} & "
            f"{goal_mean:.2f}$\\pm${goal_sem:.2f} & "
            f"{action_mean:.2f}$\\pm${action_sem:.2f} \\\\",
        )

    # Create the LaTeX table
    latex_table = f"""\\begin{{table}}[t]
\\caption{{{caption}}}
\\label{{{label}}}
\\centering
\\begin{{tabular}}{{clrrrr}}
\\toprule
\\textbf{{ID}} & \\textbf{{Preference}} & \\textbf{{Correctness}} & \\textbf{{Goal Acc.}} & \\textbf{{Action Acc.}} \\\\
\\midrule
{chr(10).join(table_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    with Path(output_path).open("w") as f:
        f.write(latex_table)
    logger.info("Scenario-aggregated LaTeX table saved to %s", output_path)

    # Also print the table to the console
    logger.info("\nGenerated scenario-aggregated LaTeX table:")
    logger.info(latex_table)


if __name__ == "__main__":
    app()
