"""Analysis utilities for AXS evaluation results."""

import logging
import pickle
import re
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from util import LLMModels, get_combined_score

import axs

app = typer.Typer()
logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> dict:
    """Parse the filename to extract evaluation details."""
    rex = r"evaluate_(?P<eval_llm>[^_]+)_(?P<gen_llm>[^_]+)(?:_(?P<features>features))?(?:_(?P<interrogation>interrogation))?(?:_(?P<context>context))?\.pkl"  # noqa: E501
    match = re.match(rex, filename)
    if match:
        return {
            "eval_llm": match.group("eval_llm"),
            "gen_llm": match.group("gen_llm"),
            "features": bool(match.group("features")),
            "interrogation": bool(match.group("interrogation")),
            "context": bool(match.group("context")),
        }
    # Handle non-evaluate file names
    gen_llm_match = re.match(r"(?P<gen_llm>[^_]+)_.*\.pkl", filename)
    return {
        "eval_llm": None,
        "gen_llm": gen_llm_match.group("gen_llm") if gen_llm_match else None,
        "features": False,
        "interrogation": False,
        "context": False,
    }


def extract_fluent_scores(fluent_data: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Extract fluency scores from the evaluation results for all explanations."""
    scores_list = []

    for idx, fluent_item in enumerate(fluent_data):
        scores = fluent_item.get("scores", {})

        all_scores = np.array(list(scores.values()))

        score_data = {
            "explanation_idx": idx,  # This will be the only explanation_idx
            "sufficient_detail": scores.get("SufficientDetail", 0),
            "satisfying": scores.get("Satisfying", 0),
            "complete": scores.get("Complete", 0),
            "trust": scores.get("Trust", 0),
            "score": np.exp(np.log(all_scores).mean()),
        }
        scores_list.append(score_data)

    return scores_list


def extract_correct_scores(
    correct_data: list[dict[str, Any]],
    exclude_last: bool = False,
) -> list[dict[str, float]]:
    """Extract correctness scores from the evaluation results for all explanations.

    Args:
        correct_data: List of dictionaries containing correctness evaluation data
        exclude_last: Whether to exclude the final explanation from the extraction

    Returns:
        List of dictionaries containing extracted scores for each explanation

    """
    scores_list = []

    # If exclude_last is True and we have data, exclude the last item
    data_to_process = (
        correct_data[:-1] if exclude_last and correct_data else correct_data
    )

    for idx, correct_item in enumerate(data_to_process):
        scores = correct_item.get("scores", {})

        score_data = {
            "explanation_idx": idx,  # This will be the only explanation_idx
            "score": scores.get("Correct", -1),
        }
        scores_list.append(score_data)

    return scores_list


def extract_actionable_scores(
    actionable_data: list[dict[str, Any]],
) -> list[dict[str, int]]:
    """Extract actionability scores from the evaluation results for all explanations."""
    scores_list = []

    for idx, actionable_item in enumerate(actionable_data):
        scores = actionable_item.get("scores", {})

        goal_score = int(scores.get("Goal", -1) == 0)
        maneuver_score = int(scores.get("Maneuver", -1) == 0)

        score_data = {
            "explanation_idx": idx,  # This will be the only explanation_idx
            "goal": goal_score,
            "maneuver": maneuver_score,
            "score": (goal_score + maneuver_score) / 2,
        }
        scores_list.append(score_data)

    return scores_list


def load_results_to_dataframe(  # noqa: PLR0913
    eval_model: LLMModels | str = "claude35",
    gen_model: LLMModels | str = "all",
    scenario: int = -1,
    features: bool | None = None,
    interrogation: bool | None = None,
    context: bool | None = None,
    exclude_last: bool = False,
) -> pd.DataFrame:
    """Load all evaluation results into a pandas dataframe.

    Args:
        eval_model: The evaluation model to filter by
        gen_model: The generation model to filter by
        scenario: The scenario ID to filter by (-1 for all)
        features: Whether to filter by feature evaluation
        interrogation: Whether to filter by interrogation
        context: Whether to filter by context
        exclude_last: Whether to exclude final explanation from correctness extraction

    Returns:
        A pandas dataframe with all results

    """
    base_dir = Path("output", "igp2")

    # Find all scenario directories
    if scenario != -1:
        scenario_dirs = [base_dir / f"scenario{scenario}" / "results"]
    else:
        scenario_dirs = list(base_dir.glob("scenario*/results"))

    # Sort by scenario number
    scenario_dirs.sort(key=lambda x: int(x.parent.name.replace("scenario", "")))

    all_data = []

    logger.info("Found %d scenario directories", len(scenario_dirs))

    for scenario_dir in scenario_dirs:
        scenario_id = int(scenario_dir.parent.name.replace("scenario", ""))

        # Find all evaluation result files
        result_files = list(scenario_dir.glob("evaluate_*.pkl"))

        logger.info(
            "Found %d result files in scenario %d",
            len(result_files),
            scenario_id,
        )

        for result_file in result_files:
            file_info = parse_filename(result_file.name)

            # Apply filters if needed
            if eval_model != "all" and file_info["eval_llm"] != eval_model:
                continue
            if gen_model != "all" and file_info["gen_llm"] != gen_model:
                continue
            if features is not None and file_info["features"] != features:
                continue
            if (
                interrogation is not None
                and file_info["interrogation"] != interrogation
            ):
                continue
            if context is not None and file_info["context"] != context:
                continue

            # Load the evaluation results
            try:
                with result_file.open("rb") as f:
                    eval_results = pickle.load(f)

                logger.info("Loaded %d results from %s", len(eval_results), result_file)

                # Process each evaluation result
                for result_id, result in enumerate(eval_results, 1):
                    # Extract parameter values
                    param = result.get("param", {})

                    # Extract scores
                    fluent_scores_list = extract_fluent_scores(result.get("fluent", []))
                    correct_scores_list = extract_correct_scores(
                        result.get("correct", []),
                        exclude_last=exclude_last,
                    )

                    # Extract actionable scores (with and without explanation)
                    actionable_exp_scores_list = extract_actionable_scores(
                        result.get("actionable_exp", []),
                    )
                    actionable_no_exp_scores_list = extract_actionable_scores(
                        result.get("actionable_no_exp", []),
                    )

                    # Calculate combined score an array with scores for each explanation
                    combined_scores = get_combined_score(result, kind="combined")

                    # Make sure combined_scores is a list
                    if not isinstance(combined_scores, list) and not isinstance(
                        combined_scores,
                        np.ndarray,
                    ):
                        combined_scores = [combined_scores]

                    # Convert to list if it's a numpy array
                    if isinstance(combined_scores, np.ndarray):
                        combined_scores = combined_scores.tolist()

                    # Get list of features
                    features_list = list(param.get("verbalizer_features", []))
                    high_complexity = 2
                    if param.get("complexity", 1) == high_complexity:
                        features_list.append("complexity")

                    # Create base data for each row (without combined score)
                    base_data = {
                        "scenario_id": scenario_id,
                        "eval_llm": file_info["eval_llm"],
                        "gen_llm": file_info["gen_llm"],
                        "features_eval": file_info["features"],
                        "interrogation": file_info["interrogation"],
                        "context": file_info["context"],
                        "n_max": param.get("n_max", -1),
                        "complexity": param.get("complexity", -1),
                        "features": ",".join(features_list),
                        "result_id": result_id,
                    }

                    # Create separate rows for combined scores as their own score type
                    for idx, combined_score in enumerate(combined_scores):
                        row_data = {
                            **base_data,
                            "explanation_idx": idx,
                            "score_type": "combined",
                            "combined_score": combined_score,
                        }
                        all_data.append(row_data)

                    # Add rows for fluent scores
                    for fluent_score in fluent_scores_list:
                        explanation_idx = fluent_score.pop("explanation_idx")
                        row_data = {
                            **base_data,
                            "explanation_idx": explanation_idx,
                            "score_type": "fluent",
                            **{f"fluent_{k}": v for k, v in fluent_score.items()},
                        }
                        all_data.append(row_data)

                    # Add rows for correct scores
                    for correct_score in correct_scores_list:
                        explanation_idx = correct_score.pop("explanation_idx")
                        row_data = {
                            **base_data,
                            "explanation_idx": explanation_idx,
                            "score_type": "correct",
                            **{f"correct_{k}": v for k, v in correct_score.items()},
                        }
                        all_data.append(row_data)

                    # Add rows for actionable scores with explanation
                    for actionable_exp_score in actionable_exp_scores_list:
                        explanation_idx = actionable_exp_score.pop("explanation_idx")
                        row_data = {
                            **base_data,
                            "explanation_idx": explanation_idx,
                            "score_type": "actionable_exp",
                            **{
                                f"actionable_exp_{k}": v
                                for k, v in actionable_exp_score.items()
                            },
                        }
                        all_data.append(row_data)

                    # Add rows for actionable scores without explanation
                    for actionable_no_exp_score in actionable_no_exp_scores_list:
                        explanation_idx = actionable_no_exp_score.pop("explanation_idx")
                        row_data = {
                            **base_data,
                            "explanation_idx": -1,  # Special value for no explanation
                            "score_type": "actionable_no_exp",
                            **{
                                f"actionable_no_exp_{k}": v
                                for k, v in actionable_no_exp_score.items()
                            },
                        }
                        all_data.append(row_data)

            except (FileNotFoundError, EOFError) as e:
                logger.exception("Error loading file %s: %s", result_file, e)  # noqa: TRY401
                continue

    # Create dataframe from all data
    df_results = pd.DataFrame(all_data)
    logger.info("Created dataframe with %d rows", len(df_results))

    return df_results


@app.command()
def df(  # noqa: PLR0913
    output_path: Annotated[
        str,
        typer.Option("--output", "-o", help="Path to save the DataFrame as a CSV file"),
    ] = "eval_results.csv",
    eval_model: Annotated[
        LLMModels,
        typer.Option("--eval-model", "-e", help="Evaluation model to filter by"),
    ] = "claude35",
    gen_model: Annotated[
        LLMModels,
        typer.Option("--gen-model", "-g", help="Generation model to filter by"),
    ] = "all",
    scenario: Annotated[
        int,
        typer.Option("--scenario", "-s", help="Scenario ID to filter by (-1 for all)"),
    ] = -1,
    features: Annotated[
        bool | None,
        typer.Option(
            "--features",
            "-f",
            help="Whether to filter by feature evaluation",
        ),
    ] = False,
    interrogation: Annotated[
        bool | None,
        typer.Option(
            "--interrogation",
            "-i",
            help="Whether to filter by interrogation",
        ),
    ] = False,
    context: Annotated[
        bool | None,
        typer.Option("--context", "-c", help="Whether to filter by context"),
    ] = False,
    exclude_last: Annotated[
        bool,
        typer.Option(
            "--exclude-last",
            "-x",
            help="Whether to exclude the final explanation from correctness extraction",
        ),
    ] = False,
) -> pd.DataFrame:
    """Create a dataframe with all evaluation results and save to CSV."""
    eval_model = eval_model.value
    gen_model = gen_model.value

    logger.info(
        "Loading results for eval_model=%s, gen_model=%s, scenario=%s",
        eval_model,
        gen_model,
        scenario,
    )

    df_results = load_results_to_dataframe(
        eval_model=eval_model,
        gen_model=gen_model,
        scenario=scenario,
        features=features,
        interrogation=interrogation,
        context=context,
        exclude_last=exclude_last,
    )

    # Save to CSV
    output_file = Path(output_path)
    df_results.to_csv(output_file, index=False)
    logger.info("Saved dataframe to %s", output_file)

    # Print summary statistics
    console = Console()

    # Print dataframe info
    console.print("[bold green]DataFrame Info:[/bold green]")
    console.print(f"Shape: {df_results.shape[0]} rows x {df_results.shape[1]} columns")

    # Print column names
    console.print("\n[bold green]Columns:[/bold green]")
    for col in df_results.columns:
        console.print(f"- {col}")

    # Print scenario distribution
    if "scenario_id" in df_results.columns:
        console.print("\n[bold green]Scenarios:[/bold green]")
        scenario_counts = df_results["scenario_id"].value_counts().sort_index()
        table = Table("Scenario ID", "Count")
        for scenario_id, count in scenario_counts.items():
            table.add_row(str(scenario_id), str(count))
        console.print(table)

    return df_results


@app.command()
def all_df(  # noqa: PLR0913
    output_path: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Path to save the full DataFrame as a CSV file",
        ),
    ] = "all_eval_results.csv",
    eval_models: Annotated[
        list[str] | None,
        typer.Option(
            "--eval-models",
            "-e",
            help=(
                "List of evaluation models to include "
                "(comma-separated, 'all' for all models)"
            ),
        ),
    ] = None,
    features: Annotated[
        bool | None,
        typer.Option(
            "--features",
            "-f",
            help="Whether to filter by feature evaluation",
        ),
    ] = False,
    interrogation: Annotated[
        bool | None,
        typer.Option(
            "--interrogation",
            "-i",
            help="Whether to filter by interrogation",
        ),
    ] = False,
    context: Annotated[
        bool | None,
        typer.Option(
            "--context",
            "-c",
            help="Whether to filter by context",
        ),
    ] = False,
    exclude_last: Annotated[
        bool,
        typer.Option(
            "--exclude-last",
            "-x",
            help="Whether to exclude the final explanation from correctness extraction",
        ),
    ] = False,
) -> pd.DataFrame:
    """Dataframe with all results from all scenarios and generation models.

    This command automatically finds all available scenarios and generation models,
    and creates a single dataframe containing all evaluation results across all configs.

    You can filter results by specifying whether to include only entries with/without
    features, interrogation, or context using the --features, --interrogation, and
    --context flags.
    """
    if eval_models is None:
        eval_models = ["claude35"]

    console = Console()
    base_dir = Path("output", "igp2")

    # Find all scenario directories
    scenario_dirs = list(base_dir.glob("scenario*/results"))
    scenarios = [
        int(dir_path.parent.name.replace("scenario", "")) for dir_path in scenario_dirs
    ]
    scenarios.sort()

    logger.info("Found %d scenarios: %s", len(scenarios), scenarios)
    console.print(
        f"[bold green]Found {len(scenarios)} scenarios:[/bold green] {scenarios}",
    )

    # Find all unique generation models
    gen_models = set()
    eval_model_set = set()

    # Scan all evaluation files to find unique generation and evaluation models
    for scenario_dir in scenario_dirs:
        for result_file in scenario_dir.glob("evaluate_*.pkl"):
            file_info = parse_filename(result_file.name)
            if file_info["gen_llm"]:
                gen_models.add(file_info["gen_llm"])
            if file_info["eval_llm"]:
                eval_model_set.add(file_info["eval_llm"])

    gen_models = sorted(gen_models)
    eval_models = sorted(eval_model_set)

    logger.info("Found %d generation models: %s", len(gen_models), gen_models)
    logger.info("Found %d evaluation models: %s", len(eval_models), eval_models)

    console.print(
        f"[bold green]Found {len(gen_models)} genmodels:[/bold green] {gen_models}",
    )
    console.print(
        f"[bold green]Found {len(eval_models)} eval models:[/bold green] {eval_models}",
    )

    # Filter evaluation models if specified
    if eval_models != ["all"]:
        filtered_eval_models = [model for model in eval_models if model in eval_models]
        if not filtered_eval_models:
            logger.warning(
                "None of the specified eval models %s were found. Using all models.",
                eval_models,
            )
            filtered_eval_models = eval_models
    else:
        filtered_eval_models = eval_models

    logger.info("Using evaluation models: %s", filtered_eval_models)
    console.print(
        f"[bold green]Using evaluation models:[/bold green] {filtered_eval_models}",
    )

    # Create progress table
    progress_table = Table(title="Loading Progress")
    progress_table.add_column("Scenario", style="cyan")
    progress_table.add_column("Gen Model", style="green")
    progress_table.add_column("Eval Model", style="yellow")
    progress_table.add_column("Status", style="magenta")

    all_dataframes = []
    total_results = 0

    console.print("\n[bold]Starting data collection...[/bold]")

    # Load data for each combination
    for scenario in scenarios:
        for gen_model in gen_models:
            for eval_model in filtered_eval_models:
                try:
                    # Load data for this combination
                    df_results = load_results_to_dataframe(
                        eval_model=eval_model,
                        gen_model=gen_model,
                        scenario=scenario,
                        features=features,
                        interrogation=interrogation,
                        context=context,
                        exclude_last=exclude_last,
                    )

                    if not df_results.empty:
                        all_dataframes.append(df_results)
                        total_results += len(df_results)
                        status = f"Loaded {len(df_results)} results"
                    else:
                        status = "No results found"

                except Exception as e:
                    logger.exception(
                        "Error loading scenario %s, gen_model %s, eval_model %s",
                        scenario,
                        gen_model,
                        eval_model,
                    )
                    status = f"Error: {str(e)[:30]}..."

                # Update progress
                progress_table.add_row(str(scenario), gen_model, eval_model, status)

    console.print(progress_table)

    if not all_dataframes:
        console.print(
            "[bold red]No data found for the specified parameters.[/bold red]",
        )
        return None

    # Combine all dataframes
    consolidated_df = pd.concat(all_dataframes, ignore_index=True)
    console.print(
        f"\n[bold green]Created dataframe of {len(consolidated_df)} rows[/bold green]",
    )

    # Save to CSV
    output_file = Path(output_path)
    consolidated_df.to_csv(output_file, index=False)
    console.print(
        f"[bold green]Saved consolidated dataframe to {output_file}[/bold green]",
    )

    # Print summary statistics
    console.print("\n[bold green]Data Summary:[/bold green]")

    # Scenario distribution
    console.print("\n[bold]Distribution by Scenario:[/bold]")
    scenario_counts = consolidated_df["scenario_id"].value_counts().sort_index()
    scenario_table = Table("Scenario ID", "Count", "Percentage")
    for scenario_id, count in scenario_counts.items():
        percentage = (count / len(consolidated_df)) * 100
        scenario_table.add_row(str(scenario_id), str(count), f"{percentage:.1f}%")
    console.print(scenario_table)

    # Generation model distribution
    console.print("\n[bold]Distribution by Generation Model:[/bold]")
    gen_model_counts = consolidated_df["gen_llm"].value_counts().sort_index()
    gen_model_table = Table("Generation Model", "Count", "Percentage")
    for gen_model, count in gen_model_counts.items():
        percentage = (count / len(consolidated_df)) * 100
        gen_model_table.add_row(gen_model, str(count), f"{percentage:.1f}%")
    console.print(gen_model_table)

    # Evaluation model distribution
    console.print("\n[bold]Distribution by Evaluation Model:[/bold]")
    eval_model_counts = consolidated_df["eval_llm"].value_counts().sort_index()
    eval_model_table = Table("Evaluation Model", "Count", "Percentage")
    for eval_model, count in eval_model_counts.items():
        percentage = (count / len(consolidated_df)) * 100
        eval_model_table.add_row(eval_model, str(count), f"{percentage:.1f}%")
    console.print(eval_model_table)

    return consolidated_df


@app.command()
def analyze(
    csv_file: Annotated[
        str,
        typer.Argument(help="Path to the CSV file containing evaluation results"),
    ],
    group_by: Annotated[
        str,
        typer.Option(
            "--group-by",
            "-g",
            help=(
                "Column to group by for analysis "
                "(e.g., scenario_id, gen_llm, features, result_id)"
            ),
        ),
    ] = "scenario_id",
    score_type: Annotated[
        str | None,
        typer.Option(
            "--score-type",
            "-t",
            help=(
                "Filter by score type (combined, fluent, correct, actionable_exp, "
                "actionable_no_exp). Leave empty for all types."
            ),
        ),
    ] = None,
    output_path: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to save analysis results as CSV",
        ),
    ] = None,
    explanation_idx: Annotated[
        int,
        typer.Option(
            "--explanation-idx",
            "-e",
            help="Filter by explanation index (-1 for all)",
        ),
    ] = -1,
) -> pd.DataFrame:
    """Analyze scores from a previously created CSV file with the new format.

    This command provides detailed analysis of evaluation results, including:
    - Score distribution by type
    - Statistical summaries per group
    - Comparison between features and conditions
    - Performance profiles across scenarios
    """
    console = Console()
    df_results = pd.read_csv(csv_file)

    console.print(
        f"[bold green]Loaded dataframe with {len(df_results)} rows[/bold green]",
    )

    # Apply filters if specified
    if score_type:
        df_results = df_results[df_results["score_type"] == score_type]
        console.print(
            f"[bold]Filtered to score_type '{score_type}': {len(df_results)} rows remaining[/bold]",  # noqa: E501
        )

    if explanation_idx >= 0:
        df_results = df_results[df_results["explanation_idx"] == explanation_idx]
        console.print(
            f"[bold]Filtered to explanation_idx {explanation_idx}: {len(df_results)} rows remaining[/bold]",  # noqa: E501
        )

    if df_results.empty:
        console.print("[bold red]No data matches the specified filters.[/bold red]")
        return pd.DataFrame()

    # Overall data summary
    console.print("\n[bold blue]===== OVERALL DATA SUMMARY =====[/bold blue]")

    # Score type distribution
    console.print("\n[bold]Distribution by Score Type:[/bold]")
    type_counts = df_results["score_type"].value_counts().sort_index()
    type_table = Table("Score Type", "Count", "Percentage")
    for type_name, count in type_counts.items():
        percentage = (count / len(df_results)) * 100
        type_table.add_row(str(type_name), str(count), f"{percentage:.1f}%")
    console.print(type_table)

    # Features distribution
    if "features" in df_results.columns:
        # Count occurrences of each feature across all rows
        all_features = []
        for feature_list in df_results["features"].dropna():
            all_features.extend(feature_list.split(","))

        feature_counts = pd.Series(all_features).value_counts().sort_index()

        console.print("\n[bold]Feature Distribution:[/bold]")
        feature_table = Table("Feature", "Count", "Percentage")
        for feature, count in feature_counts.items():
            percentage = (count / len(all_features)) * 100
            feature_table.add_row(feature, str(count), f"{percentage:.1f}%")
        console.print(feature_table)

    # Create groupwise analysis
    console.print(
        f"\n[bold blue]===== ANALYSIS BY {group_by.upper()} =====[/bold blue]",
    )

    # Detect numeric columns based on score type
    score_prefixes = [
        "combined_score",
        "fluent_",
        "correct_",
        "actionable_exp_",
        "actionable_no_exp_",
    ]
    numeric_columns = [
        col
        for col in df_results.columns
        if any(col.startswith(prefix) for prefix in score_prefixes)
    ]

    # For grouping by score_type, we need special handling
    if group_by == "score_type":
        group_results = {}

        for score_type_val in df_results["score_type"].unique():
            subset = df_results[df_results["score_type"] == score_type_val]

            # Get columns relevant to this score type
            relevant_cols = [
                col
                for col in numeric_columns
                if col.startswith(f"{score_type_val}_") or col == "combined_score"
            ]
            if not relevant_cols and score_type_val == "combined":
                relevant_cols = ["combined_score"]

            if relevant_cols:
                stats = (
                    subset[relevant_cols].agg(["mean", "std", "count", "min", "max"]).T
                )
                group_results[score_type_val] = stats

        # Print results for each score type
        for score_type_val, stats in group_results.items():
            console.print(f"\n[bold]Score Type: {score_type_val}[/bold]")

            stats_table = Table("Metric", "Mean", "Std", "Count", "Min", "Max")
            for metric, row in stats.iterrows():
                clean_metric = (
                    metric.replace(f"{score_type_val}_", "")
                    if score_type_val != "combined"
                    else metric
                )
                stats_table.add_row(
                    clean_metric,
                    f"{row['mean']:.3f}",
                    f"{row['std']:.3f}",
                    f"{row['count']:.0f}",
                    f"{row['min']:.3f}",
                    f"{row['max']:.3f}",
                )
            console.print(stats_table)
    else:
        # Regular group-by analysis
        grouped = df_results.groupby([group_by, "score_type"])

        # For each group, compute statistics on relevant numeric columns
        group_results = {}
        for (group_val, score_type_val), group_df in grouped:
            if group_val not in group_results:
                group_results[group_val] = {}

            # Get columns relevant to this score type
            relevant_cols = [
                col
                for col in numeric_columns
                if col.startswith(f"{score_type_val}_") or col == "combined_score"
            ]
            if not relevant_cols and score_type_val == "combined":
                relevant_cols = ["combined_score"]

            if relevant_cols:
                stats = (
                    group_df[relevant_cols]
                    .agg(["mean", "std", "count", "min", "max"])
                    .T
                )
                group_results[group_val][score_type_val] = stats

        # Print results for each group
        for group_val in sorted(group_results.keys()):
            console.print(f"\n[bold]{group_by}: {group_val}[/bold]")

            for score_type_val, stats in group_results[group_val].items():
                console.print(f"\n[cyan]Score Type: {score_type_val}[/cyan]")

                stats_table = Table("Metric", "Mean", "Std", "Count", "Min", "Max")
                for metric, row in stats.iterrows():
                    clean_metric = (
                        metric.replace(f"{score_type_val}_", "")
                        if score_type_val != "combined"
                        else metric
                    )
                    stats_table.add_row(
                        clean_metric,
                        f"{row['mean']:.3f}",
                        f"{row['std']:.3f}",
                        f"{row['count']:.0f}",
                        f"{row['min']:.3f}",
                        f"{row['max']:.3f}",
                    )
                console.print(stats_table)

    # Feature impact analysis
    if "features" in df_results.columns and "combined_score" in df_results.columns:
        console.print("\n[bold blue]===== FEATURE IMPACT ANALYSIS =====[/bold blue]")

        # Extract all unique features
        unique_features = set()
        for feature_list in df_results["features"].dropna():
            unique_features.update(feature_list.split(","))

        # Create binary columns for each feature
        for feature in unique_features:
            df_results[f"has_{feature}"] = df_results["features"].apply(
                lambda x, feature=feature: 1 if feature in str(x).split(",") else 0,
            )

        # Analyze impact of features on combined scores
        combined_df = df_results[df_results["score_type"] == "combined"]

        if not combined_df.empty:
            feature_impact = []
            for feature in unique_features:
                with_feature = combined_df[combined_df[f"has_{feature}"] == 1][
                    "combined_score"
                ]
                without_feature = combined_df[combined_df[f"has_{feature}"] == 0][
                    "combined_score"
                ]

                if len(with_feature) > 0 and len(without_feature) > 0:
                    impact = {
                        "Feature": feature,
                        "With_Mean": with_feature.mean(),
                        "Without_Mean": without_feature.mean(),
                        "Difference": with_feature.mean() - without_feature.mean(),
                        "With_Count": len(with_feature),
                        "Without_Count": len(without_feature),
                    }
                    feature_impact.append(impact)

            if feature_impact:
                impact_df = pd.DataFrame(feature_impact).sort_values(
                    "Difference",
                    ascending=False,
                )

                impact_table = Table(
                    "Feature",
                    "With Feature",
                    "Without Feature",
                    "Difference",
                    "With Count",
                    "Without Count",
                )
                for _, row in impact_df.iterrows():
                    impact_table.add_row(
                        row["Feature"],
                        f"{row['With_Mean']:.3f}",
                        f"{row['Without_Mean']:.3f}",
                        f"{row['Difference']:.3f}",
                        f"{row['With_Count']}",
                        f"{row['Without_Count']}",
                    )
                console.print(impact_table)

    # Cross-scenario analysis if possible
    if "scenario_id" in df_results.columns and group_by != "scenario_id":
        console.print("\n[bold blue]===== CROSS-SCENARIO ANALYSIS =====[/bold blue]")

        # Only look at combined scores for simplicity
        combined_df = df_results[df_results["score_type"] == "combined"]

        if not combined_df.empty:
            # Calculate mean combined score by scenario
            scenario_means = (
                combined_df.groupby("scenario_id")["combined_score"]
                .mean()
                .sort_values(ascending=False)
            )

            scenario_table = Table("Scenario ID", "Mean Combined Score", "Count")
            for scenario_id, mean_score in scenario_means.items():
                count = len(combined_df[combined_df["scenario_id"] == scenario_id])
                scenario_table.add_row(
                    str(scenario_id),
                    f"{mean_score:.3f}",
                    str(count),
                )
            console.print(scenario_table)

    # Save results if output path is provided
    if output_path:
        if group_by == "score_type":
            # Combine all score type results
            result_dfs = []
            for score_type_val, stats in group_results.items():
                stats_reset = stats.reset_index()
                stats_reset["score_type"] = score_type_val
                result_dfs.append(stats_reset)

            if result_dfs:
                output_df = pd.concat(result_dfs)
                output_df.to_csv(output_path, index=False)
                console.print(f"\n[green]Analysis saved to {output_path}[/green]")
        else:
            # Create a complex DataFrame with hierarchical index
            all_stats = []

            for group_val, score_types in group_results.items():
                for score_type_val, stats in score_types.items():
                    stats_reset = stats.reset_index()
                    stats_reset[group_by] = group_val
                    stats_reset["score_type"] = score_type_val
                    all_stats.append(stats_reset)

            if all_stats:
                output_df = pd.concat(all_stats)
                output_df.to_csv(output_path, index=False)
                console.print(f"\n[green]Analysis saved to {output_path}[/green]")

    # Return the transformed data for further programmatic analysis
    if group_by == "score_type":
        return pd.concat(
            list(group_results.values()),
            keys=group_results.keys(),
        )

    # Create a multi-level DataFrame with hierarchical index
    result_parts = []
    for score_types in group_results.values():
        group_part = pd.concat(
            list(score_types.values()),
            keys=score_types.keys(),
            names=["score_type"],
        )
        result_parts.append(group_part)

    if result_parts:
        return pd.concat(result_parts, keys=group_results.keys(), names=[group_by])
    return pd.DataFrame()


if __name__ == "__main__":
    axs.util.init_logging(level="INFO")
    app()
