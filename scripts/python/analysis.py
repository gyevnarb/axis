"""Analysis utilities for AXS evaluation results."""

import logging
import pickle
import re
from pathlib import Path
from typing import Annotated, Any

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
    rex = r"evaluate_(?P<eval_llm>[^_]+)_(?P<gen_llm>[^_]+)(?:_(?P<features>features))?(?:_(?P<interrogation>interrogation))?(?:_(?P<context>context))?\.pkl"
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
        explanation = fluent_item.get("explanation", "")
        explanation_truncated = (
            explanation[:100] + "..." if len(explanation) > 100 else explanation
        )

        score_data = {
            "explanation_idx": idx,
            # "explanation": explanation_truncated,
            "sufficient_detail": scores.get("SufficientDetail", 0),
            "satisfying": scores.get("Satisfying", 0),
            "complete": scores.get("Complete", 0),
            "trust": scores.get("Trust", 0),
        }
        scores_list.append(score_data)

    return scores_list


def extract_correct_scores(
    correct_data: list[dict[str, Any]],
) -> list[dict[str, float]]:
    """Extract correctness scores from the evaluation results for all explanations."""
    scores_list = []

    for idx, correct_item in enumerate(correct_data):
        scores = correct_item.get("scores", {})
        explanation = correct_item.get("explanation", "")
        explanation_truncated = (
            explanation[:100] + "..." if len(explanation) > 100 else explanation
        )

        score_data = {
            "explanation_idx": idx,
            # "explanation": explanation_truncated,
            "correct": scores.get("Correct", 0),
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
        explanation = actionable_item.get("explanation", "")
        explanation_truncated = (
            explanation[:100] + "..." if len(explanation) > 100 else explanation
        )

        score_data = {
            "explanation_idx": idx,
            # "explanation": explanation_truncated,
            "goal": scores.get("Goal", -1),
            "maneuver": scores.get("Maneuver", -1),
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
) -> pd.DataFrame:
    """Load all evaluation results into a pandas dataframe.

    Args:
        eval_model: The evaluation model to filter by
        gen_model: The generation model to filter by
        scenario: The scenario ID to filter by (-1 for all)
        features: Whether to filter by feature evaluation
        interrogation: Whether to filter by interrogation
        context: Whether to filter by context

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
                for result in eval_results:
                    # Extract parameter values
                    param = result.get("param", {})

                    # Extract scores
                    fluent_scores_list = extract_fluent_scores(result.get("fluent", []))
                    correct_scores_list = extract_correct_scores(
                        result.get("correct", []),
                    )

                    # Extract actionable scores (with and without explanation)
                    actionable_exp_scores_list = extract_actionable_scores(
                        result.get("actionable_exp", []),
                    )
                    actionable_no_exp_scores_list = extract_actionable_scores(
                        result.get("actionable_no_exp", []),
                    )

                    # Calculate combined score
                    combined_score = get_combined_score(result, kind="combined")

                    # Get list of features
                    features_list = list(param.get("verbalizer_features", []))
                    high_complexity = 2
                    if param.get("complexity", 1) == high_complexity:
                        features_list.append("complexity")

                    # Create base data for each row
                    base_data = {
                        "scenario_id": scenario_id,
                        "eval_llm": file_info["eval_llm"],
                        "gen_llm": file_info["gen_llm"],
                        "features_eval": file_info["features"],
                        "interrogation": file_info["interrogation"],
                        "context": file_info["context"],
                        "n_max": param.get("n_max", -1),
                        "complexity": param.get("complexity", -1),
                        # "features": ",".join(features_list),
                        "combined_score": combined_score.mean()
                        if isinstance(combined_score, list)
                        else 0,
                    }

                    # Add rows for fluent scores
                    for fluent_score in fluent_scores_list:
                        row_data = {
                            **base_data,
                            **{f"fluent_{k}": v for k, v in fluent_score.items()},
                        }
                        all_data.append(row_data)

                    # Add rows for correct scores
                    for correct_score in correct_scores_list:
                        row_data = {
                            **base_data,
                            **{f"correct_{k}": v for k, v in correct_score.items()},
                        }
                        all_data.append(row_data)

                    # Add rows for actionable scores with explanation
                    for actionable_exp_score in actionable_exp_scores_list:
                        row_data = {
                            **base_data,
                            **{
                                f"actionable_exp_{k}": v
                                for k, v in actionable_exp_score.items()
                            },
                        }
                        all_data.append(row_data)

                    # Add rows for actionable scores without explanation
                    for actionable_no_exp_score in actionable_no_exp_scores_list:
                        row_data = {
                            **base_data,
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
    ] = None,
    interrogation: Annotated[
        bool | None,
        typer.Option(
            "--interrogation",
            "-i",
            help="Whether to filter by interrogation",
        ),
    ] = None,
    context: Annotated[
        bool | None,
        typer.Option("--context", "-c", help="Whether to filter by context"),
    ] = None,
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
def all_df(
    output_path: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Path to save the consolidated DataFrame as a CSV file",
        ),
    ] = "consolidated_eval_results.csv",
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
) -> pd.DataFrame:
    """Dataframe with all results from all scenarios and generation models.

    This command automatically finds all available scenarios and generation models,
    and creates a single dataframe containing all evaluation results across all configs.
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
                "(e.g., scenario_id, gen_llm, eval_llm, features)"
            ),
        ),
    ] = "scenario_id",
) -> pd.DataFrame:
    """Analyze scores from a previously created CSV file."""
    df_results = pd.read_csv(csv_file)

    # Group by the specified column and calculate mean, std for numeric columns
    grouped = df_results.groupby(group_by).agg(["mean", "std"])

    # Flatten the column hierarchy
    grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]

    # Print the results
    console = Console()
    console.print(f"[bold green]Score Analysis by {group_by}:[/bold green]")

    # Create a table for each group
    for group_value in df_results[group_by].unique():
        group_data = grouped.loc[group_value]

        table = Table(title=f"{group_by}: {group_value}")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Std", style="yellow")

        # Add rows for important metrics
        for metric_prefix in ["combined_score", "fluent", "correct", "actionable"]:
            for col in grouped.columns:
                if col.startswith(f"{metric_prefix}") and col.endswith("_mean"):
                    metric_name = col.replace("_mean", "")
                    mean_val = f"{group_data[col]:.3f}"
                    std_val = f"{group_data[metric_name + '_std']:.3f}"
                    table.add_row(metric_name, mean_val, std_val)

        console.print(table)
        console.print("\n")

    return grouped


if __name__ == "__main__":
    axs.util.init_logging(level="INFO")
    app()
