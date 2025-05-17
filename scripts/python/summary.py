"""Summary functions for the AXS results."""

import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from util import (
    MODEL_NAME_MAP,
    LLMModels,
    extract_all_explanations,
    extract_all_queries,
)

import axs

app = typer.Typer()
logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> dict:
    """Parse the filename to extract evaluation details."""
    rex = (
        r"evaluate_(?P<eval_llm>[^_]+)_(?P<gen_llm>[^_]+)_(?P<features>features)?"
        r"(?P<interrogation>_interrogation)?(?P<context>_context)?\.pkl"
    )
    match = re.match(rex, filename)
    if match:
        return {
            "Evaluation LLM": match.group("eval_llm"),
            "Generation LLM": match.group("gen_llm"),
            "Feature Evaluation": "Yes" if match.group("features") else "No",
            "Interrogation": "Yes" if match.group("interrogation") else "No",
            "Context": "Yes" if match.group("context") else "No",
        }
    # Handle non-evaluate file names
    gen_llm_match = re.match(r"(?P<gen_llm>[^_]+)_.*\.pkl", filename)
    return {
        "Evaluation LLM": "-",  # Default value for non-evaluate files
        "Generation LLM": gen_llm_match.group("gen_llm") if gen_llm_match else "-",
        "Feature Evaluation": "Yes" if "features" in filename else "No",
        "Interrogation": "Yes" if "_interrogation" in filename else "No",
        "Context": "Yes" if "_context" in filename else "No",
    }


@app.command()
def files(
    model: Annotated[
        LLMModels | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to filter results by.",
        ),
    ] = None,
    scenario: Annotated[
        int,
        typer.Option(
            "--scenario",
            "-s",
            help="Scenario to filter results by.",
        ),
    ] = -1,
    evaluation: Annotated[
        bool,
        typer.Option(
            "-e",
            "--evaluation",
            help="Whether to filter for evaluation files.",
        ),
    ] = False,
) -> False:
    """Print the files of all scenario results directories in a table.

    Args:
        show_file_length (bool): Whether to include a column for file length.

    """
    show_file_length = True
    base_dir = Path("output", "igp2")
    console = Console()
    table = Table(title="Results Directory Contents (All Scenarios)")

    # Add table columns
    table.add_column("Scenario", justify="center", style="cyan", no_wrap=True)
    table.add_column("File Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Result Type", justify="center", style="magenta")
    if not evaluation:
        table.add_column("Evaluation LLM", justify="center", style="magenta")
    table.add_column("Generation LLM", justify="center", style="green")
    table.add_column("Interrogation", justify="center", style="blue")
    table.add_column("Explanations", justify="center", style="red")
    if show_file_length:
        table.add_column("File Length (N)", justify="right", style="yellow")

    # Check if the base directory exists
    if not base_dir.exists():
        console.print(f"[red]Error:[/red] Base directory '{base_dir}' does not exist.")
        raise typer.Exit(code=1)

    # Iterate through all scenario directories
    for scenario_dir in sorted(base_dir.glob("scenario*/results")):
        scenario_name = scenario_dir.parent.name
        if not scenario_dir.exists():
            continue

        # Iterate through files in the results directory
        for file in sorted(scenario_dir.glob("*.pkl")):
            if evaluation and "evaluate" not in file.name:
                continue
            if model and model.value not in file.name:
                continue
            if scenario != -1 and scenario_name != f"scenario{scenario}":
                continue

            parsed_data = parse_filename(file.name)
            result_type = file.name.split("_")[0] if "_" in file.name else "None"

            # Change "evaluate" to "feature" if the file indicates feature evaluation
            result_name = ""
            if result_type == "evaluate":
                result_name += "eval"
            else:
                result_name += "gen"
            if "features" in file.name:
                result_name += "_feats"

            if show_file_length:
                with file.open("rb") as f:
                    results = pickle.load(f)
                    file_length = len(results)
                    if "evaluate" in file.name:
                        # For evaluation files, count the number of explanations
                        n_explanations = tuple(
                            len(r["actionable_exp"]) for r in results
                        )
                    else:
                        n_explanations = tuple(
                            len(extract_all_explanations(r["messages"]))
                            for r in results
                        )

            if parsed_data:
                if not evaluation:
                    row = [
                        scenario_name.replace("scenario", ""),
                        file.name,
                        result_name,
                        parsed_data["Evaluation LLM"],
                        parsed_data["Generation LLM"],
                        parsed_data["Interrogation"],
                    ]
                else:
                    row = [
                        scenario_name.replace("scenario", ""),
                        file.name,
                        result_name,
                        parsed_data["Generation LLM"],
                        parsed_data["Interrogation"],
                    ]
                if show_file_length:
                    row.append(str(n_explanations))
                    row.append(str(file_length))
                table.add_row(*row)
            else:
                row = [
                    scenario_name.replace("scenario", ""),
                    file.name,
                    result_name,
                    "-",
                    "-",
                    "-",
                    "-",
                ]
                if show_file_length:
                    row.append(str(file_length))
                table.add_row(*row)

    console.print(table)


@app.command()
def explanations(
    scenario: Annotated[
        int,
        typer.Option("--scenario", "-s", help="Scenario to filter results by."),
    ] = -1,
    generation_model: Annotated[
        LLMModels | None,
        typer.Option("--model", "-m", help="Generation model to filter results by."),
    ] = None,
    interrogation: Annotated[
        bool,
        typer.Option(
            help="Whether to filter results by interrogation.",
        ),
    ] = True,
    context: Annotated[
        bool,
        typer.Option(
            help="Whether to filter results by context.",
        ),
    ] = True,
    features: Annotated[
        bool,
        typer.Option(
            help="Whether to filter results by features.",
        ),
    ] = True,
) -> None:
    """Display the contents of a results file in a rich table."""
    # Infer the results file path
    if generation_model is None:
        generation_model = LLMModels.all_model
        save_name = "*"
    else:
        save_name = f"{generation_model.value}"
    save_name += "_features" if features else ""
    save_name += "_interrogation" if interrogation else ""
    save_name += "_context" if context else ""

    if scenario == -1:
        save_paths = list(
            Path("output", "igp2").glob(f"scenario*/results/{save_name}.pkl"),
        )
    else:
        save_paths = list(
            Path("output", "igp2").glob(f"scenario{scenario}/results/{save_name}.pkl"),
        )

    if not save_paths:
        logger.error("Error: Save name '%s' does not exist.", save_name)
        raise typer.Exit(code=0)

    if not interrogation:
        save_paths = [path for path in save_paths if "_interrogation" not in path.name]
    if not context:
        save_paths = [path for path in save_paths if "_context" not in path.name]
    if not features:
        save_paths = [path for path in save_paths if "_features" not in path.name]

    # Order save paths by scenario ID
    save_paths.sort(key=lambda x: int(x.parent.parent.name.replace("scenario", "")))

    # Load results from the file
    logger.info("Loading results from %d files", len(save_paths))

    # Create a rich table
    param_keys = {"n_max", "complexity", "verbalizer_features"}
    table = Table(
        title=(
            f"Model: {generation_model.value}; Interrogation: {interrogation}; "
            f"Content: {context}; Features: {features}"
        ),
        padding=1,
    )
    table.add_column("Scenario", justify="center", style="cyan", no_wrap=True)
    table.add_column(
        "File Name", justify="left", style="magenta", max_width=20
    )  # New column
    for key in sorted(param_keys):  # Add a column for each param key
        table.add_column(
            key.capitalize(),
            justify="left",
            style="cyan",
            max_width=20,
        )
    table.add_column("N_exp", justify="right", style="green")
    table.add_column("User Prompt", justify="left", style="green", max_width=20)
    table.add_column("Explanation", justify="left", style="yellow")

    for result_file in save_paths:
        try:
            with result_file.open("rb") as f:
                results = pickle.load(f)
        except FileNotFoundError as e:
            typer.echo(f"Error loading results file: {e}")
            raise typer.Exit(code=0) from e

        if not results:
            typer.echo("No results found in the file.")
            raise typer.Exit(code=0)

        # Get scenario ID
        scenario_id = result_file.parent.parent.name.replace("scenario", "")

        # Populate the table with results
        for result in results:
            param = result.get("param", {})
            user_prompt = result.get("prompt", "N/A")  # Extract user prompt
            if isinstance(user_prompt, axs.Prompt):
                user_prompt = user_prompt.template
            explanation = result.get(
                "explanation",
                "N/A",
            )

            # Add a row with values for each param key
            row = [scenario_id, result_file.name]  # Include file name
            row.extend([str(param.get(key, "N/A")) for key in sorted(param_keys)])
            row.append(str(len(extract_all_explanations(result["messages"]))))
            row.append(user_prompt)  # Add user prompt to the row
            row.extend([str(explanation)])
            table.add_row(*row)

    # Print the table
    console = Console()
    console.print(table)


@app.command()
def models() -> None:
    """List all available LLM models with a description."""
    console = Console()
    table = Table(title="Available LLM Models")

    # Add table columns
    table.add_column("Model Name", justify="center", style="cyan", no_wrap=True)
    table.add_column("Human-Readable Name", justify="left", style="green")
    table.add_column("Description", justify="left", style="yellow")

    # Define descriptions for each model
    model_descriptions = {
        "llama70b": "A large-scale language model optimized for general-purpose tasks.",
        "qwen72b": "A high-capacity model designed for advanced reasoning and comprehension.",
        "gpt41": "An advanced version of GPT-4, known for its accuracy and fluency.",
        "gpt4o": "An optimized variant of GPT-4 for specific tasks.",
        "gpt41mini": "A lightweight version of GPT-4.1 for resource-constrained environments.",
        "o1": "A compact model designed for quick and efficient text processing.",
        "claude35": "A conversational AI model focused on natural dialogue generation.",
        "claude37": "An enhanced version of Claude 3.5 with improved reasoning capabilities.",
        "deepseekv3": "A model specialized in deep search and retrieval tasks.",
        "deepseekr1": "A robust model for retrieval-based tasks with high accuracy.",
        "all": "Represents all available models for batch operations.",
    }

    # Populate the table with model data
    for model, human_readable_name in MODEL_NAME_MAP.items():
        description = model_descriptions.get(model, "No description available.")
        table.add_row(model, human_readable_name, description)

    # Print the table
    console.print(table)


@app.command()
def queries_by_scenario(
    scenario: Annotated[
        int,
        typer.Option(
            "--scenario",
            "-s",
            help="Scenario to filter by (-1 for all scenarios)",
        ),
    ] = -1,
    output_path: Annotated[
        str,
        typer.Option("--output", "-o", help="Path to save the plot"),
    ] = "output/igp2/plots/query_types_by_scenario.pdf",
) -> None:
    """Plot the aggregated count of query types by scenario.

    This command analyzes result files and counts the occurrences of different query
    types (add, remove, whatif, what) for each scenario, aggregating across all LLMs.
    Results are displayed in a bar plot with scenarios on the x-axis, counts on the
    y-axis, and different colors for each query type.
    """
    # Base directory containing scenario results
    base_dir = Path("output", "igp2")

    # Check if the base directory exists
    if not base_dir.exists():
        logger.error("Base directory '%s' does not exist.", base_dir)
        raise typer.Exit(code=1)

    # Dictionary to store query counts by scenario and query type
    # Structure: {scenario_id: {query_type: count}}
    scenario_query_counts = defaultdict(lambda: defaultdict(int))

    # List to keep track of unique query types and scenarios found
    query_types_found = set()
    scenarios_found = set()

    # Get all scenario directories
    scenario_dirs = list(base_dir.glob("scenario*/results"))
    scenario_dirs.sort(key=lambda x: int(x.parent.name.replace("scenario", "")))

    # Filter by scenario if specified
    if scenario != -1:
        scenario_dirs = [
            d for d in scenario_dirs if d.parent.name == f"scenario{scenario}"
        ]

    if not scenario_dirs:
        logger.error("No matching scenario directories found.")
        raise typer.Exit(code=1)

    # Process each scenario directory
    for scenario_dir in scenario_dirs:
        scenario_id = int(scenario_dir.parent.name.replace("scenario", ""))
        scenarios_found.add(scenario_id)

        # Look for result files
        result_files = list(scenario_dir.glob("*.pkl"))

        # Skip if no result files found
        if not result_files:
            logger.warning("No result files found in %s", scenario_dir)
            continue

        # Process each result file
        for file_path in result_files:
            # Skip evaluation files
            if "evaluate" in file_path.name or "features" in file_path.name:
                continue

            try:
                # Load the results file
                with file_path.open("rb") as f:
                    results = pickle.load(f)

                # Process each result
                for result in results:
                    # Skip if no messages
                    if "messages" not in result:
                        continue

                    # Extract queries from messages
                    queries = extract_all_queries(result["messages"])

                    # Count each query type for this scenario
                    for query in queries:
                        scenario_query_counts[scenario_id][query] += 1
                        query_types_found.add(query)

            except Exception as e:
                logger.exception("Error processing file %s: %s", file_path, e)

    # Sort the found scenarios and query types for consistent plotting
    scenarios_sorted = sorted(scenarios_found)
    query_types_sorted = sorted(query_types_found)

    # Check if we found any queries
    if not scenario_query_counts:
        logger.error("No queries found in the results.")
        raise typer.Exit(code=1)

    # Calculate error bars based on standard deviation across models
    # This requires running a separate collection per model
    model_scenario_query_counts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    # Get all model names
    model_names = set()

    # Process each scenario directory again to collect data per model
    for scenario_dir in scenario_dirs:
        scenario_id = int(scenario_dir.parent.name.replace("scenario", ""))

        # Look for result files
        result_files = list(scenario_dir.glob("*.pkl"))

        # Process each result file
        for file_path in result_files:
            # Skip evaluation files
            if "evaluate" in file_path.name or "features" in file_path.name:
                continue

            # Extract model name from filename
            model_name = file_path.stem.split("_")[0]
            model_names.add(model_name)

            try:
                # Load the results file
                with file_path.open("rb") as f:
                    results = pickle.load(f)

                # Process each result
                for result in results:
                    # Skip if no messages
                    if "messages" not in result:
                        continue

                    # Extract queries from messages
                    queries = extract_all_queries(result["messages"])

                    # Count each query type for this scenario and model
                    for query in queries:
                        model_scenario_query_counts[model_name][scenario_id][query] += 1

            except Exception as e:
                logger.exception(
                    "Error processing file %s for error bars: %s", file_path, e
                )

    # Set publication-ready style for the plot
    plt.style.use("seaborn-v0_8-whitegrid")

    # Set up the figure and axes - using NeurIPS single column width (approx 3.5 inches)
    fig, ax = plt.subplots(figsize=(4.5, 2))

    # Define a color-blind friendly palette (better for print)
    # Using a colorblind-friendly palette from ColorBrewer
    color_palette = ["#66a61e", "#d95f02", "#e7298a", "#1b9e77", "#7570b3", "#e6ab02"]
    # Ensure we have enough colors
    if len(query_types_sorted) > len(color_palette):
        # Fall back to a colormap if we have more query types than colors
        color_map = plt.cm.tab10(np.arange(len(query_types_sorted)))
    else:
        # Use our defined palette
        color_map = [color_palette[i] for i in range(len(query_types_sorted))]

    # Set bar width - slightly thinner for better appearance in small figure
    bar_width = 0.7 / len(query_types_sorted)

    # Set up x-axis positions
    x = np.arange(len(scenarios_sorted))

    # Plot bars for each query type
    for i, query_type in enumerate(query_types_sorted):
        # Get counts for this query type across all scenarios
        counts = [
            scenario_query_counts[scenario_id].get(query_type, 0)
            for scenario_id in scenarios_sorted
        ]

        # Calculate error bars (standard deviation across different models)
        errors = []
        for scenario_id in scenarios_sorted:
            # Get counts for this query type and scenario across all models
            model_counts = [
                model_scenario_query_counts[model][scenario_id].get(query_type, 0)
                for model in model_names
                if model
            ]
            # Calculate standard error of mean (SEM) instead of standard deviation
            if len(model_counts) > 1:
                # SEM = standard deviation / sqrt(n)
                error = np.std(model_counts, ddof=1) / np.sqrt(len(model_counts))
            else:
                error = 0
            errors.append(error)

        # Calculate position for this group of bars
        positions = (
            x
            + bar_width * i
            - (len(query_types_sorted) * bar_width / 2)
            + bar_width / 2
        )

        # Plot the bars with hatch patterns and error bars
        ax.bar(
            positions,
            counts,
            width=bar_width,
            label=query_type.capitalize(),
            color=color_map[i] if isinstance(color_map[i], str) else color_map[i],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            hatch=["", "x", "+", ".", "/", "\\"][
                i % 6
            ],  # Add hatches for print differentiation
            yerr=errors,
            capsize=2,  # Size of error bar caps
            error_kw={
                "elinewidth": 0.7,
                "capthick": 0.7,
            },  # Thinner error bars for publication
        )

    # No title for academic publication
    # ax.set_title("Query Type Distribution by Scenario", fontsize=9)

    # Set labels with appropriate font sizes for publication
    # ax.set_xlabel("Scenario ID", fontsize=8)
    # ax.set_ylabel("Count", fontsize=8)

    # Refined grid for academic publication
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Set x-tick positions and labels with smaller font size
    ax.set_xticks(x)
    ax.set_xticklabels(["#" + str(s) for s in scenarios_sorted], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)

    # Add legend above the figure with multiple columns
    legend = ax.legend(
        fontsize=6,
        title_fontsize=7,
        title="Query Types",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),  # Position above the plot
        ncol=len(
            query_types_sorted
        ),  # Use multiple columns to fit all query types horizontally
        frameon=True,
        framealpha=1,
        edgecolor="black",
    )

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the legend on top

    # Save the figure - increase DPI for print quality
    plt.savefig(output_path, dpi=600, bbox_inches="tight", format="pdf")
    logger.info("Plot saved to %s", output_path)

    # Show some statistics
    total_queries = sum(
        sum(counts.values()) for counts in scenario_query_counts.values()
    )

    logger.info("Total queries found: %d", total_queries)
    logger.info("Query types: %s", query_types_sorted)
    logger.info("Scenarios: %s", scenarios_sorted)

    # Output per-scenario statistics
    for scenario_id in scenarios_sorted:
        scenario_total = sum(scenario_query_counts[scenario_id].values())
        logger.info(
            "Scenario %d: %d total queries - %s",
            scenario_id,
            scenario_total,
            {k: v for k, v in scenario_query_counts[scenario_id].items()},
        )


@app.command()
def queries_by_model(
    scenario: Annotated[
        int,
        typer.Option(
            "--scenario",
            "-s",
            help="Scenario to filter by (-1 for all scenarios)",
        ),
    ] = -1,
    output_path: Annotated[
        str,
        typer.Option("--output", "-o", help="Path to save the plot"),
    ] = "output/igp2/plots/query_types_by_model.pdf",
) -> None:
    """Plot the aggregated count of query types by model.

    This command analyzes result files and counts the occurrences of different query
    types (add, remove, whatif, what) for each model, aggregating across all scenarios.
    Results are displayed in a bar plot with models on the x-axis, counts on the
    y-axis, and different colors for each query type. Standard error of mean (SEM)
    error bars are included to show variability across scenarios.
    """
    # Base directory containing scenario results
    base_dir = Path("output", "igp2")

    # Check if the base directory exists
    if not base_dir.exists():
        logger.error("Base directory '%s' does not exist.", base_dir)
        raise typer.Exit(code=1)

    # Dictionary to store query counts by model and query type
    # Structure: {model_name: {query_type: count}}
    model_query_counts = defaultdict(lambda: defaultdict(int))

    # Dictionary to store per-scenario query counts by model and query type
    # Structure: {model_name: {scenario_id: {query_type: count}}}
    model_scenario_query_counts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    # List to keep track of unique query types and models found
    query_types_found = set()
    models_found = set()
    scenarios_found = set()

    # Get all scenario directories
    scenario_dirs = list(base_dir.glob("scenario*/results"))
    scenario_dirs.sort(key=lambda x: int(x.parent.name.replace("scenario", "")))

    # Filter by scenario if specified
    if scenario != -1:
        scenario_dirs = [
            d for d in scenario_dirs if d.parent.name == f"scenario{scenario}"
        ]

    if not scenario_dirs:
        logger.error("No matching scenario directories found.")
        raise typer.Exit(code=1)

    # Process each scenario directory
    for scenario_dir in scenario_dirs:
        scenario_id = int(scenario_dir.parent.name.replace("scenario", ""))
        scenarios_found.add(scenario_id)

        # Look for result files
        result_files = list(scenario_dir.glob("*.pkl"))

        # Skip if no result files found
        if not result_files:
            logger.warning("No result files found in %s", scenario_dir)
            continue

        # Process each result file
        for file_path in result_files:
            # Skip evaluation files
            if "evaluate" in file_path.name or "features" in file_path.name:
                continue

            # Extract model name from filename
            model_name = file_path.stem.split("_")[0]
            models_found.add(model_name)

            try:
                # Load the results file
                with file_path.open("rb") as f:
                    results = pickle.load(f)

                # Process each result
                for result in results:
                    # Skip if no messages
                    if "messages" not in result:
                        continue

                    # Extract queries from messages
                    queries = extract_all_queries(result["messages"])

                    # Count each query type for this model
                    for query in queries:
                        model_query_counts[model_name][query] += 1
                        model_scenario_query_counts[model_name][scenario_id][query] += 1
                        query_types_found.add(query)

            except Exception as e:
                logger.exception("Error processing file %s: %s", file_path, e)

    # Sort the found models and query types for consistent plotting
    models_found.remove("qwen72b")  # Remove qwen72b from the list
    models_sorted = sorted(models_found)
    query_types_sorted = sorted(query_types_found)

    # Check if we found any queries
    if not model_query_counts:
        logger.error("No queries found in the results.")
        raise typer.Exit(code=1)

    # Set publication-ready style for the plot
    plt.style.use("seaborn-v0_8-whitegrid")

    # Set up the figure and axes - using NeurIPS single column width (approx 3.5 inches)
    fig, ax = plt.subplots(figsize=(4.5, 1.5))

    # Define a color-blind friendly palette (better for print)
    # Using a colorblind-friendly palette from ColorBrewer
    color_palette = ["#66a61e", "#d95f02", "#e7298a", "#7570b3", "#1b9e77", "#e6ab02"]
    # Ensure we have enough colors
    if len(query_types_sorted) > len(color_palette):
        # Fall back to a colormap if we have more query types than colors
        color_map = plt.cm.tab10(np.arange(len(query_types_sorted)))
    else:
        # Use our defined palette
        color_map = [color_palette[i] for i in range(len(query_types_sorted))]

    # Set bar width - slightly thinner for better appearance in small figure
    bar_width = 0.7 / len(query_types_sorted)

    # Set up x-axis positions
    x = np.arange(len(models_sorted))

    # Calculate total counts per model for normalization
    model_totals = {
        model_name: sum(model_query_counts[model_name].values())
        for model_name in models_sorted
    }

    # Plot bars for each query type
    for i, query_type in enumerate(query_types_sorted):
        # Get proportions (instead of raw counts) for this query type across all models
        proportions = []
        for model_name in models_sorted:
            count = model_query_counts[model_name].get(query_type, 0)
            total = model_totals[model_name]
            proportion = (
                (count / total) * 100 if total > 0 else 0
            )  # Convert to percentage
            proportions.append(proportion)

        # Calculate error bars (standard error of mean across different scenarios)
        errors = []
        for model_name in models_sorted:
            # Get proportional counts for this query type and model across all scenarios
            scenario_proportions = []
            for scenario_id in scenarios_found:
                query_count = model_scenario_query_counts[model_name][scenario_id].get(
                    query_type, 0
                )
                scenario_total = sum(
                    model_scenario_query_counts[model_name][scenario_id].values()
                )
                if scenario_total > 0:
                    proportion = (
                        query_count / scenario_total
                    ) * 100  # Convert to percentage
                    scenario_proportions.append(proportion)

            # Calculate standard error of mean (SEM)
            if len(scenario_proportions) > 1:
                # SEM = standard deviation / sqrt(n)
                error = np.std(scenario_proportions, ddof=1) / np.sqrt(
                    len(scenario_proportions)
                )
            else:
                error = 0
            errors.append(error)

        # Calculate position for this group of bars
        positions = (
            x
            + bar_width * i
            - (len(query_types_sorted) * bar_width / 2)
            + bar_width / 2
        )

        # Plot the bars with hatch patterns and error bars
        ax.bar(
            positions,
            proportions,  # Use the proportions list calculated above
            width=bar_width,
            label=query_type.capitalize(),
            color=color_map[i] if isinstance(color_map[i], str) else color_map[i],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            hatch=["", "x", "+", ".", "/", "\\"][
                i % 6
            ],  # Add hatches for print differentiation
            yerr=errors,
            capsize=2,  # Size of error bar caps
            error_kw={
                "elinewidth": 0.7,
                "capthick": 0.7,
            },  # Thinner error bars for publication
        )

    # Refined grid for academic publication
    ax.grid(True, alpha=0.8, linestyle="--", linewidth=0.5)

    # Set x-tick positions and labels with smaller font size
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_NAME_MAP.get(m, m) for m in models_sorted],
        fontsize=7,
        # rotation=45,
        ha="center",
    )
    ax.tick_params(axis="y", labelsize=7)

    # Format y-axis with percentage symbols
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter())

    # Add legend above the figure with multiple columns
    ax.legend(
        fontsize=6,
        title_fontsize=7,
        title="Query Types",
        loc="center right",
        bbox_to_anchor=(1.2, 0.5),  # Position above the plot
        # ncol=len(query_types_sorted),
        frameon=True,
        framealpha=0.25,
        edgecolor="black",
    )

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)  # Make room for the legend on top

    # Save the figure - increase DPI for print quality
    plt.savefig(output_path, dpi=600, bbox_inches="tight", format="pdf")
    logger.info("Plot saved to %s", output_path)

    # Show some statistics
    total_queries = sum(sum(counts.values()) for counts in model_query_counts.values())

    logger.info("Total queries found: %d", total_queries)
    logger.info("Query types: %s", query_types_sorted)
    logger.info("Models: %s", models_sorted)

    # Output per-model statistics
    for model_name in models_sorted:
        model_total = sum(model_query_counts[model_name].values())
        logger.info(
            "Model %s: %d total queries - %s",
            model_name,
            model_total,
            dict(model_query_counts[model_name].items()),
        )


if __name__ == "__main__":
    axs.util.init_logging(
        level="INFO",
        warning_only=[
            "igp2.core.velocitysmoother",
            "matplotlib",
            "httpcore",
            "openai",
            "httpx",
        ],
    )
    app()
