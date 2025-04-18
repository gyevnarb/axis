"""Summary functions for the AXS results."""

import logging
import pickle
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from util import LLMModels

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
    show_file_length: Annotated[
        bool,
        typer.Option(
            "-l",
            "--length",
            help="Whether to show number of items in each file. Slow",
        ),
    ] = False,
) -> False:
    """Print the files of all scenario results directories in a table.

    Args:
        show_file_length (bool): Whether to include a column for file length.

    """
    base_dir = Path("output", "igp2")
    console = Console()
    table = Table(title="Results Directory Contents (All Scenarios)")

    # Add table columns
    table.add_column("Scenario", justify="center", style="cyan", no_wrap=True)
    table.add_column("File Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Result Type", justify="center", style="magenta")
    table.add_column("Evaluation LLM", justify="center", style="magenta")
    table.add_column("Generation LLM", justify="center", style="green")
    table.add_column("Interrogation", justify="center", style="blue")
    table.add_column("Context", justify="center", style="red")
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
                    file_length = len(pickle.load(f))

            if parsed_data:
                row = [
                    scenario_name.replace("scenario", ""),
                    file.name,
                    result_name,
                    parsed_data["Evaluation LLM"],
                    parsed_data["Generation LLM"],
                    parsed_data["Interrogation"],
                    parsed_data["Context"],
                ]
                if show_file_length:
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

    # Order save paths by scenario ID
    save_paths.sort(key=lambda x: int(x.parent.parent.name.replace("scenario", "")))

    # Load results from the file
    logger.info("Loading results from %d files", len(save_paths))

    # Create a rich table
    param_keys = {"n_max", "complexity", "verbalizer_features"}
    table = Table(
        title=(f"Model: {generation_model.value}; Interrogation: {interrogation}; "
               f"Content: {context}; Features: {features}"),
        padding=1,
    )
    table.add_column("Scenario", justify="center", style="cyan", no_wrap=True)
    for key in sorted(param_keys):  # Add a column for each param key
        table.add_column(
            key.capitalize(),
            justify="left",
            style="cyan",
            max_width=20,
        )
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
            explanation = result.get(
                "explanation",
                "N/A",
            )

            # Add a row with values for each param key
            row = [scenario_id]
            row.extend([str(param.get(key, "N/A")) for key in sorted(param_keys)])
            row.extend([str(explanation)])
            table.add_row(*row)

    # Print the table
    console = Console()
    console.print(table)


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
