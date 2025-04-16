"""Obtain scores from results for a given scenario and model."""

import pickle
from pathlib import Path
from typing import Annotated

import typer
from util import LLMModels

app = typer.Typer()


@app.command()
def run(
    model: Annotated[
        LLMModels,
        typer.Option("-m", "--model", help="The LLM model whose output to score."),
    ] = "llama-70b",
    interrogation: Annotated[
        bool,
        typer.Option(help="Whether to use interrogation."),
    ] = True,
    context: Annotated[
        bool,
        typer.Option(help="Whether to add context to prompts."),
    ] = True,
) -> None:
    """Get scores from results for every available scenario."""
    save_name = f"evaluate_{model.value}_features"
    if interrogation:
        save_name += "_interrogation"
    if context:
        save_name += "_context"

    for scenario in range(10):
        save_path = Path(f"output/igp2/scenario{scenario}/results/{save_name}.pkl")
        try:
            with save_path.open("rb") as f:
                eval_results = pickle.load(f)
        except FileNotFoundError:
            print(f"Scenario {scenario} file {save_path} not found.")
            continue

        for eval_dict in eval_results:


if __name__ == "__main__":
    app()