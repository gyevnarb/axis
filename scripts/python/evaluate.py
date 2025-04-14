"""Evaluate explanations with Cluade 3.7."""

import json
import pickle
import re
from pathlib import Path
from typing import Annotated

import typer
from util import LLMModels

import axs

app = typer.Typer()

with Path("data/igp2/evaluation/fluent/system.txt").open() as f:
    fluency_system = f.read()
with Path("data/igp2/evaluation/fluent/prompt.txt").open() as f:
    fluency_prompt = axs.Prompt(f.read())
with Path("data/igp2/evaluation/correct/system.txt").open() as f:
    correct_system = f.read()
with Path("data/igp2/evaluation/correct/exp_prompt.txt").open() as f:
    correct_exp_prompt = axs.Prompt(f.read())
with Path("data/igp2/evaluation/correct/no_exp_prompt.txt").open() as f:
    correct_no_exp_prompt = axs.Prompt(f.read())


def get_fluency_score(
    results: dict,
    llm: axs.LLMWrapper,
) -> float:
    """Get fluency score from LLM."""
    scenario_context = results["context"]["context"]
    question = results["user_prompt"]
    explanation = results["explanation"]

    prompt = fluency_prompt.fill(
        scenario=scenario_context, question=question, explanation=explanation,
    )

    messages = [
        {"role": "system", "content": fluency_system},
        {"role": "user", "content": prompt},
    ]
    response, usage = llm.chat(messages)
    content = response[0]["content"]
    score = int(re.search(r"Score: (\d+)", content).group(1))
    justification = re.search(r"Justification: (.*)", content).group(1)

    results["fluency"] = {
        "score": score,
        "reason": justification,
    }
    return results


def get_correct_score(
    use_explanation: bool,
    goals,
    actions,
    results: dict,
    llm: axs.LLMWrapper,
) -> float:
    """Get fluency score from LLM."""
    scenario_context = results["context"]["context"]

    if use_explanation:
        question = results["user_prompt"]
        explanation = results["explanation"]
        prompt = correct_exp_prompt.fill(
            scenario=scenario_context,
            question=question,
            explanation=explanation,
        )
    else:
        prompt = correct_no_exp_prompt.fill(
            scenario=scenario_context, goals=goals, actions=actions,
        )

    messages = [
        {"role": "system", "content": correct_system},
        {"role": "user", "content": prompt},
    ]
    response, usage = llm.chat(messages)
    content = response[0]["content"]
    score = int(re.search(r"Score: (\d+)", content).group(1))
    justification = re.search(r"Justification: (.*)", content).group(1)

    results["fluency"] = {
        "score": score,
        "reason": justification,
    }
    return results


@app.command()
def main(
    scenario: Annotated[
        int,
        typer.Argument(help="The scenario to evaluate.", min=0, max=9),
    ] = 1,
    model: Annotated[
        LLMModels,
        typer.Argument(help="The LLM model to use."),
    ] = "claude-3.5",
) -> None:
    """Evaluate AXSAgent generated explanations with Claude 3.7."""
    if model.value not in ["claude-3.5", "claude-3.7"]:
        error_msg = f"{model.value} is not supported. Use 'claude-3.5' or 'claude-3.7'."
        raise ValueError(error_msg)

    config_path = Path(f"data/igp2/configs/scenario{scenario}.json")
    config = axs.Config(config_path)

    # Load LLM interaction
    with Path("scripts/python/llm_configs.json").open("r") as f:
        llm_configs = json.load(f)
    sampling_params = llm_configs.pop("sampling_params")
    config.llm.config_dict["sampling_params"].update(sampling_params)
    config.llm.config_dict.update(llm_configs[model.value])
    llm = axs.LLMWrapper(config.llm)

    # Load possible goals and actions for the scenario

    # Load results
    # Find file names in results directory with name "final_*.pkl"
    results_dir = Path("output", "igp2", "scenario{scenario}", "results")
    results_files = list(results_dir.glob("final_*.pkl"))
    if not results_files:
        error_msg = f"No results files found in {results_dir}"
        raise FileNotFoundError(error_msg)
    for file in results_files:
        with file.open("rb") as f:
            results = pickle.load(f)
        fluency = get_fluency_score(results, llm)
        correct = get_correct_score(results, llm)


if __name__ == "__main__":
    app()
