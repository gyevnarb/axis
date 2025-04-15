"""Evaluate explanations with Cluade 3.7."""

import json
import logging
import pickle
import random
import re
from pathlib import Path
from typing import Annotated

import typer
from util import LLMModels

import axs

app = typer.Typer()
logger = logging.getLogger(__name__)


def get_fluent_score(
    results: dict,
    llm: axs.LLMWrapper,
    prompts: dict[str, axs.Prompt],
) -> dict[str, int]:
    """Get fluency score from LLM."""
    logger.info("Getting fluency score...")

    scenario_context = results["context"]["context"]
    question = results["user_prompt"]
    explanation = results["explanation"]

    prompt = prompts["prompt"].fill(
        scenario=scenario_context,
        question=question,
        explanation=explanation,
    )

    logger.debug("Prompt: %s", prompt)

    messages = [
        {"role": "system", "content": prompts["system"].fill()},
        {"role": "user", "content": prompt},
    ]
    response, usage = llm.chat(messages)
    content = response[0]["content"]

    sufficient_detail = int(re.search(r"SufficientDetail: (\d+)", content).group(1))
    satisfying = int(re.search(r"Satisfying: (\d+)", content).group(1))
    complete = int(re.search(r"Complete: (\d+)", content).group(1))
    trust = int(re.search(r"Trust: (\d+)", content).group(1))

    logger.info(
        "Fluency scores: SufficientDetail: %s, Satisfying: %s, Complete: %s, Trust: %s",
        sufficient_detail,
        satisfying,
        complete,
        trust,
        extra={"markup": True},
    )

    return {
        "scores": {
            "SufficientDetail": sufficient_detail,
            "Satisfying": satisfying,
            "Complete": complete,
            "Trust": trust,
        },
        "response": content,
        "usage": usage,
    }


def get_correct_score(
    results: dict,
    llm: axs.LLMWrapper,
    prompts: dict[str, axs.Prompt],
    ground_truth: str,
) -> dict[str, int]:
    """Get fluency score from LLM."""
    logger.info("Getting correctness score...")

    scenario_context = results["context"]["context"]
    question = results["user_prompt"]
    explanation = results["explanation"]

    prompt = prompts["prompt"].fill(
        scenario=scenario_context,
        question=question,
        explanation=explanation,
        ground_truth=ground_truth,
    )

    logger.debug(
        "Prompt for correctness score:\n%s",
        prompt,
        extra={"markup": True},
    )

    messages = [
        {"role": "system", "content": prompts["system"].fill()},
        {"role": "user", "content": prompt},
    ]
    response, usage = llm.chat(messages)
    content = response[0]["content"]

    score = int(re.search(r"Score: (\d+)", content).group(1))

    logger.info(
        "Correctness score: %s",
        score,
        extra={"markup": True},
    )

    return {
        "scores": {
            "Correct": score,
        },
        "response": content,
        "usage": usage,
    }


def random_order_string(items: dict[str, str]) -> str:
    """Return a random order string from a dictionary."""
    items_list = list(items.items())
    random.shuffle(items_list)
    remapped_items = {k: items_list[k][1] for k in range(len(items_list))}
    shuffle_mapping = {i: int(v[0]) for i, v in enumerate(items_list)}
    return "\n".join([f"{k}. {v}" for k, v in remapped_items.items()]), shuffle_mapping


def get_actionable_score(
    results: dict,
    llm: axs.LLMWrapper,
    prompts: dict[str, axs.Prompt],
    goals_actions: dict[str, str],
    use_explanation: bool,
) -> float:
    """Get fluency score from LLM."""
    logger.info("Getting actionable score...")
    if use_explanation:
        logger.info("Using explanation for actionable score.")
    else:
        logger.info("Not using explanation for actionable score.")

    scenario_context = results["context"]["context"]

    goals_str, goal_order = random_order_string(goals_actions["goals"])
    maneuvers_str, mans_order = random_order_string(goals_actions["maneuvers"])

    if use_explanation:
        explanation = results["explanation"]
        prompt = prompts["exp_prompt"].fill(
            scenario=scenario_context,
            explanation=explanation,
            goals=goals_str,
            maneuvers=maneuvers_str,
        )
    else:
        prompt = prompts["no_exp_prompt"].fill(
            scenario=scenario_context,
            goals=goals_str,
            maneuvers=maneuvers_str,
        )

    logger.debug(
        "Prompt for actionable score:\n%s",
        prompt,
        extra={"markup": True},
    )

    messages = [
        {"role": "system", "content": prompts["system"].fill()},
        {"role": "user", "content": prompt},
    ]
    response, usage = llm.chat(messages)
    content = response[0]["content"]
    goal_selection = int(re.search(r"Goal: (\d+)", content).group(1))
    goal_selection_mapped = goal_order[goal_selection]
    maneuver_selection = int(re.search(r"Maneuver: (\d+)", content).group(1))
    maneuver_selection_mapped = mans_order[maneuver_selection]

    logger.info(
        "Actionable scores (remapped): Goal: %s, Maneuver: %s",
        goal_selection_mapped,
        maneuver_selection_mapped,
        extra={"markup": True},
    )

    return {
        "scores": {
            "Goal": goal_selection_mapped,
            "Maneuver": maneuver_selection_mapped,
        },
        "reason": content,
        "usage": usage,
    }


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
    results_file: Annotated[
        str | None,
        typer.Option(
            "--results-file",
            "-r",
            help="Name of the results file to evaluate.",
        ),
    ] = None,
) -> None:
    """Evaluate AXSAgent generated explanations with Claude 3.7."""
    if model.value not in ["claude-3.5", "claude-3.7"]:
        error_msg = f"{model.value} is not supported. Use 'claude-3.5' or 'claude-3.7'."
        raise ValueError(error_msg)

    model = LLMModels.llama_70b

    # Read all prompts into a single dictionary with appropriate keys
    prompt_files = Path("data/igp2/evaluation").glob("*/*.txt")
    prompts = {}
    for prompt_file in prompt_files:
        with prompt_file.open("r") as f:
            prompt_dir = prompt_file.parent.stem
            if prompt_dir not in prompts:
                prompts[prompt_dir] = {}
            prompt_name = prompt_file.stem
            prompts[prompt_dir][prompt_name] = axs.Prompt(f.read())

    # Load scenario config
    config_path = Path(f"data/igp2/configs/scenario{scenario}.json")
    config = axs.Config(config_path)

    save_name = f"{model.value}_feature_evaluate"
    save_path = Path(config.output_dir, "results", f"{save_name}.pkl")
    axs.util.init_logging(
        level="DEBUG",
        warning_only=[
            "igp2.core.velocitysmoother",
            "matplotlib",
            "httpcore",
            "openai",
            "httpx",
        ],
        log_dir=Path(config.output_dir, "logs"),
        log_name=save_name,
    )

    # Load LLM interaction
    with Path("scripts/python/llm_configs.json").open("r") as f:
        llm_configs = json.load(f)
    sampling_params = llm_configs.pop("sampling_params")
    sampling_params["max_tokens"] = 1024
    config.llm.config_dict["sampling_params"].update(sampling_params)
    config.llm.config_dict.update(llm_configs[model.value])
    llm = axs.LLMWrapper(config.llm)

    # Load possible goals and actions for the scenario
    with Path("data/igp2/evaluation/goals_actions.json").open("r") as f:
        goals_actions = json.load(f)
    goals_actions = goals_actions[str(scenario)]

    # Load results
    results_path = Path("output/igp2", f"scenario{scenario}/results/{results_file}")
    with results_path.open("rb") as f:
        results = pickle.load(f)

    scores_results = []
    for result in results:
        scores = {}
        scores["actionable_exp"] = get_actionable_score(
            result, llm, prompts["actionable"], goals_actions, use_explanation=True,
        )
        scores["actionable_no_exp"] = get_actionable_score(
            result, llm, prompts["actionable"], goals_actions, use_explanation=False,
        )
        scores["correct"] = get_correct_score(
            result,
            llm,
            prompts["correct"],
            goals_actions["correct"],
        )
        scores["fluent"] = get_fluent_score(result, llm, prompts["fluent"])
        scores_results.append(scores)

        with save_path.open("wb") as f:
            pickle.dump(scores_results, f)
            logger.info("Results saved to %s", save_path)


if __name__ == "__main__":
    app()
