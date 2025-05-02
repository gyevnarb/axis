"""Evaluate explanations with Cluade 3.7."""

import json
import logging
import pickle
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from util import LLMModels, extract_all_explanations, random_order_string

import axs

app = typer.Typer()
logger = logging.getLogger(__name__)


class ExplanationKind(str, Enum):
    """Explanation kind for evaluation."""

    final = "final"
    all = "all"


def get_fluent_score(
    results: dict,
    llm: axs.LLMWrapper,
    prompts: dict[str, axs.Prompt],
    explanation_kind: ExplanationKind,
) -> dict[str, int]:
    """Get fluency score from LLM."""
    logger.info("Getting fluency score...")

    scenario_context = results["context"]["context"]
    question = results["user_prompt"]
    if explanation_kind == ExplanationKind.final:
        explanations = [results["explanation"]]
    elif explanation_kind == ExplanationKind.all:
        explanations = extract_all_explanations(results["messages"])
    else:
        error_msg = f"Unknown explanation kind: {explanation_kind.value}"
        logger.error(error_msg)

    ret = []
    for i, explanation in enumerate(explanations):
        logger.info(
            "Evaluating fluency for explanation %s/%s: %s",
            i + 1,
            len(explanations),
            explanation,
            extra={"markup": True},
        )

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

        sufficient_detail = int(
            re.search(r"\$ SufficientDetail: (\d+)", content).group(1),
        )
        satisfying = int(re.search(r"\$ Satisfying: (\d+)", content).group(1))
        complete = int(re.search(r"\$ Complete: (\d+)", content).group(1))
        trust = int(re.search(r"\$ Trust: (\d+)", content).group(1))

        logger.info(
            "Fluency scores: SufficientDetail: %s, Satisfying: %s, Complete: %s, Trust: %s",  # noqa: E501
            sufficient_detail,
            satisfying,
            complete,
            trust,
            extra={"markup": True},
        )

        ret.append(
            {
                "scores": {
                    "SufficientDetail": sufficient_detail,
                    "Satisfying": satisfying,
                    "Complete": complete,
                    "Trust": trust,
                },
                "explanation": explanation,
                "response": content,
                "usage": usage,
            },
        )
    return ret


def get_correct_score(
    results: dict,
    llm: axs.LLMWrapper,
    prompts: dict[str, axs.Prompt],
    ground_truth: str,
    explanation_kind: ExplanationKind,
) -> list[dict[str, int]]:
    """Get fluency score from LLM."""
    logger.info("Getting correctness score...")

    scenario_context = results["context"]["context"]
    question = results["user_prompt"]
    if explanation_kind == ExplanationKind.final:
        explanations = [results["explanation"]]
    elif explanation_kind == ExplanationKind.all:
        explanations = extract_all_explanations(results["messages"])
    else:
        error_msg = f"Unknown explanation kind: {explanation_kind.value}"
        logger.error(error_msg)

    ret = []
    for i, explanation in enumerate(explanations):
        logger.info(
            "Evaluating correctness for explanation %s/%s: %s",
            i + 1,
            len(explanations),
            explanation,
            extra={"markup": True},
        )

        if explanation == "":
            continue

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

        score = int(re.search(r"\$ Score: (\d+)", content).group(1))

        logger.info(
            "Correctness score: %s",
            score,
            extra={"markup": True},
        )

        ret.append(
            {
                "scores": {
                    "Correct": score,
                },
                "explanation": explanation,
                "response": content,
                "usage": usage,
            },
        )
    return ret


def get_actionable_score(  # noqa: PLR0913
    results: dict,
    llm: axs.LLMWrapper,
    prompts: dict[str, axs.Prompt],
    goals_actions: dict[str, str],
    use_explanation: bool,
    explanation_kind: ExplanationKind,
) -> list[dict[str, Any]]:
    """Get fluency score from LLM."""

    def _score(_prompt: str) -> dict[str, Any]:
        logger.debug(
            "Prompt for actionable score:\n%s",
            _prompt,
            extra={"markup": True},
        )

        messages = [
            {"role": "system", "content": prompts["system"].fill()},
            {"role": "user", "content": _prompt},
        ]
        response, usage = llm.chat(messages)
        content = response[0]["content"]
        goal_selection = int(re.search(r"\$ Goal: (\d+)", content).group(1))
        goal_selection_mapped = goal_order[goal_selection]
        maneuver_selection = int(re.search(r"\$ Maneuver: (\d+)", content).group(1))
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

    logger.info("Getting actionable score...")
    if use_explanation:
        logger.info("Using explanation for actionable score.")
    else:
        logger.info("Not using explanation for actionable score.")

    scenario_context = results["context"]["context"]

    goals_str, goal_order = random_order_string(goals_actions["goals"])
    maneuvers_str, mans_order = random_order_string(goals_actions["maneuvers"])

    if not use_explanation:
        prompt = prompts["no_exp_prompt"].fill(
            scenario=scenario_context,
            goals=goals_str,
            maneuvers=maneuvers_str,
        )
        score_dict = _score(prompt)
        score_dict["explanation"] = ""
        return [score_dict]

    if explanation_kind == ExplanationKind.final:
        explanations = [results["explanation"]]
    elif explanation_kind == ExplanationKind.all:
        explanations = extract_all_explanations(results["messages"])
    else:
        error_msg = f"Unknown explanation kind: {explanation_kind.value}"
        raise ValueError(error_msg)

    ret = []
    for i, explanation in enumerate(explanations):
        logger.info(
            "Evaluating actionability for explanation %s/%s: %s",
            i + 1,
            len(explanations),
            explanation,
            extra={"markup": True},
        )

        if explanation == "":
            continue

        prompt = prompts["exp_prompt"].fill(
            scenario=scenario_context,
            explanation=explanation,
            goals=goals_str,
            maneuvers=maneuvers_str,
        )
        score_dict = _score(prompt)
        score_dict["explanation"] = explanation
        ret.append(score_dict)
    return ret


@app.command()
def main(
    scenario: Annotated[
        int,
        typer.Option(
            "-s",
            "--scenario",
            help="The scenario to evaluate.",
            min=0,
            max=9,
        ),
    ] = 1,
    model: Annotated[
        LLMModels,
        typer.Option("-m", "--model", help="The LLM model to use for evaluation."),
    ] = "claude35",
    results_file: Annotated[
        str | None,
        typer.Option(
            "--results-file",
            "-r",
            help="Name of the results file to evaluate.",
        ),
    ] = None,
    explanation_kind: Annotated[
        ExplanationKind,
        typer.Option(
            "-e",
            "--explanation-kind",
            help="The explanations to use for evaluation.",
        ),
    ] = ExplanationKind.final,
) -> None:
    """Evaluate AXSAgent generated explanations with Claude or Llama (for debug)."""
    results_file = Path(results_file)

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

    save_name = f"evaluate_{model.value}_{results_file.stem}"
    save_path = Path(config.output_dir, "results", f"{save_name}.pkl")
    axs.util.init_logging(
        level="DEBUG",
        warning_only=[
            "igp2.core.velocitysmoother",
            "matplotlib",
            "httpcore",
            "openai",
            "anthropic",
            "httpx",
        ],
        log_dir=Path(config.output_dir, "logs"),
        log_name=save_name,
    )

    logger.info(
        "Evaluating scenario %s with %s on %s",
        scenario,
        model.value,
        results_file,
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
    logger.info("Loaded %s results from %s", len(results), results_path)

    if save_path.exists():
        with save_path.open("rb") as f:
            try:
                scores_results = pickle.load(f)
            except EOFError:
                scores_results = []
    else:
        scores_results = []
    logger.info("Loaded %s saves from %s", len(scores_results), save_path)

    for scores_result in scores_results:
        if "truncate" in scores_result["param"]:
            del scores_result["param"]["truncate"]
        if "truncate" in scores_result:
            del scores_result["truncate"]

    for result in results:
        scores = {}

        new_params = result["param"]

        if any(
            new_params == res["param"] and res["prompt"] == result["prompt"]
            for res in scores_results
        ):
            logger.info("Already evaluated %s", new_params)
            continue

        scores["prompt"] = result["prompt"]
        scores["param"] = new_params
        scores["actionable_exp"] = get_actionable_score(
            result,
            llm,
            prompts["actionable"],
            goals_actions,
            use_explanation=False,
            explanation_kind=explanation_kind,
        )
        scores["actionable_no_exp"] = get_actionable_score(
            result,
            llm,
            prompts["actionable"],
            goals_actions,
            use_explanation=True,
            explanation_kind=explanation_kind,
        )
        scores["correct"] = get_correct_score(
            result,
            llm,
            prompts["correct"],
            goals_actions["correct"],
            explanation_kind=explanation_kind,
        )
        scores["fluent"] = get_fluent_score(
            result,
            llm,
            prompts["fluent"],
            explanation_kind=explanation_kind,
        )

        scores_results.append(scores)

        with save_path.open("wb") as f:
            pickle.dump(scores_results, f)
            logger.info("Results saved to %s", save_path)


if __name__ == "__main__":
    app()
