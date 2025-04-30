"""Get all combinations of experimetal conditions."""

import enum
import json
import logging
import pickle
import random
import re
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain, combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np
import typer
from scipy.special import factorial

import axs

logger = logging.getLogger(__name__)
app = typer.Typer()


MODEL_NAME_MAP = {
    "llama70b": "LLaMA-3.3-70B",
    "qwen72b": "Qwen-2.5-72B",
    "gpt41": "GPT-4.1",
    "gpt4o": "GPT-4o",
    "gpt41mini": "GPT-4.1-mini",
    "o1": "o1",
    "claude35": "Claude 3.5",
    "claude37": "Claude 3.7",
    "deepseekv3": "DeepSeek-V3",
    "deepseekr1": "DeepSeek-R1",
    "all": "All Models",
}

FEATURE_LABELS = {
    "add_actions": "Actions",
    "add_layout": "Layout",
    "add_macro_actions": "Macros",
    "add_observations": "Observ's",
    "complexity": "Complex",
    "truncate": "Memory Truncation",
}


class LLMModels(enum.Enum):
    """Enum for LLM models."""

    llama_70b = "llama70b"
    qwen_72b = "qwen72b"
    gpt_4_1 = "gpt41"
    gpt_4o = "gpt4o"
    gpt_4_1_mini = "gpt41mini"
    o1 = "o1"
    claude_3_5 = "claude35"
    claude_3_7 = "claude37"
    deepseek_v3 = "deepseekv3"
    deepseek_r1 = "deepseekr1"
    all_model = "all"


def get_agent(config: axs.Config) -> axs.AXSAgent:
    """Create an AXS agent with the given configuration."""
    env = axs.util.load_env(config.env, config.env.render_mode)
    env.reset(seed=config.env.seed)
    logger.info("Created environment %s", config.env.name)

    agent_policies = axs.registry.get(config.env.policy_type).create(env)
    return axs.AXSAgent(config, agent_policies)


def get_save_paths(ctx: typer.Context, features: bool = True) -> list[Path]:
    """Load results from the specified directory.

    Args:
        ctx (typer.Context): Typer context.
        features (bool): Whether working with feature evaluations.

    """
    generation_model = ctx.obj["model"]
    eval_model = ctx.obj["eval_model"]
    interrogation = ctx.obj["interrogation"]
    context = ctx.obj["context"]
    scenario = ctx.obj["scenario"]

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

    save_name = (
        "evaluate_*" if eval_model.value == "all" else f"evaluate_{eval_model.value}"
    )
    save_name += (
        f"_{generation_model.value}" if generation_model.value != "all" else "_*"
    )
    if features:
        save_name += "_features"
    if interrogation:
        save_name += "_interrogation"
    if context:
        save_name += "_context"

    if scenario == -1:
        save_paths = list(
            Path("output", "igp2").glob(f"scenario*/results/{save_name}.pkl"),
        )
    else:
        save_paths = list(
            Path("output", "igp2", f"scenario{scenario}").glob(
                f"results/{save_name}.pkl",
            ),
        )
    if not save_paths:
        logger.error("No results found for the given parameters.")
        return []

    if not features:
        save_paths = [file for file in save_paths if "features" not in str(file)]
    if not interrogation:
        save_paths = [file for file in save_paths if "interrogation" not in str(file)]
    if not context:
        save_paths = [file for file in save_paths if "context" not in str(file)]

    logger.info(
        "Found %d results files: %s",
        len(save_paths),
        list(map(str, save_paths)),
    )
    return save_paths


def load_eval_results(
    ctx: typer.Context,
    save_path: Path,
) -> tuple[dict[str, Any], tuple[str, str, str]] | tuple[None, tuple[None, None, None]]:
    """Load evaluation results from the specified file."""
    scenario = ctx.obj["scenario"]

    sid = re.search(r"scenario(\d+)", str(save_path)).group(1)
    if scenario != -1 and int(sid) != scenario:
        logger.info("Not loading: %s", str(save_path))
        return None, (None, None, None)
    eval_model_str, gen_model_str = str(save_path.stem).split("_")[1:3]
    logger.info(
        "Processed eval model: %s; Gen model: %s",
        eval_model_str,
        gen_model_str,
    )
    with save_path.open("rb") as f:
        eval_results = pickle.load(f)
        return eval_results, (sid, eval_model_str, gen_model_str)


def powerset(iterable: Iterable) -> list[tuple]:
    """Generate the powerset of a given iterable.

    Args:
        iterable (Iterable): The input iterable.

    """
    s = list(iterable)
    # Change iteration to range(1, len(s)+1) to exclude empty set
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_params(  # noqa: PLR0913
    scenarios: list[int],
    complexity: list[int],
    models: list[str],
    features: list[str],
    use_interrogation: bool,
    use_context: bool,
    n_max: int,
) -> list[axs.Config]:
    """Get all combinations of experimetal conditions."""
    if not use_interrogation and not use_context:
        error_msg = "At least one of use_interrogation or use_context must be True."
        raise ValueError(error_msg)
    if not use_interrogation and n_max != 0:
        error_msg = "n_max must be 0 when use_interrogation is False."
        raise ValueError(error_msg)

    verbalizer_features = [
        "add_layout",
        "add_observations",
        "add_actions",
        "add_macro_actions",
    ]
    with Path("scripts", "python", "llm_configs.json").open() as f:
        llm_configs = json.load(f)
    sampling_params = llm_configs.pop("sampling_params")
    llm_configs = {k: v for k, v in llm_configs.items() if k in models}

    configs = []
    for scenario in scenarios:
        with Path("data", "igp2", "configs", f"scenario{scenario}.json").open() as f:
            config_dict = json.load(f)

        config_dict["llm"]["sampling_params"].update(sampling_params)
        config_dict["axs"]["use_interrogation"] = use_interrogation
        config_dict["axs"]["use_context"] = use_context
        config_dict["axs"]["n_max"] = n_max

        # Create LLM config combinations
        for llm_config in llm_configs.values():
            for c in complexity:
                for vf in powerset(verbalizer_features):
                    if features is not None and set(vf) != set(features):
                        continue
                    new_config_dict = deepcopy(config_dict)
                    new_config_dict["axs"]["complexity"] = c
                    new_config_dict["llm"].update(llm_config)
                    if vf != ():
                        vf_dict = new_config_dict["axs"]["verbalizer"]
                        vf_dict["params"] = dict.fromkeys(
                            vf,
                            True,
                        )
                        vf_dict["params"]["add_rewards"] = True
                        vf_dict["params"]["subsample"] = 3
                        params = {
                            "complexity": c,
                            "verbalizer_features": vf,
                            "llm_config": llm_config,
                            "scenario": scenario,
                            "use_interrogation": use_interrogation,
                            "use_context": use_context,
                            "n_max": n_max,
                            "config": axs.Config(new_config_dict),
                        }
                        configs.append(params)
    return configs


def extract_all_explanations(messages: list[dict[str, str]]) -> list[str]:
    """Extract all explanations from the messages.

    Args:
        messages (list): List of messages. passed to the LLM.

    """
    ret = []
    for prev_msg, curr_msg in pairwise(messages):
        if (
            prev_msg["role"] == "user"
            and curr_msg["role"] == "assistant"
            and (
                "explanation stage" in prev_msg["content"].lower()
                or "final explanation" in prev_msg["content"].lower()
            )
        ):
            ret.append(curr_msg["content"])
    return ret


def random_order_string(items: dict[str, str]) -> str:
    """Return a random order string from a dictionary."""
    items_list = list(items.items())
    random.shuffle(items_list)
    remapped_items = {k: items_list[k][1] for k in range(len(items_list))}
    shuffle_mapping = {i: int(v[0]) for i, v in enumerate(items_list)}
    return "\n".join([f"{k}. {v}" for k, v in remapped_items.items()]), shuffle_mapping


def get_combined_score(
    eval_result: dict[str, list[Any] | Any],
    kind: str = "correct",
) -> list[float]:
    """Get combined score for fluency and correctness using geometric mean.

    Args:
        eval_result (dict): Evaluation result dictionary.
        kind (str): The kind of metric to calculate. Either 'correct' or 'combined'.

    """
    if isinstance(eval_result["correct"], list):
        correct_scores = np.array(
            [res["scores"]["Correct"] for res in eval_result["correct"]],
        )
        fluent_scores = np.array(
            [list(res["scores"].values()) for res in eval_result["fluent"]],
        )
    else:
        correct_scores = np.array(
            [eval_result["correct"]["scores"]["Correct"]],
        )
        fluent_scores = np.array(
            [list(eval_result["fluent"]["scores"].values())],
        )

    if kind == "correct":
        return correct_scores
    if kind == "combined":
        scores = np.append(fluent_scores, correct_scores[..., None], axis=1)
        return np.exp(np.log(scores).mean(axis=1))

    error_msg = "Invalid kind. Must be 'correct' or 'combined'."
    raise ValueError(error_msg)


def get_shapley_values(features_scores: list[str, float]) -> dict[str, float]:
    """Calculate Shapley values for each feature based on scores."""
    shapley_values = {}
    unique_features = {feature for features in features_scores for feature in features}

    n = len(unique_features)

    for feature in unique_features:
        value = 0
        for subset_features, subset_score in features_scores.items():
            if feature in subset_features:
                continue

            s = len(subset_features)
            subset_with_feature = frozenset({feature}.union(subset_features))
            factor = factorial(s) * factorial(n - s - 1) / factorial(n)
            marignal_contribution = features_scores[subset_with_feature] - subset_score
            value += factor * marignal_contribution
        shapley_values[feature] = value
    return shapley_values


def get_actionable_values(
    scores: list[str, tuple[int, int]],
) -> dict[str, dict[str, float]]:
    """Calculate accuracy of feature on actionability."""
    actionable_values = {}
    for explanation_given, features_scores in scores.items():
        val = {}
        for features, (goal_correct, maneuver_correct) in features_scores.items():
            for feature in features:
                if feature not in val:
                    val[feature] = {"goal": 0, "maneuver": 0, "total": 0}
                val[feature]["total"] += 1
                val[feature]["goal"] += goal_correct
                val[feature]["maneuver"] += maneuver_correct
        for feature, sum_correct in val.items():
            total = sum_correct["total"]
            val[feature]["goal"] = sum_correct["goal"] / total
            val[feature]["maneuver"] = sum_correct["maneuver"] / total
        actionable_values[explanation_given] = val
    return actionable_values


def get_actionable_accuracy(eval_result: dict[str, Any]) -> tuple[int, int]:
    """Get actionablity test accuracy from evaluation result.

    Args:
        eval_result (dict): Evaluation result dictionary.

    """
    if isinstance(eval_result, list):
        eval_result = eval_result[0]
    actionable = eval_result["scores"]
    goal_correct = actionable["Goal"] == 0  # Correct answer is always zero
    maneuver_correct = actionable["Maneuver"] == 0

    return int(goal_correct), int(maneuver_correct)


if __name__ == "__main__":
    app()
