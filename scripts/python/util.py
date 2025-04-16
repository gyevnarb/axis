"""Get all combinations of experimetal conditions."""

import enum
import json
import logging
import random
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain, combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Patch, Polygon
from scipy.special import factorial

import axs

logger = logging.getLogger(__name__)


class LLMModels(enum.Enum):
    """Enum for LLM models."""

    llama_70b = "llama-70b"
    qwen_72b = "qwen-72b"
    gpt_4o = "gpt-4o"
    gpt_o1 = "gpt-o1"
    claude_3_5 = "claude-3.5"
    claude_3_7 = "claude-3.7"
    deepseek_v3 = "deepseek-v3"
    deepseek_r1 = "deepseek-r1"


def get_agent(config: axs.Config) -> axs.AXSAgent:
    """Create an AXS agent with the given configuration."""
    env = axs.util.load_env(config.env, config.env.render_mode)
    env.reset(seed=config.env.seed)
    logger.info("Created environment %s", config.env.name)

    agent_policies = axs.registry.get(config.env.policy_type).create(env)
    return axs.AXSAgent(config, agent_policies)


def powerset(iterable: Iterable) -> list[tuple]:
    """Generate the powerset of a given iterable.

    Args:
        iterable (Iterable): The input iterable.

    """
    s = list(iterable)
    # Change iteration to range(1, len(s)+1) to exclude empty set
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_params(
    scenarios: list[int],
    complexity: list[int],
    models: list[str],
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
                    new_config_dict = deepcopy(config_dict)
                    new_config_dict["axs"]["complexity"] = c
                    new_config_dict["llm"].update(llm_config)
                    if "add_actions" not in vf and "add_macro_actions" not in vf:
                        continue
                    if use_context and vf != ():
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
                    elif not use_context and vf == ():
                        new_config_dict["axs"]["verbalizer"]["params"] = dict.fromkeys(
                            verbalizer_features,
                            False,
                        )
                        new_config_dict["axs"]["verbalizer"]["params"]["subsample"] = 3
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
                        break
    return configs


def random_order_string(items: dict[str, str]) -> str:
    """Return a random order string from a dictionary."""
    items_list = list(items.items())
    random.shuffle(items_list)
    remapped_items = {k: items_list[k][1] for k in range(len(items_list))}
    shuffle_mapping = {i: int(v[0]) for i, v in enumerate(items_list)}
    return "\n".join([f"{k}. {v}" for k, v in remapped_items.items()]), shuffle_mapping


def get_combined_score(eval_result: dict[str, Any]) -> float:
    """Get combined score for fluency and correctness using geometric mean.

    Args:
        eval_result (dict): Evaluation result dictionary.

    """
    correct = eval_result["correct"]["scores"]
    fluent = eval_result["fluent"]["scores"]
    correct = np.array(list(correct.values()))
    fluent = np.array(list(fluent.values()))

    # Calculate the geometric mean of the scores
    scores = np.concatenate((fluent, correct))
    return np.exp(np.log(scores).mean())


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
    actionable = eval_result["scores"]
    goal_correct = actionable["Goal"] == 0  # Correct answer is always zero
    maneuver_correct = actionable["Maneuver"] == 0

    return int(goal_correct), int(maneuver_correct)


def plot_shapley_waterfall(combined_shapley: dict[str, float]) -> None:
    """Create waterfall plot with arrow indicators for feature contributions."""
    # Set publication-quality plot styling
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    rcParams["font.size"] = 10
    rcParams["axes.titlesize"] = 11
    rcParams["axes.labelsize"] = 10
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 8
    rcParams["figure.dpi"] = 300

    # Define human-readable labels for features
    feature_labels = {
        "add_actions": "Raw Actions",
        "add_layout": "Road Layout",
        "add_macro_actions": "Macro Actions",
        "add_observations": "Raw Observations",
        "complexity": "High Complexity",
        "truncate": "Memory Truncation",
    }

    # Sort features by mean Shapley value
    sorted_features = sorted(
        combined_shapley.items(),
        key=lambda x: abs(x[1]["mean"]),
    )

    # Extract data for plotting
    features = [feature for feature, _ in sorted_features]
    means = [data["mean"] for _, data in sorted_features]
    stds = [data["std"] for _, data in sorted_features]

    # More appropriate figure dimensions for academic publications
    fig_width = 7.0  # Further increased width to accommodate labels and error bars
    fig_height = min(
        5.0, max(3.5, len(features) * 0.6)
    )  # Increased height for better spacing
    fig, ax = plt.figure(figsize=(fig_width, fig_height)), plt.gca()

    # Convert feature names to human-readable labels
    readable_features = [feature_labels.get(feature, feature) for feature in features]

    # Labels for y-axis (vertical layout)
    labels = ["Baseline", *readable_features, "Total"]

    # Define arrow parameters - slightly smaller for academic papers
    arrow_width = 0.4  # height of the arrow body in data coordinates

    # Calculate cumulative values for waterfall plot
    cumulative = 0
    lefts = []
    widths = []
    colors = []

    # Add baseline value at the beginning (not an arrow)
    lefts.append(0)
    widths.append(0)
    colors.append("#e0e0e0")  # Light gray is better for print

    # Process each feature with publication-friendly colors
    for mean in means:
        lefts.append(cumulative)
        widths.append(mean)
        # More muted colors for academic publications
        colors.append("#1b9e77" if mean > 0 else "#d95f02")  # ColorBrewer palette
        cumulative += mean

    # Add total value at the end
    lefts.append(0)
    widths.append(cumulative)
    colors.append("#7570b3")  # Different but harmonious color for total

    # Draw arrows for each bar
    for i, (left, width, color) in enumerate(zip(lefts, widths, colors, strict=True)):
        y_pos = i

        # Skip the baseline (it's just a point)
        if i == 0:
            continue

        # Draw rectangles with arrow heads for the rest
        if width != 0:  # Only draw arrows for non-zero values
            direction = 1 if width > 0 else -1
            abs_width = abs(width)

            # Calculate arrow head size (proportional to bar width but with min/max lims)
            head_size = min(max(abs_width * 0.1, 0.01), 0.04)

            if i < len(lefts) - 1:  # For feature bars
                # Create arrow shape using polygon vertices
                if direction > 0:  # Right-pointing arrow
                    x_points = [
                        left,
                        left + abs_width - head_size,
                        left + abs_width - head_size,
                        left + abs_width,
                        left + abs_width - head_size,
                        left + abs_width - head_size,
                        left,
                    ]
                    y_points = [
                        y_pos - arrow_width / 2,
                        y_pos - arrow_width / 2,
                        y_pos - arrow_width,
                        y_pos,
                        y_pos + arrow_width,
                        y_pos + arrow_width / 2,
                        y_pos + arrow_width / 2,
                    ]
                else:  # Left-pointing arrow
                    x_points = [
                        left,
                        left - abs_width + head_size,
                        left - abs_width + head_size,
                        left - abs_width,
                        left - abs_width + head_size,
                        left - abs_width + head_size,
                        left,
                    ]
                    y_points = [
                        y_pos - arrow_width / 2,
                        y_pos - arrow_width / 2,
                        y_pos - arrow_width,
                        y_pos,
                        y_pos + arrow_width,
                        y_pos + arrow_width / 2,
                        y_pos + arrow_width / 2,
                    ]
            else:  # For total bar (no arrow head)
                # Regular rectangle for the total
                x_points = [left, left + abs_width, left + abs_width, left]
                y_points = [
                    y_pos - arrow_width / 2,
                    y_pos - arrow_width / 2,
                    y_pos + arrow_width / 2,
                    y_pos + arrow_width / 2,
                ]

            # Create and add the polygon with thinner edge
            arrow = Polygon(
                np.column_stack([x_points, y_points]),
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,  # Thinner edge for publication
            )
            ax.add_patch(arrow)

            # Add value labels NEXT TO each arrow instead of on it
            # Only for feature arrows (not the total)
            if i < len(lefts) - 1 and abs_width > 0.01:
                value_text = f"{width:+.1%}"

                # Position text to the right of positive arrows and left of negative ars
                if width > 0:
                    text_x = left + abs_width + 0.01  # Offset to right
                    ha = "left"
                else:
                    text_x = left - 0.01  # Offset to left
                    ha = "right"

                # Black text for better readability
                plt.text(
                    text_x,
                    y_pos,
                    value_text,
                    ha=ha,
                    va="center",
                    color="black",
                    fontsize=9,
                    fontweight="bold",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.7,
                        "edgecolor": "none",
                        "pad": 1,
                    },
                )

    # Add error bars for standard deviation - thinner for academic style
    y_positions = range(1, len(features) + 1)  # Skip baseline and total
    plt.errorbar(
        [
            lf + w if w > 0 else lf
            for lf, w in zip(lefts[1:-1], widths[1:-1], strict=True)
        ],
        y_positions,
        xerr=stds,
        fmt="none",
        capsize=3,
        color="black",
        linewidth=0.8,
        capthick=0.8,
        alpha=0.7,
    )

    # Set plot limits and labels
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # Calculate adequate x-limits to make room for labels and error bars
    x_min = min(0, *lefts) - max(stds) + 0.1  # Account for error bars on negative side
    x_max = max(
        lefts[-1] + widths[-1],
        *[lf + w for lf, w in zip(lefts, widths, strict=True)],
    ) + max(stds)
    ax.set_xlim(x_min, x_max)

    # Formatting for academic publication
    # Use a more descriptive, precise title
    ax.set_xlabel("Contribution to Performance Score", fontsize=10)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Add a subtle grid
    ax.grid(visible=True, linestyle="-", alpha=0.15, axis="x")

    # Create legend with cleaner appearance - positioned at the top
    legend_elements = [
        Patch(
            facecolor="#1b9e77",
            edgecolor="black",
            linewidth=0.5,
            label="Positive Impact",
        ),
        Patch(
            facecolor="#d95f02",
            edgecolor="black",
            linewidth=0.5,
            label="Negative Impact",
        ),
        Patch(
            facecolor="#7570b3",
            edgecolor="black",
            linewidth=0.5,
            label="Total Impact",
        ),
    ]
    # Position legend above the plot
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,  # Three columns for more compact legend
        bbox_to_anchor=(0.5, 1.12),  # Position above the plot
        framealpha=0.9,
        edgecolor="lightgray",
    )

    # Show total value as text next to the total bar (only once)
    plt.text(
        cumulative + 0.01,
        len(labels) - 1,
        f"{cumulative:.1%}",
        ha="left",
        va="center",
        color="black",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
    )

    # Add a vertical line at x=0 for reference
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    # Tight layout for proper spacing, with more padding at the top for legend
    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.92])

    # Create directory if it doesn't exist
    save_dir = Path("output", "igp2", "plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with relevant parameters
    save_name = "shapley_waterfall"

    # Save the plot in high resolution for publication
    plt.savefig(save_dir / f"{save_name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(save_dir / f"{save_name}.pdf", bbox_inches="tight")

    msg = f"Saved waterfall plot to {save_dir}/{save_name}.png and .pdf"
    logger.info(msg)
    plt.close()
