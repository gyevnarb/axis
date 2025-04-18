"""Get all combinations of experimetal conditions."""

import enum
import json
import logging
import pickle
import random
import re
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain, combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import typer
from matplotlib import rcParams
from matplotlib.patches import Patch, Polygon
from scipy.special import factorial

import axs

logger = logging.getLogger(__name__)
app = typer.Typer()


MODEL_NAME_MAP = {
    "llama70b": "LLaMA-3.3 70B",
    "qwen72b": "Qwen-2.5 72B",
    "gpt41": "GPT-4.1",
    "gpt4o": "GPT-4o",
    "gpt4omini": "GPT-4o-mini",
    "o1": "o1",
    "claude35": "Claude 3.5",
    "claude37": "Claude 3.7",
    "deepseekv3": "DeepSeek-V3",
    "deepseekr1": "DeepSeek-R1",
}

FEATURE_LABELS = {
    "add_actions": "Raw Actions",
    "add_layout": "Road Layout",
    "add_macro_actions": "Macro Actions",
    "add_observations": "Raw Observations",
    "complexity": "High Complexity",
    "truncate": "Memory Truncation",
}


class LLMModels(enum.Enum):
    """Enum for LLM models."""

    llama_70b = "llama70b"
    qwen_72b = "qwen72b"
    gpt_4_1 = "gpt41"
    gpt_4o = "gpt4o"
    gpt_4o_mini = "gpt4omini"
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


def get_save_paths(ctx: typer.Context) -> list[Path]:
    """Load results from the specified directory.

    Args:
        ctx (typer.Context): Typer context.
        eval_model (LLMModels): Evaluation model.

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


def get_params(
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
                    if "add_actions" not in vf and "add_macro_actions" not in vf:
                        continue
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
    return correct[0]
    return np.sqrt(correct[0] * fluent.mean())


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


def plot_shapley_waterfall(
    combined_shapley: dict[str, float],
    ctx: typer.Context,
) -> None:
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
        5.0,
        max(3.5, len(features) * 0.6),
    )  # Increased height for better spacing
    fig, ax = plt.figure(figsize=(fig_width, fig_height)), plt.gca()

    # Convert feature names to human-readable labels
    readable_features = [FEATURE_LABELS.get(feature, feature) for feature in features]

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

    eps = 0.001  # Small offset to avoid overlap with the arrow

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

            # Calculate arrow head size (prop to bar width but with min/max lims)
            head_size = max(abs_width * 0.1, eps)

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
                if direction > 0:
                    x_points = [left, left + abs_width, left + abs_width, left]
                else:
                    x_points = [left, left - abs_width, left - abs_width, left]
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
            if i < len(lefts) - 1 and abs_width > eps:
                value_text = f"{width:+.1%}"

                # Position text to the right of positive arrows and left of negative ars
                if width > 0:
                    text_x = left + abs_width + 0.01  # Offset to right
                    ha = "left"
                else:
                    text_x = left - abs_width - 0.01  # Offset to left
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
    x_min = (
        min(
            lefts[-1] + widths[-1],
            *[lf + w for lf, w in zip(lefts, widths, strict=True)],
        )
        - 0.12
    )  # Account for error bars on negative side
    x_max = (
        max(
            lefts[-1] + widths[-1],
            *[lf + w for lf, w in zip(lefts, widths, strict=True)],
        )
        + 0.12
    )  # + max(stds)
    ax.set_xlim(x_min, x_max)

    # Formatting for academic publication
    # Use a more descriptive, precise title
    ax.set_xlabel("Contribution to Reward", fontsize=10)
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
    s_name = ctx.obj["scenario"]
    s_name = f"{s_name}" if s_name != -1 else "all"
    gen_model = ctx.obj["model"].value
    gen_model = gen_model if gen_model != "all" else "all"
    eval_model = ctx.obj["eval_model"].value
    eval_model = eval_model if eval_model != "all" else "all"
    legend_title = (
        f"Scenario: {s_name}; "
        f"Eval: {MODEL_NAME_MAP.get(eval_model, eval_model)}; "
        f"Gen: {MODEL_NAME_MAP.get(gen_model, gen_model)}"
    )

    legend = ax.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,  # Three columns for more compact legend
        bbox_to_anchor=(0.5, 1.22),  # Position above the plot
        framealpha=0.9,
        title=legend_title,
        edgecolor="lightgray",
    )
    plt.setp(plt.setp(legend.get_title(), fontsize="small"))

    # Show total value as text next to the total bar
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
    save_name = f"shapley_s{s_name}_{eval_model}_{gen_model}"

    # Save the plot in high resolution for publication
    plt.savefig(save_dir / f"{save_name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(save_dir / f"{save_name}.pdf", bbox_inches="tight")

    msg = f"Saved waterfall plot to {save_dir}/{save_name}.png and .pdf"
    logger.info(msg)
    plt.close()


def plot_actionable_barplot(
    combined_actionable: dict[str, dict[str, float]],
    ctx: typer.Context,
) -> None:
    """Plot actionable values for features with and without explanations as a barplot.

    Args:
        combined_actionable: Dictionary with feature actionability scores
        ctx: Typer context for scenario and model information.

    """
    # Set publication-quality plot styling
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.dpi"] = 300

    # Get unique features from both conditions
    all_features = set()
    for explanation_given in ["actionable_exp", "actionable_no_exp"]:
        all_features.update(combined_actionable[explanation_given].keys())
    all_features = list(all_features)

    # Define human-readable labels for features
    feature_labels = {
        "add_actions": "Raw\nActions",
        "add_layout": "Road\nLayout",
        "add_macro_actions": "Macro\nActions",
        "add_observations": "Raw\nObservations",
        "complexity": "High\nComplexity",
        "truncate": "Memory\nTruncation",
    }

    # Calculate differences for ordering (average of goal and maneuver differences)
    differences = []
    for feature in all_features:
        # Get mean scores for both conditions and both metrics
        goal_exp = combined_actionable["actionable_exp"].get(
            feature,
            {"goal": {"mean": 0}},
        )["goal"]["mean"]
        goal_no_exp = combined_actionable["actionable_no_exp"].get(
            feature,
            {"goal": {"mean": 0}},
        )["goal"]["mean"]

        maneuver_exp = combined_actionable["actionable_exp"].get(
            feature,
            {"maneuver": {"mean": 0}},
        )["maneuver"]["mean"]
        maneuver_no_exp = combined_actionable["actionable_no_exp"].get(
            feature,
            {"maneuver": {"mean": 0}},
        )["maneuver"]["mean"]

        # Calculate average difference
        avg_diff = ((goal_exp - goal_no_exp) + (maneuver_exp - maneuver_no_exp)) / 2
        differences.append((feature, avg_diff))

    # Sort features by difference (descending)
    sorted_features = [f for f, _ in sorted(differences, key=lambda x: -x[1])]

    # Create human-readable labels for x-axis
    readable_features = [feature_labels.get(f, f) for f in sorted_features]

    # Prepare data for plotting - sizing for academic paper format
    fig_width = 7.0  # Width in inches (standard journal column width)
    fig_height = 6.0  # Height in inches

    fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    width = 0.35  # width of the bars
    x = np.arange(len(sorted_features))

    # Use publication-friendly colors
    color_exp = "#1b9e77"  # ColorBrewer green - "With Explanation"
    color_no_exp = "#d95f02"  # ColorBrewer orange - "No Explanation"
    color_diff = "#7570b3"  # ColorBrewer purple - "Difference"

    # Plot goal and maneuver scores separately
    for i, key in enumerate(["goal", "maneuver"]):
        means_exp = [
            combined_actionable["actionable_exp"].get(f, {key: {"mean": 0}})[key][
                "mean"
            ]
            for f in sorted_features
        ]
        stds_exp = [
            combined_actionable["actionable_exp"].get(f, {key: {"std": 0}})[key]["std"]
            for f in sorted_features
        ]

        means_no_exp = [
            combined_actionable["actionable_no_exp"].get(f, {key: {"mean": 0}})[key][
                "mean"
            ]
            for f in sorted_features
        ]
        stds_no_exp = [
            combined_actionable["actionable_no_exp"].get(f, {key: {"std": 0}})[key][
                "std"
            ]
            for f in sorted_features
        ]

        # Calculate proper y-axis limit to ensure difference labels are visible
        max_height = max(
            [
                max_val + std + 0.15  # Add padding of 0.15 for academic formatting
                for max_val, std in zip(
                    np.maximum(means_exp, means_no_exp),
                    np.maximum(stds_exp, stds_no_exp),
                    strict=True,
                )
            ],
        )

        # Plot bars with thinner error bars
        axs[i].bar(
            x - width / 2,
            means_exp,
            width,
            label="With Explanation",
            yerr=stds_exp,
            capsize=3,
            color=color_exp,
            edgecolor="black",
            linewidth=0.5,
        )
        axs[i].bar(
            x + width / 2,
            means_no_exp,
            width,
            label="No Explanation",
            yerr=stds_no_exp,
            capsize=3,
            color=color_no_exp,
            edgecolor="black",
            linewidth=0.5,
        )

        # Set y-limit to make room for difference annotations
        axs[i].set_ylim(0, max_height)

        # Add difference values as text with connecting lines
        for j in range(len(sorted_features)):
            diff = means_exp[j] - means_no_exp[j]
            # Position the text at a fixed height within the visible area
            text_height = 0.9 * max_height

            # Add a line connecting the two bars
            line_height = max(means_exp[j], means_no_exp[j]) + max(
                stds_exp[j],
                stds_no_exp[j],
            )
            axs[i].plot(
                [j - width / 2, j + width / 2],
                [line_height, line_height],
                "k-",
                linewidth=0.5,
            )
            axs[i].plot(
                [j, j],
                [line_height, text_height],
                "k--",
                linewidth=0.5,
                alpha=0.6,
            )

            # Add the difference text with a subtle colored background
            axs[i].text(
                j,
                text_height,
                f"Δ: {diff:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                bbox={
                    "facecolor": color_diff,
                    "alpha": 0.2,
                    "edgecolor": "gray",
                    "boxstyle": "round,pad=0.2",
                    "linewidth": 0.5,
                },
            )

        s_name = ctx.obj["scenario"]
        s_name = f"{s_name}" if s_name != -1 else "all"
        gen_model = ctx.obj["model"].value
        gen_model = gen_model if gen_model != "all" else "all"
        eval_model = ctx.obj["eval_model"].value
        eval_model = eval_model if eval_model != "all" else "all"

        axs[i].set_ylabel(f"{key.capitalize()} Accuracy")
        axs[i].set_title(f"{key.capitalize()} Actionability by Feature")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(readable_features, rotation=45, ha="right")
        axs[i].grid(
            axis="y",
            linestyle="-",
            alpha=0.15,
        )  # Subtle grid for academic style

    # Create custom legend with only the needed entries
    handles = [
        Patch(
            facecolor=color_exp,
            edgecolor="black",
            linewidth=0.5,
            label="With Explanation",
        ),
        Patch(
            facecolor=color_no_exp,
            edgecolor="black",
            linewidth=0.5,
            label="No Explanation",
        ),
        Patch(
            facecolor=color_diff,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.2,
            label="Difference (Δ)",
        ),
    ]

    # Position legend above the plot in a single row (better for academic papers)
    legend_title = (
        f"Scenario: {s_name}; Eval model: {MODEL_NAME_MAP.get(eval_model, eval_model)};"
        f" Gen model: {MODEL_NAME_MAP.get(gen_model, gen_model)}"
)
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
        title=legend_title,
    )
    plt.setp(plt.setp(legend.get_title(), fontsize="small"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the legend

    # Save plot to output directory
    output_dir = Path("output", "igp2", "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_name = f"actionable_s{s_name}_{eval_model}_{gen_model}"

    # Save in both PNG and PDF formats for academic publishing
    plt.savefig(
        output_dir / f"{save_name}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.savefig(output_dir / f"{save_name}.pdf", bbox_inches="tight")

    logger.info(
        "Actionability barplot saved to %s (.png and .pdf)",
        output_dir / f"{save_name}",
    )
    plt.close()


if __name__ == "__main__":
    app()
