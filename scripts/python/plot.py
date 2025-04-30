"""Plotting utilities for the AXS-IGP2 project."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import typer
from matplotlib import rcParams
from matplotlib.patches import Patch, Polygon
from util import FEATURE_LABELS, MODEL_NAME_MAP

logger = logging.getLogger(__name__)


def plot_combined_scores(scores: list[dict[str, Any]], ctx: typer.Context) -> None:
    """Plot combined scores with error intervals in NeurIPS style.

    Args:
        scores (list): List of dictionaries containing scores, scenarios, and models.
        ctx (typer.Context): Typer context for scenario and model information.

    """
    # Aggregate scores by index
    combined_scores = np.array([score["score"] for score in scores])
    indices = range(combined_scores.shape[1])

    # Calculate mean and standard error for each index
    means = np.array(combined_scores).mean(axis=0)
    errors = np.std(combined_scores, axis=0) / np.sqrt(len(combined_scores))

    # Set up NeurIPS-style plot
    plt.figure(figsize=(8, 6))
    plt.plot(indices, means, label="Combined Score", color="blue", linewidth=2)
    plt.fill_between(
        indices,
        means - errors,
        means + errors,
        color="blue",
        alpha=0.2,
        label="Error Interval",
    )

    # Add labels and title
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Combined Score", fontsize=12)
    plt.title("Combined Scores Across Indices", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    # Add legend
    plt.legend(fontsize=10)

    # Save the plot as a high-resolution image
    # Tight layout for proper spacing, with more padding at the top for legend
    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.92])

    # Create directory if it doesn't exist
    save_dir = Path("output", "igp2", "plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with relevant parameters
    s_name = ctx.obj["scenario"]
    s_name = f"{s_name}" if s_name != -1 else "all"
    gen_model = ctx.obj["model"].value
    gen_model = gen_model if gen_model != "all" else "all"
    eval_model = ctx.obj["eval_model"].value
    eval_model = eval_model if eval_model != "all" else "all"
    save_name = f"evolution_s{s_name}_{eval_model}_{gen_model}"

    # Save the plot in high resolution for publication
    plt.savefig(save_dir / f"{save_name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(save_dir / f"{save_name}.pdf", bbox_inches="tight")

    plt.show()


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
    fig_width = 4.0  # Further increased width to accommodate labels and error bars
    fig_height = min(
        2.5,
        max(3.5, len(features) * 0.6),
    )  # Increased height for better spacing
    _, ax = plt.figure(figsize=(fig_width, fig_height)), plt.gca()

    # Convert feature names to human-readable labels
    readable_features = [FEATURE_LABELS.get(feature, feature) for feature in features]

    # Labels for y-axis (vertical layout)
    labels = [*readable_features, "Total"]  # Removed "Baseline"

    # Define arrow parameters - slightly smaller for academic papers
    arrow_width = 0.4  # height of the arrow body in data coordinates

    # Calculate cumulative values for waterfall plot
    cumulative = 0
    lefts = []
    widths = []
    colors = []

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
    y_positions = range(len(features))  # Adjusted to skip baseline
    plt.errorbar(
        [
            lf + w if w > 0 else lf
            for lf, w in zip(lefts[:-1], widths[:-1], strict=True)
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

    s_name = ctx.obj["scenario"]
    s_name = f"{s_name}" if s_name != -1 else "all"
    gen_model = ctx.obj["model"].value
    gen_model = gen_model if gen_model != "all" else "all"
    eval_model = ctx.obj["eval_model"].value
    eval_model = eval_model if eval_model != "all" else "all"

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
        - 0.15
    )  # Account for error bars on negative side
    x_max = (
        max(
            lefts[-1] + widths[-1],
            *[lf + w for lf, w in zip(lefts, widths, strict=True)],
        )
        + 0.15
    )  # + max(stds)
    ax.set_xlim(x_min, x_max)

    # Formatting for academic publication
    # Use a more descriptive, precise title
    ax.set_xlabel(f"Contribution to Reward ({MODEL_NAME_MAP[gen_model]})", fontsize=10)
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
            label="Total",
        ),
    ]

    add_legend = False
    if add_legend:
        # Position legend above the plot
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
    plt.tight_layout(pad=0.05, rect=[0, 0, 1, 0.92])

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
