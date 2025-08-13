"""
Core plotting functions for DriftDataset visualization.

This module provides functions to visualize DriftDataset objects with
drift point annotations, feature relationships, and temporal patterns.
"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from rich.console import Console

console = Console()


def plot_dataset(
    dataset: "DriftDataset",
    features: Optional[List[str]] = None,
    plot_type: str = "auto",
    show_drift_points: bool = True,
    show_concept_segments: bool = False,
    color_by_target: bool = True,
    figsize: tuple = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Create a visualization of a DriftDataset.

    Args:
        dataset: DriftDataset to visualize
        features: List of feature names to plot. If None, uses first 2-3 features
        plot_type: Type of plot ("auto", "scatter", "time_series", "pairplot")
        show_drift_points: Whether to highlight drift points
        show_concept_segments: Whether to highlight concept segments
        color_by_target: Whether to color points by target variable
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object

    Raises:
        ValueError: If dataset is empty or invalid features specified
    """
    # Input validation
    if len(dataset.X) == 0:
        raise ValueError("Dataset is empty")

    if features is not None:
        invalid_features = [f for f in features if f not in dataset.X.columns]
        if invalid_features:
            raise ValueError(f"Feature '{invalid_features[0]}' not found in dataset")

    # Auto-select features if not specified
    if features is None:
        features = list(dataset.X.columns)[: min(3, len(dataset.X.columns))]

    # Auto-select plot type based on number of features and data characteristics
    if plot_type == "auto":
        plot_type = _auto_select_plot_type(dataset, features)

    # Create figure
    fig, axes = plt.subplots(figsize=figsize)

    # Generate the appropriate plot
    if plot_type == "scatter":
        _create_scatter_plot(dataset, features, axes, color_by_target)
    elif plot_type == "time_series":
        _create_time_series_plot(dataset, features, axes, color_by_target)
    elif plot_type == "pairplot":
        fig = _create_pairplot(dataset, features, color_by_target, figsize)
        axes = fig.get_axes()
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    # Add drift annotations
    if show_drift_points:
        _add_drift_point_annotations(dataset, axes, plot_type)

    if show_concept_segments:
        _add_concept_segment_highlights(dataset, axes, plot_type)

    # Add title and metadata
    _add_plot_metadata(dataset, fig, plot_type, features)

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]✓[/green] Plot saved to {save_path}")

    return fig


def _auto_select_plot_type(dataset: "DriftDataset", features: List[str]) -> str:
    """Automatically select the best plot type based on data characteristics."""
    n_features = len(features)
    n_samples = len(dataset.X)

    if n_features == 1:
        return "time_series"
    elif n_features == 2:
        return "scatter"
    elif n_features >= 3 and n_samples < 10000:
        return "pairplot"
    else:
        return "time_series"


def _create_scatter_plot(dataset: "DriftDataset", features: List[str], axes, color_by_target: bool) -> None:
    """Create a 2D scatter plot."""
    if len(features) < 2:
        raise ValueError("Scatter plot requires at least 2 features")

    x_data = dataset.X[features[0]]
    y_data = dataset.X[features[1]]

    if color_by_target and len(dataset.y.unique()) <= 10:
        # Color by target for categorical targets with few classes
        for target_val in sorted(dataset.y.unique()):
            mask = dataset.y == target_val
            axes.scatter(x_data[mask], y_data[mask], label=f"Class {target_val}", alpha=0.6, s=20)
        axes.legend()
    else:
        axes.scatter(x_data, y_data, alpha=0.6, s=20)

    axes.set_xlabel(features[0])
    axes.set_ylabel(features[1])
    axes.grid(True, alpha=0.3)


def _create_time_series_plot(dataset: "DriftDataset", features: List[str], axes, color_by_target: bool) -> None:
    """Create a time series plot showing features over sample index."""
    sample_indices = np.arange(len(dataset.X))

    # Plot each feature
    for i, feature in enumerate(features[:5]):  # Limit to 5 features for readability
        feature_data = dataset.X[feature]
        axes.plot(sample_indices, feature_data, label=feature, alpha=0.7)

    axes.set_xlabel("Sample Index")
    axes.set_ylabel("Feature Value")
    axes.legend()
    axes.grid(True, alpha=0.3)

    # Add target coloring as background if requested
    if color_by_target and len(dataset.y.unique()) <= 10:
        _add_target_background_coloring(dataset, axes)


def _create_pairplot(dataset: "DriftDataset", features: List[str], color_by_target: bool, figsize: tuple) -> Figure:
    """Create a pairwise plot matrix."""
    n_features = min(len(features), 4)  # Limit for readability
    selected_features = features[:n_features]

    fig, axes = plt.subplots(n_features, n_features, figsize=(figsize[0], figsize[1]))

    if n_features == 1:
        axes = np.array([[axes]])
    elif n_features == 2:
        axes = axes.reshape(2, 2)

    for i, feat_y in enumerate(selected_features):
        for j, feat_x in enumerate(selected_features):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                if color_by_target and len(dataset.y.unique()) <= 10:
                    for target_val in sorted(dataset.y.unique()):
                        mask = dataset.y == target_val
                        ax.hist(dataset.X[feat_x][mask], alpha=0.6, bins=30, label=f"Class {target_val}")
                else:
                    ax.hist(dataset.X[feat_x], alpha=0.6, bins=30)
                ax.set_xlabel(feat_x)
            else:
                # Off-diagonal: scatter plot
                x_data = dataset.X[feat_x]
                y_data = dataset.X[feat_y]

                if color_by_target and len(dataset.y.unique()) <= 10:
                    for target_val in sorted(dataset.y.unique()):
                        mask = dataset.y == target_val
                        ax.scatter(x_data[mask], y_data[mask], alpha=0.6, s=10, label=f"Class {target_val}")
                else:
                    ax.scatter(x_data, y_data, alpha=0.6, s=10)

                ax.set_xlabel(feat_x)
                ax.set_ylabel(feat_y)

            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _add_drift_point_annotations(dataset: "DriftDataset", axes, plot_type: str) -> None:
    """Add vertical lines and annotations for drift points."""
    drift_points = dataset.drift_metadata.get("drift_points", [])
    drift_types = dataset.drift_metadata.get("drift_types", [])

    if not drift_points:
        return

    # Handle different axis structures
    axes_list = [axes] if not hasattr(axes, "__iter__") else axes.flatten() if hasattr(axes, "flatten") else axes

    colors = ["red", "orange", "purple", "brown", "pink"]

    for i, point in enumerate(drift_points):
        color = colors[i % len(colors)]
        drift_type = drift_types[i] if i < len(drift_types) else "drift"

        if plot_type in ["time_series"]:
            # For time series plots, add vertical lines at sample indices
            for ax in axes_list:
                if hasattr(ax, "axvline"):
                    ax.axvline(x=point, color=color, linestyle="--", alpha=0.7, linewidth=2)
                    ax.annotate(
                        f"{drift_type.title()} Drift",
                        xy=(point, ax.get_ylim()[1] * 0.9),
                        xytext=(10, 0),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                        fontsize=8,
                    )
        elif plot_type in ["scatter"]:
            # For scatter plots, we can't meaningfully show temporal drift points
            # Instead, we'll add a text annotation about the drift points
            for ax in axes_list:
                if hasattr(ax, "text"):
                    drift_info = f"{len(drift_points)} drift points at samples: {', '.join(map(str, drift_points))}"
                    ax.text(
                        0.02,
                        0.98,
                        drift_info,
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                        verticalalignment="top",
                        fontsize=9,
                    )


def _add_concept_segment_highlights(dataset: "DriftDataset", axes, plot_type: str) -> None:
    """Add background highlights for different concept segments."""
    segments = dataset.get_concept_segments()

    if len(segments) <= 1:
        return

    # Handle different axis structures
    axes_list = [axes] if not hasattr(axes, "__iter__") else axes.flatten() if hasattr(axes, "flatten") else axes

    colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral", "lightpink"]

    for i, (start, end) in enumerate(segments):
        color = colors[i % len(colors)]

        for ax in axes_list:
            if hasattr(ax, "axvspan") and plot_type in ["time_series"]:
                ax.axvspan(start, end, alpha=0.2, color=color, label=f"Concept {i+1}")


def _add_target_background_coloring(dataset: "DriftDataset", axes) -> None:
    """Add subtle background coloring based on target variable."""
    sample_indices = np.arange(len(dataset.y))

    # Group consecutive samples with same target
    current_target = dataset.y.iloc[0]
    start_idx = 0

    colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral"]

    for i in range(1, len(dataset.y)):
        if dataset.y.iloc[i] != current_target:
            # Target changed, add background span
            color = colors[int(current_target) % len(colors)]
            axes.axvspan(start_idx, i - 1, alpha=0.1, color=color)

            current_target = dataset.y.iloc[i]
            start_idx = i

    # Add final span
    color = colors[int(current_target) % len(colors)]
    axes.axvspan(start_idx, len(dataset.y) - 1, alpha=0.1, color=color)


def _add_plot_metadata(dataset: "DriftDataset", fig: Figure, plot_type: str, features: List[str]) -> None:
    """Add title and metadata information to the plot."""
    drift_info = ""
    drift_points = dataset.drift_metadata.get("drift_points", [])
    if drift_points:
        drift_info = f" | {len(drift_points)} drift points"

    title = f"{dataset.name} ({plot_type.replace('_', ' ').title()})\n"
    title += f"{len(dataset.X)} samples, {len(features)} features{drift_info}"

    fig.suptitle(title, fontsize=14, y=0.98)
