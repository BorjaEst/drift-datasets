#!/usr/bin/env python3
"""
Gradual Drift Pattern Example

This example demonstrates gradual drift patterns where concepts change
smoothly over a transition period. Shows how to configure, generate,
and analyze datasets with different gradual drift characteristics.

Expected output:
- Hyperplane dataset with 3 gradual drift events
- Visualization of smooth transitions between concepts
- Analysis of transition periods and drift rates
- Comparison of different transition durations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drift_datasets as dd


def create_gradual_drift_dataset():
    """Create a dataset with gradual concept drift patterns."""

    print("🔄 Creating dataset with gradual drift patterns...")

    config = {
        "dataset": {
            "name": "gradual_drift_example",
            "type": "synthetic",
            "source": "capymoa",
            "generator": "HyperplaneGenerator",
            "description": "Hyperplane with gradual concept transitions",
        },
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2, "temporal": True},
        "features": [
            {"name": "x1", "type": "continuous", "role": "feature"},
            {"name": "x2", "type": "continuous", "role": "feature"},
            {"name": "x3", "type": "continuous", "role": "feature"},
            {"name": "x4", "type": "continuous", "role": "feature"},
            {"name": "x5", "type": "continuous", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"},
        ],
        "generator_config": {
            "n_instances": 20000,
            "n_dimensions": 5,
            "n_drifting_dimensions": 3,
            "noise_percentage": 0.1,
            "random_seed": 42,
        },
        "drift_config": {
            "drift_points": [5000, 10000, 15000],
            "drift_types": ["concept", "concept", "concept"],
            "drift_patterns": ["gradual", "gradual", "gradual"],
            # Different transition durations for comparison
            "transition_durations": [500, 1000, 2000],  # Fast, medium, slow transitions
            "drift_intensities": [0.4, 0.6, 0.8],  # Increasing intensity
            "stable_periods": True,
        },
    }

    dataset = dd.create_dataset(config)

    print(f"✅ Dataset created successfully!")
    print(f"   Shape: {dataset.X.shape}")
    print(f"   Drift points: {dataset.drift_metadata.drift_points}")
    print(f"   Transition durations: {dataset.drift_metadata.transition_durations}")

    return dataset


def analyze_gradual_drift_characteristics(dataset):
    """Analyze the characteristics of gradual drift in the dataset."""

    print("\n📊 Analyzing gradual drift characteristics...")

    drift_points = dataset.drift_metadata.drift_points
    transition_durations = dataset.drift_metadata.transition_durations

    # Analyze transition periods
    print("   Transition Period Analysis:")
    for i, (drift_point, duration) in enumerate(zip(drift_points, transition_durations)):
        transition_start = drift_point
        transition_end = drift_point + duration
        transition_rate = 1.0 / duration if duration > 0 else float("inf")

        print(f"     Drift {i+1}:")
        print(f"       Start: sample {transition_start:,}")
        print(f"       End: sample {transition_end:,}")
        print(f"       Duration: {duration:,} samples")
        print(f"       Rate: {transition_rate:.6f} per sample")

    # Analyze concept segments
    segments = dataset.get_concept_segments()
    print(f"\n   Concept Segments:")
    for i, (start, end) in enumerate(segments):
        duration = end - start
        segment_type = "stable" if i == 0 or i == len(segments) - 1 else "mixed"
        print(f"     Segment {i}: samples {start:,}-{end:,} (duration: {duration:,}, type: {segment_type})")


def calculate_drift_metrics(dataset, window_size=200):
    """Calculate metrics to quantify drift behavior during transitions."""

    print(f"\n📈 Calculating drift metrics (window size: {window_size})...")

    X = dataset.X
    y = dataset.y
    drift_points = dataset.drift_metadata.drift_points
    transition_durations = dataset.drift_metadata.transition_durations

    metrics = {"sample_index": [], "concept_stability": [], "feature_variance": [], "class_distribution": [], "prediction_confidence": []}

    # Calculate metrics using sliding windows
    for i in range(0, len(X) - window_size, window_size // 4):  # 75% overlap
        window_X = X.iloc[i : i + window_size]
        window_y = y.iloc[i : i + window_size]
        window_center = i + window_size // 2

        # Concept stability (inverse of variance in features)
        feature_vars = window_X.var().mean()
        stability = 1.0 / (1.0 + feature_vars)  # Normalized stability score

        # Feature variance across window
        avg_variance = window_X.var().mean()

        # Class distribution balance (distance from 50-50)
        class_dist = window_y.value_counts(normalize=True)
        if len(class_dist) == 2:
            balance = 1.0 - abs(class_dist.iloc[0] - 0.5) * 2  # 1.0 = perfect balance
        else:
            balance = 0.0

        # Simple prediction confidence proxy (consistency of class assignment)
        # In real applications, this would use actual model predictions
        confidence = balance  # Placeholder - higher balance suggests higher confidence

        metrics["sample_index"].append(window_center)
        metrics["concept_stability"].append(stability)
        metrics["feature_variance"].append(avg_variance)
        metrics["class_distribution"].append(balance)
        metrics["prediction_confidence"].append(confidence)

    return pd.DataFrame(metrics)


def visualize_gradual_drift(dataset, metrics_df):
    """Create comprehensive visualizations of gradual drift patterns."""

    print("\n📈 Creating gradual drift visualizations...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle("Gradual Concept Drift Analysis", fontsize=16, fontweight="bold")

    X = dataset.X
    y = dataset.y
    drift_points = dataset.drift_metadata.drift_points
    transition_durations = dataset.drift_metadata.transition_durations
    sample_indices = np.arange(len(X))

    # Plot 1: Feature evolution over time
    ax1 = axes[0, 0]
    # Show first 3 features for clarity
    for i in range(min(3, X.shape[1])):
        feature_name = X.columns[i]
        # Use rolling mean for smoother visualization
        rolling_mean = X[feature_name].rolling(window=200, center=True).mean()
        ax1.plot(sample_indices, rolling_mean, label=f"{feature_name} (smoothed)", alpha=0.8)

    # Mark drift points and transition periods
    colors = ["red", "orange", "purple"]
    for i, (drift_point, duration) in enumerate(zip(drift_points, transition_durations)):
        # Transition period
        ax1.axvspan(
            drift_point,
            drift_point + duration,
            alpha=0.3,
            color=colors[i % len(colors)],
            label=f"Transition {i+1} ({duration} samples)" if i < 3 else "",
        )
        # Drift point
        ax1.axvline(drift_point, color=colors[i % len(colors)], linestyle="--", alpha=0.8, linewidth=2)

    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Feature Value")
    ax1.set_title("Feature Evolution During Gradual Drift")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Class distribution over time
    ax2 = axes[0, 1]
    window_size = 500
    class_props = []
    window_centers = []

    for i in range(0, len(y) - window_size, window_size // 2):
        window_y = y.iloc[i : i + window_size]
        prop_class_1 = (window_y == 1).mean()
        class_props.append(prop_class_1)
        window_centers.append(i + window_size // 2)

    ax2.plot(window_centers, class_props, "b-", linewidth=2, alpha=0.8, label="Class 1 Proportion")
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="Equal Distribution")

    # Mark transitions
    for i, (drift_point, duration) in enumerate(zip(drift_points, transition_durations)):
        ax2.axvspan(drift_point, drift_point + duration, alpha=0.3, color=colors[i % len(colors)])
        ax2.axvline(drift_point, color=colors[i % len(colors)], linestyle="--", alpha=0.8, linewidth=2)

    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Class 1 Proportion")
    ax2.set_title(f"Class Distribution Evolution (window={window_size})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Plot 3: Concept stability metrics
    ax3 = axes[1, 0]
    ax3.plot(metrics_df["sample_index"], metrics_df["concept_stability"], "g-", linewidth=2, alpha=0.8, label="Concept Stability")
    ax3.plot(metrics_df["sample_index"], metrics_df["class_distribution"], "orange", linewidth=2, alpha=0.8, label="Class Balance")

    # Mark transitions
    for i, (drift_point, duration) in enumerate(zip(drift_points, transition_durations)):
        ax3.axvspan(drift_point, drift_point + duration, alpha=0.3, color=colors[i % len(colors)])
        ax3.axvline(drift_point, color=colors[i % len(colors)], linestyle="--", alpha=0.8, linewidth=2)

    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("Metric Value")
    ax3.set_title("Drift Detection Metrics Over Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Plot 4: Feature variance during transitions
    ax4 = axes[1, 1]
    ax4.plot(metrics_df["sample_index"], metrics_df["feature_variance"], "purple", linewidth=2, alpha=0.8, label="Feature Variance")

    # Mark transitions with different colors for each duration
    for i, (drift_point, duration) in enumerate(zip(drift_points, transition_durations)):
        ax4.axvspan(
            drift_point, drift_point + duration, alpha=0.3, color=colors[i % len(colors)], label=f"Transition {i+1} ({duration} samples)"
        )
        ax4.axvline(drift_point, color=colors[i % len(colors)], linestyle="--", alpha=0.8, linewidth=2)

    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Feature Variance")
    ax4.set_title("Feature Variance During Transitions")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Transition comparison
    ax5 = axes[2, 0]
    transition_data = []

    for i, (drift_point, duration) in enumerate(zip(drift_points, transition_durations)):
        # Extract data from transition period
        start_idx = max(0, drift_point - 100)  # Before transition
        end_idx = min(len(X), drift_point + duration + 100)  # After transition

        transition_X = X.iloc[start_idx:end_idx]
        transition_indices = np.arange(start_idx, end_idx) - drift_point  # Relative to drift point

        # Calculate feature mean during transition
        feature_mean = transition_X.iloc[:, 0].rolling(window=50, center=True).mean()

        ax5.plot(transition_indices, feature_mean, linewidth=2, label=f"Drift {i+1} (duration: {duration})", color=colors[i % len(colors)])

        # Mark transition period
        ax5.axvspan(0, duration, alpha=0.2, color=colors[i % len(colors)])

    ax5.axvline(0, color="black", linestyle="--", alpha=0.8, label="Drift Start")
    ax5.set_xlabel("Sample Index (relative to drift point)")
    ax5.set_ylabel("Feature Mean")
    ax5.set_title("Transition Period Comparison")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Drift timeline
    ax6 = axes[2, 1]
    segments = dataset.get_concept_segments()
    segment_colors = ["lightblue", "lightcoral", "lightgreen", "gold"]

    # Draw stable periods
    for i, (start, end) in enumerate(segments):
        duration = end - start
        ax6.barh(
            0.5,
            duration,
            left=start,
            height=0.3,
            color=segment_colors[i % len(segment_colors)],
            alpha=0.7,
            edgecolor="black",
            label=f"Concept {i}" if i < 4 else "",
        )

        # Add concept label
        mid_point = start + duration // 2
        ax6.text(mid_point, 0.5, f"C{i}", ha="center", va="center", fontweight="bold", fontsize=10)

    # Draw transition periods
    for i, (drift_point, duration) in enumerate(zip(drift_points, transition_durations)):
        ax6.barh(
            0,
            duration,
            left=drift_point,
            height=0.3,
            color=colors[i % len(colors)],
            alpha=0.8,
            edgecolor="darkred",
            linewidth=2,
            label=f"Transition {i+1}" if i < 3 else "",
        )

        # Add transition label
        mid_point = drift_point + duration // 2
        ax6.text(mid_point, 0, f"T{i+1}\n{duration}", ha="center", va="center", fontweight="bold", fontsize=8, color="white")

    ax6.set_xlabel("Sample Index")
    ax6.set_title("Drift Timeline - Concepts and Transitions")
    ax6.set_xlim(0, len(X))
    ax6.set_ylim(-0.3, 0.8)
    ax6.set_yticks([0, 0.5])
    ax6.set_yticklabels(["Transitions", "Concepts"])
    ax6.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax6.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    # Save visualization
    output_path = "docs/examples/output/gradual_drift_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Visualization saved to: {output_path}")

    plt.show()


def compare_transition_speeds(dataset):
    """Compare the effects of different transition durations."""

    print("\n⚡ Comparing transition speed effects...")

    drift_points = dataset.drift_metadata.drift_points
    transition_durations = dataset.drift_metadata.transition_durations

    print("   Transition Speed Analysis:")

    for i, (drift_point, duration) in enumerate(zip(drift_points, transition_durations)):
        speed = 1.0 / duration if duration > 0 else float("inf")
        transition_type = "instantaneous" if duration == 0 else ("fast" if duration < 750 else ("medium" if duration < 1500 else "slow"))

        print(f"     Drift {i+1}:")
        print(f"       Duration: {duration:,} samples")
        print(f"       Speed: {speed:.6f} per sample")
        print(f"       Type: {transition_type}")
        print(f"       Detection difficulty: {'Easy' if duration > 1000 else 'Medium' if duration > 500 else 'Hard'}")


def main():
    """Main function demonstrating gradual drift patterns."""

    print("🚀 Gradual Drift Pattern Example")
    print("=" * 40)

    # Step 1: Create dataset with gradual drift
    dataset = create_gradual_drift_dataset()

    # Step 2: Analyze drift characteristics
    analyze_gradual_drift_characteristics(dataset)

    # Step 3: Calculate drift metrics
    metrics_df = calculate_drift_metrics(dataset)

    # Step 4: Create visualizations
    visualize_gradual_drift(dataset, metrics_df)

    # Step 5: Compare transition speeds
    compare_transition_speeds(dataset)

    # Step 6: Export results
    import os

    os.makedirs("docs/examples/output", exist_ok=True)

    # Export dataset
    combined_data = pd.concat([dataset.X, dataset.y], axis=1)
    csv_path = "docs/examples/output/gradual_drift_dataset.csv"
    combined_data.to_csv(csv_path, index=False)

    # Export metrics
    metrics_path = "docs/examples/output/gradual_drift_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n💾 Files exported:")
    print(f"   Dataset: {csv_path}")
    print(f"   Metrics: {metrics_path}")

    print("\n✅ Example completed successfully!")
    print("\nKey insights:")
    print("1. Longer transition durations create smoother concept changes")
    print("2. Feature variance increases during transition periods")
    print("3. Class distribution changes gradually during transitions")
    print("4. Different transition speeds require different detection strategies")


if __name__ == "__main__":
    main()
