#!/usr/bin/env python3
"""
Drift Analysis and Visualization Example

This example demonstrates comprehensive analysis and visualization
of concept drift in datasets. Shows various techniques for understanding
drift behavior, measuring drift characteristics, and creating
publication-quality visualizations.

Expected output:
- Statistical analysis of drift events and characteristics
- Multiple visualization types for different aspects of drift
- Quantitative metrics for drift magnitude and detection
- Export of analysis results for further research
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import drift_datasets as dd

warnings.filterwarnings("ignore")


def create_analysis_dataset():
    """Create a dataset with diverse drift patterns for analysis."""

    print("🔬 Creating comprehensive dataset for drift analysis...")

    config = {
        "dataset": {
            "name": "drift_analysis_example",
            "type": "synthetic",
            "source": "capymoa",
            "generator": "HyperplaneGenerator",
            "description": "Multi-drift dataset for comprehensive analysis",
        },
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2, "temporal": True},
        "features": [{"name": f"feature_{i}", "type": "continuous", "role": "feature"} for i in range(1, 7)]
        + [{"name": "target", "type": "categorical", "role": "target"}],
        "generator_config": {
            "n_instances": 15000,
            "n_features": 6,
            "random_seed": 42,
        },
        "drift_config": {
            "drift_points": [3000, 6000, 9000, 12000],
            "drift_types": ["concept", "covariate", "concept", "prior"],
            "drift_patterns": ["abrupt", "gradual", "abrupt", "abrupt"],
            "drift_intensities": [0.7, 0.4, 0.9, 0.3],
        },
    }

    dataset = dd.create_dataset(config)

    print(f"✅ Dataset created: {dataset.X.shape[0]:,} samples, {dataset.X.shape[1]} features")
    print(f"   Drift events: {len(dataset.drift_metadata.drift_points)}")

    return dataset


def calculate_drift_metrics(dataset, window_size=300):
    """Calculate comprehensive drift detection metrics."""

    print(f"\n📊 Calculating drift metrics (window size: {window_size})...")

    X = dataset.X
    y = dataset.y
    drift_points = dataset.drift_metadata.drift_points
    drift_types = dataset.drift_metadata.drift_types

    metrics = {
        "sample_index": [],
        "statistical_distance": [],
        "feature_variance_ratio": [],
        "class_balance_change": [],
        "pca_drift_magnitude": [],
        "hellinger_distance": [],
        "ks_statistic": [],
        "concept_stability": [],
    }

    # Standardize features for consistent analysis
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Calculate metrics using sliding windows
    for i in range(window_size, len(X) - window_size, window_size // 2):
        window_center = i

        # Current window
        curr_X = X_scaled.iloc[i - window_size // 2 : i + window_size // 2]
        curr_y = y.iloc[i - window_size // 2 : i + window_size // 2]

        # Previous window for comparison
        if i >= window_size:
            prev_X = X_scaled.iloc[i - window_size : i - window_size // 2]
            prev_y = y.iloc[i - window_size : i - window_size // 2]
        else:
            prev_X = curr_X
            prev_y = curr_y

        # 1. Statistical distance (Euclidean distance between means)
        stat_distance = np.linalg.norm(curr_X.mean() - prev_X.mean())

        # 2. Feature variance ratio
        curr_var = curr_X.var().mean()
        prev_var = prev_X.var().mean()
        var_ratio = curr_var / (prev_var + 1e-8)  # Avoid division by zero

        # 3. Class balance change
        curr_balance = curr_y.value_counts(normalize=True)
        prev_balance = prev_y.value_counts(normalize=True)

        # Ensure both have same classes
        all_classes = sorted(set(curr_y) | set(prev_y))
        curr_probs = [curr_balance.get(cls, 0) for cls in all_classes]
        prev_probs = [prev_balance.get(cls, 0) for cls in all_classes]

        balance_change = np.sum(np.abs(np.array(curr_probs) - np.array(prev_probs)))

        # 4. PCA drift magnitude
        try:
            combined_X = pd.concat([prev_X, curr_X])
            pca = PCA(n_components=2)
            pca_transform = pca.fit_transform(combined_X)

            prev_pca = pca_transform[: len(prev_X)]
            curr_pca = pca_transform[len(prev_X) :]

            pca_drift = np.linalg.norm(np.mean(curr_pca, axis=0) - np.mean(prev_pca, axis=0))
        except:
            pca_drift = 0.0

        # 5. Hellinger distance (for discrete distributions)
        hellinger_dist = 0.5 * np.sum((np.sqrt(curr_probs) - np.sqrt(prev_probs)) ** 2)
        hellinger_dist = np.sqrt(hellinger_dist)

        # 6. Kolmogorov-Smirnov statistic (for first feature)
        try:
            ks_stat, _ = stats.ks_2samp(prev_X.iloc[:, 0], curr_X.iloc[:, 0])
        except:
            ks_stat = 0.0

        # 7. Concept stability (inverse of feature drift)
        stability = 1.0 / (1.0 + stat_distance)

        metrics["sample_index"].append(window_center)
        metrics["statistical_distance"].append(stat_distance)
        metrics["feature_variance_ratio"].append(var_ratio)
        metrics["class_balance_change"].append(balance_change)
        metrics["pca_drift_magnitude"].append(pca_drift)
        metrics["hellinger_distance"].append(hellinger_dist)
        metrics["ks_statistic"].append(ks_stat)
        metrics["concept_stability"].append(stability)

    return pd.DataFrame(metrics)


def analyze_drift_events(dataset, metrics_df):
    """Analyze individual drift events and their characteristics."""

    print("\n🔍 Analyzing individual drift events...")

    drift_points = dataset.drift_metadata.drift_points
    drift_types = dataset.drift_metadata.drift_types
    drift_patterns = dataset.drift_metadata.drift_patterns
    # Ensure transition_durations has the same length as drift_points
    transition_durations = dataset.drift_metadata.get("transition_durations", [])
    if len(transition_durations) != len(drift_points):
        transition_durations = [0] * len(drift_points)

    event_analysis = []

    for i, (drift_point, drift_type, pattern, duration) in enumerate(zip(drift_points, drift_types, drift_patterns, transition_durations)):
        print(f"\n   Drift Event {i+1} at sample {drift_point:,}:")
        print(f"     Type: {drift_type}")
        print(f"     Pattern: {pattern}")
        print(f"     Duration: {duration} samples")

        # Find metrics around drift point
        window_start = max(0, drift_point - 500)
        window_end = min(len(dataset.X), drift_point + duration + 500)

        event_metrics = metrics_df[(metrics_df["sample_index"] >= window_start) & (metrics_df["sample_index"] <= window_end)]

        if not event_metrics.empty:
            # Calculate event characteristics
            max_stat_dist = event_metrics["statistical_distance"].max()
            max_ks_stat = event_metrics["ks_statistic"].max()
            min_stability = event_metrics["concept_stability"].min()
            max_balance_change = event_metrics["class_balance_change"].max()

            print(f"     Max statistical distance: {max_stat_dist:.3f}")
            print(f"     Max KS statistic: {max_ks_stat:.3f}")
            print(f"     Min concept stability: {min_stability:.3f}")
            print(f"     Max class balance change: {max_balance_change:.3f}")

            event_analysis.append(
                {
                    "event_id": i + 1,
                    "drift_point": drift_point,
                    "drift_type": drift_type,
                    "pattern": pattern,
                    "duration": duration,
                    "max_statistical_distance": max_stat_dist,
                    "max_ks_statistic": max_ks_stat,
                    "min_stability": min_stability,
                    "max_balance_change": max_balance_change,
                }
            )

    return pd.DataFrame(event_analysis)


def create_comprehensive_visualizations(dataset, metrics_df, event_analysis):
    """Create comprehensive drift analysis visualizations."""

    print("\n📈 Creating comprehensive drift visualizations...")

    # Set up the plot grid
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)

    X = dataset.X
    y = dataset.y
    drift_points = dataset.drift_metadata.drift_points
    drift_types = dataset.drift_metadata.drift_types
    drift_patterns = dataset.drift_metadata.drift_patterns
    # Ensure transition_durations has the same length as drift_points
    transition_durations = dataset.drift_metadata.get("transition_durations", [])
    if len(transition_durations) != len(drift_points):
        transition_durations = [0] * len(drift_points)
    sample_indices = np.arange(len(X))

    # Color scheme for drift types
    drift_colors = {"concept": "red", "covariate": "blue", "prior": "green", "mixed": "purple"}

    # Plot 1: Feature evolution over time
    ax1 = fig.add_subplot(gs[0, :])
    for i, feature in enumerate(X.columns[:3]):  # Show first 3 features
        rolling_mean = X[feature].rolling(window=200, center=True).mean()
        ax1.plot(sample_indices, rolling_mean, label=f"{feature} (smoothed)", alpha=0.8)

    # Mark drift points
    for i, (drift_point, drift_type) in enumerate(zip(drift_points, drift_types)):
        color = drift_colors.get(drift_type, "black")
        ax1.axvline(
            drift_point,
            color=color,
            linestyle="--",
            alpha=0.8,
            label=f"{drift_type} drift" if i == 0 or drift_type not in [dt for dt in drift_types[:i]] else "",
        )

    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Feature Value")
    ax1.set_title("Feature Evolution Over Time with Drift Events")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Drift detection metrics
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(metrics_df["sample_index"], metrics_df["statistical_distance"], "b-", linewidth=2)
    for drift_point in drift_points:
        ax2.axvline(drift_point, color="red", linestyle="--", alpha=0.8)
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Statistical Distance")
    ax2.set_title("Statistical Distance (Drift Indicator)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: KS statistic
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(metrics_df["sample_index"], metrics_df["ks_statistic"], "g-", linewidth=2)
    for drift_point in drift_points:
        ax3.axvline(drift_point, color="red", linestyle="--", alpha=0.8)
    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("KS Statistic")
    ax3.set_title("Kolmogorov-Smirnov Test Statistic")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Concept stability
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(metrics_df["sample_index"], metrics_df["concept_stability"], "purple", linewidth=2)
    for drift_point in drift_points:
        ax4.axvline(drift_point, color="red", linestyle="--", alpha=0.8)
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Stability Score")
    ax4.set_title("Concept Stability Over Time")
    ax4.grid(True, alpha=0.3)

    # Plot 5: Class distribution heatmap
    ax5 = fig.add_subplot(gs[2, :])
    window_size = 200
    class_evolution = []
    window_centers = []

    for i in range(0, len(y) - window_size, window_size // 2):
        window_y = y.iloc[i : i + window_size]
        class_dist = window_y.value_counts(normalize=True)

        # Ensure consistent class order
        all_classes = sorted(y.unique())
        class_props = [class_dist.get(cls, 0) for cls in all_classes]
        class_evolution.append(class_props)
        window_centers.append(i + window_size // 2)

    class_evolution = np.array(class_evolution).T
    im = ax5.imshow(class_evolution, aspect="auto", cmap="viridis", extent=[0, len(X), 0, len(all_classes)])
    ax5.set_xlabel("Sample Index")
    ax5.set_ylabel("Class")
    ax5.set_title("Class Distribution Evolution (Darker = Higher Proportion)")
    ax5.set_yticks(range(len(all_classes)))
    ax5.set_yticklabels([f"Class {cls}" for cls in all_classes])

    # Mark drift points
    for drift_point in drift_points:
        ax5.axvline(drift_point, color="red", linestyle="--", alpha=0.8, linewidth=2)

    plt.colorbar(im, ax=ax5, label="Class Proportion")

    # Plot 6: PCA visualization by concept segments
    ax6 = fig.add_subplot(gs[3, 0])

    # Apply PCA to the entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Color by concept segments
    segments = dataset.get_concept_segments()
    segment_colors = ["skyblue", "lightcoral", "lightgreen", "gold", "pink"]

    for i, (start, end) in enumerate(segments):
        segment_X = X_pca[start:end]
        ax6.scatter(segment_X[:, 0], segment_X[:, 1], c=segment_colors[i % len(segment_colors)], alpha=0.6, s=10, label=f"Concept {i}")

    ax6.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
    ax6.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)")
    ax6.set_title("PCA Visualization by Concept Segments")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Drift event characteristics
    ax7 = fig.add_subplot(gs[3, 1])

    metrics_for_plot = ["max_statistical_distance", "max_ks_statistic", "max_balance_change"]
    x_pos = np.arange(len(event_analysis))

    for i, metric in enumerate(metrics_for_plot):
        values = event_analysis[metric].values
        ax7.bar(x_pos + i * 0.25, values, width=0.25, alpha=0.7, label=metric.replace("max_", ""))

    ax7.set_xlabel("Drift Event")
    ax7.set_ylabel("Metric Value")
    ax7.set_title("Drift Event Characteristics")
    ax7.set_xticks(x_pos + 0.25)
    ax7.set_xticklabels([f"Event {i+1}" for i in range(len(event_analysis))])
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis="y")

    # Plot 8: Feature correlation evolution
    ax8 = fig.add_subplot(gs[3, 2])

    # Calculate correlation for different periods
    correlation_evolution = []
    time_points = []

    window_size = 1000
    for i in range(0, len(X) - window_size, window_size):
        window_X = X.iloc[i : i + window_size]
        corr_matrix = window_X.corr()
        # Use mean absolute correlation as summary metric
        mean_abs_corr = corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        correlation_evolution.append(mean_abs_corr)
        time_points.append(i + window_size // 2)

    ax8.plot(time_points, correlation_evolution, "orange", linewidth=2)
    for drift_point in drift_points:
        ax8.axvline(drift_point, color="red", linestyle="--", alpha=0.8)

    ax8.set_xlabel("Sample Index")
    ax8.set_ylabel("Mean |Correlation|")
    ax8.set_title("Feature Correlation Evolution")
    ax8.grid(True, alpha=0.3)

    # Plot 9: Drift timeline with detailed annotations
    ax9 = fig.add_subplot(gs[4, :])

    # Draw concept segments
    for i, (start, end) in enumerate(segments):
        duration = end - start
        ax9.barh(
            1, duration, left=start, height=0.4, color=segment_colors[i % len(segment_colors)], alpha=0.7, edgecolor="black", linewidth=1
        )

        # Add concept label
        mid_point = start + duration // 2
        ax9.text(mid_point, 1, f"Concept {i}\n{duration:,} samples", ha="center", va="center", fontweight="bold", fontsize=9)

    # Draw drift events
    for i, (drift_point, drift_type, pattern, duration) in enumerate(zip(drift_points, drift_types, drift_patterns, transition_durations)):
        color = drift_colors.get(drift_type, "black")

        # Draw drift marker
        ax9.scatter(drift_point, 0.5, color=color, s=100, marker="v", zorder=5, edgecolor="black", linewidth=1)

        # Add drift annotation
        ax9.text(
            drift_point,
            0.2,
            f"D{i+1}\n{drift_type}\n{pattern}",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
        )

        # Draw transition period if gradual
        if duration > 0:
            ax9.barh(0.5, duration, left=drift_point, height=0.2, color=color, alpha=0.5, edgecolor="black")

    ax9.set_xlabel("Sample Index")
    ax9.set_title("Detailed Drift Timeline")
    ax9.set_xlim(0, len(X))
    ax9.set_ylim(0, 1.5)
    ax9.set_yticks([0.5, 1])
    ax9.set_yticklabels(["Transitions", "Concepts"])
    ax9.grid(True, alpha=0.3, axis="x")

    # Plot 10: Statistical significance tests
    ax10 = fig.add_subplot(gs[5, 0])

    # Perform statistical tests around drift points
    p_values = []
    drift_positions = []

    for drift_point in drift_points:
        before_start = max(0, drift_point - 500)
        before_end = drift_point
        after_start = drift_point
        after_end = min(len(X), drift_point + 500)

        if before_end > before_start and after_end > after_start:
            before_data = X.iloc[before_start:before_end, 0]
            after_data = X.iloc[after_start:after_end, 0]

            # Perform Mann-Whitney U test
            try:
                _, p_value = stats.mannwhitneyu(before_data, after_data, alternative="two-sided")
                p_values.append(-np.log10(p_value + 1e-10))  # -log10(p-value)
                drift_positions.append(drift_point)
            except:
                pass

    if p_values:
        bars = ax10.bar(range(len(p_values)), p_values, color=[drift_colors.get(dt, "gray") for dt in drift_types[: len(p_values)]])
        ax10.axhline(-np.log10(0.05), color="red", linestyle="--", alpha=0.7, label="p=0.05")
        ax10.axhline(-np.log10(0.01), color="red", linestyle="-", alpha=0.7, label="p=0.01")
        ax10.set_xlabel("Drift Event")
        ax10.set_ylabel("-log₁₀(p-value)")
        ax10.set_title("Statistical Significance of Drift Events")
        ax10.set_xticks(range(len(p_values)))
        ax10.set_xticklabels([f"Event {i+1}" for i in range(len(p_values))])
        ax10.legend()
        ax10.grid(True, alpha=0.3)

    # Plot 11: Drift detection performance simulation
    ax11 = fig.add_subplot(gs[5, 1])

    # Simulate drift detection using simple threshold on statistical distance
    threshold = metrics_df["statistical_distance"].quantile(0.95)
    detected_drifts = []

    for idx, distance in zip(metrics_df["sample_index"], metrics_df["statistical_distance"]):
        if distance > threshold:
            detected_drifts.append(idx)

    # Remove consecutive detections (keep only first in sequence)
    filtered_detections = []
    if detected_drifts:
        filtered_detections.append(detected_drifts[0])
        for detection in detected_drifts[1:]:
            if detection - filtered_detections[-1] > 200:  # Minimum gap between detections
                filtered_detections.append(detection)

    # Calculate detection performance
    true_positives = 0
    false_positives = len(filtered_detections)
    detection_delays = []

    for detection in filtered_detections:
        # Check if this detection is close to any true drift point
        for drift_point in drift_points:
            if abs(detection - drift_point) <= 300:  # Detection tolerance
                true_positives += 1
                false_positives -= 1
                detection_delays.append(abs(detection - drift_point))
                break

    false_negatives = len(drift_points) - true_positives

    # Plot detection results
    ax11.plot(metrics_df["sample_index"], metrics_df["statistical_distance"], "b-", alpha=0.7, linewidth=1)
    ax11.axhline(threshold, color="orange", linestyle="--", alpha=0.7, label=f"Threshold ({threshold:.3f})")

    # Mark true drift points
    for drift_point in drift_points:
        ax11.axvline(
            drift_point, color="green", linestyle="-", alpha=0.8, linewidth=2, label="True Drift" if drift_point == drift_points[0] else ""
        )

    # Mark detections
    for detection in filtered_detections:
        ax11.axvline(
            detection, color="red", linestyle=":", alpha=0.8, linewidth=2, label="Detection" if detection == filtered_detections[0] else ""
        )

    ax11.set_xlabel("Sample Index")
    ax11.set_ylabel("Statistical Distance")
    ax11.set_title(f"Drift Detection Simulation\nTP:{true_positives}, FP:{false_positives}, FN:{false_negatives}")
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # Plot 12: Summary statistics
    ax12 = fig.add_subplot(gs[5, 2])
    ax12.axis("off")

    # Create summary text
    summary_text = f"""
DRIFT ANALYSIS SUMMARY

Dataset Characteristics:
• Total samples: {len(X):,}
• Features: {X.shape[1]}
• Classes: {len(y.unique())}
• Drift events: {len(drift_points)}

Drift Event Types:
• Concept: {drift_types.count('concept')}
• Covariate: {drift_types.count('covariate')}
• Prior: {drift_types.count('prior')}

Detection Performance:
• True Positives: {true_positives}
• False Positives: {false_positives}
• False Negatives: {false_negatives}
• Avg Detection Delay: {np.mean(detection_delays):.0f} samples

Key Metrics:
• Max Stat Distance: {metrics_df['statistical_distance'].max():.3f}
• Max KS Statistic: {metrics_df['ks_statistic'].max():.3f}
• Min Stability: {metrics_df['concept_stability'].min():.3f}
• Max Balance Change: {metrics_df['class_balance_change'].max():.3f}
    """

    ax12.text(
        0.1,
        0.9,
        summary_text,
        transform=ax12.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.suptitle("Comprehensive Concept Drift Analysis", fontsize=20, fontweight="bold", y=0.98)

    # Save the comprehensive visualization
    output_path = "docs/examples/output/comprehensive_drift_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Comprehensive analysis saved to: {output_path}")

    plt.show()


def export_analysis_results(dataset, metrics_df, event_analysis):
    """Export all analysis results for further research."""

    print("\n💾 Exporting analysis results...")

    import os

    os.makedirs("docs/examples/output", exist_ok=True)

    # Export dataset
    combined_data = pd.concat([dataset.X, dataset.y], axis=1)
    dataset_path = "docs/examples/output/drift_analysis_dataset.csv"
    combined_data.to_csv(dataset_path, index=False)

    # Export metrics
    metrics_path = "docs/examples/output/drift_analysis_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Export event analysis
    events_path = "docs/examples/output/drift_event_analysis.csv"
    event_analysis.to_csv(events_path, index=False)

    # Export summary statistics
    import json

    summary = {
        "dataset_info": {
            "samples": len(dataset.X),
            "features": dataset.X.shape[1],
            "classes": len(dataset.y.unique()),
            "drift_events": len(dataset.drift_metadata.drift_points),
        },
        "drift_metadata": {
            "drift_points": dataset.drift_metadata.drift_points,
            "drift_types": dataset.drift_metadata.drift_types,
            "drift_patterns": dataset.drift_metadata.drift_patterns,
            "transition_durations": dataset.drift_metadata.get("transition_durations", [0] * len(dataset.drift_metadata.drift_points)),
        },
        "analysis_summary": {
            "max_statistical_distance": float(metrics_df["statistical_distance"].max()),
            "max_ks_statistic": float(metrics_df["ks_statistic"].max()),
            "min_concept_stability": float(metrics_df["concept_stability"].min()),
            "max_class_balance_change": float(metrics_df["class_balance_change"].max()),
        },
    }

    summary_path = "docs/examples/output/drift_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"   Dataset: {dataset_path}")
    print(f"   Metrics: {metrics_path}")
    print(f"   Event analysis: {events_path}")
    print(f"   Summary: {summary_path}")


def main():
    """Main function for comprehensive drift analysis example."""

    print("🚀 Comprehensive Drift Analysis Example")
    print("=" * 45)

    # Step 1: Create dataset for analysis
    dataset = create_analysis_dataset()

    # Step 2: Calculate drift metrics
    metrics_df = calculate_drift_metrics(dataset)

    # Step 3: Analyze drift events
    event_analysis = analyze_drift_events(dataset, metrics_df)

    # Step 4: Create comprehensive visualizations
    create_comprehensive_visualizations(dataset, metrics_df, event_analysis)

    # Step 5: Export results
    export_analysis_results(dataset, metrics_df, event_analysis)

    print("\n✅ Comprehensive drift analysis completed!")
    print("\nKey insights:")
    print("1. Statistical distance effectively identifies drift events")
    print("2. Different drift types show distinct metric signatures")
    print("3. Gradual drifts show extended periods of instability")
    print("4. PCA visualization reveals concept clustering patterns")
    print("5. Multiple metrics provide robust drift characterization")

    print("\nNext steps:")
    print("1. Use these metrics to train drift detection algorithms")
    print("2. Compare detection performance across different methods")
    print("3. Analyze the trade-offs between detection delay and accuracy")
    print("4. Develop adaptive threshold mechanisms for different drift types")


if __name__ == "__main__":
    main()
