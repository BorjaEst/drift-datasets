#!/usr/bin/env python3
"""
Basic Real-World Dataset Usage

This example demonstrates how to work with real-world datasets from the
UCI ML Repository using drift-datasets. Shows how to load, analyze, and
optionally inject synthetic drift into real-world data.

Expected output:
- Iris dataset loaded from UCI ML Repository
- Basic statistical analysis and visualization
- Optional synthetic drift injection demonstration
- Export to standard formats for further analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import drift_datasets as dd


def load_iris_dataset():
    """Load the classic Iris dataset from UCI ML Repository."""

    print("🌸 Loading Iris dataset from UCI ML Repository...")

    # Configuration for Iris dataset
    config = {
        "dataset": {
            "name": "iris_example",
            "type": "real_world",
            "source": "ucimlrepo",
            "description": "Classic Iris flower classification dataset",
        },
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 3, "temporal": False},
        "uci_config": {"dataset_id": 53, "as_frame": True},  # UCI dataset ID for Iris
    }

    # Load dataset
    dataset = dd.create_dataset(config)

    # Display basic information
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {dataset.X.shape}")
    print(f"   Features: {list(dataset.X.columns)}")
    print(f"   Target classes: {sorted(dataset.y.unique())}")
    print(f"   Class names: {dataset.y.value_counts().index.tolist()}")

    return dataset


def analyze_real_world_dataset(dataset):
    """Analyze characteristics of the real-world dataset."""

    print("\n📊 Real-World Dataset Analysis:")

    # Feature statistics
    print("   Feature Statistics:")
    print(dataset.X.describe().round(3))

    # Class distribution
    class_dist = dataset.y.value_counts().sort_index()
    print(f"\n   Class Distribution:")
    for class_val, count in class_dist.items():
        percentage = (count / len(dataset.y)) * 100
        print(f"     {class_val}: {count} samples ({percentage:.1f}%)")

    # Feature correlations
    print(f"\n   Feature Correlations:")
    corr_matrix = dataset.X.corr()
    print(corr_matrix.round(3))

    # Check for missing values
    missing_values = dataset.X.isnull().sum()
    print(f"\n   Missing Values:")
    if missing_values.sum() == 0:
        print("     No missing values found ✅")
    else:
        for feature, count in missing_values.items():
            if count > 0:
                print(f"     {feature}: {count} missing values")


def inject_synthetic_drift(dataset):
    """Demonstrate synthetic drift injection into real-world data."""

    print("\n🔄 Injecting synthetic drift into real-world dataset...")

    # Configuration for drift injection
    drift_config = {
        "dataset": {"name": "iris_with_drift", "type": "mixed", "description": "Iris dataset with synthetic drift injection"},
        "metadata": {
            "dimension": "multivariate",
            "labeling": "supervised",
            "n_classes": 3,
            "temporal": True,  # Now has temporal dimension due to drift
        },
        "real_world_config": {"base_dataset": dataset, "repeat_factor": 10},  # Repeat the dataset 10 times for more samples
        "drift_config": {
            "drift_points": [300, 600, 900, 1200],
            "drift_types": ["covariate", "concept", "covariate", "prior"],
            "drift_patterns": ["gradual", "abrupt", "gradual", "abrupt"],
            "transition_durations": [50, 0, 75, 0],
            "drift_intensities": [0.3, 0.8, 0.4, 0.6],
        },
    }

    # Create dataset with injected drift
    drift_dataset = dd.create_dataset(drift_config)

    print(f"✅ Drift injection completed!")
    print(f"   New shape: {drift_dataset.X.shape}")
    print(f"   Drift points: {drift_dataset.drift_metadata.drift_points}")
    print(f"   Drift types: {drift_dataset.drift_metadata.drift_types}")

    return drift_dataset


def visualize_real_world_dataset(original_dataset, drift_dataset=None):
    """Create visualizations for real-world dataset analysis."""

    print("\n📈 Creating visualizations...")

    if drift_dataset is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Real-World Dataset Analysis: Iris (Original vs. With Drift)", fontsize=16, fontweight="bold")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Real-World Dataset Analysis: Iris Dataset", fontsize=16, fontweight="bold")
        # Reshape for consistent indexing
        axes = axes.flatten()

    # Original dataset visualizations
    X_orig = original_dataset.X
    y_orig = original_dataset.y

    # Plot 1: Feature distributions
    ax1 = axes[0, 0] if drift_dataset else axes[0]
    feature_names = X_orig.columns

    for i, feature in enumerate(feature_names):
        ax1.hist(X_orig[feature], alpha=0.7, label=feature, bins=20)

    ax1.set_xlabel("Feature Value")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Original Dataset - Feature Distributions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Pairwise feature scatter (first two features)
    ax2 = axes[0, 1] if drift_dataset else axes[1]
    scatter = ax2.scatter(X_orig.iloc[:, 0], X_orig.iloc[:, 1], c=y_orig.astype("category").cat.codes, alpha=0.7, cmap="viridis")
    ax2.set_xlabel(feature_names[0])
    ax2.set_ylabel(feature_names[1])
    ax2.set_title("Original Dataset - Feature Space")
    plt.colorbar(scatter, ax=ax2, label="Class")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Class distribution
    ax3 = axes[0, 2] if drift_dataset else axes[2]
    class_counts = y_orig.value_counts().sort_index()
    bars = ax3.bar(range(len(class_counts)), class_counts.values, color=["skyblue", "lightcoral", "lightgreen"])
    ax3.set_xlabel("Class")
    ax3.set_ylabel("Count")
    ax3.set_title("Original Dataset - Class Distribution")
    ax3.set_xticks(range(len(class_counts)))
    ax3.set_xticklabels(class_counts.index)

    # Add value labels on bars
    for bar, value in zip(bars, class_counts.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(value), ha="center", va="bottom")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Correlation heatmap
    ax4 = axes[1, 0] if drift_dataset else axes[3]
    corr_matrix = X_orig.corr()
    im = ax4.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax4.set_xticks(range(len(feature_names)))
    ax4.set_yticks(range(len(feature_names)))
    ax4.set_xticklabels(feature_names, rotation=45)
    ax4.set_yticklabels(feature_names)
    ax4.set_title("Original Dataset - Feature Correlations")

    # Add correlation values as text
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax4.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
            )

    plt.colorbar(im, ax=ax4, label="Correlation")

    # Additional plots for drift dataset
    if drift_dataset is not None:
        X_drift = drift_dataset.X
        y_drift = drift_dataset.y
        drift_points = drift_dataset.drift_metadata.drift_points

        # Plot 5: Time series with drift points
        ax5 = axes[1, 1]
        sample_indices = np.arange(len(X_drift))

        # Plot first feature over time
        ax5.plot(sample_indices, X_drift.iloc[:, 0], alpha=0.7, linewidth=0.5, label=feature_names[0])
        ax5.plot(sample_indices, X_drift.iloc[:, 1], alpha=0.7, linewidth=0.5, label=feature_names[1])

        # Mark drift points
        for i, drift_point in enumerate(drift_points):
            ax5.axvline(drift_point, color="red", linestyle="--", alpha=0.8, label="Drift Point" if i == 0 else "")

        ax5.set_xlabel("Sample Index")
        ax5.set_ylabel("Feature Value")
        ax5.set_title("Dataset with Drift - Features Over Time")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Drift timeline
        ax6 = axes[1, 2]
        segments = drift_dataset.get_concept_segments()
        colors = ["skyblue", "lightcoral", "lightgreen", "gold", "pink"]
        drift_types = drift_dataset.drift_metadata.drift_types

        for i, (start, end) in enumerate(segments):
            duration = end - start
            color = colors[i % len(colors)]
            ax6.barh(0, duration, left=start, height=0.5, color=color, alpha=0.7, edgecolor="black", label=f"Segment {i}")

            # Add segment annotation
            mid_point = start + duration // 2
            ax6.text(mid_point, 0, f"S{i}\n{duration}", ha="center", va="center", fontweight="bold", fontsize=8)

        # Mark drift points and types
        for i, (drift_point, drift_type) in enumerate(zip(drift_points, drift_types)):
            ax6.axvline(drift_point, color="red", linestyle="--", alpha=0.8, linewidth=2)
            ax6.text(drift_point, 0.3, f"{drift_type[:4]}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="red")

        ax6.set_xlabel("Sample Index")
        ax6.set_title("Drift Timeline - Segments and Drift Types")
        ax6.set_xlim(0, len(X_drift))
        ax6.set_ylim(-0.3, 0.5)
        ax6.set_yticks([])
        ax6.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    # Save the plot
    output_filename = "real_world_with_drift.png" if drift_dataset else "real_world_analysis.png"
    output_path = f"docs/examples/output/{output_filename}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Visualization saved to: {output_path}")

    plt.show()


def export_datasets(original_dataset, drift_dataset=None):
    """Export datasets to various formats."""

    print("\n💾 Exporting datasets...")

    import os

    os.makedirs("docs/examples/output", exist_ok=True)

    # Export original dataset
    original_data = pd.concat([original_dataset.X, original_dataset.y], axis=1)
    original_csv_path = "docs/examples/output/iris_original.csv"
    original_data.to_csv(original_csv_path, index=False)
    print(f"   Original dataset exported to: {original_csv_path}")

    # Export drift dataset if available
    if drift_dataset is not None:
        drift_data = pd.concat([drift_dataset.X, drift_dataset.y], axis=1)
        drift_csv_path = "docs/examples/output/iris_with_drift.csv"
        drift_data.to_csv(drift_csv_path, index=False)
        print(f"   Drift dataset exported to: {drift_csv_path}")

        # Export drift metadata
        import json

        metadata = {
            "original_info": {
                "shape": list(original_dataset.X.shape),
                "n_classes": len(original_dataset.y.unique()),
                "feature_names": list(original_dataset.X.columns),
            },
            "drift_info": {
                "shape": list(drift_dataset.X.shape),
                "drift_points": drift_dataset.drift_metadata.drift_points,
                "drift_types": drift_dataset.drift_metadata.drift_types,
                "drift_patterns": drift_dataset.drift_metadata.drift_patterns,
                "concept_segments": [(int(start), int(end)) for start, end in drift_dataset.get_concept_segments()],
            },
        }

        metadata_path = "docs/examples/output/iris_drift_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"   Drift metadata exported to: {metadata_path}")


def main():
    """Main example function for real-world dataset usage."""

    print("🚀 Real-World Dataset Usage Example")
    print("=" * 45)

    # Step 1: Load real-world dataset
    original_dataset = load_iris_dataset()

    # Step 2: Analyze original dataset
    analyze_real_world_dataset(original_dataset)

    # Step 3: Demonstrate drift injection (optional)
    print("\n" + "=" * 45)
    user_input = input("Would you like to demonstrate drift injection? (y/n): ").lower().strip()

    if user_input in ["y", "yes"]:
        drift_dataset = inject_synthetic_drift(original_dataset)

        # Step 4: Visualize both datasets
        visualize_real_world_dataset(original_dataset, drift_dataset)

        # Step 5: Export both datasets
        export_datasets(original_dataset, drift_dataset)

    else:
        # Just visualize and export original dataset
        visualize_real_world_dataset(original_dataset)
        export_datasets(original_dataset)

    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("1. Examine the exported CSV files")
    print("2. Try loading other UCI datasets by changing the dataset_id")
    print("3. Experiment with different drift injection parameters")
    print("4. Use the datasets for drift detection algorithm evaluation")


if __name__ == "__main__":
    main()
