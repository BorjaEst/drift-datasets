#!/usr/bin/env python3
"""
Basic Synthetic Dataset Generation

This example demonstrates how to generate simple synthetic datasets using
drift-datasets with CapyMOA generators. Shows the fundamental workflow
from configuration to dataset creation.

Expected output:
- Sine wave dataset with 3 abrupt concept drifts
- 10,000 samples with 2 features and binary target
- Ground truth drift points at samples 2500, 5000, 7500
- Visualization of the generated data and drift points
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drift_datasets as dd


def create_basic_sine_dataset():
    """Generate a simple sine dataset with abrupt drifts."""

    print("🌊 Creating basic synthetic dataset...")

    # Create configuration programmatically
    config = {
        "dataset": {
            "name": "basic_sine_example",
            "type": "synthetic",
            "source": "capymoa",
            "generator": "SineGenerator",
            "description": "Basic sine dataset for demonstration",
        },
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2, "temporal": True},
        "features": [
            {"name": "x", "type": "continuous", "role": "feature", "description": "X coordinate from sine function"},
            {"name": "y", "type": "continuous", "role": "feature", "description": "Y coordinate from cosine function"},
            {"name": "class", "type": "categorical", "role": "target", "description": "Binary classification target"},
        ],
        "generator_config": {
            "n_instances": 10000,
            "classification_function": 1,
            "has_noise": False,
            "balance_classes": True,
            "random_seed": 42,
        },
        "drift_config": {
            "drift_points": [2500, 5000, 7500],
            "drift_types": ["concept", "concept", "concept"],
            "drift_patterns": ["abrupt", "abrupt", "abrupt"],
            "stable_periods": True,
        },
    }

    # Generate dataset
    dataset = dd.create_dataset(config)

    # Display basic information
    print(f"✅ Dataset created successfully!")
    print(f"   Shape: {dataset.X.shape}")
    print(f"   Features: {list(dataset.X.columns)}")
    print(f"   Target classes: {dataset.y.unique()}")
    print(f"   Drift points: {dataset.drift_metadata.drift_points}")

    return dataset


def analyze_dataset(dataset):
    """Analyze the generated dataset characteristics."""

    print("\n📊 Dataset Analysis:")

    # Basic statistics
    print("   Feature Statistics:")
    print(dataset.X.describe().round(3))

    # Class distribution
    class_dist = dataset.y.value_counts().sort_index()
    print(f"\n   Class Distribution:")
    for class_val, count in class_dist.items():
        percentage = (count / len(dataset.y)) * 100
        print(f"     Class {class_val}: {count:,} samples ({percentage:.1f}%)")

    # Drift segments
    segments = dataset.get_concept_segments()
    print(f"\n   Concept Segments:")
    for i, (start, end) in enumerate(segments):
        duration = end - start
        print(f"     Segment {i}: samples {start:,}-{end:,} (duration: {duration:,})")


def visualize_dataset(dataset):
    """Create visualizations of the dataset and drift points."""

    print("\n📈 Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Basic Synthetic Dataset - Sine Wave with Concept Drift", fontsize=16, fontweight="bold")

    # Extract data
    X = dataset.X
    y = dataset.y
    drift_points = dataset.drift_metadata.drift_points

    # Plot 1: Feature space colored by class
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X["x"], X["y"], c=y, alpha=0.6, cmap="viridis", s=1)
    ax1.set_xlabel("X (sine)")
    ax1.set_ylabel("Y (cosine)")
    ax1.set_title("Feature Space (colored by class)")
    plt.colorbar(scatter, ax=ax1, label="Class")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time series of features with drift points
    ax2 = axes[0, 1]
    sample_indices = np.arange(len(X))
    ax2.plot(sample_indices, X["x"], label="X feature", alpha=0.7, linewidth=0.5)
    ax2.plot(sample_indices, X["y"], label="Y feature", alpha=0.7, linewidth=0.5)

    # Mark drift points
    for i, drift_point in enumerate(drift_points):
        ax2.axvline(drift_point, color="red", linestyle="--", alpha=0.8, label="Drift Point" if i == 0 else "")

    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Feature Value")
    ax2.set_title("Feature Values Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Class distribution over time
    ax3 = axes[1, 0]
    window_size = 500
    class_proportions = []
    window_centers = []

    for i in range(0, len(y) - window_size, window_size // 2):
        window_y = y.iloc[i : i + window_size]
        prop_class_1 = (window_y == 1).mean()
        class_proportions.append(prop_class_1)
        window_centers.append(i + window_size // 2)

    ax3.plot(window_centers, class_proportions, "b-", linewidth=2, label="Class 1 Proportion")
    ax3.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="Equal Classes")

    # Mark drift points
    for i, drift_point in enumerate(drift_points):
        ax3.axvline(drift_point, color="red", linestyle="--", alpha=0.8, label="Drift Point" if i == 0 else "")

    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("Class 1 Proportion")
    ax3.set_title(f"Class Distribution Over Time (window={window_size})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)

    # Plot 4: Concept segments visualization
    ax4 = axes[1, 1]
    segments = dataset.get_concept_segments()
    colors = ["skyblue", "lightcoral", "lightgreen", "gold"]

    for i, (start, end) in enumerate(segments):
        duration = end - start
        ax4.barh(0, duration, left=start, height=0.5, color=colors[i % len(colors)], alpha=0.7, label=f"Concept {i}", edgecolor="black")

        # Add text annotation
        mid_point = start + duration // 2
        ax4.text(mid_point, 0, f"C{i}\n{duration:,}", ha="center", va="center", fontweight="bold")

    # Mark drift points
    for drift_point in drift_points:
        ax4.axvline(drift_point, color="red", linestyle="--", alpha=0.8, linewidth=2)

    ax4.set_xlabel("Sample Index")
    ax4.set_title("Concept Segments Timeline")
    ax4.set_xlim(0, len(X))
    ax4.set_ylim(-0.3, 0.3)
    ax4.set_yticks([])
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    # Save the plot
    output_path = "docs/examples/output/basic_synthetic_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Visualization saved to: {output_path}")

    plt.show()


def export_dataset(dataset):
    """Export the dataset to various formats."""

    print("\n💾 Exporting dataset...")

    # Create output directory
    import os

    os.makedirs("docs/examples/output", exist_ok=True)

    # Export to CSV
    csv_path = "docs/examples/output/basic_synthetic_dataset.csv"
    combined_data = pd.concat([dataset.X, dataset.y], axis=1)
    combined_data.to_csv(csv_path, index=False)
    print(f"   Dataset exported to CSV: {csv_path}")

    # Export metadata as JSON
    import json

    metadata_path = "docs/examples/output/basic_synthetic_metadata.json"

    metadata = {
        "dataset_info": {
            "name": "basic_sine_example",
            "shape": list(dataset.X.shape),
            "n_classes": len(dataset.y.unique()),
            "feature_names": list(dataset.X.columns),
            "target_name": dataset.y.name,
        },
        "drift_metadata": {
            "drift_points": dataset.drift_metadata.drift_points,
            "drift_types": dataset.drift_metadata.drift_types,
            "drift_patterns": dataset.drift_metadata.drift_patterns,
            "n_concepts": len(dataset.get_concept_segments()),
            "concept_segments": [(int(start), int(end)) for start, end in dataset.get_concept_segments()],
        },
        "feature_statistics": dataset.X.describe().to_dict(),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata exported to JSON: {metadata_path}")


def main():
    """Main example function demonstrating basic synthetic dataset generation."""

    print("🚀 Basic Synthetic Dataset Generation Example")
    print("=" * 50)

    # Step 1: Create dataset
    dataset = create_basic_sine_dataset()

    # Step 2: Analyze dataset characteristics
    analyze_dataset(dataset)

    # Step 3: Visualize the data and drift patterns
    visualize_dataset(dataset)

    # Step 4: Export for further analysis
    export_dataset(dataset)

    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("1. Examine the generated CSV file with your preferred data analysis tool")
    print("2. Review the metadata JSON for drift detection evaluation")
    print("3. Use the visualization to understand drift behavior")
    print("4. Try modifying the configuration to experiment with different drift patterns")


if __name__ == "__main__":
    main()
