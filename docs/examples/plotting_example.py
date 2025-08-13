"""
Plotting Example: Visualize drift datasets

This example demonstrates how to visualize DriftDataset objects with
different plot types and options to understand data and drift patterns.
"""

from pathlib import Path

import drift_datasets


def main():
    """Run the plotting example."""
    print("=== Drift Datasets Plotting Example ===\n")

    # Step 1: Create a sample dataset with drift
    print("Step 1: Creating dataset with multiple drift points...")
    config = {
        "dataset": {"name": "plotting_example", "type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2, "temporal": True},
        "features": [
            {"name": "x", "type": "continuous", "role": "feature", "description": "Sine-based feature"},
            {"name": "y", "type": "continuous", "role": "feature", "description": "Cosine-based feature"},
            {"name": "class", "type": "categorical", "role": "target", "description": "Binary classification"},
        ],
        "generator_config": {"n_instances": 2000, "classification_function": 1, "random_seed": 42},
        "drift_config": {
            "drift_points": [500, 1000, 1500],
            "drift_types": ["concept", "concept", "concept"],
            "drift_patterns": ["abrupt", "gradual", "abrupt"],
            "drift_intensities": [0.8, 0.6, 0.9],
        },
    }

    dataset = drift_datasets.create_dataset(config)
    print(f"✓ Dataset created: {dataset.X.shape} features, {len(dataset.drift_metadata.drift_points)} drift points")

    # Step 2: Create different types of plots
    print("\nStep 2: Creating different plot types...")

    # Save plots in the current directory
    output_dir = Path(".")

    # 2a: Basic scatter plot with drift points
    print("  • Creating 2D scatter plot...")
    fig1 = drift_datasets.plot_dataset(
        dataset,
        plot_type="scatter",
        show_drift_points=True,
        color_by_target=True,
        save_path=output_dir / "scatter_plot.png",
    )
    print(f"    ✓ Saved scatter plot: scatter_plot.png")

    # 2b: Time series plot showing features over time
    print("  • Creating time series plot...")
    fig2 = drift_datasets.plot_dataset(
        dataset,
        plot_type="time_series",
        show_drift_points=True,
        show_concept_segments=True,
        save_path=output_dir / "time_series_plot.png",
    )
    print(f"    ✓ Saved time series plot: time_series_plot.png")

    # 2c: Auto-selected plot type
    print("  • Creating auto-selected plot...")
    fig3 = drift_datasets.plot_dataset(
        dataset,
        plot_type="auto",  # Let the function choose the best plot type
        show_drift_points=True,
        save_path=output_dir / "auto_plot.png",
    )
    print(f"    ✓ Saved auto plot: auto_plot.png")

    # 2d: Plot specific features only
    print("  • Creating plot with selected features...")
    fig4 = drift_datasets.plot_dataset(
        dataset,
        features=["feature_0"],  # Plot only first feature
        plot_type="time_series",
        show_drift_points=True,
        save_path=output_dir / "single_feature_plot.png",
    )
    print(f"    ✓ Saved single feature plot: single_feature_plot.png")

    print(f"\n✓ All plots saved to: {Path.cwd()}")

    # Step 3: Demonstrate with different dataset types
    print("\nStep 3: Plotting different dataset types...")

    # 3a: Hyperplane dataset (high-dimensional)
    hyperplane_config = {
        "dataset": {"name": "hyperplane_viz", "type": "synthetic", "source": "capymoa", "generator": "HyperplaneGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "features": [{"name": f"dim_{i}", "type": "continuous", "role": "feature"} for i in range(5)]
        + [{"name": "target", "type": "categorical", "role": "target"}],
        "generator_config": {"n_instances": 1000, "n_features": 5, "random_seed": 42},
        "drift_config": {"drift_points": [500], "drift_types": ["concept"], "drift_patterns": ["gradual"]},
    }

    hyperplane_dataset = drift_datasets.create_dataset(hyperplane_config)

    # For high-dimensional data, the function will auto-select time series
    fig_hd = drift_datasets.plot_dataset(
        hyperplane_dataset,
        plot_type="auto",
        show_drift_points=True,
        color_by_target=True,
    )
    print("  ✓ Created high-dimensional dataset plot (auto-selected time series)")

    # Step 4: Show plot customization options
    print("\nStep 4: Demonstrating plot customization...")

    # Custom figure size and no color coding
    fig_custom = drift_datasets.plot_dataset(
        dataset,
        plot_type="scatter",
        show_drift_points=True,
        show_concept_segments=False,
        color_by_target=False,  # All points same color
        figsize=(15, 10),  # Custom figure size
    )
    print("  ✓ Created custom-sized plot without target coloring")

    # Step 5: Analysis workflow integration
    print("\nStep 5: Integration with analysis workflow...")

    # Split the dataset and plot both parts
    train_data, test_data = dataset.split_temporal(ratio=0.7)

    fig_train = drift_datasets.plot_dataset(train_data, plot_type="scatter", show_drift_points=True, color_by_target=True)
    print(f"  ✓ Created training set plot: {train_data.X.shape} samples")

    fig_test = drift_datasets.plot_dataset(test_data, plot_type="scatter", show_drift_points=True, color_by_target=True)
    print(f"  ✓ Created test set plot: {test_data.X.shape} samples")

    # Step 6: Show different visualization for drift analysis
    print("\nStep 6: Drift-specific visualizations...")

    # Get concept segments for analysis
    segments = dataset.get_concept_segments()
    print(f"  • Dataset has {len(segments)} concept segments:")
    for i, (start, end) in enumerate(segments):
        print(f"    - Segment {i+1}: samples {start}-{end} ({end-start+1} samples)")

    # Visualize concept segments
    fig_segments = drift_datasets.plot_dataset(
        dataset,
        plot_type="time_series",
        show_drift_points=False,  # Hide individual drift points
        show_concept_segments=True,  # Highlight concept segments instead
        color_by_target=True,
    )
    print("  ✓ Created concept segments visualization")

    print("\n🎉 Plotting example completed!")
    print("\nKey features demonstrated:")
    print("• Multiple plot types: scatter, time_series, auto-selection")
    print("• Drift point annotations with different styles")
    print("• Concept segment highlighting")
    print("• Target-based color coding")
    print("• Feature selection and customization")
    print("• Integration with dataset analysis workflow")
    print("\nTip: Use plot_type='auto' to let the function choose the best visualization!")


if __name__ == "__main__":
    main()
