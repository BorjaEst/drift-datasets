"""
Quickstart Example: Generate your first drift dataset

This example demonstrates the basic workflow for creating and using
drift datasets for machine learning research.
"""

import tempfile
from pathlib import Path

import drift_datasets


def main():
    """Run the quickstart example."""
    print("=== Drift Datasets Quickstart Example ===\n")

    # Step 1: Create a simple synthetic dataset configuration
    print("Step 1: Creating dataset configuration...")
    config = {
        "dataset": {"name": "quickstart_example", "type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "features": [
            {"name": "x", "type": "continuous", "role": "feature"},
            {"name": "y", "type": "continuous", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"},
        ],
        "generator_config": {"n_instances": 1000, "classification_function": 1, "random_seed": 42},
        "drift_config": {"drift_points": [500], "drift_types": ["concept"], "drift_patterns": ["abrupt"]},
    }
    print(f"✓ Configuration created for {config['dataset']['name']}")

    # Step 2: Generate the dataset
    print("\nStep 2: Generating dataset...")
    dataset = drift_datasets.create_dataset(config)
    print(f"✓ Dataset generated: {dataset.name}")
    print(f"  - Shape: {dataset.X.shape}")
    print(f"  - Target shape: {dataset.y.shape}")
    print(f"  - Drift points: {dataset.drift_metadata.drift_points}")

    # Step 3: Access and explore the data
    print("\nStep 3: Exploring dataset...")

    # Basic dataset information
    info = dataset.info()
    print(f"✓ Dataset info: {info['n_instances']} instances, {info['n_features']} features")

    # Feature analysis
    description = dataset.describe()
    print(f"✓ Feature analysis completed for {len(description['features'])} features")

    # Drift detection features (excludes targets, timestamps, etc.)
    modeling_features = dataset.get_drift_detection_features()
    print(f"✓ Modeling features: {list(modeling_features.columns)}")

    # Step 4: Check drift characteristics
    print("\nStep 4: Analyzing drift characteristics...")

    # Check if specific samples are drift points
    is_drift_500 = dataset.is_drift_point(500, tolerance=10)
    print(f"✓ Sample 500 is near drift: {is_drift_500}")

    # Get concept segments
    segments = dataset.get_concept_segments()
    print(f"✓ Concept segments: {len(segments)} concepts")
    for i, (start, end) in enumerate(segments):
        print(f"  - Concept {i}: samples {start}-{end} ({end-start} samples)")

    # Step 5: Save the dataset
    print("\nStep 5: Saving dataset...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_paths = dataset.save(tmp_dir, formats=["parquet", "csv", "json"])
        print(f"✓ Dataset saved in {len(saved_paths)} formats:")
        for path in saved_paths:
            print(f"  - {Path(path).name}")

        # Load it back
        parquet_path = next(p for p in saved_paths if p.endswith(".parquet"))
        loaded_dataset = drift_datasets.DriftDataset.load(parquet_path)
        print(f"✓ Dataset loaded: {loaded_dataset.name}")

    # Step 6: Prepare for machine learning
    print("\nStep 6: Preparing for ML workflows...")

    # Split dataset temporally
    train_dataset, test_dataset = dataset.split_temporal(ratio=0.7)
    print(f"✓ Temporal split: train={train_dataset.X.shape}, test={test_dataset.X.shape}")

    # Get clean features for modeling
    X_train = train_dataset.get_drift_detection_features()
    y_train = train_dataset.y
    X_test = test_dataset.get_drift_detection_features()
    y_test = test_dataset.y

    print(f"✓ Training data prepared: X={X_train.shape}, y={y_train.shape}")
    print(f"✓ Test data prepared: X={X_test.shape}, y={y_test.shape}")

    print("\n🎉 Quickstart completed! You now have a drift dataset ready for ML research.")
    print("\nNext steps:")
    print("- Try different generators: HyperplaneGenerator, STAGGERGenerator")
    print("- Experiment with drift patterns: gradual, continuous_gradual")
    print("- Use real-world datasets with UCI repository")
    print("- Create mixed datasets combining multiple sources")


if __name__ == "__main__":
    main()
