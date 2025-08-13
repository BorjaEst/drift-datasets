"""
Synthetic Dataset Examples: Comprehensive guide to synthetic data generation

This example demonstrates different synthetic generators and drift patterns
available in the drift_datasets library.
"""

import tempfile
from pathlib import Path

import drift_datasets


def sine_generator_example():
    """Demonstrate SineGenerator with abrupt drift."""
    print("=== SineGenerator Example (Abrupt Drift) ===")

    config = {
        "dataset": {"name": "sine_abrupt_drift", "type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "features": [
            {"name": "x", "type": "continuous", "role": "feature"},
            {"name": "y", "type": "continuous", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"},
        ],
        "generator_config": {"n_instances": 5000, "classification_function": 1, "random_seed": 42, "has_noise": False},
        "drift_config": {"drift_points": [1000, 3000], "drift_types": ["concept", "concept"], "drift_patterns": ["abrupt", "abrupt"]},
    }

    dataset = drift_datasets.create_dataset(config)
    print(f"✓ Generated {dataset.name}: {dataset.X.shape}")
    print(f"✓ Drift points: {dataset.metadata['drift_points']}")

    # Analyze concept segments
    segments = dataset.get_concept_segments()
    for i, (start, end) in enumerate(segments):
        print(f"  - Concept {i}: samples {start}-{end}")

    return dataset


def hyperplane_generator_example():
    """Demonstrate HyperplaneGenerator with continuous gradual drift."""
    print("\n=== HyperplaneGenerator Example (Continuous Drift) ===")

    config = {
        "dataset": {"name": "hyperplane_continuous_drift", "type": "synthetic", "source": "capymoa", "generator": "HyperplaneGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "generator_config": {
            "n_instances": 10000,
            "n_dimensions": 5,
            "n_drifting_dimensions": 3,
            "noise_percentage": 0.05,
            "random_seed": 42,
        },
        "drift_config": {"drift_patterns": ["continuous_gradual"], "rotation_speed": 0.001, "continuous_drift": True},
    }

    dataset = drift_datasets.create_dataset(config)
    print(f"✓ Generated {dataset.name}: {dataset.X.shape}")
    print(f"✓ Continuous drift with rotation speed: 0.001")
    print(f"✓ Drifting dimensions: 3 out of {dataset.X.shape[1]}")

    return dataset


def stagger_generator_example():
    """Demonstrate STAGGERGenerator with multiple abrupt concept changes."""
    print("\n=== STAGGERGenerator Example (Multiple Concepts) ===")

    config = {
        "dataset": {"name": "stagger_multiple_concepts", "type": "synthetic", "source": "capymoa", "generator": "STAGGERGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "generator_config": {"n_instances": 6000, "random_seed": 42},
        "drift_config": {"drift_points": [2000, 4000], "drift_types": ["concept", "concept"], "drift_patterns": ["abrupt", "abrupt"]},
    }

    dataset = drift_datasets.create_dataset(config)
    print(f"✓ Generated {dataset.name}: {dataset.X.shape}")
    print(f"✓ STAGGER concepts with drift at: {dataset.metadata['drift_points']}")

    # Show concept changes
    segments = dataset.get_concept_segments()
    for i, (start, end) in enumerate(segments):
        concept_data = dataset.X.iloc[start:end]
        print(f"  - Concept {i}: {len(concept_data)} samples")

    return dataset


def gradual_drift_example():
    """Demonstrate gradual drift with transition periods."""
    print("\n=== Gradual Drift Example (Smooth Transitions) ===")

    config = {
        "dataset": {"name": "sine_gradual_drift", "type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "generator_config": {"n_instances": 8000, "classification_function": 1, "random_seed": 42},
        "drift_config": {
            "drift_points": [2000, 6000],
            "drift_types": ["concept", "concept"],
            "drift_patterns": ["gradual", "gradual"],
            "transition_durations": [500, 800],  # Research-friendly parameter
            "drift_intensities": [0.8, 0.6],  # Research-friendly parameter
        },
    }

    dataset = drift_datasets.create_dataset(config)
    print(f"✓ Generated {dataset.name}: {dataset.X.shape}")
    print(f"✓ Gradual drift with transition durations: [500, 800] samples")
    print(f"✓ Drift intensities: [0.8, 0.6]")

    # Analyze drift characteristics
    for i, drift_point in enumerate(dataset.metadata["drift_points"]):
        print(f"  - Drift {i+1}: starts at sample {drift_point}")
        print(f"    Duration: {config['drift_config']['transition_durations'][i]} samples")
        print(f"    Intensity: {config['drift_config']['drift_intensities'][i]}")

    return dataset


def noise_and_complexity_example():
    """Demonstrate datasets with different noise levels and complexity."""
    print("\n=== Noise and Complexity Example ===")

    # High noise dataset
    noisy_config = {
        "dataset": {"name": "noisy_hyperplane", "type": "synthetic", "source": "capymoa", "generator": "HyperplaneGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "generator_config": {
            "n_instances": 5000,
            "n_dimensions": 10,
            "n_drifting_dimensions": 10,
            "noise_percentage": 0.2,  # 20% noise
            "random_seed": 42,
        },
        "drift_config": {"drift_points": [2500], "drift_types": ["concept"], "drift_patterns": ["abrupt"]},
    }

    noisy_dataset = drift_datasets.create_dataset(noisy_config)
    print(f"✓ Noisy dataset: {noisy_dataset.name}")
    print(f"✓ High-dimensional: {noisy_dataset.X.shape[1]} dimensions")
    print(f"✓ Noise level: 20%")

    # Clean dataset for comparison
    clean_config = noisy_config.copy()
    clean_config["dataset"]["name"] = "clean_hyperplane"
    clean_config["generator_config"]["noise_percentage"] = 0.0

    clean_dataset = drift_datasets.create_dataset(clean_config)
    print(f"✓ Clean dataset: {clean_dataset.name}")
    print(f"✓ Same structure: {clean_dataset.X.shape}")
    print(f"✓ No noise: 0%")

    return noisy_dataset, clean_dataset


def expertsystems_reproduction_example():
    """Reproduce datasets from ExpertSystems comparative study."""
    print("\n=== ExpertSystems Paper Reproduction ===")

    # ExpertSystems Sine configuration
    sine_config = {
        "dataset": {"name": "expertsystems_sine", "type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "generator_config": {"n_instances": 50000, "classification_function": 1, "random_seed": 42, "has_noise": False},
        "drift_config": {
            "drift_points": [10000, 25000, 40000],
            "drift_types": ["concept", "concept", "concept"],
            "drift_patterns": ["abrupt", "abrupt", "abrupt"],
            "concept_reversal": True,  # Classification reverses after each drift
        },
    }

    sine_dataset = drift_datasets.create_dataset(sine_config)
    print(f"✓ ExpertSystems Sine: {sine_dataset.X.shape}")
    print(f"✓ Concept reversal at: {sine_dataset.metadata['drift_points']}")

    # ExpertSystems Hyperplane configuration
    hyperplane_config = {
        "dataset": {"name": "expertsystems_hyperplane", "type": "synthetic", "source": "capymoa", "generator": "HyperplaneGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "generator_config": {
            "n_instances": 100000,
            "n_dimensions": 10,
            "n_drifting_dimensions": 10,
            "noise_percentage": 0.05,
            "random_seed": 42,
        },
        "drift_config": {
            "drift_patterns": ["continuous_gradual"],
            "rotation_speed": 0.001,  # Hyp(0.001) from paper
            "continuous_drift": True,
        },
    }

    hyperplane_dataset = drift_datasets.create_dataset(hyperplane_config)
    print(f"✓ ExpertSystems Hyperplane: {hyperplane_dataset.X.shape}")
    print(f"✓ Continuous rotation at speed: 0.001")

    return sine_dataset, hyperplane_dataset


def compare_datasets_example():
    """Compare different synthetic datasets characteristics."""
    print("\n=== Dataset Comparison ===")

    datasets = []

    # Generate different datasets
    generators = ["SineGenerator", "HyperplaneGenerator", "STAGGERGenerator"]

    for generator in generators:
        config = {
            "dataset": {"name": f"comparison_{generator.lower()}", "type": "synthetic", "source": "capymoa", "generator": generator},
            "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
            "generator_config": {"n_instances": 3000, "random_seed": 42},
            "drift_config": {"drift_points": [1500], "drift_types": ["concept"], "drift_patterns": ["abrupt"]},
        }

        # Add generator-specific parameters
        if generator == "HyperplaneGenerator":
            config["generator_config"].update({"n_dimensions": 5, "n_drifting_dimensions": 3, "noise_percentage": 0.05})
        elif generator == "SineGenerator":
            config["generator_config"]["classification_function"] = 1

        dataset = drift_datasets.create_dataset(config)
        datasets.append(dataset)

        print(f"✓ {generator}: {dataset.X.shape}")

    # Compare characteristics
    print("\nDataset Characteristics:")
    for dataset in datasets:
        info = dataset.info()
        description = dataset.describe()

        print(f"\n{dataset.name}:")
        print(f"  - Instances: {info['n_instances']}")
        print(f"  - Features: {info['n_features']}")
        print(f"  - Classes: {len(dataset.y.unique())}")
        print(f"  - Drift points: {dataset.metadata['drift_points']}")
        print(f"  - Feature types: {[f['type'] for f in description['features']]}")


def main():
    """Run all synthetic dataset examples."""
    print("=== Synthetic Dataset Examples ===\n")

    # Basic generators
    sine_dataset = sine_generator_example()
    hyperplane_dataset = hyperplane_generator_example()
    stagger_dataset = stagger_generator_example()

    # Advanced drift patterns
    gradual_dataset = gradual_drift_example()
    noisy_dataset, clean_dataset = noise_and_complexity_example()

    # Research reproduction
    es_sine, es_hyperplane = expertsystems_reproduction_example()

    # Dataset comparison
    compare_datasets_example()

    print("\n🎉 All synthetic examples completed!")
    print("\nKey takeaways:")
    print("- Different generators create distinct data patterns")
    print("- Drift patterns control how concepts change over time")
    print("- Research parameters translate to implementation settings")
    print("- Noise and dimensionality affect dataset complexity")
    print("- ExpertSystems configurations enable research reproduction")

    # Save example datasets
    print("\nSaving example datasets...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        example_datasets = [
            ("sine_abrupt", sine_dataset),
            ("hyperplane_continuous", hyperplane_dataset),
            ("stagger_multiple", stagger_dataset),
            ("gradual_transitions", gradual_dataset),
        ]

        for name, dataset in example_datasets:
            paths = dataset.save(tmp_dir, formats=["parquet"])
            print(f"✓ Saved {name}: {Path(paths[0]).name}")

    print("\nNext: Try real-world datasets with 02_real_world_datasets.py")


if __name__ == "__main__":
    main()
