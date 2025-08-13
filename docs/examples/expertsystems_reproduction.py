#!/usr/bin/env python3
"""
ExpertSystems Paper Dataset Reproduction

This example reproduces the exact datasets used in the ExpertSystems
comparative study on concept drift detection methods. Demonstrates
how to create publication-quality research datasets with proper
ground truth metadata.

Expected output:
- All 7 datasets from the ExpertSystems comparative study
- Proper drift configurations matching paper specifications
- Statistical validation of dataset characteristics
- Export in formats suitable for drift detection evaluation

Reference:
"A comparative study on concept drift detectors"
Expert Systems with Applications
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drift_datasets as dd


def create_sine_dataset():
    """Reproduce the Sine dataset from ExpertSystems paper."""

    print("🌊 Creating ExpertSystems Sine dataset...")

    config = {
        "dataset": {
            "name": "expertsystems_sine",
            "type": "synthetic",
            "source": "capymoa",
            "generator": "SineGenerator",
            "description": "Sine dataset from ExpertSystems comparative study",
        },
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2, "temporal": True},
        "features": [
            {"name": "x", "type": "continuous", "role": "feature"},
            {"name": "y", "type": "continuous", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"},
        ],
        "generator_config": {
            "n_instances": 50000,
            "classification_function": 1,
            "has_noise": False,
            "balance_classes": True,
            "random_seed": 42,
        },
        "drift_config": {
            "drift_points": [10000, 25000, 40000],
            "drift_types": ["concept", "concept", "concept"],
            "drift_patterns": ["abrupt", "abrupt", "abrupt"],
            "concept_reversal": True,
        },
    }

    return dd.create_dataset(config)


def create_hyperplane_datasets():
    """Reproduce Hyperplane datasets (slow and fast rotation)."""

    print("🔄 Creating ExpertSystems Hyperplane datasets...")

    # Hyperplane with slow rotation - Hyp(0.001)
    hyp_slow_config = {
        "dataset": {
            "name": "expertsystems_hyperplane_slow",
            "type": "synthetic",
            "source": "capymoa",
            "generator": "HyperplaneGenerator",
            "description": "Slow rotating hyperplane Hyp(0.001)",
        },
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2, "temporal": True},
        "features": [{"name": f"x{i}", "type": "continuous", "role": "feature"} for i in range(1, 11)]
        + [{"name": "class", "type": "categorical", "role": "target"}],
        "generator_config": {
            "n_instances": 100000,
            "n_dimensions": 10,
            "n_drifting_dimensions": 10,
            "noise_percentage": 0.05,
            "random_seed": 42,
        },
        "drift_config": {
            "drift_types": ["concept"],
            "drift_patterns": ["continuous_gradual"],
            "rotation_speed": 0.001,
            "continuous_drift": True,
            "stable_periods": False,
        },
    }

    # Hyperplane with fast rotation - Hyp(0.1)
    hyp_fast_config = hyp_slow_config.copy()
    hyp_fast_config["dataset"]["name"] = "expertsystems_hyperplane_fast"
    hyp_fast_config["dataset"]["description"] = "Fast rotating hyperplane Hyp(0.1)"
    hyp_fast_config["drift_config"]["rotation_speed"] = 0.1

    hyp_slow = dd.create_dataset(hyp_slow_config)
    hyp_fast = dd.create_dataset(hyp_fast_config)

    return hyp_slow, hyp_fast


def create_mixed_datasets():
    """Reproduce Mixed datasets with different transition durations."""

    print("🔀 Creating ExpertSystems Mixed datasets...")

    # Mixed dataset with 200-sample transitions - Mixed(200)
    mixed_200_config = {
        "dataset": {
            "name": "expertsystems_mixed_200",
            "type": "synthetic",
            "source": "capymoa",
            "generator": "MixedGenerator",
            "description": "Mixed attributes Mixed(200)",
        },
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2, "temporal": True},
        "features": [
            {"name": "boolean_1", "type": "categorical", "role": "feature"},
            {"name": "boolean_2", "type": "categorical", "role": "feature"},
            {"name": "numeric_1", "type": "continuous", "role": "feature"},
            {"name": "numeric_2", "type": "continuous", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"},
        ],
        "generator_config": {
            "n_instances": 40000,
            "n_boolean_attributes": 2,
            "n_numeric_attributes": 2,
            "verification_conditions": 3,
            "random_seed": 42,
        },
        "drift_config": {
            "drift_points": [10000, 20000, 30000],
            "drift_types": ["concept", "concept", "concept"],
            "drift_patterns": ["intermittent_gradual", "intermittent_gradual", "intermittent_gradual"],
            "transition_durations": [200, 200, 200],
        },
    }

    # Mixed dataset with 1000-sample transitions - Mixed(1000)
    mixed_1000_config = mixed_200_config.copy()
    mixed_1000_config["dataset"]["name"] = "expertsystems_mixed_1000"
    mixed_1000_config["dataset"]["description"] = "Mixed attributes Mixed(1000)"
    mixed_1000_config["drift_config"]["transition_durations"] = [1000, 1000, 1000]

    mixed_200 = dd.create_dataset(mixed_200_config)
    mixed_1000 = dd.create_dataset(mixed_1000_config)

    return mixed_200, mixed_1000


def create_stagger_dataset():
    """Reproduce STAGGER concepts dataset."""

    print("🎯 Creating ExpertSystems STAGGER dataset...")

    config = {
        "dataset": {
            "name": "expertsystems_stagger",
            "type": "synthetic",
            "source": "capymoa",
            "generator": "STAGGERGenerator",
            "description": "STAGGER concepts dataset",
        },
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2, "temporal": True},
        "features": [
            {"name": "size", "type": "categorical", "role": "feature"},
            {"name": "color", "type": "categorical", "role": "feature"},
            {"name": "shape", "type": "categorical", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"},
        ],
        "generator_config": {"n_instances": 30000, "classification_function": 1, "balance_classes": True, "random_seed": 42},
        "drift_config": {
            "drift_points": [10000, 20000],
            "drift_types": ["concept", "concept"],
            "drift_patterns": ["abrupt", "abrupt"],
            "concept_cycle": [1, 2, 3],
        },
    }

    return dd.create_dataset(config)


def create_electricity_dataset():
    """Reproduce Electricity dataset with optional drift injection."""

    print("⚡ Creating ExpertSystems Electricity dataset...")

    config = {
        "dataset": {
            "name": "expertsystems_electricity",
            "type": "real_world",
            "source": "ucimlrepo",
            "description": "Electricity dataset from ExpertSystems study",
        },
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2, "temporal": True},
        "uci_config": {"dataset_id": 321, "as_frame": True},  # Electricity dataset ID
        # Optional synthetic drift injection
        "drift_config": {"drift_points": [15000, 30000], "drift_types": ["covariate", "concept"], "drift_patterns": ["gradual", "abrupt"]},
    }

    return dd.create_dataset(config)


def validate_expertsystems_datasets(datasets):
    """Validate that generated datasets match ExpertSystems specifications."""

    print("\n✅ Validating ExpertSystems dataset specifications...")

    # Expected specifications from the paper
    expected_specs = {
        "sine": {"size": 50000, "features": 2, "drift_points": [10000, 25000, 40000], "drift_type": "abrupt"},
        "hyperplane_slow": {"size": 100000, "features": 10, "continuous_drift": True, "rotation_speed": 0.001},
        "hyperplane_fast": {"size": 100000, "features": 10, "continuous_drift": True, "rotation_speed": 0.1},
        "mixed_200": {"size": 40000, "features": 4, "drift_points": [10000, 20000, 30000], "transition_duration": 200},
        "mixed_1000": {"size": 40000, "features": 4, "drift_points": [10000, 20000, 30000], "transition_duration": 1000},
        "stagger": {"size": 30000, "features": 3, "drift_points": [10000, 20000], "concepts": 3},
        "electricity": {"features": 8, "classes": 2, "real_world": True},
    }

    validation_results = {}

    for name, dataset in datasets.items():
        expected = expected_specs.get(name, {})
        results = {"passed": True, "checks": []}

        print(f"\n   Validating {name.upper()}:")

        # Check dataset size
        if "size" in expected:
            actual_size = len(dataset.X)
            expected_size = expected["size"]
            passed = actual_size == expected_size
            results["checks"].append(f"Size: {actual_size:,} (expected: {expected_size:,}) {'✓' if passed else '✗'}")
            if not passed:
                results["passed"] = False

        # Check number of features
        if "features" in expected:
            actual_features = dataset.X.shape[1]
            expected_features = expected["features"]
            passed = actual_features == expected_features
            results["checks"].append(f"Features: {actual_features} (expected: {expected_features}) {'✓' if passed else '✗'}")
            if not passed:
                results["passed"] = False

        # Check drift points
        if "drift_points" in expected and hasattr(dataset.drift_metadata, "drift_points"):
            actual_drift_points = dataset.drift_metadata.drift_points
            expected_drift_points = expected["drift_points"]
            passed = actual_drift_points == expected_drift_points
            results["checks"].append(f"Drift points: {actual_drift_points} (expected: {expected_drift_points}) {'✓' if passed else '✗'}")
            if not passed:
                results["passed"] = False

        # Check continuous drift
        if "continuous_drift" in expected:
            actual_continuous = getattr(dataset.drift_metadata, "continuous_drift", False)
            expected_continuous = expected["continuous_drift"]
            passed = actual_continuous == expected_continuous
            results["checks"].append(f"Continuous drift: {actual_continuous} (expected: {expected_continuous}) {'✓' if passed else '✗'}")
            if not passed:
                results["passed"] = False

        validation_results[name] = results

        # Print individual checks
        for check in results["checks"]:
            print(f"     {check}")

    # Summary
    print(f"\n   Validation Summary:")
    passed_count = sum(1 for result in validation_results.values() if result["passed"])
    total_count = len(validation_results)
    print(f"     {passed_count}/{total_count} datasets passed validation")

    return validation_results


def analyze_drift_characteristics(datasets):
    """Analyze drift characteristics across all ExpertSystems datasets."""

    print("\n📊 Analyzing drift characteristics...")

    analysis = {"dataset_summary": {}, "drift_patterns": {}, "statistical_properties": {}}

    for name, dataset in datasets.items():
        print(f"\n   {name.upper()}:")

        # Basic dataset properties
        n_samples, n_features = dataset.X.shape
        n_classes = len(dataset.y.unique())

        # Drift analysis
        if hasattr(dataset.drift_metadata, "drift_points"):
            drift_points = dataset.drift_metadata.drift_points
            drift_types = getattr(dataset.drift_metadata, "drift_types", [])
            drift_patterns = getattr(dataset.drift_metadata, "drift_patterns", [])
            n_drifts = len(drift_points)

            # Concept segments
            segments = dataset.get_concept_segments()
            avg_segment_length = np.mean([end - start for start, end in segments])

            print(f"     Samples: {n_samples:,}, Features: {n_features}, Classes: {n_classes}")
            print(f"     Drift events: {n_drifts}")
            print(f"     Drift types: {drift_types}")
            print(f"     Drift patterns: {drift_patterns}")
            print(f"     Avg segment length: {avg_segment_length:.0f}")

            analysis["dataset_summary"][name] = {
                "samples": n_samples,
                "features": n_features,
                "classes": n_classes,
                "drift_events": n_drifts,
                "avg_segment_length": avg_segment_length,
            }

        else:
            print(f"     Samples: {n_samples:,}, Features: {n_features}, Classes: {n_classes}")
            print(f"     No drift metadata (real-world dataset)")

            analysis["dataset_summary"][name] = {"samples": n_samples, "features": n_features, "classes": n_classes, "real_world": True}

        # Statistical properties
        feature_stats = dataset.X.describe()
        class_distribution = dataset.y.value_counts(normalize=True).to_dict()

        analysis["statistical_properties"][name] = {
            "feature_means": feature_stats.loc["mean"].to_dict(),
            "feature_stds": feature_stats.loc["std"].to_dict(),
            "class_distribution": class_distribution,
        }

    return analysis


def create_comparative_visualization(datasets):
    """Create comparative visualizations for all ExpertSystems datasets."""

    print("\n📈 Creating comparative visualizations...")

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle("ExpertSystems Datasets - Comparative Analysis", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    dataset_names = list(datasets.keys())

    for i, (name, dataset) in enumerate(datasets.items()):
        if i >= 9:  # Maximum 9 subplots
            break

        ax = axes[i]
        X = dataset.X
        y = dataset.y

        # For datasets with temporal dimension, show time series
        if hasattr(dataset.drift_metadata, "drift_points"):
            # Time series plot of first feature
            sample_indices = np.arange(len(X))
            if X.shape[1] > 0:
                ax.plot(sample_indices, X.iloc[:, 0], alpha=0.7, linewidth=0.5)

                # Mark drift points
                drift_points = dataset.drift_metadata.drift_points
                for drift_point in drift_points:
                    ax.axvline(drift_point, color="red", linestyle="--", alpha=0.8)

            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Feature Value")

        else:
            # For real-world datasets, show feature distribution
            if X.shape[1] > 1:
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y.astype("category").cat.codes, alpha=0.6, s=1)
                ax.set_xlabel(X.columns[0])
                ax.set_ylabel(X.columns[1])
            else:
                ax.hist(X.iloc[:, 0], bins=50, alpha=0.7)
                ax.set_xlabel(X.columns[0])
                ax.set_ylabel("Frequency")

        ax.set_title(f'{name.replace("_", " ").title()}\n({len(X):,} samples)')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(datasets), 9):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save visualization
    output_path = "docs/examples/output/expertsystems_comparative_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Comparative visualization saved to: {output_path}")

    plt.show()


def export_expertsystems_datasets(datasets, analysis):
    """Export all ExpertSystems datasets and analysis results."""

    print("\n💾 Exporting ExpertSystems datasets...")

    os.makedirs("docs/examples/output/expertsystems", exist_ok=True)

    export_summary = {"datasets_exported": [], "total_samples": 0, "export_timestamp": pd.Timestamp.now().isoformat()}

    for name, dataset in datasets.items():
        # Export dataset to CSV
        combined_data = pd.concat([dataset.X, dataset.y], axis=1)
        csv_path = f"docs/examples/output/expertsystems/{name}.csv"
        combined_data.to_csv(csv_path, index=False)

        # Export metadata
        metadata = {
            "dataset_info": {
                "name": name,
                "shape": list(dataset.X.shape),
                "feature_names": list(dataset.X.columns),
                "target_name": dataset.y.name,
                "n_classes": len(dataset.y.unique()),
            }
        }

        # Add drift metadata if available
        if hasattr(dataset.drift_metadata, "drift_points"):
            metadata["drift_metadata"] = {
                "drift_points": dataset.drift_metadata.drift_points,
                "drift_types": getattr(dataset.drift_metadata, "drift_types", []),
                "drift_patterns": getattr(dataset.drift_metadata, "drift_patterns", []),
                "concept_segments": [(int(start), int(end)) for start, end in dataset.get_concept_segments()],
            }

        metadata_path = f"docs/examples/output/expertsystems/{name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        export_summary["datasets_exported"].append(
            {"name": name, "csv_file": csv_path, "metadata_file": metadata_path, "samples": len(dataset.X)}
        )
        export_summary["total_samples"] += len(dataset.X)

        print(f"   {name}: {len(dataset.X):,} samples → {csv_path}")

    # Export analysis results
    analysis_path = "docs/examples/output/expertsystems/analysis_results.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Export summary
    summary_path = "docs/examples/output/expertsystems/export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(export_summary, f, indent=2)

    print(f"\n   Analysis results: {analysis_path}")
    print(f"   Export summary: {summary_path}")
    print(f"   Total samples exported: {export_summary['total_samples']:,}")


def main():
    """Main function for ExpertSystems dataset reproduction."""

    print("🚀 ExpertSystems Paper Dataset Reproduction")
    print("=" * 50)

    datasets = {}

    # Step 1: Create all ExpertSystems datasets
    print("📂 Creating ExpertSystems datasets...")

    datasets["sine"] = create_sine_dataset()
    datasets["hyperplane_slow"], datasets["hyperplane_fast"] = create_hyperplane_datasets()
    datasets["mixed_200"], datasets["mixed_1000"] = create_mixed_datasets()
    datasets["stagger"] = create_stagger_dataset()

    # Note: Electricity dataset requires UCI ML repo access
    try:
        datasets["electricity"] = create_electricity_dataset()
    except Exception as e:
        print(f"   ⚠️  Could not create Electricity dataset: {e}")
        print("   This is normal if UCI ML repo is not accessible")

    print(f"\n✅ Created {len(datasets)} datasets successfully!")

    # Step 2: Validate dataset specifications
    validation_results = validate_expertsystems_datasets(datasets)

    # Step 3: Analyze drift characteristics
    analysis = analyze_drift_characteristics(datasets)

    # Step 4: Create comparative visualizations
    create_comparative_visualization(datasets)

    # Step 5: Export datasets and results
    export_expertsystems_datasets(datasets, analysis)

    print("\n✅ ExpertSystems reproduction completed successfully!")
    print("\nDatasets created:")
    for name, dataset in datasets.items():
        print(f"  • {name}: {len(dataset.X):,} samples, {dataset.X.shape[1]} features")

    print("\nNext steps for research:")
    print("1. Use these datasets to evaluate drift detection algorithms")
    print("2. Compare results with the original ExpertSystems paper")
    print("3. Implement the same evaluation methodology")
    print("4. Extend the study with additional drift detectors")


if __name__ == "__main__":
    main()
