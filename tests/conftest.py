"""
Test configuration and shared fixtures for drift-datasets.

This module provides session-scoped fixtures for CapyMOA mocks, UCI repository mocks,
test data, and common test utilities used across all test categories.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import toml


@pytest.fixture(scope="session")
def capymoa_interface_mock():
    """Mock CapyMOAInterface for factory testing."""
    with patch("drift_datasets.factory.CapyMOAInterface") as mock_interface_class:
        mock_interface = Mock()

        # Create a mock generator that returns data based on configuration
        def create_mock_generator(generator_name, generator_config):
            # Validate generator name like the real implementation
            if generator_name not in ["SineGenerator", "HyperplaneGenerator", "STAGGERGenerator", "SEAGenerator"]:
                raise ValueError(f"Unsupported generator: {generator_name}")

            mock_generator = Mock()
            n_instances = generator_config.get("n_instances", 1000)

            # Return tuple format (X, y) as expected by the factory
            mock_generator.generate.return_value = (
                pd.DataFrame(
                    {
                        "x": np.random.RandomState(42).random(n_instances),
                        "y": np.random.RandomState(42).random(n_instances),
                    }
                ),
                pd.Series(np.random.RandomState(42).randint(0, 2, n_instances), name="target"),
            )
            return mock_generator

        # Mock the create_generator method to return our mock generator factory
        mock_interface.create_generator.side_effect = create_mock_generator
        mock_interface.get_generator_parameters.return_value = ["n_instances", "random_seed", "noise_level", "classification_function"]
        mock_interface.check_availability.return_value = (True, {"version": "1.0.0", "generators": ["SineGenerator"]})
        mock_interface.translate_parameters.return_value = {"numInstances": 1000, "randomSeed": 42}

        mock_interface_class.return_value = mock_interface
        yield mock_interface_class


@pytest.fixture(scope="session")
def capymoa_service():
    """Mock CapyMOA service for deterministic test execution."""
    with patch("drift_datasets.generators.synthetic.CapyMOAService") as mock_service:
        # Configure mock to return deterministic synthetic data
        mock_instance = Mock()

        # Configure service with generators attribute
        mock_instance.generators = {
            "SineGenerator": Mock(),
            "HyperplaneGenerator": Mock(),
            "STAGGERGenerator": Mock(),
            "SEAGenerator": Mock(),
        }

        mock_instance.generate_sine_dataset.return_value = {
            "X": np.random.RandomState(42).random((1000, 2)),
            "y": np.random.RandomState(42).randint(0, 2, 1000),
            "drift_points": [500],
            "metadata": {"generator": "SineGenerator"},
        }
        mock_instance.generate_hyperplane_dataset.return_value = {
            "X": np.random.RandomState(42).random((1000, 10)),
            "y": np.random.RandomState(42).randint(0, 2, 1000),
            "drift_points": [],  # Continuous drift
            "metadata": {"generator": "HyperplaneGenerator", "rotation_speed": 0.001},
        }
        mock_instance.generate_stagger_dataset.return_value = {
            "X": np.random.RandomState(42).random((1000, 3)),
            "y": np.random.RandomState(42).randint(0, 2, 1000),
            "drift_points": [333, 666],
            "metadata": {"generator": "STAGGERGenerator"},
        }

        mock_service.return_value = mock_instance
        yield mock_service


@pytest.fixture(scope="session")
def uci_repository_service():
    """Mock UCI ML Repository service for deterministic test execution."""
    with patch("drift_datasets.generators.real_world.UCIService") as mock_service:
        # Mock electricity dataset (ID: 321)
        mock_instance = Mock()
        mock_instance.load_dataset.return_value = {
            "X": pd.DataFrame(
                {
                    "date": pd.date_range("2000-01-01", periods=1000, freq="D"),
                    "demand": np.random.RandomState(42).normal(1000, 100, 1000),
                    "price": np.random.RandomState(42).normal(50, 10, 1000),
                }
            ),
            "y": pd.Series(np.random.RandomState(42).randint(0, 2, 1000), name="class"),
            "metadata": {"dataset_id": 321, "name": "Electricity Market", "task": "classification", "n_instances": 1000, "n_features": 3},
        }

        mock_service.return_value = mock_instance
        yield mock_service


@pytest.fixture(scope="session")
def parameter_translator():
    """Mock parameter translator for research parameter conversion."""
    # Use a simpler mock that doesn't require the module to exist
    mock_translator = Mock()

    # Define translation rules
    def translate_research_params(drift_config):
        translations = {"transition_durations": "drift_widths", "drift_intensities": "drift_alphas"}

        capymoa_params = {}
        for research_param, capymoa_param in translations.items():
            if research_param in drift_config:
                if research_param == "drift_intensities":
                    # Convert research intensities to CapyMOA alphas
                    intensities = drift_config[research_param]
                    capymoa_params[capymoa_param] = (
                        [1.0 if i == "complete" else 0.5 if i == "moderate" else 0.1 for i in intensities]
                        if isinstance(intensities, list)
                        else [float(intensities)]
                    )
                else:
                    capymoa_params[capymoa_param] = drift_config[research_param]

        return capymoa_params

    mock_translator.translate_research_params.side_effect = translate_research_params
    yield mock_translator


@pytest.fixture
def sample_toml_configs(tmp_path):
    """Generate sample TOML configuration files for testing."""
    configs = {}

    # Synthetic Sine dataset configuration
    sine_config = {
        "dataset": {"name": "test_sine", "type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "features": [
            {"name": "x", "type": "continuous", "role": "feature"},
            {"name": "y", "type": "continuous", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"},
        ],
        "generator_config": {"n_instances": 1000, "classification_function": 1, "random_seed": 42},
        "drift_config": {"drift_points": [500], "drift_types": ["concept"], "drift_patterns": ["abrupt"]},
    }

    # Real-world UCI dataset configuration
    uci_config = {
        "dataset": {"name": "test_electricity", "type": "real_world", "source": "ucimlrepo"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "uci_config": {"dataset_id": 321, "preprocessing": ["normalize", "temporal_order"]},
        "drift_config": {
            "drift_points": [400, 800],
            "drift_types": ["covariate", "concept"],
            "drift_patterns": ["gradual", "abrupt"],
            "drift_simulation": "concept_shift",
        },
    }

    # Hyperplane dataset configuration
    hyperplane_config = {
        "dataset": {"name": "test_hyperplane", "type": "synthetic", "source": "capymoa", "generator": "HyperplaneGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "features": [{"name": f"feature_{i}", "type": "continuous", "role": "feature"} for i in range(10)]
        + [{"name": "class", "type": "categorical", "role": "target"}],
        "generator_config": {"n_instances": 1000, "n_features": 10, "random_seed": 42},
        "drift_config": {"drift_points": [500], "drift_types": ["concept"], "drift_patterns": ["gradual"]},
    }

    # STAGGER dataset configuration
    stagger_config = {
        "dataset": {"name": "test_stagger", "type": "synthetic", "source": "capymoa", "generator": "STAGGERGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "features": [
            {"name": "size", "type": "categorical", "role": "feature"},
            {"name": "color", "type": "categorical", "role": "feature"},
            {"name": "shape", "type": "categorical", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"},
        ],
        "generator_config": {"n_instances": 1000, "concept_index": 1, "random_seed": 42},
        "drift_config": {"drift_points": [333, 666], "drift_types": ["concept", "concept"], "drift_patterns": ["abrupt", "abrupt"]},
    }

    # SEA dataset configuration
    sea_config = {
        "dataset": {"name": "test_sea", "type": "synthetic", "source": "capymoa", "generator": "SEAGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "features": [
            {"name": "x1", "type": "continuous", "role": "feature"},
            {"name": "x2", "type": "continuous", "role": "feature"},
            {"name": "x3", "type": "continuous", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"},
        ],
        "generator_config": {"n_instances": 1000, "threshold": 8, "noise_percentage": 0.1, "random_seed": 42},
        "drift_config": {"drift_points": [500], "drift_types": ["concept"], "drift_patterns": ["abrupt"]},
    }

    # Mixed dataset configuration
    mixed_config = {
        "dataset": {"name": "test_mixed", "type": "mixed"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "mixed_config": {
            "combination_type": "sequential",
            "components": [
                {"source": "capymoa", "generator": "SineGenerator", "n_instances": 500, "weight": 0.5},
                {"source": "ucimlrepo", "dataset_id": 321, "n_instances": 500, "weight": 0.5},
            ],
        },
        "drift_config": {"drift_points": [250, 750], "drift_types": ["concept", "covariate"], "drift_patterns": ["abrupt", "gradual"]},
    }

    # ExpertSystems configurations
    expertsystems_sine_config = {
        "dataset": {"name": "expertsystems_sine", "type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "generator_config": {"n_instances": 50000, "classification_function": 1, "random_seed": 42},
        "drift_config": {
            "drift_points": [10000, 25000, 40000],
            "drift_types": ["concept", "concept", "concept"],
            "drift_patterns": ["abrupt", "abrupt", "abrupt"],
            "concept_reversal": True,
        },
    }

    expertsystems_hyperplane_config = {
        "dataset": {"name": "expertsystems_hyperplane", "type": "synthetic", "source": "capymoa", "generator": "HyperplaneGenerator"},
        "generator_config": {
            "n_instances": 100000,
            "n_dimensions": 10,
            "n_drifting_dimensions": 10,
            "noise_percentage": 0.05,
            "random_seed": 42,
        },
        "drift_config": {
            "drift_points": [25000, 50000, 75000],
            "drift_patterns": ["gradual", "gradual", "gradual"],
            "drift_types": ["concept", "concept", "concept"],
            "continuous_drift": True,
        },
    }

    # Write TOML files
    config_files = {
        "sine": sine_config,
        "uci": uci_config,
        "hyperplane": hyperplane_config,
        "stagger": stagger_config,
        "sea": sea_config,
        "mixed": mixed_config,
        "expertsystems_sine": expertsystems_sine_config,
        "expertsystems_hyperplane": expertsystems_hyperplane_config,
    }

    for name, config in config_files.items():
        config_file = tmp_path / f"{name}.toml"
        with open(config_file, "w") as f:
            toml.dump(config, f)
        configs[name] = str(config_file)

    return configs


@pytest.fixture
def expertsystems_configs(sample_toml_configs):
    """ExpertSystems paper configurations for testing."""
    return {
        "sine_expertsystems": sample_toml_configs["expertsystems_sine"],
        "hyperplane_expertsystems": sample_toml_configs["expertsystems_hyperplane"],
        "stagger_expertsystems": sample_toml_configs["expertsystems_sine"],  # Reuse for now
    }


@pytest.fixture
def sample_drift_dataset():
    """Create a sample DriftDataset for testing."""
    # This will be imported once the DriftDataset class is implemented
    # For now, return a mock structure
    return {
        "X": pd.DataFrame(
            {"feature_0": np.random.RandomState(42).normal(0, 1, 1000), "feature_1": np.random.RandomState(42).normal(0, 1, 1000)}
        ),
        "y": pd.Series(np.random.RandomState(42).randint(0, 2, 1000), name="target"),
        "name": "test_dataset",
        "source_type": "synthetic",
        "drift_metadata": {
            "drift_points": [500],
            "drift_types": ["concept"],
            "drift_patterns": ["abrupt"],
            "affected_features": [[0, 1]],
            "transition_durations": [0],
            "drift_intensities": [1.0],
        },
        "dataset_metadata": {
            "dimension": "multivariate",
            "labeling": "supervised",
            "n_classes": 2,
            "features": [
                {"name": "feature_0", "type": "continuous", "role": "feature"},
                {"name": "feature_1", "type": "continuous", "role": "feature"},
                {"name": "target", "type": "categorical", "role": "target"},
            ],
        },
    }


@pytest.fixture
def expected_dataset_shapes():
    """Expected dataset shapes for validation."""
    return {
        "small": {"n_instances": 1000, "n_features": 2},
        "medium": {"n_instances": 10000, "n_features": 5},
        "large": {"n_instances": 100000, "n_features": 10},
        "expertsystems_sine": {"n_instances": 50000, "n_features": 4},
        "expertsystems_hyperplane": {"n_instances": 100000, "n_features": 10},
    }


@pytest.fixture
def validation_utilities():
    """Utility functions for test validation."""

    def validate_dataset_structure(dataset, expected_shape):
        """Validate dataset has expected structure and shape."""
        assert hasattr(dataset, "X"), "Dataset must have X (features) attribute"
        assert hasattr(dataset, "y"), "Dataset must have y (targets) attribute"
        assert hasattr(dataset, "drift_metadata"), "Dataset must have drift_metadata"
        assert hasattr(dataset, "dataset_metadata"), "Dataset must have dataset_metadata"

        assert dataset.X.shape[0] == expected_shape["n_instances"], f"Expected {expected_shape['n_instances']} instances"
        assert dataset.X.shape[1] == expected_shape["n_features"], f"Expected {expected_shape['n_features']} features"
        assert len(dataset.y) == expected_shape["n_instances"], "Target length must match instances"

    def validate_drift_metadata(drift_metadata, expected_drifts):
        """Validate drift metadata structure and content."""
        assert "drift_points" in drift_metadata, "Drift metadata must contain drift_points"
        assert "drift_types" in drift_metadata, "Drift metadata must contain drift_types"
        assert "drift_patterns" in drift_metadata, "Drift metadata must contain drift_patterns"

        assert len(drift_metadata["drift_points"]) == len(expected_drifts), "Drift points count mismatch"
        assert all(isinstance(point, int) for point in drift_metadata["drift_points"]), "Drift points must be integers"

    def validate_feature_metadata(dataset_metadata, expected_features):
        """Validate feature metadata structure."""
        assert "features" in dataset_metadata, "Dataset metadata must contain features"
        assert len(dataset_metadata["features"]) == len(expected_features), "Feature count mismatch"

        for feature in dataset_metadata["features"]:
            assert "name" in feature, "Feature must have name"
            assert "type" in feature, "Feature must have type"
            assert "role" in feature, "Feature must have role"

    def validate_drift_intensities(drift_metadata, expected_range):
        """Validate drift intensities are within expected range."""
        intensities = drift_metadata.get("drift_intensities", [])
        for intensity in intensities:
            assert expected_range[0] <= intensity <= expected_range[1], f"Drift intensity {intensity} outside range {expected_range}"

    return {
        "validate_dataset_structure": validate_dataset_structure,
        "validate_drift_metadata": validate_drift_metadata,
        "validate_feature_metadata": validate_feature_metadata,
        "validate_drift_intensities": validate_drift_intensities,
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark thresholds for testing."""
    return {
        "generation_time": {"1k_samples": 5.0, "10k_samples": 30.0, "100k_samples": 300.0},  # seconds  # seconds  # seconds
        "memory_usage": {"1k_samples": 100, "10k_samples": 1000, "100k_samples": 8000},  # MB  # MB  # MB
    }


@pytest.fixture
def error_scenarios():
    """Common error scenarios for negative testing."""
    return {
        "invalid_config": {
            "missing_required_field": {
                "metadata": {"dimension": "multivariate"}
                # Missing dataset section
            },
            "invalid_generator": {
                "dataset": {"name": "test", "type": "synthetic", "generator": "InvalidGenerator"},
                "generator_config": {"n_instances": 1000, "random_seed": 42},
                "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
            },
            "invalid_drift_pattern": {"drift_config": {"drift_patterns": ["invalid_pattern"]}},
        },
        "synthetic_generation": {
            "conflicting_parameters": {
                "dataset": {"name": "test", "type": "synthetic", "generator": "SineGenerator"},
                "generator_config": {"n_instances": 1000, "n_features": 5, "n_dimensions": 10},  # Conflicting
                "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
            },
            "invalid_parameter_range": {
                "dataset": {"name": "test", "type": "synthetic", "generator": "SineGenerator"},
                "generator_config": {"n_instances": 1000, "noise_level": 1.5},  # > 1.0
                "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
            },
        },
        "network_errors": {
            "uci_connection_error": ConnectionError("Unable to connect to UCI repository"),
            "capymoa_runtime_error": RuntimeError("CapyMOA generation failed"),
        },
    }
