"""
Functional test for research parameter support end-to-end.

Tests REQ-006: Drift Point Specification with research parameters.
"""

import pytest

import drift_datasets as dd


class TestResearchParameterEndToEnd:
    """Test research parameters work end-to-end."""

    def test_create_dataset_with_research_parameters(self, sample_toml_configs):
        """Test creating a dataset with research parameters."""
        # Use assets-based config for better isolation
        if "research_params" in sample_toml_configs:
            dataset = dd.create_dataset(sample_toml_configs["research_params"])
        else:
            # Fallback to inline config
            config = {
                "dataset": {
                    "name": "test_research_params",
                    "type": "synthetic",
                    "source": "capymoa",
                    "generator": "HyperplaneGenerator",
                    "description": "Test dataset with research parameters",
                },
                "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
                "features": [
                    {"name": "feature_0", "type": "continuous", "role": "feature"},
                    {"name": "feature_1", "type": "continuous", "role": "feature"},
                    {"name": "feature_2", "type": "continuous", "role": "feature"},
                    {"name": "target", "type": "categorical", "role": "target"},
                ],
                "generator_config": {
                    "n_instances": 1000,
                    "n_features": 3,
                    "random_seed": 42,
                },
                "drift_config": {
                    "drift_points": [300, 600],
                    "drift_types": ["concept", "concept"],
                    "drift_patterns": ["gradual", "abrupt"],
                    "transition_durations": [100, 0],  # Research parameter
                    "drift_intensities": [0.5, 0.8],  # Research parameter
                    "affected_features": [[0, 1], [2]],  # Research parameter
                },
            }
            dataset = dd.create_dataset(config)

        # Verify dataset was created
        assert dataset.X.shape == (1000, 3)
        assert len(dataset.y) == 1000

        # Verify drift metadata contains research parameters
        assert dataset.drift_metadata.drift_points == [300, 600]
        assert dataset.drift_metadata.transition_durations == [100, 0]
        assert dataset.drift_metadata.drift_intensities == [0.5, 0.8]
        assert dataset.drift_metadata.affected_features == [[0, 1], [2]]

    def test_create_dataset_research_param_validation(self, sample_toml_configs):
        """Test that research parameter validation works in dataset creation."""
        config = {
            "dataset": {
                "name": "test_validation",
                "type": "synthetic",
                "source": "capymoa",
                "generator": "HyperplaneGenerator",
            },
            "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
            "features": [
                {"name": "feature_0", "type": "continuous", "role": "feature"},
                {"name": "feature_1", "type": "continuous", "role": "feature"},
                {"name": "target", "type": "categorical", "role": "target"},
            ],
            "generator_config": {
                "n_instances": 500,
                "n_features": 2,
                "random_seed": 42,
            },
            "drift_config": {
                "drift_points": [200, 400],
                "transition_durations": [50],  # Length mismatch - should fail
                "drift_intensities": [0.5, 0.8],
            },
        }

        with pytest.raises(ValueError) as exc_info:
            dd.create_dataset(config)

        assert "transition_durations length" in str(exc_info.value)

    def test_create_dataset_backward_compatibility(self, sample_toml_configs):
        """Test that existing CapyMOA parameters still work."""
        config = {
            "dataset": {
                "name": "test_backward_compat",
                "type": "synthetic",
                "source": "capymoa",
                "generator": "HyperplaneGenerator",
            },
            "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
            "features": [
                {"name": "feature_0", "type": "continuous", "role": "feature"},
                {"name": "feature_1", "type": "continuous", "role": "feature"},
                {"name": "target", "type": "categorical", "role": "target"},
            ],
            "generator_config": {
                "n_instances": 500,
                "n_features": 2,
                "random_seed": 42,
            },
            "drift_config": {
                "drift_points": [200],
                "drift_types": ["concept"],
                "drift_patterns": ["abrupt"],
                # No research parameters - should still work
            },
        }

        dataset = dd.create_dataset(config)

        assert dataset.X.shape == (500, 2)
        assert len(dataset.y) == 500
        assert dataset.drift_metadata.drift_points == [200]
        assert dataset.drift_metadata.transition_durations == []  # Default empty list

    def test_mixed_research_and_capymoa_parameters(self, sample_toml_configs):
        """Test mixing research and CapyMOA parameters (research takes precedence)."""
        config = {
            "dataset": {
                "name": "test_mixed_params",
                "type": "synthetic",
                "source": "capymoa",
                "generator": "HyperplaneGenerator",
            },
            "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
            "features": [
                {"name": "feature_0", "type": "continuous", "role": "feature"},
                {"name": "feature_1", "type": "continuous", "role": "feature"},
                {"name": "target", "type": "categorical", "role": "target"},
            ],
            "generator_config": {
                "n_instances": 500,
                "n_features": 2,
                "random_seed": 42,
            },
            "drift_config": {
                "drift_points": [200],
                "drift_types": ["concept"],
                "transition_durations": [75],  # Research parameter
                "drift_widths": [50],  # CapyMOA parameter (should be overridden)
                "drift_intensities": [0.6],  # Research parameter
                "drift_alphas": [0.3],  # CapyMOA parameter (should be overridden)
            },
        }

        dataset = dd.create_dataset(config)

        # Research parameters should be preserved
        assert dataset.drift_metadata.transition_durations == [75]
        assert dataset.drift_metadata.drift_intensities == [0.6]
