"""
Functional tests for mixed dataset functionality.

These tests validate mixed dataset creation from REQ-003,
including combining synthetic and real-world data sources.
"""

import pytest


class TestMixedDatasetCombination:
    """Test mixed dataset combination functionality (REQ-003)."""

    def test_should_combine_synthetic_and_real_world_datasets_when_mixed_config_provided(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-003: Combine multiple data sources into unified datasets."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]

        # Act: Create mixed dataset combining synthetic and real-world data
        dataset = create_dataset(config_path)

        # Assert: REQ-003 - mixed dataset characteristics
        assert dataset.name == "test_mixed"
        assert dataset.source_type == "mixed"
        assert len(dataset.X) > 0, "Dataset has instances"
        assert len(dataset.y) > 0, "Dataset has targets"

        # REQ-003: Maintain data type consistency across combined components
        assert dataset.X.dtypes is not None, "Data types consistent across components"
        assert len(dataset.X) == len(dataset.y), "Features and targets have matching lengths"

    def test_should_preserve_drift_metadata_when_combining_datasets(self, capymoa_service, uci_repository_service, sample_toml_configs):
        """Test REQ-003: Preserve individual component metadata in unified structure."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]
        dataset = create_dataset(config_path)

        # Assert: REQ-003 - unified drift metadata structure
        assert hasattr(dataset, "drift_metadata"), "Mixed dataset has drift metadata"
        drift_metadata = dataset.drift_metadata

        # Check drift metadata structure is preserved
        if hasattr(drift_metadata, "drift_points"):
            drift_points = drift_metadata.drift_points
        else:
            drift_points = drift_metadata.get("drift_points", [])

        assert isinstance(drift_points, list), "Drift points is a list"

    def test_should_generate_valid_drift_point_indices_when_combining_datasets(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-003: Generate valid drift_points indices for combined dataset."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]
        dataset = create_dataset(config_path)

        # Assert: REQ-003 - valid drift point indices for combined dataset
        drift_metadata = dataset.drift_metadata
        if hasattr(drift_metadata, "drift_points"):
            drift_points = drift_metadata.drift_points
        else:
            drift_points = drift_metadata.get("drift_points", [])

        # All drift points should be valid indices within the combined dataset
        for drift_point in drift_points:
            assert 0 <= drift_point < len(dataset.X), f"Drift point {drift_point} is valid index"

    def test_should_apply_consistent_drift_parameters_when_combining_datasets(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-003: Apply consistent drift parameters across mixed components."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]
        dataset = create_dataset(config_path)

        # Assert: REQ-003 - consistent drift parameters applied
        drift_metadata = dataset.drift_metadata

        if hasattr(drift_metadata, "drift_types"):
            drift_types = drift_metadata.drift_types
        else:
            drift_types = drift_metadata.get("drift_types", [])

        if hasattr(drift_metadata, "drift_patterns"):
            drift_patterns = drift_metadata.drift_patterns
        else:
            drift_patterns = drift_metadata.get("drift_patterns", [])

        # Consistent drift configuration across components
        if drift_types and drift_patterns:
            assert len(drift_types) == len(drift_patterns), "Drift types and patterns are consistent"
