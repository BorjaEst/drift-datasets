"""
Functional tests for performance benchmarks and comprehensive testing.

These tests validate performance requirements from REQ-014 through REQ-017.
"""

import time

import pytest


class TestPerformanceBenchmarks:
    """Test performance benchmarks functionality (REQ-014)."""

    def test_should_generate_100k_synthetic_samples_within_30_seconds_when_performance_tested(
        self, capymoa_interface_mock, sample_toml_configs, tmp_path
    ):
        """Test REQ-014: Generate 100K synthetic samples in <30 seconds."""
        import tempfile

        import toml

        from drift_datasets import create_dataset

        # Arrange: Create 100K sample configuration
        large_config = {
            "dataset": {"name": "test_large_sine", "type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
            "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
            "generator_config": {"n_instances": 100000, "classification_function": 1, "random_seed": 42},
            "drift_config": {"drift_points": [50000], "drift_types": ["concept"], "drift_patterns": ["abrupt"]},
        }

        config_file = tmp_path / "large_config.toml"
        with open(config_file, "w") as f:
            toml.dump(large_config, f)

        # Act: Measure large dataset generation time
        start_time = time.time()
        dataset = create_dataset(str(config_file))
        generation_time = time.time() - start_time

        # Assert: REQ-014 performance targets met
        assert generation_time < 30.0, f"Generation took {generation_time:.2f}s, should be under 30s"
        assert len(dataset.X) == 100000, "Generated 100K samples"
        assert len(dataset.y) == 100000, "Generated 100K targets"

    def test_should_load_uci_datasets_within_10_seconds_when_performance_tested(self, uci_repository_service, sample_toml_configs):
        """Test REQ-014: Load UCI datasets in <10 seconds."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci"]

        # Act: Measure UCI dataset loading time
        start_time = time.time()
        dataset = create_dataset(config_path)
        loading_time = time.time() - start_time

        # Assert: REQ-014 UCI loading performance
        assert loading_time < 10.0, f"Loading took {loading_time:.2f}s, should be under 10s"
        assert len(dataset.X) > 0, "Dataset loaded successfully"

    def test_should_parse_toml_configuration_within_1_second_when_performance_tested(self, sample_toml_configs):
        """Test REQ-014: Parse TOML configurations in <1 second."""
        from drift_datasets.config import ConfigurationManager

        config_path = sample_toml_configs["sine"]

        # Act: Measure TOML parsing time
        start_time = time.time()
        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_path)
        parsing_time = time.time() - start_time

        # Assert: REQ-014 TOML parsing performance
        assert parsing_time < 1.0, f"Parsing took {parsing_time:.2f}s, should be under 1s"
        assert config is not None, "Configuration loaded successfully"

    def test_should_filter_features_within_5_seconds_when_performance_tested(self, capymoa_interface_mock, sample_toml_configs, tmp_path):
        """Test REQ-014: Feature filtering operations complete in <5 seconds for large datasets."""
        import tempfile

        import toml

        from drift_datasets import create_dataset

        # Arrange: Create larger dataset for feature filtering test
        large_config = {
            "dataset": {"name": "test_feature_filter", "type": "synthetic", "source": "capymoa", "generator": "HyperplaneGenerator"},
            "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
            "generator_config": {"n_instances": 10000, "n_features": 50, "random_seed": 42},
            "drift_config": {"drift_points": [5000], "drift_types": ["concept"], "drift_patterns": ["abrupt"]},
        }

        config_file = tmp_path / "filter_config.toml"
        with open(config_file, "w") as f:
            toml.dump(large_config, f)

        dataset = create_dataset(str(config_file))

        # Act: Measure feature filtering time
        start_time = time.time()
        continuous_features = dataset.get_continuous_features()
        filtering_time = time.time() - start_time

        # Assert: REQ-014 filtering performance
        assert filtering_time < 5.0, f"Filtering took {filtering_time:.2f}s, should be under 5s"
        assert len(continuous_features) > 0, "Feature filtering successful"


class TestErrorHandlingAndLogging:
    """Test error handling and logging functionality (REQ-015)."""

    def test_should_raise_file_not_found_error_when_config_file_missing(self):
        """Test REQ-015: FileNotFoundError for missing configuration files."""
        from drift_datasets import create_dataset

        # Act & Assert: Missing file error
        with pytest.raises(FileNotFoundError) as exc_info:
            create_dataset("/nonexistent/config.toml")

        assert "/nonexistent/config.toml" in str(exc_info.value), "Error message contains full path"

    def test_should_raise_value_error_when_invalid_dataset_type_provided(self, tmp_path):
        """Test REQ-015: ValueError for invalid dataset types."""
        import toml

        from drift_datasets import create_dataset

        # Arrange: Invalid dataset type configuration
        invalid_config = {
            "dataset": {"name": "test_invalid", "type": "invalid_type", "source": "capymoa"},
            "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        }

        config_file = tmp_path / "invalid_config.toml"
        with open(config_file, "w") as f:
            toml.dump(invalid_config, f)

        # Act & Assert: Invalid type error
        with pytest.raises(ValueError) as exc_info:
            create_dataset(str(config_file))

        assert "invalid_type" in str(exc_info.value).lower(), "Error message mentions invalid type"


class TestReproducibilityAndTesting:
    """Test deterministic dataset generation functionality (REQ-016)."""

    def test_should_generate_identical_datasets_when_same_config_and_seed_used(self, capymoa_interface_mock, sample_toml_configs):
        """Test REQ-016: Deterministic dataset generation with same configuration and random seed."""
        import numpy as np

        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Generate two datasets with same configuration
        dataset1 = create_dataset(config_path)
        dataset2 = create_dataset(config_path)

        # Assert: REQ-016 bit-identical results
        assert np.array_equal(dataset1.X.values, dataset2.X.values), "X matrices are identical"
        assert np.array_equal(dataset1.y.values, dataset2.y.values), "y vectors are identical"
        assert dataset1.drift_metadata.drift_points == dataset2.drift_metadata.drift_points, "Drift points are identical"


class TestUsabilityAndDocumentation:
    """Test intuitive API design functionality (REQ-018)."""

    def test_should_provide_single_entry_point_when_creating_dataset(self, sample_toml_configs):
        """Test REQ-018: Single entry point create_dataset(config_path) -> DriftDataset."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Use single entry point
        dataset = create_dataset(config_path)

        # Assert: REQ-018 single entry point works
        assert dataset is not None, "Dataset created through single entry point"
        assert hasattr(dataset, "X"), "Dataset has features"
        assert hasattr(dataset, "y"), "Dataset has targets"
        assert hasattr(dataset, "drift_metadata"), "Dataset has drift metadata"

    def test_should_provide_convenience_methods_when_accessing_features(self, capymoa_interface_mock, sample_toml_configs):
        """Test REQ-018: Convenience methods for feature access without manual filtering."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]
        dataset = create_dataset(config_path)

        # Act: Use convenience methods
        continuous_features = dataset.get_continuous_features()
        features_by_role = dataset.get_features_by_role("feature")

        # Assert: REQ-018 convenience methods work
        assert continuous_features is not None, "get_continuous_features() works"
        assert features_by_role is not None, "get_features_by_role() works"


class TestFeatureDiscoveryAndInspection:
    """Test feature discovery and inspection functionality (REQ-019)."""

    def test_should_provide_feature_summary_when_inspecting_dataset(self, capymoa_interface_mock, sample_toml_configs):
        """Test REQ-019: summarize_features() returns comprehensive feature overview."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]
        dataset = create_dataset(config_path)

        # Act: Get feature summary
        summary = dataset.summarize_features()

        # Assert: REQ-019 comprehensive feature overview
        assert summary is not None, "Feature summary provided"
        assert "features" in summary or len(summary) > 0, "Summary contains feature information"
