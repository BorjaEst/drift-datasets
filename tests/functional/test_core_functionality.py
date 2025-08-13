"""
Functional tests for core dataset generation and management functionality.

These tests validate the primary user workflows from REQ-001 through REQ-006,
including factory method usage, DriftDataset object creation, and configuration parsing.
"""

from pathlib import Path

import pytest
import toml


class TestCoreDatasetGeneration:
    """Test core dataset generation functionality (REQ-001)."""

    def test_should_create_synthetic_dataset_when_valid_config_provided(self, capymoa_service, sample_toml_configs):
        """Test REQ-001: Factory method creates DriftDataset objects with ground truth metadata."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Create synthetic dataset using factory method
        dataset = create_dataset(config_path)

        # Assert: Dataset has required structure and metadata
        assert hasattr(dataset, "X"), "Dataset must have feature matrix X"
        assert hasattr(dataset, "y"), "Dataset must have target vector y"
        assert hasattr(dataset, "name"), "Dataset must have name attribute"
        assert hasattr(dataset, "source_type"), "Dataset must have source_type"
        assert hasattr(dataset, "drift_metadata"), "Dataset must have drift_metadata"
        assert hasattr(dataset, "dataset_metadata"), "Dataset must have dataset_metadata"

        assert dataset.name == "test_sine"
        assert dataset.source_type == "synthetic"
        assert dataset.X.shape[0] == 1000  # n_instances from config
        assert dataset.X.shape[1] == 2  # features from sine generator
        assert len(dataset.y) == 1000

    def test_should_create_real_world_dataset_when_valid_uci_config_provided(self, uci_repository_service, sample_toml_configs):
        """Test REQ-001: Factory method handles real-world dataset creation."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci"]

        # Act: Create real-world dataset
        dataset = create_dataset(config_path)

        # Assert: Dataset created with UCI data structure
        assert dataset.name == "test_electricity"
        assert dataset.source_type == "real_world"
        assert hasattr(dataset, "X"), "Real-world dataset must have features"
        assert hasattr(dataset, "y"), "Real-world dataset must have targets"
        assert 321 in str(dataset.dataset_metadata), "Dataset metadata should reference UCI ID"

    def test_should_create_mixed_dataset_when_valid_mixed_config_provided(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-001: Factory method handles mixed dataset creation."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]

        # Act: Create mixed dataset
        dataset = create_dataset(config_path)

        # Assert: Mixed dataset combines synthetic and real-world data
        assert dataset.name == "test_mixed"
        assert dataset.source_type == "mixed"
        assert dataset.X.shape[0] == 1000  # Combined instances
        assert "sequential" in str(dataset.dataset_metadata), "Should indicate sequential combination"

    def test_should_fail_when_invalid_config_provided(self, error_scenarios):
        """Test REQ-001: Factory method fails gracefully with invalid configuration."""
        from drift_datasets import create_dataset

        # Test missing required field
        with pytest.raises(ValueError, match="Missing required field 'dataset'"):
            create_dataset(error_scenarios["invalid_config"]["missing_required_field"])

        # Test invalid generator
        with pytest.raises(ValueError, match="Unsupported generator: InvalidGenerator"):
            create_dataset(error_scenarios["invalid_config"]["invalid_generator"])


class TestDriftDatasetObject:
    """Test DriftDataset object functionality (REQ-002)."""

    def test_should_provide_feature_access_when_dataset_created(self, sample_drift_dataset):
        """Test REQ-002: DriftDataset provides structured access to features."""
        from drift_datasets.models import DriftDataset

        # Act: Create DriftDataset from sample data
        dataset = DriftDataset(**sample_drift_dataset)

        # Assert: Feature access works correctly
        assert dataset.X.shape == (1000, 2), "Feature matrix has expected shape"
        assert list(dataset.X.columns) == ["feature_0", "feature_1"], "Features have correct names"
        assert dataset.y.name == "target", "Target has correct name"

        # Test feature metadata access
        feature_names = [f["name"] for f in dataset.dataset_metadata["features"] if f["role"] == "feature"]
        assert feature_names == ["feature_0", "feature_1"], "Feature metadata matches data"

    def test_should_provide_target_access_when_dataset_created(self, sample_drift_dataset):
        """Test REQ-002: DriftDataset provides structured access to targets."""
        from drift_datasets.models import DriftDataset

        # Act: Create DriftDataset from sample data
        dataset = DriftDataset(**sample_drift_dataset)

        # Assert: Target access works correctly
        assert len(dataset.y) == 1000, "Target vector has expected length"
        assert dataset.y.dtype == "int64", "Target has correct data type"
        assert set(dataset.y.unique()) <= {0, 1}, "Target contains valid class labels"

        # Test target metadata
        target_feature = next(f for f in dataset.dataset_metadata["features"] if f["role"] == "target")
        assert target_feature["name"] == "target", "Target metadata matches data"

    def test_should_provide_drift_metadata_when_dataset_created(self, sample_drift_dataset, validation_utilities):
        """Test REQ-002: DriftDataset provides comprehensive drift metadata."""
        from drift_datasets.models import DriftDataset

        # Act: Create DriftDataset from sample data
        dataset = DriftDataset(**sample_drift_dataset)

        # Assert: Drift metadata structure is valid
        validation_utilities["validate_drift_metadata"](dataset.drift_metadata, expected_drifts=[500])

        assert dataset.drift_metadata["drift_points"] == [500], "Drift points correctly stored"
        assert dataset.drift_metadata["drift_types"] == ["concept"], "Drift types correctly stored"
        assert dataset.drift_metadata["drift_patterns"] == ["abrupt"], "Drift patterns correctly stored"
        assert dataset.drift_metadata["drift_intensities"] == [1.0], "Drift intensities correctly stored"

    def test_should_provide_dataset_metadata_when_dataset_created(self, sample_drift_dataset, validation_utilities):
        """Test REQ-002: DriftDataset provides comprehensive dataset metadata."""
        from drift_datasets.models import DriftDataset

        # Act: Create DriftDataset from sample data
        dataset = DriftDataset(**sample_drift_dataset)

        # Assert: Dataset metadata structure is valid
        expected_features = [
            {"name": "feature_0", "type": "continuous", "role": "feature"},
            {"name": "feature_1", "type": "continuous", "role": "feature"},
            {"name": "target", "type": "categorical", "role": "target"},
        ]
        validation_utilities["validate_feature_metadata"](dataset.dataset_metadata, expected_features)

        assert dataset.dataset_metadata["dimension"] == "multivariate", "Dimension correctly identified"
        assert dataset.dataset_metadata["labeling"] == "supervised", "Labeling correctly identified"
        assert dataset.dataset_metadata["n_classes"] == 2, "Number of classes correctly identified"


class TestConfigurationParsing:
    """Test TOML configuration parsing functionality (REQ-003)."""

    def test_should_parse_toml_file_when_valid_config_provided(self, sample_toml_configs):
        """Test REQ-003: Parse TOML configuration files for reproducible generation."""
        from drift_datasets.config import ConfigurationParser

        config_path = sample_toml_configs["sine"]

        # Act: Parse TOML configuration
        parser = ConfigurationParser()
        config = parser.parse_config(config_path)

        # Assert: Configuration parsed correctly
        assert config["dataset"]["name"] == "test_sine", "Dataset name parsed correctly"
        assert config["dataset"]["type"] == "synthetic", "Dataset type parsed correctly"
        assert config["dataset"]["generator"] == "SineGenerator", "Generator parsed correctly"

        assert config["generator_config"]["n_instances"] == 1000, "Instance count parsed correctly"
        assert config["generator_config"]["random_seed"] == 42, "Random seed parsed correctly"

        assert config["drift_config"]["drift_points"] == [500], "Drift points parsed correctly"
        assert config["drift_config"]["drift_types"] == ["concept"], "Drift types parsed correctly"

    def test_should_validate_configuration_structure_when_parsing_config(self, sample_toml_configs):
        """Test REQ-003: Validate configuration structure during parsing."""
        from drift_datasets.config import ConfigurationParser

        config_path = sample_toml_configs["uci"]

        # Act: Parse and validate configuration
        parser = ConfigurationParser()
        config = parser.parse_config(config_path)
        is_valid, validation_errors = parser.validate_config(config)

        # Assert: Configuration validation works
        assert is_valid == True, f"Configuration should be valid: {validation_errors}"
        assert validation_errors == [], "No validation errors for valid config"

        # Test required sections present
        assert "dataset" in config, "Dataset section required"
        assert "metadata" in config, "Metadata section required"
        assert "uci_config" in config, "UCI config section required for real-world datasets"

    def test_should_detect_invalid_configuration_structure_when_parsing_bad_config(self, tmp_path, error_scenarios):
        """Test REQ-003: Detect and report invalid configuration structure."""
        from drift_datasets.config import ConfigurationParser

        # Create invalid config file
        invalid_config_path = tmp_path / "invalid.toml"
        with open(invalid_config_path, "w") as f:
            toml.dump(error_scenarios["invalid_config"]["missing_required_field"], f)

        # Act: Parse invalid configuration
        parser = ConfigurationParser()

        # Assert: Invalid configuration detected
        with pytest.raises(ValueError, match="Missing required field 'dataset'"):
            config = parser.parse_config(str(invalid_config_path))
            parser.validate_config(config)

    def test_should_handle_nested_configuration_sections_when_parsing_complex_config(self, sample_toml_configs):
        """Test REQ-003: Handle complex nested configuration sections."""
        from drift_datasets.config import ConfigurationParser

        config_path = sample_toml_configs["mixed"]

        # Act: Parse complex mixed configuration
        parser = ConfigurationParser()
        config = parser.parse_config(config_path)

        # Assert: Nested sections parsed correctly
        assert "mixed_config" in config, "Mixed config section present"
        assert "components" in config["mixed_config"], "Components section present"
        assert len(config["mixed_config"]["components"]) == 2, "Two components configured"

        # Validate component structure
        component1 = config["mixed_config"]["components"][0]
        assert component1["source"] == "capymoa", "First component source correct"
        assert component1["generator"] == "SineGenerator", "First component generator correct"
        assert component1["weight"] == 0.5, "First component weight correct"


class TestDatasetPersistence:
    """Test dataset saving and loading functionality (REQ-004)."""

    def test_should_save_dataset_to_disk_when_save_method_called(self, sample_drift_dataset, tmp_path):
        """Test REQ-004: Save DriftDataset objects to disk in multiple formats."""
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset
        dataset = DriftDataset(**sample_drift_dataset)
        output_dir = tmp_path / "saved_datasets"

        # Act: Save dataset
        saved_paths = dataset.save(output_dir, formats=["parquet", "csv", "json"])

        # Assert: Files created successfully
        assert len(saved_paths) == 3, "Should create 3 format files"
        assert (output_dir / "test_dataset.parquet").exists(), "Parquet file created"
        assert (output_dir / "test_dataset.csv").exists(), "CSV file created"
        assert (output_dir / "test_dataset_metadata.json").exists(), "JSON metadata created"

        # Verify file contents
        import pandas as pd

        loaded_parquet = pd.read_parquet(saved_paths[0])
        assert loaded_parquet.shape == (1000, 3), "Parquet contains correct data shape"

    def test_should_load_dataset_from_disk_when_load_method_called(self, sample_drift_dataset, tmp_path):
        """Test REQ-004: Load DriftDataset objects from disk."""
        from drift_datasets.models import DriftDataset

        # Arrange: Save dataset first
        original_dataset = DriftDataset(**sample_drift_dataset)
        output_dir = tmp_path / "saved_datasets"
        saved_paths = original_dataset.save(output_dir, formats=["parquet"])

        # Act: Load dataset from disk
        loaded_dataset = DriftDataset.load(saved_paths[0])

        # Assert: Loaded dataset matches original
        assert loaded_dataset.name == original_dataset.name, "Name preserved"
        assert loaded_dataset.source_type == original_dataset.source_type, "Source type preserved"
        assert loaded_dataset.X.shape == original_dataset.X.shape, "Feature shape preserved"
        assert len(loaded_dataset.y) == len(original_dataset.y), "Target length preserved"
        assert loaded_dataset.drift_metadata == original_dataset.drift_metadata, "Drift metadata preserved"

    def test_should_preserve_metadata_when_saving_and_loading_dataset(self, sample_drift_dataset, tmp_path):
        """Test REQ-004: Preserve complete metadata during save/load cycles."""
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset with rich metadata
        dataset = DriftDataset(**sample_drift_dataset)
        output_dir = tmp_path / "saved_datasets"

        # Act: Save and load dataset
        saved_paths = dataset.save(output_dir, formats=["parquet"])
        loaded_dataset = DriftDataset.load(saved_paths[0])

        # Assert: All metadata preserved
        assert loaded_dataset.drift_metadata["drift_points"] == [500], "Drift points preserved"
        assert loaded_dataset.drift_metadata["drift_types"] == ["concept"], "Drift types preserved"
        assert loaded_dataset.dataset_metadata["dimension"] == "multivariate", "Dataset dimension preserved"
        assert loaded_dataset.dataset_metadata["n_classes"] == 2, "Number of classes preserved"


class TestDatasetValidation:
    """Test dataset validation functionality (REQ-005)."""

    def test_should_validate_dataset_structure_when_dataset_created(self, sample_drift_dataset, validation_utilities):
        """Test REQ-005: Validate dataset structure and metadata consistency."""
        from drift_datasets.models import DriftDataset

        # Act: Create dataset (validation should occur during creation)
        dataset = DriftDataset(**sample_drift_dataset)

        # Assert: Validation passes for valid dataset
        validation_utilities["validate_dataset_structure"](dataset, {"n_instances": 1000, "n_features": 2})

        # Explicit validation call
        is_valid, errors = dataset.validate()
        assert is_valid == True, f"Dataset should be valid: {errors}"
        assert errors == [], "No validation errors for valid dataset"

    def test_should_detect_inconsistent_data_when_validating_malformed_dataset(self, sample_drift_dataset):
        """Test REQ-005: Detect inconsistencies between data and metadata."""
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset with inconsistent metadata
        malformed_data = sample_drift_dataset.copy()
        malformed_data["dataset_metadata"]["n_classes"] = 5  # Wrong class count

        # Act: Validate malformed dataset
        dataset = DriftDataset(**malformed_data)
        is_valid, errors = dataset.validate()

        # Assert: Validation detects inconsistency
        assert is_valid == False, "Validation should fail for inconsistent data"
        assert any("class count mismatch" in error.lower() for error in errors), "Should detect class count mismatch"

    def test_should_validate_drift_metadata_consistency_when_validating_dataset(self, sample_drift_dataset):
        """Test REQ-005: Validate drift metadata consistency with dataset size."""
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset with invalid drift points
        invalid_drift_data = sample_drift_dataset.copy()
        invalid_drift_data["drift_metadata"]["drift_points"] = [2000]  # Beyond dataset size

        # Act: Validate dataset with invalid drift points
        dataset = DriftDataset(**invalid_drift_data)
        is_valid, errors = dataset.validate()

        # Assert: Validation detects invalid drift points
        assert is_valid == False, "Validation should fail for drift points beyond dataset size"
        assert any("drift point" in error.lower() for error in errors), "Should detect invalid drift points"


class TestDatasetInformation:
    """Test dataset information and statistics functionality (REQ-006)."""

    def test_should_provide_basic_statistics_when_info_method_called(self, sample_drift_dataset):
        """Test REQ-006: Provide comprehensive dataset information and statistics."""
        from drift_datasets.models import DriftDataset

        # Act: Create dataset and get information
        dataset = DriftDataset(**sample_drift_dataset)
        info = dataset.info()

        # Assert: Information includes basic statistics
        assert "n_instances" in info, "Info includes instance count"
        assert "n_features" in info, "Info includes feature count"
        assert "n_classes" in info, "Info includes class count"
        assert "class_distribution" in info, "Info includes class distribution"

        assert info["n_instances"] == 1000, "Correct instance count"
        assert info["n_features"] == 2, "Correct feature count"
        assert info["n_classes"] == 2, "Correct class count"

    def test_should_provide_drift_statistics_when_info_method_called(self, sample_drift_dataset):
        """Test REQ-006: Provide detailed drift statistics and analysis."""
        from drift_datasets.models import DriftDataset

        # Act: Create dataset and get drift information
        dataset = DriftDataset(**sample_drift_dataset)
        info = dataset.info()

        # Assert: Information includes drift statistics
        assert "drift_analysis" in info, "Info includes drift analysis"
        assert "n_drift_points" in info["drift_analysis"], "Drift analysis includes drift point count"
        assert "drift_intervals" in info["drift_analysis"], "Drift analysis includes intervals"

        drift_analysis = info["drift_analysis"]
        assert drift_analysis["n_drift_points"] == 1, "Correct drift point count"
        assert drift_analysis["drift_intervals"] == [500, 500], "Correct drift intervals"

    def test_should_provide_feature_statistics_when_describe_method_called(self, sample_drift_dataset):
        """Test REQ-006: Provide detailed feature statistics and descriptions."""
        from drift_datasets.models import DriftDataset

        # Act: Create dataset and get feature descriptions
        dataset = DriftDataset(**sample_drift_dataset)
        description = dataset.describe()

        # Assert: Description includes feature statistics
        assert "features" in description, "Description includes feature statistics"
        assert "target" in description, "Description includes target statistics"

        feature_stats = description["features"]
        assert "feature_0" in feature_stats, "Statistics for feature_0 present"
        assert "feature_1" in feature_stats, "Statistics for feature_1 present"

        # Check statistical measures
        f0_stats = feature_stats["feature_0"]
        assert "mean" in f0_stats, "Mean statistic present"
        assert "std" in f0_stats, "Standard deviation present"
        assert "min" in f0_stats, "Minimum value present"
        assert "max" in f0_stats, "Maximum value present"
