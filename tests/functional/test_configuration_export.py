"""
Functional tests for configuration management and dataset export functionality.

These tests validate configuration management from REQ-018 through REQ-020,
including TOML configuration handling, parameter validation, and export formats.
"""

from pathlib import Path

import pytest


class TestConfigurationManagement:
    """Test configuration management functionality (REQ-018)."""

    def test_should_load_toml_configuration_when_config_file_provided(self, sample_toml_configs, tmp_path):
        """Test REQ-018: Load and parse TOML configuration files with comprehensive validation."""
        from drift_datasets.config import ConfigurationManager

        config_path = sample_toml_configs["sine"]

        # Act: Load configuration
        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_path)

        # Assert: Configuration loaded successfully
        assert config is not None, "Configuration loaded"
        assert "dataset" in config, "Dataset section present"
        assert "generator_config" in config, "Generator config section present"
        assert "drift_config" in config, "Drift config section present"
        assert "metadata" in config, "Metadata section present"

        # Test configuration structure
        dataset_config = config["dataset"]
        assert dataset_config["name"] == "test_sine", "Dataset name correct"
        assert dataset_config["type"] == "synthetic", "Dataset type correct"
        assert dataset_config["generator"] == "SineGenerator", "Generator correct"

    def test_should_validate_configuration_schema_when_loading_config(self, sample_toml_configs):
        """Test REQ-018: Validate configuration against predefined schema."""
        from drift_datasets.config import ConfigurationManager

        config_path = sample_toml_configs["sine"]

        # Act: Load and validate configuration
        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_path)
        validation_result = config_manager.validate_schema(config)

        # Assert: Schema validation successful
        assert validation_result["is_valid"] == True, f"Schema validation failed: {validation_result['errors']}"
        assert validation_result["errors"] == [], "No validation errors"
        assert validation_result["warnings"] == [], "No validation warnings"

        # Test required fields validation
        required_fields = validation_result["validated_fields"]["required"]
        assert "dataset.name" in required_fields, "Dataset name validated"
        assert "dataset.type" in required_fields, "Dataset type validated"
        assert "dataset.generator" in required_fields, "Generator validated"

    def test_should_detect_invalid_configuration_when_schema_validation_fails(self, tmp_path, error_scenarios):
        """Test REQ-018: Detect invalid configuration and provide helpful error messages."""
        from drift_datasets.config import ConfigurationManager

        # Create invalid configuration file
        invalid_config = error_scenarios["invalid_config"]["missing_required_field"]
        invalid_config_path = tmp_path / "invalid.toml"

        import toml

        with open(invalid_config_path, "w") as f:
            toml.dump(invalid_config, f)

        # Act: Load invalid configuration
        config_manager = ConfigurationManager()

        # Assert: Invalid configuration detected
        with pytest.raises(ValueError, match="Configuration validation failed"):
            config = config_manager.load_config(str(invalid_config_path))
            config_manager.validate_schema(config)

    def test_should_support_configuration_inheritance_when_base_configs_used(self, tmp_path, sample_toml_configs):
        """Test REQ-018: Support configuration inheritance and composition."""
        from drift_datasets.config import ConfigurationManager

        # Create base configuration
        base_config_path = tmp_path / "base.toml"
        child_config_path = tmp_path / "child.toml"

        import toml

        # Base configuration
        base_config = {
            "dataset": {"type": "synthetic", "generator": "SineGenerator"},
            "generator_config": {"n_instances": 1000, "random_seed": 42},
        }

        # Child configuration that extends base
        child_config = {
            "extends": str(base_config_path),
            "dataset": {"name": "child_sine"},  # Override name
            "generator_config": {"noise_level": 0.2},  # Add new parameter
        }

        with open(base_config_path, "w") as f:
            toml.dump(base_config, f)
        with open(child_config_path, "w") as f:
            toml.dump(child_config, f)

        # Act: Load child configuration with inheritance
        config_manager = ConfigurationManager()
        config = config_manager.load_config(str(child_config_path))

        # Assert: Configuration inheritance working
        assert config["dataset"]["name"] == "child_sine", "Name overridden"
        assert config["dataset"]["type"] == "synthetic", "Type inherited from base"
        assert config["dataset"]["generator"] == "SineGenerator", "Generator inherited from base"
        assert config["generator_config"]["n_instances"] == 1000, "Instance count inherited"
        assert config["generator_config"]["noise_level"] == 0.2, "Noise level added in child"

    def test_should_support_environment_variable_substitution_when_config_uses_variables(self, tmp_path):
        """Test REQ-018: Support environment variable substitution in configuration."""
        import os

        from drift_datasets.config import ConfigurationManager

        # Set environment variable
        os.environ["DATASET_SIZE"] = "2000"
        os.environ["RANDOM_SEED"] = "123"

        # Create configuration with environment variables
        config_with_env = {
            "dataset": {"name": "env_test", "type": "synthetic", "generator": "SineGenerator"},
            "generator_config": {"n_instances": "${DATASET_SIZE}", "random_seed": "${RANDOM_SEED}"},
        }

        config_path = tmp_path / "env_config.toml"
        import toml

        with open(config_path, "w") as f:
            toml.dump(config_with_env, f)

        # Act: Load configuration with environment substitution
        config_manager = ConfigurationManager()
        config = config_manager.load_config(str(config_path), substitute_env=True)

        # Assert: Environment variables substituted
        assert config["generator_config"]["n_instances"] == 2000, "Environment variable substituted for instance count"
        assert config["generator_config"]["random_seed"] == 123, "Environment variable substituted for random seed"

        # Cleanup
        del os.environ["DATASET_SIZE"]
        del os.environ["RANDOM_SEED"]


class TestParameterValidation:
    """Test parameter validation functionality (REQ-019)."""

    def test_should_validate_generator_parameters_when_creating_synthetic_dataset(self, sample_toml_configs):
        """Test REQ-019: Validate generator-specific parameters against allowed ranges."""
        from drift_datasets.config import ParameterValidator

        config_path = sample_toml_configs["sine"]

        # Act: Load config and validate parameters
        from drift_datasets.config import ConfigurationManager

        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_path)

        validator = ParameterValidator()
        validation_result = validator.validate_generator_parameters("SineGenerator", config["generator_config"])

        # Assert: Parameter validation successful
        assert validation_result["is_valid"] == True, f"Parameter validation failed: {validation_result['errors']}"
        assert validation_result["errors"] == [], "No parameter errors"

        # Test specific parameter validations
        validated_params = validation_result["validated_parameters"]
        assert validated_params["n_instances"]["is_valid"] == True, "Instance count valid"
        assert validated_params["random_seed"]["is_valid"] == True, "Random seed valid"
        assert validated_params["noise_level"]["is_valid"] == True, "Noise level valid"

    def test_should_detect_invalid_parameter_ranges_when_validating_generator_config(self):
        """Test REQ-019: Detect parameters outside valid ranges."""
        from drift_datasets.config import ParameterValidator

        # Invalid parameters
        invalid_params = {
            "n_instances": -100,  # Negative instance count
            "random_seed": "not_a_number",  # Invalid type
            "noise_level": 2.0,  # Outside valid range [0, 1]
        }

        # Act: Validate invalid parameters
        validator = ParameterValidator()
        validation_result = validator.validate_generator_parameters("SineGenerator", invalid_params)

        # Assert: Invalid parameters detected
        assert validation_result["is_valid"] == False, "Invalid parameters detected"
        assert len(validation_result["errors"]) >= 3, "Multiple errors detected"

        # Check specific error types
        error_messages = [error["message"] for error in validation_result["errors"]]
        assert any("negative" in msg.lower() for msg in error_messages), "Negative instance count error"
        assert any("type" in msg.lower() for msg in error_messages), "Type error detected"
        assert any("range" in msg.lower() for msg in error_messages), "Range error detected"

    def test_should_validate_drift_configuration_parameters_when_setting_drift(self, sample_toml_configs):
        """Test REQ-019: Validate drift-specific configuration parameters."""
        from drift_datasets.config import ParameterValidator

        config_path = sample_toml_configs["sine"]

        # Act: Load config and validate drift parameters
        from drift_datasets.config import ConfigurationManager

        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_path)

        validator = ParameterValidator()
        validation_result = validator.validate_drift_parameters(config["drift_config"], config["generator_config"])

        # Assert: Drift parameter validation successful
        assert validation_result["is_valid"] == True, f"Drift validation failed: {validation_result['errors']}"
        assert validation_result["errors"] == [], "No drift parameter errors"

        # Test drift-specific validations
        drift_validations = validation_result["drift_validations"]
        assert drift_validations["drift_points_within_bounds"] == True, "Drift points within dataset bounds"
        assert drift_validations["drift_intensities_valid"] == True, "Drift intensities in valid range"
        assert drift_validations["drift_types_supported"] == True, "Drift types are supported"

    def test_should_validate_uci_dataset_parameters_when_configuring_real_world_dataset(self, sample_toml_configs):
        """Test REQ-019: Validate UCI dataset-specific parameters."""
        from drift_datasets.config import ParameterValidator

        config_path = sample_toml_configs["uci"]

        # Act: Load config and validate UCI parameters
        from drift_datasets.config import ConfigurationManager

        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_path)

        validator = ParameterValidator()
        validation_result = validator.validate_uci_parameters(config["uci_config"])

        # Assert: UCI parameter validation successful
        assert validation_result["is_valid"] == True, f"UCI validation failed: {validation_result['errors']}"
        assert validation_result["errors"] == [], "No UCI parameter errors"

        # Test UCI-specific validations
        uci_validations = validation_result["uci_validations"]
        assert uci_validations["dataset_id_valid"] == True, "UCI dataset ID valid"
        assert uci_validations["preprocessing_options_valid"] == True, "Preprocessing options valid"

    def test_should_provide_parameter_suggestions_when_validation_fails(self):
        """Test REQ-019: Provide helpful suggestions for invalid parameters."""
        from drift_datasets.config import ParameterValidator

        # Parameters with common mistakes
        problematic_params = {"n_instances": 50, "noise_level": 1.5, "random_seed": -1}  # Too small  # Too high  # Negative seed

        # Act: Validate parameters and get suggestions
        validator = ParameterValidator()
        validation_result = validator.validate_generator_parameters("SineGenerator", problematic_params)

        # Assert: Suggestions provided for invalid parameters
        assert validation_result["is_valid"] == False, "Invalid parameters detected"

        suggestions = validation_result["suggestions"]
        assert len(suggestions) >= 3, "Suggestions provided for all invalid parameters"

        # Check suggestion content
        n_instances_suggestion = next(s for s in suggestions if "n_instances" in s["parameter"])
        assert "minimum recommended" in n_instances_suggestion["suggestion"].lower(), "Minimum value suggestion"

        noise_level_suggestion = next(s for s in suggestions if "noise_level" in s["parameter"])
        assert "range" in noise_level_suggestion["suggestion"].lower(), "Range constraint suggestion"


class TestExportFormats:
    """Test dataset export formats functionality (REQ-020)."""

    def test_should_export_to_arff_format_when_arff_export_requested(self, sample_drift_dataset, tmp_path):
        """Test REQ-020: Export datasets to ARFF format for Weka compatibility."""
        from drift_datasets.export import ARFFExporter
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset
        dataset = DriftDataset(**sample_drift_dataset)
        output_path = tmp_path / "test_dataset.arff"

        # Act: Export to ARFF format
        exporter = ARFFExporter()
        export_result = exporter.export(dataset, str(output_path))

        # Assert: ARFF export successful
        assert export_result["success"] == True, "ARFF export successful"
        assert output_path.exists(), "ARFF file created"
        assert export_result["format"] == "arff", "Correct export format"

        # Verify ARFF file content
        with open(output_path, "r") as f:
            arff_content = f.read()

        assert "@relation test_dataset" in arff_content, "ARFF relation header present"
        assert "@attribute feature_0 numeric" in arff_content, "Feature attributes defined"
        assert "@attribute feature_1 numeric" in arff_content, "Feature attributes defined"
        assert "@attribute target {0,1}" in arff_content, "Target attribute defined"
        assert "@data" in arff_content, "Data section present"

    def test_should_export_to_csv_format_when_csv_export_requested(self, sample_drift_dataset, tmp_path):
        """Test REQ-020: Export datasets to CSV format."""
        from drift_datasets.export import CSVExporter
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset
        dataset = DriftDataset(**sample_drift_dataset)
        output_path = tmp_path / "test_dataset.csv"

        # Act: Export to CSV format
        exporter = CSVExporter()
        export_result = exporter.export(dataset, str(output_path))

        # Assert: CSV export successful
        assert export_result["success"] == True, "CSV export successful"
        assert output_path.exists(), "CSV file created"
        assert export_result["format"] == "csv", "Correct export format"

        # Verify CSV file content
        import pandas as pd

        df = pd.read_csv(output_path)
        assert len(df) == 1000, "Correct number of rows"
        assert len(df.columns) == 3, "Correct number of columns"
        assert list(df.columns) == ["feature_0", "feature_1", "target"], "Correct column names"

    def test_should_export_to_parquet_format_when_parquet_export_requested(self, sample_drift_dataset, tmp_path):
        """Test REQ-020: Export datasets to Parquet format."""
        from drift_datasets.export import ParquetExporter
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset
        dataset = DriftDataset(**sample_drift_dataset)
        output_path = tmp_path / "test_dataset.parquet"

        # Act: Export to Parquet format
        exporter = ParquetExporter()
        export_result = exporter.export(dataset, str(output_path))

        # Assert: Parquet export successful
        assert export_result["success"] == True, "Parquet export successful"
        assert output_path.exists(), "Parquet file created"
        assert export_result["format"] == "parquet", "Correct export format"

        # Verify Parquet file content
        import pandas as pd

        df = pd.read_parquet(output_path)
        assert len(df) == 1000, "Correct number of rows"
        assert len(df.columns) == 3, "Correct number of columns"
        assert list(df.columns) == ["feature_0", "feature_1", "target"], "Correct column names"

    def test_should_export_metadata_separately_when_metadata_export_requested(self, sample_drift_dataset, tmp_path):
        """Test REQ-020: Export dataset metadata separately in JSON format."""
        from drift_datasets.export import MetadataExporter
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset
        dataset = DriftDataset(**sample_drift_dataset)
        output_path = tmp_path / "test_dataset_metadata.json"

        # Act: Export metadata
        exporter = MetadataExporter()
        export_result = exporter.export(dataset, str(output_path))

        # Assert: Metadata export successful
        assert export_result["success"] == True, "Metadata export successful"
        assert output_path.exists(), "Metadata file created"
        assert export_result["format"] == "json", "Correct metadata format"

        # Verify metadata file content
        import json

        with open(output_path, "r") as f:
            metadata = json.load(f)

        assert "drift_metadata" in metadata, "Drift metadata present"
        assert "dataset_metadata" in metadata, "Dataset metadata present"
        assert metadata["drift_metadata"]["drift_points"] == [500], "Drift points preserved"
        assert metadata["dataset_metadata"]["n_classes"] == 2, "Class count preserved"

    def test_should_support_batch_export_when_multiple_formats_requested(self, sample_drift_dataset, tmp_path):
        """Test REQ-020: Support batch export to multiple formats simultaneously."""
        from drift_datasets.export import BatchExporter
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset
        dataset = DriftDataset(**sample_drift_dataset)
        output_dir = tmp_path / "exports"

        # Act: Batch export to multiple formats
        exporter = BatchExporter()
        export_result = exporter.export_multiple(dataset, str(output_dir), formats=["csv", "parquet", "arff", "json"])

        # Assert: Batch export successful
        assert export_result["success"] == True, "Batch export successful"
        assert len(export_result["exported_files"]) == 4, "Four files exported"

        # Verify all files created
        assert (output_dir / "test_dataset.csv").exists(), "CSV file created"
        assert (output_dir / "test_dataset.parquet").exists(), "Parquet file created"
        assert (output_dir / "test_dataset.arff").exists(), "ARFF file created"
        assert (output_dir / "test_dataset_metadata.json").exists(), "Metadata file created"

        # Test export summary
        export_summary = export_result["export_summary"]
        assert export_summary["total_formats"] == 4, "Total format count correct"
        assert export_summary["successful_exports"] == 4, "All exports successful"
        assert export_summary["failed_exports"] == 0, "No failed exports"

    def test_should_preserve_data_integrity_when_round_trip_export_import(self, sample_drift_dataset, tmp_path):
        """Test REQ-020: Preserve data integrity during export/import round trips."""
        from drift_datasets.export import ParquetExporter
        from drift_datasets.models import DriftDataset

        # Arrange: Create original dataset
        original_dataset = DriftDataset(**sample_drift_dataset)
        export_path = tmp_path / "roundtrip_dataset.parquet"

        # Act: Export and reload dataset
        exporter = ParquetExporter()
        export_result = exporter.export(original_dataset, str(export_path))
        reloaded_dataset = DriftDataset.load(str(export_path))

        # Assert: Data integrity preserved
        import numpy as np

        assert np.array_equal(original_dataset.X.values, reloaded_dataset.X.values), "Features preserved"
        assert np.array_equal(original_dataset.y.values, reloaded_dataset.y.values), "Targets preserved"
        assert original_dataset.name == reloaded_dataset.name, "Name preserved"
        assert original_dataset.source_type == reloaded_dataset.source_type, "Source type preserved"

        # Test metadata preservation
        assert original_dataset.drift_metadata == reloaded_dataset.drift_metadata, "Drift metadata preserved"
        assert original_dataset.dataset_metadata == reloaded_dataset.dataset_metadata, "Dataset metadata preserved"
