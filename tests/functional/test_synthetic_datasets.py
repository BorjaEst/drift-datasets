"""
Functional tests for synthetic dataset generation functionality.

These tests validate synthetic dataset generation from REQ-007 through REQ-011,
including CapyMOA integration, ExpertSystems compatibility, and ground truth metadata.
"""

import pytest


class TestSyntheticDatasetGeneration:
    """Test synthetic dataset generation functionality (REQ-007)."""

    def test_should_create_sine_dataset_when_sine_generator_requested(self, capymoa_service, sample_toml_configs):
        """Test REQ-007: Generate synthetic datasets using CapyMOA generators - Sine dataset."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Generate Sine dataset
        dataset = create_dataset(config_path)

        # Assert: Sine dataset characteristics
        assert dataset.name == "test_sine"
        assert dataset.source_type == "synthetic"
        assert dataset.X.shape == (1000, 2), "Sine dataset has 2 features"
        assert len(dataset.y) == 1000, "Correct number of instances"
        assert set(dataset.y.unique()) == {0, 1}, "Binary classification"

        # Verify sine-specific metadata
        generator_info = dataset.dataset_metadata["generator"]
        assert generator_info["name"] == "SineGenerator", "Correct generator recorded"
        assert generator_info["noise_level"] == 0.1, "Noise level recorded"

    def test_should_create_hyperplane_dataset_when_hyperplane_generator_requested(self, capymoa_service, sample_toml_configs):
        """Test REQ-007: Generate synthetic datasets using CapyMOA generators - Hyperplane dataset."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["hyperplane"]

        # Act: Generate Hyperplane dataset
        dataset = create_dataset(config_path)

        # Assert: Hyperplane dataset characteristics
        assert dataset.name == "test_hyperplane"
        assert dataset.source_type == "synthetic"
        assert dataset.X.shape[1] == 10, "Hyperplane dataset has 10 features"
        assert len(dataset.y) == 1000, "Correct number of instances"

        # Verify hyperplane-specific metadata
        generator_info = dataset.dataset_metadata["generator"]
        assert generator_info["name"] == "HyperplaneGenerator", "Correct generator recorded"
        assert generator_info["n_features"] == 10, "Feature count recorded"

    def test_should_create_stagger_dataset_when_stagger_generator_requested(self, capymoa_service, sample_toml_configs):
        """Test REQ-007: Generate synthetic datasets using CapyMOA generators - STAGGER dataset."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["stagger"]

        # Act: Generate STAGGER dataset
        dataset = create_dataset(config_path)

        # Assert: STAGGER dataset characteristics
        assert dataset.name == "test_stagger"
        assert dataset.source_type == "synthetic"
        assert dataset.X.shape[1] == 3, "STAGGER dataset has 3 features"
        assert len(dataset.y) == 1000, "Correct number of instances"

        # Verify STAGGER-specific metadata
        generator_info = dataset.dataset_metadata["generator"]
        assert generator_info["name"] == "STAGGERGenerator", "Correct generator recorded"
        assert "concept_index" in generator_info, "Concept index recorded"

    def test_should_create_sea_dataset_when_sea_generator_requested(self, capymoa_service, sample_toml_configs):
        """Test REQ-007: Generate synthetic datasets using CapyMOA generators - SEA dataset."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sea"]

        # Act: Generate SEA dataset
        dataset = create_dataset(config_path)

        # Assert: SEA dataset characteristics
        assert dataset.name == "test_sea"
        assert dataset.source_type == "synthetic"
        assert dataset.X.shape[1] == 3, "SEA dataset has 3 features"
        assert len(dataset.y) == 1000, "Correct number of instances"

        # Verify SEA-specific metadata
        generator_info = dataset.dataset_metadata["generator"]
        assert generator_info["name"] == "SEAGenerator", "Correct generator recorded"
        assert "threshold" in generator_info, "Threshold parameter recorded"


class TestCapyMOAIntegration:
    """Test CapyMOA integration functionality (REQ-008)."""

    def test_should_interface_with_capymoa_when_synthetic_dataset_requested(self, capymoa_service):
        """Test REQ-008: Interface with CapyMOA library for deterministic generation."""
        from drift_datasets.generators import CapyMOAInterface

        # Act: Initialize CapyMOA interface
        interface = CapyMOAInterface()
        generator = interface.create_generator("SineGenerator", {"random_seed": 42, "noise_level": 0.1})

        # Assert: CapyMOA interface working
        assert generator is not None, "Generator created successfully"
        assert hasattr(generator, "generate"), "Generator has generate method"

        # Test generator parameters
        params = interface.get_generator_parameters("SineGenerator")
        assert "random_seed" in params, "Random seed parameter available"
        assert "noise_level" in params, "Noise level parameter available"

    def test_should_handle_capymoa_parameter_translation_when_creating_generators(self, capymoa_service, parameter_translator):
        """Test REQ-008: Handle parameter translation between our API and CapyMOA."""
        from drift_datasets.generators import CapyMOAInterface

        # Arrange: Define our parameter format
        our_params = {"n_instances": 1000, "random_seed": 42, "noise_level": 0.1, "drift_points": [500]}

        # Act: Translate parameters using interface
        interface = CapyMOAInterface()
        capymoa_params = interface.translate_parameters("SineGenerator", our_params)

        # Assert: Parameters translated correctly
        assert "numInstances" in capymoa_params, "Instance count translated"
        assert "randomSeed" in capymoa_params, "Random seed translated"
        assert "noiseLevel" in capymoa_params, "Noise level translated"
        assert capymoa_params["numInstances"] == 1000, "Correct instance count"

    def test_should_validate_capymoa_availability_when_creating_synthetic_datasets(self, capymoa_service):
        """Test REQ-008: Validate CapyMOA library availability and version."""
        from drift_datasets.generators import CapyMOAInterface

        # Act: Check CapyMOA availability
        interface = CapyMOAInterface()
        is_available, version_info = interface.check_availability()

        # Assert: CapyMOA validation working
        assert is_available == True, "CapyMOA should be available for testing"
        assert "version" in version_info, "Version information provided"
        assert version_info["version"] >= "1.0.0", "Minimum version requirement met"

    def test_should_handle_capymoa_errors_gracefully_when_generation_fails(self, capymoa_service, error_scenarios):
        """Test REQ-008: Handle CapyMOA generation errors gracefully."""
        from drift_datasets.generators import CapyMOAInterface

        interface = CapyMOAInterface()

        # Test invalid generator name
        with pytest.raises(ValueError, match="Unsupported generator: InvalidGenerator"):
            interface.create_generator("InvalidGenerator", {})

        # Test invalid parameters
        with pytest.raises(ValueError, match="Invalid parameter.*for generator"):
            interface.create_generator("SineGenerator", {"invalid_param": "value"})


class TestExpertSystemsCompatibility:
    """Test ExpertSystems paper compatibility functionality (REQ-009)."""

    def test_should_reproduce_expertsystems_sine_dataset_when_configured_correctly(self, capymoa_service, expertsystems_configs):
        """Test REQ-009: Reproduce ExpertSystems paper datasets with exact parameters."""
        from drift_datasets import create_dataset

        # Act: Create ExpertSystems-compatible Sine dataset
        dataset = create_dataset(expertsystems_configs["sine_expertsystems"])

        # Assert: ExpertSystems Sine dataset characteristics
        assert dataset.name == "ExpertSystems_Sine"
        assert dataset.X.shape == (100000, 2), "ExpertSystems uses 100k instances"
        assert len(dataset.drift_metadata["drift_points"]) == 4, "4 drift points as in paper"
        assert dataset.drift_metadata["drift_points"] == [25000, 50000, 75000], "Exact drift points from paper"

        # Verify ExpertSystems compatibility metadata
        compatibility_info = dataset.dataset_metadata["compatibility"]
        assert compatibility_info["source_paper"] == "ExpertSystems", "Paper reference recorded"
        assert compatibility_info["paper_section"] == "5.1", "Section reference recorded"

    def test_should_reproduce_expertsystems_hyperplane_dataset_when_configured_correctly(self, capymoa_service, expertsystems_configs):
        """Test REQ-009: Reproduce ExpertSystems Hyperplane dataset."""
        from drift_datasets import create_dataset

        # Act: Create ExpertSystems-compatible Hyperplane dataset
        dataset = create_dataset(expertsystems_configs["hyperplane_expertsystems"])

        # Assert: ExpertSystems Hyperplane dataset characteristics
        assert dataset.name == "ExpertSystems_Hyperplane"
        assert dataset.X.shape == (100000, 10), "10 features as in paper"
        assert dataset.drift_metadata["drift_intensities"] == [0.0, 0.5, 1.0, 0.9], "Exact intensities from paper"

        # Verify hyperplane-specific ExpertSystems parameters
        generator_info = dataset.dataset_metadata["generator"]
        assert generator_info["magnitude_change"] == 0.9, "Magnitude change from paper"
        assert generator_info["n_drift_features"] == 5, "Half features change as in paper"

    def test_should_reproduce_expertsystems_stagger_dataset_when_configured_correctly(self, capymoa_service, expertsystems_configs):
        """Test REQ-009: Reproduce ExpertSystems STAGGER dataset."""
        from drift_datasets import create_dataset

        # Act: Create ExpertSystems-compatible STAGGER dataset
        dataset = create_dataset(expertsystems_configs["stagger_expertsystems"])

        # Assert: ExpertSystems STAGGER dataset characteristics
        assert dataset.name == "ExpertSystems_STAGGER"
        assert len(dataset.drift_metadata["concept_definitions"]) == 3, "3 concepts as in paper"

        # Verify concept transitions match paper
        concept_transitions = dataset.drift_metadata["concept_transitions"]
        expected_transitions = ["C1->C2", "C2->C3", "C3->C1"]
        assert concept_transitions == expected_transitions, "Exact concept transitions from paper"

    def test_should_provide_expertsystems_citation_when_compatibility_enabled(self, expertsystems_configs):
        """Test REQ-009: Provide proper citation information for ExpertSystems compatibility."""
        from drift_datasets import create_dataset

        # Act: Create ExpertSystems-compatible dataset
        dataset = create_dataset(expertsystems_configs["sine_expertsystems"])

        # Assert: Citation information provided
        citation_info = dataset.get_citation_info()
        assert "ExpertSystems" in citation_info["primary_paper"], "Primary paper citation"
        assert "drift_datasets" in citation_info["library_paper"], "Library citation"
        assert citation_info["bibtex"] is not None, "BibTeX citation provided"

        # Verify citation components
        bibtex = citation_info["bibtex"]
        assert "@article" in bibtex, "BibTeX format correct"
        assert "2014" in bibtex, "Publication year included"


class TestGroundTruthMetadata:
    """Test ground truth metadata functionality (REQ-010)."""

    def test_should_provide_precise_drift_timing_when_dataset_generated(self, capymoa_service, sample_toml_configs):
        """Test REQ-010: Provide precise ground truth metadata for drift timing."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Generate dataset with drift
        dataset = create_dataset(config_path)

        # Assert: Precise drift timing provided
        drift_metadata = dataset.drift_metadata
        assert "drift_points" in drift_metadata, "Drift points specified"
        assert "drift_timing_precision" in drift_metadata, "Timing precision specified"
        assert drift_metadata["drift_timing_precision"] == "instance_level", "Instance-level precision"

        # Test drift boundary detection
        drift_boundaries = drift_metadata["drift_boundaries"]
        assert len(drift_boundaries) == 1, "One drift boundary for one drift point"
        assert drift_boundaries[0]["start"] == 500, "Drift start point precise"
        assert drift_boundaries[0]["end"] == 500, "Abrupt drift (start == end)"

    def test_should_provide_drift_intensity_measurements_when_dataset_generated(
        self, capymoa_service, sample_toml_configs, validation_utilities
    ):
        """Test REQ-010: Provide quantitative drift intensity measurements."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["hyperplane"]

        # Act: Generate dataset with measured drift intensity
        dataset = create_dataset(config_path)

        # Assert: Drift intensity measurements provided
        drift_metadata = dataset.drift_metadata
        validation_utilities["validate_drift_intensities"](drift_metadata, expected_range=(0.0, 1.0))

        assert "drift_intensities" in drift_metadata, "Drift intensities specified"
        assert "intensity_calculation_method" in drift_metadata, "Calculation method specified"

        # Test intensity values
        intensities = drift_metadata["drift_intensities"]
        assert all(0.0 <= intensity <= 1.0 for intensity in intensities), "Valid intensity range"
        assert drift_metadata["intensity_calculation_method"] == "parameter_change_magnitude", "Correct method"

    def test_should_provide_affected_features_when_dataset_generated(self, capymoa_service, sample_toml_configs):
        """Test REQ-010: Identify which features are affected by each drift."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["hyperplane"]

        # Act: Generate dataset with feature-level drift information
        dataset = create_dataset(config_path)

        # Assert: Affected features identified
        drift_metadata = dataset.drift_metadata
        assert "affected_features" in drift_metadata, "Affected features specified"

        affected_features = drift_metadata["affected_features"]
        assert len(affected_features) == len(drift_metadata["drift_points"]), "One per drift point"

        # Test feature specification
        first_drift_features = affected_features[0]
        assert "feature_indices" in first_drift_features, "Feature indices specified"
        assert "feature_names" in first_drift_features, "Feature names specified"
        assert "change_type" in first_drift_features, "Change type specified"

    def test_should_provide_pre_post_drift_statistics_when_dataset_generated(self, capymoa_service, sample_toml_configs):
        """Test REQ-010: Provide statistical comparison before and after drift."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Generate dataset with drift statistics
        dataset = create_dataset(config_path)

        # Assert: Pre/post drift statistics available
        drift_stats = dataset.get_drift_statistics()
        assert len(drift_stats) == 1, "Statistics for one drift point"

        first_drift_stats = drift_stats[0]
        assert "pre_drift_stats" in first_drift_stats, "Pre-drift statistics"
        assert "post_drift_stats" in first_drift_stats, "Post-drift statistics"
        assert "statistical_distance" in first_drift_stats, "Statistical distance measure"

        # Test statistical measures
        pre_stats = first_drift_stats["pre_drift_stats"]
        post_stats = first_drift_stats["post_drift_stats"]
        assert "mean" in pre_stats, "Mean statistics available"
        assert "covariance" in pre_stats, "Covariance statistics available"
        assert pre_stats["mean"] != post_stats["mean"], "Drift detected in means"


class TestSyntheticDatasetValidation:
    """Test synthetic dataset validation functionality (REQ-011)."""

    def test_should_validate_generator_parameters_when_creating_synthetic_dataset(self, capymoa_service, sample_toml_configs):
        """Test REQ-011: Validate generator parameters and dataset consistency."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Create dataset (validation occurs during creation)
        dataset = create_dataset(config_path)

        # Assert: Generator parameters validated
        generator_validation = dataset.validate_generator_consistency()
        assert generator_validation["is_valid"] == True, "Generator consistency validated"
        assert generator_validation["parameter_check"] == "passed", "Parameters validated"

        # Test specific parameter validation
        params = dataset.dataset_metadata["generator"]
        assert params["random_seed"] == 42, "Random seed preserved"
        assert 0.0 <= params["noise_level"] <= 1.0, "Noise level in valid range"

    def test_should_detect_parameter_inconsistencies_when_validating_synthetic_dataset(self, error_scenarios):
        """Test REQ-011: Detect inconsistencies in synthetic dataset generation."""
        from drift_datasets import create_dataset

        # Test conflicting parameters
        with pytest.raises(ValueError, match="Conflicting parameters"):
            create_dataset(error_scenarios["synthetic_generation"]["conflicting_parameters"])

        # Test out-of-range parameters
        with pytest.raises(ValueError, match="Parameter.*out of valid range"):
            create_dataset(error_scenarios["synthetic_generation"]["invalid_parameter_range"])

    def test_should_verify_drift_implementation_when_validating_synthetic_dataset(self, capymoa_service, sample_toml_configs):
        """Test REQ-011: Verify drift is implemented correctly in synthetic data."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Create dataset and verify drift implementation
        dataset = create_dataset(config_path)
        drift_verification = dataset.verify_drift_implementation()

        # Assert: Drift correctly implemented
        assert drift_verification["drift_detected"] == True, "Drift detection successful"
        assert drift_verification["drift_location_accurate"] == True, "Drift location accurate"
        assert drift_verification["drift_magnitude_correct"] == True, "Drift magnitude correct"

        # Test statistical tests for drift detection
        statistical_tests = drift_verification["statistical_tests"]
        assert statistical_tests["ks_test"]["p_value"] < 0.05, "KS test detects distribution change"
        assert statistical_tests["chi2_test"]["p_value"] < 0.05, "Chi-square test detects class distribution change"

    def test_should_validate_reproducibility_when_using_random_seeds(self, capymoa_service, sample_toml_configs):
        """Test REQ-011: Validate reproducibility with random seeds."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Create same dataset twice with same seed
        dataset1 = create_dataset(config_path)
        dataset2 = create_dataset(config_path)

        # Assert: Datasets are identical
        import numpy as np

        assert np.array_equal(dataset1.X.values, dataset2.X.values), "Features identical with same seed"
        assert np.array_equal(dataset1.y.values, dataset2.y.values), "Targets identical with same seed"
        assert dataset1.drift_metadata == dataset2.drift_metadata, "Metadata identical with same seed"

        # Test different seeds produce different data
        config_different_seed = sample_toml_configs["sine_different_seed"]
        dataset3 = create_dataset(config_different_seed)
        assert not np.array_equal(dataset1.X.values, dataset3.X.values), "Different seeds produce different data"
