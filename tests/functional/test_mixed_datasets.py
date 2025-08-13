"""
Functional tests for mixed dataset functionality.

These tests validate mixed dataset creation from REQ-015 through REQ-017,
including combining synthetic and real-world data, temporal mixing strategies, and validation.
"""

import pytest


class TestMixedDatasetCreation:
    """Test mixed dataset creation functionality (REQ-015)."""

    def test_should_create_sequential_mixed_dataset_when_sequential_strategy_configured(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-015: Create mixed datasets by combining synthetic and real-world data - Sequential strategy."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]

        # Act: Create sequential mixed dataset
        dataset = create_dataset(config_path)

        # Assert: Sequential mixed dataset characteristics
        assert dataset.name == "test_mixed"
        assert dataset.source_type == "mixed"
        assert dataset.X.shape[0] == 1000, "Combined instance count"

        # Verify mixing metadata
        mixing_info = dataset.dataset_metadata["mixing"]
        assert mixing_info["strategy"] == "sequential", "Sequential mixing strategy"
        assert mixing_info["transition_point"] == 500, "Transition at 500 instances"
        assert len(mixing_info["components"]) == 2, "Two data components"

        # Test component information
        components = mixing_info["components"]
        synthetic_component = components[0]
        real_component = components[1]

        assert synthetic_component["source_type"] == "synthetic", "First component is synthetic"
        assert synthetic_component["instances"] == 500, "First component has 500 instances"
        assert real_component["source_type"] == "real_world", "Second component is real-world"
        assert real_component["instances"] == 500, "Second component has 500 instances"

    def test_should_create_interleaved_mixed_dataset_when_interleaved_strategy_configured(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-015: Create mixed datasets with interleaved strategy."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed_interleaved"]

        # Act: Create interleaved mixed dataset
        dataset = create_dataset(config_path)

        # Assert: Interleaved mixed dataset characteristics
        mixing_info = dataset.dataset_metadata["mixing"]
        assert mixing_info["strategy"] == "interleaved", "Interleaved mixing strategy"
        assert mixing_info["interleave_pattern"] == "alternating", "Alternating pattern"
        assert mixing_info["block_size"] == 50, "Block size of 50 instances"

        # Verify interleaving in dataset metadata
        instance_sources = dataset.dataset_metadata["instance_sources"]
        assert len(instance_sources) == 1000, "Source tracking for all instances"

        # Check alternating pattern
        first_block_sources = instance_sources[:50]
        second_block_sources = instance_sources[50:100]
        assert all(source == "synthetic" for source in first_block_sources), "First block is synthetic"
        assert all(source == "real_world" for source in second_block_sources), "Second block is real-world"

    def test_should_create_weighted_mixed_dataset_when_weighted_strategy_configured(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-015: Create mixed datasets with weighted sampling strategy."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed_weighted"]

        # Act: Create weighted mixed dataset
        dataset = create_dataset(config_path)

        # Assert: Weighted mixed dataset characteristics
        mixing_info = dataset.dataset_metadata["mixing"]
        assert mixing_info["strategy"] == "weighted_sampling", "Weighted sampling strategy"

        weights = mixing_info["component_weights"]
        assert len(weights) == 2, "Two component weights"
        assert abs(weights[0] + weights[1] - 1.0) < 0.01, "Weights sum to 1.0"

        # Verify weight distribution in final dataset
        instance_sources = dataset.dataset_metadata["instance_sources"]
        synthetic_count = sum(1 for source in instance_sources if source == "synthetic")
        real_count = sum(1 for source in instance_sources if source == "real_world")

        expected_synthetic = int(1000 * weights[0])
        expected_real = int(1000 * weights[1])

        assert abs(synthetic_count - expected_synthetic) <= 50, "Synthetic instances match weight"
        assert abs(real_count - expected_real) <= 50, "Real-world instances match weight"

    def test_should_handle_feature_alignment_when_combining_different_datasets(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-015: Handle feature alignment when combining datasets with different schemas."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed_different_features"]

        # Act: Create mixed dataset with different feature schemas
        dataset = create_dataset(config_path)

        # Assert: Feature alignment handled
        alignment_info = dataset.dataset_metadata["feature_alignment"]
        assert "alignment_strategy" in alignment_info, "Alignment strategy recorded"
        assert alignment_info["alignment_strategy"] == "intersection", "Using intersection strategy"

        # Check common features identified
        common_features = alignment_info["common_features"]
        assert len(common_features) > 0, "Common features identified"
        assert dataset.X.shape[1] == len(common_features), "Dataset uses only common features"

        # Verify feature mapping
        feature_mapping = alignment_info["feature_mapping"]
        assert len(feature_mapping) == 2, "Mapping for both components"
        assert all(
            len(mapping["original_features"]) >= len(mapping["aligned_features"]) for mapping in feature_mapping
        ), "Feature reduction during alignment"


class TestMixedDatasetValidation:
    """Test mixed dataset validation functionality (REQ-016)."""

    def test_should_validate_component_compatibility_when_creating_mixed_dataset(
        self, capymoa_service, uci_repository_service, sample_toml_configs, validation_utilities
    ):
        """Test REQ-016: Validate compatibility between dataset components."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]

        # Act: Create mixed dataset (validation occurs during creation)
        dataset = create_dataset(config_path)

        # Assert: Component compatibility validated
        compatibility_report = dataset.validate_component_compatibility()
        assert compatibility_report["components_compatible"] == True, "Components are compatible"
        assert compatibility_report["feature_compatibility"] == "aligned", "Features aligned successfully"
        assert compatibility_report["target_compatibility"] == "compatible", "Targets are compatible"

        # Test specific compatibility checks
        validation_utilities["validate_mixed_dataset_consistency"](dataset, expected_components=2, expected_total_instances=1000)

    def test_should_detect_incompatible_components_when_validating_mixed_dataset(self, error_scenarios):
        """Test REQ-016: Detect incompatible components in mixed dataset configuration."""
        from drift_datasets import create_dataset

        # Test incompatible feature types
        with pytest.raises(ValueError, match="Incompatible feature types"):
            create_dataset(error_scenarios["mixed_dataset"]["incompatible_features"])

        # Test incompatible target types
        with pytest.raises(ValueError, match="Incompatible target types"):
            create_dataset(error_scenarios["mixed_dataset"]["incompatible_targets"])

        # Test mismatched dimensions
        with pytest.raises(ValueError, match="Mismatched feature dimensions"):
            create_dataset(error_scenarios["mixed_dataset"]["mismatched_dimensions"])

    def test_should_validate_mixing_strategy_parameters_when_creating_mixed_dataset(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-016: Validate mixing strategy parameters."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed_weighted"]

        # Act: Create dataset and validate mixing parameters
        dataset = create_dataset(config_path)
        mixing_validation = dataset.validate_mixing_parameters()

        # Assert: Mixing parameters validated
        assert mixing_validation["weights_valid"] == True, "Component weights are valid"
        assert mixing_validation["strategy_valid"] == True, "Mixing strategy is valid"
        assert mixing_validation["parameters_consistent"] == True, "Parameters are consistent"

        # Test weight constraints
        weights = dataset.dataset_metadata["mixing"]["component_weights"]
        assert all(0.0 <= weight <= 1.0 for weight in weights), "Weights in valid range"
        assert abs(sum(weights) - 1.0) < 0.001, "Weights sum to 1.0"

    def test_should_verify_mixed_dataset_statistics_when_validating_dataset(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-016: Verify statistical properties of mixed datasets."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]

        # Act: Create mixed dataset and verify statistics
        dataset = create_dataset(config_path)
        statistics_validation = dataset.validate_mixed_statistics()

        # Assert: Mixed dataset statistics are valid
        assert statistics_validation["distribution_continuity"] == "valid", "Distribution continuity maintained"
        assert statistics_validation["class_balance_preserved"] == True, "Class balance preserved"
        assert statistics_validation["feature_distributions_valid"] == True, "Feature distributions valid"

        # Test component boundary statistics
        boundary_stats = statistics_validation["component_boundaries"]
        for boundary in boundary_stats:
            assert boundary["transition_smoothness"] >= 0.8, "Smooth transitions between components"
            assert boundary["statistical_consistency"] == True, "Statistical consistency at boundaries"


class TestMixedDatasetMetadata:
    """Test mixed dataset metadata functionality (REQ-017)."""

    def test_should_provide_comprehensive_component_metadata_when_mixed_dataset_created(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-017: Provide comprehensive metadata for mixed dataset components."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]

        # Act: Create mixed dataset
        dataset = create_dataset(config_path)

        # Assert: Component metadata available
        component_metadata = dataset.dataset_metadata["component_metadata"]
        assert len(component_metadata) == 2, "Metadata for both components"

        # Test synthetic component metadata
        synthetic_meta = component_metadata[0]
        assert synthetic_meta["component_type"] == "synthetic", "Synthetic component identified"
        assert "generator_info" in synthetic_meta, "Generator information provided"
        assert "generation_parameters" in synthetic_meta, "Generation parameters recorded"
        assert synthetic_meta["instances_contributed"] == 500, "Instance contribution tracked"

        # Test real-world component metadata
        real_meta = component_metadata[1]
        assert real_meta["component_type"] == "real_world", "Real-world component identified"
        assert "source_dataset" in real_meta, "Source dataset information provided"
        assert "preprocessing_applied" in real_meta, "Preprocessing information recorded"
        assert real_meta["instances_contributed"] == 500, "Instance contribution tracked"

    def test_should_track_instance_provenance_when_mixed_dataset_created(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-017: Track instance-level provenance in mixed datasets."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]

        # Act: Create mixed dataset
        dataset = create_dataset(config_path)

        # Assert: Instance provenance tracked
        provenance = dataset.get_instance_provenance()
        assert len(provenance) == 1000, "Provenance for all instances"

        # Test provenance information
        first_instance = provenance[0]
        assert "component_id" in first_instance, "Component ID tracked"
        assert "component_type" in first_instance, "Component type tracked"
        assert "original_index" in first_instance, "Original index tracked"
        assert "generation_timestamp" in first_instance, "Generation timestamp tracked"

        # Verify provenance distribution
        synthetic_instances = [p for p in provenance if p["component_type"] == "synthetic"]
        real_instances = [p for p in provenance if p["component_type"] == "real_world"]
        assert len(synthetic_instances) == 500, "Correct synthetic instance count"
        assert len(real_instances) == 500, "Correct real-world instance count"

    def test_should_provide_mixing_statistics_when_mixed_dataset_analyzed(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-017: Provide detailed mixing statistics and analysis."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed_interleaved"]

        # Act: Create mixed dataset and get statistics
        dataset = create_dataset(config_path)
        mixing_stats = dataset.get_mixing_statistics()

        # Assert: Mixing statistics provided
        assert "component_distribution" in mixing_stats, "Component distribution statistics"
        assert "mixing_pattern_analysis" in mixing_stats, "Mixing pattern analysis"
        assert "transition_analysis" in mixing_stats, "Transition analysis"

        # Test component distribution
        component_dist = mixing_stats["component_distribution"]
        assert component_dist["synthetic_percentage"] == 50.0, "50% synthetic instances"
        assert component_dist["real_world_percentage"] == 50.0, "50% real-world instances"

        # Test pattern analysis
        pattern_analysis = mixing_stats["mixing_pattern_analysis"]
        assert pattern_analysis["pattern_type"] == "interleaved", "Correct pattern identified"
        assert pattern_analysis["regularity_score"] >= 0.8, "High regularity in pattern"

    def test_should_support_component_filtering_when_analyzing_mixed_dataset(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-017: Support filtering and analysis by component type."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]

        # Act: Create mixed dataset and filter components
        dataset = create_dataset(config_path)

        # Test synthetic component filtering
        synthetic_subset = dataset.filter_by_component("synthetic")
        assert len(synthetic_subset.X) == 500, "Synthetic subset has correct size"
        assert all(
            provenance["component_type"] == "synthetic" for provenance in synthetic_subset.get_instance_provenance()
        ), "All instances are synthetic"

        # Test real-world component filtering
        real_subset = dataset.filter_by_component("real_world")
        assert len(real_subset.X) == 500, "Real-world subset has correct size"
        assert all(
            provenance["component_type"] == "real_world" for provenance in real_subset.get_instance_provenance()
        ), "All instances are real-world"

        # Test component analysis
        synthetic_analysis = synthetic_subset.analyze_component()
        real_analysis = real_subset.analyze_component()

        assert synthetic_analysis["component_type"] == "synthetic", "Synthetic analysis correct"
        assert "generator_characteristics" in synthetic_analysis, "Generator characteristics provided"
        assert real_analysis["component_type"] == "real_world", "Real-world analysis correct"
        assert "dataset_characteristics" in real_analysis, "Dataset characteristics provided"

    def test_should_validate_mixed_dataset_reproducibility_when_using_seeds(
        self, capymoa_service, uci_repository_service, sample_toml_configs
    ):
        """Test REQ-017: Validate reproducibility of mixed dataset generation."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["mixed"]

        # Act: Create same mixed dataset twice
        dataset1 = create_dataset(config_path)
        dataset2 = create_dataset(config_path)

        # Assert: Mixed datasets are reproducible
        import numpy as np

        assert np.array_equal(dataset1.X.values, dataset2.X.values), "Features identical with same configuration"
        assert np.array_equal(dataset1.y.values, dataset2.y.values), "Targets identical with same configuration"

        # Test provenance reproducibility
        provenance1 = dataset1.get_instance_provenance()
        provenance2 = dataset2.get_instance_provenance()

        for p1, p2 in zip(provenance1, provenance2):
            assert p1["component_type"] == p2["component_type"], "Component types match"
            assert p1["component_id"] == p2["component_id"], "Component IDs match"
            assert p1["original_index"] == p2["original_index"], "Original indices match"

        # Test mixing metadata reproducibility
        mixing1 = dataset1.dataset_metadata["mixing"]
        mixing2 = dataset2.dataset_metadata["mixing"]
        assert mixing1 == mixing2, "Mixing metadata identical"
