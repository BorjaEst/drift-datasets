"""
Functional tests for real-world dataset integration functionality.

These tests validate real-world dataset integration from REQ-012 through REQ-014,
including UCI ML Repository integration, dataset preprocessing, and metadata enrichment.
"""

import pytest


class TestRealWorldDatasetIntegration:
    """Test real-world dataset integration functionality (REQ-012)."""

    def test_should_fetch_electricity_dataset_when_uci_config_provided(self, uci_repository_service, sample_toml_configs):
        """Test REQ-012: Integrate real-world datasets from UCI ML Repository - Electricity dataset."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci"]

        # Act: Fetch Electricity dataset from UCI
        dataset = create_dataset(config_path)

        # Assert: Electricity dataset characteristics
        assert dataset.name == "test_electricity"
        assert dataset.source_type == "real_world"
        assert dataset.dataset_metadata["uci_id"] == 321, "Correct UCI repository ID"
        assert dataset.X.shape[1] == 8, "Electricity dataset has 8 features"
        assert set(dataset.y.unique()) <= {0, 1}, "Binary classification (UP/DOWN)"

        # Verify UCI-specific metadata
        uci_info = dataset.dataset_metadata["uci_metadata"]
        assert uci_info["dataset_name"] == "ElectricityLoadDiagrams20112014", "Official UCI name"
        assert uci_info["repository_url"] is not None, "Repository URL recorded"

    def test_should_fetch_covertype_dataset_when_covertype_config_provided(self, uci_repository_service, sample_toml_configs):
        """Test REQ-012: Integrate Covertype dataset from UCI."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["covertype"]

        # Act: Fetch Covertype dataset from UCI
        dataset = create_dataset(config_path)

        # Assert: Covertype dataset characteristics
        assert dataset.name == "test_covertype"
        assert dataset.source_type == "real_world"
        assert dataset.dataset_metadata["uci_id"] == 31, "Correct UCI repository ID"
        assert dataset.X.shape[1] == 54, "Covertype dataset has 54 features"
        assert dataset.dataset_metadata["n_classes"] == 7, "7 forest cover types"

        # Verify feature types
        feature_metadata = dataset.dataset_metadata["features"]
        continuous_features = [f for f in feature_metadata if f["role"] == "feature" and f["type"] == "continuous"]
        categorical_features = [f for f in feature_metadata if f["role"] == "feature" and f["type"] == "binary"]
        assert len(continuous_features) == 10, "10 continuous features"
        assert len(categorical_features) == 44, "44 binary features"

    def test_should_fetch_poker_hand_dataset_when_poker_config_provided(self, uci_repository_service, sample_toml_configs):
        """Test REQ-012: Integrate Poker Hand dataset from UCI."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["poker"]

        # Act: Fetch Poker Hand dataset from UCI
        dataset = create_dataset(config_path)

        # Assert: Poker Hand dataset characteristics
        assert dataset.name == "test_poker"
        assert dataset.source_type == "real_world"
        assert dataset.dataset_metadata["uci_id"] == 158, "Correct UCI repository ID"
        assert dataset.X.shape[1] == 10, "Poker dataset has 10 features (5 cards × 2 attributes)"
        assert dataset.dataset_metadata["n_classes"] == 10, "10 poker hand classes"

    def test_should_handle_large_dataset_streaming_when_memory_limited(self, uci_repository_service, sample_toml_configs):
        """Test REQ-012: Handle large datasets through streaming interface."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["large_dataset"]

        # Act: Create large dataset with streaming
        dataset = create_dataset(config_path, streaming=True)

        # Assert: Streaming interface available
        assert hasattr(dataset, "stream"), "Dataset provides streaming interface"
        assert hasattr(dataset.stream, "__iter__"), "Stream is iterable"
        assert hasattr(dataset.stream, "batch_size"), "Stream has configurable batch size"

        # Test streaming functionality
        first_batch = next(iter(dataset.stream))
        assert "X" in first_batch, "Batch contains features"
        assert "y" in first_batch, "Batch contains targets"
        assert len(first_batch["X"]) <= dataset.stream.batch_size, "Batch size respected"


class TestUCIRepositoryIntegration:
    """Test UCI ML Repository integration functionality (REQ-013)."""

    def test_should_validate_uci_dataset_availability_when_fetching_dataset(self, uci_repository_service):
        """Test REQ-013: Validate dataset availability from UCI ML Repository."""
        from drift_datasets.data_sources import UCIRepository

        # Act: Check dataset availability
        uci_repo = UCIRepository()
        is_available = uci_repo.is_dataset_available(321)  # Electricity dataset
        dataset_info = uci_repo.get_dataset_info(321)

        # Assert: Dataset availability validation
        assert is_available == True, "Electricity dataset should be available"
        assert dataset_info["id"] == 321, "Correct dataset ID"
        assert dataset_info["name"] == "ElectricityLoadDiagrams20112014", "Correct dataset name"
        assert "download_url" in dataset_info, "Download URL provided"

    def test_should_handle_uci_repository_errors_when_dataset_unavailable(self, uci_repository_service):
        """Test REQ-013: Handle UCI repository errors gracefully."""
        from drift_datasets.data_sources import UCIRepository

        uci_repo = UCIRepository()

        # Test nonexistent dataset
        is_available = uci_repo.is_dataset_available(99999)
        assert is_available == False, "Nonexistent dataset should be unavailable"

        with pytest.raises(ValueError, match="Dataset with ID 99999 not found"):
            uci_repo.get_dataset_info(99999)

    def test_should_cache_uci_datasets_when_repeatedly_accessed(self, uci_repository_service, tmp_path):
        """Test REQ-013: Cache UCI datasets for efficient repeated access."""
        from drift_datasets.data_sources import UCIRepository

        # Arrange: Configure cache directory
        cache_dir = tmp_path / "uci_cache"
        uci_repo = UCIRepository(cache_dir=str(cache_dir))

        # Act: Fetch same dataset twice
        data1 = uci_repo.fetch_dataset(321)
        data2 = uci_repo.fetch_dataset(321)  # Should use cache

        # Assert: Caching behavior
        assert cache_dir.exists(), "Cache directory created"
        cache_files = list(cache_dir.glob("*.pkl"))
        assert len(cache_files) >= 1, "Cache file created"

        # Verify cached data is identical
        assert data1.shape == data2.shape, "Cached data identical"
        assert (data1 == data2).all().all(), "Cached data values identical"

    def test_should_validate_uci_data_integrity_when_downloading(self, uci_repository_service):
        """Test REQ-013: Validate data integrity during UCI dataset downloads."""
        from drift_datasets.data_sources import UCIRepository

        # Act: Fetch dataset with integrity validation
        uci_repo = UCIRepository()
        data, integrity_report = uci_repo.fetch_dataset_with_validation(321)

        # Assert: Data integrity validated
        assert integrity_report["checksum_valid"] == True, "Data checksum validated"
        assert integrity_report["format_valid"] == True, "Data format validated"
        assert integrity_report["completeness_valid"] == True, "Data completeness validated"

        # Test data structure
        assert data is not None, "Data successfully downloaded"
        assert len(data) > 0, "Dataset contains data"


class TestRealWorldDatasetPreprocessing:
    """Test real-world dataset preprocessing functionality (REQ-014)."""

    def test_should_handle_missing_values_when_preprocessing_uci_dataset(self, uci_repository_service, sample_toml_configs):
        """Test REQ-014: Handle missing values in real-world datasets appropriately."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci_with_missing"]

        # Act: Create dataset with missing value handling
        dataset = create_dataset(config_path)

        # Assert: Missing values handled
        assert not dataset.X.isnull().any().any(), "No missing values in final dataset"

        # Check preprocessing metadata
        preprocessing_info = dataset.dataset_metadata["preprocessing"]
        assert "missing_value_strategy" in preprocessing_info, "Missing value strategy recorded"
        assert preprocessing_info["missing_value_strategy"] == "impute_median", "Correct strategy applied"
        assert "missing_value_counts" in preprocessing_info, "Original missing counts recorded"

    def test_should_normalize_features_when_requested_in_config(self, uci_repository_service, sample_toml_configs):
        """Test REQ-014: Apply feature normalization as specified in configuration."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci_normalized"]

        # Act: Create dataset with normalization
        dataset = create_dataset(config_path)

        # Assert: Features normalized
        for column in dataset.X.columns:
            if dataset.X[column].dtype in ["float64", "int64"]:
                column_mean = dataset.X[column].mean()
                column_std = dataset.X[column].std()
                assert abs(column_mean) < 0.1, f"Column {column} should be centered near zero"
                assert abs(column_std - 1.0) < 0.1, f"Column {column} should have unit variance"

        # Check normalization metadata
        preprocessing_info = dataset.dataset_metadata["preprocessing"]
        assert "normalization_method" in preprocessing_info, "Normalization method recorded"
        assert preprocessing_info["normalization_method"] == "standardize", "Correct normalization applied"

    def test_should_encode_categorical_features_when_present_in_dataset(self, uci_repository_service, sample_toml_configs):
        """Test REQ-014: Encode categorical features appropriately."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci_categorical"]

        # Act: Create dataset with categorical encoding
        dataset = create_dataset(config_path)

        # Assert: Categorical features encoded
        categorical_features = [
            f for f in dataset.dataset_metadata["features"] if f["role"] == "feature" and f["original_type"] == "categorical"
        ]

        for cat_feature in categorical_features:
            feature_name = cat_feature["name"]
            if feature_name in dataset.X.columns:
                # Check if feature is now numeric
                assert dataset.X[feature_name].dtype in ["int64", "float64"], f"Categorical feature {feature_name} encoded"

        # Check encoding metadata
        preprocessing_info = dataset.dataset_metadata["preprocessing"]
        assert "categorical_encoding" in preprocessing_info, "Categorical encoding recorded"
        assert "label_encoders" in preprocessing_info, "Label encoder mappings stored"

    def test_should_detect_and_handle_outliers_when_configured(self, uci_repository_service, sample_toml_configs):
        """Test REQ-014: Detect and handle outliers in real-world datasets."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci_outlier_detection"]

        # Act: Create dataset with outlier handling
        dataset = create_dataset(config_path)

        # Assert: Outliers detected and handled
        preprocessing_info = dataset.dataset_metadata["preprocessing"]
        assert "outlier_detection" in preprocessing_info, "Outlier detection performed"
        assert "outliers_detected" in preprocessing_info["outlier_detection"], "Outlier count recorded"
        assert "outlier_handling_method" in preprocessing_info["outlier_detection"], "Handling method recorded"

        # Check dataset consistency after outlier handling
        outlier_handling = preprocessing_info["outlier_detection"]["outlier_handling_method"]
        if outlier_handling == "remove":
            original_size = preprocessing_info["original_size"]
            assert len(dataset.X) < original_size, "Outliers removed from dataset"
        elif outlier_handling == "cap":
            # Verify capping was applied (no extreme values)
            for column in dataset.X.select_dtypes(include=["float64", "int64"]).columns:
                q99 = dataset.X[column].quantile(0.99)
                q01 = dataset.X[column].quantile(0.01)
                assert dataset.X[column].max() <= q99 * 1.1, f"Column {column} values capped at high end"
                assert dataset.X[column].min() >= q01 * 0.9, f"Column {column} values capped at low end"

    def test_should_preserve_temporal_order_when_processing_temporal_datasets(self, uci_repository_service, sample_toml_configs):
        """Test REQ-014: Preserve temporal order in time-series datasets."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci_temporal"]

        # Act: Create temporal dataset
        dataset = create_dataset(config_path)

        # Assert: Temporal order preserved
        temporal_info = dataset.dataset_metadata["temporal"]
        assert "temporal_order_preserved" in temporal_info, "Temporal order tracking"
        assert temporal_info["temporal_order_preserved"] == True, "Order preserved during processing"
        assert "timestamp_column" in temporal_info, "Timestamp column identified"

        # Check temporal consistency
        if "timestamp" in dataset.X.columns:
            timestamps = dataset.X["timestamp"]
            assert timestamps.is_monotonic_increasing, "Timestamps in ascending order"

    def test_should_validate_preprocessing_steps_when_multiple_operations_applied(self, uci_repository_service, sample_toml_configs):
        """Test REQ-014: Validate preprocessing pipeline with multiple operations."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci_full_preprocessing"]

        # Act: Create dataset with full preprocessing pipeline
        dataset = create_dataset(config_path)

        # Assert: All preprocessing steps recorded and validated
        preprocessing_info = dataset.dataset_metadata["preprocessing"]
        assert "pipeline_steps" in preprocessing_info, "Pipeline steps recorded"

        pipeline_steps = preprocessing_info["pipeline_steps"]
        expected_steps = ["missing_value_imputation", "outlier_detection", "normalization", "categorical_encoding"]
        for expected_step in expected_steps:
            assert expected_step in [step["name"] for step in pipeline_steps], f"Step {expected_step} in pipeline"

        # Verify pipeline order
        step_names = [step["name"] for step in pipeline_steps]
        assert step_names.index("missing_value_imputation") < step_names.index("normalization"), "Correct pipeline order"
        assert step_names.index("categorical_encoding") < step_names.index("normalization"), "Categorical encoding before normalization"

        # Test preprocessing reversibility information
        assert "reversible_operations" in preprocessing_info, "Reversibility information provided"
        reversible_ops = preprocessing_info["reversible_operations"]
        assert "normalization" in reversible_ops, "Normalization is reversible"
        assert reversible_ops["normalization"]["parameters"] is not None, "Normalization parameters stored for reversal"
