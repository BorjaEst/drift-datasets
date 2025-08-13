"""
Functional tests for real-world dataset integration functionality.

These tests validate real-world dataset integration from REQ-002 and REQ-013,
including UCI ML Repository integration and basic dataset loading.
Aligned with current implementation capabilities and REQUIREMENTS.md.
"""

import pytest


class TestRealWorldDatasetIntegration:
    """Test real-world dataset integration functionality (REQ-002)."""

    def test_should_fetch_electricity_dataset_when_uci_config_provided(self, uci_repository_service, sample_toml_configs):
        """Test REQ-002: Integrate real-world datasets from UCI ML Repository - Electricity dataset."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["uci"]

        # Act: Fetch Electricity dataset from UCI
        dataset = create_dataset(config_path)

        # Assert: Electricity dataset characteristics
        assert dataset.name == "test_electricity"
        assert dataset.source_type == "real_world"
        assert hasattr(dataset, "X"), "Dataset has feature matrix"
        assert hasattr(dataset, "y"), "Dataset has target vector"

        # Verify basic structure
        assert len(dataset.X) > 0, "Dataset contains data"
        assert len(dataset.y) > 0, "Dataset contains targets"
        assert len(dataset.X) == len(dataset.y), "Features and targets have same length"


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

    def test_should_handle_uci_repository_errors_when_dataset_unavailable(self, uci_repository_service):
        """Test REQ-013: Handle UCI repository errors gracefully."""
        from drift_datasets.data_sources import UCIRepository

        uci_repo = UCIRepository()

        # Test nonexistent dataset
        is_available = uci_repo.is_dataset_available(99999)
        assert is_available == False, "Nonexistent dataset should be unavailable"

        with pytest.raises(ValueError, match="Dataset with ID 99999 not found"):
            uci_repo.get_dataset_info(99999)

    def test_should_validate_uci_data_integrity_when_downloading(self, uci_repository_service):
        """Test REQ-013: Basic data integrity validation."""
        from drift_datasets.data_sources import UCIRepository

        # Act: Fetch dataset with basic validation
        uci_repo = UCIRepository()
        data, integrity_report = uci_repo.fetch_dataset_with_validation(321)

        # Assert: Basic data integrity
        assert integrity_report["format_valid"] == True, "Data format validated"
        assert data is not None, "Data successfully retrieved"
        assert len(data) > 0, "Dataset contains data"
