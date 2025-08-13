"""
Unit tests for DriftMetadata research parameter support.

Tests REQ-011A: Pydantic v2 Model Validation with research parameters.
"""

import pytest
from pydantic import ValidationError

from drift_datasets.models.dataset import DriftMetadata


class TestDriftMetadataResearchParameters:
    """Test DriftMetadata model with research parameters."""

    def test_drift_metadata_accepts_transition_durations(self):
        """Test that DriftMetadata accepts transition_durations parameter."""
        # RED: This should pass once we implement the feature
        metadata = DriftMetadata(drift_points=[1000, 2000], drift_types=["concept", "concept"], transition_durations=[500, 750])

        assert metadata.transition_durations == [500, 750]
        assert len(metadata.transition_durations) == len(metadata.drift_points)

    def test_drift_metadata_accepts_drift_intensities(self):
        """Test that DriftMetadata accepts drift_intensities parameter."""
        metadata = DriftMetadata(drift_points=[1000, 2000], drift_types=["concept", "concept"], drift_intensities=[0.5, 0.8])

        assert metadata.drift_intensities == [0.5, 0.8]
        assert len(metadata.drift_intensities) == len(metadata.drift_points)

    def test_drift_metadata_validates_transition_durations_length(self):
        """Test that transition_durations length must match drift_points length."""
        with pytest.raises(ValidationError) as exc_info:
            DriftMetadata(drift_points=[1000, 2000], drift_types=["concept", "concept"], transition_durations=[500])  # Length mismatch

        assert "transition_durations length" in str(exc_info.value)

    def test_drift_metadata_validates_drift_intensities_range(self):
        """Test that drift_intensities values must be in [0.0, 1.0] range."""
        with pytest.raises(ValidationError) as exc_info:
            DriftMetadata(drift_points=[1000], drift_types=["concept"], drift_intensities=[1.5])  # Out of range

        assert "drift_intensities" in str(exc_info.value)

    def test_drift_metadata_validates_drift_intensities_length(self):
        """Test that drift_intensities length must match drift_points length."""
        with pytest.raises(ValidationError) as exc_info:
            DriftMetadata(drift_points=[1000, 2000], drift_types=["concept", "concept"], drift_intensities=[0.5])  # Length mismatch

        assert "drift_intensities length" in str(exc_info.value)

    def test_drift_metadata_optional_research_parameters(self):
        """Test that research parameters are optional."""
        metadata = DriftMetadata(drift_points=[1000, 2000], drift_types=["concept", "concept"])

        assert metadata.transition_durations == []
        assert metadata.drift_intensities == []

    def test_drift_metadata_with_all_research_parameters(self):
        """Test DriftMetadata with all research parameters."""
        metadata = DriftMetadata(
            drift_points=[1000, 2000, 3000],
            drift_types=["concept", "concept", "covariate"],
            drift_patterns=["gradual", "abrupt", "gradual"],
            transition_durations=[500, 0, 750],
            drift_intensities=[0.3, 1.0, 0.6],
            affected_features=[[0, 1], [2], [0, 2]],
        )

        assert len(metadata.drift_points) == 3
        assert len(metadata.transition_durations) == 3
        assert len(metadata.drift_intensities) == 3
        assert len(metadata.affected_features) == 3

        assert metadata.transition_durations == [500, 0, 750]
        assert metadata.drift_intensities == [0.3, 1.0, 0.6]

    def test_drift_metadata_backward_compatibility(self):
        """Test that existing CapyMOA parameters still work."""
        metadata = DriftMetadata(drift_points=[1000, 2000], drift_types=["concept", "concept"], drift_patterns=["gradual", "abrupt"])

        assert metadata.drift_points == [1000, 2000]
        assert metadata.drift_types == ["concept", "concept"]
        assert metadata.drift_patterns == ["gradual", "abrupt"]
