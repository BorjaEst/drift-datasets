"""
Tests for plotting functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_datasets.models import DriftDataset
from drift_datasets.plotting import plot_dataset


class TestPlotDataset:
    """Test the plot_dataset function."""

    def test_plot_basic_dataset(self, sample_drift_dataset):
        """Test basic plotting of a drift dataset."""
        # Should return a matplotlib figure
        fig = plot_dataset(sample_drift_dataset)

        assert fig is not None
        # Check it's a matplotlib figure
        assert hasattr(fig, "get_axes")
        assert len(fig.get_axes()) > 0

    def test_plot_dataset_with_drift_points(self, sample_drift_dataset):
        """Test plotting with drift points visualization."""
        fig = plot_dataset(sample_drift_dataset, show_drift_points=True)

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) > 0

    def test_plot_dataset_feature_selection(self, sample_drift_dataset):
        """Test plotting with specific feature selection."""
        feature_names = list(sample_drift_dataset.X.columns)

        # Test with first two features
        fig = plot_dataset(sample_drift_dataset, features=feature_names[:2])

        assert fig is not None

    def test_plot_dataset_color_by_target(self, sample_drift_dataset):
        """Test plotting colored by target variable."""
        fig = plot_dataset(sample_drift_dataset, color_by_target=True)

        assert fig is not None

    def test_plot_dataset_save_figure(self, sample_drift_dataset):
        """Test saving plot to file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_plot.png"

            fig = plot_dataset(sample_drift_dataset, save_path=str(save_path))

            assert fig is not None
            assert save_path.exists()

    def test_plot_dataset_2d_scatter(self, sample_drift_dataset):
        """Test 2D scatter plot for datasets with 2 features."""
        # Ensure we have exactly 2 features for 2D plotting
        X_2d = sample_drift_dataset.X.iloc[:, :2]
        dataset_2d = DriftDataset(
            X=X_2d,
            y=sample_drift_dataset.y,
            name="test_2d",
            source_type="synthetic",
            drift_metadata=sample_drift_dataset.drift_metadata,
            dataset_metadata=sample_drift_dataset.dataset_metadata,
        )

        fig = plot_dataset(dataset_2d, plot_type="scatter")

        assert fig is not None

    def test_plot_dataset_time_series(self, sample_drift_dataset):
        """Test time series plot showing features over time."""
        fig = plot_dataset(sample_drift_dataset, plot_type="time_series")

        assert fig is not None

    def test_plot_dataset_with_concept_segments(self, sample_drift_dataset):
        """Test plotting with concept segments highlighted."""
        fig = plot_dataset(sample_drift_dataset, show_concept_segments=True)

        assert fig is not None

    def test_plot_dataset_invalid_features(self, sample_drift_dataset):
        """Test error handling for invalid feature names."""
        with pytest.raises(ValueError, match="Feature 'invalid_feature' not found"):
            plot_dataset(sample_drift_dataset, features=["invalid_feature"])

    def test_plot_dataset_empty_dataset(self):
        """Test error handling for empty dataset."""
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=int)

        dataset = DriftDataset(X=empty_X, y=empty_y, name="empty", source_type="synthetic")

        with pytest.raises(ValueError, match="Dataset is empty"):
            plot_dataset(dataset)
