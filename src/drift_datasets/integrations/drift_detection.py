"""
Drift detection integration adapter.

This module provides functionality to integrate DriftDataset objects
with drift detection libraries.
"""


class DriftDetectionStream:
    """Streaming interface for drift detection algorithms."""

    def __init__(self, X, y, true_drift_points):
        self.X = X
        self.y = y
        self.true_drift_points = true_drift_points

    def __iter__(self):
        """Iterate over dataset instances."""
        for i in range(len(self.X)):
            if hasattr(self.X, "iloc"):
                features = self.X.iloc[i]
            else:
                features = self.X[i]

            if hasattr(self.y, "iloc"):
                target = self.y.iloc[i]
            else:
                target = self.y[i]

            yield features, target


class DriftDetectionAdapter:
    """Adapter for drift detection integration."""

    def prepare_for_drift_detection(self, dataset):
        """Prepare dataset for drift detection algorithms."""
        # Extract drift points from dataset metadata
        drift_points = []
        if hasattr(dataset, "drift_metadata") and dataset.drift_metadata:
            drift_points = dataset.drift_metadata.drift_points

        return DriftDetectionStream(dataset.X, dataset.y, drift_points)

    def to_drift_detector(self, dataset):
        """Convert DriftDataset to drift detector-compatible format."""
        raise NotImplementedError("Drift detection integration not yet implemented")
