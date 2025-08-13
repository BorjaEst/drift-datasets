"""
River integration adapter.

This module provides functionality to convert DriftDataset objects to
formats compatible with River streaming workflows.
"""

from typing import Any, Dict, Iterator

import numpy as np


class RiverStream:
    """River-compatible streaming dataset wrapper."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self.X)):
            # Get row data properly from DataFrame/Series
            if hasattr(self.X, "iloc"):
                x_sample = self.X.iloc[i]
            else:
                x_sample = self.X[i]

            if hasattr(self.y, "iloc"):
                y_sample = self.y.iloc[i]
            else:
                y_sample = self.y[i]

            # Convert to dict format expected by River
            if hasattr(x_sample, "to_dict"):
                x_dict = x_sample.to_dict()
            elif hasattr(x_sample, "__iter__") and not isinstance(x_sample, (str, bytes)):
                # Convert array-like to dict with numeric keys
                x_dict = {j: float(val) for j, val in enumerate(x_sample)}
            else:
                x_dict = {"feature_0": float(x_sample)}

            yield {"x": x_dict, "y": int(y_sample)}


class RiverAdapter:
    """Adapter for River integration."""

    def to_river_stream(self, dataset) -> RiverStream:
        """
        Convert DriftDataset to River-compatible streaming format.

        Args:
            dataset: DriftDataset object

        Returns:
            RiverStream object
        """
        return RiverStream(dataset.X, dataset.y)

    def to_river(self, dataset):
        """Convert DriftDataset to River-compatible format."""
        raise NotImplementedError("River integration not yet implemented")
