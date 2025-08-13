"""
Scikit-learn integration adapter.

This module provides functionality to convert DriftDataset objects to
formats compatible with scikit-learn workflows.
"""

from typing import Tuple

import numpy as np
import pandas as pd


class SKLearnAdapter:
    """Adapter for scikit-learn integration."""

    def to_sklearn(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DriftDataset to sklearn-compatible format.

        Args:
            dataset: DriftDataset object

        Returns:
            Tuple of (X, y) as numpy arrays
        """
        X = dataset.X.values if hasattr(dataset.X, "values") else np.array(dataset.X)
        y = dataset.y.values if hasattr(dataset.y, "values") else np.array(dataset.y)

        return X, y
