"""
PyTorch integration adapter.

This module provides functionality to convert DriftDataset objects to
formats compatible with PyTorch workflows.
"""

from typing import Any, Tuple

import numpy as np


class PyTorchDataset:
    """PyTorch-compatible dataset wrapper."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PyTorchAdapter:
    """Adapter for PyTorch integration."""

    def to_torch_dataset(self, dataset) -> PyTorchDataset:
        """
        Convert DriftDataset to PyTorch-compatible dataset.

        Args:
            dataset: DriftDataset object

        Returns:
            PyTorchDataset object
        """
        X = dataset.X.values if hasattr(dataset.X, "values") else np.array(dataset.X)
        y = dataset.y.values if hasattr(dataset.y, "values") else np.array(dataset.y)

        return PyTorchDataset(X, y)

    def to_torch(self, dataset):
        """Convert DriftDataset to PyTorch-compatible format."""
        raise NotImplementedError("PyTorch integration not yet implemented")
