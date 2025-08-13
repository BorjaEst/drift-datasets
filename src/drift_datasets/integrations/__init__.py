"""
Integration adapters for external ML libraries.

This module provides adapters to convert DriftDataset objects to formats
compatible with popular machine learning libraries like scikit-learn, PyTorch, etc.
"""

from .drift_detection import DriftDetectionAdapter
from .pytorch import PyTorchAdapter
from .river import RiverAdapter
from .sklearn import SKLearnAdapter

__all__ = [
    "SKLearnAdapter",
    "PyTorchAdapter",
    "RiverAdapter",
    "DriftDetectionAdapter",
]
