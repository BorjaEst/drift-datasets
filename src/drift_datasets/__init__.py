"""
Drift Datasets Library

A comprehensive library for generating and managing datasets with concept drift
for research and benchmarking in machine learning.
"""

from .config import ConfigurationManager, ParameterValidator
from .data_sources import UCIRepository, UCIService
from .factory import create_dataset
from .models import DriftDataset

__version__ = "0.1.0"

__all__ = ["create_dataset", "DriftDataset", "ConfigurationManager", "ParameterValidator", "UCIRepository", "UCIService"]
