"""
Drift Datasets Literals Module

This module provides type literal definitions for the Drift Datasets package.
It includes type definitions for logging levels, drift simulation types, feature characteristics,
dataset classifications, synthetic data generators, preprocessing options, and methods for simulating drift.
The literals ensure consistent usage of string values throughout the application.
"""

from typing import Literal

# Log levels
LogLevel = Literal["debug", "info", "warning", "error", "critical"]

# Drift classification types
DriftType = Literal["covariate", "concept", "prior", "none"]
DriftPattern = Literal["abrupt", "gradual", "recurring", "incremental"]

# Feature types
FeatureType = Literal["continuous", "categorical", "mixed"]
FeatureRole = Literal["feature", "target", "timestamp", "identifier", "metadata", "exclude"]

# Dataset types
DatasetType = Literal["synthetic", "real_world", "mixed"]
DatasetDimension = Literal["univariate", "multivariate"]
DatasetLabeling = Literal["supervised", "unsupervised", "semi-supervised"]
DatasetSource = Literal["capymoa", "ucimlrepo"]

# Synthetic data generators (CapyMOA)
SyntheticGenerator = Literal[
    "SineGenerator",
    "HyperplaneGenerator",
    "STAGGERGenerator",
    "RandomTreeGenerator",
    "SEAGenerator",
    "AgrawalGenerator",
    "LEDGenerator",
]

# Real-world dataset preprocessing options
PreprocessingOption = Literal[
    "normalize",
    "standardize",
    "temporal_order",
    "remove_missing",
    "impute_missing",
]

# Drift simulation methods for real-world data
DriftSimulationMethod = Literal[
    "feature_rotation",
    "label_noise",
    "feature_scaling",
    "concept_shift",
    "sampling_bias",
]

# Mixed dataset combination types
MixedCombinationType = Literal[
    "sequential",
    "interleaved",
    "hierarchical",
]
