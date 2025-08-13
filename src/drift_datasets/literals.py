from typing import Literal

# Drift classification types
DriftType = Literal["covariate", "concept", "prior", "none"]
DriftPattern = Literal["abrupt", "gradual", "recurring", "incremental"]

# Data types
DataType = Literal["continuous", "categorical", "mixed"]
DataDimension = Literal["univariate", "multivariate"]
DataLabeling = Literal["supervised", "unsupervised", "semi-supervised"]

# Dataset types
DatasetType = Literal["synthetic", "real_world", "mixed"]
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

# Log levels
LogLevel = Literal["debug", "info", "warning", "error", "critical"]
