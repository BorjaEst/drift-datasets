# API Reference

Complete reference for all public classes, methods, and functions in drift-datasets.

## Overview

The drift-datasets library provides a simple, high-level API centered around the `create_dataset()` function and the `DriftDataset` class. All functionality is accessible through these main entry points.

## Main Entry Point

### create_dataset()

```python
def create_dataset(
    config: Union[str, Dict, Path],
    validate: bool = True,
    cache_dir: Optional[str] = None
) -> DriftDataset
```

Creates a drift dataset from configuration file or dictionary.

**Parameters**:

- **config** (str | dict | Path): Configuration file path or dictionary
- **validate** (bool, optional): Whether to validate configuration. Default: True
- **cache_dir** (str, optional): Directory for caching external data. Default: None

**Returns**:

- **DriftDataset**: Complete dataset object with X, y, and metadata

**Raises**:

- **FileNotFoundError**: Configuration file not found
- **ValueError**: Invalid configuration parameters
- **ConnectionError**: External API failures (UCI, CapyMOA)
- **RuntimeError**: Dataset generation failures

**Examples**:

```python
import drift_datasets as dd

# From TOML configuration file
dataset = dd.create_dataset("config.toml")

# From Python dictionary
config = {
    "dataset": {"name": "test", "type": "synthetic"},
    "generator_config": {"n_instances": 1000}
}
dataset = dd.create_dataset(config)

# With validation disabled (faster, but risky)
dataset = dd.create_dataset(config, validate=False)

# With custom cache directory
dataset = dd.create_dataset(config, cache_dir="./my_cache")
```

## Core Data Classes

### DriftDataset

Main dataset class containing features, targets, and comprehensive metadata.

```python
class DriftDataset:
    """
    Unified dataset object for concept drift research.
    
    Attributes:
        X (pd.DataFrame): Feature matrix (n_samples, n_features)
        y (pd.Series): Target vector (n_samples,)
        dataset_metadata (DatasetMetadata): Feature and dataset information
        drift_metadata (DriftMetadata): Ground truth drift information
        name (str): Dataset identifier
        source_type (str): Generation method ("synthetic", "real_world", "mixed")
        generator_info (dict): Generator-specific information
        config (dict): Original configuration
    """
```

#### Properties

**Core Data Access**:

```python
@property
def X(self) -> pd.DataFrame:
    """Feature matrix with shape (n_samples, n_features)"""

@property  
def y(self) -> pd.Series:
    """Target vector with shape (n_samples,)"""

@property
def dataset_metadata(self) -> DatasetMetadata:
    """Dataset and feature metadata"""

@property
def drift_metadata(self) -> DriftMetadata:
    """Ground truth drift information"""
```

**Dataset Information**:

```python
@property
def name(self) -> str:
    """Dataset identifier from configuration"""

@property
def source_type(self) -> str:
    """Generation method: 'synthetic', 'real_world', or 'mixed'"""

@property
def generator_info(self) -> Dict:
    """Generator-specific metadata and parameters"""

@property
def config(self) -> Dict:
    """Original configuration dictionary"""
```

#### Feature Access Methods

**Role-Based Filtering**:

```python
def get_drift_detection_features(self) -> pd.DataFrame:
    """
    Get features appropriate for drift detection modeling.
    
    Excludes features with roles: target, timestamp, identifier, metadata, exclude
    
    Returns:
        pd.DataFrame: Features suitable for drift detection
    
    Example:
        >>> dataset = create_dataset(config)
        >>> X_modeling = dataset.get_drift_detection_features()
        >>> print(f"Modeling features: {list(X_modeling.columns)}")
    """

def get_features_by_role(self, role: str) -> pd.DataFrame:
    """
    Get features matching specific role.
    
    Args:
        role: Feature role ("feature", "target", "timestamp", etc.)
    
    Returns:
        pd.DataFrame: Features with matching role
    
    Example:
        >>> timestamps = dataset.get_features_by_role("timestamp")
        >>> targets = dataset.get_features_by_role("target")
    """
```

**Type-Based Filtering**:

```python
def get_continuous_features(self) -> pd.DataFrame:
    """
    Get all continuous (numerical) features.
    
    Returns:
        pd.DataFrame: Features with type="continuous"
    
    Example:
        >>> continuous = dataset.get_continuous_features()
        >>> print(f"Continuous features: {list(continuous.columns)}")
    """

def get_categorical_features(self) -> pd.DataFrame:
    """
    Get all categorical features.
    
    Returns:
        pd.DataFrame: Features with type="categorical"
    
    Example:
        >>> categorical = dataset.get_categorical_features()
        >>> print(f"Categorical features: {list(categorical.columns)}")
    """
```

#### Drift Analysis Methods

**Drift Point Analysis**:

```python
def is_drift_point(self, idx: int, tolerance: int = 0) -> bool:
    """
    Check if sample index is at or near a drift point.
    
    Args:
        idx: Sample index to check
        tolerance: Distance tolerance for "near" drift points
    
    Returns:
        bool: True if sample is within tolerance of a drift point
    
    Example:
        >>> dataset.is_drift_point(500)  # Exact drift point
        True
        >>> dataset.is_drift_point(495, tolerance=10)  # Near drift point
        True
        >>> dataset.is_drift_point(100)  # Not near drift
        False
    """

def get_concept_segments(self) -> List[Tuple[int, int]]:
    """
    Get stable concept segments between drift points.
    
    Returns:
        List[Tuple[int, int]]: List of (start, end) indices for each concept
    
    Example:
        >>> segments = dataset.get_concept_segments()
        >>> print(f"Concept segments: {segments}")
        [(0, 500), (500, 1000)]  # Drift at sample 500
    """

def get_drift_type_at(self, idx: int) -> str:
    """
    Get drift type at specific sample index.
    
    Args:
        idx: Sample index
    
    Returns:
        str: Drift type ("concept", "covariate", "prior", "none")
    
    Example:
        >>> drift_type = dataset.get_drift_type_at(500)
        >>> print(f"Drift type at 500: {drift_type}")
    """

def get_affected_features_at(self, idx: int) -> List[int]:
    """
    Get feature indices affected by drift at sample index.
    
    Args:
        idx: Sample index
    
    Returns:
        List[int]: Feature indices affected by drift
    
    Example:
        >>> affected = dataset.get_affected_features_at(500)
        >>> print(f"Affected features: {affected}")
    """
```

**Validation Methods**:

```python
def validate_drift_metadata(self) -> Dict[str, Any]:
    """
    Validate consistency of drift metadata.
    
    Returns:
        Dict with keys:
        - 'valid' (bool): Whether metadata is consistent
        - 'issues' (List[str]): List of validation issues found
    
    Example:
        >>> validation = dataset.validate_drift_metadata()
        >>> if not validation['valid']:
        ...     for issue in validation['issues']:
        ...         print(f"Issue: {issue}")
    """
```

#### Data Manipulation Methods

**Temporal Splitting**:

```python
def split_temporal(
    self, 
    ratio: float, 
    preserve_drift: bool = True
) -> Tuple['DriftDataset', 'DriftDataset']:
    """
    Split dataset temporally while preserving drift structure.
    
    Args:
        ratio: Fraction for training set (0.0-1.0)
        preserve_drift: Whether to preserve drift metadata in splits
    
    Returns:
        Tuple[DriftDataset, DriftDataset]: (train_dataset, test_dataset)
    
    Example:
        >>> train, test = dataset.split_temporal(ratio=0.7)
        >>> print(f"Train: {len(train.X)}, Test: {len(test.X)}")
        >>> print(f"Train drifts: {train.drift_metadata.drift_points}")
    """
```

#### Feature Information Methods

**Feature Metadata Access**:

```python
def get_feature_info(self, feature_name: str) -> FeatureMetadata:
    """
    Get detailed metadata for specific feature.
    
    Args:
        feature_name: Name of the feature
    
    Returns:
        FeatureMetadata: Complete feature metadata object
    
    Example:
        >>> info = dataset.get_feature_info("temperature")
        >>> print(f"Type: {info.type}, Role: {info.role}")
        >>> print(f"Description: {info.description}")
    """

def summarize_features(self) -> Dict[str, Any]:
    """
    Get comprehensive feature summary for dataset exploration.
    
    Returns:
        Dict with keys:
        - 'total_features' (int): Total number of features
        - 'feature_types' (Dict[str, int]): Count by type
        - 'feature_roles' (Dict[str, int]): Count by role
        - 'modeling_features' (List[str]): Features for modeling
        - 'excluded_features' (List[str]): Features to exclude
    
    Example:
        >>> summary = dataset.summarize_features()
        >>> print(f"Modeling features: {summary['modeling_features']}")
        >>> print(f"Feature types: {summary['feature_types']}")
    """
```

## Metadata Classes

### DatasetMetadata

Container for dataset-level metadata and feature information.

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class DatasetMetadata(BaseModel):
    """Dataset-level metadata and feature specifications."""
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    dimension: str = Field(..., pattern="^(univariate|multivariate)$")  # "univariate" or "multivariate"
    labeling: str = Field(..., pattern="^(supervised|unsupervised|semi-supervised)$")   # Learning paradigm
    n_classes: Optional[int] = Field(None, ge=2)  # Number of classes for classification
    task_type: Optional[str] = Field(None, pattern="^(classification|regression)$")  # Task type
    temporal: bool = False  # Whether dataset has temporal ordering
    features: List[FeatureMetadata]  # Per-feature metadata
```

### FeatureMetadata

Metadata for individual features.

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class FeatureMetadata(BaseModel):
    """Metadata for individual dataset features."""
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    name: str = Field(..., min_length=1, max_length=100)  # Feature name
    type: str = Field(..., pattern="^(continuous|categorical|mixed)$")  # Feature type
    role: str = Field(..., pattern="^(feature|target|timestamp|identifier|metadata|exclude)$")  # Feature role
    description: Optional[str] = Field(None, max_length=500)  # Human-readable description
    missing_values: bool = False  # Whether feature can have missing values
    unique_values: Optional[int] = Field(None, ge=1)  # Approximate number of unique values
```

### DriftMetadata

Ground truth information about concept drift.

```python
from pydantic import BaseModel, Field, ConfigDict, validator
from typing import List, Optional

class DriftMetadata(BaseModel):
    """Ground truth drift metadata."""
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    drift_points: List[int] = Field(..., min_items=0)  # Sample indices where drift occurs
    drift_types: List[str]   # Type of each drift
    drift_patterns: List[str]  # Pattern of each drift
    drift_intensities: Optional[List[str]] = None  # Intensity of each drift
    transition_durations: Optional[List[int]] = None  # Duration of each transition
    affected_features: Optional[List[List[int]]] = None  # Features affected by each drift
    drift_widths: Optional[List[int]] = None  # CapyMOA width parameters
    drift_alphas: Optional[List[float]] = None  # CapyMOA alpha parameters
    concepts: Optional[List[str]] = None  # Human-readable concept labels
    
    @validator('drift_types')
    def validate_drift_types(cls, v, values):
        """Validate drift_types length matches drift_points."""
        drift_points = values.get('drift_points', [])
        if len(v) != len(drift_points):
            raise ValueError(f"drift_types length ({len(v)}) must match drift_points length ({len(drift_points)})")
        return v
    
    @validator('drift_patterns')
    def validate_drift_patterns(cls, v, values):
        """Validate drift_patterns length matches drift_points."""
        drift_points = values.get('drift_points', [])
        if len(v) != len(drift_points):
            raise ValueError(f"drift_patterns length ({len(v)}) must match drift_points length ({len(drift_points)})")
        return v
```

## Constants and Enums

### Supported Values

```python
# Dataset types
DATASET_TYPES = ["synthetic", "real_world", "mixed"]

# Feature types
FEATURE_TYPES = ["continuous", "categorical", "mixed"]

# Feature roles
FEATURE_ROLES = ["feature", "target", "timestamp", "identifier", "metadata", "exclude"]

# Drift types
DRIFT_TYPES = ["covariate", "concept", "prior", "none"]

# Drift patterns
DRIFT_PATTERNS = ["abrupt", "gradual", "continuous_gradual", "intermittent_gradual", "recurring", "incremental"]

# Dimension types
DIMENSION_TYPES = ["univariate", "multivariate"]

# Labeling types
LABELING_TYPES = ["supervised", "unsupervised", "semi-supervised"]

# CapyMOA generators
SUPPORTED_GENERATORS = [
    "SineGenerator", "HyperplaneGenerator", "STAGGERGenerator",
    "RandomTreeGenerator", "SEAGenerator", "AgrawalGenerator", "LEDGenerator"
]
```

## Generator-Specific Classes

### SyntheticGenerator

Wrapper for CapyMOA synthetic dataset generators.

```python
class SyntheticGenerator:
    """Generator for synthetic datasets using CapyMOA."""
    
    def __init__(
        self, 
        generator_name: str,
        generator_config: Dict,
        drift_config: Optional[Dict] = None
    ):
        """
        Initialize synthetic generator.
        
        Args:
            generator_name: CapyMOA generator name
            generator_config: Generator-specific parameters
            drift_config: Drift pattern specification
        """
    
    def generate(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic dataset.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (features, targets)
        """
```

### RealWorldGenerator  

Wrapper for UCI ML Repository dataset loading.

```python
class RealWorldGenerator:
    """Generator for real-world datasets from UCI repository."""
    
    def __init__(
        self,
        uci_config: Dict,
        drift_config: Optional[Dict] = None
    ):
        """
        Initialize real-world dataset generator.
        
        Args:
            uci_config: UCI dataset parameters
            drift_config: Optional drift injection parameters
        """
    
    def generate(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load real-world dataset with optional drift injection.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (features, targets)
        """
```

## Usage Patterns

### Common Workflows

**Basic Dataset Generation**:

```python
import drift_datasets as dd

# Generate simple synthetic dataset
config = {
    "dataset": {"name": "simple", "type": "synthetic", "generator": "SineGenerator"},
    "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
    "generator_config": {"n_instances": 1000, "random_seed": 42}
}

dataset = dd.create_dataset(config)
print(f"Generated dataset: {dataset.X.shape}")
```

**Drift Detection Research**:

```python
# Load dataset with drift
dataset = dd.create_dataset("drift_config.toml")

# Get features for modeling
X = dataset.get_drift_detection_features()
y = dataset.y

# Analyze drift points
for i, drift_point in enumerate(dataset.drift_metadata.drift_points):
    drift_type = dataset.drift_metadata.drift_types[i]
    print(f"Drift at sample {drift_point}: {drift_type}")

# Split for evaluation
train, test = dataset.split_temporal(ratio=0.7)
```

**Feature Analysis**:

```python
# Explore dataset features
summary = dataset.summarize_features()
print(f"Total features: {summary['total_features']}")
print(f"Modeling features: {summary['modeling_features']}")

# Get specific feature types
continuous = dataset.get_continuous_features()
categorical = dataset.get_categorical_features()

# Detailed feature information
for feature_name in dataset.X.columns:
    info = dataset.get_feature_info(feature_name)
    print(f"{feature_name}: {info.type} ({info.role})")
```

This comprehensive API reference covers all public interfaces in drift-datasets. For implementation examples, see the [examples directory](examples/) and [quickstart tutorial](quickstart.md).
