# Drift-Datasets Architecture

## Project Overview

**drift-datasets** is a simplified Python library for creating standardized concept drift datasets with ground truth metadata. The library reads TOML configuration files and returns unified dataset objects containing feature matrix (X), target vector (y), and comprehensive drift metadata indicating when and how much drift occurs.

## Core Concepts

### Unified Dataset Model

The library provides a single, unified dataset object that contains:

- **Unified Data**: Single feature matrix (X) and target vector (y) for the entire dataset
- **Drift Ground Truth**: Metadata indicating exact drift points, intensities, and types
- **Configuration-Driven**: TOML files define all dataset parameters for reproducibility
- **Analysis Ready**: Pre-structured data with drift annotations for immediate analysis

### Supported Dataset Types

1. **Synthetic Datasets**: Generated using CapyMOA (Sine, Hyperplane, STAGGER)
2. **Real-World Datasets**: From UCI Repository with simulated drift injection
3. **Mixed Datasets**: Temporal combinations of multiple generators

## Technology Stack

### Core Dependencies

```toml
[dependencies]
pydantic = "^2.0.0"     # Data validation and serialization
pandas = "^2.0.0"       # Data manipulation
numpy = "^1.21.0"       # Numerical computations
capymoa = "^0.10.0"     # MOA streaming algorithms
ucimlrepo = "^0.0.7"    # UCI ML Repository access
toml = "^0.10.2"        # Configuration parsing

[dev-dependencies]
pytest = "^7.0.0"       # Testing framework
pytest-cov = "^3.0.0"   # Coverage reporting
```

### Simplified Architecture

```
TOML Config → Parser → Generator → Unified Dataset Object
                         ↓
              [X, y, drift_metadata]
```

## Dataset Object Model

### DriftDataset Structure

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import pandas as pd

class DriftMetadata(BaseModel):
    """Ground truth drift information."""
    drift_points: List[int]              # Sample indices where drift occurs
    drift_intensities: Optional[List[float]]  # Drift strength (0.0 to 1.0)
    drift_types: Optional[List[str]]     # Type: 'abrupt', 'gradual', 'recurring'
    concepts: Optional[List[str]]        # Concept labels for each segment

class DriftDataset(BaseModel):
    """Unified drift dataset with ground truth metadata."""
    
    # Core data
    X: pd.DataFrame                      # Feature matrix (entire dataset)
    y: pd.Series                        # Target vector (entire dataset)
    
    # Drift ground truth
    drift_metadata: DriftMetadata       # When and how much drift occurs
    
    # Dataset information  
    name: str                           # Human-readable name
    source_type: str                    # synthetic | real_world | mixed
    generator_info: Dict[str, Any]      # Generator configuration
    config: Dict[str, Any]              # Original TOML configuration
```

### Key Methods

- `get_concept_segments()`: Returns (start, end) indices for each concept
- `get_segment_data(idx)`: Extract features/targets for a specific concept
- `is_drift_point(idx, tolerance)`: Check if sample is near drift point
- `split_temporal(ratio)`: Split while preserving drift structure

## Configuration Schema

### Basic Configuration

```toml
[dataset]
name = "sine_1000"
type = "synthetic"              # synthetic | real_world | mixed
source = "capymoa"             # capymoa | ucimlrepo
generator = "SineGenerator"

[generator_config]
n_instances = 10000
classification_function = 1
has_noise = true
noise_level = 0.1
random_seed = 42

[drift_config]
drift_points = [2000, 5000, 8000]      # Sample indices where drift occurs
drift_types = ["abrupt", "gradual", "abrupt"]
drift_intensities = [0.8, 0.5, 0.9]    # Optional: drift strength
concepts = ["sine_1", "sine_2", "sine_3", "sine_4"]  # Optional: concept names
```

### Real-World Dataset Example

```toml
[dataset]
name = "electricity_drift"
type = "real_world"
source = "ucimlrepo"

[uci_config]
dataset_id = 321
preprocessing = ["normalize", "temporal_order"]

[drift_simulation]
inject_drift = true
drift_points = [1000, 3000]
drift_method = "feature_rotation"
drift_intensities = [0.6, 0.4]
```

## Implementation

### Factory Pattern

```python
from drift_datasets.models.dataset import DriftDataset, DriftMetadata

def create_drift_dataset(config_path: str) -> DriftDataset:
    """Main entry point: TOML config → DriftDataset object"""
    config = load_toml_config(config_path)
    
    if config['dataset']['type'] == 'synthetic':
        return SyntheticGenerator(config).create()
    elif config['dataset']['type'] == 'real_world':
        return RealWorldGenerator(config).create()
    elif config['dataset']['type'] == 'mixed':
        return MixedGenerator(config).create()
    else:
        raise ValueError(f"Unknown dataset type: {config['dataset']['type']}")
```

### Generator Implementation

```python
class SyntheticGenerator:
    """Generate synthetic datasets with known drift points."""
    
    def __init__(self, config: dict):
        self.config = config
        self.generator_config = config['generator_config']
        self.drift_config = config.get('drift_config', {})
    
    def create(self) -> DriftDataset:
        # Generate data using CapyMOA
        X, y = self._generate_data()
        
        # Create drift metadata
        drift_metadata = DriftMetadata(
            drift_points=self.drift_config.get('drift_points', []),
            drift_intensities=self.drift_config.get('drift_intensities'),
            drift_types=self.drift_config.get('drift_types'),
            concepts=self.drift_config.get('concepts')
        )
        
        return DriftDataset(
            X=X, y=y,
            drift_metadata=drift_metadata,
            name=self.config['dataset']['name'],
            source_type='synthetic',
            generator_info=self.generator_config,
            config=self.config
        )
```

## Usage Examples

### Basic Usage

```python
import drift_datasets as dd

# Load dataset from configuration
dataset = dd.create_drift_dataset("configs/sine_drift.toml")

# Access unified data
X, y = dataset.X, dataset.y  # Entire dataset
print(f"Dataset: {dataset.name}")
print(f"Shape: {X.shape}, Drift points: {dataset.drift_metadata.drift_points}")

# Analyze drift structure
for i, (start, end) in enumerate(dataset.get_concept_segments()):
    X_segment, y_segment = dataset.get_segment_data(i)
    print(f"Concept {i}: samples {start}-{end}, shape {X_segment.shape}")

# Check if specific samples are near drift points
drift_samples = [i for i in range(len(X)) if dataset.is_drift_point(i, tolerance=10)]
print(f"Samples near drift: {len(drift_samples)}")
```

### Temporal Splitting

```python
# Split while preserving drift structure
train_dataset, test_dataset = dataset.split_temporal(split_ratio=0.7)

print(f"Train: {train_dataset.n_samples} samples, {train_dataset.n_drift_points} drift points")
print(f"Test: {test_dataset.n_samples} samples, {test_dataset.n_drift_points} drift points")

# Use for drift detection evaluation
X_train, y_train = train_dataset.X, train_dataset.y
X_test, y_test = test_dataset.X, test_dataset.y
```

## Testing Strategy

### Test-Driven Development

```python
def test_drift_dataset_structure():
    """Validate unified dataset structure with drift metadata."""
    dataset = create_drift_dataset("configs/test_sine.toml")
    
    # Test unified structure
    assert isinstance(dataset.X, pd.DataFrame)
    assert isinstance(dataset.y, pd.Series)
    assert len(dataset.X) == len(dataset.y)
    
    # Test drift metadata
    assert isinstance(dataset.drift_metadata.drift_points, list)
    assert all(0 <= dp < len(dataset.X) for dp in dataset.drift_metadata.drift_points)
    
    # Test concept segmentation
    segments = dataset.get_concept_segments()
    assert len(segments) == len(dataset.drift_metadata.drift_points) + 1

def test_temporal_splitting_preserves_drift():
    """Ensure temporal splitting maintains drift structure."""
    dataset = create_drift_dataset("configs/test_drift.toml")
    train, test = dataset.split_temporal(0.7)
    
    # Verify split maintains drift information
    total_drift_points = len(dataset.drift_metadata.drift_points)
    split_drift_points = len(train.drift_metadata.drift_points) + len(test.drift_metadata.drift_points)
    assert split_drift_points <= total_drift_points  # Some drift points might be at boundary
```

## Benefits of Simplified Design

### Advantages

1. **Unified Interface**: Single X, y structure is intuitive and widely compatible
2. **Ground Truth Clarity**: Drift metadata explicitly shows when/where/how much drift occurs
3. **Flexible Splitting**: Temporal splitting preserves drift structure when needed
4. **Simplified Testing**: Easier to validate single dataset object vs multiple splits
5. **Memory Efficient**: No data duplication across reference/test splits
6. **Analysis Ready**: Direct compatibility with scikit-learn and other ML libraries

## Conclusion

This simplified architecture removes unnecessary complexity while improving the core functionality. The unified dataset model with explicit drift metadata provides clearer ground truth information and is more compatible with standard ML workflows. The TOML configuration system remains for reproducibility, but the output is a single, comprehensive dataset object rather than multiple fragmented pieces.
