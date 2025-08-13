# drift-datasets

A Python library for generating standardized concept drift datasets with ground truth metadata for machine learning research. The library provides unified dataset objects containing feature matrices, target vectors, and comprehensive drift annotations through simple TOML configuration.

## Features

- **Synthetic Datasets**: Generate datasets using CapyMOA (SineGenerator, HyperplaneGenerator, STAGGER, etc.)
- **Real-World Datasets**: Load UCI ML Repository datasets with optional drift injection
- **Mixed Datasets**: Combine multiple data sources with unified drift patterns
- **Ground Truth Metadata**: Precise drift point locations, types, and characteristics
- **Feature-Level Metadata**: Detailed type and role information for proper drift detection
- **TOML Configuration**: Reproducible dataset generation through configuration files
- **Research-Ready**: Methods optimized for drift detection algorithm evaluation

## Installation

```bash
pip install drift-datasets
```

**Requirements:**

- Python ≥3.10
- Java Runtime Environment (required for CapyMOA synthetic generation)

## Quick Start

### 1. Create a Configuration File

**Synthetic Dataset (synthetic_sine.toml):**

```toml
[dataset]
name = "sine_drift_demo"
type = "synthetic"
source = "capymoa"
generator = "SineGenerator"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "feature_0"
type = "continuous"
role = "feature"
description = "Primary sine wave component"

[[features]]
name = "feature_1"
type = "continuous"
role = "feature"
description = "Secondary sine wave component"

[[features]]
name = "target"
type = "categorical"
role = "target"
description = "Binary classification target"

[generator_config]
n_instances = 5000
classification_function = 1
has_noise = true
noise_level = 0.1
random_seed = 42

[drift_config]
drift_points = [1000, 3000]
drift_types = ["concept", "covariate"]
drift_patterns = ["abrupt", "gradual"]
drift_widths = [0, 500]
drift_alphas = [0.0, 0.3]
```

**Real-World Dataset (uci_electricity.toml):**

```toml
[dataset]
name = "electricity_drift"
type = "real_world"
source = "ucimlrepo"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "date"
type = "mixed"
role = "timestamp"
description = "Timestamp of measurement"

[[features]]
name = "day_of_week"
type = "categorical"
role = "feature"
description = "Day of week (1-7)"

[[features]]
name = "period"
type = "continuous"
role = "feature"
description = "Time period within day"

[[features]]
name = "nswprice"
type = "continuous"
role = "feature"
description = "NSW electricity price"

[[features]]
name = "class"
type = "categorical"
role = "target"
description = "Price movement classification"

[uci_config]
dataset_id = 321
preprocessing = ["normalize", "temporal_order"]

[drift_simulation]
inject_drift = true
drift_points = [1000, 2000]
drift_types = ["covariate", "concept"]
drift_patterns = ["abrupt", "gradual"]
drift_method = "feature_rotation"
affected_features = [[1, 2, 3], [2, 3]]
```

### 2. Load and Explore Dataset

```python
import drift_datasets as dd

# Load dataset from configuration
dataset = dd.create_drift_dataset("synthetic_sine.toml")

# Explore dataset structure
print(f"Dataset: {dataset.name}")
print(f"Shape: X={dataset.X.shape}, y={dataset.y.shape}")
print(f"Drift points: {dataset.drift_metadata.drift_points}")
print(f"Drift types: {dataset.drift_metadata.drift_types}")

# Feature summary
feature_summary = dataset.summarize_features()
print(f"Modeling features: {feature_summary['modeling_features']}")
print(f"Features to exclude: {feature_summary['features_to_exclude']}")
```

### 3. Prepare Data for Drift Detection

```python
# Get features appropriate for drift detection (excludes timestamps, IDs, etc.)
X_drift = dataset.get_drift_detection_features()
y = dataset.y

print(f"Features for drift detection: {list(X_drift.columns)}")
print(f"Data shape: {X_drift.shape}")

# Split temporally while preserving drift structure  
train_dataset, test_dataset = dataset.split_temporal(ratio=0.7)
X_train = train_dataset.get_drift_detection_features()
X_test = test_dataset.get_drift_detection_features()
y_train = train_dataset.y
y_test = test_dataset.y

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train drift points: {train_dataset.drift_metadata.drift_points}")
print(f"Test drift points: {test_dataset.drift_metadata.drift_points}")
```

### 4. Analyze Drift Characteristics

```python
# Check for drift at specific samples
for i in range(0, len(dataset.X), 500):
    if dataset.is_drift_point(i, tolerance=10):
        drift_type = dataset.get_drift_type_at(i)
        affected_features = dataset.get_affected_features_at(i)
        print(f"Drift at sample {i}: type={drift_type}, features={affected_features}")

# Get concept segments
segments = dataset.get_concept_segments()
for i, (start, end) in enumerate(segments):
    print(f"Concept {i}: samples {start}-{end} ({end-start} samples)")

# Validate drift metadata consistency
validation = dataset.validate_drift_metadata()
print(f"Metadata valid: {validation['valid']}")
if not validation['valid']:
    print(f"Issues: {validation['issues']}")
```

## Advanced Usage

### Feature Type-Specific Access

```python
# Get only continuous features (for statistical drift detection methods)
X_continuous = dataset.get_continuous_features()
print(f"Continuous features: {list(X_continuous.columns)}")

# Get only categorical features (for categorical drift detection methods)
X_categorical = dataset.get_categorical_features()
print(f"Categorical features: {list(X_categorical.columns)}")

# Get features by role
modeling_features = dataset.get_features_by_role("feature")
timestamp_features = dataset.get_features_by_role("timestamp")
```

### Detailed Feature Information

```python
# Inspect individual feature metadata
for feature_name in dataset.X.columns:
    info = dataset.get_feature_info(feature_name)
    print(f"{feature_name}: {info.type} ({info.role}) - {info.description}")

# Find features by characteristics
continuous_features = dataset.get_features_by_type("continuous")
excluded_features = dataset.get_features_by_role("exclude")
```

### Mixed Dataset Configuration

```toml
[dataset]
name = "mixed_temporal_drift"
type = "mixed"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 3

[mixed_config]
combination_type = "sequential"

    [[mixed_config.components]]
    source = "capymoa"
    generator = "SineGenerator"
    n_instances = 3000
    weight = 0.4

    [[mixed_config.components]]
    source = "ucimlrepo"
    dataset_id = 45
    n_instances = 2000
    weight = 0.6

[drift_config]
drift_points = [1500, 4000]
drift_types = ["concept", "covariate"]
drift_patterns = ["abrupt", "gradual"]
```

## Configuration Reference

### Dataset Types

- **synthetic**: Generated using CapyMOA generators
- **real_world**: Loaded from UCI ML Repository  
- **mixed**: Combination of multiple sources

### Feature Types

- **continuous**: Numerical data with meaningful distances (temperature, price, age)
- **categorical**: Discrete categories (colors, days, binary yes/no)
- **mixed**: Combined continuous/categorical aspects (timestamps)

### Feature Roles

- **feature**: Used for modeling and drift detection
- **target**: Prediction target (excluded from feature matrix)
- **timestamp**: Temporal data (excluded from drift detection)
- **identifier**: Unique IDs (excluded from modeling)
- **metadata**: Descriptive info (excluded from modeling)
- **exclude**: Explicitly excluded from all analysis

### Drift Types

- **covariate**: Changes in input feature distribution P(X)
- **concept**: Changes in conditional distribution P(y|X)
- **prior**: Changes in class distribution P(y)
- **none**: No drift (stable concept)

### Drift Patterns

- **abrupt**: Instantaneous change at drift point
- **gradual**: Smooth transition over specified width
- **recurring**: Periodic return to previous concepts
- **incremental**: Continuous small changes

## Supported Generators

### CapyMOA Synthetic Generators

- **SineGenerator**: Sine wave-based classification with configurable functions
- **HyperplaneGenerator**: Rotating hyperplane classification  
- **STAGGERGenerator**: STAGGER concepts with discrete attributes
- **RandomTreeGenerator**: Random decision tree-based concepts
- **SEAGenerator**: SEA concepts with abrupt changes
- **AgrawalGenerator**: Agrawal functions for classification
- **LEDGenerator**: LED display digit recognition with noise

### UCI Repository Datasets

Access to 400+ real-world datasets including:

- Electricity market data (ID: 321)
- Adult census data (ID: 2)
- Forest cover types (ID: 31)
- Poker hands (ID: 158)
- And many more...

## API Reference

### Main Entry Point

```python
create_drift_dataset(config_path: str) -> DriftDataset
```

Load dataset from TOML configuration file.

### DriftDataset Methods

**Core Data Access:**

- `.X: pd.DataFrame` - Feature matrix
- `.y: pd.Series` - Target vector
- `.dataset_metadata: DatasetMetadata` - Feature and structure metadata
- `.drift_metadata: DriftMetadata` - Ground truth drift information

**Feature Filtering:**

- `.get_drift_detection_features() -> pd.DataFrame` - Features suitable for drift detection
- `.get_continuous_features() -> pd.DataFrame` - Only continuous features
- `.get_categorical_features() -> pd.DataFrame` - Only categorical features
- `.get_features_by_type(feature_type: str) -> pd.DataFrame` - Filter by type
- `.get_features_by_role(role: str) -> pd.DataFrame` - Filter by role

**Feature Information:**

- `.get_feature_info(name: str) -> FeatureMetadata` - Detailed feature metadata
- `.summarize_features() -> Dict[str, Any]` - Overview of all features

**Drift Analysis:**

- `.is_drift_point(idx: int, tolerance: int = 0) -> bool` - Check if sample near drift
- `.get_concept_segments() -> List[Tuple[int, int]]` - Concept boundary indices
- `.get_drift_type_at(idx: int) -> str` - Drift type at sample index
- `.get_affected_features_at(idx: int) -> List[int]` - Features affected by drift
- `.validate_drift_metadata() -> Dict[str, Any]` - Verify metadata consistency

**Data Splitting:**

- `.split_temporal(ratio: float) -> Tuple[DriftDataset, DriftDataset]` - Train/test split

## Performance

- **Synthetic Generation**: <30 seconds for 100K samples
- **UCI Loading**: <10 seconds for datasets up to 1M samples
- **Memory Usage**: <2GB for 1M sample datasets with metadata
- **Configuration Parsing**: <1 second for complex TOML files

## Development

### Requirements

- Python ≥3.10
- Java Runtime Environment (for CapyMOA)
- See `requirements.txt` for Python dependencies

### Development Setup

```bash
# Clone repository
git clone https://github.com/BorjaEst/drift-datasets.git
cd drift-datasets

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/ --cov=drift_datasets

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
src/drift_datasets/
├── __init__.py           # Main exports
├── literals.py           # Type literals and enums  
├── factory.py           # create_drift_dataset() entry point
├── models/              # Data model definitions
├── generators/          # Dataset generators (synthetic, real_world, mixed)
└── utils/              # Configuration parsing and utilities

tests/
├── conftest.py         # Shared fixtures
├── synthetic/          # Synthetic dataset tests
├── real_world/         # UCI integration tests
├── mixed/              # Mixed dataset tests
└── functional/         # End-to-end workflow tests
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes following TDD practices:
   - Write functional tests first in appropriate test directory
   - Implement feature to make tests pass
   - Ensure all existing tests still pass
4. Format code (`black`, `isort`) and add type hints
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)  
7. Open a Pull Request

### Testing Philosophy

This project follows Test-Driven Development (TDD) with emphasis on functional tests:

- **60% Functional Tests**: End-to-end user workflows
- **30% Integration Tests**: External API interactions  
- **10% Unit Tests**: Individual component behavior

All new features must include functional tests demonstrating real research workflows.

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) file for details.

## Citation

If you use drift-datasets in your research, please cite:

```bibtex
@software{drift_datasets,
  title={drift-datasets: Standardized Concept Drift Datasets for Machine Learning Research},
  author={Borja Esteban},
  url={https://github.com/BorjaEst/drift-datasets},
  year={2025}
}
```

## Links

- **Homepage**: <https://github.com/BorjaEst/drift-datasets>
- **Documentation**: <https://drift-datasets.readthedocs.io>
- **PyPI**: <https://pypi.org/project/drift-datasets/>
- **Issues**: <https://github.com/BorjaEst/drift-datasets/issues>
