# drift-datasets

Generate standardized concept drift datasets with ground truth metadata for machine learning research through simple TOML configuration.

[![Version](https://img.shields.io/pypi/v/drift-datasets?color=blue)](https://pypi.org/project/drift-datasets/)
[![Python](https://img.shields.io/pypi/pyversions/drift-datasets)](https://pypi.org/project/drift-datasets/)
[![License](https://img.shields.io/github/license/BorjaEst/drift-datasets)](./LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow-status/BorjaEst/drift-datasets/tests.yml?branch=main&label=tests)](https://github.com/BorjaEst/drift-datasets/actions)
[![Coverage](https://img.shields.io/codecov/c/github/BorjaEst/drift-datasets)](https://codecov.io/gh/BorjaEst/drift-datasets)
[![Downloads](https://img.shields.io/pypi/dm/drift-datasets)](https://pypi.org/project/drift-datasets/)

## Overview

**drift-datasets** is a Python library for generating standardized concept drift datasets with comprehensive ground truth metadata. It serves machine learning researchers, data scientists, and algorithm developers who need reproducible datasets for evaluating drift detection methods, adaptive learning algorithms, and streaming data analysis.

The library addresses the challenge of inconsistent drift dataset generation across research studies by providing:

- Unified dataset objects containing feature matrices, targets, and detailed drift annotations
- Configuration-driven reproducible generation through TOML files
- Support for both synthetic and real-world datasets with optional drift injection
- Research-friendly parameter translation for common drift simulation frameworks

**Who it's for:**

- ML researchers evaluating drift detection algorithms
- Data scientists prototyping adaptive learning systems
- Algorithm developers benchmarking streaming methods
- Educators teaching concept drift concepts

**Why it exists:**
Different studies use varying base learners, datasets, metrics, and comparison methods, making it difficult to choose appropriate drift detection methods for specific situations. This library standardizes drift dataset generation with ground truth metadata for reliable, reproducible research.

## Key Features

- **Synthetic Datasets**: Generate datasets using CapyMOA (SineGenerator, HyperplaneGenerator, STAGGER, etc.)
- **Real-World Datasets**: Load UCI ML Repository datasets with optional drift injection
- **Mixed Datasets**: Combine multiple data sources with unified drift patterns
- **Ground Truth Metadata**: Precise drift point locations, types, and characteristics
- **Feature-Level Metadata**: Detailed type and role information for proper drift detection
- **Research Parameter Translation**: Automatically convert research-friendly parameters to implementation parameters
- **ExpertSystems Compatibility**: Full support for reproducing comparative drift detection studies
- **TOML Configuration**: Reproducible dataset generation through simple configuration files

## Architecture

The library follows a factory pattern with three main generators:

1. **SyntheticGenerator**: Creates synthetic datasets via CapyMOA with configurable drift patterns
2. **RealWorldGenerator**: Loads UCI datasets with optional drift injection
3. **MixedGenerator**: Combines multiple sources with unified drift metadata

All generators produce unified `DriftDataset` objects containing feature matrices (X), targets (y), and comprehensive drift metadata. A `ParameterTranslator` automatically converts research-friendly parameters to implementation-specific values.

**Architecture diagram**: [TODO: Add ./docs/architecture.png when available]

## Tech Stack and Requirements

- **Language**: Python ≥3.10
- **Core Dependencies**:
  - CapyMOA ≥0.10.0 (synthetic dataset generation)
  - ucimlrepo ≥0.0.7 (real-world dataset access)
  - NumPy ≥2.2.0, pandas ≥2.3.0, SciPy ≥1.15.0 (data manipulation)
- **System Requirements**:
  - Java Runtime Environment (required for CapyMOA synthetic generation)
  - Internet connection (for UCI dataset downloads)
  - 2GB+ RAM (for large dataset generation)

## Installation

### Prerequisites

Install Java Runtime Environment:

```bash
# Ubuntu/Debian
sudo apt-get install default-jre

# macOS
brew install openjdk

# Verify installation
java -version
```

### Install Package

```bash
# From PyPI (recommended)
pip install drift-datasets

# From source (development)
git clone https://github.com/BorjaEst/drift-datasets.git
cd drift-datasets
pip install -e .
```

## Quickstart

Generate your first drift dataset in under 2 minutes:

### 1. Create Configuration File

Create `my_dataset.toml`:

```toml
[dataset]
name = "quickstart_example"
type = "synthetic"
source = "capymoa"
generator = "SineGenerator"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "x"
type = "continuous"
role = "feature"

[[features]]
name = "y"
type = "continuous"
role = "feature"

[[features]]
name = "class"
type = "categorical"
role = "target"

[generator_config]
n_instances = 1000
classification_function = 1
random_seed = 42

[drift_config]
drift_points = [500]
drift_types = ["concept"]
drift_patterns = ["abrupt"]
```

### 2. Generate Dataset

```python
import drift_datasets

# Load configuration and generate dataset
dataset = drift_datasets.create_dataset("my_dataset.toml")

# Access data
print(f"Dataset shape: {dataset.X.shape}")
print(f"Target shape: {dataset.y.shape}")
print(f"Drift points: {dataset.drift_metadata.drift_points}")

# Use for drift detection research
X_train = dataset.get_drift_detection_features()  # Excludes target/metadata
is_drift = dataset.is_drift_point(500, tolerance=10)  # Check if sample 500 is near drift
```

**Expected output:**

```
Dataset shape: (1000, 2)
Target shape: (1000,)
Drift points: [500]
```

## Configuration

The library uses TOML configuration files with no environment variables required.

### Configuration Structure

| Section | Type | Required | Description |
|---------|------|----------|-------------|
| `[dataset]` | dict | Yes | Dataset metadata and type specification |
| `[metadata]` | dict | Yes | Dataset dimensionality and labeling information |
| `[[features]]` | list | Yes | Per-feature metadata specification |
| `[generator_config]` | dict | Conditional | Generator-specific parameters (synthetic datasets) |
| `[drift_config]` | dict | Optional | Drift pattern specification |
| `[uci_config]` | dict | Conditional | UCI dataset parameters (real-world datasets) |
| `[mixed_config]` | dict | Conditional | Mixed dataset combination rules |

### Default Values

- `drift_types`: `["none"]` (no drift)
- `drift_patterns`: `["abrupt"]`
- `affected_features`: `"all"`
- `random_seed`: `42`
- `noise_level`: `0.0`

## Usage

### Library/API Usage

```python
import drift_datasets

# Basic synthetic dataset
dataset = drift_datasets.create_dataset("config.toml")

# Access data components
X = dataset.X                    # Feature matrix (pandas.DataFrame)
y = dataset.y                    # Target vector (pandas.Series)
metadata = dataset.drift_metadata  # Ground truth drift information

# Research workflow methods
features = dataset.get_drift_detection_features()  # Features for modeling
segments = dataset.get_concept_segments()          # [(start, end), ...] 
is_drift = dataset.is_drift_point(idx=100)        # Check drift proximity

# Temporal splitting preserving drift structure
train, test = dataset.split_temporal(ratio=0.7)

# Feature filtering by role/type
continuous = dataset.get_continuous_features()
targets_only = dataset.get_features_by_role("target")
```

### Common Patterns

```python
# Pattern 1: ExpertSystems paper reproduction
config = {
    "dataset": {"name": "sine_expertsystems", "type": "synthetic", 
                "source": "capymoa", "generator": "SineGenerator"},
    "generator_config": {"n_instances": 50000, "random_seed": 42},
    "drift_config": {
        "drift_points": [10000, 25000],
        "drift_patterns": ["abrupt", "abrupt"],
        "concept_reversal": True  # Classification reverses after each drift
    }
}

# Pattern 2: Real-world dataset with injected drift  
config = {
    "dataset": {"name": "electricity_drift", "type": "real_world", "source": "ucimlrepo"},
    "uci_config": {"dataset_id": 321},  # Electricity market dataset
    "drift_config": {
        "drift_points": [15000],
        "drift_simulation": "concept_shift",
        "drift_intensity": 0.3
    }
}

# Pattern 3: Research-friendly parameter translation
config = {
    "drift_config": {
        "drift_patterns": ["continuous_gradual"],
        "transition_durations": [1000],    # Research parameter
        "drift_intensities": [0.8],        # Research parameter  
        # Automatically translates to CapyMOA drift_widths=[1000], drift_alphas=[0.8]
    }
}
```

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

## Data and Interfaces

### Input Formats

- **TOML Configuration**: Primary input format for reproducible generation
- **Dictionary Configuration**: Programmatic configuration via Python dicts
- **UCI Dataset IDs**: Integer identifiers for real-world datasets

### Output Formats

**DriftDataset Object**:

```python
class DriftDataset:
    X: pd.DataFrame              # Feature matrix (n_samples, n_features)
    y: pd.Series                 # Target vector (n_samples,)
    name: str                    # Dataset identifier
    source_type: DatasetType     # "synthetic" | "real_world" | "mixed"
    dataset_metadata: DatasetMetadata
    drift_metadata: DriftMetadata
    config: dict                 # Original configuration
```

**DriftMetadata Schema**:

```python
class DriftMetadata:
    drift_points: List[int]              # Sample indices of drift occurrences
    drift_types: List[DriftType]         # ["covariate", "concept", "prior", "none"]
    drift_patterns: List[DriftPattern]   # ["abrupt", "gradual", "continuous_gradual", ...]
    affected_features: List[List[int]]   # Feature indices per drift
    # Research-friendly parameters
    transition_durations: List[int]      # Duration in samples for each transition
    drift_intensities: List[float]       # Magnitude of change [0.0-1.0]
```

### Export Formats

```python
# Export to common formats (TODO: Implement in REQ-016, REQ-017)
dataset.to_arff("output.arff")           # Weka ARFF format
dataset.to_csv("output.csv")             # CSV with metadata
dataset.to_parquet("output.parquet")     # Efficient binary format
```

## Examples

### Scenario 1: Drift Detection Algorithm Evaluation

```python
# Generate dataset with known drift points
dataset = drift_datasets.create_dataset({
    "dataset": {"name": "evaluation_set", "type": "synthetic", 
                "source": "capymoa", "generator": "HyperplaneGenerator"},
    "generator_config": {"n_instances": 10000, "n_features": 10, "random_seed": 123},
    "drift_config": {
        "drift_points": [2500, 5000, 7500],
        "drift_patterns": ["abrupt", "gradual", "abrupt"],
        "transition_durations": [0, 500, 0]
    }
})

# Evaluate your drift detector
X = dataset.get_drift_detection_features()
y = dataset.y

# Your drift detection algorithm
predicted_drifts = your_drift_detector.fit_predict(X, y)

# Compare with ground truth
actual_drifts = dataset.drift_metadata.drift_points
evaluation_metrics = compute_drift_detection_metrics(predicted_drifts, actual_drifts)
```

### Scenario 2: ExpertSystems Paper Reproduction

```python
# Reproduce Sine dataset from ExpertSystems comparative study
sine_config = {
    "dataset": {"name": "expertsystems_sine", "type": "synthetic",
                "source": "capymoa", "generator": "SineGenerator"},
    "generator_config": {"n_instances": 50000, "classification_function": 1, "random_seed": 42},
    "drift_config": {
        "drift_points": [10000, 25000, 40000],
        "drift_types": ["concept", "concept", "concept"],
        "drift_patterns": ["abrupt", "abrupt", "abrupt"],
        "concept_reversal": True  # Classification reverses after each drift
    }
}

sine_dataset = drift_datasets.create_dataset(sine_config)

# Reproduce Hyperplane dataset with continuous gradual drift
hyp_config = {
    "dataset": {"name": "expertsystems_hyperplane", "type": "synthetic",
                "source": "capymoa", "generator": "HyperplaneGenerator"},  
    "generator_config": {"n_instances": 100000, "n_dimensions": 10, 
                        "n_drifting_dimensions": 10, "noise_percentage": 0.05},
    "drift_config": {
        "drift_patterns": ["continuous_gradual"],  # No discrete drift points
        "rotation_speed": 0.001,                   # Slow continuous rotation
        "continuous_drift": True
    }
}

hyp_dataset = drift_datasets.create_dataset(hyp_config)
```

## Testing

### Run Tests Locally

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=drift_datasets --cov-report=html

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests

# Run performance benchmarks
pytest -m performance --benchmark-only
```

### Test Coverage

- **Target Coverage**: ≥90% for all production code
- **Current Status**: TODO: Set up coverage reporting
- **Test Categories**: Unit (60%), Integration (30%), Performance (10%)

### Test Data

Test datasets are located in `tests/assets/`:

- Sample TOML configurations
- Mock CapyMOA responses  
- Small UCI dataset samples
- Expected output fixtures

## Linting and Formatting

```bash
# Format code
black src tests
isort src tests

# Lint code
ruff check src tests
mypy src

# Pre-commit hooks (recommended)
pre-commit install
pre-commit run --all-files
```

**Tools configured**:

- **Black**: Code formatting (line length: 140)
- **isort**: Import sorting
- **Ruff**: Fast Python linter
- **mypy**: Static type checking

## Build and CI/CD

### Build Package

```bash
# Build distribution packages
python -m build

# Upload to PyPI (maintainers only)
twine upload dist/*
```

### CI/CD Pipeline

TODO: Set up GitHub Actions workflow with:

- **Tests**: Run on Python 3.10+ across Ubuntu, macOS, Windows
- **Coverage**: Upload coverage reports to Codecov
- **Linting**: Code quality checks with ruff, black, mypy
- **Security**: Dependency vulnerability scanning
- **Release**: Automated PyPI publishing on tags

### Versioning Strategy

- **Scheme**: Semantic versioning (MAJOR.MINOR.PATCH)
- **Current**: v0.1.0 (development/alpha)
- **Releases**: Manual tagging triggers automated PyPI upload

## Performance

### Benchmarks and Limits

| Dataset Size | Generation Time | Memory Usage | Notes |
|--------------|-----------------|--------------|-------|
| 10K samples | <5 seconds | <100MB | Typical research size |
| 100K samples | <30 seconds | <1GB | Large experiments |
| 1M samples | <5 minutes | <8GB | Maximum recommended |

### Known Bottlenecks

- **CapyMOA JVM startup**: 2-3 second overhead per generation
- **UCI API rate limiting**: ~1 request/second for downloads
- **Memory scaling**: Linear with dataset size for in-memory operations

### Profiling

```python
# Enable performance logging
import logging
logging.getLogger('drift_datasets').setLevel(logging.DEBUG)

# Profile generation time
import time
start = time.time()
dataset = drift_datasets.create_dataset(config)
print(f"Generation took {time.time() - start:.2f} seconds")
```

## Security

### Threat Model

- **Configuration injection**: Malicious TOML files could execute arbitrary code
- **Network requests**: UCI API calls could be intercepted or return malicious data
- **File system access**: Generated datasets written to user-specified paths

### Security Measures

- **Input validation**: All configuration parameters validated against schemas
- **Network timeouts**: UCI API calls have 30-second timeouts
- **Safe defaults**: No file system access unless explicitly configured
- **No authentication required**: Library operates locally without credentials

### Secrets Management

No secrets or API keys required. All external data sources (UCI, CapyMOA) are publicly accessible.

## Observability

### Logging

```python
import logging

# Configure logging levels
logging.getLogger('drift_datasets').setLevel(logging.INFO)

# Available log levels:
# DEBUG: Parameter translations, CapyMOA interactions
# INFO: Dataset generation progress, timing information  
# WARNING: Performance degradation, fallback behaviors
# ERROR: Generation failures, network issues
# CRITICAL: System-level failures
```

### Metrics

TODO: Implement metrics collection for:

- Dataset generation latency by type/size
- Memory usage patterns
- UCI API response times
- CapyMOA generation success rates

### Tracing

No distributed tracing implemented. All operations are local Python function calls.

## Compatibility

### Operating Systems

- **Linux**: Full support (tested on Ubuntu 20.04+)
- **macOS**: Full support (tested on macOS 11+)  
- **Windows**: Partial support (CapyMOA requires manual Java setup)

### Python Versions

- **Supported**: Python 3.10, 3.11, 3.12
- **Recommended**: Python 3.11+ for best performance
- **EOL Policy**: Drop support 6 months after Python version EOL

### Breaking Changes Policy

- **Major versions** (1.0, 2.0, etc.): Breaking API changes allowed
- **Minor versions** (1.1, 1.2, etc.): Backwards compatible feature additions
- **Patch versions** (1.1.1, 1.1.2, etc.): Bug fixes only
- **Deprecation**: 2 minor version warning period for removed features

## Internationalization/Accessibility

- **Language**: English-only error messages and documentation
- **Character encoding**: UTF-8 for all text data
- **Accessibility**: Not applicable (library, not user interface)

## Roadmap and Status

### Maturity and Stability

- **Current Status**: Alpha (v0.1.0) - API subject to change
- **Stability**: Core functionality implemented, extensive testing in progress
- **Production Ready**: Not recommended for production use until v1.0.0

### Planned Work

**v0.2.0 (Q2 2025)**:

- Complete test suite implementation (REQ-017)
- CI/CD pipeline setup
- Performance optimizations
- Export format support (REQ-016, REQ-017)

**v0.3.0 (Q3 2025)**:

- Advanced drift patterns (REQ-022)
- Mixed dataset combinations (REQ-008, REQ-009)
- Comprehensive documentation

**v1.0.0 (Q4 2025)**:

- API stability guarantee
- Full ExpertSystems paper compatibility
- Production deployment guides
- Long-term support commitment

## Contributing

We welcome contributions! Here's how to get started:

### Quick Start

1. Fork the repository on GitHub
2. Clone your fork locally
3. Install development dependencies: `pip install -e ".[dev]"`
4. Create a feature branch: `git checkout -b feature-name`
5. Make changes and add tests
6. Run tests: `pytest`
7. Format code: `black . && isort .`
8. Submit a pull request

### Development Setup

```bash
git clone https://github.com/BorjaEst/drift-datasets.git
cd drift-datasets
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

**Detailed guidelines**: [TODO: Create CONTRIBUTING.md]

## Code of Conduct

TODO: Add Code of Conduct link when created. We expect all contributors to follow professional, inclusive behavior standards.

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

**SPDX-License-Identifier**: GPL-3.0-or-later

See the [LICENSE](./LICENSE) file for full license text.

### License Summary

- ✅ Use, modify, distribute freely
- ✅ Private and commercial use allowed  
- ✅ Patent protection included
- ⚠️ Must disclose source code for derivative works
- ⚠️ Same license required for derivative works
- ❌ No liability or warranty provided

## Support and Contact

### Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/BorjaEst/drift-datasets/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/BorjaEst/drift-datasets/discussions)
- **Documentation**: [In-code examples and API reference](./docs/)

### Security Contact

Report security vulnerabilities privately to: **<boressan@outlook.com>**

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if available)

We aim to acknowledge security reports within 48 hours.

### Maintainers

- **Borja Esteban** ([@BorjaEst](https://github.com/BorjaEst)) - Project maintainer and primary developer

## FAQ and Troubleshooting

### Frequently Asked Questions

**Q: Do I need Java installed to use this library?**
A: Yes, Java Runtime Environment is required for CapyMOA synthetic dataset generation. Real-world datasets from UCI don't require Java.

**Q: Can I use this library without internet access?**
A: Partially. Synthetic dataset generation works offline, but UCI real-world datasets require internet connectivity for initial download.

**Q: How do I reproduce datasets from research papers?**
A: Use the ExpertSystems configuration examples in the documentation. The library includes pre-configured templates for common research datasets.

**Q: What's the difference between drift_patterns: "gradual" vs "continuous_gradual"?**
A: "gradual" has discrete drift points with smooth transitions, while "continuous_gradual" represents ongoing change without stable periods (like rotating hyperplanes).

**Q: Can I generate datasets larger than 1M samples?**
A: Technically yes, but memory usage becomes significant. Consider streaming generation or breaking into smaller datasets for very large experiments.

### Common Issues and Fixes

**Issue**: `ImportError: No module named 'capymoa'`

```bash
# Fix: Install CapyMOA dependency
pip install capymoa>=0.10.0
```

**Issue**: `RuntimeError: Java Runtime Environment not found`

```bash
# Fix: Install Java JRE
# Ubuntu: sudo apt-get install default-jre
# macOS: brew install openjdk
# Verify: java -version
```

**Issue**: `ConnectionError: Unable to connect to UCI repository`

```bash
# Fix: Check internet connection and retry
# Alternative: Use cached datasets or synthetic generation
```

**Issue**: `ValueError: Invalid drift_pattern 'custom_pattern'`

```python
# Fix: Use supported drift patterns
valid_patterns = ["abrupt", "gradual", "continuous_gradual", "intermittent_gradual", "recurring", "incremental"]
```

**Issue**: Memory errors with large datasets

```python
# Fix: Reduce dataset size or use streaming approach
config["generator_config"]["n_instances"] = 100000  # Instead of 1000000
```

## Changelog

[TODO: Create CHANGELOG.md] - Link to detailed version history and release notes.

## Glossary

- **Concept Drift**: Changes in the statistical properties of data over time, affecting model performance
- **Drift Point**: Specific sample index where concept drift begins to occur
- **Drift Type**: Classification of drift cause - covariate (input distribution), concept (relationship), or prior (class distribution)
- **Ground Truth**: Known, verified drift locations and characteristics for evaluation purposes
- **CapyMOA**: Java-based framework for streaming data analysis and synthetic dataset generation
- **UCI Repository**: University of California Irvine Machine Learning Repository of real-world datasets
- **Feature Role**: Purpose classification of dataset columns (feature, target, timestamp, etc.)
- **Transition Duration**: Time period over which drift occurs (research-friendly parameter)
- **ExpertSystems**: Reference comparative study for drift detection methods requiring specific dataset configurations

```

### 2. Load and Explore Dataset

```python
import drift_datasets as dd

# Load dataset from configuration
dataset = dd.create_dataset("synthetic_sine.toml")

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
- **continuous_gradual**: Continuous drift with no stable periods (ExpertSystems Hyperplane)
- **intermittent_gradual**: Gradual transitions with stable periods between changes (ExpertSystems Mixed)
- **recurring**: Periodic return to previous concepts
- **incremental**: Continuous small changes

### Research-Friendly Drift Parameters

**Basic Parameters:**

- **drift_points**: Sample indices where drift occurs `[1000, 3000]`
- **drift_types**: Type classification per drift point `["concept", "covariate"]`
- **drift_patterns**: Pattern behavior per drift point `["abrupt", "continuous_gradual"]`

**Advanced Research Parameters:**

- **transition_durations**: How long each transition takes `[0, 500]` (0 = instant)
- **drift_intensities**: Magnitude of change `["complete", "moderate", "mild"]`
- **affected_features**: Feature indices affected by each drift `[[0, 1], [1, 2]]`
- **concept_reversal**: Boolean for classification reversal `true/false`
- **stable_periods**: Enable stable concepts between transitions `true/false`
- **rotation_speed**: For hyperplane datasets `0.001` (slow) to `0.1` (fast)

**CapyMOA Compatibility Parameters (Advanced):**

- **drift_widths**: Direct CapyMOA width parameter `[0, 500]`
- **drift_alphas**: Direct CapyMOA alpha parameter `[1.0, 0.3]`

*Note: Research parameters automatically translate to CapyMOA parameters. Advanced users can specify CapyMOA parameters directly for fine control.*

### ExpertSystems Paper Dataset Configurations

For reproducing the ExpertSystems comparative study datasets:

**Sine Dataset (Abrupt Drift):**

```toml
[dataset]
name = "expertsystems_sine"
type = "synthetic" 
source = "capymoa"
generator = "SineGenerator"

[generator_config]
n_instances = 50000
classification_function = 1
has_noise = false
random_seed = 42

[drift_config]
drift_points = [10000, 25000, 40000]  # Multiple concept changes
drift_types = ["concept", "concept", "concept"]
drift_patterns = ["abrupt", "abrupt", "abrupt"]
concept_reversal = true  # Classification reverses after each drift
```

**Hyperplane Dataset (Continuous Gradual Drift):**

```toml
[dataset]
name = "expertsystems_hyperplane"
type = "synthetic"
source = "capymoa" 
generator = "HyperplaneGenerator"

[generator_config]
n_instances = 100000
n_dimensions = 10
n_drifting_dimensions = 10
noise_percentage = 0.05
random_seed = 42

[drift_config]
drift_types = ["concept"]  # Single continuous drift
drift_patterns = ["continuous_gradual"]
rotation_speed = 0.001  # Hyp(0.001) from paper
continuous_drift = true  # No discrete drift points, ongoing change
```

**STAGGER Dataset (Abrupt Drift):**

```toml
[dataset]
name = "expertsystems_stagger"
type = "synthetic"
source = "capymoa"
generator = "STAGGERGenerator"

[generator_config] 
n_instances = 50000
random_seed = 42

[drift_config]
drift_points = [10000, 25000]  # Boolean function changes
drift_types = ["concept", "concept"]
drift_patterns = ["abrupt", "abrupt"]
concept_functions = ["function_1", "function_2", "function_3"]  # Three STAGGER functions
```

**Mixed Dataset (Intermittent Gradual Drift):**

```toml
[dataset]
name = "expertsystems_mixed"
type = "synthetic"
source = "capymoa"
generator = "MixedGenerator"

[generator_config]
n_instances = 60000
boolean_attributes = 2
numerical_attributes = 2  
random_seed = 42

[drift_config]
drift_points = [15000, 35000]
drift_types = ["concept", "concept"] 
drift_patterns = ["intermittent_gradual", "intermittent_gradual"]
transition_durations = [1000, 1000]  # Mixed(1000) stable periods
verification_conditions = 3  # Three verification conditions
stable_periods = true  # Stable concepts between gradual transitions
```

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
create_dataset(config_path: str) -> DriftDataset
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
├── factory.py           # create_dataset() entry point
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
