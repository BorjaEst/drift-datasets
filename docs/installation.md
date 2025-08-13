# Installation Guide

This guide covers system setup, prerequisites, and installation methods for drift-datasets.

## System Requirements

### Operating Systems

- **Linux**: Full support (tested on Ubuntu 20.04+)
- **macOS**: Full support (tested on macOS 11+)
- **Windows**: Partial support (CapyMOA requires manual Java setup)

### Python Versions

- **Supported**: Python 3.10, 3.11, 3.12
- **Recommended**: Python 3.11+ for best performance
- **Minimum**: Python 3.10

### Hardware Requirements

| Dataset Size | RAM Required | Storage | Processing Time |
|--------------|--------------|---------|-----------------|
| 10K samples | 100MB | 10MB | <5 seconds |
| 100K samples | 1GB | 100MB | <30 seconds |
| 1M samples | 8GB | 1GB | <5 minutes |

## Prerequisites

### Java Runtime Environment (Required for CapyMOA)

CapyMOA requires Java Runtime Environment for synthetic dataset generation.

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install default-jre

# Verify installation
java -version
```

#### macOS

```bash
# Using Homebrew
brew install openjdk

# Or using macOS system Java
# Download from https://www.oracle.com/java/technologies/downloads/

# Verify installation
java -version
```

#### Windows

1. Download Java from [Oracle](https://www.oracle.com/java/technologies/downloads/) or [OpenJDK](https://openjdk.org/)
2. Install following the installer wizard
3. Add Java to your PATH environment variable
4. Verify in Command Prompt: `java -version`

#### Expected Java Output

```bash
$ java -version
openjdk version "11.0.20" 2023-07-18
OpenJDK Runtime Environment (build 11.0.20+8-Ubuntu-1ubuntu120.04)
OpenJDK 64-Bit Server VM (build 11.0.20+8-Ubuntu-1ubuntu120.04, mixed mode, sharing)
```

### Git (Optional, for development)

```bash
# Ubuntu/Debian
sudo apt-get install git

# macOS
brew install git

# Windows
# Download from https://git-scm.com/downloads
```

## Installation Methods

### Method 1: PyPI Installation (Recommended)

Install the latest stable version from PyPI:

```bash
pip install drift-datasets
```

#### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install drift-datasets
pip install drift-datasets
```

#### Upgrade Existing Installation

```bash
pip install --upgrade drift-datasets
```

### Method 2: Development Installation

Install from source for development or latest features:

```bash
# Clone repository
git clone https://github.com/BorjaEst/drift-datasets.git
cd drift-datasets

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Method 3: Specific Version Installation

```bash
# Install specific version
pip install drift-datasets==0.1.0

# Install pre-release version
pip install --pre drift-datasets
```

## Dependency Installation

### Core Dependencies

Core dependencies are automatically installed:

- **pydantic v2**: All data validation (never manual validation)
- **rich**: All console output (never print())
- **rich-click**: CLI applications  
- **pytest**: Testing
- **CapyMOA** (≥0.10.0): Synthetic dataset generation
- **NumPy** (≥2.2.0): Numerical computations
- **pandas** (≥2.3.0): Data manipulation
- **SciPy** (≥1.15.0): Scientific computing

### Development Dependencies

For contributors and developers:

```bash
pip install -e ".[dev]"
```

This installs additional packages:

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **isort**: Import sorting
- **ruff**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

### Optional Dependencies

#### Documentation Dependencies

```bash
pip install -e ".[docs]"
```

#### Benchmark Dependencies

```bash
pip install -e ".[benchmark]"
```

## Verification

### Basic Installation Check

```python
import drift_datasets

# Check version
print(f"drift-datasets version: {drift_datasets.__version__}")

# Check dependencies
import capymoa
import numpy as np
import pandas as pd
import scipy

print("All core dependencies loaded successfully!")
```

### Java Integration Check

```python
import drift_datasets

# Test CapyMOA integration (requires Java)
try:
    dataset = drift_datasets.create_dataset({
        "dataset": {"name": "test", "type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
        "metadata": {"dimension": "multivariate", "labeling": "supervised", "n_classes": 2},
        "features": [
            {"name": "x", "type": "continuous", "role": "feature"},
            {"name": "y", "type": "continuous", "role": "feature"},
            {"name": "class", "type": "categorical", "role": "target"}
        ],
        "generator_config": {"n_instances": 100, "classification_function": 1, "random_seed": 42}
    })
    print(f"Java integration successful! Generated dataset with {len(dataset.X)} samples")
except Exception as e:
    print(f"Java integration failed: {e}")
```

### Full System Test

```python
# Create test configuration file
config = """
[dataset]
name = "installation_test"
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
"""

# Save configuration
with open("test_config.toml", "w") as f:
    f.write(config)

# Test dataset generation
import drift_datasets
dataset = drift_datasets.create_dataset("test_config.toml")

# Validate results
assert dataset.X.shape == (1000, 2), f"Expected (1000, 2), got {dataset.X.shape}"
assert len(dataset.y) == 1000, f"Expected 1000 targets, got {len(dataset.y)}"
assert dataset.drift_metadata.drift_points == [500], f"Expected drift at 500, got {dataset.drift_metadata.drift_points}"

print("✅ Full system test passed!")

# Cleanup
import os
os.remove("test_config.toml")
```

## Troubleshooting

### Common Issues

#### Java Not Found

**Error**: `RuntimeError: Java Runtime Environment not found`

**Solution**:

1. Install Java Runtime Environment (see Prerequisites section)
2. Verify `java -version` works in terminal
3. Restart your Python environment
4. Check JAVA_HOME environment variable if needed

#### Import Errors

**Error**: `ImportError: No module named 'drift_datasets'`

**Solution**:

```bash
# Check if installed
pip list | grep drift-datasets

# If not found, install
pip install drift-datasets

# Check virtual environment activation
which python
which pip
```

#### Version Conflicts

**Error**: `pkg_resources.VersionConflict: drift-datasets X.Y.Z conflicts with requirement drift-datasets>=A.B.C`

**Solution**:

```bash
# Upgrade to latest version
pip install --upgrade drift-datasets

# Or install specific version
pip install drift-datasets==A.B.C

# Check for dependency conflicts
pip check
```

#### CapyMOA Integration Issues

**Error**: CapyMOA import or generation failures

**Solution**:

1. Verify Java installation and version (Java 11+ recommended)
2. Check CapyMOA version: `pip show capymoa`
3. Update CapyMOA: `pip install --upgrade capymoa`
4. Test CapyMOA directly:

```python
from capymoa.stream import SineGenerator
generator = SineGenerator()
```

#### Memory Issues

**Error**: `MemoryError` with large datasets

**Solution**:

1. Reduce dataset size in configuration: `n_instances = 10000`
2. Use 64-bit Python if on 32-bit system
3. Monitor memory usage: `htop` (Linux/macOS) or Task Manager (Windows)
4. Consider streaming approaches for very large datasets

### Platform-Specific Issues

#### Windows-Specific

**Issue**: CapyMOA not working on Windows

**Solutions**:

1. Ensure Java is properly installed and in PATH
2. Use PowerShell or Command Prompt (not Git Bash)
3. Check Windows Defender/antivirus isn't blocking Java
4. Try installing from conda-forge if pip fails:

```bash
conda install -c conda-forge drift-datasets
```

#### macOS-Specific

**Issue**: Java installation issues on macOS

**Solutions**:

1. Use official Oracle JDK or OpenJDK
2. Check security settings allow Java execution
3. Verify Architecture compatibility (Intel vs Apple Silicon)

#### Linux-Specific

**Issue**: Permission or package manager issues

**Solutions**:

1. Use virtual environments to avoid system conflicts
2. Install Java through system package manager
3. Check for missing system libraries:

```bash
sudo apt-get install python3-dev build-essential
```

## Environment Setup

### IDE Configuration

#### VS Code

Recommended extensions:

- Python
- Pylance
- Python Docstring Generator
- TOML Language Support

Settings.json:

```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

#### PyCharm

1. Configure Python interpreter to virtual environment
2. Enable pytest as test runner
3. Configure Black as code formatter
4. Set TOML file association

### Development Environment

```bash
# Clone repository
git clone https://github.com/BorjaEst/drift-datasets.git
cd drift-datasets

# Create development environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest

# Check code quality
black . && isort . && ruff check . && mypy src/
```

## Next Steps

After successful installation:

1. **Quick Start**: Follow the [Quickstart Tutorial](quickstart.md)
2. **Configuration**: Learn [TOML Configuration](configuration.md)
3. **Examples**: Explore the [examples directory](examples/)
4. **API Reference**: Browse the [API documentation](api-reference.md)

## Getting Help

If you encounter installation issues:

1. **Check logs**: Look for specific error messages
2. **Search issues**: [GitHub Issues](https://github.com/BorjaEst/drift-datasets/issues)
3. **Ask questions**: [GitHub Discussions](https://github.com/BorjaEst/drift-datasets/discussions)
4. **Report bugs**: Create a detailed issue with:
   - Operating system and version
   - Python version
   - Java version
   - Full error traceback
   - Steps to reproduce
