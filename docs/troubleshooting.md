# Troubleshooting Guide

Common issues, solutions, and debugging techniques for drift-datasets.

## Common Installation Issues

### Java Runtime Environment Problems

#### Error: "RuntimeError: Java Runtime Environment not found"

**Symptoms**:

```text
RuntimeError: Java Runtime Environment not found. CapyMOA requires Java to generate synthetic datasets.
```

**Causes**:

- Java not installed
- Java not in system PATH
- Incorrect Java version

**Solutions**:

1. **Install Java Runtime Environment**:

   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install default-jre
   
   # macOS (using Homebrew)
   brew install openjdk
   
   # Windows: Download from https://www.oracle.com/java/technologies/downloads/
   ```

2. **Verify Java Installation**:

   ```bash
   java -version
   ```

   Expected output:

   ```text
   openjdk version "11.0.20" 2023-07-18
   OpenJDK Runtime Environment (build 11.0.20+8-Ubuntu-1ubuntu120.04)
   OpenJDK 64-Bit Server VM (build 11.0.20+8-Ubuntu-1ubuntu120.04, mixed mode, sharing)
   ```

3. **Fix PATH Issues** (if Java installed but not found):

   ```bash
   # Linux/macOS: Add to ~/.bashrc or ~/.zshrc
   export JAVA_HOME=/usr/lib/jvm/default-java  # Adjust path as needed
   export PATH=$JAVA_HOME/bin:$PATH
   
   # Windows: Use System Properties > Environment Variables
   # Set JAVA_HOME to Java installation directory
   # Add %JAVA_HOME%\bin to PATH
   ```

4. **Test Java with Python**:

   ```python
   import subprocess
   from rich.console import Console
   
   console = Console()
   
   try:
       result = subprocess.run(['java', '-version'], 
                             capture_output=True, text=True)
       console.print(f"Java available: {result.returncode == 0}")
       console.print(f"Java version: {result.stderr}")  # Java prints version to stderr
   except FileNotFoundError:
       console.print("[red]Java not found in PATH[/red]")
   ```

#### Error: "UnsupportedClassVersionError" or Java Version Issues

**Symptoms**:

```text
java.lang.UnsupportedClassVersionError: ... has been compiled by a more recent version of the Java Runtime
```

**Causes**:

- Java version too old for CapyMOA requirements
- Multiple Java versions causing conflicts

**Solutions**:

1. **Check Java Version Requirements**:

   ```python
   # CapyMOA requires Java 11 or later
   import subprocess
   from rich.console import Console
   
   console = Console()
   result = subprocess.run(['java', '-version'], capture_output=True, text=True)
   console.print(result.stderr)
   ```

2. **Update Java to Newer Version**:

   ```bash
   # Ubuntu: Install Java 11 or later
   sudo apt-get install openjdk-11-jre
   
   # macOS: Use Homebrew
   brew install openjdk@11
   
   # Set default Java version (Ubuntu)
   sudo update-alternatives --config java
   ```

3. **Resolve Multiple Java Versions**:

   ```bash
   # Linux: List available Java versions
   update-alternatives --list java
   
   # Set specific version
   sudo update-alternatives --set java /usr/lib/jvm/java-11-openjdk-amd64/bin/java
   
   # macOS: Use Java version manager
   export JAVA_HOME=$(/usr/libexec/java_home -v 11)
   ```

### Package Installation Issues

#### Error: "ImportError: No module named 'drift_datasets'"

**Symptoms**:

```python
>>> import drift_datasets
ImportError: No module named 'drift_datasets'
```

**Solutions**:

1. **Verify Installation**:

   ```bash
   pip list | grep drift-datasets
   # Should show: drift-datasets    0.1.0
   ```

2. **Install Package**:

   ```bash
   pip install drift-datasets
   ```

3. **Check Virtual Environment**:

   ```bash
   # Verify you're in the correct environment
   which python
   which pip
   
   # Activate virtual environment if needed
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

4. **Development Installation**:

   ```bash
   # If installing from source
   git clone https://github.com/BorjaEst/drift-datasets.git
   cd drift-datasets
   pip install -e .
   ```

#### Error: Dependency Version Conflicts

**Symptoms**:

```text
ERROR: pip's dependency resolver does not currently have a solution
```

**Solutions**:

1. **Update pip**:

   ```bash
   pip install --upgrade pip
   ```

2. **Create Fresh Environment**:

   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install drift-datasets
   ```

3. **Install with Specific Versions**:

   ```bash
   pip install drift-datasets --no-deps
   pip install pandas>=2.3.0 numpy>=2.2.0 scipy>=1.15.0 capymoa>=0.10.0
   ```

## Configuration Issues

### TOML Configuration Errors

#### Error: "ValueError: Missing required section [dataset]"

**Symptoms**:

```text
ValueError: Missing required section [dataset] in configuration
```

**Causes**:

- Incorrect TOML syntax
- Missing required configuration sections

**Solutions**:

1. **Validate TOML Syntax**:

   ```python
   import tomli
   
   try:
       with open("config.toml", "rb") as f:
           config = tomli.load(f)
       print("TOML syntax is valid")
   except tomli.TOMLDecodeError as e:
       print(f"TOML syntax error: {e}")
   ```

2. **Check Required Sections**:

   ```toml
   # Minimum required configuration
   [dataset]
   name = "test_dataset"
   type = "synthetic"
   source = "capymoa"
   generator = "SineGenerator"
   
   [metadata]
   dimension = "multivariate"
   labeling = "supervised"
   n_classes = 2
   
   [[features]]
   name = "feature1"
   type = "continuous"
   role = "feature"
   
   [[features]]
   name = "target"
   type = "categorical"
   role = "target"
   
   [generator_config]
   n_instances = 1000
   random_seed = 42
   ```

3. **Debug Configuration**:

   ```python
   import drift_datasets
   
   try:
       dataset = drift_datasets.create_dataset("config.toml")
   except ValueError as e:
       print(f"Configuration error: {e}")
       # Fix the specific issue mentioned in the error
   ```

#### Error: "ValueError: dataset.type must be one of ['synthetic', 'real_world', 'mixed']"

**Symptoms**:

```text
ValueError: dataset.type must be one of ['synthetic', 'real_world', 'mixed'], got 'invalid_type'
```

**Solution**:

Fix the dataset type in configuration:

```toml
[dataset]
type = "synthetic"  # Must be: synthetic, real_world, or mixed
```

#### Error: Feature Configuration Issues

**Common Feature Errors**:

1. **Missing target feature for supervised learning**:

   ```text
   ValueError: Supervised datasets require at least one feature with role='target'
   ```

   **Solution**:

   ```toml
   [[features]]
   name = "target_column"
   type = "categorical"  # or "continuous" for regression
   role = "target"
   ```

2. **Invalid feature type or role**:

   ```text
   ValueError: feature.type must be one of ['continuous', 'categorical', 'mixed']
   ```

   **Solution**:

   ```toml
   [[features]]
   name = "my_feature"
   type = "continuous"  # Must be: continuous, categorical, or mixed
   role = "feature"     # Must be: feature, target, timestamp, identifier, metadata, exclude
   ```

### Generator Configuration Issues

#### Error: "ValueError: SineGenerator requires classification_function parameter"

**Symptoms**:

```text
ValueError: SineGenerator requires classification_function parameter
```

**Solution**:

Add required generator parameters:

```toml
[generator_config]
n_instances = 1000
classification_function = 1  # Required for SineGenerator (0-3)
has_noise = false
random_seed = 42
```

#### Error: Invalid Generator Parameters

**Symptoms**:

```text
ValueError: Invalid parameter 'invalid_param' for HyperplaneGenerator
```

**Solutions**:

1. **Check Generator Documentation**:

   ```python
   # Each generator has specific required parameters
   generators_params = {
       "SineGenerator": ["n_instances", "classification_function"],
       "HyperplaneGenerator": ["n_instances", "n_dimensions"],
       "STAGGERGenerator": ["n_instances", "classification_function"],
       # ... etc
   }
   ```

2. **Use Valid Parameters**:

   ```toml
   # HyperplaneGenerator example
   [generator_config]
   n_instances = 10000
   n_dimensions = 5
   n_drifting_dimensions = 3
   noise_percentage = 0.05
   random_seed = 42
   ```

## Runtime Issues

### Dataset Generation Failures

#### Error: "ConnectionError: Unable to connect to UCI repository"

**Symptoms**:

```text
ConnectionError: Unable to connect to UCI repository. Check internet connection.
```

**Causes**:

- No internet connection
- UCI API temporarily unavailable
- Firewall blocking requests

**Solutions**:

1. **Check Internet Connection**:

   ```python
   import requests
   
   try:
       response = requests.get("https://httpbin.org/get", timeout=10)
       print("Internet connection OK")
   except requests.exceptions.RequestException as e:
       print(f"Connection issue: {e}")
   ```

2. **Test UCI API Directly**:

   ```python
   from ucimlrepo import fetch_ucirepo
   
   try:
       # Try to fetch a small dataset
       dataset = fetch_ucirepo(id=53)  # Iris dataset
       print("UCI API accessible")
   except Exception as e:
       print(f"UCI API error: {e}")
   ```

3. **Use Cached Data**:

   ```python
   # Configure cache directory for offline use
   dataset = drift_datasets.create_dataset(
       config,
       cache_dir="./uci_cache"  # Downloads cached here
   )
   ```

4. **Switch to Synthetic Data**:

   ```toml
   # Temporarily use synthetic data instead of UCI
   [dataset]
   type = "synthetic"  # Instead of "real_world"
   source = "capymoa"
   generator = "SineGenerator"
   ```

#### Error: "RuntimeError: CapyMOA generation failed"

**Symptoms**:

```text
RuntimeError: CapyMOA generation failed: Java exception occurred
```

**Causes**:

- Invalid generator parameters
- Java memory issues
- CapyMOA version incompatibility

**Solutions**:

1. **Validate Generator Parameters**:

   ```python
   # Test with minimal parameters first
   config = {
       "dataset": {"type": "synthetic", "generator": "SineGenerator"},
       "generator_config": {
           "n_instances": 100,  # Start small
           "classification_function": 1,
           "random_seed": 42
       }
   }
   ```

2. **Check Java Memory**:

   ```bash
   # Increase Java heap size if needed
   export JAVA_OPTS="-Xmx4g"  # 4GB heap
   ```

3. **Update CapyMOA**:

   ```bash
   pip install --upgrade capymoa
   ```

4. **Debug with Direct CapyMOA**:

   ```python
   # Test CapyMOA directly
   from capymoa.stream.generator import SineGenerator
   
   try:
       generator = SineGenerator(
           classification_function=1,
           random_seed=42
       )
       print("CapyMOA generator created successfully")
   except Exception as e:
       print(f"CapyMOA error: {e}")
   ```

### Memory and Performance Issues

#### Error: "MemoryError" with Large Datasets

**Symptoms**:

```text
MemoryError: Unable to allocate array with shape (1000000, 100)
```

**Causes**:

- Dataset too large for available memory
- Inefficient memory usage

**Solutions**:

1. **Reduce Dataset Size**:

   ```toml
   [generator_config]
   n_instances = 10000  # Reduce from larger number
   ```

2. **Monitor Memory Usage**:

   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   
   print(f"Memory before: {process.memory_info().rss / 1024**2:.1f} MB")
   dataset = drift_datasets.create_dataset(config)
   print(f"Memory after: {process.memory_info().rss / 1024**2:.1f} MB")
   ```

3. **Use Chunked Processing**:

   ```python
   # Generate dataset in chunks if possible
   def generate_chunked_dataset(config, chunk_size=10000):
       total_instances = config["generator_config"]["n_instances"]
       chunks = []
       
       for i in range(0, total_instances, chunk_size):
           chunk_config = config.copy()
           chunk_config["generator_config"]["n_instances"] = min(chunk_size, total_instances - i)
           
           chunk_dataset = drift_datasets.create_dataset(chunk_config)
           chunks.append((chunk_dataset.X, chunk_dataset.y))
       
       # Combine chunks
       all_X = pd.concat([chunk[0] for chunk in chunks], ignore_index=True)
       all_y = pd.concat([chunk[1] for chunk in chunks], ignore_index=True)
       
       return all_X, all_y
   ```

#### Error: Generation Taking Too Long

**Symptoms**:

- Dataset generation hangs or takes excessive time
- No progress indication

**Solutions**:

1. **Enable Progress Logging**:

   ```python
   import logging
   
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger('drift_datasets')
   logger.setLevel(logging.INFO)
   
   dataset = drift_datasets.create_dataset(config)
   ```

2. **Start with Smaller Dataset**:

   ```python
   # Test with small dataset first
   test_config = config.copy()
   test_config["generator_config"]["n_instances"] = 1000
   
   start_time = time.time()
   test_dataset = drift_datasets.create_dataset(test_config)
   elapsed = time.time() - start_time
   
   print(f"Small dataset took {elapsed:.1f}s")
   # Estimate time for full dataset
   full_size = config["generator_config"]["n_instances"]
   estimated_time = elapsed * (full_size / 1000)
   print(f"Estimated time for full dataset: {estimated_time:.1f}s")
   ```

## Data Issues

### Data Validation Errors

#### Error: "ValidationError: X.shape[0] != y.shape[0]"

**Symptoms**:

```text
ValidationError: Feature matrix and target vector have mismatched lengths
```

**Causes**:

- Internal generation error
- Data corruption during processing

**Solutions**:

1. **Regenerate Dataset**:

   ```python
   # Try regenerating with different random seed
   config["generator_config"]["random_seed"] = 123
   dataset = drift_datasets.create_dataset(config)
   ```

2. **Validate Data Manually**:

   ```python
   dataset = drift_datasets.create_dataset(config)
   
   print(f"X shape: {dataset.X.shape}")
   print(f"y shape: {dataset.y.shape}")
   print(f"Lengths match: {len(dataset.X) == len(dataset.y)}")
   ```

#### Error: Drift Points Out of Bounds

**Symptoms**:

```text
ValueError: drift_points [5000] contain indices >= dataset length (1000)
```

**Solutions**:

Fix drift points in configuration:

```toml
[generator_config]
n_instances = 10000  # Ensure sufficient samples

[drift_config]
drift_points = [5000]  # Must be < n_instances
```

### Feature Access Issues

#### Error: "KeyError: 'feature_name' not found"

**Symptoms**:

```text
KeyError: 'temperature' not found in dataset features
```

**Solutions**:

1. **Check Available Features**:

   ```python
   dataset = drift_datasets.create_dataset(config)
   
   print("Available features:", list(dataset.X.columns))
   print("Feature info:")
   for name in dataset.X.columns:
       info = dataset.get_feature_info(name)
       print(f"  {name}: {info.type} ({info.role})")
   ```

2. **Fix Feature Names in Configuration**:

   ```toml
   [[features]]
   name = "correct_feature_name"  # Match actual feature names
   type = "continuous"
   role = "feature"
   ```

## Debugging Techniques

### Enable Debug Logging

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable drift_datasets debug logging
drift_logger = logging.getLogger('drift_datasets')
drift_logger.setLevel(logging.DEBUG)

# Enable CapyMOA debug logging
capymoa_logger = logging.getLogger('capymoa')
capymoa_logger.setLevel(logging.DEBUG)

# Generate dataset with full logging
dataset = drift_datasets.create_dataset(config)
```

### Configuration Debugging

```python
def debug_config(config):
    """Debug configuration issues."""
    
    print("=== Configuration Debug ===")
    
    # Check required sections
    required_sections = ['dataset', 'metadata', 'features']
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required section: {section}")
        else:
            print(f"✅ Found section: {section}")
    
    # Check dataset type
    if 'dataset' in config:
        dataset_type = config['dataset'].get('type')
        valid_types = ['synthetic', 'real_world', 'mixed']
        if dataset_type in valid_types:
            print(f"✅ Valid dataset type: {dataset_type}")
        else:
            print(f"❌ Invalid dataset type: {dataset_type}")
    
    # Check features
    if 'features' in config:
        features = config['features']
        target_features = [f for f in features if f.get('role') == 'target']
        
        if len(target_features) > 0:
            print(f"✅ Found {len(target_features)} target feature(s)")
        else:
            print("❌ No target features found for supervised learning")
    
    # Check generator config
    dataset_type = config.get('dataset', {}).get('type')
    if dataset_type == 'synthetic':
        if 'generator_config' in config:
            print("✅ Generator config found")
        else:
            print("❌ Missing generator_config for synthetic dataset")

# Usage
debug_config(config)
```

### Step-by-Step Generation

```python
def debug_generation(config):
    """Debug dataset generation step by step."""
    
    try:
        print("Step 1: Validating configuration...")
        drift_datasets.validate_config(config)
        print("✅ Configuration valid")
        
        print("Step 2: Creating generator...")
        # Internal generator creation would be here
        print("✅ Generator created")
        
        print("Step 3: Generating data...")
        dataset = drift_datasets.create_dataset(config, validate=False)
        print(f"✅ Data generated: {dataset.X.shape}")
        
        print("Step 4: Validating result...")
        validation = dataset.validate_drift_metadata()
        if validation['valid']:
            print("✅ Result validation passed")
        else:
            print(f"❌ Validation issues: {validation['issues']}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error at current step: {e}")
        raise

# Usage
dataset = debug_generation(config)
```

### System Information

```python
def system_info():
    """Print system information for troubleshooting."""
    
    import sys
    import platform
    import subprocess
    
    print("=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.architecture()}")
    
    # Java version
    try:
        java_version = subprocess.run(['java', '-version'], 
                                    capture_output=True, text=True)
        print(f"Java: Available (returncode: {java_version.returncode})")
        if java_version.stderr:
            print(f"Java version: {java_version.stderr.split()[2]}")
    except FileNotFoundError:
        print("Java: Not found")
    
    # Package versions
    try:
        import drift_datasets
        print(f"drift-datasets: {drift_datasets.__version__}")
    except:
        print("drift-datasets: Not installed")
    
    try:
        import capymoa
        print(f"capymoa: {capymoa.__version__}")
    except:
        print("capymoa: Not installed")
    
    try:
        import pandas as pd
        print(f"pandas: {pd.__version__}")
    except:
        print("pandas: Not installed")

# Usage
system_info()
```

## Getting Help

If you're still experiencing issues after trying these solutions:

1. **Search existing issues**: [GitHub Issues](https://github.com/BorjaEst/drift-datasets/issues)
2. **Create detailed bug report**: Include system info, full error traceback, and minimal reproducible example
3. **Ask in discussions**: [GitHub Discussions](https://github.com/BorjaEst/drift-datasets/discussions)
4. **Check documentation**: [Documentation Index](index.md)

### Bug Report Template

```markdown
## Bug Report

**System Information**:
- OS: [Ubuntu 20.04, macOS 12, Windows 10, etc.]
- Python: [3.10, 3.11, 3.12]
- Java: [11, 17, etc. or "Not installed"]
- drift-datasets: [0.1.0]

**Configuration**:

```toml
[dataset]
# Your configuration here
```

**Error Message**:

```text
Full error traceback here
```

**Expected Behavior**:
[What you expected to happen]

**Actual Behavior**:  
[What actually happened]

**Steps to Reproduce**:

1. Step 1
2. Step 2  
3. Step 3

**Additional Context**:

```text
Any other relevant information
```

```

This troubleshooting guide should help resolve most common issues with drift-datasets. For complex problems, don't hesitate to reach out for community support.
