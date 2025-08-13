# Configuration Guide

Complete reference for TOML configuration files used to generate drift datasets. This guide covers all configuration sections, parameters, and examples.

## Configuration File Structure

Configuration files use TOML format with the following structure:

```toml
[dataset]        # Required: Basic dataset metadata
[metadata]       # Required: Dataset characteristics
[[features]]     # Required: Per-feature specifications (array)
[generator_config]   # Conditional: Synthetic dataset parameters
[drift_config]   # Optional: Drift pattern specifications
[uci_config]     # Conditional: UCI dataset parameters
[mixed_config]   # Conditional: Mixed dataset combination rules
```

## Required Sections

### [dataset] Section

Basic dataset identification and type specification.

```toml
[dataset]
name = "my_dataset"           # String: Dataset identifier
type = "synthetic"            # Enum: "synthetic", "real_world", "mixed"
source = "capymoa"            # String: Data source ("capymoa", "ucimlrepo")
generator = "SineGenerator"   # String: Generator name (type-specific)
description = "Optional description"  # String: Human-readable description
```

#### Required Fields

- **name**: Unique identifier for the dataset (used in output metadata)
- **type**: Dataset generation method
  - `"synthetic"`: Generated using CapyMOA generators
  - `"real_world"`: Loaded from UCI ML Repository
  - `"mixed"`: Combination of multiple sources

#### Type-Specific Fields

**For type="synthetic"**:

- **source**: Must be `"capymoa"`
- **generator**: CapyMOA generator name (see Generator-Specific Parameters section)

**For type="real_world"**:

- **source**: Must be `"ucimlrepo"`
- Generator not required (loaded from UCI)

**For type="mixed"**:

- **source**: Not required
- **generator**: Not required

### [metadata] Section

Dataset characteristics and learning task specification.

```toml
[metadata]
dimension = "multivariate"    # Enum: "univariate", "multivariate"
labeling = "supervised"       # Enum: "supervised", "unsupervised", "semi-supervised"
n_classes = 2                 # Integer: Number of target classes (supervised only)
task_type = "classification"  # Enum: "classification", "regression" (optional)
temporal = true               # Boolean: Whether data has temporal ordering (optional)
```

#### Metadata Required Fields

- **dimension**: Data dimensionality
  - `"univariate"`: Single feature
  - `"multivariate"`: Multiple features
- **labeling**: Learning paradigm
  - `"supervised"`: Has labeled targets
  - `"unsupervised"`: No targets
  - `"semi-supervised"`: Partially labeled

#### Conditional Fields

- **n_classes**: Required for supervised classification tasks
- **task_type**: Inferred from features if not specified
- **temporal**: Affects splitting and drift interpretation

### [[features]] Section

Feature-level metadata specification. This is an array of feature objects.

```toml
[[features]]
name = "temperature"          # String: Feature name
type = "continuous"           # Enum: "continuous", "categorical", "mixed"
role = "feature"              # Enum: "feature", "target", "timestamp", "identifier", "metadata", "exclude"
description = "Temperature in Celsius"  # String: Human-readable description (optional)
missing_values = false        # Boolean: Whether feature can have missing values (optional)
unique_values = 50           # Integer: Approximate number of unique values (optional)

[[features]]
name = "season"
type = "categorical"
role = "feature"
description = "Season of the year"
unique_values = 4

[[features]]
name = "target_class"
type = "categorical"
role = "target"
description = "Classification target"
```

#### Required Fields per Feature

- **name**: Column name in resulting dataset
- **type**: Data type classification
  - `"continuous"`: Numerical data with meaningful distances
  - `"categorical"`: Discrete categories or labels
  - `"mixed"`: Combination of continuous/categorical aspects
- **role**: Purpose in modeling workflow
  - `"feature"`: Used for modeling and drift detection
  - `"target"`: Prediction target (excluded from feature matrix)
  - `"timestamp"`: Temporal information (excluded from drift detection)
  - `"identifier"`: Unique IDs (excluded from modeling)
  - `"metadata"`: Descriptive information (excluded from modeling)
  - `"exclude"`: Explicitly excluded from all analysis

#### Feature Role Impact

| Role | In X Matrix | In Drift Detection | In get_features_by_role() |
|------|-------------|-------------------|---------------------------|
| feature | ✅ | ✅ | ✅ |
| target | ❌ | ❌ | ✅ |
| timestamp | ✅ | ❌ | ✅ |
| identifier | ✅ | ❌ | ✅ |
| metadata | ✅ | ❌ | ✅ |
| exclude | ❌ | ❌ | ✅ |

#### Validation Rules

- At least one feature with `role="target"` for supervised datasets
- Feature names must be unique within the dataset
- Type and role combinations must be logical (e.g., targets shouldn't be timestamps)

## Generator-Specific Sections

### [generator_config] Section

Parameters for synthetic dataset generation. Required for `type="synthetic"`.

#### Common Parameters (All Generators)

```toml
[generator_config]
n_instances = 10000          # Integer: Number of samples to generate
random_seed = 42             # Integer: Random seed for reproducibility
noise_percentage = 0.0       # Float: Noise level (0.0-1.0)
```

#### Generator-Specific Parameters

**SineGenerator**:

```toml
[generator_config]
n_instances = 1000
classification_function = 1   # Integer: Which sine function (0-3)
has_noise = false            # Boolean: Add noise to data
balance_classes = true       # Boolean: Balance class distribution
random_seed = 42
```

**HyperplaneGenerator**:

```toml
[generator_config]
n_instances = 10000
n_dimensions = 5             # Integer: Total feature dimensions
n_drifting_dimensions = 3    # Integer: Features affected by drift
noise_percentage = 0.05      # Float: Percentage of noise
random_seed = 42
```

**STAGGERGenerator**:

```toml
[generator_config]
n_instances = 5000
classification_function = 1   # Integer: Which STAGGER concept (1-3)
balance_classes = true       # Boolean: Balance class distribution
random_seed = 42
```

**SEAGenerator**:

```toml
[generator_config]
n_instances = 8000
classification_function = 1   # Integer: Which SEA function (1-4)
noise_percentage = 0.1       # Float: Noise level
balance_classes = true       # Boolean: Balance class distribution
random_seed = 42
```

**AgrawalGenerator**:

```toml
[generator_config]
n_instances = 12000
classification_function = 1   # Integer: Which Agrawal function (0-9)
balance_classes = true       # Boolean: Balance class distribution
perturbation = 0.05          # Float: Perturbation level
random_seed = 42
```

**RandomTreeGenerator**:

```toml
[generator_config]
n_instances = 15000
n_classes = 3                # Integer: Number of target classes
n_features_tree = 10         # Integer: Features in decision tree
n_features_nominal = 5       # Integer: Nominal features
max_tree_depth = 5           # Integer: Maximum tree depth
leaf_fraction = 0.15         # Float: Fraction of leaves
random_seed = 42
```

**LEDGenerator**:

```toml
[generator_config]
n_instances = 6000
n_irrelevant_attributes = 7  # Integer: Irrelevant features to add
has_noise = true             # Boolean: Add noise to LED digits
random_seed = 42
```

### [uci_config] Section

Parameters for real-world dataset loading. Required for `type="real_world"`.

```toml
[uci_config]
dataset_id = 321             # Integer: UCI dataset ID
data_home = "./data"         # String: Local cache directory (optional)
fetch_original = false       # Boolean: Fetch original vs preprocessed (optional)
return_X_y = false          # Boolean: Return tuple vs Bunch object (optional)
as_frame = true             # Boolean: Return pandas vs numpy (optional)
```

#### UCI Required Fields

- **dataset_id**: UCI ML Repository dataset identifier

#### Optional Fields

- **data_home**: Directory for caching downloaded datasets
- **fetch_original**: Whether to fetch original format
- **return_X_y**: Format of returned data
- **as_frame**: Whether to use pandas DataFrames

#### Popular UCI Dataset IDs

| Dataset | ID | Samples | Features | Classes | Task |
|---------|----|---------:|----------:|---------:|------|
| Electricity | 321 | 45,312 | 8 | 2 | Classification |
| Forest Cover | 31 | 581,012 | 54 | 7 | Classification |
| Poker Hand | 158 | 1,025,010 | 10 | 10 | Classification |
| Census Income | 20 | 48,842 | 14 | 2 | Classification |
| Nursery | 76 | 12,960 | 8 | 5 | Classification |

### [mixed_config] Section

Parameters for combining multiple data sources. Required for `type="mixed"`.

```toml
[mixed_config]
combination_type = "sequential"  # Enum: "sequential", "interleaved", "hierarchical"
weights = [0.6, 0.4]            # Array: Relative weights for each component
transition_points = [5000]       # Array: Sample indices where sources change
blend_regions = [100]           # Array: Number of samples for smooth transitions
random_seed = 42                # Integer: Random seed for deterministic mixing

# Component datasets (array of dataset configurations)
[[mixed_config.components]]
type = "synthetic"
source = "capymoa"
generator = "SineGenerator"
weight = 0.6
# ... (full generator config)

[[mixed_config.components]]
type = "real_world"
source = "ucimlrepo"
dataset_id = 321
weight = 0.4
# ... (full UCI config)
```

## Drift Configuration

### [drift_config] Section

Drift pattern specification. Optional section - if omitted, no drift is injected.

#### Basic Drift Parameters

```toml
[drift_config]
drift_points = [1000, 5000]     # Array: Sample indices where drift occurs
drift_types = ["concept", "covariate"]  # Array: Type of each drift
drift_patterns = ["abrupt", "gradual"]  # Array: Pattern of each drift
```

#### Research-Friendly Parameters

```toml
[drift_config]
# Basic drift specification
drift_points = [2000, 6000, 10000]
drift_types = ["concept", "covariate", "prior"]
drift_patterns = ["abrupt", "gradual", "abrupt"]

# Advanced drift control
transition_durations = [0, 500, 0]      # Array: How long each transition takes
drift_intensities = ["complete", "moderate", "mild"]  # Array: Magnitude of change
affected_features = [[0, 1], [1, 2], [0, 1, 2]]     # Array: Features affected by each drift

# Special drift behaviors
concept_reversal = true          # Boolean: Classification reversal after drift
stable_periods = true           # Boolean: Stable concepts between transitions
continuous_drift = false        # Boolean: Ongoing change vs discrete points

# Generator-specific parameters
rotation_speed = 0.001          # Float: Hyperplane rotation speed (Hyperplane)
```

#### Drift Types

- **concept**: Changes in conditional distribution P(y|X) - relationship between features and target
- **covariate**: Changes in input distribution P(X) - feature marginal distributions
- **prior**: Changes in class distribution P(y) - target class frequencies
- **none**: No drift - stable concept

#### Drift Patterns

- **abrupt**: Instantaneous change at drift point
- **gradual**: Smooth transition over specified duration
- **continuous_gradual**: Ongoing change with no stable periods (e.g., rotating hyperplane)
- **intermittent_gradual**: Gradual changes with stable periods between
- **recurring**: Periodic return to previous concepts
- **incremental**: Continuous small changes

#### Advanced Parameters

**Transition Durations**:

- Research-friendly parameter specifying how long drift takes
- `0` = instant change, `500` = change over 500 samples
- Automatically translated to CapyMOA width parameters

**Drift Intensities**:

- `"complete"`: Full concept change
- `"moderate"`: Partial concept change  
- `"mild"`: Subtle concept change
- Automatically translated to CapyMOA alpha parameters

**Affected Features**:

- Specify which feature indices are impacted by each drift
- `[[0, 1], [1, 2]]` = first drift affects features 0,1; second affects 1,2
- `"all"` = all features affected by all drifts

#### CapyMOA Compatibility Parameters

For advanced users who need direct control over CapyMOA parameters:

```toml
[drift_config]
# Direct CapyMOA parameters (bypasses research parameter translation)
drift_widths = [0, 500]        # Array: CapyMOA width parameters
drift_alphas = [1.0, 0.3]      # Array: CapyMOA alpha parameters
```

## Complete Examples

### Example 1: Simple Synthetic Dataset

```toml
[dataset]
name = "simple_sine"
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
n_instances = 5000
classification_function = 1
has_noise = false
random_seed = 42

[drift_config]
drift_points = [2500]
drift_types = ["concept"]
drift_patterns = ["abrupt"]
```

### Example 2: Multi-Drift Hyperplane Dataset

```toml
[dataset]
name = "hyperplane_multi_drift"
type = "synthetic"
source = "capymoa"
generator = "HyperplaneGenerator"
description = "Rotating hyperplane with multiple drift types"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2
temporal = true

[[features]]
name = "dim_1"
type = "continuous"
role = "feature"
description = "First hyperplane dimension"

[[features]]
name = "dim_2"
type = "continuous"
role = "feature"
description = "Second hyperplane dimension"

[[features]]
name = "dim_3"
type = "continuous"
role = "feature"
description = "Third hyperplane dimension"

[[features]]
name = "dim_4"
type = "continuous"
role = "feature"
description = "Fourth hyperplane dimension"

[[features]]
name = "dim_5"
type = "continuous"
role = "feature"
description = "Fifth hyperplane dimension"

[[features]]
name = "timestamp"
type = "continuous"
role = "timestamp"
description = "Sample generation time"

[[features]]
name = "class"
type = "categorical"
role = "target"
description = "Binary classification target"

[generator_config]
n_instances = 20000
n_dimensions = 5
n_drifting_dimensions = 3
noise_percentage = 0.05
random_seed = 123

[drift_config]
drift_points = [5000, 10000, 15000]
drift_types = ["concept", "covariate", "concept"]
drift_patterns = ["gradual", "abrupt", "continuous_gradual"]
transition_durations = [1000, 0, 500]
drift_intensities = ["moderate", "complete", "mild"]
affected_features = [[0, 1, 2], [2, 3, 4], [0, 2, 4]]
rotation_speed = 0.005
```

### Example 3: Real-World Dataset with Injected Drift

```toml
[dataset]
name = "electricity_with_drift"
type = "real_world"
source = "ucimlrepo"
description = "Electricity dataset with synthetic drift injection"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2
temporal = true

# Features auto-detected from UCI metadata, but can be overridden
[[features]]
name = "date"
type = "mixed"
role = "timestamp"
description = "Date of electricity measurement"

[[features]]
name = "day"
type = "categorical"
role = "feature"
description = "Day of week"

[[features]]
name = "period"
type = "continuous"
role = "feature"
description = "Period within day"

[[features]]
name = "nswprice"
type = "continuous"
role = "feature"
description = "NSW electricity price"

[[features]]
name = "nswdemand"
type = "continuous"
role = "feature"
description = "NSW electricity demand"

[[features]]
name = "vicprice"
type = "continuous"
role = "feature"
description = "VIC electricity price"

[[features]]
name = "vicdemand"
type = "continuous"
role = "feature"
description = "VIC electricity demand"

[[features]]
name = "transfer"
type = "continuous"
role = "feature"
description = "Transfer between states"

[[features]]
name = "class"
type = "categorical"
role = "target"
description = "Price movement direction"

[uci_config]
dataset_id = 321
data_home = "./data/uci_cache"
as_frame = true

[drift_config]
drift_points = [15000, 30000]
drift_types = ["covariate", "concept"]
drift_patterns = ["gradual", "abrupt"]
transition_durations = [2000, 0]
affected_features = [[1, 2, 3], [3, 4, 5, 6, 7]]
```

### Example 4: Mixed Dataset Combination

```toml
[dataset]
name = "mixed_temporal_evolution"
type = "mixed"
description = "Combination of synthetic and real-world data"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

# Mixed datasets inherit feature definitions from components
# But can be explicitly defined for consistency

[[features]]
name = "synthetic_x"
type = "continuous"
role = "feature"

[[features]]
name = "synthetic_y"
type = "continuous"
role = "feature"

[[features]]
name = "real_feature_1"
type = "continuous"
role = "feature"

[[features]]
name = "real_feature_2"
type = "categorical"
role = "feature"

[[features]]
name = "class"
type = "categorical"
role = "target"

[mixed_config]
combination_type = "sequential"
transition_points = [8000]
blend_regions = [500]
random_seed = 42

[[mixed_config.components]]
type = "synthetic"
source = "capymoa"
generator = "SineGenerator"
weight = 0.6

[mixed_config.components.generator_config]
n_instances = 8000
classification_function = 1
random_seed = 42

[[mixed_config.components]]
type = "real_world"
source = "ucimlrepo"
weight = 0.4

[mixed_config.components.uci_config]
dataset_id = 321

[drift_config]
drift_points = [4000, 8000, 12000]
drift_types = ["concept", "covariate", "concept"]
drift_patterns = ["abrupt", "gradual", "abrupt"]
transition_durations = [0, 1000, 0]
```

## Validation and Error Handling

### Configuration Validation

The library validates configurations and provides detailed error messages:

```python
import drift_datasets
from rich.console import Console

console = Console()

try:
    dataset = drift_datasets.create_dataset("config.toml")
except ValueError as e:
    console.print(f"[red]Configuration error:[/red] {e}")
except FileNotFoundError as e:
    console.print(f"[red]File error:[/red] {e}")
```

### Common Validation Errors

**Missing Required Sections**:

```text
ValueError: Missing required section [dataset] in configuration
```

**Invalid Dataset Type**:

```text
ValueError: dataset.type must be one of ['synthetic', 'real_world', 'mixed'], got 'invalid'
```

**Feature Role Validation**:

```text
ValueError: Supervised datasets require at least one feature with role='target'
```

**Drift Configuration Errors**:

```text
ValueError: drift_types length (2) must match drift_points length (3)
```

**Generator Parameter Errors**:

```text
ValueError: SineGenerator requires classification_function parameter
```

### Best Practices

1. **Start simple**: Begin with basic configurations and add complexity gradually
2. **Validate early**: Test configurations with small n_instances first
3. **Use meaningful names**: Choose descriptive dataset and feature names
4. **Document configurations**: Include description fields for clarity
5. **Version control**: Store configurations in git for reproducibility
6. **Consistent seeds**: Use fixed random_seed values for deterministic results

## Parameter Translation Reference

### Research → CapyMOA Translation

| Research Parameter | CapyMOA Parameter | Conversion |
|-------------------|-------------------|------------|
| transition_durations = [500] | width = 500 | Direct mapping |
| drift_intensities = ["moderate"] | alpha = 0.5 | complete=1.0, moderate=0.5, mild=0.1 |
| drift_patterns = ["gradual"] | Enables width parameter | abrupt sets width=0 |
| rotation_speed = 0.001 | Generator-specific | Direct to HyperplaneGenerator |

### ExpertSystems Paper Compatibility

Parameters from the ExpertSystems comparative study:

**Sine Dataset**:

```toml
[generator_config]
classification_function = 1
concept_reversal = true

[drift_config]
drift_points = [10000, 25000, 40000]
drift_patterns = ["abrupt", "abrupt", "abrupt"]
```

**Hyperplane Dataset**:

```toml
[generator_config]
rotation_speed = 0.001  # Hyp(0.001) from paper

[drift_config]
drift_patterns = ["continuous_gradual"]
continuous_drift = true
```

This comprehensive configuration guide covers all aspects of dataset generation through TOML files. For specific use cases, refer to the [examples directory](examples/) and [ExpertSystems compatibility guide](expertsystems.md).
