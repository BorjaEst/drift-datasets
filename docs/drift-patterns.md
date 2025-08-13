# Drift Patterns Guide

Complete guide to concept drift types, patterns, and their implementation in drift-datasets. This covers theoretical foundations, configuration options, and practical examples for research applications.

## Theoretical Foundation

### What is Concept Drift?

Concept drift refers to changes in the statistical properties of data over time that affect the performance of machine learning models. Understanding drift types and patterns is essential for developing robust streaming learning systems.

### Taxonomy of Concept Drift

#### By Source of Change

**Covariate Drift (P(X) changes)**:

- Input feature distributions change
- Relationship between features and target remains stable
- Example: Customer demographics shift, but buying behavior patterns stay the same

**Concept Drift (P(y|X) changes)**:

- Relationship between features and target changes
- Input distributions may remain stable
- Example: Economic conditions change how income affects loan default risk

**Prior Drift (P(y) changes)**:

- Class distribution changes
- Both input features and relationships may stay stable
- Example: Seasonal changes in product demand categories

#### By Temporal Behavior

**Real Drift vs Virtual Drift**:

- **Real drift**: Actual changes in underlying data generating process
- **Virtual drift**: Apparent changes due to sampling effects or external factors

**Recurring vs Novel Concepts**:

- **Recurring**: Return to previously seen concepts
- **Novel**: Introduction of entirely new concepts

## Drift Patterns

### Pattern Classification

drift-datasets supports six main drift patterns, each with specific use cases and research applications.

### 1. Abrupt Drift

**Characteristics**:

- Instantaneous change at specific points
- No transition period between concepts
- Clear, discrete concept boundaries

**Mathematical Model**:

```text
P_t(y|X) = {
  P_old(y|X)  if t < t_drift
  P_new(y|X)  if t >= t_drift
}
```

**Configuration**:

```toml
[drift_config]
drift_points = [5000]
drift_types = ["concept"]
drift_patterns = ["abrupt"]
transition_durations = [0]  # Optional: explicitly specify instant change
```

**Research Applications**:

- Algorithm change-point detection evaluation
- Sudden environmental changes
- System failure scenarios
- Policy or regulatory changes

**Example: STAGGER Dataset**

```toml
[dataset]
name = "stagger_abrupt"
type = "synthetic"
source = "capymoa"
generator = "STAGGERGenerator"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "size"
type = "categorical"
role = "feature"

[[features]]
name = "color"
type = "categorical"
role = "feature"

[[features]]
name = "shape"
type = "categorical"
role = "feature"

[[features]]
name = "class"
type = "categorical"
role = "target"

[generator_config]
n_instances = 6000
classification_function = 1
random_seed = 42

[drift_config]
drift_points = [2000, 4000]
drift_types = ["concept", "concept"]
drift_patterns = ["abrupt", "abrupt"]
concept_reversal = true  # Cycle through STAGGER concepts
```

### 2. Gradual Drift

**Characteristics**:

- Smooth transition over specified duration
- Both old and new concepts present during transition
- Probabilistic mixing of concepts

**Mathematical Model**:

```
P_t(y|X) = (1 - α_t) * P_old(y|X) + α_t * P_new(y|X)

where α_t = (t - t_start) / duration for t_start <= t <= t_end
```

**Configuration**:

```toml
[drift_config]
drift_points = [5000]
drift_types = ["concept"]
drift_patterns = ["gradual"]
transition_durations = [1000]  # 1000 samples for transition
drift_intensities = ["moderate"]  # Controls mixing rate
```

**Research Applications**:

- Gradual environmental changes
- Slow system degradation
- Seasonal transitions
- Adaptive algorithm evaluation

**Example: Gradual Hyperplane Drift**

```toml
[dataset]
name = "hyperplane_gradual"
type = "synthetic"
source = "capymoa"
generator = "HyperplaneGenerator"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "x1"
type = "continuous"
role = "feature"

[[features]]
name = "x2"
type = "continuous"
role = "feature"

[[features]]
name = "x3"
type = "continuous"
role = "feature"

[[features]]
name = "x4"
type = "continuous"
role = "feature"

[[features]]
name = "x5"
type = "continuous"
role = "feature"

[[features]]
name = "class"
type = "categorical"
role = "target"

[generator_config]
n_instances = 15000
n_dimensions = 5
n_drifting_dimensions = 3
noise_percentage = 0.05
random_seed = 42

[drift_config]
drift_points = [5000, 10000]
drift_types = ["concept", "concept"]
drift_patterns = ["gradual", "gradual"]
transition_durations = [2000, 1500]
affected_features = [[0, 1, 2], [2, 3, 4]]
```

### 3. Continuous Gradual Drift

**Characteristics**:

- Ongoing, continuous change without stable periods
- No discrete drift points
- Constant evolution of concepts

**Mathematical Model**:

```
P_t(y|X) = P(y|X, θ_t)
where θ_t changes continuously over time
```

**Configuration**:

```toml
[drift_config]
drift_types = ["concept"]
drift_patterns = ["continuous_gradual"]
continuous_drift = true
rotation_speed = 0.001  # For hyperplane datasets
```

**Research Applications**:

- Rotating hyperplane experiments (ExpertSystems paper)
- Continuously evolving environments
- Non-stationary optimization problems
- Adaptive learning in dynamic environments

**Example: ExpertSystems Hyperplane**

```toml
[dataset]
name = "expertsystems_hyperplane"
type = "synthetic"
source = "capymoa"
generator = "HyperplaneGenerator"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "x1"
type = "continuous"
role = "feature"

[[features]]
name = "x2"
type = "continuous"
role = "feature"

[[features]]
name = "x3"
type = "continuous"
role = "feature"

[[features]]
name = "x4"
type = "continuous"
role = "feature"

[[features]]
name = "x5"
type = "continuous"
role = "feature"

[[features]]
name = "x6"
type = "continuous"
role = "feature"

[[features]]
name = "x7"
type = "continuous"
role = "feature"

[[features]]
name = "x8"
type = "continuous"
role = "feature"

[[features]]
name = "x9"
type = "continuous"
role = "feature"

[[features]]
name = "x10"
type = "continuous"
role = "feature"

[[features]]
name = "class"
type = "categorical"
role = "target"

[generator_config]
n_instances = 100000
n_dimensions = 10
n_drifting_dimensions = 10
noise_percentage = 0.05
random_seed = 42

[drift_config]
drift_types = ["concept"]
drift_patterns = ["continuous_gradual"]
continuous_drift = true
rotation_speed = 0.001  # Hyp(0.001) configuration
```

### 4. Intermittent Gradual Drift

**Characteristics**:

- Gradual transitions followed by stable periods
- Alternating between change and stability phases
- Predictable or unpredictable timing patterns

**Mathematical Model**:

```
P_t(y|X) = {
  P_stable(y|X)                           if t in stable_periods
  (1 - α_t) * P_old(y|X) + α_t * P_new(y|X)  if t in transition_periods
}
```

**Configuration**:

```toml
[drift_config]
drift_points = [3000, 8000, 13000]
drift_types = ["concept", "concept", "concept"]
drift_patterns = ["intermittent_gradual", "intermittent_gradual", "intermittent_gradual"]
transition_durations = [1000, 1000, 1000]
stable_periods = [2000, 2000, 2000]
```

**Research Applications**:

- Mixed datasets from ExpertSystems paper
- Seasonal business patterns
- Economic cycles
- System maintenance cycles

**Example: Mixed Attribute Dataset**

```toml
[dataset]
name = "mixed_intermittent"
type = "synthetic"
source = "capymoa"
generator = "RandomTreeGenerator"  # Use supported generator for mixed attributes

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "boolean_1"
type = "categorical"
role = "feature"

[[features]]
name = "boolean_2"
type = "categorical"
role = "feature"

[[features]]
name = "numeric_1"
type = "continuous"
role = "feature"

[[features]]
name = "numeric_2"
type = "continuous"
role = "feature"

[[features]]
name = "class"
type = "categorical"
role = "target"

[generator_config]
n_instances = 20000
n_boolean_attributes = 2
n_numeric_attributes = 2
verification_conditions = 3
random_seed = 42

[drift_config]
drift_points = [5000, 10000, 15000]
drift_types = ["concept", "concept", "concept"]
drift_patterns = ["intermittent_gradual", "intermittent_gradual", "intermittent_gradual"]
transition_durations = [1000, 1000, 1000]
stable_periods = [1000, 1000, 1000]  # Mixed(1000) configuration
```

### 5. Recurring Drift

**Characteristics**:

- Periodic return to previously seen concepts
- Cyclic pattern of concept changes
- Predictable or seasonal recurrence

**Mathematical Model**:

```
P_t(y|X) = P_concept[k](y|X)
where k = f(t) cycles through concept indices
```

**Configuration**:

```toml
[drift_config]
drift_points = [2000, 4000, 6000, 8000]
drift_types = ["concept", "concept", "concept", "concept"]
drift_patterns = ["recurring", "recurring", "recurring", "recurring"]
concept_cycle = [0, 1, 2, 0]  # Return to concept 0
recurrence_period = 2000
```

**Research Applications**:

- Seasonal datasets
- Daily/weekly/monthly patterns
- Cyclic business processes
- Memory-based adaptive algorithms

**Example: Seasonal Classification**

```toml
[dataset]
name = "seasonal_recurring"
type = "synthetic"
source = "capymoa"
generator = "SineGenerator"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "temperature"
type = "continuous"
role = "feature"
description = "Seasonal temperature"

[[features]]
name = "humidity"
type = "continuous"
role = "feature"
description = "Seasonal humidity"

[[features]]
name = "season"
type = "categorical"
role = "timestamp"
description = "Season identifier"

[[features]]
name = "activity_class"
type = "categorical"
role = "target"
description = "Seasonal activity classification"

[generator_config]
n_instances = 12000
classification_function = 1
has_noise = true
random_seed = 42

[drift_config]
drift_points = [3000, 6000, 9000, 12000]  # Quarterly changes
drift_types = ["concept", "concept", "concept", "concept"]
drift_patterns = ["recurring", "recurring", "recurring", "recurring"]
concept_cycle = ["spring", "summer", "autumn", "winter"]
recurrence_period = 3000
```

### 6. Incremental Drift

**Characteristics**:

- Small, continuous changes over time
- No discrete change points
- Cumulative effect becomes significant over time

**Mathematical Model**:

```
P_t(y|X) = P(y|X, θ_0 + δ * t)
where δ represents small incremental changes
```

**Configuration**:

```toml
[drift_config]
drift_types = ["concept"]
drift_patterns = ["incremental"]
drift_rate = 0.0001  # Small change per sample
cumulative_drift = true
```

**Research Applications**:

- Gradual system aging
- Slow environmental changes
- Long-term trend analysis
- Incremental learning evaluation

**Example: Incremental System Degradation**

```toml
[dataset]
name = "system_degradation"
type = "synthetic"
source = "capymoa"
generator = "RandomTreeGenerator"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 3

[[features]]
name = "sensor_1"
type = "continuous"
role = "feature"

[[features]]
name = "sensor_2"
type = "continuous"
role = "feature"

[[features]]
name = "sensor_3"
type = "continuous"
role = "feature"

[[features]]
name = "system_age"
type = "continuous"
role = "timestamp"

[[features]]
name = "health_status"
type = "categorical"
role = "target"

[generator_config]
n_instances = 50000
n_classes = 3
n_features_tree = 3
max_tree_depth = 5
random_seed = 42

[drift_config]
drift_types = ["concept"]
drift_patterns = ["incremental"]
drift_rate = 0.00005  # Very slow degradation
cumulative_drift = true
noise_injection = 0.01  # Small noise increase over time
```

## Advanced Drift Configuration

### Multi-Type Drift Scenarios

Real-world scenarios often involve multiple types of drift occurring simultaneously or sequentially.

```toml
[drift_config]
drift_points = [5000, 15000, 25000]
drift_types = ["covariate", "concept", "prior"]
drift_patterns = ["gradual", "abrupt", "gradual"]
transition_durations = [1000, 0, 2000]
affected_features = [[0, 1], [1, 2, 3], [3, 4]]
drift_intensities = ["moderate", "complete", "mild"]
```

### Combined Drift with Real and Virtual Components

```toml
[drift_config]
# Real concept drift
drift_points = [8000, 16000]
drift_types = ["concept", "concept"]
drift_patterns = ["gradual", "abrupt"]

# Virtual drift through sampling effects
sampling_drift = true
sample_bias_points = [4000, 12000, 20000]
bias_intensities = [0.1, 0.2, 0.15]
```

### Feature-Specific Drift Control

```toml
[drift_config]
drift_points = [10000]
drift_types = ["concept"]
drift_patterns = ["gradual"]
transition_durations = [2000]

# Fine-grained feature control
affected_features = [[0, 2, 4]]  # Only features 0, 2, 4 drift
feature_drift_rates = [0.8, 1.0, 0.6]  # Different rates per feature
feature_drift_delays = [0, 500, 1000]  # Staggered drift start times
```

## Pattern Comparison and Selection

### Pattern Selection Guidelines

| Pattern | Use Case | Detection Difficulty | Research Focus |
|---------|----------|---------------------|-----------------|
| Abrupt | Sudden changes, system failures | Easy | Change-point detection |
| Gradual | Environmental changes | Moderate | Adaptive windowing |
| Continuous Gradual | Evolving environments | Hard | Online learning |
| Intermittent Gradual | Seasonal/cyclic changes | Moderate | Pattern recognition |
| Recurring | Seasonal data | Easy (if periodic) | Memory-based methods |
| Incremental | Long-term trends | Very Hard | Trend detection |

### Complexity Levels

**Simple (Beginner)**:

- Single drift point
- Abrupt or gradual patterns
- Clear concept boundaries
- Binary classification

**Intermediate (Research)**:

- Multiple drift points
- Mixed drift types
- Feature-specific drift
- Multi-class problems

**Advanced (Expert)**:

- Continuous/incremental patterns
- Combined real/virtual drift
- Recurring with variations
- High-dimensional scenarios

### Performance Characteristics

| Pattern | Memory Requirements | Generation Time | Realism |
|---------|-------------------|-----------------|---------|
| Abrupt | Low | Fast | Medium |
| Gradual | Medium | Medium | High |
| Continuous Gradual | High | Slow | Very High |
| Intermittent Gradual | Medium | Medium | High |
| Recurring | Medium | Medium | Medium |
| Incremental | High | Slow | Very High |

## Validation and Analysis

### Drift Metadata Validation

```python
import drift_datasets

# Load dataset with complex drift
dataset = drift_datasets.create_dataset("complex_drift.toml")

# Validate drift consistency
validation = dataset.validate_drift_metadata()
if not validation['valid']:
    print("Issues found:")
    for issue in validation['issues']:
        print(f"  - {issue}")

# Analyze drift characteristics
segments = dataset.get_concept_segments()
print(f"Number of concept segments: {len(segments)}")
for i, (start, end) in enumerate(segments):
    duration = end - start
    print(f"Segment {i}: samples {start}-{end} (duration: {duration})")
```

### Pattern-Specific Analysis

```python
# Check pattern-specific properties
drift_meta = dataset.drift_metadata

for i, (point, pattern) in enumerate(zip(drift_meta.drift_points, drift_meta.drift_patterns)):
    print(f"Drift {i}: point={point}, pattern={pattern}")
    
    if pattern == "gradual":
        duration = drift_meta.transition_durations[i]
        print(f"  Transition duration: {duration} samples")
        
    elif pattern == "continuous_gradual":
        print(f"  Continuous drift - no discrete segments")
        
    elif pattern == "recurring":
        cycle = drift_meta.concept_cycle[i] if hasattr(drift_meta, 'concept_cycle') else None
        print(f"  Recurring to concept: {cycle}")
```

### Statistical Analysis

```python
import pandas as pd
import numpy as np
from scipy import stats

# Analyze feature distributions across concepts
X = dataset.get_drift_detection_features()
segments = dataset.get_concept_segments()

for feature in X.columns:
    print(f"\nFeature: {feature}")
    
    distributions = []
    for i, (start, end) in enumerate(segments):
        segment_data = X[feature].iloc[start:end]
        distributions.append(segment_data)
        
        mean = segment_data.mean()
        std = segment_data.std()
        print(f"  Segment {i}: mean={mean:.3f}, std={std:.3f}")
    
    # Statistical tests between segments
    if len(distributions) >= 2:
        stat, p_value = stats.ks_2samp(distributions[0], distributions[-1])
        print(f"  KS test (first vs last): statistic={stat:.3f}, p-value={p_value:.3f}")
```

## Research Applications

### ExpertSystems Paper Reproduction

The ExpertSystems comparative study used specific drift configurations:

**Sine Dataset (Abrupt Concept Drift)**:

- Multiple abrupt concept changes
- Classification function reversal
- Clear change-point evaluation

**Hyperplane Dataset (Continuous Gradual Drift)**:

- Rotating hyperplane with Hyp(0.001) parameter
- No discrete drift points
- Continuous adaptation evaluation

**Mixed Dataset (Intermittent Gradual Drift)**:

- Gradual transitions with stable periods
- Mixed(200) and Mixed(1000) configurations
- Pattern recognition evaluation

### Modern Research Extensions

**Multi-Source Drift**:

```toml
[dataset]
name = "multi_source_drift"
type = "mixed"

[mixed_config]
combination_type = "hierarchical"

# Different drift patterns per source
[[mixed_config.components]]
type = "synthetic"
drift_pattern = "continuous_gradual"

[[mixed_config.components]]
type = "real_world"
drift_pattern = "intermittent_gradual"
```

**High-Dimensional Drift**:

```toml
[generator_config]
n_dimensions = 100
n_drifting_dimensions = 20

[drift_config]
affected_features = [[0, 5, 10, 15], [20, 25, 30, 35]]  # Sparse drift
feature_interaction_drift = true
```

This comprehensive guide covers all drift patterns supported by drift-datasets. For implementation details, see the [Configuration Guide](configuration.md), and for research applications, see the [ExpertSystems Compatibility Guide](expertsystems.md).
