# ExpertSystems Paper Compatibility

Complete guide for reproducing datasets and configurations from the ExpertSystems comparative study on drift detection methods.

## Overview

The ExpertSystems paper presents a comprehensive evaluation of concept drift detection methods using standardized datasets. This guide shows how to reproduce those exact configurations using drift-datasets.

### Paper Reference

**Title**: "A comparative study on concept drift detectors"  
**Authors**: [Authors from the ExpertSystems paper]  
**Journal**: Expert Systems with Applications  
**Year**: [Publication year]  
**DOI**: [Paper DOI]

### ExpertSystems Dataset Overview

The paper evaluates drift detection methods on five main dataset categories:

1. **Sine Dataset** - Abrupt concept drift with classification reversal
2. **Hyperplane Dataset** - Continuous gradual drift with rotation
3. **Mixed Dataset** - Intermittent gradual drift with stable periods
4. **STAGGER Dataset** - Discrete concept changes with attribute interactions
5. **Real-World Datasets** - UCI datasets with natural or injected drift

## Sine Dataset Reproduction

### Paper Description

The Sine dataset generates data using a sine and cosine wave combination with classification reversal after each drift point.

**Original Paper Parameters**:

- Dataset size: 50,000 samples
- Drift points: 10,000, 25,000, 40,000
- Drift type: Abrupt concept drift
- Behavior: Classification function reverses after each drift

### drift-datasets Configuration

#### Complete Sine Configuration

```toml
[dataset]
name = "expertsystems_sine"
type = "synthetic"
source = "capymoa"
generator = "SineGenerator"
description = "Sine dataset from ExpertSystems comparative study"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2
temporal = true

[[features]]
name = "x"
type = "continuous"
role = "feature"
description = "X coordinate from sine function"

[[features]]
name = "y"
type = "continuous"
role = "feature"
description = "Y coordinate from cosine function"

[[features]]
name = "class"
type = "categorical"
role = "target"
description = "Binary classification target"

[generator_config]
n_instances = 50000
classification_function = 1
has_noise = false
balance_classes = true
random_seed = 42

[drift_config]
drift_points = [10000, 25000, 40000]
drift_types = ["concept", "concept", "concept"]
drift_patterns = ["abrupt", "abrupt", "abrupt"]
concept_reversal = true
stable_periods = true
```

#### Python Generation

```python
import drift_datasets as dd
from rich.console import Console

console = Console()

# Generate ExpertSystems Sine dataset
sine_dataset = dd.create_dataset("expertsystems_sine.toml")

# Verify configuration matches paper
console.print(f"Dataset size: {len(sine_dataset.X)} (expected: 50,000)")
console.print(f"Drift points: {sine_dataset.drift_metadata.drift_points}")
console.print(f"Drift types: {sine_dataset.drift_metadata.drift_types}")

# Analyze concept segments
segments = sine_dataset.get_concept_segments()
console.print("Concept segments:")
for i, (start, end) in enumerate(segments):
    duration = end - start
    console.print(f"  Concept {i}: samples {start}-{end} (duration: {duration:,})")
```

#### Expected Output

```text
Dataset size: 50000 (expected: 50,000)
Drift points: [10000, 25000, 40000]
Drift types: ['concept', 'concept', 'concept']
Concept segments:
  Concept 0: samples 0-10000 (duration: 10,000)
  Concept 1: samples 10000-25000 (duration: 15,000)
  Concept 2: samples 25000-40000 (duration: 15,000)
  Concept 3: samples 40000-50000 (duration: 10,000)
```

## Hyperplane Dataset Reproduction

### Hyperplane Dataset Details

Rotating hyperplane in d-dimensional space with continuous gradual drift.

**Original Paper Parameters**:

- Dataset size: 100,000 samples
- Dimensions: 10 (all drifting)
- Rotation speed: Hyp(0.001) slow, Hyp(0.1) fast
- Noise: 5%
- Drift type: Continuous gradual (no stable periods)

### Configuration for drift-datasets

#### Hyperplane Slow Rotation - Hyp(0.001)

```toml
[dataset]
name = "expertsystems_hyperplane_slow"
type = "synthetic"
source = "capymoa"
generator = "HyperplaneGenerator"
description = "Slow rotating hyperplane Hyp(0.001) from ExpertSystems study"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2
temporal = true

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
stable_periods = false  # No stable periods - continuous change
```

#### Hyperplane Fast Rotation - Hyp(0.1)

```toml
[dataset]
name = "expertsystems_hyperplane_fast"
type = "synthetic"
source = "capymoa"
generator = "HyperplaneGenerator"
description = "Fast rotating hyperplane Hyp(0.1) from ExpertSystems study"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

# Same features as slow rotation...

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
rotation_speed = 0.1  # Hyp(0.1) configuration
stable_periods = false
```

#### Python Generation and Analysis

```python
from rich.console import Console

console = Console()

# Generate both hyperplane variants
hyp_slow = dd.create_dataset("expertsystems_hyperplane_slow.toml")
hyp_fast = dd.create_dataset("expertsystems_hyperplane_fast.toml")

# Verify configurations
console.print("[bold]Slow Hyperplane Hyp(0.001):[/bold]")
console.print(f"  Size: {len(hyp_slow.X):,} samples")
console.print(f"  Features: {hyp_slow.X.shape[1]} dimensions")
console.print(f"  Continuous drift: {hyp_slow.drift_metadata.continuous_drift}")

console.print("\n[bold]Fast Hyperplane Hyp(0.1):[/bold]")
console.print(f"  Size: {len(hyp_fast.X):,} samples") 
console.print(f"  Features: {hyp_fast.X.shape[1]} dimensions")

# Analyze drift behavior - continuous drift has no discrete segments
console.print(f"\nSlow rotation segments: {hyp_slow.get_concept_segments()}")
console.print(f"Fast rotation segments: {hyp_fast.get_concept_segments()}")
```

## Mixed Dataset Reproduction

### Mixed Dataset Details

Mixed boolean and numerical attributes with gradual probability changes between stable periods.

**Original Paper Parameters**:

- Attributes: 2 boolean + 2 numerical
- Verification conditions: Three boolean conditions
- Stable periods: Mixed(200), Mixed(1000) configurations
- Drift type: Intermittent gradual with stable concept segments

### Configuration for drift-datasets

#### Mixed Dataset - Mixed(200)

```toml
[dataset]
name = "expertsystems_mixed_200"
type = "synthetic"
source = "capymoa"
generator = "RandomTreeGenerator"
description = "Mixed attributes Mixed(200) from ExpertSystems study"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "boolean_1"
type = "categorical"
role = "feature"
description = "First boolean attribute"

[[features]]
name = "boolean_2"
type = "categorical"
role = "feature"
description = "Second boolean attribute"

[[features]]
name = "numeric_1"
type = "continuous"
role = "feature"
description = "First numerical attribute"

[[features]]
name = "numeric_2"
type = "continuous"
role = "feature"
description = "Second numerical attribute"

[[features]]
name = "class"
type = "categorical"
role = "target"

[generator_config]
n_instances = 40000
n_boolean_attributes = 2
n_numeric_attributes = 2
verification_conditions = 3
random_seed = 42

[drift_config]
drift_points = [10000, 20000, 30000]
drift_types = ["concept", "concept", "concept"]
drift_patterns = ["intermittent_gradual", "intermittent_gradual", "intermittent_gradual"]
transition_durations = [200, 200, 200]  # Mixed(200) parameter
stable_periods = [5000, 5000, 5000]
```

#### Mixed Dataset - Mixed(1000)

```toml
[dataset]
name = "expertsystems_mixed_1000"
type = "synthetic"
source = "capymoa"
generator = "RandomTreeGenerator"
description = "Mixed attributes Mixed(1000) from ExpertSystems study"

# Same metadata and features as Mixed(200)...

[generator_config]
n_instances = 40000
n_boolean_attributes = 2
n_numeric_attributes = 2
verification_conditions = 3
random_seed = 42

[drift_config]
drift_points = [10000, 20000, 30000]
drift_types = ["concept", "concept", "concept"]
drift_patterns = ["intermittent_gradual", "intermittent_gradual", "intermittent_gradual"]
transition_durations = [1000, 1000, 1000]  # Mixed(1000) parameter
stable_periods = [5000, 5000, 5000]
```

## STAGGER Dataset Reproduction

### STAGGER Dataset Details

Three boolean attributes with different classification concepts.

**Original Paper Parameters**:

- Attributes: size, color, shape (all boolean/categorical)
- Concepts: Three different STAGGER concepts
- Drift points: Multiple abrupt changes
- Classification rules change at each drift point

### Configuration for STAGGER Dataset

```toml
[dataset]
name = "expertsystems_stagger"
type = "synthetic"
source = "capymoa"
generator = "STAGGERGenerator"
description = "STAGGER concepts from ExpertSystems study"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "size"
type = "categorical"
role = "feature"
description = "Object size (small, medium, large)"

[[features]]
name = "color"
type = "categorical"
role = "feature"
description = "Object color (red, green, blue)"

[[features]]
name = "shape"
type = "categorical"
role = "feature"
description = "Object shape (square, circle, triangle)"

[[features]]
name = "class"
type = "categorical"
role = "target"

[generator_config]
n_instances = 30000
classification_function = 1  # Start with STAGGER concept 1
balance_classes = true
random_seed = 42

[drift_config]
drift_points = [10000, 20000]
drift_types = ["concept", "concept"]
drift_patterns = ["abrupt", "abrupt"]
concept_cycle = [1, 2, 3]  # Cycle through STAGGER concepts
```

## Real-World Dataset Integration

### Real-World Dataset Details

UCI datasets with natural drift or synthetic drift injection.

**Common Real-World Datasets in ExpertSystems**:

- Electricity dataset
- Forest Cover dataset  
- Poker Hand dataset

### Real-World Configuration

#### Electricity Dataset with Drift

```toml
[dataset]
name = "expertsystems_electricity"
type = "real_world"
source = "ucimlrepo"
description = "Electricity dataset from ExpertSystems study"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2
temporal = true

# Features auto-detected from UCI metadata
[[features]]
name = "date"
type = "mixed"
role = "timestamp"

[[features]]
name = "day"
type = "categorical"
role = "feature"

[[features]]
name = "period"
type = "continuous"
role = "feature"

[[features]]
name = "nswprice"
type = "continuous"
role = "feature"

[[features]]
name = "nswdemand"
type = "continuous"
role = "feature"

[[features]]
name = "vicprice"
type = "continuous"
role = "feature"

[[features]]
name = "vicdemand"
type = "continuous"
role = "feature"

[[features]]
name = "transfer"
type = "continuous"
role = "feature"

[[features]]
name = "class"
type = "categorical"
role = "target"

[uci_config]
dataset_id = 321
as_frame = true

# Optional: inject additional synthetic drift
[drift_config]
drift_points = [15000, 30000]
drift_types = ["covariate", "concept"]
drift_patterns = ["gradual", "abrupt"]
```

## Complete Reproduction Script

### Python Script for All Datasets

```python
import drift_datasets as dd
import pandas as pd
import matplotlib.pyplot as plt

def reproduce_expertsystems_datasets():
    """Generate all ExpertSystems datasets for comparative study."""
    
    datasets = {}
    
    # 1. Sine Dataset
    print("Generating Sine dataset...")
    datasets['sine'] = dd.create_dataset("expertsystems_sine.toml")
    
    # 2. Hyperplane Datasets
    print("Generating Hyperplane datasets...")
    datasets['hyperplane_slow'] = dd.create_dataset("expertsystems_hyperplane_slow.toml")
    datasets['hyperplane_fast'] = dd.create_dataset("expertsystems_hyperplane_fast.toml")
    
    # 3. Mixed Datasets
    print("Generating Mixed datasets...")
    datasets['mixed_200'] = dd.create_dataset("expertsystems_mixed_200.toml")
    datasets['mixed_1000'] = dd.create_dataset("expertsystems_mixed_1000.toml")
    
    # 4. STAGGER Dataset
    print("Generating STAGGER dataset...")
    datasets['stagger'] = dd.create_dataset("expertsystems_stagger.toml")
    
    # 5. Electricity Dataset
    print("Loading Electricity dataset...")
    datasets['electricity'] = dd.create_dataset("expertsystems_electricity.toml")
    
    return datasets

def validate_expertsystems_datasets(datasets):
    """Validate generated datasets match ExpertSystems specifications."""
    
    print("\n=== Dataset Validation ===")
    
    # Expected specifications from paper
    expected = {
        'sine': {'size': 50000, 'features': 2, 'drift_points': [10000, 25000, 40000]},
        'hyperplane_slow': {'size': 100000, 'features': 10, 'continuous_drift': True},
        'hyperplane_fast': {'size': 100000, 'features': 10, 'continuous_drift': True},
        'mixed_200': {'size': 40000, 'features': 4, 'drift_points': [10000, 20000, 30000]},
        'mixed_1000': {'size': 40000, 'features': 4, 'drift_points': [10000, 20000, 30000]},
        'stagger': {'size': 30000, 'features': 3, 'drift_points': [10000, 20000]},
        'electricity': {'features': 8, 'classes': 2}
    }
    
    for name, dataset in datasets.items():
        exp = expected[name]
        
        print(f"\n{name.upper()} Dataset:")
        print(f"  Size: {len(dataset.X):,} (expected: {exp.get('size', 'variable'):,})")
        print(f"  Features: {dataset.X.shape[1]} (expected: {exp.get('features', 'variable')})")
        
        if hasattr(dataset.drift_metadata, 'drift_points') and exp.get('drift_points'):
            print(f"  Drift points: {dataset.drift_metadata.drift_points}")
            print(f"  Expected: {exp['drift_points']}")
            
        if exp.get('continuous_drift'):
            is_continuous = getattr(dataset.drift_metadata, 'continuous_drift', False)
            print(f"  Continuous drift: {is_continuous} (expected: True)")

def analyze_drift_characteristics(datasets):
    """Analyze drift characteristics for research applications."""
    
    print("\n=== Drift Analysis ===")
    
    for name, dataset in datasets.items():
        print(f"\n{name.upper()}:")
        
        # Concept segments
        segments = dataset.get_concept_segments()
        print(f"  Concept segments: {len(segments)}")
        
        # Drift types and patterns
        if hasattr(dataset.drift_metadata, 'drift_types'):
            types = dataset.drift_metadata.drift_types
            patterns = dataset.drift_metadata.drift_patterns
            print(f"  Drift types: {types}")
            print(f"  Drift patterns: {patterns}")
        
        # Feature analysis
        continuous = dataset.get_continuous_features()
        categorical = dataset.get_categorical_features()
        print(f"  Continuous features: {len(continuous.columns)}")
        print(f"  Categorical features: {len(categorical.columns)}")

# Execute reproduction
if __name__ == "__main__":
    datasets = reproduce_expertsystems_datasets()
    validate_expertsystems_datasets(datasets)
    analyze_drift_characteristics(datasets)
    
    print("\n✅ ExpertSystems dataset reproduction complete!")
```

## Research Usage Patterns

### Drift Detection Evaluation

```python
def evaluate_drift_detector(datasets, detector_class):
    """Evaluate drift detector on ExpertSystems datasets."""
    
    results = {}
    
    for name, dataset in datasets.items():
        print(f"Evaluating on {name} dataset...")
        
        # Prepare data for drift detection
        X = dataset.get_drift_detection_features()
        y = dataset.y
        
        # Initialize detector
        detector = detector_class()
        
        # Simulate online learning scenario
        detected_drifts = []
        
        for i in range(len(X)):
            # Update detector with new sample
            is_drift = detector.add_element(X.iloc[i], y.iloc[i])
            
            if is_drift:
                detected_drifts.append(i)
        
        # Compare with ground truth
        ground_truth = dataset.drift_metadata.drift_points
        
        results[name] = {
            'detected': detected_drifts,
            'ground_truth': ground_truth,
            'precision': calculate_precision(detected_drifts, ground_truth),
            'recall': calculate_recall(detected_drifts, ground_truth),
            'delay': calculate_detection_delay(detected_drifts, ground_truth)
        }
    
    return results
```

### Comparative Study Framework

```python
def run_comparative_study(detectors, datasets):
    """Run comparative study following ExpertSystems methodology."""
    
    results_matrix = {}
    
    for detector_name, detector_class in detectors.items():
        print(f"Testing {detector_name}...")
        
        detector_results = evaluate_drift_detector(datasets, detector_class)
        results_matrix[detector_name] = detector_results
    
    # Generate comparison tables
    create_results_table(results_matrix)
    create_performance_plots(results_matrix)
    
    return results_matrix

# Usage
detectors = {
    'ADWIN': AdwinDetector,
    'DDM': DDMDetector,
    'EDDM': EDDMDetector,
    'Page-Hinkley': PageHinkleyDetector
}

expertsystems_datasets = reproduce_expertsystems_datasets()
study_results = run_comparative_study(detectors, expertsystems_datasets)
```

## Paper Compatibility Checklist

### ✅ Required Configurations

- [ ] **Sine Dataset**: 50K samples, 3 abrupt drifts at 10K/25K/40K
- [ ] **Hyperplane Slow**: 100K samples, Hyp(0.001), continuous drift
- [ ] **Hyperplane Fast**: 100K samples, Hyp(0.1), continuous drift  
- [ ] **Mixed 200**: 40K samples, intermittent drift, 200-sample transitions
- [ ] **Mixed 1000**: 40K samples, intermittent drift, 1000-sample transitions
- [ ] **STAGGER**: 30K samples, 3 concepts, abrupt changes
- [ ] **Electricity**: UCI dataset 321, optional drift injection

### ✅ Validation Criteria

- [ ] Dataset sizes match paper specifications
- [ ] Drift points occur at correct sample indices
- [ ] Drift types (concept/covariate/prior) correctly specified
- [ ] Drift patterns (abrupt/gradual/continuous) properly implemented
- [ ] Feature types and roles correctly defined
- [ ] Random seeds produce reproducible results

### ✅ Research Integration

- [ ] Ground truth drift metadata available for evaluation
- [ ] Compatible with standard drift detection frameworks
- [ ] Supports online learning simulation workflows
- [ ] Enables comparative performance analysis
- [ ] Provides consistent evaluation methodology

## Paper Citation

When using these configurations in research, please cite both the original ExpertSystems paper and drift-datasets:

```bibtex
@article{expertsystems_drift_study,
    title={A comparative study on concept drift detectors},
    author={[Authors]},
    journal={Expert Systems with Applications},
    year={[Year]},
    doi={[DOI]}
}

@software{drift_datasets,
    title={drift-datasets: Standardized concept drift datasets with ground truth metadata},
    author={Borja Esteban},
    year={2025},
    url={https://github.com/BorjaEst/drift-datasets}
}
```

This comprehensive guide enables exact reproduction of ExpertSystems study configurations using drift-datasets, ensuring consistent and comparable results across different research studies.
