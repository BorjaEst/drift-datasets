# Quickstart Tutorial

Generate your first drift dataset in under 2 minutes! This tutorial covers the essential steps to get started with drift-datasets.

## Prerequisites

- Python 3.10+ installed
- Java Runtime Environment installed (for synthetic datasets)
- drift-datasets package installed (`pip install drift-datasets`)

If you haven't installed these yet, see the [Installation Guide](installation.md).

## 1. Create Your First Configuration

Create a file called `my_first_dataset.toml` with this content:

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

### Configuration Breakdown

- **Dataset section**: Defines a synthetic dataset using CapyMOA's SineGenerator
- **Metadata section**: Specifies multivariate supervised learning with 2 classes
- **Features section**: Three features - two continuous (x, y) and one categorical target (class)
- **Generator config**: 1000 samples, specific classification function, fixed random seed
- **Drift config**: Single concept drift at sample 500 with abrupt pattern

## 2. Generate Your Dataset

```python
import drift_datasets
from rich.console import Console

console = Console()

# Generate dataset from configuration file
dataset = drift_datasets.create_dataset("my_first_dataset.toml")

# Inspect the basic structure
console.print(f"Dataset name: {dataset.name}")
console.print(f"Dataset shape: X={dataset.X.shape}, y={dataset.y.shape}")
console.print(f"Features: {list(dataset.X.columns)}")
console.print(f"Target classes: {sorted(dataset.y.unique())}")
```

**Expected output**:

```text
Dataset name: quickstart_example
Dataset shape: X=(1000, 2), y=(1000,)
Features: ['x', 'y']
Target classes: [0, 1]
```

## 3. Explore Drift Metadata

```python
from rich.console import Console

console = Console()

# Access drift information
drift_meta = dataset.drift_metadata
console.print(f"Drift points: {drift_meta.drift_points}")
console.print(f"Drift types: {drift_meta.drift_types}")
console.print(f"Drift patterns: {drift_meta.drift_patterns}")

# Check if specific samples are near drift points
console.print(f"Sample 500 is drift point: {dataset.is_drift_point(500)}")
console.print(f"Sample 495 near drift (tolerance=10): {dataset.is_drift_point(495, tolerance=10)}")
console.print(f"Sample 100 near drift: {dataset.is_drift_point(100)}")

# Get concept segments
segments = dataset.get_concept_segments()
console.print(f"Concept segments: {segments}")
```

**Expected output**:

```text
Drift points: [500]
Drift types: ['concept']
Drift patterns: ['abrupt']
Sample 500 is drift point: True
Sample 495 near drift (tolerance=10): True
Sample 100 near drift: False
Concept segments: [(0, 500), (500, 1000)]
```

## 4. Prepare Data for Modeling

```python
from rich.console import Console

console = Console()

# Get features suitable for drift detection (excludes target and metadata)
X_features = dataset.get_drift_detection_features()
y_target = dataset.y

console.print(f"Modeling features: {list(X_features.columns)}")
console.print(f"Feature matrix shape: {X_features.shape}")
console.print(f"Target shape: {y_target.shape}")

# Split temporally while preserving drift structure
train_dataset, test_dataset = dataset.split_temporal(ratio=0.7)
X_train = train_dataset.get_drift_detection_features()
X_test = test_dataset.get_drift_detection_features()
y_train = train_dataset.y
y_test = test_dataset.y

console.print(f"Train split: X={X_train.shape}, y={y_train.shape}")
console.print(f"Test split: X={X_test.shape}, y={y_test.shape}")
console.print(f"Train drift points: {train_dataset.drift_metadata.drift_points}")
console.print(f"Test drift points: {test_dataset.drift_metadata.drift_points}")
```

**Expected output**:

```text
Modeling features: ['x', 'y']
Feature matrix shape: (1000, 2)
Target shape: (1000,)
Train split: X=(700, 2), y=(700,)
Test split: X=(300, 2), y=(300,)
Train drift points: []
Test drift points: [200]
```

## 5. Visualize Your Dataset (Optional)

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot feature evolution over time
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

# Plot feature x over time
ax1.plot(dataset.X.index, dataset.X['x'], alpha=0.7, color='blue')
ax1.axvline(x=500, color='red', linestyle='--', alpha=0.7, label='Drift Point')
ax1.set_ylabel('Feature X')
ax1.set_title('Feature Evolution Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot feature y over time
ax2.plot(dataset.X.index, dataset.X['y'], alpha=0.7, color='green')
ax2.axvline(x=500, color='red', linestyle='--', alpha=0.7, label='Drift Point')
ax2.set_ylabel('Feature Y')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot class distribution over time (sliding window)
window_size = 50
class_means = []
indices = []
for i in range(window_size, len(dataset.y)):
    window = dataset.y.iloc[i-window_size:i]
    class_means.append(window.mean())
    indices.append(i)

ax3.plot(indices, class_means, color='orange', linewidth=2)
ax3.axvline(x=500, color='red', linestyle='--', alpha=0.7, label='Drift Point')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Class Mean (50-sample window)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Basic Drift Detection Example

Here's a simple example of using your dataset for drift detection research:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulate online learning scenario
model = RandomForestClassifier(n_estimators=50, random_state=42)
batch_size = 100
accuracies = []
sample_indices = []

# Process data in batches
for i in range(0, len(X_features), batch_size):
    batch_X = X_features.iloc[i:i+batch_size]
    batch_y = y_target.iloc[i:i+batch_size]
    
    if i == 0:
        # Initial training
        model.fit(batch_X, batch_y)
    else:
        # Evaluate on current batch before updating
        pred_y = model.predict(batch_X)
        accuracy = accuracy_score(batch_y, pred_y)
        accuracies.append(accuracy)
        sample_indices.append(i + batch_size // 2)  # Mid-point of batch
        
        # Update model with new data
        model.fit(batch_X, batch_y)

# Plot accuracy evolution
plt.figure(figsize=(10, 6))
plt.plot(sample_indices, accuracies, marker='o', linewidth=2, markersize=4)
plt.axvline(x=500, color='red', linestyle='--', alpha=0.7, label='Known Drift Point')
plt.xlabel('Sample Index')
plt.ylabel('Batch Accuracy')
plt.title('Model Performance Over Time (Concept Drift Detection)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.1)
plt.show()

print(f"Accuracy drop near drift point: {min(accuracies[4:7]):.3f}")
print(f"Average accuracy before drift: {np.mean(accuracies[:4]):.3f}")
print(f"Average accuracy after drift: {np.mean(accuracies[7:]):.3f}")
```

## Next Steps: More Advanced Examples

### Example 1: Real-World Dataset with Drift Injection

```toml
# Create electricity_drift.toml
[dataset]
name = "electricity_drift"
type = "real_world"
source = "ucimlrepo"

[metadata]
dimension = "multivariate"
labeling = "supervised"

[uci_config]
dataset_id = 321  # Electricity dataset

[drift_config]
drift_points = [15000, 30000]
drift_types = ["covariate", "concept"]
drift_patterns = ["gradual", "abrupt"]
transition_durations = [1000, 0]
```

```python
# Generate real-world dataset with synthetic drift
real_dataset = drift_datasets.create_dataset("electricity_drift.toml")
print(f"Real-world dataset: {real_dataset.X.shape}")
print(f"Original features: {len([f for f in real_dataset.dataset_metadata.features if f.role == 'feature'])}")
```

### Example 2: Multiple Drift Types

```toml
# Create multi_drift.toml
[dataset]
name = "multi_drift_scenario"
type = "synthetic"
source = "capymoa"
generator = "HyperplaneGenerator"

[metadata]
dimension = "multivariate"
labeling = "supervised"
n_classes = 2

[[features]]
name = "feature_1"
type = "continuous"
role = "feature"

[[features]]
name = "feature_2"
type = "continuous"
role = "feature"

[[features]]
name = "feature_3"
type = "continuous"
role = "feature"

[[features]]
name = "class"
type = "categorical"
role = "target"

[generator_config]
n_instances = 5000
n_dimensions = 3
noise_percentage = 0.05
random_seed = 42

[drift_config]
drift_points = [1000, 2500, 4000]
drift_types = ["concept", "covariate", "prior"]
drift_patterns = ["abrupt", "gradual", "abrupt"]
transition_durations = [0, 500, 0]
affected_features = [[0, 1], [1, 2], [0, 1, 2]]
```

### Example 3: Configuration from Python Dictionary

```python
# Define configuration programmatically
config = {
    "dataset": {
        "name": "programmatic_dataset",
        "type": "synthetic",
        "source": "capymoa",
        "generator": "SEAGenerator"
    },
    "metadata": {
        "dimension": "multivariate",
        "labeling": "supervised",
        "n_classes": 2
    },
    "features": [
        {"name": "x1", "type": "continuous", "role": "feature"},
        {"name": "x2", "type": "continuous", "role": "feature"},
        {"name": "x3", "type": "continuous", "role": "feature"},
        {"name": "class", "type": "categorical", "role": "target"}
    ],
    "generator_config": {
        "n_instances": 2000,
        "noise_percentage": 0.1,
        "random_seed": 123
    },
    "drift_config": {
        "drift_points": [800, 1600],
        "drift_types": ["concept", "concept"],
        "drift_patterns": ["abrupt", "gradual"],
        "transition_durations": [0, 200]
    }
}

# Generate from dictionary
sea_dataset = drift_datasets.create_dataset(config)
print(f"SEA dataset generated: {sea_dataset.X.shape}")
```

## Common Patterns Summary

After completing this tutorial, you should understand these key patterns:

1. **TOML Configuration**: Structure and sections for defining datasets
2. **Dataset Generation**: Using `create_dataset()` function
3. **Metadata Access**: Exploring drift points, types, and patterns
4. **Feature Filtering**: Getting appropriate features for modeling
5. **Temporal Splitting**: Preserving chronological order and drift structure
6. **Drift Detection**: Basic workflow for evaluating drift detection methods

## Troubleshooting Quick Fixes

**Java not found**: Ensure Java is installed and `java -version` works
**Import errors**: Verify installation with `pip list | grep drift-datasets`
**Configuration errors**: Check TOML syntax and required sections
**Empty datasets**: Verify generator_config parameters are valid
**Memory issues**: Reduce n_instances for initial testing

## What's Next?

- **Deep dive**: [Configuration Guide](configuration.md) for all options
- **Research focus**: [ExpertSystems Compatibility](expertsystems.md) for paper reproduction  
- **Advanced patterns**: [Drift Patterns Guide](drift-patterns.md) for complex scenarios
- **API reference**: [Complete API Documentation](api-reference.md)

Congratulations! You've successfully generated your first drift dataset. The concepts you've learned here form the foundation for all drift-datasets workflows.
