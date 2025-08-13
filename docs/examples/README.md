# Examples Directory

This directory contains practical examples demonstrating how to use drift-datasets for various research and development scenarios.

## Example Categories

### 📊 Basic Usage Examples

- **[basic_synthetic.py](basic_synthetic.py)** - Generate simple synthetic datasets
- **[basic_real_world.py](basic_real_world.py)** - Work with UCI ML Repository datasets
- **[basic_mixed.py](basic_mixed.py)** - Combine synthetic and real-world data

### 🌊 Drift Pattern Examples

- **[abrupt_drift.py](abrupt_drift.py)** - Instantaneous concept changes
- **[gradual_drift.py](gradual_drift.py)** - Smooth transitions between concepts
- **[continuous_drift.py](continuous_drift.py)** - Ongoing changes without stable periods
- **[recurring_drift.py](recurring_drift.py)** - Cyclic concept patterns

### 🔬 Research Use Cases

- **[expertsystems_reproduction.py](expertsystems_reproduction.py)** - Reproduce research paper datasets
- **[drift_detection_benchmark.py](drift_detection_benchmark.py)** - Create benchmarks for drift detectors
- **[comparative_study.py](comparative_study.py)** - Multi-dataset comparative analysis
- **[custom_evaluation.py](custom_evaluation.py)** - Custom evaluation scenarios

### ⚙️ Configuration Examples

- **[complex_config.toml](configs/complex_config.toml)** - Advanced multi-drift configuration
- **[research_config.toml](configs/research_config.toml)** - Research-oriented parameters
- **[benchmark_config.toml](configs/benchmark_config.toml)** - Benchmark suite configuration
- **[validation_config.toml](configs/validation_config.toml)** - Validation and testing setup

### 📈 Analysis and Visualization

- **[drift_analysis.py](drift_analysis.py)** - Analyze drift characteristics
- **[visualization_examples.py](visualization_examples.py)** - Plot drift patterns and data
- **[statistical_analysis.py](statistical_analysis.py)** - Statistical drift analysis
- **[metadata_exploration.py](metadata_exploration.py)** - Explore ground truth metadata

### 🔧 Integration Examples

- **[sklearn_integration.py](sklearn_integration.py)** - Integrate with scikit-learn workflows
- **[streaming_simulation.py](streaming_simulation.py)** - Simulate streaming data scenarios
- **[evaluation_framework.py](evaluation_framework.py)** - Build evaluation frameworks
- **[export_formats.py](export_formats.py)** - Export to different formats

## Quick Start

Each example is self-contained and includes:

- Complete working code
- Detailed comments explaining each step
- Expected output descriptions
- Configuration files when applicable
- Visualization code for relevant examples

## Running Examples

```bash
# Install dependencies
pip install drift-datasets[examples]

# Run a basic example
python docs/examples/basic_synthetic.py

# Run with custom configuration
python docs/examples/gradual_drift.py --config configs/custom.toml

# Generate all example datasets
python docs/examples/generate_all_examples.py
```

## Example Data

Generated example datasets are saved to `docs/examples/output/` directory with:

- CSV files for each dataset
- Metadata JSON files with drift information
- Visualization plots (PNG/PDF)
- Configuration files used for generation

## Contributing Examples

To add new examples:

1. Create a descriptive Python file in the appropriate category
2. Include complete docstrings and comments
3. Add any required configuration files to `configs/`
4. Update this README with the new example
5. Test the example thoroughly

## Example Template

```python
#!/usr/bin/env python3
"""
Example Title: Brief description

This example demonstrates [specific feature/use case].
Shows how to [key learning objectives].

Expected output:
- [Describe what the example produces]
- [Key metrics or visualizations]
"""

import drift_datasets as dd
import pandas as pd
import matplotlib.pyplot as plt

def main():
    \"\"\"Main example function with clear steps.\"\"\"
    # Step 1: Configuration
    # Step 2: Generation  
    # Step 3: Analysis
    # Step 4: Output

if __name__ == "__main__":
    main()
```
