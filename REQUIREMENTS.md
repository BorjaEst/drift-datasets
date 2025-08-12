# Requirements for Concept Drift Dataset Generator

## Project Overview

This document specifies requirements for a Python library that generates synthetic datasets with concept drift patterns, designed to reproduce the exact datasets used in "A comparative study on concept drift detectors" (Gonçalves et al., 2014). The system follows Test-Driven Development (TDD) principles to ensure research-grade accuracy and reliability.

## TDD Development Approach

### Core TDD Principles

- **RED**: Write failing tests that specify exact dataset behavior
- **GREEN**: Implement minimal code to make tests pass  
- **REFACTOR**: Improve code while maintaining green tests

### Test-First Requirements

- Each module requires comprehensive test suite before implementation
- Tests must verify exact mathematical properties and drift patterns
- Statistical properties must be validated through executable tests
- Edge cases and boundary conditions must have test coverage

## Module 1: Core Dataset Foundation

### REQ-CORE-001: Base Dataset Interface

**Description**: Define abstract base interface for all dataset generators.

**Test Specification**:

```python
def test_dataset_interface():
    dataset = DatasetGenerator()
    assert hasattr(dataset, 'generate')
    assert hasattr(dataset, 'get_drift_points')
    assert hasattr(dataset, 'get_metadata')
```

**Acceptance Criteria**:

- All dataset generators must implement common interface
- Interface must support instance generation
- Interface must provide drift point information
- Interface must expose dataset metadata

### REQ-CORE-002: Instance Generation Engine

**Description**: Core engine for generating data instances with deterministic behavior.

**Requirements**:

- Generate instances with configurable number of examples
- Support streaming generation (one instance at a time)
- Maintain deterministic behavior with seed control
- Validate instance format and types

### REQ-CORE-003: Label Generation System

**Description**: System for generating classification labels based on concept definitions.

**Requirements**:

- Binary classification support (primary requirement)
- Multi-class support (future extension)
- Label consistency within concept periods
- Support for concept transitions

## Module 2: Abrupt Drift Datasets

### REQ-ABRUPT-101: Sine Dataset Generator

**Description**: Generate Sine datasets with abrupt concept drift as defined in Gonçalves et al. (2014).

**Mathematical Definition Test**:

```python
def test_sine_mathematical_definition():
    generator = SineDatasetGenerator()
    x, y = generator.generate_instance()
    # Test: points below curve y = sin(x) are positive
    expected_label = 1 if y < math.sin(x) else 0
    assert generator.get_label(x, y, concept=0) == expected_label
```

**Requirements**:

- Generate coordinates (x, y) uniformly in [0,1] interval
- Classification rule: positive if y < sin(x), negative otherwise  
- Drift behavior: classification reverses after each drift point
- Add 2 irrelevant attributes with random values in [0,1]
- Support Sine(1000) and Sine(5000) configurations
- Ensure balanced positive/negative examples

**Drift Points Test**:

```python
def test_sine_drift_points():
    generator = SineDatasetGenerator(context_size=1000)
    drift_points = generator.get_drift_points()
    assert drift_points == [1000, 2000, 3000]  # Every 1000 instances
```

### REQ-ABRUPT-102: Stagger Dataset Generator

**Description**: Generate Stagger datasets with categorical attributes and abrupt concept changes.

**Attribute Definition Test**:

```python
def test_stagger_attributes():
    generator = StaggerDatasetGenerator()
    instance = generator.generate_instance()
    assert instance.shape in ['triangle', 'circle', 'rectangle']
    assert instance.color in ['green', 'blue', 'red']
    assert instance.size in ['small', 'medium', 'large']
```

**Requirements**:

- Three categorical attributes: shape, color, size (3 values each)
- Total combinations: 27 (3³)
- Three concept definitions:
  1. Concept 1: color=red AND size=small
  2. Concept 2: color=green OR shape=circle  
  3. Concept 3: size=medium OR size=large
- Support Stagger(1) and Stagger(20) configurations
- Noise-free dataset (100% accuracy on concept function)

**Concept Transition Test**:

```python
def test_stagger_concept_transitions():
    generator = StaggerDatasetGenerator()
    # Test concept 1 to 2: 11 same, 16 different
    same_count, different_count = generator.analyze_transition(1, 2)
    assert same_count == 11
    assert different_count == 16
```

## Module 3: Gradual Drift Datasets

### REQ-GRADUAL-201: Hyperplane Dataset Generator

**Description**: Generate rotating hyperplane datasets with gradual concept drift.

**Hyperplane Equation Test**:

```python
def test_hyperplane_equation():
    generator = HyperplaneDatasetGenerator(dimensions=10)
    instance, label = generator.generate_instance()
    weights = generator.get_current_weights()
    sum_product = sum(w * x for w, x in zip(weights[:-1], instance))
    expected_label = 1 if sum_product >= weights[-1] else 0
    assert label == expected_label
```

**Requirements**:

- d-dimensional rotating hyperplane (default d=10)
- Classification: ∑(wi * xi) ≥ w0 → positive class
- Continuous rotation with configurable speed
- 5% noise injection
- 10% probability of direction reversal
- Support Hyp(0.1) and Hyp(0.001) configurations

**Rotation Test**:

```python
def test_hyperplane_rotation():
    generator = HyperplaneDatasetGenerator(change_rate=0.1)
    weights_t0 = generator.get_current_weights()
    generator.advance_time(100)  # 100 instances
    weights_t100 = generator.get_current_weights()
    assert weights_t0 != weights_t100  # Weights should change
```

### REQ-GRADUAL-202: Mixed Dataset Generator

**Description**: Generate Mixed datasets with gradual concept transitions using probability-based drift.

**Condition Test**:

```python
def test_mixed_conditions():
    generator = MixedDatasetGenerator()
    v, w, x, y = 1, 0, 0.3, 0.4
    conditions = generator.evaluate_conditions(v, w, x, y)
    # Condition 1: v (True), Condition 2: w (False), 
    # Condition 3: y < 0.5 + 0.3*sin(3*pi*x)
    expected_condition_3 = y < (0.5 + 0.3 * math.sin(3 * math.pi * x))
    assert conditions[2] == expected_condition_3
```

**Requirements**:

- 2 boolean attributes (v, w) + 2 numerical attributes (x, y)
- Three conditions:
  1. v (boolean value)
  2. w (boolean value)  
  3. y < 0.5 + 0.3×sin(3πx)
- Positive label if ≥2 conditions satisfied
- Gradual drift: probability transition between old/new concepts
- Support Mixed(200) and Mixed(1000) width configurations
- Noise-free dataset

**Gradual Transition Test**:

```python
def test_mixed_gradual_transition():
    generator = MixedDatasetGenerator(width=200)
    # At drift start: 100% old concept
    prob_old = generator.get_concept_probability(drift_start=1000, current=1000)
    assert prob_old == 1.0
    # At drift middle: 50% each
    prob_old = generator.get_concept_probability(drift_start=1000, current=1100) 
    assert abs(prob_old - 0.5) < 0.01
    # At drift end: 100% new concept
    prob_old = generator.get_concept_probability(drift_start=1000, current=1200)
    assert prob_old == 0.0
```

## Module 4: Configuration Management

### REQ-CONFIG-301: Dataset Configuration System

**Description**: Centralized system for managing dataset parameters and presets.

**Configuration Test**:

```python
def test_dataset_configurations():
    configs = {
        'Sine(1000)': {'type': 'sine', 'context_size': 1000, 't': 1, 'd': 100},
        'Sine(5000)': {'type': 'sine', 'context_size': 5000, 't': 10, 'd': 100},
        'Stagger(1)': {'type': 'stagger', 'context_size': 40, 't': 1, 'd': 100},
        'Stagger(20)': {'type': 'stagger', 'context_size': 40, 't': 20, 'd': 100}
    }
    for name, config in configs.items():
        generator = DatasetFactory.create(name, config)
        assert generator.get_config() == config
```

**Requirements**:

- Support predefined configurations from paper
- Allow custom configuration parameters
- Validate configuration compatibility
- Provide configuration templates

### REQ-CONFIG-302: Parameter Validation System

**Description**: Comprehensive parameter validation with informative error messages.

**Parameter Validation Test**:

```python
def test_parameter_validation():
    with pytest.raises(ValueError, match="Context size must be positive"):
        SineDatasetGenerator(context_size=-1)
    
    with pytest.raises(ValueError, match="Training instances must be positive"):
        SineDatasetGenerator(t=0)
```

**Requirements**:

- Validate all parameters at initialization
- Provide clear error messages for invalid parameters
- Support parameter ranges and constraints
- Default parameter values match paper specifications

## Module 5: Data Quality and Validation

### REQ-QUALITY-401: Statistical Properties Validation

**Description**: Ensure generated datasets maintain proper statistical properties.

**Class Balance Test**:

```python
def test_class_balance():
    generator = SineDatasetGenerator()
    labels = [generator.generate_instance()[1] for _ in range(10000)]
    positive_ratio = sum(labels) / len(labels)
    assert 0.45 <= positive_ratio <= 0.55  # Approximately balanced
```

**Requirements**:

- Maintain approximate class balance (45-55% each class)
- Validate statistical distributions match specifications
- Test for proper randomness and no unwanted patterns
- Verify noise levels where specified

### REQ-QUALITY-402: Drift Point Accuracy Validation

**Description**: Ensure concept drifts occur exactly at specified instance counts.

**Drift Detection Test**:

```python
def test_drift_point_accuracy():
    generator = SineDatasetGenerator(context_size=1000)
    instances = []
    for i in range(2500):  # Span multiple drifts
        instances.append(generator.generate_instance())
    
    # Verify drift occurs exactly at specified points
    assert instances[999][1] != instances[1001][1] or instances[999][0] == instances[1001][0]  # Classification should change at drift
```

**Requirements**:

- Drift occurs at exact specified instance counts
- Concept changes are immediate (abrupt) or gradual as specified
- No drift occurs within stable concept periods
- Drift points are reproducible with same seed

## Module 6: Export and Compatibility

### REQ-EXPORT-501: Data Format Support

**Description**: Support multiple data export formats for different ML frameworks.

**Format Test**:

```python
def test_output_formats():
    generator = SineDatasetGenerator()
    
    # CSV format
    csv_data = generator.export_csv(1000)
    assert isinstance(csv_data, str)
    assert csv_data.count('\n') == 1001  # Header + 1000 instances
    
    # ARFF format (Weka compatibility)
    arff_data = generator.export_arff(1000)
    assert '@relation' in arff_data
    assert '@data' in arff_data
    
    # NumPy arrays
    X, y = generator.export_numpy(1000)
    assert X.shape == (1000, 4)  # 4 attributes for Sine
    assert y.shape == (1000,)
```

**Requirements**:

- CSV export with headers
- ARFF format for Weka compatibility  
- NumPy array export for scikit-learn
- JSON export for metadata
- Streaming output for large datasets

### REQ-EXPORT-502: MOA Framework Integration

**Description**: Seamless integration with the MOA (Massive Online Analysis) framework.

**MOA Integration Test**:

```python
def test_moa_compatibility():
    generator = SineDatasetGenerator()
    moa_stream = generator.to_moa_stream()
    
    # Test MOA interface compliance
    assert hasattr(moa_stream, 'nextInstance')
    assert hasattr(moa_stream, 'hasMoreInstances')
    assert hasattr(moa_stream, 'getHeader')
```

**Requirements**:

- Generate MOA-compatible data streams
- Support MOA evaluation framework requirements
- Match original paper's MOA implementations
- Provide MOA metadata and headers

## Module 7: Performance and Scalability

### REQ-PERFORMANCE-601: Memory Efficiency

**Description**: Efficient memory usage for large dataset generation.

**Memory Test**:

```python
def test_memory_efficiency():
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    generator = SineDatasetGenerator()
    # Generate large dataset
    for _ in range(100000):
        generator.generate_instance()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
```

**Requirements**:

- Constant memory usage for streaming generation
- Efficient storage of concept parameters only
- No memory leaks during long generations
- Configurable batch processing for large datasets

### REQ-PERFORMANCE-602: Generation Speed Optimization

**Description**: High-performance instance generation for large-scale experiments.

**Performance Test**:

```python
def test_generation_speed():
    generator = SineDatasetGenerator()
    start_time = time.time()
    
    for _ in range(10000):
        generator.generate_instance()
    
    end_time = time.time()
    instances_per_second = 10000 / (end_time - start_time)
    assert instances_per_second > 1000  # At least 1000 instances/second
```

**Requirements**:

- Generate at least 1000 instances per second
- Efficient mathematical computations
- Minimal overhead per instance
- Scalable to millions of instances

## Module 8: Reproducibility and Testing

### REQ-REPRODUCIBILITY-701: Deterministic Behavior

**Description**: Ensure reproducible dataset generation across platforms and runs.

**Reproducibility Test**:

```python
def test_deterministic_generation():
    generator1 = SineDatasetGenerator(seed=42)
    generator2 = SineDatasetGenerator(seed=42)
    
    instances1 = [generator1.generate_instance() for _ in range(100)]
    instances2 = [generator2.generate_instance() for _ in range(100)]
    
    assert instances1 == instances2  # Exact reproduction
```

**Requirements**:

- Deterministic generation with seed control
- Reproducible across different runs and platforms
- Seed affects all random components consistently
- Support for random seed generation and logging

### REQ-REPRODUCIBILITY-702: Comprehensive Test Coverage

**Description**: Maintain comprehensive test suite with high coverage standards.

**Test Coverage Requirements**:

- Unit tests for each dataset type (>95% coverage)
- Integration tests for complete workflows
- Property-based tests for mathematical correctness
- Performance regression tests
- Cross-platform compatibility tests

**Test Structure**:

```
tests/
├── unit/
│   ├── test_sine_dataset.py
│   ├── test_stagger_dataset.py
│   ├── test_hyperplane_dataset.py
│   ├── test_mixed_dataset.py
├── integration/
│   ├── test_moa_compatibility.py
│   ├── test_export_formats.py
├── property/
│   ├── test_mathematical_properties.py
│   ├── test_statistical_properties.py
└── performance/
    ├── test_memory_usage.py
    ├── test_generation_speed.py
```

## Implementation Guidelines

### TDD Development Process

1. **Start with Interface Tests**
   - Define abstract base classes through tests
   - Specify method signatures and return types
   - Validate error handling behavior

2. **Mathematical Property Tests**
   - Test exact mathematical formulas
   - Verify concept definitions
   - Validate drift transition logic

3. **Statistical Property Tests**  
   - Test class distributions
   - Verify noise levels
   - Validate randomness properties

4. **Integration Tests**
   - Test complete dataset generation workflows
   - Verify export format correctness
   - Test MOA framework compatibility

5. **Performance Tests**
   - Benchmark generation speed
   - Monitor memory usage
   - Test scalability limits

### Quality Assurance Standards

**Code Quality**:

- Type hints for all public methods
- Comprehensive docstrings with examples
- Consistent naming conventions
- Error handling with informative messages

**Documentation**:

- API documentation with examples
- Mathematical formulations explained
- Configuration guides
- Performance benchmarks

**Continuous Integration**:

- Automated test suite execution
- Code coverage reporting (>95%)
- Performance regression detection
- Multi-platform testing (Linux, Windows, macOS)

## Acceptance Criteria

### Dataset Validation Criteria

1. **Mathematical Correctness**: All generated instances must satisfy exact mathematical definitions from the paper
2. **Drift Accuracy**: Concept drifts must occur at precisely specified instance counts
3. **Statistical Properties**: Generated data must match expected distributions and class balances
4. **Reproducibility**: Identical seeds must produce identical datasets across runs and platforms
5. **Performance**: Must generate at least 1000 instances per second with constant memory usage
6. **Compatibility**: Must integrate seamlessly with MOA framework and export to standard formats

### Test Coverage Requirements

- **Unit Tests**: 100% coverage of core generation logic
- **Integration Tests**: All export formats and framework integrations
- **Property Tests**: Mathematical and statistical properties validated
- **Performance Tests**: Memory and speed benchmarks established
- **Cross-platform Tests**: Verified on Linux, Windows, and macOS

### Documentation Requirements

- **API Documentation**: Complete with examples and type hints
- **Mathematical Specifications**: All formulas and concepts explained
- **Configuration Guide**: All parameters documented with examples
- **Performance Benchmarks**: Speed and memory usage characteristics
