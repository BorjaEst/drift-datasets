# Requirements Specification

Detailed functional requirements for drift-datasets library. This document maps directly to the REQUIREMENTS.md and provides comprehensive specification of system capabilities and acceptance criteria.

## Document Overview

This specification defines all functional requirements (REQ-001 through REQ-023) for the drift-datasets library. Each requirement includes acceptance criteria, implementation details, and validation procedures.

## Core Dataset Generation Requirements

### REQ-001: Synthetic Dataset Creation

**Description**: System shall generate synthetic datasets with configurable drift patterns using CapyMOA generators.

**Priority**: Critical
**Type**: Functional
**Category**: Core Generation

#### Acceptance Criteria

1. **Generator Support**: Support all major CapyMOA generators
   - ✅ SineGenerator
   - ✅ HyperplaneGenerator  
   - ✅ STAGGERGenerator
   - ✅ RandomTreeGenerator
   - ✅ SEAGenerator
   - ✅ AgrawalGenerator
   - ✅ LEDGenerator

2. **Size Matching**: Generated dataset size exactly matches configured n_instances

   ```python
   # Test case
   config = {"generator_config": {"n_instances": 10000}}
   dataset = create_dataset(config)
   assert len(dataset.X) == 10000
   assert len(dataset.y) == 10000
   ```

3. **Reproducibility**: Random seed produces deterministic, reproducible datasets

   ```python
   # Test case
   config = {"generator_config": {"random_seed": 42}}
   dataset1 = create_dataset(config)
   dataset2 = create_dataset(config)
   assert dataset1.X.equals(dataset2.X)
   assert dataset1.y.equals(dataset2.y)
   ```

4. **Data Consistency**: Generated X, y have matching lengths and valid types

   ```python
   # Test case
   dataset = create_dataset(config)
   assert dataset.X.shape[0] == len(dataset.y)
   assert isinstance(dataset.X, pd.DataFrame)
   assert isinstance(dataset.y, pd.Series)
   ```

5. **Noise Injection**: Support configurable noise_level parameter (0.0-1.0)

   ```python
   # Test case
   config = {"generator_config": {"noise_level": 0.1}}
   dataset = create_dataset(config)
   # Noise validation depends on generator implementation
   ```

#### Implementation Notes

- Uses CapyMOA Java integration through Python bindings
- Requires Java Runtime Environment
- Generator parameters validated against CapyMOA specifications
- Error handling for invalid parameters and Java runtime issues

#### Validation Methods

```python
def test_synthetic_generation():
    """Test synthetic dataset generation requirements"""
    config = {
        "dataset": {"type": "synthetic", "source": "capymoa", "generator": "SineGenerator"},
        "generator_config": {"n_instances": 1000, "random_seed": 42}
    }
    
    dataset = create_dataset(config)
    
    # REQ-001 validation
    assert dataset.X.shape[0] == 1000
    assert len(dataset.y) == 1000
    assert dataset.X.shape[0] == len(dataset.y)
```

### REQ-002: Real-World Dataset Loading

**Description**: System shall load real-world datasets from UCI ML Repository with optional drift injection.

**Priority**: Critical  
**Type**: Functional
**Category**: Core Generation

#### Acceptance Criteria

1. **UCI Dataset Loading**: Successfully load datasets by dataset_id

   ```python
   # Test case - Electricity dataset
   config = {"uci_config": {"dataset_id": 321}}
   dataset = create_dataset(config)
   assert dataset.X.shape[0] > 0
   assert len(dataset.y) > 0
   ```

2. **Data Characteristic Preservation**: Maintain original data types and structure

   ```python
   # Test case
   dataset = create_dataset(uci_config)
   # Verify feature types match UCI metadata
   for feature in dataset.dataset_metadata.features:
       assert feature.type in ["continuous", "categorical", "mixed"]
   ```

3. **Metadata Mapping**: Map UCI metadata to internal schema

   ```python
   # Test case
   dataset = create_dataset(uci_config)
   meta = dataset.dataset_metadata
   assert meta.dimension in ["univariate", "multivariate"]
   assert meta.labeling in ["supervised", "unsupervised", "semi-supervised"]
   ```

4. **Missing Value Handling**: Handle datasets with missing values per UCI specs

   ```python
   # Test case for datasets with missing values
   dataset = create_dataset(uci_config_with_missing)
   # Validate missing value handling according to UCI documentation
   ```

5. **Temporal Ordering**: Maintain temporal order when present in source data

   ```python
   # Test case for time-series datasets
   dataset = create_dataset(temporal_uci_config)
   if dataset.dataset_metadata.temporal:
       # Verify chronological ordering preserved
       pass
   ```

#### Popular UCI Datasets Support

| Dataset | ID | Samples | Features | Classes | Status |
|---------|----|---------:|----------:|---------:|--------|
| Electricity | 321 | 45,312 | 8 | 2 | ✅ Supported |
| Forest Cover | 31 | 581,012 | 54 | 7 | ✅ Supported |
| Poker Hand | 158 | 1,025,010 | 10 | 10 | ✅ Supported |
| Census Income | 20 | 48,842 | 14 | 2 | ✅ Supported |

#### Error Handling

- ConnectionError for UCI API failures
- ValueError for invalid dataset IDs
- Graceful degradation for offline operation
- Clear error messages for network issues

### REQ-003: Mixed Dataset Combination

**Description**: System shall combine multiple data sources into unified datasets with consistent drift metadata.

**Priority**: High
**Type**: Functional  
**Category**: Advanced Generation

#### Acceptance Criteria

1. **Combination Pattern Support**:
   - ✅ Sequential: Datasets placed end-to-end
   - ✅ Interleaved: Datasets mixed sample-by-sample
   - ✅ Hierarchical: Nested dataset combinations

2. **Data Type Consistency**: Maintain consistent types across combined components

   ```python
   # Test case
   mixed_dataset = create_dataset(mixed_config)
   # All components must have compatible feature types
   for feature_name in mixed_dataset.X.columns:
       feature_types = [comp.get_feature_type(feature_name) for comp in components]
       assert all(t == feature_types[0] for t in feature_types)
   ```

3. **Metadata Preservation**: Individual component metadata maintained in unified structure

   ```python
   # Test case
   mixed_dataset = create_dataset(mixed_config)
   assert hasattr(mixed_dataset.metadata, 'component_metadata')
   assert len(mixed_dataset.metadata.component_metadata) == len(components)
   ```

4. **Drift Parameter Consistency**: Apply consistent drift parameters across components

   ```python
   # Test case
   mixed_config = {
       "drift_config": {"drift_points": [5000]},
       "mixed_config": {"components": [comp1, comp2]}
   }
   mixed_dataset = create_dataset(mixed_config)
   assert mixed_dataset.drift_metadata.drift_points == [5000]
   ```

5. **Valid Drift Indices**: Generate valid drift_points indices for combined dataset

   ```python
   # Test case
   mixed_dataset = create_dataset(mixed_config)
   for drift_point in mixed_dataset.drift_metadata.drift_points:
       assert 0 <= drift_point < len(mixed_dataset.X)
   ```

## Configuration and Parsing Requirements

### REQ-004: TOML Configuration Parsing

**Description**: System shall parse and validate TOML configuration files with comprehensive error reporting.

**Priority**: Critical
**Type**: Functional
**Category**: Configuration

#### Acceptance Criteria

1. **Required Section Parsing**: Parse all required sections correctly

   ```toml
   # Test configuration
   [dataset]
   name = "test"
   type = "synthetic"
   
   [metadata]
   dimension = "multivariate"
   labeling = "supervised"
   
   [generator_config]
   n_instances = 1000
   
   [drift_config]
   drift_points = [500]
   ```

2. **Dataset Type Validation**: Validate dataset.type against allowed values

   ```python
   # Test case
   valid_types = ["synthetic", "real_world", "mixed"]
   for valid_type in valid_types:
       config = {"dataset": {"type": valid_type}}
       # Should not raise ValueError
       validate_config(config)
   
   # Invalid type should raise ValueError
   with pytest.raises(ValueError, match="dataset.type must be one of"):
       validate_config({"dataset": {"type": "invalid"}})
   ```

3. **Metadata Validation**: Validate metadata fields against specifications

   ```python
   # Test case
   valid_dimensions = ["univariate", "multivariate"]
   valid_labelings = ["supervised", "unsupervised", "semi-supervised"]
   
   for dim in valid_dimensions:
       for label in valid_labelings:
           config = {"metadata": {"dimension": dim, "labeling": label}}
           validate_config(config)
   ```

4. **Error Reporting**: Report specific validation errors with context

   ```python
   # Test case
   invalid_config = {"dataset": {"type": "invalid_type"}}
   
   with pytest.raises(ValueError) as exc_info:
       validate_config(invalid_config)
   
   error_msg = str(exc_info.value)
   assert "dataset.type" in error_msg
   assert "invalid_type" in error_msg
   ```

5. **File Error Handling**: Raise FileNotFoundError for missing files

   ```python
   # Test case
   with pytest.raises(FileNotFoundError):
       create_dataset("nonexistent_config.toml")
   ```

### REQ-005: Feature Metadata Specification

**Description**: System shall support detailed feature-level metadata specification in configuration.

**Priority**: High
**Type**: Functional
**Category**: Configuration

#### Acceptance Criteria

1. **Feature Array Parsing**: Parse [[features]] array with required fields

   ```toml
   # Test configuration
   [[features]]
   name = "temperature"
   type = "continuous"
   role = "feature"
   description = "Temperature in Celsius"
   ```

2. **Type Validation**: Validate type field against allowed values

   ```python
   # Test case
   valid_types = ["continuous", "categorical", "mixed"]
   for feature_type in valid_types:
       feature_config = {"type": feature_type}
       validate_feature_config(feature_config)
   ```

3. **Role Validation**: Validate role field against allowed values

   ```python
   # Test case
   valid_roles = ["feature", "target", "timestamp", "identifier", "metadata", "exclude"]
   for role in valid_roles:
       feature_config = {"role": role}
       validate_feature_config(feature_config)
   ```

4. **Optional Field Support**: Support optional fields with appropriate defaults

   ```python
   # Test case
   feature_config = {
       "name": "optional_feature",
       "type": "continuous",
       "role": "feature",
       "missing_values": True,  # Optional
       "unique_values": 50      # Optional
   }
   validate_feature_config(feature_config)
   ```

5. **Target Feature Requirement**: Enforce at least one target feature for supervised datasets

   ```python
   # Test case - Should pass
   supervised_config = {
       "metadata": {"labeling": "supervised"},
       "features": [
           {"name": "x", "type": "continuous", "role": "feature"},
           {"name": "y", "type": "categorical", "role": "target"}
       ]
   }
   validate_config(supervised_config)
   
   # Test case - Should fail
   no_target_config = {
       "metadata": {"labeling": "supervised"},
       "features": [{"name": "x", "type": "continuous", "role": "feature"}]
   }
   with pytest.raises(ValueError, match="supervised datasets require.*target"):
       validate_config(no_target_config)
   ```

6. **Exclusion Handling**: Allow features with role="exclude" to be omitted from X matrix

   ```python
   # Test case
   config_with_excluded = {
       "features": [
           {"name": "include_me", "type": "continuous", "role": "feature"},
           {"name": "exclude_me", "type": "continuous", "role": "exclude"},
           {"name": "target", "type": "categorical", "role": "target"}
       ]
   }
   dataset = create_dataset(config_with_excluded)
   assert "include_me" in dataset.X.columns
   assert "exclude_me" not in dataset.X.columns
   ```

## Drift Metadata and Ground Truth Requirements

### REQ-006: Drift Point Specification

**Description**: System shall create precise drift point metadata with ground truth information.

**Priority**: Critical
**Type**: Functional  
**Category**: Drift Metadata

#### Acceptance Criteria

1. **Valid Drift Indices**: drift_points list contains valid sample indices

   ```python
   # Test case
   dataset = create_dataset(config_with_drift)
   for drift_point in dataset.drift_metadata.drift_points:
       assert 0 <= drift_point < len(dataset.X)
   ```

2. **Consistent Array Lengths**: drift_types and drift_patterns match drift_points length

   ```python
   # Test case
   drift_meta = dataset.drift_metadata
   assert len(drift_meta.drift_types) == len(drift_meta.drift_points)
   assert len(drift_meta.drift_patterns) == len(drift_meta.drift_points)
   ```

3. **Valid Drift Types**: drift_types contains only allowed values

   ```python
   # Test case
   valid_types = ["covariate", "concept", "prior", "none"]
   drift_meta = dataset.drift_metadata
   for drift_type in drift_meta.drift_types:
       assert drift_type in valid_types
   ```

4. **Valid Drift Patterns**: drift_patterns contains only allowed values

   ```python
   # Test case
   valid_patterns = ["abrupt", "gradual", "recurring", "incremental"]
   drift_meta = dataset.drift_metadata
   for pattern in drift_meta.drift_patterns:
       assert pattern in valid_patterns
   ```

5. **Optional Parameter Support**: Handle optional drift_widths and drift_alphas

   ```python
   # Test case
   config_with_capymoa_params = {
       "drift_config": {
           "drift_points": [1000],
           "drift_widths": [500],
           "drift_alphas": [0.5]
       }
   }
   dataset = create_dataset(config_with_capymoa_params)
   assert hasattr(dataset.drift_metadata, 'drift_widths')
   assert hasattr(dataset.drift_metadata, 'drift_alphas')
   ```

6. **Affected Features Specification**: Optional affected_features for each drift

   ```python
   # Test case
   config_with_affected = {
       "drift_config": {
           "drift_points": [1000, 2000],
           "affected_features": [[0, 1], [1, 2]]
       }
   }
   dataset = create_dataset(config_with_affected)
   assert len(dataset.drift_metadata.affected_features) == 2
   ```

### REQ-007: Drift Analysis Methods

**Description**: System shall provide analysis methods for drift detection research workflows.

**Priority**: High
**Type**: Functional
**Category**: Analysis Methods

#### Acceptance Criteria

1. **Drift Point Proximity**: is_drift_point(idx, tolerance) returns boolean

   ```python
   # Test case
   dataset = create_dataset(config_with_drift_at_500)
   
   assert dataset.is_drift_point(500) == True  # Exact match
   assert dataset.is_drift_point(495, tolerance=10) == True  # Within tolerance
   assert dataset.is_drift_point(400, tolerance=10) == False  # Outside tolerance
   ```

2. **Concept Segmentation**: get_concept_segments() returns valid segments

   ```python
   # Test case
   dataset = create_dataset(config_with_drift_at_500)
   segments = dataset.get_concept_segments()
   
   assert isinstance(segments, list)
   assert all(isinstance(seg, tuple) and len(seg) == 2 for seg in segments)
   assert segments == [(0, 500), (500, 1000)]  # For drift at 500, n_instances=1000
   ```

3. **Drift Type Lookup**: get_drift_type_at(idx) returns correct drift type

   ```python
   # Test case
   dataset = create_dataset(config_with_concept_drift_at_500)
   
   assert dataset.get_drift_type_at(500) == "concept"
   assert dataset.get_drift_type_at(400) == "none"  # Before drift
   assert dataset.get_drift_type_at(600) == "none"  # After drift (stable period)
   ```

4. **Affected Features Lookup**: get_affected_features_at(idx) returns feature indices

   ```python
   # Test case
   config = {
       "drift_config": {
           "drift_points": [500],
           "affected_features": [[0, 2]]
       }
   }
   dataset = create_dataset(config)
   
   affected = dataset.get_affected_features_at(500)
   assert affected == [0, 2]
   ```

5. **Metadata Validation**: validate_drift_metadata() ensures consistency

   ```python
   # Test case
   dataset = create_dataset(valid_config)
   validation = dataset.validate_drift_metadata()
   
   assert validation['valid'] == True
   assert isinstance(validation['issues'], list)
   assert len(validation['issues']) == 0
   ```

6. **Edge Case Handling**: Methods handle edge cases gracefully

   ```python
   # Test case
   dataset = create_dataset(config)
   
   # Out of bounds indices
   assert dataset.is_drift_point(-1) == False
   assert dataset.is_drift_point(len(dataset.X)) == False
   
   # Empty drift_points
   no_drift_dataset = create_dataset(no_drift_config)
   assert no_drift_dataset.get_concept_segments() == [(0, len(no_drift_dataset.X))]
   ```

## Data Access and Filtering Requirements

### REQ-008: Feature Role-Based Filtering

**Description**: System shall filter features based on their role for appropriate modeling workflows.

**Priority**: High
**Type**: Functional
**Category**: Data Access

#### Acceptance Criteria

1. **Drift Detection Features**: get_drift_detection_features() excludes non-modeling features

   ```python
   # Test case
   config = {
       "features": [
           {"name": "feature1", "type": "continuous", "role": "feature"},
           {"name": "target", "type": "categorical", "role": "target"},
           {"name": "timestamp", "type": "continuous", "role": "timestamp"},
           {"name": "id", "type": "continuous", "role": "identifier"},
           {"name": "notes", "type": "categorical", "role": "metadata"},
           {"name": "excluded", "type": "continuous", "role": "exclude"}
       ]
   }
   dataset = create_dataset(config)
   
   drift_features = dataset.get_drift_detection_features()
   assert list(drift_features.columns) == ["feature1"]  # Only modeling features
   ```

2. **Type-Based Filtering**: get_continuous_features() and get_categorical_features()

   ```python
   # Test case
   dataset = create_dataset(mixed_type_config)
   
   continuous = dataset.get_continuous_features()
   categorical = dataset.get_categorical_features()
   
   # Verify all continuous features have type="continuous"
   for col in continuous.columns:
       feature_info = dataset.get_feature_info(col)
       assert feature_info.type == "continuous"
   
   # Verify all categorical features have type="categorical"
   for col in categorical.columns:
       feature_info = dataset.get_feature_info(col)
       assert feature_info.type == "categorical"
   ```

3. **Role-Based Filtering**: get_features_by_role(role) returns matching features

   ```python
   # Test case
   dataset = create_dataset(multi_role_config)
   
   feature_cols = dataset.get_features_by_role("feature")
   target_cols = dataset.get_features_by_role("target")
   timestamp_cols = dataset.get_features_by_role("timestamp")
   
   # Verify role filtering
   for col in feature_cols.columns:
       feature_info = dataset.get_feature_info(col)
       assert feature_info.role == "feature"
   ```

4. **Index Consistency**: Filtered DataFrames maintain sample order and index

   ```python
   # Test case
   dataset = create_dataset(config)
   original_index = dataset.X.index
   
   filtered = dataset.get_drift_detection_features()
   assert filtered.index.equals(original_index)
   
   # Verify chronological order maintained
   assert filtered.index.is_monotonic_increasing
   ```

5. **Empty Result Handling**: Return valid empty DataFrames for no matches

   ```python
   # Test case - Config with no continuous features
   categorical_only_config = {
       "features": [
           {"name": "cat1", "type": "categorical", "role": "feature"},
           {"name": "cat2", "type": "categorical", "role": "target"}
       ]
   }
   dataset = create_dataset(categorical_only_config)
   
   continuous = dataset.get_continuous_features()
   assert isinstance(continuous, pd.DataFrame)
   assert len(continuous.columns) == 0
   assert len(continuous) == len(dataset.X)  # Same number of rows
   ```

### REQ-009: Temporal Data Splitting

**Description**: System shall split datasets temporally while preserving drift structure and chronological order.

**Priority**: High
**Type**: Functional
**Category**: Data Access

#### Acceptance Criteria

1. **Ratio-Based Splitting**: split_temporal(ratio) creates correct train/test sizes

   ```python
   # Test case
   dataset = create_dataset({"generator_config": {"n_instances": 1000}})
   
   train, test = dataset.split_temporal(ratio=0.7)
   
   assert len(train.X) == 700
   assert len(test.X) == 300
   assert len(train.y) == 700
   assert len(test.y) == 300
   ```

2. **Chronological Order**: Training set contains first portion, test set remainder

   ```python
   # Test case
   dataset = create_dataset(temporal_config)
   
   train, test = dataset.split_temporal(ratio=0.6)
   
   # Training set: samples 0-599
   # Test set: samples 600-999
   assert train.X.index.min() == 0
   assert train.X.index.max() < test.X.index.min()
   assert test.X.index.max() == dataset.X.index.max()
   ```

3. **Drift Point Distribution**: Drift points appropriately distributed between splits

   ```python
   # Test case
   config = {
       "generator_config": {"n_instances": 1000},
       "drift_config": {"drift_points": [300, 700]}
   }
   dataset = create_dataset(config)
   
   train, test = dataset.split_temporal(ratio=0.6)  # Split at 600
   
   # Drift at 300 should be in training set
   assert 300 in train.drift_metadata.drift_points
   
   # Drift at 700 should be in test set, adjusted for split
   adjusted_test_drifts = [dp - 600 for dp in dataset.drift_metadata.drift_points if dp >= 600]
   assert 100 in test.drift_metadata.drift_points  # 700 - 600 = 100
   ```

4. **Complete Metadata Preservation**: Both splits contain valid DriftDataset objects

   ```python
   # Test case
   dataset = create_dataset(full_config)
   
   train, test = dataset.split_temporal(ratio=0.8)
   
   # Verify both are valid DriftDataset objects
   assert isinstance(train, DriftDataset)
   assert isinstance(test, DriftDataset)
   
   # Verify metadata preservation
   assert train.dataset_metadata.features == dataset.dataset_metadata.features
   assert test.dataset_metadata.features == dataset.dataset_metadata.features
   ```

5. **Feature Metadata Consistency**: All feature metadata preserved in both splits

   ```python
   # Test case
   train, test = dataset.split_temporal(ratio=0.7)
   
   # Feature metadata should be identical
   for feature_name in dataset.X.columns:
       original_info = dataset.get_feature_info(feature_name)
       train_info = train.get_feature_info(feature_name)
       test_info = test.get_feature_info(feature_name)
       
       assert train_info.type == original_info.type
       assert train_info.role == original_info.role
       assert test_info.type == original_info.type
       assert test_info.role == original_info.role
   ```

## Data Model and Validation Requirements

### REQ-010: Unified Dataset Structure

**Description**: System shall provide unified DriftDataset objects containing X, y, and comprehensive metadata.

**Priority**: Critical
**Type**: Functional
**Category**: Data Model

#### Acceptance Criteria

1. **Correct Data Types**: X is DataFrame, y is Series with matching lengths

   ```python
   # Test case
   dataset = create_dataset(config)
   
   assert isinstance(dataset.X, pd.DataFrame)
   assert isinstance(dataset.y, pd.Series)
   assert dataset.X.shape[0] == len(dataset.y)
   ```

2. **Metadata Objects**: dataset_metadata and drift_metadata properly populated

   ```python
   # Test case
   dataset = create_dataset(config)
   
   assert hasattr(dataset, 'dataset_metadata')
   assert hasattr(dataset, 'drift_metadata')
   assert isinstance(dataset.dataset_metadata, DatasetMetadata)
   assert isinstance(dataset.drift_metadata, DriftMetadata)
   ```

3. **Required Fields**: name, source_type, generator_info, config populated

   ```python
   # Test case
   config = {
       "dataset": {"name": "test_dataset", "type": "synthetic", "generator": "SineGenerator"}
   }
   dataset = create_dataset(config)
   
   assert dataset.name == "test_dataset"
   assert dataset.source_type == "synthetic"
   assert dataset.generator_info is not None
   assert dataset.config == config
   ```

4. **Immutability**: Object immutable after creation

   ```python
   # Test case
   dataset = create_dataset(config)
   
   # Attempting to modify should raise error or be ignored
   with pytest.raises(AttributeError):
       dataset.X = pd.DataFrame()  # Should not be settable
   
   # Or verify data doesn't change
   original_X = dataset.X.copy()
   # Attempt modification
   try:
       dataset.X.iloc[0, 0] = 999999
   except:
       pass
   # Verify no change
   assert dataset.X.equals(original_X) or dataset.X.iloc[0, 0] != 999999
   ```

### REQ-011: Data Consistency Validation

**Description**: System shall enforce data consistency rules across all dataset components.

**Priority**: Critical
**Type**: Functional
**Category**: Validation

#### Acceptance Criteria

1. **Sample Count Matching**: X.shape[0] == y.shape[0]

   ```python
   # Test case
   dataset = create_dataset(config)
   assert dataset.X.shape[0] == len(dataset.y)
   ```

2. **Drift Point Bounds**: All drift_points in valid range [0, X.shape[0])

   ```python
   # Test case
   dataset = create_dataset(config_with_drift)
   
   for drift_point in dataset.drift_metadata.drift_points:
       assert 0 <= drift_point < len(dataset.X)
   ```

3. **Drift Array Consistency**: Drift arrays have matching lengths

   ```python
   # Test case
   dataset = create_dataset(config_with_drift)
   drift_meta = dataset.drift_metadata
   
   assert len(drift_meta.drift_types) == len(drift_meta.drift_points)
   assert len(drift_meta.drift_patterns) == len(drift_meta.drift_points)
   ```

4. **Target Feature Correspondence**: Target features correspond to y values

   ```python
   # Test case
   dataset = create_dataset(supervised_config)
   
   target_features = [f for f in dataset.dataset_metadata.features if f.role == "target"]
   assert len(target_features) >= 1  # At least one target
   
   # For single target, y should match target column values
   if len(target_features) == 1:
       target_name = target_features[0].name
       # Target column should not be in X but correspond to y
       assert target_name not in dataset.X.columns
   ```

5. **Feature Exclusion Consistency**: Excluded features not in X columns

   ```python
   # Test case
   config_with_exclusions = {
       "features": [
           {"name": "include", "type": "continuous", "role": "feature"},
           {"name": "exclude", "type": "continuous", "role": "exclude"},
           {"name": "target", "type": "categorical", "role": "target"}
       ]
   }
   dataset = create_dataset(config_with_exclusions)
   
   assert "include" in dataset.X.columns
   assert "exclude" not in dataset.X.columns
   assert "target" not in dataset.X.columns
   ```

6. **Class Count Validation**: n_classes matches unique y values for classification

   ```python
   # Test case
   config = {
       "metadata": {"n_classes": 3, "labeling": "supervised"},
       "generator_config": {"n_classes": 3}
   }
   dataset = create_dataset(config)
   
   unique_classes = len(dataset.y.unique())
   expected_classes = dataset.dataset_metadata.n_classes
   assert unique_classes == expected_classes
   ```

## Performance and Quality Requirements

### REQ-014: Performance Benchmarks

**Description**: System shall meet specified performance targets for dataset generation and manipulation.

**Priority**: Medium
**Type**: Performance
**Category**: Quality

#### Acceptance Criteria

1. **100K Sample Generation**: Complete in <30 seconds

   ```python
   # Performance test
   import time
   
   config = {"generator_config": {"n_instances": 100000}}
   
   start_time = time.time()
   dataset = create_dataset(config)
   elapsed = time.time() - start_time
   
   assert elapsed < 30, f"Generation took {elapsed:.1f}s, expected <30s"
   ```

2. **UCI Dataset Loading**: Load up to 1M samples in <10 seconds

   ```python
   # Performance test
   import time
   
   large_uci_config = {"uci_config": {"dataset_id": 31}}  # Forest Cover dataset
   
   start_time = time.time()
   dataset = create_dataset(large_uci_config)
   elapsed = time.time() - start_time
   
   if len(dataset.X) <= 1000000:
       assert elapsed < 10, f"Loading took {elapsed:.1f}s, expected <10s"
   ```

3. **Memory Usage**: Use <2GB memory for 1M sample datasets

   ```python
   # Memory test (requires psutil)
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   
   # Measure baseline memory
   baseline_memory = process.memory_info().rss
   
   # Generate large dataset
   config = {"generator_config": {"n_instances": 1000000}}
   dataset = create_dataset(config)
   
   # Measure memory after generation
   peak_memory = process.memory_info().rss
   memory_used = peak_memory - baseline_memory
   
   # Should use less than 2GB (2 * 1024^3 bytes)
   assert memory_used < 2 * 1024**3, f"Used {memory_used / 1024**3:.1f}GB, expected <2GB"
   ```

4. **Configuration Parsing**: Parse TOML configs in <1 second

   ```python
   # Performance test
   import time
   
   complex_config = create_complex_toml_config()  # Large configuration
   
   start_time = time.time()
   config = parse_toml_config(complex_config)
   elapsed = time.time() - start_time
   
   assert elapsed < 1.0, f"Parsing took {elapsed:.1f}s, expected <1s"
   ```

5. **Feature Filtering**: Complete in <5 seconds for 1M samples

   ```python
   # Performance test
   import time
   
   large_dataset = create_dataset({"generator_config": {"n_instances": 1000000}})
   
   start_time = time.time()
   filtered = large_dataset.get_drift_detection_features()
   elapsed = time.time() - start_time
   
   assert elapsed < 5.0, f"Filtering took {elapsed:.1f}s, expected <5s"
   ```

6. **Temporal Splitting**: Preserve performance for large datasets

   ```python
   # Performance test
   import time
   
   large_dataset = create_dataset({"generator_config": {"n_instances": 1000000}})
   
   start_time = time.time()
   train, test = large_dataset.split_temporal(ratio=0.7)
   elapsed = time.time() - start_time
   
   assert elapsed < 10.0, f"Splitting took {elapsed:.1f}s, expected <10s"
   ```

### REQ-015: Error Handling and Logging

**Description**: System shall provide comprehensive error handling with informative messages and logging.

**Priority**: High
**Type**: Quality
**Category**: Error Handling

#### Acceptance Criteria

1. **File Errors**: FileNotFoundError for missing configuration files

   ```python
   # Test case
   with pytest.raises(FileNotFoundError) as exc_info:
       create_dataset("nonexistent_config.toml")
   
   error_msg = str(exc_info.value)
   assert "nonexistent_config.toml" in error_msg
   ```

2. **Validation Errors**: ValueError for invalid configurations

   ```python
   # Test case
   invalid_config = {"dataset": {"type": "invalid_type"}}
   
   with pytest.raises(ValueError) as exc_info:
       create_dataset(invalid_config)
   
   error_msg = str(exc_info.value)
   assert "dataset.type" in error_msg
   assert "invalid_type" in error_msg
   ```

3. **Connection Errors**: ConnectionError for external API failures

   ```python
   # Test case (requires network mocking)
   with mock.patch('ucimlrepo.fetch_ucirepo', side_effect=ConnectionError("Network error")):
       uci_config = {"uci_config": {"dataset_id": 321}}
       
       with pytest.raises(ConnectionError) as exc_info:
           create_dataset(uci_config)
       
       assert "Network error" in str(exc_info.value)
   ```

4. **Index Errors**: IndexError for invalid sample indices

   ```python
   # Test case
   dataset = create_dataset(config)
   
   with pytest.raises(IndexError):
       dataset.get_drift_type_at(-1)  # Negative index
   
   with pytest.raises(IndexError):
       dataset.get_drift_type_at(len(dataset.X))  # Out of bounds
   ```

5. **Runtime Errors**: RuntimeError for CapyMOA generation failures

   ```python
   # Test case (requires CapyMOA mocking)
   with mock.patch('capymoa.SineGenerator', side_effect=RuntimeError("Java error")):
       sine_config = {
           "dataset": {"type": "synthetic", "generator": "SineGenerator"},
           "generator_config": {"n_instances": 1000}
       }
       
       with pytest.raises(RuntimeError) as exc_info:
           create_dataset(sine_config)
       
       error_msg = str(exc_info.value)
       assert "Java error" in error_msg
   ```

6. **Informative Logging**: INFO and DEBUG level logging for operations

   ```python
   # Test case
   import logging
   from io import StringIO
   
   log_capture = StringIO()
   handler = logging.StreamHandler(log_capture)
   logger = logging.getLogger('drift_datasets')
   logger.addHandler(handler)
   logger.setLevel(logging.INFO)
   
   dataset = create_dataset(config)
   
   log_output = log_capture.getvalue()
   assert "Dataset generation" in log_output or "Generation progress" in log_output
   ```

## Reproducibility and Testing Requirements

### REQ-016: Deterministic Dataset Generation

**Description**: System shall generate identical datasets given identical configurations and random seeds.

**Priority**: Critical
**Type**: Quality
**Category**: Reproducibility

#### Acceptance Criteria

1. **Bit-Identical X, y Matrices**: Same config + seed = identical data

   ```python
   # Test case
   config = {
       "dataset": {"type": "synthetic", "generator": "SineGenerator"},
       "generator_config": {"n_instances": 1000, "random_seed": 42}
   }
   
   dataset1 = create_dataset(config)
   dataset2 = create_dataset(config)
   
   # DataFrames should be identical
   pd.testing.assert_frame_equal(dataset1.X, dataset2.X)
   pd.testing.assert_series_equal(dataset1.y, dataset2.y)
   ```

2. **Consistent Drift Points**: Drift occurs at exactly same sample indices

   ```python
   # Test case
   config_with_drift = {
       "generator_config": {"random_seed": 123},
       "drift_config": {"drift_points": [500, 1500]}
   }
   
   dataset1 = create_dataset(config_with_drift)
   dataset2 = create_dataset(config_with_drift)
   
   assert dataset1.drift_metadata.drift_points == dataset2.drift_metadata.drift_points
   ```

3. **CapyMOA Determinism**: CapyMOA generators produce deterministic output

   ```python
   # Test case for each supported generator
   generators = ["SineGenerator", "HyperplaneGenerator", "STAGGERGenerator", 
                 "SEAGenerator", "AgrawalGenerator", "RandomTreeGenerator", "LEDGenerator"]
   
   for generator_name in generators:
       config = {
           "dataset": {"type": "synthetic", "generator": generator_name},
           "generator_config": {"n_instances": 100, "random_seed": 999}
       }
       
       dataset1 = create_dataset(config)
       dataset2 = create_dataset(config)
       
       pd.testing.assert_frame_equal(dataset1.X, dataset2.X, 
                                   check_dtype=False)  # Allow minor type differences
   ```

4. **UCI Dataset Consistency**: UCI loading returns consistent results

   ```python
   # Test case
   uci_config = {"uci_config": {"dataset_id": 321}}
   
   dataset1 = create_dataset(uci_config)
   dataset2 = create_dataset(uci_config)
   
   # Data should be identical (excluding external changes)
   pd.testing.assert_frame_equal(dataset1.X, dataset2.X)
   pd.testing.assert_series_equal(dataset1.y, dataset2.y)
   ```

5. **Configuration Parsing Consistency**: Same TOML = same internal representation

   ```python
   # Test case
   toml_content = """
   [dataset]
   name = "test"
   type = "synthetic"
   
   [generator_config]
   n_instances = 1000
   random_seed = 42
   """
   
   # Parse multiple times
   config1 = parse_toml_string(toml_content)
   config2 = parse_toml_string(toml_content)
   
   assert config1 == config2
   ```

### REQ-017: Comprehensive Test Coverage

**Description**: System shall maintain high test coverage with functional tests for all user workflows.

**Priority**: High
**Type**: Quality
**Category**: Testing

#### Test Coverage Requirements

1. **Functional Test Coverage**: All REQ-XXX requirements have corresponding tests

   ```python
   # Example functional test structure
   class TestREQ001SyntheticGeneration:
       def test_generator_support(self):
           """Test all supported CapyMOA generators"""
           pass
       
       def test_size_matching(self):
           """Test n_instances matching"""
           pass
       
       def test_reproducibility(self):
           """Test random seed determinism"""
           pass
   ```

2. **Integration Test Coverage**: External dependencies validated

   ```python
   # Example integration tests
   class TestCapyMOAIntegration:
       def test_java_runtime_available(self):
           """Test Java runtime integration"""
           pass
       
       def test_generator_initialization(self):
           """Test CapyMOA generator creation"""
           pass
   
   class TestUCIIntegration:
       def test_api_connectivity(self):
           """Test UCI API access"""
           pass
   ```

3. **Unit Test Coverage**: Error cases and edge conditions

   ```python
   # Example unit tests
   class TestConfigValidation:
       def test_invalid_dataset_type(self):
           """Test error handling for invalid types"""
           pass
       
       def test_missing_required_sections(self):
           """Test missing configuration sections"""
           pass
   ```

4. **Coverage Metrics**: >=90% coverage for production code

   ```bash
   # Coverage measurement
   pytest --cov=drift_datasets --cov-report=html --cov-fail-under=90
   ```

5. **Realistic Test Data**: All public APIs tested with realistic scenarios

   ```python
   # Test data requirements
   class TestDataRealism:
       def test_realistic_drift_scenarios(self):
           """Test with realistic drift patterns"""
           pass
       
       def test_real_world_dataset_sizes(self):
           """Test with realistic dataset sizes"""
           pass
   ```

6. **Offline Testing**: Mock external dependencies for reliable testing

   ```python
   # Example mocking for offline tests
   @mock.patch('capymoa.SineGenerator')
   def test_synthetic_generation_mocked(self, mock_generator):
       """Test synthetic generation with mocked CapyMOA"""
       pass
   
   @mock.patch('ucimlrepo.fetch_ucirepo')
   def test_uci_loading_mocked(self, mock_fetch):
       """Test UCI loading with mocked API"""
       pass
   ```

This comprehensive requirements specification provides the foundation for implementing and validating all drift-datasets functionality. Each requirement includes detailed acceptance criteria and test cases to ensure reliable, well-tested implementation.
