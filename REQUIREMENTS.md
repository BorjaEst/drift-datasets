# REQUIREMENTS

## Core Dataset Generation Requirements

### REQ-001: Synthetic Dataset Creation

**Description**: System shall generate synthetic datasets with configurable drift patterns using CapyMOA generators.
**Acceptance Criteria**:

- Support SineGenerator, HyperplaneGenerator, STAGGERGenerator, RandomTreeGenerator, SEAGenerator, AgrawalGenerator, LEDGenerator
- Generated dataset size matches configured n_instances (e.g., 10000 samples)
- Random seed produces deterministic, reproducible datasets
- Generated X, y have matching lengths
- Support noise injection with configurable noise_level parameter

### REQ-002: Real-World Dataset Loading

**Description**: System shall load real-world datasets from UCI ML Repository with optional drift injection.
**Acceptance Criteria**:

- Successfully load UCI datasets by dataset_id (e.g., dataset_id=321 for electricity)
- Preserve original data characteristics and feature types
- Map UCI metadata (feature_types, task, characteristics) to internal schema
- Handle datasets with missing values according to UCI specifications
- Maintain temporal ordering when present in source data

### REQ-003: Mixed Dataset Combination

**Description**: System shall combine multiple data sources into unified datasets with consistent drift metadata.
**Acceptance Criteria**:

- Support sequential, interleaved, and hierarchical combination patterns
- Maintain data type consistency across combined components
- Preserve individual component metadata in unified structure
- Apply consistent drift parameters across mixed components
- Generate valid drift_points indices for combined dataset

## Configuration and Parsing Requirements

### REQ-004: TOML Configuration Parsing

**Description**: System shall parse and validate TOML configuration files with comprehensive error reporting.
**Acceptance Criteria**:

- Parse required sections: [dataset], [metadata], [generator_config], [drift_config]
- Validate dataset.type in ["synthetic", "real_world", "mixed"]
- Validate metadata.dimension in ["univariate", "multivariate"]
- Validate metadata.labeling in ["supervised", "unsupervised", "semi-supervised"]
- Report specific validation errors with line numbers and field names
- Raise FileNotFoundError for missing configuration files

### REQ-005: Feature Metadata Specification

**Description**: System shall support detailed feature-level metadata specification in configuration.
**Acceptance Criteria**:

- Parse [[features]] array with name, type, role, description fields
- Validate type in ["continuous", "categorical", "mixed"]
- Validate role in ["feature", "target", "timestamp", "identifier", "metadata", "exclude"]
- Support optional fields: missing_values (boolean), unique_values (integer)
- Enforce at least one feature with role="target" for supervised datasets
- Allow features with role="exclude" to be omitted from X matrix

## Drift Metadata and Ground Truth Requirements

### REQ-006: Drift Point Specification

**Description**: System shall create precise drift point metadata with ground truth information.
**Acceptance Criteria**:

- drift_points list contains valid sample indices within [0, len(X))
- drift_types list matches drift_points length with values ["covariate", "concept", "prior", "none"]
- drift_patterns list matches drift_points length with values ["abrupt", "gradual", "recurring", "incremental"]
- Optional drift_widths (CapyMOA width parameter) and drift_alphas (CapyMOA alpha parameter)
- Optional affected_features specifies feature indices impacted by each drift
- Optional concepts list provides human-readable concept labels

### REQ-007: Drift Analysis Methods

**Description**: System shall provide analysis methods for drift detection research workflows.
**Acceptance Criteria**:

- is_drift_point(idx, tolerance) returns boolean indicating proximity to drift
- get_concept_segments() returns List[Tuple[int, int]] with (start, end) indices
- get_drift_type_at(idx) returns drift type at specific sample index
- get_affected_features_at(idx) returns feature indices affected by drift at sample
- validate_drift_metadata() ensures consistency of all drift information
- Methods handle edge cases (idx out of bounds, empty drift_points)

## Data Access and Filtering Requirements

### REQ-008: Feature Role-Based Filtering

**Description**: System shall filter features based on their role for appropriate modeling workflows.
**Acceptance Criteria**:

- get_drift_detection_features() excludes features with role in ["target", "timestamp", "identifier", "metadata", "exclude"]
- get_continuous_features() returns only features with type="continuous"
- get_categorical_features() returns only features with type="categorical"
- get_features_by_role(role) returns features matching specific role
- Filtered DataFrames maintain sample order and index consistency
- Empty results return valid empty DataFrames with correct structure

### REQ-009: Temporal Data Splitting

**Description**: System shall split datasets temporally while preserving drift structure and chronological order.
**Acceptance Criteria**:

- split_temporal(ratio) creates train/test split at specified ratio (e.g., 0.7)
- Training set contains first ratio portion of samples in chronological order
- Test set contains remaining samples maintaining temporal sequence
- Drift points distributed appropriately between train/test sets
- Both splits contain valid DriftDataset objects with complete metadata
- Split preserves all feature metadata and dataset characteristics

## Data Model and Validation Requirements

### REQ-010: Unified Dataset Structure

**Description**: System shall provide unified DriftDataset objects containing X, y, and comprehensive metadata.
**Acceptance Criteria**:

- X is pandas.DataFrame with shape (n_samples, n_features)
- y is pandas.Series with shape (n_samples,) matching X length
- dataset_metadata contains DatasetMetadata with feature list and properties
- drift_metadata contains DriftMetadata with ground truth drift information
- name, source_type, generator_info, config fields populated correctly
- Object immutable after creation (no modification of core data)

### REQ-011: Data Consistency Validation

**Description**: System shall enforce data consistency rules across all dataset components.
**Acceptance Criteria**:

- X.shape[0] == y.shape[0] (matching sample counts)
- All drift_points values in range [0, X.shape[0])
- len(drift_types) == len(drift_points) == len(drift_patterns)
- Features with role="target" correspond to y values
- Features with role="exclude" do not appear in X columns
- n_classes matches unique values in y for classification tasks

## External Integration Requirements

### REQ-012: CapyMOA Integration

**Description**: System shall integrate with CapyMOA for synthetic dataset generation with native drift support.
**Acceptance Criteria**:

- Initialize CapyMOA generators with configuration parameters
- Handle CapyMOA-specific parameters (width, alpha) for drift injection
- Generate datasets with controlled concept drift at specified points
- Map CapyMOA generator outputs to pandas DataFrame/Series format
- Handle CapyMOA failures gracefully with meaningful error messages
- Require Java runtime availability for CapyMOA operations

### REQ-013: UCI Repository Integration

**Description**: System shall integrate with UCI ML Repository for real-world dataset access and metadata mapping.
**Acceptance Criteria**:

- Connect to UCI repository API with dataset_id parameter
- Load dataset features and targets into pandas structures
- Map UCI metadata fields to internal DatasetMetadata schema
- Handle UCI API failures with appropriate ConnectionError exceptions
- Support offline operation when external APIs unavailable (graceful degradation)
- Cache metadata mapping to reduce API calls during development

## Performance and Quality Requirements

### REQ-014: Performance Benchmarks

**Description**: System shall meet specified performance targets for dataset generation and manipulation.
**Acceptance Criteria**:

- Generate 100K synthetic samples in <30 seconds
- Load UCI datasets up to 1M samples in <10 seconds
- Use <2GB memory for 1M sample datasets with metadata
- Parse TOML configurations in <1 second
- Feature filtering operations complete in <5 seconds for 1M samples
- Temporal splitting preserves performance for large datasets

### REQ-015: Error Handling and Logging

**Description**: System shall provide comprehensive error handling with informative messages and logging.
**Acceptance Criteria**:

- FileNotFoundError for missing configuration files with full path
- ValueError for invalid dataset types, generator names, or malformed configs
- ConnectionError for external API failures (CapyMOA, UCI)
- IndexError for invalid sample indices in drift operations
- RuntimeError for CapyMOA generation failures with diagnostic information
- INFO level logging for dataset generation progress and timing
- DEBUG level logging for external API interactions and parameter details

## Reproducibility and Testing Requirements

### REQ-016: Deterministic Dataset Generation

**Description**: System shall generate identical datasets given identical configurations and random seeds.
**Acceptance Criteria**:

- Same TOML config with same random_seed produces bit-identical X, y matrices
- Drift points occur at exactly the same sample indices across runs
- CapyMOA generators produce deterministic output with fixed seeds
- UCI dataset loading returns consistent results (excluding external changes)
- Configuration parsing yields identical internal representations

### REQ-017: Comprehensive Test Coverage

**Description**: System shall maintain high test coverage with functional tests for all user workflows.
**Acceptance Criteria**:

- Functional tests cover all REQ-XXX requirements with concrete scenarios
- Integration tests validate CapyMOA and UCI API interactions
- Unit tests cover error cases and edge conditions
- Test coverage >= 90% for all production code
- All public APIs have corresponding test scenarios with realistic data
- Mock external dependencies for reliable offline testing

## Usability and Documentation Requirements

### REQ-018: Intuitive API Design

**Description**: System shall provide user-friendly methods for common drift detection research workflows.
**Acceptance Criteria**:

- Single entry point: create_drift_dataset(config_path) -> DriftDataset
- Convenience methods for feature access without manual filtering
- Self-documenting method names and parameters
- Comprehensive docstrings with examples for all public methods
- Type hints for all public interfaces
- Consistent naming conventions aligned with pandas/sklearn patterns

### REQ-019: Feature Discovery and Inspection

**Description**: System shall support easy exploration and understanding of dataset characteristics.
**Acceptance Criteria**:

- summarize_features() returns comprehensive feature overview
- get_feature_info(name) provides detailed metadata for specific features
- Methods to query features by type, role, or characteristics
- Human-readable descriptions for all features when available
- Clear indication of which features are suitable for modeling
- Validation of feature metadata consistency with actual data
