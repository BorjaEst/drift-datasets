"""
Shared pytest fixtures for drift-datasets test suite.

This module provides session-scoped and module-scoped fixtures that are shared
across all test modules. Following TDD principles, these fixtures provide
the foundation for testing dataset generation behavior.
"""

import os
import tempfile
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture(scope="session")
def test_seed():
    """Provide consistent random seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session") 
def temporary_directory():
    """Provide temporary directory for file export tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="session")
def sample_configurations():
    """Provide standard dataset configurations from the paper."""
    return {
        'Sine(1000)': {
            'type': 'sine',
            'context_size': 1000,
            'training_instances': 1,
            'drift_instances': 100,
            'irrelevant_attributes': 2
        },
        'Sine(5000)': {
            'type': 'sine', 
            'context_size': 5000,
            'training_instances': 10,
            'drift_instances': 100,
            'irrelevant_attributes': 2
        },
        'Stagger(1)': {
            'type': 'stagger',
            'context_size': 40,
            'training_instances': 1,
            'drift_instances': 100,
            'noise_level': 0.0
        },
        'Stagger(20)': {
            'type': 'stagger',
            'context_size': 40,
            'training_instances': 20, 
            'drift_instances': 100,
            'noise_level': 0.0
        },
        'Hyp(0.1)': {
            'type': 'hyperplane',
            'dimensions': 10,
            'change_rate': 0.1,
            'noise_level': 0.05,
            'magnitude_change': 0.0
        },
        'Hyp(0.001)': {
            'type': 'hyperplane',
            'dimensions': 10,
            'change_rate': 0.001,
            'noise_level': 0.05,
            'magnitude_change': 0.0
        },
        'Mixed(200)': {
            'type': 'mixed',
            'width': 200,
            'drift_probability': 'sigmoid',
            'noise_level': 0.0
        },
        'Mixed(1000)': {
            'type': 'mixed',
            'width': 1000,
            'drift_probability': 'sigmoid',
            'noise_level': 0.0
        }
    }


@pytest.fixture(scope="session")
def statistical_validation_params():
    """Parameters for statistical property validation."""
    return {
        'class_balance_tolerance': 0.05,  # 45%-55% acceptable range
        'sample_size_large': 10000,      # Large sample for statistical tests
        'sample_size_medium': 1000,      # Medium sample for performance tests
        'sample_size_small': 100,        # Small sample for unit tests
        'significance_level': 0.05       # Alpha for statistical tests
    }


@pytest.fixture(scope="session")
def performance_thresholds():
    """Performance thresholds for benchmark tests."""
    return {
        'min_instances_per_second': 1000,      # Minimum generation speed
        'max_memory_increase_mb': 100,         # Maximum memory increase
        'max_generation_time_ms': 1.0,         # Maximum time per instance
        'max_batch_processing_time_s': 10.0    # Maximum batch processing time
    }


@pytest.fixture(scope="module")
def mock_random_state():
    """Provide controlled random state for testing deterministic behavior."""
    def _create_mock(seed: int = 42):
        """Create a mock random state with predictable sequences."""
        np.random.seed(seed)
        mock = Mock()
        mock.uniform = Mock(side_effect=lambda low=0, high=1, size=None: 
                           np.random.uniform(low, high, size))
        mock.randint = Mock(side_effect=lambda low, high, size=None:
                           np.random.randint(low, high, size))
        mock.choice = Mock(side_effect=lambda a, size=None, p=None:
                          np.random.choice(a, size, p=p))
        mock.normal = Mock(side_effect=lambda loc=0, scale=1, size=None:
                          np.random.normal(loc, scale, size))
        return mock
    
    return _create_mock


@pytest.fixture
def sample_sine_data():
    """Provide realistic sine dataset sample for testing."""
    return {
        'features': np.array([
            [0.5, 0.3, 0.1, 0.8],    # Below sine curve: positive
            [0.2, 0.8, 0.4, 0.6],    # Above sine curve: negative  
            [0.7, 0.5, 0.2, 0.9],    # Near sine curve
            [0.1, 0.1, 0.3, 0.7],    # Clear below: positive
            [0.9, 0.9, 0.5, 0.4]     # Clear above: negative
        ]),
        'labels': np.array([1, 0, 1, 1, 0]),
        'drift_points': [1000, 2000, 3000],
        'concept_sequence': [0, 1, 0, 1]
    }


@pytest.fixture
def sample_stagger_data():
    """Provide realistic stagger dataset sample for testing."""
    return {
        'features': [
            {'shape': 'triangle', 'color': 'red', 'size': 'small'},      # Concept 1: True
            {'shape': 'circle', 'color': 'blue', 'size': 'large'},       # Concept 2: True
            {'shape': 'rectangle', 'color': 'green', 'size': 'medium'},  # Concept 3: True
            {'shape': 'triangle', 'color': 'blue', 'size': 'large'},     # Mixed
            {'shape': 'rectangle', 'color': 'red', 'size': 'medium'}     # Mixed
        ],
        'labels': np.array([1, 1, 1, 0, 0]),
        'attribute_values': {
            'shape': ['triangle', 'circle', 'rectangle'],
            'color': ['red', 'green', 'blue'],
            'size': ['small', 'medium', 'large']
        },
        'concept_definitions': {
            1: lambda shape, color, size: color == 'red' and size == 'small',
            2: lambda shape, color, size: color == 'green' or shape == 'circle',
            3: lambda shape, color, size: size == 'medium' or size == 'large'
        }
    }


@pytest.fixture
def sample_hyperplane_data():
    """Provide realistic hyperplane dataset sample for testing."""
    return {
        'dimensions': 10,
        'features': np.random.RandomState(42).uniform(0, 1, (100, 10)),
        'initial_weights': np.array([0.1, -0.2, 0.3, -0.1, 0.2, 
                                   -0.3, 0.1, 0.2, -0.1, 0.05, 0.0]),
        'change_rate': 0.1,
        'noise_level': 0.05,
        'expected_rotation': True
    }


@pytest.fixture  
def sample_mixed_data():
    """Provide realistic mixed dataset sample for testing."""
    return {
        'boolean_attributes': np.array([
            [1, 0], [0, 1], [1, 1], [0, 0], [1, 0]
        ]),
        'numerical_attributes': np.array([
            [0.3, 0.4], [0.7, 0.2], [0.5, 0.6], [0.1, 0.8], [0.9, 0.3]
        ]),
        'expected_conditions': [
            [True, False, True],    # v=1, w=0, y<0.5+0.3*sin(3π*0.3)
            [False, True, True],    # v=0, w=1, y<0.5+0.3*sin(3π*0.7)  
            [True, True, False],    # v=1, w=1, y>0.5+0.3*sin(3π*0.5)
            [False, False, False],  # v=0, w=0, y>0.5+0.3*sin(3π*0.1)
            [True, False, True]     # v=1, w=0, y<0.5+0.3*sin(3π*0.9)
        ],
        'expected_labels': [1, 1, 1, 0, 1],  # >=2 conditions satisfied
        'width': 200,
        'drift_transition_positions': [1000, 1200]
    }


@pytest.fixture
def export_format_samples():
    """Provide expected export format samples for testing."""
    return {
        'csv_header': 'x,y,a3,a4,class',
        'csv_sample_line': '0.5,0.3,0.1,0.8,1',
        'arff_header': '@relation sine_dataset',
        'arff_attributes': [
            '@attribute x numeric',
            '@attribute y numeric', 
            '@attribute a3 numeric',
            '@attribute a4 numeric',
            '@attribute class {0,1}'
        ],
        'arff_data_marker': '@data',
        'json_metadata_keys': [
            'dataset_type', 'num_attributes', 'num_classes',
            'drift_points', 'generation_timestamp'
        ]
    }


@pytest.fixture
def moa_interface_requirements():
    """Define required MOA framework interface methods."""
    return {
        'required_methods': [
            'nextInstance', 'hasMoreInstances', 'getHeader',
            'estimatedRemainingInstances', 'isRestartable', 'restart'
        ],
        'header_attributes': [
            'numAttributes', 'numClasses', 'classIndex'
        ],
        'instance_attributes': [
            'numAttributes', 'classValue', 'value'
        ]
    }


# Test data validation helpers
@pytest.fixture
def validation_helpers():
    """Provide helper functions for validating test data."""
    
    def validate_class_balance(labels: np.ndarray, tolerance: float = 0.05) -> bool:
        """Validate class distribution is approximately balanced."""
        if len(labels) == 0:
            return False
        
        positive_ratio = np.mean(labels)
        return abs(positive_ratio - 0.5) <= tolerance
    
    def validate_feature_range(features: np.ndarray, min_val: float = 0.0, 
                             max_val: float = 1.0) -> bool:
        """Validate features are within expected range."""
        return np.all(features >= min_val) and np.all(features <= max_val)
    
    def validate_drift_timing(drift_points: List[int], 
                            instance_positions: List[int]) -> bool:
        """Validate drifts occur at exact specified positions."""
        return all(pos in instance_positions for pos in drift_points)
    
    def calculate_concept_accuracy(predictions: np.ndarray, 
                                 ground_truth: np.ndarray) -> float:
        """Calculate prediction accuracy for concept validation."""
        if len(predictions) != len(ground_truth):
            return 0.0
        return np.mean(predictions == ground_truth)
    
    return {
        'validate_class_balance': validate_class_balance,
        'validate_feature_range': validate_feature_range, 
        'validate_drift_timing': validate_drift_timing,
        'calculate_concept_accuracy': calculate_concept_accuracy
    }


# Mathematical property fixtures
@pytest.fixture
def mathematical_constants():
    """Provide mathematical constants used in dataset generation."""
    return {
        'pi': np.pi,
        'sin_period': 2 * np.pi,
        'uniform_range': (0.0, 1.0),
        'hyperplane_rotation_threshold': 1e-6,
        'statistical_significance': 0.05,
        'floating_point_tolerance': 1e-10
    }


@pytest.fixture(autouse=True)
def reset_random_state():
    """Automatically reset random state before each test for reproducibility."""
    np.random.seed(42)
    yield
    # Cleanup after test if needed
