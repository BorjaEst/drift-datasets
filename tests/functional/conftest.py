"""
Functional test fixtures and configuration.

This module provides feature-specific fixtures for functional tests,
building on the session-scoped fixtures from the main conftest.py.
"""

from pathlib import Path

import pytest


@pytest.fixture
def mock_drift_detector():
    """Provide a mock drift detector for testing integration."""

    class MockDriftDetector:
        def __init__(self):
            self.detected_drifts = []
            self.sensitivity = 0.01

        def update(self, features, target):
            """Process new instance and detect drift."""
            # Simple mock logic: detect drift at instance 500
            if len(self.detected_drifts) == 0 and hasattr(self, "_instance_count"):
                if self._instance_count == 500:
                    self.detected_drifts.append(self._instance_count)

            if not hasattr(self, "_instance_count"):
                self._instance_count = 0
            self._instance_count += 1

        def has_drift(self):
            """Check if drift was detected."""
            return len(self.detected_drifts) > 0

        def get_drift_points(self):
            """Get detected drift points."""
            return self.detected_drifts.copy()

    return MockDriftDetector()


@pytest.fixture
def sample_sklearn_workflow():
    """Provide sample sklearn workflow components."""
    return {
        "preprocessing": {"steps": ["StandardScaler", "PCA"], "parameters": {"StandardScaler": {}, "PCA": {"n_components": 2}}},
        "model": {"algorithm": "LogisticRegression", "parameters": {"random_state": 42, "max_iter": 1000}},
        "evaluation": {"cv_folds": 5, "metrics": ["accuracy", "f1_score", "roc_auc"]},
    }


@pytest.fixture
def sample_pytorch_config():
    """Provide sample PyTorch configuration."""
    return {
        "dataloader": {"batch_size": 32, "shuffle": True, "num_workers": 0},  # For testing
        "model": {"architecture": "MLP", "layers": [10, 5, 2], "activation": "ReLU", "dropout": 0.1},
        "training": {"epochs": 10, "learning_rate": 0.001, "optimizer": "Adam"},
    }


@pytest.fixture
def sample_river_config():
    """Provide sample River (online learning) configuration."""
    return {
        "model": {"algorithm": "LogisticRegression", "parameters": {"optimizer": "SGD", "l2": 0.01}},
        "metrics": ["Accuracy", "F1Score", "Recall"],
        "evaluation": {"window_size": 100, "drift_detection": True},
    }


@pytest.fixture
def integration_test_data():
    """Provide test data specifically for integration tests."""
    import numpy as np
    import pandas as pd

    # Generate synthetic data that works well with various ML libraries
    np.random.seed(42)
    n_samples = 1000

    # Features: mix of continuous and discrete
    X = pd.DataFrame(
        {
            "continuous_1": np.random.normal(0, 1, n_samples),
            "continuous_2": np.random.normal(1, 0.5, n_samples),
            "discrete_1": np.random.choice([0, 1, 2], n_samples),
            "discrete_2": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
    )

    # Target: binary classification
    y = pd.Series((X["continuous_1"] + X["continuous_2"] > 0.5).astype(int), name="target")

    return {
        "X": X,
        "y": y,
        "feature_types": {"continuous_1": "continuous", "continuous_2": "continuous", "discrete_1": "ordinal", "discrete_2": "binary"},
    }


@pytest.fixture
def mock_external_libraries():
    """Mock external library imports for testing integration points."""

    class MockSKLearn:
        class LinearModel:
            class LogisticRegression:
                def __init__(self, **kwargs):
                    self.params = kwargs
                    self.fitted = False

                def fit(self, X, y):
                    self.fitted = True
                    return self

                def predict(self, X):
                    if not self.fitted:
                        raise ValueError("Model not fitted")
                    return [0] * len(X)

                def predict_proba(self, X):
                    if not self.fitted:
                        raise ValueError("Model not fitted")
                    return [[0.6, 0.4] for _ in range(len(X))]

        class ModelSelection:
            @staticmethod
            def train_test_split(X, y, **kwargs):
                split_idx = int(len(X) * 0.8)
                return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    class MockTorch:
        class Utils:
            class Data:
                class Dataset:
                    def __init__(self, X, y):
                        self.X = X
                        self.y = y

                    def __len__(self):
                        return len(self.X)

                    def __getitem__(self, idx):
                        return self.X[idx], self.y[idx]

                class DataLoader:
                    def __init__(self, dataset, **kwargs):
                        self.dataset = dataset
                        self.batch_size = kwargs.get("batch_size", 32)

                    def __iter__(self):
                        for i in range(0, len(self.dataset), self.batch_size):
                            batch_X = [self.dataset[j][0] for j in range(i, min(i + self.batch_size, len(self.dataset)))]
                            batch_y = [self.dataset[j][1] for j in range(i, min(i + self.batch_size, len(self.dataset)))]
                            yield batch_X, batch_y

    class MockRiver:
        class LinearModel:
            class LogisticRegression:
                def __init__(self, **kwargs):
                    self.params = kwargs

                def predict_one(self, x):
                    return 0 if sum(x.values()) < 0 else 1

                def learn_one(self, x, y):
                    pass

        class Metrics:
            class Accuracy:
                def __init__(self):
                    self.total = 0
                    self.correct = 0

                def update(self, y_true, y_pred):
                    self.total += 1
                    if y_true == y_pred:
                        self.correct += 1

                def get(self):
                    return self.correct / self.total if self.total > 0 else 0.0

    return {"sklearn": MockSKLearn(), "torch": MockTorch(), "river": MockRiver()}


@pytest.fixture
def integration_error_scenarios():
    """Provide error scenarios specific to integration testing."""
    return {
        "sklearn_integration": {
            "incompatible_data_types": {
                "features": ["string_feature", "complex_feature"],
                "error_type": "TypeError",
                "expected_message": "Input contains unsupported data type",
            },
            "missing_target_values": {"target_has_nan": True, "error_type": "ValueError", "expected_message": "Target contains NaN values"},
        },
        "pytorch_integration": {
            "tensor_dimension_mismatch": {
                "expected_dims": 2,
                "actual_dims": 3,
                "error_type": "RuntimeError",
                "expected_message": "Dimension mismatch",
            },
            "batch_size_mismatch": {"dataloader_batch_size": 32, "model_expected_batch": 64, "error_type": "RuntimeError"},
        },
        "river_integration": {
            "streaming_interruption": {
                "interruption_point": 500,
                "error_type": "ConnectionError",
                "recovery_strategy": "checkpoint_and_resume",
            },
            "concept_drift_handling": {"drift_points": [300, 600, 900], "adaptation_required": True},
        },
    }


@pytest.fixture
def performance_test_configs():
    """Provide configurations for performance testing."""
    return {
        "small_dataset": {"size": 1000, "features": 5, "max_generation_time": 2.0, "max_memory_usage": 50},  # seconds  # MB
        "medium_dataset": {"size": 10000, "features": 10, "max_generation_time": 10.0, "max_memory_usage": 200},
        "large_dataset": {"size": 100000, "features": 20, "max_generation_time": 60.0, "max_memory_usage": 1000},
        "streaming_dataset": {
            "size": 1000000,
            "chunk_size": 1000,
            "max_chunk_processing_time": 0.1,
            "max_memory_usage": 100,  # Should remain constant
        },
    }


@pytest.fixture
def benchmark_data():
    """Provide benchmark data for comparison testing."""
    return {
        "baseline_performance": {
            "sine_generator_1k": {"time": 0.5, "memory": 25},
            "hyperplane_generator_1k": {"time": 0.7, "memory": 30},
            "uci_electricity_load": {"time": 2.0, "memory": 100},
            "mixed_dataset_1k": {"time": 1.2, "memory": 40},
        },
        "regression_thresholds": {
            "performance_degradation_max": 0.2,  # 20% slower than baseline
            "memory_increase_max": 0.3,  # 30% more memory than baseline
        },
        "quality_metrics": {"min_accuracy_synthetic": 0.85, "min_drift_detection_precision": 0.9, "min_data_completeness": 0.99},
    }
