"""
Synthetic data generators for drift_datasets library.

This module provides the interface to CapyMOA generators and handles
synthetic dataset generation with drift patterns.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class CapyMOAService:
    """Mock CapyMOA service for testing and development."""

    def __init__(self):
        self.generators = {
            "SineGenerator": self._create_sine_generator,
            "HyperplaneGenerator": self._create_hyperplane_generator,
            "STAGGERGenerator": self._create_stagger_generator,
            "SEAGenerator": self._create_sea_generator,
        }

    def create_generator(self, generator_name: str, parameters: Dict[str, Any]):
        """Create a generator instance with specified parameters."""
        if generator_name not in self.generators:
            raise ValueError(f"Unsupported generator: {generator_name}")

        return self.generators[generator_name](parameters)

    def _create_sine_generator(self, parameters: Dict[str, Any]):
        """Create Sine dataset generator."""
        return SineGenerator(parameters)

    def _create_hyperplane_generator(self, parameters: Dict[str, Any]):
        """Create Hyperplane dataset generator."""
        return HyperplaneGenerator(parameters)

    def _create_stagger_generator(self, parameters: Dict[str, Any]):
        """Create STAGGER dataset generator."""
        return STAGGERGenerator(parameters)

    def _create_sea_generator(self, parameters: Dict[str, Any]):
        """Create SEA dataset generator."""
        return SEAGenerator(parameters)


class SineGenerator:
    """Sine dataset generator."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.random_seed = parameters.get("random_seed", 42)
        self.n_instances = parameters.get("n_instances", 1000)
        self.noise_level = parameters.get("noise_level", 0.1)
        self.classification_function = parameters.get("classification_function", 1)  # Accept but ignore for now

        np.random.seed(self.random_seed)

    def generate(self, drift_points: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate Sine dataset with optional drift points."""
        # Generate features
        X = pd.DataFrame({"feature_0": np.random.uniform(-1, 1, self.n_instances), "feature_1": np.random.uniform(-1, 1, self.n_instances)})

        # Generate targets based on sine function
        y_values = []
        for i in range(self.n_instances):
            # Basic sine function classification
            value = np.sin(X.iloc[i, 0]) + np.sin(X.iloc[i, 1])

            # Add drift at specified points
            if drift_points:
                for drift_point in drift_points:
                    if i >= drift_point:
                        # Introduce concept drift by changing the decision boundary
                        value = -value
                        break

            # Add noise
            value += np.random.normal(0, self.noise_level)

            # Convert to binary classification
            y_values.append(1 if value > 0 else 0)

        y = pd.Series(y_values, name="target")

        return X, y


class HyperplaneGenerator:
    """Hyperplane dataset generator."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.random_seed = parameters.get("random_seed", 42)
        self.n_instances = parameters.get("n_instances", 1000)
        self.n_features = parameters.get("n_features", 10)

        np.random.seed(self.random_seed)

    def generate(self, drift_points: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate Hyperplane dataset with optional drift points."""
        # Generate features
        feature_names = [f"feature_{i}" for i in range(self.n_features)]
        X = pd.DataFrame(np.random.randn(self.n_instances, self.n_features), columns=feature_names)

        # Generate hyperplane weights
        weights = np.random.randn(self.n_features)

        # Generate targets
        y_values = []
        for i in range(self.n_instances):
            # Compute hyperplane decision
            decision_value = np.dot(X.iloc[i].values, weights)

            # Add drift at specified points
            if drift_points:
                for drift_point in drift_points:
                    if i >= drift_point:
                        # Introduce drift by changing weights
                        weights = -weights * 0.9  # Reverse and scale
                        break

            y_values.append(1 if decision_value > 0 else 0)

        y = pd.Series(y_values, name="target")

        return X, y


class STAGGERGenerator:
    """STAGGER concept dataset generator."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.random_seed = parameters.get("random_seed", 42)
        self.n_instances = parameters.get("n_instances", 1000)

        np.random.seed(self.random_seed)

    def generate(self, drift_points: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate STAGGER dataset with concept drift."""
        # Generate 3 categorical features (size, color, shape)
        sizes = np.random.choice(["small", "medium", "large"], self.n_instances)
        colors = np.random.choice(["red", "green", "blue"], self.n_instances)
        shapes = np.random.choice(["circle", "square", "triangle"], self.n_instances)

        X = pd.DataFrame({"feature_0": pd.Categorical(sizes), "feature_1": pd.Categorical(colors), "feature_2": pd.Categorical(shapes)})

        # Convert to numeric for processing
        X_numeric = pd.get_dummies(X)

        # STAGGER concepts
        concepts = [
            lambda s, c, sh: (s == "small" and c == "red"),  # Concept 1
            lambda s, c, sh: (c == "green" or sh == "circle"),  # Concept 2
            lambda s, c, sh: (s == "medium" or s == "large"),  # Concept 3
        ]

        # Generate targets based on active concept
        y_values = []
        current_concept = 0

        for i in range(self.n_instances):
            # Check for concept drift
            if drift_points:
                for j, drift_point in enumerate(drift_points):
                    if i >= drift_point:
                        current_concept = (j + 1) % len(concepts)

            # Apply current concept
            concept_func = concepts[current_concept]
            result = concept_func(sizes[i], colors[i], shapes[i])
            y_values.append(1 if result else 0)

        y = pd.Series(y_values, name="target")

        # Return numeric version for consistency
        return X_numeric, y


class SEAGenerator:
    """SEA dataset generator."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.random_seed = parameters.get("random_seed", 42)
        self.n_instances = parameters.get("n_instances", 1000)
        self.threshold = parameters.get("threshold", 8.0)

        np.random.seed(self.random_seed)

    def generate(self, drift_points: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate SEA dataset with threshold-based drift."""
        # Generate 3 features
        X = pd.DataFrame(
            {
                "feature_0": np.random.uniform(0, 10, self.n_instances),
                "feature_1": np.random.uniform(0, 10, self.n_instances),
                "feature_2": np.random.uniform(0, 10, self.n_instances),
            }
        )

        # Generate targets based on threshold
        y_values = []
        current_threshold = self.threshold

        for i in range(self.n_instances):
            # Check for drift
            if drift_points:
                for drift_point in drift_points:
                    if i >= drift_point:
                        # Change threshold to create concept drift
                        current_threshold = 10 - self.threshold
                        break

            # SEA concept: f1 + f2 <= threshold
            decision_value = X.iloc[i, 0] + X.iloc[i, 1]
            y_values.append(1 if decision_value <= current_threshold else 0)

        y = pd.Series(y_values, name="target")

        return X, y


class CapyMOAInterface:
    """Interface to CapyMOA library for synthetic dataset generation."""

    def __init__(self):
        self.service = CapyMOAService()
        self.parameter_mappings = {
            "SineGenerator": {"n_instances": "numInstances", "random_seed": "randomSeed", "noise_level": "noiseLevel"},
            "HyperplaneGenerator": {"n_instances": "numInstances", "random_seed": "randomSeed", "n_features": "numFeatures"},
        }

    def create_generator(self, generator_name: str, parameters: Dict[str, Any]):
        """Create a generator with parameter validation."""
        # Validate generator exists
        if generator_name not in ["SineGenerator", "HyperplaneGenerator", "STAGGERGenerator", "SEAGenerator"]:
            raise ValueError(f"Unsupported generator: {generator_name}")

        # Validate parameters
        self._validate_parameters(generator_name, parameters)

        return self.service.create_generator(generator_name, parameters)

    def translate_parameters(self, generator_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Translate our parameter names to CapyMOA parameter names."""
        if generator_name not in self.parameter_mappings:
            return parameters

        mapping = self.parameter_mappings[generator_name]
        translated = {}

        for our_name, capymoa_name in mapping.items():
            if our_name in parameters:
                translated[capymoa_name] = parameters[our_name]

        # Add any unmapped parameters
        for key, value in parameters.items():
            if key not in mapping:
                translated[key] = value

        return translated

    def get_generator_parameters(self, generator_name: str) -> List[str]:
        """Get list of available parameters for a generator."""
        if generator_name == "SineGenerator":
            return ["n_instances", "random_seed", "noise_level", "classification_function"]
        elif generator_name == "HyperplaneGenerator":
            return ["n_instances", "random_seed", "n_features"]
        elif generator_name == "STAGGERGenerator":
            return ["n_instances", "random_seed"]
        elif generator_name == "SEAGenerator":
            return ["n_instances", "random_seed", "threshold"]
        else:
            return []

    def check_availability(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if CapyMOA is available and return version info."""
        # For testing purposes, always return available
        return True, {"version": "1.0.0", "generators": list(self.service.generators.keys())}

    def _validate_parameters(self, generator_name: str, parameters: Dict[str, Any]):
        """Validate parameters for a specific generator."""
        available_params = self.get_generator_parameters(generator_name)

        for param_name in parameters.keys():
            if param_name not in available_params:
                raise ValueError(f"Invalid parameter '{param_name}' for generator {generator_name}")
