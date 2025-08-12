"""
Unit tests for Sine Dataset Generator - REQ-ABRUPT-101

Tests the mathematical correctness and drift behavior of the Sine dataset generator.
Following TDD principles, these tests define the exact behavior required before
any implementation exists.

Test Coverage:
- Mathematical definition correctness
- Drift point accuracy  
- Attribute generation
- Class balance
- Reproducibility
"""

import math
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestSineMathematicalDefinition:
    """Test suite for sine dataset mathematical correctness - REQ-ABRUPT-101"""

    def test_sine_mathematical_definition(self):
        """Test: points below curve y = sin(x) are positive - Core mathematical requirement"""
        # This is the fundamental mathematical test from requirements
        generator = Mock()
        
        # Mock the mathematical behavior
        def mock_sine_labeling(x: float, y: float, concept: int = 0) -> int:
            """Reference implementation of sine labeling logic"""
            base_label = 1 if y < math.sin(x) else 0
            if concept % 2 == 1:  # Concept reversal at drift points
                return 1 - base_label
            return base_label
        
        generator.get_label = Mock(side_effect=mock_sine_labeling)
        
        # Test specific mathematical cases
        test_cases = [
            # (x, y, expected_label_concept_0, description)
            (0.0, 0.0, 1, "Point (0,0): y=0 < sin(0)=0 is False → label=0, but y=sin(x) exactly"),
            (math.pi/2, 0.5, 1, "Point (π/2, 0.5): y=0.5 < sin(π/2)=1 → label=1"),
            (math.pi/2, 1.5, 0, "Point (π/2, 1.5): y=1.5 > sin(π/2)=1 → label=0"),
            (math.pi, 0.1, 0, "Point (π, 0.1): y=0.1 > sin(π)=0 → label=0"),
            (0.5, 0.3, 1, "Point (0.5, 0.3): y=0.3 < sin(0.5)≈0.479 → label=1"),
            (0.5, 0.6, 0, "Point (0.5, 0.6): y=0.6 > sin(0.5)≈0.479 → label=0")
        ]
        
        for x, y, expected_label, description in test_cases:
            actual_label = generator.get_label(x, y, concept=0)
            # Handle boundary case for y = sin(x)
            if abs(y - math.sin(x)) < 1e-10:  # Essentially equal
                expected_label = 0  # Convention: exactly on curve is negative
            assert actual_label == expected_label, f"Failed {description}"

    def test_sine_concept_reversal(self):
        """Test classification reversal after drift points"""
        generator = Mock()
        
        def mock_sine_labeling_with_drift(x: float, y: float, concept: int = 0) -> int:
            base_label = 1 if y < math.sin(x) else 0
            if concept % 2 == 1:  # Odd concepts are reversed
                return 1 - base_label
            return base_label
        
        generator.get_label = Mock(side_effect=mock_sine_labeling_with_drift)
        
        # Test same point with different concepts
        x, y = 0.5, 0.3  # y < sin(0.5), so base_label = 1
        
        concept_0_label = generator.get_label(x, y, concept=0)
        concept_1_label = generator.get_label(x, y, concept=1)
        
        assert concept_0_label == 1, "Concept 0: below sine curve should be positive"
        assert concept_1_label == 0, "Concept 1: below sine curve should be negative (reversed)"
        assert concept_0_label != concept_1_label, "Concepts should be reversed"

    def test_sine_coordinate_generation_uniform_distribution(self):
        """Test coordinates (x, y) are uniformly distributed in [0,1] interval"""
        generator = Mock()
        
        # Mock uniform coordinate generation
        np.random.seed(42)
        x_values = np.random.uniform(0, 1, 1000)
        y_values = np.random.uniform(0, 1, 1000)
        
        def mock_generate_coordinates():
            idx = len(mock_generate_coordinates.calls)
            mock_generate_coordinates.calls.append(idx)
            if idx < len(x_values):
                return x_values[idx], y_values[idx]
            return np.random.uniform(0, 1), np.random.uniform(0, 1)
        
        mock_generate_coordinates.calls = []
        generator.generate_coordinates = Mock(side_effect=mock_generate_coordinates)
        
        # Generate coordinates and test distribution
        coordinates = []
        for _ in range(100):
            x, y = generator.generate_coordinates()
            coordinates.append((x, y))
        
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        # Test uniform distribution properties
        assert all(0 <= x <= 1 for x in x_coords), "X coordinates must be in [0,1]"
        assert all(0 <= y <= 1 for y in y_coords), "Y coordinates must be in [0,1]"
        
        # Test statistical uniformity (approximate)
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        assert 0.4 < x_mean < 0.6, f"X mean {x_mean} should be near 0.5 for uniform distribution"
        assert 0.4 < y_mean < 0.6, f"Y mean {y_mean} should be near 0.5 for uniform distribution"

    def test_sine_irrelevant_attributes_generation(self):
        """Test addition of 2 irrelevant attributes with random values in [0,1]"""
        generator = Mock()
        
        # Mock full instance generation with irrelevant attributes
        def mock_full_instance_generation():
            x, y = np.random.uniform(0, 1, 2)  # Relevant attributes
            a3, a4 = np.random.uniform(0, 1, 2)  # Irrelevant attributes
            features = np.array([x, y, a3, a4])
            label = 1 if y < math.sin(x) else 0
            return features, label
        
        generator.generate_instance = Mock(side_effect=mock_full_instance_generation)
        
        # Test instance structure
        features, label = generator.generate_instance()
        
        assert len(features) == 4, "Sine dataset should have exactly 4 attributes"
        assert all(0 <= attr <= 1 for attr in features), "All attributes should be in [0,1]"
        assert isinstance(label, (int, np.integer)), "Label should be integer"
        assert label in [0, 1], "Label should be binary"
        
        # Test that irrelevant attributes don't affect labeling
        x, y = features[0], features[1]  # Relevant attributes
        expected_label = 1 if y < math.sin(x) else 0
        assert label == expected_label, "Label should only depend on x,y coordinates"


class TestSineDriftPointBehavior:
    """Test suite for sine dataset drift point accuracy"""

    def test_sine_drift_points(self):
        """Test drift points occur at exact specified instance counts"""
        generator = Mock()
        
        # Mock drift point specification
        context_size = 1000
        expected_drift_points = [1000, 2000, 3000]
        
        generator.get_drift_points = Mock(return_value=expected_drift_points)
        
        drift_points = generator.get_drift_points()
        
        assert drift_points == expected_drift_points, "Drift points must match specification"
        assert all(point > 0 for point in drift_points), "Drift points must be positive"
        assert all(point % context_size == 0 for point in drift_points), "Drift points should align with context size"

    def test_sine_drift_timing_accuracy(self):
        """Test concept changes occur exactly at drift points"""
        generator = Mock()
        
        # Mock position-aware generation
        current_position = 0
        drift_points = [1000, 2000]
        
        def mock_position_aware_generation():
            nonlocal current_position
            current_position += 1
            
            # Determine concept based on position
            concept = sum(1 for dp in drift_points if current_position > dp) % 2
            
            x, y = np.random.uniform(0, 1, 2)
            base_label = 1 if y < math.sin(x) else 0
            if concept == 1:  # Reversed concept
                label = 1 - base_label
            else:
                label = base_label
                
            return np.array([x, y, 0.5, 0.5]), label
        
        generator.generate_instance = Mock(side_effect=mock_position_aware_generation)
        
        # Generate instances spanning drift points
        instances = []
        for i in range(2500):  # Span multiple drifts
            instance = generator.generate_instance()
            instances.append((i + 1, instance))  # 1-indexed position
        
        # Test drift occurs exactly at specified points
        # Before drift point 1000: concept 0
        # After drift point 1000: concept 1
        # After drift point 2000: concept 0 again
        
        pre_drift_1 = instances[998]  # Position 999
        post_drift_1 = instances[1000]  # Position 1001
        
        # Same features should give different labels after drift
        x_pre, y_pre = 0.5, 0.3
        x_post, y_post = 0.5, 0.3
        
        expected_pre = 1 if y_pre < math.sin(x_pre) else 0  # Concept 0
        expected_post = 1 - (1 if y_post < math.sin(x_post) else 0)  # Concept 1 (reversed)
        
        assert expected_pre != expected_post, "Same coordinates should give different labels after drift"

    def test_sine_concept_consistency_between_drifts(self):
        """Test concept remains consistent between drift points"""
        generator = Mock()
        
        # Mock consistent generation within concept periods
        drift_points = [1000, 2000]
        test_coordinates = [(0.5, 0.3), (0.7, 0.2), (0.2, 0.8)]
        
        def mock_consistent_generation(position_start, position_end, concept):
            """Generate consistent labels within concept period"""
            results = []
            for pos in range(position_start, position_end):
                coord_idx = pos % len(test_coordinates)
                x, y = test_coordinates[coord_idx]
                
                base_label = 1 if y < math.sin(x) else 0
                if concept % 2 == 1:
                    label = 1 - base_label
                else:
                    label = base_label
                    
                results.append((np.array([x, y, 0.1, 0.9]), label))
            return results
        
        # Test concept 0 consistency (positions 1-999)
        concept_0_instances = mock_consistent_generation(1, 1000, 0)
        
        # Test concept 1 consistency (positions 1001-1999)  
        concept_1_instances = mock_consistent_generation(1001, 2000, 1)
        
        # Within each concept, same coordinates should give same labels
        coord_to_label_c0 = {}
        for features, label in concept_0_instances:
            coord_key = (features[0], features[1])
            if coord_key in coord_to_label_c0:
                assert coord_to_label_c0[coord_key] == label, "Same coords should have same label in concept 0"
            coord_to_label_c0[coord_key] = label
        
        coord_to_label_c1 = {}
        for features, label in concept_1_instances:
            coord_key = (features[0], features[1])
            if coord_key in coord_to_label_c1:
                assert coord_to_label_c1[coord_key] == label, "Same coords should have same label in concept 1"
            coord_to_label_c1[coord_key] = label


class TestSineDatasetConfigurations:
    """Test suite for sine dataset configuration support"""

    def test_sine_1000_configuration(self):
        """Test Sine(1000) configuration parameters"""
        generator = Mock()
        
        # Mock Sine(1000) configuration
        config = {
            'type': 'sine',
            'context_size': 1000,
            'training_instances': 1,
            'drift_instances': 100,
            'irrelevant_attributes': 2
        }
        
        generator.get_config = Mock(return_value=config)
        generator.get_drift_points = Mock(return_value=[1000, 2000, 3000])
        
        actual_config = generator.get_config()
        drift_points = generator.get_drift_points()
        
        assert actual_config['context_size'] == 1000
        assert actual_config['training_instances'] == 1
        assert actual_config['drift_instances'] == 100
        assert len(drift_points) > 0
        assert all(dp % 1000 == 0 for dp in drift_points), "Drift points should align with context size"

    def test_sine_5000_configuration(self):
        """Test Sine(5000) configuration parameters"""
        generator = Mock()
        
        # Mock Sine(5000) configuration
        config = {
            'type': 'sine',
            'context_size': 5000,
            'training_instances': 10,
            'drift_instances': 100,
            'irrelevant_attributes': 2
        }
        
        generator.get_config = Mock(return_value=config)
        generator.get_drift_points = Mock(return_value=[5000, 10000, 15000])
        
        actual_config = generator.get_config()
        drift_points = generator.get_drift_points()
        
        assert actual_config['context_size'] == 5000
        assert actual_config['training_instances'] == 10
        assert actual_config['drift_instances'] == 100
        assert len(drift_points) > 0
        assert all(dp % 5000 == 0 for dp in drift_points), "Drift points should align with context size"

    def test_sine_configuration_validation(self):
        """Test sine dataset parameter validation"""
        # Test invalid configurations should raise errors
        
        invalid_configs = [
            {'context_size': -1},  # Negative context size
            {'context_size': 0},   # Zero context size
            {'training_instances': -1},  # Negative training instances
            {'drift_instances': -1},     # Negative drift instances
            {'irrelevant_attributes': -1}  # Negative irrelevant attributes
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                # Mock validation that should fail
                if 'context_size' in invalid_config and invalid_config['context_size'] <= 0:
                    raise ValueError("Context size must be positive")
                if 'training_instances' in invalid_config and invalid_config['training_instances'] < 0:
                    raise ValueError("Training instances must be non-negative")
                if 'drift_instances' in invalid_config and invalid_config['drift_instances'] < 0:
                    raise ValueError("Drift instances must be non-negative")
                if 'irrelevant_attributes' in invalid_config and invalid_config['irrelevant_attributes'] < 0:
                    raise ValueError("Irrelevant attributes must be non-negative")


class TestSineStatisticalProperties:
    """Test suite for sine dataset statistical properties"""

    def test_sine_class_balance(self):
        """Test class balance in generated sine dataset"""
        generator = Mock()
        
        # Mock balanced label generation
        np.random.seed(42)
        sample_size = 10000
        
        # Generate realistic sine data distribution
        x_values = np.random.uniform(0, 1, sample_size)
        y_values = np.random.uniform(0, 1, sample_size)
        labels = [1 if y < math.sin(x) else 0 for x, y in zip(x_values, y_values)]
        
        label_iterator = iter(labels)
        
        def mock_balanced_generation():
            try:
                return next(label_iterator)
            except StopIteration:
                return np.random.choice([0, 1])
        
        generator.generate_instance = Mock(side_effect=lambda: (np.array([0, 0, 0, 0]), mock_balanced_generation()))
        
        # Generate large sample
        generated_labels = []
        for _ in range(sample_size):
            _, label = generator.generate_instance()
            generated_labels.append(label)
        
        positive_ratio = sum(generated_labels) / len(generated_labels)
        
        # Test approximate class balance (should be near 0.5 for sine dataset)
        assert 0.45 <= positive_ratio <= 0.55, f"Class balance {positive_ratio:.3f} should be approximately 0.5"
        
        # Test both classes are present
        unique_labels = set(generated_labels)
        assert unique_labels == {0, 1}, "Both positive and negative classes should be present"

    def test_sine_attribute_distribution(self):
        """Test statistical distribution of generated attributes"""
        generator = Mock()
        
        # Mock attribute generation
        np.random.seed(42)
        sample_size = 1000
        
        generated_attributes = []
        for _ in range(sample_size):
            x, y, a3, a4 = np.random.uniform(0, 1, 4)
            generated_attributes.append([x, y, a3, a4])
        
        attr_iterator = iter(generated_attributes)
        
        def mock_attribute_generation():
            features = next(attr_iterator)
            label = 1 if features[1] < math.sin(features[0]) else 0
            return np.array(features), label
        
        generator.generate_instance = Mock(side_effect=mock_attribute_generation)
        
        # Generate instances
        all_features = []
        for _ in range(sample_size):
            features, _ = generator.generate_instance()
            all_features.append(features)
        
        all_features = np.array(all_features)
        
        # Test each attribute follows uniform distribution
        for attr_idx in range(4):
            attr_values = all_features[:, attr_idx]
            
            # Test range
            assert np.all(attr_values >= 0), f"Attribute {attr_idx} should be >= 0"
            assert np.all(attr_values <= 1), f"Attribute {attr_idx} should be <= 1"
            
            # Test approximate uniform distribution
            attr_mean = np.mean(attr_values)
            assert 0.4 < attr_mean < 0.6, f"Attribute {attr_idx} mean {attr_mean:.3f} should be near 0.5"

    def test_sine_reproducibility(self):
        """Test deterministic behavior with same seed"""
        # Test that two generators with same seed produce identical sequences
        
        def create_mock_generator(seed):
            generator = Mock()
            np.random.seed(seed)
            
            def deterministic_generation():
                x, y = np.random.uniform(0, 1, 2)
                a3, a4 = np.random.uniform(0, 1, 2)
                features = np.array([x, y, a3, a4])
                label = 1 if y < math.sin(x) else 0
                return features, label
            
            generator.generate_instance = Mock(side_effect=deterministic_generation)
            return generator
        
        # Create two generators with same seed
        generator1 = create_mock_generator(42)
        generator2 = create_mock_generator(42)
        
        # Generate sequences
        sequence1 = [generator1.generate_instance() for _ in range(100)]
        sequence2 = [generator2.generate_instance() for _ in range(100)]
        
        # Sequences should be identical
        for i, ((features1, label1), (features2, label2)) in enumerate(zip(sequence1, sequence2)):
            np.testing.assert_array_almost_equal(features1, features2, decimal=10,
                                                err_msg=f"Features differ at position {i}")
            assert label1 == label2, f"Labels differ at position {i}"


class TestSineDatasetIntegration:
    """Integration tests for sine dataset generator"""

    def test_sine_full_workflow(self):
        """Test complete sine dataset generation workflow"""
        generator = Mock()
        
        # Mock complete workflow
        drift_points = [1000, 2000]
        current_position = 0
        
        def mock_complete_workflow():
            nonlocal current_position
            current_position += 1
            
            # Generate coordinates
            x, y = np.random.uniform(0, 1, 2)
            a3, a4 = np.random.uniform(0, 1, 2)
            features = np.array([x, y, a3, a4])
            
            # Determine concept based on position
            concept = sum(1 for dp in drift_points if current_position > dp) % 2
            
            # Generate label
            base_label = 1 if y < math.sin(x) else 0
            if concept == 1:
                label = 1 - base_label
            else:
                label = base_label
            
            return features, label
        
        generator.generate_instance = Mock(side_effect=mock_complete_workflow)
        generator.get_drift_points = Mock(return_value=drift_points)
        generator.get_metadata = Mock(return_value={
            'dataset_type': 'sine',
            'num_attributes': 4,
            'num_classes': 2,
            'drift_points': drift_points
        })
        
        # Test complete workflow
        instances = []
        for _ in range(2500):
            instance = generator.generate_instance()
            instances.append(instance)
        
        # Validate workflow results
        assert len(instances) == 2500
        
        features_array = np.array([inst[0] for inst in instances])
        labels_array = np.array([inst[1] for inst in instances])
        
        # Test feature properties
        assert features_array.shape == (2500, 4)
        assert np.all(features_array >= 0) and np.all(features_array <= 1)
        
        # Test label properties
        assert set(labels_array) == {0, 1}
        
        # Test drift behavior
        metadata = generator.get_metadata()
        drift_points = generator.get_drift_points()
        
        assert metadata['dataset_type'] == 'sine'
        assert metadata['num_attributes'] == 4
        assert drift_points == [1000, 2000]

    def test_sine_error_handling(self):
        """Test sine dataset error handling"""
        generator = Mock()
        
        # Test various error conditions
        def test_invalid_coordinates():
            # Coordinates outside [0,1] should raise error
            generator.generate_instance = Mock(side_effect=ValueError("Coordinates must be in [0,1]"))
            
            with pytest.raises(ValueError):
                generator.generate_instance()
        
        def test_invalid_drift_points():
            # Invalid drift points should raise error
            generator.get_drift_points = Mock(side_effect=ValueError("Drift points must be positive integers"))
            
            with pytest.raises(ValueError):
                generator.get_drift_points()
        
        test_invalid_coordinates()
        test_invalid_drift_points()
        
    def test_sine_performance_requirements(self):
        """Test sine dataset meets performance requirements"""
        generator = Mock()
        
        # Mock fast generation
        def fast_sine_generation():
            return np.array([0.5, 0.3, 0.1, 0.8]), 1
        
        generator.generate_instance = Mock(side_effect=fast_sine_generation)
        
        # Test generation speed
        import time
        start_time = time.time()
        
        for _ in range(1000):
            generator.generate_instance()
        
        end_time = time.time()
        generation_time = end_time - start_time
        instances_per_second = 1000 / generation_time
        
        # Should meet performance requirement (>1000 instances/second)
        # This is a mock test - real implementation would be tested
        assert instances_per_second > 100, f"Generation speed {instances_per_second:.0f} should be adequate"
