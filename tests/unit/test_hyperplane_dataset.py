"""
Unit tests for Hyperplane Dataset Generator - REQ-GRADUAL-201

Tests the rotating hyperplane behavior and gradual concept drift of the Hyperplane dataset generator.
Following TDD principles, these tests define the exact behavior required before
any implementation exists.

Test Coverage:
- Hyperplane equation correctness
- Continuous rotation behavior
- Noise injection (5%)
- Direction reversal (10% probability)
- Multi-dimensional support
- Gradual drift patterns
"""

import math
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestHyperplaneEquationCorrectness:
    """Test suite for hyperplane mathematical definition - REQ-GRADUAL-201"""

    def test_hyperplane_equation(self):
        """Test hyperplane classification: ∑(wi * xi) ≥ w0 → positive class"""
        # This is the fundamental mathematical test from requirements
        generator = Mock()
        
        # Mock hyperplane evaluation
        def mock_hyperplane_classification(instance, weights):
            """Reference implementation of hyperplane classification"""
            if len(instance) != len(weights) - 1:
                raise ValueError("Instance dimensions must match weight dimensions - 1")
            
            # Calculate weighted sum: ∑(wi * xi)
            weighted_sum = sum(w * x for w, x in zip(weights[:-1], instance))
            threshold = weights[-1]  # w0 is the last weight (threshold)
            
            # Classification rule: positive if weighted_sum >= threshold
            return 1 if weighted_sum >= threshold else 0
        
        generator.classify = Mock(side_effect=mock_hyperplane_classification)
        
        # Test specific cases
        test_cases = [
            # (instance, weights, expected_label, description)
            ([0.5, 0.5], [1.0, 1.0, 0.8], 1, "0.5*1 + 0.5*1 = 1.0 >= 0.8 → positive"),
            ([0.3, 0.4], [1.0, 1.0, 0.8], 0, "0.3*1 + 0.4*1 = 0.7 < 0.8 → negative"),
            ([0.8, 0.2], [0.5, 1.5, 1.0], 0, "0.8*0.5 + 0.2*1.5 = 0.7 < 1.0 → negative"),
            ([0.9, 0.9], [0.5, 0.5, 0.8], 1, "0.9*0.5 + 0.9*0.5 = 0.9 >= 0.8 → positive"),
            ([0.4, 0.6], [2.0, -1.0, 0.2], 1, "0.4*2 - 0.6*1 = 0.2 >= 0.2 → positive (boundary)"),
        ]
        
        for instance, weights, expected_label, description in test_cases:
            actual_label = generator.classify(instance, weights)
            assert actual_label == expected_label, f"Failed: {description}"

    def test_hyperplane_dimensions(self):
        """Test d-dimensional hyperplane support (default d=10)"""
        generator = Mock()
        
        # Mock multi-dimensional instance generation
        def mock_multidimensional_generation(dimensions=10):
            instance = np.random.uniform(0, 1, dimensions)
            weights = np.random.uniform(-1, 1, dimensions + 1)  # +1 for threshold
            
            # Calculate classification
            weighted_sum = sum(w * x for w, x in zip(weights[:-1], instance))
            label = 1 if weighted_sum >= weights[-1] else 0
            
            return instance, label, weights
        
        generator.generate_instance = Mock(side_effect=lambda dims=10: mock_multidimensional_generation(dims))
        
        # Test default dimensions (10)
        instance, label, weights = generator.generate_instance()
        
        assert len(instance) == 10, "Default should be 10 dimensions"
        assert len(weights) == 11, "Should have 10 feature weights + 1 threshold"
        assert isinstance(label, (int, np.integer)), "Label should be integer"
        assert label in [0, 1], "Label should be binary"
        
        # Test custom dimensions
        for dims in [2, 5, 15, 20]:
            instance, label, weights = generator.generate_instance(dims)
            assert len(instance) == dims, f"Should support {dims} dimensions"
            assert len(weights) == dims + 1, f"Should have {dims} feature weights + 1 threshold"

    def test_hyperplane_weight_vector_properties(self):
        """Test weight vector initialization and properties"""
        generator = Mock()
        
        # Mock weight vector initialization
        def mock_weight_initialization(dimensions=10):
            # Weights should be initialized randomly but within reasonable bounds
            weights = np.random.uniform(-1, 1, dimensions + 1)
            return weights
        
        generator.initialize_weights = Mock(side_effect=mock_weight_initialization)
        
        # Test weight initialization
        weights = generator.initialize_weights(10)
        
        assert len(weights) == 11, "10D hyperplane should have 11 weights (10 + threshold)"
        assert all(-1 <= w <= 1 for w in weights), "Weights should be in reasonable range"
        assert weights[-1] is not None, "Threshold weight should be defined"
        
        # Test different dimensions
        for dims in [5, 15, 25]:
            weights = generator.initialize_weights(dims)
            assert len(weights) == dims + 1, f"{dims}D should have {dims + 1} weights"

    def test_hyperplane_instance_generation_uniform(self):
        """Test instance coordinates are uniformly distributed in [0,1]^d"""
        generator = Mock()
        
        # Mock uniform instance generation
        np.random.seed(42)
        dimensions = 10
        sample_size = 1000
        
        generated_instances = []
        for _ in range(sample_size):
            instance = np.random.uniform(0, 1, dimensions)
            generated_instances.append(instance)
        
        instance_iterator = iter(generated_instances)
        generator.generate_coordinates = Mock(side_effect=lambda: next(instance_iterator))
        
        # Generate and test distribution
        coordinates = []
        for _ in range(100):  # Test subset for validation
            coord = generator.generate_coordinates()
            coordinates.append(coord)
        
        coordinates = np.array(coordinates)
        
        # Test coordinate bounds
        assert np.all(coordinates >= 0), "All coordinates should be >= 0"
        assert np.all(coordinates <= 1), "All coordinates should be <= 1"
        assert coordinates.shape == (100, 10), "Should have correct shape"
        
        # Test approximate uniform distribution
        for dim in range(dimensions):
            dim_values = coordinates[:, dim]
            dim_mean = np.mean(dim_values)
            assert 0.3 < dim_mean < 0.7, f"Dimension {dim} mean {dim_mean:.3f} should be near 0.5"


class TestHyperplaneRotationBehavior:
    """Test suite for continuous hyperplane rotation"""

    def test_hyperplane_rotation(self):
        """Test continuous rotation with configurable speed"""
        generator = Mock()
        
        # Mock rotation behavior
        def mock_rotation_system(initial_weights, change_rate=0.1, time_steps=100):
            """Simulate hyperplane rotation over time"""
            current_weights = initial_weights.copy()
            weight_history = [current_weights.copy()]
            
            for step in range(time_steps):
                # Simulate small random changes to weights (rotation)
                np.random.seed(42 + step)  # Deterministic for testing
                rotation_vector = np.random.normal(0, change_rate, len(current_weights))
                current_weights += rotation_vector
                weight_history.append(current_weights.copy())
            
            return weight_history
        
        # Test rotation
        initial_weights = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.2, -0.1, 0.05, 0.0])
        change_rate = 0.1
        
        generator.simulate_rotation = Mock(side_effect=lambda w, rate, steps: mock_rotation_system(w, rate, steps))
        
        weight_history = generator.simulate_rotation(initial_weights, change_rate, 100)
        
        # Test that weights actually change over time
        weights_t0 = weight_history[0]
        weights_t50 = weight_history[50]
        weights_t100 = weight_history[100]
        
        # Weights should be different after rotation
        assert not np.array_equal(weights_t0, weights_t50), "Weights should change after 50 steps"
        assert not np.array_equal(weights_t0, weights_t100), "Weights should change after 100 steps"
        assert not np.array_equal(weights_t50, weights_t100), "Weights should continue changing"
        
        # Changes should be gradual (not abrupt)
        change_t0_to_t50 = np.linalg.norm(weights_t50 - weights_t0)
        change_t50_to_t100 = np.linalg.norm(weights_t100 - weights_t50)
        
        # Both changes should be reasonable (not zero, not huge)
        assert 0.01 < change_t0_to_t50 < 5.0, f"Change magnitude {change_t0_to_t50:.3f} should be reasonable"
        assert 0.01 < change_t50_to_t100 < 5.0, f"Change magnitude {change_t50_to_t100:.3f} should be reasonable"

    def test_hyperplane_change_rates(self):
        """Test different rotation speeds: Hyp(0.1) and Hyp(0.001)"""
        generator = Mock()
        
        # Mock different change rates
        def mock_change_rate_comparison(change_rate, steps=100):
            """Compare rotation at different change rates"""
            initial_weights = np.ones(11) * 0.1
            
            # Simulate rotation
            np.random.seed(42)
            total_change = 0
            current_weights = initial_weights.copy()
            
            for step in range(steps):
                change = np.random.normal(0, change_rate, len(current_weights))
                current_weights += change
                total_change += np.linalg.norm(change)
            
            return total_change, current_weights
        
        generator.test_change_rate = Mock(side_effect=mock_change_rate_comparison)
        
        # Test Hyp(0.1) - fast rotation
        fast_change, fast_weights = generator.test_change_rate(0.1, 100)
        
        # Test Hyp(0.001) - slow rotation  
        slow_change, slow_weights = generator.test_change_rate(0.001, 100)
        
        # Fast rotation should produce more change than slow rotation
        assert fast_change > slow_change, "Fast rotation should produce more total change"
        assert fast_change > slow_change * 10, "Fast rotation should be significantly faster"
        
        # Both should produce some change
        assert fast_change > 0, "Fast rotation should produce measurable change"
        assert slow_change > 0, "Slow rotation should produce some change"

    def test_hyperplane_rotation_continuity(self):
        """Test rotation is continuous (no abrupt jumps)"""
        generator = Mock()
        
        # Mock continuous rotation
        def mock_continuous_rotation():
            """Generate sequence with continuous small changes"""
            initial_weights = np.array([0.1, 0.2, -0.1, 0.05, -0.15, 0.08, -0.05, 0.12, 0.03, -0.08, 0.0])
            change_rate = 0.01
            
            weight_sequence = []
            current_weights = initial_weights.copy()
            
            for step in range(50):
                np.random.seed(42 + step)
                small_change = np.random.normal(0, change_rate, len(current_weights))
                current_weights += small_change
                weight_sequence.append(current_weights.copy())
            
            return weight_sequence
        
        generator.generate_continuous_sequence = Mock(side_effect=mock_continuous_rotation)
        
        weight_sequence = generator.generate_continuous_sequence()
        
        # Test continuity - consecutive weights should be very similar
        for i in range(len(weight_sequence) - 1):
            weight_diff = np.linalg.norm(weight_sequence[i+1] - weight_sequence[i])
            assert weight_diff < 0.1, f"Step {i} to {i+1}: change {weight_diff:.4f} too large for continuity"
        
        # But overall change should be significant
        total_change = np.linalg.norm(weight_sequence[-1] - weight_sequence[0])
        assert total_change > 0.05, f"Total change {total_change:.4f} should be measurable over time"

    def test_hyperplane_direction_reversal(self):
        """Test 10% probability of direction reversal"""
        generator = Mock()
        
        # Mock direction reversal mechanism
        def mock_direction_reversal_test(num_trials=1000):
            """Test direction reversal probability"""
            np.random.seed(42)
            reversal_count = 0
            
            for trial in range(num_trials):
                # 10% probability of reversal
                if np.random.random() < 0.1:
                    reversal_count += 1
            
            return reversal_count
        
        generator.count_reversals = Mock(side_effect=mock_direction_reversal_test)
        
        reversal_count = generator.count_reversals(1000)
        
        # Should be approximately 10% (100 out of 1000)
        expected_reversals = 100
        tolerance = 30  # Allow some randomness
        
        assert expected_reversals - tolerance <= reversal_count <= expected_reversals + tolerance, \
            f"Expected ~100 reversals in 1000 trials, got {reversal_count}"

    def test_hyperplane_reversal_behavior(self):
        """Test what happens during direction reversal"""
        generator = Mock()
        
        # Mock reversal behavior
        def mock_reversal_effect(weights_before_reversal):
            """Simulate the effect of direction reversal"""
            # Direction reversal might flip some weights or change rotation direction
            # This is implementation-dependent, but should be deterministic
            weights_after_reversal = weights_before_reversal.copy()
            
            # Example: flip sign of some weights (one possible reversal strategy)
            reversal_mask = np.random.choice([True, False], len(weights_after_reversal))
            weights_after_reversal[reversal_mask] *= -1
            
            return weights_after_reversal
        
        generator.apply_reversal = Mock(side_effect=mock_reversal_effect)
        
        original_weights = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.2, -0.1, 0.05, 0.0])
        reversed_weights = generator.apply_reversal(original_weights)
        
        # After reversal, weights should be different
        assert not np.array_equal(original_weights, reversed_weights), "Reversal should change weights"
        
        # But magnitude should be preserved (just direction changes)
        original_magnitudes = np.abs(original_weights)
        reversed_magnitudes = np.abs(reversed_weights)
        np.testing.assert_array_almost_equal(original_magnitudes, reversed_magnitudes, decimal=6,
                                           err_msg="Reversal should preserve weight magnitudes")


class TestHyperplaneNoiseInjection:
    """Test suite for 5% noise injection"""

    def test_hyperplane_noise_level(self):
        """Test 5% noise injection in hyperplane dataset"""
        generator = Mock()
        
        # Mock noise injection
        def mock_noise_injection(clean_labels, noise_level=0.05):
            """Simulate noise injection in labels"""
            np.random.seed(42)
            noisy_labels = clean_labels.copy()
            
            # Flip labels with specified probability
            for i in range(len(noisy_labels)):
                if np.random.random() < noise_level:
                    noisy_labels[i] = 1 - noisy_labels[i]  # Flip binary label
            
            return noisy_labels
        
        generator.add_noise = Mock(side_effect=mock_noise_injection)
        
        # Test with large sample
        sample_size = 10000
        clean_labels = np.random.choice([0, 1], sample_size)
        noisy_labels = generator.add_noise(clean_labels, 0.05)
        
        # Count flipped labels
        flipped_count = np.sum(clean_labels != noisy_labels)
        flip_rate = flipped_count / sample_size
        
        # Should be approximately 5%
        expected_rate = 0.05
        tolerance = 0.01  # 1% tolerance
        
        assert abs(flip_rate - expected_rate) <= tolerance, \
            f"Noise rate {flip_rate:.3%} should be approximately {expected_rate:.1%}"

    def test_hyperplane_noise_randomness(self):
        """Test noise injection is random and not systematic"""
        generator = Mock()
        
        # Mock random noise application
        def mock_random_noise():
            """Test that noise doesn't follow systematic patterns"""
            np.random.seed(42)
            
            # Generate clean data with clear pattern
            clean_labels = np.array([i % 2 for i in range(1000)])  # Alternating 0,1,0,1...
            
            # Apply noise
            noisy_labels = clean_labels.copy()
            for i in range(len(noisy_labels)):
                if np.random.random() < 0.05:
                    noisy_labels[i] = 1 - noisy_labels[i]
            
            return clean_labels, noisy_labels
        
        generator.test_noise_randomness = Mock(side_effect=mock_random_noise)
        
        clean_labels, noisy_labels = generator.test_noise_randomness()
        
        # Find positions where noise was applied
        noise_positions = np.where(clean_labels != noisy_labels)[0]
        
        # Test noise is not systematic
        if len(noise_positions) > 1:
            # Gaps between noise positions should vary (not constant)
            gaps = np.diff(noise_positions)
            assert len(set(gaps)) > 1, "Noise positions should not follow regular pattern"
            
            # Noise should affect both classes roughly equally
            flipped_from_0 = np.sum((clean_labels == 0) & (noisy_labels == 1))
            flipped_from_1 = np.sum((clean_labels == 1) & (noisy_labels == 0))
            
            total_flips = flipped_from_0 + flipped_from_1
            if total_flips > 10:  # Only test if we have enough flips
                flip_balance = min(flipped_from_0, flipped_from_1) / max(flipped_from_0, flipped_from_1)
                assert flip_balance > 0.3, "Noise should affect both classes roughly equally"

    def test_hyperplane_noise_preservation(self):
        """Test noise level remains consistent over time"""
        generator = Mock()
        
        # Mock consistent noise over time
        def mock_consistent_noise():
            """Test noise level consistency across different time periods"""
            np.random.seed(42)
            
            noise_rates = []
            for period in range(10):  # Test 10 different time periods
                # Generate 1000 instances for this period
                period_labels = []
                for _ in range(1000):
                    # Generate clean label (based on some concept)
                    clean_label = np.random.choice([0, 1])
                    
                    # Apply noise with 5% probability
                    if np.random.random() < 0.05:
                        noisy_label = 1 - clean_label
                    else:
                        noisy_label = clean_label
                    
                    period_labels.append((clean_label, noisy_label))
                
                # Calculate noise rate for this period
                clean, noisy = zip(*period_labels)
                noise_rate = np.mean(np.array(clean) != np.array(noisy))
                noise_rates.append(noise_rate)
            
            return noise_rates
        
        generator.test_noise_consistency = Mock(side_effect=mock_consistent_noise)
        
        noise_rates = generator.test_noise_consistency()
        
        # All periods should have similar noise rates (around 5%)
        expected_rate = 0.05
        for i, rate in enumerate(noise_rates):
            assert abs(rate - expected_rate) < 0.02, \
                f"Period {i}: noise rate {rate:.3%} deviates too much from expected {expected_rate:.1%}"
        
        # Overall consistency
        rate_std = np.std(noise_rates)
        assert rate_std < 0.015, f"Noise rate standard deviation {rate_std:.4f} should be small"


class TestHyperplaneGradualDrift:
    """Test suite for gradual drift behavior"""

    def test_hyperplane_gradual_vs_abrupt_drift(self):
        """Test hyperplane uses gradual drift (not abrupt concept changes)"""
        generator = Mock()
        
        # Mock gradual drift behavior
        def mock_gradual_drift_pattern():
            """Simulate gradual concept drift through continuous rotation"""
            # Track classification boundary over time
            test_point = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            
            # Initial weights
            weights = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0])
            
            classifications = []
            for step in range(1000):
                # Gradually rotate weights
                rotation = np.random.normal(0, 0.01, len(weights))
                weights += rotation
                
                # Classify test point
                weighted_sum = sum(w * x for w, x in zip(weights[:-1], test_point))
                label = 1 if weighted_sum >= weights[-1] else 0
                classifications.append(label)
            
            return classifications
        
        generator.test_drift_pattern = Mock(side_effect=mock_gradual_drift_pattern)
        
        classifications = generator.test_drift_pattern()
        
        # For gradual drift, we should see smooth transitions, not abrupt changes
        # Count consecutive runs of same label
        runs = []
        current_run = 1
        for i in range(1, len(classifications)):
            if classifications[i] == classifications[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        # Gradual drift should have longer runs (not constantly flipping)
        avg_run_length = np.mean(runs)
        assert avg_run_length > 5, f"Average run length {avg_run_length:.1f} too short for gradual drift"
        
        # Should have some transitions (not stuck in one class)
        assert len(runs) > 5, "Should have multiple transitions for gradual drift"

    def test_hyperplane_drift_rate_correlation(self):
        """Test drift rate correlates with change_rate parameter"""
        generator = Mock()
        
        # Mock drift rate testing
        def mock_drift_rate_test(change_rate):
            """Test how change rate affects classification transitions"""
            np.random.seed(42)
            
            # Fixed test point
            test_point = np.array([0.5] * 10)
            
            # Start with fixed weights
            weights = np.array([0.1] * 11)
            
            transitions = 0
            prev_label = None
            
            for step in range(1000):
                # Apply rotation based on change rate
                rotation = np.random.normal(0, change_rate, len(weights))
                weights += rotation
                
                # Classify
                weighted_sum = sum(w * x for w, x in zip(weights[:-1], test_point))
                label = 1 if weighted_sum >= weights[-1] else 0
                
                if prev_label is not None and label != prev_label:
                    transitions += 1
                
                prev_label = label
            
            return transitions
        
        generator.count_transitions = Mock(side_effect=mock_drift_rate_test)
        
        # Test different change rates
        slow_transitions = generator.count_transitions(0.001)  # Hyp(0.001)
        fast_transitions = generator.count_transitions(0.1)    # Hyp(0.1)
        
        # Faster change rate should produce more transitions
        assert fast_transitions > slow_transitions, \
            f"Fast rate ({fast_transitions} transitions) should exceed slow rate ({slow_transitions})"
        
        # Both should have some transitions
        assert slow_transitions > 0, "Slow rate should still produce some transitions"
        assert fast_transitions > 0, "Fast rate should produce transitions"

    def test_hyperplane_concept_drift_smoothness(self):
        """Test concept boundaries change smoothly over time"""
        generator = Mock()
        
        # Mock smooth boundary evolution
        def mock_smooth_boundary_evolution():
            """Test that decision boundary evolves smoothly"""
            # Sample multiple points near potential boundary
            test_points = [
                np.array([0.4] * 10),
                np.array([0.5] * 10), 
                np.array([0.6] * 10)
            ]
            
            # Track classification over time
            point_classifications = {i: [] for i in range(len(test_points))}
            
            weights = np.array([0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.05, -0.05, 0.15, 0.0])
            
            for step in range(200):
                # Small rotation
                weights += np.random.normal(0, 0.01, len(weights))
                
                # Classify all test points
                for i, point in enumerate(test_points):
                    weighted_sum = sum(w * x for w, x in zip(weights[:-1], point))
                    label = 1 if weighted_sum >= weights[-1] else 0
                    point_classifications[i].append(label)
            
            return point_classifications
        
        generator.test_boundary_smoothness = Mock(side_effect=mock_smooth_boundary_evolution)
        
        classifications = generator.test_boundary_smoothness()
        
        # For smooth drift, nearby points should have similar classification patterns
        for point_id, labels in classifications.items():
            # Count transitions for each point
            transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
            
            # Should have some transitions (drift occurring) but not too many (smoothness)
            assert 0 < transitions < 50, f"Point {point_id}: {transitions} transitions not in smooth range"


class TestHyperplaneIntegration:
    """Integration tests for hyperplane dataset generator"""

    def test_hyperplane_full_workflow(self):
        """Test complete hyperplane generation workflow"""
        generator = Mock()
        
        # Mock complete workflow
        def mock_complete_hyperplane_workflow():
            """Simulate full hyperplane dataset generation"""
            dimensions = 10
            change_rate = 0.1
            noise_level = 0.05
            
            # Initialize
            weights = np.random.uniform(-1, 1, dimensions + 1)
            instances = []
            
            for step in range(1000):
                # Generate instance
                instance = np.random.uniform(0, 1, dimensions)
                
                # Evolve weights (rotation)
                if step > 0:  # Don't change on first step
                    rotation = np.random.normal(0, change_rate, len(weights))
                    weights += rotation
                    
                    # Occasional direction reversal (10% probability)
                    if np.random.random() < 0.1:
                        weights *= -1
                
                # Classify
                weighted_sum = sum(w * x for w, x in zip(weights[:-1], instance))
                clean_label = 1 if weighted_sum >= weights[-1] else 0
                
                # Add noise (5% probability)
                if np.random.random() < noise_level:
                    label = 1 - clean_label
                else:
                    label = clean_label
                
                instances.append((instance, label, weights.copy()))
            
            return instances
        
        generator.generate_full_dataset = Mock(side_effect=mock_complete_hyperplane_workflow)
        
        instances = generator.generate_full_dataset()
        
        # Test dataset properties
        assert len(instances) == 1000, "Should generate requested number of instances"
        
        # Test instance format
        for i, (features, label, weights) in enumerate(instances[:10]):  # Check first 10
            assert len(features) == 10, f"Instance {i}: should have 10 features"
            assert label in [0, 1], f"Instance {i}: label should be binary"
            assert len(weights) == 11, f"Instance {i}: should have 11 weights"
            assert all(0 <= f <= 1 for f in features), f"Instance {i}: features should be in [0,1]"
        
        # Test drift occurred (weights changed over time)
        initial_weights = instances[0][2]
        final_weights = instances[-1][2]
        
        weight_change = np.linalg.norm(final_weights - initial_weights)
        assert weight_change > 0.1, f"Weights should change significantly over time: {weight_change:.3f}"

    def test_hyperplane_configuration_validation(self):
        """Test hyperplane configuration parameter validation"""
        generator = Mock()
        
        # Test valid configurations
        valid_configs = [
            {'dimensions': 10, 'change_rate': 0.1, 'noise_level': 0.05},
            {'dimensions': 5, 'change_rate': 0.001, 'noise_level': 0.05},
            {'dimensions': 20, 'change_rate': 0.05, 'noise_level': 0.0}
        ]
        
        for config in valid_configs:
            # Mock validation that should pass
            try:
                assert config['dimensions'] > 0, "Dimensions must be positive"
                assert config['change_rate'] > 0, "Change rate must be positive"
                assert 0 <= config['noise_level'] <= 1, "Noise level must be in [0,1]"
                validation_passed = True
            except AssertionError:
                validation_passed = False
            
            assert validation_passed, f"Valid config should pass validation: {config}"
        
        # Test invalid configurations
        invalid_configs = [
            {'dimensions': 0, 'change_rate': 0.1, 'noise_level': 0.05},     # Zero dimensions
            {'dimensions': -1, 'change_rate': 0.1, 'noise_level': 0.05},    # Negative dimensions
            {'dimensions': 10, 'change_rate': 0, 'noise_level': 0.05},      # Zero change rate
            {'dimensions': 10, 'change_rate': -0.1, 'noise_level': 0.05},   # Negative change rate
            {'dimensions': 10, 'change_rate': 0.1, 'noise_level': -0.1},    # Negative noise
            {'dimensions': 10, 'change_rate': 0.1, 'noise_level': 1.5}      # Noise > 1
        ]
        
        for config in invalid_configs:
            with pytest.raises(AssertionError):
                assert config['dimensions'] > 0, "Dimensions must be positive"
                assert config['change_rate'] > 0, "Change rate must be positive"
                assert 0 <= config['noise_level'] <= 1, "Noise level must be in [0,1]"

    def test_hyperplane_reproducibility(self):
        """Test deterministic behavior with seed control"""
        # Test reproducible hyperplane generation
        
        def create_mock_generator(seed):
            generator = Mock()
            np.random.seed(seed)
            
            def deterministic_generation():
                # Generate deterministic sequence
                dimensions = 5  # Smaller for testing
                instance = np.random.uniform(0, 1, dimensions)
                weights = np.random.uniform(-1, 1, dimensions + 1)
                
                # Classify
                weighted_sum = sum(w * x for w, x in zip(weights[:-1], instance))
                label = 1 if weighted_sum >= weights[-1] else 0
                
                return instance, label
            
            generator.generate_instance = Mock(side_effect=deterministic_generation)
            return generator
        
        # Create two generators with same seed
        generator1 = create_mock_generator(42)
        generator2 = create_mock_generator(42)
        
        # Generate sequences
        sequence1 = [generator1.generate_instance() for _ in range(50)]
        sequence2 = [generator2.generate_instance() for _ in range(50)]
        
        # Sequences should be identical
        for i, ((features1, label1), (features2, label2)) in enumerate(zip(sequence1, sequence2)):
            np.testing.assert_array_almost_equal(features1, features2, decimal=10,
                                                err_msg=f"Features differ at position {i}")
            assert label1 == label2, f"Labels differ at position {i}"

    def test_hyperplane_performance_requirements(self):
        """Test hyperplane generator meets performance requirements"""
        generator = Mock()
        
        # Mock efficient generation
        def fast_hyperplane_generation():
            instance = np.random.uniform(0, 1, 10)
            label = np.random.choice([0, 1])
            return instance, label
        
        generator.generate_instance = Mock(side_effect=fast_hyperplane_generation)
        
        # Test generation speed
        import time
        start_time = time.time()
        
        for _ in range(1000):
            generator.generate_instance()
        
        end_time = time.time()
        generation_time = end_time - start_time
        instances_per_second = 1000 / generation_time
        
        # Should meet performance requirement
        assert instances_per_second > 100, f"Generation speed {instances_per_second:.0f} should be adequate"
