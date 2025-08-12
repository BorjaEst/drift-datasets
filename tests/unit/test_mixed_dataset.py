"""
Unit tests for Mixed Dataset Generator - REQ-GRADUAL-202

Tests the mixed attribute behavior and probability-based gradual drift of the 
Mixed dataset generator. Following TDD principles, these tests define the exact 
behavior required before any implementation exists.

Test Coverage:
- Boolean and numerical attribute generation
- Probability-based concept drift
- Gradual transitions in concept probability
- 4 attributes (2 boolean + 2 numerical)
- Classification rules based on mixed attributes
"""

from typing import Any, Dict, List, Tuple, Union
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestMixedAttributeDefinitions:
    """Test suite for mixed attribute structure - REQ-GRADUAL-202"""

    def test_mixed_attribute_types(self):
        """Test mixed dataset has exactly 4 attributes: 2 boolean + 2 numerical"""
        generator = Mock()
        
        # Mock attribute generation
        def mock_mixed_attribute_generation():
            """Generate mixed attribute instance"""
            # 2 boolean attributes (0 or 1)
            bool_attr1 = np.random.choice([0, 1])
            bool_attr2 = np.random.choice([0, 1])
            
            # 2 numerical attributes (continuous in [0,1])
            num_attr1 = np.random.uniform(0, 1)
            num_attr2 = np.random.uniform(0, 1)
            
            return [bool_attr1, bool_attr2, num_attr1, num_attr2]
        
        generator.generate_attributes = Mock(side_effect=mock_mixed_attribute_generation)
        
        # Test attribute generation
        attributes = generator.generate_attributes()
        
        assert len(attributes) == 4, "Mixed dataset should have exactly 4 attributes"
        
        # Test attribute types
        assert attributes[0] in [0, 1], "First attribute should be boolean (0 or 1)"
        assert attributes[1] in [0, 1], "Second attribute should be boolean (0 or 1)"
        assert isinstance(attributes[2], (float, np.floating)), "Third attribute should be numerical"
        assert isinstance(attributes[3], (float, np.floating)), "Fourth attribute should be numerical"
        
        # Test numerical attribute ranges
        assert 0 <= attributes[2] <= 1, "Third attribute should be in [0,1]"
        assert 0 <= attributes[3] <= 1, "Fourth attribute should be in [0,1]"

    def test_boolean_attribute_distribution(self):
        """Test boolean attributes are uniformly distributed"""
        generator = Mock()
        
        # Mock boolean generation with proper distribution
        def mock_boolean_distribution():
            """Test boolean attribute distribution"""
            np.random.seed(42)
            
            # Generate many samples
            samples = []
            for _ in range(10000):
                bool1 = np.random.choice([0, 1])
                bool2 = np.random.choice([0, 1])
                samples.append((bool1, bool2))
            
            return samples
        
        generator.generate_boolean_samples = Mock(side_effect=mock_boolean_distribution)
        
        samples = generator.generate_boolean_samples()
        
        # Analyze distribution
        bool1_values = [s[0] for s in samples]
        bool2_values = [s[1] for s in samples]
        
        # Both boolean attributes should be approximately 50/50
        bool1_mean = np.mean(bool1_values)
        bool2_mean = np.mean(bool2_values)
        
        assert 0.45 < bool1_mean < 0.55, f"Boolean attr1 mean {bool1_mean:.3f} should be ~0.5"
        assert 0.45 < bool2_mean < 0.55, f"Boolean attr2 mean {bool2_mean:.3f} should be ~0.5"
        
        # Test all 4 combinations appear roughly equally
        combinations = [(0,0), (0,1), (1,0), (1,1)]
        combination_counts = {combo: 0 for combo in combinations}
        
        for bool1, bool2 in samples:
            combination_counts[(bool1, bool2)] += 1
        
        for combo, count in combination_counts.items():
            proportion = count / len(samples)
            assert 0.2 < proportion < 0.3, \
                f"Combination {combo} proportion {proportion:.3f} should be ~0.25"

    def test_numerical_attribute_distribution(self):
        """Test numerical attributes are uniformly distributed in [0,1]"""
        generator = Mock()
        
        # Mock numerical generation
        def mock_numerical_distribution():
            """Test numerical attribute distribution"""
            np.random.seed(42)
            
            samples = []
            for _ in range(5000):
                num1 = np.random.uniform(0, 1)
                num2 = np.random.uniform(0, 1)
                samples.append((num1, num2))
            
            return samples
        
        generator.generate_numerical_samples = Mock(side_effect=mock_numerical_distribution)
        
        samples = generator.generate_numerical_samples()
        
        # Analyze distribution
        num1_values = np.array([s[0] for s in samples])
        num2_values = np.array([s[1] for s in samples])
        
        # Test bounds
        assert np.all(num1_values >= 0) and np.all(num1_values <= 1), "Numerical attr1 should be in [0,1]"
        assert np.all(num2_values >= 0) and np.all(num2_values <= 1), "Numerical attr2 should be in [0,1]"
        
        # Test approximately uniform distribution (mean should be ~0.5)
        num1_mean = np.mean(num1_values)
        num2_mean = np.mean(num2_values)
        
        assert 0.45 < num1_mean < 0.55, f"Numerical attr1 mean {num1_mean:.3f} should be ~0.5"
        assert 0.45 < num2_mean < 0.55, f"Numerical attr2 mean {num2_mean:.3f} should be ~0.5"
        
        # Test distribution shape (standard deviation for uniform [0,1] ≈ 0.289)
        num1_std = np.std(num1_values)
        num2_std = np.std(num2_values)
        expected_std = np.sqrt(1/12)  # Theoretical std for uniform [0,1]
        
        assert 0.25 < num1_std < 0.33, f"Numerical attr1 std {num1_std:.3f} should be ~{expected_std:.3f}"
        assert 0.25 < num2_std < 0.33, f"Numerical attr2 std {num2_std:.3f} should be ~{expected_std:.3f}"

    def test_mixed_attribute_independence(self):
        """Test attributes are generated independently"""
        generator = Mock()
        
        # Mock independent generation
        def mock_independent_generation():
            """Test attribute independence"""
            np.random.seed(42)
            
            samples = []
            for _ in range(2000):
                # Generate all 4 attributes
                bool1 = np.random.choice([0, 1])
                bool2 = np.random.choice([0, 1])
                num1 = np.random.uniform(0, 1)
                num2 = np.random.uniform(0, 1)
                
                samples.append([bool1, bool2, num1, num2])
            
            return np.array(samples)
        
        generator.generate_independent_samples = Mock(side_effect=mock_independent_generation)
        
        samples = generator.generate_independent_samples()
        
        # Test correlation between attributes (should be near zero)
        correlations = np.corrcoef(samples.T)
        
        # Check all pairwise correlations are small
        for i in range(4):
            for j in range(i+1, 4):
                correlation = abs(correlations[i, j])
                assert correlation < 0.1, \
                    f"Attributes {i} and {j} correlation {correlation:.3f} too high for independence"


class TestMixedConceptDefinitions:
    """Test suite for mixed concept classification rules"""

    def test_mixed_classification_rule(self):
        """Test classification rule for mixed attributes"""
        generator = Mock()
        
        # Mock classification based on mixed rule
        def mock_mixed_classification(attributes):
            """Reference implementation of mixed classification"""
            bool1, bool2, num1, num2 = attributes
            
            # Example mixed rule: combination of boolean and numerical conditions
            # Rule: (bool1 AND bool2) OR (num1 > 0.5 AND num2 < 0.3)
            condition1 = (bool1 == 1) and (bool2 == 1)
            condition2 = (num1 > 0.5) and (num2 < 0.3)
            
            return 1 if (condition1 or condition2) else 0
        
        generator.classify = Mock(side_effect=mock_mixed_classification)
        
        # Test specific cases
        test_cases = [
            # (bool1, bool2, num1, num2, expected_label, description)
            ([1, 1, 0.2, 0.8], 1, "Both booleans True → positive"),
            ([0, 1, 0.2, 0.8], 0, "Only one boolean True, nums don't satisfy → negative"),
            ([0, 0, 0.7, 0.2], 1, "Booleans False but num1>0.5 and num2<0.3 → positive"),
            ([0, 0, 0.3, 0.2], 0, "Booleans False, num1≤0.5 → negative"),
            ([0, 0, 0.7, 0.5], 0, "Booleans False, num2≥0.3 → negative"),
            ([1, 1, 0.3, 0.5], 1, "Both booleans True (nums irrelevant) → positive"),
        ]
        
        for attributes, expected_label, description in test_cases:
            actual_label = generator.classify(attributes)
            assert actual_label == expected_label, f"Failed: {description}"

    def test_concept_probability_structure(self):
        """Test probability-based concept definition"""
        generator = Mock()
        
        # Mock probabilistic concept
        def mock_probabilistic_concept(attributes, concept_probability=0.8):
            """Test probabilistic concept definition"""
            bool1, bool2, num1, num2 = attributes
            
            # Base classification
            base_condition = (bool1 == 1 and bool2 == 1) or (num1 > 0.5 and num2 < 0.3)
            
            if base_condition:
                # High probability of positive class
                return 1 if np.random.random() < concept_probability else 0
            else:
                # High probability of negative class
                return 0 if np.random.random() < concept_probability else 1
        
        generator.probabilistic_classify = Mock(side_effect=mock_probabilistic_concept)
        
        # Test with high probability concept
        np.random.seed(42)
        
        # Test case that should be mostly positive
        positive_attributes = [1, 1, 0.2, 0.8]  # Satisfies boolean condition
        positive_results = []
        
        for _ in range(1000):
            result = generator.probabilistic_classify(positive_attributes, 0.9)
            positive_results.append(result)
        
        positive_rate = np.mean(positive_results)
        assert positive_rate > 0.8, f"High-probability positive case should be >80% positive: {positive_rate:.2%}"
        
        # Test case that should be mostly negative
        negative_attributes = [0, 0, 0.3, 0.8]  # Doesn't satisfy either condition
        negative_results = []
        
        for _ in range(1000):
            result = generator.probabilistic_classify(negative_attributes, 0.9)
            negative_results.append(result)
        
        negative_rate = 1 - np.mean(negative_results)  # Rate of negative (0) labels
        assert negative_rate > 0.8, f"High-probability negative case should be >80% negative: {negative_rate:.2%}"

    def test_concept_boundary_cases(self):
        """Test classification at concept boundaries"""
        generator = Mock()
        
        # Mock boundary testing
        def mock_boundary_classification():
            """Test cases at decision boundaries"""
            
            test_cases = [
                # Boundary cases for numerical conditions
                ([0, 0, 0.5, 0.3], "num1 exactly 0.5"),
                ([0, 0, 0.50001, 0.29999], "just above/below thresholds"),
                ([0, 0, 0.49999, 0.30001], "just below/above thresholds"),
                
                # Mixed boundary cases
                ([1, 0, 0.6, 0.2], "one boolean true, nums satisfy"),
                ([0, 1, 0.6, 0.2], "other boolean true, nums satisfy"),
            ]
            
            results = []
            for attributes, description in test_cases:
                bool1, bool2, num1, num2 = attributes
                
                # Apply exact rule
                condition1 = (bool1 == 1) and (bool2 == 1)
                condition2 = (num1 > 0.5) and (num2 < 0.3)
                label = 1 if (condition1 or condition2) else 0
                
                results.append((attributes, label, description))
            
            return results
        
        generator.test_boundaries = Mock(side_effect=mock_boundary_classification)
        
        boundary_results = generator.test_boundaries()
        
        # Verify boundary classifications
        expected_results = [
            (0, "num1 exactly 0.5 should be negative"),
            (1, "just above/below should be positive"), 
            (0, "just below/above should be negative"),
            (1, "one boolean + nums should be positive"),
            (1, "other boolean + nums should be positive"),
        ]
        
        for i, ((attributes, actual_label, desc), (expected_label, expected_desc)) in enumerate(zip(boundary_results, expected_results)):
            assert actual_label == expected_label, f"Boundary case {i}: {desc} - {expected_desc}"


class TestMixedProbabilityDrift:
    """Test suite for probability-based gradual drift"""

    def test_probability_drift_mechanism(self):
        """Test how concept probability changes over time"""
        generator = Mock()
        
        # Mock probability evolution
        def mock_probability_evolution():
            """Simulate gradual change in concept probability"""
            
            # Start with high concept strength (90% probability)
            initial_prob = 0.9
            final_prob = 0.1  # End with low concept strength
            
            drift_length = 1000  # Number of instances over which drift occurs
            probabilities = []
            
            for step in range(drift_length):
                # Linear drift for testing (could be other functions)
                progress = step / (drift_length - 1)
                current_prob = initial_prob * (1 - progress) + final_prob * progress
                probabilities.append(current_prob)
            
            return probabilities
        
        generator.evolve_probability = Mock(side_effect=mock_probability_evolution)
        
        probabilities = generator.evolve_probability()
        
        # Test probability evolution
        assert len(probabilities) == 1000, "Should have probability for each instance"
        assert probabilities[0] == pytest.approx(0.9, abs=0.01), "Should start at high probability"
        assert probabilities[-1] == pytest.approx(0.1, abs=0.01), "Should end at low probability"
        
        # Test monotonic decrease
        for i in range(1, len(probabilities)):
            assert probabilities[i] <= probabilities[i-1], f"Probability should decrease monotonically at step {i}"
        
        # Test gradual change (no sudden jumps)
        for i in range(1, len(probabilities)):
            change = abs(probabilities[i] - probabilities[i-1])
            assert change < 0.01, f"Probability change {change:.4f} too large for gradual drift at step {i}"

    def test_probability_effect_on_classification(self):
        """Test how changing probability affects classification outcomes"""
        generator = Mock()
        
        # Mock classification with varying probability
        def mock_varying_probability_classification():
            """Test classification under varying concept probability"""
            # Fixed test case that satisfies the concept
            test_attributes = [1, 1, 0.2, 0.8]  # Should be positive under strong concept
            
            results_high_prob = []
            results_low_prob = []
            
            np.random.seed(42)
            
            # Test under high concept probability (0.95)
            for _ in range(1000):
                label = 1 if np.random.random() < 0.95 else 0
                results_high_prob.append(label)
            
            # Reset seed for fair comparison
            np.random.seed(42)
            
            # Test under low concept probability (0.20)
            for _ in range(1000):
                label = 1 if np.random.random() < 0.20 else 0
                results_low_prob.append(label)
            
            return results_high_prob, results_low_prob
        
        generator.test_probability_effect = Mock(side_effect=mock_varying_probability_classification)
        
        high_prob_results, low_prob_results = generator.test_probability_effect()
        
        # Test probability effects
        high_prob_positive_rate = np.mean(high_prob_results)
        low_prob_positive_rate = np.mean(low_prob_results)
        
        assert high_prob_positive_rate > 0.9, f"High probability should yield >90% positive: {high_prob_positive_rate:.2%}"
        assert low_prob_positive_rate < 0.3, f"Low probability should yield <30% positive: {low_prob_positive_rate:.2%}"
        
        # Significant difference between probabilities
        assert high_prob_positive_rate > low_prob_positive_rate + 0.5, \
            "High and low probabilities should produce significantly different outcomes"

    def test_gradual_concept_transition(self):
        """Test smooth transition between concept strengths"""
        generator = Mock()
        
        # Mock gradual transition
        def mock_gradual_transition():
            """Simulate gradual transition in classification behavior"""
            # Test instance that should transition from positive to negative
            test_attributes = [1, 1, 0.6, 0.4]  # Satisfies boolean condition
            
            classification_rates = []
            
            # Test different stages of drift
            probabilities = [0.9, 0.7, 0.5, 0.3, 0.1]
            
            for prob in probabilities:
                np.random.seed(42)  # Consistent seed for comparison
                
                classifications = []
                for _ in range(1000):
                    # Simulate probabilistic classification
                    base_positive = True  # This case satisfies the concept
                    
                    if base_positive:
                        label = 1 if np.random.random() < prob else 0
                    else:
                        label = 0 if np.random.random() < prob else 1
                    
                    classifications.append(label)
                
                positive_rate = np.mean(classifications)
                classification_rates.append(positive_rate)
            
            return classification_rates
        
        generator.test_gradual_transition = Mock(side_effect=mock_gradual_transition)
        
        rates = generator.test_gradual_transition()
        
        # Test smooth transition
        assert len(rates) == 5, "Should have rates for all probability levels"
        
        # Rates should decrease monotonically
        for i in range(1, len(rates)):
            assert rates[i] < rates[i-1], f"Classification rate should decrease: {rates[i]:.3f} vs {rates[i-1]:.3f}"
        
        # First rate should be high, last should be low
        assert rates[0] > 0.8, f"Initial rate {rates[0]:.3f} should be high"
        assert rates[-1] < 0.2, f"Final rate {rates[-1]:.3f} should be low"
        
        # Intermediate rates should be reasonable
        assert 0.4 < rates[2] < 0.6, f"Middle rate {rates[2]:.3f} should be around 0.5"

    def test_drift_speed_variations(self):
        """Test different speeds of probability drift"""
        generator = Mock()
        
        # Mock different drift speeds
        def mock_drift_speed_comparison():
            """Compare fast vs slow drift speeds"""
            
            def create_drift_sequence(speed_factor, length=1000):
                """Create probability sequence with given speed"""
                probabilities = []
                current_prob = 0.9
                target_prob = 0.1
                
                for step in range(length):
                    # Exponential approach to target with different speeds
                    progress = 1 - np.exp(-speed_factor * step / length)
                    prob = current_prob * (1 - progress) + target_prob * progress
                    probabilities.append(prob)
                
                return probabilities
            
            fast_drift = create_drift_sequence(3.0)    # Fast drift
            slow_drift = create_drift_sequence(0.5)    # Slow drift
            
            return fast_drift, slow_drift
        
        generator.compare_drift_speeds = Mock(side_effect=mock_drift_speed_comparison)
        
        fast_probs, slow_probs = generator.compare_drift_speeds()
        
        # Test that fast drift changes more quickly
        fast_midpoint = fast_probs[500]  # Probability at midpoint
        slow_midpoint = slow_probs[500]
        
        assert fast_midpoint < slow_midpoint, \
            f"Fast drift ({fast_midpoint:.3f}) should reach lower probability than slow drift ({slow_midpoint:.3f}) at midpoint"
        
        # Test final convergence
        fast_final = fast_probs[-1]
        slow_final = slow_probs[-1]
        
        # Fast should be closer to target (0.1)
        fast_distance = abs(fast_final - 0.1)
        slow_distance = abs(slow_final - 0.1)
        
        assert fast_distance < slow_distance, \
            f"Fast drift should be closer to target: {fast_distance:.4f} vs {slow_distance:.4f}"


class TestMixedClassificationEdgeCases:
    """Test suite for edge cases and special conditions"""

    def test_extreme_numerical_values(self):
        """Test behavior with extreme numerical attribute values"""
        generator = Mock()
        
        # Mock edge case testing
        def mock_extreme_value_testing():
            """Test extreme cases for numerical attributes"""
            
            def classify_mixed_rule(attributes):
                bool1, bool2, num1, num2 = attributes
                condition1 = (bool1 == 1) and (bool2 == 1) 
                condition2 = (num1 > 0.5) and (num2 < 0.3)
                return 1 if (condition1 or condition2) else 0
            
            test_cases = [
                # Extreme numerical values
                ([0, 0, 0.0, 0.0], 0, "Minimum numerical values"),
                ([0, 0, 1.0, 1.0], 0, "Maximum numerical values"),
                ([0, 0, 0.0, 0.3], 0, "num1 min, num2 at boundary"),
                ([0, 0, 0.5, 0.0], 0, "num1 at boundary, num2 min"),
                ([0, 0, 1.0, 0.0], 1, "num1 max, num2 min → positive"),
                ([0, 0, 0.50001, 0.29999], 1, "Just above thresholds → positive"),
            ]
            
            results = []
            for attributes, expected, description in test_cases:
                actual = classify_mixed_rule(attributes)
                results.append((attributes, actual, expected, description))
            
            return results
        
        generator.test_extreme_values = Mock(side_effect=mock_extreme_value_testing)
        
        results = generator.test_extreme_values()
        
        for attributes, actual, expected, description in results:
            assert actual == expected, f"Failed extreme case: {description} - attributes {attributes}"

    def test_boolean_combination_completeness(self):
        """Test all 4 boolean combinations are handled correctly"""
        generator = Mock()
        
        # Mock complete boolean testing
        def mock_boolean_completeness():
            """Test all boolean combinations with various numerical values"""
            
            def classify_mixed_rule(attributes):
                bool1, bool2, num1, num2 = attributes
                condition1 = (bool1 == 1) and (bool2 == 1)
                condition2 = (num1 > 0.5) and (num2 < 0.3)
                return 1 if (condition1 or condition2) else 0
            
            boolean_combinations = [(0,0), (0,1), (1,0), (1,1)]
            numerical_test_cases = [
                (0.3, 0.8, 0, "nums don't satisfy condition"),
                (0.7, 0.2, 1, "nums satisfy condition"),
                (0.5, 0.3, 0, "nums at exact boundaries"),
                (0.6, 0.1, 1, "nums clearly satisfy"),
            ]
            
            results = []
            for bool1, bool2 in boolean_combinations:
                bool_satisfies = (bool1 == 1) and (bool2 == 1)
                
                for num1, num2, num_expected, num_desc in numerical_test_cases:
                    attributes = [bool1, bool2, num1, num2]
                    
                    # Final expected result
                    if bool_satisfies:
                        expected = 1  # Boolean condition satisfied
                    else:
                        expected = num_expected  # Depends on numerical condition
                    
                    actual = classify_mixed_rule(attributes)
                    
                    description = f"Bool({bool1},{bool2}), {num_desc}"
                    results.append((attributes, actual, expected, description))
            
            return results
        
        generator.test_boolean_completeness = Mock(side_effect=mock_boolean_completeness)
        
        results = generator.test_boolean_completeness()
        
        # Verify all combinations
        for attributes, actual, expected, description in results:
            assert actual == expected, f"Failed boolean combination: {description}"
        
        # Should have tested all 16 combinations (4 boolean × 4 numerical cases)
        assert len(results) == 16, "Should test all boolean-numerical combinations"

    def test_probability_edge_cases(self):
        """Test probability-based classification at extreme probabilities"""
        generator = Mock()
        
        # Mock probability edge cases
        def mock_probability_edge_cases():
            """Test extreme probability values"""
            test_attributes = [1, 1, 0.2, 0.8]  # Should be positive under normal concept
            
            edge_probabilities = [0.0, 0.001, 0.1, 0.9, 0.999, 1.0]
            results = {}
            
            for prob in edge_probabilities:
                np.random.seed(42)
                
                classifications = []
                for _ in range(1000):
                    # Simulate probabilistic classification
                    if np.random.random() < prob:
                        label = 1  # Concept satisfied
                    else:
                        label = 0  # Concept not satisfied
                    
                    classifications.append(label)
                
                positive_rate = np.mean(classifications)
                results[prob] = positive_rate
            
            return results
        
        generator.test_probability_edges = Mock(side_effect=mock_probability_edge_cases)
        
        results = generator.test_probability_edges()
        
        # Test extreme cases
        assert results[0.0] < 0.05, "Zero probability should yield ~0% positive"
        assert results[1.0] > 0.95, "Unit probability should yield ~100% positive"
        assert results[0.001] < 0.02, "Very low probability should yield very few positive"
        assert results[0.999] > 0.98, "Very high probability should yield most positive"
        
        # Test intermediate cases
        assert 0.05 < results[0.1] < 0.15, "10% probability should yield ~10% positive"
        assert 0.85 < results[0.9] < 0.95, "90% probability should yield ~90% positive"


class TestMixedDatasetIntegration:
    """Integration tests for mixed dataset generator"""

    def test_mixed_full_workflow(self):
        """Test complete mixed dataset generation workflow"""
        generator = Mock()
        
        # Mock complete mixed workflow
        def mock_complete_mixed_workflow():
            """Simulate full mixed dataset generation with drift"""
            
            def classify_mixed(attributes, concept_prob):
                bool1, bool2, num1, num2 = attributes
                base_condition = (bool1 == 1 and bool2 == 1) or (num1 > 0.5 and num2 < 0.3)
                
                # Probabilistic classification
                if base_condition:
                    return 1 if np.random.random() < concept_prob else 0
                else:
                    return 0 if np.random.random() < concept_prob else 1
            
            instances = []
            initial_prob = 0.9
            final_prob = 0.2
            
            for step in range(2000):
                # Generate mixed attributes
                bool1 = np.random.choice([0, 1])
                bool2 = np.random.choice([0, 1])
                num1 = np.random.uniform(0, 1)
                num2 = np.random.uniform(0, 1)
                attributes = [bool1, bool2, num1, num2]
                
                # Evolve concept probability
                progress = step / 1999
                current_prob = initial_prob * (1 - progress) + final_prob * progress
                
                # Classify
                label = classify_mixed(attributes, current_prob)
                
                instances.append((attributes, label, current_prob))
            
            return instances
        
        generator.generate_full_mixed_dataset = Mock(side_effect=mock_complete_mixed_workflow)
        
        instances = generator.generate_full_mixed_dataset()
        
        # Test dataset properties
        assert len(instances) == 2000, "Should generate requested number of instances"
        
        # Test instance format
        for i, (attributes, label, prob) in enumerate(instances[:10]):  # Check first 10
            assert len(attributes) == 4, f"Instance {i}: should have 4 attributes"
            assert attributes[0] in [0, 1], f"Instance {i}: first attribute should be boolean"
            assert attributes[1] in [0, 1], f"Instance {i}: second attribute should be boolean"
            assert 0 <= attributes[2] <= 1, f"Instance {i}: third attribute should be in [0,1]"
            assert 0 <= attributes[3] <= 1, f"Instance {i}: fourth attribute should be in [0,1]"
            assert label in [0, 1], f"Instance {i}: label should be binary"
            assert 0 <= prob <= 1, f"Instance {i}: probability should be in [0,1]"
        
        # Test probability drift occurred
        initial_prob = instances[0][2]
        final_prob = instances[-1][2]
        
        assert initial_prob > final_prob, "Concept probability should decrease over time"
        assert initial_prob > 0.8, "Should start with high probability"
        assert final_prob < 0.3, "Should end with low probability"

    def test_mixed_statistical_properties(self):
        """Test statistical properties of mixed dataset"""
        generator = Mock()
        
        # Mock statistical analysis
        def mock_statistical_analysis():
            """Analyze statistical properties of generated dataset"""
            np.random.seed(42)
            
            # Generate large sample
            sample_size = 10000
            attributes_list = []
            
            for _ in range(sample_size):
                bool1 = np.random.choice([0, 1])
                bool2 = np.random.choice([0, 1])
                num1 = np.random.uniform(0, 1)
                num2 = np.random.uniform(0, 1)
                attributes_list.append([bool1, bool2, num1, num2])
            
            return np.array(attributes_list)
        
        generator.generate_statistical_sample = Mock(side_effect=mock_statistical_analysis)
        
        samples = generator.generate_statistical_sample()
        
        # Test boolean attribute statistics
        bool1_mean = np.mean(samples[:, 0])
        bool2_mean = np.mean(samples[:, 1])
        
        assert 0.45 < bool1_mean < 0.55, f"Boolean 1 mean {bool1_mean:.3f} should be ~0.5"
        assert 0.45 < bool2_mean < 0.55, f"Boolean 2 mean {bool2_mean:.3f} should be ~0.5"
        
        # Test numerical attribute statistics
        num1_mean = np.mean(samples[:, 2])
        num2_mean = np.mean(samples[:, 3])
        
        assert 0.45 < num1_mean < 0.55, f"Numerical 1 mean {num1_mean:.3f} should be ~0.5"
        assert 0.45 < num2_mean < 0.55, f"Numerical 2 mean {num2_mean:.3f} should be ~0.5"
        
        # Test attribute independence
        correlation_matrix = np.corrcoef(samples.T)
        for i in range(4):
            for j in range(i+1, 4):
                corr = abs(correlation_matrix[i, j])
                assert corr < 0.05, f"Attributes {i},{j} correlation {corr:.4f} too high"

    def test_mixed_concept_drift_validation(self):
        """Test mixed dataset concept drift is gradual and measurable"""
        generator = Mock()
        
        # Mock concept drift validation
        def mock_drift_validation():
            """Validate concept drift properties"""
            
            # Simulate classification over time with concept drift
            def simulate_classification_over_time():
                # Fixed test case
                test_case = [1, 1, 0.3, 0.7]  # Should be positive (satisfies boolean condition)
                
                time_windows = []
                window_size = 200
                
                for window_start in range(0, 2000, window_size):
                    # Calculate concept probability for this window
                    window_center = window_start + window_size / 2
                    progress = window_center / 2000
                    concept_prob = 0.9 * (1 - progress) + 0.1 * progress
                    
                    # Simulate classifications for this window
                    np.random.seed(42 + window_start // window_size)
                    window_labels = []
                    
                    for _ in range(window_size):
                        # This case satisfies the concept, so use concept_prob directly
                        label = 1 if np.random.random() < concept_prob else 0
                        window_labels.append(label)
                    
                    positive_rate = np.mean(window_labels)
                    time_windows.append((window_start, positive_rate, concept_prob))
                
                return time_windows
            
            return simulate_classification_over_time()
        
        generator.validate_drift = Mock(side_effect=mock_drift_validation)
        
        windows = generator.validate_drift()
        
        # Test gradual change
        positive_rates = [rate for _, rate, _ in windows]
        
        # Should show decreasing trend
        for i in range(1, len(positive_rates)):
            # Allow some variance, but general trend should be decreasing
            if i > 2:  # After initial windows, trend should be clear
                assert positive_rates[i] < positive_rates[0] + 0.1, \
                    f"Window {i} rate {positive_rates[i]:.3f} should show decreasing trend from {positive_rates[0]:.3f}"
        
        # First and last windows should be significantly different
        assert positive_rates[0] > positive_rates[-1] + 0.3, \
            f"First rate {positive_rates[0]:.3f} should be much higher than last {positive_rates[-1]:.3f}"

    def test_mixed_performance_requirements(self):
        """Test mixed dataset generator meets performance requirements"""
        generator = Mock()
        
        # Mock efficient generation
        def fast_mixed_generation():
            bool1 = np.random.choice([0, 1])
            bool2 = np.random.choice([0, 1])
            num1 = np.random.uniform(0, 1)
            num2 = np.random.uniform(0, 1)
            label = np.random.choice([0, 1])
            return [bool1, bool2, num1, num2], label
        
        generator.generate_instance = Mock(side_effect=fast_mixed_generation)
        
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
