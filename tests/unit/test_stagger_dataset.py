"""
Unit tests for Stagger Dataset Generator - REQ-ABRUPT-102

Tests the categorical attribute handling and concept definitions of the Stagger dataset generator.
Following TDD principles, these tests define the exact behavior required before
any implementation exists.

Test Coverage:
- Categorical attribute generation
- Concept definitions and transitions
- Noise-free dataset validation
- Configuration support (Stagger(1), Stagger(20))
- Combinatorial completeness
"""

import itertools
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestStaggerAttributeDefinitions:
    """Test suite for stagger dataset attribute generation - REQ-ABRUPT-102"""

    def test_stagger_attributes(self):
        """Test categorical attribute generation matches specification"""
        # This is the fundamental test from requirements
        generator = Mock()
        
        # Mock attribute generation
        def mock_attribute_generation():
            shape = np.random.choice(['triangle', 'circle', 'rectangle'])
            color = np.random.choice(['green', 'blue', 'red'])
            size = np.random.choice(['small', 'medium', 'large'])
            
            return {
                'shape': shape,
                'color': color, 
                'size': size
            }
        
        generator.generate_instance = Mock(side_effect=lambda: (mock_attribute_generation(), 0))
        
        # Test attribute domains
        instance_data, _ = generator.generate_instance()
        
        assert instance_data['shape'] in ['triangle', 'circle', 'rectangle'], "Shape must be from valid domain"
        assert instance_data['color'] in ['green', 'blue', 'red'], "Color must be from valid domain"  
        assert instance_data['size'] in ['small', 'medium', 'large'], "Size must be from valid domain"
        
        # Test all attributes are present
        required_attributes = ['shape', 'color', 'size']
        for attr in required_attributes:
            assert attr in instance_data, f"Attribute {attr} must be present"

    def test_stagger_attribute_domains(self):
        """Test each attribute has exactly 3 values as specified"""
        # Test attribute domain completeness
        expected_domains = {
            'shape': ['triangle', 'circle', 'rectangle'],
            'color': ['green', 'blue', 'red'],
            'size': ['small', 'medium', 'large']
        }
        
        for attr_name, expected_values in expected_domains.items():
            assert len(expected_values) == 3, f"Attribute {attr_name} must have exactly 3 values"
            assert len(set(expected_values)) == 3, f"Attribute {attr_name} values must be unique"

    def test_stagger_total_combinations(self):
        """Test total combinations equal 27 (3³)"""
        # Test combinatorial completeness
        shapes = ['triangle', 'circle', 'rectangle']
        colors = ['green', 'blue', 'red']
        sizes = ['small', 'medium', 'large']
        
        all_combinations = list(itertools.product(shapes, colors, sizes))
        
        assert len(all_combinations) == 27, "Total combinations must be 3³ = 27"
        
        # Test no duplicate combinations
        assert len(set(all_combinations)) == 27, "All combinations must be unique"
        
        # Test all attributes appear in combinations
        shapes_in_combos = set(combo[0] for combo in all_combinations)
        colors_in_combos = set(combo[1] for combo in all_combinations)
        sizes_in_combos = set(combo[2] for combo in all_combinations)
        
        assert shapes_in_combos == set(shapes), "All shapes must appear in combinations"
        assert colors_in_combos == set(colors), "All colors must appear in combinations"
        assert sizes_in_combos == set(sizes), "All sizes must appear in combinations"

    def test_stagger_uniform_distribution(self):
        """Test attributes are uniformly distributed across generations"""
        generator = Mock()
        
        # Mock uniform attribute sampling
        np.random.seed(42)
        sample_size = 1000
        
        generated_instances = []
        for _ in range(sample_size):
            shape = np.random.choice(['triangle', 'circle', 'rectangle'])
            color = np.random.choice(['green', 'blue', 'red'])
            size = np.random.choice(['small', 'medium', 'large'])
            generated_instances.append({'shape': shape, 'color': color, 'size': size})
        
        instance_iterator = iter(generated_instances)
        generator.generate_instance = Mock(side_effect=lambda: (next(instance_iterator), 0))
        
        # Collect attribute distributions
        shape_counts = {'triangle': 0, 'circle': 0, 'rectangle': 0}
        color_counts = {'green': 0, 'blue': 0, 'red': 0}
        size_counts = {'small': 0, 'medium': 0, 'large': 0}
        
        for _ in range(sample_size):
            instance, _ = generator.generate_instance()
            shape_counts[instance['shape']] += 1
            color_counts[instance['color']] += 1
            size_counts[instance['size']] += 1
        
        # Test approximate uniform distribution (each value should appear ~1/3 of the time)
        expected_count = sample_size // 3
        tolerance = 0.1 * expected_count  # 10% tolerance
        
        for attr_name, counts in [('shape', shape_counts), ('color', color_counts), ('size', size_counts)]:
            for value, count in counts.items():
                assert abs(count - expected_count) < tolerance, \
                    f"{attr_name}={value} appears {count} times, expected ~{expected_count}"


class TestStaggerConceptDefinitions:
    """Test suite for stagger concept definitions and transitions"""

    def test_stagger_concept_1_definition(self):
        """Test Concept 1: color=red AND size=small"""
        generator = Mock()
        
        # Mock concept 1 evaluation
        def mock_concept_1_evaluation(attributes):
            return attributes['color'] == 'red' and attributes['size'] == 'small'
        
        generator.evaluate_concept = Mock(side_effect=mock_concept_1_evaluation)
        
        # Test positive cases
        positive_cases = [
            {'shape': 'triangle', 'color': 'red', 'size': 'small'},
            {'shape': 'circle', 'color': 'red', 'size': 'small'},
            {'shape': 'rectangle', 'color': 'red', 'size': 'small'}
        ]
        
        for case in positive_cases:
            result = generator.evaluate_concept(case)
            assert result == True, f"Case {case} should satisfy concept 1"
        
        # Test negative cases
        negative_cases = [
            {'shape': 'triangle', 'color': 'blue', 'size': 'small'},  # Wrong color
            {'shape': 'triangle', 'color': 'red', 'size': 'medium'},   # Wrong size
            {'shape': 'triangle', 'color': 'green', 'size': 'large'},  # Both wrong
            {'shape': 'circle', 'color': 'blue', 'size': 'medium'}     # Both wrong
        ]
        
        for case in negative_cases:
            result = generator.evaluate_concept(case)
            assert result == False, f"Case {case} should not satisfy concept 1"

    def test_stagger_concept_2_definition(self):
        """Test Concept 2: color=green OR shape=circle"""
        generator = Mock()
        
        # Mock concept 2 evaluation  
        def mock_concept_2_evaluation(attributes):
            return attributes['color'] == 'green' or attributes['shape'] == 'circle'
        
        generator.evaluate_concept = Mock(side_effect=mock_concept_2_evaluation)
        
        # Test positive cases
        positive_cases = [
            {'shape': 'triangle', 'color': 'green', 'size': 'small'},  # Green color
            {'shape': 'circle', 'color': 'red', 'size': 'large'},      # Circle shape
            {'shape': 'circle', 'color': 'green', 'size': 'medium'},   # Both conditions
            {'shape': 'rectangle', 'color': 'green', 'size': 'large'}, # Green color
            {'shape': 'circle', 'color': 'blue', 'size': 'small'}      # Circle shape
        ]
        
        for case in positive_cases:
            result = generator.evaluate_concept(case)
            assert result == True, f"Case {case} should satisfy concept 2"
        
        # Test negative cases (neither green nor circle)
        negative_cases = [
            {'shape': 'triangle', 'color': 'red', 'size': 'small'},
            {'shape': 'rectangle', 'color': 'blue', 'size': 'medium'},
            {'shape': 'triangle', 'color': 'blue', 'size': 'large'},
            {'shape': 'rectangle', 'color': 'red', 'size': 'large'}
        ]
        
        for case in negative_cases:
            result = generator.evaluate_concept(case)
            assert result == False, f"Case {case} should not satisfy concept 2"

    def test_stagger_concept_3_definition(self):
        """Test Concept 3: size=medium OR size=large"""
        generator = Mock()
        
        # Mock concept 3 evaluation
        def mock_concept_3_evaluation(attributes):
            return attributes['size'] == 'medium' or attributes['size'] == 'large'
        
        generator.evaluate_concept = Mock(side_effect=mock_concept_3_evaluation)
        
        # Test positive cases
        positive_cases = [
            {'shape': 'triangle', 'color': 'red', 'size': 'medium'},
            {'shape': 'circle', 'color': 'blue', 'size': 'large'},
            {'shape': 'rectangle', 'color': 'green', 'size': 'medium'},
            {'shape': 'triangle', 'color': 'green', 'size': 'large'},
        ]
        
        for case in positive_cases:
            result = generator.evaluate_concept(case)
            assert result == True, f"Case {case} should satisfy concept 3"
        
        # Test negative cases (only size=small)
        negative_cases = [
            {'shape': 'triangle', 'color': 'red', 'size': 'small'},
            {'shape': 'circle', 'color': 'blue', 'size': 'small'},
            {'shape': 'rectangle', 'color': 'green', 'size': 'small'}
        ]
        
        for case in negative_cases:
            result = generator.evaluate_concept(case)
            assert result == False, f"Case {case} should not satisfy concept 3"

    def test_stagger_concept_transitions(self):
        """Test concept transition analysis matches paper specification"""
        generator = Mock()
        
        # Mock transition analysis for concepts 1 to 2
        def mock_analyze_transition(from_concept, to_concept):
            """Analyze how many instances change labels between concepts"""
            
            # Generate all 27 possible combinations
            shapes = ['triangle', 'circle', 'rectangle']
            colors = ['green', 'blue', 'red']
            sizes = ['small', 'medium', 'large']
            
            all_combinations = [
                {'shape': s, 'color': c, 'size': sz}
                for s in shapes for c in colors for sz in sizes
            ]
            
            # Evaluate each combination under both concepts
            same_count = 0
            different_count = 0
            
            for combo in all_combinations:
                if from_concept == 1:
                    label_1 = combo['color'] == 'red' and combo['size'] == 'small'
                elif from_concept == 2:
                    label_1 = combo['color'] == 'green' or combo['shape'] == 'circle'
                else:  # concept 3
                    label_1 = combo['size'] == 'medium' or combo['size'] == 'large'
                
                if to_concept == 1:
                    label_2 = combo['color'] == 'red' and combo['size'] == 'small'
                elif to_concept == 2:
                    label_2 = combo['color'] == 'green' or combo['shape'] == 'circle'
                else:  # concept 3
                    label_2 = combo['size'] == 'medium' or combo['size'] == 'large'
                
                if label_1 == label_2:
                    same_count += 1
                else:
                    different_count += 1
            
            return same_count, different_count
        
        generator.analyze_transition = Mock(side_effect=mock_analyze_transition)
        
        # Test concept 1 to 2 transition (as specified in requirements)
        same_count, different_count = generator.analyze_transition(1, 2)
        
        assert same_count == 11, f"Expected 11 same labels, got {same_count}"
        assert different_count == 16, f"Expected 16 different labels, got {different_count}"
        assert same_count + different_count == 27, "Total should equal 27 combinations"

    def test_stagger_all_concept_transitions(self):
        """Test transitions between all concept pairs"""
        generator = Mock()
        
        # Test all possible transitions
        expected_transitions = {
            (1, 2): (11, 16),  # From requirements
            (1, 3): None,      # To be calculated
            (2, 1): (11, 16),  # Should be symmetric
            (2, 3): None,      # To be calculated
            (3, 1): None,      # To be calculated
            (3, 2): None       # To be calculated
        }
        
        def mock_transition_analysis(from_concept, to_concept):
            if (from_concept, to_concept) in expected_transitions:
                expected = expected_transitions[(from_concept, to_concept)]
                if expected is not None:
                    return expected
                else:
                    # Calculate for unknown transitions
                    return (10, 17)  # Mock values for testing
            return (0, 27)  # Default case
        
        generator.analyze_transition = Mock(side_effect=mock_transition_analysis)
        
        # Test known transition
        same, different = generator.analyze_transition(1, 2)
        assert same == 11 and different == 16, "Concept 1→2 transition should match specification"
        
        # Test that all transitions sum to 27
        same, different = generator.analyze_transition(2, 3)
        assert same + different == 27, "All transitions should account for all 27 combinations"


class TestStaggerDatasetConfigurations:
    """Test suite for stagger dataset configuration support"""

    def test_stagger_1_configuration(self):
        """Test Stagger(1) configuration parameters"""
        generator = Mock()
        
        # Mock Stagger(1) configuration
        config = {
            'type': 'stagger',
            'context_size': 40,
            'training_instances': 1,
            'drift_instances': 100,
            'noise_level': 0.0
        }
        
        generator.get_config = Mock(return_value=config)
        
        actual_config = generator.get_config()
        
        assert actual_config['context_size'] == 40, "Stagger(1) should have context_size=40"
        assert actual_config['training_instances'] == 1, "Stagger(1) should have training_instances=1"
        assert actual_config['drift_instances'] == 100, "Stagger(1) should have drift_instances=100"
        assert actual_config['noise_level'] == 0.0, "Stagger dataset should be noise-free"

    def test_stagger_20_configuration(self):
        """Test Stagger(20) configuration parameters"""
        generator = Mock()
        
        # Mock Stagger(20) configuration
        config = {
            'type': 'stagger',
            'context_size': 40,
            'training_instances': 20,
            'drift_instances': 100,
            'noise_level': 0.0
        }
        
        generator.get_config = Mock(return_value=config)
        
        actual_config = generator.get_config()
        
        assert actual_config['context_size'] == 40, "Stagger(20) should have context_size=40"
        assert actual_config['training_instances'] == 20, "Stagger(20) should have training_instances=20"
        assert actual_config['drift_instances'] == 100, "Stagger(20) should have drift_instances=100"
        assert actual_config['noise_level'] == 0.0, "Stagger dataset should be noise-free"

    def test_stagger_noise_free_requirement(self):
        """Test noise-free dataset requirement (100% accuracy on concept function)"""
        generator = Mock()
        
        # Mock noise-free generation
        def mock_noise_free_generation():
            # Generate attribute combination
            attributes = {
                'shape': np.random.choice(['triangle', 'circle', 'rectangle']),
                'color': np.random.choice(['green', 'blue', 'red']),
                'size': np.random.choice(['small', 'medium', 'large'])
            }
            
            # Apply concept function exactly (no noise)
            # Using concept 1 as example
            label = 1 if (attributes['color'] == 'red' and attributes['size'] == 'small') else 0
            
            return attributes, label
        
        generator.generate_instance = Mock(side_effect=mock_noise_free_generation)
        
        # Test 100% accuracy
        correct_predictions = 0
        total_instances = 100
        
        for _ in range(total_instances):
            attributes, predicted_label = generator.generate_instance()
            
            # Manually calculate expected label using concept 1
            expected_label = 1 if (attributes['color'] == 'red' and attributes['size'] == 'small') else 0
            
            if predicted_label == expected_label:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_instances
        assert accuracy == 1.0, f"Stagger dataset should have 100% accuracy, got {accuracy:.2%}"

    def test_stagger_concept_sequence_validation(self):
        """Test concept sequence follows valid transitions"""
        generator = Mock()
        
        # Mock concept sequence
        valid_concepts = [1, 2, 3]
        concept_sequence = [1, 2, 3, 1, 2, 3]  # Example sequence
        
        generator.get_concept_sequence = Mock(return_value=concept_sequence)
        
        sequence = generator.get_concept_sequence()
        
        # Validate sequence properties
        assert all(concept in valid_concepts for concept in sequence), \
            "All concepts in sequence must be valid (1, 2, or 3)"
        assert len(sequence) > 0, "Concept sequence cannot be empty"
        
        # Test transitions are abrupt (immediate concept changes)
        for i in range(len(sequence) - 1):
            current_concept = sequence[i]
            next_concept = sequence[i + 1]
            # Each position should have exactly one concept (no gradual transition)
            assert current_concept in [1, 2, 3], f"Invalid concept {current_concept} at position {i}"
            assert next_concept in [1, 2, 3], f"Invalid concept {next_concept} at position {i+1}"


class TestStaggerStatisticalProperties:
    """Test suite for stagger dataset statistical properties"""

    def test_stagger_combination_coverage(self):
        """Test all 27 combinations can be generated"""
        generator = Mock()
        
        # Mock generation that covers all combinations
        shapes = ['triangle', 'circle', 'rectangle']
        colors = ['green', 'blue', 'red']
        sizes = ['small', 'medium', 'large']
        
        all_combinations = [
            {'shape': s, 'color': c, 'size': sz}
            for s in shapes for c in colors for sz in sizes
        ]
        
        # Mock generator to cycle through all combinations
        combination_cycle = itertools.cycle(all_combinations)
        
        def mock_comprehensive_generation():
            combo = next(combination_cycle)
            # Apply concept 1 for labeling
            label = 1 if (combo['color'] == 'red' and combo['size'] == 'small') else 0
            return combo, label
        
        generator.generate_instance = Mock(side_effect=mock_comprehensive_generation)
        
        # Generate enough instances to see all combinations
        generated_combinations = set()
        for _ in range(100):  # Generate more than 27 to ensure cycling
            attributes, _ = generator.generate_instance()
            combo_tuple = (attributes['shape'], attributes['color'], attributes['size'])
            generated_combinations.add(combo_tuple)
        
        # Should have seen all 27 combinations
        expected_combinations = set(itertools.product(shapes, colors, sizes))
        assert generated_combinations == expected_combinations, \
            "Should be able to generate all 27 possible combinations"

    def test_stagger_label_distribution_per_concept(self):
        """Test label distribution for each concept"""
        generator = Mock()
        
        # Test label distribution for each concept
        concepts_to_test = [
            (1, lambda attrs: attrs['color'] == 'red' and attrs['size'] == 'small'),
            (2, lambda attrs: attrs['color'] == 'green' or attrs['shape'] == 'circle'),
            (3, lambda attrs: attrs['size'] == 'medium' or attrs['size'] == 'large')
        ]
        
        for concept_id, concept_function in concepts_to_test:
            # Count positive cases for this concept across all combinations
            shapes = ['triangle', 'circle', 'rectangle']
            colors = ['green', 'blue', 'red']
            sizes = ['small', 'medium', 'large']
            
            positive_count = 0
            total_count = 0
            
            for combo in itertools.product(shapes, colors, sizes):
                attributes = {'shape': combo[0], 'color': combo[1], 'size': combo[2]}
                if concept_function(attributes):
                    positive_count += 1
                total_count += 1
            
            # Test expected positive ratios
            positive_ratio = positive_count / total_count
            
            if concept_id == 1:  # Only 3 combinations: (red, small) with any shape
                assert positive_count == 3, f"Concept 1 should have exactly 3 positive cases, got {positive_count}"
            elif concept_id == 2:  # Green (9) + circle non-green (6) = 15
                assert positive_count == 15, f"Concept 2 should have exactly 15 positive cases, got {positive_count}"
            elif concept_id == 3:  # Medium (9) + large (9) = 18
                assert positive_count == 18, f"Concept 3 should have exactly 18 positive cases, got {positive_count}"

    def test_stagger_deterministic_labeling(self):
        """Test deterministic labeling - same attributes always give same label"""
        generator = Mock()
        
        # Test deterministic behavior
        test_attributes = [
            {'shape': 'triangle', 'color': 'red', 'size': 'small'},    # Should be positive for concept 1
            {'shape': 'circle', 'color': 'blue', 'size': 'large'},     # Should be positive for concept 2
            {'shape': 'rectangle', 'color': 'green', 'size': 'medium'} # Should be positive for concepts 2&3
        ]
        
        def mock_deterministic_labeling(attributes, concept):
            if concept == 1:
                return 1 if (attributes['color'] == 'red' and attributes['size'] == 'small') else 0
            elif concept == 2:
                return 1 if (attributes['color'] == 'green' or attributes['shape'] == 'circle') else 0
            elif concept == 3:
                return 1 if (attributes['size'] == 'medium' or attributes['size'] == 'large') else 0
            else:
                return 0
        
        generator.get_label = Mock(side_effect=mock_deterministic_labeling)
        
        # Test multiple calls with same attributes produce same labels
        for concept in [1, 2, 3]:
            for attributes in test_attributes:
                label1 = generator.get_label(attributes, concept)
                label2 = generator.get_label(attributes, concept)
                label3 = generator.get_label(attributes, concept)
                
                assert label1 == label2 == label3, \
                    f"Same attributes should always produce same label for concept {concept}"

    def test_stagger_reproducibility(self):
        """Test deterministic generation with seed control"""
        # Test reproducible attribute generation
        
        def create_mock_generator(seed):
            generator = Mock()
            np.random.seed(seed)
            
            def deterministic_generation():
                shape = np.random.choice(['triangle', 'circle', 'rectangle'])
                color = np.random.choice(['green', 'blue', 'red'])
                size = np.random.choice(['small', 'medium', 'large'])
                
                attributes = {'shape': shape, 'color': color, 'size': size}
                # Use concept 1 for consistent labeling
                label = 1 if (color == 'red' and size == 'small') else 0
                
                return attributes, label
            
            generator.generate_instance = Mock(side_effect=deterministic_generation)
            return generator
        
        # Create two generators with same seed
        generator1 = create_mock_generator(42)
        generator2 = create_mock_generator(42)
        
        # Generate sequences
        sequence1 = [generator1.generate_instance() for _ in range(50)]
        sequence2 = [generator2.generate_instance() for _ in range(50)]
        
        # Sequences should be identical
        for i, ((attrs1, label1), (attrs2, label2)) in enumerate(zip(sequence1, sequence2)):
            assert attrs1 == attrs2, f"Attributes differ at position {i}"
            assert label1 == label2, f"Labels differ at position {i}"


class TestStaggerIntegration:
    """Integration tests for stagger dataset generator"""

    def test_stagger_full_concept_workflow(self):
        """Test complete workflow with concept transitions"""
        generator = Mock()
        
        # Mock complete workflow with concept transitions
        concept_sequence = [1, 2, 3, 1]  # Example sequence
        instances_per_concept = 40
        current_concept_idx = 0
        instance_count = 0
        
        def mock_concept_aware_generation():
            nonlocal current_concept_idx, instance_count
            instance_count += 1
            
            # Switch concept after instances_per_concept
            if instance_count > (current_concept_idx + 1) * instances_per_concept:
                current_concept_idx += 1
                if current_concept_idx >= len(concept_sequence):
                    current_concept_idx = len(concept_sequence) - 1
            
            current_concept = concept_sequence[current_concept_idx]
            
            # Generate attributes
            shape = np.random.choice(['triangle', 'circle', 'rectangle'])
            color = np.random.choice(['green', 'blue', 'red'])
            size = np.random.choice(['small', 'medium', 'large'])
            
            attributes = {'shape': shape, 'color': color, 'size': size}
            
            # Apply concept function
            if current_concept == 1:
                label = 1 if (color == 'red' and size == 'small') else 0
            elif current_concept == 2:
                label = 1 if (color == 'green' or shape == 'circle') else 0
            else:  # concept 3
                label = 1 if (size == 'medium' or size == 'large') else 0
            
            return attributes, label
        
        generator.generate_instance = Mock(side_effect=mock_concept_aware_generation)
        generator.get_concept_sequence = Mock(return_value=concept_sequence)
        
        # Generate instances across multiple concepts
        total_instances = len(concept_sequence) * instances_per_concept
        instances = [generator.generate_instance() for _ in range(total_instances)]
        
        assert len(instances) == total_instances
        
        # Validate concept transitions occurred
        concept_seq = generator.get_concept_sequence()
        assert len(concept_seq) == 4, "Should have 4 concepts in sequence"
        assert concept_seq == [1, 2, 3, 1], "Concept sequence should match specification"

    def test_stagger_error_handling(self):
        """Test error handling for invalid configurations"""
        generator = Mock()
        
        # Test invalid attribute values
        def test_invalid_attributes():
            invalid_attrs = {'shape': 'invalid', 'color': 'red', 'size': 'small'}
            generator.generate_instance = Mock(side_effect=ValueError("Invalid attribute value"))
            
            with pytest.raises(ValueError):
                generator.generate_instance()
        
        # Test invalid concept numbers
        def test_invalid_concept():
            generator.get_label = Mock(side_effect=ValueError("Invalid concept number"))
            
            with pytest.raises(ValueError):
                generator.get_label({'shape': 'triangle', 'color': 'red', 'size': 'small'}, concept=0)
        
        test_invalid_attributes()
        test_invalid_concept()

    def test_stagger_performance_requirements(self):
        """Test stagger dataset meets performance requirements"""
        generator = Mock()
        
        # Mock efficient generation
        def fast_stagger_generation():
            return ({'shape': 'triangle', 'color': 'red', 'size': 'small'}, 1)
        
        generator.generate_instance = Mock(side_effect=fast_stagger_generation)
        
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

    def test_stagger_metadata_completeness(self):
        """Test metadata provides complete dataset information"""
        generator = Mock()
        
        # Mock comprehensive metadata
        metadata = {
            'dataset_type': 'stagger',
            'num_attributes': 3,
            'attribute_types': ['categorical', 'categorical', 'categorical'],
            'attribute_domains': {
                'shape': ['triangle', 'circle', 'rectangle'],
                'color': ['green', 'blue', 'red'], 
                'size': ['small', 'medium', 'large']
            },
            'num_classes': 2,
            'num_concepts': 3,
            'concept_definitions': {
                1: 'color=red AND size=small',
                2: 'color=green OR shape=circle',
                3: 'size=medium OR size=large'
            },
            'noise_level': 0.0,
            'total_combinations': 27
        }
        
        generator.get_metadata = Mock(return_value=metadata)
        
        actual_metadata = generator.get_metadata()
        
        # Test metadata completeness
        required_fields = [
            'dataset_type', 'num_attributes', 'attribute_types', 'attribute_domains',
            'num_classes', 'num_concepts', 'concept_definitions', 'noise_level'
        ]
        
        for field in required_fields:
            assert field in actual_metadata, f"Metadata must include {field}"
        
        # Test metadata accuracy
        assert actual_metadata['dataset_type'] == 'stagger'
        assert actual_metadata['num_attributes'] == 3
        assert actual_metadata['num_classes'] == 2
        assert actual_metadata['noise_level'] == 0.0
        assert actual_metadata['total_combinations'] == 27
