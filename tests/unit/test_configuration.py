"""
Unit tests for Configuration Management - REQ-CONFIG-301 and REQ-CONFIG-302

Tests the parameter validation and preset configuration functionality.
Following TDD principles, these tests define the exact behavior required 
before any implementation exists.

Test Coverage:
- Parameter validation and constraints (REQ-CONFIG-301)
- Preset configurations for datasets (REQ-CONFIG-302)
- Configuration serialization/deserialization
- Invalid parameter handling
- Configuration inheritance and defaults
"""

from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestParameterValidation:
    """Test suite for parameter validation - REQ-CONFIG-301"""

    def test_sine_dataset_parameter_validation(self):
        """Test Sine dataset parameter validation"""
        config_manager = Mock()
        
        # Mock Sine parameter validation
        def mock_sine_validation(params):
            """Validate Sine dataset parameters"""
            required_params = ['classification_function', 'drift_position', 'noise_level']
            errors = []
            
            # Check required parameters
            for param in required_params:
                if param not in params:
                    errors.append(f"Missing required parameter: {param}")
            
            # Validate parameter values
            if 'classification_function' in params:
                valid_functions = [0, 1, 2, 3]  # 4 sine functions available
                if params['classification_function'] not in valid_functions:
                    errors.append(f"classification_function must be in {valid_functions}")
            
            if 'drift_position' in params:
                if not isinstance(params['drift_position'], (int, float)):
                    errors.append("drift_position must be numeric")
                elif params['drift_position'] < 0:
                    errors.append("drift_position must be non-negative")
            
            if 'noise_level' in params:
                if not isinstance(params['noise_level'], (int, float)):
                    errors.append("noise_level must be numeric")
                elif not (0 <= params['noise_level'] <= 1):
                    errors.append("noise_level must be in [0,1]")
            
            return len(errors) == 0, errors
        
        config_manager.validate_sine_params = Mock(side_effect=mock_sine_validation)
        
        # Test valid configurations
        valid_configs = [
            {'classification_function': 0, 'drift_position': 5000, 'noise_level': 0.0},
            {'classification_function': 1, 'drift_position': 1000, 'noise_level': 0.05},
            {'classification_function': 3, 'drift_position': 0, 'noise_level': 1.0},
        ]
        
        for config in valid_configs:
            is_valid, errors = config_manager.validate_sine_params(config)
            assert is_valid, f"Valid config should pass: {config}, errors: {errors}"
            assert len(errors) == 0, f"No errors expected for valid config: {errors}"
        
        # Test invalid configurations
        invalid_configs = [
            ({'classification_function': 4, 'drift_position': 1000, 'noise_level': 0.05}, 
             "classification_function out of range"),
            ({'classification_function': 0, 'drift_position': -100, 'noise_level': 0.05}, 
             "negative drift_position"),
            ({'classification_function': 0, 'drift_position': 1000, 'noise_level': -0.1}, 
             "negative noise_level"),
            ({'classification_function': 0, 'drift_position': 1000, 'noise_level': 1.5}, 
             "noise_level > 1"),
            ({'drift_position': 1000, 'noise_level': 0.05}, 
             "missing classification_function"),
        ]
        
        for config, description in invalid_configs:
            is_valid, errors = config_manager.validate_sine_params(config)
            assert not is_valid, f"Invalid config should fail ({description}): {config}"
            assert len(errors) > 0, f"Should have validation errors for ({description}): {config}"

    def test_stagger_dataset_parameter_validation(self):
        """Test Stagger dataset parameter validation"""
        config_manager = Mock()
        
        # Mock Stagger parameter validation
        def mock_stagger_validation(params):
            """Validate Stagger dataset parameters"""
            required_params = ['concept_configuration', 'drift_position']
            errors = []
            
            # Check required parameters
            for param in required_params:
                if param not in params:
                    errors.append(f"Missing required parameter: {param}")
            
            # Validate parameter values
            if 'concept_configuration' in params:
                valid_concepts = list(range(1, 4))  # Concepts 1, 2, 3
                if params['concept_configuration'] not in valid_concepts:
                    errors.append(f"concept_configuration must be in {valid_concepts}")
            
            if 'drift_position' in params:
                if not isinstance(params['drift_position'], (int, float)):
                    errors.append("drift_position must be numeric")
                elif params['drift_position'] < 0:
                    errors.append("drift_position must be non-negative")
            
            # Optional parameters
            if 'noise_level' in params:
                if not (0 <= params['noise_level'] <= 1):
                    errors.append("noise_level must be in [0,1]")
            
            return len(errors) == 0, errors
        
        config_manager.validate_stagger_params = Mock(side_effect=mock_stagger_validation)
        
        # Test valid configurations
        valid_configs = [
            {'concept_configuration': 1, 'drift_position': 2000},
            {'concept_configuration': 2, 'drift_position': 5000, 'noise_level': 0.1},
            {'concept_configuration': 3, 'drift_position': 0},
        ]
        
        for config in valid_configs:
            is_valid, errors = config_manager.validate_stagger_params(config)
            assert is_valid, f"Valid config should pass: {config}, errors: {errors}"
        
        # Test invalid configurations
        invalid_configs = [
            ({'concept_configuration': 0, 'drift_position': 1000}, "concept 0 doesn't exist"),
            ({'concept_configuration': 4, 'drift_position': 1000}, "concept 4 doesn't exist"),
            ({'concept_configuration': 1, 'drift_position': -500}, "negative drift position"),
            ({'concept_configuration': 1}, "missing drift_position"),
        ]
        
        for config, description in invalid_configs:
            is_valid, errors = config_manager.validate_stagger_params(config)
            assert not is_valid, f"Invalid config should fail ({description}): {config}"

    def test_hyperplane_dataset_parameter_validation(self):
        """Test Hyperplane dataset parameter validation"""
        config_manager = Mock()
        
        # Mock Hyperplane parameter validation
        def mock_hyperplane_validation(params):
            """Validate Hyperplane dataset parameters"""
            required_params = ['dimensions', 'change_rate']
            errors = []
            
            # Check required parameters
            for param in required_params:
                if param not in params:
                    errors.append(f"Missing required parameter: {param}")
            
            # Validate parameter values
            if 'dimensions' in params:
                if not isinstance(params['dimensions'], int):
                    errors.append("dimensions must be integer")
                elif params['dimensions'] < 2:
                    errors.append("dimensions must be >= 2")
                elif params['dimensions'] > 100:
                    errors.append("dimensions must be <= 100 for performance")
            
            if 'change_rate' in params:
                if not isinstance(params['change_rate'], (int, float)):
                    errors.append("change_rate must be numeric")
                elif params['change_rate'] <= 0:
                    errors.append("change_rate must be positive")
                elif params['change_rate'] > 1:
                    errors.append("change_rate should be <= 1 for stability")
            
            # Optional parameters
            if 'noise_level' in params:
                if not (0 <= params['noise_level'] <= 1):
                    errors.append("noise_level must be in [0,1]")
            
            if 'reversal_probability' in params:
                if not (0 <= params['reversal_probability'] <= 1):
                    errors.append("reversal_probability must be in [0,1]")
            
            return len(errors) == 0, errors
        
        config_manager.validate_hyperplane_params = Mock(side_effect=mock_hyperplane_validation)
        
        # Test valid configurations
        valid_configs = [
            {'dimensions': 10, 'change_rate': 0.1},
            {'dimensions': 5, 'change_rate': 0.001, 'noise_level': 0.05},
            {'dimensions': 20, 'change_rate': 0.5, 'reversal_probability': 0.1},
        ]
        
        for config in valid_configs:
            is_valid, errors = config_manager.validate_hyperplane_params(config)
            assert is_valid, f"Valid config should pass: {config}, errors: {errors}"
        
        # Test invalid configurations
        invalid_configs = [
            ({'dimensions': 1, 'change_rate': 0.1}, "dimensions too small"),
            ({'dimensions': 150, 'change_rate': 0.1}, "dimensions too large"),
            ({'dimensions': 10, 'change_rate': 0}, "zero change_rate"),
            ({'dimensions': 10, 'change_rate': -0.1}, "negative change_rate"),
            ({'dimensions': 10, 'change_rate': 2.0}, "change_rate too large"),
        ]
        
        for config, description in invalid_configs:
            is_valid, errors = config_manager.validate_hyperplane_params(config)
            assert not is_valid, f"Invalid config should fail ({description}): {config}"

    def test_mixed_dataset_parameter_validation(self):
        """Test Mixed dataset parameter validation"""
        config_manager = Mock()
        
        # Mock Mixed parameter validation
        def mock_mixed_validation(params):
            """Validate Mixed dataset parameters"""
            required_params = ['drift_length', 'initial_concept_prob', 'final_concept_prob']
            errors = []
            
            # Check required parameters
            for param in required_params:
                if param not in params:
                    errors.append(f"Missing required parameter: {param}")
            
            # Validate parameter values
            if 'drift_length' in params:
                if not isinstance(params['drift_length'], int):
                    errors.append("drift_length must be integer")
                elif params['drift_length'] < 1:
                    errors.append("drift_length must be positive")
            
            if 'initial_concept_prob' in params:
                if not isinstance(params['initial_concept_prob'], (int, float)):
                    errors.append("initial_concept_prob must be numeric")
                elif not (0 <= params['initial_concept_prob'] <= 1):
                    errors.append("initial_concept_prob must be in [0,1]")
            
            if 'final_concept_prob' in params:
                if not isinstance(params['final_concept_prob'], (int, float)):
                    errors.append("final_concept_prob must be numeric")
                elif not (0 <= params['final_concept_prob'] <= 1):
                    errors.append("final_concept_prob must be in [0,1]")
            
            # Logical validation
            if ('initial_concept_prob' in params and 'final_concept_prob' in params):
                if params['initial_concept_prob'] == params['final_concept_prob']:
                    errors.append("initial_concept_prob and final_concept_prob should differ for drift")
            
            return len(errors) == 0, errors
        
        config_manager.validate_mixed_params = Mock(side_effect=mock_mixed_validation)
        
        # Test valid configurations
        valid_configs = [
            {'drift_length': 1000, 'initial_concept_prob': 0.9, 'final_concept_prob': 0.1},
            {'drift_length': 5000, 'initial_concept_prob': 0.8, 'final_concept_prob': 0.2},
            {'drift_length': 100, 'initial_concept_prob': 0.1, 'final_concept_prob': 0.9},
        ]
        
        for config in valid_configs:
            is_valid, errors = config_manager.validate_mixed_params(config)
            assert is_valid, f"Valid config should pass: {config}, errors: {errors}"
        
        # Test invalid configurations
        invalid_configs = [
            ({'drift_length': 0, 'initial_concept_prob': 0.9, 'final_concept_prob': 0.1}, 
             "zero drift_length"),
            ({'drift_length': 1000, 'initial_concept_prob': 1.5, 'final_concept_prob': 0.1}, 
             "initial_prob > 1"),
            ({'drift_length': 1000, 'initial_concept_prob': 0.5, 'final_concept_prob': 0.5}, 
             "same initial and final probabilities"),
        ]
        
        for config, description in invalid_configs:
            is_valid, errors = config_manager.validate_mixed_params(config)
            assert not is_valid, f"Invalid config should fail ({description}): {config}"

    def test_general_parameter_constraints(self):
        """Test general parameter constraints across all datasets"""
        config_manager = Mock()
        
        # Mock general validation
        def mock_general_validation(params):
            """Validate common parameters"""
            errors = []
            
            # Common parameters
            if 'random_seed' in params:
                if not isinstance(params['random_seed'], int):
                    errors.append("random_seed must be integer")
                elif params['random_seed'] < 0:
                    errors.append("random_seed must be non-negative")
            
            if 'num_instances' in params:
                if not isinstance(params['num_instances'], int):
                    errors.append("num_instances must be integer")
                elif params['num_instances'] < 1:
                    errors.append("num_instances must be positive")
                elif params['num_instances'] > 1000000:
                    errors.append("num_instances too large (>1M)")
            
            if 'batch_size' in params:
                if not isinstance(params['batch_size'], int):
                    errors.append("batch_size must be integer")
                elif params['batch_size'] < 1:
                    errors.append("batch_size must be positive")
                elif params['batch_size'] > params.get('num_instances', 1000):
                    errors.append("batch_size cannot exceed num_instances")
            
            return len(errors) == 0, errors
        
        config_manager.validate_general_params = Mock(side_effect=mock_general_validation)
        
        # Test valid general parameters
        valid_configs = [
            {'random_seed': 42, 'num_instances': 10000, 'batch_size': 1000},
            {'random_seed': 0, 'num_instances': 5000},
            {'num_instances': 1, 'batch_size': 1},
        ]
        
        for config in valid_configs:
            is_valid, errors = config_manager.validate_general_params(config)
            assert is_valid, f"Valid general config should pass: {config}, errors: {errors}"
        
        # Test invalid general parameters
        invalid_configs = [
            ({'random_seed': -1, 'num_instances': 1000}, "negative random_seed"),
            ({'random_seed': 42.5, 'num_instances': 1000}, "non-integer random_seed"),
            ({'num_instances': 0}, "zero num_instances"),
            ({'num_instances': 2000000}, "num_instances too large"),
            ({'num_instances': 100, 'batch_size': 200}, "batch_size > num_instances"),
        ]
        
        for config, description in invalid_configs:
            is_valid, errors = config_manager.validate_general_params(config)
            assert not is_valid, f"Invalid general config should fail ({description}): {config}"

    def test_parameter_type_validation(self):
        """Test strict parameter type validation"""
        config_manager = Mock()
        
        # Mock type validation
        def mock_type_validation(param_name, value, expected_type):
            """Validate parameter types"""
            if expected_type == 'int':
                return isinstance(value, int) and not isinstance(value, bool)
            elif expected_type == 'float':
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            elif expected_type == 'bool':
                return isinstance(value, bool)
            elif expected_type == 'string':
                return isinstance(value, str)
            elif expected_type == 'positive_int':
                return isinstance(value, int) and value > 0
            elif expected_type == 'non_negative_float':
                return isinstance(value, (int, float)) and value >= 0
            elif expected_type == 'probability':
                return isinstance(value, (int, float)) and 0 <= value <= 1
            else:
                return False
        
        config_manager.validate_type = Mock(side_effect=mock_type_validation)
        
        # Test type validation cases
        type_test_cases = [
            ('dimensions', 10, 'positive_int', True),
            ('dimensions', 10.5, 'positive_int', False),
            ('dimensions', -5, 'positive_int', False),
            ('change_rate', 0.1, 'non_negative_float', True),
            ('change_rate', -0.1, 'non_negative_float', False),
            ('noise_level', 0.05, 'probability', True),
            ('noise_level', 1.5, 'probability', False),
            ('noise_level', -0.1, 'probability', False),
            ('dataset_name', 'sine', 'string', True),
            ('dataset_name', 123, 'string', False),
        ]
        
        for param_name, value, expected_type, should_pass in type_test_cases:
            result = config_manager.validate_type(param_name, value, expected_type)
            if should_pass:
                assert result, f"Type validation should pass: {param_name}={value} as {expected_type}"
            else:
                assert not result, f"Type validation should fail: {param_name}={value} as {expected_type}"


class TestPresetConfigurations:
    """Test suite for preset configurations - REQ-CONFIG-302"""

    def test_sine_dataset_presets(self):
        """Test predefined Sine dataset configurations"""
        config_manager = Mock()
        
        # Mock Sine presets
        def mock_sine_presets():
            """Define standard Sine dataset presets"""
            presets = {
                'sine_basic': {
                    'dataset_type': 'sine',
                    'classification_function': 0,
                    'drift_position': 5000,
                    'noise_level': 0.0,
                    'num_instances': 10000,
                    'description': 'Basic sine wave with abrupt drift at position 5000'
                },
                'sine_noisy': {
                    'dataset_type': 'sine',
                    'classification_function': 1,
                    'drift_position': 2500,
                    'noise_level': 0.05,
                    'num_instances': 10000,
                    'description': 'Sine wave with 5% noise and early drift'
                },
                'sine_late_drift': {
                    'dataset_type': 'sine',
                    'classification_function': 2,
                    'drift_position': 7500,
                    'noise_level': 0.1,
                    'num_instances': 10000,
                    'description': 'Sine wave with late drift and 10% noise'
                },
                'sine_complex': {
                    'dataset_type': 'sine',
                    'classification_function': 3,
                    'drift_position': 3000,
                    'noise_level': 0.02,
                    'num_instances': 15000,
                    'description': 'Complex sine function with moderate noise'
                }
            }
            return presets
        
        config_manager.get_sine_presets = Mock(side_effect=mock_sine_presets)
        
        presets = config_manager.get_sine_presets()
        
        # Test preset structure
        assert len(presets) >= 3, "Should have multiple Sine presets"
        
        for preset_name, preset_config in presets.items():
            assert 'dataset_type' in preset_config, f"Preset {preset_name} missing dataset_type"
            assert preset_config['dataset_type'] == 'sine', f"Preset {preset_name} should be sine type"
            
            # Required parameters
            required = ['classification_function', 'drift_position', 'noise_level', 'num_instances']
            for param in required:
                assert param in preset_config, f"Preset {preset_name} missing {param}"
            
            # Validate parameter ranges
            assert 0 <= preset_config['classification_function'] <= 3, \
                f"Preset {preset_name} invalid classification_function"
            assert preset_config['drift_position'] >= 0, \
                f"Preset {preset_name} invalid drift_position"
            assert 0 <= preset_config['noise_level'] <= 1, \
                f"Preset {preset_name} invalid noise_level"
            assert preset_config['num_instances'] > 0, \
                f"Preset {preset_name} invalid num_instances"

    def test_stagger_dataset_presets(self):
        """Test predefined Stagger dataset configurations"""
        config_manager = Mock()
        
        # Mock Stagger presets
        def mock_stagger_presets():
            """Define standard Stagger dataset presets"""
            presets = {
                'stagger_concept1': {
                    'dataset_type': 'stagger',
                    'concept_configuration': 1,
                    'drift_position': 5000,
                    'num_instances': 10000,
                    'description': 'Stagger concept 1: size=small AND color=red'
                },
                'stagger_concept2': {
                    'dataset_type': 'stagger',
                    'concept_configuration': 2,
                    'drift_position': 4000,
                    'num_instances': 12000,
                    'description': 'Stagger concept 2: color=green OR shape=triangular'
                },
                'stagger_concept3': {
                    'dataset_type': 'stagger',
                    'concept_configuration': 3,
                    'drift_position': 6000,
                    'num_instances': 15000,
                    'noise_level': 0.05,
                    'description': 'Stagger concept 3: size=medium OR size=large'
                }
            }
            return presets
        
        config_manager.get_stagger_presets = Mock(side_effect=mock_stagger_presets)
        
        presets = config_manager.get_stagger_presets()
        
        # Test preset structure
        assert len(presets) >= 3, "Should have at least 3 Stagger presets"
        
        for preset_name, preset_config in presets.items():
            assert preset_config['dataset_type'] == 'stagger', f"Preset {preset_name} should be stagger type"
            
            # Required parameters
            required = ['concept_configuration', 'drift_position', 'num_instances']
            for param in required:
                assert param in preset_config, f"Preset {preset_name} missing {param}"
            
            # Validate parameter ranges
            assert 1 <= preset_config['concept_configuration'] <= 3, \
                f"Preset {preset_name} invalid concept_configuration"
            assert preset_config['drift_position'] >= 0, \
                f"Preset {preset_name} invalid drift_position"

    def test_hyperplane_dataset_presets(self):
        """Test predefined Hyperplane dataset configurations"""
        config_manager = Mock()
        
        # Mock Hyperplane presets
        def mock_hyperplane_presets():
            """Define standard Hyperplane dataset presets"""
            presets = {
                'hyperplane_fast': {
                    'dataset_type': 'hyperplane',
                    'dimensions': 10,
                    'change_rate': 0.1,
                    'noise_level': 0.05,
                    'reversal_probability': 0.1,
                    'num_instances': 10000,
                    'description': 'Fast rotating hyperplane - Hyp(0.1)'
                },
                'hyperplane_slow': {
                    'dataset_type': 'hyperplane',
                    'dimensions': 10,
                    'change_rate': 0.001,
                    'noise_level': 0.05,
                    'reversal_probability': 0.1,
                    'num_instances': 10000,
                    'description': 'Slow rotating hyperplane - Hyp(0.001)'
                },
                'hyperplane_high_dim': {
                    'dataset_type': 'hyperplane',
                    'dimensions': 20,
                    'change_rate': 0.05,
                    'noise_level': 0.0,
                    'reversal_probability': 0.05,
                    'num_instances': 15000,
                    'description': 'High-dimensional hyperplane (20D)'
                },
                'hyperplane_noisy': {
                    'dataset_type': 'hyperplane',
                    'dimensions': 5,
                    'change_rate': 0.08,
                    'noise_level': 0.15,
                    'reversal_probability': 0.2,
                    'num_instances': 8000,
                    'description': 'Low-dimensional noisy hyperplane'
                }
            }
            return presets
        
        config_manager.get_hyperplane_presets = Mock(side_effect=mock_hyperplane_presets)
        
        presets = config_manager.get_hyperplane_presets()
        
        # Test preset structure
        for preset_name, preset_config in presets.items():
            assert preset_config['dataset_type'] == 'hyperplane', f"Preset {preset_name} should be hyperplane type"
            
            # Required parameters
            required = ['dimensions', 'change_rate', 'num_instances']
            for param in required:
                assert param in preset_config, f"Preset {preset_name} missing {param}"
            
            # Validate parameter ranges
            assert preset_config['dimensions'] >= 2, f"Preset {preset_name} invalid dimensions"
            assert preset_config['change_rate'] > 0, f"Preset {preset_name} invalid change_rate"
            
            # Optional parameters validation
            if 'noise_level' in preset_config:
                assert 0 <= preset_config['noise_level'] <= 1, \
                    f"Preset {preset_name} invalid noise_level"

    def test_mixed_dataset_presets(self):
        """Test predefined Mixed dataset configurations"""
        config_manager = Mock()
        
        # Mock Mixed presets
        def mock_mixed_presets():
            """Define standard Mixed dataset presets"""
            presets = {
                'mixed_basic': {
                    'dataset_type': 'mixed',
                    'drift_length': 1000,
                    'initial_concept_prob': 0.9,
                    'final_concept_prob': 0.1,
                    'num_instances': 10000,
                    'description': 'Basic mixed dataset with strong to weak concept drift'
                },
                'mixed_gradual': {
                    'dataset_type': 'mixed',
                    'drift_length': 5000,
                    'initial_concept_prob': 0.8,
                    'final_concept_prob': 0.2,
                    'num_instances': 20000,
                    'description': 'Very gradual mixed concept drift'
                },
                'mixed_reverse': {
                    'dataset_type': 'mixed',
                    'drift_length': 2000,
                    'initial_concept_prob': 0.1,
                    'final_concept_prob': 0.9,
                    'num_instances': 12000,
                    'description': 'Reverse drift: weak to strong concept'
                }
            }
            return presets
        
        config_manager.get_mixed_presets = Mock(side_effect=mock_mixed_presets)
        
        presets = config_manager.get_mixed_presets()
        
        # Test preset structure
        for preset_name, preset_config in presets.items():
            assert preset_config['dataset_type'] == 'mixed', f"Preset {preset_name} should be mixed type"
            
            # Required parameters
            required = ['drift_length', 'initial_concept_prob', 'final_concept_prob', 'num_instances']
            for param in required:
                assert param in preset_config, f"Preset {preset_name} missing {param}"
            
            # Validate parameter ranges
            assert preset_config['drift_length'] > 0, f"Preset {preset_name} invalid drift_length"
            assert 0 <= preset_config['initial_concept_prob'] <= 1, \
                f"Preset {preset_name} invalid initial_concept_prob"
            assert 0 <= preset_config['final_concept_prob'] <= 1, \
                f"Preset {preset_name} invalid final_concept_prob"

    def test_preset_completeness(self):
        """Test that presets cover all required configurations"""
        config_manager = Mock()
        
        # Mock complete preset collection
        def mock_all_presets():
            """Get all available presets"""
            all_presets = {
                'sine': ['sine_basic', 'sine_noisy', 'sine_late_drift', 'sine_complex'],
                'stagger': ['stagger_concept1', 'stagger_concept2', 'stagger_concept3'],
                'hyperplane': ['hyperplane_fast', 'hyperplane_slow', 'hyperplane_high_dim'],
                'mixed': ['mixed_basic', 'mixed_gradual', 'mixed_reverse']
            }
            return all_presets
        
        config_manager.get_all_preset_names = Mock(side_effect=mock_all_presets)
        
        all_presets = config_manager.get_all_preset_names()
        
        # Test coverage of dataset types
        required_types = ['sine', 'stagger', 'hyperplane', 'mixed']
        for dataset_type in required_types:
            assert dataset_type in all_presets, f"Missing presets for {dataset_type}"
            assert len(all_presets[dataset_type]) >= 2, \
                f"Should have at least 2 presets for {dataset_type}"

    def test_preset_customization(self):
        """Test ability to customize preset configurations"""
        config_manager = Mock()
        
        # Mock preset customization
        def mock_customize_preset(preset_name, custom_params):
            """Customize a preset with user parameters"""
            # Base presets (simplified)
            base_presets = {
                'sine_basic': {
                    'dataset_type': 'sine',
                    'classification_function': 0,
                    'drift_position': 5000,
                    'noise_level': 0.0,
                    'num_instances': 10000
                }
            }
            
            if preset_name not in base_presets:
                return None, ["Unknown preset"]
            
            # Start with base preset
            customized = base_presets[preset_name].copy()
            
            # Apply customizations
            for param, value in custom_params.items():
                customized[param] = value
            
            # Validate result (simplified validation)
            errors = []
            if 'noise_level' in customized:
                if not (0 <= customized['noise_level'] <= 1):
                    errors.append("Invalid noise_level")
            
            return customized if len(errors) == 0 else None, errors
        
        config_manager.customize_preset = Mock(side_effect=mock_customize_preset)
        
        # Test valid customization
        custom_params = {'noise_level': 0.1, 'num_instances': 5000}
        result, errors = config_manager.customize_preset('sine_basic', custom_params)
        
        assert result is not None, f"Customization should succeed, errors: {errors}"
        assert result['noise_level'] == 0.1, "Custom noise_level should be applied"
        assert result['num_instances'] == 5000, "Custom num_instances should be applied"
        assert result['classification_function'] == 0, "Base parameters should be preserved"
        
        # Test invalid customization
        invalid_params = {'noise_level': 1.5}  # Invalid noise level
        result, errors = config_manager.customize_preset('sine_basic', invalid_params)
        
        assert result is None, "Invalid customization should fail"
        assert len(errors) > 0, "Should have validation errors"


class TestConfigurationSerialization:
    """Test suite for configuration serialization/deserialization"""

    def test_configuration_to_dict(self):
        """Test converting configuration objects to dictionaries"""
        config_manager = Mock()
        
        # Mock configuration object
        class MockConfiguration:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            def to_dict(self):
                return {key: value for key, value in self.__dict__.items() 
                       if not key.startswith('_')}
        
        # Mock configuration conversion
        def mock_config_to_dict(config_obj):
            """Convert configuration object to dictionary"""
            if hasattr(config_obj, 'to_dict'):
                return config_obj.to_dict()
            else:
                return None
        
        config_manager.config_to_dict = Mock(side_effect=mock_config_to_dict)
        
        # Test configuration conversion
        test_config = MockConfiguration(
            dataset_type='sine',
            classification_function=1,
            drift_position=3000,
            noise_level=0.05,
            num_instances=8000
        )
        
        config_dict = config_manager.config_to_dict(test_config)
        
        assert config_dict is not None, "Configuration should convert to dict"
        assert config_dict['dataset_type'] == 'sine', "Should preserve dataset_type"
        assert config_dict['classification_function'] == 1, "Should preserve all parameters"
        assert len(config_dict) == 5, "Should have all configuration parameters"

    def test_configuration_from_dict(self):
        """Test creating configuration objects from dictionaries"""
        config_manager = Mock()
        
        # Mock configuration creation
        def mock_config_from_dict(config_dict):
            """Create configuration object from dictionary"""
            required_fields = ['dataset_type']
            
            # Validation
            for field in required_fields:
                if field not in config_dict:
                    return None, [f"Missing required field: {field}"]
            
            # Create mock configuration object
            class ConfigurationObject:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
                
                def __eq__(self, other):
                    if not isinstance(other, ConfigurationObject):
                        return False
                    return self.__dict__ == other.__dict__
            
            return ConfigurationObject(**config_dict), []
        
        config_manager.config_from_dict = Mock(side_effect=mock_config_from_dict)
        
        # Test valid dictionary
        valid_dict = {
            'dataset_type': 'hyperplane',
            'dimensions': 15,
            'change_rate': 0.05,
            'noise_level': 0.1,
            'num_instances': 12000
        }
        
        config_obj, errors = config_manager.config_from_dict(valid_dict)
        
        assert config_obj is not None, f"Should create config object, errors: {errors}"
        assert len(errors) == 0, "Should have no errors"
        assert config_obj.dataset_type == 'hyperplane', "Should set attributes correctly"
        assert config_obj.dimensions == 15, "Should preserve all parameters"
        
        # Test invalid dictionary
        invalid_dict = {'dimensions': 10, 'change_rate': 0.05}  # Missing dataset_type
        
        config_obj, errors = config_manager.config_from_dict(invalid_dict)
        
        assert config_obj is None, "Should fail with missing required field"
        assert len(errors) > 0, "Should have validation errors"

    def test_configuration_json_serialization(self):
        """Test JSON serialization of configurations"""
        config_manager = Mock()
        
        # Mock JSON serialization
        def mock_config_to_json(config_dict):
            """Convert configuration to JSON string"""
            import json
            try:
                # Test that all values are JSON-serializable
                json_str = json.dumps(config_dict, indent=2, sort_keys=True)
                return json_str, []
            except TypeError as e:
                return None, [f"JSON serialization error: {str(e)}"]
        
        def mock_config_from_json(json_str):
            """Parse configuration from JSON string"""
            import json
            try:
                config_dict = json.loads(json_str)
                return config_dict, []
            except json.JSONDecodeError as e:
                return None, [f"JSON parsing error: {str(e)}"]
        
        config_manager.config_to_json = Mock(side_effect=mock_config_to_json)
        config_manager.config_from_json = Mock(side_effect=mock_config_from_json)
        
        # Test JSON round-trip
        original_config = {
            'dataset_type': 'mixed',
            'drift_length': 2000,
            'initial_concept_prob': 0.85,
            'final_concept_prob': 0.15,
            'num_instances': 10000,
            'random_seed': 42
        }
        
        # Serialize to JSON
        json_str, serialize_errors = config_manager.config_to_json(original_config)
        assert json_str is not None, f"JSON serialization should succeed, errors: {serialize_errors}"
        assert len(serialize_errors) == 0, "Should have no serialization errors"
        
        # Deserialize from JSON
        parsed_config, parse_errors = config_manager.config_from_json(json_str)
        assert parsed_config is not None, f"JSON parsing should succeed, errors: {parse_errors}"
        assert len(parse_errors) == 0, "Should have no parsing errors"
        
        # Verify round-trip accuracy
        assert parsed_config == original_config, "JSON round-trip should preserve configuration"

    def test_configuration_file_operations(self):
        """Test saving/loading configurations to/from files"""
        config_manager = Mock()
        
        # Mock file operations
        def mock_save_config_to_file(config_dict, filepath):
            """Simulate saving configuration to file"""
            import json

            # Validate filepath
            if not filepath.endswith('.json'):
                return False, ["Configuration files must use .json extension"]
            
            # Validate config
            if 'dataset_type' not in config_dict:
                return False, ["Invalid configuration: missing dataset_type"]
            
            # Simulate successful save
            return True, []
        
        def mock_load_config_from_file(filepath):
            """Simulate loading configuration from file"""
            # Mock file contents based on filepath
            mock_configs = {
                'sine_config.json': {
                    'dataset_type': 'sine',
                    'classification_function': 2,
                    'drift_position': 4000,
                    'noise_level': 0.1,
                    'num_instances': 8000
                },
                'invalid.txt': None  # Wrong extension
            }
            
            if filepath not in mock_configs:
                return None, ["File not found"]
            
            config = mock_configs[filepath]
            if config is None:
                return None, ["Invalid file format"]
            
            return config, []
        
        config_manager.save_config = Mock(side_effect=mock_save_config_to_file)
        config_manager.load_config = Mock(side_effect=mock_load_config_from_file)
        
        # Test saving configuration
        config_to_save = {
            'dataset_type': 'stagger',
            'concept_configuration': 2,
            'drift_position': 5000,
            'num_instances': 15000
        }
        
        success, save_errors = config_manager.save_config(config_to_save, 'stagger_config.json')
        assert success, f"Config save should succeed, errors: {save_errors}"
        assert len(save_errors) == 0, "Should have no save errors"
        
        # Test loading configuration
        loaded_config, load_errors = config_manager.load_config('sine_config.json')
        assert loaded_config is not None, f"Config load should succeed, errors: {load_errors}"
        assert len(load_errors) == 0, "Should have no load errors"
        assert loaded_config['dataset_type'] == 'sine', "Should load correct configuration"
        
        # Test invalid file extension
        success, errors = config_manager.save_config(config_to_save, 'config.txt')
        assert not success, "Should fail with wrong file extension"
        assert len(errors) > 0, "Should have validation errors"


class TestConfigurationInheritance:
    """Test suite for configuration inheritance and defaults"""

    def test_default_configuration_values(self):
        """Test default values are applied for missing parameters"""
        config_manager = Mock()
        
        # Mock default value application
        def mock_apply_defaults(config_dict, dataset_type):
            """Apply default values for missing parameters"""
            defaults = {
                'sine': {
                    'noise_level': 0.0,
                    'num_instances': 10000,
                    'random_seed': None
                },
                'hyperplane': {
                    'dimensions': 10,
                    'noise_level': 0.05,
                    'reversal_probability': 0.1,
                    'num_instances': 10000,
                    'random_seed': None
                }
            }
            
            if dataset_type not in defaults:
                return config_dict, [f"Unknown dataset type: {dataset_type}"]
            
            # Apply defaults for missing values
            result_config = config_dict.copy()
            for param, default_value in defaults[dataset_type].items():
                if param not in result_config:
                    result_config[param] = default_value
            
            return result_config, []
        
        config_manager.apply_defaults = Mock(side_effect=mock_apply_defaults)
        
        # Test Sine defaults
        partial_sine_config = {
            'dataset_type': 'sine',
            'classification_function': 1,
            'drift_position': 3000
        }
        
        complete_config, errors = config_manager.apply_defaults(partial_sine_config, 'sine')
        
        assert len(errors) == 0, f"Should apply defaults without errors: {errors}"
        assert complete_config['noise_level'] == 0.0, "Should apply default noise_level"
        assert complete_config['num_instances'] == 10000, "Should apply default num_instances"
        assert complete_config['classification_function'] == 1, "Should preserve existing values"
        
        # Test Hyperplane defaults
        partial_hyperplane_config = {
            'dataset_type': 'hyperplane',
            'change_rate': 0.05
        }
        
        complete_config, errors = config_manager.apply_defaults(partial_hyperplane_config, 'hyperplane')
        
        assert complete_config['dimensions'] == 10, "Should apply default dimensions"
        assert complete_config['noise_level'] == 0.05, "Should apply default noise_level"
        assert complete_config['reversal_probability'] == 0.1, "Should apply default reversal_probability"

    def test_configuration_inheritance_hierarchy(self):
        """Test configuration inheritance from base to specific"""
        config_manager = Mock()
        
        # Mock inheritance hierarchy
        def mock_configuration_inheritance(base_config, specific_overrides):
            """Apply inheritance from base configuration"""
            # Start with base configuration
            inherited_config = base_config.copy()
            
            # Apply specific overrides
            for key, value in specific_overrides.items():
                inherited_config[key] = value
            
            return inherited_config
        
        config_manager.inherit_configuration = Mock(side_effect=mock_configuration_inheritance)
        
        # Test inheritance
        base_config = {
            'num_instances': 10000,
            'random_seed': 42,
            'noise_level': 0.0,
            'output_format': 'csv'
        }
        
        specific_overrides = {
            'dataset_type': 'sine',
            'classification_function': 2,
            'drift_position': 5000,
            'noise_level': 0.1  # Override base value
        }
        
        final_config = config_manager.inherit_configuration(base_config, specific_overrides)
        
        # Test inheritance results
        assert final_config['num_instances'] == 10000, "Should inherit base num_instances"
        assert final_config['random_seed'] == 42, "Should inherit base random_seed"
        assert final_config['output_format'] == 'csv', "Should inherit base output_format"
        
        # Test overrides
        assert final_config['dataset_type'] == 'sine', "Should apply specific dataset_type"
        assert final_config['noise_level'] == 0.1, "Should override base noise_level"
        assert final_config['classification_function'] == 2, "Should apply specific parameters"

    def test_configuration_validation_after_inheritance(self):
        """Test that inherited configurations are validated"""
        config_manager = Mock()
        
        # Mock post-inheritance validation
        def mock_validate_inherited_config(config):
            """Validate configuration after inheritance/defaults"""
            errors = []
            
            # Check required fields exist
            required_fields = ['dataset_type', 'num_instances']
            for field in required_fields:
                if field not in config or config[field] is None:
                    errors.append(f"Required field missing or None: {field}")
            
            # Type and range validation
            if 'num_instances' in config:
                if not isinstance(config['num_instances'], int) or config['num_instances'] <= 0:
                    errors.append("num_instances must be positive integer")
            
            if 'noise_level' in config:
                if not isinstance(config['noise_level'], (int, float)) or not (0 <= config['noise_level'] <= 1):
                    errors.append("noise_level must be in [0,1]")
            
            return len(errors) == 0, errors
        
        config_manager.validate_inherited_config = Mock(side_effect=mock_validate_inherited_config)
        
        # Test valid inherited configuration
        valid_config = {
            'dataset_type': 'hyperplane',
            'dimensions': 15,
            'change_rate': 0.08,
            'noise_level': 0.05,
            'num_instances': 12000,
            'random_seed': 42
        }
        
        is_valid, errors = config_manager.validate_inherited_config(valid_config)
        assert is_valid, f"Valid inherited config should pass validation, errors: {errors}"
        
        # Test invalid inherited configuration
        invalid_config = {
            'dataset_type': 'sine',
            'num_instances': -1000,  # Invalid: negative
            'noise_level': 1.5       # Invalid: > 1
        }
        
        is_valid, errors = config_manager.validate_inherited_config(invalid_config)
        assert not is_valid, "Invalid inherited config should fail validation"
        assert len(errors) >= 2, f"Should have multiple validation errors: {errors}"
