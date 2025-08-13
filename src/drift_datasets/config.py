"""
Configuration management for drift_datasets library.

This module handles TOML configuration file parsing, validation, and management
for reproducible dataset generation.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import toml


class ConfigurationParser:
    """Simple TOML configuration parser for compatibility."""

    def __init__(self):
        self._manager = ConfigurationManager()

    def parse_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse TOML configuration file.

        Args:
            config_path: Path to TOML configuration file

        Returns:
            Parsed configuration dictionary
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        return toml.load(config_path)

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_list)

        Raises:
            ValueError: If configuration is invalid
        """
        validation_result = self._manager.validate_schema(config)
        is_valid = validation_result["is_valid"]
        errors = validation_result["errors"]

        if not is_valid and errors:
            # Raise ValueError with first error message for TDD compatibility
            first_error = errors[0] if isinstance(errors[0], str) else str(errors[0])
            raise ValueError(first_error)

        return is_valid, errors


class ConfigurationManager:
    """Manages TOML configuration loading and validation."""

    def __init__(self):
        self.schema_validators = {
            "synthetic": self._validate_synthetic_config,
            "real_world": self._validate_real_world_config,
            "mixed": self._validate_mixed_config,
        }

    def load_config(self, config_path: Union[str, Path], substitute_env: bool = False, validate: bool = True) -> Dict[str, Any]:
        """
        Load TOML configuration file with optional environment variable substitution.

        Args:
            config_path: Path to TOML configuration file
            substitute_env: Whether to substitute environment variables
            validate: Whether to validate the final configuration

        Returns:
            Parsed configuration dictionary
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load base configuration
        config = toml.load(config_path)

        # Handle inheritance if 'extends' is present
        if "extends" in config:
            base_config_path = Path(config["extends"])
            if not base_config_path.is_absolute():
                base_config_path = config_path.parent / base_config_path

            base_config = self.load_config(base_config_path, substitute_env, validate=False)

            # Merge configurations (child overrides parent)
            config = self._merge_configs(base_config, config)

            # Remove 'extends' key from final config
            config.pop("extends", None)

        # Substitute environment variables if requested
        if substitute_env:
            config = self._substitute_environment_variables(config)

        # Validate the final configuration only if requested
        if validate:
            validation_result = self.validate_schema(config)
            if not validation_result["is_valid"]:
                raise ValueError(f"Configuration validation failed: {validation_result['errors']}")

        return config

    def validate_schema(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against schema requirements.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Validation result with is_valid flag and error list
        """
        errors = []
        warnings = []
        validated_fields = {"required": [], "optional": []}

        # Check required top-level sections
        required_sections = ["dataset"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required field '{section}'")
            else:
                validated_fields["required"].append(section)

        if "dataset" in config:
            dataset_config = config["dataset"]

            # Check required dataset fields
            required_dataset_fields = ["name", "type"]
            for field in required_dataset_fields:
                if field not in dataset_config:
                    errors.append(f"Missing required field 'dataset.{field}'")
                else:
                    validated_fields["required"].append(f"dataset.{field}")

            # Validate dataset type and corresponding configuration
            if "type" in dataset_config:
                dataset_type = dataset_config["type"]
                if dataset_type in self.schema_validators:
                    type_errors, type_warnings = self.schema_validators[dataset_type](config, validated_fields)
                    errors.extend(type_errors)
                    warnings.extend(type_warnings)
                else:
                    errors.append(f"Unsupported dataset type: {dataset_type}")

        return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings, "validated_fields": validated_fields}

    def _validate_synthetic_config(self, config: Dict[str, Any], validated_fields: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """Validate synthetic dataset configuration."""
        errors = []
        warnings = []

        dataset_config = config.get("dataset", {})

        # Check for generator specification
        if "generator" not in dataset_config:
            errors.append("Synthetic datasets require 'generator' specification")
        else:
            validated_fields["required"].append("dataset.generator")

        # Check for generator_config section
        if "generator_config" not in config:
            errors.append("Synthetic datasets require 'generator_config' section")
        else:
            generator_config = config["generator_config"]

            # Check for required generator parameters
            if "n_instances" not in generator_config:
                errors.append("Generator config missing 'n_instances'")

            if "random_seed" not in generator_config:
                warnings.append("Generator config missing 'random_seed' - results may not be reproducible")

        return errors, warnings

    def _validate_real_world_config(self, config: Dict[str, Any], validated_fields: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """Validate real-world dataset configuration."""
        errors = []
        warnings = []

        # Check for UCI configuration
        if "uci_config" not in config:
            errors.append("Real-world datasets require 'uci_config' section")
        else:
            uci_config = config["uci_config"]

            if "dataset_id" not in uci_config:
                errors.append("UCI config missing 'dataset_id'")

        return errors, warnings

    def _validate_mixed_config(self, config: Dict[str, Any], validated_fields: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """Validate mixed dataset configuration."""
        errors = []
        warnings = []

        # Check for mixed_config section
        if "mixed_config" not in config:
            errors.append("Mixed datasets require 'mixed_config' section")
        else:
            mixed_config = config["mixed_config"]

            if "components" not in mixed_config:
                errors.append("Mixed config missing 'components' list")
            elif len(mixed_config["components"]) < 2:
                errors.append("Mixed datasets require at least 2 components")

        return errors, warnings

    def _merge_configs(self, base: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()

        for key, value in child.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _substitute_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values."""

        def substitute_value(value):
            if isinstance(value, str):
                # Look for ${VAR_NAME} patterns
                pattern = r"\$\{([^}]+)\}"
                matches = re.findall(pattern, value)

                for var_name in matches:
                    env_value = os.getenv(var_name)
                    if env_value is not None:
                        value = value.replace(f"${{{var_name}}}", env_value)

                # If the whole value was a single variable, try to convert type
                if value.isdigit():
                    return int(value)
                else:
                    try:
                        return float(value)
                    except ValueError:
                        return value

            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value

        return substitute_value(config)


class ParameterValidator:
    """Validates parameters for different dataset generators and configurations."""

    def __init__(self):
        self.generator_schemas = {
            "SineGenerator": {
                "n_instances": {"type": int, "min": 100, "max": 1000000, "required": True},
                "random_seed": {"type": int, "min": 0, "max": 2**31 - 1, "required": False},
                "noise_level": {"type": float, "min": 0.0, "max": 1.0, "required": False},
            },
            "HyperplaneGenerator": {
                "n_instances": {"type": int, "min": 100, "max": 1000000, "required": True},
                "random_seed": {"type": int, "min": 0, "max": 2**31 - 1, "required": False},
                "n_features": {"type": int, "min": 2, "max": 100, "required": True},
            },
            "STAGGERGenerator": {
                "n_instances": {"type": int, "min": 100, "max": 1000000, "required": True},
                "random_seed": {"type": int, "min": 0, "max": 2**31 - 1, "required": False},
            },
        }

    def validate_generator_parameters(self, generator_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters for a specific generator.

        Args:
            generator_name: Name of the generator
            parameters: Parameters to validate

        Returns:
            Validation result dictionary
        """
        errors = []
        validated_parameters = {}
        suggestions = []

        if generator_name not in self.generator_schemas:
            errors.append(f"Unknown generator: {generator_name}")
            return {"is_valid": False, "errors": errors, "validated_parameters": {}, "suggestions": []}

        schema = self.generator_schemas[generator_name]

        for param_name, param_config in schema.items():
            if param_name in parameters:
                value = parameters[param_name]
                param_validation = self._validate_parameter(param_name, value, param_config)
                validated_parameters[param_name] = param_validation

                if not param_validation["is_valid"]:
                    errors.extend(param_validation["errors"])
                    for error in param_validation["errors"]:
                        if "suggestion" in error:
                            suggestions.append({"parameter": param_name, "suggestion": error["suggestion"]})
            else:
                # Parameter not provided - check if it's required
                is_required = param_config.get("required", True)
                if is_required:
                    validated_parameters[param_name] = {"is_valid": False, "errors": ["Parameter not provided"]}
                    errors.append({"message": f"Parameter '{param_name}' is required for {generator_name}"})
                    suggestions.append(
                        {"parameter": param_name, "suggestion": f"Parameter '{param_name}' is required for {generator_name}"}
                    )
                else:
                    validated_parameters[param_name] = {"is_valid": True, "errors": []}

        return {"is_valid": len(errors) == 0, "errors": errors, "validated_parameters": validated_parameters, "suggestions": suggestions}

    def validate_drift_parameters(self, drift_config: Dict[str, Any], generator_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate drift configuration parameters."""
        errors = []
        drift_validations = {}

        n_instances = generator_config.get("n_instances", 1000)

        # Validate drift points
        drift_points = drift_config.get("drift_points", [])
        drift_validations["drift_points_within_bounds"] = all(0 < point < n_instances for point in drift_points)

        if not drift_validations["drift_points_within_bounds"]:
            errors.append("Drift points must be within dataset bounds (0 < point < n_instances)")

        # Validate drift intensities
        drift_intensities = drift_config.get("drift_intensities", [])
        drift_validations["drift_intensities_valid"] = all(0.0 <= intensity <= 1.0 for intensity in drift_intensities)

        if not drift_validations["drift_intensities_valid"]:
            errors.append("Drift intensities must be in range [0.0, 1.0]")

        # Validate drift types
        supported_drift_types = ["concept", "feature", "prior"]
        drift_types = drift_config.get("drift_types", [])
        drift_validations["drift_types_supported"] = all(dtype in supported_drift_types for dtype in drift_types)

        if not drift_validations["drift_types_supported"]:
            errors.append(f"Unsupported drift types. Supported types: {supported_drift_types}")

        return {"is_valid": len(errors) == 0, "errors": errors, "drift_validations": drift_validations}

    def validate_uci_parameters(self, uci_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate UCI dataset parameters."""
        errors = []
        uci_validations = {}

        # Validate dataset ID
        dataset_id = uci_config.get("dataset_id")
        if dataset_id is not None:
            uci_validations["dataset_id_valid"] = isinstance(dataset_id, int) and dataset_id > 0
            if not uci_validations["dataset_id_valid"]:
                errors.append("UCI dataset_id must be a positive integer")
        else:
            errors.append("UCI config missing 'dataset_id'")
            uci_validations["dataset_id_valid"] = False

        # Validate preprocessing options
        preprocessing_options = uci_config.get("preprocessing", [])
        supported_options = ["normalize", "temporal_order", "normalization", "missing_values", "categorical_encoding"]

        uci_validations["preprocessing_options_valid"] = True
        for option in preprocessing_options:
            if option not in supported_options:
                uci_validations["preprocessing_options_valid"] = False
                errors.append(f"Unsupported preprocessing option: {option}")

        return {"is_valid": len(errors) == 0, "errors": errors, "uci_validations": uci_validations}

    def _validate_parameter(self, name: str, value: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single parameter against its configuration."""
        errors = []

        # Check type
        expected_type = config.get("type")
        if expected_type and not isinstance(value, expected_type):
            errors.append(
                {
                    "message": f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
                    "suggestion": f"Convert '{name}' to {expected_type.__name__}",
                }
            )

        # Check range for numeric types
        if isinstance(value, (int, float)):
            if "min" in config and value < config["min"]:
                message = f"Parameter '{name}' value {value} is below minimum {config['min']}"
                if value < 0:
                    message = f"Parameter '{name}' has negative value {value}, minimum is {config['min']}"
                errors.append(
                    {
                        "message": message,
                        "suggestion": f"Set '{name}' to at least {config['min']} (minimum recommended: {config['min']})",
                    }
                )

            if "max" in config and value > config["max"]:
                errors.append(
                    {
                        "message": f"Parameter '{name}' value {value} exceeds maximum {config['max']} (out of valid range)",
                        "suggestion": f"Set '{name}' to at most {config['max']} (valid range: [{config.get('min', 'no limit')}, {config['max']}])",
                    }
                )

        return {"is_valid": len(errors) == 0, "errors": errors}
