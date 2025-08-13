"""
Core factory functions for drift_datasets library.

This module provides the main entry points for creating datasets
from TOML configurations.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from rich.console import Console

from .config import ConfigurationManager
from .data_sources import UCIRepository
from .generators import CapyMOAInterface
from .models import DriftDataset

console = Console()


def create_dataset(config_path: Union[str, Path, Dict[str, Any]], **kwargs) -> DriftDataset:
    """
    Create a DriftDataset from a TOML configuration file.

    This is the main factory function that handles synthetic, real-world,
    and mixed dataset creation based on configuration specifications.

    Args:
        config_path: Path to TOML configuration file or dictionary configuration
        **kwargs: Additional options (streaming, cache, etc.)

    Returns:
        DriftDataset instance with data and metadata

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If configuration file doesn't exist
    """
    # Load and validate configuration
    config_manager = ConfigurationManager()

    if isinstance(config_path, dict):
        # Direct dictionary configuration
        config = config_path
        # Validate the dictionary configuration
        validation_result = config_manager.validate_schema(config)
        if not validation_result["is_valid"]:
            raise ValueError(f"Configuration validation failed: {'; '.join(validation_result['errors'])}")
    else:
        # File path configuration
        config = config_manager.load_config(config_path)

    # Extract main configuration sections
    dataset_config = config["dataset"]
    dataset_type = dataset_config["type"]
    dataset_name = dataset_config["name"]

    # Route to appropriate creation function based on type
    if dataset_type == "synthetic":
        return _create_synthetic_dataset(dataset_name, config)
    elif dataset_type == "real_world":
        return _create_real_world_dataset(dataset_name, config)
    elif dataset_type == "mixed":
        return _create_mixed_dataset(dataset_name, config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _create_synthetic_dataset(name: str, config: Dict[str, Any]) -> DriftDataset:
    """Create a synthetic dataset using CapyMOA generators."""
    dataset_config = config["dataset"]
    generator_config = config.get("generator_config", {})
    drift_config = config.get("drift_config", {})
    metadata_config = config.get("metadata", {})

    # Get generator name
    generator_name = dataset_config["generator"]

    # Create CapyMOA interface and generator
    capymoa_interface = CapyMOAInterface()
    generator = capymoa_interface.create_generator(generator_name, generator_config)

    # Generate data with drift points
    drift_points = drift_config.get("drift_points", [])
    X, y = generator.generate(drift_points)

    # Build drift metadata
    drift_metadata = {
        "drift_points": drift_points,
        "drift_types": drift_config.get("drift_types", ["concept"] * len(drift_points)),
        "drift_patterns": drift_config.get("drift_patterns", ["abrupt"] * len(drift_points)),
        "drift_intensities": drift_config.get("drift_intensities", [1.0] * len(drift_points)),
    }

    # Build dataset metadata
    dataset_metadata = {
        "generator": {"name": generator_name, **generator_config},
        "features": _build_feature_metadata(X, y),
        "dimension": "multivariate" if len(X.columns) > 1 else "univariate",
        "labeling": "supervised",
        "n_classes": len(y.unique()),
    }

    # Add any additional metadata
    dataset_metadata.update(metadata_config)

    return DriftDataset(X=X, y=y, name=name, source_type="synthetic", drift_metadata=drift_metadata, dataset_metadata=dataset_metadata)


def _create_real_world_dataset(name: str, config: Dict[str, Any]) -> DriftDataset:
    """Create a real-world dataset from UCI repository."""
    uci_config = config.get("uci_config", {})
    drift_config = config.get("drift_config", {})
    metadata_config = config.get("metadata", {})

    # Get dataset ID from configuration
    dataset_id = uci_config.get("dataset_id")
    if dataset_id is None:
        raise ValueError("dataset_id is required in uci_config for real-world datasets")

    # Initialize UCI repository and load dataset
    uci_repository = UCIRepository()

    try:
        console.print(f"[blue]Loading UCI dataset {dataset_id}...[/blue]")

        # Check if dataset is available
        if not uci_repository.is_dataset_available(dataset_id):
            raise ValueError(f"UCI dataset {dataset_id} is not available")

        # Fetch the dataset
        data = uci_repository.fetch_dataset(dataset_id)
        info = uci_repository.get_dataset_info(dataset_id)

        # Separate features and target
        X = data.drop(columns=["target"])
        y = data["target"]

        console.print(f"[green]✓[/green] Successfully loaded UCI dataset: {info['name']}")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load UCI dataset {dataset_id}: {e}")
        raise ConnectionError(f"Unable to connect to UCI repository for dataset {dataset_id}: {e}")

    # Build drift metadata (real-world datasets typically don't have ground truth drift)
    drift_metadata = {
        "drift_points": drift_config.get("drift_points", []),
        "drift_types": drift_config.get("drift_types", []),
        "drift_patterns": drift_config.get("drift_patterns", []),
        "drift_intensities": drift_config.get("drift_intensities", []),
    }

    # Build dataset metadata
    dataset_metadata = {
        "source": "uci",
        "uci_id": dataset_id,
        "uci_metadata": {
            "dataset_name": info["name"],
            "repository_url": info.get("url", f"https://archive.ics.uci.edu/dataset/{dataset_id}"),
            "task": info["task"],
            "original_n_instances": info.get("n_instances", len(X)),
            "original_n_features": info.get("n_features", len(X.columns)),
        },
        "features": _build_feature_metadata(X, y),
        "dimension": "multivariate" if len(X.columns) > 1 else "univariate",
        "labeling": "supervised" if info["task"] in ["classification", "regression"] else "unsupervised",
        "n_classes": len(y.unique()) if info["task"] == "classification" else None,
    }

    # Add any additional metadata from configuration
    dataset_metadata.update(metadata_config)

    return DriftDataset(X=X, y=y, name=name, source_type="real_world", drift_metadata=drift_metadata, dataset_metadata=dataset_metadata)


def _create_mixed_dataset(name: str, config: Dict[str, Any]) -> DriftDataset:
    """Create a mixed dataset combining synthetic and real-world data."""
    mixed_config = config.get("mixed_config", {})
    components = mixed_config.get("components", [])
    metadata_config = config.get("metadata", {})

    if len(components) < 2:
        raise ValueError("Mixed datasets require at least 2 components")

    # Generate components
    component_datasets = []
    total_instances = 0

    for i, component in enumerate(components):
        component_config = {
            "dataset": {"name": f"{name}_component_{i}", "type": component["source"], "generator": component.get("generator")}
        }

        if component["source"] == "capymoa":
            component_config["generator_config"] = {
                "n_instances": component.get("n_instances", 500),
                "random_seed": component.get("random_seed", 42 + i),
            }
            component_dataset = _create_synthetic_dataset(f"{name}_component_{i}", component_config)
        else:
            # Real-world dataset component (UCI)
            component_config["uci_config"] = component.get("uci_config", {"dataset_id": 321})
            component_dataset = _create_real_world_dataset(f"{name}_component_{i}", component_config)

            # Truncate to desired size if needed
            desired_instances = component.get("n_instances", 500)
            if len(component_dataset.X) > desired_instances:
                # Create new truncated dataset instead of modifying existing
                truncated_X = component_dataset.X.iloc[:desired_instances]
                truncated_y = component_dataset.y.iloc[:desired_instances]

                component_dataset = DriftDataset(
                    X=truncated_X,
                    y=truncated_y,
                    name=component_dataset.name,
                    source_type=component_dataset.source_type,
                    drift_metadata=component_dataset.drift_metadata,
                    dataset_metadata=component_dataset.dataset_metadata,
                )

        component_datasets.append(component_dataset)
        total_instances += len(component_dataset.X)

    # Combine datasets (sequential strategy for now)
    X_combined = pd.concat([ds.X for ds in component_datasets], ignore_index=True)
    y_combined = pd.concat([ds.y for ds in component_datasets], ignore_index=True)

    # Build mixed dataset metadata
    mixing_info = {
        "strategy": mixed_config.get("strategy", "sequential"),
        "transition_point": len(component_datasets[0].X) if len(component_datasets) >= 2 else 0,
        "components": [{"source_type": ds.source_type, "instances": len(ds.X)} for ds in component_datasets],
    }

    # Build drift metadata (if any components have drift)
    drift_metadata = {}
    all_drift_points = []
    offset = 0

    for ds in component_datasets:
        if ds.drift_metadata.get("drift_points"):
            # Adjust drift points for combined dataset
            adjusted_points = [point + offset for point in ds.drift_metadata["drift_points"]]
            all_drift_points.extend(adjusted_points)
        offset += len(ds.X)

    if all_drift_points:
        drift_metadata["drift_points"] = all_drift_points
        drift_metadata["drift_types"] = ["concept"] * len(all_drift_points)
        drift_metadata["drift_patterns"] = ["abrupt"] * len(all_drift_points)
        drift_metadata["drift_intensities"] = [1.0] * len(all_drift_points)

    # Build dataset metadata
    dataset_metadata = {
        "mixing": mixing_info,
        "component_metadata": [ds.dataset_metadata for ds in component_datasets],
        "features": _build_feature_metadata(X_combined, y_combined),
        "dimension": "multivariate" if len(X_combined.columns) > 1 else "univariate",
        "labeling": "supervised",
        "n_classes": len(y_combined.unique()),
    }

    dataset_metadata.update(metadata_config)

    return DriftDataset(
        X=X_combined, y=y_combined, name=name, source_type="mixed", drift_metadata=drift_metadata, dataset_metadata=dataset_metadata
    )


def _build_feature_metadata(X: pd.DataFrame, y: pd.Series) -> list:
    """Build feature metadata from DataFrame."""
    features = []

    # Add feature columns
    for col in X.columns:
        feature_info = {"name": col, "role": "feature"}

        if pd.api.types.is_numeric_dtype(X[col]):
            feature_info["type"] = "continuous"
        elif pd.api.types.is_categorical_dtype(X[col]):
            feature_info["type"] = "categorical"
        else:
            feature_info["type"] = "categorical"  # Default for object types

        features.append(feature_info)

    # Add target
    target_info = {"name": y.name or "target", "role": "target"}

    if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
        target_info["type"] = "continuous"
    else:
        target_info["type"] = "categorical"

    features.append(target_info)

    return features
