"""
Core dataset models for drift_datasets library.

This module provides the main DriftDataset class that represents datasets
with comprehensive drift metadata and ground truth information.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeatureMetadata(BaseModel):
    """Feature-level metadata."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="allow")

    feature_names: List[str] = Field(default_factory=list, description="Names of features")
    feature_types: List[str] = Field(default_factory=list, description="Data types of features")

    def get(self, key: str, default=None):
        """Dictionary-style get method for backward compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """Dictionary-style access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Dictionary-style 'in' operator for backward compatibility."""
        return hasattr(self, key)


class DatasetMetadata(BaseModel):
    """Core dataset metadata information."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="allow")

    name: str = Field(min_length=1, description="Dataset name")
    source_type: str = Field(pattern=r"^(synthetic|real_world|mixed|unknown)$", description="Origin of the dataset")
    n_samples: int = Field(ge=1, description="Number of samples")
    n_features: int = Field(ge=1, description="Number of features")
    n_classes: Optional[int] = Field(default=None, ge=2, description="Number of classes (None for regression)")
    dimension: Optional[str] = Field(default=None, description="Dataset dimensionality")
    labeling: Optional[str] = Field(default=None, description="Labeling type")
    features: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Feature definitions")

    def get(self, key: str, default=None):
        """Dictionary-style get method for backward compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """Dictionary-style access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Dictionary-style 'in' operator for backward compatibility."""
        return hasattr(self, key)


class DriftMetadata(BaseModel):
    """Ground truth drift metadata."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="allow")

    drift_points: List[int] = Field(default_factory=list, description="Sample indices where drift occurs")
    drift_types: List[str] = Field(default_factory=list, description="Types of drift at each point")
    drift_patterns: List[str] = Field(default_factory=list, description="Patterns of drift at each point")
    drift_intensities: List[float] = Field(default_factory=list, description="Intensities of drift at each point")
    affected_features: Optional[List[List[int]]] = Field(default_factory=list, description="Features affected by each drift point")
    transition_durations: Optional[List[int]] = Field(default_factory=list, description="Duration of drift transitions")

    @model_validator(mode="after")
    def validate_drift_consistency(self) -> "DriftMetadata":
        """Ensure all drift lists have consistent lengths and valid values."""
        n_points = len(self.drift_points)

        if self.drift_types and len(self.drift_types) != n_points:
            raise ValueError(f"drift_types length ({len(self.drift_types)}) must match drift_points length ({n_points})")

        if self.drift_patterns and len(self.drift_patterns) != n_points:
            raise ValueError(f"drift_patterns length ({len(self.drift_patterns)}) must match drift_points length ({n_points})")

        if self.drift_intensities and len(self.drift_intensities) != n_points:
            raise ValueError(f"drift_intensities length ({len(self.drift_intensities)}) must match drift_points length ({n_points})")

        # Validate drift_intensities range
        for i, intensity in enumerate(self.drift_intensities):
            if not (0.0 <= intensity <= 1.0):
                raise ValueError(f"drift_intensities[{i}] = {intensity} must be in range [0.0, 1.0]")

        if self.affected_features and len(self.affected_features) != n_points:
            raise ValueError(f"affected_features length ({len(self.affected_features)}) must match drift_points length ({n_points})")

        if self.transition_durations and len(self.transition_durations) != n_points:
            raise ValueError(f"transition_durations length ({len(self.transition_durations)}) must match drift_points length ({n_points})")

        return self

    def get(self, key: str, default=None):
        """Dictionary-style get method for backward compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """Dictionary-style access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Dictionary-style 'in' operator for backward compatibility."""
        return hasattr(self, key)


class DriftDataset(BaseModel):
    """
    Core dataset class that represents datasets with drift metadata.

    This class provides structured access to features, targets, and comprehensive
    metadata about concept drift patterns and dataset characteristics.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, arbitrary_types_allowed=True  # Allow pandas DataFrame/Series
    )

    X: pd.DataFrame = Field(..., description="Feature matrix as pandas DataFrame")
    y: pd.Series = Field(..., description="Target vector as pandas Series")
    name: str = Field(min_length=1, max_length=100, description="Dataset name")
    source_type: str = Field(pattern=r"^(synthetic|real_world|mixed|unknown)$", description="Type of dataset")
    drift_metadata: Union[DriftMetadata, Dict[str, Any]] = Field(
        default_factory=dict, description="Metadata about drift patterns and ground truth"
    )
    dataset_metadata: Union[DatasetMetadata, Dict[str, Any]] = Field(
        default_factory=dict, description="Metadata about dataset characteristics"
    )

    @model_validator(mode="before")
    def convert_metadata_dicts(cls, values):
        """Convert dictionary metadata to Pydantic models."""
        if isinstance(values, dict):
            # Convert drift_metadata if it's a dictionary
            if "drift_metadata" in values and isinstance(values["drift_metadata"], dict):
                values["drift_metadata"] = DriftMetadata(**values["drift_metadata"])

            # Convert dataset_metadata if it's a dictionary
            if "dataset_metadata" in values and isinstance(values["dataset_metadata"], dict):
                # Extract core fields for DatasetMetadata
                dataset_metadata_dict = values["dataset_metadata"].copy()

                # Get required fields from the dataset itself if not provided
                if "X" in values and "y" in values:
                    dataset_metadata_dict.setdefault("name", values.get("name", "dataset"))
                    dataset_metadata_dict.setdefault("source_type", values.get("source_type", "unknown"))
                    dataset_metadata_dict.setdefault("n_samples", len(values["X"]))
                    dataset_metadata_dict.setdefault("n_features", len(values["X"].columns))
                    dataset_metadata_dict.setdefault("n_classes", len(values["y"].unique()))

                values["dataset_metadata"] = DatasetMetadata(**dataset_metadata_dict)

        return values

    @model_validator(mode="after")
    def validate_data_consistency(self) -> "DriftDataset":
        """Validate that X and y have consistent dimensions and drift metadata is valid."""
        # Check X and y dimensions match
        if len(self.X) != len(self.y):
            raise ValueError(f"Feature matrix and target vector have different lengths: X={len(self.X)}, y={len(self.y)}")

        # Handle both DriftMetadata objects and dictionaries
        if isinstance(self.drift_metadata, DriftMetadata):
            drift_points = self.drift_metadata.drift_points
            drift_types = self.drift_metadata.drift_types
            drift_patterns = self.drift_metadata.drift_patterns
        else:
            drift_points = self.drift_metadata.get("drift_points", [])
            drift_types = self.drift_metadata.get("drift_types", [])
            drift_patterns = self.drift_metadata.get("drift_patterns", [])

        # Validate drift points are within bounds
        for point in drift_points:
            if not isinstance(point, int) or point < 0 or point >= len(self.X):
                raise ValueError(f"Drift point {point} is not a valid index for dataset of size {len(self.X)}")

        # Validate drift metadata consistency (only check if there are drift points and types/patterns)
        if drift_types and len(drift_types) != len(drift_points):
            raise ValueError(f"drift_types length ({len(drift_types)}) must match drift_points length ({len(drift_points)})")

        if drift_patterns and len(drift_patterns) != len(drift_points):
            raise ValueError(f"drift_patterns length ({len(drift_patterns)}) must match drift_points length ({len(drift_points)})")

        return self

    def info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.

        Returns:
            Dictionary with dataset statistics and information
        """
        # Basic statistics
        info_dict = {
            "n_instances": len(self.X),
            "n_features": len(self.X.columns),
            "n_classes": len(self.y.unique()),
            "class_distribution": self.y.value_counts().to_dict(),
        }

        # Drift analysis
        drift_analysis = {
            "n_drift_points": len(self.drift_metadata.get("drift_points", [])),
            "drift_intervals": self._calculate_drift_intervals(),
        }
        info_dict["drift_analysis"] = drift_analysis

        return info_dict

    def describe(self) -> Dict[str, Any]:
        """
        Get detailed statistical description of the dataset.

        Returns:
            Dictionary with feature and target statistics
        """
        description = {"features": {}, "target": {}}

        # Feature statistics
        for column in self.X.columns:
            if pd.api.types.is_numeric_dtype(self.X[column]):
                description["features"][column] = {
                    "mean": float(self.X[column].mean()),
                    "std": float(self.X[column].std()),
                    "min": float(self.X[column].min()),
                    "max": float(self.X[column].max()),
                    "median": float(self.X[column].median()),
                }
            else:
                description["features"][column] = {
                    "unique_values": int(self.X[column].nunique()),
                    "most_common": str(self.X[column].mode().iloc[0]) if not self.X[column].mode().empty else None,
                }

        # Target statistics
        description["target"] = {
            "unique_values": int(self.y.nunique()),
            "distribution": self.y.value_counts().to_dict(),
            "type": "categorical" if pd.api.types.is_categorical_dtype(self.y) else "continuous",
        }

        return description

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset consistency and integrity.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check basic data consistency
        if len(self.X) != len(self.y):
            errors.append("Feature matrix and target vector have different lengths")

        # Check metadata consistency
        if self.dataset_metadata.get("n_classes") != len(self.y.unique()):
            errors.append("Class count mismatch between metadata and actual data")

        # Check drift points are within dataset bounds
        drift_points = self.drift_metadata.get("drift_points", [])
        for point in drift_points:
            if point >= len(self.X):
                errors.append(f"Drift point {point} is beyond dataset size {len(self.X)}")

        # Check for missing values
        if self.X.isnull().any().any():
            errors.append("Feature matrix contains missing values")

        if self.y.isnull().any():
            errors.append("Target vector contains missing values")

        return len(errors) == 0, errors

    def save(self, output_dir: Union[str, Path], formats: Optional[List[str]] = None) -> List[str]:
        """
        Save dataset to disk in specified formats.

        Args:
            output_dir: Directory to save files
            formats: List of formats to export (parquet, csv, json)

        Returns:
            List of saved file paths
        """
        if formats is None:
            formats = ["parquet"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for format_name in formats:
            if format_name == "parquet":
                # Combine features and targets for parquet
                combined_df = self.X.copy()
                combined_df[self.y.name or "target"] = self.y

                parquet_path = output_dir / f"{self.name}.parquet"
                combined_df.to_parquet(parquet_path, index=False)
                saved_paths.append(str(parquet_path))

            elif format_name == "csv":
                # Combine features and targets for CSV
                combined_df = self.X.copy()
                combined_df[self.y.name or "target"] = self.y

                csv_path = output_dir / f"{self.name}.csv"
                combined_df.to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))

            elif format_name == "json":
                # Save metadata as JSON
                def convert_to_dict(obj):
                    """Convert Pydantic models to dictionaries for JSON serialization."""
                    if hasattr(obj, "model_dump"):
                        return obj.model_dump()
                    return str(obj)

                metadata = {
                    "name": self.name,
                    "source_type": self.source_type,
                    "drift_metadata": (
                        self.drift_metadata.model_dump() if hasattr(self.drift_metadata, "model_dump") else self.drift_metadata
                    ),
                    "dataset_metadata": (
                        self.dataset_metadata.model_dump() if hasattr(self.dataset_metadata, "model_dump") else self.dataset_metadata
                    ),
                }

                json_path = output_dir / f"{self.name}_metadata.json"
                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=convert_to_dict)
                saved_paths.append(str(json_path))

        return saved_paths

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "DriftDataset":
        """
        Load dataset from disk.

        Args:
            file_path: Path to dataset file

        Returns:
            DriftDataset instance
        """
        file_path = Path(file_path)

        if file_path.suffix == ".parquet":
            # Load parquet file
            df = pd.read_parquet(file_path)

            # Try to load metadata
            metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Separate features and target
                export_info = metadata.get("_export_info", {})
                target_col = export_info.get("target_column", "target")
                if target_col in df.columns:
                    y = df[target_col]
                    X = df.drop(columns=[target_col])
                else:
                    # Assume last column is target
                    y = df.iloc[:, -1]
                    X = df.iloc[:, :-1]

                # Convert dictionaries to Pydantic models
                drift_metadata_dict = metadata.get("drift_metadata", {})
                dataset_metadata_dict = metadata.get("dataset_metadata", {})

                # Create DriftMetadata object
                if drift_metadata_dict:
                    drift_metadata_obj = DriftMetadata(**drift_metadata_dict)
                else:
                    drift_metadata_obj = DriftMetadata()

                # Create DatasetMetadata object
                if dataset_metadata_dict:
                    dataset_metadata_obj = DatasetMetadata(
                        name=metadata.get("name", file_path.stem),
                        source_type=metadata.get("source_type", "unknown"),
                        n_samples=len(X),
                        n_features=len(X.columns),
                        n_classes=len(y.unique()),
                        **{
                            k: v
                            for k, v in dataset_metadata_dict.items()
                            if k not in ["name", "source_type", "n_samples", "n_features", "n_classes"]
                        },
                    )
                else:
                    dataset_metadata_obj = DatasetMetadata(
                        name=metadata.get("name", file_path.stem),
                        source_type=metadata.get("source_type", "unknown"),
                        n_samples=len(X),
                        n_features=len(X.columns),
                        n_classes=len(y.unique()),
                    )

                return cls(
                    X=X,
                    y=y,
                    name=metadata.get("name", file_path.stem),
                    source_type=metadata.get("source_type", "unknown"),
                    drift_metadata=drift_metadata_obj,
                    dataset_metadata=dataset_metadata_obj,
                )
            else:
                # No metadata file, create basic structure
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]

                # Create basic Pydantic models
                drift_metadata_obj = DriftMetadata()
                dataset_metadata_obj = DatasetMetadata(
                    name=file_path.stem, source_type="unknown", n_samples=len(X), n_features=len(X.columns), n_classes=len(y.unique())
                )

                return cls(
                    X=X,
                    y=y,
                    name=file_path.stem,
                    source_type="unknown",
                    drift_metadata=drift_metadata_obj,
                    dataset_metadata=dataset_metadata_obj,
                )

        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def is_drift_point(self, idx: int, tolerance: int = 0) -> bool:
        """
        Check if index is near a drift point.

        REQ-007: Drift Analysis Methods - returns boolean indicating proximity to drift

        Args:
            idx: Sample index to check
            tolerance: Tolerance around drift points (default 0 = exact match)

        Returns:
            True if idx is within tolerance of a drift point
        """
        drift_points = self.drift_metadata.get("drift_points", [])

        for point in drift_points:
            if abs(idx - point) <= tolerance:
                return True

        return False

    def get_concept_segments(self) -> List[Tuple[int, int]]:
        """
        Get concept segments between drift points.

        REQ-007: Drift Analysis Methods - returns List[Tuple[int, int]] with (start, end) indices

        Returns:
            List of (start_idx, end_idx) tuples for each concept segment
        """
        drift_points = self.drift_metadata.get("drift_points", [])

        if not drift_points:
            # No drift points, entire dataset is one segment
            return [(0, len(self.X) - 1)]

        segments = []
        start_idx = 0

        for point in sorted(drift_points):
            if point > start_idx:
                segments.append((start_idx, point - 1))
                start_idx = point

        # Add final segment after last drift point
        if start_idx < len(self.X):
            segments.append((start_idx, len(self.X) - 1))

        return segments

    def get_drift_type_at(self, idx: int) -> Optional[str]:
        """
        Get drift type at specific sample index.

        REQ-007: Drift Analysis Methods - returns drift type at specific sample index

        Args:
            idx: Sample index to check

        Returns:
            Drift type at the given index, or None if no drift at that point
        """
        if not 0 <= idx < len(self.X):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.X)}")

        drift_points = self.drift_metadata.get("drift_points", [])
        drift_types = self.drift_metadata.get("drift_types", [])

        try:
            drift_idx = drift_points.index(idx)
            if drift_idx < len(drift_types):
                return drift_types[drift_idx]
        except ValueError:
            pass  # Index not in drift_points

        return None

    def get_affected_features_at(self, idx: int) -> Optional[List[int]]:
        """
        Get feature indices affected by drift at specific sample.

        REQ-007: Drift Analysis Methods - returns feature indices affected by drift at sample

        Args:
            idx: Sample index to check

        Returns:
            List of feature indices affected by drift, or None if no drift at that point
        """
        if not 0 <= idx < len(self.X):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.X)}")

        drift_points = self.drift_metadata.get("drift_points", [])
        affected_features = self.drift_metadata.get("affected_features", [])

        try:
            drift_idx = drift_points.index(idx)
            if drift_idx < len(affected_features) and affected_features[drift_idx]:
                return affected_features[drift_idx]
        except ValueError:
            pass  # Index not in drift_points

        return None

    def validate_drift_metadata(self) -> Tuple[bool, List[str]]:
        """
        Validate consistency of all drift information.

        REQ-007: Drift Analysis Methods - ensures consistency of all drift information

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        drift_points = self.drift_metadata.get("drift_points", [])
        drift_types = self.drift_metadata.get("drift_types", [])
        drift_patterns = self.drift_metadata.get("drift_patterns", [])
        drift_intensities = self.drift_metadata.get("drift_intensities", [])
        affected_features = self.drift_metadata.get("affected_features", [])

        # Check drift points are within dataset bounds
        for point in drift_points:
            if not isinstance(point, int) or point < 0 or point >= len(self.X):
                errors.append(f"Drift point {point} is not a valid index for dataset of size {len(self.X)}")

        # Check drift points are sorted
        if drift_points != sorted(drift_points):
            errors.append("Drift points must be in ascending order")

        # Check list length consistency
        n_drift_points = len(drift_points)

        if drift_types and len(drift_types) != n_drift_points:
            errors.append(f"drift_types length ({len(drift_types)}) must match drift_points length ({n_drift_points})")

        if drift_patterns and len(drift_patterns) != n_drift_points:
            errors.append(f"drift_patterns length ({len(drift_patterns)}) must match drift_points length ({n_drift_points})")

        if drift_intensities and len(drift_intensities) != n_drift_points:
            errors.append(f"drift_intensities length ({len(drift_intensities)}) must match drift_points length ({n_drift_points})")

        if affected_features and len(affected_features) != n_drift_points:
            errors.append(f"affected_features length ({len(affected_features)}) must match drift_points length ({n_drift_points})")

        # Validate drift types
        valid_drift_types = {"covariate", "concept", "prior", "none"}
        for i, drift_type in enumerate(drift_types):
            if drift_type not in valid_drift_types:
                errors.append(f"Invalid drift type '{drift_type}' at index {i}. Must be one of: {valid_drift_types}")

        # Validate drift patterns
        valid_drift_patterns = {"abrupt", "gradual", "recurring", "incremental"}
        for i, pattern in enumerate(drift_patterns):
            if pattern not in valid_drift_patterns:
                errors.append(f"Invalid drift pattern '{pattern}' at index {i}. Must be one of: {valid_drift_patterns}")

        # Validate affected features
        n_features = len(self.X.columns)
        for i, features in enumerate(affected_features):
            if features:  # Skip None/empty lists
                for feature_idx in features:
                    if not isinstance(feature_idx, int) or feature_idx < 0 or feature_idx >= n_features:
                        errors.append(f"Invalid feature index {feature_idx} in affected_features[{i}]. Must be 0 <= idx < {n_features}")

        return len(errors) == 0, errors

    def split_temporal(self, ratio: float = 0.7) -> Tuple["DriftDataset", "DriftDataset"]:
        """
        Split dataset temporally while preserving drift structure.

        REQ-009: Temporal Data Splitting - creates train/test split at specified ratio
        while preserving chronological order and drift structure

        Args:
            ratio: Ratio for train/test split (e.g., 0.7 means 70% train, 30% test)

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not 0 < ratio < 1:
            raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")

        split_idx = int(len(self.X) * ratio)

        # Split features and targets
        X_train = self.X.iloc[:split_idx].copy()
        X_test = self.X.iloc[split_idx:].copy()
        y_train = self.y.iloc[:split_idx].copy()
        y_test = self.y.iloc[split_idx:].copy()

        # Distribute drift points between splits
        drift_points = self.drift_metadata.get("drift_points", [])
        drift_types = self.drift_metadata.get("drift_types", [])
        drift_patterns = self.drift_metadata.get("drift_patterns", [])
        drift_intensities = self.drift_metadata.get("drift_intensities", [])

        train_drift_points = []
        train_drift_types = []
        train_drift_patterns = []
        train_drift_intensities = []

        test_drift_points = []
        test_drift_types = []
        test_drift_patterns = []
        test_drift_intensities = []

        for i, point in enumerate(drift_points):
            if point < split_idx:
                # Drift point in training set
                train_drift_points.append(point)
                if i < len(drift_types):
                    train_drift_types.append(drift_types[i])
                if i < len(drift_patterns):
                    train_drift_patterns.append(drift_patterns[i])
                if i < len(drift_intensities):
                    train_drift_intensities.append(drift_intensities[i])
            else:
                # Drift point in test set - adjust index
                test_drift_points.append(point - split_idx)
                if i < len(drift_types):
                    test_drift_types.append(drift_types[i])
                if i < len(drift_patterns):
                    test_drift_patterns.append(drift_patterns[i])
                if i < len(drift_intensities):
                    test_drift_intensities.append(drift_intensities[i])

        # Create drift metadata for splits
        if isinstance(self.drift_metadata, DriftMetadata):
            train_drift_metadata = DriftMetadata(
                drift_points=train_drift_points,
                drift_types=train_drift_types,
                drift_patterns=train_drift_patterns,
                drift_intensities=train_drift_intensities,
            )
            test_drift_metadata = DriftMetadata(
                drift_points=test_drift_points,
                drift_types=test_drift_types,
                drift_patterns=test_drift_patterns,
                drift_intensities=test_drift_intensities,
            )
        else:
            train_drift_metadata = {
                "drift_points": train_drift_points,
                "drift_types": train_drift_types,
                "drift_patterns": train_drift_patterns,
                "drift_intensities": train_drift_intensities,
            }
            test_drift_metadata = {
                "drift_points": test_drift_points,
                "drift_types": test_drift_types,
                "drift_patterns": test_drift_patterns,
                "drift_intensities": test_drift_intensities,
            }

        # Create dataset metadata for splits
        if isinstance(self.dataset_metadata, DatasetMetadata):
            train_dataset_metadata = DatasetMetadata(
                name=f"{self.dataset_metadata.name}_train",
                source_type=self.dataset_metadata.source_type,
                n_samples=len(X_train),
                n_features=len(X_train.columns),
                n_classes=len(y_train.unique()),
                dimension=self.dataset_metadata.dimension,
                labeling=self.dataset_metadata.labeling,
                features=self.dataset_metadata.features,
            )
            test_dataset_metadata = DatasetMetadata(
                name=f"{self.dataset_metadata.name}_test",
                source_type=self.dataset_metadata.source_type,
                n_samples=len(X_test),
                n_features=len(X_test.columns),
                n_classes=len(y_test.unique()),
                dimension=self.dataset_metadata.dimension,
                labeling=self.dataset_metadata.labeling,
                features=self.dataset_metadata.features,
            )
        else:
            train_dataset_metadata = {
                **self.dataset_metadata,
                "name": f"{self.name}_train",
                "n_samples": len(X_train),
                "n_features": len(X_train.columns),
                "n_classes": len(y_train.unique()),
            }
            test_dataset_metadata = {
                **self.dataset_metadata,
                "name": f"{self.name}_test",
                "n_samples": len(X_test),
                "n_features": len(X_test.columns),
                "n_classes": len(y_test.unique()),
            }

        # Create train and test datasets
        train_dataset = DriftDataset(
            X=X_train,
            y=y_train,
            name=f"{self.name}_train",
            source_type=self.source_type,
            drift_metadata=train_drift_metadata,
            dataset_metadata=train_dataset_metadata,
        )

        test_dataset = DriftDataset(
            X=X_test,
            y=y_test,
            name=f"{self.name}_test",
            source_type=self.source_type,
            drift_metadata=test_drift_metadata,
            dataset_metadata=test_dataset_metadata,
        )

        return train_dataset, test_dataset

    def get_continuous_features(self) -> pd.DataFrame:
        """
        Get features with continuous (numerical) types.

        REQ-008: Feature Role-Based Filtering - returns only features with type="continuous"

        Returns:
            DataFrame with only continuous features
        """
        continuous_cols = []

        # Check if we have feature metadata
        features = self.dataset_metadata.get("features", [])
        if features:
            # Use metadata to identify continuous features
            for feature in features:
                if feature.get("type") == "continuous" and feature.get("role") == "feature":
                    feature_name = feature.get("name")
                    if feature_name in self.X.columns:
                        continuous_cols.append(feature_name)
        else:
            # Fall back to pandas dtype detection
            continuous_cols = [col for col in self.X.columns if pd.api.types.is_numeric_dtype(self.X[col])]

        return self.X[continuous_cols] if continuous_cols else pd.DataFrame()

    def get_categorical_features(self) -> pd.DataFrame:
        """
        Get features with categorical types.

        REQ-008: Feature Role-Based Filtering - returns only features with type="categorical"

        Returns:
            DataFrame with only categorical features
        """
        categorical_cols = []

        # Check if we have feature metadata
        features = self.dataset_metadata.get("features", [])
        if features:
            # Use metadata to identify categorical features
            for feature in features:
                if feature.get("type") == "categorical" and feature.get("role") == "feature":
                    feature_name = feature.get("name")
                    if feature_name in self.X.columns:
                        categorical_cols.append(feature_name)
        else:
            # Fall back to pandas dtype detection
            categorical_cols = [col for col in self.X.columns if not pd.api.types.is_numeric_dtype(self.X[col])]

        return self.X[categorical_cols] if categorical_cols else pd.DataFrame()

    def get_features_by_role(self, role: str) -> pd.DataFrame:
        """
        Get features filtered by their role.

        REQ-008: Feature Role-Based Filtering - returns features matching specific role

        Args:
            role: Feature role to filter by (feature, target, timestamp, identifier, metadata, exclude)

        Returns:
            DataFrame with features matching the specified role
        """
        role_cols = []

        # Check if we have feature metadata
        features = self.dataset_metadata.get("features", [])
        if features:
            for feature in features:
                if feature.get("role") == role:
                    feature_name = feature.get("name")
                    if feature_name in self.X.columns:
                        role_cols.append(feature_name)
        else:
            # If no metadata, assume all columns are features if role="feature"
            if role == "feature":
                role_cols = list(self.X.columns)

        return self.X[role_cols] if role_cols else pd.DataFrame()

    def get_drift_detection_features(self) -> pd.DataFrame:
        """
        Get features suitable for drift detection.

        REQ-008: Feature Role-Based Filtering - excludes features with role in
        ["target", "timestamp", "identifier", "metadata", "exclude"]

        Returns:
            DataFrame with features suitable for drift detection
        """
        excluded_roles = {"target", "timestamp", "identifier", "metadata", "exclude"}
        drift_detection_cols = []

        # Check if we have feature metadata
        features = self.dataset_metadata.get("features", [])
        if features:
            for feature in features:
                role = feature.get("role", "feature")
                if role not in excluded_roles:
                    feature_name = feature.get("name")
                    if feature_name in self.X.columns:
                        drift_detection_cols.append(feature_name)
        else:
            # If no metadata, assume all columns are suitable
            drift_detection_cols = list(self.X.columns)

        return self.X[drift_detection_cols] if drift_detection_cols else pd.DataFrame()

    def summarize_features(self) -> Dict[str, Any]:
        """
        Get comprehensive feature overview.

        REQ-019: Feature Discovery and Inspection - returns comprehensive feature overview

        Returns:
            Dictionary with comprehensive feature summary
        """
        summary = {
            "n_features": len(self.X.columns),
            "feature_names": list(self.X.columns),
            "features": {},
            "by_type": {"continuous": 0, "categorical": 0, "mixed": 0},
            "by_role": {"feature": 0, "target": 0, "timestamp": 0, "identifier": 0, "metadata": 0, "exclude": 0},
        }

        # Check if we have feature metadata
        features = self.dataset_metadata.get("features", [])
        if features:
            # Use detailed metadata
            for feature in features:
                name = feature.get("name", "unknown")
                if name in self.X.columns or feature.get("role") == "target":
                    feature_type = feature.get("type", "continuous")
                    feature_role = feature.get("role", "feature")

                    summary["features"][name] = {
                        "type": feature_type,
                        "role": feature_role,
                        "description": feature.get("description", ""),
                        "missing_values": feature.get("missing_values", False),
                        "unique_values": feature.get("unique_values", None),
                    }

                    # Update counts
                    summary["by_type"][feature_type] = summary["by_type"].get(feature_type, 0) + 1
                    summary["by_role"][feature_role] = summary["by_role"].get(feature_role, 0) + 1
        else:
            # Fall back to basic analysis
            for col in self.X.columns:
                feature_type = "continuous" if pd.api.types.is_numeric_dtype(self.X[col]) else "categorical"

                summary["features"][col] = {
                    "type": feature_type,
                    "role": "feature",
                    "description": "",
                    "missing_values": self.X[col].isnull().any(),
                    "unique_values": int(self.X[col].nunique()),
                }

                summary["by_type"][feature_type] += 1
                summary["by_role"]["feature"] += 1

        return summary

    def get_feature_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed metadata for specific feature.

        REQ-019: Feature Discovery and Inspection - provides detailed metadata for specific features

        Args:
            name: Name of the feature to get info for

        Returns:
            Dictionary with detailed feature information
        """
        if name not in self.X.columns and name != (self.y.name or "target"):
            raise ValueError(f"Feature '{name}' not found in dataset")

        # Check if we have feature metadata
        features = self.dataset_metadata.get("features", [])
        feature_metadata = None

        for feature in features:
            if feature.get("name") == name:
                feature_metadata = feature
                break

        if feature_metadata:
            info = {
                "name": name,
                "type": feature_metadata.get("type", "unknown"),
                "role": feature_metadata.get("role", "unknown"),
                "description": feature_metadata.get("description", ""),
                "missing_values": feature_metadata.get("missing_values", False),
                "unique_values": feature_metadata.get("unique_values", None),
            }
        else:
            # Generate basic info from data
            if name in self.X.columns:
                col_data = self.X[name]
                info = {
                    "name": name,
                    "type": "continuous" if pd.api.types.is_numeric_dtype(col_data) else "categorical",
                    "role": "feature",
                    "description": "",
                    "missing_values": col_data.isnull().any(),
                    "unique_values": int(col_data.nunique()),
                }
            else:
                # Target variable
                info = {
                    "name": name,
                    "type": "categorical",
                    "role": "target",
                    "description": "",
                    "missing_values": self.y.isnull().any(),
                    "unique_values": int(self.y.nunique()),
                }

        # Add statistical information if numeric
        if name in self.X.columns and pd.api.types.is_numeric_dtype(self.X[name]):
            col_data = self.X[name]
            info["statistics"] = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "median": float(col_data.median()),
            }
        elif name == (self.y.name or "target") and pd.api.types.is_numeric_dtype(self.y):
            info["statistics"] = {
                "mean": float(self.y.mean()),
                "std": float(self.y.std()),
                "min": float(self.y.min()),
                "max": float(self.y.max()),
                "median": float(self.y.median()),
            }

        return info

    def _calculate_drift_intervals(self) -> List[int]:
        """Calculate intervals between drift points."""
        drift_points = self.drift_metadata.get("drift_points", [])
        if not drift_points:
            return [len(self.X)]

        intervals = []
        prev_point = 0
        for point in drift_points:
            intervals.append(point - prev_point)
            prev_point = point

        # Add final interval
        intervals.append(len(self.X) - prev_point)

        return intervals

    def get(self, key: str, default=None):
        """Dictionary-style get method for backward compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """Dictionary-style access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Dictionary-style 'in' operator for backward compatibility."""
        return hasattr(self, key)
