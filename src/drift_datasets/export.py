"""
Dataset export functionality for drift_datasets library.

This module provides exporters for different file formats including ARFF, CSV,
Parquet, and JSON for compatibility with various machine learning tools.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class ARFFExporter:
    """Export datasets to ARFF format for Weka compatibility."""

    def export(self, dataset, output_path: str) -> Dict[str, Any]:
        """
        Export dataset to ARFF format.

        Args:
            dataset: DriftDataset object or dataset dictionary
            output_path: Path to output ARFF file

        Returns:
            Export result dictionary with success flag and format
        """
        output_path = Path(output_path)

        # Handle both DriftDataset objects and dictionaries
        if hasattr(dataset, "X"):
            X = dataset.X
            y = dataset.y
            name = getattr(dataset, "name", "dataset")
        else:
            X = dataset["X"]
            y = dataset["y"]
            name = dataset.get("name", "dataset")

        # Combine features and target
        full_data = X.copy()
        full_data["target"] = y

        # Generate ARFF content
        arff_content = self._generate_arff_content(full_data, name)

        # Write to file
        with open(output_path, "w") as f:
            f.write(arff_content)

        return {"success": True, "format": "arff"}

    def _generate_arff_content(self, data: pd.DataFrame, dataset_name: str) -> str:
        """Generate ARFF file content from DataFrame."""
        lines = []

        # Header
        lines.append(f"@relation {dataset_name}")
        lines.append("")

        # Attributes
        for column in data.columns:
            if data[column].dtype in ["object", "category"] or column == "target":
                # Categorical attribute
                unique_values = sorted(data[column].unique())
                values_str = ",".join(str(v) for v in unique_values)
                lines.append(f"@attribute {column} {{{values_str}}}")
            else:
                # Numeric attribute
                lines.append(f"@attribute {column} numeric")

        lines.append("")
        lines.append("@data")

        # Data rows
        for _, row in data.iterrows():
            row_values = []
            for value in row:
                if pd.isna(value):
                    row_values.append("?")
                else:
                    row_values.append(str(value))
            lines.append(",".join(row_values))

        return "\n".join(lines)


class CSVExporter:
    """Export datasets to CSV format."""

    def export(self, dataset, output_path: str) -> Dict[str, Any]:
        """
        Export dataset to CSV format.

        Args:
            dataset: DriftDataset object or dataset dictionary
            output_path: Path to output CSV file

        Returns:
            Export result dictionary with success flag and format
        """
        output_path = Path(output_path)

        # Handle both DriftDataset objects and dictionaries
        if hasattr(dataset, "X"):
            X = dataset.X
            y = dataset.y
        else:
            X = dataset["X"]
            y = dataset["y"]

        # Combine features and target
        full_data = X.copy()
        full_data["target"] = y

        # Write to CSV
        full_data.to_csv(output_path, index=False)

        return {"success": True, "format": "csv"}


class ParquetExporter:
    """Export datasets to Parquet format."""

    def export(self, dataset, output_path: str) -> Dict[str, Any]:
        """
        Export dataset to Parquet format.

        Args:
            dataset: DriftDataset object or dataset dictionary
            output_path: Path to output Parquet file

        Returns:
            Export result dictionary with success flag and format
        """
        output_path = Path(output_path)

        # Handle both DriftDataset objects and dictionaries
        if hasattr(dataset, "X"):
            X = dataset.X
            y = dataset.y
            name = getattr(dataset, "name", "dataset")
            source_type = getattr(dataset, "source_type", "unknown")
            drift_metadata = dataset.drift_metadata
            dataset_metadata = getattr(dataset, "dataset_metadata", {})
        else:
            X = dataset["X"]
            y = dataset["y"]
            name = dataset.get("name", "dataset")
            source_type = dataset.get("source_type", "unknown")
            drift_metadata = dataset.get("drift_metadata", {})
            dataset_metadata = dataset.get("dataset_metadata", {})

        # Combine features and target
        full_data = X.copy()
        full_data["target"] = y

        # Write to Parquet
        full_data.to_parquet(output_path, index=False)

        # Create companion metadata file
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"

        # Convert Pydantic models to dictionaries for JSON serialization
        def convert_pydantic_to_dict(obj):
            """Convert Pydantic models to dictionaries for JSON serialization."""
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return obj

        metadata = {
            "name": name,
            "source_type": source_type,
            "drift_metadata": convert_pydantic_to_dict(drift_metadata),
            "dataset_metadata": convert_pydantic_to_dict(dataset_metadata),
            "_export_info": {"target_column": "target"},
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {"success": True, "format": "parquet"}


class MetadataExporter:
    """Export dataset metadata to JSON format."""

    def export(self, dataset, output_path: str) -> Dict[str, Any]:
        """
        Export dataset metadata to JSON format.

        Args:
            dataset: DriftDataset object or dataset dictionary
            output_path: Path to output JSON file

        Returns:
            Export result dictionary with success flag and format
        """
        output_path = Path(output_path)

        # Extract metadata from dataset
        if hasattr(dataset, "drift_metadata"):
            # DriftDataset object
            drift_metadata = dataset.drift_metadata
            if hasattr(drift_metadata, "drift_points"):
                # drift_metadata is an object
                drift_data = {
                    "drift_points": drift_metadata.drift_points,
                    "drift_types": drift_metadata.drift_types,
                    "drift_patterns": drift_metadata.drift_patterns,
                    "drift_intensities": drift_metadata.drift_intensities,
                }
            else:
                # drift_metadata is a dict
                drift_data = {
                    "drift_points": drift_metadata.get("drift_points", []),
                    "drift_types": drift_metadata.get("drift_types", []),
                    "drift_patterns": drift_metadata.get("drift_patterns", []),
                    "drift_intensities": drift_metadata.get("drift_intensities", []),
                }

            metadata = {
                "name": getattr(dataset, "name", "dataset"),
                "drift_metadata": drift_data,
                "dataset_metadata": {
                    "n_samples": len(dataset.X),
                    "n_features": len(dataset.X.columns),
                    "n_classes": len(dataset.y.unique()) if hasattr(dataset, "y") else 0,
                    "feature_names": list(dataset.X.columns),
                },
            }
        else:
            # Dataset dictionary
            metadata = {
                "name": dataset.get("name", "dataset"),
                "drift_metadata": dataset.get("drift_metadata", {}),
                "dataset_metadata": {
                    "n_samples": len(dataset["X"]),
                    "n_features": len(dataset["X"].columns),
                    "n_classes": len(dataset["y"].unique()) if "y" in dataset else 0,
                    "feature_names": list(dataset["X"].columns),
                },
            }

        # Write to JSON
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {"success": True, "format": "json"}


class BatchExporter:
    """Export datasets to multiple formats simultaneously."""

    def __init__(self):
        self.exporters = {
            "arff": ARFFExporter(),
            "csv": CSVExporter(),
            "parquet": ParquetExporter(),
            "json": MetadataExporter(),
        }

    def export_multiple(self, dataset, output_dir: str, formats: list) -> Dict[str, Any]:
        """
        Export dataset to multiple formats.

        Args:
            dataset: DriftDataset object or dataset dictionary
            output_dir: Directory to write output files
            formats: List of formats to export ("arff", "csv", "parquet", "json")

        Returns:
            Export result dictionary with success flag and results for each format
        """
        return self.export(dataset, output_dir, formats)

    def export(self, dataset, output_dir: str, formats: list) -> Dict[str, Any]:
        """
        Export dataset to multiple formats.

        Args:
            dataset: DriftDataset object or dataset dictionary
            output_dir: Directory to write output files
            formats: List of formats to export ("arff", "csv", "parquet", "json")

        Returns:
            Export result dictionary with success flag and results for each format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Get dataset name for file naming
        if hasattr(dataset, "name"):
            dataset_name = dataset.name
        else:
            dataset_name = dataset.get("name", "dataset")

        results = {}
        all_success = True
        exported_files = []

        for format_type in formats:
            if format_type not in self.exporters:
                results[format_type] = {"success": False, "error": f"Unsupported format: {format_type}"}
                all_success = False
                continue

            # Determine output file path
            if format_type == "json":
                output_path = output_dir / f"{dataset_name}_metadata.json"
            else:
                output_path = output_dir / f"{dataset_name}.{format_type}"

            try:
                exporter = self.exporters[format_type]
                result = exporter.export(dataset, str(output_path))
                results[format_type] = result
                if result["success"]:
                    exported_files.append(str(output_path))
                else:
                    all_success = False
            except Exception as e:
                results[format_type] = {"success": False, "error": str(e)}
                all_success = False

        return {
            "success": all_success,
            "results": results,
            "exported_formats": list(formats),
            "exported_files": exported_files,
            "export_summary": {
                "total_formats": len(formats),
                "successful_exports": len(exported_files),
                "failed_exports": len(formats) - len(exported_files),
            },
        }
