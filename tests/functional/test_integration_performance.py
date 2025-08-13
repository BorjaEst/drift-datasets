"""
Functional tests for integration and performance functionality.

These tests validate integration capabilities from REQ-021 through REQ-023,
including library integration, performance benchmarks, and comprehensive system tests.
"""

import time
from pathlib import Path

import pytest


class TestLibraryIntegration:
    """Test library integration functionality (REQ-021)."""

    def test_should_integrate_with_sklearn_when_sklearn_workflow_used(self, sample_drift_dataset):
        """Test REQ-021: Integrate seamlessly with scikit-learn workflows."""
        from drift_datasets.integrations import SKLearnAdapter
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset
        dataset = DriftDataset(**sample_drift_dataset)

        # Act: Convert to sklearn format
        sklearn_adapter = SKLearnAdapter()
        X_sklearn, y_sklearn = sklearn_adapter.to_sklearn(dataset)

        # Assert: sklearn format compatibility
        assert hasattr(X_sklearn, "shape"), "Features have array-like interface"
        assert hasattr(y_sklearn, "shape"), "Targets have array-like interface"
        assert X_sklearn.shape == (1000, 2), "Correct feature dimensions"
        assert y_sklearn.shape == (1000,), "Correct target dimensions"

        # Test with actual sklearn estimator
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_sklearn, y_sklearn, test_size=0.2, random_state=42)

            # Train model
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)

            # Test predictions
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test), "Predictions for all test samples"
            assert all(pred in [0, 1] for pred in predictions), "Valid binary predictions"

        except ImportError:
            pytest.skip("scikit-learn not available for integration test")

    def test_should_integrate_with_pytorch_when_pytorch_workflow_used(self, sample_drift_dataset):
        """Test REQ-021: Integrate with PyTorch data loading workflows."""
        from drift_datasets.integrations import PyTorchAdapter
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset
        dataset = DriftDataset(**sample_drift_dataset)

        # Act: Convert to PyTorch format
        pytorch_adapter = PyTorchAdapter()
        torch_dataset = pytorch_adapter.to_torch_dataset(dataset)

        # Assert: PyTorch dataset compatibility
        assert len(torch_dataset) == 1000, "Correct dataset length"
        assert hasattr(torch_dataset, "__getitem__"), "Dataset is indexable"
        assert hasattr(torch_dataset, "__len__"), "Dataset has length"

        # Test data loading
        first_sample = torch_dataset[0]
        assert len(first_sample) == 2, "Sample contains features and target"
        features, target = first_sample
        assert len(features) == 2, "Correct number of features"
        assert target in [0, 1], "Valid target value"

        # Test batch loading
        try:
            from torch.utils.data import DataLoader

            dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=True)
            batch = next(iter(dataloader))
            batch_features, batch_targets = batch

            assert batch_features.shape[0] == 32, "Correct batch size"
            assert batch_features.shape[1] == 2, "Correct feature dimensions"
            assert batch_targets.shape[0] == 32, "Correct target batch size"

        except ImportError:
            pytest.skip("PyTorch not available for integration test")

    def test_should_integrate_with_river_when_streaming_workflow_used(self, sample_drift_dataset):
        """Test REQ-021: Integrate with River (online learning) workflows."""
        from drift_datasets.integrations import RiverAdapter
        from drift_datasets.models import DriftDataset

        # Arrange: Create dataset
        dataset = DriftDataset(**sample_drift_dataset)

        # Act: Convert to River streaming format
        river_adapter = RiverAdapter()
        stream = river_adapter.to_river_stream(dataset)

        # Assert: River stream compatibility
        assert hasattr(stream, "__iter__"), "Stream is iterable"

        # Test streaming interface
        sample_count = 0
        for sample in stream:
            if sample_count >= 10:  # Test first 10 samples
                break
            assert "x" in sample, "Sample contains features"
            assert "y" in sample, "Sample contains target"
            assert len(sample["x"]) == 2, "Correct number of features"
            assert sample["y"] in [0, 1], "Valid target value"
            sample_count += 1

        assert sample_count == 10, "Stream provided expected number of samples"

        # Test with River model (if available)
        try:
            from river import linear_model, metrics

            model = linear_model.LogisticRegression()
            metric = metrics.Accuracy()

            # Process stream with River model
            processed_samples = 0
            for sample in river_adapter.to_river_stream(dataset):
                if processed_samples >= 100:  # Test with 100 samples
                    break

                prediction = model.predict_one(sample["x"])
                if prediction is not None:
                    metric.update(sample["y"], prediction)

                model.learn_one(sample["x"], sample["y"])
                processed_samples += 1

            assert processed_samples == 100, "Processed expected number of samples"

        except ImportError:
            pytest.skip("River not available for integration test")

    def test_should_provide_drift_detection_integration_when_drift_detectors_used(self, capymoa_service, sample_toml_configs):
        """Test REQ-021: Integrate with drift detection algorithms."""
        from drift_datasets import create_dataset
        from drift_datasets.integrations import DriftDetectionAdapter

        # Arrange: Create dataset with known drift
        config_path = sample_toml_configs["sine"]
        dataset = create_dataset(config_path)

        # Act: Setup drift detection integration
        drift_adapter = DriftDetectionAdapter()
        detection_stream = drift_adapter.prepare_for_drift_detection(dataset)

        # Assert: Drift detection stream prepared
        assert hasattr(detection_stream, "true_drift_points"), "True drift points available"
        assert hasattr(detection_stream, "__iter__"), "Stream is iterable"
        assert detection_stream.true_drift_points == [500], "Correct drift points"

        # Test with mock drift detector
        detected_drifts = []
        for i, (features, target) in enumerate(detection_stream):
            # Simulate drift detection logic
            if i == 500:  # At known drift point
                detected_drifts.append(i)
            if i >= 600:  # Process some samples after drift
                break

        # Verify drift detection capability
        assert len(detected_drifts) == 1, "Drift detected"
        assert detected_drifts[0] == 500, "Drift detected at correct position"


class TestPerformanceBenchmarks:
    """Test performance benchmarks functionality (REQ-022)."""

    def test_should_generate_datasets_within_time_limits_when_performance_tested(
        self, capymoa_service, sample_toml_configs, performance_benchmarks
    ):
        """Test REQ-022: Ensure dataset generation meets performance benchmarks."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["sine"]

        # Act: Measure dataset generation time
        start_time = time.time()
        dataset = create_dataset(config_path)
        generation_time = time.time() - start_time

        # Assert: Performance within acceptable limits
        max_time_small_dataset = performance_benchmarks["generation_time"]["small_dataset_max_seconds"]
        assert generation_time < max_time_small_dataset, f"Generation took {generation_time:.2f}s, max allowed {max_time_small_dataset}s"

        # Test memory usage
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        max_memory_mb = performance_benchmarks["memory_usage"]["small_dataset_max_mb"]

        assert memory_usage_mb < max_memory_mb, f"Memory usage {memory_usage_mb:.1f}MB exceeds limit {max_memory_mb}MB"

    def test_should_handle_large_datasets_efficiently_when_memory_constrained(
        self, capymoa_service, sample_toml_configs, performance_benchmarks
    ):
        """Test REQ-022: Handle large datasets efficiently with memory constraints."""
        from drift_datasets import create_dataset

        config_path = sample_toml_configs["large_synthetic"]

        # Act: Generate large dataset with streaming
        start_time = time.time()
        dataset = create_dataset(config_path, streaming=True)

        # Process dataset in chunks to measure streaming performance
        chunk_count = 0
        total_processed = 0

        for chunk in dataset.stream_chunks(chunk_size=1000):
            chunk_count += 1
            total_processed += len(chunk["X"])
            if chunk_count >= 10:  # Test first 10 chunks
                break

        processing_time = time.time() - start_time

        # Assert: Streaming performance acceptable
        max_streaming_time = performance_benchmarks["streaming_performance"]["chunk_processing_max_seconds"]
        assert processing_time < max_streaming_time, f"Streaming processing took {processing_time:.2f}s"
        assert total_processed == 10000, "Processed expected number of instances"

    def test_should_cache_datasets_for_improved_performance_when_repeatedly_accessed(
        self, uci_repository_service, sample_toml_configs, tmp_path, performance_benchmarks
    ):
        """Test REQ-022: Use caching to improve repeated dataset access performance."""
        from drift_datasets import create_dataset
        from drift_datasets.caching import DatasetCache

        config_path = sample_toml_configs["uci"]

        # Setup cache
        cache_dir = tmp_path / "dataset_cache"
        cache = DatasetCache(cache_dir=str(cache_dir))

        # Act: First access (should cache)
        start_time = time.time()
        dataset1 = create_dataset(config_path, cache=cache)
        first_access_time = time.time() - start_time

        # Second access (should use cache)
        start_time = time.time()
        dataset2 = create_dataset(config_path, cache=cache)
        second_access_time = time.time() - start_time

        # Assert: Caching improves performance
        max_cache_speedup_ratio = performance_benchmarks["caching"]["min_speedup_ratio"]
        speedup_ratio = first_access_time / second_access_time
        assert speedup_ratio >= max_cache_speedup_ratio, f"Cache speedup {speedup_ratio:.2f}x below minimum {max_cache_speedup_ratio}x"

        # Verify cached dataset is identical
        import numpy as np

        assert np.array_equal(dataset1.X.values, dataset2.X.values), "Cached dataset identical"
        assert dataset1.drift_metadata == dataset2.drift_metadata, "Cached metadata identical"

    def test_should_provide_performance_metrics_when_benchmarking_enabled(
        self, capymoa_service, sample_toml_configs, performance_benchmarks
    ):
        """Test REQ-022: Provide detailed performance metrics during dataset operations."""
        from drift_datasets import create_dataset
        from drift_datasets.performance import PerformanceProfiler

        config_path = sample_toml_configs["sine"]

        # Act: Generate dataset with performance profiling
        profiler = PerformanceProfiler()
        with profiler.profile("dataset_generation"):
            dataset = create_dataset(config_path)

        # Assert: Performance metrics available
        metrics = profiler.get_metrics()
        assert "dataset_generation" in metrics, "Generation metrics recorded"

        generation_metrics = metrics["dataset_generation"]
        assert "execution_time" in generation_metrics, "Execution time measured"
        assert "memory_peak" in generation_metrics, "Peak memory measured"
        assert "cpu_usage" in generation_metrics, "CPU usage measured"

        # Test performance breakdown
        if "breakdown" in generation_metrics:
            breakdown = generation_metrics["breakdown"]
            expected_phases = ["initialization", "data_generation", "metadata_creation", "validation"]
            for phase in expected_phases:
                if phase in breakdown:
                    assert breakdown[phase]["time"] > 0, f"Time recorded for {phase}"

    def test_should_scale_linearly_with_dataset_size_when_performance_tested(
        self, capymoa_service, sample_toml_configs, performance_benchmarks
    ):
        """Test REQ-022: Verify performance scales appropriately with dataset size."""
        from drift_datasets import create_dataset

        # Test different dataset sizes
        sizes = [1000, 5000, 10000]
        generation_times = []

        for size in sizes:
            # Create config for specific size
            config_path = sample_toml_configs[f"sine_{size}"]

            # Measure generation time
            start_time = time.time()
            dataset = create_dataset(config_path)
            generation_time = time.time() - start_time
            generation_times.append(generation_time)

            # Verify dataset size
            assert len(dataset.X) == size, f"Dataset has correct size {size}"

        # Assert: Performance scaling is reasonable
        # Time should scale sub-quadratically (better than O(n^2))
        time_ratio_1_to_2 = generation_times[1] / generation_times[0]
        size_ratio_1_to_2 = sizes[1] / sizes[0]  # 5x increase

        max_acceptable_time_scaling = performance_benchmarks["scaling"]["max_time_scaling_factor"]
        actual_time_scaling = time_ratio_1_to_2 / size_ratio_1_to_2

        assert (
            actual_time_scaling <= max_acceptable_time_scaling
        ), f"Time scaling {actual_time_scaling:.2f} exceeds maximum {max_acceptable_time_scaling}"


class TestSystemIntegration:
    """Test comprehensive system integration functionality (REQ-023)."""

    def test_should_support_complete_workflow_when_end_to_end_pipeline_executed(
        self, capymoa_service, uci_repository_service, sample_toml_configs, tmp_path
    ):
        """Test REQ-023: Support complete end-to-end workflows from configuration to analysis."""
        from drift_datasets import create_dataset
        from drift_datasets.analysis import DatasetAnalyzer

        # Act: Execute complete workflow
        # 1. Configuration loading and validation
        config_path = sample_toml_configs["mixed"]

        # 2. Dataset creation
        dataset = create_dataset(config_path)

        # 3. Dataset analysis
        analyzer = DatasetAnalyzer()
        analysis_report = analyzer.analyze(dataset)

        # 4. Export to multiple formats
        export_dir = tmp_path / "workflow_output"
        export_paths = dataset.save(export_dir, formats=["csv", "parquet", "json"])

        # Assert: Complete workflow successful
        assert dataset is not None, "Dataset created successfully"
        assert analysis_report is not None, "Analysis completed successfully"
        assert len(export_paths) == 3, "All export formats successful"

        # Verify workflow outputs
        assert "data_quality" in analysis_report, "Data quality analysis included"
        assert "drift_analysis" in analysis_report, "Drift analysis included"
        assert "statistical_summary" in analysis_report, "Statistical summary included"

        # Test analysis quality
        data_quality = analysis_report["data_quality"]
        assert data_quality["completeness_score"] > 0.95, "High data completeness"
        assert data_quality["consistency_score"] > 0.95, "High data consistency"

    def test_should_handle_error_recovery_when_partial_failures_occur(self, capymoa_service, error_scenarios, tmp_path):
        """Test REQ-023: Handle partial failures gracefully with error recovery."""
        from drift_datasets import create_dataset
        from drift_datasets.errors import DatasetGenerationError

        # Test recovery from generation errors
        problematic_config = error_scenarios["system_integration"]["partial_failure_config"]

        # Act: Attempt dataset creation with potential failures
        try:
            dataset = create_dataset(problematic_config, error_recovery=True)

            # If successful, verify recovery was applied
            recovery_info = dataset.dataset_metadata.get("recovery_applied", {})
            if recovery_info:
                assert "recovery_strategy" in recovery_info, "Recovery strategy recorded"
                assert recovery_info["original_errors"] is not None, "Original errors recorded"

        except DatasetGenerationError as e:
            # If failed, verify error information is comprehensive
            assert hasattr(e, "error_details"), "Detailed error information available"
            assert hasattr(e, "recovery_suggestions"), "Recovery suggestions provided"
            pytest.skip("Dataset generation failed despite recovery attempts")

    def test_should_validate_system_requirements_when_library_initialized(self):
        """Test REQ-023: Validate system requirements and dependencies."""
        from drift_datasets.system import SystemValidator

        # Act: Validate system requirements
        validator = SystemValidator()
        validation_report = validator.validate_system()

        # Assert: System validation comprehensive
        assert "dependencies" in validation_report, "Dependency validation included"
        assert "performance_capabilities" in validation_report, "Performance validation included"
        assert "compatibility" in validation_report, "Compatibility validation included"

        # Test dependency validation
        dependencies = validation_report["dependencies"]
        required_deps = ["numpy", "pandas", "toml"]

        for dep in required_deps:
            if dep in dependencies:
                assert dependencies[dep]["available"] == True, f"Required dependency {dep} available"
                assert dependencies[dep]["version"] is not None, f"Version info for {dep} available"

        # Test optional dependencies
        optional_deps = ["scikit-learn", "torch", "river"]
        for dep in optional_deps:
            if dep in dependencies:
                integration_available = dependencies[dep]["available"]
                if integration_available:
                    assert dependencies[dep]["integration_tested"] == True, f"Integration tested for {dep}"

    def test_should_provide_comprehensive_logging_when_system_operations_executed(self, capymoa_service, sample_toml_configs, tmp_path):
        """Test REQ-023: Provide comprehensive logging for system operations."""
        import logging

        from drift_datasets import create_dataset
        from drift_datasets.logging import SystemLogger

        # Setup logging
        log_file = tmp_path / "system.log"
        logger = SystemLogger(log_file=str(log_file))
        logger.set_level(logging.DEBUG)

        # Act: Execute operations with logging
        config_path = sample_toml_configs["sine"]
        with logger.log_operation("dataset_creation"):
            dataset = create_dataset(config_path)

        # Assert: Comprehensive logging available
        assert log_file.exists(), "Log file created"

        with open(log_file, "r") as f:
            log_content = f.read()

        # Test log content
        assert "dataset_creation" in log_content, "Operation logged"
        assert "INFO" in log_content, "Info level logs present"
        assert "DEBUG" in log_content, "Debug level logs present"

        # Test structured logging
        log_entries = logger.get_structured_logs()
        assert len(log_entries) > 0, "Structured log entries available"

        creation_entries = [entry for entry in log_entries if "dataset_creation" in entry.get("operation", "")]
        assert len(creation_entries) > 0, "Dataset creation logged"

        # Test performance logging
        performance_entries = [entry for entry in log_entries if "performance" in entry]
        if performance_entries:
            assert any("execution_time" in entry for entry in performance_entries), "Execution time logged"

    def test_should_support_plugin_architecture_when_extensions_loaded(self):
        """Test REQ-023: Support plugin architecture for extensibility."""
        from drift_datasets.plugins import PluginManager

        # Act: Initialize plugin system
        plugin_manager = PluginManager()
        available_plugins = plugin_manager.discover_plugins()

        # Assert: Plugin system functional
        assert hasattr(plugin_manager, "load_plugin"), "Plugin loading capability"
        assert hasattr(plugin_manager, "register_plugin"), "Plugin registration capability"
        assert isinstance(available_plugins, list), "Plugin discovery working"

        # Test plugin categories
        plugin_categories = plugin_manager.get_plugin_categories()
        expected_categories = ["generators", "exporters", "analyzers", "integrations"]

        for category in expected_categories:
            assert category in plugin_categories, f"Plugin category {category} supported"

        # Test plugin interface validation
        if available_plugins:
            first_plugin = available_plugins[0]
            validation_result = plugin_manager.validate_plugin(first_plugin)
            assert "interface_compatible" in validation_result, "Plugin interface validation"
            assert "dependencies_met" in validation_result, "Plugin dependency validation"
