"""
Unit tests for generator parameter translation.

Tests REQ-012A: Generator Parameter Translation.
"""

from unittest.mock import Mock, patch

import pytest

from drift_datasets.generators.synthetic import CapyMOAInterface


class TestGeneratorParameterTranslation:
    """Test generator parameter translation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.interface = CapyMOAInterface()

    def test_translate_research_parameters_transition_durations(self):
        """Test translation of transition_durations to drift_widths."""
        drift_config = {"drift_points": [1000, 2000], "transition_durations": [500, 750]}

        translated = self.interface._translate_research_parameters("HyperplaneGenerator", drift_config)

        assert "drift_widths" in translated
        assert translated["drift_widths"] == [500, 750]
        assert "transition_durations" in translated  # Original preserved

    def test_translate_research_parameters_drift_intensities(self):
        """Test translation of drift_intensities to drift_alphas."""
        drift_config = {"drift_points": [1000, 2000], "drift_intensities": [0.3, 0.8]}

        translated = self.interface._translate_research_parameters("HyperplaneGenerator", drift_config)

        assert "drift_alphas" in translated
        assert translated["drift_alphas"] == [0.3, 0.8]
        assert "drift_intensities" in translated  # Original preserved

    def test_translate_research_parameters_precedence(self):
        """Test that research parameters take precedence over CapyMOA parameters."""
        drift_config = {
            "drift_points": [1000, 2000],
            "transition_durations": [500, 750],  # Research parameter
            "drift_widths": [200, 300],  # CapyMOA parameter (should be overridden)
        }

        translated = self.interface._translate_research_parameters("HyperplaneGenerator", drift_config)

        assert translated["drift_widths"] == [500, 750]  # Research parameter wins

    def test_translate_research_parameters_no_research_params(self):
        """Test that translation works when no research parameters present."""
        drift_config = {"drift_points": [1000, 2000], "drift_widths": [200, 300]}  # Only CapyMOA parameters

        translated = self.interface._translate_research_parameters("HyperplaneGenerator", drift_config)

        assert translated["drift_widths"] == [200, 300]  # CapyMOA parameters preserved

    def test_validate_research_parameters_length_mismatch(self):
        """Test validation of research parameter lengths."""
        parameters = {"drift_points": [1000, 2000], "transition_durations": [500]}  # Length mismatch

        with pytest.raises(ValueError) as exc_info:
            self.interface._validate_research_parameters("HyperplaneGenerator", parameters)

        assert "transition_durations length" in str(exc_info.value)

    def test_validate_research_parameters_intensity_range(self):
        """Test validation of drift_intensities range."""
        parameters = {"drift_points": [1000], "drift_intensities": [1.5]}  # Out of range

        with pytest.raises(ValueError) as exc_info:
            self.interface._validate_research_parameters("HyperplaneGenerator", parameters)

        assert "drift_intensities" in str(exc_info.value)
        assert "range [0.0, 1.0]" in str(exc_info.value)

    def test_validate_research_parameters_valid_config(self):
        """Test validation passes for valid research parameters."""
        parameters = {"drift_points": [1000, 2000], "transition_durations": [500, 750], "drift_intensities": [0.3, 0.8]}

        # Should not raise any exception
        self.interface._validate_research_parameters("HyperplaneGenerator", parameters)

    def test_enhanced_parameter_validation(self):
        """Test that enhanced validation includes research parameters."""
        # This tests that the existing _validate_parameters method
        # now also validates research parameters

        parameters = {
            "n_instances": 1000,
            "n_features": 5,
            "drift_points": [500],
            "transition_durations": [250],
            "drift_intensities": [0.5],
        }

        # Should not raise any exception
        self.interface._validate_parameters("HyperplaneGenerator", parameters)

    def test_enhanced_parameter_validation_invalid_research_param(self):
        """Test that enhanced validation rejects invalid research parameters."""
        parameters = {"n_instances": 1000, "n_features": 5, "drift_points": [500], "transition_durations": [250, 300]}  # Length mismatch

        with pytest.raises(ValueError) as exc_info:
            self.interface._validate_parameters("HyperplaneGenerator", parameters)

        assert "transition_durations length" in str(exc_info.value)
