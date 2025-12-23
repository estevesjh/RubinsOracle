"""Utility functions for Rubin's Oracle."""

from __future__ import annotations

from rubin_oracle.utils.data import (
    add_time_features,
    check_missing_values,
    compute_temp_mean,
    get_frequency,
    prepare_regular_frequency,
    validate_input,
)
from rubin_oracle.utils.frequency import FrequencyConverter, SampleConverter
from rubin_oracle.utils.metrics import MetricsCalculator
from rubin_oracle.utils.output import OutputFormatter
from rubin_oracle.utils.persistence import ModelPersistence
from rubin_oracle.utils.postprocessing import PostProcessor
from rubin_oracle.utils.seasonality import SeasonalityConverter

__all__ = [
    # Data utilities
    "add_time_features",
    "check_missing_values",
    "compute_temp_mean",
    "get_frequency",
    "prepare_regular_frequency",
    "validate_input",
    # New helper classes
    "FrequencyConverter",
    "SampleConverter",
    "MetricsCalculator",
    "OutputFormatter",
    "SeasonalityConverter",
    "ModelPersistence",
    "PostProcessor",
]
