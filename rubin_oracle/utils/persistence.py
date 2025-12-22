"""Model persistence utilities for Rubin's Oracle forecasters.

This module provides utilities for saving and loading forecaster models,
configs, and metadata across different model types (Prophet, NeuralProphet, Ensemble).
"""

import json
from pathlib import Path
from typing import Any, Union, cast

import pandas as pd

from rubin_oracle.config import (
    EnsembleConfig,
    NeuralProphetConfig,
    ProphetConfig,
)


class ModelPersistence:
    """Static utility class for model persistence operations.

    Handles saving and loading of configs (YAML), Prophet models (JSON),
    NeuralProphet models (torch), and metadata (JSON).
    """

    @staticmethod
    def save_config(
        config: Union[ProphetConfig, NeuralProphetConfig, EnsembleConfig], path: Union[str, Path]
    ) -> None:
        """Save config to YAML file.

        Args:
            config: Forecaster config object
            path: File path to save YAML config

        Raises:
            IOError: If file cannot be written
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(path))

    @staticmethod
    def load_config(
        path: Union[str, Path], config_class: type
    ) -> Union[ProphetConfig, NeuralProphetConfig, EnsembleConfig]:
        """Load config from YAML file.

        Args:
            path: File path to load YAML config
            config_class: Config class to instantiate (ProphetConfig, NeuralProphetConfig, EnsembleConfig)

        Returns:
            Loaded config object

        Raises:
            IOError: If file cannot be read
            ValueError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise OSError(f"Config file not found: {path}")

        return cast(
            Union[ProphetConfig, NeuralProphetConfig, EnsembleConfig],
            config_class.from_yaml(str(path)),
        )

    @staticmethod
    def save_prophet_model(model, path: Union[str, Path]) -> None:
        """Save Prophet model to JSON file.

        Args:
            model: Prophet model object
            path: File path to save model

        Raises:
            IOError: If file cannot be written
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prophet has built-in serialization
        with open(path, "w") as f:
            json.dump(model.stan_backend.model_name, f)

        # Also save the model using its method if available
        if hasattr(model, "save"):
            model.save(str(path))

    @staticmethod
    def load_prophet_model(path: Union[str, Path]) -> Any:
        """Load Prophet model from file.

        Args:
            path: File path to load model

        Returns:
            Loaded Prophet model

        Raises:
            IOError: If file cannot be read
        """
        from prophet import Prophet

        path = Path(path)
        if not path.exists():
            raise OSError(f"Model file not found: {path}")

        # Prophet deserialization
        if hasattr(Prophet, "load"):
            return Prophet.load(str(path))

        raise NotImplementedError("Prophet.load() not available")

    @staticmethod
    def save_neuralprophet_model(model, path: Union[str, Path]) -> None:
        """Save NeuralProphet model using torch checkpoint.

        Args:
            model: NeuralProphet model object
            path: File path to save model

        Raises:
            IOError: If file cannot be written
        """
        from neuralprophet import save

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # NeuralProphet save function
        save(model, str(path))

    @staticmethod
    def load_neuralprophet_model(path: Union[str, Path]) -> Any:
        """Load NeuralProphet model from torch checkpoint.

        Args:
            path: File path to load model

        Returns:
            Loaded NeuralProphet model

        Raises:
            IOError: If file cannot be read
        """
        from neuralprophet import load

        path = Path(path)
        if not path.exists():
            raise OSError(f"Model file not found: {path}")

        return load(str(path))

    @staticmethod
    def save_metadata(metadata: dict, path: Union[str, Path]) -> None:
        """Save metadata to JSON file.

        Args:
            metadata: Metadata dictionary
            path: File path to save metadata

        Raises:
            IOError: If file cannot be written
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert timestamps to ISO format strings for JSON serialization
        metadata_serializable = {}
        for key, value in metadata.items():
            if isinstance(value, pd.Timestamp):
                metadata_serializable[key] = value.isoformat()
            else:
                metadata_serializable[key] = value

        with open(path, "w") as f:
            json.dump(metadata_serializable, f, indent=2)

    @staticmethod
    def load_metadata(path: Union[str, Path]) -> dict[Any, Any]:
        """Load metadata from JSON file.

        Args:
            path: File path to load metadata

        Returns:
            Metadata dictionary

        Raises:
            IOError: If file cannot be read
        """
        path = Path(path)
        if not path.exists():
            raise OSError(f"Metadata file not found: {path}")

        with open(path) as f:
            return cast(dict[Any, Any], json.load(f))
