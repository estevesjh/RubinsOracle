Rubin's Oracle Documentation
=============================

**Rubin's Oracle** is a unified weather forecasting package for Rubin Observatory using Prophet and NeuralProphet.

This package provides a clean, modern interface for time series forecasting with support for:

* Prophet-based forecasting
* NeuralProphet deep learning models
* Signal decomposition and preprocessing
* Ensemble methods
* Walk-forward validation

Quick Start
-----------

Installation::

    pip install rubin-oracle

Basic usage::

    from rubin_oracle import ProphetForecaster
    from rubin_oracle.config import ProphetConfig

    # Create forecaster
    config = ProphetConfig()
    forecaster = ProphetForecaster(config)

    # Fit and predict
    forecaster.fit(train_df)
    forecast_df = forecaster.predict(horizon_days=7)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/forecasters
   api/config
   api/preprocessing
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
