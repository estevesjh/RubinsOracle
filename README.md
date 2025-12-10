# Rubin's Oracle: Unified Weather Forecasting

A Python package providing a unified interface for time series forecasting using Prophet and NeuralProphet models, designed for weather forecasting at Cerro Pachon (Rubin Observatory).

This repository contains the implementation for the paper *"Forecasting Sunset Temperatures for Dome Thermal Management at the Vera C. Rubin Observatory"* along with LaTeX source, analysis code, and supporting data products.

## Features

- **Unified API**: Consistent interface across Prophet and NeuralProphet models
- **Type-Safe Configuration**: Pydantic-based configs with YAML support
- **Standardized Output**: All models return the same forecast schema
- **Easy Persistence**: Save/load trained models with configurations
- **Data Utilities**: Built-in validation and preprocessing helpers

## Installation

### Basic Installation (Prophet only)

```bash
pip install -e .
```

### With NeuralProphet Support

```bash
pip install -e ".[neural]"
```

### Development Installation

```bash
pip install -e ".[all]"
```

## Quick Start

### Using Prophet

```python
from rubin_oracle import ProphetForecaster
from rubin_oracle.config import ProphetConfig
import pandas as pd

# Create configuration
config = ProphetConfig(
    lag_days=48,
    n_forecast=24,
    daily_seasonality=True,
)

# Or load from YAML
config = ProphetConfig.from_yaml("rubin_oracle/configs/prophet_default.yaml")

# Prepare data (must have 'ds' and 'y' columns)
df = pd.DataFrame({
    'ds': pd.date_range('2024-01-01', periods=1000, freq='h'),
    'y': your_temperature_data
})

# Fit and predict
forecaster = ProphetForecaster(config)
forecaster.fit(df)
predictions = forecaster.predict(periods=24)

# Get standardized output
standardized = forecaster.standardize_output(predictions)
print(standardized.head())
#          ds      yhat  yhat_lower  yhat_upper  step
# 0  2024-...  15.23        14.85        15.61     1
# 1  2024-...  15.45        15.05        15.85     2
```

### Using NeuralProphet

```python
from rubin_oracle import NeuralProphetForecaster
from rubin_oracle.config import NeuralProphetConfig

# Create configuration
config = NeuralProphetConfig(
    lag_days=48,
    n_forecast=24,
    epochs=20,
    ar_layers=[64, 32],  # Deep AR network
)

# Fit model
forecaster = NeuralProphetForecaster(config)
forecaster.fit(train_df)

# Predict (NeuralProphet needs recent data for AR)
predictions = forecaster.predict(recent_df)  # Must have ≥48 hours
standardized = forecaster.standardize_output(predictions)
```

## Configuration

All configuration is done through Pydantic models with YAML support.

### Shared Parameters (BaseForecasterConfig)

```yaml
lag_days: 48              # Historical window (Prophet: training, NeuralProphet: n_lags)
n_forecast: 48            # Forecast horizon (NeuralProphet: n_forecasts)

# Seasonality
yearly_seasonality: false
weekly_seasonality: false
daily_seasonality: true
seasonality_mode: additive  # or multiplicative

# Trend
growth: linear            # linear, logistic, or flat
n_changepoints: 12
changepoints_range: 0.85
```

### Prophet-Specific Parameters

```yaml
changepoint_prior_scale: 0.05    # Trend flexibility
seasonality_prior_scale: 10.0    # Seasonality strength
holidays_prior_scale: 0.0        # Holiday effects
interval_width: 0.68             # Uncertainty intervals (~1 sigma)
```

### NeuralProphet-Specific Parameters

```yaml
# AR configuration
ar_layers: [64, 32]      # Hidden layers (empty = linear AR)
ar_reg: 1.0              # AR regularization

# Training
epochs: 15
batch_size: 128
learning_rate: 0.003
loss_func: SmoothL1Loss
optimizer: AdamW

# Uncertainty
quantiles: [0.16, 0.84]  # ~1 sigma

# Data handling
drop_missing: true
impute_missing: true
```

## Standardized Output Schema

All forecasters produce identical output after calling `standardize_output()`:

| Column | Type | Description |
|--------|------|-------------|
| `ds` | datetime | Target timestamp (what the forecast is FOR) |
| `yhat` | float | Point forecast |
| `yhat_lower` | float | Lower uncertainty bound |
| `yhat_upper` | float | Upper uncertainty bound |
| `step` | int | Forecast horizon (1, 2, 3, ...) |

## Saving and Loading Models

```python
# Save
forecaster.save("models/my_model")
# Creates: models/my_model/model.json (or model/ for NeuralProphet)
#          models/my_model/config.yaml

# Load
loaded = ProphetForecaster.load("models/my_model")
predictions = loaded.predict(periods=24)
```

## Package Structure

```
rubin_oracle/
├── __init__.py          # Public API
├── base.py              # Forecaster Protocol
├── config.py            # Pydantic configurations
├── models/
│   ├── prophet.py       # ProphetForecaster
│   └── neural_prophet.py # NeuralProphetForecaster
├── utils/
│   └── data.py          # Data validation & preprocessing
└── configs/             # Example YAML configs
    ├── prophet_default.yaml
    ├── neural_prophet_default.yaml
    └── neural_prophet_deep.yaml
```

## Examples

See [examples/basic_usage.py](examples/basic_usage.py) for complete examples including:
- Basic Prophet usage
- NeuralProphet with autoregressive components
- Comparing forecasters
- Model persistence

Run the examples:

```bash
python examples/basic_usage.py
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black rubin_oracle/
ruff check rubin_oracle/
```

### Type Checking

```bash
mypy rubin_oracle/
```

## Requirements

- Python ≥ 3.9
- pandas ≥ 2.0
- pydantic ≥ 2.0
- pyyaml ≥ 6.0
- prophet ≥ 1.1
- neuralprophet ≥ 0.7 (optional)
- torch ≥ 1.13 (optional, for NeuralProphet)

## Citation

If you use this package in your research, please cite:

```bibtex
@article{rubin_weather_forecast,
  title={Forecasting Sunset Temperatures for Dome Thermal Management at the Vera C. Rubin Observatory},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Links

- [Rubin Observatory](https://rubinobservatory.org/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [NeuralProphet Documentation](https://neuralprophet.com/)

