"""Quick verification script to test package installation and imports."""

import sys


def test_basic_imports():
    """Test basic package imports."""
    print("Testing basic imports...")
    try:
        from rubin_oracle import (
            Forecaster,
            BaseForecasterConfig,
            ProphetConfig,
            NeuralProphetConfig,
            ProphetForecaster,
        )
        print("✓ Basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Basic import failed: {e}")
        return False


def test_prophet_import():
    """Test Prophet-specific imports."""
    print("\nTesting Prophet imports...")
    try:
        from rubin_oracle import ProphetForecaster
        from rubin_oracle.config import ProphetConfig
        print("✓ Prophet imports successful")
        return True
    except ImportError as e:
        print(f"✗ Prophet import failed: {e}")
        return False


def test_neural_prophet_import():
    """Test NeuralProphet-specific imports."""
    print("\nTesting NeuralProphet imports...")
    try:
        from rubin_oracle import NeuralProphetForecaster
        from rubin_oracle.config import NeuralProphetConfig
        print("✓ NeuralProphet imports successful")
        return True
    except ImportError as e:
        print(f"⚠ NeuralProphet not available (optional): {e}")
        return None  # Optional dependency


def test_config_creation():
    """Test configuration creation."""
    print("\nTesting configuration creation...")
    try:
        from rubin_oracle.config import ProphetConfig, NeuralProphetConfig

        # Test ProphetConfig
        prophet_config = ProphetConfig(
            lag_days=48,
            n_forecast=24,
            daily_seasonality=True,
        )
        print(f"✓ ProphetConfig created: {prophet_config.name}")

        # Test NeuralProphetConfig
        neural_config = NeuralProphetConfig(
            lag_days=48,
            n_forecast=24,
            epochs=10,
        )
        print(f"✓ NeuralProphetConfig created: {neural_config.name}")

        return True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False


def test_yaml_configs():
    """Test loading YAML configurations."""
    print("\nTesting YAML config loading...")
    try:
        from pathlib import Path
        from rubin_oracle.config import ProphetConfig, NeuralProphetConfig

        # Test Prophet YAML
        prophet_yaml = Path("rubin_oracle/configs/prophet_default.yaml")
        if prophet_yaml.exists():
            prophet_config = ProphetConfig.from_yaml(prophet_yaml)
            print(f"✓ Prophet YAML loaded: {prophet_config.name}")
        else:
            print(f"⚠ Prophet YAML not found: {prophet_yaml}")

        # Test NeuralProphet YAML
        neural_yaml = Path("rubin_oracle/configs/neural_prophet_default.yaml")
        if neural_yaml.exists():
            neural_config = NeuralProphetConfig.from_yaml(neural_yaml)
            print(f"✓ NeuralProphet YAML loaded: {neural_config.name}")
        else:
            print(f"⚠ NeuralProphet YAML not found: {neural_yaml}")

        return True
    except Exception as e:
        print(f"✗ YAML loading failed: {e}")
        return False


def test_data_utils():
    """Test data utilities."""
    print("\nTesting data utilities...")
    try:
        import pandas as pd
        import numpy as np
        from rubin_oracle.utils import validate_input, prepare_regular_frequency

        # Create sample data
        df = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=100, freq='h'),
            'y': np.random.randn(100),
        })

        # Test validation
        validated = validate_input(df)
        print(f"✓ Data validation successful: {len(validated)} rows")

        # Test frequency preparation
        regular = prepare_regular_frequency(df, freq='h')
        print(f"✓ Frequency preparation successful: {len(regular)} rows")

        return True
    except Exception as e:
        print(f"✗ Data utils test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Rubin's Oracle - Installation Verification")
    print("=" * 60)

    results = []
    results.append(("Basic Imports", test_basic_imports()))
    results.append(("Prophet Import", test_prophet_import()))
    results.append(("NeuralProphet Import", test_neural_prophet_import()))
    results.append(("Config Creation", test_config_creation()))
    results.append(("YAML Loading", test_yaml_configs()))
    results.append(("Data Utils", test_data_utils()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    optional = sum(1 for _, result in results if result is None)

    for name, result in results:
        status = "✓ PASS" if result is True else "✗ FAIL" if result is False else "⚠ SKIP"
        print(f"{status:8} - {name}")

    print(f"\nTests: {passed} passed, {failed} failed, {optional} skipped")

    if failed == 0:
        print("\n🎉 All required tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
