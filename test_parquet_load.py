"""Test script to verify validate_input works with parquet data."""

import pandas as pd
from rubin_oracle.utils import validate_input

# Load the parquet file
df = pd.read_parquet('./data/master_dataset.parquet')

print("Loaded parquet file:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Test validate_input
try:
    df_validated = validate_input(df)
    print("\n✓ validate_input succeeded!")
    print(f"Validated shape: {df_validated.shape}")
    print(f"Columns after validation: {df_validated.columns.tolist()}")
    print(f"\nFirst few validated rows:")
    print(df_validated.head())

    # Check if 'y' column exists now
    if 'y' in df_validated.columns:
        print(f"\n✓ 'y' column successfully created from 'tempMean'")
        print(f"y range: [{df_validated['y'].min():.2f}, {df_validated['y'].max():.2f}]")
        print(f"y missing: {df_validated['y'].isna().sum()} / {len(df_validated)}")

except Exception as e:
    print(f"\n✗ validate_input failed: {e}")
    import traceback
    traceback.print_exc()
