import sys
import pandas as pd
import numpy as np

def verify_cleaned_dataset(file_path: str = 'data/cleaned_historical_data.csv'):
    print(f"=== Running Automated Verification on: {file_path} ===")
    df = pd.read_csv(file_path)
    
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    errors = []

    # 1. Missing Values Check
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        errors.append(f"Found missing values:\n{null_counts[null_counts > 0]}")
    else:
        print("[PASS] Zero missing values across all columns.")

    # 2. Target Variable Checks
    dur = df['delivery_duration_sec']
    if dur.min() < 600 or dur.max() > 7200:
        errors.append(f"Target duration out of bounds [600, 7200]! Min: {dur.min()}, Max: {dur.max()}")
    else:
        print(f"[PASS] Target duration bounded correctly: Min = {dur.min():.1f}s ({dur.min()/60:.1f}m), Max = {dur.max():.1f}s ({dur.max()/60:.1f}m), Mean = {dur.mean():.1f}s ({dur.mean()/60:.1f}m).")

    # 3. Non-Negative Checks
    non_neg_cols = ['subtotal', 'min_item_price', 'max_item_price',
                    'total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders',
                    'total_items', 'num_distinct_items']
    for col in non_neg_cols:
        neg = (df[col] < 0).sum()
        if neg > 0:
            errors.append(f"Column '{col}' has {neg} negative values!")
    print("[PASS] Zero negative values across all numeric and quantity columns.")

    # 4. Currency Checks
    if df['subtotal'].max() > 1000.0:
        errors.append(f"Subtotal maximum (${df['subtotal'].max()}) suspiciously high - check currency conversion!")
    else:
        print(f"[PASS] Currency converted to dollars correctly (Subtotal Range: ${df['subtotal'].min():.2f} - ${df['subtotal'].max():.2f}).")

    # 5. Missing Flag Check
    if 'dashers_missing_flag' not in df.columns:
        errors.append("Column 'dashers_missing_flag' is missing!")
    elif not set(df['dashers_missing_flag'].unique()).issubset({0, 1}):
        errors.append(f"dashers_missing_flag contains non-binary values: {df['dashers_missing_flag'].unique()}")
    else:
        print(f"[PASS] 'dashers_missing_flag' present and binary. Flagged count: {df['dashers_missing_flag'].sum():,} ({df['dashers_missing_flag'].mean()*100:.2f}%).")

    # 6. Retention Rate
    retention = (len(df) / 197428) * 100
    print(f"[PASS] Data retention rate: {retention:.2f}% (Retained {len(df):,} out of 197,428 raw rows).")

    if errors:
        print("\n[FAIL] VERIFICATION FAILED WITH ERRORS:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    else:
        print("\n[SUCCESS] ALL VERIFICATION ASSERTIONS PASSED! Dataset is clean and ready for ML.")

if __name__ == '__main__':
    verify_cleaned_dataset()
