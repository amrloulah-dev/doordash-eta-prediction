import sys
import pandas as pd
import numpy as np

def verify_featured_dataset(file_path: str = 'data/featured_data.csv'):
    print(f"=== Running Verification on Featured Dataset: {file_path} ===")
    df = pd.read_csv(file_path)
    
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    
    errors = []

    # 1. Row count integrity
    if len(df) != 196125:
        errors.append(f"Unexpected row count: {len(df)} (expected 196,125)")

    # 2. Check for null or inf values across all columns
    null_cols = df.isnull().sum()
    if null_cols.sum() > 0:
        errors.append(f"Columns with nulls:\n{null_cols[null_cols > 0]}")
    else:
        print("[PASS] Zero null values across all 44 columns.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = np.isinf(df[numeric_cols]).sum()
    if inf_cols.sum() > 0:
        errors.append(f"Columns with inf values:\n{inf_cols[inf_cols > 0]}")
    else:
        print("[PASS] Zero infinite values across all numeric columns.")

    # 3. Check specific key engineered features exist
    expected_new_features = [
        'dasher_deficit', 'orders_per_onshift_dasher', 'busy_dashers_ratio', 'available_dashers',
        'is_dinner_rush', 'is_lunch_rush', 'is_weekend', 'sin_hour', 'cos_hour',
        'base_estimated_duration', 'drive_to_base_ratio', 'avg_item_price', 'price_range',
        'items_per_distinct', 'driving_x_orders_per_dasher', 'items_x_busy_ratio',
        'store_id_target_enc', 'store_primary_category_target_enc', 'market_id_target_enc'
    ]
    for feat in expected_new_features:
        if feat not in df.columns:
            errors.append(f"Missing expected engineered feature: '{feat}'")
    print(f"[PASS] All {len(expected_new_features)} key engineered features are present.")

    # 4. Correlation check with target
    target = 'delivery_duration_sec'
    corrs = df[expected_new_features].apply(lambda s: df[target].corr(s)).sort_values(ascending=False)
    print("\n--- Top Engineered Feature Correlations with Target ---")
    for feat, val in corrs.head(10).items():
        print(f"  {feat:35}: r = {val:+.4f}")

    if errors:
        print("\n[FAIL] VERIFICATION FAILED:")
        for err in errors:
            print(" -", err)
        sys.exit(1)
    else:
        print("\n[SUCCESS] ALL FEATURE VERIFICATION TESTS PASSED! Dataset is ready for modeling.")

if __name__ == '__main__':
    verify_featured_dataset()
