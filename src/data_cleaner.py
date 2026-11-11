import os
import pandas as pd
import numpy as np

class DoorDashDataCleaner:
    def __init__(self, raw_path: str = 'data/historical_data.csv', output_path: str = 'data/cleaned_historical_data.csv'):
        self.raw_path = raw_path
        self.output_path = output_path

    def load_data(self) -> pd.DataFrame:
        print(f"[1/6] Loading raw data from: {self.raw_path}")
        df = pd.read_csv(self.raw_path)
        print(f"      Initial shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def clean_target_and_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[2/6] Cleaning target variable and timestamps...")
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

        # 1. Drop rows with missing delivery time
        initial_len = len(df)
        df = df.dropna(subset=['actual_delivery_time']).copy()
        dropped_null_target = initial_len - len(df)
        print(f"      Dropped null actual_delivery_time: {dropped_null_target} rows")

        # 2. Drop 2014 corrupt timestamp
        pre_2014_len = len(df)
        df = df[df['created_at'].dt.year >= 2015].copy()
        dropped_2014 = pre_2014_len - len(df)
        print(f"      Dropped corrupt 2014 timestamps: {dropped_2014} rows")

        # 3. Compute target duration in seconds
        df['delivery_duration_sec'] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds()

        # 4. Filter duration to realistic operational bounds: [600s (10m), 7200s (120m)]
        pre_filter_len = len(df)
        df = df[(df['delivery_duration_sec'] >= 600) & (df['delivery_duration_sec'] <= 7200)].copy()
        dropped_outliers = pre_filter_len - len(df)
        print(f"      Filtered target to [600s, 7200s]: dropped {dropped_outliers} outlier rows")
        print(f"      Rows remaining: {len(df):,}")
        return df

    def handle_prices_and_negatives(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[3/6] Handling prices, negative values, and volume bounds...")
        # 1. Filter out subtotal == 0
        pre_sub_len = len(df)
        df = df[df['subtotal'] > 0].copy()
        dropped_zero_sub = pre_sub_len - len(df)
        print(f"      Dropped subtotal == 0: {dropped_zero_sub} rows")

        # 2. Clip negative values to 0
        clip_lower_zero = [
            'min_item_price',
            'total_onshift_dashers',
            'total_busy_dashers',
            'total_outstanding_orders'
        ]
        for col in clip_lower_zero:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"      Clipped {neg_count} negative values in '{col}' to 0")
                df[col] = df[col].clip(lower=0)

        # 3. Clip extreme order bulkiness
        items_clipped = (df['total_items'] > 50).sum()
        if items_clipped > 0:
            print(f"      Clipped {items_clipped} extreme total_items values at 50")
            df['total_items'] = df['total_items'].clip(upper=50)

        return df

    def impute_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[4/6] Imputing missing values with domain-aware strategies...")

        # 1. Flag missing dasher telemetry before imputation
        df['dashers_missing_flag'] = df['total_onshift_dashers'].isnull().astype(np.int32)
        print(f"      Added 'dashers_missing_flag' (missing count: {df['dashers_missing_flag'].sum():,})")

        # 2. Recover store_primary_category via store_id lookup
        known_store_cats = (
            df.dropna(subset=['store_primary_category'])
            .groupby('store_id')['store_primary_category']
            .agg(lambda x: x.mode().iloc[0] if not x.empty else 'Unknown')
        )
        pre_missing_cats = df['store_primary_category'].isnull().sum()
        df['store_primary_category'] = df['store_primary_category'].fillna(df['store_id'].map(known_store_cats))
        recovered_cats = pre_missing_cats - df['store_primary_category'].isnull().sum()
        print(f"      Recovered {recovered_cats:,} store categories via store_id lookup ({recovered_cats/pre_missing_cats*100:.1f}%)")
        df['store_primary_category'] = df['store_primary_category'].fillna('Unknown')

        # 3. Recover market_id via store_id lookup
        known_store_mkts = (
            df.dropna(subset=['market_id'])
            .groupby('store_id')['market_id']
            .agg(lambda x: x.mode().iloc[0] if not x.empty else 2.0)
        )
        pre_missing_mkts = df['market_id'].isnull().sum()
        df['market_id'] = df['market_id'].fillna(df['store_id'].map(known_store_mkts))
        recovered_mkts = pre_missing_mkts - df['market_id'].isnull().sum()
        print(f"      Recovered {recovered_mkts:,} market_id via store_id lookup ({recovered_mkts/pre_missing_mkts*100:.1f}%)")
        overall_mkt_mode = df['market_id'].mode().iloc[0]
        df['market_id'] = df['market_id'].fillna(overall_mkt_mode)

        # 4. Impute order_protocol with mode
        protocol_mode = df['order_protocol'].mode().iloc[0]
        missing_proto = df['order_protocol'].isnull().sum()
        df['order_protocol'] = df['order_protocol'].fillna(protocol_mode)
        print(f"      Imputed {missing_proto:,} missing order_protocol with mode ({protocol_mode})")

        # 5. Impute estimated driving duration with market median
        market_drive_median = df.groupby('market_id')['estimated_store_to_consumer_driving_duration'].transform('median')
        missing_drive = df['estimated_store_to_consumer_driving_duration'].isnull().sum()
        df['estimated_store_to_consumer_driving_duration'] = df['estimated_store_to_consumer_driving_duration'].fillna(market_drive_median)
        # Fallback to overall median if market median has null
        df['estimated_store_to_consumer_driving_duration'] = df['estimated_store_to_consumer_driving_duration'].fillna(df['estimated_store_to_consumer_driving_duration'].median())
        print(f"      Imputed {missing_drive:,} missing driving durations with market median")

        # 6. Impute dasher telemetry grouped by (market_id, hour_of_day)
        df['hour_of_day'] = df['created_at'].dt.hour
        dasher_cols = ['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders']
        
        for col in dasher_cols:
            grouped_median = df.groupby(['market_id', 'hour_of_day'])[col].transform('median')
            df[col] = df[col].fillna(grouped_median)
            # Fallback to market median
            df[col] = df[col].fillna(df.groupby('market_id')[col].transform('median'))
            # Global fallback
            df[col] = df[col].fillna(df[col].median())

        print("      Imputed dasher metrics by (market_id, hour_of_day) median successfully")
        return df

    def standardize_and_optimize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[5/6] Standardizing currency to dollars and optimizing datatypes...")
        # Convert cents to dollars
        currency_cols = ['subtotal', 'min_item_price', 'max_item_price']
        for col in currency_cols:
            df[col] = (df[col] / 100.0).round(2)

        # Optimize datatypes
        df['market_id'] = df['market_id'].astype(np.int32)
        df['order_protocol'] = df['order_protocol'].astype(np.int32)
        df['store_id'] = df['store_id'].astype(np.int32)
        df['total_items'] = df['total_items'].astype(np.int32)
        df['num_distinct_items'] = df['num_distinct_items'].astype(np.int32)
        df['estimated_order_place_duration'] = df['estimated_order_place_duration'].astype(np.int32)

        float_cols = [
            'subtotal', 'min_item_price', 'max_item_price',
            'total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders',
            'estimated_store_to_consumer_driving_duration', 'delivery_duration_sec'
        ]
        for col in float_cols:
            df[col] = df[col].astype(np.float32)

        df['store_primary_category'] = df['store_primary_category'].astype(str)

        return df

    def run_pipeline(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.clean_target_and_dates(df)
        df = self.handle_prices_and_negatives(df)
        df = self.impute_missing_features(df)
        df = self.standardize_and_optimize_types(df)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        print(f"[6/6] Saving cleaned data to: {self.output_path}")
        df.to_csv(self.output_path, index=False)
        print(f"      Saved successfully! Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"      Retained {(len(df) / 197428) * 100:.2f}% of raw dataset.")
        return df

if __name__ == '__main__':
    cleaner = DoorDashDataCleaner()
    cleaner.run_pipeline()
