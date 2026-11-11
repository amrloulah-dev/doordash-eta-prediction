import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class DoorDashFeatureEngineer:
    """
    Feature Engineering Pipeline for DoorDash Delivery ETA Prediction.
    Computes domain-specific ratios, temporal encodings, order complexity indices,
    interactions, and leak-free Out-Of-Fold (OOF) smoothed target encodings.
    """
    def __init__(self,
                 input_path: str = 'data/cleaned_historical_data.csv',
                 output_path: str = 'data/featured_data.csv',
                 target_col: str = 'delivery_duration_sec'):
        self.input_path = input_path
        self.output_path = output_path
        self.target_col = target_col
        self.global_encodings = {}

    def load_data(self) -> pd.DataFrame:
        print(f"[1/6] Loading cleaned dataset from: {self.input_path}")
        df = pd.read_csv(self.input_path)
        df['created_at'] = pd.to_datetime(df['created_at'])
        print(f"      Rows: {len(df):,}, Columns: {len(df.columns)}")
        return df

    def engineer_supply_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[2/6] Engineering Dasher supply, demand, and congestion features...")
        # 1. Dasher deficit (orders - drivers)
        df['dasher_deficit'] = df['total_outstanding_orders'] - df['total_onshift_dashers']

        # 2. Workload per on-shift driver
        df['orders_per_onshift_dasher'] = df['total_outstanding_orders'] / (df['total_onshift_dashers'] + 1.0)

        # 3. Fleet utilization rate
        df['busy_dashers_ratio'] = df['total_busy_dashers'] / (df['total_onshift_dashers'] + 1.0)

        # 4. Available/free drivers buffer
        df['available_dashers'] = (df['total_onshift_dashers'] - df['total_busy_dashers']).clip(lower=0.0)

        # 5. Congestion pressure per busy dasher
        df['orders_per_busy_dasher'] = df['total_outstanding_orders'] / (df['total_busy_dashers'] + 1.0)

        return df

    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[3/6] Engineering calendar, rush-hour, and cyclical temporal features...")
        dt = df['created_at'].dt
        df['hour_of_day'] = dt.hour
        df['day_of_week'] = dt.dayofweek
        df['day_of_month'] = dt.day

        # Rush hours
        # Hours 1, 2, 3 UTC correspond to US dinner peak (5 PM - 8 PM local time)
        df['is_dinner_rush'] = df['hour_of_day'].isin([1, 2, 3]).astype(np.int32)
        # Hours 18, 19, 20 UTC correspond to US lunch peak (11 AM - 2 PM local time)
        df['is_lunch_rush'] = df['hour_of_day'].isin([18, 19, 20]).astype(np.int32)
        # Weekends (Saturday=5, Sunday=6)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(np.int32)

        # Cyclical continuous transformation of hours (0 to 23)
        df['sin_hour'] = np.sin(2.0 * np.pi * df['hour_of_day'] / 24.0).astype(np.float32)
        df['cos_hour'] = np.cos(2.0 * np.pi * df['hour_of_day'] / 24.0).astype(np.float32)

        # Cyclical day of week (0 to 6)
        df['sin_dayofweek'] = np.sin(2.0 * np.pi * df['day_of_week'] / 7.0).astype(np.float32)
        df['cos_dayofweek'] = np.cos(2.0 * np.pi * df['day_of_week'] / 7.0).astype(np.float32)

        return df

    def engineer_order_complexity_and_logistics(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[4/6] Engineering order complexity, price structure, and baseline travel times...")
        # 1. Base duration estimate (place + driving)
        df['base_estimated_duration'] = df['estimated_order_place_duration'] + df['estimated_store_to_consumer_driving_duration']

        # 2. Ratio of driving time to base duration
        df['drive_to_base_ratio'] = df['estimated_store_to_consumer_driving_duration'] / (df['base_estimated_duration'] + 1e-5)

        # 3. Average price per item (fine dining vs budget fast food)
        df['avg_item_price'] = df['subtotal'] / (df['total_items'] + 1e-5)

        # 4. Item price dispersion within the order
        df['price_range'] = df['max_item_price'] - df['min_item_price']

        # 5. Ratio of total items to distinct items (dish batching)
        df['items_per_distinct'] = df['total_items'] / (df['num_distinct_items'] + 1e-5)

        # 6. Subtotal per distinct dish
        df['subtotal_per_distinct'] = df['subtotal'] / (df['num_distinct_items'] + 1e-5)

        # Cross Interactions
        # Bottleneck interaction: long driving distance coupled with driver shortage
        df['driving_x_orders_per_dasher'] = (
            df['estimated_store_to_consumer_driving_duration'] * df['orders_per_onshift_dasher']
        ).astype(np.float32)

        # Bulk order during high fleet utilization
        df['items_x_busy_ratio'] = (df['total_items'] * df['busy_dashers_ratio']).astype(np.float32)

        return df

    def apply_leak_free_target_encoding(self, df: pd.DataFrame, n_splits: int = 5, smoothing: float = 10.0) -> pd.DataFrame:
        """
        Computes Bayesian-smoothed Out-Of-Fold (OOF) target encoding for categorical columns
        to guarantee ZERO data leakage during model training.
        """
        print("[5/6] Computing leak-free Out-Of-Fold (OOF) smoothed target encodings...")
        target = df[self.target_col]
        global_mean = target.mean()
        print(f"      Global Target Mean: {global_mean:.2f} seconds ({global_mean/60:.2f} minutes)")

        target_cols = ['store_id', 'store_primary_category', 'market_id']
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for col in target_cols:
            oof_col_name = f"{col}_target_enc"
            df[oof_col_name] = np.nan

            # 1. K-Fold Out-of-fold encoding for training rows
            for train_idx, val_idx in kf.split(df):
                train_subset = df.iloc[train_idx]
                val_subset = df.iloc[val_idx]

                # Compute statistics on training fold ONLY
                stats = train_subset.groupby(col)[self.target_col].agg(['count', 'mean'])
                counts = stats['count']
                means = stats['mean']
                
                # Bayesian smoothed mean
                smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
                df.iloc[val_idx, df.columns.get_loc(oof_col_name)] = val_subset[col].map(smoothed)

            # Fill any unseen categories in validation folds with global mean
            df[oof_col_name] = df[oof_col_name].fillna(global_mean).astype(np.float32)

            # 2. Also compute and save the full global mapping for future unseen test inference
            full_stats = df.groupby(col)[self.target_col].agg(['count', 'mean'])
            full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
            self.global_encodings[col] = {
                'global_mean': float(global_mean),
                'smoothing': float(smoothing),
                'mapping': {str(k): float(v) for k, v in full_smoothed.items()}
            }

            corr = df[[oof_col_name, self.target_col]].corr().iloc[0, 1]
            print(f"      Generated '{oof_col_name}': OOF Pearson correlation with target = {corr:+.3f}")

        # Save global encodings mappings to JSON for inference
        enc_path = 'data/target_encodings.json'
        os.makedirs(os.path.dirname(enc_path), exist_ok=True)
        with open(enc_path, 'w') as f:
            json.dump(self.global_encodings, f, indent=2)
        print(f"      Saved inference encodings mapping to: {enc_path}")

        return df

    def run_pipeline(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.engineer_supply_demand_features(df)
        df = self.engineer_temporal_features(df)
        df = self.engineer_order_complexity_and_logistics(df)
        df = self.apply_leak_free_target_encoding(df)

        # Optimize datatypes for ML
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype(np.float32)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        print(f"[6/6] Saving feature-engineered dataset to: {self.output_path}")
        df.to_csv(self.output_path, index=False)
        print(f"      Done! Total final columns: {len(df.columns)}")
        print(f"      Output size: {os.path.getsize(self.output_path) / (1024**2):.2f} MB")
        return df

if __name__ == '__main__':
    engineer = DoorDashFeatureEngineer()
    engineer.run_pipeline()
