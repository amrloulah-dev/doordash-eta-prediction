import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class DoorDashPredictor:
    def __init__(self,
                 model_path: str = 'models/best_eta_model.joblib',
                 encodings_path: str = 'data/target_encodings.json'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first!")
        if not os.path.exists(encodings_path):
            raise FileNotFoundError(f"Encodings file not found at: {encodings_path}.")

        # Load trained bundle
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.feature_cols = bundle['feature_cols']
        self.metrics = bundle.get('metrics', {})

        # Load target encodings mappings
        with open(encodings_path, 'r') as f:
            self.encodings = json.load(f)

    def _get_target_encoding(self, category_type: str, key_val: any) -> float:
        enc_info = self.encodings.get(category_type, {})
        global_mean = enc_info.get('global_mean', 2826.15)
        mapping = enc_info.get('mapping', {})
        return float(mapping.get(str(key_val), global_mean))

    def preprocess_order(self, order: dict) -> pd.DataFrame:
        # Datetime handling
        created_at = order.get('created_at')
        if created_at is None:
            dt = datetime.now()
        elif isinstance(created_at, str):
            dt = pd.to_datetime(created_at)
        else:
            dt = created_at

        hour = dt.hour
        dayofweek = dt.weekday()
        day = dt.day

        # Raw values
        market_id = int(order.get('market_id', 2))
        store_id = int(order.get('store_id', 1))
        category = str(order.get('store_primary_category', 'american')).lower()
        order_protocol = int(order.get('order_protocol', 1))
        total_items = min(int(order.get('total_items', 3)), 50)
        subtotal = float(order.get('subtotal', 25.0))
        num_distinct_items = int(order.get('num_distinct_items', 2))
        min_item_price = max(float(order.get('min_item_price', 5.0)), 0.0)
        max_item_price = float(order.get('max_item_price', 15.0))
        
        onshift = max(float(order.get('total_onshift_dashers', 30.0)), 0.0)
        busy = max(float(order.get('total_busy_dashers', 25.0)), 0.0)
        outstanding = max(float(order.get('total_outstanding_orders', 35.0)), 0.0)
        
        place_duration = int(order.get('estimated_order_place_duration', 251))
        drive_duration = float(order.get('estimated_store_to_consumer_driving_duration', 550.0))
        missing_flag = int(order.get('dashers_missing_flag', 0))

        # Engineered features
        dasher_deficit = outstanding - onshift
        orders_per_onshift = outstanding / (onshift + 1.0)
        busy_ratio = busy / (onshift + 1.0)
        available = max(onshift - busy, 0.0)
        orders_per_busy = outstanding / (busy + 1.0)

        is_dinner = 1 if hour in [1, 2, 3] else 0
        is_lunch = 1 if hour in [18, 19, 20] else 0
        is_weekend = 1 if dayofweek in [5, 6] else 0

        sin_hour = np.sin(2.0 * np.pi * hour / 24.0)
        cos_hour = np.cos(2.0 * np.pi * hour / 24.0)
        sin_dow = np.sin(2.0 * np.pi * dayofweek / 7.0)
        cos_dow = np.cos(2.0 * np.pi * dayofweek / 7.0)

        base_est = place_duration + drive_duration
        drive_to_base = drive_duration / (base_est + 1e-5)
        avg_item_price = subtotal / (total_items + 1e-5)
        price_range = max_item_price - min_item_price
        items_per_distinct = total_items / (num_distinct_items + 1e-5)
        subtotal_per_distinct = subtotal / (num_distinct_items + 1e-5)

        driving_x_orders = drive_duration * orders_per_onshift
        items_x_busy = total_items * busy_ratio

        # Target encodings
        store_enc = self._get_target_encoding('store_id', store_id)
        cat_enc = self._get_target_encoding('store_primary_category', category)
        mkt_enc = self._get_target_encoding('market_id', market_id)

        features = {
            'market_id': market_id,
            'store_id': store_id,
            'order_protocol': order_protocol,
            'total_items': total_items,
            'subtotal': subtotal,
            'num_distinct_items': num_distinct_items,
            'min_item_price': min_item_price,
            'max_item_price': max_item_price,
            'total_onshift_dashers': onshift,
            'total_busy_dashers': busy,
            'total_outstanding_orders': outstanding,
            'estimated_order_place_duration': place_duration,
            'estimated_store_to_consumer_driving_duration': drive_duration,
            'dashers_missing_flag': missing_flag,
            'hour_of_day': hour,
            'dasher_deficit': dasher_deficit,
            'orders_per_onshift_dashers': orders_per_onshift,
            'busy_dashers_ratio': busy_ratio,
            'available_dashers': available,
            'orders_per_busy_dasher': orders_per_busy,
            'day_of_week': dayofweek,
            'day_of_month': day,
            'is_dinner_rush': is_dinner,
            'is_lunch_rush': is_lunch,
            'is_weekend': is_weekend,
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,
            'sin_dayofweek': sin_dow,
            'cos_dayofweek': cos_dow,
            'base_estimated_duration': base_est,
            'drive_to_base_ratio': drive_to_base,
            'avg_item_price': avg_item_price,
            'price_range': price_range,
            'items_per_distinct': items_per_distinct,
            'subtotal_per_distinct': subtotal_per_distinct,
            'driving_x_orders_per_dasher': driving_x_orders,
            'items_x_busy_ratio': items_x_busy,
            'store_id_target_enc': store_enc,
            'store_primary_category_target_enc': cat_enc,
            'market_id_target_enc': mkt_enc
        }

        # Order dataframe strictly by self.feature_cols
        # In case of minor naming discrepancy:
        if 'orders_per_onshift_dasher' in self.feature_cols and 'orders_per_onshift_dasher' not in features:
            features['orders_per_onshift_dasher'] = orders_per_onshift

        row_df = pd.DataFrame([features])
        return row_df[self.feature_cols]

    def predict(self, order: dict) -> dict:
        X = self.preprocess_order(order)
        pred_minutes = float(self.model.predict(X)[0])
        pred_seconds = int(round(pred_minutes * 60))

        min_range = max(round(pred_minutes - 5.0, 1), 10.0)
        max_range = round(pred_minutes + 5.0, 1)

        return {
            'predicted_duration_minutes': round(pred_minutes, 1),
            'predicted_duration_seconds': pred_seconds,
            'expected_window': f"{min_range} - {max_range} mins",
            'confidence_margin': "±5.0 mins"
        }

def run_sample_demo():
    print("=== DoorDash ETA Predictor Interactive Demo ===")
    predictor = DoorDashPredictor()

    sample_order = {
        'market_id': 2,
        'store_id': 3560,
        'store_primary_category': 'pizza',
        'order_protocol': 1,
        'total_items': 4,
        'subtotal': 38.50,
        'num_distinct_items': 3,
        'min_item_price': 6.00,
        'max_item_price': 18.00,
        'total_onshift_dashers': 45,
        'total_busy_dashers': 42,
        'total_outstanding_orders': 58,
        'estimated_order_place_duration': 251,
        'estimated_store_to_consumer_driving_duration': 650,
        'created_at': '2015-02-15 19:30:00'
    }

    print("\nSample Order Details:")
    for k, v in sample_order.items():
        print(f"  {k}: {v}")

    result = predictor.predict(sample_order)
    print("\n--- Model Prediction Result ---")
    print(f"  Predicted Delivery Time: {result['predicted_duration_minutes']} minutes ({result['predicted_duration_seconds']} seconds)")
    print(f"  Estimated Window:        {result['expected_window']}")
    print(f"  Confidence Margin:       {result['confidence_margin']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DoorDash ETA Inference Tool")
    parser.add_argument('--demo', action='store_true', default=True, help="Run a sample order prediction demo")
    args = parser.parse_args()

    if args.demo:
        run_sample_demo()
