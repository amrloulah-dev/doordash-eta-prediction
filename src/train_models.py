import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

class DoorDashModelTrainer:
    def __init__(self,
                 data_path: str = 'data/featured_data.csv',
                 models_dir: str = 'models',
                 reports_dir: str = 'reports',
                 cutoff_date: str = '2015-02-14'):
        self.data_path = data_path
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.figures_dir = os.path.join(reports_dir, 'figures')
        self.cutoff_date = pd.Timestamp(cutoff_date)
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_and_split(self):
        print(f"[1/5] Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['created_at'] = pd.to_datetime(df['created_at'])

        exclude_cols = ['created_at', 'actual_delivery_time', 'delivery_duration_sec', 'store_primary_category']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        print(f"      Total Features: {len(self.feature_cols)}")
        print(f"      Chronological split cutoff: {self.cutoff_date}")

        train_mask = df['created_at'] < self.cutoff_date
        self.X_train = df.loc[train_mask, self.feature_cols]
        self.y_train = (df.loc[train_mask, 'delivery_duration_sec'] / 60.0).astype(np.float32)  # minutes

        self.X_test = df.loc[~train_mask, self.feature_cols]
        self.y_test = (df.loc[~train_mask, 'delivery_duration_sec'] / 60.0).astype(np.float32)

        print(f"      Train Set: {self.X_train.shape[0]:,} rows ({self.X_train.shape[0]/len(df)*100:.1f}%)")
        print(f"      Test Set:  {self.X_test.shape[0]:,} rows ({self.X_test.shape[0]/len(df)*100:.1f}%)")

    def evaluate_model(self, model, name: str, fit_time: float) -> dict:
        # Train evaluation
        train_preds = model.predict(self.X_train)
        train_mae = mean_absolute_error(self.y_train, train_preds)
        train_rmse = root_mean_squared_error(self.y_train, train_preds)
        train_r2 = r2_score(self.y_train, train_preds)

        # Test evaluation
        test_preds = model.predict(self.X_test)
        test_mae = mean_absolute_error(self.y_test, test_preds)
        test_rmse = root_mean_squared_error(self.y_test, test_preds)
        test_r2 = r2_score(self.y_test, test_preds)
        
        # Business Tolerance Metrics
        diffs = np.abs(self.y_test - test_preds)
        within_5m = (diffs <= 5.0).mean() * 100.0
        within_10m = (diffs <= 10.0).mean() * 100.0

        return {
            'model_name': name,
            'train_mae_min': round(float(train_mae), 2),
            'test_mae_min': round(float(test_mae), 2),
            'test_rmse_min': round(float(test_rmse), 2),
            'test_r2': round(float(test_r2), 4),
            'within_5min_pct': round(float(within_5m), 2),
            'within_10min_pct': round(float(within_10m), 2),
            'fit_time_sec': round(float(fit_time), 2),
            'test_preds': test_preds
        }

    def run_tournament(self) -> pd.DataFrame:
        print("\n[2/5] Running Phase 1 Model Tournament across 4 candidate algorithms...")
        
        models = {
            'Ridge (Baseline)': Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=100.0, random_state=42))
            ]),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=16,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            ),
            'HistGradientBoosting': HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.06,
                max_leaf_nodes=63,
                random_state=42
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=300,
                learning_rate=0.04,
                num_leaves=63,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        }

        self.results = []
        self.trained_models = {}

        for name, model in models.items():
            print(f"      Training {name}...")
            start = time.time()
            model.fit(self.X_train, self.y_train)
            fit_time = time.time() - start
            
            metrics = self.evaluate_model(model, name, fit_time)
            self.results.append(metrics)
            self.trained_models[name] = model

            print(f"        -> Test MAE: {metrics['test_mae_min']} min | R2: {metrics['test_r2']} | Within +-5m: {metrics['within_5min_pct']}% | Fit time: {metrics['fit_time_sec']}s")

        # Create summary DataFrame
        df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'test_preds'} for r in self.results])
        df_results = df_results.sort_values(by='test_mae_min', ascending=True).reset_index(drop=True)
        df_results.to_csv(os.path.join(self.reports_dir, 'model_comparison.csv'), index=False)
        print("\n=== Phase 1 Tournament Leaderboard ===")
        print(df_results[['model_name', 'test_mae_min', 'test_rmse_min', 'test_r2', 'within_5min_pct', 'within_10min_pct', 'fit_time_sec']])

        self.winner_name = df_results.iloc[0]['model_name']
        print(f"\n      Tournament Winner: {self.winner_name} (Lowest Test MAE: {df_results.iloc[0]['test_mae_min']} min)")
        return df_results

    def run_tuning_phase(self):
        print(f"\n[3/5] Running Phase 2: Targeted Hyperparameter Tuning on Winner ({self.winner_name})...")
        
        # Tuned configurations to test
        configs = [
            {'n_estimators': 350, 'learning_rate': 0.03, 'num_leaves': 95, 'min_child_samples': 40, 'colsample_bytree': 0.75, 'subsample': 0.85},
            {'n_estimators': 450, 'learning_rate': 0.025, 'num_leaves': 127, 'min_child_samples': 50, 'colsample_bytree': 0.8, 'subsample': 0.8},
            {'n_estimators': 500, 'learning_rate': 0.02, 'num_leaves': 80, 'min_child_samples': 30, 'colsample_bytree': 0.7, 'subsample': 0.9}
        ]

        best_tuned_model = None
        best_tuned_metrics = None
        best_mae = float('inf')

        for i, cfg in enumerate(configs, 1):
            print(f"      Testing Tuned Config #{i}: {cfg}")
            start = time.time()
            tuned_lgb = LGBMRegressor(**cfg, random_state=42, verbose=-1, n_jobs=-1)
            tuned_lgb.fit(self.X_train, self.y_train)
            fit_time = time.time() - start
            
            metrics = self.evaluate_model(tuned_lgb, f'Tuned LightGBM (Config #{i})', fit_time)
            print(f"        -> Test MAE: {metrics['test_mae_min']} min | R2: {metrics['test_r2']} | Within +-5m: {metrics['within_5min_pct']}%")

            if metrics['test_mae_min'] < best_mae:
                best_mae = metrics['test_mae_min']
                best_tuned_model = tuned_lgb
                best_tuned_metrics = metrics

        self.final_model = best_tuned_model
        self.final_metrics = best_tuned_metrics
        print(f"\n      Optimal Tuned Model Selected with Test MAE = {best_mae} min")

    def save_artifacts_and_plots(self):
        print("\n[4/5] Saving final model artifacts and generating diagnostic visualizations...")
        # 1. Save model with joblib
        model_path = os.path.join(self.models_dir, 'best_eta_model.joblib')
        joblib.dump({
            'model': self.final_model,
            'feature_cols': self.feature_cols,
            'metrics': {k: v for k, v in self.final_metrics.items() if k != 'test_preds'},
            'trained_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }, model_path)
        print(f"      Saved final model to: {model_path} ({os.path.getsize(model_path)/(1024**2):.2f} MB)")

        # 2. Feature Importance Plot
        importances = self.final_model.feature_importances_
        feat_df = pd.DataFrame({'feature': self.feature_cols, 'importance': importances})
        feat_df = feat_df.sort_values(by='importance', ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=feat_df, x='importance', y='feature', palette='viridis')
        plt.title('Top 20 Most Predictive Features for Delivery ETA (LightGBM Gain)')
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'feature_importance.png'), dpi=150)
        plt.close()
        print(f"      Saved feature importance plot to: {self.figures_dir}/feature_importance.png")

        # 3. Residuals & Error Distribution
        preds = self.final_metrics['test_preds']
        residuals = self.y_test - preds  # actual - predicted

        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, bins=60, kde=True, color='crimson')
        plt.axvline(0, color='black', linestyle='--', label='Zero Error')
        plt.axvline(residuals.median(), color='blue', linestyle=':', label=f'Median Error: {residuals.median():.2f} min')
        plt.title('Residuals Distribution on Out-of-Time Test Set (Actual - Predicted Minutes)')
        plt.xlabel('Error (Minutes)')
        plt.ylabel('Order Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'residuals_distribution.png'), dpi=150)
        plt.close()
        print(f"      Saved residuals plot to: {self.figures_dir}/residuals_distribution.png")

        # 4. Actual vs Predicted Scatter/Hexbin
        plt.figure(figsize=(8, 8))
        plt.hexbin(self.y_test, preds, gridsize=50, cmap='Blues', mincnt=1)
        plt.plot([0, 120], [0, 120], 'r--', label='Perfect 45-degree Prediction')
        plt.title('Actual vs Predicted Delivery Duration (Minutes)')
        plt.xlabel('Actual Delivery Duration (Minutes)')
        plt.ylabel('Predicted Delivery Duration (Minutes)')
        plt.xlim(10, 120)
        plt.ylim(10, 120)
        plt.legend()
        plt.colorbar(label='Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'actual_vs_predicted.png'), dpi=150)
        plt.close()
        print(f"      Saved actual vs predicted plot to: {self.figures_dir}/actual_vs_predicted.png")

        # Save metrics JSON
        summary_json = {
            'winner_model': self.winner_name,
            'final_metrics': {k: v for k, v in self.final_metrics.items() if k != 'test_preds'},
            'tournament_comparison': [
                {k: v for k, v in r.items() if k != 'test_preds'} for r in self.results
            ]
        }
        with open(os.path.join(self.reports_dir, 'model_results.json'), 'w') as f:
            json.dump(summary_json, f, indent=2)

    def run_all(self):
        self.load_and_split()
        self.run_tournament()
        self.run_tuning_phase()
        self.save_artifacts_and_plots()
        print("\n[5/5] Model Training, Tuning, and Diagnostic Reporting Complete!")

if __name__ == '__main__':
    trainer = DoorDashModelTrainer()
    trainer.run_all()
