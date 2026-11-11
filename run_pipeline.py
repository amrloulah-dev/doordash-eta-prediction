"""
DoorDash Delivery ETA Prediction - Master Pipeline Orchestrator

Usage:
    python run_pipeline.py --step all       # Run end-to-end (clean -> features -> train -> verify -> demo)
    python run_pipeline.py --step clean     # Run data cleaning
    python run_pipeline.py --step features  # Run feature engineering
    python run_pipeline.py --step train     # Run model tournament & tuning
    python run_pipeline.py --step verify    # Run automated sanity assertions
    python run_pipeline.py --step demo      # Run sample order inference demo
"""

import os
import sys
import argparse
import time

def step_clean():
    print("\n" + "="*70)
    print("STEP 1: DATA CLEANING & PREPROCESSING")
    print("="*70)
    from src.data_cleaner import DoorDashDataCleaner
    from src.verify_cleaning import verify_cleaned_dataset
    cleaner = DoorDashDataCleaner()
    cleaner.run_pipeline()
    verify_cleaned_dataset()

def step_features():
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING & TARGET ENCODING")
    print("="*70)
    from src.feature_engineer import DoorDashFeatureEngineer
    from src.verify_features import verify_featured_dataset
    engineer = DoorDashFeatureEngineer()
    engineer.run_pipeline()
    verify_featured_dataset()

def step_train():
    print("\n" + "="*70)
    print("STEP 3: MODEL TOURNAMENT & TUNING")
    print("="*70)
    from src.train_models import DoorDashModelTrainer
    trainer = DoorDashModelTrainer()
    trainer.run_all()

def step_verify():
    print("\n" + "="*70)
    print("RUNNING AUTOMATED VERIFICATION SUITE")
    print("="*70)
    from src.verify_cleaning import verify_cleaned_dataset
    from src.verify_features import verify_featured_dataset
    verify_cleaned_dataset()
    verify_featured_dataset()
    print("\n[SUCCESS] All verification tests passed cleanly!")

def step_demo():
    print("\n" + "="*70)
    print("RUNNING INTERACTIVE PREDICTION DEMO")
    print("="*70)
    from predict import run_sample_demo
    run_sample_demo()

def main():
    parser = argparse.ArgumentParser(description="DoorDash ETA Machine Learning Pipeline")
    parser.add_argument('--step', choices=['all', 'clean', 'features', 'train', 'verify', 'demo'],
                        default='all', help="Pipeline step to execute")
    args = parser.parse_args()

    start_time = time.time()
    print(f"Starting DoorDash ML Pipeline with step: '{args.step}'...")

    if args.step == 'clean':
        step_clean()
    elif args.step == 'features':
        step_features()
    elif args.step == 'train':
        step_train()
    elif args.step == 'verify':
        step_verify()
    elif args.step == 'demo':
        step_demo()
    elif args.step == 'all':
        step_clean()
        step_features()
        step_train()
        step_verify()
        step_demo()

    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"PIPELINE EXECUTION COMPLETED IN {total_time:.2f} SECONDS")
    print("="*70)

if __name__ == '__main__':
    main()
