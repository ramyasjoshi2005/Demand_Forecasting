import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta
from config import RAW_DATA_PATH

def generate_data(output_path):
    print("SIMULATED DATA — FOR DEVELOPMENT ONLY")
    print("Generating synthetic dataset mapping the Kaggle challenge structure...")
    np.random.seed(42)
    
    start_date = pd.to_datetime('2013-01-01')
    end_date = pd.to_datetime('2017-12-31')
    dates = pd.date_range(start_date, end_date)
    
    stores = range(1, 11)
    items = range(1, 51)
    
    date_features = pd.DataFrame({'date': dates})
    date_features['dayofweek'] = date_features['date'].dt.dayofweek
    date_features['dayofyear'] = date_features['date'].dt.dayofyear
    date_features['year'] = date_features['date'].dt.year
    
    date_features['trend'] = (date_features['year'] - 2013) * 5
    weekly_seasonality = {0: 1.0, 1: 1.1, 2: 1.2, 3: 1.3, 4: 1.5, 5: 1.8, 6: 1.9}
    date_features['weekly'] = date_features['dayofweek'].map(weekly_seasonality)
    date_features['yearly'] = np.sin(2 * np.pi * date_features['dayofyear'] / 365.25 - np.pi/2) * 5 + 15
    date_features['date_effect'] = date_features['trend'] + date_features['yearly']
    
    store_base = np.random.uniform(0.8, 1.2, size=len(stores) + 1)
    item_base = np.random.uniform(5, 50, size=len(items) + 1)
    
    idx = pd.MultiIndex.from_product([dates, stores, items], names=['date', 'store', 'item'])
    df = pd.DataFrame(index=idx).reset_index()
    
    df = pd.merge(df, date_features[['date', 'weekly', 'date_effect']], on='date', how='left')
    df['store_effect'] = df['store'].map(lambda x: store_base[x])
    df['item_effect'] = df['item'].map(lambda x: item_base[x])
    
    noise = np.random.normal(0, 2, size=len(df))
    df['sales'] = (df['item_effect'] * df['store_effect'] * df['weekly']) + df['date_effect'] + noise
    df['sales'] = df['sales'].clip(lower=0).round().astype(int)
    
    df = df[['date', 'store', 'item', 'sales']]
    df.sort_values(by=['date', 'store', 'item'], inplace=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset generated at {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data management script")
    parser.add_argument('--simulate', action='store_true', help="Generate synthetic data for development.")
    args = parser.parse_args()
    
    if os.path.exists(RAW_DATA_PATH):
        print(f"Found real dataset at {RAW_DATA_PATH}.")
        if args.simulate:
            print("Warning: Real dataset already exists. Not overwriting with simulated data.")
    else:
        if args.simulate:
            generate_data(RAW_DATA_PATH)
        else:
            print(f"Error: Dataset not found at {RAW_DATA_PATH}.")
            print("Please download the Kaggle Store Item Demand Forecasting `train.csv` and place it in the data/raw/ directory.")
            print("Alternatively, use the --simulate flag to generate synthetic development data: python src/download_or_simulate_data.py --simulate")
            sys.exit(1)
