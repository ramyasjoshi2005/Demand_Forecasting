import pandas as pd
import sys
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
import os

def load_data(filepath=RAW_DATA_PATH):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} does not exist. Please download the dataset.")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    
    # Data Validation
    required_columns = {'date', 'store', 'item', 'sales'}
    if not required_columns.issubset(df.columns):
        print(f"Error: Missing required columns. Expected {required_columns}.")
        sys.exit(1)
        
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)
        
    if not pd.api.types.is_numeric_dtype(df['sales']):
        print("Error: 'sales' column must be numeric.")
        sys.exit(1)
        
    missing_vals = df.isnull().sum().sum()
    if missing_vals > 0:
        print(f"Warning: Found {missing_vals} missing values.")
        
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        print(f"Warning: Found {dup_rows} duplicate rows. Dropping them.")
        df = df.drop_duplicates()
        
    num_stores = df['store'].nunique()
    num_items = df['item'].nunique()
    num_records = len(df)
    min_date = df['date'].min().strftime('%Y-%m-%d')
    max_date = df['date'].max().strftime('%Y-%m-%d')
    
    print("--- DATASET SUMMARY ---")
    print(f"Records: {num_records:,}")
    print(f"Stores: {num_stores}")
    print(f"Items: {num_items}")
    print(f"Date Range: {min_date} to {max_date}")
    print("-----------------------")
    
    return df

def basic_preprocessing(df):
    """
    Sort by date, store, item to ensure time series integrity.
    """
    df = df.sort_values(by=['store', 'item', 'date']).reset_index(drop=True)
    return df

def save_processed_data(df, filepath=PROCESSED_DATA_PATH):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved processed data to {filepath}")

if __name__ == "__main__":
    df = load_data()
    df = basic_preprocessing(df)
    save_processed_data(df)
