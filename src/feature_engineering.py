import pandas as pd
import numpy as np

def create_calendar_features(df):
    """
    Create calendar/date related features.
    """
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['quarter'] = df['date'].dt.quarter
    return df

def create_lag_features(df, target='sales'):
    """
    Create lag features grouped independently for each store x item time series.
    """
    df = df.copy()
    group = df.groupby(['store', 'item'])[target]
    
    df['lag_1'] = group.shift(1)
    df['lag_7'] = group.shift(7)
    df['lag_14'] = group.shift(14)
    df['lag_28'] = group.shift(28)
    
    # Optional lag 365 if we have sufficient history (e.g. 5 years)
    if len(df['date'].unique()) > 365:
        df['lag_365'] = group.shift(365)
        
    return df

def create_rolling_features(df, target='sales'):
    """
    Create rolling mean and std features independently for each store x item.
    Must shift by 1 first to avoid target leakage!
    Uses the pattern: sales.shift(1).rolling(...)
    """
    df = df.copy()
    # Shift target by 1 to ensure rolling features only use historical data
    shifted = df.groupby(['store', 'item'])[target].shift(1)
    
    # We apply rolling on the shifted series, grouped by store and item
    df['rolling_mean_7'] = shifted.groupby([df['store'], df['item']]).transform(lambda x: x.rolling(7).mean())
    df['rolling_mean_14'] = shifted.groupby([df['store'], df['item']]).transform(lambda x: x.rolling(14).mean())
    df['rolling_mean_28'] = shifted.groupby([df['store'], df['item']]).transform(lambda x: x.rolling(28).mean())
    
    df['rolling_std_7'] = shifted.groupby([df['store'], df['item']]).transform(lambda x: x.rolling(7).std())
    df['rolling_std_28'] = shifted.groupby([df['store'], df['item']]).transform(lambda x: x.rolling(28).std())
    
    return df

def engineer_features(df):
    """
    Main function to run all feature engineering steps.
    """
    print("Creating calendar features...")
    df = create_calendar_features(df)
    print("Creating lag features...")
    df = create_lag_features(df)
    print("Creating rolling features...")
    df = create_rolling_features(df)
    return df

if __name__ == "__main__":
    from data_preprocessing import load_data, basic_preprocessing, save_processed_data
    df = load_data()
    df = basic_preprocessing(df)
    df = engineer_features(df)
    save_processed_data(df)
