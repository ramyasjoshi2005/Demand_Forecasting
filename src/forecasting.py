import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from config import PROCESSED_DATA_PATH, LIGHTGBM_MODEL_PATH
from feature_engineering import create_lag_features, create_rolling_features
import os

_model = None
_features = None
_df = None

def _load_resources():
    global _model, _features, _df
    if _model is None:
        if not os.path.exists(LIGHTGBM_MODEL_PATH):
            raise FileNotFoundError("LightGBM model not found. Please train first.")
        _model = lgb.Booster(model_file=LIGHTGBM_MODEL_PATH)
        
        with open('models/lgbm_features.json', 'r') as f:
            _features = json.load(f)
            
        _df = pd.read_csv(PROCESSED_DATA_PATH)
        _df['date'] = pd.to_datetime(_df['date'])

def get_historical_and_forecast(store_id, item_id, horizon, historical_days=30):
    _load_resources()
    
    # Filter for specific store/item
    store_item_df = _df[(_df['store'] == store_id) & (_df['item'] == item_id)].sort_values('date').copy()
    
    if store_item_df.empty:
        return None, None
        
    last_date = store_item_df['date'].max()
    historical_start = last_date - pd.Timedelta(days=historical_days)
    
    history_df = store_item_df[store_item_df['date'] > historical_start].copy()
    
    # Fully recursive forecasting loop
    forecasts = []
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    
    current_df = store_item_df.copy()
    
    for future_date in future_dates:
        # Create a new row with basic calendar features
        new_row = pd.DataFrame({
            'date': [future_date],
            'store': [store_id],
            'item': [item_id],
            'sales': [np.nan], # Target is unknown
            'year': [future_date.year],
            'month': [future_date.month],
            'day': [future_date.day],
            'day_of_week': [future_date.dayofweek],
            'week_of_year': [future_date.isocalendar().week],
            'day_of_year': [future_date.dayofyear],
            'is_weekend': [1 if future_date.dayofweek >= 5 else 0],
            'quarter': [future_date.quarter]
        })
        
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        # Recalculate lag and rolling features
        current_df = create_lag_features(current_df)
        current_df = create_rolling_features(current_df)
        
        # Get the row we need to predict
        predict_row = current_df.iloc[-1:]
        
        pred = _model.predict(predict_row[_features])[0]
        pred = max(0, round(pred))
        forecasts.append(pred)
        
        # Fill in the prediction as 'sales' so the next iteration can use it for lags
        current_df.loc[current_df.index[-1], 'sales'] = pred
        
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': forecasts
    })
    
    return history_df[['date', 'sales']], forecast_df
