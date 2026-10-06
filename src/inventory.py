import numpy as np
import pandas as pd
from config import PROCESSED_DATA_PATH, VALID_START, VALID_END, LIGHTGBM_MODEL_PATH, RESIDUAL_START, RESIDUAL_END
import lightgbm as lgb
import os
import json

_df = None
_model = None
_features = None

def _load_data_for_error():
    global _df, _model, _features
    if _df is None:
        _df = pd.read_csv(PROCESSED_DATA_PATH)
        _df['date'] = pd.to_datetime(_df['date'])
        
        if os.path.exists(LIGHTGBM_MODEL_PATH):
            _model = lgb.Booster(model_file=LIGHTGBM_MODEL_PATH)
        if os.path.exists('models/lgbm_features.json'):
            with open('models/lgbm_features.json', 'r') as f:
                _features = json.load(f)

def get_forecast_error(store_id, item_id, start=None, end=None):
    """
    Calculate forecast error (std of one-step residuals) on a residual window
    for a specific store x item. Defaults to the validation period used for
    LightGBM early stopping (documented limitation — not an independent holdout).
    """
    _load_data_for_error()
    if _df is None or _model is None or _features is None:
        return 5.0 # Fallback default

    start = start or RESIDUAL_START or VALID_START
    end = end or RESIDUAL_END or VALID_END
    valid_mask = (_df['date'] >= start) & (_df['date'] <= end)
    item_mask = (_df['store'] == store_id) & (_df['item'] == item_id)
    
    sub_df = _df[valid_mask & item_mask].dropna(subset=_features + ['sales'])
    
    if len(sub_df) < 10:
        # Fallback to store-level error
        store_mask = (_df['store'] == store_id)
        sub_df = _df[valid_mask & store_mask].dropna(subset=_features + ['sales'])
        
        if len(sub_df) < 10:
            # Fallback to global error
            sub_df = _df[valid_mask].dropna(subset=_features + ['sales'])
            
    if len(sub_df) == 0:
        return 5.0
        
    preds = _model.predict(sub_df[_features])
    residuals = sub_df['sales'].values - preds
    return np.std(residuals)

def calculate_safety_stock(forecast_error, lead_time_days, z_score):
    """
    Calculate safety stock using standard inventory policy formula:
    Safety Stock = z * Forecast Error * sqrt(Lead Time)
    
    This is a simplified inventory policy assuming normally distributed, independent errors.
    """
    return max(0, int(np.round(z_score * forecast_error * np.sqrt(lead_time_days))))

def calculate_recommended_stock(expected_demand_during_lead_time, safety_stock):
    """
    Recommended Stock = Expected Demand During Lead Time + Safety Stock
    """
    return int(np.round(expected_demand_during_lead_time)) + safety_stock

def get_z_score(service_level):
    """
    Map service level % to z-score (assuming normal distribution of errors).
    """
    mapping = {
        80: 0.84,
        85: 1.04,
        90: 1.28,
        95: 1.645,
        99: 2.33
    }
    return mapping.get(service_level, 1.645)
