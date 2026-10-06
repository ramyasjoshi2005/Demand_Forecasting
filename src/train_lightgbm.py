import pandas as pd
import lightgbm as lgb
import json
import os
from config import PROCESSED_DATA_PATH, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END, LIGHTGBM_MODEL_PATH
from evaluation import evaluate_model

def get_splits(df):
    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    valid_mask = (df['date'] >= VALID_START) & (df['date'] <= VALID_END)
    test_mask = (df['date'] >= TEST_START) & (df['date'] <= TEST_END)
    
    drop_cols = ['date', 'sales']
    features = [c for c in df.columns if c not in drop_cols]

    X_train, y_train = df[train_mask][features], df[train_mask]['sales']
    X_valid, y_valid = df[valid_mask][features], df[valid_mask]['sales']
    X_test_full, y_test_full = df[test_mask][features], df[test_mask]['sales']

    return X_train, y_train, X_valid, y_valid, X_test_full, y_test_full, features

def train_lgbm():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna().copy()
    
    print("--- Time-Based Split ---")
    print(f"Training:   {TRAIN_START} -> {TRAIN_END}")
    print(f"Validation: {VALID_START} -> {VALID_END}")
    print(f"Test:       {TEST_START} -> {TEST_END}")
    print("------------------------")
    
    X_train, y_train, X_valid, y_valid, X_test_full, y_test_full, features = get_splits(df)
    
    os.makedirs('models', exist_ok=True)
    with open('models/lgbm_features.json', 'w') as f:
        json.dump(features, f)
        
    print("Training LightGBM...")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    model.save_model(LIGHTGBM_MODEL_PATH)
    
    # Evaluate full for general performance
    print("\nEvaluating LightGBM on Full Test Set...")
    preds_full = model.predict(X_test_full)
    metrics_full = evaluate_model(y_test_full.values, preds_full, "LightGBM (Full)")
    with open('models/lgbm_full_test_metrics.json', 'w') as f:
        json.dump(metrics_full, f)
        
    print("\nEvaluating LightGBM on all 500 store-item series (test period only)...")
    metrics_eval = metrics_full
    
    # Update model comparison CSV
    comp_file = 'models/model_comparison.csv'

    # Write or append
    comp_df = pd.DataFrame([{
        'model': 'LightGBM',
        'evaluation_population': 'all 500 series',
        'MAE': metrics_eval['mae'],
        'RMSE': metrics_eval['rmse'],
        'MAPE': metrics_eval['mape'],
        'sMAPE': metrics_eval['smape'],
        'test_start': TEST_START,
        'test_end': TEST_END
    }])
    
    if os.path.exists(comp_file):
        existing = pd.read_csv(comp_file)
        existing = existing[existing['model'] != 'LightGBM']
        pd.concat([existing, comp_df]).to_csv(comp_file, index=False)
    else:
        comp_df.to_csv(comp_file, index=False)
        
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    importance.to_csv('models/lgbm_importance.csv', index=False)

if __name__ == "__main__":
    train_lgbm()
