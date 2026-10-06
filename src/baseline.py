import pandas as pd
from config import PROCESSED_DATA_PATH, TEST_START, TEST_END
from evaluation import evaluate_model
import json
import os

BASELINE_NAME = "Seasonal-naive / same-day-last-week baseline"
LEGACY_BASELINE_NAMES = {
    "Seasonal Naive",
    "Same-day-last-week seasonal-naive baseline",
    BASELINE_NAME,
}


def load_test_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()
    return test_df


def run_baseline(test_df):
    """
    Seasonal-naive / same-day-last-week baseline: predict sales(t) using lag_7
    (demand on the same weekday one week earlier).
    lag_1 (previous day) is computed only as a diagnostic, not the reported baseline.
    """
    test_df = test_df.dropna(subset=['lag_7', 'lag_1', 'sales']).copy()

    y_true = test_df['sales'].values
    y_pred_lag7 = test_df['lag_7'].values
    y_pred_lag1 = test_df['lag_1'].values

    metrics_lag7 = evaluate_model(y_true, y_pred_lag7, BASELINE_NAME)
    evaluate_model(y_true, y_pred_lag1, "Diagnostic (previous-day lag_1)")

    os.makedirs('models', exist_ok=True)
    with open('models/baseline_metrics.json', 'w') as f:
        json.dump({
            'model': BASELINE_NAME,
            'method': 'predict sales(t) = lag_7 = sales(t-7)',
            'evaluation_population': 'all 500 series',
            **metrics_lag7,
        }, f)

    n_series = test_df[['store', 'item']].drop_duplicates().shape[0]
    print(f"Baseline evaluation: {n_series} store-item series, {len(test_df)} rows")

    comp_file = 'models/model_comparison.csv'
    comp_df = pd.DataFrame([{
        'model': BASELINE_NAME,
        'evaluation_population': 'all 500 series',
        'MAE': metrics_lag7['mae'],
        'RMSE': metrics_lag7['rmse'],
        'MAPE': metrics_lag7['mape'],
        'sMAPE': metrics_lag7['smape'],
        'test_start': TEST_START,
        'test_end': TEST_END,
    }])

    if os.path.exists(comp_file):
        existing = pd.read_csv(comp_file)
        existing = existing[~existing['model'].isin(LEGACY_BASELINE_NAMES)]
        pd.concat([existing, comp_df]).to_csv(comp_file, index=False)
    else:
        comp_df.to_csv(comp_file, index=False)

    return metrics_lag7


if __name__ == "__main__":
    test_df = load_test_data()
    run_baseline(test_df)
