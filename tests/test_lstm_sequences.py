import pandas as pd
import numpy as np
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from train_lstm import create_sequences


def test_lstm_sequence_predicts_last_window_day_without_using_that_day_sales():
    n = 40
    dates = pd.date_range('2017-01-01', periods=n)
    rng = np.random.default_rng(0)
    sales = rng.integers(10, 50, size=n).astype(float)
    # leakage-safe lag_1 = previous sales
    lag_1 = np.concatenate([[np.nan], sales[:-1]])
    df = pd.DataFrame({
        'date': dates,
        'store': 1,
        'item': 1,
        'sales': sales,
        'sales_scaled': sales,
        'lag_1': lag_1,
    }).dropna()

    seqs, stores, items, targets, target_dates, true_sales = create_sequences(df, ['lag_1'], seq_len=7)
    # last timestep lag_1 equals previous day's sales, not the target day's sales
    last_lag = seqs[0, -1, 0]
    target_sales = true_sales[0]
    assert last_lag != target_sales
    # target date is the last date in the first complete window
    assert pd.Timestamp(target_dates[0]) == df['date'].iloc[6]
    assert targets[0] == df['sales_scaled'].iloc[6]
    assert np.isclose(last_lag, df['lag_1'].iloc[6])
