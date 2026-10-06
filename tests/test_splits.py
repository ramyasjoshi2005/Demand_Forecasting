import pandas as pd
import numpy as np
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from train_lightgbm import get_splits


def test_chronological_split_no_overlap():
    dates = pd.date_range('2013-01-01', '2017-12-31', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'store': 1,
        'item': 1,
        'sales': 1.0,
        'lag_1': 1.0,
    })
    X_train, y_train, X_valid, y_valid, X_test, y_test, features = get_splits(df)

    train_dates = df.loc[X_train.index, 'date']
    valid_dates = df.loc[X_valid.index, 'date']
    test_dates = df.loc[X_test.index, 'date']

    assert train_dates.max() <= pd.Timestamp(TRAIN_END)
    assert valid_dates.min() >= pd.Timestamp(VALID_START)
    assert valid_dates.max() <= pd.Timestamp(VALID_END)
    assert test_dates.min() >= pd.Timestamp(TEST_START)
    assert test_dates.max() <= pd.Timestamp(TEST_END)
    assert set(train_dates) & set(valid_dates) == set()
    assert set(valid_dates) & set(test_dates) == set()
    assert set(train_dates) & set(test_dates) == set()
    assert 'sales' not in features
    assert 'date' not in features


def test_split_order_is_strictly_chronological():
    assert pd.Timestamp(TRAIN_END) < pd.Timestamp(VALID_START)
    assert pd.Timestamp(VALID_END) < pd.Timestamp(TEST_START)
