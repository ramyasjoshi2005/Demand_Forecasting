import numpy as np
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from evaluation import calculate_smape, calculate_rmse


def test_smape_symmetric_and_zero_safe():
    y_true = np.array([10.0, 0.0, 5.0])
    y_pred = np.array([10.0, 0.0, 7.0])
    smape = calculate_smape(y_true, y_pred)
    # (0,0) is dropped because the sMAPE denominator is 0; remaining terms: 0 and |5-7|/6
    expected = np.mean([0.0, (2.0 / 6.0)]) * 100
    assert np.isclose(smape, expected)


def test_rmse_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert np.isclose(calculate_rmse(y_true, y_pred), np.sqrt(4 / 3))
