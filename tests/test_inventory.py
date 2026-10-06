import numpy as np
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from inventory import calculate_safety_stock, calculate_recommended_stock, get_z_score


def test_z_scores_for_reported_service_levels():
    assert get_z_score(90) == 1.28
    assert get_z_score(95) == 1.645
    assert get_z_score(99) == 2.33


def test_safety_stock_formula():
    error = 10.0
    lead_time = 4
    z = 1.645
    expected = int(np.round(z * error * np.sqrt(lead_time)))
    assert calculate_safety_stock(error, lead_time, z) == expected
    assert calculate_safety_stock(0, 7, 1.645) == 0
    assert calculate_safety_stock(-5, 7, 1.645) == 0


def test_recommended_stock_is_demand_plus_safety():
    assert calculate_recommended_stock(100.4, 12) == 100 + 12
    assert calculate_recommended_stock(10.6, 3) == 11 + 3
