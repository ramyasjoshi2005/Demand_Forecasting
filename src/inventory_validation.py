import pandas as pd
import numpy as np
import os
import json
from config import (
    PROCESSED_DATA_PATH, TEST_START, TEST_END,
    RESIDUAL_START, RESIDUAL_END,
    RESIDUAL_SENSITIVITY_START, RESIDUAL_SENSITIVITY_END,
    LIGHTGBM_MODEL_PATH,
)
from inventory import get_forecast_error, get_z_score, calculate_safety_stock, calculate_recommended_stock


def _evaluate_coverage(test_df, features, model, lead_time, service_levels, residual_start, residual_end, window_name):
    results = {
        sl: {'covered': 0, 'total': 0, 'safety_stock_sum': 0, 'rec_stock_sum': 0, 'lead_demand_sum': 0}
        for sl in service_levels
    }
    error_cache = {}
    grouped = test_df.groupby(['store', 'item'])

    for (store, item), group in grouped:
        group = group.reset_index(drop=True)
        key = (store, item, residual_start, residual_end)
        if key not in error_cache:
            error_cache[key] = get_forecast_error(store, item, start=residual_start, end=residual_end)
        error_std = error_cache[key]

        # Decision origin i uses one-step forecasts for days i+1 ... i+lead_time.
        # Those one-step features include realized sales after the origin, so this is
        # rolling one-step coverage — not an open-loop recursive lead-time forecast.
        for i in range(len(group) - lead_time):
            lead_time_forecast = group.loc[i + 1:i + lead_time, 'forecast'].sum()
            lead_time_actual = group.loc[i + 1:i + lead_time, 'sales'].sum()

            for sl in service_levels:
                z = get_z_score(sl)
                ss = calculate_safety_stock(error_std, lead_time, z)
                rec_stock = calculate_recommended_stock(lead_time_forecast, ss)

                results[sl]['total'] += 1
                results[sl]['safety_stock_sum'] += ss
                results[sl]['rec_stock_sum'] += rec_stock
                results[sl]['lead_demand_sum'] += lead_time_forecast
                if lead_time_actual <= rec_stock:
                    results[sl]['covered'] += 1

    summary = []
    for sl in service_levels:
        total = results[sl]['total']
        coverage = results[sl]['covered'] / total * 100 if total > 0 else 0
        avg_ss = results[sl]['safety_stock_sum'] / total if total > 0 else 0
        avg_rec = results[sl]['rec_stock_sum'] / total if total > 0 else 0
        avg_demand = results[sl]['lead_demand_sum'] / total if total > 0 else 0
        summary.append({
            'residual_window': window_name,
            'residual_start': residual_start,
            'residual_end': residual_end,
            'target_service_level': sl,
            'empirical_service_level': coverage,
            'coverage_gap': coverage - sl,
            'avg_safety_stock': avg_ss,
            'avg_recommended_stock': avg_rec,
            'avg_lead_time_demand': avg_demand,
            'n_decisions': total,
        })
    return summary


def validate_inventory_policy(lead_time=7, service_levels=[90, 95, 99]):
    print("Running inventory validation on test period...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].sort_values(['store', 'item', 'date'])

    import lightgbm as lgb

    if not os.path.exists(LIGHTGBM_MODEL_PATH):
        print("LightGBM model not found. Cannot validate.")
        return

    model = lgb.Booster(model_file=LIGHTGBM_MODEL_PATH)

    with open('models/lgbm_features.json', 'r') as f:
        features = json.load(f)

    test_df = test_df.dropna(subset=features + ['sales']).copy()
    test_df['forecast'] = model.predict(test_df[features])

    primary = _evaluate_coverage(
        test_df, features, model, lead_time, service_levels,
        RESIDUAL_START, RESIDUAL_END, 'validation_full_early_stopping_window',
    )
    sensitivity = _evaluate_coverage(
        test_df, features, model, lead_time, service_levels,
        RESIDUAL_SENSITIVITY_START, RESIDUAL_SENSITIVITY_END, 'validation_jul_sep_sensitivity',
    )

    df_summary = pd.DataFrame(primary + sensitivity)
    df_summary.to_csv('models/inventory_validation.csv', index=False)
    print("Inventory validation complete.")
    print(df_summary)
    return df_summary


if __name__ == "__main__":
    validate_inventory_policy()
