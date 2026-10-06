import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from feature_engineering import create_lag_features, create_rolling_features

def test_feature_engineering():
    # Create mock dataframe with 2 stores, 2 items, 35 days
    dates = pd.date_range('2017-01-01', periods=35)
    stores = [1, 2]
    items = [1, 2]
    
    idx = pd.MultiIndex.from_product([dates, stores, items], names=['date', 'store', 'item'])
    df = pd.DataFrame(index=idx).reset_index()
    df['sales'] = np.random.randint(10, 100, size=len(df))
    df = df.sort_values(by=['store', 'item', 'date']).reset_index(drop=True)
    
    # Apply features
    df_lag = create_lag_features(df)
    df_feat = create_rolling_features(df_lag)
    
    # Test lag independence
    # The first row of store 2 item 1 should have lag_1 as NaN, NOT the last row of store 1 item 2
    store2_item1_first_row = df_feat[(df_feat['store']==2) & (df_feat['item']==1)].iloc[0]
    assert pd.isna(store2_item1_first_row['lag_1']), "Leakage: lag_1 crossed store/item boundaries!"
    
    # Test rolling independence
    assert pd.isna(store2_item1_first_row['rolling_mean_7']), "Leakage: rolling_mean_7 crossed boundaries!"
    
    # Test rolling doesn't use current target
    # rolling_mean_7 for day 8 should be the mean of days 1-7, NOT days 2-8
    s1i1 = df_feat[(df_feat['store']==1) & (df_feat['item']==1)].reset_index(drop=True)
    day8_rolling = s1i1.loc[7, 'rolling_mean_7']
    expected_rolling = s1i1.loc[0:6, 'sales'].mean()
    
    assert np.isclose(day8_rolling, expected_rolling), f"Leakage: Rolling mean included current target! Got {day8_rolling}, expected {expected_rolling}"

    assert s1i1.loc[1, 'lag_1'] == s1i1.loc[0, 'sales']
    assert s1i1.loc[7, 'lag_7'] == s1i1.loc[0, 'sales']
    assert s1i1.loc[7, 'lag_1'] == s1i1.loc[6, 'sales']

    day8_std = s1i1.loc[7, 'rolling_std_7']
    expected_std = s1i1.loc[0:6, 'sales'].std(ddof=1)
    assert np.isclose(day8_std, expected_std)

    print("SUCCESS: Feature engineering is leakage-safe and properly grouped.")

if __name__ == "__main__":
    test_feature_engineering()
