import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from config import PROCESSED_DATA_PATH
from inventory import calculate_safety_stock, calculate_recommended_stock, get_z_score, get_forecast_error
from forecasting import get_historical_and_forecast

st.set_page_config(page_title="Demand Forecasting & Inventory", layout="wide")

@st.cache_data
def load_full_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

@st.cache_data
def load_metrics():
    comp_file = 'models/model_comparison.csv'
    if os.path.exists(comp_file):
        return pd.read_csv(comp_file)
    return None

df = load_full_data()
metrics_df = load_metrics()

st.title("Demand Forecasting & Inventory Stock Recommendation")

st.markdown("**Dataset: Kaggle Store Item Demand Forecasting**")
if metrics_df is not None and not metrics_df.empty:
    eval_start = metrics_df['test_start'].iloc[0]
    eval_end = metrics_df['test_end'].iloc[0]
    st.markdown(f"**Evaluation Period:** `{eval_start} → {eval_end}`")

if df is None:
    st.warning("Processed dataset not found. Please place the real Kaggle 'train.csv' in data/raw/ and run the preprocessing scripts.")
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Demand Analysis", "Model Performance", "Demand Forecast", "Stock Recommendation", "Business Insights"
    ])

    with tab1:
        st.header("Project Overview")
        st.write("""
        ### Business Problem
        A store needs to estimate how much inventory to order while balancing the risk of stockouts against excess inventory. 
        If it stocks too little, it can lose sales. If it stocks too much, it can hold unnecessary inventory. 
        The system forecasts demand and converts that forecast into a recommended stock level.
        """)
        
        st.subheader("Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Stores", df['store'].nunique())
        col3.metric("Items", df['item'].nunique())
        col4.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")

    with tab2:
        st.header("Demand Analysis (EDA)")
        st.write("### Total Sales over Time")
        daily_sales = df.groupby('date')['sales'].sum().reset_index()
        fig_trend = px.line(daily_sales, x='date', y='sales', title="Daily Total Sales")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Weekly Seasonality")
            weekly = df.groupby('day_of_week')['sales'].mean().reset_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekly['Day'] = weekly['day_of_week'].map(lambda x: day_names[x])
            fig_week = px.bar(weekly, x='Day', y='sales', title="Average Sales by Day of Week")
            st.plotly_chart(fig_week, use_container_width=True)
            
        with col2:
            st.write("### Store Comparison")
            store_sales = df.groupby('store')['sales'].sum().reset_index()
            fig_store = px.bar(store_sales, x='store', y='sales', title="Total Sales by Store")
            st.plotly_chart(fig_store, use_container_width=True)

    with tab3:
        st.header("Model Performance")
        if metrics_df is not None and not metrics_df.empty:
            st.write("A time-based chronological split was used. Random splitting would allow future information to leak into training, producing overly optimistic results.")
            
            st.subheader("Model Comparison (Fair Evaluation)")
            st.markdown("**Fair comparison population:** all 500 store × item series, chronological test period Oct–Dec 2017.")
            eval_start = metrics_df['test_start'].iloc[0]
            eval_end = metrics_df['test_end'].iloc[0]
            st.markdown(f"**Test period:** `{eval_start} → {eval_end}`")
            st.dataframe(metrics_df[['model', 'MAE', 'RMSE', 'MAPE', 'sMAPE']], use_container_width=True)
            
            # Show full test set LightGBM diagnostic if available
            full_test_path = 'models/lgbm_full_test_metrics.json'
            if os.path.exists(full_test_path):
                st.subheader("Full test-set LightGBM diagnostic")
                st.markdown("*Note: Evaluated across all 500 store-item series.*")
                with open(full_test_path, 'r') as f:
                    full_metrics = json.load(f)
                full_metrics_df = pd.DataFrame([full_metrics])
                # Format to match uppercase headers
                full_metrics_df = full_metrics_df.rename(columns={'mae': 'MAE', 'rmse': 'RMSE', 'mape': 'MAPE', 'smape': 'sMAPE'})
                st.dataframe(full_metrics_df, use_container_width=True)
            
            if os.path.exists('models/lgbm_importance.csv'):
                st.subheader("LightGBM Feature Importance")
                fi = pd.read_csv('models/lgbm_importance.csv').head(10)
                fig_fi = px.bar(fi, x='importance', y='feature', orientation='h', title="Top 10 Features")
                fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Results not available — train models on the real dataset first.")

    with tab4:
        st.header("Demand Forecast")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_store = st.selectbox("Store", sorted(df['store'].unique()), key='forecast_store')
        with col2:
            selected_item = st.selectbox("Item", sorted(df['item'].unique()), key='forecast_item')
        with col3:
            horizon = st.selectbox("Forecast Horizon (days)", [7, 14, 30])
            
        if st.button("Generate Forecast"):
            try:
                history, forecast = get_historical_and_forecast(selected_store, selected_item, horizon)
                if history is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=history['date'], y=history['sales'], name="Historical Demand", line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['forecast'], name="Forecast", line=dict(color='orange', dash='dot')))
                    fig.update_layout(title=f"Demand Forecast for Store {selected_store}, Item {selected_item}", xaxis_title="Date", yaxis_title="Sales")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Forecast Table")
                    st.dataframe(forecast, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating forecast: {e}")

    with tab5:
        st.header("Inventory / Stock Recommendation")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            inv_store = st.selectbox("Store", sorted(df['store'].unique()), key='inv_store')
        with col2:
            inv_item = st.selectbox("Item", sorted(df['item'].unique()), key='inv_item')
        with col3:
            lead_time = st.number_input("Lead Time (days)", min_value=1, max_value=30, value=7)
        with col4:
            service_level = st.selectbox("Service Level (%)", [90, 95, 99], index=1)
            
        st.write("---")
        st.write("**Simplified Inventory Policy:** Safety stock is extra inventory held to protect against uncertainty in demand and forecast error. `√Lead Time` is used under the standard independent-error assumption.")
        
        if st.button("Calculate Recommendation"):
            try:
                history, forecast = get_historical_and_forecast(inv_store, inv_item, lead_time)
                
                # Dynamic forecast error based on residual std deviation for this specific store/item
                error_std = get_forecast_error(inv_store, inv_item)
                
                z = get_z_score(service_level)
                ss = calculate_safety_stock(error_std, lead_time, z)
                
                expected_demand = forecast['forecast'].sum()
                recommended = calculate_recommended_stock(expected_demand, ss)
                
                st.subheader(f"Output for Store {inv_store}, Item {inv_item}")
                
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Expected Demand (Lead Time)", f"{expected_demand:.0f} units")
                mc2.metric("Safety Stock", f"{ss} units")
                mc3.metric("Recommended Stock Level", f"{recommended} units")
                
                st.info(f"Formula: Recommended Stock = {expected_demand:.0f} (Demand) + {ss} (Safety Stock).  Estimated local standard error: {error_std:.2f}")
            except Exception as e:
                st.error(f"Error calculating inventory: {e}. Are models trained?")
                
        st.write("---")
        st.header("Inventory Policy Validation")
        st.write("The stock recommendation was evaluated on the test period by measuring how often actual lead-time demand remained below the recommended stock level.")
        
        val_path = 'models/inventory_validation.csv'
        if os.path.exists(val_path):
            val_df = pd.read_csv(val_path)
            primary = val_df
            if 'residual_window' in val_df.columns:
                primary = val_df[val_df['residual_window'] == 'validation_full_early_stopping_window']
                st.caption("Primary coverage uses residual σ from the full Jan–Sep 2017 validation window (same period as LightGBM early stopping).")
            st.dataframe(primary.style.format({
                'empirical_service_level': '{:.1f}%',
                'coverage_gap': '{:.1f}%',
                'avg_safety_stock': '{:.1f}',
                'avg_recommended_stock': '{:.1f}',
                'avg_lead_time_demand': '{:.1f}'
            }), use_container_width=True)

            fig = px.bar(primary, x='target_service_level', y='empirical_service_level',
                         title="Target vs Empirical Service Level",
                         labels={'target_service_level': 'Target Service Level (%)', 'empirical_service_level': 'Empirical Coverage (%)'})
            fig.add_scatter(x=primary['target_service_level'], y=primary['target_service_level'], mode='lines', name='Target', line=dict(color='red', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)

            if 'residual_window' in val_df.columns:
                sens = val_df[val_df['residual_window'] == 'validation_jul_sep_sensitivity']
                if not sens.empty:
                    st.subheader("Sensitivity: Jul–Sep residual window")
                    st.dataframe(sens, use_container_width=True)

            st.info("Limitations: (1) Forecast-error σ is estimated on the validation period also used for early stopping, so coverage can be slightly optimistic. (2) Lead-time demand is the sum of rolling one-step LightGBM forecasts, which use realized sales after the order origin; this is more optimistic than a recursive open-loop forecast from the decision date.")
        else:
            st.info("Inventory validation results not available. Run validation script.")

    with tab6:
        st.header("Business Insights")
        best_store = df.groupby('store')['sales'].sum().idxmax()
        best_item = df.groupby('item')['sales'].sum().idxmax()
        
        st.write(f"- **Highest-demand store:** Store {best_store}")
        st.write(f"- **Highest-demand item:** Item {best_item}")
        st.write("- **Seasonal patterns:** Clear weekly seasonality, typical peaks around specific times of the year.")
        st.write("- **Model application:** By modeling Store-Item error independently, safety stock is precisely calibrated to localized volatility instead of a global average.")
