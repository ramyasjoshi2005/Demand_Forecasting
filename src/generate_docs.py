import pandas as pd
import json
import os
import sys


def _fmt_pct(x):
    return f"{x:.2f}%"


def generate():
    comp_file = 'models/model_comparison.csv'
    if not os.path.exists(comp_file):
        print("model_comparison.csv not found. Train models first.")
        sys.exit(1)

    df = pd.read_csv(comp_file)
    fair_df = df[df['evaluation_population'] == 'all 500 series'].copy()

    def row_for(name_substr):
        hit = fair_df[fair_df['model'].str.contains(name_substr, case=False, na=False)]
        if hit.empty:
            return None
        return hit.iloc[-1]

    baseline_row = row_for('seasonal-naive')
    if baseline_row is None:
        baseline_row = row_for('baseline')
    lgbm_row = row_for('LightGBM')
    lstm_row = row_for('LSTM')

    baseline_smape = float(baseline_row['sMAPE']) if baseline_row is not None else float('nan')
    lgbm_smape = float(lgbm_row['sMAPE']) if lgbm_row is not None else float('nan')
    lstm_smape = float(lstm_row['sMAPE']) if lstm_row is not None else float('nan')
    improvement = ((baseline_smape - lgbm_smape) / baseline_smape) * 100 if baseline_smape else float('nan')

    test_start = fair_df['test_start'].values[0]
    test_end = fair_df['test_end'].values[0]

    table_md = "| Model | Evaluation Population | MAE | RMSE | MAPE | sMAPE |\n"
    table_md += "|---|---|---|---|---|---|\n"
    for _, row in fair_df.iterrows():
        table_md += (
            f"| {row['model']} | {row['evaluation_population']} | "
            f"{row['MAE']:.2f} | {row['RMSE']:.2f} | {row['MAPE']:.2f}% | {row['sMAPE']:.2f}% |\n"
        )

    lstm_cfg = ""
    prev_note = ""
    if os.path.exists('models/lstm_metrics.json'):
        with open('models/lstm_metrics.json', 'r') as f:
            lm = json.load(f)
        lstm_cfg = f"""## LSTM Training Configuration
The LSTM is a one-layer network (hidden size {lm.get('hidden_size', 64)}) with store and item embeddings, trained with Adam on MSE of a scaled target.

| Setting | Value |
|---|---|
| Sequence length | {lm.get('sequence_length', 30)} |
| Max epochs | {lm.get('max_epochs', 'n/a')} |
| Patience | {lm.get('patience', 'n/a')} |
| Batch size | {lm.get('batch_size', 'n/a')} |
| Learning rate | {lm.get('learning_rate', 'n/a')} |
| Seed | {lm.get('seed', 42)} |
| Epochs run | {lm.get('epochs_trained', 'n/a')} |
| Best epoch (min validation loss) | {lm.get('best_epoch', 'n/a')} |
| Stop reason | {lm.get('stop_reason', 'n/a')} |
| Test rows / series | {lm.get('n_test_rows', 'n/a')} / {lm.get('n_test_series', 'n/a')} |

Early stopping monitors **validation MSE** (Jan–Sep 2017). The checkpoint with the lowest validation loss is restored before test evaluation. Test data is not used for stopping or checkpoint selection.
"""
    if os.path.exists('models/lstm_metrics_previous.json'):
        with open('models/lstm_metrics_previous.json', 'r') as f:
            prev = json.load(f)
        prev_note = (
            f"A previous LSTM checkpoint on this repo scored sMAPE {prev.get('smape', float('nan')):.2f}% "
            f"(best epoch {prev.get('best_epoch')}, epochs run {prev.get('epochs_trained')}). "
            "That run built sequences inside each split in isolation, so the first 30 test days were dropped "
            "and predictions were rounded before scoring. The current LSTM number uses the full Oct–Dec 2017 "
            "test dates (same chronological window as LightGBM) and the same unrounded sMAPE definition.\n\n"
        )

    val_md = ""
    resume_inv = ""
    val_file = 'models/inventory_validation.csv'
    if os.path.exists(val_file):
        val_df = pd.read_csv(val_file)
        primary = val_df
        if 'residual_window' in val_df.columns:
            primary = val_df[val_df['residual_window'] == 'validation_full_early_stopping_window']
        val_md = "## Inventory Policy Validation\n"
        val_md += (
            "The stock recommendation was evaluated on the test period by measuring how often actual "
            "lead-time demand remained below the recommended stock level. Reported service levels are "
            "90%, 95%, and 99% only.\n\n"
        )
        val_md += "| Target Service Level | Empirical Coverage |\n|---|---|\n"
        for _, row in primary.iterrows():
            val_md += f"| {row['target_service_level']}% | {row['empirical_service_level']:.1f}% |\n"

        sl95 = primary[primary['target_service_level'] == 95]['empirical_service_level'].values
        if len(sl95) > 0:
            resume_inv = (
                f"\n- Validated inventory policy on a holdout set, achieving {sl95[0]:.1f}% "
                "empirical coverage for a 95% target service level."
            )

        if 'residual_window' in val_df.columns:
            sens = val_df[val_df['residual_window'] == 'validation_jul_sep_sensitivity']
            if not sens.empty:
                val_md += (
                    "\nA sensitivity run estimates residual σ on Jul–Sep 2017 only (still inside the "
                    "early-stopping validation window, not a fully independent period):\n\n"
                    "| Target Service Level | Empirical Coverage (Jul–Sep σ) |\n|---|---|\n"
                )
                for _, row in sens.iterrows():
                    val_md += f"| {row['target_service_level']}% | {row['empirical_service_level']:.1f}% |\n"

    readme_content = f"""# Demand Forecasting & Inventory Stock Recommendation System

## Project Overview
This project builds an end-to-end machine learning system that forecasts future product demand and converts those predictions into practical inventory stock recommendations. The goal is to avoid both overstocking (which holds unnecessary capital) and understocking (which leads to lost sales).

## Business Problem
A store needs to estimate how much inventory to order while balancing the risk of stockouts against excess inventory. By accurately forecasting demand, we can calculate a recommended stock level.

- **Underestimating demand**: Leads to stockouts and lost revenue.
- **Overestimating demand**: Leads to excess inventory, taking up warehouse space and working capital.

## Dataset
**Primary experiments use the Store Item Demand Forecasting dataset from Kaggle** (2013–2017, 913,000 daily records, 10 stores × 50 items = 500 series).

A synthetic-data generator `src/download_or_simulate_data.py --simulate` exists only for local development when the Kaggle file is unavailable. Synthetic results are **not** used for reported metrics or resume claims.

## Methodology
1. **EDA**: Trends, weekly seasonality, and store/item behavior.
2. **Chronological split**: Train 2013–2016, validation Jan–Sep 2017, test Oct–Dec 2017.
3. **Feature engineering**: Lags, rolling statistics, and calendar features, computed independently per `store × item`.
4. **Baseline**: Seasonal-naive / same-day-last-week (`sales(t) = lag_7`).
5. **LightGBM**: Gradient boosting on the leakage-safe tabular features; early stopping on validation RMSE.
6. **LSTM**: PyTorch sequence model with store/item embeddings; early stopping on validation MSE; best checkpoint restored.
7. **Evaluation**: MAE, RMSE, MAPE, and sMAPE on the same 500 series and the same test window.
8. **Inventory**: Expected lead-time demand + `z × forecast_error × sqrt(lead time)`.

## Why a Chronological Split?
Random splitting would mix future days into training. Train/validation/test dates are strictly ordered and do not overlap. The test set is not used for early stopping, checkpointing, or LightGBM model selection.

## Feature Engineering and Leakage Prevention
- **Calendar**: year, month, day, day of week, week of year, day of year, weekend flag, quarter.
- **Lags**: 1, 7, 14, 28, and 365, grouped by `store × item`.
- **Rolling mean/std**: `groupby(store, item).shift(1).rolling(...)` so the current day's sales never enter that day's features.
- Features are built on the full history, then **split by date**. Because lags and rolling windows only look backward, this does not leak the current or future target into `X`.

## Models

### Seasonal-naive / same-day-last-week baseline
The reported baseline predicts each test day using `lag_7` (demand on the same weekday one week earlier). It is **not** a seasonal-naive lag-365 method. A previous-day (`lag_1`) diagnostic is computed in code but is not the headline baseline.

### LightGBM
Trained on 2013–2016 with early stopping against Jan–Sep 2017. Final metrics are computed once on Oct–Dec 2017. Hyperparameters were not re-tuned against the test set.

### LSTM
Input: length-30 sequences of scaled lag/rolling/calendar features; last timestep is the forecast date. Store and item IDs are embeddings, not numeric inputs. Target sales are standard-scaled on the training period only.

{lstm_cfg}
{prev_note}## Model Comparison
All models below are scored with the same `evaluation.calculate_smape` implementation on all 500 store–item series and the Oct–Dec 2017 test period.

Test period: {test_start} to {test_end}

{table_md}
LightGBM is the strongest model on this dataset, with a {improvement:.1f}% sMAPE reduction versus the same-day-last-week baseline. The LSTM is a fair neural baseline, not a tuned competitor: it uses the same chronological split and is not optimized to beat LightGBM.

## Inventory Recommendation
`Recommended stock = expected demand during lead time + safety stock`

`Safety stock = z × forecast_error × sqrt(lead time)`

- `z` maps to the target service level under a normal-error assumption: 90% → 1.28, 95% → 1.645, 99% → 2.33.
- `forecast_error` is the store–item standard deviation of LightGBM one-step residuals on the validation window (fallback to store-level, then global, if a series is too short).
- Default evaluated lead time is 7 days.
- Dashboard recommendations use recursive LightGBM forecasts from the last observed date (information available at forecast time).
- The **coverage study** sums rolling one-step LightGBM test forecasts over the next `lead_time` days. Those one-step features include realized sales after the order origin, so coverage is more optimistic than a fully recursive open-loop lead-time forecast. Residual σ does **not** use test-set residuals.

{val_md}
## Limitations
- **No external factors**: promotions, holidays, prices, and weather are absent.
- **Simplified inventory policy**: independent, normally distributed errors and fixed lead time.
- **Residual window is not independent**: primary σ comes from Jan–Sep 2017, which is also the LightGBM/LSTM early-stopping set. Coverage can be slightly optimistic. A Jul–Sep sensitivity is reported but is still inside that validation window. A production system would estimate uncertainty on a dedicated residual / rolling-backtest period.
- **Coverage evaluation is rolling one-step, not open-loop**: see above.
- **LSTM vs LightGBM is not an identical feature-use comparison**: LightGBM sees a single day's tabular features; the LSTM sees a 30-day window ending on that day.
- LightGBM still uses integer `store` and `item` as numeric features (trees can split on them). The LSTM uses embeddings instead.

## Future Improvements
- Holiday and promotion flags.
- Probabilistic forecasts instead of residual-based safety stock.
- A residual period that is disjoint from model selection.
- Recursive (open-loop) lead-time coverage evaluation.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Obtain Data:**
   Place the real Kaggle `train.csv` in `data/raw/`
3. **Run Preprocessing & Feature Engineering:**
   ```bash
   $env:PYTHONPATH=".;$PWD/src"
   python src/data_preprocessing.py
   python src/feature_engineering.py
   ```
4. **Train / evaluate models:**
   ```bash
   python src/baseline.py
   python src/train_lightgbm.py
   python src/train_lstm.py
   python src/inventory_validation.py
   python src/generate_docs.py
   ```
5. **Tests:**
   ```bash
   pytest tests -q
   ```
6. **Dashboard:**
   ```bash
   python -m streamlit run app.py
   ```
"""
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

    resume_content = f"""- Developed a demand forecasting and inventory recommendation system using leakage-safe time-series features and LightGBM, reducing sMAPE from {baseline_smape:.2f}% to {lgbm_smape:.2f}% versus a seasonal-naive / same-day-last-week baseline on a chronological holdout.
- Evaluated LightGBM against a same-day-last-week baseline and an LSTM (embeddings, early stopping, shared Oct–Dec 2017 test window) on all 500 store-item series; LightGBM had the lowest sMAPE ({lgbm_smape:.2f}% vs LSTM {lstm_smape:.2f}%).
- Converted demand forecasts into configurable safety-stock recommendations using lead-time demand, localized forecast residuals and service-level targets.{resume_inv}
"""
    with open('resume_bullets.md', 'w', encoding='utf-8') as f:
        f.write(resume_content)

    interview_content = f"""# Interview Notes

### Business
**1. What problem does this project solve?**
It forecasts future demand to help a store estimate how much inventory to order while balancing the risk of stockouts against excess inventory.

**2. Why is demand forecasting important?**
It directly affects profit. Better forecasts mean lower inventory costs and higher product availability.

**3. What happens if demand is underestimated?**
Stockouts occur, leading to lost sales and unhappy customers.

**4. What happens if demand is overestimated?**
Excess inventory ties up working capital, increases storage costs, and risks product obsolescence or spoilage.

### Time Series
**5. Why can't you randomly split the data?**
Random splitting causes data leakage. You could train on a Wednesday and test on a Tuesday of the same week, meaning the model "sees the future." You must use a chronological split.

**6. What is a lag feature?**
A lag feature is the value of the target variable from a past time step (e.g., sales from exactly 7 days ago).

**7. What is a rolling mean?**
An average calculated over a moving window of recent days (e.g., average sales over the last 14 days).

**8. Why are lag features useful?**
They capture recent trends and autocorrelation (what happened recently is a good predictor of what will happen today).

**9. Why is lag 365 useful?**
It captures annual seasonality directly. If sales spike on a specific date every year, lag 365 gives the model that exact historical value to learn from.

**10. What is seasonality?**
Repeating patterns in the data occurring at fixed intervals, like higher sales on weekends (weekly) or higher sales in summer (yearly).

**11. What is the difference between trend and seasonality?**
Trend is the long-term direction of the data (e.g., sales increasing over 5 years). Seasonality is the short-term repeating pattern.

### Modeling
**12. Why did you use a seasonal-naive / same-day-last-week baseline?**
It is the actual implemented baseline: `sales(t) = lag_7`. A more complex model is only useful if it beats last week's same weekday. This project does **not** use lag-365 as the reported naive forecast.

**13. Why LightGBM?**
It's a gradient boosting framework that is highly efficient, handles tabular data extremely well, and is usually the state-of-the-art for tabular forecasting tasks.

**14. Why LSTM?**
LSTM is a neural sequence model. Here it is a fair neural baseline with store/item embeddings, scaled targets, validation-based early stopping, and the same test window as LightGBM — not a claim that deep learning should win on this dataset.

**15. Which model performed better?**
LightGBM (sMAPE {lgbm_smape:.2f}%) beat both the same-day-last-week baseline ({baseline_smape:.2f}%) and the LSTM ({lstm_smape:.2f}%) on all 500 series, Oct–Dec 2017.

**16. How was the LSTM trained?**
Config-driven: max 40 epochs, patience 7, checkpoint on lowest validation MSE, seed 42. Training stops on early stopping or max epochs; the best checkpoint is restored. Sequences use history across split boundaries so the test score covers the full Oct–Dec 2017 window.

**17. If the LSTM still loses, what do you say?**
Lag/rolling features already encode most of the short-term structure. A one-layer LSTM on 30-step windows did not beat a well-regularized tree model on this relatively regular retail series. That is an empirical result, not a failure of "trying deep learning."

### Evaluation
**18. Why RMSE?**
Root Mean Squared Error penalizes large errors, which matter for inventory because a massive misforecast is costly.

**19. What is MAPE?**
Mean Absolute Percentage Error is the average absolute error as a percentage of actual sales.

**20. What problem can MAPE have with zero actual sales?**
Division by zero / exploding percentages when actuals are 0.

**21. What is sMAPE?**
Symmetric MAPE divides by the average of |actual| and |prediction|, which is defined as long as both are not zero.

**22. Why should the test set represent future data?**
Deployment predicts the future. The test set must simulate that.

### Inventory
**23. What is safety stock?**
Extra inventory held to protect against demand uncertainty and forecast error.

**24. What does the z-score represent?**
It maps to service level under a normal-error assumption (90% → 1.28, 95% → 1.645, 99% → 2.33).

**25. Why does safety stock use forecast error?**
If the forecast were perfect, you would not need safety stock. Larger residual σ implies more buffer.

**26. How do you know your stock recommendation works?**
Empirical coverage on the test period for 90%, 95%, and 99% targets, using LightGBM one-step forecasts and validation residual σ.

**27. Is your inventory formula production-ready?**
No. It is a simplified independent-normal, fixed-lead-time policy.

**28. What is the limitation of your safety-stock validation?**
(1) Residual σ is estimated on the same Jan–Sep 2017 window used for early stopping, so coverage can be slightly optimistic. (2) Coverage sums rolling one-step forecasts that observe sales after the order date; a true order would need an open-loop forecast from the origin.

### Practical
**29. What would you improve if you had more data?**
External regressors: promotions, holidays, prices, weather.

**30. How would promotions affect forecasting?**
They create spikes that lags alone cannot explain; you need an explicit promo feature or a causal overlay.

**31. How would you handle a new product with no sales history?**
Cold start: map to similar items, or use attribute-level models / hierarchical pooling.

**32. How would you handle a sudden demand spike?**
Rolling features adapt slowly; you would add outlier handling, human overrides, or event flags.
"""
    with open('INTERVIEW_NOTES.md', 'w', encoding='utf-8') as f:
        f.write(interview_content)

    print("Documentation successfully regenerated based on real metrics.")


if __name__ == "__main__":
    generate()
