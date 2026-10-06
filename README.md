# Demand Forecasting & Inventory Stock Recommendation System

## Project Overview
This project builds an end-to-end machine learning system that forecasts future product demand and converts those predictions into practical inventory stock recommendations. The goal is to avoid both overstocking (which holds unnecessary capital) and understocking (which leads to lost sales).

## Business Problem
A store needs to estimate how much inventory to order while balancing the risk of stockouts against excess inventory. By accurately forecasting demand, we can calculate a recommended stock level. 

- **Underestimating demand**: Leads to stockouts and lost revenue.
- **Overestimating demand**: Leads to excess inventory, taking up warehouse space and working capital.

## Dataset
**The primary experiments use the Store Item Demand Forecasting dataset from Kaggle.**

*(Note: A synthetic-data generator `src/download_or_simulate_data.py --simulate` is included only for development and testing when the real dataset is unavailable. Synthetic results are not used for the reported model performance or resume claims.)*

## Methodology
1. **EDA**: Understand trends, weekly seasonality, and store/item behavior.
2. **Time-Based Split**: Split data chronologically to prevent data leakage.
3. **Feature Engineering**: Create safe lag features, rolling statistics, and calendar features completely independently per `store x item`.
4. **Baseline Model**: Predict based on "Same day last week" to establish a minimum performance benchmark.
5. **Machine Learning (LightGBM)**: Fast, accurate gradient boosting on the engineered features.
6. **Deep Learning (LSTM)**: PyTorch-based sequence modeling to capture temporal dependencies (Simple LSTM -> Dropout -> Linear).
7. **Evaluation**: Compare models on a holdout test set representing the future using a fair evaluation population.
8. **Inventory Recommendation**: Convert expected demand + safety stock into a recommended stock level.

## Why Time-Based Split?
A chronological split (e.g., Training on 2013-2016, Testing on late 2017) was used because random splitting can allow information from the future to influence training and produce overly optimistic results. We must evaluate models strictly on future data.

## Feature Engineering
We engineered:
- **Calendar Features**: Day of week, month, year, weekend indicator.
- **Lag Features**: Past sales (1, 7, 14, 28 days ago) to capture recent history.
- **Rolling Features**: Moving averages and standard deviations. *Crucially*, these use `.shift(1).rolling(...)` and are calculated over `store x item` groups to ensure no future information leaks into the features.

## LSTM Experiment
The initial LSTM experiment was intentionally simple. The model was subsequently improved using scaled targets, store/item embeddings, longer training (up to 40 epochs) with early stopping, and a larger evaluation population.

## Model Comparison
Models are evaluated on MAE, RMSE, MAPE, and sMAPE. A fair model comparison is performed where all models are evaluated on the exact same population subset.

Because the sequence model is computationally more expensive, the experimental LSTM comparison was conducted on a predefined evaluation population. LightGBM and the Baseline were evaluated on the exact same population for a fair model comparison.

### Results
Test Period: 2017-10-01 to 2017-12-31

| Model | Evaluation Population | MAE | RMSE | MAPE | sMAPE |
|---|---|---|---|---|---|
| Seasonal Naive | all 500 series | 9.08 | 12.05 | 19.92% | 18.56% |
| Same-day-last-week seasonal-naive baseline | all 500 series | 9.08 | 12.05 | 19.92% | 18.56% |
| LightGBM | all 500 series | 5.94 | 7.69 | 13.09% | 12.52% |
| LSTM | all 500 series | 6.45 | 8.60 | 13.97% | 13.85% |

LightGBM outperformed both the seasonal-naive baseline and the LSTM on the shared evaluation population, achieving a 32.6% improvement in sMAPE over the baseline.

## Inventory Recommendation
The stock recommendation uses a standard inventory policy:
`Expected demand during lead time + Safety stock = Recommended stock`

Where **Safety Stock** is:
`Safety Stock = z * forecast_error * sqrt(Lead Time)`
- `z` corresponds to the desired service level (e.g., 95% = 1.645)
- `forecast_error` is approximated using the localized standard deviation of residuals for the specific `store x item` on the validation set.
- `sqrt(Lead Time)` scales the error over the days until restock.

*(This is a simplified inventory policy assuming independent and normally distributed errors).*

## Inventory Policy Validation
The stock recommendation was evaluated on the test period by measuring how often actual lead-time demand remained below the recommended stock level.

| Target Service Level | Empirical Coverage |
|---|---|
| 90.0% | 93.0% |
| 95.0% | 96.8% |
| 99.0% | 99.5% |


## Limitations
- **No external factors**: Promotions, holidays, price changes, and weather are not included.
- **Simplified inventory policy**: The formula assumes normally distributed, independent errors.
- **Static lead time**: Supplier delivery times often vary in the real world.
- **Residual Estimate Limitation**: Forecast-error estimates for the inventory policy are derived from the validation period, which is also used for model early stopping. This can make the estimated error slightly optimistic. A production implementation would estimate inventory uncertainty using a separate rolling backtest or out-of-sample residual period.

## Future Improvements
- Include holiday and promotion flags.
- Implement probabilistic forecasting instead of point forecasts.
- Use the hierarchical structure (Store -> Item) for grouped forecasting.
- Add dynamic pricing optimization.

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
4. **Train Models:**
   ```bash
   python src/baseline.py
   python src/train_lightgbm.py
   python src/train_lstm.py
   python src/inventory_validation.py
   ```
5. **Run Dashboard:**
   ```bash
   python -m streamlit run app.py
   ```
