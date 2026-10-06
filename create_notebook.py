import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Demand Forecasting Exploratory Data Analysis
This notebook explores the dataset for the Demand Forecasting project.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('../data/raw/train.csv')
df['date'] = pd.to_datetime(df['date'])
df.head()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Overall Demand"""))

cells.append(nbf.v4.new_code_cell("""
daily_sales = df.groupby('date')['sales'].sum().reset_index()
plt.figure(figsize=(15, 6))
plt.plot(daily_sales['date'], daily_sales['sales'])
plt.title('Total Daily Sales over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Seasonality"""))

cells.append(nbf.v4.new_code_cell("""
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(x='day_of_week', y='sales', data=df, ax=ax1, estimator=np.mean)
ax1.set_title('Average Sales by Day of Week')

sns.barplot(x='month', y='sales', data=df, ax=ax2, estimator=np.mean)
ax2.set_title('Average Sales by Month')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Store and Item Behavior"""))

cells.append(nbf.v4.new_code_cell("""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
store_sales = df.groupby('store')['sales'].sum().reset_index()
sns.barplot(x='store', y='sales', data=store_sales, ax=ax1)
ax1.set_title('Total Sales by Store')

item_sales = df.groupby('item')['sales'].sum().reset_index().sort_values('sales', ascending=False).head(10)
sns.barplot(x='item', y='sales', data=item_sales, ax=ax2, order=item_sales['item'])
ax2.set_title('Top 10 Items by Total Sales')
plt.show()
"""))

nb['cells'] = cells

with open('notebooks/demand_forecasting_eda.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
