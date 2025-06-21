# 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 📁 Load and Prepare Data
file_path = r'Consumption Dataset - Dataset.csv'
df = pd.read_csv(file_path)

# Rename columns to standardized names
df.rename(columns={
    'Date Time Served': 'date',
    'Bar Name': 'location_id',
    'Brand Name': 'item_id',
    'Consumed (ml)': 'consumption'
}, inplace=True)

# Convert data types
df['date'] = pd.to_datetime(df['date'])
df['item_id'] = df['item_id'].astype(str)
df['location_id'] = df['location_id'].astype(str)
df['consumption'] = df['consumption'].fillna(0)

# Aggregate daily consumption
daily = df.groupby(['date', 'item_id', 'location_id'])['consumption'].sum().reset_index()

# Create time-based features
daily['day_of_week'] = daily['date'].dt.dayofweek
daily['month'] = daily['date'].dt.month

# Forecast function
def forecast_group(data, item_id, location_id):
    group = data[(data['item_id'] == item_id) & (data['location_id'] == location_id)]
    if len(group) < 10:
        print(f"⚠️ Not enough records for item {item_id} @ location {location_id}")
        return None, None

    X = group[['day_of_week', 'month']]
    y = group['consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    print(f"📌 RMSE for item '{item_id}' at location '{location_id}': {rmse:.2f}")
    return model, group

# Recommend par level
def recommend_par(model, group, buffer=1.2, days=7):
    future = pd.DataFrame({
        'day_of_week': list(range(days)),
        'month': [group['month'].iloc[-1]] * days
    })
    forecast = model.predict(future)
    total_demand = forecast.sum()
    par = int(np.ceil(total_demand * buffer))
    print(f"📦 Forecasted 7-day demand: {total_demand:.2f} → Recommended Par Level: {par}")
    return forecast, par

# Select item and location for demo
sample_item = df['item_id'].iloc[0]
sample_location = df['location_id'].iloc[0]

# Forecast and recommend
model, group = forecast_group(daily, sample_item, sample_location)
if model:
    forecast, par_level = recommend_par(model, group)

    # Plot result
    future_dates = pd.date_range(group['date'].max() + pd.Timedelta(days=1), periods=7)
    plt.figure(figsize=(10, 5))
    plt.plot(group['date'], group['consumption'], label='Historical', linewidth=2)
    plt.plot(future_dates, forecast, label='Forecasted', linestyle='--', marker='o')
    plt.title(f"Forecast for '{sample_item}' at '{sample_location}'")
    plt.xlabel("Date")
    plt.ylabel("Consumption (ml)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
