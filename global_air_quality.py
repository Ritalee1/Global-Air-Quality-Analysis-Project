#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# Load Dataset
file_path = "global_air_quality_cleaned.csv"
try:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: File '{file_path}' not found.")
    exit()

# Display dataset preview
print("📌 Dataset Preview:")
print(df.head())

# Keep only relevant columns
columns_to_keep = ["Location", "Period", "Pollution_Value", "Pollution_Low", "Pollution_High", "Indicator"]
df = df[columns_to_keep]

print("📌 Dataset After Selecting Important Columns:")
print(df.head())

# Check for missing values
df.dropna(inplace=True)
print("📌 Missing Values After Cleaning:")
print(df.isna().sum())

# Summary Statistics
print("📌 Dataset Summary:")
print(df.describe())

# Boxplot for Pollution Levels
plt.figure(figsize=(12, 6))
sns.boxplot(x="Indicator", y="Pollution_Value", data=df)
plt.xticks(rotation=90)
plt.title("Pollution Levels Across Indicators")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Pollution Levels")
plt.show()

# Pollution Trends by Country
top_polluted = df.groupby("Location")["Pollution_Value"].mean().sort_values(ascending=False).head(10).index
df_top = df[df["Location"].isin(top_polluted)]
plt.figure(figsize=(12,6))
sns.lineplot(data=df_top, x="Period", y="Pollution_Value", hue="Location")
plt.title("Pollution Trends Over Time (Top 10 Polluted Locations)")
plt.legend(title="Location", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Predict Future Pollution Trends Using Linear Regression
df_regression = df.groupby("Period")["Pollution_Value"].mean().reset_index()
X = df_regression["Period"].values.reshape(-1, 1)
y = df_regression["Pollution_Value"].values
model = LinearRegression()
model.fit(X, y)
future_years = np.array(range(2020, 2031)).reshape(-1, 1)
future_predictions = model.predict(future_years)
plt.figure(figsize=(12,6))
plt.scatter(X, y, label="Actual Data", color="blue")
plt.plot(future_years, future_predictions, label="Predicted Trend", color="red", linestyle="dashed")
plt.xlabel("Year")
plt.ylabel("Average Pollution Value")
plt.title("Predicted Pollution Trends (2020-2030)")
plt.legend()
plt.show()

# ARIMA-Based Pollution Forecasting
df_timeseries = df.groupby("Period")["Pollution_Value"].mean().reset_index()
df_timeseries.set_index("Period", inplace=True)
model_arima = ARIMA(df_timeseries, order=(2,1,2))
model_arima_fit = model_arima.fit()
forecast_years = list(range(2020, 2031))
forecast_arima = model_arima_fit.forecast(steps=len(forecast_years))
plt.figure(figsize=(12,6))
plt.plot(df_timeseries, label="Actual Pollution Levels", marker="o")
plt.plot(forecast_years, forecast_arima, label="Predicted Pollution (ARIMA)", linestyle="dashed", color="red")
plt.xlabel("Year")
plt.ylabel("Average Pollution Value")
plt.title("ARIMA Forecast: Predicted Pollution Trends (2020-2030)")
plt.legend()
plt.show()

# Prophet-Based Forecasting
df_prophet = df.groupby("Period")["Pollution_Value"].mean().reset_index()
df_prophet.rename(columns={"Period": "ds", "Pollution_Value": "y"}, inplace=True)
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")
model_prophet = Prophet(yearly_seasonality=True)
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=11, freq="Y")
forecast_prophet = model_prophet.predict(future)
plt.figure(figsize=(12,6))
plt.plot(df_prophet["ds"], df_prophet["y"], label="Actual Pollution Levels", marker="o", color="blue")
plt.plot(forecast_prophet["ds"], forecast_prophet["yhat"], linestyle="dashed", color="red", label="Predicted Pollution (Prophet)")
plt.fill_between(forecast_prophet["ds"], forecast_prophet["yhat_lower"], forecast_prophet["yhat_upper"], color="red", alpha=0.2)
plt.xlabel("Year")
plt.ylabel("Average Pollution Value")
plt.title("Prophet Forecast: Pollution Trends (2020-2030)")
plt.legend()
plt.show()

# Prophet vs. ARIMA Comparison
plt.figure(figsize=(12,6))
plt.plot(df_timeseries.index, df_timeseries["Pollution_Value"], label="Actual Pollution Levels", marker="o", color="blue")
plt.plot(forecast_prophet["ds"], forecast_prophet["yhat"], linestyle="dashed", color="red", label="Prophet Prediction")
plt.plot(forecast_years, forecast_arima, linestyle="dashed", color="green", label="ARIMA Prediction")
plt.xlabel("Year")
plt.ylabel("Average Pollution Value")
plt.title("Comparison: ARIMA vs Prophet Forecast (2020-2030)")
plt.legend()
plt.grid(True)
plt.show()
