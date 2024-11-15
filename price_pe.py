# -*- coding: utf-8 -*-
"""price_pe.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17tB-p_abhPZfIylFqqOVn-ztA15ZzI-Z
"""

# Ensure the necessary libraries are imported
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf  # For gathering stock price data

# Fetch stock data function (unchanged)
def fetch_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="10y")  # 5 years of historical data
    return hist

# Feature engineering function (unchanged)
def calculate_features(df):
    df['Return'] = df['Close'].pct_change()  # Daily returns
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day moving average
    df['SMA_200'] = df['Close'].rolling(window=200).mean()  # 200-day moving average
    df['Volatility'] = df['Close'].rolling(window=50).std()  # 50-day volatility
    df['P/E_Ratio'] = df['Close'] / (df['Volume'] * df['Close'])  # Simplified P/E approximation
    df = df.dropna()  # Remove rows with NaN values due to rolling calculations
    return df

# Apply feature engineering to stock data
stock_data = fetch_stock_data("HCC.NS")
stock_data = calculate_features(stock_data)

stock_data.head()

stock_data.tail()

# Prepare Features and Targets
features = stock_data[['Return', 'SMA_50', 'SMA_200', 'Volatility']]
price_target = stock_data['Close']  # For price prediction
pe_target = stock_data['P/E_Ratio']  # For P/E ratio prediction

# Drop rows with NaN or infinite values in features or targets
features.replace([np.inf, -np.inf], np.nan, inplace=True)
price_target.replace([np.inf, -np.inf], np.nan, inplace=True)
pe_target.replace([np.inf, -np.inf], np.nan, inplace=True)

# Create a DataFrame to help in dropping rows with NaN values
data = pd.concat([features, price_target, pe_target], axis=1)

# Drop rows with any NaN values
data.dropna(inplace=True)

data

# Reassign features and targets after cleaning
features = data.drop(columns=[price_target.name, pe_target.name])  # Drop target columns
price_target = data[price_target.name]  # Reassign cleaned price target
pe_target = data[pe_target.name]  # Reassign cleaned P/E target

pe_target

price_target

# Train-test split
X_train, X_test, y_train_price, y_test_price = train_test_split(features, price_target, test_size=0.2, random_state=42)
_, _, y_train_pe, y_test_pe = train_test_split(features, pe_target, test_size=0.2, random_state=42)

# Modeling: Linear Regression for Price Prediction
lr_price_model = LinearRegression()
lr_price_model.fit(X_train, y_train_price)

# Random Forest Model for Price Prediction
rf_price_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_price_model.fit(X_train, y_train_price)

# Modeling: Linear Regression for P/E Ratio Prediction
lr_pe_model = LinearRegression()
lr_pe_model.fit(X_train, y_train_pe)

# Random Forest Model for P/E Ratio Prediction
rf_pe_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pe_model.fit(X_train, y_train_pe)

# Predictions and Evaluation
# Price Prediction Evaluation
lr_price_predictions = lr_price_model.predict(X_test)
lr_price_predictions

rf_price_predictions = rf_price_model.predict(X_test)
rf_price_predictions

lr_price_mae = mean_absolute_error(y_test_price, lr_price_predictions)
rf_price_mae = mean_absolute_error(y_test_price, rf_price_predictions)
print(f"Linear Regression Price MAE: {lr_price_mae}")
print(f"Random Forest Price MAE: {rf_price_mae}")

# P/E Ratio Prediction Evaluation
lr_pe_predictions = lr_pe_model.predict(X_test)
rf_pe_predictions = rf_pe_model.predict(X_test)

lr_pe_mae = mean_absolute_error(y_test_pe, lr_pe_predictions)
rf_pe_mae = mean_absolute_error(y_test_pe, rf_pe_predictions)
print(f"Linear Regression P/E MAE: {lr_pe_mae}")
print(f"Random Forest P/E MAE: {rf_pe_mae}")

# Next Day Prediction (demonstration)
next_day_features = features.iloc[-1].values.reshape(1, -1)  # Get last available row of features
lr_next_day_price_pred = lr_price_model.predict(next_day_features)
rf_next_day_price_pred = rf_price_model.predict(next_day_features)
lr_next_day_pe_pred = lr_pe_model.predict(next_day_features)
rf_next_day_pe_pred = rf_pe_model.predict(next_day_features)

print(f"Linear Regression Next-Day Price Prediction: {lr_next_day_price_pred[0]}")
print(f"Random Forest Next-Day Price Prediction: {rf_next_day_price_pred[0]}")
print(f"Linear Regression Next-Day P/E Ratio Prediction: {lr_next_day_pe_pred[0]}")
print(f"Random Forest Next-Day P/E Ratio Prediction: {rf_next_day_pe_pred[0]}")

















# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf  # For gathering stock price data

# Step 1: Data Collection
# Collecting stock price data for historical fundamentals and stock prices
def fetch_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="10y")  # 5 years of historical data
    return hist
# Example: Fetching historical stock data for Reliance Industries on NSE
stock_data1 = fetch_stock_data("HCC.NS")
print(stock_data1.tail())

# Step 2: Feature Engineering
# Example features: calculate historical returns, moving averages, and volatility
def calculate_features(df):
    df['Return'] = df['Close'].pct_change()  # Daily returns
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day moving average
    df['SMA_200'] = df['Close'].rolling(window=200).mean()  # 200-day moving average
    df['Volatility'] = df['Close'].rolling(window=50).std()  # 50-day volatility (standard deviation)
    df['P/E_Ratio'] = df['Close'] / (df['Volume'] * df['Close'])  # Simplified P/E approximation
    df = df.dropna()  # Remove rows with NaN values due to rolling calculations
    return df

# Additional check for NaN or infinite values
print("Checking for NaN values in features:")
print(features.isna().sum())
print("\nChecking for infinite values in features:")
print(np.isfinite(features).all())

print("Checking for NaN values in target:")
print(target.isna().sum())
print("\nChecking for infinite values in target:")
print(np.isfinite(target).all())

# Remove any remaining rows with NaN or infinite values in features or target
features = features.dropna()
target = target[features.index]  # Align target with cleaned features

# Ensure no infinite values remain
features = features[np.isfinite(features).all(axis=1)]
target = target[features.index]  # Align target with cleaned features

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train.shape,X_test.shape

y_train.shape,y_test.shape

# Step 4: Modeling
# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest Model for comparison
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Predictions and Evaluation
# Make predictions
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

print(lr_predictions)

print(rf_predictions)

# Evaluate the models
lr_mae = mean_absolute_error(y_test, lr_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print(f"Linear Regression MAE: {lr_mae}")
print(f"Random Forest MAE: {rf_mae}")

# Print RMSE for deeper comparison
lr_rmse = mean_squared_error(y_test, lr_predictions, squared=False)
rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Random Forest RMSE: {rf_rmse}")

# Step 6: Make a Prediction for the Next Day (for demonstration)
next_day_features = features.iloc[-1].values.reshape(1, -1)  # Get last available row of features
lr_next_day_pred = lr_model.predict(next_day_features)
rf_next_day_pred = rf_model.predict(next_day_features)

print(f"Linear Regression Next-Day Prediction: {lr_next_day_pred[0]}")
print(f"Random Forest Next-Day Prediction: {rf_next_day_pred[0]}")











