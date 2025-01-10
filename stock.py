from dotenv import load_dotenv
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

load_dotenv()

def get_stock_history(ticker):
    return yf.Ticker(ticker).history(start="2000-01-01", end="2025-01-01", interval="1d", actions=False)

history = get_stock_history("AAPL")

scaler = preprocessing.MinMaxScaler()
columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']
history_scaled = pd.DataFrame(scaler.fit_transform(history[columns_to_scale]), columns=columns_to_scale, index=history.index)

history_scaled['Target'] = history_scaled['Close'].shift(-1)
history_scaled = history_scaled.dropna()

X = history_scaled.drop(columns=['Target'])
y = history_scaled['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_lr = model.predict(X_test)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)
print(f"Linear Regression Baseline MSE: {lr_mse}")
print(f"Linear regression R-squared value is: {lr_r2}")

# Random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
print(f"Random forest mse: {rf_mse}")



# Training the model


# Visualizing
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred_lr, label="linear regression predicted model", linestyle="--")
plt.legend()
plt.title("Model Predictions vs. Actual")
plt.show()