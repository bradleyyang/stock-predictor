import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def get_stockprices(ticker):
    return yf.Ticker(ticker).history(period='5d', interval="1m", actions=False)

stockprices = get_stockprices("AAPL")

scaler = preprocessing.MinMaxScaler()
stockprices.drop(axis=1, columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
columns_to_scale = ['Close']
stockprices_scaled = pd.DataFrame(scaler.fit_transform(stockprices[['Close']]), columns=['Close'], index=stockprices.index)

stockprices_scaled['Target'] = stockprices_scaled['Close'].shift(-1)
stockprices_scaled = stockprices_scaled.dropna()

X = stockprices_scaled[['Close']]
y = stockprices_scaled['Target']

train_ratio = 0.8
train_size = int(len(stockprices_scaled) * train_ratio)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)
print(f"Linear Regression Baseline MSE: {lr_mse}")
print(f"Linear regression R-squared value is: {lr_r2}")

y_pred_lr = np.array(y_pred_lr).reshape(-1, 1)
y_pred_unscaled = scaler.inverse_transform(y_pred_lr)
y_test = np.array(y_test).reshape(-1, 1)
y_test_unscaled = scaler.inverse_transform(y_test)
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
print(f"stock price predictions mse: {mse}")