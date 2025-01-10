from dotenv import load_dotenv
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_lr = model.predict(X_test)
lr_mse = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression Baseline MSE: {lr_mse}")

