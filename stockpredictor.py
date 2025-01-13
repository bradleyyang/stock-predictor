import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from tensorflow.python.keras import Sequential
from keras.src.layers import Dense, Dropout, LSTM


def stochastic_oscillator(data: pd.DataFrame, period=14):
    highs = data['High']
    lows = data['Low']
    closes = data['Close']
    highs_max = highs.rolling(window=period).max()
    lows_min = lows.rolling(window=period).min()
    fast_k = 100 * ((closes - lows_min) / (highs_max - lows_min))
    slow_k = fast_k.rolling(window=3).mean()
    return fast_k, slow_k


def calculate_macd(prices: pd.Series, short_window=12, long_window=26, signal_window=9):
    short_ema = prices.ewm(
        span=short_window, min_periods=short_window, adjust=False).mean()
    long_ema = prices.ewm(
        span=long_window, min_periods=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(
        span=signal_window, min_periods=signal_window, adjust=False).mean()
    return macd_line, signal_line


def calculate_rsi(prices: pd.Series, period=14):
    delta = prices.diff()
    delta = delta.dropna()
    up, down = delta.clip(lower=0), delta.clip(upper=0, lower=None)
    ema_up = up.ewm(alpha=1/period, min_periods=period).mean()
    ema_down = down.abs().ewm(alpha=1/period, min_periods=period).mean()
    rs = ema_up / ema_down
    rsi = 100 - 100 / (1 + rs)
    return rsi


def get_stockprices(ticker):
    return yf.Ticker(ticker).history(period='10y', interval="1wk", actions=False)

# ================================================================================


stockprices = get_stockprices("BMO")
stockprices = stockprices.astype(float)

# Calculating features
stockprices['CMA'] = stockprices['Close'].expanding().mean()
stockprices['EMA'] = stockprices['Close'].ewm(span=5).mean()
stockprices['RSI'] = calculate_rsi(stockprices['Close'])
macd_line, signal_line = calculate_macd(stockprices['Close'])
stockprices['MACD'] = macd_line
stockprices['signal_line'] = signal_line
fast_k, slow_k = stochastic_oscillator(stockprices)
stockprices['fast_k'] = fast_k
stockprices['slow_k'] = slow_k

stockprices['Close_Lagged'] = stockprices['Close'].shift(1)
stockprices['Return'] = stockprices['Close'] / stockprices['Close_Lagged'] - 1
stockprices['ROC'] = stockprices['Close'].pct_change(periods=5)

stockprices['Target'] = stockprices['Close'].shift(-1)
stockprices.dropna(inplace=True)
features = ['High', 'Low', 'Close', 'Open', 'RSI', 'MACD', 'Volume', 'CMA',
            'EMA', 'signal_line', 'fast_k', 'slow_k', 'Close_Lagged', 'Return', 'ROC']
# features = ['High', 'Low', 'Close', 'Open', 'Volume', 'Close_Lagged', 'Return', 'ROC']

train_ratio = 0.8
train_size = int(len(stockprices[features]) * train_ratio)
X = stockprices[features]
y = stockprices['Target']
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

x_scaler = preprocessing.MinMaxScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = preprocessing.MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))


baseline_model = RandomForestRegressor()
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_baseline)
r2 = r2_score(y_test, y_pred_baseline)
print(f"MSE: {mse}")
print(f"r2: {r2}")

y_pred_baseline_series = pd.Series(y_pred_baseline, index=y_test.index)
y_pred_baseline_series.plot(label="predicted")
y_test.plot(label="actual")
plt.legend()
plt.show()

# LSTM
