import pandas as pd

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