import pandas as pd
from prophet import Prophet
import yfinance as yf
import numpy as np
from prophet.plot import plot_plotly, plot_components_plotly
import json

def get_stock(ticker: str, period: str, interval: str):
    data = yf.Ticker(ticker).history(period=period, interval=interval, actions=False)
    data.drop(axis=1, columns=['Volume', 'Open', 'High', 'Low'], inplace=True)
    data.rename(columns={'Close': 'y'}, inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'ds'}, inplace=True)
    data['ds'] = data['ds'].dt.tz_localize(None)
    data.dropna(inplace=True)
    return data

def get_predictions(periods: int, freq: str, data: pd.DataFrame):
    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=periods, freq=freq)

    forecast = model.predict(future)

    predicted_data = forecast[['ds', 'yhat']].copy()
    predicted_data["ds"] = predicted_data["ds"].dt.strftime('%Y-%m-%d')
    predicted_json = predicted_data.to_dict(orient="records")
    return predicted_json

def get_stockprices(ticker: str, period: str, interval: str, future_periods: int, freq: str):
    data = yf.Ticker(ticker).history(period=period, interval=interval, actions=False)
    data.drop(axis=1, columns=['Volume', 'Open', 'High', 'Low'], inplace=True)
    data.rename(columns={'Close': 'y'}, inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'ds'}, inplace=True)
    data['ds'] = data['ds'].dt.tz_localize(None)
    data.dropna(inplace=True)

    m = Prophet()
    m.fit(data)

    future = m.make_future_dataframe(periods=future_periods, freq=freq)

    forecast = m.predict(future)

    predicted_data = forecast[['ds', 'yhat']].copy()
    predicted_data["ds"] = predicted_data["ds"].dt.strftime('%Y-%m-%d')
    predicted_json = predicted_data.to_dict(orient="records")
    return json.dumps(predicted_json, indent=2)

    # fig1 = m.plot(forecast)
    # fig2 = m.plot_components(forecast)

    # fig1 = plot_plotly(m, forecast)
    # fig2 = plot_components_plotly(m, forecast)
    # fig1.show()

stock_data = get_stock("AAPL", "1mo", "1d")
print(get_predictions(10, "B", stock_data))



