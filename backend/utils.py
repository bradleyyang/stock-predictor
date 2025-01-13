import pandas as pd
from prophet import Prophet
import yfinance as yf

def get_stock(ticker: str, period: str, interval: str):
    try:
        data = yf.Ticker(ticker).history(
            period=period, interval=interval, actions=False)
        data.drop(axis=1, columns=['Volume', 'Open',
                  'High', 'Low'], inplace=True)
        data.rename(columns={'Close': 'y'}, inplace=True)
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'ds'}, inplace=True)
        data['ds'] = data['ds'].dt.tz_localize(None)
        data.dropna(inplace=True)
        return data
    except:
        return None

def get_predictions(periods: int, freq: str, data: pd.DataFrame):
    try:
        model = Prophet()
        model.fit(data)

        future = model.make_future_dataframe(periods=periods, freq=freq)

        forecast = model.predict(future)

        predicted_data = forecast[['ds', 'yhat']].copy()
        predicted_data["ds"] = predicted_data["ds"].dt.strftime('%Y-%m-%d')
        predicted_json = predicted_data.to_dict(orient="records")
        return predicted_json
    except:
        return None
