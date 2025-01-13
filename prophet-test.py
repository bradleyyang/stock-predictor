import pandas as pd
from prophet import Prophet
import yfinance as yf
import numpy as np
# import matplotlib.pyplot as plt
from prophet.plot import plot_plotly, plot_components_plotly


data = yf.Ticker('BMO').history(period='max', interval="1mo", actions=False)
data.drop(axis=1, columns=['Volume', 'Open', 'High', 'Low'], inplace=True)
data.rename(columns={'Close': 'y'}, inplace=True)
data.reset_index(inplace=True)
data.rename(columns={'Date': 'ds'}, inplace=True)
data['ds'] = data['ds'].dt.tz_localize(None)
data.dropna(inplace=True)

m = Prophet()
m.fit(data)

future = m.make_future_dataframe(periods=24, freq='M')

forecast = m.predict(future)

# fig1 = m.plot(forecast)
# fig2 = m.plot_components(forecast)

fig1 = plot_plotly(m, forecast)
fig2 = plot_components_plotly(m, forecast)
fig1.show()



