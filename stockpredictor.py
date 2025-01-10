from dotenv import load_dotenv
import requests, os
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
symbol = 'TSLA'
interval = '5min'
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={alpha_vantage_api_key}"

response = requests.get(url)
data = response.json()

time_series = data.get(f"Time Series ({interval})", {})

df = pd.DataFrame.from_dict(time_series, orient='index')
df.columns = ['open', 'high', 'low', 'close', 'volume']
df.index = pd.to_datetime(df.index)
df = df.astype(float)
df = df.sort_index()

df = df.ffill()
df = df.interpolate(method='linear')
df = df.dropna()
df = df.drop_duplicates()

# all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
# missing_dates = set(all_dates) - set(df.index)
# print("missing dates: ", missing_dates)
# print("number of missing dates: \n", )
# print(len(missing_dates))

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
print(scaled)






