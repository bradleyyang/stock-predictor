from dotenv import load_dotenv
import requests, os
import pandas as pd

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
df = df.astype(float)
df = df.sort_index()

print(df)