import pandas as pd
import yfinance as yf

def download_stockprices(ticker, period, interval):
    data = yf.Ticker(ticker).history(period=period, interval=interval, actions=False)
    data.to_csv(f"{ticker}_historical_data.csv")

def get_stockprices_from_csv(ticker: str) -> pd.DataFrame:
    return pd.DataFrame(pd.read_csv(f"{ticker}_historical_data.csv"))