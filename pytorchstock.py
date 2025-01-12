import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker = "AAPL"
stockprices = yf.Ticker(ticker).history(period="1d", interval="1m", actions=False)
stockprices['Target'] = stockprices['Close'].shift(-1)
stockprices = stockprices.dropna()

