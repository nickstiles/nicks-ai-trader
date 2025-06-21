import yfinance as yf
import pandas as pd

# Download S&P 500 tickers from Wikipedia
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
tickers = sp500["Symbol"].tolist()

# Some tickers use '.' instead of '-' in Yahoo syntax
tickers = [t.replace(".", "-") for t in tickers]

pd.DataFrame(tickers, columns=["ticker"]).to_csv("tickers.csv", index=False)