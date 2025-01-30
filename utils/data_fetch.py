import yfinance as yf
import pandas as pd
import os
import sys

from portfolio import Portfolio


def get_index_components(index_symbol: str) -> list:
    """
    Get component stocks of a specified index
    Supported indices: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)
    """
    try:
        index = yf.Ticker(index_symbol)
        print(index.funds_data.top_holdings)
    except Exception as e:
        print(f"Error fetching index components: {e}")
        return []


def fetch_data(portfolio: Portfolio) -> pd.DataFrame:
    """
    Fetch OHLC data for portfolio tickers and/or index components
    """

    tickers = yf.Tickers(portfolio.tickers)
    data = tickers.history(period=portfolio.period,
                           interval=portfolio.interval)
    pwd = os.getcwd()
    data_path = os.path.join(pwd, 'yf_data')
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data_file = os.path.join(data_path, 'ohlc.csv')
        data.to_csv(data_file)
    return data


if __name__ == '__main__':
    portfolio = Portfolio.default()
    index_components = get_index_components(portfolio.index)
    print(index_components)
    data = fetch_data(portfolio)
    print(data)
