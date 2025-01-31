from config.data_settings import DataSettings
from utils.data_related.data_fetch import fetch_data
from utils.data_related.portfolio import Portfolio
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
import sys

project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize a series using z-score normalization
    """
    return (series - series.mean()) / series.std()


def process_ohlc_data(data: pd.DataFrame, tickers: List[str], output_dir: str, factors: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Process OHLC data and save individual stock data
    Args:
        data: Multi-index DataFrame from yfinance
        output_dir: Directory to save individual stock CSV files
        factors: List of factors to extract (e.g., ["Open", "High", "Low", "Close", "Volume"])
    Returns:
        Dictionary containing normalization statistics for each stock
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    stats = {}

    # Process each ticker
    for ticker in tickers:
        # Initialize DataFrame for this ticker
        ticker_data = pd.DataFrame(index=data.index)
        stats[ticker] = {}

        # Process each factor
        for factor in factors:
            try:
                # Get data for this ticker and factor
                series = data[factor][ticker]

                # Store statistics
                stats[ticker][factor] = {
                    'mean': series.mean(),
                    'std': series.std()
                }

                # Normalize and add to ticker data
                normalized_series = normalize_series(series)
                ticker_data[factor] = normalized_series
            except KeyError:
                print(f"Warning: Could not find {factor} data for {ticker}")
                continue

        # Save ticker data to CSV
        output_path = os.path.join(output_dir, f"{ticker}_ohlcv.csv")
        ticker_data.to_csv(output_path)
        print(f"Saved normalized data for {ticker} to {output_path}")

    return stats


def inverse_normalize(value: float, mean: float, std: float) -> float:
    """
    Inverse normalize a value using stored statistics
    """
    return (value * std) + mean


if __name__ == "__main__":
    # Get settings
    data_settings = DataSettings.default()
    portfolio = Portfolio.default()

    # Fetch data
    print("Fetching data...")
    raw_data = fetch_data(portfolio)

    # Process data
    print("Processing and normalizing data...")
    stats = process_ohlc_data(
        data=raw_data,
        tickers=portfolio.tickers,
        output_dir="yf_data",
        factors=data_settings.factors
    )

    print("Processing complete. Normalization statistics:")
