from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from config.data_settings import DataSettings
from config.model_settings import ModelSettings
from config.training_settings import TrainingSettings
import pandas as pd
import numpy as np
import torch


class TimeSeriesDataset(Dataset):
    def __init__(self, data, past_len, forecast_len, factors):
        """
        Initialize TimeSeriesDataset with sliding window
        Args:
            data: DataFrame with OHLC data
            past_len: Number of days for historical data
            forecast_len: Number of days to forecast
            factors: List of factors to use (e.g., ["Open", "High", "Low", "Close", "Volume"])

        Each item in the dataset will contain:
            x_past: Historical data of shape [past_len, len(factors)]
            x_future: Future data without Close price of shape [forecast_len, len(factors)-1]
            y: Future Close prices of shape [forecast_len]
        """
        self.data = data
        self.past_len = past_len
        self.forecast_len = forecast_len
        self.factors = factors
        # All factors except Close
        self.future_factors = [f for f in factors if f != 'Close']

        # Calculate valid start indices
        # We need past_len days for input and forecast_len days for target
        self.valid_indices = range(len(data) - (past_len + forecast_len))

    def __len__(self):
        """Returns the number of valid sequences in the dataset"""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Get a single sequence from the dataset
        Args:
            idx: Index of the sequence

        Returns:
            tuple (x_past, x_future, y) where:
                x_past: Historical data tensor of shape [past_len, num_factors]
                x_future: Future data tensor without Close of shape [forecast_len, num_factors-1]
                y: Future Close prices tensor of shape [forecast_len]

        Example:
            If past_len=60, forecast_len=30, and factors=["Open", "High", "Low", "Close", "Volume"]:
            x_past shape: [60, 5]    # 60 days of history, all 5 features
            x_future shape: [30, 4]  # 30 days of future, 4 features (no Close)
            y shape: [30]            # 30 days of future Close prices
        """
        # Get starting index for this sequence
        start_idx = self.valid_indices[idx]
        past_end_idx = start_idx + self.past_len
        future_end_idx = past_end_idx + self.forecast_len

        # Get the sequences
        past_sequence = self.data[start_idx:past_end_idx]
        future_sequence = self.data[past_end_idx:future_end_idx]

        # Create feature tensors
        x_past = torch.FloatTensor(past_sequence[self.factors].values)
        x_future = torch.FloatTensor(
            future_sequence[self.future_factors].values)
        y = torch.FloatTensor(future_sequence['Close'].values)

        return x_past, x_future, y


def load_data(data_settings: DataSettings, model_settings: ModelSettings, training_settings: TrainingSettings):
    """
    Load and prepare data for training
    Args:
        data_settings: DataSettings object containing data configuration
        model_settings: ModelSettings object containing model configuration
        training_settings: TrainingSettings object containing training configuration
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data
    data = pd.read_csv(
        f"yf_data/{data_settings.portfolio_data.tickers[0]}_ohlcv.csv")
    data.set_index('Date', inplace=True)

    # Split data into train, validation, and test sets
    # First split: separate out test set
    train_val_data, test_data = train_test_split(
        data,
        test_size=data_settings.test_ratio,
        shuffle=False  # Keep temporal order
    )

    # Second split: divide remaining data into train and validation
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=data_settings.val_ratio/(1-data_settings.test_ratio),
        shuffle=False  # Keep temporal order
    )

    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_data,
        past_len=model_settings.past_len,
        forecast_len=model_settings.forecast_len,
        factors=data_settings.factors
    )

    val_dataset = TimeSeriesDataset(
        val_data,
        past_len=model_settings.past_len,
        forecast_len=model_settings.forecast_len,
        factors=data_settings.factors
    )

    test_dataset = TimeSeriesDataset(
        test_data,
        past_len=model_settings.past_len,
        forecast_len=model_settings.forecast_len,
        factors=data_settings.factors
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_settings.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_settings.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_settings.batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
