import pandas as pd
import datetime as dt
from util import get_data


def BollingerBands(df: pd.DataFrame, symbol: str, window_size=20) -> pd.DataFrame:
    rolling_mean = df[symbol].rolling(window=window_size).mean()
    rolling_std = df[symbol].rolling(window=window_size).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    bbp = (df[symbol] - lower_band) / (upper_band - lower_band)
    return pd.DataFrame({'Price': df[symbol], 'Upper Band': upper_band, 'Lower Band': lower_band, 'BBP': bbp})


def RSI(df: pd.DataFrame, symbol: str, lookback=14) -> pd.Series:
    daily_returns = df[symbol].pct_change().dropna()
    gains = daily_returns.where(daily_returns > 0, 0.0)
    losses = -daily_returns.where(daily_returns < 0, 0.0)
    avg_gain = gains.rolling(window=lookback).mean()
    avg_loss = losses.rolling(window=lookback).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.DataFrame({'RSI': rsi})


def MACD(df: pd.DataFrame, symbol: str, short_window=12, long_window=26, signal_window=9) -> pd.DataFrame:
    ema_short = df[symbol].ewm(span=short_window, adjust=False).mean()
    ema_long = df[symbol].ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({'MACD Line': macd_line, 'Signal Line': signal_line, 'Histogram': histogram})


def Momentum(df: pd.DataFrame, symbol: str, lookback=14) -> pd.Series:
    momentum = df[symbol] / df[symbol].shift(lookback) - 1
    return pd.DataFrame({'Momentum': momentum})


def ExponentialMovingAverage(df: pd.DataFrame, symbol: str, window_size=20) -> pd.DataFrame:
    ema = df[symbol].ewm(span=window_size, adjust=False).mean()
    return pd.DataFrame({'EMA': ema})


def StochasticOscillator(df: pd.DataFrame, symbol: str, lookback=14) -> pd.DataFrame:
    low_min = df[symbol].rolling(window=lookback).min()
    high_max = df[symbol].rolling(window=lookback).max()
    k = 100 * ((df[symbol] - low_min) / (high_max - low_min))
    d = k.rolling(window=3).mean()
    return pd.DataFrame({'K': k, 'D': d})


def AverageTrueRange(df: pd.DataFrame, symbol: str, period=14) -> pd.DataFrame:
    high = df[symbol]
    low = df[symbol]
    close = df[symbol]
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return pd.DataFrame({'ATR': atr})


def OnBalanceVolume(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    volume = get_data([symbol], df.index, colname='Volume')[symbol]
    price_change = df[symbol].diff()
    obv = (volume * (price_change > 0).astype(int) -
           volume * (price_change < 0).astype(int)).cumsum()
    return pd.DataFrame({'OBV': obv})


def RateOfChange(df: pd.DataFrame, symbol: str, period=12) -> pd.DataFrame:
    roc = ((df[symbol] - df[symbol].shift(period)) / df[symbol].shift(period)) * 100
    return pd.DataFrame({'ROC': roc})


def CommodityChannelIndex(df: pd.DataFrame, symbol: str, period=20) -> pd.DataFrame:
    typical_price = df[symbol]
    moving_average = typical_price.rolling(window=period).mean()
    mean_deviation = abs(typical_price - moving_average).rolling(window=period).mean()
    cci = (typical_price - moving_average) / (0.015 * mean_deviation)
    return pd.DataFrame({'CCI': cci})


def AverageDirectionalIndex(df: pd.DataFrame, symbol: str, period=14) -> pd.DataFrame:
    high = df[symbol]
    low = df[symbol]
    close = df[symbol]
    plus_dm = high.diff()
    minus_dm = low.diff()
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return pd.DataFrame({'ADX': adx})


def plot_data(df: pd.DataFrame, title="Indicator", xlabel="Date", ylabel="Value"):
    import matplotlib.pyplot as plt
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(f"{title}.png")
    plt.close()


def plot_indicators(df: pd.DataFrame, symbol: str):
    indicators = {
        "Bollinger Bands": BollingerBands,
        "RSI": RSI,
        "MACD": MACD,
        "Momentum": Momentum,
        "Stochastic Oscillator": StochasticOscillator,
        "ATR": AverageTrueRange,
        "OBV": OnBalanceVolume,
        "ROC": RateOfChange,
        "CCI": CommodityChannelIndex,
        "ADX": AverageDirectionalIndex
    }
    for name, func in indicators.items():
        plot_data(func(df, symbol), title=f"{symbol} {name}", xlabel="Date", ylabel=name)


def run():
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    df_prices = get_data([symbol], pd.date_range(start_date, end_date)).dropna()
    plot_indicators(df_prices, symbol)


if __name__ == '__main__':
    run()
