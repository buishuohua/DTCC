import pandas as pd
import datetime as dt
# local imports
from util import get_data


def MACD(df: pd.DataFrame, symbol: str, short_window=12, long_window=26, signal_window=9)->pd.DataFrame:
    # Calculate the short / long term exponential moving average (short_window EMA)
    short_ema = df[symbol].ewm(span=short_window, adjust=False).mean()
    long_ema = df[symbol].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal
    buy_signal = (macd.shift(1) < signal.shift(1)) & (macd > signal)
    sell_signal = (macd.shift(1) > signal.shift(1)) & (macd < signal)
    df_macd = pd.DataFrame({'MACD': macd, 'Signal Line': signal, 'Histogram': histogram, 'Buy Signal': buy_signal, 'Sell Signal': sell_signal})

    return df_macd

def BollingerBands(df: pd.DataFrame, symbol: str, window=20, num_std=2) -> pd.DataFrame:
    rolling_mean = df[symbol].rolling(window=window).mean()
    rolling_std = df[symbol].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    bbp = (df[symbol] - lower_band) / (upper_band - lower_band)

    # Create a DataFrame that contains the original price, rolling mean, upper band, lower band, and signals
    df_bollinger = pd.DataFrame({
        str(symbol): df[symbol],
        'Rolling Mean': rolling_mean,
        'Upper Band': upper_band,
        'Lower Band': lower_band,
        'Signal': bbp
    })

    return df_bollinger

def RSI(df: pd.DataFrame, window=14)->pd.DataFrame:
    # Calculate daily returns
    daily_returns = df.diff()
    # Calculate up days and down days
    up_days = daily_returns.where(daily_returns > 0, 0)
    down_days = -daily_returns.where(daily_returns < 0, 0)
    # Calculate average gain and average loss
    avg_gain = up_days.rolling(window=window).mean()
    avg_loss = down_days.rolling(window=window).mean()
    # Calculate relative strength
    rs = avg_gain / (avg_loss + 1e-10)
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi


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


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):  		  	   		 	   		  		  		    	 		 		   		 		  
    import matplotlib.pyplot as plt  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    """Plot stock prices with a custom title and meaningful axis labels."""  		  	   		 	   		  		  		    	 		 		   		 		  
    ax = df.plot(title=title, fontsize=12)  		  	   		 	   		  		  		    	 		 		   		 		  
    ax.set_xlabel(xlabel)  		  	   		 	   		  		  		    	 		 		   		 		  
    ax.set_ylabel(ylabel)  	
    plt.savefig(f"{title}.png")	  	   		 	   		  		  		    	 		 		   		 		  
    # plt.show()  
    plt.close()

def plot_indicators(df_prices: pd.DataFrame, symbol: str):
    plot_data(BollingerBands(df_prices, symbol), title=f"{symbol} BollingerBands", xlabel="Date", ylabel="Value")
    plot_data(MACD(df_prices, symbol), title= f"{symbol} MACD", xlabel="Date", ylabel="Value")
    plot_data(RSI(df_prices), title= f"{symbol} RSI", xlabel="Date", ylabel="Value")

def run():
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    df_prices = get_data([symbol], pd.date_range(start_date, end_date), addSPY=False).dropna(axis=0)
    plot_indicators(df_prices, symbol)
    
if __name__ == '__main__':
    run()
    