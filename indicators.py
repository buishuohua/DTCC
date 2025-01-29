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
    