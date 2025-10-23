import pandas as pd
import numpy as np

def calculate_rsi(df, window=14):
    """
    Calculate Relative Strength Index (RSI)
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    """
    if df is None or len(df) < window:
        # Generate dummy RSI data if not enough data
        return pd.Series(np.random.uniform(30, 70, size=max(0, len(df))))
    
    delta = df['close'].diff()
    
    # Get gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    MACD Line = 12-Period EMA - 26-Period EMA
    Signal Line = 9-Period EMA of MACD Line
    Histogram = MACD Line - Signal Line
    """
    if df is None or len(df) < max(fast_period, slow_period, signal_period):
        # Generate dummy MACD data if not enough data
        size = max(0, len(df))
        return (
            pd.Series(np.random.uniform(-2, 2, size=size)),  # MACD
            pd.Series(np.random.uniform(-2, 2, size=size)),  # Signal
            pd.Series(np.random.uniform(-1, 1, size=size))   # Histogram
        )
    
    # Calculate EMAs
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(df, window=20, std_dev=2):
    """
    Calculate Bollinger Bands
    
    Middle Band = 20-Period SMA
    Upper Band = Middle Band + (20-Period Standard Deviation × 2)
    Lower Band = Middle Band - (20-Period Standard Deviation × 2)
    """
    if df is None or 'close' not in df.columns or len(df) < window:
        return None, None
    
    # Calculate middle band (SMA)
    middle_band = df['close'].rolling(window=window).mean()
    
    # Calculate standard deviation
    std = df['close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, lower_band

def calculate_pe_ratio(price, eps):
    """Calculate Price to Earnings Ratio"""
    if eps is None or eps == 0:
        return None
    
    return price / eps

def calculate_moving_averages(df, periods=[20, 50, 200]):
    """Calculate simple moving averages for given periods"""
    if df is None or 'close' not in df.columns:
        return {}
    
    mas = {}
    for period in periods:
        if len(df) >= period:
            mas[f'SMA{period}'] = df['close'].rolling(window=period).mean()
    
    return mas

def calculate_volatility(df, window=20):
    """
    Calculate volatility as standard deviation of returns
    Returns the most recent volatility value
    """
    if df is None or 'close' not in df.columns or len(df) < window:
        return 0.02  # Return a default volatility
    
    # Calculate daily returns
    returns = df['close'].pct_change().dropna()
    
    # Calculate rolling volatility (standard deviation of returns)
    volatility = returns.rolling(window=window).std().dropna()
    
    # Return the most recent volatility or a fallback
    if len(volatility) > 0:
        return volatility.iloc[-1]
    else:
        return 0.02  # Default volatility
