import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_indian_rsi(df, period=14):
    """
    Calculate RSI for Indian stocks with market-specific adjustments
    """
    if len(df) < period:
        return pd.Series(index=df.index, dtype=float)
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use exponential moving average for Indian markets (more responsive)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_indian_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD for Indian stocks with adjusted parameters
    """
    if len(df) < slow:
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)
    
    # Use exponential moving averages
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def calculate_indian_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands for Indian stocks
    """
    if len(df) < period:
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)
    
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, lower_band

def calculate_indian_volume_profile(df, period=20):
    """
    Calculate volume profile indicators for Indian markets
    """
    if len(df) < period:
        return pd.Series(index=df.index, dtype=float)
    
    # Volume moving average
    volume_ma = df['volume'].rolling(window=period).mean()
    
    # Volume ratio (current volume vs average)
    volume_ratio = df['volume'] / volume_ma
    
    return volume_ratio

def calculate_indian_momentum_indicators(df):
    """
    Calculate momentum indicators specific to Indian market behavior
    """
    indicators = {}
    
    # Price momentum (5-day, 10-day, 20-day)
    indicators['momentum_5'] = df['close'].pct_change(5) * 100
    indicators['momentum_10'] = df['close'].pct_change(10) * 100
    indicators['momentum_20'] = df['close'].pct_change(20) * 100
    
    # Rate of Change (ROC)
    indicators['roc_10'] = ((df['close'] / df['close'].shift(10)) - 1) * 100
    indicators['roc_20'] = ((df['close'] / df['close'].shift(20)) - 1) * 100
    
    # Price relative to moving averages
    sma_20 = df['close'].rolling(window=20).mean()
    sma_50 = df['close'].rolling(window=50).mean()
    
    indicators['price_sma20'] = (df['close'] / sma_20 - 1) * 100
    indicators['price_sma50'] = (df['close'] / sma_50 - 1) * 100
    
    return indicators

def calculate_indian_volatility_indicators(df, period=20):
    """
    Calculate volatility indicators for Indian markets
    """
    indicators = {}
    
    # Historical volatility
    returns = df['close'].pct_change()
    indicators['volatility'] = returns.rolling(window=period).std() * np.sqrt(252) * 100
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    indicators['atr'] = true_range.rolling(window=period).mean()
    
    # Volatility ratio
    short_vol = returns.rolling(window=10).std() * np.sqrt(252) * 100
    long_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
    indicators['vol_ratio'] = short_vol / long_vol
    
    return indicators

def calculate_indian_sector_indicators(df, sector):
    """
    Calculate sector-specific indicators for Indian stocks
    """
    indicators = {}
    
    # Banking sector specific indicators
    if sector in ['Banking', 'Financial Services', 'NBFC']:
        # Interest rate sensitivity
        indicators['rate_sensitivity'] = df['close'].rolling(window=20).corr(
            pd.Series(range(len(df)), index=df.index)  # Dummy rate proxy
        )
        
        # Credit growth proxy (volume-based)
        indicators['credit_growth'] = df['volume'].rolling(window=20).mean()
    
    # IT sector specific indicators
    elif sector in ['IT', 'Technology']:
        # Dollar sensitivity (using volume as proxy)
        indicators['dollar_sensitivity'] = df['volume'].rolling(window=20).std()
        
        # Client concentration risk (price volatility)
        indicators['client_risk'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    # Pharma sector specific indicators
    elif sector in ['Pharma', 'Healthcare']:
        # Regulatory risk (volatility-based)
        indicators['regulatory_risk'] = df['close'].rolling(window=20).std()
        
        # Pipeline strength (momentum-based)
        indicators['pipeline_strength'] = df['close'].pct_change(20)
    
    # FMCG sector specific indicators
    elif sector in ['FMCG', 'Consumer Goods']:
        # Consumer sentiment (volume-based)
        indicators['consumer_sentiment'] = df['volume'].rolling(window=20).mean()
        
        # Price elasticity (price-volume relationship)
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        indicators['price_elasticity'] = price_change.rolling(window=20).corr(volume_change)
    
    # Auto sector specific indicators
    elif sector in ['Automobile', 'Auto']:
        # Seasonal factors (month-based)
        df['month'] = df.index.month
        indicators['seasonal_factor'] = df.groupby('month')['close'].transform('mean')
        
        # Raw material sensitivity (volatility-based)
        indicators['raw_material_sensitivity'] = df['close'].rolling(window=20).std()
    
    # Energy sector specific indicators
    elif sector in ['Oil & Gas', 'Power', 'Energy']:
        # Energy price sensitivity (using price momentum)
        indicators['energy_sensitivity'] = df['close'].pct_change(5)
        
        # Government policy impact (volatility-based)
        indicators['policy_impact'] = df['close'].rolling(window=20).std()
    
    return indicators

def calculate_indian_market_sentiment(df):
    """
    Calculate market sentiment indicators for Indian stocks
    """
    indicators = {}
    
    # Advance-Decline ratio proxy (using price momentum)
    indicators['advance_decline'] = (df['close'] > df['close'].shift(1)).rolling(window=20).mean()
    
    # Market breadth (using volume)
    indicators['market_breadth'] = (df['volume'] > df['volume'].rolling(window=20).mean()).rolling(window=20).mean()
    
    # Fear-Greed index (using volatility and momentum)
    volatility = df['close'].rolling(window=20).std()
    momentum = df['close'].pct_change(20)
    
    # Normalize to 0-100 scale
    vol_norm = (volatility - volatility.rolling(window=50).min()) / (volatility.rolling(window=50).max() - volatility.rolling(window=50).min())
    mom_norm = (momentum - momentum.rolling(window=50).min()) / (momentum.rolling(window=50).max() - momentum.rolling(window=50).min())
    
    indicators['fear_greed'] = (1 - vol_norm + mom_norm) * 50  # 0-100 scale
    
    return indicators

def get_indian_stock_analysis(df, symbol, sector=None):
    """
    Get comprehensive analysis for Indian stocks
    """
    if df is None or len(df) < 50:
        return None
    
    analysis = {}
    
    # Basic technical indicators
    analysis['rsi'] = calculate_indian_rsi(df)
    analysis['macd'], analysis['macd_signal'], analysis['macd_histogram'] = calculate_indian_macd(df)
    analysis['bb_upper'], analysis['bb_lower'] = calculate_indian_bollinger_bands(df)
    analysis['volume_ratio'] = calculate_indian_volume_profile(df)
    
    # Momentum indicators
    momentum = calculate_indian_momentum_indicators(df)
    analysis.update(momentum)
    
    # Volatility indicators
    volatility = calculate_indian_volatility_indicators(df)
    analysis.update(volatility)
    
    # Market sentiment
    sentiment = calculate_indian_market_sentiment(df)
    analysis.update(sentiment)
    
    # Sector-specific indicators
    if sector:
        sector_indicators = calculate_indian_sector_indicators(df, sector)
        analysis.update(sector_indicators)
    
    # Generate signals
    analysis['signals'] = generate_indian_signals(analysis, df)
    
    return analysis

def generate_indian_signals(analysis, df):
    """
    Generate trading signals based on Indian market analysis
    """
    signals = {}
    
    # RSI signals
    rsi = analysis['rsi']
    signals['rsi_oversold'] = rsi < 30
    signals['rsi_overbought'] = rsi > 70
    signals['rsi_bullish'] = (rsi > 50) & (rsi.shift(1) <= 50)
    signals['rsi_bearish'] = (rsi < 50) & (rsi.shift(1) >= 50)
    
    # MACD signals
    macd = analysis['macd']
    macd_signal = analysis['macd_signal']
    signals['macd_bullish'] = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
    signals['macd_bearish'] = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))
    
    # Bollinger Bands signals
    bb_upper = analysis['bb_upper']
    bb_lower = analysis['bb_lower']
    signals['bb_breakout'] = df['close'] > bb_upper
    signals['bb_breakdown'] = df['close'] < bb_lower
    signals['bb_squeeze'] = (bb_upper - bb_lower) < (bb_upper - bb_lower).rolling(window=20).mean() * 0.8
    
    # Volume signals
    volume_ratio = analysis['volume_ratio']
    signals['volume_surge'] = volume_ratio > 2.0
    signals['volume_dry'] = volume_ratio < 0.5
    
    # Momentum signals
    momentum_5 = analysis['momentum_5']
    momentum_20 = analysis['momentum_20']
    signals['momentum_bullish'] = (momentum_5 > 0) & (momentum_20 > 0)
    signals['momentum_bearish'] = (momentum_5 < 0) & (momentum_20 < 0)
    
    # Volatility signals
    volatility = analysis['volatility']
    signals['high_volatility'] = volatility > volatility.rolling(window=50).quantile(0.8)
    signals['low_volatility'] = volatility < volatility.rolling(window=50).quantile(0.2)
    
    return signals

def get_indian_stock_summary(analysis, df, symbol):
    """
    Get a summary of Indian stock analysis
    """
    if not analysis:
        return None
    
    summary = {
        'symbol': symbol,
        'current_price': df['close'].iloc[-1],
        'price_change': df['close'].iloc[-1] - df['close'].iloc[-2],
        'price_change_pct': ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100,
        'volume': df['volume'].iloc[-1],
        'volume_ratio': analysis['volume_ratio'].iloc[-1],
        'rsi': analysis['rsi'].iloc[-1],
        'macd': analysis['macd'].iloc[-1],
        'volatility': analysis['volatility'].iloc[-1],
        'momentum_5': analysis['momentum_5'].iloc[-1],
        'momentum_20': analysis['momentum_20'].iloc[-1],
        'fear_greed': analysis['fear_greed'].iloc[-1] if 'fear_greed' in analysis else 50,
        'signals': analysis['signals']
    }
    
    return summary
