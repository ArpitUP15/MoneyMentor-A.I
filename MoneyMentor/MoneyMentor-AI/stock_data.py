import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import time

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
BASE_URL = "https://www.alphavantage.co/query"

# Cache for API responses to avoid rate limiting
cache = {}
cache_expiry = {}
CACHE_DURATION = 300  # 5 minutes

def get_from_cache_or_api(url, params):
    """Get data from cache or API with caching"""
    cache_key = str(params)
    current_time = time.time()
    
    # Return from cache if still valid
    if cache_key in cache and cache_key in cache_expiry:
        if current_time < cache_expiry[cache_key]:
            return cache[cache_key]
    
    # Make API request
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            print(f"API Error: {data['Error Message']}")
            return None
        
        if "Note" in data and "API call frequency" in data["Note"]:
            print(f"API Rate Limit Warning: {data['Note']}")
        
        # Cache the response
        cache[cache_key] = data
        cache_expiry[cache_key] = current_time + CACHE_DURATION
        
        return data
    except Exception as e:
        print(f"API request error: {e}")
        return None

def search_stocks(query):
    """Search for stocks by symbol or name"""
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": query,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    data = get_from_cache_or_api(BASE_URL, params)
    
    if data and "bestMatches" in data:
        results = []
        for match in data["bestMatches"]:
            results.append({
                "symbol": match["1. symbol"],
                "name": match["2. name"],
                "type": match["3. type"],
                "region": match["4. region"],
                "currency": match["8. currency"]
            })
        return results
    
    # Return empty list or demo data if using demo API key
    if ALPHA_VANTAGE_API_KEY == "demo":
        return [
            {"symbol": "AAPL", "name": "Apple Inc.", "type": "Equity", "region": "United States", "currency": "USD"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "type": "Equity", "region": "United States", "currency": "USD"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "type": "Equity", "region": "United States", "currency": "USD"},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "type": "Equity", "region": "United States", "currency": "USD"}
        ]
    
    return []

def get_stock_quote(symbol):
    """Get current stock quote with basic information"""
    # Try global quote endpoint first
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    quote_data = get_from_cache_or_api(BASE_URL, params)
    
    # Get company overview for additional data
    overview_params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    overview_data = get_from_cache_or_api(BASE_URL, overview_params)
    
    # If demo key, handle with synthetic data
    if ALPHA_VANTAGE_API_KEY == "demo" or (not quote_data or "Global Quote" not in quote_data or not quote_data["Global Quote"]):
        # For demo purposes, use the daily data to create a quote
        daily_data = get_stock_daily_data(symbol)
        if daily_data is not None and not daily_data.empty:
            last_row = daily_data.iloc[-1]
            prev_row = daily_data.iloc[-2] if len(daily_data) > 1 else last_row
            
            # Create synthetic quote
            return {
                "symbol": symbol,
                "name": get_company_name(symbol, overview_data),
                "price": last_row['close'],
                "price_change": last_row['close'] - prev_row['close'],
                "price_change_percent": ((last_row['close'] / prev_row['close']) - 1) * 100,
                "volume": int(last_row['volume']),
                "market_cap": int(last_row['close'] * get_shares_outstanding(symbol, overview_data)),
                "pe": get_pe_ratio(symbol, overview_data, last_row['close']),
                "52w_high": daily_data['high'].max(),
                "52w_low": daily_data['low'].min()
            }
        return None
    
    # Extract data from the API response
    quote = quote_data["Global Quote"]
    
    # Format quote data
    price = float(quote["05. price"])
    prev_close = float(quote["08. previous close"])
    
    formatted_quote = {
        "symbol": symbol,
        "name": get_company_name(symbol, overview_data),
        "price": price,
        "price_change": price - prev_close,
        "price_change_percent": ((price / prev_close) - 1) * 100 if prev_close > 0 else 0,
        "volume": int(float(quote["06. volume"])),
        "market_cap": get_market_cap(symbol, overview_data, price),
        "pe": get_pe_ratio(symbol, overview_data, price),
        "52w_high": get_52w_high(symbol),
        "52w_low": get_52w_low(symbol)
    }
    
    return formatted_quote

def get_company_name(symbol, overview_data=None):
    """Get company name from overview data or use symbol as fallback"""
    if overview_data and "Name" in overview_data:
        return overview_data["Name"]
    
    # Use a dictionary for commonly used symbols
    common_names = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "JPM": "JPMorgan Chase & Co.",
        "V": "Visa Inc.",
        "JNJ": "Johnson & Johnson"
    }
    
    return common_names.get(symbol, f"{symbol} Stock")

def get_market_cap(symbol, overview_data=None, current_price=None):
    """Calculate market cap from shares outstanding and price"""
    if overview_data and "MarketCapitalization" in overview_data:
        try:
            return float(overview_data["MarketCapitalization"])
        except (ValueError, TypeError):
            pass
    
    # Use current price and shares outstanding if available
    if current_price and overview_data and "SharesOutstanding" in overview_data:
        try:
            shares = float(overview_data["SharesOutstanding"])
            return current_price * shares
        except (ValueError, TypeError):
            pass
    
    # Fallback values for demo
    demo_market_caps = {
        "AAPL": 2800000000000,  # 2.8T
        "MSFT": 2700000000000,  # 2.7T
        "GOOGL": 1700000000000,  # 1.7T
        "AMZN": 1500000000000,  # 1.5T
        "META": 900000000000,   # 900B
        "TSLA": 800000000000,   # 800B
        "NVDA": 1200000000000,  # 1.2T
        "JPM": 500000000000,    # 500B
        "V": 480000000000,      # 480B
        "JNJ": 380000000000     # 380B
    }
    
    return demo_market_caps.get(symbol, 100000000000)  # Default to 100B

def get_shares_outstanding(symbol, overview_data=None):
    """Get shares outstanding from overview data or use fallback"""
    if overview_data and "SharesOutstanding" in overview_data:
        try:
            return float(overview_data["SharesOutstanding"])
        except (ValueError, TypeError):
            pass
    
    # Fallback values for demo
    demo_shares = {
        "AAPL": 16000000000,  # 16B
        "MSFT": 7500000000,   # 7.5B
        "GOOGL": 6500000000,  # 6.5B
        "AMZN": 10000000000,  # 10B
        "META": 2500000000,   # 2.5B
        "TSLA": 3200000000,   # 3.2B
        "NVDA": 2500000000,   # 2.5B
        "JPM": 3000000000,    # 3B
        "V": 2000000000,      # 2B
        "JNJ": 2600000000     # 2.6B
    }
    
    return demo_shares.get(symbol, 1000000000)  # Default to 1B

def get_pe_ratio(symbol, overview_data=None, current_price=None):
    """Get P/E ratio from overview data or calculate from price and EPS"""
    if overview_data and "PERatio" in overview_data:
        try:
            pe = float(overview_data["PERatio"])
            if pe > 0:
                return pe
        except (ValueError, TypeError):
            pass
    
    # Calculate P/E from price and EPS if available
    if current_price and overview_data and "EPS" in overview_data:
        try:
            eps = float(overview_data["EPS"])
            if eps > 0:
                return current_price / eps
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    
    # Fallback values for demo
    demo_pe = {
        "AAPL": 28.5,
        "MSFT": 32.1,
        "GOOGL": 25.3,
        "AMZN": 40.2,
        "META": 22.7,
        "TSLA": 75.3,
        "NVDA": 60.8,
        "JPM": 15.2,
        "V": 30.5,
        "JNJ": 18.9
    }
    
    return demo_pe.get(symbol, 25.0)  # Default to 25

def get_52w_high(symbol):
    """Get 52-week high price"""
    # This would ideally come from API but we'll use calculated value from daily data
    daily_data = get_stock_daily_data(symbol)
    if daily_data is not None and len(daily_data) > 0:
        # Get data from the past year
        one_year_ago = datetime.now() - timedelta(days=365)
        filtered_data = daily_data[daily_data.index >= one_year_ago]
        
        if len(filtered_data) > 0:
            return filtered_data['high'].max()
    
    # Fallback demo values
    demo_52w_high = {
        "AAPL": 198.23,
        "MSFT": 378.53,
        "GOOGL": 143.71,
        "AMZN": 170.83,
        "META": 326.20,
        "TSLA": 299.29,
        "NVDA": 502.66,
        "JPM": 159.38,
        "V": 250.06,
        "JNJ": 175.97
    }
    
    return demo_52w_high.get(symbol, 100.0)  # Default value

def get_52w_low(symbol):
    """Get 52-week low price"""
    # This would ideally come from API but we'll use calculated value from daily data
    daily_data = get_stock_daily_data(symbol)
    if daily_data is not None and len(daily_data) > 0:
        # Get data from the past year
        one_year_ago = datetime.now() - timedelta(days=365)
        filtered_data = daily_data[daily_data.index >= one_year_ago]
        
        if len(filtered_data) > 0:
            return filtered_data['low'].min()
    
    # Fallback demo values
    demo_52w_low = {
        "AAPL": 124.17,
        "MSFT": 213.43,
        "GOOGL": 85.57,
        "AMZN": 88.09,
        "META": 88.09,
        "TSLA": 101.81,
        "NVDA": 108.13,
        "JPM": 123.11,
        "V": 174.60,
        "JNJ": 144.95
    }
    
    return demo_52w_low.get(symbol, 50.0)  # Default value

def get_stock_intraday_data(symbol, interval="5min"):
    """Get intraday stock data"""
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "compact",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    data = get_from_cache_or_api(BASE_URL, params)
    
    # If demo or error, generate synthetic data
    if ALPHA_VANTAGE_API_KEY == "demo" or data is None or f"Time Series ({interval})" not in data:
        return generate_synthetic_intraday_data(symbol)
    
    # Parse the data from API response
    time_series_key = f"Time Series ({interval})"
    
    df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Rename columns and convert to float
    df.columns = [col.split(". ")[1] for col in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    return df

def get_stock_daily_data(symbol, outputsize="compact"):
    """Get daily stock data"""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    data = get_from_cache_or_api(BASE_URL, params)
    
    # If demo or error, generate synthetic data
    if ALPHA_VANTAGE_API_KEY == "demo" or data is None or "Time Series (Daily)" not in data:
        return generate_synthetic_daily_data(symbol)
    
    # Parse the data from API response
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Rename columns and convert to float
    df.columns = [col.split(". ")[1] for col in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    return df

def get_stock_weekly_data(symbol):
    """Get weekly stock data"""
    params = {
        "function": "TIME_SERIES_WEEKLY",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    data = get_from_cache_or_api(BASE_URL, params)
    
    # If demo or error, generate synthetic data
    if ALPHA_VANTAGE_API_KEY == "demo" or data is None or "Weekly Time Series" not in data:
        return generate_synthetic_weekly_data(symbol)
    
    # Parse the data from API response
    df = pd.DataFrame.from_dict(data["Weekly Time Series"], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Rename columns and convert to float
    df.columns = [col.split(". ")[1] for col in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    return df

def generate_synthetic_intraday_data(symbol):
    """Generate synthetic intraday data for demo purposes"""
    # Use base price from a dictionary for common symbols
    base_prices = {
        "AAPL": 180.0,
        "MSFT": 340.0,
        "GOOGL": 135.0,
        "AMZN": 150.0,
        "META": 300.0,
        "TSLA": 250.0,
        "NVDA": 450.0,
        "JPM": 145.0,
        "V": 230.0,
        "JNJ": 155.0
    }
    
    base_price = base_prices.get(symbol, 100.0)
    
    # Get current date and create a range of times for the current day
    today = datetime.now().date()
    market_open = datetime.combine(today, datetime.min.time().replace(hour=9, minute=30))
    market_close = datetime.combine(today, datetime.min.time().replace(hour=16, minute=0))
    
    if datetime.now() < market_open:
        # Before market open, show yesterday's data
        market_open = market_open - timedelta(days=1)
        market_close = market_close - timedelta(days=1)
    elif datetime.now() > market_close:
        # After market close but still today
        pass
    else:
        # During market hours, only show until current time
        market_close = datetime.now()
    
    # Create time range with 5-minute intervals
    time_range = pd.date_range(start=market_open, end=market_close, freq='5min')
    
    # Generate random walk prices
    np.random.seed(hash(symbol) % 10000)  # Use symbol as seed for consistent randomness
    
    # Calculate volatility based on symbol (some stocks are more volatile)
    volatility = 0.001  # Base volatility
    if symbol in ["TSLA", "NVDA", "AMZN"]:
        volatility = 0.002  # Higher volatility
    
    # Generate a random walk
    returns = np.random.normal(0, volatility, size=len(time_range))
    returns[0] = 0  # Start at 0
    price_changes = np.cumsum(returns)
    
    # Generate OHLC data
    prices = base_price * (1 + price_changes)
    
    # Create small variations for open, high, low from close
    df = pd.DataFrame(index=time_range)
    df['close'] = prices
    df['open'] = np.zeros(len(time_range))
    df['high'] = np.zeros(len(time_range))
    df['low'] = np.zeros(len(time_range))
    
    # Generate open, high, low with realistic relationships
    for i in range(len(df)):
        if i == 0:
            # First interval open is the previous day's close with small gap
            gap = np.random.normal(0, 0.002)
            df.iloc[i, df.columns.get_loc('open')] = base_price * (1 + gap)
        else:
            # Open is previous close
            df.iloc[i, df.columns.get_loc('open')] = df.iloc[i-1, df.columns.get_loc('close')]
        
        # High and low are variations of close
        high_var = abs(np.random.normal(0, 0.001))
        low_var = abs(np.random.normal(0, 0.001))
        
        close_price = df.iloc[i, df.columns.get_loc('close')]
        open_price = df.iloc[i, df.columns.get_loc('open')]
        
        # Determine high and low based on whether price went up or down
        if close_price >= open_price:  # Price went up
            df.iloc[i, df.columns.get_loc('high')] = close_price + high_var * base_price
            df.iloc[i, df.columns.get_loc('low')] = open_price - low_var * base_price
        else:  # Price went down
            df.iloc[i, df.columns.get_loc('high')] = open_price + high_var * base_price
            df.iloc[i, df.columns.get_loc('low')] = close_price - low_var * base_price
    
    # Generate volume
    avg_volume = 1000000  # Base average volume
    if symbol in ["AAPL", "TSLA", "AMZN", "MSFT"]:
        avg_volume = 5000000  # Higher volume for popular stocks
    
    # Volume has a U-shape during the day (higher at open and close)
    time_factors = np.linspace(0, 1, len(time_range))
    u_shape = 1 - 0.5 * np.sin(time_factors * np.pi)
    volume_base = avg_volume * u_shape
    
    # Add randomness to volume
    df['volume'] = volume_base * np.random.uniform(0.7, 1.3, size=len(time_range))
    df['volume'] = df['volume'].astype(int)
    
    # Ensure columns order is consistent
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    return df

def generate_synthetic_daily_data(symbol):
    """Generate synthetic daily data for demo purposes"""
    # Use base price from a dictionary for common symbols
    base_prices = {
        "AAPL": 180.0,
        "MSFT": 340.0,
        "GOOGL": 135.0,
        "AMZN": 150.0,
        "META": 300.0,
        "TSLA": 250.0,
        "NVDA": 450.0,
        "JPM": 145.0,
        "V": 230.0,
        "JNJ": 155.0
    }
    
    base_price = base_prices.get(symbol, 100.0)
    
    # Create date range for 180 days up to today
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Generate random walk prices
    np.random.seed(hash(symbol) % 10000)  # Use symbol as seed for consistent randomness
    
    # Calculate volatility based on symbol
    volatility = 0.012  # Base daily volatility
    if symbol in ["TSLA", "NVDA", "AMZN"]:
        volatility = 0.025  # Higher volatility
    
    # Add a slight upward bias
    drift = 0.0003
    if symbol in ["NVDA", "MSFT", "AAPL"]:
        drift = 0.0006  # Stronger upward trend for certain stocks
    
    # Generate a random walk with drift
    returns = np.random.normal(drift, volatility, size=len(date_range))
    price_changes = np.cumsum(returns)
    
    # Generate OHLC data
    prices = base_price * (1 + price_changes)
    
    df = pd.DataFrame(index=date_range)
    df['close'] = prices
    df['open'] = np.zeros(len(date_range))
    df['high'] = np.zeros(len(date_range))
    df['low'] = np.zeros(len(date_range))
    
    # Generate open, high, low with realistic relationships
    for i in range(len(df)):
        if i == 0:
            # First day open is slightly different from base price
            gap = np.random.normal(0, 0.005)
            df.iloc[i, df.columns.get_loc('open')] = base_price * (1 + gap)
        else:
            # Open has a small gap from previous close
            gap = np.random.normal(0, 0.005)
            df.iloc[i, df.columns.get_loc('open')] = df.iloc[i-1, df.columns.get_loc('close')] * (1 + gap)
        
        # High and low are variations from open and close
        high_var = abs(np.random.normal(0, 0.01))
        low_var = abs(np.random.normal(0, 0.01))
        
        close_price = df.iloc[i, df.columns.get_loc('close')]
        open_price = df.iloc[i, df.columns.get_loc('open')]
        
        # Determine high and low
        if close_price >= open_price:  # Price went up
            df.iloc[i, df.columns.get_loc('high')] = max(close_price, open_price) + high_var * base_price
            df.iloc[i, df.columns.get_loc('low')] = min(close_price, open_price) - low_var * base_price
        else:  # Price went down
            df.iloc[i, df.columns.get_loc('high')] = max(close_price, open_price) + high_var * base_price
            df.iloc[i, df.columns.get_loc('low')] = min(close_price, open_price) - low_var * base_price
    
    # Generate volume
    avg_volume = 20000000  # Base average volume
    if symbol in ["AAPL", "TSLA", "AMZN", "MSFT"]:
        avg_volume = 80000000  # Higher volume for popular stocks
    
    # Add some randomness to volume
    df['volume'] = avg_volume * np.random.uniform(0.5, 1.5, size=len(date_range))
    
    # Add some volume spikes for earnings days (approximately every 90 days)
    earnings_days = np.random.choice(range(len(date_range)), size=2, replace=False)
    for day in earnings_days:
        df.iloc[day, df.columns.get_loc('volume')] = df.iloc[day, df.columns.get_loc('volume')] * 3
    
    df['volume'] = df['volume'].astype(int)
    
    # Ensure columns order is consistent
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    return df

def generate_synthetic_weekly_data(symbol):
    """Generate synthetic weekly data for demo purposes"""
    # Get daily data and resample to weekly
    daily_data = generate_synthetic_daily_data(symbol)
    
    if daily_data is not None and not daily_data.empty:
        # Resample to weekly
        weekly_ohlc = daily_data.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return weekly_ohlc
    
    return None
