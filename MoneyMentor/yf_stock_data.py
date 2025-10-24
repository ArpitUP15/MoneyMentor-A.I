import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

# Cache for API responses to avoid rate limiting
cache = {}
cache_expiry = {}
CACHE_DURATION = 300  # 5 minutes

def get_from_cache_or_api(func, *args, **kwargs):
    """Get data from cache or API with caching"""
    cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
    current_time = time.time()
    
    # Return from cache if still valid
    if cache_key in cache and cache_key in cache_expiry:
        if current_time < cache_expiry[cache_key]:
            return cache[cache_key]
    
    # Execute function
    try:
        data = func(*args, **kwargs)
        
        # Cache the response
        cache[cache_key] = data
        cache_expiry[cache_key] = current_time + CACHE_DURATION
        
        return data
    except Exception as e:
        print(f"API request error: {e}")
        return None

def search_stocks(query):
    """Search for stocks by symbol or name using yfinance"""
    if not query or len(query) < 1:
        return []
    
    results = []
    query_upper = query.upper().strip()
    
    # First, try direct ticker lookup
    try:
        ticker = yf.Ticker(query_upper)
        info = ticker.info
        
        # Check if we got valid data
        if info and len(info) > 1 and 'symbol' in info:
            results.append({
                "symbol": info['symbol'],
                "name": info.get('shortName', info.get('longName', info['symbol'])),
                "type": "Equity",
                "region": info.get('country', 'United States'),
                "currency": info.get('currency', 'USD')
            })
            return results
    except Exception as e:
        print(f"Direct lookup failed for {query_upper}: {e}")
    
    # If direct lookup fails, try with common suffixes
    suffixes = ['', '.TO', '.L', '.HK', '.T', '.PA', '.DE', '.AS', '.BR', '.V']
    
    for suffix in suffixes:
        try:
            test_symbol = query_upper + suffix
            ticker = yf.Ticker(test_symbol)
            info = ticker.info
            
            if info and len(info) > 1 and 'symbol' in info:
                results.append({
                    "symbol": info['symbol'],
                    "name": info.get('shortName', info.get('longName', info['symbol'])),
                    "type": "Equity",
                    "region": info.get('country', 'United States'),
                    "currency": info.get('currency', 'USD')
                })
                # Limit to 5 results to avoid too many API calls
                if len(results) >= 5:
                    break
        except:
            continue
    
    # If still no results, try with expanded common ticker list
    if not results:
        expanded_tickers = [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", 
            "WMT", "JNJ", "PG", "UNH", "HD", "BAC", "MA", "XOM", "PFE", "DIS", "NFLX",
            "ADBE", "CSCO", "PEP", "AVGO", "ORCL", "ACN", "NKE", "CMCSA", "CRM", "ABT",
            "TMO", "COST", "DHR", "VZ", "NEE", "LLY", "ABBV", "MRK", "TXN", "QCOM",
            "HON", "SPGI", "LOW", "UNP", "UPS", "CAT", "RTX", "IBM", "GE", "F",
            "AMD", "INTC", "PYPL", "ADP", "T", "CVX", "KO", "PEP", "WFC", "C",
            "GS", "AXP", "BLK", "SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "AGG"
        ]
        
        query_lower = query.lower()
        
        # Search by ticker symbol
        for symbol in expanded_tickers:
            if query_upper in symbol:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if info and 'symbol' in info:
                        results.append({
                            "symbol": info['symbol'],
                            "name": info.get('shortName', info.get('longName', info['symbol'])),
                            "type": "Equity",
                            "region": info.get('country', 'United States'),
                            "currency": info.get('currency', 'USD')
                        })
                        if len(results) >= 10:
                            break
                except:
                    continue
        
        # Search by company name
        for symbol in expanded_tickers:
            if symbol in [r['symbol'] for r in results]:
                continue  # Skip if already found
                
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and 'symbol' in info:
                    name = info.get('shortName', info.get('longName', ''))
                    if name and query_lower in name.lower():
                        results.append({
                            "symbol": info['symbol'],
                            "name": name,
                            "type": "Equity",
                            "region": info.get('country', 'United States'),
                            "currency": info.get('currency', 'USD')
                        })
                        if len(results) >= 10:
                            break
            except:
                continue
    
    return results[:10]  # Return max 10 results

def get_stock_quote(symbol):
    """Get current stock quote with basic information"""
    # Use a function to get data that we can cache
    def fetch_quote(symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we have valid info
            if not info or len(info) < 2:
                print(f"No valid info for {symbol}")
                return None
            
            # Get recent pricing data - try different periods
            hist = None
            for period in ["2d", "5d", "1mo"]:
                try:
                    hist = ticker.history(period=period)
                    if not hist.empty and len(hist) >= 2:
                        break
                except:
                    continue
            
            if hist is None or hist.empty:
                print(f"No historical data for {symbol}")
                return None
            
            # Current and previous day
            current = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else current
            
            # Calculate price change
            price = float(current['Close'])
            prev_close = float(prev['Close'])
            price_change = price - prev_close
            price_change_percent = (price / prev_close - 1) * 100 if prev_close > 0 else 0
            
            # Get market cap
            market_cap = info.get('marketCap', 0)
            if market_cap is None:
                market_cap = 0
            
            # Get 52-week high/low from historical data if not in info
            hist_52w_high = hist['High'].max() if len(hist) > 0 else price
            hist_52w_low = hist['Low'].min() if len(hist) > 0 else price
            
            # Format quote data
            return {
                "symbol": symbol,
                "name": info.get('shortName', info.get('longName', symbol)),
                "price": price,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "volume": int(current['Volume']) if 'Volume' in current else 0,
                "market_cap": market_cap,
                "pe": info.get('trailingPE', info.get('forwardPE', None)),
                "52w_high": info.get('fiftyTwoWeekHigh', hist_52w_high),
                "52w_low": info.get('fiftyTwoWeekLow', hist_52w_low)
            }
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None
    
    return get_from_cache_or_api(fetch_quote, symbol)

def get_company_name(symbol, overview_data=None):
    """Get company name from overview data or use symbol as fallback"""
    if overview_data and 'name' in overview_data:
        return overview_data['name']
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('shortName', info.get('longName', symbol))
    except:
        # Fallback
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

def get_stock_intraday_data(symbol, interval="5m"):
    """Get intraday stock data"""
    # Map interval to yfinance format
    yf_interval = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "60min": "60m"
    }.get(interval, "5m")
    
    # Use a function to get data that we can cache
    def fetch_intraday(symbol, interval):
        try:
            ticker = yf.Ticker(symbol)
            # Get data for last day with specified interval
            df = ticker.history(period="1d", interval=interval)
            
            if df.empty:
                return None
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df
        except Exception as e:
            print(f"Error getting intraday data for {symbol}: {e}")
            return None
    
    return get_from_cache_or_api(fetch_intraday, symbol, yf_interval)

def get_stock_daily_data(symbol, period="1y"):
    """Get daily stock data"""
    # Use a function to get data that we can cache
    def fetch_daily(symbol, period):
        try:
            ticker = yf.Ticker(symbol)
            
            # Try different periods if the requested one fails
            periods_to_try = [period, "1y", "6mo", "3mo", "1mo"]
            df = None
            
            for p in periods_to_try:
                try:
                    df = ticker.history(period=p, interval="1d")
                    if not df.empty:
                        break
                except:
                    continue
            
            if df is None or df.empty:
                print(f"No daily data available for {symbol}")
                return None
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df
        except Exception as e:
            print(f"Error getting daily data for {symbol}: {e}")
            return None
    
    return get_from_cache_or_api(fetch_daily, symbol, period)

def get_stock_weekly_data(symbol):
    """Get weekly stock data"""
    # Use a function to get data that we can cache
    def fetch_weekly(symbol):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y", interval="1wk")
            
            if df.empty:
                return None
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df
        except Exception as e:
            print(f"Error getting weekly data for {symbol}: {e}")
            return None
    
    return get_from_cache_or_api(fetch_weekly, symbol)