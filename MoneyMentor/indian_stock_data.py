import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests
import json

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

# Indian stock market data
INDIAN_STOCKS = {
    # NSE Large Cap Stocks
    "RELIANCE": {"name": "Reliance Industries Ltd", "sector": "Oil & Gas", "symbol": "RELIANCE.NS"},
    "TCS": {"name": "Tata Consultancy Services Ltd", "sector": "IT", "symbol": "TCS.NS"},
    "HDFCBANK": {"name": "HDFC Bank Ltd", "sector": "Banking", "symbol": "HDFCBANK.NS"},
    "INFY": {"name": "Infosys Ltd", "sector": "IT", "symbol": "INFY.NS"},
    "HINDUNILVR": {"name": "Hindustan Unilever Ltd", "sector": "FMCG", "symbol": "HINDUNILVR.NS"},
    "ITC": {"name": "ITC Ltd", "sector": "FMCG", "symbol": "ITC.NS"},
    "SBIN": {"name": "State Bank of India", "sector": "Banking", "symbol": "SBIN.NS"},
    "BHARTIARTL": {"name": "Bharti Airtel Ltd", "sector": "Telecom", "symbol": "BHARTIARTL.NS"},
    "KOTAKBANK": {"name": "Kotak Mahindra Bank Ltd", "sector": "Banking", "symbol": "KOTAKBANK.NS"},
    "LT": {"name": "Larsen & Toubro Ltd", "sector": "Engineering", "symbol": "LT.NS"},
    "ASIANPAINT": {"name": "Asian Paints Ltd", "sector": "Paints", "symbol": "ASIANPAINT.NS"},
    "MARUTI": {"name": "Maruti Suzuki India Ltd", "sector": "Automobile", "symbol": "MARUTI.NS"},
    "AXISBANK": {"name": "Axis Bank Ltd", "sector": "Banking", "symbol": "AXISBANK.NS"},
    "NESTLEIND": {"name": "Nestle India Ltd", "sector": "FMCG", "symbol": "NESTLEIND.NS"},
    "ULTRACEMCO": {"name": "UltraTech Cement Ltd", "sector": "Cement", "symbol": "ULTRACEMCO.NS"},
    "TITAN": {"name": "Titan Company Ltd", "sector": "Consumer Goods", "symbol": "TITAN.NS"},
    "POWERGRID": {"name": "Power Grid Corporation of India Ltd", "sector": "Power", "symbol": "POWERGRID.NS"},
    "NTPC": {"name": "NTPC Ltd", "sector": "Power", "symbol": "NTPC.NS"},
    "SUNPHARMA": {"name": "Sun Pharmaceutical Industries Ltd", "sector": "Pharma", "symbol": "SUNPHARMA.NS"},
    "TECHM": {"name": "Tech Mahindra Ltd", "sector": "IT", "symbol": "TECHM.NS"},
    "WIPRO": {"name": "Wipro Ltd", "sector": "IT", "symbol": "WIPRO.NS"},
    "HCLTECH": {"name": "HCL Technologies Ltd", "sector": "IT", "symbol": "HCLTECH.NS"},
    "ONGC": {"name": "Oil and Natural Gas Corporation Ltd", "sector": "Oil & Gas", "symbol": "ONGC.NS"},
    "COALINDIA": {"name": "Coal India Ltd", "sector": "Mining", "symbol": "COALINDIA.NS"},
    "TATAMOTORS": {"name": "Tata Motors Ltd", "sector": "Automobile", "symbol": "TATAMOTORS.NS"},
    "BAJFINANCE": {"name": "Bajaj Finance Ltd", "sector": "NBFC", "symbol": "BAJFINANCE.NS"},
    "BAJAJFINSV": {"name": "Bajaj Finserv Ltd", "sector": "Financial Services", "symbol": "BAJAJFINSV.NS"},
    "DRREDDY": {"name": "Dr. Reddy's Laboratories Ltd", "sector": "Pharma", "symbol": "DRREDDY.NS"},
    "CIPLA": {"name": "Cipla Ltd", "sector": "Pharma", "symbol": "CIPLA.NS"},
    "APOLLOHOSP": {"name": "Apollo Hospitals Enterprise Ltd", "sector": "Healthcare", "symbol": "APOLLOHOSP.NS"},
    "DIVISLAB": {"name": "Divi's Laboratories Ltd", "sector": "Pharma", "symbol": "DIVISLAB.NS"},
    "EICHERMOT": {"name": "Eicher Motors Ltd", "sector": "Automobile", "symbol": "EICHERMOT.NS"},
    "HEROMOTOCO": {"name": "Hero MotoCorp Ltd", "sector": "Automobile", "symbol": "HEROMOTOCO.NS"},
    "BAJAJ-AUTO": {"name": "Bajaj Auto Ltd", "sector": "Automobile", "symbol": "BAJAJ-AUTO.NS"},
    "M&M": {"name": "Mahindra & Mahindra Ltd", "sector": "Automobile", "symbol": "M&M.NS"},
    "TATACONSUM": {"name": "Tata Consumer Products Ltd", "sector": "FMCG", "symbol": "TATACONSUM.NS"},
    "BRITANNIA": {"name": "Britannia Industries Ltd", "sector": "FMCG", "symbol": "BRITANNIA.NS"},
    "DABUR": {"name": "Dabur India Ltd", "sector": "FMCG", "symbol": "DABUR.NS"},
    "GODREJCP": {"name": "Godrej Consumer Products Ltd", "sector": "FMCG", "symbol": "GODREJCP.NS"},
    "PIDILITIND": {"name": "Pidilite Industries Ltd", "sector": "Chemicals", "symbol": "PIDILITIND.NS"},
    "ADANIPORTS": {"name": "Adani Ports and Special Economic Zone Ltd", "sector": "Infrastructure", "symbol": "ADANIPORTS.NS"},
    "ADANIENT": {"name": "Adani Enterprises Ltd", "sector": "Diversified", "symbol": "ADANIENT.NS"},
    "ADANIGREEN": {"name": "Adani Green Energy Ltd", "sector": "Power", "symbol": "ADANIGREEN.NS"},
    "ADANITRANS": {"name": "Adani Transmission Ltd", "sector": "Power", "symbol": "ADANITRANS.NS"},
    "GRASIM": {"name": "Grasim Industries Ltd", "sector": "Textiles", "symbol": "GRASIM.NS"},
    "JSWSTEEL": {"name": "JSW Steel Ltd", "sector": "Steel", "symbol": "JSWSTEEL.NS"},
    "TATASTEEL": {"name": "Tata Steel Ltd", "sector": "Steel", "symbol": "TATASTEEL.NS"},
    "HINDALCO": {"name": "Hindalco Industries Ltd", "sector": "Metals", "symbol": "HINDALCO.NS"},
    "VEDL": {"name": "Vedanta Ltd", "sector": "Mining", "symbol": "VEDL.NS"},
    "JINDALSTEL": {"name": "Jindal Steel & Power Ltd", "sector": "Steel", "symbol": "JINDALSTEL.NS"},
    "SAIL": {"name": "Steel Authority of India Ltd", "sector": "Steel", "symbol": "SAIL.NS"},
    "BHEL": {"name": "Bharat Heavy Electricals Ltd", "sector": "Engineering", "symbol": "BHEL.NS"},
    "SIEMENS": {"name": "Siemens Ltd", "sector": "Engineering", "symbol": "SIEMENS.NS"},
    "ABBOTINDIA": {"name": "Abbott India Ltd", "sector": "Pharma", "symbol": "ABBOTINDIA.NS"},
    "BIOCON": {"name": "Biocon Ltd", "sector": "Pharma", "symbol": "BIOCON.NS"},
    "LUPIN": {"name": "Lupin Ltd", "sector": "Pharma", "symbol": "LUPIN.NS"},
    "CADILAHC": {"name": "Cadila Healthcare Ltd", "sector": "Pharma", "symbol": "CADILAHC.NS"},
    "TORNTPHARM": {"name": "Torrent Pharmaceuticals Ltd", "sector": "Pharma", "symbol": "TORNTPHARM.NS"},
    "AUROPHARMA": {"name": "Aurobindo Pharma Ltd", "sector": "Pharma", "symbol": "AUROPHARMA.NS"},
    "GLENMARK": {"name": "Glenmark Pharmaceuticals Ltd", "sector": "Pharma", "symbol": "GLENMARK.NS"},
    "MINDTREE": {"name": "Mindtree Ltd", "sector": "IT", "symbol": "MINDTREE.NS"},
    "LTI": {"name": "Larsen & Toubro Infotech Ltd", "sector": "IT", "symbol": "LTI.NS"},
    "MPHASIS": {"name": "Mphasis Ltd", "sector": "IT", "symbol": "MPHASIS.NS"},
    "PERSISTENT": {"name": "Persistent Systems Ltd", "sector": "IT", "symbol": "PERSISTENT.NS"},
    "COFORGE": {"name": "Coforge Ltd", "sector": "IT", "symbol": "COFORGE.NS"},
    "MINDTREE": {"name": "Mindtree Ltd", "sector": "IT", "symbol": "MINDTREE.NS"},
    "LALPATHLAB": {"name": "Dr. Lal PathLabs Ltd", "sector": "Healthcare", "symbol": "LALPATHLAB.NS"},
    "FORTIS": {"name": "Fortis Healthcare Ltd", "sector": "Healthcare", "symbol": "FORTIS.NS"},
    "MAXHEALTH": {"name": "Max Healthcare Institute Ltd", "sector": "Healthcare", "symbol": "MAXHEALTH.NS"},
    "NARAYANK": {"name": "Narayana Hrudayalaya Ltd", "sector": "Healthcare", "symbol": "NARAYANK.NS"},
    "METROPOLIS": {"name": "Metropolis Healthcare Ltd", "sector": "Healthcare", "symbol": "METROPOLIS.NS"},
    "REDINGTON": {"name": "Redington India Ltd", "sector": "Technology", "symbol": "REDINGTON.NS"},
    "TCS": {"name": "Tata Consultancy Services Ltd", "sector": "IT", "symbol": "TCS.NS"},
    "INFY": {"name": "Infosys Ltd", "sector": "IT", "symbol": "INFY.NS"},
    "WIPRO": {"name": "Wipro Ltd", "sector": "IT", "symbol": "WIPRO.NS"},
    "HCLTECH": {"name": "HCL Technologies Ltd", "sector": "IT", "symbol": "HCLTECH.NS"},
    "TECHM": {"name": "Tech Mahindra Ltd", "sector": "IT", "symbol": "TECHM.NS"},
    "MINDTREE": {"name": "Mindtree Ltd", "sector": "IT", "symbol": "MINDTREE.NS"},
    "LTI": {"name": "Larsen & Toubro Infotech Ltd", "sector": "IT", "symbol": "LTI.NS"},
    "MPHASIS": {"name": "Mphasis Ltd", "sector": "IT", "symbol": "MPHASIS.NS"},
    "PERSISTENT": {"name": "Persistent Systems Ltd", "sector": "IT", "symbol": "PERSISTENT.NS"},
    "COFORGE": {"name": "Coforge Ltd", "sector": "IT", "symbol": "COFORGE.NS"}
}

def search_indian_stocks(query):
    """Search for Indian stocks by symbol or name"""
    if not query or len(query) < 1:
        return []
    
    query_upper = query.upper().strip()
    results = []
    
    # Search by symbol
    for symbol, data in INDIAN_STOCKS.items():
        if query_upper in symbol:
            results.append({
                "symbol": symbol,
                "name": data["name"],
                "sector": data["sector"],
                "type": "Equity",
                "region": "India",
                "currency": "INR",
                "yf_symbol": data["symbol"]
            })
    
    # Search by company name
    for symbol, data in INDIAN_STOCKS.items():
        if query_upper in data["name"].upper():
            # Avoid duplicates
            if not any(r["symbol"] == symbol for r in results):
                results.append({
                    "symbol": symbol,
                    "name": data["name"],
                    "sector": data["sector"],
                    "type": "Equity",
                    "region": "India",
                    "currency": "INR",
                    "yf_symbol": data["symbol"]
                })
    
    # Search by sector
    for symbol, data in INDIAN_STOCKS.items():
        if query_upper in data["sector"].upper():
            # Avoid duplicates
            if not any(r["symbol"] == symbol for r in results):
                results.append({
                    "symbol": symbol,
                    "name": data["name"],
                    "sector": data["sector"],
                    "type": "Equity",
                    "region": "India",
                    "currency": "INR",
                    "yf_symbol": data["symbol"]
                })
    
    return results[:20]  # Return max 20 results

def get_indian_stock_quote(symbol):
    """Get current Indian stock quote with basic information"""
    # Get the yfinance symbol
    if symbol in INDIAN_STOCKS:
        yf_symbol = INDIAN_STOCKS[symbol]["symbol"]
    else:
        # Try with .NS suffix
        yf_symbol = f"{symbol}.NS"
    
    def fetch_quote(symbol, yf_symbol):
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            # Check if we have valid info
            if not info or len(info) < 2:
                print(f"No valid info for {symbol}")
                return None
            
            # Get recent pricing data
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
                "52w_low": info.get('fiftyTwoWeekLow', hist_52w_low),
                "currency": "INR",
                "exchange": "NSE"
            }
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None
    
    return get_from_cache_or_api(fetch_quote, symbol, yf_symbol)

def get_indian_stock_daily_data(symbol, period="1y"):
    """Get daily Indian stock data"""
    # Get the yfinance symbol
    if symbol in INDIAN_STOCKS:
        yf_symbol = INDIAN_STOCKS[symbol]["symbol"]
    else:
        # Try with .NS suffix
        yf_symbol = f"{symbol}.NS"
    
    def fetch_daily(symbol, yf_symbol, period):
        try:
            ticker = yf.Ticker(yf_symbol)
            
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
    
    return get_from_cache_or_api(fetch_daily, symbol, yf_symbol, period)

def get_indian_stock_intraday_data(symbol, interval="5m"):
    """Get intraday Indian stock data"""
    # Get the yfinance symbol
    if symbol in INDIAN_STOCKS:
        yf_symbol = INDIAN_STOCKS[symbol]["symbol"]
    else:
        # Try with .NS suffix
        yf_symbol = f"{symbol}.NS"
    
    def fetch_intraday(symbol, yf_symbol, interval):
        try:
            ticker = yf.Ticker(yf_symbol)
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
    
    return get_from_cache_or_api(fetch_intraday, symbol, yf_symbol, interval)

def get_indian_stock_weekly_data(symbol):
    """Get weekly Indian stock data"""
    # Get the yfinance symbol
    if symbol in INDIAN_STOCKS:
        yf_symbol = INDIAN_STOCKS[symbol]["symbol"]
    else:
        # Try with .NS suffix
        yf_symbol = f"{symbol}.NS"
    
    def fetch_weekly(symbol, yf_symbol):
        try:
            ticker = yf.Ticker(yf_symbol)
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
    
    return get_from_cache_or_api(fetch_weekly, symbol, yf_symbol)

def get_popular_indian_stocks():
    """Get list of popular Indian stocks for quick access"""
    popular_stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ITC", "SBIN", 
        "BHARTIARTL", "KOTAKBANK", "LT", "ASIANPAINT", "MARUTI", "AXISBANK",
        "NESTLEIND", "ULTRACEMCO", "TITAN", "POWERGRID", "NTPC", "SUNPHARMA", "TECHM"
    ]
    
    return [
        {
            "symbol": symbol,
            "name": INDIAN_STOCKS[symbol]["name"],
            "sector": INDIAN_STOCKS[symbol]["sector"],
            "yf_symbol": INDIAN_STOCKS[symbol]["symbol"]
        }
        for symbol in popular_stocks
    ]

def get_indian_sectors():
    """Get list of Indian market sectors"""
    sectors = set()
    for stock_data in INDIAN_STOCKS.values():
        sectors.add(stock_data["sector"])
    return sorted(list(sectors))
