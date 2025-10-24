#!/usr/bin/env python3
"""
Test script for Indian stock market functionality
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indian_stock_data import (
    search_indian_stocks,
    get_indian_stock_quote,
    get_indian_stock_daily_data,
    get_popular_indian_stocks,
    get_indian_sectors
)

from indian_technical_indicators import (
    get_indian_stock_analysis,
    get_indian_stock_summary
)

def test_indian_stock_search():
    """Test Indian stock search functionality"""
    print("Testing Indian stock search...")
    
    # Test search by symbol
    results = search_indian_stocks("RELIANCE")
    print(f"Search for 'RELIANCE': {len(results)} results")
    if results:
        print(f"First result: {results[0]}")
    
    # Test search by company name
    results = search_indian_stocks("Tata")
    print(f"Search for 'Tata': {len(results)} results")
    if results:
        print(f"First result: {results[0]}")
    
    # Test search by sector
    results = search_indian_stocks("IT")
    print(f"Search for 'IT': {len(results)} results")
    if results:
        print(f"First result: {results[0]}")

def test_indian_stock_data():
    """Test Indian stock data fetching"""
    print("\nTesting Indian stock data fetching...")
    
    # Test with RELIANCE
    try:
        quote = get_indian_stock_quote("RELIANCE")
        if quote:
            print(f"RELIANCE quote: {quote['name']} - ₹{quote['price']:.2f}")
        else:
            print("RELIANCE quote: No data available")
    except Exception as e:
        print(f"Error getting RELIANCE quote: {e}")
    
    # Test with TCS
    try:
        quote = get_indian_stock_quote("TCS")
        if quote:
            print(f"TCS quote: {quote['name']} - ₹{quote['price']:.2f}")
        else:
            print("TCS quote: No data available")
    except Exception as e:
        print(f"Error getting TCS quote: {e}")

def test_indian_stock_analysis():
    """Test Indian stock technical analysis"""
    print("\nTesting Indian stock technical analysis...")
    
    # Get daily data for RELIANCE
    try:
        df = get_indian_stock_daily_data("RELIANCE", period="3mo")
        if df is not None and not df.empty:
            print(f"RELIANCE data: {len(df)} days")
            
            # Get technical analysis
            analysis = get_indian_stock_analysis(df, "RELIANCE", "Oil & Gas")
            if analysis:
                print("Technical analysis completed successfully")
                
                # Get summary
                summary = get_indian_stock_summary(analysis, df, "RELIANCE")
                if summary:
                    print(f"Current RSI: {summary['rsi']:.2f}")
                    print(f"Current MACD: {summary['macd']:.4f}")
                    print(f"Volatility: {summary['volatility']:.2f}%")
            else:
                print("Technical analysis failed")
        else:
            print("No data available for RELIANCE")
    except Exception as e:
        print(f"Error in technical analysis: {e}")

def test_popular_stocks():
    """Test popular Indian stocks"""
    print("\nTesting popular Indian stocks...")
    
    popular = get_popular_indian_stocks()
    print(f"Popular stocks: {len(popular)}")
    
    for i, stock in enumerate(popular[:5]):
        print(f"{i+1}. {stock['symbol']} - {stock['name']} ({stock['sector']})")

def test_sectors():
    """Test Indian market sectors"""
    print("\nTesting Indian market sectors...")
    
    sectors = get_indian_sectors()
    print(f"Available sectors: {len(sectors)}")
    print(f"Sectors: {', '.join(sectors[:10])}...")

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Indian Stock Market Functionality")
    print("=" * 50)
    
    try:
        test_indian_stock_search()
        test_indian_stock_data()
        test_indian_stock_analysis()
        test_popular_stocks()
        test_sectors()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
