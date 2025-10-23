# Indian Stock Market Integration for MoneyMentor-AI

## Overview

The MoneyMentor-AI application has been successfully extended to support the Indian stock market, providing comprehensive analysis for all major companies listed on NSE (National Stock Exchange) and BSE (Bombay Stock Exchange).

## Features Added

### 1. Indian Stock Market Data Integration
- **NSE Support**: Full support for NSE-listed stocks with `.NS` suffix
- **BSE Support**: Support for BSE-listed stocks with `.BO` suffix
- **Real-time Data**: Live stock quotes, historical data, and market information
- **Currency Support**: All prices displayed in INR (Indian Rupees)

### 2. Comprehensive Stock Database
- **80+ Major Indian Stocks**: Including all Nifty 50 and Sensex 30 companies
- **Sector Classification**: 23 different sectors including IT, Banking, FMCG, Pharma, etc.
- **Company Information**: Full company names, sectors, and market details

### 3. Enhanced Search Functionality
- **Symbol Search**: Search by stock symbols (e.g., "RELIANCE", "TCS")
- **Company Name Search**: Search by company names (e.g., "Tata", "Reliance")
- **Sector Search**: Search by industry sectors (e.g., "IT", "Banking")
- **Smart Suggestions**: Real-time search suggestions with company details

### 4. Indian Market-Specific Technical Analysis
- **RSI with Indian Market Adjustments**: Optimized for Indian market volatility
- **MACD with Sector-Specific Parameters**: Different parameters for different sectors
- **Bollinger Bands**: Adapted for Indian market characteristics
- **Volume Analysis**: Indian market-specific volume indicators
- **Momentum Indicators**: 5-day, 10-day, and 20-day momentum analysis
- **Volatility Indicators**: Historical volatility and ATR calculations

### 5. Sector-Specific Analysis
- **Banking Sector**: Interest rate sensitivity, credit growth indicators
- **IT Sector**: Dollar sensitivity, client concentration risk
- **Pharma Sector**: Regulatory risk, pipeline strength indicators
- **FMCG Sector**: Consumer sentiment, price elasticity analysis
- **Auto Sector**: Seasonal factors, raw material sensitivity
- **Energy Sector**: Energy price sensitivity, government policy impact

### 6. Market Sentiment Analysis
- **Fear-Greed Index**: Indian market-specific sentiment indicators
- **Advance-Decline Ratio**: Market breadth analysis
- **Volume Profile**: Trading activity analysis
- **Momentum Analysis**: Short-term and medium-term momentum

### 7. AI-Powered Predictions for Indian Stocks
- **Stock-Specific Biases**: Individual stock characteristics and market position
- **Sector-Based Predictions**: Different prediction models for different sectors
- **Indian Market Factors**: Consideration of Indian market-specific factors
- **Accuracy Metrics**: Performance tracking for Indian stocks

## Supported Indian Stocks

### Major Large-Cap Stocks
- **RELIANCE** - Reliance Industries Ltd (Oil & Gas)
- **TCS** - Tata Consultancy Services Ltd (IT)
- **HDFCBANK** - HDFC Bank Ltd (Banking)
- **INFY** - Infosys Ltd (IT)
- **HINDUNILVR** - Hindustan Unilever Ltd (FMCG)
- **ITC** - ITC Ltd (FMCG)
- **SBIN** - State Bank of India (Banking)
- **BHARTIARTL** - Bharti Airtel Ltd (Telecom)
- **KOTAKBANK** - Kotak Mahindra Bank Ltd (Banking)
- **LT** - Larsen & Toubro Ltd (Engineering)

### IT Sector Stocks
- **TCS** - Tata Consultancy Services Ltd
- **INFY** - Infosys Ltd
- **WIPRO** - Wipro Ltd
- **HCLTECH** - HCL Technologies Ltd
- **TECHM** - Tech Mahindra Ltd
- **MINDTREE** - Mindtree Ltd
- **LTI** - Larsen & Toubro Infotech Ltd
- **MPHASIS** - Mphasis Ltd
- **PERSISTENT** - Persistent Systems Ltd
- **COFORGE** - Coforge Ltd

### Banking & Financial Services
- **HDFCBANK** - HDFC Bank Ltd
- **SBIN** - State Bank of India
- **KOTAKBANK** - Kotak Mahindra Bank Ltd
- **AXISBANK** - Axis Bank Ltd
- **BAJFINANCE** - Bajaj Finance Ltd
- **BAJAJFINSV** - Bajaj Finserv Ltd

### Pharma & Healthcare
- **SUNPHARMA** - Sun Pharmaceutical Industries Ltd
- **DRREDDY** - Dr. Reddy's Laboratories Ltd
- **CIPLA** - Cipla Ltd
- **APOLLOHOSP** - Apollo Hospitals Enterprise Ltd
- **DIVISLAB** - Divi's Laboratories Ltd
- **BIOCON** - Biocon Ltd
- **LUPIN** - Lupin Ltd
- **CADILAHC** - Cadila Healthcare Ltd

### FMCG & Consumer Goods
- **HINDUNILVR** - Hindustan Unilever Ltd
- **ITC** - ITC Ltd
- **NESTLEIND** - Nestle India Ltd
- **TITAN** - Titan Company Ltd
- **TATACONSUM** - Tata Consumer Products Ltd
- **BRITANNIA** - Britannia Industries Ltd
- **DABUR** - Dabur India Ltd
- **GODREJCP** - Godrej Consumer Products Ltd

### Automobile Sector
- **MARUTI** - Maruti Suzuki India Ltd
- **TATAMOTORS** - Tata Motors Ltd
- **EICHERMOT** - Eicher Motors Ltd
- **HEROMOTOCO** - Hero MotoCorp Ltd
- **BAJAJ-AUTO** - Bajaj Auto Ltd
- **M&M** - Mahindra & Mahindra Ltd

### Infrastructure & Engineering
- **LT** - Larsen & Toubro Ltd
- **ASIANPAINT** - Asian Paints Ltd
- **ULTRACEMCO** - UltraTech Cement Ltd
- **POWERGRID** - Power Grid Corporation of India Ltd
- **NTPC** - NTPC Ltd
- **BHEL** - Bharat Heavy Electricals Ltd
- **SIEMENS** - Siemens Ltd

### Energy & Utilities
- **RELIANCE** - Reliance Industries Ltd
- **ONGC** - Oil and Natural Gas Corporation Ltd
- **COALINDIA** - Coal India Ltd
- **POWERGRID** - Power Grid Corporation of India Ltd
- **NTPC** - NTPC Ltd
- **ADANIGREEN** - Adani Green Energy Ltd

## How to Use

### 1. Market Selection
- Choose "Indian Markets" from the market selection radio buttons
- The interface will switch to Indian stock market mode

### 2. Stock Search
- Use the search box to find Indian stocks by:
  - Symbol (e.g., "RELIANCE")
  - Company name (e.g., "Tata")
  - Sector (e.g., "IT")

### 3. Popular Indian Stocks
- Quick access buttons for major Indian stocks
- Organized by market cap and popularity

### 4. Technical Analysis
- Indian market-specific RSI, MACD, and Bollinger Bands
- Sector-specific indicators and analysis
- Volume analysis tailored for Indian markets

### 5. AI Predictions
- Stock-specific prediction models
- Sector-based analysis
- Indian market factors consideration

## Technical Implementation

### Files Added/Modified
1. **indian_stock_data.py** - Indian stock market data integration
2. **indian_technical_indicators.py** - Indian market-specific technical analysis
3. **app.py** - Updated main application with Indian market support
4. **prediction_model.py** - Enhanced with Indian stock predictions

### Key Features
- **Caching**: 5-minute cache for API responses to avoid rate limiting
- **Error Handling**: Robust error handling for data fetching
- **Performance**: Optimized for Indian market data characteristics
- **Scalability**: Easy to add more Indian stocks and sectors

## Testing

A comprehensive test suite is included in `test_indian_stocks.py` that verifies:
- Stock search functionality
- Data fetching for major Indian stocks
- Technical analysis calculations
- Popular stocks and sectors

Run the test with:
```bash
python test_indian_stocks.py
```

## Future Enhancements

1. **BSE Integration**: Full BSE support with `.BO` suffix
2. **More Stocks**: Expand to include mid-cap and small-cap stocks
3. **Sector ETFs**: Support for sector-specific ETFs
4. **Options Data**: Support for Indian options and derivatives
5. **News Integration**: Indian financial news sentiment analysis
6. **Regulatory Updates**: Real-time regulatory and policy impact analysis

## Support

The Indian stock market integration is fully functional and ready for use. All major NSE stocks are supported with comprehensive technical analysis and AI-powered predictions tailored for the Indian market characteristics.

For any issues or feature requests, please refer to the main MoneyMentor-AI documentation.
