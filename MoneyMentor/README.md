# MoneyMentor-AI

A real-time stock analysis platform with AI-powered insights for active traders and investors.

## Features

- **Real-time Stock Data**: Get live data from Yahoo Finance
- **Technical Analysis**: Calculate key indicators like RSI, MACD, and Bollinger Bands
- **Stock Watchlists**: Create and manage watchlists of your favorite stocks
- **AI Predictions**: Get AI-powered trend predictions and trading signals
- **Sentiment Analysis**: Analyze news sentiment for stocks

## Installation

1. Install dependencies:
```
pip install -r dependencies.txt
```

2. Run the application:
```
streamlit run app.py
```

## Usage

- Use the search bar to find stocks by symbol or name
- Switch between Dashboard and Watchlists views
- Create watchlists and add stocks to track
- Analyze stock charts with different time frames
- View technical indicators and sentiment analysis

## Dependencies

- numpy
- pandas
- plotly
- psycopg2-binary
- requests
- sqlalchemy
- streamlit
- yfinance